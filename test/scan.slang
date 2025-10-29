#define SORT_BITS_PER_PASS 4
#define SORT_BIN_COUNT (1 << SORT_BITS_PER_PASS)
#define ELEMENTS_PER_THREAD 4
#define THREADGROUP_SIZE 128

struct ScanPass
{
    RWStructuredBuffer<uint> scan_src;
    RWStructuredBuffer<uint> scan_dst;
    RWStructuredBuffer<uint> scan_scratch;
    uint config_shift_bit;
    uint config_num_keys;
    int config_num_blocks_per_threadgroup;
    uint config_num_thread_groups;
    uint config_num_threadgroups_with_additional_block;
    uint config_num_reduce_threadgroup_per_bin;
    uint config_num_scan_values;
};

groupshared uint lds_sum[THREADGROUP_SIZE];
uint block_scan_prefix(uint local_sum, uint local_id)
{
    // Do wave local scan-prefix
    uint wave_prefixed = WavePrefixSum(local_sum);

    // Since we are dealing with thread group sizes greater than HW wave size, we need to account for what wave we are in.
    uint wave_id = local_id / WaveGetLaneCount();
    uint lane_id = WaveGetLaneIndex();

    // Last element in a wave writes out partial sum to LDS
    if (lane_id == WaveGetLaneCount() - 1)
        lds_sum[wave_id] = wave_prefixed + local_sum;

    GroupMemoryBarrierWithGroupSync();

    // First wave prefixes partial sums
    if (wave_id == 0)
        lds_sum[local_id] = WavePrefixSum(lds_sum[local_id]);

    GroupMemoryBarrierWithGroupSync();

    // Add the partial sums back to each wave prefix
    wave_prefixed += lds_sum[wave_id];

    return wave_prefixed;
}

ParameterBlock<ScanPass> pass;
groupshared int lds[ELEMENTS_PER_THREAD][THREADGROUP_SIZE];
void scan_prefix(uint num_values_to_scan, uint local_id, uint group_id, uint bin_offset, uint base_index, bool add_partial_sum)
{
    uint i;

    for (i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
        uint data_index = base_index + (i * THREADGROUP_SIZE) + local_id;

        // NOTE(@chan):
        // local_id: [0, 1, 2, ..., 127]
        // col: ([0, 128, 256, 384] + local_id) / 4 => [0, 32, 64, 96] + local_id / 4 => 
        //       local_id 0~3 => 0, 4~7 => 1, ..., 124~127 => 31
        //       local_id == 0: [0, 32, 64, 96]
        //       local_id == 1: [0, 32, 64, 96]
        //       local_id == 4: [1, 33, 65, 97]
        // row: ([0, 128, 256, 384] + local_id) % 4 => local_id % 4 => [0, 1, 2, 3]
        // lds[0][0] = scan_src[bin_offset + 0] from thread 0
        // lds[1][0] = scan_src[bin_offset + 1] from thread 1
        // lds[2][0] = scan_src[bin_offset + 2] from thread 2
        // lds[3][0] = scan_src[bin_offset + 3] from thread 3
        // lds[0][1] = scan_src[bin_offset + 4] from thread 4
        // lds[1][1] = scan_src[bin_offset + 5] from thread 5
        // ...
        // lds[0][32] = scan_src[bin_offset + 128] // from thread 0
        // lds[1][32] = scan_src[bin_offset + 129] // from thread 1
        // ...
        // lds[0][64] = scan_src[bin_offset + 256] // from thread 0
        // ...        
        uint col = ((i * THREADGROUP_SIZE) + local_id) / ELEMENTS_PER_THREAD;
        uint row = ((i * THREADGROUP_SIZE) + local_id) % ELEMENTS_PER_THREAD;
        lds[row][col] = (data_index < num_values_to_scan) ? pass.scan_src[bin_offset + data_index] : 0;
    }

    GroupMemoryBarrierWithGroupSync();

    // local scan-prefix for current thread
    // after this lds[0..3][local_id] exclusive prefix array for ELEMENTS_PER_THREAD values.
    uint threadgroup_sum = 0;
    for (i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
        uint tmp = lds[i][local_id];
        lds[i][local_id] = threadgroup_sum;
        threadgroup_sum += tmp;
    }

    // scan prefix partial sums
    threadgroup_sum = block_scan_prefix(threadgroup_sum, local_id);

    // Add reduced partial sums if requested
    uint partial_sum = 0;
    if (add_partial_sum)
    {
        partial_sum = pass.scan_scratch[group_id];
    }

    // Add the block scanned-prefixes back in
    for (i = 0; i < ELEMENTS_PER_THREAD; ++i)
        lds[i][local_id] += threadgroup_sum;

    GroupMemoryBarrierWithGroupSync();

    // Perform coalesced writes to scan dst
    for (i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
        uint data_index = base_index + (i * THREADGROUP_SIZE) + local_id;
        uint col = ((i * THREADGROUP_SIZE) + local_id) / ELEMENTS_PER_THREAD;
        uint row = ((i * THREADGROUP_SIZE) + local_id) % ELEMENTS_PER_THREAD;
        if (data_index < num_values_to_scan)
            pass.scan_dst[bin_offset + data_index] = lds[row][col] + partial_sum;
    }
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
void scan_pass(uint3 local_id: SV_GroupThreadID, uint3 group_id: SV_GroupID)
{
    // NOTE(@chan): 
    // scan_src is the reduce_table from the count pass
    // scan_dst is the reduce_table from the count pass as well.
    // scan_pass does not use scan_scratch.
    // scan_pass is dispatched with (1, 1, 1), which means group_id.x is always 0.
    // Therefore base_index is always 0.
    // uint base_index = ELEMENTS_PER_THREAD * THREADGROUP_SIZE * group_id.x;
    scan_prefix(pass.config_num_scan_values, local_id.x, group_id.x, 0, 0, false);
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
void scan_add_pass(uint3 local_id: SV_GroupThreadID, uint3 group_id: SV_GroupID)
{
    // NOTE(@chan):
    // scan_src is the sum_table from the count pass
    // scan_dst is the sum_table from the count pass as well.
    // scan_scratch is the reduce_table updated by scan_pass. scan_scratch is used in scan_add_pass.
    uint bin_id = group_id.x / pass.config_num_reduce_threadgroup_per_bin;
    uint bin_offset = bin_id * pass.config_num_thread_groups;
    uint base_index = (group_id.x % pass.config_num_reduce_threadgroup_per_bin) * ELEMENTS_PER_THREAD * THREADGROUP_SIZE;
    scan_prefix(pass.config_num_thread_groups, local_id.x, group_id.x, bin_offset, base_index, true);
}