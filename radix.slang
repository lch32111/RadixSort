#define SORT_BITS_PER_PASS 4u
#define SORT_BIN_COUNT (1u << SORT_BITS_PER_PASS)
#define ELEMENTS_PER_THREAD 4u
#define THREADGROUP_SIZE 128u

struct RadixSortConfig
{
    uint num_keys;
    uint shift_bit;
    uint num_blocks_per_threadgroup;
    uint num_thread_groups;
    uint num_threadgroups_with_additional_block;
    uint num_reduce_threadgroup_per_bin;
    uint num_scan_values;
}

struct RadixSort
{
    RWStructuredBuffer<uint> src_data;
    RWStructuredBuffer<uint> dst_data;

    RWStructuredBuffer<uint> sum_table;
    RWStructuredBuffer<uint> sum_reduce_table;

    ConstantBuffer<RadixSortConfig> config;
};

ParameterBlock<RadixSort> g_sort;

groupshared uint lds_histogram[THREADGROUP_SIZE * SORT_BIN_COUNT];
groupshared uint lds_sums[THREADGROUP_SIZE];

[numthreads(THREADGROUP_SIZE, 1, 1)]
void count_pass(uint3 local_id3 : SV_GroupThreadID, uint group_id3 : SV_GroupID)
{
    uint local_id = local_id3.x;
    uint group_id = group_id3.x;

    for (uint i = 0; i < SORT_BIN_COUNT; ++i)
        lds_histogram[(i * THREADGROUP_SIZE) + local_id] = 0;

    // Wait for everyone to catch up
    GroupMemoryBarrierWithGroupSync();

    uint block_size = ELEMENTS_PER_THREAD * THREADGROUP_SIZE;

    uint threadgroup_block_start = (block_size * g_sort.config.num_blocks_per_threadgroup * group_id);
    uint num_blocks_to_process = g_sort.config.num_blocks_per_threadgroup;

    if (group_id >= g_sort.config.num_blocks_per_threadgroup - g_sort.config.num_threadgroups_with_additional_block)
    {
        // this thread in this thread group need to process one more block
        threadgroup_block_start += (group_id - (g_sort.config.num_thread_groups - g_sort.config.num_threadgroups_with_additional_block)) * block_size;
        ++num_blocks_to_process;
    }

    uint block_index = threadgroup_block_start + local_id;
    for (uint block_count = 0; block_count < num_blocks_to_process; ++block_count, block_index += block_size)
    {
        uint data_index = block_index;

        // pre-load the key values in order to hide some of the read latency
        uint src_keys[ELEMENTS_PER_THREAD];

        [[ForceUnroll]]
        for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i)
        {
            if (data_index < g_sort.config.num_keys)
                src_keys[i] = g_sort.src_data[data_index];

            data_index += THREADGROUP_SIZE;
        }

        data_index = block_index;

        [[ForceUnroll]]
        for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i)
        {
            if (data_index < g_sort.config.num_keys)
            {
                uint local_key = (src_keys[i] >> g_sort.config.shift_bit) & 0xFu;

                // NOTE(@chan): I guess there would be no need for InterlockedAdd here if each thread has BINs??
                // InterlockedAdd(histogram[(local_key * THREADGROUP_SIZE) + local_id], 1);
                lds_histogram[(local_key * THREADGROUP_SIZE) + local_id] += 1;
            }

            data_index += THREADGROUP_SIZE;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (local_id < SORT_BIN_COUNT)
    {
        // NOTE(@chan): now each local_id means the local_key
        // sums up each thread count value for this local_key
        uint sum = 0;
        for (int i = 0; i < THREADGROUP_SIZE; ++i)
        {
            sum += lds_histogram[local_id * THREADGROUP_SIZE + i];
        }

        // NOTE(@chan):
        // The layout would be like this
        // Bin0 [Group0, Group1, ...]
        // Bin1 [Group0, Group1, ...]
        // ...
        g_sort.sum_table[local_id * g_sort.config.num_thread_groups + group_id] = sum;
    }
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
void count_reduce_pass(uint3 local_id3 : SV_GroupThreadID, uint3 group_id3 : SV_GroupID)
{
    uint local_id = local_id3.x;
    uint group_id = group_id3.x;

    // NOTE(@chan): each group on this reduce pass cares one bin.
    // To know what bin this group will handle, we get bin_id.
    // After that, we get exact offset of bin x start by bin_id * num_thread_groups
    uint bin_id = group_id / g_sort.config.num_reduce_threadgroup_per_bin;
    uint bin_offset = bin_id * g_sort.config.num_thread_groups;

    // NOTE(@chan): Out of reduce threadgroups for bin X,
    // what is the base index of this group?
    uint base_index = (group_id % g_sort.config.num_reduce_threadgroup_per_bin) * ELEMENTS_PER_THREAD * THREADGROUP_SIZE;

    uint threadgroup_sum = 0;
    for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
        // NOTE(@chan): Out of elements in this reduce threadgroup for bin X,
        // what this thread handles
        uint data_index = base_index + (i * THREADGROUP_SIZE) + local_id;
        if (data_index < g_sort.config.num_thread_groups)
        {
            threadgroup_sum += g_sort.sum_table[bin_offset + data_index];
        }
    }

    // Reduce across the entirety of the thread group
    uint lane_count = WaveGetLaneCount();
    uint wave_reduced = WaveActiveSum(threadgroup_sum);

    uint wave_id = local_id.x / lane_count;
    if (WaveIsFirstLane())
        lds_sums[wave_id] = wave_reduced;

    // NOTE(@chan): let all the threads write lds_sums
    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0) // NOTE(@chan): only the first wave gets the threadgroup sums
    {
        // NOTE(@chan): As AMD code says, it will not work well when the lane size is less tha or equal to 8,
        // because the local_id.x (currently here a lane index) cannot handle the other waves.
        // Let's say we have 128 threadgroup size and 8 lanes, then there would be 32 waves.
        // the thread id from 0 to 7 is in the first wave and it can see only from 0 to 7 waves, except waves from 8 to 31 waves.
        // In this we should set a ELEMENTS_PER_LANE (here 4 in this example) and then calculate it.
        // But I don't know I can do it in wave level or not (I guess in CUDA you can do it.)
        // If we find a HW with this setup, then we can start to fix this.
        uint wave_count = THREADGROUP_SIZE / lane_count;
        wave_reduced = WaveActiveSum(local_id < wave_count ? lds_sums[local_id] : 0);
    }

    // NOTE(@chan): as the first wave holds the threadgroup sums, we let the first thread write the sum
    if (local_id == 0)
        g_sort.sum_reduce_table[group_id] = wave_reduced;
}