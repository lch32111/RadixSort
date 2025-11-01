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
    uint num_threadgroups_with_additional_blocks;
    uint num_reduce_threadgroup_per_bin;
    uint num_scan_values;
}

struct RadixSort
{
    RWStructuredBuffer<uint> src_data;
    RWStructuredBuffer<uint> dst_data;

    RWStructuredBuffer<uint> sum_table;
    RWStructuredBuffer<uint> sum_reduce_table;

#ifdef RADIXSORT_PAYLOAD
    RWStructuredBuffer<uint> src_payload;
    RWStructuredBuffer<uint> dst_payload;
#endif

    ConstantBuffer<RadixSortConfig> config;
};

ParameterBlock<RadixSort> g_sort;

groupshared uint lds_histogram[THREADGROUP_SIZE * SORT_BIN_COUNT];
groupshared uint lds_sum[THREADGROUP_SIZE];
groupshared uint lds_prefix[ELEMENTS_PER_THREAD][THREADGROUP_SIZE];

groupshared uint lds_bin_offset_cache[THREADGROUP_SIZE];
groupshared uint lds_bin_local_histogram[SORT_BIN_COUNT];
groupshared uint lds_scratch[THREADGROUP_SIZE];

[numthreads(THREADGROUP_SIZE, 1, 1)]
void count_pass(uint local_id : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    for (uint i = 0; i < SORT_BIN_COUNT; ++i)
        lds_histogram[(i * THREADGROUP_SIZE) + local_id] = 0;

    // Wait for everyone to catch up
    GroupMemoryBarrierWithGroupSync();

    uint block_size = ELEMENTS_PER_THREAD * THREADGROUP_SIZE;

    uint threadgroup_block_start = (block_size * g_sort.config.num_blocks_per_threadgroup * group_id);
    uint num_blocks_to_process = g_sort.config.num_blocks_per_threadgroup;

    if (group_id >= g_sort.config.num_thread_groups - g_sort.config.num_threadgroups_with_additional_blocks)
    {
        // this thread in this thread group need to process one more block
        threadgroup_block_start += (group_id - (g_sort.config.num_thread_groups - g_sort.config.num_threadgroups_with_additional_blocks)) * block_size;
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
                // InterlockedAdd(lds_histogram[(local_key * THREADGROUP_SIZE) + local_id], 1);
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
        for (uint i = 0; i < THREADGROUP_SIZE; ++i)
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
void count_reduce_pass(uint local_id : SV_GroupThreadID, uint group_id : SV_GroupID)
{
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

    uint wave_id = local_id / lane_count;
    if (WaveIsFirstLane())
        lds_sum[wave_id] = wave_reduced;

    // NOTE(@chan): let all the threads write lds_sum
    GroupMemoryBarrierWithGroupSync();

    if (wave_id == 0) // NOTE(@chan): only the first wave gets the threadgroup sums
    {
        // NOTE(@chan): As AMD code says, it will not work well when the lane size is less tha or equal to 8,
        // because the local_id (currently here a lane index) cannot handle the other waves.
        // Let's say we have 128 threadgroup size and 8 lanes, then there would be 32 waves.
        // the thread id from 0 to 7 is in the first wave and it can see only from 0 to 7 waves, except waves from 8 to 31 waves.
        // In this we should set a ELEMENTS_PER_LANE (here 4 in this example) and then calculate it.
        // But I don't know I can do it in wave level or not (I guess in CUDA you can do it.)
        // If we find a HW with this setup, then we can start to fix this.
        uint wave_count = THREADGROUP_SIZE / lane_count;
        wave_reduced = WaveActiveSum(local_id < wave_count ? lds_sum[local_id] : 0);
    }

    // NOTE(@chan): as the first wave holds the threadgroup sums, we let the first thread write the sum
    if (local_id == 0)
        g_sort.sum_reduce_table[group_id] = wave_reduced;
}

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

void scan_prefix(uint num_values_to_scan, uint local_id, uint group_id, uint bin_offset, uint base_index, bool add_partial_sum,
                 RWStructuredBuffer<uint> scan_src,
                 RWStructuredBuffer<uint> scan_dst,
                 RWStructuredBuffer<uint> scan_scratch)
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
        lds_prefix[row][col] = (data_index < num_values_to_scan) ? scan_src[bin_offset + data_index] : 0;
    }

    GroupMemoryBarrierWithGroupSync();

    // local scan-prefix for current thread
    // after this lds_prefix[0..3][local_id] exclusive prefix array for ELEMENTS_PER_THREAD values.
    uint threadgroup_sum = 0;
    for (i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
        uint tmp = lds_prefix[i][local_id];
        lds_prefix[i][local_id] = threadgroup_sum;
        threadgroup_sum += tmp;
    }

    // scan prefix partial sums
    threadgroup_sum = block_scan_prefix(threadgroup_sum, local_id);

    // Add reduced partial sums if requested
    uint partial_sum = 0;
    if (add_partial_sum)
    {
        partial_sum = scan_scratch[group_id];
    }

    // Add the block scanned-prefixes back in
    for (i = 0; i < ELEMENTS_PER_THREAD; ++i)
        lds_prefix[i][local_id] += threadgroup_sum;

    GroupMemoryBarrierWithGroupSync();

    // Perform coalesced writes to scan dst
    for (i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
        uint data_index = base_index + (i * THREADGROUP_SIZE) + local_id;
        uint col = ((i * THREADGROUP_SIZE) + local_id) / ELEMENTS_PER_THREAD;
        uint row = ((i * THREADGROUP_SIZE) + local_id) % ELEMENTS_PER_THREAD;
        if (data_index < num_values_to_scan)
            scan_dst[bin_offset + data_index] = lds_prefix[row][col] + partial_sum;
    }
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
void scan_pass(uint local_id : SV_GroupThreadID, uint group_id : SV_GroupID)
{
    // NOTE(@chan):
    // scan_src is the reduce_table from the count pass
    // scan_dst is the reduce_table from the count pass as well.
    // scan_pass does not use scan_scratch.
    // scan_pass is dispatched with (1, 1, 1), which means group_id.x is always 0.
    // Therefore base_index is always 0.
    // uint base_index = ELEMENTS_PER_THREAD * THREADGROUP_SIZE * group_id.x;
    scan_prefix(g_sort.config.num_scan_values, local_id, group_id, 0, 0, false,
                g_sort.sum_reduce_table,
                g_sort.sum_reduce_table,
                g_sort.sum_reduce_table);
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
void scan_add_pass(uint local_id: SV_GroupThreadID, uint group_id: SV_GroupID)
{
    // NOTE(@chan):
    // scan_src is the sum_table from the count pass
    // scan_dst is the sum_table from the count pass as well.
    // scan_scratch is the reduce_table updated by scan_pass. scan_scratch is used in scan_add_pass.
    uint bin_id = group_id / g_sort.config.num_reduce_threadgroup_per_bin;
    uint bin_offset = bin_id * g_sort.config.num_thread_groups;
    uint base_index = (group_id % g_sort.config.num_reduce_threadgroup_per_bin) * ELEMENTS_PER_THREAD * THREADGROUP_SIZE;
    scan_prefix(g_sort.config.num_thread_groups, local_id, group_id, bin_offset, base_index, true,
                g_sort.sum_table,
                g_sort.sum_table,
                g_sort.sum_reduce_table);
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
void scatter_pass(uint local_id: SV_GroupThreadID, uint group_id: SV_GroupID)
{
    // Load the sort bin threadgroup offsets into LDS for faster referencing
    if (local_id < SORT_BIN_COUNT)
    {
        // NOTE(@chan): currently the sum table has a prescan result of each bin (sum_table[SORT_BIN_COUNT][NUM_THREADGROUPS_TO_RUN])
        // if local_id is 0 across thread groups,
        //      group[0] -> sum_table[0]
        //      group[1] -> sum_table[1]
        //      ...
        //      group[num_thread_gorups - 1] -> sum_table[num_thread_groups - 1]
        // if local_id.x is 1 across thread groups,
        //      group[0] -> sum_table[512] -> sum_table[1][0]
        //      group[1] -> sum_table[513] -> sum_table[1][1]
        // from local_id.x 0 ~ THREADGROUP_SIZE - 1 in group0
        //      local_id 0 sum_table[0] -> sum_table[0][0]
        //      local_id 1 sum_table[512] -> sum_table[1][0]
        //      local_id 2 sum_table[1024] -> sum_table[2][0]
        //      local_id 3 sum_table[1536] -> sum_table[3][0]
        //      local_id 4 sum_table[2048] -> sum_table[4][0] ...
        //      ...
        //      local_id 15 sum_table[7680] -> sum_table[15][0]
        // Therefore, each thread i in a group x takes values sum_table[i][x]
        lds_bin_offset_cache[local_id] = g_sort.sum_table[local_id * g_sort.config.num_thread_groups + group_id];
    }

    GroupMemoryBarrierWithGroupSync();

    uint block_size = ELEMENTS_PER_THREAD * THREADGROUP_SIZE;
    uint threadgroup_block_start = (block_size * g_sort.config.num_blocks_per_threadgroup * group_id);
    uint num_blocks_to_process = g_sort.config.num_blocks_per_threadgroup;

    if (group_id >= g_sort.config.num_thread_groups - g_sort.config.num_threadgroups_with_additional_blocks)
    {
        threadgroup_block_start += (group_id - (g_sort.config.num_thread_groups - g_sort.config.num_threadgroups_with_additional_blocks)) * block_size;
        num_blocks_to_process += 1;
    }

    // Get the block start index for this thread
    uint block_index = threadgroup_block_start + local_id;
    for (uint block_count = 0; block_count < num_blocks_to_process; ++block_count, block_index += block_size)
    {
        uint data_index = block_index;

        // Pre-load the key values in order to hide some of the read latency
        uint src_keys[ELEMENTS_PER_THREAD];
        src_keys[0] = (data_index < g_sort.config.num_keys) ? g_sort.src_data[data_index] : 0xFFFFFFFFu;
        data_index += THREADGROUP_SIZE;

        src_keys[1] = (data_index < g_sort.config.num_keys) ? g_sort.src_data[data_index] : 0xFFFFFFFFu;
        data_index += THREADGROUP_SIZE;

        src_keys[2] = (data_index < g_sort.config.num_keys) ? g_sort.src_data[data_index] : 0xFFFFFFFFu;
        data_index += THREADGROUP_SIZE;

        src_keys[3] = (data_index < g_sort.config.num_keys) ? g_sort.src_data[data_index] : 0xFFFFFFFFu;

#ifdef RADIXSORT_PAYLOAD
        data_index = block_index;

        uint src_values[ELEMENTS_PER_THREAD];
        src_values[0] = (data_index < g_sort.config.num_keys) ? g_sort.src_payload[data_index] : 0u;
        data_index += THREADGROUP_SIZE;

        src_values[1] = (data_index < g_sort.config.num_keys) ? g_sort.src_payload[data_index] : 0u;
        data_index += THREADGROUP_SIZE;

        src_values[2] = (data_index < g_sort.config.num_keys) ? g_sort.src_payload[data_index] : 0u;
        data_index += THREADGROUP_SIZE;

        src_values[3] = (data_index < g_sort.config.num_keys) ? g_sort.src_payload[data_index] : 0u;
#endif
        data_index = block_index;
        for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i)
        {
            // clear the local histogram
            if (local_id < SORT_BIN_COUNT)
                lds_bin_local_histogram[local_id] = 0;

            uint local_key = src_keys[i];
#ifdef RADIXSORT_PAYLOAD
            uint local_value = src_values[i];
#endif

            // Sort the keys locally in LDS
            for (uint bit_shift = 0; bit_shift < SORT_BITS_PER_PASS; bit_shift += 2)
            {
                // Figure out the key index
                uint key_index = (local_key >> g_sort.config.shift_bit) & 0xFu;
                uint bit_key = (key_index >> bit_shift) & 0x3u;

                // Create a packed histogram
                // NOTE(@chan):
                // bit_key == 0 -> 0x      1 ->          1 -> 256^0 -> sum0
                // bit_key == 1 -> 0x    100 ->        256 -> 256^1 -> sum1
                // bit_key == 2 -> 0x  10000 ->     65,536 -> 256^2 -> sum2
                // bit_key == 3 -> 0x1000000 -> 16,777,216 -> 256^3 -> sum3
                // Let's say each thread in a thread group size (128) holds the same bit_key
                // - bit_key == 0 -> 0x1 -> 1 * 128 -> 128 -> 0x00 ~ 0x80
                // - bit_key == 1 -> 0x100 -> 256 * 128 -> 32,768 -> 0x100 ~ 0x8000
                // - bit_key == 2 -> 0x10000 -> 65,536 * 128 -> 8,388,608 -> 0x1_0000 ~ 0x80_0000
                // - bit_key == 3-> 0x1000000 -> 16,777,216 * 128 -> 2,147,483,648 -> 0x100_0000 ~ 0x8000_0000
                // This means that we can accumulate packed_histogram and then count what bit_key appears on each thread.
                uint packed_histogram = 1u << (bit_key * 8u);

                // sum up all the packed keys (generates counted offsets up to current thread group)
                // NOTE(@chan):
                // Right after block_scan_prefix, local_sum has
                // 0xprefix_bin3 | prefix_bin2 | prefix_bin1 | prefix_bin0
                uint local_sum = block_scan_prefix(packed_histogram, local_id.x);

                // Last thread stores the updated histogram counts for the thread group
                // Scratch = 0xsum3|sum2|sum1|sum0 for thread group
                // NOTE(@chan): as the scan prefix does not include the last element, we add the last thread's packed_histogram
                //              to get a total sum.
                if (local_id == (THREADGROUP_SIZE - 1))
                    lds_scratch[0] = local_sum + packed_histogram;

                // Wait for everyone to catch up
                GroupMemoryBarrierWithGroupSync();

                // Load the sums value for the thread group
                packed_histogram = lds_scratch[0];

                // Add prefix offsets for all 4 bit "keys" (packed_histogram = 0xsum2_1_0|sum1_0|sum0|0)
                // NOTE(@chan):
                // packed_histogram << 8  : 0xsum2|sum1|sum0|0
                // packed_histogram << 16 : 0xsum1|sum0|   0|0
                // packed_histogram << 24 : 0xsum0|   0|   0|0
                //                          0xsum_2_1_0 | sum1_0| sum0 | 0
                packed_histogram = (packed_histogram << 8) + (packed_histogram << 16) + (packed_histogram << 24);

                // Calculate the proper offset for this thread's value
                // NOTE(@chan):
                // local_sum:           0xprefix_bin3 | prefix_bin2 | prefix_bin1 | prefix_bin0
                // packed_histogram:    0xsum_2_1_0   | sum1_0      |   sum0      |     0
                // This addition gives you an offset of a specific bin with bit_key.
                local_sum += packed_histogram;

                // calculate target offset
                uint key_offset = (local_sum >> (bit_key * 8)) & 0xFFu;

                // Re-arrange the keys (store, sync, load)
                lds_sum[key_offset] = local_key;
                GroupMemoryBarrierWithGroupSync();
                local_key = lds_sum[local_id];

                GroupMemoryBarrierWithGroupSync();

#ifdef RADIXSORT_PAYLOAD
                lds_sum[key_offset] = local_value;
                GroupMemoryBarrierWithGroupSync();
                local_value = lds_sum[local_id];

                GroupMemoryBarrierWithGroupSync();
#endif
            }

            // Need to recalculate the key_index on this thread now that values have been copied around the thread group
            // NOTE(@chan): finally we get a locally sorted key in a thread group
            uint key_index = (local_key >> g_sort.config.shift_bit) & 0xFu;
            
            // Reconstruct histogram
            InterlockedAdd(lds_bin_local_histogram[key_index], 1);

            GroupMemoryBarrierWithGroupSync();

            // Prefix histogram
            uint histogram_prefix_sum = WavePrefixSum(local_id < SORT_BIN_COUNT ? lds_bin_local_histogram[local_id] : 0);

            // Broadcast prefix-sum via LDS
            if (local_id < SORT_BIN_COUNT)
                lds_scratch[local_id] = histogram_prefix_sum;

            // Get the global offset for this key out of the cache
            uint global_offset = lds_bin_offset_cache[key_index];

            GroupMemoryBarrierWithGroupSync();

            // Get the local offset (at this point the keys are all in increasing order from 0 -> num bins in localID 0 -> thread group size)
            uint local_offset = local_id - lds_scratch[key_index];

            // Write to detination
            uint total_offset = global_offset + local_offset;
            if (data_index < g_sort.config.num_keys && total_offset < g_sort.config.num_keys)
            {
                g_sort.dst_data[total_offset] = local_key;
#ifdef RADIXSORT_PAYLOAD
                g_sort.dst_payload[total_offset] = local_value;
#endif
            }

            GroupMemoryBarrierWithGroupSync();

            // Update the cached histogram for the next set of entries
            if (local_id < SORT_BIN_COUNT)
                lds_bin_offset_cache[local_id] += lds_bin_local_histogram[local_id];

            data_index += THREADGROUP_SIZE;
        }
    }
}