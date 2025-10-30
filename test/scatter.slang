// import sgl.device.print;

#define SORT_BITS_PER_PASS 4
#define SORT_BIN_COUNT (1 << SORT_BITS_PER_PASS)
#define ELEMENTS_PER_THREAD 4
#define THREADGROUP_SIZE 128 // NOTE(@chan): THREADGGROUP_SIZE should be bigger than SORT_BIN_COUNT -> check local_histogram

struct ScatterPass
{
    RWStructuredBuffer<uint> src_buffer;
    RWStructuredBuffer<uint> dst_buffer;
    RWStructuredBuffer<uint> sum_table;
    uint config_shift_bit;
    uint config_num_keys;
    int config_num_blocks_per_threadgroup;
    uint config_num_thread_groups;
    uint config_num_threadgroups_with_additional_block;
    uint config_num_reduce_threadgroup_per_bin;
    uint config_num_scan_values;
};

ParameterBlock<ScatterPass> pass;

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

groupshared uint bin_offset_cache[THREADGROUP_SIZE];
groupshared uint local_histogram[SORT_BIN_COUNT];
groupshared uint lds_scratch[THREADGROUP_SIZE];
[numthreads(THREADGROUP_SIZE, 1, 1)]
void scatter_pass(uint3 local_id : SV_GroupThreadID, uint3 group_id : SV_GroupID)
{
    // Load the sort bin threadgroup offsets into LDS for faster referencing
    if (local_id.x < SORT_BIN_COUNT)
    {
        // NOTE(@chan): currently the sum table has a prescan result of each bin (sum_table[SORT_BIN_COUNT][NUM_THREADGROUPS_TO_RUN])
        // if local_id.x is 0 across thread groups, 
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
        bin_offset_cache[local_id.x] = pass.sum_table[local_id.x * pass.config_num_thread_groups + group_id.x];
    }

    GroupMemoryBarrierWithGroupSync();

    uint block_size = ELEMENTS_PER_THREAD * THREADGROUP_SIZE;

    uint threadgroup_block_start = (block_size * pass.config_num_blocks_per_threadgroup * group_id.x);
    uint num_blocks_to_process = pass.config_num_blocks_per_threadgroup;

    if (group_id.x >= pass.config_num_thread_groups - pass.config_num_threadgroups_with_additional_block)
    {
        threadgroup_block_start += (group_id.x - (pass.config_num_thread_groups - pass.config_num_threadgroups_with_additional_block)) * block_size;
        num_blocks_to_process += 1;
    }

    // Get the block start index for this thread
    uint block_index = threadgroup_block_start + local_id.x;

    // Count value occurences
    for (uint block_count = 0; block_count < num_blocks_to_process; ++block_count, block_index += block_size)
    {
        uint data_index = block_index;

        // Pre-load the key values in order to hide some of the read latency
        uint src_keys[ELEMENTS_PER_THREAD];
        src_keys[0] = (data_index < pass.config_num_keys) ? pass.src_buffer[data_index] : 0xFFFFFFFFu;
        data_index += THREADGROUP_SIZE;

        src_keys[1] = (data_index < pass.config_num_keys) ? pass.src_buffer[data_index] : 0xFFFFFFFFu;
        data_index += THREADGROUP_SIZE;

        src_keys[2] = (data_index < pass.config_num_keys) ? pass.src_buffer[data_index] : 0xFFFFFFFFu;
        data_index += THREADGROUP_SIZE;

        src_keys[3] = (data_index < pass.config_num_keys) ? pass.src_buffer[data_index] : 0xFFFFFFFFu;

        data_index = block_index;
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
        {
            // clear the local histogram
            if (local_id.x < SORT_BIN_COUNT)
                local_histogram[local_id.x] = 0;

            uint local_key = src_keys[i];

            // Sort the keys locally in LDS
            for (uint bit_shift = 0; bit_shift < SORT_BITS_PER_PASS; bit_shift += 2)
            {
                // Figure out the key index
                // uint key_index = (local_key >> pass.config_shift_bit) & 0xFu;
                uint key_index = local_key;
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
                if (local_id.x == (THREADGROUP_SIZE - 1))
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
                local_key = lds_sum[local_id.x];

                GroupMemoryBarrierWithGroupSync();
            }

            // Need to recalculate the key_index on this thread now that values have been copied around the thread group
            // uint key_index = (local_key >> pass.config_shift_bit) & 0xFu;
            // NOTE(@chan): finally we get a locally sorted key in a thread group
            uint key_index = local_key;

            // Reconstruct histogram
            InterlockedAdd(local_histogram[key_index], 1);

            GroupMemoryBarrierWithGroupSync();

            // Prefix histogram
            uint histogram_prefix_sum = WavePrefixSum(local_id.x < SORT_BIN_COUNT ? local_histogram[local_id.x] : 0);

            // Broadcast prefix-sum via LDS
            if (local_id.x < SORT_BIN_COUNT)
                lds_scratch[local_id.x] = histogram_prefix_sum;

            // Get the global offset for this key out of the cache
            uint global_offset = bin_offset_cache[key_index];

            GroupMemoryBarrierWithGroupSync();

            // Get the local offset (at this point the keys are all in increasing order from 0 -> num bins in localID 0 -> thread group size)
            uint local_offset = local_id.x - lds_scratch[key_index];

            // Write to detination
            uint total_offset = global_offset + local_offset;
            if (data_index < pass.config_num_keys && total_offset < pass.config_num_keys)
            {
                pass.dst_buffer[total_offset] = local_key;
            }

            GroupMemoryBarrierWithGroupSync();

            // Update the cached histogram for the next set of entries
            if (local_id.x < SORT_BIN_COUNT)
                bin_offset_cache[local_id.x] += local_histogram[local_id.x];

            data_index += THREADGROUP_SIZE;
        }
    }
}