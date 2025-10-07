// import sgl.device.print;

// https://github.com/GPUOpen-Effects/FidelityFX-ParallelSort/blob/master/ffx-parallelsort/FFX_ParallelSort.h
struct CountPass
{
    RWStructuredBuffer<uint> keys;
    RWStructuredBuffer<uint> sum_table;
    uint config_shift_bit;
    uint config_num_keys;
    int config_num_blocks_per_threadgroup;
    uint config_num_thread_groups;
    uint config_num_threadgroups_with_additional_block;
    // uint num_reduce_threadgroup_per_bin;
    // uint num_scan_values;
}

ParameterBlock<CountPass> pass;

#define SORT_BITS_PER_PASS 4
#define SORT_BIN_COUNT (1 << SORT_BITS_PER_PASS)
#define ELEMENTS_PER_THREAD 4
#define THREADGROUP_SIZE 128

// Shared memory: one count table per thread
groupshared uint histogram[THREADGROUP_SIZE * SORT_BIN_COUNT];

[numthreads(THREADGROUP_SIZE, 1, 1)]
void compute_main(uint3 local_id : SV_GroupThreadID, uint3 group_id : SV_GroupID)
{
    for (int i = 0; i < SORT_BIN_COUNT; ++i)
        histogram[(i * THREADGROUP_SIZE) + local_id.x] = 0;

    // Wait for everyone to catch up
    GroupMemoryBarrierWithGroupSync();

    // print("{}-{}: {} {} {}\n", group_id.x, local_id.x, pass.config_shift_bit, pass.config_num_keys, pass.config_num_blocks_per_threadgroup);

    int block_size = ELEMENTS_PER_THREAD * THREADGROUP_SIZE;

    uint threadgroup_block_start = (block_size * pass.config_num_blocks_per_threadgroup * group_id.x);
    uint num_blocks_to_process = pass.config_num_blocks_per_threadgroup;

    uint additional_block_thread_group_id = group_id.x - pass.config_num_thread_groups + pass.config_num_threadgroups_with_additional_block;
    if (additional_block_thread_group_id >= 0)
    {
        // this thread in this thread group need to process one more block
        threadgroup_block_start += additional_block_thread_group_id * block_size;
        ++num_blocks_to_process;
    }

    uint block_index = threadgroup_block_start + local_id.x;

    for (uint block_count = 0; block_count < num_blocks_to_process; ++block_count, block_index += block_size)
    {
        uint data_index = block_index;

        // pre-load the key values in order to hide some of the read latency
        uint src_keys[ELEMENTS_PER_THREAD];
        src_keys[0] = pass.keys[data_index];
        src_keys[1] = pass.keys[data_index + THREADGROUP_SIZE];
        src_keys[2] = pass.keys[data_index + THREADGROUP_SIZE * 2];
        src_keys[3] = pass.keys[data_index + THREADGROUP_SIZE * 3];

        for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i)
        {
            if (data_index < pass.config_num_keys)
            {
                // uint local_key = (src_keys[i] >> sort_config.shift_bit) & 0xF;
                uint local_key = src_keys[i];
                // NOTE(@chan): I guess there would be no need for InterlockedAdd here if each thread has BINs??
                InterlockedAdd(histogram[(local_key * THREADGROUP_SIZE) + local_id.x], 1);
                // histogram[(local_key * THREADGROUP_SIZE) + local_id.x] += 1;
                data_index += THREADGROUP_SIZE;
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (local_id.x < SORT_BIN_COUNT)
    {
        // NOTE(@chan): now each local_id means the local_key
        // sums up each thread count value for this local_key
        uint sum = 0;
        for (int i = 0; i < THREADGROUP_SIZE; ++i)
        {
            sum += histogram[local_id.x * THREADGROUP_SIZE + i];
        }

        // The layout would be like this
        // Bin0 [Group0, Group1, ...]
        // Bin1 [Group0, Group1, ...]
        // ...
        pass.sum_table[local_id.x * pass.config_num_thread_groups + group_id.x] = sum;
    }
}