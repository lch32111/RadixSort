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
}

ParameterBlock<ScanPass> pass;

void scan_prefix(uint num_values_to_scan, uint local_id, uint group_id, uint bin_offset, uint base_index, bool add_partial_sum)
{
    uint i;
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
void scan_pass(uint3 local_id: SV_GroupThreadID, uint3 group_id: SV_GroupID)
{
    // NOTE(@chan): 
    // scan_src is the reduce_table from the count pass
    // scan_dst is the reduce_table from the count pass as well.
    // scan_pass does not use scan_scratch.
    uint base_index = ELEMENTS_PER_THREAD * THREADGROUP_SIZE * group_id.x;
    scan_prefix(pass.config_num_scan_values, local_id.x, group_id.x, 0, base_index, false);
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