SORT_BITS_PER_PASS = 4
SORT_BIN_COUNT = (1 << SORT_BITS_PER_PASS)
ELEMENTS_PER_THREAD = 4
THREADGROUP_SIZE = 128 # need to be tuned according to the device capability
MAX_THREAD_GROUPS = 800 # need to be tuned according to the device capability

class RadixDispatchConfig:
    def __init__(self, num_keys):
        self.block_size = ELEMENTS_PER_THREAD * THREADGROUP_SIZE
        self.num_blocks = (num_keys + self.block_size - 1) // self.block_size
        self.num_threadgroups_to_run = MAX_THREAD_GROUPS
        self.num_blocks_per_threadgroup = self.num_blocks // self.num_threadgroups_to_run
        self.num_threadgroups_with_additional_blocks = self.num_blocks % self.num_threadgroups_to_run

        if self.num_blocks < self.num_threadgroups_to_run:
            self.num_blocks_per_threadgroup = 1
            self.num_threadgroups_to_run = self.num_blocks
            self.num_threadgroups_with_additional_blocks = 0

        # NOTE(@chan):
        # After the count pass, the sum_table layout would be like this
        # SumTable
        # Bin0: [Group0, Group1, Group2, ...]
        # Bin1: [Group0, Group1, Group2, ...]
        # In the count reduce pass, we want to sum up the elements in each bin.
        # As block_size is the number of elements that one thread group can handle,
        # If each bin0 contains counts from groups more than block_size,
        # Then we use more than one threadgroup to sum for each bin.
        # For example, If Bin0 has 1024 entries and block_size 512, then we need to use 2 threadgroups to sum it.
        # However we don't get a global count yet, after this reduce pass, we will get on this example:.
        # Bin0: [RGroup0, RGroup1]
        # Bin1: [RGroup0, RGroup1]
        # ...
        # This will be considered in scan pass later.
        # The total elements of the reduced_table is same as self.num_reduce_threadgroups_to_run
        if self.block_size > self.num_threadgroups_to_run:
            self.num_reduce_threadgroups_to_run = SORT_BIN_COUNT
        else:
            self.num_reduce_threadgroups_to_run = SORT_BIN_COUNT * ((self.num_threadgroups_to_run + self.block_size - 1) // self.block_size)
        self.num_reduce_threadgroup_per_bin = self.num_reduce_threadgroups_to_run // SORT_BIN_COUNT
        self.num_scan_values = self.num_reduce_threadgroups_to_run

        # Scan Pass on the reduced_table will be dispatched with (1, 1, 1) thread groups and (THREADGROUP_SIZE, 1, 1) threads.
        # Each thread handle ELEMENTS_PER_THREAD values.
        # Therefore, num_scan_values should not exceed block_size.
        assert self.num_scan_values <= self.block_size