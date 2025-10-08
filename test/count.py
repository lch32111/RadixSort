import slangpy as spy
import numpy as np
from pathlib import Path

SORT_BITS_PER_PASS = 4
SORT_BIN_COUNT = (1 << SORT_BITS_PER_PASS)
ELEMENTS_PER_THREAD = 4
THREADGROUP_SIZE = 128

EXAMPLE_DIR = Path(__file__).parent

device = spy.Device(
    enable_debug_layers=True,
    # enable_print=True,
    compiler_options={"include_paths": [EXAMPLE_DIR]},
)

count_program = device.load_program("count.slang", ["count_pass"])
count_reduce_program = device.load_program("count.slang", ["count_reduce_pass"])
count_pass_kernel = device.create_compute_kernel(count_program)
count_reduce_pass_kernel = device.create_compute_kernel(count_reduce_program)

NUM_KEYS = (1 << 20) + 1337
MAX_THREAD_GROUPS = 800

block_size = ELEMENTS_PER_THREAD * THREADGROUP_SIZE
num_blocks = (NUM_KEYS + block_size - 1) // block_size
num_threadgroups_to_run = MAX_THREAD_GROUPS
num_blocks_per_threadgroup = num_blocks // num_threadgroups_to_run
num_threadgroups_with_additional_blocks = num_blocks % num_threadgroups_to_run

if num_blocks < num_threadgroups_to_run:
    num_blocks_per_threadgroup = 1
    num_threadgroups_to_run = num_blocks
    num_threadgroups_with_additional_blocks = 0

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
# However we don't get a global count yet, after this reduce pass, we will get on this example.
# Bin0: [RGroup0, RGroup1]
if block_size > num_threadgroups_to_run:
    num_reduce_threadgroups_to_run = SORT_BIN_COUNT
else:
    num_reduce_threadgroups_to_run = SORT_BIN_COUNT * (num_threadgroups_to_run + block_size - 1) // block_size
num_reduce_threadgroup_per_bin = num_reduce_threadgroups_to_run // SORT_BIN_COUNT
num_scan_values = num_reduce_threadgroups_to_run

print("Key Count: ", NUM_KEYS)
print("Num Blocks: ", num_blocks)
print("Num Count ThreadGroups: ", num_threadgroups_to_run)
print("Num Count Blocks Per ThreadGroup: ", num_blocks_per_threadgroup)
print("Num Count ThreadGroups with Additional Blocks: ", num_threadgroups_with_additional_blocks)
print("Num Reduce ThreadGroups: ", num_reduce_threadgroups_to_run)
print("Num Reduce ThreadGroup per Bin: ", num_reduce_threadgroup_per_bin)

np.random.seed(20251002)
keys = np.random.randint(0, SORT_BIN_COUNT, size=NUM_KEYS, dtype=np.uint32)
sum_table = np.zeros(SORT_BIN_COUNT * num_threadgroups_to_run, dtype=np.uint32)
reduce_table = np.zeros(SORT_BIN_COUNT * num_reduce_threadgroup_per_bin, dtype=np.uint32)

keys_buffer = device.create_buffer(
    usage=spy.BufferUsage.unordered_access,
    data=keys
)

sum_table_buffer = device.create_buffer(
    usage=spy.BufferUsage.unordered_access,
    data=sum_table
)
reduce_table_buffer = device.create_buffer(
    usage=spy.BufferUsage.unordered_access,
    data=reduce_table
)

HAS_TIMEQUERY_FEATURE = device.has_feature(spy.Feature.timestamp_query)
if HAS_TIMEQUERY_FEATURE:
    query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=2)

command_encoder = device.create_command_encoder()
if HAS_TIMEQUERY_FEATURE:
    command_encoder.write_timestamp(query_pool, 0)

cpu_timer = spy.Timer()

command_encoder.set_buffer_state(keys_buffer, spy.ResourceState.shader_resource)
command_encoder.set_buffer_state(sum_table_buffer, spy.ResourceState.unordered_access)
command_encoder.set_buffer_state(reduce_table_buffer, spy.ResourceState.unordered_access)
with command_encoder.begin_compute_pass() as pass_encoder:
    shader_object = pass_encoder.bind_pipeline(count_pass_kernel.pipeline)
    cursor = spy.ShaderCursor(shader_object)["pass"]

    cursor["config_shift_bit"] = 0
    cursor["config_num_keys"] = NUM_KEYS
    cursor["config_num_blocks_per_threadgroup"] = num_blocks_per_threadgroup
    cursor["config_num_thread_groups"] = num_threadgroups_to_run
    cursor["config_num_threadgroups_with_additional_block"] = num_threadgroups_with_additional_blocks
    cursor["config_num_reduce_threadgroup_per_bin"] = num_reduce_threadgroup_per_bin
    cursor["config_num_scan_values"] = num_scan_values

    cursor["keys"] = keys_buffer
    cursor["sum_table"] = sum_table_buffer
    cursor["reduce_table"] = reduce_table_buffer
    pass_encoder.dispatch_compute([num_threadgroups_to_run, 1, 1])

# barrier on the sum table
command_encoder.set_buffer_state(sum_table_buffer, spy.ResourceState.shader_resource)
command_encoder.set_buffer_state(reduce_table_buffer, spy.ResourceState.unordered_access)

with command_encoder.begin_compute_pass() as pass_encoder:
    shader_object = pass_encoder.bind_pipeline(count_reduce_pass_kernel.pipeline)
    cursor = spy.ShaderCursor(shader_object)["pass"]

    cursor["config_shift_bit"] = 0
    cursor["config_num_keys"] = NUM_KEYS
    cursor["config_num_blocks_per_threadgroup"] = num_blocks_per_threadgroup
    cursor["config_num_thread_groups"] = num_threadgroups_to_run
    cursor["config_num_threadgroups_with_additional_block"] = num_threadgroups_with_additional_blocks
    cursor["config_num_reduce_threadgroup_per_bin"] = num_reduce_threadgroup_per_bin
    cursor["config_num_scan_values"] = num_scan_values

    cursor["keys"] = keys_buffer
    cursor["sum_table"] = sum_table_buffer
    cursor["reduce_table"] = reduce_table_buffer
    pass_encoder.dispatch_compute([num_reduce_threadgroups_to_run, 1, 1])

if HAS_TIMEQUERY_FEATURE:
    command_encoder.write_timestamp(query_pool, 1)
id = device.submit_command_buffer(command_encoder.finish())
device.wait_for_idle()

cpu_time_elapsed = cpu_timer.elapsed_ms()
print(f"{cpu_time_elapsed} ms elapsed on cpu timer")

if HAS_TIMEQUERY_FEATURE:
    timers = np.array(query_pool.get_timestamp_results(0, 2))
    timers /= device.info.timestamp_frequency
    diff = timers[1] - timers[0]
    print(f"{diff} seconds elapsed on gpu query")

expected = np.bincount(keys, minlength=SORT_BIN_COUNT)
sum_table_result = sum_table_buffer.to_numpy().view(np.uint32)
sum_table_result = sum_table_result.reshape(-1, num_threadgroups_to_run).sum(axis=1, dtype=np.uint32)

print(sum_table_result)
print(expected)
assert np.array_equal(sum_table_result, expected)

reduce_table_result = reduce_table_buffer.to_numpy().view(np.uint32)
reduce_table_result = reduce_table_result.reshape(-1, num_reduce_threadgroup_per_bin).sum(axis=1, dtype=np.uint32)
print(reduce_table_result)
assert np.array_equal(reduce_table_result, expected)