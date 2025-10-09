import slangpy as spy
import numpy as np
from pathlib import Path
import RadixConstants as Const

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

config = Const.RadixDispatchConfig(NUM_KEYS)

print("Key Count: ", NUM_KEYS)
print("Num Blocks: ", config.num_blocks)
print("Num Count ThreadGroups: ", config.num_threadgroups_to_run)
print("Num Count Blocks Per ThreadGroup: ", config.num_blocks_per_threadgroup)
print("Num Count ThreadGroups with Additional Blocks: ", config.num_threadgroups_with_additional_blocks)
print("Num Reduce ThreadGroups: ", config.num_reduce_threadgroups_to_run)
print("Num Reduce ThreadGroup per Bin: ", config.num_reduce_threadgroup_per_bin)

np.random.seed(20251002)
keys = np.random.randint(0, Const.SORT_BIN_COUNT, size=NUM_KEYS, dtype=np.uint32)
sum_table = np.zeros(Const.SORT_BIN_COUNT * config.num_threadgroups_to_run, dtype=np.uint32)
reduce_table = np.zeros(Const.SORT_BIN_COUNT * config.num_reduce_threadgroup_per_bin, dtype=np.uint32)

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
    cursor["config_num_blocks_per_threadgroup"] = config.num_blocks_per_threadgroup
    cursor["config_num_thread_groups"] = config.num_threadgroups_to_run
    cursor["config_num_threadgroups_with_additional_block"] = config.num_threadgroups_with_additional_blocks
    cursor["config_num_reduce_threadgroup_per_bin"] = config.num_reduce_threadgroup_per_bin
    cursor["config_num_scan_values"] = config.num_scan_values

    cursor["keys"] = keys_buffer
    cursor["sum_table"] = sum_table_buffer
    cursor["reduce_table"] = reduce_table_buffer
    pass_encoder.dispatch_compute([config.num_threadgroups_to_run, 1, 1])

# barrier on the sum table
command_encoder.set_buffer_state(sum_table_buffer, spy.ResourceState.shader_resource)
command_encoder.set_buffer_state(reduce_table_buffer, spy.ResourceState.unordered_access)

with command_encoder.begin_compute_pass() as pass_encoder:
    shader_object = pass_encoder.bind_pipeline(count_reduce_pass_kernel.pipeline)
    cursor = spy.ShaderCursor(shader_object)["pass"]

    cursor["config_shift_bit"] = 0
    cursor["config_num_keys"] = NUM_KEYS
    cursor["config_num_blocks_per_threadgroup"] = config.num_blocks_per_threadgroup
    cursor["config_num_thread_groups"] = config.num_threadgroups_to_run
    cursor["config_num_threadgroups_with_additional_block"] = config.num_threadgroups_with_additional_blocks
    cursor["config_num_reduce_threadgroup_per_bin"] = config.num_reduce_threadgroup_per_bin
    cursor["config_num_scan_values"] = config.num_scan_values

    cursor["keys"] = keys_buffer
    cursor["sum_table"] = sum_table_buffer
    cursor["reduce_table"] = reduce_table_buffer
    pass_encoder.dispatch_compute([config.num_reduce_threadgroups_to_run, 1, 1])

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

expected = np.bincount(keys, minlength=Const.SORT_BIN_COUNT)
sum_table_result = sum_table_buffer.to_numpy().view(np.uint32)
sum_table_result = sum_table_result.reshape(-1, config.num_threadgroups_to_run).sum(axis=1, dtype=np.uint32)

print(sum_table_result)
print(expected)
assert np.array_equal(sum_table_result, expected)

reduce_table_result = reduce_table_buffer.to_numpy().view(np.uint32)
reduce_table_result = reduce_table_result.reshape(-1, config.num_reduce_threadgroup_per_bin).sum(axis=1, dtype=np.uint32)
print(reduce_table_result)
assert np.array_equal(reduce_table_result, expected)