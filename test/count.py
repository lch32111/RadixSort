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

program = device.load_program("count.slang", ["compute_main"])
kernel = device.create_compute_kernel(program)

NUM_KEYS = 1 << 16
MAX_THREAD_GROUPS = 800

block_size = ELEMENTS_PER_THREAD * THREADGROUP_SIZE
num_blocks = (NUM_KEYS + block_size - 1) // block_size
num_threadgroups_to_run = MAX_THREAD_GROUPS
blocks_per_threadgroup = num_blocks // num_threadgroups_to_run
num_threadgroups_with_additional_blocks = num_blocks % num_threadgroups_to_run

if num_blocks < num_threadgroups_to_run:
    blocks_per_threadgroup = 1
    num_threadgroups_to_run = num_blocks
    num_threadgroups_with_additional_blocks = 0

np.random.seed(20251002)
keys = np.random.randint(0, SORT_BIN_COUNT, size=NUM_KEYS, dtype=np.uint32)
sum_table = np.zeros(SORT_BIN_COUNT * num_threadgroups_to_run, dtype=np.uint32)

keys_buffer = device.create_buffer(
    usage=spy.BufferUsage.unordered_access,
    data=keys
)

sum_table_buffer = device.create_buffer(
    usage=spy.BufferUsage.unordered_access,
    data=sum_table
)
command_encoder = device.create_command_encoder()
with command_encoder.begin_compute_pass() as pass_encoder:
    shader_object = pass_encoder.bind_pipeline(kernel.pipeline)
    cursor = spy.ShaderCursor(shader_object)["pass"]

    cursor["config_shift_bit"] = 0
    cursor["config_num_keys"] = NUM_KEYS
    cursor["config_num_blocks_per_threadgroup"] = blocks_per_threadgroup
    cursor["config_num_thread_groups"] = num_threadgroups_to_run
    cursor["config_num_threadgroups_with_additional_block"] = num_threadgroups_with_additional_blocks

    cursor["keys"] = keys_buffer
    cursor["sum_table"] = sum_table_buffer

    pass_encoder.dispatch_compute([num_threadgroups_to_run, 1, 1])
id = device.submit_command_buffer(command_encoder.finish())
device.wait_for_submit(id)

expected = np.bincount(keys, minlength=SORT_BIN_COUNT)
result = sum_table_buffer.to_numpy().view(np.uint32)
result = result.reshape(-1, num_threadgroups_to_run).sum(axis=1, dtype=np.uint32)

print(result)
print(expected)
assert np.array_equal(result, expected)