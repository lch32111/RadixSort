import slangpy as spy
import numpy as np
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent

device = spy.Device(
    enable_debug_layers=True,
    compiler_options={"include_paths": [EXAMPLE_DIR]},
)

program = device.load_program("count.slang", ["main"])
kernel = device.create_compute_kernel(program)

NUM_KEYS = 512
NUM_BINS = 256
THREADS_PER_GROUP = 16

np.random.seed(20251002)
keys = np.random.randint(0, 256, size=NUM_KEYS, dtype=np.uint32)
global_counts = np.zeros(NUM_BINS, dtype=np.uint32)


keys_buffer = device.create_buffer(
    usage=spy.BufferUsage.shader_resource,
    data=keys
)

global_counts_buffer = device.create_buffer(
    usage=spy.BufferUsage.unordered_access,
    data=global_counts
)
command_encoder = device.create_command_encoder()
with command_encoder.begin_compute_pass() as pass_encoder:
    shader_object = pass_encoder.bind_pipeline(kernel.pipeline)
    cursor = spy.ShaderCursor(shader_object)
    cursor["keys"] = keys_buffer
    cursor["global_counts"] = global_counts_buffer
    num_groups = (NUM_KEYS + THREADS_PER_GROUP - 1) // THREADS_PER_GROUP
    pass_encoder.dispatch([num_groups, 1, 1])
id = device.submit_command_buffer(command_encoder.finish())
device.wait_for_submit(id)

expected = np.bincount(keys, minlength=NUM_BINS)
result = global_counts_buffer.to_numpy().view(np.uint32)
print(result)
print(expected)
assert np.array_equal(result, expected)