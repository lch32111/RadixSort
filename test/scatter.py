import slangpy as spy
import numpy as np
import scan

config = scan.count.config
device = scan.count.device

scatter_pass_program = device.load_program("scatter.slang", ["scatter_pass"])
scatter_pass_kernel = device.create_compute_kernel(scatter_pass_program)

out_buffer = device.create_buffer(
    usage=spy.BufferUsage.unordered_access,
    data=np.zeros((scan.count.NUM_KEYS), dtype=np.uint32)
)

command_encoder = device.create_command_encoder()
with command_encoder.begin_compute_pass() as pass_encoder:
    shader_object = pass_encoder.bind_pipeline(scatter_pass_kernel.pipeline)
    cursor = spy.ShaderCursor(shader_object)["pass"]

    cursor["config_shift_bit"] = 0
    cursor["config_num_keys"] = scan.count.NUM_KEYS
    cursor["config_num_blocks_per_threadgroup"] = config.num_blocks_per_threadgroup
    cursor["config_num_thread_groups"] = config.num_threadgroups_to_run
    cursor["config_num_threadgroups_with_additional_block"] = config.num_threadgroups_with_additional_blocks
    cursor["config_num_reduce_threadgroup_per_bin"] = config.num_reduce_threadgroup_per_bin
    cursor["config_num_scan_values"] = config.num_scan_values

    cursor["src_buffer"] = scan.count.keys_buffer
    cursor["dst_buffer"] = out_buffer
    cursor["sum_table"] = scan.count.sum_table_buffer

    pass_encoder.dispatch_compute([config.num_threadgroups_to_run, 1, 1])

id = device.submit_command_buffer(command_encoder.finish())
device.wait_for_idle()

sorted_keys = np.sort(scan.count.keys)
out_buffer_result = out_buffer.to_numpy().view(np.uint32)
print(out_buffer_result)
assert np.array_equal(out_buffer_result, sorted_keys)
