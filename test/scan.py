import slangpy as spy
import numpy as np
import count

# scan constitutes of two passes: scan pass and scan add pass.
#
# In the scan pass,
# ScanSrc as input is the reduced_table from the count reduce pass.
# ScanDst is the reduced_table, which is same as ScanSrc here.
# This is dispatched by (1, 1, 1) thread groups with (THREADGROUP_SIZE, 1, 1) threads.
# The number of elements in reduced_table is config.num_scan_values, which is same as config.num_reduce_threadgroups_to_run.

scan_pass_program = count.device.load_program("scan.slang", ["scan_pass"])
scan_pass_kernel = count.device.create_compute_kernel(scan_pass_program)

scan_add_pass_program = count.device.load_program("scan.slang", ["scan_add_pass"])
scan_add_pass_kernel = count.device.create_compute_kernel(scan_add_pass_program)

command_encoder = count.device.create_command_encoder()
with command_encoder.begin_compute_pass() as pass_encoder:
    shader_object = pass_encoder.bind_pipeline(scan_pass_kernel.pipeline)
    cursor = spy.ShaderCursor(shader_object)["pass"]

    cursor["config_shift_bit"] = 0
    cursor["config_num_keys"] = count.NUM_KEYS
    cursor["config_num_blocks_per_threadgroup"] = count.config.num_blocks_per_threadgroup
    cursor["config_num_thread_groups"] = count.config.num_threadgroups_to_run
    cursor["config_num_threadgroups_with_additional_block"] = count.config.num_threadgroups_with_additional_blocks
    cursor["config_num_reduce_threadgroup_per_bin"] = count.config.num_reduce_threadgroup_per_bin
    cursor["config_num_scan_values"] = count.config.num_scan_values

    cursor["scan_src"] = count.reduce_table_buffer
    cursor["scan_dst"] = count.reduce_table_buffer
    cursor["scan_scratch"] = count.reduce_table_buffer # just put it for generating a bind descriptor
    pass_encoder.dispatch_compute([1, 1, 1])

    # NOTE(@chan): slangpy might track its resource usage for automatic barriers
    # command_encoder.set_buffer_state(count.reduce_table_buffer, spy.ResourceState.shader_resource)

    shader_object = pass_encoder.bind_pipeline(scan_add_pass_kernel.pipeline)
    cursor = spy.ShaderCursor(shader_object)["pass"]

    cursor["config_shift_bit"] = 0
    cursor["config_num_keys"] = count.NUM_KEYS
    cursor["config_num_blocks_per_threadgroup"] = count.config.num_blocks_per_threadgroup
    cursor["config_num_thread_groups"] = count.config.num_threadgroups_to_run
    cursor["config_num_threadgroups_with_additional_block"] = count.config.num_threadgroups_with_additional_blocks
    cursor["config_num_reduce_threadgroup_per_bin"] = count.config.num_reduce_threadgroup_per_bin
    cursor["config_num_scan_values"] = count.config.num_scan_values

    cursor["scan_src"] = count.sum_table_buffer
    cursor["scan_dst"] = count.sum_table_buffer
    cursor["scan_scratch"] = count.reduce_table_buffer
    pass_encoder.dispatch_compute([count.config.num_reduce_threadgroups_to_run, 1, 1])

id = count.device.submit_command_buffer(command_encoder.finish())
count.device.wait_for_idle()

expected_scan_reduce_table = np.concatenate((np.array([0], dtype=np.uint32), np.cumsum(count.original_reduce_table_result.flatten(), axis=0, dtype=np.uint32)[:-1]))
expected_scan_reduce_table = expected_scan_reduce_table.reshape(count.original_reduce_table_result.shape)
scan_reduce_table_result = count.reduce_table_buffer.to_numpy().view(np.uint32).reshape(-1, count.config.num_reduce_threadgroup_per_bin)
print(scan_reduce_table_result)
assert np.array_equal(scan_reduce_table_result, expected_scan_reduce_table)

expected_scan_sum_table = np.hstack([np.zeros((count.Const.SORT_BIN_COUNT, 1), dtype=np.uint32), np.cumsum(count.original_sum_table_result, axis=1, dtype=np.uint32)[:, :-1]])
expected_scan_sum_table[:] += expected_scan_reduce_table[:, 0][..., np.newaxis]
scan_add_sum_table_result = count.sum_table_buffer.to_numpy().view(np.uint32).reshape(count.Const.SORT_BIN_COUNT, -1)
print(scan_add_sum_table_result)
assert np.array_equal(scan_add_sum_table_result, expected_scan_sum_table)

