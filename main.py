from pathlib import Path
import slangpy as spy
import numpy as np

HERE_DIR = Path(__file__).parent
    
# Assume RadixSorter is used in a frame of rendering to sort multiple key buffers
class RadixSorter:
    class RadixSortTask:
        def __init__(self, radix_sorter, keys: np.ndarray, payloads: np.ndarray):
            assert keys.dtype == np.uint32 and payloads.dtype == np.uint32

            self.sorter = radix_sorter

            self.num_keys = keys.nbytes // 4

            self.block_size = self.sorter.ELEMENTS_PER_THREAD * self.sorter.THREADGROUP_SIZE
            self.num_blocks = (self.num_keys + self.block_size - 1) // self.block_size
            self.num_threadgroups_to_run = self.sorter.MAX_THREAD_GROUPS
            self.num_blocks_per_threadgroup = self.num_blocks // self.num_threadgroups_to_run
            self.num_threadgroups_with_additional_blocks = self.num_blocks % self.num_threadgroups_to_run

            if self.num_blocks < self.num_threadgroups_to_run:
                self.num_blocks_per_threadgroup = 1
                self.num_threadgroups_to_run = self.num_blocks
                self.num_threadgroups_with_additional_blocks = 0

            if self.block_size > self.num_threadgroups_to_run:
                self.num_reduce_threadgroups_to_run = self.sorter.SORT_BIN_COUNT
            else:
                self.num_reduce_threadgroups_to_run = self.sorter.SORT_BIN_COUNT * ((self.num_threadgroups_to_run + self.block_size - 1) // self.block_size)
            self.num_reduce_threadgroup_per_bin = self.num_reduce_threadgroups_to_run // self.sorter.SORT_BIN_COUNT
            self.num_scan_values = self.num_reduce_threadgroups_to_run

            assert self.num_scan_values <= self.block_size

            device = self.sorter.device
            self.requested_key = keys
            
            self.keys_buffer1 = device.create_buffer(
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=keys
            )
            self.keys_buffer2 = device.create_buffer(
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(self.num_keys, dtype=np.uint32)
            )
            if self.sorter.is_sort_with_payload:
                self.requested_payload = payloads
                self.payloads_buffer1 = device.create_buffer(
                    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                    data=payloads
                )
                self.payloads_buffer2 = device.create_buffer(
                    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                    data=np.zeros(self.num_keys, dtype=np.uint32)
                )
            self.sum_table_buffer = device.create_buffer(
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(self.sorter.SORT_BIN_COUNT * self.num_threadgroups_to_run, dtype=np.uint32)
            )
            self.sum_reduce_table_buffer = device.create_buffer(
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                data=np.zeros(self.sorter.SORT_BIN_COUNT * self.num_reduce_threadgroup_per_bin, dtype=np.uint32)
            )
           
            # Only set one validation as true, because it will exit this program.
            # I only validate keys, not payloads.
            self.DO_VALIDATION_INDEX = 7 # loop index for validation
            self.DO_VALIDATION_COUNTING = False
            self.DO_VALIDATION_SCAN = False
            self.DO_VALIDATION_SCATTER = False
            self.DO_VALIDATE_RADIX_SORT = False # validate after all the loops
            
            if self.DO_VALIDATION_SCAN or self.DO_VALIDATION_COUNTING:
                # only validate the first pass 
                self.keys_cpu = keys.to_numpy().view(np.uint32)

                self.keys_temp = self.keys_cpu.copy()
                for shift_bit in range(0, self.sorter.KEY_BIT, self.sorter.SORT_BIT_PER_PASS):
                    if shift_bit == self.DO_VALIDATION_INDEX * self.sorter.SORT_BIT_PER_PASS:
                        break

                    keys_bits = (self.keys_temp >> shift_bit) & 0xF
                    sort_indices = np.argsort(keys_bits, stable=True)
                    self.keys_temp = self.keys_temp[sort_indices]

                keys_bits = (self.keys_temp >> shift_bit) & 0xF

                group_results = []
                for group_id in range(self.num_threadgroups_to_run):
                    start_index = group_id * self.num_blocks_per_threadgroup * self.block_size

                    num_blocks_for_this_group = self.num_blocks_per_threadgroup
                    if group_id >= self.num_threadgroups_to_run - self.num_threadgroups_with_additional_blocks:
                        start_index += (group_id - (self.num_threadgroups_to_run - self.num_threadgroups_with_additional_blocks)) * self.block_size
                        num_blocks_for_this_group += 1
                    end_index = start_index +  num_blocks_for_this_group * self.block_size
                    if (end_index > self.num_keys):
                        end_index = self.num_keys

                    group_bins = np.bincount(keys_bits[start_index:end_index], minlength=self.sorter.SORT_BIN_COUNT).astype(np.uint32)
                    group_results.append(group_bins)

                self.expected_sum_table = np.stack(group_results, axis=1)

                group_results = []
                for bin_index in range(self.sorter.SORT_BIN_COUNT):
                    bin_result = []
                    for group_id in range(self.num_reduce_threadgroup_per_bin):
                        start_index = group_id * self.block_size
                        end_index = (group_id + 1) * self.block_size
                        if end_index > self.num_threadgroups_to_run:
                            end_index = self.num_threadgroups_to_run

                        one_reduce_threadgroup_result = self.expected_sum_table[bin_index, start_index:end_index].sum()
                        bin_result.append(one_reduce_threadgroup_result)
                    bin_result = np.asarray(bin_result, dtype=np.uint32)
                    group_results.append(bin_result)
                self.expected_sum_reduce_table = np.stack(group_results, axis=0)
                
            if self.DO_VALIDATION_SCAN:
                self.expected_scan_reduce_table = np.concatenate(
                    (np.array([0], dtype=np.uint32), 
                     np.cumsum(self.expected_sum_reduce_table.flatten(), axis=0, dtype=np.uint32)[:-1])
                )
                self.expected_scan_reduce_table = self.expected_scan_reduce_table.reshape(self.expected_sum_reduce_table.shape)

                self.expected_scan_add_sum_table = np.hstack([
                    np.zeros((self.sorter.SORT_BIN_COUNT, 1), dtype=np.uint32),
                    np.cumsum(self.expected_sum_table, axis=1, dtype=np.uint32)[:, :-1]
                ])
                self.expected_scan_add_sum_table[:] += self.expected_scan_reduce_table[:, 0][..., np.newaxis]
                self.expected_scan_add_sum_table = self.expected_scan_add_sum_table.reshape(-1, self.num_threadgroups_to_run)

            if self.DO_VALIDATION_SCATTER:
                self.keys_cpu = keys.to_numpy().view(np.uint32)
                
                self.keys_temp = self.keys_cpu.copy()
                for shift_bit in range(0, self.sorter.KEY_BIT, self.sorter.SORT_BIT_PER_PASS):
                    keys_bits = (self.keys_temp >> shift_bit) & 0xF
                    sort_indices = np.argsort(keys_bits, stable=True)
                    self.keys_temp = self.keys_temp[sort_indices]

                    if shift_bit == self.DO_VALIDATION_INDEX * self.sorter.SORT_BIT_PER_PASS:
                        break

                self.expected_scatter_sorted_result = self.keys_temp

            if self.DO_VALIDATE_RADIX_SORT:
                self.keys_cpu = keys.to_numpy().view(np.uint32)
                self.expected_sorted_result = np.sort(self.keys_cpu, stable=True)


        def execute(self, command_encoder: spy.CommandEncoder):
            key_buffers = [self.keys_buffer1, self.keys_buffer2]
            if self.sorter.is_sort_with_payload:
                payload_buffers = [self.payloads_buffer1, self.payloads_buffer2]
            
            src_key_index = 0
            dst_key_index = 1

            # NOTE(@chan): I comment out barrier codes that I guess slangpy manages barrier automatically.
            # But I want to keep the codes in the case of porting this code into a native graphics API.

            config_data = {
                "num_keys": self.num_keys,
                "shift_bit": 0,
                "num_blocks_per_threadgroup": self.num_blocks_per_threadgroup,
                "num_thread_groups": self.num_threadgroups_to_run,
                "num_threadgroups_with_additional_blocks": self.num_threadgroups_with_additional_blocks,
                "num_reduce_threadgroup_per_bin": self.num_reduce_threadgroup_per_bin,
                "num_scan_values": self.num_scan_values
            }

            pass_encoder = command_encoder.begin_compute_pass()
            for shift_bit in range(0, self.sorter.KEY_BIT, self.sorter.SORT_BIT_PER_PASS):
                # config data update
                config_data["shift_bit"] = shift_bit
                
                count_pass_shader_object = pass_encoder.bind_pipeline(self.sorter.count_pass_pipeline)
                count_pass_cursor = spy.ShaderCursor(count_pass_shader_object)["g_sort"]
                count_pass_cursor["src_data"] = key_buffers[src_key_index]
                count_pass_cursor["dst_data"] = key_buffers[dst_key_index]
                count_pass_cursor["sum_table"] = self.sum_table_buffer
                count_pass_cursor["sum_reduce_table"] = self.sum_reduce_table_buffer
                if self.sorter.is_sort_with_payload:
                    count_pass_cursor["src_payload"] = payload_buffers[src_key_index]
                    count_pass_cursor["dst_payload"] = payload_buffers[dst_key_index]
                count_pass_cursor["config"] = config_data
                pass_encoder.dispatch_compute([self.num_threadgroups_to_run, 1, 1])

                # command_encoder.set_buffer_state(self.sum_table_buffer, spy.ResourceState.shader_resource)

                count_reduce_pass_shader_object = pass_encoder.bind_pipeline(self.sorter.count_reduce_pass_pipeline)
                count_reduce_pass_cursor = spy.ShaderCursor(count_reduce_pass_shader_object)["g_sort"]
                count_reduce_pass_cursor["src_data"] = key_buffers[src_key_index]
                count_reduce_pass_cursor["dst_data"] = key_buffers[dst_key_index]
                count_reduce_pass_cursor["sum_table"] = self.sum_table_buffer
                count_reduce_pass_cursor["sum_reduce_table"] = self.sum_reduce_table_buffer
                if self.sorter.is_sort_with_payload:
                    count_reduce_pass_cursor["src_payload"] = payload_buffers[src_key_index]
                    count_reduce_pass_cursor["dst_payload"] = payload_buffers[dst_key_index]
                count_reduce_pass_cursor["config"] = config_data
                pass_encoder.dispatch_compute([self.num_reduce_threadgroups_to_run, 1, 1])

                if self.DO_VALIDATION_COUNTING and\
                   self.DO_VALIDATION_INDEX * self.sorter.SORT_BIT_PER_PASS == shift_bit:
                    pass_encoder.end()
                    device = self.sorter.device
                    device.submit_command_buffer(command_encoder.finish())
                    device.wait_for_idle()
                    # device.flush_print()

                    sum_table_result = self.sum_table_buffer.to_numpy().view(np.uint32)
                    sum_table_result = sum_table_result.reshape(-1, self.num_threadgroups_to_run)
                    reduce_table_result = self.sum_reduce_table_buffer.to_numpy().view(np.uint32)
                    reduce_table_result = reduce_table_result.reshape(-1, self.num_reduce_threadgroup_per_bin)
                    assert np.array_equal(self.expected_sum_table, sum_table_result)
                    assert np.array_equal(self.expected_sum_reduce_table, reduce_table_result)
                    exit(0)

                scan_pass_shader_object = pass_encoder.bind_pipeline(self.sorter.scan_pass_pipeline)
                scan_pass_cursor = spy.ShaderCursor(scan_pass_shader_object)["g_sort"]
                scan_pass_cursor["src_data"] = key_buffers[src_key_index]
                scan_pass_cursor["dst_data"] = key_buffers[dst_key_index]
                scan_pass_cursor["sum_table"] = self.sum_table_buffer
                scan_pass_cursor["sum_reduce_table"] = self.sum_reduce_table_buffer
                if self.sorter.is_sort_with_payload:
                    scan_pass_cursor["src_payload"] = payload_buffers[src_key_index]
                    scan_pass_cursor["dst_payload"] = payload_buffers[dst_key_index]
                scan_pass_cursor["config"] = config_data
                pass_encoder.dispatch_compute([1, 1, 1])

                # command_encoder.set_buffer_state(count.reduce_table_buffer, spy.ResourceState.shader_resource)
                scan_add_pass_shader_object = pass_encoder.bind_pipeline(self.sorter.scan_add_pass_pipeline)
                scan_add_pass_cursor = spy.ShaderCursor(scan_add_pass_shader_object)["g_sort"]
                scan_add_pass_cursor["src_data"] = key_buffers[src_key_index]
                scan_add_pass_cursor["dst_data"] = key_buffers[dst_key_index]
                scan_add_pass_cursor["sum_table"] = self.sum_table_buffer
                scan_add_pass_cursor["sum_reduce_table"] = self.sum_reduce_table_buffer
                if self.sorter.is_sort_with_payload:
                    scan_add_pass_cursor["src_payload"] = payload_buffers[src_key_index]
                    scan_add_pass_cursor["dst_payload"] = payload_buffers[dst_key_index]
                scan_add_pass_cursor["config"] = config_data
                pass_encoder.dispatch_compute([self.num_reduce_threadgroups_to_run, 1, 1])

                if self.DO_VALIDATION_SCAN and\
                   self.DO_VALIDATION_INDEX * self.sorter.SORT_BIT_PER_PASS == shift_bit:
                    pass_encoder.end()
                    device = self.sorter.device
                    device.submit_command_buffer(command_encoder.finish())
                    device.wait_for_idle()

                    scan_reduce_table_result = self.sum_reduce_table_buffer.to_numpy().view(np.uint32)
                    scan_reduce_table_result = scan_reduce_table_result.reshape(-1, self.num_reduce_threadgroup_per_bin)
                    scan_add_sum_table_result = self.sum_table_buffer.to_numpy().view(np.uint32)
                    scan_add_sum_table_result = scan_add_sum_table_result.reshape(-1, self.num_threadgroups_to_run)
                    assert np.array_equal(self.expected_scan_reduce_table, scan_reduce_table_result)
                    assert np.array_equal(self.expected_scan_add_sum_table, scan_add_sum_table_result)
                    exit(0)
                
                scatter_pass_shader_object = pass_encoder.bind_pipeline(self.sorter.scatter_pass_pipeline)
                scatter_pass_cursor = spy.ShaderCursor(scatter_pass_shader_object)["g_sort"]
                scatter_pass_cursor["src_data"] = key_buffers[src_key_index]
                scatter_pass_cursor["dst_data"] = key_buffers[dst_key_index]
                scatter_pass_cursor["sum_table"] = self.sum_table_buffer
                scatter_pass_cursor["sum_reduce_table"] = self.sum_reduce_table_buffer
                if self.sorter.is_sort_with_payload:
                    scatter_pass_cursor["src_payload"] = payload_buffers[src_key_index]
                    scatter_pass_cursor["dst_payload"] = payload_buffers[dst_key_index]
                scatter_pass_cursor["config"] = config_data
                pass_encoder.dispatch_compute([self.num_threadgroups_to_run, 1, 1])

                if self.DO_VALIDATION_SCATTER and\
                   self.DO_VALIDATION_INDEX * self.sorter.SORT_BIT_PER_PASS == shift_bit:
                    pass_encoder.end()
                    device = self.sorter.device
                    device.submit_command_buffer(command_encoder.finish())
                    device.wait_for_idle()

                    sorted_result = key_buffers[dst_key_index].to_numpy().view(np.uint32)
                    assert np.array_equal(self.expected_scatter_sorted_result, sorted_result)
                    exit(0)

                temp_key_index = src_key_index
                src_key_index = dst_key_index
                dst_key_index = temp_key_index

            pass_encoder.end()
            if self.DO_VALIDATE_RADIX_SORT:
                device = self.sorter.device
                device.submit_command_buffer(command_encoder.finish())
                device.wait_for_idle()

                final_sorted_key_index = src_key_index
                dst_buffer = key_buffers[final_sorted_key_index]
                sorted_result = dst_buffer.to_numpy().view(np.uint32)
                assert np.array_equal(self.expected_sorted_result, sorted_result)
                exit(0)

        def get_result_buffers(self):
            result = [self.keys_buffer1]
            if self.sorter.is_sort_with_payload:
                result.append(self.payloads_buffer1)
            return result
        
    def __init__(self, device : spy.Device, is_sort_with_payload: bool):
        self.device = device
        self.is_sort_with_payload = is_sort_with_payload

        self.KEY_BIT = 32 # uint32
        self.SORT_BIT_PER_PASS = 4
        self.SORT_BIN_COUNT = (1 << self.SORT_BIT_PER_PASS)
        self.ELEMENTS_PER_THREAD = 4
        self.THREADGROUP_SIZE = 128 # need to be tuned according to the device capability
        self.MAX_THREAD_GROUPS = 800 # need to be tuned according to the device capability

        radix_sort_module = device.load_module("radix.slang")
        self.count_pass_pipeline = device.create_compute_pipeline(
            device.link_program([radix_sort_module], [radix_sort_module.entry_point("count_pass")])
        )
        self.count_reduce_pass_pipeline = device.create_compute_pipeline(
            device.link_program([radix_sort_module], [radix_sort_module.entry_point("count_reduce_pass")])
        )
        self.scan_pass_pipeline = device.create_compute_pipeline(
            device.link_program([radix_sort_module], [radix_sort_module.entry_point("scan_pass")])
        )
        self.scan_add_pass_pipeline = device.create_compute_pipeline(
            device.link_program([radix_sort_module], [radix_sort_module.entry_point("scan_add_pass")])
        )
        self.scatter_pass_pipeline = device.create_compute_pipeline(
            device.link_program([radix_sort_module], [radix_sort_module.entry_point("scatter_pass")])
        )

        self.requested_sorts = {}

    # setup relevant data
    def begin_frame(self, command_encoder: spy.CommandEncoder):
        for key, sort_task in self.requested_sorts.items():
            sort_task.execute(command_encoder)


    # call for all requests of sort_keys...
    def end_frame(self, command_encoder: spy.CommandEncoder):
        result_buffer_dict = {}
        for key, sort_task in self.requested_sorts.items():
            result_buffer_dict[key] = sort_task.get_result_buffers()

        del self.requested_sorts
        self.request_sort = { }

        return result_buffer_dict

    # RadixSorter currently supports uint32_t type for keys. (it could be float later)
    def request_sort_keys(self, label, keys : np.ndarray, payloads : np.ndarray):
        self.requested_sorts[label] = RadixSorter.RadixSortTask(self, keys, payloads)


if __name__ == "__main__":
    np.random.seed(20251101)
    # np.random.seed(20251002)

    DO_SORT_WITH_PAYLOAD = True
    DO_SORT_WITH_WAVEFREE = True
    RADIXSORT_PAYLOAD_MACRO = "RADIXSORT_PAYLOAD" if DO_SORT_WITH_PAYLOAD else ""
    RADIXSORT_WAVEFREE_MACRO = "RADIXSORT_WAVEFREE" if DO_SORT_WITH_WAVEFREE else ""

    device = spy.Device(
        enable_debug_layers=True,
        # enable_print=True,
        compiler_options={
            "include_paths": [HERE_DIR],
            "defines" : {
                RADIXSORT_PAYLOAD_MACRO : "1",
                RADIXSORT_WAVEFREE_MACRO : "1",
            }
        },
    )
    command_encoder = device.create_command_encoder()

    sorter = RadixSorter(device, DO_SORT_WITH_PAYLOAD)

    TEST_COUNT = 10
    # NUM_KEYS = 32
    # NUM_KEYS = 1024 + 317
    # NUM_KEYS = 512 * 512
    NUM_KEYS = (1 << 20) + 1179
    # NUM_KEYS = (1 << 25) + 1337

    answer_dict = {}
    for test_i in range(TEST_COUNT):
        keys = np.random.randint(0, 0xFFFFFFFF, size=NUM_KEYS, dtype=np.uint32)

        if DO_SORT_WITH_PAYLOAD:
            payloads = np.random.randint(0, 0xFFFFFFFF, size=NUM_KEYS, dtype=np.uint32)
        else:
            payloads = None

        test_name = f"test_{test_i}"
        
        sorter.request_sort_keys(test_name, keys, payloads)
        
        keys_sort_indices = np.argsort(keys, stable=True)
        answer_dict[test_name] = [keys[keys_sort_indices]]
        if DO_SORT_WITH_PAYLOAD:
            answer_dict[test_name].append(payloads[keys_sort_indices])

    sorter.begin_frame(command_encoder)
    results = sorter.end_frame(command_encoder)

    device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()

    answer_results = ""
    for test_name, result_buffer_gpus in results.items():
        key_result_buffer_cpu = result_buffer_gpus[0].to_numpy().view(np.uint32)
        key_answer_buffer = answer_dict[test_name][0]
        answer_results += f'{test_name} Key Result: {np.array_equal(key_result_buffer_cpu, key_answer_buffer)}\n'
        if DO_SORT_WITH_PAYLOAD:
            payload_result_buffer_cpu = result_buffer_gpus[1].to_numpy().view(np.uint32)
            payload_answer_buffer = answer_dict[test_name][1]
            answer_results += f'{test_name} Payload Result: {np.array_equal(payload_result_buffer_cpu, payload_answer_buffer)}\n'
    
    del command_encoder
    del sorter
    del device
    print(answer_results)
    

    