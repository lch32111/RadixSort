from pathlib import Path
import slangpy as spy

HERE_DIR = Path(__file__).parent

if __name__ == "__main__":
    device = spy.Device(
        enable_debug_layers=True,
        compiler_options={"include_paths": [HERE_DIR]},
    )

    count_pass_kernel = device.create_compute_kernel(
        device.load_program("radix.slang", ["count_pass"])
    )
    count_reduce_pass_kernel = device.create_compute_kernel(
        device.load_program("radix.slang", ["count_reduce_pass"])
    )
    print(count_pass_kernel)
    print(count_reduce_pass_kernel)