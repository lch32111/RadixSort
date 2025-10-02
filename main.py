from pathlib import Path
import slangpy as spy

HERE_DIR = Path(__file__).parent

if __name__ == "__main__":
    device = spy.Device(
        enable_debug_layers=True,
        compiler_options={"include_paths": [HERE_DIR]},
    )