# GPU-accelerated parsing
#
# Demonstrates ``loads[target="gpu"]`` and ``load[target="gpu"]`` for
# GPU parsing on machines with a CUDA / ROCm accelerator.
#
# Apple Silicon caveat
# --------------------
# The GPU pipeline relies on raw-pointer kernels that the Metal
# backend does not currently support. ``loads[target="gpu"]`` raises
# on Apple Silicon by default; recompile with
# ``-D JSON_GPU_ALLOW_APPLE_FALLBACK=1`` to opt back into a silent
# CPU fallback (useful for typechecking / dev-on-Mac). On Apple
# Silicon without the flag this example detects the platform and
# prints a guidance message instead of crashing.
#
# Performance
# -----------
# GPU parsing wins on large documents (MB+ sized). For small inputs,
# CPU parsing is faster because of kernel launch overhead and host
# <-> device data transfer.

from std.sys import has_apple_gpu_accelerator, is_defined

from json import loads, load, dumps, Value


comptime _APPLE_FALLBACK = is_defined["JSON_GPU_ALLOW_APPLE_FALLBACK"]()


def _demo() raises:
    print("1. Basic GPU parsing:")
    var json_str = '{"message": "Hello from GPU!", "count": 42}'
    var data = loads[target="gpu"](json_str)
    print("   Input:", json_str)
    print("   Parsed:", dumps(data))
    print()

    print("2. Parsing nested structures:")
    var nested_json = """{
        "users": [
            {"id": 1, "name": "Alice", "scores": [95, 87, 92]},
            {"id": 2, "name": "Bob", "scores": [88, 91, 85]},
            {"id": 3, "name": "Charlie", "scores": [90, 93, 89]}
        ],
        "metadata": {
            "total_users": 3,
            "generated_at": "2024-01-01T00:00:00Z"
        }
    }"""
    var nested_data = loads[target="gpu"](nested_json)
    print("   Parsed successfully!")
    print("   Result:", dumps(nested_data))
    print()

    print("3. GPU parsing from file:")
    with open("gpu_test.json", "w") as f:
        _ = f.write(nested_json)

    with open("gpu_test.json", "r") as f:
        var file_data = load[target="gpu"](f)
        print("   Loaded from file successfully!")
        var keys = file_data.object_keys()
        print("   Object keys:", ", ".join(keys))
    print()

    print("4. CPU vs GPU comparison:")
    var test_json = '{"x": 1, "y": 2, "z": 3}'
    var cpu_result = loads(test_json)
    var gpu_result = loads[target="gpu"](test_json)
    print("   CPU result:", dumps(cpu_result))
    print("   GPU result:", dumps(gpu_result))
    print()

    print("Note: GPU parsing excels with large JSON documents (MB+ sized).")
    print("For small inputs, CPU parsing is typically faster due to")
    print("GPU kernel launch overhead and data transfer costs.")


def main() raises:
    print("GPU-Accelerated JSON Parsing")
    print("=" * 40)
    print()

    comptime if not _APPLE_FALLBACK:
        if has_apple_gpu_accelerator():
            print(
                "Detected Apple Silicon without"
                " -D JSON_GPU_ALLOW_APPLE_FALLBACK=1."
            )
            print(
                "loads[target='gpu'] would raise here. To run this example"
                " on this Mac, recompile with:"
            )
            print(
                "    pixi run mojo -D JSON_GPU_ALLOW_APPLE_FALLBACK=1 -I ."
                " examples/advanced/gpu_parsing.mojo"
            )
            print(
                "Or use 'pixi run example-gpu' which sets the flag for you."
            )
            return

    _demo()
