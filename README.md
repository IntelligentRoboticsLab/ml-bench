# ml-bench 🔬

A tool for benchmarking inference time of onnx models using the
[OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) runtime.

The benchmarks are run using [criterion](https://github.com/bheisler/criterion.rs), running a forward pass on the model
`--samples` times. Once the benchmark is complete, the results are printed to stdout.


## Usage

```sh
ml-bench <model> [--samples <num_samples>]
```

## Building

### macOS

When compiling from macOS to `x86-64-unknown-linux-gnu`, create a `.cargo/config.toml` file with the following content:

```toml
[target.x86_64-unknown-linux-gnu]
linker = "x86_64-unknown-linux-gnu-gcc"
ar = "x86_64-unknown-linux-gnu-ar"

[env]
TARGET_CC = "x86_64-unknown-linux-gnu-gcc"
TARGET_CXX = "x86_64-unknown-linux-gnu-g++"
CC = "/opt/homebrew/opt/x86_64-unknown-linux-gnu/bin/x86_64-linux-gnu-gcc"
CXX = "/opt/homebrew/opt/x86_64-unknown-linux-gnu/bin/x86_64-linux-gnu-g++"
```
