SubCuber
----------
SubCuber is a compiler for Strassen-like algorithms to fast CUDA code.
This repository contains the source code, CUDA kernels, tests, examples, and benchmark evaluation scripts.

Requirements
------------
The CUDA build expects a CUDA toolkit installation and a C++20-capable host compiler.
By default, the Makefiles use:

```bash
CUDA_HOME=/usr/local/cuda
NVCC=/usr/local/cuda/bin/nvcc
CXX=g++
```

You can override these on the command line, for example:

```bash
make CUDA_HOME=/path/to/cuda CXX=/path/to/g++
```


Submodules
----------------------

Update git submodules and apply CUTLASS patch

```bash
git submodule update --recursive
git apply --directory cutlass/ cutlass.patch
```


Build `kernel_runner`
---------------------
From the repository root, run:

```bash
make
```

This builds all registered runner objects and writes outputs under root-level `build/`:

```text
build/
|-- kernel_runner
`-- obj/kernel_runner/
```

Useful build variants:

```bash
# Build the default kernel runner
make all

# Build the runner without CUDA declarations enabled in the runner objects
make no_cuda_declarations

# Remove root-level kernel_runner build artifacts
make clean
```

The build can be tuned with Make variables:

```bash
make SPLIT_COMPILE=8
make PRESUM_LEVEL_2_SPLIT_COMPILE=16
make CUDA_HOME=/usr/local/cuda-12.4
```

Run `kernel_runner`
-------------------
The runner requires the GEMM problem size, data type, GPU architecture, Strassen level, iteration count, warmup count, and number of CUDA streams.

```bash
./build/kernel_runner \
	--m=4096 \
	--n=4096 \
	--k=4096 \
	--dtype=f32 \
	--gpu_arch=ampere \
	--strassen_level=1 \
	--iterations=10 \
	--warmup=2 \
	--streams=7
```

Supported values:

```text
--dtype=f32|f16|fp64
--gpu_arch=volta|ampere|hopper
--strassen_level=0|1|2|all
```

Optional filtering:

```bash
./build/kernel_runner \
	--m=4096 --n=4096 --k=4096 \
	--dtype=f32 --gpu_arch=ampere --strassen_level=all \
	--iterations=10 --warmup=2 --streams=7 \
	--kernel_regex='presum'
```

For the full usage line:

```bash
./build/kernel_runner --help
```

Build Tests
-----------
From the repository root, build all CUDA GoogleTest binaries with:

```bash
make -C tests
```

The test Makefile writes test binaries directly into `tests/`.
You can also build one test binary by naming its target:

```bash
make -C tests test_ampere_f32_strassen_winograd_tile
make -C tests test_hopper_f32_strassen_winograd_presum
```

Run Tests
---------
Run the full test suite:

```bash
make -C tests run-tests
```

Run tests for one GPU family:

```bash
make -C tests run-volta-tests
make -C tests run-ampere-tests
make -C tests run-hopper-tests
```

Run one test binary through Make:

```bash
make -C tests run-test_ampere_f32_strassen_winograd_tile
```

Or run a compiled test binary directly:

```bash
cd tests
./test_ampere_f32_strassen_winograd_tile
```

Clean test artifacts:

```bash
make -C tests clean
```

Run Experiments
---------------
The repo has scripts to run the whole evaluation pipeline.
See `ArtifactEval.md` for details.


Current Status
----------------

We are actively working on optimizing our implementation and supporting larger cases. Here is a list of all things that we are working on for now. We are obviously happy to receive contributions from the open source community.

1. Add Cooperative Strassen GeMM Kernels for H100/H200: Currently, we only support Pingpong kernels because on our H200 system this kernel runs fastest on most cases. However, on H100, Cooperative kernels are fastest. We will soon support Cooperative Strassen GeMM Kernels.

2. Add GeMM kernels for Blackwell.

3. Overhaul Level 2 schedules to support larger Strassen family of algorithms.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
