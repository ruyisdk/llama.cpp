# Build and run Qwen3 with Ztt

This guide builds llama.cpp for RISC-V Linux with Ztt support and runs Qwen3-0.6B Q4_K_M with a Ztt-enabled QEMU. The integration uses [TORQ-Tile](https://github.com/XUANTIE-RV/torq-tile) for Q4_K/Q6_K GEMM in the ggml CPU backend. Both `GGML_CPU_TORQ_TILE` and `GGML_CPU_TORQ_TILE_ZTT` must be enabled; they default to OFF. Other tensor types and operations use the existing GGML CPU paths.

## Requirements

- A Linux host with Git, CMake 3.16 or newer, Ninja, `curl`, and `sha256sum`.
- A RISC-V Linux toolchain with Ztt ISA support and the required `<riscv_ztt.h>` API. The static build needs a Linux sysroot, startup objects, and static C/C++ runtime libraries for the `lp64d` ABI.
- A Ztt/AME v0.6 QEMU Linux-user build supporting the CPU properties shown below.

The example configuration was exercised with Xuantie Clang `24.0.0git` (`ztt-dev B-20260916`) and QEMU `8.2.94` from the AME v0.6 development build dated 2026-10-16. Public development resources are linked in the [Ztt GNU/LLVM announcement](https://lists.riscv.org/g/tech-attached-matrix-extension/message/930) and the [QEMU ame-0.6 branch](https://github.com/ruyisdk/qemu/tree/ame-0.6). Their correspondence to the tested development binaries has not been verified.

## Get the source and configure the tools

For a new checkout:

```bash
git clone --branch ztt https://github.com/ruyisdk/llama.cpp.git
cd llama.cpp
```

Run the remaining commands from the repository root. Set these paths to your installed tools:

```bash
export ZTT_TOOLCHAIN=/path/to/ztt-linux-toolchain
export QEMU=/path/to/ztt-qemu-riscv64
```

The commands below use the Clang configuration described above. If the compiler needs explicit target, sysroot, or runtime library paths, supply a CMake toolchain file with `-DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake`. GCC requires separate validation; select its Linux cross-compiler pair through `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`. Use a separate build directory when switching toolchains.

## Build

```bash
cmake -S . -B build-ztt -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER="$ZTT_TOOLCHAIN/bin/clang" \
    -DCMAKE_CXX_COMPILER="$ZTT_TOOLCHAIN/bin/clang++" \
    -DCMAKE_C_FLAGS='-march=rv64gcv -mabi=lp64d' \
    -DCMAKE_CXX_FLAGS='-march=rv64gcv -mabi=lp64d' \
    -DCMAKE_EXE_LINKER_FLAGS=-static \
    -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
    -DGGML_CPU_REPACK=ON -DGGML_RVV=ON \
    -DGGML_CPU_TORQ_TILE=ON -DGGML_CPU_TORQ_TILE_ZTT=ON \
    -DGGML_RV_ZFH=OFF -DGGML_RV_ZVFH=OFF -DGGML_RV_ZVFBFWMA=OFF \
    -DGGML_RV_ZICBOP=OFF -DGGML_RV_ZIHINTPAUSE=OFF \
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-ztt --parallel 8 --target llama-completion
```

CMake checks the compiler's Ztt support and fetches the TORQ-Tile revision pinned in [ggml/src/ggml-cpu/CMakeLists.txt](../ggml/src/ggml-cpu/CMakeLists.txt). To use a local checkout of that revision, pass `-DFETCHCONTENT_SOURCE_DIR_TORQ_TILE_DOWNLOAD=/path/to/torq-tile`. Only the selected kernels are compiled; the full TORQ-Tile library, tests, and benchmarks are not built. Fetching and compiler probing are skipped when the integration is disabled.

`TORQ_TILE_ZTT_COMPILE_OPTIONS` controls the specialized kernel target. Its default is `-march=rv64gcv_ztt_zttuds32b_zttm16_ztta4;-mabi=lp64d`, selecting UDS32, 16 matrix registers, and 4 accumulator registers. Generic GGML sources use RV64GCV. A different compiler or hardware profile may require overriding this option with a complete semicolon-separated flag list that matches the runtime.

## Download the model

This fixed Qwen3-0.6B Q4_K_M model contains both Q4_K and Q6_K weights:

```bash
curl -L --fail 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf' -o Qwen3-0.6B-Q4_K_M.gguf
echo 'ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a  Qwen3-0.6B-Q4_K_M.gguf' | sha256sum -c -
export MODEL="$PWD/Qwen3-0.6B-Q4_K_M.gguf"
```

## Run Qwen3

The short generation example uses NLEN8, VLEN128, and the UDS32/M16/A4 profile selected at build time:

```bash
export QEMU_CPU='rv64,v=true,vlen=128,elen=64,vext_spec=v1.0,x-ztt=true,amenlen=8,ameudsz=32,ame-num-mregs=16,ame-num-accregs=4'

GGML_TORQ_TILE_DISABLE=0 "$QEMU" -cpu "$QEMU_CPU" \
    build-ztt/bin/llama-completion -m "$MODEL" \
    -p 'The capital of France is' -n 8 -t 1 -tb 1 -c 512 -ngl 0 \
    --temp 0 --seed 1 --no-warmup --no-conversation \
    --no-display-prompt --simple-io -fa off -ctk f16 -ctv f16
```

With the tested setup, the eight generated tokens were:

```text
Paris. The capital of France is also
```

Check the startup log for `TORQ_TILE = 1`, `ZTT = 1`, and the `CPU_TORQ_TILE` model buffer. These show that the backend is available and eligible weights are packed; operation dispatch still depends on tensor type and shape. Weight conversion and prepacking happen during model loading, and the packed weights are reused during inference.

This command runs the text-generation example. `llama-completion` timing output does not provide phase instruction counts; those require a separate counting harness and QEMU plugins.

## Runtime controls and integration notes

To use ordinary GGML in the same binary, repeat the launch with `GGML_TORQ_TILE_DISABLE=1`. The setting is read once per process and disables both weight packing and dispatch. An unset value, empty string, `0`, `false`, `no`, or `off` enables the extension; other values disable it.

This is a target-specific build without automatic Ztt hardware detection. `GGML_CPU_ALL_VARIANTS` is not supported. The runtime must support the compiled Ztt profile; NLEN is read at runtime and must be a power of two no greater than 128, and UDS must match the compile-time profile.

- Initial packed-buffer selection requires contiguous 2D Q4_K or Q6_K weights. F32 activations are quantized in blocks of 256, so K must be a positive multiple of 256. The GEMM handles M/N tails.
- The packed weight buffer supports `GET_ROWS`, including embeddings. A complete single-row slice can also be cast to a contiguous F32 vector on the CPU. Supported reshaped weights and strided or batched inputs use a decoded CPU fallback when they cannot use Ztt GEMM. This does not imply scheduler support for arbitrary reshaped or batched weights.
- There is no Ztt `MUL_MAT_ID` path. Packed weights require complete-tensor loading; partial updates, raw tensor readback, and copying packed weights to another backend are unsupported.

Activation quantization and kernel accumulation can differ numerically from generic GGML. Current model validation has not met the per-row NMSE limit of `1e-4` against generic GGML, so model-level numerical equivalence is not established. Validate fixed-input logits or perplexity for the intended workload.
