# llama.cpp + Llama 3 on V100 — Setup Guide

This guide builds llama.cpp from source with CUDA support, runs Llama 3 inference on
V100 GPUs via a quantized GGUF model.

> **Tested target:** `discslab-server2`, NVIDIA V100 (Volta, compute capability **7.0**),
> 16 GB or 32 GB. Assumes `/mnt/nobackup/jchen/` already configured per
> [`README_vllm.md`](./README_vllm.md) Section 1 (HF cache, conda envs, pip cache).

> **TL;DR — reproducing a known-good environment:** once an env is working, use
> [`requirements_llamacpp.txt`](./requirements_llamacpp.txt) plus the build flags in
> Section 6 to recreate it. Step-by-step sections below exist to build that file and
> explain the V100-specific knobs.

---

## 0. Why llama.cpp on V100 (and how it differs from vLLM)

llama.cpp is the easier path on Volta:

| Concern | vLLM 0.5.4 | llama.cpp |
|---|---|---|
| BF16 weights on a no-BF16 GPU | manual `dtype="float16"` | irrelevant — GGUF uses INT quantizations |
| Flash Attention 2 (Ampere+) | manual `XFORMERS` backend | irrelevant — has own attention kernels |
| 16 GB V100 fits Llama-3-8B? | only with quantization or TP=2 | **yes** at Q4_K_M (~4.7 GB) or Q5_K_M (~5.7 GB) |
| Throughput at batch=1 | high | similar / slightly lower |
| Throughput at large batch | much higher (paged KV cache) | lower (no continuous batching) |
| Setup complexity | high (Python deps, version pins) | low (one C++ build, one GGUF download) |

For a single 16 GB V100 doing single-stream inference, llama.cpp is the more
forgiving choice. For batched serving, vLLM wins.

V100-specific knob you must set: **`CMAKE_CUDA_ARCHITECTURES=70`** (Volta = SM 7.0).
The default architecture list in llama.cpp's CMake doesn't always include 70 in
recent commits; setting it explicitly avoids the "no kernel image is available
for execution on the device" runtime error.

Verify hardware:

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
# Expect compute_cap = 7.0
nvcc --version
# Expect CUDA 12.x (we'll use 12.1)
gcc --version
# Expect gcc >= 11 for CUDA 12.x
cmake --version
# Expect cmake >= 3.20
```

If any of those are missing or too old, see Section 3.

---

## 1. Storage layout (already done if you followed README_vllm.md)

We reuse the exact same `/mnt/nobackup/jchen/` layout. New additions for llama.cpp:

```
/mnt/nobackup/jchen/
├── conda_envs/llamacpp-v100/                      # conda env (this guide)
├── llama.cpp/                                     # source clone + build dir
└── llama3_models/                                 # pre-quantized GGUFs
    └── Meta-Llama-3-8B-Instruct-Q4_K_M.gguf       # ~4.7 GB
```

The home quota / `HF_HOME` setup from `README_vllm.md` Sections 1a–1e still applies.
If you skipped that and home is tight, go back and run those sections first.

Re-confirm space:

```bash
quota -s                       # home should still be well under 12 GB
df -h /mnt/nobackup            # need ~10 GB free for source build + one GGUF
```

---

## 2. Create a fresh conda env for tooling

The C++ build doesn't need conda, but we use a small conda env for:

* `huggingface_hub` (downloading the GGUF or the source HF model)
* optional `convert_hf_to_gguf.py` (if you want to quantize from raw safetensors)

Keep it separate from `vllm-v100` so the two never fight over deps.

### 2a. Create + activate (mind the discslab-server2 quirks)

```bash
source /usr/local/pkgs/anaconda/etc/profile.d/conda.sh
conda create -n llamacpp-v100 python=3.10 -y
conda activate llamacpp-v100
```

### 2b. Verify (same trap as before — the prompt lies)

```bash
echo $CONDA_PREFIX
# expect:  /mnt/nobackup/jchen/conda_envs/llamacpp-v100
which python
# expect:  /mnt/nobackup/jchen/conda_envs/llamacpp-v100/bin/python
python --version
# expect:  Python 3.10.x
```

If any of those don't match, re-source the conda hook (Section 2 of README_vllm.md
covers why VSCode terminals need this).

### 2c. Install tooling and persist `HF_HOME`

```bash
python -m pip install -U "huggingface_hub[cli]"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/hf_cache.sh" <<'EOF'
export HF_HOME=/mnt/nobackup/jchen/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
EOF
conda deactivate && conda activate llamacpp-v100
echo $HF_HOME    # /mnt/nobackup/jchen/hf_cache
```

You'll re-use the same HF cache as vLLM, so any model already downloaded is shared.

---

## 3. Confirm system build dependencies

llama.cpp needs:

* `cmake` ≥ 3.20
* a C++17 compiler (`gcc` ≥ 11 recommended for CUDA 12.x)
* CUDA toolkit (we'll use whatever is at `/usr/local/pkgs/cuda/latest`)
* `git`, `make`

Check:

```bash
cmake --version
gcc --version
nvcc --version
ls -ld /usr/local/pkgs/cuda/latest
```

If `cmake` is missing or too old, install it into the conda env (no sudo needed):

```bash
python -m pip install "cmake>=3.27" ninja
```

This puts a fresh `cmake` and `ninja` in `$CONDA_PREFIX/bin/`. Verify:

```bash
which cmake     # /mnt/nobackup/jchen/conda_envs/llamacpp-v100/bin/cmake
cmake --version
```

If `nvcc` isn't on PATH, add it:

```bash
export PATH=/usr/local/pkgs/cuda/latest/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/pkgs/cuda/latest/lib64:$LD_LIBRARY_PATH
```

(You can persist these in the same `activate.d/hf_cache.sh` you wrote in Section 2c.)

---

## 4. Clone and build llama.cpp with CUDA + V100 support

### 4a. Clone

```bash
cd /mnt/nobackup/jchen/
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git log -1 --oneline    # record the commit you built — useful for reproducibility
```

### 4b. Configure and build

The two flags that matter on V100:

* `-DGGML_CUDA=ON` — turn on the CUDA backend.
* `-DCMAKE_CUDA_ARCHITECTURES=70` — generate kernels for SM 7.0 (Volta).

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

Expect 5–15 minutes on first build. The CUDA compile of `ggml-cuda.cu` is the long
step — that's normal.

### 4c. Verify the binaries built and have CUDA enabled

```bash
ls build/bin/ | grep -E '^llama-(cli|server|bench|quantize)$'
# expect: llama-bench  llama-cli  llama-quantize  llama-server

./build/bin/llama-cli --version
# expect a line mentioning CUDA / GPU build info
```

Quick GPU-detection check:

```bash
./build/bin/llama-cli --list-devices
# expect at least one CUDA device labeled 'Tesla V100-...'
```

If `--list-devices` shows only `CPU`, CUDA support didn't compile in — recheck
Section 4b's flags and that `nvcc` was on PATH during the cmake configure step.

---

## 5. Get a Llama 3 GGUF

Two options. **Option A** (download a pre-quantized GGUF) is simpler and recommended;
**Option B** (convert + quantize from your existing HF cache) is useful if you want
custom quantizations.

### 5a. Option A — download a pre-quantized GGUF (recommended)

We'll use `bartowski/Meta-Llama-3-8B-Instruct-GGUF`, a well-maintained ungated repo
with multiple quantization levels.

```bash
mkdir -p /mnt/nobackup/jchen/llama3_models

"$CONDA_PREFIX/bin/hf" download \
    bartowski/Meta-Llama-3-8B-Instruct-GGUF \
    Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    --local-dir /mnt/nobackup/jchen/llama3_models
```

Quantization picker for V100:

| File | Size | Best for |
|---|---|---|
| `*-Q4_K_M.gguf` | ~4.7 GB | 16 GB V100, lots of context headroom |
| `*-Q5_K_M.gguf` | ~5.7 GB | 16 GB V100, slightly higher quality |
| `*-Q8_0.gguf` | ~8.5 GB | 32 GB V100, near-FP16 quality |
| `*-f16.gguf` | ~16 GB | 32 GB V100, no quality loss |

Verify:

```bash
ls -lh /mnt/nobackup/jchen/llama3_models/*.gguf
```

### 5b. Option B — convert + quantize from the HF safetensors cache

If you already downloaded `meta-llama/Meta-Llama-3-8B-Instruct` for vLLM, you have
the BF16 safetensors in `/mnt/nobackup/jchen/hf_cache/...`. Convert and quantize:

```bash
# llama.cpp ships a converter script; install its Python deps:
python -m pip install -r /mnt/nobackup/jchen/llama.cpp/requirements.txt

# Resolve the snapshot dir (HF stores models under a hash):
SNAPSHOT=$(ls -d /mnt/nobackup/jchen/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/*/ | head -1)
echo "Source: $SNAPSHOT"

# Convert BF16 safetensors -> FP16 GGUF
python /mnt/nobackup/jchen/llama.cpp/convert_hf_to_gguf.py "$SNAPSHOT" \
    --outfile /mnt/nobackup/jchen/llama3_models/Meta-Llama-3-8B-Instruct-f16.gguf \
    --outtype f16

# Quantize FP16 -> Q4_K_M
/mnt/nobackup/jchen/llama.cpp/build/bin/llama-quantize \
    /mnt/nobackup/jchen/llama3_models/Meta-Llama-3-8B-Instruct-f16.gguf \
    /mnt/nobackup/jchen/llama3_models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    Q4_K_M
```

Either option produces the same `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`.

---

## 6. Smoke test

Run a single-prompt generation, with **all layers offloaded to the GPU**:

```bash
cd /mnt/nobackup/jchen/llama.cpp

./build/bin/llama-cli \
    -m /mnt/nobackup/jchen/llama3_models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    -n 128 \
    -p "Explain in two sentences why V100 cannot run Flash Attention 2."
```

Flag cheat sheet:

| Flag | Meaning |
|---|---|
| `-m` | path to GGUF |
| `-ngl 99` | offload up to 99 layers to GPU (Llama-3-8B has 32; 99 just means "all") |
| `-c 4096` | KV cache / context window |
| `-n 128` | max new tokens |
| `-p` | prompt |

While it runs, in another terminal:

```bash
nvidia-smi    # should show llama-cli using GPU memory and a non-zero util %
```

If `nvidia-smi` shows the model loaded on CPU only (or `-ngl 99` errors), CUDA
support didn't link in — re-run Section 4b.

Expected first-token latency on a V100 with Q4_K_M, all layers on GPU: **a couple
hundred ms**. Steady-state throughput: **~50–80 tok/s** at batch=1.

---

## 7. Reproducible env via `requirements_llamacpp.txt`

llama.cpp itself is reproduced via the **git commit + cmake flags**, not pip.
Capture both.

### 7a. Freeze the conda env

```bash
python -m pip freeze > /home/2020/jchen213/llama3_on_v100/requirements_llamacpp.txt
```

This file is small — only `huggingface_hub`, optionally the converter's deps,
and possibly `cmake`/`ninja` if you installed them into the env.

### 7b. Record the llama.cpp build provenance

Append the git SHA and build flags to the freeze file (or keep it next to the
README):

```bash
cd /mnt/nobackup/jchen/llama.cpp
{
    echo "# llama.cpp commit:"
    git rev-parse HEAD
    echo "# cmake flags:"
    echo "  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release"
    echo "# build:"
    echo "  cmake --build build --config Release -j \$(nproc)"
} >> /home/2020/jchen213/llama3_on_v100/requirements_llamacpp.txt
```

### 7c. Reproduce on a fresh box

```bash
# Conda env + Python deps
conda create -n llamacpp-v100 python=3.10 -y
conda activate llamacpp-v100
python -m pip install -r /home/2020/jchen213/llama3_on_v100/requirements_llamacpp.txt

# llama.cpp build (use the commit SHA recorded in 7b)
git clone https://github.com/ggml-org/llama.cpp.git /mnt/nobackup/jchen/llama.cpp
cd /mnt/nobackup/jchen/llama.cpp
git checkout <SHA from requirements_llamacpp.txt>
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cmake` errors with `Could NOT find CUDAToolkit` | `nvcc` not on PATH during configure. `export PATH=/usr/local/pkgs/cuda/latest/bin:$PATH` then re-run `cmake -B build ...`. |
| Build succeeds but runtime says `no kernel image is available for execution on the device` | `CMAKE_CUDA_ARCHITECTURES` didn't include 70. Re-configure with `-DCMAKE_CUDA_ARCHITECTURES=70` and rebuild from scratch (`rm -rf build && cmake -B build ...`). |
| `llama-cli --list-devices` shows only CPU | CUDA didn't compile in — likely `-DGGML_CUDA=ON` was missing or `nvcc` wasn't found. Re-run Section 4b. |
| `error while loading shared libraries: libcudart.so.12` | `LD_LIBRARY_PATH` doesn't include CUDA's `lib64`. `export LD_LIBRARY_PATH=/usr/local/pkgs/cuda/latest/lib64:$LD_LIBRARY_PATH`. |
| Slow generation (~5 tok/s) | Layers running on CPU. Ensure `-ngl 99` is passed and `nvidia-smi` shows non-zero memory + utilization. |
| OOM on 16 GB V100 with Q8_0 | Q8_0 is 8.5 GB; KV cache for 8K context is another ~2 GB. Switch to Q5_K_M or Q4_K_M, or lower `-c` to 2048. |
| `gcc: error: unrecognized command-line option '-std=c++20'` | gcc too old for the C++ standard llama.cpp asks for. Use a newer toolchain: `module load gcc/11` if the host has Modules, or `conda install -n llamacpp-v100 -c conda-forge gxx_linux-64=11 gcc_linux-64=11`. |
| `convert_hf_to_gguf.py` import errors | Install its requirements: `python -m pip install -r /mnt/nobackup/jchen/llama.cpp/requirements.txt`. |
| `huggingface-cli` deprecation warning / `error: invalid choice: 'download'` | Same issue as in `README_vllm.md`. Use `"$CONDA_PREFIX/bin/hf" download ...`. |

---

## Layout summary (after a successful setup)

```
$HOME (≤ 12 GB quota)
├── ~/.condarc                          # already configured for /mnt/nobackup
├── ~/.config/pip/pip.conf              # already configured for /mnt/nobackup
└── ~/llama3_on_v100/                   # this guide + scripts (small)

/mnt/nobackup/jchen/  (TB-scale, no quota)
├── conda_envs/llamacpp-v100/           # the env (~500 MB)
├── llama.cpp/                          # source + build/ (~2 GB after build)
└── llama3_models/
    └── Meta-Llama-3-8B-Instruct-Q4_K_M.gguf   # ~4.7 GB
```

---

## What's Next

You now have two parallel ways to run Llama 3 on V100:

* **vLLM** for high-throughput batched inference — see [`README_vllm.md`](./README_vllm.md).
* **llama.cpp** for single-stream, low-setup, low-memory inference — this guide.

Pick based on workload: many concurrent users → vLLM; one user, modest GPU → llama.cpp.
