# llama.cpp + Llama 3 on V100 — Setup Guide

This guide builds llama.cpp from source with CUDA support on a V100, converts the
Llama-3-8B safetensors you already have on disk into a quantized GGUF, and runs a
smoke test.

> **Tested target:** `discslab-server2`, NVIDIA V100 (Volta, compute capability **7.0**),
> 16 GB or 32 GB.
>
> **Assumes you've already done [`README_vllm.md`](./README_vllm.md)**, specifically:
>
> - `/mnt/nobackup/jchen/` is configured (`conda_envs/`, `conda_pkgs/`, `pip_cache/`,
>   `hf_cache/`) and `~/.condarc` + `~/.config/pip/pip.conf` point at it.
> - `meta-llama/Meta-Llama-3-8B-Instruct` is downloaded into
>   `/mnt/nobackup/jchen/hf_cache/hub/...`.
> - You know the discslab-server2 quirks: source the conda hook explicitly, trust
>   `$CONDA_PREFIX` not the prompt, use `python -m pip`, use `"$CONDA_PREFIX/bin/hf"`
>   instead of `huggingface-cli`.

---

## 0. Why llama.cpp on V100 (and how it differs from vLLM)

| Concern | vLLM 0.5.4 | llama.cpp |
|---|---|---|
| BF16 weights on a no-BF16 GPU | manual `dtype="float16"` | irrelevant — GGUF uses INT quantizations |
| Flash Attention 2 | manual `XFORMERS` backend | irrelevant — has own attention kernels |
| 16 GB V100 fits Llama-3-8B? | only with quantization or TP=2 | **yes** at Q4_K_M (~4.7 GB) or Q5_K_M (~5.7 GB) |
| Throughput at large batch | high (paged KV cache) | lower (no continuous batching) |
| Setup complexity | high | low |

The one V100-specific knob you **must** set: **`CMAKE_CUDA_ARCHITECTURES=70`** (Volta = SM 7.0).
The default arch list in recent llama.cpp commits doesn't always include 70, and you'll get
"no kernel image is available for execution on the device" at runtime if it's missing.

---

## 1. Create a conda env for the conversion script

The C++ build doesn't need conda, but `convert_hf_to_gguf.py` (the safetensors → GGUF
converter) needs `torch`, `transformers`, `numpy`, etc. Keep this isolated from
`vllm-v100` so they can't fight over deps.

```bash
source /usr/local/pkgs/anaconda/etc/profile.d/conda.sh
conda create -n llamacpp-v100 python=3.10 -y
conda activate llamacpp-v100

# Sanity check (the trap from README_vllm.md Section 2c):
echo $CONDA_PREFIX     # /mnt/nobackup/jchen/conda_envs/llamacpp-v100
which python           # ...llamacpp-v100/bin/python

# Reuse the same HF cache as vLLM
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/hf_cache.sh" <<'EOF'
export HF_HOME=/mnt/nobackup/jchen/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
EOF
conda deactivate && conda activate llamacpp-v100
echo $HF_HOME          # /mnt/nobackup/jchen/hf_cache
```

The HF cache snippet means `convert_hf_to_gguf.py` will read the safetensors that
vLLM downloaded earlier — no re-download.

---

## 2. Confirm build dependencies

> **Heads-up for `discslab-server2`:** the CUDA toolkit lives at `/usr/local/cuda-12.8/`,
> **not** at `/usr/local/pkgs/cuda/latest/` (that path is on `$PATH` by default but the
> directory is a dead reference — `nvcc` is not actually there). You must point at
> `/usr/local/cuda-12.8/bin` explicitly, or cmake's CUDA detection will fail with
> `No CMAKE_CUDA_COMPILER could be found`.

```bash
ls /usr/local/cuda-12.8/bin/nvcc       # confirm it exists
/usr/local/cuda-12.8/bin/nvcc --version
cmake --version                        # need >= 3.20
gcc --version                          # need >= 11 for CUDA 12.x
```

If `cmake` is missing or too old, install into the env (no sudo needed):

```bash
python -m pip install "cmake>=3.27" ninja
hash -r                # forget cached path of any system cmake
which cmake            # should resolve to .../llamacpp-v100/bin/cmake
```

**Prepend the real CUDA path to PATH** (this is the step that fixes
`No CMAKE_CUDA_COMPILER could be found`):

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
which nvcc            # /usr/local/cuda-12.8/bin/nvcc
```

To make this permanent for the env, append to the same activate.d snippet:

```bash
cat >> "$CONDA_PREFIX/etc/conda/activate.d/hf_cache.sh" <<'EOF'
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
EOF
```

---

## 3. Clone and build llama.cpp with CUDA + V100 support

Clone into the project directory (next to this README):

```bash
cd ~/llama3_on_v100
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git log -1 --oneline   # record the commit — useful for reproducibility
```

> **Quota note:** the source clone is ~150 MB but the `build/` directory grows to
> ~2–3 GB during compilation. With a 12 GB home quota, this will fit but tightens
> things. If quota becomes an issue, redirect the build dir off home with
> `cmake -B /mnt/nobackup/jchen/llamacpp_build ...` and the same for `cmake --build`.

Configure and build (the **two flags that matter on V100**: `-DGGML_CUDA=ON` and
`-DCMAKE_CUDA_ARCHITECTURES=70`):

```bash
# If you hit "No CMAKE_CUDA_COMPILER" earlier, nuke the failed configure first:
rm -rf build

cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j $(nproc)
```

First build takes 5–15 minutes (compiling `ggml-cuda.cu` is the long step).

Verify CUDA actually linked in:

```bash
ls build/bin/ | grep -E '^llama-(cli|server|bench|quantize)$'
# expect: llama-bench  llama-cli  llama-quantize  llama-server

./build/bin/llama-cli --list-devices
# expect at least one CUDA device labeled 'Tesla V100-...'
```

If `--list-devices` shows only CPU, CUDA support didn't compile in. Common cause:
`nvcc` wasn't on PATH during the cmake configure step. Re-export PATH (Section 2),
nuke and reconfigure: `rm -rf build && cmake -B build ...`.

---

## 4. Convert the existing safetensors → FP16 GGUF

The HF model is already at `/mnt/nobackup/jchen/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/`.
We'll convert it to **FP16 GGUF** — the same numeric precision vLLM ends up using on
V100 (since V100 has no native BF16, both stacks effectively run the model in FP16).
**No quantization** — that would change the weights; we want to run the original model.

> *(If you only have a 16 GB V100, the FP16 file (~16 GB) won't fit alongside KV
> cache. In that case you'd quantize to Q5_K_M / Q4_K_M with `llama-quantize` — see
> the troubleshooting note. On a 32 GB V100 you don't need to.)*

### 4a. Install the converter's Python deps

```bash
python -m pip install -r ~/llama3_on_v100/llama.cpp/requirements.txt
```

This pulls `torch`, `transformers`, `numpy`, `sentencepiece`, `gguf`, etc. into the
`llamacpp-v100` env (isolated from `vllm-v100`).

### 4b. Convert BF16 safetensors → FP16 GGUF

```bash
mkdir -p /mnt/nobackup/jchen/llama3_models

# Resolve the cached snapshot dir (HF stores models under a content hash):
SNAPSHOT=$(ls -d /mnt/nobackup/jchen/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/*/ | head -1)
echo "Source snapshot: $SNAPSHOT"

python ~/llama3_on_v100/llama.cpp/convert_hf_to_gguf.py "$SNAPSHOT" \
    --outfile /mnt/nobackup/jchen/llama3_models/Meta-Llama-3-8B-Instruct-f16.gguf \
    --outtype f16
```

Output is ~16 GB. Takes a few minutes — mostly disk-bound.

```bash
ls -lh /mnt/nobackup/jchen/llama3_models/
# expect Meta-Llama-3-8B-Instruct-f16.gguf, ~15-16 GB
```

---

## 5. Smoke test

```bash
cd ~/llama3_on_v100/llama.cpp

./build/bin/llama-cli \
    -m /mnt/nobackup/jchen/llama3_models/Meta-Llama-3-8B-Instruct-f16.gguf \
    -ngl 99 \
    -c 4096 \
    -n 128 \
    -p "Explain in two sentences why V100 cannot run Flash Attention 2."
```

Flag cheat sheet:

| Flag | Meaning |
|---|---|
| `-m` | path to GGUF |
| `-ngl 99` | offload up to 99 layers to GPU (Llama-3-8B has 32; "99" just means "all") |
| `-c 4096` | KV cache / context window |
| `-n 128` | max new tokens |
| `-p` | prompt |

Reality check from another shell while it runs:

```bash
nvidia-smi    # llama-cli should show ~16 GB GPU memory used + non-zero util %
```

Expected on a 32 GB V100 with FP16, all layers on GPU: **~30–50 tok/s** steady-state
at batch=1. (FP16 is ~2× slower than Q4_K_M because each generated token reads ~2× the
weight bytes — but it's the original-precision model.)

If `nvidia-smi` shows zero GPU activity (or the model loads on CPU only), CUDA
didn't link in — go back to Section 3.

---

## 6. Reproducible env

llama.cpp itself is reproduced via the **git commit + cmake flags**, not pip. Capture both:

```bash
# Conda env freeze
python -m pip freeze > /home/2020/jchen213/llama3_on_v100/requirements_llamacpp.txt

# Append llama.cpp build provenance to the same file
cd ~/llama3_on_v100/llama.cpp
{
    echo ""
    echo "# llama.cpp commit:"
    echo "#   $(git rev-parse HEAD)"
    echo "# cmake configure:"
    echo "#   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release"
    echo "# cmake build:"
    echo "#   cmake --build build --config Release -j \$(nproc)"
} >> /home/2020/jchen213/llama3_on_v100/requirements_llamacpp.txt
```

To reproduce on a fresh box:

```bash
conda create -n llamacpp-v100 python=3.10 -y
conda activate llamacpp-v100
python -m pip install -r /home/2020/jchen213/llama3_on_v100/requirements_llamacpp.txt

git clone https://github.com/ggml-org/llama.cpp.git ~/llama3_on_v100/llama.cpp
cd ~/llama3_on_v100/llama.cpp
git checkout <SHA from requirements_llamacpp.txt>
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cmake` errors with `Could NOT find CUDAToolkit` | `nvcc` not on PATH during configure. `export PATH=/usr/local/cuda-12.8/bin:$PATH`, then `rm -rf build && cmake -B build ...`. |
| Build succeeds but runtime says `no kernel image is available for execution on the device` | `CMAKE_CUDA_ARCHITECTURES` didn't include 70. Re-configure with `-DCMAKE_CUDA_ARCHITECTURES=70` and rebuild from scratch (`rm -rf build`). |
| `llama-cli --list-devices` shows only CPU | CUDA didn't compile in — likely `-DGGML_CUDA=ON` was missing or `nvcc` wasn't found. Re-run Section 3. |
| `error while loading shared libraries: libcudart.so.12` | `LD_LIBRARY_PATH` doesn't include CUDA's `lib64`. `export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH`. |
| Slow generation (~5 tok/s) | Layers running on CPU. Pass `-ngl 99` and confirm `nvidia-smi` shows non-zero memory + utilization. |
| OOM loading FP16 GGUF on a 16 GB V100 | The full FP16 model is ~16 GB; KV cache for 4K context adds ~1 GB on top. A 16 GB V100 can't fit it. **Quantize** to a smaller GGUF: `~/llama3_on_v100/llama.cpp/build/bin/llama-quantize $WORKDIR/llama3_models/Meta-Llama-3-8B-Instruct-f16.gguf $WORKDIR/llama3_models/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf Q5_K_M` (Q5_K_M ~5.7 GB) or `Q4_K_M` (~4.7 GB). Then run `llama-cli` with the quantized path. |
| `gcc: error: unrecognized command-line option '-std=c++20'` | gcc too old for the C++ standard llama.cpp asks for. `module load gcc/11` if Modules are available, or `conda install -n llamacpp-v100 -c conda-forge gxx_linux-64=11 gcc_linux-64=11`. |
| `convert_hf_to_gguf.py` import errors (`gguf`, `safetensors`, etc.) | `python -m pip install -r ~/llama3_on_v100/llama.cpp/requirements.txt`. |
| `convert_hf_to_gguf.py: error: Model class ... is not supported` | llama.cpp commit too old for newer Llama-3 config fields. `git pull` in `~/llama3_on_v100/llama.cpp` and rebuild. |
| `which cmake` still points at a stale system cmake after `pip install cmake` | Run `hash -r` to clear bash's path cache, then `which cmake` again. |

---

## Layout summary

```
$HOME (≤ 12 GB quota)
└── ~/llama3_on_v100/                     # this guide + scripts (small)

/mnt/nobackup/jchen/  (TB-scale, no quota)
├── conda_envs/llamacpp-v100/             # ~1.5 GB after converter deps land
├── llama.cpp/                            # source + build/  (~2 GB after build)
└── llama3_models/
    └── Meta-Llama-3-8B-Instruct-f16.gguf       # ~16 GB (the original-precision model)
```

---

## What's Next

You now have two parallel ways to run Llama 3 on V100 — pick by workload:

* **vLLM** ([`README_vllm.md`](./README_vllm.md)) — high-throughput batched inference.
* **llama.cpp** (this guide) — single-stream, low-setup, low-memory inference.
