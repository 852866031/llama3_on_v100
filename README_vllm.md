# vLLM + Llama 3 on V100 — Setup Guide

This guide installs vLLM in a fresh conda environment, runs Llama 3 inference on V100
GPUs, and runs a simple throughput benchmark.

> **Tested target:** `discslab-server2`, NVIDIA V100 (Volta, compute capability **7.0**),
> 16 GB or 32 GB. The guide assumes the discslab home quota (~12 GB) and the
> `/mnt/nobackup` scratch mount; if you're on a different host, adapt Section 1.

---

## 0. V100 Constraints (read first)

V100 is older than what current LLM stacks assume by default. Three things matter:

1. **No native BF16.** Llama 3 weights ship as BF16, but you must run them as FP16 on
   V100. Pass `dtype="float16"` to vLLM (or `--dtype half` on the CLI).
2. **No Flash Attention 2.** FA2 requires Ampere (SM 8.0+). Force the xFormers backend
   with `VLLM_ATTENTION_BACKEND=XFORMERS`.
3. **VRAM is tight.** Llama-3-8B in FP16 is ~16 GB just for weights. On a 16 GB V100
   you must either quantize (AWQ/GPTQ) or shard with `tensor_parallel_size>=2`. On a
   32 GB V100 it fits comfortably with a few GB of KV cache.

Stick with **vLLM 0.5.x**. It is the last line that reliably supports Volta with the
xFormers fallback. Newer releases (0.6+) have regressed or dropped V100 paths.

Check what you have before starting:

```bash
nvidia-smi                                  # confirm V100s and driver
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

---

## 1. Storage layout on `discslab-server2`

This is the **most important** section — get it wrong and `pip install torch` will
exhaust your home quota mid-install. On `discslab-server2`:

| Mount | Capacity | Purpose |
|---|---|---|
| `$HOME` (`/home/2020/<user>`) | **12 GB user quota** | Dotfiles, scripts, VSCode server. **Not** for envs or models. |
| `/mnt/nobackup` | 7.0 TB shared (~2.4 TB free) | Conda envs, pkg caches, pip wheel cache, HF model cache. **Everything bulky goes here.** |

### 1a. Check current usage

```bash
quota -s                                     # home quota: USED vs QUOTA
df -h /mnt/nobackup                          # confirm space available
du --max-depth=1 -h ~ 2>/dev/null | sort -h | tail -10   # what's eating home
```

You want at least **3–4 GB free in home** before starting the install (pip uses transient
space for wheel extraction even when the cache and target dirs are elsewhere).

### 1b. Free home space if needed

Common safe-to-delete caches in `$HOME` on this server:

```bash
rm -rf ~/.cache/pip            # pip wheel cache (regenerates on demand) — often 1–2 GB
rm -rf ~/.npm                  # npm cache — typically 0.5 GB
# Optional, only if you don't have an active VSCode remote session:
# rm -rf ~/.vscode-server      # ~1 GB; will regenerate on next VSCode reconnect
```

If you need much more headroom, you can move bulky user data off home:

```bash
mv ~/others /mnt/nobackup/jchen/others && ln -s /mnt/nobackup/jchen/others ~/others
```

### 1c. Configure conda to use `/mnt/nobackup` for envs and pkgs

Conda's defaults put envs in `~/.conda/envs/` and downloaded pkg tarballs in
`~/.conda/pkgs/`. Both will overflow home. Redirect them:

```bash
mkdir -p /mnt/nobackup/jchen/conda_envs
mkdir -p /mnt/nobackup/jchen/conda_pkgs
conda config --prepend envs_dirs /mnt/nobackup/jchen/conda_envs
conda config --prepend pkgs_dirs /mnt/nobackup/jchen/conda_pkgs
```

Verify — the `/mnt/nobackup/...` entries **must be first**:

```bash
conda config --show envs_dirs pkgs_dirs
# envs_dirs:
#   - /mnt/nobackup/jchen/conda_envs       <-- first
#   - /home/2020/jchen213/.conda/envs
#   - /usr/local/pkgs/anaconda/envs
# pkgs_dirs:
#   - /mnt/nobackup/jchen/conda_pkgs       <-- first
#   - /home/2020/jchen213/.conda/pkgs
```

### 1d. Configure the pip cache

Pip's wheel cache also defaults to `~/.cache/pip`. Pin it to `/mnt/nobackup` globally
(applies to every env and every pip invocation):

```bash
mkdir -p /mnt/nobackup/jchen/pip_cache ~/.config/pip
printf '[global]\ncache-dir = /mnt/nobackup/jchen/pip_cache\n' > ~/.config/pip/pip.conf
```

### 1e. Set the Hugging Face cache

Llama-3-8B is ~16 GB on disk and absolutely must not land in home. Set `HF_HOME` so
both `huggingface-cli` and vLLM read/write from `/mnt/nobackup`:

```bash
mkdir -p /mnt/nobackup/jchen/hf_cache
export HF_HOME=/mnt/nobackup/jchen/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
```

We'll make this persistent per-env in Section 2c.

---

## 2. Create the Conda Environment

### 2a. Create

With Section 1c done, this lands on `/mnt/nobackup` automatically:

```bash
conda create -n vllm-v100 python=3.10 -y
```

Python 3.10 is the safest choice for vLLM 0.5.x wheels.

### 2b. Activate (the careful version — VSCode terminals need this)

VSCode's integrated terminal often skips `~/.bashrc`, so `conda activate` will set the
shell prompt but *not* update `PATH` or `CONDA_PREFIX`. Source the conda hook
explicitly first:

```bash
source /usr/local/pkgs/anaconda/etc/profile.d/conda.sh
conda activate vllm-v100
```

### 2c. Verify the env actually activated (do not skip — the prompt lies)

```bash
echo $CONDA_PREFIX
# expect:  /mnt/nobackup/jchen/conda_envs/vllm-v100
which python
# expect:  /mnt/nobackup/jchen/conda_envs/vllm-v100/bin/python
python --version
# expect:  Python 3.10.x
python -m pip --version
# expect:  pip ... from /mnt/nobackup/jchen/conda_envs/vllm-v100/lib/python3.10/.../pip
```

The prompt prefix `(/mnt/nobackup/jchen/conda_envs/vllm-v100)` is **not** sufficient
proof of activation — `conda` updates the prompt cosmetically even when activation
didn't take. Trust `$CONDA_PREFIX` and `which python`.

If `which python` starts with `/home/2020/...` or `/usr/...`, **stop**. Either
Section 1c didn't take effect, or the conda hook wasn't sourced (re-run 2b).
Installing torch from here will exhaust your home quota.

> **Why we use `python -m pip` everywhere from now on:** even after correct activation,
> a stale `~/.local/bin/pip` shim from a prior `pip --user` install can shadow the env's
> `pip` because `~/.local/bin` may sit earlier in `PATH`. `python -m pip` invokes pip
> through *this* python, bypassing PATH entirely. If you'd rather fix it once and use
> `pip` directly, run: `rm -f ~/.local/bin/pip ~/.local/bin/pip3 ~/.local/bin/pip3.*`

### 2d. Persist `HF_HOME` for this env

Drop a snippet into the env's `activate.d/` so `HF_HOME` is set automatically every time
you `conda activate vllm-v100`:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/hf_cache.sh" <<'EOF'
export HF_HOME=/mnt/nobackup/jchen/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
EOF

# Reactivate so the snippet runs in this shell too
conda deactivate && conda activate vllm-v100
echo $HF_HOME
# /mnt/nobackup/jchen/hf_cache
```

---

## 3. Install PyTorch (CUDA 12.1 build)

V100 supports CUDA 11.8 and 12.1. Use the cu121 wheel that vLLM 0.5.x is built against.
**Use `python -m pip`** (see Section 2c for why) so the install targets the env's
site-packages, not `~/.local`:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

If pip says `Defaulting to user installation because normal site-packages is not
writeable` — **abort**. That message means `python` is system Python 3.8, not the env.
Go back to Section 2b and re-source the conda hook.

This is the install step that previously blew out home quota. With Sections 1c–1d done,
the wheels (~2 GB during download, ~2.5 GB extracted into site-packages) all land on
`/mnt/nobackup`. While it's running, you can sanity-check from another shell:

```bash
du -sh /mnt/nobackup/jchen/conda_envs/vllm-v100/lib/python3.10/site-packages/torch
quota -s        # home should not be growing
```

Verify the GPU is visible from PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# Expect: True Tesla V100-... (7, 0)
```

If `(7, 0)` does not print, stop — the rest of this guide assumes a Volta device.

---

## 4. Install vLLM

```bash
python -m pip install vllm==0.5.4
```

This pulls a matching `xformers` wheel automatically. If pip complains about a torch
version mismatch, recreate the env and reinstall torch first (Section 3), then vLLM.

Smoke-check the import:

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 5. Get Llama 3 Weights

`meta-llama/Meta-Llama-3-8B-Instruct` is **gated**. You need to:

1. Accept the license at <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>.
2. Create a token at <https://huggingface.co/settings/tokens> (read scope is enough).
3. Log in from the shell:

```bash
python -m pip install -U "huggingface_hub[cli]"
huggingface-cli login    # paste the token
```

Pre-download so the first vLLM run doesn't stall on a 16 GB pull. Because `HF_HOME`
points to `/mnt/nobackup/jchen/`, the weights land there automatically:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
# verify it landed on /mnt/nobackup
du -sh /mnt/nobackup/jchen/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct
```

**Ungated alternative** (same weights, no license gate, useful for quick testing):
`NousResearch/Meta-Llama-3-8B-Instruct`. Substitute that string anywhere the official
model id appears below.

> Note: passing the model **id** (e.g. `"meta-llama/Meta-Llama-3-8B-Instruct"`) to
> `LLM(...)` is correct — vLLM resolves it through `HF_HOME` and reuses the cached
> snapshot. You don't need to hardcode the absolute `/mnt/nobackup/...` path. If you
> prefer a fully explicit path, `huggingface-cli download ... --local-dir
> /mnt/nobackup/jchen/llama3_models/Meta-Llama-3-8B-Instruct` works too, and you'd then pass
> that directory as `model=`.

---

## 6. Smoke Test (offline inference)

Create `vllm_smoke.py`:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="float16",            # V100 has no BF16
    gpu_memory_utilization=0.90,
    max_model_len=4096,         # lower if you OOM
    enforce_eager=True,         # avoid CUDA-graph issues on Volta
    # tensor_parallel_size=2,   # uncomment for two 16 GB V100s
)

sampling = SamplingParams(temperature=0.7, max_tokens=128)
out = llm.generate(
    ["Explain in two sentences why V100 cannot run Flash Attention 2."],
    sampling,
)
print(out[0].outputs[0].text)
```

Run it with the xFormers backend forced on:

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
python vllm_smoke.py
```

You should see ~50 tokens of generated text. If you get a Flash Attention error,
the env var didn't take effect — re-export it in the same shell.

---

## 7. Throughput Benchmark

This batches 64 prompts of 128 output tokens and reports tokens/sec. It's
intentionally minimal so it works without cloning the vLLM repo.

Create `vllm_bench.py`:

```python
import time
from vllm import LLM, SamplingParams

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
N_PROMPTS = 64
MAX_TOKENS = 128

llm = LLM(
    model=MODEL,
    dtype="float16",
    gpu_memory_utilization=0.90,
    max_model_len=2048,
    enforce_eager=True,
)

prompts = ["Write a short story about a robot learning to paint."] * N_PROMPTS
sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)

# Warm up (JITs kernels, fills KV cache pool)
llm.generate(prompts[:4], sampling)

t0 = time.time()
outputs = llm.generate(prompts, sampling)
elapsed = time.time() - t0

total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f"Model:       {MODEL}")
print(f"Requests:    {N_PROMPTS}")
print(f"Out tokens:  {total_out_tokens}")
print(f"Elapsed:     {elapsed:.2f} s")
print(f"Throughput:  {total_out_tokens / elapsed:.2f} output tok/s")
print(f"Req/s:       {N_PROMPTS / elapsed:.2f}")
```

Run:

```bash
VLLM_ATTENTION_BACKEND=XFORMERS python vllm_bench.py
```

Rough expectations on a single 32 GB V100 (FP16, eager, xFormers):

| Metric              | Ballpark           |
|---------------------|--------------------|
| Output throughput   | ~600–1100 tok/s    |
| Per-request latency | dominated by batch |

Numbers vary with prompt length, batch size, and `max_model_len`. Treat the script
above as a starting point — bump `N_PROMPTS` to saturate the scheduler.

---

## 8. (Optional) Run vLLM as an OpenAI-compatible Server

For an HTTP endpoint instead of offline batch:

```bash
VLLM_ATTENTION_BACKEND=XFORMERS \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --port 8000
```

Test with curl:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "prompt": "Hello,",
    "max_tokens": 32
  }'
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Disk quota exceeded` during `pip install torch` | `envs_dirs` not pointing at `/mnt/nobackup` (Section 1c) **or** pip cache not redirected (Section 1d). Run `which python` — if it's under `/home/...`, the env is in the wrong place. Remove with `conda env remove -n vllm-v100`, redo Section 1c, then Section 2. |
| `bfloat16 is only supported on GPUs with compute capability >= 8.0` | Pass `dtype="float16"` (or `--dtype half`). |
| `FlashAttention only supports Ampere GPUs or newer` | `export VLLM_ATTENTION_BACKEND=XFORMERS`. |
| OOM on 16 GB V100 | Lower `max_model_len` (e.g. 2048), lower `gpu_memory_utilization` to 0.85, set `tensor_parallel_size=2`, or load an AWQ build (`python -m pip install autoawq` and use a community `*-AWQ` repo). |
| CUDA graph capture failure / illegal memory access | Keep `enforce_eager=True`. |
| 401 / gated repo | `huggingface-cli whoami` to confirm login; recheck license acceptance, or switch to `NousResearch/Meta-Llama-3-8B-Instruct`. |
| `xformers` wheel mismatch on install | Recreate the env, install the exact torch version in Section 3 first, then `python -m pip install vllm==0.5.4`. |
| `Defaulting to user installation because normal site-packages is not writeable` (pip writes to `~/.local/lib/python3.8/`) | The shell is using system Python 3.8, not the env's Python 3.10 — `conda activate` updated only the prompt. Re-run Section 2b (`source /usr/local/pkgs/anaconda/etc/profile.d/conda.sh && conda activate vllm-v100`), then verify Section 2c. Always invoke pip as `python -m pip` to be safe. |
| `pip install ...` says "Requirement already satisfied" but `python -c "import torch"` fails with `ModuleNotFoundError` | `pip` resolved to a stale `~/.local/bin/pip` shim targeting an old Python (e.g. 3.8) while `python` resolves to the env's 3.10 — they're out of sync. Use `python -m pip install ...` instead, or `rm -f ~/.local/bin/pip ~/.local/bin/pip3 ~/.local/bin/pip3.*` to remove the shims. Also clean any stale `~/.local/lib/python3.X/site-packages/torch` (was 1.2 GB on this server) — it's a dead duplicate now. |
| Conda hook not sourced in VSCode terminal | VSCode's integrated terminal uses an `--init-file` that often skips `~/.bashrc`. Either `source /usr/local/pkgs/anaconda/etc/profile.d/conda.sh` at the start of each shell, or add `[ -f ~/.bashrc ] && source ~/.bashrc` to `~/.bash_profile`, or set the VSCode terminal profile to `bash -l`. |
| Disk fills up in `$HOME` after downloading the model | `HF_HOME` wasn't exported. `echo $HF_HOME` should print `/mnt/nobackup/jchen/hf_cache`. Re-run the activate.d snippet from Section 2c, then `rm -rf ~/.cache/huggingface` to reclaim space. |
| `conda env list` shows env under `/home/...` despite `--prepend` | Check `cat ~/.condarc` directly — the prepend may have been a no-op if the line was already present. Edit `~/.condarc` by hand to ensure `/mnt/nobackup/jchen/conda_envs` is the first entry under `envs_dirs`. |

---

## Layout summary (after a successful setup)

```
$HOME (≤ 12 GB quota)
├── ~/.condarc                          # points conda at /mnt/nobackup
├── ~/.config/pip/pip.conf              # points pip cache at /mnt/nobackup
└── ~/llama3_on_v100/                   # this guide + scripts (small)

/mnt/nobackup/jchen/  (TB-scale, no quota)
├── conda_envs/vllm-v100/               # the env (~7–8 GB after install)
├── conda_pkgs/                         # conda pkg tarball cache
├── pip_cache/                          # pip wheel cache
└── hf_cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/   # ~16 GB
```

---

## What's Next

The companion guide `README_llamacpp.md` covers the same task with **llama.cpp**,
which is more forgiving on older GPUs and supports tighter quantizations
(Q4_K_M, Q5_K_M) that fit Llama-3-8B comfortably on a 16 GB V100.
