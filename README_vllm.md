# vLLM + Llama 3 on V100 — Setup Guide

This guide installs vLLM in a fresh conda environment, runs Llama 3 inference on V100
GPUs, and runs a simple throughput benchmark.

> Tested target: NVIDIA V100 (Volta, compute capability **7.0**), 16 GB or 32 GB.

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

## 1. Create the Conda Environment

```bash
conda create -n vllm-v100 python=3.10 -y
conda activate vllm-v100
```

Python 3.10 is the safest choice for vLLM 0.5.x wheels.

### 1a. Pin model storage to `/mnt/nobackup/jchen/`

The home filesystem is too small for 16 GB model weights. Route the Hugging Face cache
to `/mnt/nobackup/jchen/` so every download (and every vLLM load) reads/writes there.

```bash
mkdir -p /mnt/nobackup/jchen/hf_cache
export HF_HOME=/mnt/nobackup/jchen/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub        # belt-and-suspenders for older libs
export TRANSFORMERS_CACHE=$HF_HOME/hub
```

Make it persistent for this conda env so you don't have to remember on every login:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/hf_cache.sh" <<'EOF'
export HF_HOME=/mnt/nobackup/jchen/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
EOF
```

After this, `conda activate vllm-v100` automatically points the cache at
`/mnt/nobackup/jchen/`. Verify:

```bash
echo $HF_HOME
# /mnt/nobackup/jchen/hf_cache
```

Models will land at `/mnt/nobackup/jchen/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/...`.

---

## 2. Install PyTorch (CUDA 12.1 build)

V100 supports CUDA 11.8 and 12.1. Use the cu121 wheel that vLLM 0.5.x is built against:

```bash
pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible from PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# Expect: True Tesla V100-... (7, 0)
```

If `(7, 0)` does not print, stop — the rest of this guide assumes a Volta device.

---

## 3. Install vLLM

```bash
pip install vllm==0.5.4
```

This pulls a matching `xformers` wheel automatically. If pip complains about a torch
version mismatch, recreate the env and reinstall torch first (Step 2), then vLLM.

Smoke-check the import:

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 4. Get Llama 3 Weights

`meta-llama/Meta-Llama-3-8B-Instruct` is **gated**. You need to:

1. Accept the license at <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>.
2. Create a token at <https://huggingface.co/settings/tokens> (read scope is enough).
3. Log in from the shell:

```bash
pip install -U "huggingface_hub[cli]"
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
> /mnt/nobackup/jchen/models/Meta-Llama-3-8B-Instruct` works too, and you'd then pass
> that directory as `model=`.

---

## 5. Smoke Test (offline inference)

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

## 6. Throughput Benchmark

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

## 7. (Optional) Run vLLM as an OpenAI-compatible Server

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
| `bfloat16 is only supported on GPUs with compute capability >= 8.0` | Pass `dtype="float16"` (or `--dtype half`). |
| `FlashAttention only supports Ampere GPUs or newer` | `export VLLM_ATTENTION_BACKEND=XFORMERS`. |
| OOM on 16 GB V100 | Lower `max_model_len` (e.g. 2048), lower `gpu_memory_utilization` to 0.85, set `tensor_parallel_size=2`, or load an AWQ build (`pip install autoawq` and use a community `*-AWQ` repo). |
| CUDA graph capture failure / illegal memory access | Keep `enforce_eager=True`. |
| 401 / gated repo | `huggingface-cli whoami` to confirm login; recheck license acceptance, or switch to `NousResearch/Meta-Llama-3-8B-Instruct`. |
| `xformers` wheel mismatch on install | Recreate the env, install the exact torch version in Step 2 first, then `pip install vllm==0.5.4`. |
| Disk fills up in `$HOME` after downloading the model | `HF_HOME` wasn't exported. `echo $HF_HOME` should print `/mnt/nobackup/jchen/hf_cache`. Re-run the activate.d snippet from Step 1a, then `rm -rf ~/.cache/huggingface` to reclaim space. |

---

## What's Next

The companion guide `README_llamacpp.md` covers the same task with **llama.cpp**,
which is more forgiving on older GPUs and supports tighter quantizations
(Q4_K_M, Q5_K_M) that fit Llama-3-8B comfortably on a 16 GB V100.
