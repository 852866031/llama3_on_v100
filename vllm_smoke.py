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