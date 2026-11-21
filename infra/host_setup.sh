set -x
python3 -m venv /data/venv && /data/venv/bin/pip -q install -U "huggingface_hub[hf_xet]"
HF_HOME=/data/hf /data/venv/bin/hf download Qwen/Qwen3-8B --exclude "*.pth" &
docker pull vllm/vllm-openai:v0.34.0
docker pull lmsysorg/sglang:v0.5.23
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0
wait
docker images --digests | grep -E "vllm|sglang|tensorrt"
du -sh /data/hf
echo SETUP_DONE
