# Step 2 host setup: g6e.48xlarge (8x L40S), Deep Learning AMI with Docker + NVIDIA runtime.
# Run as ubuntu: bash host_setup_step2.sh 2>&1 | tee /data/logs/setup.log
set -x
# /data on the local NVMe instance store (RAID0 over all of them) if not already mounted
if ! mountpoint -q /data; then
  DEVS=$(lsblk -dpno NAME,MODEL | grep -i "Instance Storage" | awk '{print $1}')
  N=$(echo $DEVS | wc -w)
  if [ "$N" -gt 1 ]; then sudo mdadm --create /dev/md0 --level=0 --raid-devices=$N $DEVS --run; DEV=/dev/md0; else DEV=$DEVS; fi
  sudo mkfs.ext4 -q -F $DEV && sudo mkdir -p /data && sudo mount $DEV /data && sudo chown ubuntu:ubuntu /data
fi
mkdir -p /data/logs /data/results /data/hf
python3 -m venv /data/venv && /data/venv/bin/pip -q install -U "huggingface_hub[hf_xet]" aiohttp transformers pandas pyarrow
python3 -m venv /data/aiperf-venv && /data/aiperf-venv/bin/pip -q install aiperf==0.14.0
HF_HOME=/data/hf /data/venv/bin/hf download Qwen/Qwen3-32B --exclude "*.pth" &
HF_HOME=/data/hf /data/venv/bin/hf download openai/gsm8k --repo-type dataset --include "main/*" --local-dir /data/hf/gsm8k &
docker pull vllm/vllm-openai:v0.34.0 &
docker pull lmsysorg/sglang:v0.5.23 &
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0 &
docker pull nginx:1.27.3-alpine &
wait
nvidia-smi topo -m
docker images --digests | grep -E "vllm|sglang|tensorrt|nginx"
du -sh /data/hf
/data/aiperf-venv/bin/aiperf --version
echo SETUP_DONE
