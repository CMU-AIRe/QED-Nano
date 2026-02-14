# export TOGETHER_API_KEY="<FIXME>"
# export OPENAI_API_KEY="<FIXME>"
# export GOOGLE_API_KEY="<FIXME>"
# export DEEPSEEK_API_KEY="<FIXME>"
export VLLM_API_KEY=EMPTY 

timestamp=$(date +%Y%m%d%H%M%S)
python -m vllm.entrypoints.openai.api_server \
  --model lm-provers/QED-Nano \
  --revision main \
  --host 0.0.0.0 \
  --port 8080 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 > /home/yuxiaoq/workspace/log/log/server_${timestamp}.log 2>&1 &
