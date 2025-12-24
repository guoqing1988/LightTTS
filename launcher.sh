# Enabling NVIDIA's multiprocessing services, Sometimes it can accelerate
nvidia-cuda-mps-control -d
# python -m light_tts.server.api_server --model_dir ./pretrained_models/CosyVoice2-0.5B
python -m light_tts.server.api_server --model_dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512 --port 8080