### 安装
```
git clone --recursive https://github.com/ModelTC/LightTTS.git   
cd LightTTS
uv venv --python 3.11 
uv pip install tensorrt-cu12-libs tensorrt-cu12-bindings tensorrt-cu12 --extra-index-url https://pypi.nvidia.com
uv pip install onnxruntime-gpu==1.23.2 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --index-strategy unsafe-best-match
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple --index-strategy unsafe-best-match
```


### 运行
```
python -m light_tts.server.api_server --host 0.0.0.0 --port 8080 --data_type float16 --load_trt True --gpu_memory_utilization 0.2 --max_model_len 2048 --model_dir /data/www/ComfyUI/models/cosyvoice/Fun-CosyVoice3-0.5B/FunAudioLLM/Fun-CosyVoice3-0.5B-2512                                        
```
