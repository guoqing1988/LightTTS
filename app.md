git clone --recursive https://github.com/ModelTC/LightTTS.git   
cd LightTTS
uv venv --python 3.11 
uv pip install tensorrt-cu12-libs tensorrt-cu12-bindings tensorrt-cu12 --extra-index-url https://pypi.nvidia.com
uv pip install onnxruntime-gpu==1.23.2 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --index-strategy unsafe-best-match
  
