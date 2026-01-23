git clone --recursive https://github.com/ModelTC/LightTTS.git   
uv venv --python 3.11

uv pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple \
  --index-strategy unsafe-best-match
