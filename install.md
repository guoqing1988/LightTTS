git clone --recursive https://github.com/ModelTC/LightTTS.git
cd LightTTS
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
