conda create -n blip3o python=3.11 -y
conda activate blip3o
uv pip install --upgrade pip setuptools
uv pip install -r requirements.txt
uv pip install -e .
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation