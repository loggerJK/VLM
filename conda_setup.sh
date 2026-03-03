conda env create -f ./blip.yaml -y
conda activate blip3
pip install uv
uv pip install -e .

