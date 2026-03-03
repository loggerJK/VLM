from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BLIP3o/BLIP3o-Model-8B",
    repo_type="model",
    local_dir="BLIP3o/BLIP3o-Model-8B",
    local_dir_use_symlinks=False,
)
