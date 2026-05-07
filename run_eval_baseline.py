import subprocess
import os
import sys

# 공통 메타데이터 경로 (다른 5개 체크포인트와 동일)
metadata_path = "/mnt/data1/jiwon/evaluation_metadata_count.jsonl"
script_path = "/mnt/data1/jiwon/Janus/geneval_janus_evalulate_multigpu.py"
base_out_dir = "/mnt/data1/jiwon/Janus/results_geneval_count_extended"

model_name = "deepseek-ai/Janus-Pro-7B"
ckpt_name = "janus_pro_7b_baseline"
out_dir = os.path.join(base_out_dir, ckpt_name)

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

print(f"Current Python: {sys.executable}")
print(f"================================================================")
print(f"Final Inference for Baseline using Standard Metadata (cat, tie, etc.)")
print(f"Model: {model_name}")
print(f"Output: {out_dir}")
print(f"================================================================")

port = "29640"

cmd = [
    sys.executable, "-m", "torch.distributed.run",
    "--nproc_per_node=4",
    f"--master_port={port}",
    script_path,
    metadata_path,
    "--model", model_name,
    "--outdir", out_dir
]

try:
    subprocess.run(cmd, env=env, check=True)
    print(f"Successfully finished final inference for {ckpt_name}\n")
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
