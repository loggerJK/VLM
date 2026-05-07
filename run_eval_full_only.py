import subprocess
import os
import sys

# 설정
metadata_path = "/mnt/data1/jiwon/evaluation_metadata_count.jsonl"
script_path = "/mnt/data1/jiwon/Janus/geneval_janus_evalulate_multigpu.py"
base_out_dir = "/mnt/data1/jiwon/Janus/results_geneval_count_extended"

# 누락된 체크포인트
checkpoints = [
    "/mnt/data1/jiwon/Janus/checkpoints/train[full]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600"
]

# 환경 변수 설정
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

print(f"Current Python: {sys.executable}")

for i, model_path in enumerate(checkpoints):
    parent_dir = os.path.basename(os.path.dirname(model_path))
    ckpt_name = os.path.basename(model_path)
    out_dir = os.path.join(base_out_dir, parent_dir, ckpt_name)
    
    print(f"================================================================")
    print(f"Starting evaluation for: {model_path}")
    print(f"Output: {out_dir}")
    print(f"================================================================")

    port = "29620" # Use a new port just in case
    
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=4",
        f"--master_port={port}",
        script_path,
        metadata_path,
        "--model", model_path,
        "--outdir", out_dir
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Successfully finished {ckpt_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
