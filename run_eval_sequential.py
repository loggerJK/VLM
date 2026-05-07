import subprocess
import os
import sys

# 설정
metadata_path = "/mnt/data1/jiwon/evaluation_metadata_count.jsonl"
script_path = "/mnt/data1/jiwon/Janus/geneval_janus_evalulate_multigpu.py"
base_out_dir = "/mnt/data1/jiwon/Janus/results_geneval_count_extended"

# 실행할 체크포인트 목록 (이미 완료된 첫 번째 제외)
checkpoints = [
    "/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-12000",
    "/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-point-count-concatenated]_ngpu2_bs1_accum64_lr4e-5_ep3_full/checkpoint-15000",
    "/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-points_0-20]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-15000",
    "/mnt/data1/jiwon/Janus/checkpoints/train[transformer]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600"
]

# 환경 변수 설정
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 현재 python 실행 경로 확인
print(f"Current Python: {sys.executable}")

for i, model_path in enumerate(checkpoints):
    parent_dir = os.path.basename(os.path.dirname(model_path))
    ckpt_name = os.path.basename(model_path)
    out_dir = os.path.join(base_out_dir, parent_dir, ckpt_name)
    
    print(f"================================================================")
    print(f"Starting evaluation [{i+1}/{len(checkpoints)}]")
    print(f"Model: {ckpt_name}")
    print(f"Path:  {model_path}")
    print(f"Output:{out_dir}")
    print(f"================================================================")

    # master_port를 순차적으로 변경
    port = str(29610 + i)
    
    # [수정됨] torchrun 바이너리 대신 python -m torch.distributed.run 사용
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
        print(f"Error occurred while evaluating {ckpt_name}")
        print(f"Return code: {e.returncode}")
        print("Moving to next checkpoint...\n")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Moving to next checkpoint...\n")

print("All tasks completed.")