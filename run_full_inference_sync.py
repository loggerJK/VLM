import subprocess
import os
import sys

# 설정
metadata_path = "/mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl"
script_path = "/mnt/data1/jiwon/Janus/geneval_janus_evalulate_multigpu.py"
base_out_dir = "/mnt/data1/jiwon/Janus/results_geneval_count_extended"

# 모든 대상 모델 (5개 체크포인트 + 1개 베이스라인)
models = [
    ("deepseek-ai/Janus-Pro-7B", "janus_pro_7b_baseline"),
    ("/mnt/data1/jiwon/Janus/checkpoints/train[full]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600", "train_full_ckpt600"),
    ("/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-12000", "train_transformer_only_count_ckpt12000"),
    ("/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-point-count-concatenated]_ngpu2_bs1_accum64_lr4e-5_ep3_full/checkpoint-15000", "train_transformer_only_point_count_ckpt15000"),
    ("/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-points_0-20]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-15000", "train_transformer_only_points_ckpt15000"),
    ("/mnt/data1/jiwon/Janus/checkpoints/train[transformer]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600", "train_transformer_ckpt600"),
]

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

print(f"Current Python: {sys.executable}")
print(f"Metadata Path: {metadata_path}")

for i, (model_path, folder_name) in enumerate(models):
    out_dir = os.path.join(base_out_dir, folder_name)
    
    print(f"\n[{i+1}/{len(models)}] Processing Model: {folder_name}")
    print(f"Path: {model_path}")
    
    # Run Inference
    port = str(29700 + i)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=4", f"--master_port={port}",
        script_path, metadata_path, "--model", model_path, "--outdir", out_dir
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"  [OK] Inference finished for {folder_name}")
        
        # Immediate Verification
        subdirs = [d for d in os.listdir(out_dir) if d.isdigit() and os.path.isdir(os.path.join(out_dir, d))]
        if len(subdirs) == 90:
            print(f"  [SUCCESS] Verification passed: 90 folders created in {out_dir}")
        else:
            print(f"  [ERROR] Verification failed: Found {len(subdirs)} folders in {out_dir}, expected 90.")
            
    except Exception as e:
        print(f"  [CRITICAL] Task failed for {folder_name}: {e}")

print("\n--- ALL TASKS COMPLETED BY CVLAB22 AGENT ---")
