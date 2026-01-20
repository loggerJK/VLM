import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


input_folder_list = [
    '/mnt/data1/heeji/geneval/results/lumina_base_eval',
    "/mnt/data1/heeji/geneval/lumina_lora128_counting",
    "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead",
    "/mnt/data1/heeji/geneval/lumina_base_512",
    "/mnt/data1/heeji/geneval/lumina_lora128_counting_res512",
    "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_res512",
    "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_epoch6-iter45311-step12000",
    "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_epoch8-iter41215-step15400",
       
]


overall_csv_path = '/mnt/data1/heeji/geneval/overall_summary_metrics.csv'
overall_csv_list = []

for input_folder in input_folder_list:
    
    input_file = os.path.join(input_folder, 'evaluation_results.jsonl')
    output_dir = input_folder
    
    
    # Data storage
    y_true = []
    y_pred = []
    deviations = []

    print(f"Reading results from: {input_file}")

    with open(input_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Ensure it's a counting task
                if data.get('tag') != 'counting':
                    continue
                    
                counts_info = data.get('counts_info', {})
                
                # Extract target and predicted counts
                # Assuming typically one key per counts_info for counting tasks
                for class_name, info in counts_info.items():
                    target = info['target']
                    pred = info['pred']
                    
                    y_true.append(target)
                    y_pred.append(pred)
                    deviations.append(abs(target - pred))
                    
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                continue

    if not y_true:
        print("No valid counting data found.")
        exit()

    # Calculate Metrics
    total_samples = len(y_true)
    accuracy = sum([1 for t, p in zip(y_true, y_pred) if t == p]) / total_samples
    mean_deviation = sum(deviations) / len(deviations)

    print(f"Total Samples: {total_samples}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Deviation: {mean_deviation:.4f}")

    # Confusion Matrix
    labels = sorted(list(set(y_true + y_pred)))
    # Filter labels to meaningful range (e.g. 2-10 as per previous examples, or just strict min/max)
    # Based on previous code, 2-10 seems standard for this benchmark
    labels = [l for l in labels if 2 <= l <= 10]
    if not labels:
        labels = sorted(list(set(y_true + y_pred))) # Fallback if filtering removes everything

    # Confusion Matrix (Absolute)
    cm_abs = confusion_matrix(y_true, y_pred, labels=labels)
    cm_save_path_abs = os.path.join(output_dir, 'confusion_matrix_abs.png')
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_abs, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='viridis')
    plt.xlabel('Predicted Count')
    plt.ylabel('Target Count')
    plt.title(f'Confusion Matrix (Abs) (Acc: {accuracy:.4f}, MAD: {mean_deviation:.4f})')
    plt.tight_layout()
    plt.savefig(cm_save_path_abs)
    print(f"Saved confusion matrix (abs) to: {cm_save_path_abs}")
    plt.close()

    # Confusion Matrix (Ratio)
    cm_ratio = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm_save_path_ratio = os.path.join(output_dir, 'confusion_matrix_ratio.png')

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_ratio, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='viridis', vmax=1.0)
    plt.xlabel('Predicted Count')
    plt.ylabel('Target Count')
    plt.title(f'Confusion Matrix (Ratio) (Acc: {accuracy:.4f}, MAD: {mean_deviation:.4f})')
    plt.tight_layout()
    plt.savefig(cm_save_path_ratio)
    print(f"Saved confusion matrix (ratio) to: {cm_save_path_ratio}")
    plt.close()

    # CSV Summary
    csv_path = os.path.join(output_dir, 'summary_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Mean Deviation'])
        # Using parent folder name as model name for now
        model_name = os.path.basename(output_dir)
        writer.writerow([model_name, f"{accuracy:.4f}", f"{mean_deviation:.4f}"])
        
    # Append to overall summary
    overall_csv_list.append([model_name, f"{accuracy:.4f}", f"{mean_deviation:.4f}"])

    print(f"Saved summary metrics to: {csv_path}")
    
# Write overall summary CSV
with open(overall_csv_path, 'w', newline='') as overall_csv_file:
    overall_writer = csv.writer(overall_csv_file)
    overall_writer.writerow(['Model', 'Accuracy', 'Mean Deviation'])
    for row in overall_csv_list:
        overall_writer.writerow(row)
