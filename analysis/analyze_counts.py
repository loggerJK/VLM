import os
from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define paths and config
DATASETS = {
    "pixmo_count_filtered": {
        "path": "/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-count-filtered-imgContained/train",
        "max_bin": 10
    },
    "pixmo_point_count_concat": {
        "path": "/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-point-count-concatenated-dataset-qaFixed/train",
        "max_bin": 10
    },
    "pixmo_points_filtered_0-20": {
        "path": "/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-points-filtered_0-20_imgContained/train",
        "max_bin": 20
    }
}

OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_counts(counts, max_bin=10):
    """
    Bins counts into 0, 1, ..., max_bin, and '>max_bin'.
    Returns a DataFrame with 'Label', 'Count', 'Ratio'.
    """
    counts = np.array(counts)
    
    # Bins: 0 to max_bin individually
    bins_counts = {}
    for i in range(max_bin + 1):
        bins_counts[str(i)] = np.sum(counts == i)
    
    # Rest (>max_bin)
    bins_counts[f'>{max_bin}'] = np.sum(counts > max_bin)
    
    # Convert to DataFrame
    labels = [str(i) for i in range(max_bin + 1)] + [f'>{max_bin}']
    values = [bins_counts[l] for l in labels]
    
    df = pd.DataFrame({'Label': labels, 'Count': values})
    total = df['Count'].sum()
    df['Ratio'] = df['Count'] / total
    
    return df

def plot_and_save(df, dataset_name, max_bin):
    """
    Plots Ratio and Absolute Count histograms and saves them.
    Adjusts figure width based on max_bin to ensure labels are readable.
    """
    # width = 10 if max_bin <= 10 else 15  # Make it wider for more bins
    width = 7
    
    # 1. Absolute Count
    plt.figure(figsize=(width, 6))
    plt.bar(df['Label'], df['Count'], color='skyblue', edgecolor='black')
    plt.title(f'Count Distribution (Absolute) - {dataset_name} (0-{max_bin})')
    plt.xlabel('Count Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels
    for i, v in enumerate(df['Count']):
        plt.text(i, v + (v * 0.01), str(v), ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_hist_absolute.png")
    plt.savefig(save_path)
    print(f"Saved absolute histogram to {save_path}")
    plt.close()

    # 2. Ratio
    plt.figure(figsize=(width, 6))
    plt.bar(df['Label'], df['Ratio'], color='lightgreen', edgecolor='black')
    plt.title(f'Count Distribution (Ratio) - {dataset_name} (0-{max_bin})')
    plt.xlabel('Count Value')
    plt.ylabel('Ratio')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels
    for i, v in enumerate(df['Ratio']):
        plt.text(i, v + 0.005, f"{v:.1%}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_hist_ratio.png")
    plt.savefig(save_path)
    print(f"Saved ratio histogram to {save_path}")
    plt.close()

def main():
    for name, config in DATASETS.items():
        path = config["path"]
        max_bin = config["max_bin"]
        
        print(f"Processing {name} from {path} with max_bin={max_bin}...")
        try:
            ds = load_from_disk(path)
            
            # Check for 'count' column
            if 'count' not in ds.column_names:
                print(f"Warning: 'count' column not found in {name}. Available columns: {ds.column_names}")
                continue
                
            counts = ds['count']
            df = process_counts(counts, max_bin=max_bin)
            
            # Print stats
            print(f"Statistics for {name}:")
            print(df)
            
            plot_and_save(df, name, max_bin)
            
        except Exception as e:
            print(f"Failed to process {name}: {e}")

if __name__ == "__main__":
    main()