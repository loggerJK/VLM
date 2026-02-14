# %%
from datasets import load_dataset, load_from_disk, concatenate_datasets
from datasets import ClassLabel, Value

# data1_path = '/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-count-filtered-imgContained-castInt64/train'
# data2_path = '/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-points-filtered_0-20_imgContained/train' 

# save_path = '/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-point-count-concatenated-dataset/train'


# dataset1 = load_from_disk(data1_path)
# dataset2 = load_from_disk(data2_path)

# # %%
# from datasets import load_dataset, load_from_disk, concatenate_datasets
# from datasets import ClassLabel, Value

# target_columns = ['image', 'question', 'answer', 'count',]

# # to int64
# # dataset1 = dataset1.cast_column('count', Value('int64'))
# # dataset2 = dataset2.cast_column('count', Value('int64'))

# dataset1 = dataset1.remove_columns([col for col in dataset1.column_names if col not in target_columns])
# dataset2 = dataset2.remove_columns([col for col in dataset2.column_names if col not in target_columns])

# combined_dataset = concatenate_datasets([dataset1, dataset2])

# %%
# combined_dataset_filtered = combined_dataset.filter(lambda example : example['count'] <= 10, num_proc=16)
# print(f"len(combined_dataset_filtered): {len(combined_dataset_filtered)}")

# combined_dataset_filtered.save_to_disk(save_path)

############## Combined dataset Filtering. ##############

combined_dataset_path = '/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-point-count-concatenated-dataset/train'
save_path = '/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-point-count-concatenated-dataset-qaFixed/train'

combined_dataset = load_from_disk(combined_dataset_path)

combined_dataset_filtered = combined_dataset.map(lambda example: {
    "question": example["question"].replace("Response Example : There are **<number>** of", "Response Example : There are **<number>**"),
    "answer": example["answer"].replace("** of", "**"),
}, num_proc=32)

print(f"len(combined_dataset_filtered): {len(combined_dataset_filtered)}")
combined_dataset_filtered.save_to_disk(save_path)

print("Dataset saved.")

print(combined_dataset_filtered[0])

print ("Pushing to Hub...")
combined_dataset_filtered.push_to_hub("Jiwon-Kang/pixmo-point-count-concatenated-dataset-qaFixed")
