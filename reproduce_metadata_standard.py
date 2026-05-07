import json

# 기존 5개 체크포인트가 이미 사용한 클래스 구성
classes = [
    "person", "dog", "cat", "car", "cup", 
    "chair", "book", "bottle", "apple", "tie"
]

num_to_word = {
    2: "two", 3: "three", 4: "four", 5: "five", 
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"
}

def get_plural(word):
    if word == "person": return "persons"
    if word == "tie": return "ties"
    return word + "s"

output_file = "/mnt/data1/jiwon/evaluation_metadata_count.jsonl"

with open(output_file, "w") as f:
    for count in range(2, 11): # 2 to 10
        for cls in classes:
            prompt = f"a photo of {num_to_word[count]} {get_plural(cls)}"
            
            entry = {
                "tag": "counting",
                "include": [{"class": cls, "count": count}],
                "exclude": [{"class": cls, "count": count + 1}],
                "prompt": prompt
            }
            f.write(json.dumps(entry) + "\n")

print(f"Generated 90 entries in {output_file} (Standard set for all checkpoints)")
