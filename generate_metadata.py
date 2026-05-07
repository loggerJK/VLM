import json

classes = ["person", "dog", "car", "chair", "cup", "bird", "apple", "book", "clock", "boat"]
counts = range(2, 11) # 2 to 10

num_to_word = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten"
}

def get_plural(cls):
    if cls == "person":
        return "persons" # Based on observed pattern in existing metadata
    return cls + "s"

output_file = "/mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl"

with open(output_file, "w") as f:
    for cls in classes:
        for count in counts:
            word = num_to_word[count]
            plural = get_plural(cls)
            prompt = f"a photo of {word} {plural}"
            
            entry = {
                "tag": "counting",
                "include": [{"class": cls, "count": count}],
                "exclude": [{"class": cls, "count": count + 1}],
                "prompt": prompt
            }
            f.write(json.dumps(entry) + "\n")

print(f"Generated {len(classes) * len(counts)} prompts in {output_file}")

