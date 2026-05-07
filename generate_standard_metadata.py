import json

# 공통 클래스 구성
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

# 이 파일이 모든 인퍼런스의 '진짜' 기준이 됨
output_file = "/mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl"

with open(output_file, "w") as f:
    for count in range(2, 11): # 2 to 10 (Outer loop: Count)
        for cls in classes:    # (Inner loop: Class)
            prompt = f"a photo of {num_to_word[count]} {get_plural(cls)}"
            
            entry = {
                "tag": "counting",
                "include": [{"class": cls, "count": count}],
                "exclude": [{"class": cls, "count": count + 1}],
                "prompt": prompt
            }
            f.write(json.dumps(entry) + "\n")

print(f"Standard metadata generated at: {output_file}")
print("Order: Count-first (2-10), then Class (person, dog, cat, car, cup, chair, book, bottle, apple, tie)")
