import json

data_list = [
    {
        "input_images": ["gemini_t2i_pig.png", ],
        "output_image": "gemini_pig_remove_hat.png",
        "instruction": "remove the pig's hat"
    },
    {
        "input_images": ["gemini_pig_remove_hat.png", "gemini_t2i_sunglasses.png"],
        "output_image": "gemini_combine.png",
        "instruction": "let the animal in the first image wear the item in the second image"
    }
]


file_path = "qwen_image_edit/example/data.jsonl"
with open(file_path, "w", encoding="utf-8") as f:
    for item in data_list:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
