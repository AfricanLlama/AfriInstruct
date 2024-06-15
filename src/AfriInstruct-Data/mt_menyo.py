from datasets import load_dataset
from translation import get_prompt
import json

def get_dataset():
    dataset = load_dataset("menyo20k_mt")
    result = []

    for split in ["train", "validation", "test"]:
        data = dataset[split]

        for i in range(len(data)):
            s_code = "eng"
            t_code = "yor"
            s_text = data[i]["translation"]["en"]
            t_text = data[i]["translation"]["yo"]
            s_lang = "English"
            t_lang = "Yoruba"
            s = "train" if split == "train" or split == "test" else "dev"

            result.append(get_prompt(s_lang, t_lang, s_text, t_text, s_code, t_code, "MENYO", s))

    return result

if __name__ == "__main__":
    with open(f"data-new/MENYO_dataset.json", "w") as f:
        json.dump(get_dataset(), f, ensure_ascii=False)