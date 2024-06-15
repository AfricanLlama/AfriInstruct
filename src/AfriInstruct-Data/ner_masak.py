from jinja2 import Template
import json
from tqdm import tqdm
from ner import TEMPLATE

AVAILABLE_DATA = [
    "hau",
    "ibo",
    "kin",
    "nya",
    "sna",
    "swa",
    "xho",
    "yor",
    "zul",
]

def convert_maskner(lang, split="train"):

    def group(lines):
        cur_group = []
        result = []
        for line in lines:
            if line == "\n":
                result.append(cur_group)
                cur_group = []
            else:
                cur_group.append(line.strip().split(" "))

        return result

    def to_tokens(group):
        result = "["
        for token in group:
            result += f'"{token[0]}", '
        result = result[:-2] + "]"
        return result

    def to_labelled_tokens(group):
        result = "["
        for token in group:
            try:
                result += f'("{token[0]}", "{token[1]}"), '
            except Exception as e:
                print(group)
        result = result[:-2] + "]"
        return result
    
    lines = []
    with open(f"data/MASAKANER2/{lang}/{split}.txt", "r") as f:
        lines = f.readlines()

    grouped = group(lines)
    tokens = list(map(to_tokens, grouped))
    labelled_tokens = list(map(to_labelled_tokens, grouped))

    template = Template(TEMPLATE)
    result = []

    for i in range(len(tokens)):
        result.append({
                "instruction": template.render(tokens=tokens[i]),
                "output": labelled_tokens[i],
                "lang": lang,
                "split": split,
                "source": "MasakhaNER2.0",
                "task": "NER",
        })

    return result

if __name__ == "__main__":
    result = []
    for lang in tqdm(AVAILABLE_DATA):
        splits = ["train", "dev", "test"]

        for split in splits:
            out = convert_maskner(lang, split)
            result.extend(out)

    with open(f"data-new/MASKANER_dataset.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)