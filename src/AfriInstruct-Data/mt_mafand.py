import pandas as pd
import json
from tqdm import tqdm
from globals import LANG_CODES
from translation import get_prompt

AVAILABLE_DATA = [
    {"s_code": "eng", "t_code": "amh", "splits": ["dev", "test"]},
    {"s_code": "eng", "t_code": "hau", "splits": ["train", "dev", "test"]},
    {"s_code": "eng", "t_code": "ibo", "splits": ["train", "dev", "test"]},
    {"s_code": "eng", "t_code": "kin", "splits": ["dev", "test"]},
    {"s_code": "eng", "t_code": "nya", "splits": ["dev", "test"]},
    {"s_code": "eng", "t_code": "sna", "splits": ["dev", "test"]},
    {"s_code": "eng", "t_code": "swa", "splits": ["train", "dev", "test"]},
    {"s_code": "eng", "t_code": "xho", "splits": ["dev", "test"]},
    {"s_code": "eng", "t_code": "yor", "splits": ["train", "dev", "test"]},
    {"s_code": "eng", "t_code": "zul", "splits": ["train", "dev", "test"]},
]


def convert_mafand(s_code, t_code, split="train"):
    
    df = pd.read_csv(f"data/MAFAND/{s_code}-{t_code}/{split}.tsv", delimiter='\t')
    result = []

    for i in range(len(df)):
        s_text = df[s_code][i]
        t_text = df[t_code][i]
        s = "train" if split == "train" or split == "test" else "dev"
        result.append(get_prompt(LANG_CODES[t_code], LANG_CODES[s_code], t_text, s_text, t_code, s_code, "MAFAND", s))
    
    return result

if __name__ == "__main__":

    result = []
    for data in tqdm(AVAILABLE_DATA):
        s_code = data["s_code"]
        t_code = data["t_code"]
        splits = data["splits"]

        for split in splits:
            result.extend(convert_mafand(s_code, t_code, split))

    with open(f"data-new/MAFAND_dataset.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)

