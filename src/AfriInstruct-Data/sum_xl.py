from jinja2 import Template
import json
import pandas as pd
from tqdm import tqdm
from random import randint
from globals import LANG_CODES

TEMPLATES = [
    {"template": Template("{{ text }}\n\n===\n\nWrite a summary of the text above in {{ lang }}:"), "name": "of_above"},
    {"template": Template("{{ text }}\n\nTL;DR in {{ lang }}:"), "name": "tldr"},
    {"template": Template("Article in {{ lang }}: {{ text }}\n\nSummary in {{ lang }}:"), "name": "article"},
]

AVAILABLE_DATA = [
    "amh",
    "ara",
    "eng",
    "hau",
    "ibo",
    "orm",
    "por",
    "swa",
    "tir",
    "yor",
]

def convert_xlsum(lang_code):
    
    data = pd.read_json(path_or_buf=f"data/XLSUM/{lang_code}_train.jsonl", lines=True).to_dict(orient="records")
    lang = LANG_CODES[lang_code]
    result = []
    num = min(len(data), 10000)

    for i in range(num):

        text = data[i]["text"]
        summary = data[i]["summary"]
        template = TEMPLATES[randint(0, len(TEMPLATES) - 1)]

        result.append({
                "instruction": template["template"].render(text=text, lang=lang),
                "output": summary,
                "lang": lang_code,
                "split": "train",
                "source": "XL-Sum",
                "task": "Summarization",
        })

    return result

if __name__ == "__main__":
    result = []
    for lang in tqdm(AVAILABLE_DATA):

        out = convert_xlsum(lang)
        result.extend(out)

    with open(f"data-new/XLSUM_dataset.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)