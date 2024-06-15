from translation import LANG_FLORES
from jinja2 import Template
from random import randint
import pandas as pd
import json

TEXT_FIRST = "Classify the text \"{{ text }}\" into the following topics:\n- {{ labels | join('\n- ') }}\nTopic: "
TEXT_LAST = "Given the topics of {{labels[:-1] | join(', ') }}, and {{ labels[-1] }}, specify which of them best represents the following text:\n{{ text }}\nBest:"

TEMPLATES = [{"template": Template(TEXT_FIRST), "name": "text_first"}, {"template": Template(TEXT_LAST), "name": "text_last"}]

def get_prompt(text, label, labels, lang_code, split):

    temp = randint(0, len(TEMPLATES) - 1)
    prompt = TEMPLATES[temp]
    
    return {
            "instruction": prompt["template"].render(text=text, labels=labels),
            "output": label,
            "lang": f"{lang_code}",
            "split": split,
            "source": "SIB-200",
            "task": "topic-classification",
        }

def get_labels(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def convert_sib():
    result = []

    for lang_code in LANG_FLORES.keys():
        
        flores_lang = LANG_FLORES[lang_code]
        path = f"sib-200/data/annotated/{flores_lang}"
        labels = get_labels(f"{path}/labels.txt")

        for split in ["train", "dev", "test"]:

            data_path = f"{path}/{split}.tsv"
            data = pd.read_csv(data_path, sep='\t').to_dict(orient="records")

            for i in range(len(data)):
                text = data[i]["text"]
                label = data[i]["category"]
                result.append(get_prompt(text, label, labels, lang_code, split))

    return result

if __name__ == "__main__":
    result = convert_sib()
    with open(f"data-new/SIB-200_dataset.json", "w") as f:
        json.dump(result, f, ensure_ascii=False)
