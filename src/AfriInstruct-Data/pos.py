import json
import os
from jinja2 import Template
import pandas as pd
import subprocess
from io import StringIO

pretrain_langs = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "ara": "Egyptian Arabic",
    "eng": "English",
    "fra": "French",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "mlg": "Malagasy",
    "nya": "Chichewa",
    "orm": "Afaan Oromoo",
    "por": "Portuguese",
    "som": "Somali",
    "sna": "Shona",
    "sot": "Sesotho",
    "swa": "Swahili",
    "tir": "Tigrinya",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
}

pos_langs = ['bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'mos', 'nya', 'pcm', 'sna', 'swa', 'twi', 'wol', 'xho', 'yor', 'zul']
all_langs = []
# all_langs = ['hau', 'ibo', 'kin', 'nya', 'sna', 'swa', 'xho', 'yor', 'zul']

TEMPLATE = """Study this taxonomy for classifying parts of speech:
- X: Other
- ADJ: Adjective
- ADP: Adposition
- ADV: Adverb
- AUX: Auxiliary verb
- CCONJ: Coordinating conjunction
- DET: Determiner
- INTJ: Interjection
- NOUN: Noun
- NUM: Numeral
- PART: Particle
- PRON: Pronoun
- PROPN: Proper noun
- PUNCT: Punctuation
- SCONJ: Subordinating conjunction
- SYM: Symbol
- VERB: Verb
Perform Part-of-Speech (POS) tagging on the following tokens:
{{ tokens| join('\n') }}
Answer: """

def wget_data(url):
    try:
        # Call wget command and capture output
        output = subprocess.check_output(["wget", "-qO-", url])
        return output.decode('utf-8')  # Convert bytes to string
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        return None

def create_pos():
    dataset = []
    github_link = "https://raw.githubusercontent.com/masakhane-io/masakhane-pos/main/data/"
    template = Template(TEMPLATE)
    for lang in pos_langs:
        if lang not in pretrain_langs:
            continue
        all_langs.append(lang)
        print("Processing language: ", pretrain_langs[lang])
        for split in ['train', 'dev', 'test']:
            url = github_link + lang + '/' + split + '.txt'
            data = wget_data(url).split('\n\n')
            for i, sample in enumerate(data):
                if not sample:
                    continue
                tokens = [line.split()[0] for line in sample.split('\n')]
                rendered = template.render(tokens=tokens)
                dataset.append({
                            'instruction': rendered,
                            'output': sample,
                            'lang': lang,
                            'split': split,
                            'source': 'MasakhaPOS',
                            'task': 'POS',
                        })
    return dataset

if __name__ == '__main__':
    dataset = create_pos()
    print(all_langs)

    # save to json file with proper formatting
    with open('pos.json', 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)   