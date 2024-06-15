import json
from datasets import load_dataset
import os
from jinja2 import Template
import pandas as pd
import subprocess
from io import StringIO


pretrained = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Egyptian Arabic",
    "en": "English",
    "fr": "French",
    "ha": "Hausa",
    "ig": "Igbo",
    "rw": "Kinyarwanda",
    "mg": "Malagasy",
    "ny": "Chichewa",
    "om": "Afaan Oromoo",
    "pt": "Portuguese",
    "so": "Somali",
    "sn": "Shona",
    "st": "Sesotho",
    "sw": "Swahili",
    "ti": "Tigrinya",
    "xh": "Xhosa",
    "yo": "Yoruba",
    "zu": "Zulu",
}

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

d = {v: k for k, v in pretrain_langs.items()}

d_map = {v: d[pretrained[v]] for v in pretrained}

senti_prompts = [
    "Classify the sentiment expressed in the following '{{ text }}':",
    "Is the sentiment in this '{{ text }}' positive, negative, or neutral?",
    "Analyze the sentiment expressed in the following tweet'{{ text }}'\nOptions: positive, negative, neutral",
    "Rate the sentiment of the tweet as either positive, negative, or neutral.\n {{ text }}",
    "What emotions are expressed in the following '{{ text }}'?\nClassify the emotion as either positive, negative, or neutral.",
]

afrisenti_langs = ['amh', 'hau', 'ibo', 'arq', 'ary', 'yor', 'por', 'twi', 'tso', 'tir', 'orm', 'pcm', 'kin', 'swa']
nollysenti_langs = ['ha', 'ig', 'en', 'yo']
labels = {0: 'positive', 1: 'neutral', 2: 'negative'}

def wget_content(url):
    try:
        # Call wget command and capture output
        output = subprocess.check_output(["wget", "-qO-", url])
        return output.decode('utf-8')  # Convert bytes to string
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        return None


if __name__ == '__main__':
    afrisenti_data = []
    # for lang in afrisenti_langs:
    #     if lang not in pretrain_langs:
    #         continue
    #     print("Processing language: ", pretrain_langs[lang])
    #     dataset = load_dataset("shmuhammad/AfriSenti-twitter-sentiment", lang)
    #     for split in ['train', 'validation', 'test']:
    #         for i in range(len(dataset[split])):
    #             for prompt in senti_prompts:
    #                 template = Template(prompt)
    #                 rendered = template.render(text = dataset[split][i]['tweet'])
    #                 afrisenti_data.append({
    #                             'instruction': rendered,
    #                             'output': labels[dataset[split][i]['label']],
    #                             'lang': lang,
    #                             'split': 'dev' if split == 'validation' else split,
    #                             'source': 'AfriSenti',
    #                             'task': 'Sentiment Analysis',
    #                         })
                    
    github_link = 'https://raw.githubusercontent.com/IyanuSh/NollySenti/main/data/'

    for lang in nollysenti_langs:
        for split in ['train', 'dev', 'test']:
            url = github_link + lang + '/' + split + '.tsv'
            data = StringIO(wget_content(url))
            df = pd.read_csv(data, sep='\t')
            col1 = df.columns.to_list()[0]
            for index, row in df.iterrows():
                for prompt in senti_prompts:
                    template = Template(prompt)
                    rendered = template.render(text = row[col1])
                    if type(row['sentiment']) == str:
                        afrisenti_data.append({
                                    'instruction': rendered,
                                    'output': row['sentiment'],
                                    'lang': d_map[lang],
                                    'split': split,
                                    'source': 'NollySenti',
                                    'task': 'Sentiment Analysis',
                                })
                    else:
                        print("Error in row")

    with open('senti_small.json', 'w') as f:
        json.dump(afrisenti_data, f, ensure_ascii=False) 
