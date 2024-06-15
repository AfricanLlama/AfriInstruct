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


news_langs = ['amh', 'eng', 'fra', 'hau', 'ibo', 'lin', 'lug', 'orm', 'pcm', 'run', 'sna', 'som', 'swa', 'tir', 'xho', 'yor']

all_langs = []
# all_langs = ['amh', 'eng', 'fra', 'hau', 'ibo', 'orm', 'sna', 'som', 'swa', 'tir', 'xho', 'yor']

prompts = {
    'title' : [
        "{{text}} \n\nWhich of the following sections of a newspaper would this article likely appear in? {{ answer_choices[0]| capitalize}}, {{ answer_choices[1:-1]| join(', ') }}, or {{ answer_choices[-1]}}?",
        "{{text}} \n\nWhich of these labels best describes this news article:\n{{ answer_choices| join(' $ ') }}\n\nLabel:",
        "Which of these labels best describes this news article:\n{{ answer_choices| join(' $ ') }}\n\n{{text}}\n\nLabel:"
    ],
    'text' : [
        "What topic does the following news title \"{{text}}\" belong to? {{ answer_choices[0]| capitalize}}, {{ answer_choices[1:-1]| join(', ') }}, or {{ answer_choices[-1]}}?",
        "Is this a piece of news regarding {{ answer_choices[0]| capitalize}}, {{ answer_choices[1:-1]| join(', ') }}, or {{ answer_choices[-1]}}?\n{{text}}",
    ]
}

def wget_data(url):
    try:
        # Call wget command and capture output
        output = subprocess.check_output(["wget", "-qO-", url])
        return output.decode('utf-8')  # Convert bytes to string
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        return None

def create_news():
    # load data
    dataset = []
    github_link = "https://raw.githubusercontent.com/masakhane-io/masakhane-news/main/data/"
    for lang in news_langs:
        if lang not in pretrain_langs:
            continue
        print("Processing language: ", pretrain_langs[lang])
        all_langs.append(lang)
        for split in ['train', 'dev', 'test']:
            url = github_link + lang + '/' + split + '.tsv'
            data = StringIO(wget_data(url))
            df = pd.read_csv(data, sep='\t')
            labels = df['category'].unique()
            # iterate over rows
            for index, row in df.iterrows():
                for div in prompts:
                    for prompt in prompts[div]:
                        template = Template(prompt)
                        rendered = template.render(text=row['headline'], answer_choices=labels)
                        dataset.append({
                            'instruction': rendered,
                            'output': row['category'],
                            'lang': lang,
                            'split': split,
                            'source': 'MasakhaNEWS',
                            'task': 'news_topic_classification',
                        })
    return dataset


if __name__ == '__main__':
    dataset = create_news()
    print(all_langs)

    # save to json file with proper formatting
    with open('news.json', 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)   


            