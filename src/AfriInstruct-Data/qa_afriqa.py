import json
import pandas as pd
from jinja2 import Template
from tqdm import tqdm
import os

# Template for generating prompts for the QA task
QA_TEMPLATE = Template(
    "Question in {{ lang }}: '{{ question }}'\n"
    "Translated question ({{ pivot_lang }}): '{{ translated_question }}'\n"
    "Provide the answer in {{ lang }} based on the information available."
)


# {"answer_pivot": {"answer_start": [-1], "text": ["yes"]}
# , "context": "The history of Egypt under the British lasts from 1882,
# when it was occupied by British forces during the Anglo-Egyptian War, until 1956 after the Suez Crisis,
# when the last British forces withdrew in accordance with the Anglo-Egyptian agreement of 1954.
# The first period of British rule (1882–1914) is often called the \"veiled protectorate\".
# During this time the Khedivate of Egypt remained an autonomous province of the Ottoman Empire,
# and the British occupation had no legal basis but constituted a \"de facto\" protectorate over the country.
# Egypt was thus not part of the British Empire. This state of affairs lasted until 1914 when the Ottoman Empire joined the First World War
# on the side of the Central Powers and Britain declared a protectorate over Egypt. The ruling khedive was deposed and his successor,
# Hussein Kamel, compelled to declare himself Sultan of Egypt independent of the Ottomans in December 1914.
# The formal protectorate over Egypt did not long outlast the war. It was brought to an end when the British government issued the
# Unilateral Declaration of Egyptian Independence on 28 February 1922. Shortly afterwards, Sultan Fuad I declared himself King of Egypt,
# but the British occupation continued, in accordance with several reserve clauses in the declaration of independence.
# The situation was normalised in the Anglo-Egyptian treaty of 1936, which granted Britain the right to station troops in Egypt
# for the defence of the Suez Canal, its link with India. Britain also continued to control the training of the Egyptian Army.
# During the Second World War (1939–45), Egypt came under attack from Italian Libya on account of the British presence there,
# although Egypt itself remained neutral until late in the war. After the war Egypt sought to modify the treaty,
# but it was abrogated in its entirety by an anti-British government in October 1951. After the 1952 coup d'état,
# the British agreed to withdraw their troops, and by June 1956 had done so. Britain went to war against Egypt over the Suez Canal
# in late 1956, but with insufficient international support was forced to back down.", "id": "0"
# , "question_lang": "Bushe icaalo ca Egypt caali tekwapo ne caalo cimbi?",
# "question_translated": "Has the country of Egypt been colonized before?", "title": "History of Egypt under the British"
#, "answer_lang": "Emukwai"}


# {"id": 0, "question": "Bushe icaalo ca Egypt caali tekwapo ne caalo cimbi?"
# , "translated_question": "Has the country of Egypt been colonized before?"
# , "answers": "['Emukwai']", "lang": "bem", "split": "dev", "translated_answer": "['yes']", "translation_type": "human_translation"}

NEW_QA_TEMPLATE = Template(
    "Use the following pieces of context to answer the provided question.\n"
    "{{context}}. \n Question: {{ question }}\n"
    "Provide the answer in ({{ pivot_lang }}) based on the context available."
)

def process_afriqa_data(base_file_path, gold_file_path, lang, split, pivot_lang='en'):
    with open(base_file_path, "r") as file:
        base_data = [json.loads(line) for line in file.readlines()]
    # data = json.loads(file_path)

    with open(gold_file_path, "r") as file:
        gold_data = [json.loads(line) for line in file.readlines()]


    results = []
    for base_item, gold_item in zip(base_data, gold_data):
        prompt = NEW_QA_TEMPLATE.render(
            context=gold_item['context'],
            question=base_item['question'],
            pivot_lang=pivot_lang
        )
        results.append({
            "instruction": prompt,
            "output": base_item['translated_answer'],
            "lang": lang,
            "split": split,
            "source": "AfriQA",
            "task": "QA"
        })
    return results

if __name__ == "__main__":
    base_path = "/path/africadata/masakhane_xqa/data/queries"
    gold_path = "/path/africadata/masakhane_xqa/data/gold_passages"

    languages = ["bem", "fon", "hau", "ibo", "kin", "swa", "twi", "wol", "yor", "zul"]
    splits = ["train", "dev", "test"]
    pivot_langs = {"en": "English", "fr": "French"}

    all_results = []
    for lang in tqdm(languages):
        for pivot_lang_key in pivot_langs:
            pivot_lang = pivot_langs[pivot_lang_key]
            for split in splits:
                base_file_name = f"{base_path}/{lang}/queries.afriqa.{lang}.{pivot_lang_key}.{split}.json"
                gold_file_name = f"{gold_path}/{lang}/gold_span_passages.afriqa.{lang}.{pivot_lang_key}.{split}.json"
                if os.path.exists(base_file_name) and os.path.exists(gold_file_name):
                    results = process_afriqa_data(base_file_name, gold_file_name, lang, split, pivot_lang)
                    all_results.extend(results)
                else:
                    print(f"File not found: {base_file_name}")

    with open("africadata/data-new/AFRIQA.json", "w") as outfile:
        json.dump(all_results, outfile, ensure_ascii=False)

# hau, ibo, kin, swa, yor, zul
