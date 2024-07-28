import os
import pandas as pd
import json
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from huggingface_hub import Repository

# African languages from the Aya Collection dataset
AYA_AFRICAN_LANGS_CODES = {
    "afr": "afrikaans",
    # "arq": "algerian_arabic",
    "amh": "amharic",
    # "bem": "bemba",
    "knc": "central_kanuri",
    "arz": "egyptian_arabic",
    # "fon": "fon",
    "hau": "hausa",
    "ibo": "igbo",
    "kin": "kinyarwanda",
    "ary": "moroccan_arabic",
    "por": "mozambican_portuguese",
    "ars": "najdi_arabic",
    "nso": "northern_sotho",
    "nya": "nyanja",
    "plt": "plateau_malagasy",
    "sna": "shona",
    "som": "somali",
    "sot": "southern_sotho",
    "arb": "standard_arabic",
    "swa": "swahili",
    "acq": "taizzi_adeni_arabic",
    "tmh": "tamasheq",
    "aeb": "tunisian_arabic",
    # "twi": "twi",
    "wol": "wolof",
    "xho": "xhosa",
    "yor": "yoruba",
    "zul": "zulu",
}

# Languages in llama-lang-adapt/AfriInstruct-Data
AFRI_INSTRUCT_LANG_CODES = [
    "afr-eng",
    "afr-fra",
    "amh",
    "amh-eng",
    "amh-fra",
    "ara",
    "ara-eng",
    "ara-fra",
    "eng",
    "eng-afr",
    "eng-amh",
    "eng-ara",
    "eng-eng",
    "eng-fra",
    "eng-hau",
    "eng-ibo",
    "eng-kin",
    "eng-nya",
    "eng-por",
    "eng-sna",
    "eng-som",
    "eng-sot",
    "eng-swa",
    "eng-tir",
    "eng-xho",
    "eng-yor",
    "eng-zul",
    "fra",
    "fra-afr",
    "fra-amh",
    "fra-ara",
    "fra-eng",
    "fra-fra",
    "fra-hau",
    "fra-ibo",
    "fra-kin",
    "fra-nya",
    "fra-por",
    "fra-sna",
    "fra-som",
    "fra-sot",
    "fra-swa",
    "fra-tir",
    "fra-xho",
    "fra-yor",
    "fra-zul",
    "hau",
    "hau-eng",
    "hau-fra",
    "ibo",
    "ibo-eng",
    "ibo-fra",
    "kin",
    "kin-eng",
    "kin-fra",
    "nya",
    "nya-eng",
    "nya-fra",
    "orm",
    "por",
    "por-eng",
    "por-fra",
    "sna",
    "sna-eng",
    "sna-fra",
    "som",
    "som-eng",
    "som-fra",
    "sot",
    "sot-eng",
    "sot-fra",
    "swa",
    "swa-eng",
    "swa-fra",
    "tir",
    "tir-eng",
    "tir-fra",
    "xho",
    "xho-eng",
    "xho-fra",
    "yor",
    "yor-eng",
    "yor-fra",
    "zul",
    "zul-eng",
    "zul-fra",
]

# We already have these datasets in llama-lang-adapt/AfriInstruct-Data
REPEATED_DATASETS = [
    "AfriQA-inst",
    "MasakhaNEWS-inst",
    "AfriSenti-inst"
]

def filter_by_dataset_source(example):
    """
    Filter out examples from the datasets we already have.
    """
    return example["dataset_name"] not in REPEATED_DATASETS

def filter_non_empty(example):
    """
    Filter out examples where 'instruction' or 'output' fields are empty.
    """
    return example['instruction'].strip() != '' and example['output'].strip() != ''

def replace_split(example):
    """
    Replace the 'split' field of an example to 'train'.
    """
    example['split'] = 'train'
    return example

def replace_split_batch(batch):
    """
    Replace the 'split' field for all examples in a batch to 'train'.
    """
    batch['split'] = ['train'] * len(batch['split'])
    return batch

def filter_by_language(example, target_language):
    """
    Filter examples by specified language.
    """
    return example['lang'] == target_language

######################## PUSH TO HUGGING FACE ########################

afri_instruct = load_dataset("llama-lang-adapt/AfriInstruct-Data")

# Upload the languages that are in the Aya Collection and llama-lang-adapt/AfriInstruct-Data
for lang in tqdm(AYA_AFRICAN_LANGS_CODES):
    ds = load_dataset(
        "CohereForAI/aya_collection_language_split",
        AYA_AFRICAN_LANGS_CODES[lang]
    )

    # Remove the datasets we already have in llama-lang-adapt/AfriInstruct-Data
    ds = ds.filter(filter_by_dataset_source, batched=False)

    # We will use all the data from the datasets in the Aya Collection as part of the train 
    # split in our dataset. I don't like this since we may not have data in some dev/test splits 
    ds = concatenate_datasets([ds[split] for split in ds])
    ds = ds.map(replace_split_batch, batched=True)
    ds = DatasetDict({"train": ds})

    # Rename some columns to have the same ones as llama-lang-adapt/AfriInstruct-Data
    ds = ds.rename_column(original_column_name="inputs", new_column_name="instruction")
    ds = ds.rename_column(original_column_name="targets", new_column_name="output")
    ds = ds.rename_column(original_column_name="language", new_column_name="lang")
    ds = ds.rename_column(original_column_name="dataset_name", new_column_name="source")
    ds = ds.rename_column(original_column_name="task_type", new_column_name="task")
    ds = ds.select_columns(["instruction", "output", "lang", "split", "source", "task"])

    ds_train = concatenate_datasets(
        [
            # Filter the train split of llama-lang-adapt/AfriInstruct-Data by language
            afri_instruct["train"].filter(
                lambda sample: filter_by_language(sample, lang), batched=False
            ),
            ds["train"],
        ]
    )
    ds_lang = DatasetDict({"train": ds_train})

    # Filter the validation split of llama-lang-adapt/AfriInstruct-Data by language
    ds_validation = afri_instruct["validation"].filter(
        lambda sample: filter_by_language(sample, lang), batched=False
    )
    if len(ds_validation) > 0:
        ds_lang["validation"] = ds_validation

    # Filter the test split of llama-lang-adapt/AfriInstruct-Data by language
    ds_test= afri_instruct["test"].filter(
        lambda sample: filter_by_language(sample, lang), batched=False
    )
    if len(ds_test) > 0:
        ds_lang["test"] = ds_test

    # Remove possible empty-strings
    ds_lang = ds_lang.filter(filter_non_empty, batched=False)

    ds_lang.push_to_hub(
        "davidguzmanr/AfriInstruct-language-split",
        lang,
        commit_message=f"Upload {lang} data"
    )

# Upload the languages that are only in llama-lang-adapt/AfriInstruct-Data
for lang in tqdm(set(AFRI_INSTRUCT_LANG_CODES) - set(AYA_AFRICAN_LANGS_CODES)):
    ds_train = afri_instruct["train"]
    ds_lang = DatasetDict(
        {
            # Filter the train split of llama-lang-adapt/AfriInstruct-Data by language
            "train": ds_train.filter(
                lambda sample: filter_by_language(sample, lang), batched=False
            )
        }
    )

    # Filter the validation split of llama-lang-adapt/AfriInstruct-Data by language
    ds_validation = afri_instruct["validation"].filter(
        lambda sample: filter_by_language(sample, lang), batched=False
    )
    if len(ds_validation) > 0:
        ds_lang["validation"] = ds_validation

    # Filter the test split of llama-lang-adapt/AfriInstruct-Data by language
    ds_test= afri_instruct["test"].filter(
        lambda sample: filter_by_language(sample, lang), batched=False
    )
    if len(ds_test) > 0:
        ds_lang["test"] = ds_test

    # Remove possible empty-strings
    ds_lang = ds_lang.filter(filter_non_empty, batched=False)

    ds_lang.push_to_hub(
        "davidguzmanr/AfriInstruct-language-split",
        lang,
        commit_message=f"Upload {lang} data"
    )