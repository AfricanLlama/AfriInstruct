import json

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

LANG_TO_LIST = {
    "amh": [],
    "ara": [],
    "eng": [],
    "hau": [],
    "ibo": [],
    "orm": [],
    "por": [],
    "swa": [],
    "tir": [],
    "yor": [],
}

with (open("path/XLSUM_dataset.json", "r")) as f:
    result = json.load(f)

for item in result:
    lang = item["lang"]
    LANG_TO_LIST[lang].append(item)

for lang in AVAILABLE_DATA:
    print(f"{lang}: {len(LANG_TO_LIST[lang])}")



