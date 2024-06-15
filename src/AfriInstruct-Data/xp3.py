import json
from datasets import load_dataset

pretrained = {
    "af": "Afrikaans",
    "am": "Amharic",
    # "ar": "Egyptian Arabic",
    # "en": "English",
    # "fr": "French",
    "ha": "Hausa",
    "ig": "Igbo",
    "rw": "Kinyarwanda",
    "mg": "Malagasy",
    "ny": "Chichewa",
    "om": "Afaan Oromoo",
    # "pt": "Portuguese",
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

xp3_langs = ['ak', 'ar', 'as', 'bm', 'bn', 'ca', 'code', 'en', 'es', 'eu', 'fon', 'fr', 'gu', 'hi', 'id', 'ig', 'ki', 'kn', 'lg', 'ln', 'ml', 'mr', 'ne', 'nso', 'ny', 'or', 'pa', 'pt', 'rn', 'rw', 'sn', 'st', 'sw', 'ta', 'te', 'tn', 'ts', 'tum', 'tw', 'ur', 'vi', 'wo', 'xh', 'yo', 'zh', 'zu']


if __name__ == '__main__':
    dataset =  []
    for lang in xp3_langs:
        if lang in pretrained:
            print(lang)
            xp3 = load_dataset("bigscience/xP3", lang)['train']
            print(len(xp3))
            for i in range(len(xp3)):
                dataset.append({
                            'instruction': xp3[i]['inputs'],
                            'output': xp3[i]['targets'],
                            'lang': d_map[lang],
                            'split': 'train',
                            'source': 'xP3',
                            'task': 'Multitask',
                        })

    with open('xp3.json', 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)