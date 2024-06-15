from jinja2 import Template
from random import randint

BASIC = "Translate the following text from {{ s_lang }} to {{ t_lang }}: {{ s_text }}"
DESCRIPTIVE = "Translate the following text from {{ s_lang }} to {{ t_lang }}. \n\n{{ s_lang }}: {{ s_text }} \n\n{{ t_lang }}:"
xP3 = "{{ s_text }} the previous text is in {{ s_lang }}. Here is a translation to {{ t_lang }} "

TEMPLATES = [{"template": Template(DESCRIPTIVE), "name": "descriptive"}, {"template": Template(xP3), "name": "xP3"}]

LANG_FLORES = {
    "afr": "afr_Latn",
    "amh": "amh_Ethi",
    "ara": "arz_Arab",
    "eng": "eng_Latn",
    "fra": "fra_Latn",
    "hau": "hau_Latn",
    "ibo": "ibo_Latn",
    "kin": "kin_Latn",
    "nya": "nya_Latn",
    "por": "por_Latn",
    "som": "som_Latn",
    "sna": "sna_Latn",
    "sot": "sot_Latn",
    "swa": "swh_Latn",
    "tir": "tir_Ethi",
    "xho": "xho_Latn",
    "yor": "yor_Latn",
    "zul": "zul_Latn",
}

LANG_NTREX = {
    "afr": "Afrikaans",
    "amh": "Amharic",
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
    "swa": "Swahili",
    "tir": "Tigrinya",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
}

def get_prompt(s_lang, t_lang, s_text, t_text, s_code, t_code, source, split="train"):

    temp = randint(0, len(TEMPLATES) - 1)
    prompt = TEMPLATES[temp]
    is_reverse = randint(0, 1)

    if is_reverse:
            s_lang, t_lang = t_lang, s_lang
            s_text, t_text = t_text, s_text
            s_code, t_code = t_code, s_code
    
    return {
            "instruction": prompt["template"].render(s_lang=s_lang, t_lang=t_lang, s_text=s_text),
            "output": t_text,
            "lang": f"{s_code}-{t_code}",
            "split": split,
            "source": source,
            "task": "translation",
        }