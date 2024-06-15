from datasets import load_dataset
from translation import get_prompt, LANG_FLORES
from globals import LANG_CODES
import json

DEV_LEN = 997
DEVTEST_LEN = 1012

def convert_flores():

    def african_sets():
        result = []
        for lang in LANG_FLORES.keys():
            if lang != "eng" or lang != "fra" or lang != "por":
                dataset = load_dataset("facebook/flores", LANG_FLORES[lang])
                result.append({"lang": lang, "dataset": dataset})
        return result
    
    def source_sets():
        result = []
        for lang in ['eng', 'fra']:
            dataset = load_dataset("facebook/flores", LANG_FLORES[lang])
            result.append({"lang": lang, "dataset": dataset})
        return result
    
    source_data = source_sets()
    african_data = african_sets()

    result = []
    for source in source_data:
        for target in african_data:
            for i in range(DEV_LEN):
                t_text = target["dataset"]["dev"][i]["sentence"]
                s_text = source["dataset"]["dev"][i]["sentence"]
                result.append(get_prompt(LANG_CODES[source["lang"]], LANG_CODES[target["lang"]], s_text, t_text, source["lang"], target["lang"], "FLORES"))
            for i in range(DEVTEST_LEN):
                t_text = target["dataset"]["devtest"][i]["sentence"]
                s_text = source["dataset"]["devtest"][i]["sentence"]
                result.append(get_prompt(LANG_CODES[source["lang"]], LANG_CODES[target["lang"]], s_text, t_text, source["lang"], target["lang"], "FLORES"))

    return result

if __name__ == "__main__":

    final_data = convert_flores()
    with open("data-new/FLORES_dataset.json", "w") as f:
        json.dump(final_data, f, ensure_ascii=False)
            

        