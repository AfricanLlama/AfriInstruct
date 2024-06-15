from datasets import load_dataset
from translation import get_prompt, LANG_NTREX
from globals import LANG_CODES
import json

NTREX_LEN = 1997

def convert_ntrex():

    def african_sets():
        result = []
        for lang in LANG_NTREX.keys():
            if lang != "eng" or lang != "fra" or lang != "por":
                with open(f"data/NTrex/newstest2019-ref.{lang}.txt", "r") as f:
                    lines = [x.strip() for x in f.readlines()]
                result.append({"lang": lang, "dataset": lines})
        return result
    
    def source_sets():
        result = []
        for lang in ['eng', 'fra']:
            with open(f"data/NTrex/newstest2019-ref.{lang}.txt", "r") as f:
                lines = [x.strip() for x in f.readlines()]
            result.append({"lang": lang, "dataset": lines})
        return result
    
    source_data = source_sets()
    african_data = african_sets()

    result = []
    for source in source_data:
        for target in african_data:
            for i in range(NTREX_LEN):
                t_text = target["dataset"][i]
                s_text = source["dataset"][i]
                result.append(get_prompt(LANG_CODES[source["lang"]], LANG_CODES[target["lang"]], s_text, t_text, source["lang"], target["lang"], "NTREX"))
    return result

if __name__ == "__main__":

    final_data = convert_ntrex()
    with open("data-new/NTREX_dataset.json", "w") as f:
        json.dump(final_data, f, ensure_ascii=False)
            

        