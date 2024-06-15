# AfriInstruct: Instruction Tuning of African Languages for Diverse Tasks

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Large language models (LLMs) for African languages perform worse compared to their performance in high-resource languages. To address this issue, this study introduces AfriInstruct, which specializes in African languages. AfriInstruct-Data covers various tasks and languages with instruction-tuning datasets designed to train language models. AfriInstruct-Model, developed through continual pretraining with a large African language corpus and instruction fine-tuning, demonstrates superior performance across multiple tasks. Our mixed task evaluation shows it outperforms GPT-3.5-Turbo and other baseline models of similar size. Our contributions fill a critical gap of LLM performance between high-resource and African languages.
## Languages

There are currently 19 languages covered in AfriInstruct-Data:

- Amharic (amh)
- Arabic (ara)
- English (eng)
- French (fra)
- Hausa (hau)
- Igbo (ibo)
- Kinyarwanda (kin)
- Malagasy (mlg)
- Chichewa (nya)
- Afaan Oromoo (orm)
- Portuguese (por)
- Somali (som)
- Shona (sna)
- Sesotho (sot)
- Swahili (swa)
- Tigrinya (tir)
- Xhosa (xho)
- Yoruba (yor)
- Zulu (zul)

| Source         | Task                      | Tokens      | Prompts   | Language                                                                          |
| -------------- |---------------------------| ----------- | --------- | --------------------------------------------------------------------------------- |
| MasakhaNEWS    | News Topic Classification | 6,154,176   | 90,890    | eng, fra, amh, hau, ibo, orm, sna, som, swa, tir, xho, yor                         |
| MasakhaPOS     | POS                       | 1,780,578   | 6,879     | hau, ibo, kin, nya, sna, swa, xho, yor, zul                                       |
| AfriSenti      | Sentiment Analysis        | 19,201,035  | 235,225   | amh, hau, ibo, yor, por, kin, swa                                                 |
| NollySenti     | Sentiment Analysis        | 1,213,691   | 15,100    | hau, ibo, eng, yor                                                                |
| xP3            | xP3-Multitask             | 640,745,532 | 8,314,942 | eng, ara, ibo, hau, kin, nya, sna, sot, swa, xho, yor, zul                        |
| xP3            | xP3-QA                    | 146,758,736 | 8,314,942 | eng, ara, ibo, hau, kin, nya, sna, sot, swa, xho, yor, zul                        |
| FLORES         | translation               | 5,692,402   | 72,324    | eng, fra, afr, amh, ara, hau, ibo, kin, nya, por, som, sna, sot, swa, tir, xho, yor, zul |
| MAFAND         | translation               | 4,467,767   | 66,234    | eng, amh, hau, ibo, kin, nya, sna, swa, xho, yor, zul                             |
| MasakhaNER2.0  | NER                       | 12,935,191  | 58,667    | hau, ibo, kin, nya, sna, swa, xho, yor, zul                                       |
| MENYO          | translation               | 1,225,883   | 16,703    | eng, yor                                                                          |
| XL-Sum         | Summarization             | 32,814,291  | 72,124    | eng, amh, ara, hau, ibo, orm, por, swa, tir, yor                                  |

## Dataset Download

You can easily download the dataset using the Hugging Face `datasets` library. Below are the steps to download and use the dataset:

```bash
pip install datasets
```

- The dataset is also available on [Hugging Face](https://huggingface.co/datasets/llama-lang-adapt/AfriInstruct-Data)

```python
from datasets import load_dataset
dataset = load_dataset('llama-lang-adapt/AfriInstruct-Data')
```

- Data Format:
  - instruction : Prompt 
  - output : Answer for the Instruction
  - lang : Datapoint Language e.g `bem`
  - split : Dataset Split
  - source : Dataset Source
  - task : Task Category of the Instruction e.g POS

## Environment and Repository Setup

Note that axolotl requires Python >= 3.9 and PyTorch >= 2.0.

```
conda create -n africa-it python=3.9
conda activate africa-it
pip3 install torch

git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install packaging
pip3 install -e '.[flash-attn,deepspeed]'
cd ..
```

Note that if you get an error while installing axolotl with "RuntimeError: FlashAttention is only supported on CUDA 11.6 and above", it might be because your CUDA version for torch and nvcc are mismatched. Check with `nvcc -V`. In case of mismatch, use `export PATH=/usr/local/cuda/bin:$PATH` to match the versions.

To ensure access to the models and datasets through HuggingFace, you'll want to login through the cli as well.

```
huggingface-cli login
```

## Runnning fine-tuning

```
# Preprocess the dataset using the model's tokenizer
python -m axolotl.cli.preprocess path/finetune.yml 

# Run the finetuning
accelerate launch -m axolotl.cli.train path/finetune.yml
```

Note preprocessing and finetuning will fill up your HuggingFace cache location with a lot of data. You might want to relocate your cache location with `export HF_DATASETS_CACHE="$PWD/.cache"` before running these scripts.


