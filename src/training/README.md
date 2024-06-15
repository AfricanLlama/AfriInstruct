# Setting up environment

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

# Runnning fine-tuning

```
# Preprocess the dataset using the model's tokenizer
python -m axolotl.cli.preprocess finetune.yml 

# Run the finetuning
accelerate launch -m axolotl.cli.train finetune.yml
```

Note preprocessing and finetuning will fill up your HuggingFace cache location with a lot of data. You might want to relocate your cache location with `export HF_DATASETS_CACHE="$PWD/.cache"` before running these scripts.