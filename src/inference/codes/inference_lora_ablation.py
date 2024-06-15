from unsloth import FastLanguageModel
import json
import torch
import os
# run with unsloth_env


max_seq_length = 4096
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
HAS_BFLOAT16 = torch.cuda.is_bf16_supported()
# device = torch.device('cuda:1')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "llama-lang-adapt/AfricanLLM-3-256-rank", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# model = model.to(device)

prefix = "You are very proficient in African languages, and you are very good at responding in those languages."

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def generate_responses_with_unsloth(file_path, output_dir):
    # Load prompts from a JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Handle each generated output
    results = []
    for i, entry in enumerate(data):
        prompt = entry['instruction']

        inputs = tokenizer(
        [
            alpaca_prompt.format(
                prefix, # instruction
                prompt, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        split_outputs = [text.split('### Response:') for text in decoded_outputs]
        output = split_outputs[0][-1]
        clean_output = output.replace("</s>", "")

        result = {
            "instruction": prompt,
            "output": clean_output,
            "lang": entry.get('lang', 'N/A'),  # Handle optional fields
            "split": entry.get('split', 'N/A'),
            "source": entry.get('source', 'N/A'),
            "task": entry.get('task', 'N/A')
        }
        results.append(result)

    # Save results to a new JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, os.path.basename(file_path))
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Inference complete and results saved for {file_path}")

# Example usage
benchmark_files = [
    "src/evaluation/benchmarks/benchmark_sample.json"
    "src/evaluation/benchmarks/benchmark.json",
    "src/evaluation/benchmarks/benchmark_hau_806.json",
    "src/evaluation/benchmarks/benchmark_ibo_806.json",
    "src/evaluation/benchmarks/benchmark_kin_806.json",
    "src/evaluation/benchmarks/benchmark_swa_602.json",
    "src/evaluation/benchmarks/benchmark_yor_602.json",
    "src/evaluation/benchmarks/benchmark_zul_602.json"
]
output_dir = "/mnt/disk/kuemura/africadata/AfricanLLM-3-256-rank"

for file_path in benchmark_files:
    generate_responses_with_unsloth(file_path, output_dir)
