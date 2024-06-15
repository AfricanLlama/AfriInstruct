import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

max_seq_length = 4096
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
load_in_4bit = True  
model_name = "llama-lang-adapt/african-it-lora"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

prefix = "You are very proficient in African languages, and you are very good at responding in those languages."

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def generate_responses(file_path, output_dir):
    with open(file_path, 'r') as file:
        data = json.load(file)

    results = []
    for entry in data:
        prompt = entry['instruction']

        input_text = alpaca_prompt.format(
            prefix,  # instruction
            prompt,  # input
            ""  # output - leave this blank for generation!
        )

        inputs = tokenizer(input_text, return_tensors="pt", max_length=max_seq_length, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean_output = decoded_output.split('### Response:')[-1].strip()

        result = {
            "instruction": prompt,
            "output": clean_output,
            "lang": entry.get('lang', 'N/A'),
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
    "/mnt/disk/kuemura/africadata/revised_benchmark_all_include/benchmark_sample.json",
    "/mnt/disk/kuemura/africadata/revised_benchmark_all_include/benchmark.json",
    "/mnt/disk/kuemura/africadata/revised_benchmark_all_include/benchmark_hau_806.json",
    "/mnt/disk/kuemura/africadata/revised_benchmark_all_include/benchmark_ibo_806.json",
    "/mnt/disk/kuemura/africadata/revised_benchmark_all_include/benchmark_kin_806.json",
    "/mnt/disk/kuemura/africadata/revised_benchmark_all_include/benchmark_swa_602.json",
    "/mnt/disk/kuemura/africadata/revised_benchmark_all_include/benchmark_yor_602.json",
    "/mnt/disk/kuemura/africadata/revised_benchmark_all_include/benchmark_zul_602.json"
]

output_dir = "/mnt/disk/kuemura/africadata/AfricanLLM-2-it-lora"

for file_path in benchmark_files:
    generate_responses(file_path, output_dir)
