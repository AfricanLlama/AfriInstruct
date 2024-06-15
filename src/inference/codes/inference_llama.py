import json
from vllm import LLM, SamplingParams
import torch
# Initialize vLLM with the desired model and caching settings
# llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_prefix_caching=True, enforce_eager=True)
# device = torch.device('cuda:1')

# Initialize vLLM with the desired model, caching settings, and assume a hypothetical quantization parameter
llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_prefix_caching=True, enforce_eager=True, quantization="gptq")

sampling_params = SamplingParams(temperature=0.0)

prefix = "You are very proficient in African languages, and you are very good at responding in those languages."

def generate_responses_with_vllm(file_path):
    # Load prompts from a JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    prompts = [prefix + data_point["instruction"] for data_point in data]

    # Generate responses for each prompt
    outputs = llm.generate(prompts, sampling_params)

    # Handle each generated output
    results = []
    for i, entry in enumerate(data):
        prompt = entry['instruction']

        # Access the first element of each output list if outputs are nested lists
        if isinstance(outputs[i], list):
            generated_text = outputs[i][0].outputs[0].text
        else:
            generated_text = outputs[i].outputs[0].text

        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "lang": entry.get('lang', 'N/A'),  # Handle optional fields
            "split": entry.get('split', 'N/A'),
            "source": entry.get('source', 'N/A'),
            "task": entry.get('task', 'N/A')
        }
        results.append(result)

    # Save results to a new JSON file
    output_filename = file_path.replace('.json', '_results.json')
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

for file_path in benchmark_files:
    generate_responses_with_vllm(file_path)
