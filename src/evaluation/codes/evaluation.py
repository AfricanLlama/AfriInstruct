from fuzzywuzzy import fuzz
import json
from sklearn.metrics import f1_score
from datasets import load_metric
import ast
from collections import Counter
# import evaluate as evl
from nltk.tokenize import word_tokenize
import os

def squad_f1(pred, ref):
    pred_tokens = word_tokenize(pred)
    ref_tokens = word_tokenize(ref)
    common_tokens = set(pred_tokens) & set(ref_tokens)
    if not common_tokens:
        return 0.0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1 * 100

# Load the benchmark data
def load_benchmark(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Load your model predictions (assuming they are stored similarly)
def load_predictions(filepath):
    with open(filepath, 'r') as file:
        predictions = json.load(file)
    return predictions

def extract_topic(prediction_text):
    # Placeholder function: adapt this to your needs
    # Extract the topic from a sequence, possibly using regex or other parsing methods
    return prediction_text.split()[-1]  # Simple example: assuming the topic is the last word

def get_closest_topic(topic, topics):
    max_ratio = 0
    closest_topic = None
    for t in topics:
        ratio = fuzz.partial_ratio(topic, t)
        if ratio > max_ratio:
            max_ratio = ratio
            closest_topic = t
    return closest_topic if max_ratio > 80 else None

topics = {
    'science/technology': 0,
    'travel': 1,
    'politics': 2,
    'sports': 3,
    'health': 4,
    'entertainment': 5,
    'geography': 6
}

def evaluate(benchmarks, predictions):
    chrf = load_metric('chrf', trust_remote_code=True)

    results = {
        'QA': [],
        'translation': [],
        'topic-classification': []
    }

    true_labels = []
    pred_labels = []
    
    for bench, pred in zip(benchmarks, predictions):
        if bench['task'] == 'QA':
            possible_answers = bench['output']
            predicted_answer = pred['output']

            # possible_answersをリスト形式に変換
            if isinstance(possible_answers, str):
                try:
                    possible_answers = ast.literal_eval(possible_answers)
                except:
                    possible_answers = [possible_answers]

            if isinstance(possible_answers, list):
                f1_scores = [squad_f1(str(predicted_answer), str(answer)) for answer in possible_answers]
                if f1_scores:  # Ensure f1_scores is not empty
                    best_f1 = max(f1_scores)
                    results['QA'].append(best_f1)
                else:
                    results['QA'].append(0.0)  # If f1_scores is empty, append 0
            else:
                f1_score_value = squad_f1(predicted_answer, possible_answers)
                results['QA'].append(f1_score_value)

        elif bench['task'] == 'translation':
            ref = [[bench['output']]]
            hypo = [pred['output']]
            score = chrf.compute(predictions=hypo, references=ref)['score']
            results['translation'].append(score)

        
        # refer to alex evaluation codes
        elif bench['task'] == 'topic-classification':
            true_topic = bench['output'].strip().lower()
            predicted_topic = pred['output'].strip().lower()
            
            closest_true_topic = get_closest_topic(true_topic, topics.keys())
            closest_pred_topic = get_closest_topic(predicted_topic, topics.keys())
            
            if closest_true_topic and closest_pred_topic:
                true_label = topics[closest_true_topic]
                pred_label = topics[closest_pred_topic]
                true_labels.append(true_label)
                pred_labels.append(pred_label)
    
    if true_labels and pred_labels:
        f1 = f1_score(true_labels, pred_labels, average='macro')
        results['topic-classification'].append(f1 * 100) 

    return results

def run_evaluation(benchmark_dir, prediction_dir, description):
    total_scores = {
        'QA': 0,
        'translation': 0,
        'topic-classification': 0
    }
    print(f"Evaluation Results for {description}:")

    benchmark_files = sorted([os.path.join(benchmark_dir, f) for f in os.listdir(benchmark_dir) if f.endswith('.json')])
    prediction_files = sorted([os.path.join(prediction_dir, f) for f in os.listdir(prediction_dir) if f.endswith('.json')])

    for benchmark_file, prediction_file in zip(benchmark_files, prediction_files):
        benchmark_data = load_benchmark(benchmark_file)
        predictions_data = load_predictions(prediction_file)
        evaluation_results = evaluate(benchmark_data, predictions_data)
        task_scores = []
        for task, scores in evaluation_results.items():
            average_score = sum(scores) / len(scores) if scores else 0
            total_scores[task] += average_score
            task_scores.append(average_score)
            print(f"{os.path.basename(benchmark_file)} - {task} Average Score: {average_score:.2f}")
        print("total average score:" f"{sum(task_scores) / len(task_scores)}")
    print(f"Total Scores across all tasks for {description}: {total_scores}\n")

# Example usage:
benchmark_dir = "/mnt/disk/kuemura/africadata/revised_benchmark_all_include/"
benchmark_dir_copy = "/mnt/disk/kuemura/africadata/revised_benchmark_all_include copy"
prediction_dir_llama3_high_4bit = "/mnt/disk/kuemura/africadata/AfricanLLM-3-high-rank-4bit/"
prediction_dir_llama3_high = "/mnt/disk/kuemura/africadata/AfricanLLM-3-high-rank/"
prediction_dir_llama2_low = "/mnt/disk/kuemura/africadata/AfricanLLM-2-low-rank/"
prediction_dir_llama2_cpt_low = "/mnt/disk/kuemura/africadata/AfricanLLM-2-CPT-low-rank/"
prediction_dir_llama2_cpt_high = "/mnt/disk/kuemura/africadata/AfricanLLM-2-CPT-high-rank/"
prediction_dir_llama3_extreme = "/mnt/disk/kuemura/africadata/AfricanLLM-3-extreme-rank/"

prediction_dir_llama3_32 = "/mnt/disk/kuemura/africadata/AfricanLLM-3-32-rank"
prediction_dir_llama3_64 = "/mnt/disk/kuemura/africadata/AfricanLLM-3-64-rank"
prediction_dir_llama3_128 = "/mnt/disk/kuemura/africadata/AfricanLLM-3-128-rank"
prediction_dir_llama3_256 = "/mnt/disk/kuemura/africadata/AfricanLLM-3-256-rank"
prediction_dir_llama3_512 = "/mnt/disk/kuemura/africadata/AfricanLLM-3-512-rank"

prediction_dir_llama2_it_low = "/mnt/disk/kuemura/africadata/AfricanLLM-2-it-lora"
prediction_dir_llama3_it = "/mnt/disk/kuemura/africadata/AfricanLLM-3-it"

base_llama3 = "/mnt/disk/mchen/llama-8b_results"


# run_evaluation(benchmark_dir, prediction_dir_llama3_high_4bit, "Llama3 High Rank 4-bit")
# run_evaluation(benchmark_dir, prediction_dir_llama3_high, "Llama3 High Rank")
# run_evaluation(benchmark_dir, prediction_dir_llama2_low, "Llama2 7b 32")
# run_evaluation(benchmark_dir, prediction_dir_llama2_it_low, "AfricanLLM-2-it-small")
# run_evaluation(benchmark_dir, prediction_dir_llama2_cpt_low, "Llama2 CPT Low Rank")
# run_evaluation(benchmark_dir, prediction_dir_llama2_cpt_high, "Llama2 CPT High Rank") 
# run_evaluation(benchmark_dir, prediction_dir_llama3_32, "32")
# run_evaluation(benchmark_dir, prediction_dir_llama3_64, "64")
# run_evaluation(benchmark_dir, prediction_dir_llama3_128, "128")
# run_evaluation(benchmark_dir, prediction_dir_llama3_256, "256")
# run_evaluation(benchmark_dir, prediction_dir_llama3_512, "512")

# run_evaluation(benchmark_dir, prediction_dir_llama3_it, "AfricanLLM-3-it")

# run_evaluation(benchmark_dir_copy, base_llama3, "Llama3 8b base")

# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/AfricanLLM-2-low-rank-no-quant", "Llama2 7b base 32")
# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/AfricanLLM-2-low-rank-CPT-no-quant", "Llama2 7b CPT 32")


# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/AfricanLLM-2-CPT-32-rank-900", "Llama2 7b cpt 32")
# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/AfricanLLM-2-CPT-64-rank-900", "Llama2 7b cpt 64")
# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/AfricanLLM-2-CPT-128-rank-900", "Llama2 7b cpt 128")
# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/AfricanLLM-2-CPT-256-rank-900", "Llama2 7b cpt 256")
# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/AfricanLLM-2-CPT-512-rank-900", "Llama2 7b cpt 512")
# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/AfricanLLM-2-CPT-512-rank-900", "Llama2 7b cpt 512")

# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/chat-gpt-3.5-turbo", "chat gpt 3.5 turbo")
# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/chat-gpt-4o-2024-05-13", "chat gpt 4o")
# run_evaluation(benchmark_dir, "/mnt/disk/kuemura/africadata/llama2_13b", "llama2 13b")
