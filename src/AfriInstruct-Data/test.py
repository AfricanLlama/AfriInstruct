from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer, pipeline
import re
import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import login
login("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")

model = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model)

def truncate_QA(answer, symbol="Answer: "):
    return answer[answer.find(symbol) + len(symbol):].strip()

def truncate_translation(answer, instruction):

    truncate_instr = answer[answer.find(instruction) + len(instruction):].strip()

    return truncate_instr[:truncate_instr.find("\n")].strip()

def truncate_classify(answer):
    truncate_topic = answer[answer.find("Topic: ") + 7:]

    assert truncate_topic != -1

    return truncate_topic[:truncate_topic.find("\n")].strip()

pipe = pipeline(
    "text-generation",
    model=model,
    device_map="cuda:0",
    torch_dtype=torch.float16
)

instruction_QA = "Use the following pieces of context to answer the provided question.\nThomas Joseph Odhiambo Mboya (15August 19305July 1969) was a Kenyan trade unionist, educator, Pan-Africanist, author, independence activist, and statesman. He was one of the founding fathers of the Republic of Kenya. He led the negotiations for independence at the Lancaster House Conferences and was instrumental in the formation of Kenya's independence party – the Kenya African National Union (KANU) – where he served as its first Secretary-General. He laid the foundation for Kenya's capitalist and mixed economy policies at the height of the Cold War and set up several of the country's key labour institutions. Mboya's intelligence, charm, leadership, and oratory skills won him admiration from all over the world. He gave speeches, participated in debates and interviews across the world in favour of Kenya's independence from British colonial rule. He also spoke at several rallies in the goodwill of the civil rights movement in the United States. In 1958, at the age of 28, Mboya was elected Conference Chairman at the All-African Peoples' Conference convened by Kwame Nkrumah of Ghana. He helped build to the Trade Union Movement in Kenya, Uganda and Tanzania, as well as across Africa. He also served as the Africa Representative to the International Confederation of Free Trade Unions (ICFTU). In 1959, Mboya called a conference in Lagos, Nigeria, to form the first All-Africa ICFTU labour organization.. \n Question: Eneo wanako ishi wakamba nchini Kenya linaitwaje?\nProvide the answer in (English) based on the context available."
instruction_tr = "Translate the following text from English to Chichewa. \n\nEnglish: I didn't do it.\" \n\nChichewa:"
instruction_tr = "ሌላ ጥቁር ሰዉ?! the previous text is in Amharic. Here is a translation to English "
instruction_classify = "Classify the text \"Kimwe mu bibazo bikunze kugaragara kurusha ibindi iyo ugerageza guhindura filimi mu buryo bwa DVD ni uko imwe mu bice by'amashusho bitagaragara.\" into the following topics:\n- science/technology\n- travel\n- politics\n- sports\n- health\n- entertainment\n- geography\nTopic: "

sequences_QA = pipe(
    instruction_QA,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=2000,
)[0]["generated_text"]

print(truncate_QA(sequences_QA))

sequences_tr = pipe(
    instruction_tr,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=2000,
)[0]["generated_text"]

print(sequences_tr)

print("\n")

print(truncate_translation(sequences_tr, instruction_tr))

sequences_classify = pipe(
    instruction_classify,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=2000,
)[0]["generated_text"]

print(truncate_classify(sequences_classify))