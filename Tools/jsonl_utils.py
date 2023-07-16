import json
import os
from glob import glob
from datasets import load_dataset
import re


def read_jsonl(train_fn):
    res = []
    with open(train_fn) as f:
        for i, line in enumerate(f):
            try:
                res.append(json.loads(line))
            except:
                continue
    print(f"loading from {train_fn}, there are {len(res)} samples")
    return res


def write_jsonl(data, fn):
    with open(fn, "w") as f:
        for line in data:
            print(json.dumps(line), file=f)


datasets_closed_ended = ["coursera", "quality", "topic_retrieval_longchat", "tpo", "gsm100"]
datasets_open_ended = ["financial_qa", "gov_report_summ", "legal_contract_qa", "meeting_summ", "multidoc_qa",
                       "narrative_qa", "natural_question", "news_summ", "paper_assistant", "patent_summ", "review_summ",
                       "scientific_qa", "tv_show_summ"]


def to_filename(data_save_path, task_name):
    return os.path.join(data_save_path, task_name + ".pred.jsonl")


def build_key_data_pairs(args, key_data_pairs, data_save_path):
    os.makedirs(f"Predictions/{args.metric}", exist_ok=True)
    if "llm" not in args.metric:
        os.makedirs(data_save_path, exist_ok=True)

        if args.task_name:
            data = load_dataset('L4NLP/LEval', args.task_name, split='test')
            key_data_pairs[to_filename(data_save_path, args.task_name)] = data
        elif args.task_path:
            if os.path.isdir(args.task_path):
                args.task_path = os.path.join(args.task_path, "*")
            files = glob(args.task_path)
            for file_path in files:
                data = read_jsonl(file_path)
                match = re.search(r'/([^/]*)\.jsonl', file_path)
                file_name = match.group(1)
                key_data_pairs[to_filename(data_save_path, file_name)] = data
        else:
            if args.metric == "ngram_eval":
                datasets_eval = datasets_open_ended
            else:
                datasets_eval = datasets_closed_ended
            for task_name in datasets_eval:
                data = load_dataset('L4NLP/LEval', task_name, split='test')
                key_data_pairs[to_filename(data_save_path, task_name)] = data
    else:
        for gen_data in datasets_open_ended:
            try:
                data = load_dataset('L4NLP/LEval', gen_data, split='test')
            except:
                data = read_jsonl(f"LEval-data/Open-ended-tasks/{gen_data}.jsonl")
            if args.metric == "llm_turbo_eval":
                data = [d for d in data if d["evaluation"] == "human" or d["evaluation"] == "LLM"]
            else:
                data = [d for d in data if d["evaluation"] == "LLM"]
            file_name_llm = data_save_path + ".pred.jsonl"
            if file_name_llm not in key_data_pairs:
                key_data_pairs[file_name_llm] = data
            else:
                key_data_pairs[file_name_llm] += data

def get_prompt(args):
    if args.gsm:
        prompt = "Given several question answer pairs, you need to follow a similar format to answer the last question. Make sure the response is end with The answer is _ ."
    elif "exam" not in args.metric:
        prompt = "Now you are given a very long document. Please follow the instruction after this document. These instructions may include summarizing a document, answering questions based on the document, or writing a required paragraph."
    else:
        prompt = "Now you are given a very long document. Please follow the instruction based on this document. For multi-choice questions, there could be a sinlge correct option or multiple correct options. Please only provide the letter corresponding to the answer (like A or AB) when answering. For other questions, please directly give the concise and accurate answer."
    return prompt