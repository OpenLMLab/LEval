import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "/cpfs01/user/gongshansan/xgen-7b-8k-inst"
# model_path = "Salesforce/xgen-7b-8k-inst"
open_source_model = model_path.split("/")[-1]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

# -*- coding:utf-8 -*-
import os.path
import sys
import time
# import openai
import numpy as np
import argparse
import json
from jsonl_utils import *
from tqdm import tqdm


def num_tokens_from_string(string: str) -> int:
    encoding = tokenizer(string, return_tensors="pt")
    num_tokens = len(encoding['input_ids'][0])
    return num_tokens


def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        print(len(data))
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        if 'gsm' in file_name:
            args.gsm = True

        avg_pred_len = []
        for d in data:
            if d['evaluation'] == 'human' or d['evaluation'] == 'LLM':
                continue
            for inst, out in zip(d['instructions'], d['outputs']):
                avg_pred_len.append(len(out.split()))
        print('avg pred len', np.mean(avg_pred_len))

        for d in tqdm(data):
            document = d['input']
            cnt = 0
            while num_tokens_from_string(document) > 7500:
                document = " ".join(document.split()[:7500 - cnt])  # chunk the input len into 16k tokens
                cnt += 250

            print('document len', num_tokens_from_string(document))

            instructions = d['instructions']
            outputs = d['outputs']
            i = 0

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                context = "Document is as follows. {} Instruction: {} Output: "
                if args.gsm:
                    document = inst
                elif args.metric == "ngram_eval":
                    context = "Document is as follows. {} Instruction: {} " + f"The suggested output length is around {len(out.split())} words. Output: "

                message = header + "### Human: " + prompt + context + "### Assistant:"
                save_d['prompt'] = message

                inputs = tokenizer(message.format(document, inst), return_tensors="pt").to(device)
                sample = model.generate(**inputs, do_sample=False, max_new_tokens=500, eos_token_id=50256)
                output = tokenizer.decode(sample[0])
                ret = output.strip().replace("<|endoftext|>", "").split("### Assistant:")[1].strip()

                save_d[f'{open_source_model}_pred'] = ret
                save_d['evaluation'] = d['evaluation']
                print("----------------- [output] vs [ground truth] -----------------")
                print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])

                fw.write(json.dumps(save_d) + '\n')

                i += 1
                # break
        fw.close()
        # break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval"],
                        default="auto_eval", help='metric name from "turbo_eval","gpt4_eval","auto_eval"')
    parser.add_argument('--task_path', type=str, default=None,
                        help= 'set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')

    parser.add_argument('--gsm', action='store_true', help='set this if you want to test gsm100 dataset')
    args = parser.parse_args()
    key_data_pairs = {}

    if args.task_name in datasets_closed_ended:
        args.metric = "exam_eval"
    else:
        args.metric = "ngram_eval"

    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    prompt = get_prompt(args)

    sys.exit(main())
