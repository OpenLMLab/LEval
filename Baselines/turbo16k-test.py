# -*- coding:utf-8 -*-
import os.path
import sys
import time
import openai
import numpy as np
import argparse
import tiktoken

from tqdm import tqdm
from LEval_config import *


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        sys_prompt = get_sys_prompt(args, file_name)
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        for d in tqdm(data):
            document = d['input']
            cnt = 0
            while num_tokens_from_string(document, "gpt-3.5-turbo") > max_length:
                if "code" not in file_name:
                    document = " ".join(document.split(" ")[:max_length - cnt]) # chunk the input len from right
                else:
                    document = " ".join(document.split(" ")[cnt - max_length:]) # chunk the input len from left
                cnt += 250
            
            print('document len', num_tokens_from_string(document, "gpt-3.5-turbo"))

            instructions = d['instructions']
            outputs = d['outputs']
            i = 0

            for inst, out in zip(instructions, outputs):
                messages = [{"role": "system", "content" : sys_prompt}]
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "codeU" in file_name:
                    messages.append({"role": "user", "content": document + "\n\n" + inst})
                    save_d['prompt'] = sys_prompt + inst

                elif args.metric == "exam_eval":
                    context = "Document is as follows. {} Question: {} \nPlease directly give answer without any additional output or explanation\n Answer: "
                    messages.append({"role": "user", "content": context.format(document, inst)})
                    save_d['prompt'] = sys_prompt + context
                else:
                    context = "Document is as follows. {} Instruction: {} " + f"The suggested output length is around {len(out.split())} words. Output: "
                    messages.append({"role": "user", "content": context.format(document, inst)})
                    save_d['prompt'] = sys_prompt + context

                for _ in range(10):
                    try:
                        if start_idx == 0:
                            print(messages[1]["content"])
                            print("--------------------------- end of example input ------------------")
                            input("Press Enter to confirm this is the correct input for the api call ...")
                            start_idx += 1
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-16k-0613",
                            messages=messages, 
                            max_tokens=max_new_tokens,
                            temperature=0.0001,
                        )  # get response
                        ret = response['choices'][0]['message']['content']
                        ret = ret.strip()  # get the paraphrased answer

                        save_d[f'{openai_model}_pred'] = ret
                        save_d['evaluation'] = d['evaluation']

                        # test the factuality in scientific fiction
                        if "sci_fi" in file_name:
                            text_inputs = inst.replace("based on the world described in the document.",
                                                       "based on the real-world knowledge and facts up until your last training") + "\nPlease directly give answer without any additional output or explanation \nAnswer:"
                            messages.append({"role": "user", "content": text_inputs})
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo-16k-0613",
                                messages=messages,
                                max_tokens=max_new_tokens,
                                temperature=0.0001,
                            )  # get response
                            ret = response['choices'][0]['message']['content']
                            ret = ret.strip()  # get the paraphrased answer
                            save_d[f'{openai_model}_pred'] += f" [fact: {ret}]"

                        print("----------------- [output] vs [ground truth] -----------------")
                        print('[output]:', save_d[f'{openai_model}_pred'], "\n\n" , '[ground truth]:', save_d['gt'])
                        fw.write(json.dumps(save_d) + '\n')
                        break

                    except Exception as e:  # add some logit here for retry
                        if isinstance(e, KeyboardInterrupt):
                            raise e
                        print(i, e)

                        time.sleep(0.8)
                time.sleep(1.0)
                i += 1
                # break
        fw.close()
        # break

if __name__ == "__main__":
    openai.api_key = "" # please fulfill your api key here

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval","llm_gpt4_eval","exam_eval", "ngram_eval", "human_eval"], required=True, help='metric name from ["turbo_eval","gpt4_eval","auto_eval", ...]')

    parser.add_argument('--max_length', default="16k", help='max length of the input, e.g., 2k, 16k')
    # if none, we will load from huggingface
    parser.add_argument('--task_path', type=str, default=None, help= 'set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    parser.add_argument('--task_name', type=str, default=None,
                        help='optional, if not set, we will test all. set this if you want test a specific task from huggingface, example: coursera, tpo')
    parser.add_argument('--mc_tasks', action='store_true')
    args = parser.parse_args()
    key_data_pairs = {}

    max_length = k_to_number(args.max_length) - max_new_tokens
    openai_model = "turbo-16k-0613-" + args.max_length
    data_save_path = f"Predictions/{args.metric}/{openai_model}"
    input(f"Your prediction file will be saved to: {data_save_path}  , press enter to confirm...")
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
