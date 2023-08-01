# -*- coding:utf-8 -*-
import os.path
import sys
import time
import openai
import numpy as np
import argparse
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from tqdm import tqdm
from LEval_config import *



def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        sys_prompt = get_sys_prompt(args, file_name)
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        for d in tqdm(data):
            document = d['input']
            instructions = d['instructions']
            outputs = d['outputs']
            i = 0

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = sys_prompt + context
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {} \nQuestion: {}  \nPlease directly give answer without any additonal output or explanation"
                    message = sys_prompt + context + " \nAnswer:"
                else:
                    context = "Document is as follows. {} Instruction: {} " + f"\nAnswer this question with {len(out.split())} words."
                    message =  sys_prompt + context
                save_d['prompt'] = message.replace(document, '<long input>')
                text_inputs = message.format(document, inst)
                for _ in range(10):
                    try:
                        if start_idx == 0:
                            print(f"{HUMAN_PROMPT} {text_inputs} {AI_PROMPT}")
                            print("--------------------------- end of example input ------------------")
                            input("Press Enter to confirm this is the correct input for the api call ...")
                            start_idx += 1
                        completion = anthropic.completions.create(
                            model=anthropic_model,
                            max_tokens_to_sample=max_new_tokens,
                            prompt=f"{HUMAN_PROMPT} {text_inputs} {AI_PROMPT}",
                        )
                        ret = completion.completion.strip()  # get the paraphrased answer

                        save_d[f'{anthropic_model}_pred'] = ret
                        save_d['evaluation'] = d['evaluation']
                        print("----------------- [output] vs [ground truth] -----------------")
                        print('[output]:', save_d[f'{anthropic_model}_pred'], "\n\n" , '[ground truth]:', save_d['gt'])

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
    openai.api_key = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval","llm_gpt4_eval","exam_eval", "ngram_eval", "human_eval"], required=True, help='metric name from ["turbo_eval","gpt4_eval","auto_eval", ...]')
    parser.add_argument('--task_name', type=str, default=None,
                        help='optional, if not set, we will test all. set this if you want test a specific task from huggingface, example: coursera')
    # if none, we will load from huggingface
    parser.add_argument('--task_path', type=str, default=None, help= 'set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    parser.add_argument('--version', type=str, default="2", choices=["2", "1.3"])
    parser.add_argument('--mc_tasks', action='store_true', help="test multiple choice tasks")
    args = parser.parse_args()
    key_data_pairs = {}

    anthropic = Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key="",
    )

    anthropic_model = f"claude-{args.version}"
    data_save_path = f"Predictions/{args.metric}/{anthropic_model}"
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
