# -*- coding:utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModel
import argparse
from LEval_config import *
from tqdm import tqdm



def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(file_name, "w")
        data = key_data_pairs[file_name]
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )

        sys_prompt = get_sys_prompt(args, file_name)

        for d in tqdm(data):
            document = d['input']
            cnt = 0
            while num_tokens_from_string(document, tokenizer) > args.max_length:
                document = " ".join(document.split()[:args.max_length - cnt])  # chunk the input len into 16k tokens
                cnt += 250


            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = header + sys_prompt + context
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {} \nQuestion: {} "
                    message = header + sys_prompt + context + " \nAnswer:"
                else:
                    context = "Document is as follows. {} Instruction: {} " + f"\nAnswer this question with {len(out.split())} words."
                    message = header + sys_prompt + context

                save_d['prompt'] = message.replace(document, '<long input>')
                text_inputs = message.format(document, inst)
                response, history = model.chat(tokenizer, text_inputs, history=[], do_sample=False)
                save_d[f'{open_source_model}_pred'] = response
                save_d['evaluation'] = d['evaluation']
                if start_idx <5:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print(text_inputs)
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    input()
                    start_idx += 1
                fw.write(json.dumps(save_d) + '\n')
        fw.close()
        # break

# eval:  python Evaluation/auto_eval.py --pred_file Predictions/ngram_eval/chatglm2-6b-8k/ --with_options
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        help='metric name from choices', required=True)
    parser.add_argument('--max_length', type=int, default=7500, choices=[1500, 3500, 7500])
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')
    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want to test all multiple choice tasks')
    args = parser.parse_args()
    key_data_pairs = {}

    model_path = "THUDM/chatglm2-6b"
    open_source_model = "chatglm2-6b-8k"

    if args.max_length == 1500:
        open_source_model = open_source_model.replace("8k", "2k")
    elif args.max_length == 3500:
        open_source_model = open_source_model.replace("8k", "4k")

    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    input(f"Your prediction file will be saved to: {data_save_path} \npress enter to confirm")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
