# Code adapted from https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test/blob/main/llama_rope_scaled_monkey_patch.py
from transformers import AutoTokenizer
import argparse
from LEval_config import *
from lightllm_helper import LightLLMServer, lightllm_infer
from tqdm import tqdm


def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        header = (
            "A chat between a curious user and an artificial intelligence assistant."
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )

        sys_prompt = get_sys_prompt(args, file_name)
        for d in tqdm(data):
            document = d['input']
            cnt = 0
            # truncate documents
            while num_tokens_from_string(document, tokenizer) > max_length:
                document = " ".join(document.split(" ")[:max_length - cnt])  # chunk the input len into 16k tokens
                cnt += 250

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name:
                    context = document + "\n\n" + inst
                    message = sys_prompt + context
                elif "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = header + " USER: " + sys_prompt + context
                    message += " \nASSISTANT: "
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {} \nQuestion: {} "
                    message = header + " USER: " + sys_prompt + context
                    message += " \nAnswer:"
                elif "coursera" in file_name:
                    context = "Document is as follows. {} Question: {} "
                    message = header + " USER: " + sys_prompt + context + "\n Please only give the correct options (e.g., A)."
                    message += " \nASSISTANT: "
                else:
                    context = "Document is as follows. {} \nInstruction: {} " + f"The suggested output length is around {len(out.split())} words. "
                    message = header + " USER: " + sys_prompt + context
                    message += " \nASSISTANT: My english answer is:"

                save_d['prompt'] = message.replace(document, "<long input>")
                output = lightllm_infer(message.format(document, inst), do_sample=False, max_new_tokens=max_new_tokens)
                save_d[f'{model_name}_pred'] = output
                save_d['evaluation'] = d['evaluation']
                if start_idx < 5:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{model_name}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1
                fw.write(json.dumps(save_d) + '\n')
        fw.close()
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        help='metric name from choices', required=True)
    parser.add_argument('--max_length', default="2k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument('--model_path', required=True, help='path to the downloaded model')
    parser.add_argument('--lightllm_extra_args', help="arguments passed to lightllm server")
    # set this if you do not want to use data from huggingface
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    # set this if you do not want to test a specific task
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')
    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want to test all multiple choice tasks')

    args = parser.parse_args()

    max_total_len = k_to_number(args.max_length)
    max_length = max_total_len - max_new_tokens
    model_path = args.model_path

    if model_path[-1] == "/":
        model_name = model_path.split("/")[-2]
    else:
        model_name = model_path.split("/")[-1]

    data_save_path = f"Predictions/{args.metric}/{model_name}"
    input(f"Your prediction file will be saved to: {data_save_path}  , press enter to confirm...")

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)

    lightllm_args = args.lightllm_extra_args.split() if args.lightllm_extra_args else []
    tokenizer = AutoTokenizer.from_pretrained(model_path, force_download=True, resume_download=False)
    with LightLLMServer(model_path, lightllm_args):
        main()