import torch
from transformers import AutoTokenizer
import transformers
from transformers import pipeline
import numpy as np
import argparse
from LEval_config import *
from tqdm import tqdm


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
            while num_tokens_from_string(document, tokenizer) > doc_len:
                document = " ".join(document.split(" ")[:doc_len - cnt])  # chunk the input len into 16k tokens
                cnt += 250

            instructions = d['instructions']
            outputs = d['outputs']
            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "topic" in file_name or "code" in file_name:
                    context = document + "\n\n" + inst
                    message =  sys_prompt + context
                elif args.metric == "ngram_eval":
                    context = "Document is as follows. {document} \nQuestion: {inst} " + f"The suggested output length is around {len(out.split())} words. "
                    message =  sys_prompt + context
                else:
                    context = "Document is as follows. {document} \nQuestion: {inst} "
                    message = sys_prompt + context

                try:
                    text_inputs = message.format(document=document, inst=inst)
                except:
                    text_inputs = message

                text_inputs += "\nThe answer is: "
                save_d['prompt'] = message.replace(document, '<long input>')
                pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = pipe(text_inputs, max_new_tokens=512, do_sample=False, use_cache=True)[0]['generated_text'][len(message):]
                save_d[f'{open_source_model}_pred'] = output
                save_d['evaluation'] = d['evaluation']

                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "\nAnswer:"
                    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        output = pipe(text_inputs, max_new_tokens=512, do_sample=False, use_cache=True)[0][
                                     'generated_text'][len(message):]
                    save_d[f'{open_source_model}_pred'] += f" [fact: {output}]"

                if start_idx < 5:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print("[document]:",text_inputs[:100] + "...")
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1

                fw.write(json.dumps(save_d) + '\n')
                # break
        fw.close()
        # break

"""
MPT has strict requirement on running envs
if you get an error, try installing:
triton==2.0.0.dev20221202
torch==1.13.1+cu117
transformers==4.29.2
flash-attn==1.0.3.post0
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval"], help='metric name from choices', required=True)
    parser.add_argument('--task_path', type=str, default=None,
                        help= 'set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want use multiple choice dataset')
    args = parser.parse_args()
    key_data_pairs = {}
    doc_len = 7500
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    model_path = "mosaicml/mpt-7b-storywriter"

    open_source_model = model_path.split("/")[-1]

    config = transformers.AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    config.attn_config['attn_impl'] = 'triton'
    config.init_device = device  # For fast initialization directly on GPU!
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
