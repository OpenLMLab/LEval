import math
from functools import partial

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
# -*- coding:utf-8 -*-
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from LEval_config import *
from tqdm import tqdm


class NTKRotaryEmbedding(torch.nn.Module):
    def __init__(
        self, dim, alpha, max_position=4096, max_position_embeddings = 4096, base=10000, device=None
    ):
        super().__init__()
        self.base = base
        self.alpha = alpha
        self.update_base = base
        self.dim  = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position
        # print(f"Monkey Patching condense ratio {ratio}")
        t = (
            torch.arange(
                self.max_seq_len_cached,
                device=self.inv_freq.device,
                dtype=self.inv_freq.dtype,
            )
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


    def forward(self, x, seq_len=None):
        # update the base based on prompt:
        if seq_len > self.max_seq_len_cached:
            if self.alpha == 1:
                if x.shape[2] > self.max_seq_len_cached:
                    scale_up = math.ceil((seq_len + max_new_tokens) / self.max_seq_len_cached) # prompt + max_len
                    self.update_base = self.base * (scale_up ** (self.dim / (self.dim-2)))
            else:
                self.update_base = self.base * (self.alpha ** (self.dim / (self.dim - 2)))

            # x: [bs, num_attention_heads, seq_len, head_size]
            # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
            inv_freq = 1.0 / (self.update_base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            t = (
                torch.arange(
                    seq_len, device=x.device, dtype=self.inv_freq.dtype
                )
            )
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def replace_llama_with_ntkEmb():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(
        NTKRotaryEmbedding, max_position=4096, alpha=args.ntk_alpha
    )


def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(file_name, "w")
        data = key_data_pairs[file_name]
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        sys_prompt = get_sys_prompt(args, file_name)

        for d in tqdm(data):
            document = d['input']
            cnt = 0
            while num_tokens_from_string(document, tokenizer) > max_length:
                if "code" not in file_name:
                    document = " ".join(document.split(" ")[:max_length - cnt]) # chunk the input len from right
                else:
                    document = " ".join(document.split(" ")[cnt - max_length:]) # chunk the input len from left
                cnt += 250                

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "codeU" in file_name:
                    context = document + "\n\n" + inst
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context
                elif "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {document} \nQuestion: {inst}.  Please directly give the answer without any additional output or explanation "
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
                    message += "\nAnswer:"
                else:
                    context = "Document is as follows. {document} Instruction: {inst} " + f"\nAnswer this question with {len(out.split())} words."
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
                try:
                    text_inputs = message.format(document=document, inst=inst)
                except:
                    text_inputs = message
                save_d['prompt'] = message.replace(document, "<long document>")

                inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                prompt_length = inputs.input_ids.size()[-1]
                output = tokenizer.decode(sample[0][prompt_length:])

                save_d[f'{open_source_model}_pred'] = output.replace('</s>', '')
                save_d['evaluation'] = d['evaluation']

                # test the factuality in scientific fiction
                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "Please directly answer without any additional output or explanation. \nAnswer:"
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                    sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                    prompt_length = inputs.input_ids.size()[-1]
                    output = tokenizer.decode(sample[0][prompt_length:])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        help='metric name from choices', required=True)
    parser.add_argument('--max_length', default="4k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument('--gpu', type=int, default=0)

    # for llama based model
    parser.add_argument('--scale', default='7b', choices=['7b', '13b'])

    # if you want to use NTK embedding
    parser.add_argument('--ntk_alpha', type=int, help='using fix ntk to extend ntk_alpha times context length ', default=1)
    parser.add_argument('--ntk_dyn', action='store_true', help='set this if you want to use dynamic ntk') # ntk_alpha is dynamically determined by the forward x => len(x) / 4k

    # set this if you do not want to use data from huggingface
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    # set this if you do not want to test a specific task
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')

    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want to test all multiple choice tasks')
    parser.add_argument('--flash', action='store_true', help='set this if you want to use flash attention')
    args = parser.parse_args()

    model_path = f"meta-llama/Llama-2-{args.scale}-chat-hf"
    open_source_model = f"llama2-{args.scale}-chat-" + args.max_length
    if args.ntk_alpha > 1:
        open_source_model += f"-ntkFix{args.ntk_alpha}"
        replace_llama_with_ntkEmb()
    elif args.ntk_dyn:
        open_source_model += f"-ntkDyn"
        replace_llama_with_ntkEmb()

    max_length = k_to_number(args.max_length) - max_new_tokens

    if args.flash:
        replace_llama_attn_with_flash_attn()

    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    input(f"Your prediction file will be saved to: {data_save_path}  , press enter to confirm...")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
