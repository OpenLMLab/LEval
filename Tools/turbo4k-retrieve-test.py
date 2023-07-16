# -*- coding:utf-8 -*-
import os
import sys
import time
import openai
import numpy as np
import argparse
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ElasticSearchBM25Retriever
# you need to start the elasticsearch server at http://localhost:9200 first

import json
from tqdm import tqdm
from jsonl_utils import *

def read_jsonl(train_fn):
    res = []
    with open(train_fn) as f:
        for i, line in enumerate(f):
            try:
                res.append(json.loads(line))
            except:
                continue
    print("loading ", len(res), "samples")
    return res

def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        print(len(data))

        if 'gsm' in file_name:
            args.gsm = True
        
        prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. \n\{context\} \nQuestion: \{question\}"
        i = 0

        for d in tqdm(data):
            if "llm" not in args.metric and (d['evaluation'] == 'human' or d['evaluation'] == 'LLM'):
                continue
            document = d['input']
            instructions = d['instructions']
            outputs = d['outputs']
            
            if args.gsm:
                document = "\nQuestion: ".join(instructions[0].split("\nQuestion: ")[:-1])
                instructions = ["Question: " + instructions[0].split("\nQuestion: ")[-1]]
            tmp_doc_name = f"./tmp/tmp_doc_{os.path.split(file_name)[-1].split('.')[0]}_{i}.txt"
            if not os.path.exists(tmp_doc_name):
                with open(tmp_doc_name, 'w', encoding='utf-8') as f_tmp:
                    f_tmp.write(document)

            loader = TextLoader(tmp_doc_name, encoding='utf-8')
            documents = loader.load()
            # import pdb; pdb.set_trace()
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1024, chunk_overlap=0, separator=' ')
            texts = text_splitter.split_documents(documents)

            if args.retriever == 'BM25':
                elasticsearch_url = "http://localhost:9200"
                retriever = ElasticSearchBM25Retriever.create(elasticsearch_url, f"langchain-index4-doc_{os.path.split(file_name)[-1].split('.')[0]}_{i}")
                retriever.add_texts([text.page_content for text in texts])
            else:
                embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
                docsearch = Chroma.from_documents(texts, embeddings)
                retriever = docsearch.as_retriever()

            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo-0613"), chain_type="stuff", retriever=retriever, return_source_documents=True)
            for inst, out in zip(instructions, outputs):
                if args.gsm:
                    message = f" Make sure the response is end with The answer is _ ."
                elif "ngram" in args.metric:
                    message = f" The suggested output length is around {len(out.split())} words."
                elif "topic" in file_name:
                    message = " Please directly give the concise and accurate answer."
                else:
                    message = " There could be a sinlge correct option or multiple correct options. Please only provide the letter corresponding to the answer (like A or AB) when answering."

                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                save_d['prompt'] = prompt + message
                save_d['evaluation'] = d['evaluation']

                query = inst + message
                result = qa({"query": query})

                save_d[f'{openai_model}_pred'] = result['result']
                save_d['source_documents'] = [r.page_content[:100] + '...' for r in result['source_documents']]
                
                print("----------------- [output] vs [ground truth] -----------------")
                print('[output]:', save_d[f'{openai_model}_pred'], "\n\n" , '[ground truth]:', save_d['gt'])

                fw.write(json.dumps(save_d) + '\n')
                        
                time.sleep(1.0)
            i += 1
        fw.close()
        # break

def to_filename(task_name):
    return  os.path.join(data_save_path, task_name + ".pred.jsonl")

if __name__ == "__main__":
    openai_key = ""

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval","llm_gpt4_eval","exam_eval", "ngram_eval"], default="auto_eval", help='metric name from "turbo_eval","gpt4_eval","auto_eval"')
    parser.add_argument('--task_path', type=str, default=None, help= 'set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    parser.add_argument('--task_name', type=str, default=None, help='set this if you want test a specific task from huggingface, example: coursera')
    parser.add_argument('--retriever', choices=["AdaEmbedding", "BM25"], default="BM25", help='choose retriever from "AdaEmbedding", "BM25"')
    parser.add_argument('--gsm', action='store_true', help='set this if you want to test gsm100 dataset')
    args = parser.parse_args()
    key_data_pairs = {}

    if args.task_name in datasets_closed_ended:
        args.metric = "exam_eval"
    elif args.task_name in datasets_open_ended:
        args.metric = "ngram_eval"

    if args.retriever == "BM25":
        openai_model = "retrieval-bm25-turbo-0613"
    else:
        openai_model = "retrieval-turbo-0613"
    data_save_path = f"Predictions/{args.metric}/{openai_model}"
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
