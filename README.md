<div align="center">
<img src="figs/logo.png" border="0" width=450px/>
</div>

------
### *L-Eval: Instituting Standardized Evaluation for Long Context Language Models*

L-Eval ([preview on 🤗 HuggingFace Datasets](https://huggingface.co/datasets/L4NLP/LEval) • [check our 📃 paper](https://arxiv.org/abs/2307.11088) ) is a comprehensive long-context language models evaluation suite with 18 long document tasks across multiple domains that require reasoning over long texts, including summarization, question answering, in-context learning with long CoT examples, topic retrieval, and paper writing assistance. L-Eval is a high-quality test set with 508 long documents and  2197 manually labeled query-response pairs.   
Currently, there have been great efforts invested in the expansion of context length for large language models. 
But it remains unclear whether extending the context can offer substantial gains over traditional methods such as retrieval, and to what extent it improves upon their regular (short context) counterparts  in practical downstream tasks. 

We hope L-Eval could help researchers and developers track the progress of long-context language models (LCLMs) and understand the strengths/shortcomings of different methods. We will also keep up with the **latest releases** of instruction-following LCLMs.

#### Features of this repo:
- 🧐 [How to get the data](#use)  
- 📏 [How to evaluate your models](#eval)  
- 📨 [How to submit your results](#submit)  
- 🔖 [View the Leaderboard](https://l-eval.github.io) 
- 🧭️ [Handle CUDA OOM with memory-efficient inference](#inference)
- 🖇️ [Build a retrieval-based baseline with Langchain](#tool)  
- ✏️ [Annotate & filter QA pairs from local jsonl files with web](#tool)

#### Overview:
<div align="center">
<img src="figs/lclms_bar.png" border="0" width=850px/>
</div>


## 🔥 Updates of L-Eval 
- **[2023.8.30]** We have annotated two new closed ended tasks:  (i) A [scientific fiction](https://github.com/OpenLMLab/LEval/blob/main/LEval-data/Closed-ended-tasks/sci_fi.jsonl) dataset to test the loyalty to input and (ii) a [code understanding](https://github.com/OpenLMLab/LEval/blob/main/LEval-data/Closed-ended-tasks/codeU.jsonl) dataset. Details can be found in our paper [v3](). 📢 **L-Eval** has been supported by [OpenCompass](https://github.com/internLM/OpenCompass/). You can  test L-Eval together with other benchmarks for foundation models here.
- [2023.8.17] We have tested some recently released models based on **Llama2** via NTK w/o training[[code]](https://github.com/OpenLMLab/LEval/blob/main/Baselines/llama2-chat-test.py) and PI (vicuna1.5-16k trained on ShareGPT)[[code]](https://github.com/OpenLMLab/LEval/blob/main/Baselines/vicuna-test.py).  Chatglm2-32k has also been included and the results for these models will be released soon.
- [2023.8.14] **Coursera** has been updated to improve the difficulty and please download the newest version. We're sorry for the inconvenience. We are also annotating a new **code test set**.
- [2023.8.01]  Predictions of LCLMs tested in this paper are available [here](https://drive.google.com/drive/folders/1pPbIXw0eRD_XZVMixZL4BG_SrMwFH3SH?usp=sharing) and judgements from gpt4 are available [here](https://drive.google.com/drive/folders/1bUGs-2isRLaY5xCz8k3mkKDArX6WxX0u?usp=sharing). 
We hope these can help researchers analyze different models and metrics. We also add a related work section discussing other long sequences benchmarks.  

Please check our paper [v3](https://arxiv.org/abs/2307.11088) for more details.

## Folders
The repository is structured as follows:

```bash
├── Baselines/ # scripts to generate the prediction files with baseline models
├── Baselines-light/ # scripts to generate the prediction files with 24G gpus
├── Evaluation/ # evaluation scripts
├── LEval-data/ # test samples
│   ├── Exam/ # exact match tasks (like multiple-choice)
│   │   ├── test_file.jsonl 
│   │   └── ...
│   ├── Generation/ # generation tasks
│   │   ├── test_file.jsonl
│   │   └── ...
├── Predictions/ # output of models
│   ├── exam_eval/trubo-16k-0613
│   │              ├── <task_name>.pred.jsonl
│   │              └── ... 
│   ├── llm_gpt4_eval  
│   │             ├──<model_name>.pred.jsonl
│   ├── ...
├── Tools/ # useful scripts
├── figs/ # figures
├── LICENSE
└── README.md
```

<a name="use"></a>
## Quick use
#### Step 1. Download the data 
It is easy to load the 20 test data in one line with huggingface datasets, and we give the example scripts:
```python
from datasets import load_dataset

datasets = ["coursera", "gsm100", "quality", "topic_retrieval_longchat", "tpo", "codeU", "sci_fi" ,"financial_qa", "gov_report_summ", "legal_contract_qa", "meeting_summ", "multidoc_qa", "narrative_qa", "natural_question", "news_summ", "paper_assistant", "patent_summ", "review_summ", "scientific_qa", "tv_show_summ"]

for testset in datasets:
    data = load_dataset('L4NLP/LEval', testset, split='test')
    # evaluate your model
```

You can also directly clone this repo:
```
git clone https://github.com/OpenLMLab/LEval.git
```
The test data is in `LEval-data`.

Each long document has multiple queries and corresponding responses. The format of each sample is as follows:
```json
{
    "instructions": ["What is the main goal of data science?\nA. Analyze and predict future trends\nB. Generate massive amounts of data\nC. Answer questions using data\nD. Increase the use of technology", "..."], // a list of instructions (questions need LLMs to answer)
    "outputs": ["C","A", "..."], // the ground truth or reference of corresponding instructions
    "input": "A very long document", // LLMs need to respond to instructions based on this long document.
    "source": "domain the document belongs to", // meeting, narrative_qa, etc.
    "evaluation": "Metrics used for evaluation" // e.g., exam, human, LLM, ROUGE, F1, etc.
}
```

#### Step 2. Generate your prediction files
We test all the baselines with a single 80G A800 GPU. If you encounter the OOM problem, please refer to [multiple GPUs inference](#inference). To generate the output files, you need to add a new file to `Baseline` folder and then replace the model name with your own model. An example of testing chatglm on all closed-ended tasks:
```
python Baselines/chatglm2-test.py --gpu 0 --metric exam_eval (exam_eval, ngram_eval , llm_gpt4_eval, llm_turbo_eval, human_eval)
```
where `--metric` means which metric you want to use (e.g., we use `exam_eval` for closed-ended tasks). Details about metrics in L-Eval can be found in the next section. The script will print out the path to the prediction file and you need to press enter to confirm.

#### Step 3. Evaluate the prediction file
Based on the `--metric` passed in Step 2, you can choose one of the scripts from `Evaluation/auto_eval.py`,  `Evaluation/llm_eval.py`, and `Evaluation/web_human_eval.py`. Then run the following command:
```
python Evaluation/auto_eval.py --pred_file Predictions/exam_eval/<your model>/coursera.pred.jsonl 
```
Examples of using the  `Evaluation/llm_eval.py`, and `Evaluation/web_human_eval.py` can be found [here](#eval_script)

<a name="eval"></a>
## How to Evaluate on L-Eval
In this part, we explain the metrics we used and how to run the evaluation scripts.

### Metrics used in L-Eval
L-Eval does not only contain open-ended questions (e.g.: multiple choice)  considering that in real-world applications, the generated answer  may not be exactly the same as the reference for long documents tasks. L-Eval is mainly divided into **two groups**: `Close-ended` and `Open-ended` and we use different evaluation metrics for each group.
#### Closed-ended tasks
  - Multiple Choice Question (single correct option). Example predicted answer: `A`
  - Multiple-Answer Questions (multiple correct options). Example predicted answer: `BCD`
  - Math Word Problems. Example predicted answer: `3`
  - Topic Retrieval. Example predicted answer: `The benefits of volunteering`
 
 The only evaluation metric used in these tasks takes the format of *Exact Match*  `"evaluation": "exam"` like grading exam papers.
 The total score is 100 and the score on each question is `100/(number of questions)`. For Multiple-Answer Questions, if the predicted answer does not cover all correct answers, it will only achieve a **quarter** of the score on this question. For example, if the correct answer is `ABC` and the predicted answer is `AC`, the score on this question is `0.25 * [100/(number of questions)]`.

#### Open-ended tasks 
- Summarization (Summarize a long document into a short paragraph). Example predicted answer: `This paper proposes a new method for ...`
- Abstractive Question Answering (Answer questions based on a long document). Example predicted answer: `The main goal of data science is to answer questions using data.`
- Writing Assistance (Assist in writing part of the long document). Example predicted answer: `2 Related Work\n Recent study has shown that ...`

we use the following metrics to evaluate the performance of generation tasks:
- **GPT4** Evaluation, `"evaluation": "LLM"`: We suggest battling with `turbo-16k-0613` and reporting `Win % vs turbo-16k-0613`. If your model is powerful enough, you can also directly compare it with the most powerful model `Claude-100k`, reporting `Win % vs Claude-100k`. We filter 17 long documents with **96* QA pairs for using GPT4 evaluator and the inference cost for this subset is about **$3**
- **GPT3.5** Evaluation (biased), `"evaluation": "LLM"` and  `"evaluation": "human"`: The evaluation step is similar to GPT4 evaluation which is cheaper but not accurate as GPT4. It serves as an alternative for researchers who do not have access to the GPT-4 API. We involve more samples for GPT3.5 Evaluation which is 29 long documents with 181 questions and the inference cost for this subset is about **$1**.
- **N-gram Match** Evaluation (biased),  `"evaluation": "f1" or "rouge"`: Using traditional automatic metrics like F1, ROUGE, etc. The low cost of automatic metrics makes it possible to evaluate all samples in L-Eval.
- **Human** Evaluation, ` "evaluation": "human"`: The annotators are asked to give a score from `1` to `5`. Automatic metrics can't replace the human evaluation and we filter **12 long documents with 85 questions** for human evaluation, each of which has 3 references: [human-written, GPT4-32k, and Claude-100k]([https://github.com/OpenLMLab/LEval/blob/main/Predictions/human_eval](https://github.com/OpenLMLab/LEval/blob/main/Predictions/human_eval/claude.gpt4.ref.jsonl)). you can visualize and score the results with `python Evaluation/web_for_human_eval.py`.
  
❗️**Notice**: For open-ended tasks,  models are informed of the ground truth length via a length instruction,e.g,  *We need a 20 words summary* where 20 is the length of reference answer to reduce the length bias in automatic evaluators.

**Explanation**
1. The n-gram matching metrics like f1 are very sensitive to the *length* of ground truth (length bias). In our preliminary experiments, the turbo-16k model achieved a very poor f1 score because it usually generates a very lengthy answer with an explanation which decreases the f1 score. 
To reduce the length bias, we suggest adding the length instruction (e.g., please answer with 10 words) while testing ngram metrics: *rouge* and *f1*.
2. LLM evaluators also have length biases as they tend to prefer detailed answers. In a pairwise comparison scenario, where it's impossible to feed the entire document, responses with additional or even inaccurate details may receive a higher rating. It's also challenging to judge the adequacy of a detailed summary against a one-sentence reference summary. Therefore, aligning the prediction's granularity with the ground truth ensures a more equitable assessment. 


<a name="eval_script"></a>
### Evaluation Scripts
- To run our evaluation scripts for automatic evaluation, you need to preprocess your output file in the format of `jsonl files` in [exam_eval](https://github.com/OpenLMLab/LEval/tree/main/Predictions/exam_eval/) and [ngram_eval](https://github.com/OpenLMLab/LEval/tree/main/Predictions/ngram_eval/) folders. Assuming you are going to evaluate the output of `turbo-16k-0613` on a multiple choice task `coursera`, you can run the following cmd:
```
python Evaluation/auto_eval.py --pred_file Predictions/exam_eval/turbo-16k-0613/coursera.pred.jsonl 
```

- To run our evaluation scripts for GPT4/Turbo3.5 evaluation, you have to provide the `api key` in `Evaluation/llm_eval.py` and then run:
```
python Evaluation/llm_eval.py --pred_path /path/to/<your model>.pred.jsonl  --judge_model gpt-4 (or gpt-3.5-turbo) --battle_with turbo-16k-0613 (or claude-100k)
```
where `--pred_path` means the prediction file. Example prediction files of `Claude-100k (vs turbo-16k)` are available: [for gpt4 evaluation](https://github.com/OpenLMLab/LEval/tree/main/Predictions/llm_gpt4_eval/claude-100k.pred.jsonl) and [for turbo3.5 evaluation](https://github.com/OpenLMLab/LEval/tree/main/Predictions/llm_turbo_eval/claude-100k.pred.jsonl)

- For human evaluation, we provide a very easy-to-use flask web app running on `localhost 127.0.0.1:5000`. You need to copy your prediction file `<model_name>.pred.jsonl` (samples with `evaluation: human`) to the `Predictions/human_eval` folder and then run:
```
python Evaluation/web_human_eval.py  --mode begin (or continue)
```
where `--mode` denotes whether you are starting a new evaluation or continuing your previous annotation.  Feel free to close the browser and set `--mode continue` to continue from your last annotation. Once running the script, you have to provide the annotator name and your annotation results will be saved to `Predictions/human_eval/annotation_from_<name>.jsonl`.
See the running screenshot [here](#human_demo). We  have provided the prediction files from 5 popular models as baselines for human evaluation. if you want to add outputs from other baselines, you can also move the corresponding prediction file to the `Predictions/human_eval` folder.


<a name="submit"></a>
## How to Submit
The [leaderboard](https://l-eval.github.io) contains 5 parts: `Exact Match, GPT-4 evaluator, GPT-3.5 Evaluator, F1, ROUGE`,

To submit your results on our leaderboard, you can send an email to `levalbenchmark@gmail.com`. 
#### Your submission should include 4 things:

* Metadata: Model name, number of parameters, and links to your paper/blog/GitHub/demo.
* Output files: Please submit 1 folder named with your model (e.g., `Predictions/turbo-16k-0613` ) for ngram matching evaluation and a jsonl file, e.g., `Predictions/LLM_Eval/claude100k.pred.jsonl`(The file naming format is `model_name.pred.jsonl`) for  LLM evaluation, as described in [Evaluation scripts section](#eval).
* Results: Please submit the results produced by our evaluation scripts. Results should contain all keys in the  [leaderboard](https://l-eval.github.io).
* Judgements from turbo3.5 and gpt4 (The output file produced by `llm_eval.py`)

We will randomly verify some results with the submitted output files.

#### Explanation of keys in the leaderboard

1. Keys in [Exact Match](https://l-eval.github.io)
   - `Avg`:  averaging over 4 datasets performance score.
   - `Max-Ctx`: the maximum context length of your model.
   - `Tokens`: the number of input tokens in experiments.
   - `Ret.`: whether using retrieval.
   - `PE`: whether doing prompt engineering (e.g., modifying the original prompt to improve the performance,  providing in-context examples).
   - `IDD`: whether using in-domain data (e.g.  data from qmsum, narrative_qa training set) into further finetuning. **Please don't hack this evaluation set**. But considering most of the sources are open, if your dataset potentially contains some in-domain data, you don't need to remove them. In that case, please set this value to 'yes'. If the construction of the IFT data is not transparent, you can leave it blank.
2. Keys in [F1_and ROUGE](https://l-eval.github.io) 
   - `F1 avg`:  the average over each dataset’s overall F1 score on QA-style tasks
   - `ROUGE avg`: the average over each dataset’s overall ROUGE-L score on Summarization-style tasks.
   - `Length`: the average length of the generated outputs.
3. Keys in [GPT-4/3.5 Evaluator](https://l-eval.github.io)
    - `n_wins`: number of wins including results of swapping the position of two answers.
    - `n_draws` number of draws including results of swapping the position of two answers.
    - `win % vs turbo16k` The win rate of your model in the battle with `turbo-16k-0613`
    - `Length`: the average length of the generated outputs.

<a name="inference"></a>
## Memory-efficient inference and multiple GPUs inference
### Using Flash Attention during inference 🚀
Please first try [Flash Attention](https://github.com/Dao-AILab/flash-attention) if you have a **80G** GPU. Based on our experiments, it works well when the sequence length is less than 32k (Flash-attn v2).  if you still encounter OOM, please refer to the next section.
If you are using LLaMA, we support FlashAttention in inference which can save your gpu memory, please add the param `--flash`.  The code is similar for other models.
Flash attention for Chatglm is implemented with torch2.0. Please ensure that you have successfully installed it.

If you encounter installation issues, it's likely due to the CUDA and Torch versions mismatch. Here is my running env:
```
python>=3.8
torch==1.13.1+cu117
CUDA Driver Version: 525.105.17   CUDA Toolkit: 11.7
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
[if flashAttn-v1] git checkout tags/v1.0.0 
python setup.py install
```

```
python Baselines/longchat-test.py --task_path LEval-data/Open-ended-tasks/narrative.jsonl --max_length 16k --gpu 0 --metric ngram_eval --flash 
```

### Memory-efficient inference with [LightLLM](https://github.com/ModelTC/lightllm) 🚂

Using lightLLM can make the inference procedure on a single or multiple 24G GPUs by optimizing the storage of KV cache but sacrificing inference speed.

#### Installation
1. Download L-Eval and the [data](https://github.com/OpenLMLab/LEval#step-1-download-the-data).
2. Install LightLLM according to the [official instructions](https://github.com/ModelTC/lightllm#get-started).

#### Examples of running L-Eval with LightLLM
> You must first download the model you would like to evaluate. LightLLM does not support automatic downloads yet.

> Code for running L-Eval with LightLLM is located in the `Baselines-light` directory.

The following command evaluates vicuna-7b-v1.5-16k on 4 RTX 3090 GPUs.
```bash
python Baselines-lightllm/vicuna-test.py --metric exam_eval --max_length 16k --model_path /.../.../vicuna-7b-v1.5-16k/ --lightllm_extra_args "--tp 4 --max_total_token_num 130000 --trust_remote_code --max_req_total_len 16384 --max_req_input_len 15900"
```

> You don't actually need 4 GPUs to run this example. But performance will improve with more GPUs.

`--lightllm_extra_args` are extra arguments passed to LightLLM server. View the [LightLLM documentation](https://github.com/ModelTC/lightllm/blob/main/docs/ApiServerArgs.md) for more information on how to set these arguments. `model_dir` is automatically passed and do not need to be specified again.

The script assumes LightLLM server is listening on port `8000`.

#### Known Issues
LightLLM server process might not properly terminate after the evaluation script stops. If you don't have other Python processes running, you can run `killall -HUP python` to terminate LightLLM server.


## Other Tools
<a name="tool"></a>
### Using Langchain to build retrieval-based baselines
You can use the script `turbo4k-retrieve-test.py` in `Baselines` to enhance a regular LLM with a sparser or dense retriever. An example is as follows:
```
python Baselines/turbo4k-retrieve-test.py --metric exam_eval (or ngram_eval, human_eval, llm_turbo_eval, llm_gpt4_eval) --retriever BM25 (or AdaEmbedding)
```
The retrieval-based method is implemented with [langchain](https://github.com/hwchase17/langchain). If you want to use BM25 retriever, please first install [Elasticsearch](https://github.com/elastic/elasticsearch). If you want to try ada embedding (cheap but effective), please fill your api-key.
 

### A flask-based annotation website for jsonl files
We have also released a very easy-to-use annotation website for L-Eval and make sure you have installed flask.
Firstly, you have to preprocess your files into a jsonl format which should contains 3 keys `input:str`, `instructions:list` and, `outputs:list` (see the examples in `LEval-data` folder).
To annotate new instruction-output pairs, please run the script to view and annotate the local jsonl file:
Start running the website on `127.0.0.1:5000` by:
```
python Tools/web_annotate_jsonl.py --path LEval-data/Generation/meeting_summ.jsonl --mode begin --new_pairs_num 2
```
where `--new_pairs_num` means the number of new QA pairs you want to add and `--mode` (begin or continue) means whether you want to continue from previous annotation results. 
The input file denoted by `--path` should be a `jsonl` file like the examples in `LEval-data` folder.  In this case, we annotate two new QA pairs based on the long input. After clicking `submit`, the results will be saved to the disk.

#### Example of our annotation website
<div align="center">
<img src="figs/annotation.png" border="0" width=660px/>
</div>

<a name="human_demo"></a>
#### Example of the human evaluation website
<div align="center">
<img src="figs/human_eval.png" border="0" width=660px/>
</div>
You can score the outputs from different models via the website. After completing the annotation, the result page is like:
<div align="center">
<img src="figs/res_page.png" border="0"/>
</div>

## Acknowledgement
This work is done by Fudan University and The University of Hong Kong.
Primary contributors: Chenxin An, Shansan Gong, Ming Zhong, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu.

We also thank the following people for their valuable suggestions and contributions: Siyu Ren, Zhiyong Wu,  Qinyuan Cheng, Bo Wang

**We sincerely appreciate the assistance provided by the following works for L-Eval**:
- We download the videos to form the long documents from [Coursera website](https://www.coursera.org/)
- we extract 100 math problems from  [GSM8k](https://github.com/openai/grade-school-math) and we use 8 long examples from [{chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub/blob/main/gsm8k/lib_prompt/prompt_hardest.txt)
- topic retrieval data is collected from [LongChat](https://github.com/DachengLi1/LongChat)
- QuALITY is from [their official github](https://github.com/nyu-mll/quality)
- TOEFL Practice Online data comes from [TOEFL-QA](https://github.com/iamyuanchung/TOEFL-QA/tree/master) 
Other open-sourced datasets are collected from: [gov_report](https://gov-report-data.github.io/),  [cuad](https://github.com/TheAtticusProject/cuad), [qmsum](https://github.com/Yale-LILY/QMSum),  [Multidoc2dial](https://doc2dial.github.io/multidoc2dial)
 [narrativeQA](https://github.com/deepmind/narrativeqa), [Natural Questions](https://github.com/google-research-datasets/natural-questions), [review advisor](https://github.com/neulab/ReviewAdvisor), [multi-news](https://github.com/Alex-Fabbri/Multi-News)
[bigpatent](https://evasharma.github.io/bigpatent/), [SPACE](https://github.com/stangelid/qt), [Qasper](https://github.com/allenai/qasper-led-baseline), [SummScreen](https://github.com/mingdachen/SummScreen)

Please kindly cite the [original papers](https://github.com/OpenLMLab/LEval/blob/main/citation.bib) when using L-Eval.
Thanks again for their effort!!  

## Citation
```
@misc{an2023leval,
      title={L-Eval: Instituting Standardized Evaluation for Long Context Language Models}, 
      author={Chenxin An and Shansan Gong and Ming Zhong and Mukai Li and Jun Zhang and Lingpeng Kong and Xipeng Qiu},
      year={2023},
      eprint={2307.11088},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



