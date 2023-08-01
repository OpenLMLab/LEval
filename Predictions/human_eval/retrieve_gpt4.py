import json

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


all_samples = read_jsonl("claude.gpt4.ref.jsonl")
upt_samples = []

for sample in all_samples:
    instructions = sample["instructions"]
    gpt4_outputs = sample["gpt4_outputs"]
    gts = sample["outputs"]
    for inst, gpt4, gt in zip(instructions, gpt4_outputs, gts):
        upt_samples.append({
            "query":inst,
            "gpt4-x_pred": gpt4,
            "gt":gt,
            "prompt": inst
        })
write_jsonl(upt_samples, "gpt4-32k.pred.jsonl")