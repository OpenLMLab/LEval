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


from glob import glob

files = glob("vicuna-13b-16k/*.jsonl")


for file in files:
    new_samples = []
    samples = read_jsonl(file)
    for sample in samples:
        if sample["evaluation"] == "LLM" or sample["evaluation"] == "human":
            continue
        new_samples.append(sample)
    write_jsonl(new_samples, file)
