import json
import os
from tqdm import tqdm
from pathlib import Path
import random

input_path = "/data/hongbang/cognlp/data/ner/trex/data/full_data"
output_path = "/data/hongbang/cognlp/data/ner/trex/data/processed_data"
num = 20
train,dev,test = 9,0.05,0.95

def load_json(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data

files = [os.path.join(input_path,f) for f in os.listdir(input_path) if f.endswith(".json")]
files = files[0:num]
all_datas = []
for file in tqdm(files):
    datas = load_json(file)
    for data in datas:
        text = data['text']
        entities = data['entities']
        sentences_boundaries = data['sentences_boundaries']
        words_boundaries = data["words_boundaries"]

        # 修正word跨sentence的情况
        indexes = []
        for s_start, s_end in sentences_boundaries:
            for idx, (w_start, w_end) in enumerate(words_boundaries):
                if w_start < s_start and s_start < w_end:
                    indexes.append((idx, w_start, s_start, w_end))
                    break
        for step, (idx, w_start, s_start, w_end) in enumerate(indexes):
            words_boundaries[step + idx] = [w_start, s_start]
            words_boundaries.insert(step + idx + 1, [s_start, w_end])

    all_datas.extend(datas)
length = len(all_datas)
random.shuffle(all_datas)
train_prop = int(float(train) / (train + dev + test) *length)
dev_prop = int(float(dev) / (train + dev + test) * length)

train_datas = all_datas[:train_prop]
dev_datas = all_datas[train_prop:train_prop+dev_prop]
test_datas = all_datas[train_prop+dev_prop:]

print("Write the result to files...")
processed_datas = [train_datas,dev_datas,test_datas]
file_names = ["train.json","dev.json","test.json"]
for i,file_name in enumerate(file_names):
    with open(os.path.join(output_path,file_name),"w") as f:
        json.dump(processed_datas[i],f)
print("Finished Preprocessing!")

