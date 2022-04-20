import torch
import json
import os

from typing import Dict, Optional


CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"

ANSWER_NUM_DICT = {"ufet": 10331, "onto": 89, "figer": 113, "bbn": 56}

# Specify paths here
BASE_PATH = "./"
FILE_ROOT = "../data/"
EXP_ROOT = "../model"
ONTOLOGY_DIR = "../data/ontology"

TYPE_FILES = {
    "ufet": os.path.join(ONTOLOGY_DIR, "ufet_types.txt"),
    "onto": os.path.join(ONTOLOGY_DIR,  "ontonotes_types.txt"),
    "figer": os.path.join(ONTOLOGY_DIR, "figer_types.txt"),
    "bbn": os.path.join(ONTOLOGY_DIR, "bbn_types.txt")
}


def load_vocab_dict(
  vocab_file_name: str,
  vocab_max_size: Optional[int] = None,
  start_vocab_count: Optional[int] = None,
  common_vocab_file_name: Optional[str] = None
) -> Dict[str, int]:
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    if vocab_max_size:
      text = text[:vocab_max_size]
    if common_vocab_file_name:
        print("==> adding common training set types")
        print("==> before:", len(text))
        with open(common_vocab_file_name, "r") as fc:
            common = [x.strip() for x in fc.readlines()]
        print("==> common:", len(common))
        text = list(set(text + common))
        print("==> after:", len(text))
    if start_vocab_count:
      file_content = dict(zip(text, range(0 + start_vocab_count, len(text) +
                                          start_vocab_count)))
    else:
      file_content = dict(zip(text, range(0, len(text))))
  return file_content


def load_marginals_probs(file_name, device):
  with open(file_name) as f:
    marginals = [l.strip().split("\t") for l in f]
  marginals = torch.tensor([float(m) for t, m in marginals], device=device)
  return marginals


def load_conditional_probs(file_name, type_dict, device):
  with open(file_name) as f:
    probs = [json.loads(l) for l in f]
  id2pair, id2conditional = \
    [[int(type_dict[p["x"]]), int(type_dict[p["y"]])]
     for p in probs], [float(p["p(x|y)"]) for p in probs]
  id2pair = torch.tensor(id2pair, device=device)
  id2conditional = torch.tensor(id2conditional, device=device)
  id2conditional = torch.cat(
    [
      id2conditional.unsqueeze(-1),
      (1. - id2conditional).unsqueeze(-1)
    ], dim=-1)
  return id2pair, id2conditional