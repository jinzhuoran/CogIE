"""
@Author: jinzhuan
@File: el_toolkit.py
@Desc: 
"""
from cogie import *
import threading
from ..base_toolkit import BaseToolkit
# import blink.predictor as predictor
from cogie.utils.cognet import CognetServer
from cogie.utils.util import get_all_forms
from ...models.el.crossencoder import ENT_START_TAG,ENT_END_TAG,ENT_TITLE_TAG
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import sys
from cogie.models.el.biencoder import BiEncoderRanker
from cogie.models.el.crossencoder import CrossEncoderRanker
from cogie.models.el.blink import DenseHNSWFlatIndexer
from cogie.utils.util import el_load_candidates
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import logging


class ElToolkit(BaseToolkit):
    def __init__(self, task='el', language='english', corpus=None):
        config = load_configuration()
        if language == 'english':
            if corpus is None:
                corpus = "wiki"
        self.task = task
        self.language = language
        self.corpus = corpus
        download_model(config[task][language][corpus])
        device = torch.device(config['device'])
        device_ids = config['device_id']
        max_seq_length = config['max_seq_length']
        super().__init__(device=device,device_ids=device_ids,max_seq_length=max_seq_length)
        if self.language == 'english':
            if self.corpus == 'wiki':
                path = config[task][language][corpus]['path']
                for key in config[task][language][corpus]['data']:
                    config[task][language][corpus]['data'][key] = absolute_path(path,config[task][language][corpus]['data'][key])
                biencoder_config = config[task][language][corpus]['data']['biencoder_config']
                crossencoder_config = config[task][language][corpus]['data']['crossencoder_config']
                self.biencoder_params = load_json(biencoder_config)
                self.crossencoder_params = load_json(crossencoder_config)
                self.biencoder_params["path_to_model"] = config[task][language][corpus]['data']['biencoder_model']
                self.crossencoder_params["path_to_model"] = config[task][language][corpus]['data']['crossencoder_model']
                self.biencoder = BiEncoderRanker(self.biencoder_params)
                self.crossencoder = CrossEncoderRanker(self.crossencoder_params)
                self.faiss_indexer = DenseHNSWFlatIndexer(1)
                self.faiss_indexer.deserialize_from(config[task][language][corpus]['data']['index_path'])
                self.convert_dict = {}
                title2id,id2title,id2text,wikipedia_id2local_id = el_load_candidates(config[task][language][corpus]['data']['entity_catalogue'])
                id2url = {
                    v: "https://en.wikipedia.org/wiki?curid=%s" % k
                    for k, v in wikipedia_id2local_id.items()
                }
                self.convert_dict = {
                    "title2id":title2id,
                    "id2title":id2title,
                    "id2text":id2text,
                    "wikipedia_id2local_id":wikipedia_id2local_id,
                    "id2url":id2url,
                }






    # def __init__(self, task='el', language='english', corpus=None):
    #     super().__init__()
    #     self.task = task
    #     self.language = language
    #     self.corpus = corpus
    #     config = load_configuration()
    #     download_model(config[task]['cognet'])
    #     path = config['el']['cognet']['path']
    #     file = config['el']['cognet']['data']['file']
    #     self.wikidata2wikipedia = load_json(absolute_path(path, file))
    #     self.cognet = CognetServer()
        # self.id2url, self.et_ner_model, self.et_models = predictor.get_et_predictor()

    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(ElToolkit, "_instance"):
            with ElToolkit._instance_lock:
                if not hasattr(ElToolkit, "_instance"):
                    ElToolkit._instance = object.__new__(cls)
        return ElToolkit._instance

    def run(self,ner_result):
        if self.language == 'english':
            if self.corpus == 'wiki':
                samples = []
                for mention in ner_result:
                    record = {}
                    record["label"] = 'unkonwn'
                    record["label_id"] = -1
                    for key in ["mention","context_left","context_right"]:
                        if key not in mention.keys():
                            raise ValueError("Key {} is not available in ner result!".format(key))
                        record[key] = " ".join(mention[key]).lower()
                    samples.append(record)
                _,biencoder_tensor_data = process_mention_data(samples,
                                                               self.biencoder.tokenizer,
                                                               self.biencoder_params["max_context_length"],
                                                               self.biencoder_params["max_cand_length"],
                                                               silent=True,
                                                               logger=None,
                                                               debug=self.biencoder_params["debug"],
                                                               )

                biencoder_sampler = SequentialSampler(biencoder_tensor_data)
                biencoder_dataloader = DataLoader(
                    biencoder_tensor_data, sampler=biencoder_sampler, batch_size=self.biencoder_params["eval_batch_size"]
                )
                labels, nns, scores = run_biencoder(
                    self.biencoder, biencoder_dataloader, candidate_encoding=None, top_k=10, indexer=self.faiss_indexer
                )
                context_input, candidate_input, label_input = prepare_crossencoder_data(
                    self.crossencoder.tokenizer, samples, labels, nns, self.convert_dict["id2title"], self.convert_dict["id2text"], keep_all=True,
                )
                context_input = el_modify(context_input, candidate_input, self.crossencoder_params["max_seq_length"])
                crossencoder_tensor_data = TensorDataset(context_input, label_input)
                crossencoder_sampler = SequentialSampler(crossencoder_tensor_data)
                crossencoder_dataloader = DataLoader(
                    crossencoder_tensor_data, sampler=crossencoder_sampler, batch_size=self.crossencoder_params["eval_batch_size"]
                )
                # CrossEncoder Prediction
                accuracy, index_array, unsorted_scores = run_crossencoder(
                    self.crossencoder,
                    crossencoder_dataloader,
                    logger=logging.Logger(__name__),
                    context_len=self.biencoder_params["max_context_length"],
                )
                el_result = []
                for entity_list,index_list,sample in zip(nns,index_array,ner_result):
                    e_id = entity_list[index_list[-1]]
                    e_title = self.convert_dict["id2title"][e_id]
                    e_text = self.convert_dict["id2text"][e_id]
                    e_url = self.convert_dict["id2url"][e_id]
                    el_result.append({
                        **sample,
                        "title":e_title,
                        "text":e_text,
                        "id":e_id,
                        "url":e_url,
                    })
                return el_result


    # def run(self, sentence):
    #     url = "https://en.wikipedia.org/wiki/"
        # links = predictor.run(10, *self.et_models, text=sentence, id2url=self.id2url, ner_model=self.et_ner_model)
        # for link in links:
        #     forms = get_all_forms(link["title"])
        #     cognet_link = "unk"
        #     for form in forms:
        #         wikipedia = url + form
        #         if wikipedia in self.wikidata2wikipedia:
        #             wikidata = self.wikidata2wikipedia[wikipedia]
        #             cognet_link = self.cognet.query("<" + wikidata + ">")
        #     link["cognet_link"] = cognet_link
        # return links

WORLDS = [
    'american_football',
    'doctor_who',
    'fallout',
    'final_fantasy',
    'military',
    'pro_wrestling',
    'starwars',
    'world_of_warcraft',
    'coronation_street',
    'muppets',
    'ice_hockey',
    'elder_scrolls',
    'forgotten_realms',
    'lego',
    'star_trek',
    'yugioh'
]

world_to_id = {src : k for k, src in enumerate(WORLDS)}

def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]

# Prepare Biencoder Data

def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc,
    tokenizer,
    max_seq_length,
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    context_key="context",
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)
        label_tokens = get_candidate_representation(
            label, tokenizer, max_cand_length, title,
        )
        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx)
    else:
        tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)
    return data, tensor_data


# Prepare CrossEncoder Data


def prepare_crossencoder_mentions(
    tokenizer,
    samples,
    max_context_length=32,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):

    context_input_list = []  # samples X 128

    # for sample in tqdm(samples):
    for sample in samples:
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )
        tokens_ids = context_tokens["ids"]
        context_input_list.append(tokens_ids)

    context_input_list = np.asarray(context_input_list)
    return context_input_list


def prepare_crossencoder_candidates(
    tokenizer, labels, nns, id2title, id2text, max_cand_length=128, topk=100
):

    START_TOKEN = tokenizer.cls_token
    END_TOKEN = tokenizer.sep_token

    candidate_input_list = []  # samples X topk=10 X 128
    label_input_list = []  # samples
    idx = 0
    for label, nn in zip(labels, nns):
        candidates = []

        label_id = -1
        for jdx, candidate_id in enumerate(nn[:topk]):

            if label == candidate_id:
                label_id = jdx

            rep = get_candidate_representation(
                id2text[candidate_id],
                tokenizer,
                max_cand_length,
                id2title[candidate_id],
            )
            tokens_ids = rep["ids"]

            assert len(tokens_ids) == max_cand_length
            candidates.append(tokens_ids)

        label_input_list.append(label_id)
        candidate_input_list.append(candidates)

        idx += 1
        sys.stdout.write("{}/{} \r".format(idx, len(labels)))
        sys.stdout.flush()

    label_input_list = np.asarray(label_input_list)
    candidate_input_list = np.asarray(candidate_input_list)

    return label_input_list, candidate_input_list


def filter_crossencoder_tensor_input(
    context_input_list, label_input_list, candidate_input_list
):
    # remove the - 1 : examples for which gold is not among the candidates
    context_input_list_filtered = [
        x
        for x, y, z in zip(context_input_list, candidate_input_list, label_input_list)
        if z != -1
    ]
    label_input_list_filtered = [
        z
        for x, y, z in zip(context_input_list, candidate_input_list, label_input_list)
        if z != -1
    ]
    candidate_input_list_filtered = [
        y
        for x, y, z in zip(context_input_list, candidate_input_list, label_input_list)
        if z != -1
    ]
    return (
        context_input_list_filtered,
        label_input_list_filtered,
        candidate_input_list_filtered,
    )


def prepare_crossencoder_data(
    tokenizer, samples, labels, nns, id2title, id2text, keep_all=False
):

    # encode mentions
    context_input_list = prepare_crossencoder_mentions(tokenizer, samples)

    # encode candidates (output of biencoder)
    label_input_list, candidate_input_list = prepare_crossencoder_candidates(
        tokenizer, labels, nns, id2title, id2text
    )

    if not keep_all:
        # remove examples where the gold entity is not among the candidates
        (
            context_input_list,
            label_input_list,
            candidate_input_list,
        ) = filter_crossencoder_tensor_input(
            context_input_list, label_input_list, candidate_input_list
        )
    else:
        label_input_list = [0] * len(label_input_list)

    context_input = torch.LongTensor(context_input_list)
    label_input = torch.LongTensor(label_input_list)
    candidate_input = torch.LongTensor(candidate_input_list)

    return (
        context_input,
        candidate_input,
        label_input,
    )

def run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    # for batch in tqdm(dataloader):
    for batch in dataloader:
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores


def el_modify(context_input, candidate_input, max_seq_length):
    new_input = []
    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in range(len(context_input)):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate
            sample = cur_input + cur_candidate[j][1:]
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels), outputs == labels

def evaluate(reranker, eval_dataloader, device, logger, context_length, zeshel=False, silent=True):
    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    acc = {}
    tot = {}
    world_size = len(WORLDS)
    for i in range(world_size):
        acc[i] = 0.0
        tot[i] = 0.0

    all_logits = []
    cnt = 0
    for step, batch in enumerate(iter_):
        if zeshel:
            src = batch[2]
            cnt += 1
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]
        label_input = batch[1]
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, label_input, context_length)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()

        tmp_eval_accuracy, eval_result = accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        all_logits.extend(logits)

        nb_eval_examples += context_input.size(0)
        if zeshel:
            for i in range(context_input.size(0)):
                src_w = src[i].item()
                acc[src_w] += eval_result[i]
                tot[src_w] += 1
        nb_eval_steps += 1

    normalized_eval_accuracy = -1
    if nb_eval_examples > 0:
        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    if zeshel:
        macro = 0.0
        num = 0.0
        for i in range(len(WORLDS)):
            if acc[i] > 0:
                acc[i] /= tot[i]
                macro += acc[i]
                num += 1
        if num > 0:
            logger.info("Macro accuracy: %.5f" % (macro / num))
            logger.info("Micro accuracy: %.5f" % normalized_eval_accuracy)
    else:
        if logger:
            logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)

    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
    return results

def run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, logger, context_len, zeshel=False, silent=True)
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    if accuracy > -1:
        predictions = np.argsort(logits, axis=1)
    else:
        predictions = []

    return accuracy, predictions, logits
