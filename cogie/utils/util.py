"""
@Author: jinzhuan
@File: utils.py
@Desc: 
"""
import torch
import torch.nn as nn
import numpy as np
import os
import random
import json
import pickle
from collections import OrderedDict
from pathlib import Path
import sys
import yaml
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += f"\t{k} : {str(v)}\n"
    print("\n" + info + "\n")


def init_logger(log_file=None, log_file_level=logging.DEBUG):
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def calc_metric(y_true, y_pred):
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


def find_triggers(labels):
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][1]])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]


def seed_everything(seed=527):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def prepare_device(n_gpu_use):
    if not n_gpu_use:
        device_type = 'cpu'
    else:
        n_gpu_use = n_gpu_use.split(",")
        device_type = f"cuda:{n_gpu_use[0]}"
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        device_type = 'cpu'
    if len(n_gpu_use) > n_gpu:
        msg = f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine."
        logger.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device(device_type)
    list_ids = n_gpu_use
    return device, list_ids


def model_device(n_gpu, model):
    device, device_ids = prepare_device(n_gpu)
    if len(device_ids) > 1:
        logger.info(f"current {len(device_ids)} GPUs")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device


def restore_checkpoint(resume_path, model=None):
    if isinstance(resume_path, Path):
        resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    states = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(states)
    else:
        model.load_state_dict(states)
    return [model, best, start_epoch]


def save_pickle(data, file_path):
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data, file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def save_numpy(data, file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.save(str(file_path), data)


def load_numpy(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.load(str(file_path))


def load_json(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data


def load_yaml(file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = yaml.load(f)
    return data


def json_to_text(file_path, data):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def save_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)


def load_model(model, model_path):
    if isinstance(model_path, Path):
        model_path = str(model_path)
    logging.info(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    # state = states['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(states)
    else:
        model.load_state_dict(states)
    return model


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def summary(model, *inputs, batch_size=-1, show_input=True):
    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if show_input is False and output is not None:
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out.size())[1:]
                            ][0]
                        else:
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out[0].size())[1:]
                            ][0]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-----------------------------------------------------------------------")
    if show_input is True:
        line_new = f"{'Layer (type)':>25}  {'Input Shape':>25} {'Param #':>15}"
    else:
        line_new = f"{'Layer (type)':>25}  {'Output Shape':>25} {'Param #':>15}"
    print(line_new)
    print("=======================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if show_input is True:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
        else:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )

        total_params += summary[layer]["nb_params"]
        if show_input is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]

        print(line_new)

    print("=======================================================================")
    print(f"Total params: {total_params:0,}")
    print(f"Trainable params: {trainable_params:0,}")
    print(f"Non-trainable params: {(total_params - trainable_params):0,}")
    print("-----------------------------------------------------------------------")


def module2parallel(module, device_ids):
    if len(device_ids) == 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module.to(device)
        return module
    elif len(device_ids) == 1:
        device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
        module.to(device)
        return module
    else:
        device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
        module.to(device)
        module_name = type(module).__name__
        module = getattr(sys.modules[module.__module__], module_name + "Parallel")(module, device_ids)
        return module


def get_all_forms(mention):
    forms = []
    words = mention.split(' ')
    forms.append('_'.join(words))
    for i in range(len(words)):
        words[i] = words[i][0].upper() + words[i][1:]
        forms.append('_'.join(words))
    return forms


def batch2tuple(batch):
    return tuple(zip(*batch))


def bio_tag_to_spans(tags, ignore_labels=None):
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'B':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'I' and prev_bio_tag in ('B', 'I') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'O':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]


def to_tensor(batch, device):
    data = []
    for key, value in batch.items():
        data.append(value)
    return tuple(data)


def get_samples_weight(datable, vocabulary):
    samples_weight = []
    for label_id in datable['triggers_y']:
        not_none = False
        for trigger in label_id:
            if trigger != vocabulary.to_index('O') and trigger != -1:
                not_none = True
                break
        if not_none:
            samples_weight.append(5.0)
        else:
            samples_weight.append(1.0)
    return np.array(samples_weight)

def el_load_candidates(entity_catalogue):
    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
    )

from colorama import init
from termcolor import colored

HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]


def el_print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) :]
    else:
        msg = input_sentence
        print("Failed to identify entity from text:")
    print("\n" + str(msg) + "\n")