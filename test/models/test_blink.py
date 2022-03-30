from cogie import *
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import json

# Configurations
biencoder_params = {"data_path": "",
                    "bert_model": "bert-large-uncased",
                    "model_output_path": None,
                    "context_key": "context",
                    "lowercase": True,
                    "top_k": 10,
                    "max_seq_length": 256,
                    "evaluate": False,
                    "evaluate_with_pregenerated_candidates": False,
                    "output_eval_file": None,
                    "debug": False,
                    "silent": False,
                    "train_batch_size": 8,
                    "eval_batch_size": 8,
                    "data_parallel": False,
                    "max_grad_norm": 1.0,
                    "learning_rate": 3e-05,
                    "num_train_epochs": 1,
                    "print_interval": 5,
                    "eval_interval": 40,
                    "save_interval": 1,
                    "warmup_proportion": 0.1,
                    "no_cuda": True,
                    "seed": 52313,
                    "gradient_accumulation_steps": 1,
                    "out_dim": 100,
                    "pull_from_layer": -1,
                    "type_optimization": "all_encoder_layers",
                    "add_linear": False,
                    "shuffle": False,
                    "encode_batch_size": 8,
                    "max_context_length": 32,
                    "max_cand_length": 128,
                    "path_to_model":"blink_models/biencoder_wiki_large.bin",
          }

crossencoder_params={
    "data_path": "",
    "bert_model": "bert-large-uncased",
    "model_output_path": None,
    "context_key": "context",
    "lowercase": True,
    "top_k": 100,
    "max_context_length": 32,
    "max_cand_length": 128,
    "max_seq_length": 160,
    "evaluate": False,
    "evaluate_with_pregenerated_candidates": False,
    "output_eval_file": None,
    "debug": False,
    "silent": False,
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "data_parallel": False,
    "max_grad_norm": 1.0,
    "learning_rate": 3e-05,
    "num_train_epochs": 5,
    "print_interval": 10,
    "eval_interval": 2000,
    "save_interval": 1,
    "warmup_proportion": 0.1,
    "no_cuda": True,
    "seed": 52313,
    "gradient_accumulation_steps": 1,
    "out_dim": 1,
    "pull_from_layer": -1,
    "type_optimization": "all_encoder_layers",
    "add_linear": True,
    "shuffle": False,
    "segment": True,
    "path_to_model":'blink_models/crossencoder_wiki_large.bin',
}

entity_catalogue = "blink_models/entity.jsonl"
entity_encoding = "blink_modes/all_entities_large.t7"
faiss_index = "hnsw"
index_path = "blink_models/faiss_hnsw_index.pkl"

text = "The Russian and Ukrainians delegations have now arrived at the Dolmabahce - President Erdogan’s office on the banks of the Bosphorus here in Istanbul."
top_k = 10

# Load BiEncoder
biencoder = BiEncoderRanker(biencoder_params)

# Load CrossEncoder
crossencoder = CrossEncoderRanker(crossencoder_params)

# Load Indexer and all the 5903527 entities
faiss_indexer = DenseHNSWFlatIndexer(1)
faiss_indexer.deserialize_from(index_path)
title2id,id2title,id2text,wikipedia_id2local_id = el_load_candidates(entity_catalogue)
id2url = {
    v: "https://en.wikipedia.org/wiki?curid=%s" % k
    for k, v in wikipedia_id2local_id.items()
}

# Load NER model
ner_model = Flair()

# NER prediction
ner_output_data = ner_model.predict([text])
sentences = ner_output_data["sentences"]
mentions = ner_output_data["mentions"]
samples = []
for mention in mentions:
    record = {}
    record["label"] = "unknown"
    record["label_id"] = -1
    # LOWERCASE EVERYTHING !
    record["context_left"] = sentences[mention["sent_idx"]][
                             : mention["start_pos"]
                             ].lower()
    record["context_right"] = sentences[mention["sent_idx"]][
                              mention["end_pos"]:
                              ].lower()
    record["mention"] = mention["text"].lower()
    record["start_pos"] = int(mention["start_pos"])
    record["end_pos"] = int(mention["end_pos"])
    record["sent_idx"] = mention["sent_idx"]
    samples.append(record)
el_print_colorful_text(text, samples)

# Prepare Biencoder Input
_,biencoder_tensor_data = process_mention_data(samples,
                                     biencoder.tokenizer,
                                     biencoder_params["max_context_length"],
                                     biencoder_params["max_cand_length"],
                                     silent=True,
                                     logger=None,
                                     debug=biencoder_params["debug"],)
biencoder_sampler = SequentialSampler(biencoder_tensor_data)
biencoder_dataloader = DataLoader(
    biencoder_tensor_data, sampler=biencoder_sampler, batch_size=biencoder_params["eval_batch_size"]
)

# Biencoder Prediction
labels, nns, scores = run_biencoder(
    biencoder, biencoder_dataloader, candidate_encoding=None, top_k=top_k, indexer=faiss_indexer
)
idx = 0
for entity_list, sample in zip(nns, samples):
    e_id = entity_list[0]
    e_title = id2title[e_id]
    e_text = id2text[e_id]
    e_url = id2url[e_id]
    el_print_colorful_prediction(
        idx, sample, e_id, e_title, e_text, e_url, show_url=True
    )
    idx += 1

# Prepare CrossEncoder Input
context_input, candidate_input, label_input = prepare_crossencoder_data(
    crossencoder.tokenizer, samples, labels, nns, id2title, id2text, keep_all=True,
)
context_input = el_modify(context_input,candidate_input,crossencoder_params["max_seq_length"])
crossencoder_tensor_data = TensorDataset(context_input, label_input)
crossencoder_sampler = SequentialSampler(crossencoder_tensor_data)
crossencoder_dataloader = DataLoader(
    crossencoder_tensor_data, sampler=crossencoder_sampler, batch_size=crossencoder_params["eval_batch_size"]
)


















