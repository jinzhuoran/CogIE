from cogie import *
import torch
import torch.nn as nn
import torch.optim as optim
from cogie.core.metrics import EventMetric
from cogie.core.trainer import Trainer
from cogie.io.loader.ee.ace2005_casee import ACE2005CASEELoader
from cogie.io.processor.ee.ace2005_casee import ACE2005CASEEProcessor
import argparse
torch.cuda.set_device(4)
device = torch.device('cuda:4')
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--data_path", type=str, default='datasets/FewFC', help="Path of the dataset.")
    parser.add_argument("--test_path", type=str, default='datasets/FewFC/data/test.json', help="Path of the testset.")

    parser.add_argument("--output_result_path", type=str, default='models_save/results.json')
    parser.add_argument("--output_model_path", default="./models_save/model.bin", type=str, help="Path of the output model.")

    parser.add_argument("--model_name_or_path", default="bert-base-chinese", type=str, help="Path of the output model.")
    parser.add_argument("--cache_dir", default="./plm", type=str, help="Where do you want to store the pre-trained models downloaded")
    parser.add_argument("--do_lower_case", action="store_true", help="")
    parser.add_argument("--seq_length", default=128, type=int, help="Sequence length.")

    # Training options.
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lr_bert", type=float, default=2e-5, help="Learning rate for BERT.")
    parser.add_argument("--lr_task", type=float, default=1e-4, help="Learning rate for task layers.")
    parser.add_argument("--warmup", type=float, default=0.1, help="Warm up value.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch_size.")
    parser.add_argument("--epochs_num", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=5, help="Specific steps to print prompt.")

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay value")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout on BERT")
    parser.add_argument("--decoder_dropout", type=float, default=0.3, help="Dropout on decoders")

    # Model options.
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--w3", type=float, default=1.0)
    parser.add_argument("--pow_0", type=int, default=1)
    parser.add_argument("--pow_1", type=int, default=1)
    parser.add_argument("--pow_2", type=int, default=1)

    parser.add_argument("--rp_size", type=int, default=64)
    parser.add_argument("--decoder_num_head", type=int, default=1)

    parser.add_argument("--threshold_0", type=float, default=0.5)
    parser.add_argument("--threshold_1", type=float, default=0.5)
    parser.add_argument("--threshold_2", type=float, default=0.5)
    parser.add_argument("--threshold_3", type=float, default=0.5)
    parser.add_argument("--threshold_4", type=float, default=0.5)

    parser.add_argument("--step", type=str, choices=["dev", "test"])

    args = parser.parse_args()
    return args
config = parse_args()

loader = ACE2005CASEELoader()
train_data, dev_data, test_data = loader.load_all('../../../cognlp/data/ee/ace2005casee/data')
processor = ACE2005CASEEProcessor(schema_path='../../../cognlp/data/ee/ace2005casee/data/schema.json',
                                  trigger_path='../../../cognlp/data/ee/ace2005casee/data/trigger_vocabulary.txt',
                                  argument_path='../../../cognlp/data/ee/ace2005casee/data/argument_vocabulary.txt'
                                  )
train_datable = processor.process_train(train_data)
train_dataset = DataTableSet(train_datable, to_device=False)

dev_datable = processor.process_dev(dev_data)
dev_dataset = DataTableSet(dev_datable, to_device=False)
#
# test_datable = processor.process(test_data)
# test_dataset = DataTableSet(test_datable, to_device=False)
#
model =CasEE(config,
             type_num=len(processor.get_trigger_vocabulary()),
             args_num=len(processor.get_argument_vocabulary()),
             bert_model='bert-base-cased', pos_emb_size=64,
             device=device,
             schema_id=processor.schema_id)
loss = None
optimizer = optim.Adam(model.parameters(), lr=0.00005)
metric = CASEEMetric()

trainer = Trainer(model,
                  train_dataset,
                  dev_data=dev_dataset,
                  n_epochs=100,
                  batch_size=20,
                  dev_batch_size=1,
                  loss=loss,
                  optimizer=optimizer,
                  scheduler=None,
                  metrics=metric,
                  drop_last=False,
                  gradient_accumulation_steps=1,
                  num_workers=None,
                  save_path='../../../cognlp/data/ee/ace2005casee/model',
                  save_file=None,
                  print_every=None,
                  scheduler_steps=None,
                  validate_steps=1,
                  save_steps=1000,
                  grad_norm=1.0,
                  use_tqdm=True,
                  device=device,
                  device_ids=[4],
                  collate_fn=train_dataset.to_dict,
                  dev_collate_fn=dev_dataset.to_dict,
                  callbacks=None,
                  metric_key=None,
                  writer_path='../../../cognlp/data/ee/ace2005casee/tensorboard',
                  fp16=False,
                  fp16_opt_level='O1',
                  checkpoint_path=None,
                  task='ace2005-event-test',
                  logger_path='../../../cognlp/data/ee/ace2005casee/logger')

trainer.train()