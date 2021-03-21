"""
@Author: jinzhuan
@File: tester.py
@Desc: 
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from cogie.utils import load_model, module2parallel

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Tester:
    def __init__(self, model, model_path, batch_size=32, sampler=None,
                 drop_last=False, num_workers=0, print_every=1000,
                 dev_data=None, metrics=None, metric_key=None, use_tqdm=True, device=None,
                 callbacks=None, check_code_level=0, device_ids=None):
        self.dev_data = dev_data
        self.model = model
        self.model_path = model_path
        self.metrics = metrics
        self.batch_size = batch_size
        self.sampler = sampler
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.print_every = print_every
        self.metrics = metrics
        self.use_tqdm = use_tqdm
        self.device = device
        self.device_ids = device_ids

        self.model = module2parallel(self.model, self.device_ids)
        self.model = load_model(self.model, self.model_path)
        self.dev_dataloader = DataLoader(dataset=self.dev_data, batch_size=self.batch_size, sampler=self.sampler,
                                           drop_last=self.drop_last)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            if self.use_tqdm:
                progress = enumerate(tqdm(self.dev_dataloader, desc="Evaluating"), 1)
            else:
                progress = enumerate(self.dev_dataloader, 1)
            for step, batch in progress:
                self.model.evaluate(batch, self.metrics)
            evaluate_result = self.metrics.get_metric(reset=True)
            logger.info("Evaluate result = %s", str(evaluate_result))
