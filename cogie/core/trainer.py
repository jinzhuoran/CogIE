"""
@Author: jinzhuan
@File: trainer.py
@Desc: 
"""
import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import time
import os
from cogie.core import Tester
import logging
from cogie.utils.util import save_model, load_model, module2parallel
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
            self,
            model,
            train_data,
            dev_data=None,
            n_epochs=10,
            batch_size=32,
            loss=None,
            optimizer=None,
            scheduler=None,
            metrics=None,
            train_sampler=None,
            dev_sampler=None,
            drop_last=False,
            gradient_accumulation_steps=1,
            num_workers=0,
            collate_fn=None,
            save_path=None,
            save_file=None,
            print_every=None,
            scheduler_steps=None,
            validate_steps=None,
            save_steps=None,
            grad_norm=None,
            use_tqdm=True,
            device=None,
            device_ids=None,
            callbacks=None,
            check_code_level=logging.INFO,
            metric_key=None,
            writer_path=None,
            fp16=False,
            fp16_opt_level='O1',
            seed=527,
            checkpoint_path=None,
            task='train',
            logger_path=None,
    ):
        """
        训练器构造函数
        :param model:模型
        :param train_data:训练数据
        :param dev_data:验证数据
        :param n_epochs:迭代轮数
        :param batch_size:数据大小
        :param loss:损失函数
        :param optimizer:优化器
        :param scheduler:调整学习率
        :param metrics:评价指标
        :param train_sampler:训练集采样器
        :param dev_sampler:验证集采样器
        :param drop_last:丢掉最后一个
        :param gradient_accumulation_steps:梯度累积步数
        :param num_workers:多线程加载数据
        :param print_every:打印步数
        :param save_path:保存训练参数的路径
        :param save_file:保存模型的文件名
        :param validate_steps:验证步数
        :param save_steps:保存步数
        :param grad_norm:梯度裁剪
        :param use_tqdm:是否使用tqdm
        :param device:设备
        :param device_ids:多卡 [0, 1, 2, 3]
        :param callbacks:回调函数
        :param check_code_level:
        :param metric_key:
        """
        # if seed:
        #     seed_everything(seed)

        self.train_data = train_data
        self.dev_data = dev_data
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.n_epochs = n_epochs
        self.device = device
        self.grad_norm = grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_path = save_path
        self.save_file = save_file
        self.save_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
        self.task = task
        self.logger_path = logger_path

        self.metrics = metrics
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.use_tqdm = use_tqdm
        self.callbacks = callbacks
        self.train_sampler = train_sampler
        self.dev_sampler = dev_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.device_ids = device_ids
        self.writer_path = writer_path
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.checkpoint_path = checkpoint_path
        self.check_code_level = check_code_level
        self.metric_key = metric_key

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model.to(self.device)
            self.model, self.optimizer = amp.initialize(model, optimizer, opt_level=self.fp16_opt_level)

        self.train_dataloader = DataLoader(dataset=self.train_data, batch_size=self.batch_size,
                                           sampler=self.train_sampler,drop_last=self.drop_last,
                                           collate_fn=self.collate_fn)

        self.batch_count = len(self.train_dataloader)
        if save_steps:
            self.save_steps = save_steps
        else:
            self.save_steps = self.batch_count

        if print_every:
            self.print_every = print_every
        else:
            self.print_every = self.batch_count

        if scheduler_steps:
            self.scheduler_steps = scheduler_steps
        else:
            self.scheduler_steps = self.batch_count

        if validate_steps:
            self.validate_steps = validate_steps
        else:
            self.validate_steps = self.batch_count

        if self.dev_data:
            self.dev_dataloader = DataLoader(dataset=self.dev_data, batch_size=self.batch_size,
                                             sampler=self.dev_sampler, drop_last=self.drop_last,
                                             collate_fn=self.collate_fn)
        self.model = module2parallel(self.model, self.device_ids)

        if self.writer_path:
            self.writer_path = os.path.join(self.writer_path, self.task)
            self.writer_path = os.path.join(self.writer_path, self.save_time)
            if not os.path.exists(self.writer_path):
                os.makedirs(self.writer_path)
            self.writer = SummaryWriter(self.writer_path)
        else:
            self.writer = SummaryWriter()

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger()
        self.logger.setLevel(self.check_code_level)
        if self.logger_path:
            self.logger_file = os.path.join(self.logger_path, self.task)
            self.logger_file = os.path.join(self.logger_file, self.save_time)
            if not os.path.exists(self.logger_file):
                os.makedirs(self.logger_file)
            self.logger_file = os.path.join(self.logger_file, 'logger.txt')
            file_handler = logging.handlers.TimedRotatingFileHandler(filename=self.logger_file, when='D',
                                                                     backupCount=100,
                                                                     encoding='utf-8')
            self.logger.addHandler(file_handler)

    def train(self):
        global_step = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        # Check if continuing training from a checkpoint
        if self.checkpoint_path and os.path.exists(self.checkpoint_path) and "checkpoint" in self.checkpoint_path:
            # set global_step to gobal_step of last saved checkpoint from models path
            global_step = int(self.checkpoint_path.split("-")[-1].split("/")[0])
            epochs_trained = epochs_trained + global_step // (
                    len(self.train_dataloader) // self.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(self.train_dataloader) // self.gradient_accumulation_steps)

            self.logger.info("Continuing training from checkpoint, will skip to saved global_step")
            self.logger.info("Continuing training from epoch %d", epochs_trained)
            self.logger.info("Continuing training from global step %d", global_step)
            self.logger.info("Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

            if os.path.isfile(os.path.join(self.checkpoint_path, "models.pt")):
                self.model = load_model(self.model, os.path.join(self.checkpoint_path, "models.pt"))
            if os.path.isfile(os.path.join(self.checkpoint_path, "optimizer.pt")):
                self.optimizer.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "optimizer.pt")))
            if os.path.isfile(os.path.join(self.checkpoint_path, "scheduler.pt")):
                self.scheduler.load_state_dict(torch.load(os.path.join(self.checkpoint_path, "scheduler.pt")))

        self.logger.info("Start training")
        self.logger.info("Epoch size = %d", self.n_epochs)
        self.logger.info("Batch size = %d", self.batch_size)
        self.logger.info("Global step = %d", global_step)

        total_loss = 0.0
        self.model.zero_grad()
        for epoch in range(epochs_trained, self.n_epochs + 1):

            epoch_loss = 0.0
            self.logger.info("Train epoch = %d", epoch)
            self.logger.info("Start time = %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
            self.model.train()
            if self.use_tqdm:
                progress = enumerate(tqdm(self.train_dataloader, desc="Iteration", leave=False), 1)
            else:
                progress = enumerate(self.train_dataloader, 1)

            for step, batch in progress:

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                loss = self.model.loss(batch, self.loss)
                epoch_loss += loss.item()
                total_loss += loss.item()
                self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=global_step)

                # 梯度反传
                if self.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError(
                            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # 参数更新
                if isinstance(self.gradient_accumulation_steps,
                              int) and global_step % self.gradient_accumulation_steps == 0:

                    # 梯度裁剪
                    if self.grad_norm:
                        if self.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.grad_norm)
                        else:
                            utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                # 学习率更新
                if self.scheduler and isinstance(self.scheduler_steps, int) and global_step % self.scheduler_steps == 0:
                    self.scheduler.step()
                    # If there is one global learning rate (which is the common case).
                    lr = next(iter(self.optimizer.param_groups))['lr']
                    self.logger.debug('Global step: {}, Learning rate: {}'.format(global_step, lr))
                    self.writer.add_scalar(tag='Learning rate', scalar_value=lr, global_step=global_step)

                # 打印训练信息
                if isinstance(self.print_every, int) and global_step % self.print_every == 0:
                    self.logger.debug('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'.
                                      format(epoch, self.n_epochs, step, self.batch_count, loss.item()))

                # 保存模型
                if self.save_path and isinstance(self.save_steps, int) and global_step % self.save_steps == 0:
                    save_dir = os.path.join(self.save_path, self.task)
                    save_dir = os.path.join(save_dir, self.save_time)
                    self.logger.info("Saving models step = %d", global_step)
                    output_dir = os.path.join(save_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    self.logger.debug("Saving models checkpoint to %s", output_dir)
                    save_model(model=self.model, model_path=os.path.join(output_dir, "models.pt"))
                    # logger.info("Saving trainer arguments to %s", output_dir)
                    # save_json(vars(self), os.path.join(output_dir, "trainer.json"))
                    if self.optimizer:
                        self.logger.debug("Saving optimizer states to %s", output_dir)
                        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    if self.scheduler:
                        self.logger.debug("Saving scheduler states to %s", output_dir)
                        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.dev_data and isinstance(self.validate_steps, int) and global_step % self.validate_steps == 0:
                    self.logger.info("Evaluate step = %d", global_step)
                    self.model.eval()
                    if self.use_tqdm:
                        progress = enumerate(tqdm(self.dev_dataloader, desc="Evaluating", leave=False), 1)
                    else:
                        progress = enumerate(self.dev_dataloader, 1)
                    with torch.no_grad():
                        self.model.evaluate(progress)
                        # for step, batch in progress:
                        #     self.model.evaluate(batch, self.metrics)
                    self.model.train()
                    # evaluate_result = self.metrics.get_metric()
                    # self.logger.info("Evaluate result = %s", str(evaluate_result))
                    # for key, value in evaluate_result.items():
                    #     self.writer.add_scalar(tag=key, scalar_value=value, global_step=global_step)

            self.logger.info("Epoch loss = %f", epoch_loss)
            self.logger.info("End time = %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        self.logger.info("End training")
