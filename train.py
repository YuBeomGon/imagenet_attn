# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py.
Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.
Train on ImageNet with default parameters:
.. code-block: bash
    python imagenet.py fit --model.data_path /path/to/imagenet
or show all options you can change:
.. code-block: bash
    python imagenet.py --help
    python imagenet.py fit --help
"""
import argparse
import os
from typing import Optional
import pandas as pd
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchmetrics import Accuracy, F1Score, Specificity

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import Trainer
# from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

import custom_models


parser = argparse.ArgumentParser(description='PyTorch Lightning ImageNet Training')
parser.add_argument('--data_path', metavar='DIR', default='ILSVRC/Data/CLS-LOC/',
                    help='path to dataset (default: ILSVRC/Data/CLS-LOC/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--accelerator', '--accelerator', default='gpu', type=str, help='default: gpu')

parser.add_argument('--devices', '--devices', default=4, type=int, help='number of gpus, default 2')
parser.add_argument('--img_size', default=400, type=int, help='input image resolution in swin models')
parser.add_argument('--num_classes', default=1000, type=int, help='number of classes')
parser.add_argument('--saved_dir', default='./saved_models/tunning', type=str, help='directory for model checkpoint')


class ImageNetAttnModel(LightningModule):
    """
    >>> ImageNetLightningModel(data_path='missing')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ImageNetLightningModel(
      (model): ResNet(...)
    )
    """

    def __init__(
        self,
        data_path: str,
        arch: str = "resnet18",
        pretrained: bool = False,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        workers: int = 4,
        num_classes : int = 1000,
    ):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.num_classes = num_classes
        # self.model = models.__dict__[self.arch](pretrained=self.pretrained)
        self.model = custom_models.__dict__[self.arch](pretrained=False)
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.train_acc1 = Accuracy(top_k=1)
        self.train_acc5 = Accuracy(top_k=5)
        self.eval_acc1 = Accuracy(top_k=1)
        self.eval_acc5 = Accuracy(top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss_train = F.cross_entropy(output, target)
        self.log("train_loss", loss_train)
        # update metrics
        self.train_acc1(output, target)
        self.train_acc5(output, target)
        self.log("train_acc1", self.train_acc1, prog_bar=True)
        self.log("train_acc5", self.train_acc5, prog_bar=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self.model(images)
        loss_val = F.cross_entropy(output, target)
        self.log(f"{prefix}_loss", loss_val)
        # update metrics
        self.eval_acc1(output, target)
        self.eval_acc5(output, target)
        self.log(f"{prefix}_acc1", self.eval_acc1, prog_bar=True)
        self.log(f"{prefix}_acc5", self.eval_acc5, prog_bar=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def setup(self, stage: Optional[str] = None):
        if isinstance(self.trainer.strategy, ParallelStrategy):
            # When using a single GPU per process and per `DistributedDataParallel`, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)

        if stage in (None, "fit"):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_dir = os.path.join(self.data_path, "train")
            self.train_dataset = datasets.ImageFolder(
                train_dir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
        # all stages will use the eval dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        val_dir = os.path.join(self.data_path, "val")
        self.eval_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True
        )

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    args = parser.parse_args()
    if torch.cuda.is_available() :
        args.accelerator = 'gpu'
        args.devices = torch.cuda.device_count()
        
    # tb_logger = TensorBoardLogger(save_dir="tuning_logs/" + args.arch, name=args.arch + "_my_model")
    logger_tb = TensorBoardLogger('./tuning_logs' +'/' + args.arch, name=now)
    logger_wandb = WandbLogger(project='Paps_clf', name=now, mode='online') # online or disabled    
    
    trainer_defaults = dict(
        callbacks = [
            # the PyTorch example refreshes every 10 batches
            TQDMProgressBar(refresh_rate=50),
            # save when the validation top1 accuracy improves
            ModelCheckpoint(monitor="val_acc1", mode="max",
                            dirpath=args.saved_dir + '/' + args.arch,
                            filename='paps_tunning_{epoch}_{val_acc1:.2f}'),  
            ModelCheckpoint(monitor="val_acc1", mode="max",
                            dirpath=args.saved_dir + '/' + args.arch,
                            filename='paps_tunning_best'),             
        ],    
        # plugins = "deepspeed_stage_2_offload",
        precision = 16,
        max_epochs = args.epochs,
        accelerator = args.accelerator, # auto, or select device, "gpu"
        devices = args.devices, # number of gpus
        logger = [logger_tb, logger_wandb],
        benchmark = True,
        strategy = "ddp",
        )
    
    model = ImageNetAttnModel(
        data_path=args.data_path,
        arch=args.arch,
        pretrained=False,
        workers=args.workers,
        lr = args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
    )
    
    trainer = Trainer(**trainer_defaults)
    trainer.fit(model)  
    
    trainer.test(model)