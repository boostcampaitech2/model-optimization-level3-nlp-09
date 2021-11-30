"""PyTorch trainer module.

- Author: Jongkuk Lim, Junghoon Kim
- Contact: lim.jeikei@gmail.com, placidus36@gmail.com
"""

import os
import shutil
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.cuda import device_count
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from tqdm import tqdm

from src.utils.torch_utils import save_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from ptflops import get_model_complexity_info
import copy
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
from typing import List
tl.set_backend('pytorch')
import wandb

def _get_n_data_from_dataloader(dataloader: DataLoader) -> int:
    """Get a number of data in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A number of data in dataloader
    """
    if isinstance(dataloader.sampler, SubsetRandomSampler):
        n_data = len(dataloader.sampler.indices)
    elif isinstance(dataloader.sampler, SequentialSampler):
        n_data = len(dataloader.sampler.data_source)
    else:
        n_data = len(dataloader) * dataloader.batch_size if dataloader.batch_size else 1

    return n_data


def _get_n_batch_from_dataloader(dataloader: DataLoader) -> int:
    """Get a batch number in dataloader.

    Args:
        dataloader: torch dataloader

    Returns:
        A batch number in dataloader
    """
    n_data = _get_n_data_from_dataloader(dataloader)
    n_batch = dataloader.batch_size if dataloader.batch_size else 1

    return n_data // n_batch


def _get_len_label_from_dataset(dataset: Dataset) -> int:
    """Get length of label from dataset.

    Args:
        dataset: torch dataset

    Returns:
        A number of label in set.
    """
    if isinstance(dataset, torchvision.datasets.ImageFolder) or isinstance(
        dataset, torchvision.datasets.vision.VisionDataset
    ):
        return len(dataset.classes)
    elif isinstance(dataset, torch.utils.data.Subset):
        return _get_len_label_from_dataset(dataset.dataset)
    else:
        raise NotImplementedError


class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        model_path: str,
        scaler=None,
        device: torch.device = "cpu",
        verbose: int = 1,
    ) -> None:
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
        """
        self.device = device

        self.model = model
        params_nums = count_model_params(self.model)
        print(params_nums)
        macs, params = self.get_macs(self.model)
        print(macs)

        # self.model = models.vgg11_bn(pretrained=True)
        # print(self.model.features)
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.device = device
        # print(f'decompose start!')

        for name, param in self.model.named_modules():
            if isinstance(param, nn.Conv2d):
                param.register_buffer('rank', torch.Tensor([0.5, 0.5])) # rank in, out 
        self.model= self.decompose(self.model)
        # print(self.model.features)
        # print(f'decompose end!')
        macs, params = self.get_macs(self.model)
        print(macs)
        print(f'decomposed_model: {macs}')

        params_nums = count_model_params(self.model)
        print(params_nums)
        assert False
        self.model.to(self.device)

    def get_macs(self, model):
        model.to(self.device)
        # calculate macs
        macs, params = get_model_complexity_info(model, tuple([3, 32, 32]), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        return macs, params

    def decompose(self, module: nn.Module):
        """Iterate model layers and decompose"""
        model_layers = list(module.children())
        if not model_layers:
            return None
        for i in range(len(model_layers)):
            if type(model_layers[i]) == nn.Sequential:
                decomposed_module = self.decompose(model_layers[i])
                if decomposed_module:
                    model_layers[i] = decomposed_module
            if type(model_layers[i]) == nn.Conv2d:
                model_layers[i] = self.tucker_decomposition_conv_layer(model_layers[i])
        return nn.Sequential(*model_layers)

    def tucker_decomposition_conv_layer(self,
      layer: nn.Module,
      normed_rank: List[int] = [0.5, 0.5],
  ) -> nn.Module:
      """Gets a conv layer,
      returns a nn.Sequential object with the Tucker decomposition.
      The ranks are estimated with a Python implementation of VBMF
      https://github.com/CasvandenBogaard/VBMF
      """
      if hasattr(layer, "rank"):
          normed_rank = getattr(layer, "rank")
      rank = [int(r * layer.weight.shape[i]) for i, r in enumerate(normed_rank)] # output channel * normalized rank
      rank = [max(r, 2) for r in rank]

      core, [last, first] = partial_tucker(
          layer.weight.data,
          modes=[0, 1],
          n_iter_max=2000,
          rank=rank,
          init="svd",
      )

      # A pointwise convolution that reduces the channels from S to R3
      first_layer = nn.Conv2d(
          in_channels=first.shape[0],
          out_channels=first.shape[1],
          kernel_size=1,
          stride=1,
          padding=0,
          dilation=layer.dilation,
          bias=False,
      )

      # A ular 2D convolution layer with R3 input channels
      # and R3 output channels
      core_layer = nn.Conv2d(
          in_channels=core.shape[1],
          out_channels=core.shape[0],
          kernel_size=layer.kernel_size,
          stride=layer.stride,
          padding=layer.padding,
          dilation=layer.dilation,
          bias=False,
      )

      # A pointwise convolution that increases the channels from R4 to T
      last_layer = nn.Conv2d(
          in_channels=last.shape[1],
          out_channels=last.shape[0],
          kernel_size=1,
          stride=1,
          padding=0,
          dilation=layer.dilation,
          bias=True,
      )

      if hasattr(layer, "bias") and layer.bias is not None:
          last_layer.bias.data = layer.bias.data

      first_layer.weight.data = (
          torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
      )
      last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
      core_layer.weight.data = core

      new_layers = [first_layer, core_layer, last_layer]
      return nn.Sequential(*new_layers)
    
    def train(
        self,
        train_dataloader: DataLoader,
        n_epoch: int,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, float]:
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """
        best_test_acc = -1.0
        best_test_f1 = -1.0
        num_classes = _get_len_label_from_dataset(train_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        run= wandb.init(project= 'lightweight', entity= 'quarter100', name= f'eff_b0_decomp_0.001')

        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, (data, labels) in pbar:
                data, labels = data.to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                    f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
                )
            pbar.close()

            test_loss, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )

            wandb.log({
                'train_loss': (running_loss / (batch + 1)),
                'train_acc': (correct / total) * 100,
                'train_f1': f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0),
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_f1': test_f1})
            if best_test_f1 > test_f1:
                continue
            best_test_acc = test_acc
            best_test_f1 = test_f1
            print(f"Model saved. Current best test f1: {best_test_f1:.3f}")
            save_model(
                model=self.model,
                path=self.model_path,
                data=data,
                device=self.device,
            )

            # macs, params = self.get_macs(self.model)
            # print(f'decomposed_model: {macs}')

        return best_test_acc, best_test_f1

    @torch.no_grad()
    def test(
        self, model: nn.Module, test_dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        n_batch = _get_n_batch_from_dataloader(test_dataloader)

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = _get_len_label_from_dataset(test_dataloader.dataset)
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, (data, labels) in pbar:
            data, labels = data.to(self.device), labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            running_loss += self.criterion(outputs, labels).item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy


def count_model_params(
    model: torch.nn.Module,
) -> int:
    """Count model's parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
