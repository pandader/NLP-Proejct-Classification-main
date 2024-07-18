# Copyright 2024 @ Lun Li
#
# Summary: Training utilities. In particular, ResultsMgr is a handy object to gather results during the training.

import os
import numpy as np
from enum import Enum
from typing import Any, Optional, Union
# torch
import torch
from torch import nn
from torch.types import Number
# Peft
from peft import PeftModel

### types of data loaders (register more as needed)
class DataLoaderType(Enum):
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3
    TRAINING_UNLABELED = 4

### types of loss functions (register more as needed)
class LossFuncType(Enum):
    CROSS_ENTROPY = 1
    KL_DIV = 2

### loss function getter
def get_loss_functions(loss_func_type : LossFuncType, reduce : Optional[str]='mean'):
    if loss_func_type == LossFuncType.CROSS_ENTROPY:
        return nn.CrossEntropyLoss(reduction=reduce)
    elif loss_func_type == LossFuncType.KL_DIV:
        return nn.KLDivLoss(reduction=reduce)
    else:
        raise Exception("Unsupported loss function type")

# makes dataLoader iterable
def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x

def send_to_device(data : Any, device : Any):
    return [d.to(device) for d in data]

### model saving
def save_model(epoch : int, model_name : str, model : nn.Module, path : Optional[str]=""):
  if isinstance(model, PeftModel):
      model.save_pretrained(os.path.join(path, f'{model_name}_epcoh_{epoch}'))
  else:
    export_path = os.path.join(path, f'{model_name}_epoch_{epoch}.pt')
    torch.save(model.state_dict(), export_path)

### model loading
def load_model(epoch : int, model_name : str, model, path : Optional[str]="", is_peft : Optional[bool]=False):
  if is_peft:
      import_path = os.path.join(path, f'{model_name}_epcoh_{epoch}')
      PeftModel.from_pretrained(model, import_path, is_trainable=True)
  else:
    import_path = os.path.join(path, f'{model_name}_epoch_{epoch}.pt')
    model.load_state_dict(torch.load(import_path))

### results manager
class ResultsMgr:

    def __init__(self, num_epochs : Optional[int]=1, report_freq : Optional[int]=-1):
        """
        Args:
            report_freq (int, optional): Within an epoch, every report_freq, report aggregated results. Defaults to -1 (disabled).
        """
        self.epoch_idx = 0
        self.num_epochs = num_epochs
        self.report_freq = report_freq
        self.agg_res = dict()
    
    def set_num_epochs(self, num_epochs : int):
        self.num_epochs = num_epochs

    def start_this_epoch(self):
        self.this_epoch_step = 0
        # this epoch container
        self.preds = []
        self.labels = []
        self.sup_loss = []
        self.unsup_loss = []

    def step(self, model_output : torch.Tensor, labels : torch.Tensor, loss : Union[None, torch.Tensor], **kwargs):
        # gather results for this batch
        # list of tensors
        self.preds.append(torch.argmax(model_output.detach(), dim=1).to("cpu"))
        
        self.labels.append(labels.to("cpu"))
        # list of numbers
        if loss is not None:
            self.sup_loss.append(loss.item())
        # increment
        self.this_epoch_step += 1
    
    def report(self, cur_step):
        if self.report_freq != -1 and (cur_step + 1) % self.report_freq == 0:
            # report sup-loss
            print(f'At step {cur_step + 1}, the training loss is {np.mean(self.sup_loss)}.')

    def end_this_epoch(self, verbose : Optional[bool]=True):
        preds_ = torch.concat(self.preds)
        labels_ = torch.cat(self.labels)
        accuracy = (preds_ == labels_).sum() / len(labels_)
        loss = np.mean(self.sup_loss) if len(self.sup_loss) > 0 else np.nan
        self.epoch_idx += 1
        self.agg_res[self.epoch_idx] = \
        {
            "accuracy" : accuracy,
            "loss" : loss
        }
        if verbose:
            print(f'For epoch {self.epoch_idx}, the mean sup loss is: {loss}, and accuracy is: {accuracy}.')
    
    ### some getters

    def get_agg_res(self, epoch : Optional[int]=-1):
        assert(epoch <= self.num_epochs)
        if epoch == -1:
            return self.agg_res
        else:
            return self.agg_res[epoch]
    
    def get_epoch_idx(self):
        return self.epoch_idx
    
    def get_num_epochs(self):
        return self.num_epochs