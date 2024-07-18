# Copyright 2024 @ Lun Li
#
# Summary: Training utilities. In particular, ResultsMgr is a handy object to gather results during the training.

import os
import numpy as np
from enum import Enum
from typing import Any, Optional, Union, Tuple
# torch
import torch
from torch import nn
import torch.nn.functional as F
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
    PROB_MEAN_SQ = 3

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class ProbMeanSq(object):
    '''
    Mean square loss of probability distribution:
    || dist_p - dist_q ||_2
    where dist_p/dist_q are both finite-dimensional vectors over number of classes
    '''
    def __init__(self, lambda_u):        
        self.lambda_u = lambda_u

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, num_epochs):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, self.lambda_u * linear_rampup(epoch, num_epochs)

### loss function getter
def get_loss_functions(loss_func_type : LossFuncType, reduce : Optional[str]='mean', **kwargs):
    if loss_func_type == LossFuncType.CROSS_ENTROPY:
        return nn.CrossEntropyLoss(reduction=reduce)
    elif loss_func_type == LossFuncType.KL_DIV:
        return nn.KLDivLoss(reduction=reduce)
    elif loss_func_type == LossFuncType.PROB_MEAN_SQ:
        return ProbMeanSq(kwargs['lambda_u'])
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
def save_model(
        epoch : int, 
        model_name : str, 
        model : nn.Module, 
        path : Optional[str]="", 
        aux_models : Optional[list[nn.Module]]=[]):

    if isinstance(model, PeftModel):
        model.save_pretrained(os.path.join(path, f'{model_name}_epoch_{epoch}'))
    else:
        export_path = os.path.join(path, f'{model_name}_epoch_{epoch}.pt')
        torch.save(model.state_dict(), export_path)
    
    if len(aux_models) != 0:
        for idx, m in enumerate(aux_models):
            if isinstance(m, PeftModel):
                m.save_pretrained(os.path.join(path, f'{model_name}_aux_{idx+1}_epoch_{epoch}'))
            else:
                export_path = os.path.join(path, f'{model_name}_aux_{idx+1}_epoch_{epoch}.pt')
                torch.save(m.state_dict(), export_path)

### model loading
def load_model(
        epoch : int, 
        model_name : str, 
        model, path : Optional[str]="", 
        is_peft : Optional[bool]=False, 
        aux_id : Optional[int]=-1):

    if is_peft:
        if aux_id == -1:
            import_path = os.path.join(path, f'{model_name}_epoch_{epoch}')
        else:
            import_path = os.path.join(path, f'{model_name}_aux_{aux_id}_epoch_{epoch}')
        PeftModel.from_pretrained(model, import_path, is_trainable=True)
    else:
        if aux_id == -1:
            import_path = os.path.join(path, f'{model_name}_epoch_{epoch}.pt')
        else:
            import_path = os.path.join(path, f'{model_name}_aux_{aux_id}_epoch_{epoch}.pt')
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

    def get_cur_epoch(self):
        return self.epoch_idx
    
    def start_this_epoch(self):
        self.this_epoch_step = 0
        # this epoch container
        self.preds = []
        self.labels = []
        self.sup_loss = []
        self.unsup_loss = []

    def step(self, model_output : torch.Tensor, labels : torch.Tensor, loss : Union[Tuple, list], **kwargs):
        # gather results for this batch
        # list of tensors
        self.preds.append(torch.argmax(model_output.detach(), dim=1).to("cpu"))
        self.labels.append(labels.to("cpu"))
        # list of numbers
        if loss is not None:
            self.sup_loss.append(loss[0].item())
            if loss[1] is not None:
                self.unsup_loss.append(loss[1].item())
        # increment
        self.this_epoch_step += 1
    
    def report(self, cur_step):
        if self.report_freq != -1 and (cur_step + 1) % self.report_freq == 0:
            # report sup-loss
            sup_loss_str = f'At step {cur_step + 1}, the training (sup)loss is {np.mean(self.sup_loss)}'
            # report unsup-loss (if applicable)
            unsup_loss_str = '' if len(self.unsup_loss) == 0 else f', the training unsup-loss is {np.mean(self.unsup_loss)}'
            print(sup_loss_str + unsup_loss_str + '.')

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