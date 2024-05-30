# Copyright 2024 @ Lun Li
#
# Summary: A semi-supervised framework by following "MixMatch: A Holistic Approach to Semi-Supervised Learning"
#          https://arxiv.org/pdf/1905.02249

import copy
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from typing import Optional, Any, Callable, Tuple
# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
# local
from .standard_trainer import Trainer, Evaluator, ResultsMgr
from .trainer_utilities import (
    send_to_device, save_model, repeat_dataloader, get_loss_functions, DataLoaderType, LossFuncType)
from ..optimizer import SchedulerType
from ..models import MultiLabelClassifier

# Two important utilities to ensure good batch normalization
# i.e., the proportionality of labeled vs unlabeled in every sub-group remains the same as in the sup-group
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

# Exponential moving average of model parameters
class WeightEMA(object):
    def __init__(self, 
                 models : list[nn.Module],
                 ema_models : list[nn.Module],
                 alpha : float=0.999):
        self.models = models
        self.ema_models = ema_models
        self.alpha = alpha
        self.params = {i : list(m.state_dict().values()) for i, m in enumerate(self.models)}
        self.ema_params = {i : list(m.state_dict().values()) for i, m in enumerate(self.ema_models)}
        self.wd = 0.02 * self.alpha
        for i in range(len(self.params)):
            for param, ema_param in zip(self.params[i], self.ema_params[i]):
                param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for i in range(len(self.params)):
            for param, ema_param in zip(self.params[i], self.ema_params[i]):
                if ema_param.dtype==torch.float32:
                    ema_param.mul_(self.alpha)
                    ema_param.add_(param * one_minus_alpha)
                    # customized weight decay
                    param.mul_(1 - self.wd)


class EvaluatorMixAndMatch(Evaluator):

    def __init__(self, loss_func: callable, data_loader: DataLoader):
        super().__init__(loss_func, data_loader)

    def run(self, cur_model : nn.Module, aux_models : list[nn.Module]):
        resMgr = ResultsMgr(num_epochs=1)
        cur_model.eval()
        aux_models[0].eval()
        # Predict
        with torch.no_grad():
            resMgr.start_this_epoch()
            for _, batch in enumerate(self.data_loader):
                b_input_ids, b_input_mask, b_labels = send_to_device(batch, self.device)
                result = cur_model(b_input_ids, b_input_mask)
                logits = aux_models[0](result.hidden_states[0])
                sup_loss = self.loss_func(logits, b_labels)
                sup_loss = torch.mean(sup_loss)
                resMgr.step(logits, b_labels, sup_loss)
            resMgr.end_this_epoch(verbose=False)
        val_acc = resMgr.get_agg_res(1)['accuracy']
        print(f'Validation accuracy is: {val_acc.item()}.\n')


# Main trainer
class TrainerMixAndMatch(Trainer):

    def __init__(self, 
                 model: nn.Module, 
                 aux_model : MultiLabelClassifier,
                 data_loader: dict, 
                 optimizer: Optimizer, 
                 aux_model_optimizer: Optimizer,
                 report_freq: Optional[int]= 100,
                 temperature : Optional[float]=0.5,
                 alpha : Optional[float]=0.75,
                 lambda_u : Optional[float]=75,
                 ema_decay : Optional[float]=0.999):

        super().__init__(
            model, data_loader, get_loss_functions(LossFuncType.CROSS_ENTROPY), optimizer, report_freq
        )
        ### instaniate mixAndMatch specific parameters
        self.alpha = alpha
        self.temperature = temperature
        self.aux_models = [aux_model]
        self.aux_optimizer = aux_model_optimizer
        self.num_classes = self.aux_models[0].num_labels
        self.loss_func_semi = get_loss_functions(loss_func_type=LossFuncType.PROB_MEAN_SQ, lambda_u = lambda_u)
        # since labeled data is << unlabeled data
        # the batches will be generated from these unlabeled ones
        # in the meanwhile, we repeated generate data fraom labeled ones
        self.train_sup_dataloader_iter = repeat_dataloader(self.data_loaders[DataLoaderType.TRAINING])
        self.generate_batch_based_on = DataLoaderType.TRAINING_UNLABELED
        self.num_batches_per_epoch = len(data_loader[self.generate_batch_based_on])
        # set up ema scheulder (abuse of name)
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters(): param.detach_()
        self.ema_aux_model = copy.deepcopy(self.aux_models[0])
        for param in self.ema_aux_model.parameters(): param.detach_()
        self.scheduler = WeightEMA(
            [self.model, self.aux_models[0]],
            [self.ema_model, self.ema_aux_model],
            alpha=ema_decay)
        # set up evaluator
        self.eval = EvaluatorMixAndMatch(self.loss_func, self.data_loaders[DataLoaderType.VALIDATION])
        
    # uda loss computation logic
    def calcualte_loss(self, 
                       batch_idx : int,
                       batch : Any): # batch => training unlabeled

        ### preparations
        # labeled
        b_input_ids, b_input_mask, b_labels = send_to_device(next(self.train_sup_dataloader_iter), self.device)
        # unlabeled
        b_ori_input_ids, b_ori_input_mask, b_aug_input_ids, b_aug_input_mask = send_to_device(batch, self.device)
        # batch size
        batch_size = b_input_ids.size(0)
        # transform label to one hot
        b_labels_one_hot = torch.zeros(
            batch_size, self.num_classes).scatter_(1, b_labels.view(-1, 1).long(), 1).to(self.device)

        ### forward pass but no grad
        with torch.no_grad():
            # compute guessed labels of unlabeled samples
            tmp_a = self.model(b_ori_input_ids, b_ori_input_mask)
            outputs_a = self.aux_models[0](tmp_a.hidden_states[0])
            tmp_b = self.model(b_aug_input_ids, b_aug_input_mask)
            outputs_b = self.aux_models[0](tmp_b.hidden_states[0])
            # compute average predictions across all aug of ub
            p = (torch.softmax(outputs_a, dim=1) + torch.softmax(outputs_b, dim=1)) / 2 
            # Apply temperature sharpening to the average prediction
            pt = p**(1. / self.temperature)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
            # tracking results [labeled]
            tmp_r = self.model(b_input_ids, b_input_mask)
            output_r = self.aux_models[0](tmp_r.hidden_states[0])

        ### mixup
        all_input_ids = torch.cat([b_input_ids, b_ori_input_ids, b_aug_input_ids], dim=0) # combine inputs_ids
        all_input_masks = torch.cat([b_input_mask, b_ori_input_mask, b_aug_input_mask], dim=0) # combine inputs_masks
        all_targets = torch.cat([b_labels_one_hot, targets_u, targets_u], dim=0) # combine labels
        # for NLP task, we need to get a shared representation to be able mix
        shared_rep = self.model(all_input_ids, all_input_masks)
        shared_rep = shared_rep.hidden_states[0]
        # gamma dist mix up
        l = np.random.beta(self.alpha, self.alpha)
        l = max(l, 1-l)
        idx = torch.randperm(all_input_ids.size(0))
        input_a, input_b = shared_rep, shared_rep[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        # do batch separately
        logits = [self.aux_models[0](mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(self.aux_models[0](input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        # loss calculation
        cur_epoch = self.resMgr.get_cur_epoch()
        num_epochs = self.resMgr.get_num_epochs()
        l_x, l_u, w = self.loss_func_semi(
            logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], 
            cur_epoch + batch_idx / self.num_batches_per_epoch, num_epochs
        )
        loss = l_x + w * l_u

        # return total loss, sup result (.logits), sup labels
        return loss, output_r, b_labels
    
    def set_scheduler(self, 
                      num_epochs : int,
                      schedule_type : SchedulerType, 
                      num_warmup_steps : int,
                      override_schedule : Any):

        pass

    def preprocess(self):
        self.aux_optimizer.zero_grad()

    def postprocess(self):
        self.aux_optimizer.step()

