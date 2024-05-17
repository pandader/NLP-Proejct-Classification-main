# Copyright 2024 @ Lun Li
#
# Summary: A semi-supervised framework through consistency training
#          1) requires a handful labeled data (depends on the number of classes)
#          2) requires a considerable number of unlabeled data
#          the classifier is trained against both data, where labeled data provides the benchmark
#          and unlabeled data helps to generalize the model structure 
#          Reference: https://arxiv.org/pdf/1904.12848

from tqdm import trange
from typing import Optional, Any, Callable
# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
# local
from .standard_trainer import Trainer
from .trainer_utilities import (send_to_device, save_model, repeat_dataloader, DataLoaderType)
from ..optimizer import SchedulerType

class TrainerUDA(Trainer):

    def __init__(self, 
                 model: nn.Module, 
                 data_loader: dict, 
                 loss_func_dict: dict[Callable], # sup/unsup
                 optimizer: Optimizer, 
                 report_freq: Optional[int]= 100,
                 tsa_schedule : Optional[str]="linear",
                 uda_confidence_thresh : Optional[float]=0.50,
                 uda_softmax_temp : Optional[float]=0.85,
                 uda_coeff : Optional[float]=1.):
        
        assert('sup' in loss_func_dict and 'unsup' in loss_func_dict)
        super().__init__(model, data_loader, loss_func_dict['sup'], optimizer, report_freq)
        ### instaniate uda specific parameters
        assert (tsa_schedule in ["linear", "exp", "log"])
        self.uda_coeff = uda_coeff
        self.tsa_schedule = tsa_schedule
        self.uda_confidence_thresh = uda_confidence_thresh
        self.uda_softmax_temp = uda_softmax_temp
        self.loss_fun_unsup = loss_func_dict['unsup']
        # since labeled data is << unlabeled data
        # the batches will be generated from these unlabeled ones
        # in the meanwhile, we repeated generate data fraom labeled ones
        self.train_sup_dataloader_iter = repeat_dataloader(self.data_loaders[DataLoaderType.TRAINING])
        self.generate_batch_based_on = DataLoaderType.TRAINING_UNLABELED

    # uda loss computation logic
    def calcualte_loss(self, 
                       batch_idx : int,
                       batch : Any):

        # var / device assignment
        b_input_ids, b_input_mask, b_labels = send_to_device(next(self.train_sup_dataloader_iter), self.device)
        b_ori_input_ids, b_ori_input_mask, b_aug_input_ids, b_aug_input_mask = send_to_device(batch, self.device)            
        # stack up + aug unsup data
        # NOTICE: we intentinoally leave out orig unsup data,
        #         as we shall no update params due to training orig unsup data
        input_ids = torch.cat((b_input_ids, b_aug_input_ids), dim=0)
        input_mask = torch.cat((b_input_mask, b_aug_input_mask), dim=0)
        
        # forward pass
        # call bert + classifier => logits
        result = self.model(input_ids, input_mask)

        ### sup loss [back prop]
        # tsa
        sup_size = b_labels.shape[0]
        sup_loss = self.loss_func(result.logits[:sup_size], b_labels)  # shape : train_batch_size
        num_batches = len(self.data_loaders[DataLoaderType.TRAINING_UNLABELED])
        tsa_thresh = TrainerUDA.get_tsa_thresh(
            self.tsa_schedule, 
            self.resMgr.get_epoch_idx() * num_batches + batch_idx,
            self.resMgr.get_num_epochs() * num_batches, 
            start=1./result.logits.shape[-1], 
            end=1).to(self.device)
        larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
        loss_mask = torch.ones_like(b_labels, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
        sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / \
            torch.max(torch.sum(loss_mask, dim=-1), torch.tensor(1.).to(self.device))

        ### unsup loss
        #   ori [no back prop]
        with torch.no_grad():
            ori_logits = self.model(b_ori_input_ids, b_ori_input_mask)
            ori_prob = F.softmax(ori_logits.logits, dim=-1)  # KLdiv target
            # confidence-based masking
            if self.uda_confidence_thresh != -1:
                unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > self.uda_confidence_thresh
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
            else:
                unsup_loss_mask = torch.ones(len(result.logits) - sup_size, dtype=torch.float32)
            unsup_loss_mask = unsup_loss_mask.to(self.device)
        # aug [use grad]
        # softmax temperature controlling
        uda_softmax_temp = self.uda_softmax_temp if self.uda_softmax_temp > 0 else 1.
        aug_log_prob = F.log_softmax(result.logits[sup_size:] / uda_softmax_temp, dim=-1)
        
        # KLdiv loss
        unsup_loss = torch.sum(self.loss_fun_unsup(aug_log_prob, ori_prob), dim=-1)
        unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / \
            torch.max(torch.sum(unsup_loss_mask, dim=-1), torch.tensor(1.).to(self.device))
        final_loss = sup_loss + self.uda_coeff * unsup_loss
        
        # return total loss, sup result (.logits), sup labels
        return final_loss, result.logits[:sup_size], b_labels
    
    @classmethod
    def get_tsa_thresh(cls, schedule, global_step, num_train_steps, start, end):
        training_progress = torch.tensor(float(global_step) / float(num_train_steps))
        if schedule == "linear":
            threshold = training_progress
        elif schedule == "exp":
            scale = 5
            threshold = torch.exp((training_progress - 1) * scale)
        elif schedule == "log":
            scale = 5
            threshold = 1 - torch.exp((-training_progress) * scale)
        output = threshold * (end - start) + start
        return output




