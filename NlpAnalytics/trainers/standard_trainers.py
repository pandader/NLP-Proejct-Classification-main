# Copyright 2024 @ Lun Li
#
# Summary: A standard trainer and evaluator to demonstrate a handful utilities and how they interact
#          Other trainers should subclass this one and add/override the member functions

from tqdm import trange
from typing import Optional, Any, Callable
# torch
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
# local
from .trainer_utilities import (send_to_device, save_model, DataLoaderType, ResultsMgr)
from ..utilities import get_device
from ..optimizer import SchedulerType

class Evaluator:
    '''
    For both evaluation and testing
    '''

    def __init__(self, loss_func: callable, data_loader : DataLoader):
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.device = get_device()
    
    def run(self, cur_model : nn.Module):
        resMgr = ResultsMgr(num_epochs=1)
        cur_model.eval()
        # Tracking variables
        preds, labels = [], []
        # Predict
        with torch.no_grad():
            resMgr.start_this_epoch()
            for _, batch in enumerate(self.data_loader):
                b_input_ids, b_input_mask, b_labels = send_to_device(batch, self.device)
                result = cur_model(b_input_ids, b_input_mask)
                sup_loss = self.loss_func(result.logits, b_labels)
                sup_loss = torch.mean(sup_loss)
                resMgr.step(result.logits, b_labels, sup_loss)

                # preds.append(torch.argmax(result.logits.to('cpu'), dim=1))
                # labels.append(b_labels.to('cpu'))
            resMgr.end_this_epoch(verbose=False)
        val_acc = resMgr.get_agg_res(1)['accuracy']
        print(f'Validation accuracy is: {val_acc.item()}.\n')


class Trainer:
    '''
    A vanilla Trainer
    '''
    def __init__(self,
                 model : nn.Module,
                 data_loader : dict,
                 loss_func : Callable,
                 optimizer : Optimizer,
                 report_freq : Optional[int]=100):

        self.model = model
        self.data_loaders = data_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = None
        self.device = get_device()
        self.model.to(self.device)
        # results manager
        self.resMgr = ResultsMgr(report_freq=report_freq)
        # evaluator on validation set
        self.eval = Evaluator(self.loss_func,self.data_loaders[DataLoaderType.VALIDATION])
    
    def set_scheduler(self, 
                      num_epochs : int,
                      schedule_type : SchedulerType, 
                      num_warmup_steps : int,
                      override_schedule : Any):
        
        if override_schedule is not None:
            self.scheduler = override_schedule
        else: 
            num_of_batches = len(self.data_loaders[DataLoaderType.TRAINING])
            self.scheduler = self.optimizer.compile_schedule(num_epochs * num_of_batches, schedule_type, num_warmup_steps)

    def train(self, 
              epochs : int, 
              schedule_type : Optional[SchedulerType]=SchedulerType.CONSTANT, 
              num_warmup_steps : Optional[int]=0,
              override_schedule : Optional[Any]=None,
              save_model_freq : Optional[int]=-1,
              save_loc : Optional[str]="",
              model_name : Optional[str]="this_model"):
        
        # set up scheduler
        if schedule_type != SchedulerType.CONSTANT:
            self.set_scheduler(epochs, schedule_type, num_warmup_steps, override_schedule)
        # set up results mgr
        self.resMgr.set_num_epochs(epochs)
        # start epoch
        for _ in trange(epochs, desc = "Epoch"):
            self.model.train()
            self.resMgr.start_this_epoch()
            for iter, batch in enumerate(self.data_loaders[DataLoaderType.TRAINING]):
                # 'batch' contains [0]: input ids; [1]: attention masks; [2]: labels
                b_input_ids, b_input_mask, b_labels = send_to_device(batch, self.device)
                # clear out the gradients
                self.optimizer.zero_grad()
                #forward pass
                result = self.model(b_input_ids, b_input_mask)
                # apply loss functions
                sup_loss = self.loss_func(result.logits, b_labels)
                sup_loss = torch.mean(sup_loss)
                # backward pass
                sup_loss.backward()
                # update paras
                self.optimizer.step()
                # scheduler (if applicable)
                if self.scheduler is not None:
                    self.scheduler.step()
                # gather results and report if needed
                self.resMgr.step(result.logits, b_labels, sup_loss)
                self.resMgr.report(iter)
            # summary training results
            self.resMgr.end_this_epoch()
            # evaluation step (validation)
            self.eval.run(self.model)
            # save model
            if save_model_freq != -1 and self.resMgr.get_epoch_idx() % save_model_freq == 0:
                save_model(self.resMgr.get_epoch_idx(), model_name, self.model, save_loc)
    