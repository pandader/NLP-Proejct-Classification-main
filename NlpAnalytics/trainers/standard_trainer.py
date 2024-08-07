# Copyright 2024 @ Lun Li
#
# Summary: A standard trainer/evaluator skeleton sets up the interface and demonstrates how it interacts with trainer utilities
# Other customized trainers (including semi-supervsied ones) shall subclass this one while adding/overriding key member functions

from tqdm import trange
from typing import Optional, Any, Callable
# torch
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
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
    
    def run(self, cur_model : nn.Module, aux_models : list[nn.Module]):
        resMgr = ResultsMgr(num_epochs=1)
        cur_model.eval()
        # Predict
        with torch.no_grad():
            resMgr.start_this_epoch()
            for _, batch in enumerate(self.data_loader):
                b_input_ids, b_input_mask, b_labels = send_to_device(batch, self.device)
                result = cur_model(b_input_ids, b_input_mask)
                sup_loss = self.loss_func(result.logits, b_labels)
                sup_loss = torch.mean(sup_loss)
                resMgr.step(result.logits, b_labels, (sup_loss, None))
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
        # default: generate batch based on trainnig data
        self.generate_batch_based_on = DataLoaderType.TRAINING
        # default: no auxiliary model
        self.aux_models = []

    def train(self, 
              epochs : int, 
              schedule_type : Optional[SchedulerType]=SchedulerType.CONSTANT, 
              num_warmup_steps : Optional[int]=0,
              override_schedule : Optional[Any]=None,
              save_model_freq : Optional[int]=-1,
              save_loc : Optional[str]='',
              model_name : Optional[str]='this_model'):
        
        # set up scheduler
        if schedule_type != SchedulerType.CONSTANT:
            self.set_scheduler(epochs, schedule_type, num_warmup_steps, override_schedule)
        # set up results mgr
        self.resMgr.set_num_epochs(epochs)
        # start epoch
        for _ in trange(epochs, desc = "Epoch"):
            self.model.train()
            self.resMgr.start_this_epoch()
            for iter, batch in enumerate(self.data_loaders[self.generate_batch_based_on]):
                # clear out the gradients
                self.optimizer.zero_grad()                
                # preprocess
                self.preprocess()
                # calculate loss
                loss, sup_result, sup_labels, loss_to_report = self.calcualte_loss(iter, batch)
                # backward pass
                loss.backward()
                # postprocess
                self.postprocess()
                # update paras
                self.optimizer.step()
                # scheduler (if applicable)
                if self.scheduler is not None:
                    self.scheduler.step()
                # gather results and report if needed
                self.resMgr.step(sup_result, sup_labels, loss_to_report)
                self.resMgr.report(iter)
            # summary training results
            self.resMgr.end_this_epoch()
            # evaluation step (validation)
            self.eval.run(self.model, self.aux_models)
            # save model
            if save_model_freq != -1 and self.resMgr.get_epoch_idx() % save_model_freq == 0:
                save_model(self.resMgr.get_epoch_idx(), model_name, self.model, save_loc, self.aux_models)

    def set_scheduler(self, 
                      num_epochs : int,
                      schedule_type : SchedulerType, 
                      num_warmup_steps : int,
                      override_schedule : Any):

        if override_schedule is not None:
            self.scheduler = override_schedule
        else: 
            num_of_batches = len(self.data_loaders[self.generate_batch_based_on])
            self.scheduler = self.optimizer.compile_schedule(num_epochs * num_of_batches, schedule_type, num_warmup_steps)
    
    def calcualte_loss(self, batch_idx : int, batch : Any):        
        # 'batch' contains [0]: input ids; [1]: attention masks; [2]: labels
        b_input_ids, b_input_mask, b_labels = send_to_device(batch, self.device)        
        # forward pass
        result = self.model(b_input_ids, b_input_mask)
        # apply loss functions
        sup_loss = self.loss_func(result.logits, b_labels)
        sup_loss = torch.mean(sup_loss)
        # return total loss, sup result (.logits), sup labels
        return sup_loss, result.logits, b_labels, (sup_loss, None)
    
    def preprocess(self):
        # adds flexibility: inject preprocess before calculateLoss
        pass

    def postprocess(self):
        # adds flexibility: inject postprocess after calculateLoss
        pass
    