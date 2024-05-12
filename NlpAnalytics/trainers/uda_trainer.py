from tqdm import trange
from typing import Optional, Any, Callable
# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
# local
from .standard_trainers import Trainer
from .trainer_utilities import (
     send_to_device, save_model, repeat_dataloader, DataLoaderType)
from ..optimizer import SchedulerType

class TrainerUDA(Trainer):

    def __init__(self, 
                 model: nn.Module, 
                 data_loader: dict, 
                 loss_func: dict[Callable], # sup/unsup
                 optimizer: Optimizer, 
                 report_freq: int | None = 100,
                 tsa_schedule : Optional[str]="linear",
                 uda_confidence_thresh : Optional[float]=0.50,
                 uda_softmax_temp : Optional[float]=0.85,
                 uda_coeff : Optional[float]=1.):

        ### instaniate uda specific parameters
        assert (tsa_schedule in ["linear", "exp", "log"])
        self.uda_coeff = uda_coeff
        self.tsa_schedule = tsa_schedule
        self.uda_confidence_thresh = uda_confidence_thresh
        self.uda_softmax_temp = uda_softmax_temp
        self.loss_fun_unsup = loss_func['unsup']
        super().__init__(model, data_loader, loss_func['sup'], optimizer, report_freq)

    def set_scheduler(self, 
                      num_epochs : int,
                      schedule_type : SchedulerType, 
                      num_warmup_steps : int,
                      override_schedule : Any,
                      batch_num_based_on : Optional[DataLoaderType]=DataLoaderType.TRAINING):
        
        if override_schedule is not None:
            self.scheduler = override_schedule
        else: 
            num_of_batches = len(self.data_loaders[batch_num_based_on])
            self.scheduler = self.optimizer.compile_schedule(num_epochs * num_of_batches, schedule_type, num_warmup_steps)

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

    def train(self, 
              epochs : int, 
              schedule_type : Optional[SchedulerType]=SchedulerType.CONSTANT, 
              num_warmup_steps : Optional[int]=0,
              override_schedule : Optional[Any]=None,
              save_model_freq : Optional[int]=-1,
              save_loc : Optional[str]="",
              model_name : Optional[str]="this_model"):
        
        ### will move these

        # set up scheduler
        if schedule_type != SchedulerType.CONSTANT:
            self.set_scheduler(epochs, schedule_type, num_warmup_steps, override_schedule)
        # set up results mgr
        self.resMgr.set_num_epochs(epochs)
        # since labeled data is << unlabeled data
        # the batches will be generated from these unlabeled ones
        # in the meanwhile, we repeated generate data fraom labeled ones
        train_sup_dataloader_iter = repeat_dataloader(self.data_loaders[DataLoaderType.TRAINING])
        # start epoch
        for _ in trange(epochs, desc = "Epoch"):
            self.model.train()
            self.resMgr.start_this_epoch()
            for iter, batch in enumerate(self.data_loaders[DataLoaderType.TRAINING_UNLABELED]):
                # var / device assignment
                b_input_ids, b_input_mask, b_labels = send_to_device(next(train_sup_dataloader_iter), self.device)
                b_ori_input_ids, b_ori_input_mask, b_aug_input_ids, b_aug_input_mask = send_to_device(batch, self.device)            
                # stack up + aug unsup data
                # NOTICE: we intentinoally leave out orig unsup data,
                #         as we shall no update params due to training orig unsup data
                input_ids = torch.cat((b_input_ids, b_aug_input_ids), dim=0)
                input_mask = torch.cat((b_input_mask, b_aug_input_mask), dim=0)
                # clear out the gradients
                self.optimizer.zero_grad()
                # forward pass
                # call bert + classifier => logits
                result = self.model(input_ids, input_mask)
                ########################################### LOSS ###########################################################
                ### sup loss [back prop]
                # tsa
                sup_size = b_labels.shape[0]
                sup_loss = self.loss_func(result.logits[:sup_size], b_labels)  # shape : train_batch_size
                num_batches = len(self.data_loaders[DataLoaderType.TRAINING_UNLABELED])
                tsa_thresh = TrainerUDA.get_tsa_thresh(
                    self.tsa_schedule, 
                    self.resMgr.get_epoch_idx() * num_batches + iter,
                    epochs * num_batches, 
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
                ########################################### LOSS ###########################################################

                # backward pass
                final_loss.backward()
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




