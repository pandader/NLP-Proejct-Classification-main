import os
import numpy as np
from enum import Enum
from tqdm import trange
# torch
import torch
from torch import nn
import torch.nn.functional as F
# transformers
from transformers.trainer_utils import SchedulerType
# Peft
from peft import PeftModel
# local
from uda_data import *

# makes dataLoader iterable
def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x

# port data to device
def send_to_device(data, device):
    return [d.to(device) for d in data]

# model saving
def save_model(epoch, model_name, model, path=""):
  if isinstance(model, PeftModel):
      model.save_pretrained(os.path.join(path, f'{model_name}_epcoh_{epoch}'))
  else:
    export_path = os.path.join(path, f'{model_name}_epoch_{epoch}.pt')
    torch.save(model.state_dict(), export_path)

# model loading
def load_model(epoch, model_name, model, path="", is_peft=False):
  if is_peft:
      import_path = os.path.join(path, f'{model_name}_epcoh_{epoch}')
      PeftModel.from_pretrained(model, import_path, is_trainable=True)
  else:
    import_path = os.path.join(path, f'{model_name}_epoch_{epoch}.pt')
    model.load_state_dict(torch.load(import_path))

# results manager
class ResultsMgr:

    def __init__(self, num_epochs=1, report_freq=-1):
        """
        Args:
            report_freq (int, optional): Within an epoch, every report_freq, report aggregated results. Defaults to -1 (disabled).
        """
        self.epoch_idx = 0
        self.num_epochs = num_epochs
        self.report_freq = report_freq
        self.agg_res = dict()

    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs

    def start_this_epoch(self):
        self.this_epoch_step = 0
        # this epoch container
        self.preds = []
        self.labels = []
        self.sup_loss = []
        self.unsup_loss = []

    def step(self, model_output, labels, loss, **kwargs):
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
            unsup_loss_str = '' if len(self.unsup_loss) == 0 else f', the training unsup-loss is {np.mean(self.unsup_loss)}'
            print(sup_loss_str + unsup_loss_str + '.')
            

    def end_this_epoch(self, verbose=True):
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
    
    # some getters
    def get_agg_res(self, epoch=-1):
        assert(epoch <= self.num_epochs)
        if epoch == -1:
            return self.agg_res
        else:
            return self.agg_res[epoch]
    
    def get_epoch_idx(self):
        return self.epoch_idx
    
    def get_num_epochs(self):
        return self.num_epochs

# evaluator
class Evaluator:
    '''
    for validation/testing purpose
    '''
    def __init__(self, loss_func, data_loader, device=torch.device('cpu')):
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.device = device

    def run(self, cur_model):
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

# standard trainer
class Trainer:
    
    def __init__(self, model, data_loader, loss_func, optimizer, report_freq=100, device=torch.device('cpu')):
        self.model = model
        self.data_loaders = data_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = None
        self.device = device
        self.model.to(self.device)
        # results manager
        self.resMgr = ResultsMgr(report_freq=report_freq)
        # evaluator on validation set
        self.eval = Evaluator(self.loss_func, self.data_loaders[DataLoaderType.VALIDATION], device=self.device)
        # default: generate batch based on trainnig data
        self.generate_batch_based_on = DataLoaderType.TRAINING

    def train(self, 
              epochs, 
              schedule_type=SchedulerType.CONSTANT, 
              num_warmup_steps=0,
              override_schedule=None,
              save_model_freq=-1,
              save_loc="",
              model_name="this_model"):
        
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
                # calculate loss
                loss, sup_result, sup_labels, loss_to_report = self.calcualte_loss(iter, batch)
                # backward pass
                loss.backward()
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
            self.eval.run(self.model)
            # save model
            if save_model_freq != -1 and self.resMgr.get_epoch_idx() % save_model_freq == 0:
                save_model(self.resMgr.get_epoch_idx(), model_name, self.model, save_loc)
    
    def set_scheduler(self, num_epochs, schedule_type, num_warmup_steps, override_schedule):
        if override_schedule is not None:
            self.scheduler = override_schedule
        else: 
            num_of_batches = len(self.data_loaders[self.generate_batch_based_on])
            self.scheduler = self.optimizer.compile_schedule(num_epochs * num_of_batches, schedule_type, num_warmup_steps)
    
    def calcualte_loss(self, batch_idx, batch):
        # 'batch' contains [0]: input ids; [1]: attention masks; [2]: labels
        b_input_ids, b_input_mask, b_labels = send_to_device(batch, self.device)        
        # forward pass
        result = self.model(b_input_ids, b_input_mask)
        # apply loss functions
        sup_loss = self.loss_func(result.logits, b_labels)
        sup_loss = torch.mean(sup_loss)
        # return total loss, sup result (.logits), sup labels
        return sup_loss, result.logits, b_labels, (sup_loss, None)

class TrainerUDA(Trainer):

    def __init__(self, 
                 model, 
                 data_loader, 
                 loss_func_dict, 
                 optimizer, 
                 report_freq= 100, 
                 tsa_schedule="linear", 
                 uda_confidence_thresh=0.50, 
                 uda_softmax_temp=0.85, 
                 uda_coeff=1.,
                 device=torch.device('cpu')):

        assert('sup' in loss_func_dict and 'unsup' in loss_func_dict)
        super().__init__(model, data_loader, loss_func_dict['sup'], optimizer, report_freq, device)
        ### instaniate uda specific parameters
        assert (tsa_schedule in ["linear", "exp", "log"])
        self.uda_coeff = uda_coeff
        self.tsa_schedule = tsa_schedule
        self.uda_confidence_thresh = uda_confidence_thresh
        self.uda_softmax_temp = uda_softmax_temp
        self.loss_fun_unsup = loss_func_dict['unsup']
        self.loss_fun = loss_func_dict['sup']
        # since labeled data is << unlabeled data
        # the batches will be generated from these unlabeled ones
        # in the meanwhile, we repeated generate data fraom labeled ones
        self.train_sup_dataloader_iter = repeat_dataloader(self.data_loaders[DataLoaderType.TRAINING])
        self.generate_batch_based_on = DataLoaderType.TRAINING_UNLABELED

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

    # uda loss computation logic
    def calcualte_loss(self,  batch_idx, batch):

        # var / device assignment
        b_input_ids, b_input_mask, b_labels = send_to_device(next(self.train_sup_dataloader_iter), self.device)
        b_ori_input_ids, b_ori_input_mask, b_aug_input_ids, b_aug_input_mask = send_to_device(batch, self.device)            
        # stack up + aug unsup data
        # NOTICE: we intentinoally leave out orig unsup data,
        #         as we shall not update params due to training orig unsup data
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
            self.resMgr.num_epochs * num_batches, 
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
        return final_loss, result.logits[:sup_size], b_labels, (sup_loss, unsup_loss)

