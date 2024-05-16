from tqdm import trange
from typing import Optional, Any, Callable, Tuple
# torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
# local
from .standard_trainer import Trainer, Evaluator
from .trainer_utilities import (send_to_device, save_model, repeat_dataloader, DataLoaderType)
from ..optimizer import SchedulerType
from ..models import GANPackage


class EvaluatorDomainAdapt(Evaluator):
    
    def calcualte_loss(self, model, input_ids, input_mask, **kwargs):
        result = model(input_ids, input_mask)
        _, logits, probs = kwargs['d'](result.hidden_states[0])
        return logits

class TrainerDA(Trainer):

    def __init__(self, 
                 model : nn.Module,
                 gan_pckg: GANPackage,
                 data_loader: dict,                  
                 optimizer : Optimizer, 
                 epsilon : Optional[float]=1e-8,
                 report_freq: Optional[int]=100):
                
        super().__init__(model, data_loader, None, optimizer, report_freq)
        ### instaniate gan bert specific parameters
        self.generator = gan_pckg.g
        self.discriminator = gan_pckg.d
        self.optimizer_generator = gan_pckg.g_opt
        self.optimizer_discriminator = gan_pckg.d_opt
        self.noise_size = gan_pckg.noise_size
        self.epsilon = epsilon

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
            self.set_status_of_aux_models(to_train=True) # change staus of auxiliary models
            self.resMgr.start_this_epoch()
            for iter, batch in enumerate(self.data_loaders[self.generate_batch_based_on]):
                # clear out the gradients
                self.optimizer.zero_grad()
                # calculate loss
                loss, sup_result, sup_labels = self.calcualte_loss(iter, batch)                
                # gather results and report if needed
                self.resMgr.step(sup_result, sup_labels, loss)
                self.resMgr.report(iter)
            # summary training results
            self.resMgr.end_this_epoch()
            # evaluation step (validation)
            self.model.eval()
            self.set_status_of_aux_models(to_train=False)
            self.eval.run(self.model)
            # save model
            if save_model_freq != -1 and self.resMgr.get_epoch_idx() % save_model_freq == 0:
                save_model(self.resMgr.get_epoch_idx(), model_name, self.model, save_loc)

    # gan bert loss computation logic
    def calcualte_loss(self, 
                       batch_idx : int,
                       batch : Any):

        # zero d/g gradient
        self.optimizer_generator.zero_grad()
        self.optimizer_discriminator.zero_grad()
        
        # unpack this training batch from our dataloader. 
        # 'batch' contains [0]: input ids; [1]: attention masks; [2]: labels; [3]: wehther unlabeled or labeled
        b_input_ids, b_input_mask, b_label_mask, b_labels = send_to_device(batch, self.device)        
        batch_size = b_input_ids.shape[0]
     
        # encode real data in the BERTClassifier [the degenerated version, i.e., no classifier]
        model_outputs = self.model(b_input_ids, attention_mask=b_input_mask)
        hidden_states = model_outputs.hidden_states[0] # the CLS token thingy
        
        # generate fake data
        noise = torch.zeros(batch_size, self.noise_size, device=self.device).uniform_(0, 1)
        gen_rep = self.generator(noise)

        # generate the output of the discriminator for real and fake data.
        disciminator_input = torch.cat([hidden_states, gen_rep], dim=0)
        features, logits, probs = self.discriminator(disciminator_input)

        # separate outputs
        features_list = torch.split(features, batch_size)
        d_real_features, d_fake_features = features_list[0], features_list[1]
        logits_list = torch.split(logits, batch_size)
        d_real_logits, d_fake_logits = logits_list[0], logits_list[1]
        probs_list = torch.split(probs, batch_size)
        d_real_probs, d_fake_probs = probs_list[0], probs_list[1]
        
        ### Loss
        # G-Loss
        g_loss_d = -1 * torch.mean(torch.log(1 - d_fake_probs[:,-1] + self.epsilon))
        g_feat_reg = torch.mean(torch.pow(torch.mean(d_real_features, dim=0) - torch.mean(d_fake_features, dim=0), 2))
        g_loss = g_loss_d + g_feat_reg
  
        # D-Loss
        logits = d_real_logits[:,0:-1]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # the discriminator provides an output for both labeled and unlabeled real data
        # we skip unlabeld via masking
        label2one_hot = torch.nn.functional.one_hot(b_labels, self.num_labels)
        per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)
        per_example_loss = torch.masked_select(per_example_loss, b_label_mask)
        labeled_example_count = per_example_loss.type(torch.float32).numel()

        # in case that a batch does not contain labeled examples, the "supervised loss" won't be evaluated
        if labeled_example_count == 0:
          d_l_supervised = 0
        else:
          d_l_supervised = torch.div(torch.sum(per_example_loss), labeled_example_count)
                 
        d_l_unsupervised1u = -1 * torch.mean(torch.log(1 - d_real_probs[:, -1] + self.epsilon))
        d_l_unsupervised2u = -1 * torch.mean(torch.log(d_fake_probs[:, -1] + self.epsilon))
        d_loss = d_l_supervised + d_l_unsupervised1u + d_l_unsupervised2u

        # calculate weigth updates
        # retain_graph=True is required since the underlying graph will be deleted after backward
        g_loss.backward(retain_graph=True)
        d_loss.backward() 
        
        # Apply modifications
        self.optimizer_generator.step()
        self.optimizer_discriminator.step()
        self.optimizer.step()
        # scheduler (if applicable)
        if self.scheduler is not None:
            self.scheduler.step()

        # loss, model prediction, labels 
        real_labels = torch.masked_select(b_labels, b_label_mask)
        real_preds = torch.stack([logits[idx] for idx, each in enumerate(b_label_mask) if each], dim=0)
        return d_loss, real_preds, real_labels
    
    def set_status_of_aux_models(self, to_train : Optional[bool]=True):
        if to_train:
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()