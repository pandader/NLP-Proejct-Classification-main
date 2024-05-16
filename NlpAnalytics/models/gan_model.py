# Copyright 2024 @ Lun Li
#
# Summary: GanBERT structure + FineTuning,  
#          1) Generator: produce "fake" vector representations of sentences;
#          2) Discriminator: a BERT-based classifier over k+1 categories.
#          The model is compatiable with PeftFineTuning
#
#          Reference of a vanilla ganbert: https://github.com/crux82/ganbert-pytorch

from typing import Optional, Any
# torch
import torch
from torch import nn
# huggingface
from transformers import BertPreTrainedModel, BertConfig, BertTokenizerFast, BertModel, BertForSequenceClassification, BertForMaskedLM
from transformers.modeling_outputs import SequenceClassifierOutput

        

### Standard Generator 
### as in (https://www.aclweb.org/anthology/2020.acl-main.191/)
class GanGenerator(nn.Module):

    def __init__(self, 
                 noise_size : Optional[int]=100, 
                 output_size : Optional[int]=768, 
                 hidden_sizes : Optional[list]=[768], 
                 dropout_rate : Optional[float]=0.1):
        super(GanGenerator, self).__init__()

        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend(
                [
                    nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), 
                    nn.LeakyReLU(0.2, inplace=True), 
                    nn.Dropout(dropout_rate)
                ]
            )
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep

### Standard Discriminator 
### as in (https://www.aclweb.org/anthology/2020.acl-main.191/)
class GanDiscriminator(nn.Module):

    def __init__(self, 
                 input_size : Optional[int]=768, 
                 hidden_sizes : Optional[list]=[768], 
                 num_labels : Optional[int]=2,
                 dropout_rate : Optional[float]=0.1):
        super(GanDiscriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend(
                [
                    nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), 
                    nn.LeakyReLU(0.2, inplace=True), 
                    nn.Dropout(dropout_rate)
                ]
            )
        self.layers = nn.Sequential(*layers)
        # the extra 1 accounts for the probability of this sample being fake/real.
        self.logit = nn.Linear(hidden_sizes[-1], num_labels + 1) 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs



### Gan Model Pacakge
class GANPackage:
    
    def __init__(
            self, 
            num_labels : Optional[int]=2,
            noise_size : Optional[int]=100,
            shared_dim : Optional[int]=768,
            dropout : Optional[float]=0.1,
            gen_lr : Optional[float]=5e-5,
            disc_lr : Optional[float]=5e-5):
        # noise size
        self.noise_size = noise_size
        # set up generator and discriminator
        self.g = GanGenerator(noise_size, shared_dim, [shared_dim], dropout)
        self.d = GanDiscriminator(shared_dim, [shared_dim], num_labels, dropout)
        d_vars = [v for v in self.d.parameters()]
        g_vars = [v for v in self.g.parameters()]
        # optimizer
        self.d_opt = torch.optim.AdamW(d_vars, lr=disc_lr)
        self.g_opt = torch.optim.AdamW(g_vars, lr=gen_lr)
