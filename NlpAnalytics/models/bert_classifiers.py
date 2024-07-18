# Copyright 2024 @ Lun Li
#
# Summary:
#     1. Load HF Vanilla/Pretrained Bert Models
#     2. Load HF BERT with classification head
#     3. Load HF BERT with customizable classification head

# MULTILABEL-CLASSIFIER
from enum import Enum
from typing import Any, Optional
# torch
import torch
from torch import nn
# huggingface
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (BertPreTrainedModel, BertConfig, BertTokenizerFast, BertModel, BertForSequenceClassification)
# utilities
from ..utilities import get_device

### BASE BERT MODEL
### load model and corresponding tokenizer from huggingface
class BertLoader:

    def __init__(self, 
                bert_name: str="bert-base-uncased", # pretranied model name
                config_override: Optional[BertConfig]=BertConfig(), # use user provided config
                load_tokenizer: Optional[bool]=False, # load corresponding tokenizer
                ): 
        # tokenizer
        tokenizer_name = "bert-base-uncased" if bert_name == "" else bert_name
        self.tokenizer_ = BertTokenizerFast.from_pretrained(tokenizer_name) if load_tokenizer else None        
        self.model_ = None
        self.config_ = config_override
        # bert model
        if bert_name == "":
            self.model_ = BertModel(self.config_)
            print('Warning: loading BERT with initial parameters == 0.')
        else:
            self.config_ = BertConfig.from_pretrained(bert_name, output_hidden_states=True, output_attentions=True)
            self.model_ = BertModel.from_pretrained(bert_name, self.config_)

    @property
    def config(self):
        return self.config_

    @property
    def tokenizer(self):
        return self.tokenizer_

    @property
    def model(self):
        return self.model_

### MULTI-LABEL CLASSIFICATION HEAD
### if no hidden_dims, it recovers the simple classifier as in BertForSequenceClassification
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden_dims: Optional[list]=list(), dropout: Optional[float]=0.1):
        super().__init__()
        self.num_labels_ = num_labels
        self.sequential = nn.Sequential()
        structure = [input_dim] + hidden_dims + [self.num_labels_]
        for i in range(0, len(structure)-1):
            self.sequential.add_module(f'dropout_{i}', nn.Dropout(dropout))
            self.sequential.add_module(f'linear_{i}', nn.Linear(structure[i], structure[i+1]))
            if i == len(structure) - 2:
                # not apply activation for the last layer
                break
            self.sequential.add_module(f'activation_{i}', nn.ReLU())
        self.sequential.to(get_device())
    
    @property
    def num_labels(self):
        return self.num_labels_

    def forward(self, x: torch.Tensor):
        return self.sequential(x)

### This is a customizable BERT (user can 'configure' MultiLabelClassifier class above)
class BertClassifier(BertPreTrainedModel):

    def __init__(self, 
                 bert_name: Optional[str]="bert-base-uncased", # pretriained model name
                 num_labels: Optional[int]=2, # number of output labels
                 dropout: Optional[float]=0.1, # dropout ratio
                 hidden_dims: Optional[list]=list(), # hidden dimensions in the classifier                 
                 config_override: Optional[Any]=None, # use user provided config
                 load_tokenizer: Optional[bool]=False # load corresponding tokenizer
                 ):

        bert_config = BertConfig.from_pretrained(bert_name) if config_override is None else config_override
        # if load tokenizer
        self.tokenizer_ = BertTokenizerFast.from_pretrained(bert_name) if load_tokenizer else None
        super().__init__(bert_config)
        self.num_labels = num_labels
        self.config = bert_config
        if bert_name == "":
            self.bert = BertModel(bert_config)
        else:
            self.bert = BertModel.from_pretrained(bert_name)
        # initialize customized classifier
        self.classifier = MultiLabelClassifier(self.config.hidden_size, num_labels, hidden_dims, dropout)
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def tokenizer(self):
        return self.tokenizer_
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict if return_dict is not None else self.config.use_return_dict,
        )

        logits = self.classifier(outputs.pooler_output)
        return SequenceClassifierOutput(logits=logits, hidden_states=(outputs.pooler_output,))

### BERT Classifier SELECTOR
class ClassifierType(Enum):

    BERT_CLASSIFIER_HF = 1
    BERT_CLASSIFIER = 2

### BERT + Classification head Loader
class BertClassifierLoader:

    def __init__(self, 
                 head_type: ClassifierType,
                 bert_name: Optional[str]="", # pretriained model name
                 num_labels: Optional[int]=2, # number of output labels
                 dropout: Optional[float]=0.1, # dropout ratio
                 hidden_dims: Optional[list]=list(), # hidden dimensions in the classifier
                 config_override: Optional[BertConfig]=BertConfig(), # use user provided config
                 load_tokenizer: Optional[bool]=False # load corresponding tokenizer
                ):
        
        self.tokenizer_ = None
        if head_type == ClassifierType.BERT_CLASSIFIER:
            self.model_ = BertClassifier(bert_name, num_labels, dropout, hidden_dims, config_override, load_tokenizer)
            self.tokenizer_ = self.model_.tokenizer
        elif head_type == ClassifierType.BERT_CLASSIFIER_HF:
            # HF model doesn't support multi-layer classifier
            assert(len(hidden_dims) == 0)
            self.tokenizer_ = BertTokenizerFast.from_pretrained(bert_name) if load_tokenizer else None
            self.model_ = BertForSequenceClassification.from_pretrained(bert_name, num_labels=num_labels, classifier_dropout=dropout)

    @property
    def tokenizer(self):
        return self.tokenizer_

    @property
    def model(self):
        return self.model_
