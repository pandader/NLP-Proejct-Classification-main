# Copyright 2024 @ Lun Li
#
# Summary:
#     1. Loader to set up HF Vanilla/Pretrained Bert Models;
#     2. Loader to set up HF BERT with Classification Head as well as our own customized BERT + 

# MULTILABEL-CLASSIFIER
from enum import Enum
from typing import Any, Optional, Union, Tuple
# torch
import torch
from torch import nn
# huggingface
from transformers import BertPreTrainedModel, BertConfig, BertTokenizerFast, BertModel, BertForSequenceClassification, BertForMaskedLM
from transformers.modeling_outputs import SequenceClassifierOutput

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
        cls_name = MultiLabelClassifier.__name__
        self.sequential = nn.Sequential()
        structure = [input_dim] + hidden_dims + [num_labels]
        for i in range(0, len(structure)-1):
            self.sequential.add_module(f'dropout_{i}', nn.Dropout(dropout))
            self.sequential.add_module(f'linear_{i}', nn.Linear(structure[i], structure[i+1]))
            if i == len(structure) - 2:
                # don't apply activation for the last layer
                break
            self.sequential.add_module(f'activation_{i}', nn.ReLU())
    
    def forward(self, x: torch.Tensor):
        return self.sequential(x)


# ### MODEL ASSEMBLER
# # Base BERT [from BERT LOADER]  + MultiLabelClassifier
# class BertClassifier(nn.Module):

#     def __init__(self, 
#                  bert_name: Optional[str]="", # pretriained model name
#                  num_labels: Optional[int]=2, # number of output labels
#                  dropout: Optional[float]=0.1, # dropout ratio
#                  hidden_dims: Optional[list]=list(), # hidden dimensions in the classifier
#                  config_override: Optional[BertConfig]=BertConfig(), # use user provided config
#                  load_tokenizer: Optional[bool]=False, # load corresponding tokenizer
#                 ):
#         super().__init__()
#         self.loader = BertLoader(bert_name, config_override=config_override, load_tokenizer=load_tokenizer)
#         self.bert = self.loader.model
#         self.classifier = MultiLabelClassifier(self.loader.config.hidden_size, num_labels, hidden_dims, dropout)
#         self.config = self.bert.config

#     @property
#     def tokenizer(self):
#         return self.loader.tokenizer

#     def forward(self, 
#                 input_ids: torch.Tensor, 
#                 attention_mask: torch.Tensor, 
#                 token_type_ids: Optional[torch.Tensor]=None,
#                 inputs_embeds=inputs_embeds,
#                 labels=labels,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,):
#         h = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         logits = self.classifier(h.pooler_output)
#         return SequenceClassifierOutput(logits=logits)

class BertClassifier(BertPreTrainedModel):
    def __init__(self, 
                 bert_name: Optional[str]="bert-base-uncased", # pretriained model name
                 num_labels: Optional[int]=2, # number of output labels
                 dropout: Optional[float]=0.1, # dropout ratio
                 hidden_dims: Optional[list]=list(), # hidden dimensions in the classifier                 
                 config_override: Optional[Any]=None, # use user provided config
                 load_tokenizer: Optional[bool]=False, # load corresponding tokenizer
                 ):
        bert_config = BertConfig.from_pretrained(bert_name) if config_override is None else config_override
        # if load tokenizer
        self.tokenizer_ = BertTokenizerFast.from_pretrained(bert_name) if load_tokenizer else None
        super().__init__(bert_config)
        self.num_labels = num_labels
        self.config = bert_config
        self.bert = BertModel(bert_config)
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classifier(outputs.pooler_output)
        return SequenceClassifierOutput(logits=logits)

### MODEL SELECTOR

class ClassifierType(Enum):
    BERT_CLASSIFIER_HF = 1
    BERT_CLASSIFIER = 2
    BERT_MLM_HF = 3

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
        else:
            # HF model doesn't support multi-layer classifier
            assert(len(hidden_dims) == 0)
            self.tokenizer_ = BertTokenizerFast.from_pretrained(bert_name) if load_tokenizer else None
            if head_type == ClassifierType.BERT_CLASSIFIER_HF:
                self.model_ = BertForSequenceClassification.from_pretrained(bert_name, num_labels=num_labels, classifier_dropout=dropout)
            elif head_type == ClassifierType.BERT_MLM_HF:
                self.model_ = BertForMaskedLM.from_pretrained(bert_name)

    @property
    def tokenizer(self):
        return self.tokenizer_

    @property
    def model(self):
        return self.model_








