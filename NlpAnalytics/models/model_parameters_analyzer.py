# Copyright 2024 @ Lun Li
#
# Summary:
# Parse BERT or BERTClassifier parameters. This will be useful in set up the optimizer, 
# e.g., parameter freezing, regularization
    
from enum import Enum
from typing import Optional
# torch
from torch import nn

class BertParamType(Enum):
    EMBEDER = 1
    ENCODER = 2
    POOLER = 3

class ModelParameterAnalyze:

    def __init__(self, model : nn.Module, has_classifier : Optional[bool]=True):
        self.model  = model
        self.has_classifier = has_classifier
        self.bert = self.model.bert if self.has_classifier else model

    def display_all_parameters(self):
        return [n for n, _ in self.model.named_parameters()]

    def get_classifier_parameters(self):
        if not self.has_classifier:
            raise Exception("The input model does not have classifier.")
        return [f'classifier.{n}' for n, _ in self.model.classifier.named_parameters()]        

    def get_bert_parameters(self):
        return [f'bert.{n}' for n, _ in self.bert.named_parameters()]

    def get_bert_bias_parameters(self):
         return [n for n, _ in self.bert.named_parameters(prefix='bert') if 'bias' in n]
    
    def get_bert_layer_norm_parameters(self):
        return [n for n, _ in self.bert.named_parameters(prefix = 'bert') if 'LayerNorm' in n]

    def get_bert_no_reg_parameters(self): # return no decay parames names of bert
        return self.get_bert_bias_parameters() + list(set(self.get_bert_layer_norm_parameters()) - set(self.get_bert_bias_parameters()))
    
    def get_classifier_no_reg_parameters(self): # return no decay param names of classifier
        return [param for param in self.get_parameters_by_types(['classifier']) if 'bias' in param]

    def get_parameters_by_types(self, types: Optional[list] = ['encoder','embeddings','pooler','classifier'], exclude_no_reg : Optional[bool]=True): # type can only in ['encoder','embeddings','pooler','classifier']
        all_params = self.display_all_parameters()
        exclu_params = self.get_bert_no_reg_parameters() 
        result = []
        if exclude_no_reg:
            tmp_params = list(set(all_params)-set(exclu_params))
            for n in tmp_params:
                if any(nd in n for nd in types):
                    result.append(n)
        else:
            for n in all_params:
                if any(nd in n for nd in types):
                    result.append(n)
        return result

    def get_encoder_parameters_by_layers(self, layers: int, exclude_no_reg:Optional[bool]=True):
        # bert_params = self.get_bert_parameters()
        exclu_params = self.get_bert_no_reg_parameters()
        result = []
        for n, _ in self.bert.encoder.layer[:layers].named_parameters(prefix = 'bert.encoder.layer'):
                result.append(n)
        if exclude_no_reg:
            result = list(set(result)-set(exclu_params))
        return result
    
    def get_embeddings_params(self):
        return [n for n, _ in self.bert.embeddings.named_parameters(prefix = 'bert.embeddings')]
     

