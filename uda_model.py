# torch
import torch
from torch import nn
# huggingface
from transformers import BertTokenizerFast, BertForSequenceClassification


### load bert model
def load_bert_model(model_name, num_labels, drop_out=0.1, device=torch.device('cpu')):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, classifier_dropout=drop_out).to(device)
    return tokenizer, model


# Parse BERT or BERTClassifier parameters. This will be useful in set up the optimizer, 
# e.g., parameter freezing, regularization
class ModelParameterAnalyzer:

    def __init__(self, model):
        self.model  = model
        self.bert = self.model.bert

    def display_all_parameters(self):
        return [n for n, _ in self.model.named_parameters()]

    def get_classifier_parameters(self):
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

    # type can only in ['encoder','embeddings','pooler','classifier']
    def get_parameters_by_types(self, types=['encoder','embeddings','pooler','classifier'], exclude_no_reg=True): 
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

    def get_encoder_parameters_by_layers(self, layers, exclude_no_reg=True):
        exclu_params = self.get_bert_no_reg_parameters()
        result = []
        for n, _ in self.bert.encoder.layer[:layers].named_parameters(prefix = 'bert.encoder.layer'):
                result.append(n)
        if exclude_no_reg:
            result = list(set(result)-set(exclu_params))
        return result
    
    def get_embeddings_params(self):
        return [n for n, _ in self.bert.embeddings.named_parameters(prefix = 'bert.embeddings')]
     
  
    def get_embedding_layer_norm_parameters(self):
        return [n for n, _ in self.bert.embeddings.named_parameters(prefix = 'bert.embeddings') if 'LayerNorm' in n]








