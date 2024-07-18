# Copyright 2024 @ Lun Li
#
# Summary: Introduce a BERT specific AdamW (weight_decay fixes):
#          1) user can config parameters to be optimized through functors (defaulted to train all but embedding layers)
#          2) user can config parameters to be regularized through functors (defaulted to train all bert weights (bias/LayerNorm) parameter)

from typing import Optional, Tuple
# torch
from torch import nn
# hugging face
from transformers.optimization import AdamW, get_scheduler, SchedulerType
# local
from ..models.model_parameters_analyzer import ModelParameterAnalyzer
from peft import LoraConfig, get_peft_model, PeftModel


# bert embedding and encpder freezing - 
def parameter_freezing_layers(model : nn.Module, layer_freeze : Optional[dict]=dict()):
    # default set up -- freezing embedding as always
    if len(layer_freeze) == 0: layer_freeze={'embeddings' : True, 'encoder' : None} 
    if 'embeddings' in layer_freeze and layer_freeze['embeddings']:
        for params in model.bert.embeddings.parameters():
            params.requires_grad = False
    if 'encoder' in layer_freeze and layer_freeze['encoder'] is not None:
        for i in range(layer_freeze['encoder']):
            for params in model.bert.encoder.layer[i].parameters():
                params.requires_grad = False

# lora get paramaters
def get_parameter_names(model: PeftModel, forbidden_layer_types: list):
    result = []
    for name, child in model.named_children():
        result += [f"{name}.{n}" for n in get_parameter_names(child, forbidden_layer_types) \
                   if not isinstance(child, tuple(forbidden_layer_types))]
    result += list(model._parameters.keys())
    return result

### BertAdam + Scheduler
class AdamNLP(AdamW):

    def __init__(self, model : nn.Module, grouped_params : list, lr: float, betas : Tuple[float, float], eps : float, weight_decay : float):
        self.transformed_model = model
        super().__init__(grouped_params, lr, betas, eps, weight_decay, True, False)

    @classmethod
    def newNLPAdam(cls, model: nn.Module, layer_freeze: Optional[dict]=dict(),
                    lr: Optional[float] = 1e-3, betas: Optional[Tuple[float, float]] = (0.9, 0.999),
                    eps: Optional[float] = 1e-6, weight_decay: Optional[float] = 0.01):
        
        if len(layer_freeze) == 0: layer_freeze={'embeddings' : True, 'encoder' : None}
        param_analyzer = ModelParameterAnalyzer(model)

        # grouped parameters
        param_with_decay = {'params' : list(), 'weight_decay' : weight_decay}
        param_wo_decay = {'params' : list(), 'weight_decay' : 0.0}

        # all parameters
        all_params = set(param_analyzer.display_all_parameters())
        
        # freezing embedding layer or not
        if layer_freeze['embeddings']:
            # if freezed
            if layer_freeze['encoder'] is None: 
                # encoder is not freeze
                parameter_freezing_layers(model, layer_freeze)
                freeze_param_names = param_analyzer.get_embeddings_params()
            else: 
                # encoder is freeze
                parameter_freezing_layers(model, layer_freeze)
                freeze_layers_num = layer_freeze['encoder']
                freeze_param_names = param_analyzer.get_embeddings_params() + param_analyzer.get_encoder_parameters_up_to_layer_n(freeze_layers_num, exclude_no_reg=False)
            weights_param_names = all_params - set(freeze_param_names) - set(param_analyzer.get_bert_no_reg_parameters()) - set(param_analyzer.get_classifier_no_reg_parameters())
            rest_param_names = all_params - set(freeze_param_names) - weights_param_names
        elif not layer_freeze['embeddings']:
            # if not freezed
            if layer_freeze['encoder'] is not None:
                # encoder freeze
                parameter_freezing_layers(model, layer_freeze)
                freeze_layers_num = layer_freeze['encoder']
                freeze_param_names = param_analyzer.get_encoder_parameters_up_to_layer_n(freeze_layers_num, exclude_no_reg=False)
            else: 
                # encoder nor freeze
                freeze_param_names = []
            weights_param_names = all_params - set(freeze_param_names) - set(param_analyzer.get_bert_no_reg_parameters()) - \
                set(param_analyzer.get_classifier_no_reg_parameters())- set(param_analyzer.get_embedding_layer_norm_parameters())
            rest_param_names = all_params - set(freeze_param_names) - weights_param_names
            
        for n, p in model.named_parameters():
                if n in weights_param_names:
                    param_with_decay['params'].append(p)
                elif n in rest_param_names:
                    param_wo_decay['params'].append(p)

        return cls(model, [param_with_decay, param_wo_decay], lr, betas, eps, weight_decay)

    @classmethod
    def newNLPAdam_LORA(cls, model: nn.Module, lora_config: LoraConfig, 
                        lr: Optional[float] = 1e-3, betas: Optional[Tuple[float, float]] = (0.9, 0.999),
                        eps: Optional[float] = 1e-6, weight_decay: Optional[float] = 0.01):
        peft_model = get_peft_model(model, lora_config)
        decay_params = get_parameter_names(peft_model, [nn.LayerNorm])
        decay_params = [each for each in decay_params if 'bias' not in each]
        optimzer_params = [
            {'params': [p for n, p in peft_model.named_parameters() if n in decay_params], 'weight_decay' : weight_decay},
            {'params': [p for n, p in peft_model.named_parameters() if n not in decay_params], 'weight_decay' : 0.}
        ]
        return cls(peft_model, optimzer_params, lr, betas, eps, weight_decay)

    def get_model_transformed(self):
        # if peft, we get peft model
        # if non-peft, we get original model
        return self.transformed_model

    def compile_schedule(self, 
                         total_num_steps,
                         schedule_type : Optional[SchedulerType]=SchedulerType.LINEAR, 
                         num_warmup_steps : Optional[int]=0):

        return get_scheduler(schedule_type, optimizer=self, num_warmup_steps=num_warmup_steps, num_training_steps=total_num_steps)

    @property
    def emb_param_names(self):
        return self.emb_param_names_
    
    @property
    def weights_param_names(self):
        return self.weights_param_names_
    
    @property
    def all_params(self):
        return self.all_params_

    @property
    def all_params(self):
        return self.rest_params_
