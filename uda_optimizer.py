
# Introduce a BERT specific AdamW (weight_decay fixes):
#   1) user can config parameters to be optimized through functors (defaulted to train all but embedding layers)
#   2) user can config parameters to be regularized through functors (defaulted to train all bert weights (bias/LayerNorm) parameter)

from enum import Enum
# torch
from torch import nn
# hugging face
from transformers.optimization import AdamW, get_scheduler, SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from peft import LoraConfig, get_peft_model, PeftModel
from peft.utils.peft_types import TaskType
# local
from uda_model import ModelParameterAnalyzer

### types of loss functions (register more as needed)
class LossFuncType(Enum):
    CROSS_ENTROPY = 1
    KL_DIV = 2

### loss function getter
def get_loss_functions(loss_func_type, reduce='mean'):
    if loss_func_type == LossFuncType.CROSS_ENTROPY:
        return nn.CrossEntropyLoss(reduction=reduce)
    elif loss_func_type == LossFuncType.KL_DIV:
        return nn.KLDivLoss(reduction=reduce)
    else:
        raise Exception("Unsupported loss function type")

# bert embedding and encpder freezing - 
def parameter_freezing_layers(model, layer_freeze={'embeddings':True,'encoder':None}):

    if 'embeddings' in layer_freeze and layer_freeze['embeddings']:
        for params in model.bert.embeddings.parameters():
            params.requires_grad = False

    if 'encoder' in layer_freeze and layer_freeze['encoder'] is not None:
        for i in range(layer_freeze['encoder']):
            for params in model.bert.encoder.layer[i].parameters():
                params.requires_grad = False

# lora get paramaters
def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [f"{name}.{n}" for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))]
    result += list(model._parameters.keys())
    return result

### BertAdam + Scheduler
class AdamNLP(AdamW):

    def __init__(self, model, grouped_params, lr, betas, eps, weight_decay):
        self.transformed_model = model
        super().__init__(grouped_params, lr, betas, eps, weight_decay, True, False)

    @classmethod
    def newNLPAdam(cls, 
                   model, 
                   layer_freeze={'embeddings': True, 'encoder': None},
                   lr=1e-3, betas=(0.9, 0.999),
                   eps=1e-6, 
                   weight_decay=0.01):
        
        param_analyzer = ModelParameterAnalyzer(model)

        # grouped parameters
        param_with_decay = {'params' : list(), 'weight_decay' : weight_decay}
        param_wo_decay = {'params' : list(), 'weight_decay' : 0.0}

        # all parameters
        all_params = set(param_analyzer.display_all_parameters())
        
        # when embedding layer is frozen
        if layer_freeze['embeddings']:
            if layer_freeze['encoder'] is None: 
                # encoder not freeze
                parameter_freezing_layers(model, layer_freeze)
                freeze_param_names = param_analyzer.get_embeddings_params()
            else: 
                # encoder freeze
                parameter_freezing_layers(model, layer_freeze)
                freeze_layers_num = layer_freeze['encoder']
                freeze_param_names = param_analyzer.get_embeddings_params() + param_analyzer.get_encoder_parameters_by_layers(freeze_layers_num, exclude_no_reg=False)
            weights_param_names = all_params - set(freeze_param_names) - set(param_analyzer.get_bert_no_reg_parameters()) - set(param_analyzer.get_classifier_no_reg_parameters())
            rest_param_names = all_params - set(freeze_param_names) - weights_param_names

        # when embedding layer not freeze
        elif not layer_freeze['embeddings']:
            if layer_freeze['encoder'] is not None:
                # encoder freeze
                parameter_freezing_layers(model, layer_freeze)
                freeze_layers_num = layer_freeze['encoder']
                freeze_param_names = param_analyzer.get_encoder_parameters_by_layers(freeze_layers_num, exclude_no_reg=False)
            else: 
                # encoder not freeze
                freeze_param_names = []
            weights_param_names = all_params - set(freeze_param_names) - set(param_analyzer.get_bert_no_reg_parameters())- set(param_analyzer.get_classifier_no_reg_parameters())- set(param_analyzer.get_embedding_layer_norm_parameters())
            rest_param_names = all_params - set(freeze_param_names) - weights_param_names
            
        for n, p in model.named_parameters():
            if n in weights_param_names:
                param_with_decay['params'].append(p)
            elif n in rest_param_names:
                param_wo_decay['params'].append(p)

        return cls(model, [param_with_decay, param_wo_decay], lr, betas, eps, weight_decay)

    @classmethod
    def newNLPAdam_LORA(cls, model, lora_config, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        peft_model = get_peft_model(model, lora_config)
        decay_params = get_parameter_names(peft_model, [nn.LayerNorm])
        decay_params = [each for each in decay_params if 'bias' not in each]
        optimzer_params = [
            {'params':[p for n, p in peft_model.named_parameters() if n in decay_params], 'weight_decay': weight_decay},
            {'params':[p for n, p in peft_model.named_parameters() if n not in decay_params], 'weight_decay': 0.}]
        return cls(peft_model, optimzer_params, lr, betas, eps, weight_decay)

    def get_model_transformed(self):
        # if peft, we get peft model
        # if non-peft, we get original model
        return self.transformed_model

    def compile_schedule(self, 
                         total_num_steps,
                         schedule_type=SchedulerType.LINEAR, 
                         num_warmup_steps=0):

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
