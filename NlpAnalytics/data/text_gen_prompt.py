# Copyright 2024 @ Lun Li
#
# Summary:
#     Text generation via genAI model provided by hugging face.
#     In particular, we follow the idea in UDG "Towards Zero-Label Language Learning"
#       https://arxiv.org/pdf/2109.09193
#     Instead of directly generating labels (Y) for unlabeled text (X), we leverage 
#     genAI model to generate X' via zero/few-short learning. We then use 
#     semi-supervised framework (mixmatch/uda) to train the classifier.
#
# Remark:
# create a huggingface account and fill out the form of llama3 model
# os.environ["HF_TOKEN"] = xxxxxx

import os
import numpy as np
import pandas as pd
from random import sample
from typing import Any, Optional, Callable
from torch import LongTensor, FloatTensor, eq
# torch
import torch
# huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):

    def __init__(self, stop_token_ids:list):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            print(f"Testing {input_ids[0][-len(stop_ids[0])+1:]} against {stop_ids[0][1:]}")
            if eq(input_ids[0][-len(stop_ids[0])+1:], stop_ids[0][1:]).all():
                return True
        return False

class GenAIModelLoader:
    
    ### wrap around hugging face gen ai model. Default to Llama 3
    
    def __init__(self, 
                 model_name : Optional[str]='meta-llama/Meta-Llama-3-8B-instruct', 
                 device : Optional[Any]=None,
                 root_path : Optional[str]="",
                 file_name : Optional[str]="generated_text"):
        
        """ Load genAI model and its tokenizer

        Args:
            model_name (str, optional): which gpt model. Defaults to 'meta-llama/Meta-Llama-3-8B-instruct'.
            device (Any, optional): 'cpu', 'cuda', 'mps'. Defaults to None.
            root_path (str, optional): root path to save data. Defaults to "".
            file_name (str, optional): file name of the save data. Defaults to "generated_text".
        """
        
        self.device = device
        self.path = root_path
        self.file_name = file_name
        # model/tokenizer downloading
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        if self.device is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id).to(self.device)

    def run_uda_generation(self, 
                           text_input : list, 
                           psuedo_label : str, 
                           prompt_template : Callable,
                           stop_list: list,
                           export_freq : Optional[int]=100,
                           max_new_tokens : Optional[int]=200,
                           do_sample : Optional[bool]=True, 
                           top_k : Optional[int]=50, 
                           top_p : Optional[float]=0.9, 
                           temperature : Optional[float]=0.1
                           ):
        
        count = 1
        recording_loc, recording_all = [], []
        export_freq = min(export_freq, len(text_input))
        stop_token_ids = [self.tokenizer(x,  return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
        stop_token_ids = [LongTensor(x).to(self.device) for x in stop_token_ids]
        for i, text in enumerate(text_input):
            sample_prompt = prompt_template(text, psuedo_label)
            if self.device is None:
                tokenized_inputs = self.tokenizer(sample_prompt, return_tensors='pt')
            else:
                tokenized_inputs = self.tokenizer(sample_prompt, return_tensors='pt').to(self.device)
            # text generation
            outputs = self.model.generate(
                **tokenized_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                temperature=temperature, stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_token_ids)]))
            
            # Decooding to text
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # recording
            recording_loc.append(output_text)
            if count == export_freq:
                # export loc
                pd.DataFrame(recording_loc, columns=['text'], index=None).to_csv(os.path.join(self.path, f'{self.file_name}_{i}.csv'))
                # aggregation to all
                recording_all += recording_loc
                # reset
                recording_loc = []
                count = 1
                # report
                print(f'Finished {i} samples.')
            else:
                count += 1
        return recording_all