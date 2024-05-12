# Copyright 2024 @ Lun Li
#
# Summary:
#     Text generation via GenAI Model

import os
import numpy as np
import pandas as pd
from random import sample
from typing import Any, Optional, Callable
# torch
import torch
# huggingface
from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast


class GenAIModelLoader:
    ### BLOOM model seems to have the closest performance to GPT3 models
    ### https://huggingface.co/bigscience
    
    def __init__(self, 
                 model_name : Optional[str]='bigscience/bloom-7b1', 
                 device : Optional[Any]=None,
                 root_path : Optional[str]="",
                 file_name : Optional[str]="generated_text"):
        
        """ Load genAI model and its tokenizer

        Args:
            model_name (str, optional): which gpt model. Defaults to 'bigscience/bloom-7b1'.
            device (Any, optional): 'cpu', 'cuda', 'mps'. Defaults to None.
            root_path (str, optional): root path to save data. Defaults to "".
            file_name (str, optional): file name of the save data. Defaults to "generated_text".
        """
        
        self.device = device
        self.path = root_path
        self.file_name = file_name
        # model/tokenizer downloading
        self.tokenizer = BloomTokenizerFast.from_pretrained(model_name) 
        if self.device is None:
            self.model = BloomForCausalLM.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)
        else:
            self.model = BloomForCausalLM.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id).to(self.device)

    def run_uda_generation(self, 
                           text_input : list, 
                           psuedo_label : int, 
                           prompt_template : Callable,
                           export_freq : Optional[int]=100,
                           max_new_tokens : Optional[int]=128, 
                           do_sample : Optional[bool]=True, 
                           top_k : Optional[int]=40, 
                           top_p : Optional[float]=0.9, 
                           temperature : Optional[float]=0.9):
        
        count = 1
        recording_loc, recording_all = [], []
        export_freq = min(export_freq, len(text_input))
        for i, text in enumerate(text_input):
            sample_prompt = prompt_template(text, psuedo_label)
            if self.device is None:
                tokenized_inputs = self.tokenizer.encode(sample_prompt, return_tensors='pt')
            else:
                tokenized_inputs = self.tokenizer.encode(sample_prompt, return_tensors='pt').to(self.device)
            # text generation
            outputs = self.model.generate(
                tokenized_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
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