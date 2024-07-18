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
# create a huggingface account, for gated model such as llama3, you need to feed to have HF_TOKEN

import os
from typing import Optional
# huggingface
from transformers import pipeline
# native
from ..utilities.utilities import (get_device)

TEXT_GEN_AI_DEFAULT_CONFIG = {
    'hf_token' : '',
    'task' : 'text-generation',
    'model' : 'meta-llama/Meta-Llama-3-8B-instruct',
    'temperature' : 0.9,
    'max_new_token' : 100,
    'repetition_penalty' : 1.1,
    'batch_size' : 20,
    'device' : get_device()
}

SAMPLE_QUESTION = 'I want to know if there are any retention offers for existing members'

SAMPLE_ANSWER = 'Any possible retention offers that could make continued membership more appealing'

SYSTEMP_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant to rephrase a paragraph.
<|eot_id|>
"""

class TextAugGenAI:
    
    def __init__(self, 
                 config : Optional[dict]=TEXT_GEN_AI_DEFAULT_CONFIG,
                 sample_question : Optional[str]=SAMPLE_QUESTION,
                 sample_answer : Optional[str]=SAMPLE_ANSWER):

        os.environ["HF_TOKEN"] = config['hf_token']
        self.generator = pipeline(
            task=config['task'],
            model=config['model'],
            temperature=config['temperature'],
            max_new_tokens=config['max_new_token'],
            repetition_penalty=config['repetition_penalty'],
            device=config['device'],
            batch_size=config['batch_size'])
        self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id
        self.sample_prompt = f"""
<|start_header_id|>user<|end_header_id|>
Below is an example paragraph:
{sample_question}

Please rephrase the above paragraph. Make sure you only return one paragraph and nothing more.<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{sample_answer}<|eot_id|>
"""

    def augment(self, data : list):
        prompts = []
        for each in data:
            main_body = f"""
<|start_header_id|>user<|end_header_id|>
Below is an example paragraph:
{each}.

Please paraphrase the above sentence. Make sure you only return one paragraph and nothing more.<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
            prompts.append(SYSTEMP_PROMPT + self.sample_prompt + main_body)

        # prompting
        res = self.generator(prompts)
        return [res[i][0]['generated_text'].replace(prompts[i], '') for i in range(len(prompts))]