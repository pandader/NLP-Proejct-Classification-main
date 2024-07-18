# Copyright 2024 @ Lun Li
#
# Summary:
#     We improve the NLPAUG package (https://github.com/makcedward/nlpaug) 
#     by allowing vectorized augmentation

# load package
import pandas as pd
from typing import Optional
# nlpaug
import nlpaug.augmenter.word as naw
from nlpaug.util import Action
from nlpaug.flow import Pipeline
# native
from ..utilities.utilities import (get_device)


### default config
TEXT_GEN_DEFAULT_CONFIG = {
    'name' : 'Somtimes_Piepline',
    'aug_p' : 0.8,
    'verbose' : 0,
    'include_detail' : False,
    'flow' : [
        naw.RandomWordAug(),
        naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-fr',
        to_model_name='Helsinki-NLP/opus-mt-fr-en',
        device=get_device())
    ]
}

### TextAug
class TextAug(Pipeline):
    
    def __init__(self, config : Optional[dict]=TEXT_GEN_DEFAULT_CONFIG):
        Pipeline.__init__(self, 
                          name=config['name'], 
                          action=Action.SOMETIMES, 
                          flow=config['flow'], 
                          aug_p=config['aug_p'],
                          include_detail=config['include_detail'], 
                          verbose=config['verbose'])
    
    def draw(self):
        return self.aug_p > self.prob()

    # override
    # we remove thread option, the parallellism doesn't work with joblibs
    def augment(self, data : list, n : Optional[int]=1):
        augmented_results = [self._augment(data) for _ in range(n)]
        augmented_results = [r for sub_results in augmented_results for r in sub_results if len(r) > 0]
        this_df = pd.DataFrame(columns=['org', 'aug1', 'aug2'])
        this_df['org'] = data
        this_df['aug1'] = augmented_results[0]
        this_df['aug2'] = augmented_results[1]
        return this_df