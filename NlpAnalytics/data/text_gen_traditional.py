# Copyright 2024 @ Lun Li
#
# Summary:
#     Text generation via NLPAUG package, base on random word insert and back translation.
#     you can add more augmentation way on character/word/sentence level. Go to check NLPAUG
#     official github.
#     https://github.com/makcedward/nlpaug

# load package
import os
import numpy as np
import pandas as pd
from typing import Optional, Union, Any
# nlpaug
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
from nlpaug.util import Action

class TextGen():
    def __init__(self,from_model_name: Optional[str] = None,
                 to_model_name: Optional[str] = None):
        self.from_translation = from_model_name
        self.back_translation = to_model_name

    def TextAug(self,input_text: Union[str,list], num_aug = 2, num_thread: Optional[Any] = None):
        result_list = {}
        aug = naf.Sometimes([
            naw.RandomWordAug(),
            naw.BackTranslationAug(from_model_name=self.from_translation, 
                                   to_model_name=self.back_translation)])
        
        if isinstance(input_text, list):
            for each in input_text:
                result_list[each] = aug.augment(each, n = num_aug, num_thread = 10)
            return result_list
        else:
            return aug.augment(input_text, n = num_aug)

