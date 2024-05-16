# Copyright 2024 @ Lun Li
#
# Summary:
#     Subclass of Dataset to handle NLP data.
#     It also showcases how to subclass torch.util.data.Dataset to make a customized dataset.

import os
import pandas as pd
from typing import Optional
# torch
import torch
from torch.utils.data import Dataset, TensorDataset
# huggingface
from transformers.tokenization_utils import PreTrainedTokenizer
'''
Dataset Hierachy:

    -- ConcatDataset complies a list of Datasets, (Dataset 1, Dataset 2, Dataset 3, ...) 
        -- Dataset [abstract class]
            - TensorDataset(Tensors 1, Tensors 2, Tensors 3, ... )
              Tensors i corresponds to the i-th feature of sample data, thus a collection of Tensors (each <=> 1 data point)
              The most important function for this class is __getitem__ (or equivalently [] operator) which defines the behavior
              of a[i], where a is an instaniation of TensorDataset. 
            - Define your own inheritance of Dataset, which allows you to :
              a) modify the initialization of the dataset
              b) modify the behavior of __getitem__()
        
    -- Subset gets a subset of ConcatDataset or Dataset, i.e., Subset(Dataset, selected_indices_set)

Comments:
    If you define your own subclass of Dataset (see example below: class DatasetNLP), the most important thing is to define you __getitem__ function, becaus, for a concatenation of DatasetNLP or Subset of DatasetNLP, if you use [] operator, ultimately, it delegates to the __getitem__ function of DatasetNLP


Remark: function "random_split" takes in a ConcatDataset/Dataset and split it into fractions, i.e., 
        e.g,  random_split(dataset, [3, 7], generator=torch.Generator().manual_seed(42)) => dataset1, dataset2
'''

class DatasetNLP(Dataset):

    def __init__(self, 
                 input_df : pd.DataFrame,
                 tokenizer : PreTrainedTokenizer, 
                 cols_to_tokenize : Optional[list]=["*"], 
                 cols_label : Optional[list]=[],
                 bool_col : Optional[list]=[]
                 ):
        """ Convert a dataframe (text 1, [text 2], ..., [label]) to a Dataset class
        Args:
            input_df (DataFrame): a dataframe consists of : 1) col to tokenize; 2) col to not tokenize; 3) label
            tokenizer (PretrainedTokenizer): tokenizer to convert text to tensor,
            cols_to_tokenize (list, Optional): the column(s) to tokenize, Default to all except label(s)
            col_label (list, optional): the column(s) for label. Default to no label: []
            bool_col (list, optional): the column(s) of boolean values. Default to [].
        """
        Dataset.__init__(self)
        self.input_df = input_df
        self.tokenizer = tokenizer
        self.cols_label = cols_label
        self.bool_col = bool_col
        assert len(cols_to_tokenize) >= 1
        self.cols_to_tokenize = cols_to_tokenize
        if len(cols_to_tokenize) == 1 and cols_to_tokenize[0] == "*":
            self.cols_to_not_tokenize = []
            self.cols_to_tokenize = [col for col in input_df.columns if col not in self.cols_label]
        else:
            self.cols_to_not_tokenize = [col for col in input_df.columns if col not in self.cols_to_tokenize and col not in self.cols_label]
        # set label columns
        self.run_all()

    def run_all(self, 
            max_len : Optional[int]=128, 
            padding : Optional[bool]=True, 
            truncation: Optional[bool]=True, 
            ret_token_type_ids: Optional[bool]=False):
        """ Run all: in particular columns that requires tokenizations can have an overriden setting
        Args:
            max_len (int, optional): the maximum length for the returned tensor. Defaults to 128.
            padding (bool, optional): whether applying padding to make sequence equal-length. Defaults to True.
            truncation (bool, optional): trunacte if exceeding max_len. Defaults to True.
            ret_token_type_ids (bool, optional): whether return the tensor fo token_type_ids. Default to False.
        """
        self.desc = []
        self.tensors = []
        # labels
        self.tensor_labels = []
        if len(self.cols_label) != 0:
            self.tensor_labels = [torch.LongTensor(self.input_df[label].tolist()) for label in self.cols_label]
        # column(s) to tokenize
        all_text = []
        for col in self.cols_to_tokenize:
            all_text += self.input_df[col].tolist()
        all_tensors = self.tokenizer(all_text, truncation=truncation, return_tensors='pt', padding=padding, max_length=max_len)
        df_len = len(self.input_df)
        for i, col in enumerate(self.cols_to_tokenize):
            self.desc.append(f'{col}_input_ids')
            self.tensors.append(all_tensors['input_ids'][i * df_len : (i + 1) * df_len].long())
            self.desc.append(f'{col}_attention_mask')
            self.tensors.append(all_tensors['attention_mask'][i * df_len : (i + 1) * df_len].long())
            if ret_token_type_ids:
                self.desc.append(f'{col}_token_type_ids')
                self.tensors.append(all_tensors['token_type_ids'][i * df_len : (i + 1) * df_len].long())
        # column(s) to not tokenize
        self.run_transform()
        # combine all
        for each in self.tensor_labels:
            self.tensors.append(each)
        self.desc += self.cols_label

    def run_transform(self):
        if len(self.bool_col) == 0:
            return
        for col in self.bool_col:
            self.tensors.append(torch.tensor(self.input_df[col].tolist()))

    def export_tensors_with_desc(self):
        return self.tensors, self.desc

    def export_as_tensordataset(self, file_name="", root_path=""):
        output_data = TensorDataset(*self.tensors)
        if file_name != "":
            torch.save(output_data, os.path.join(root_path, f'{file_name}.pt'))
        return output_data

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)