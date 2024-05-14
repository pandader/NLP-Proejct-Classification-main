import os
import pandas as pd
import numpy as np
import torch
from torch import Generator
from peft import LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split, RandomSampler, TensorDataset
# transformer
from transformers.optimization import AdamW, get_scheduler, SchedulerType
# native
from NlpAnalytics import *

PATH = '/Users/lunli/Library/CloudStorage/GoogleDrive-yaojn19880525@gmail.com/My Drive/Colab Notebooks/'
DATASET_NAME = 'uda_imdb_data_128'

if __name__ == '__main__':
    # sup
    train_sup_data = torch.load(os.path.join(PATH, f'data/{DATASET_NAME}/train_sup_data.pt'))
    # unsup
    train_unsup_data = torch.load(os.path.join(PATH, f'data/{DATASET_NAME}/train_unsup_data.pt'))
    train_unsup_data = TensorDataset(
    torch.cat([train_unsup_data.tensors[0][:20000], train_unsup_data.tensors[0][-20000:]]),
    torch.cat([train_unsup_data.tensors[1][:20000], train_unsup_data.tensors[1][-20000:]]),
    torch.cat([train_unsup_data.tensors[2][:20000], train_unsup_data.tensors[2][-20000:]]),
    torch.cat([train_unsup_data.tensors[3][:20000], train_unsup_data.tensors[3][-20000:]]),
)
    # valid
    valid_data = torch.load(os.path.join(PATH, f'data/{DATASET_NAME}/val_data.pt'))
    # test
    test_data = torch.load(os.path.join(PATH, f'data/{DATASET_NAME}/test_data.pt'))

    # to dataloader
    generator = Generator().manual_seed(42)
    train_sup_dataloader = DataLoader(train_sup_data, sampler=RandomSampler(train_sup_data, generator=generator), batch_size=8)
    train_unsup_dataloader = DataLoader(train_unsup_data, sampler=RandomSampler(train_unsup_data, generator=generator), batch_size=24)
    valid_dataloader = DataLoader(valid_data, sampler=RandomSampler(valid_data, generator=generator), batch_size=16)
    test_dataloader = DataLoader(test_data, sampler=RandomSampler(test_data, generator=generator), batch_size=16)

    # trainer
    ### load HF BERT Classifier
    loader = BertClassifierLoader(ClassifierType.BERT_CLASSIFIER_HF, "bert-base-uncased", 2, 0.1, load_tokenizer=True)

    datamodeler = {DataLoaderType.TRAINING: train_sup_dataloader,DataLoaderType.VALIDATION: valid_dataloader,
                DataLoaderType.TESTING:test_dataloader, DataLoaderType.TRAINING_UNLABELED:train_unsup_dataloader}

    loss_sup = get_loss_functions(LossFuncType.CROSS_ENTROPY)
    loss_unsup = get_loss_functions(LossFuncType.KL_DIV)

    loss_dict = {'sup':loss_sup, 'unsup':loss_unsup}
    ##### no lora ####
    optimizer = AdamNLP.newNLPAdam(loader.model, {'embeddings':True, 'encoder': 9}, lr = 2e-4)
    model = optimizer.get_model_transformed()
    ##### lora #####
    # lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,target_modules=["query", "key", "value"], r=1, lora_alpha=1, lora_dropout=0.1)
    # optimizer = AdamNLP.newNLPAdam_LORA(loader.model, lora_config)
    # model = optimizer.get_model_transformed()

    trainer = TrainerUDA(model, datamodeler, loss_dict, optimizer)
    trainer.train(2, schedule_type = SchedulerType.INVERSE_SQRT, save_model_freq=-1)

        