import pandas as pd
import numpy as np
import torch
from torch import Generator
from peft import LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split, RandomSampler
# transformer
from transformers.optimization import AdamW, get_scheduler, SchedulerType
# native
from NlpAnalytics import *


if __name__ == '__main__':

    # load intent dataset
    path = '/Users/lunli/Dropbox/Amex Project/'
    df_train = pd.read_csv(path+'intent_train.csv',index_col=0)
    df_test = pd.read_csv(path+'intent_test.csv',index_col=0)

    # remove the slot filling column
    df_train = df_train.drop('slot filling', axis = 1)
    df_test = df_test.drop('slot filling', axis = 1)

    # training set miss 4 labels
    encoder = LabelEncoder()
    df_train['intent_new'] = encoder.fit_transform(df_train['intent'])

    ### Load tokenizer
    tokenizer = BertLoader(load_tokenizer=True).tokenizer

    df_train_ = DatasetNLP(input_df=df_train, 
                        tokenizer=tokenizer,
                        cols_to_tokenize=['query'],  
                        cols_label=['intent_new'] )
    df_test_ = DatasetNLP(input_df=df_test, 
                        tokenizer=tokenizer,  
                        cols_label=['intent'] )

    # split to train and validation
    generator = Generator().manual_seed(42)
    df_train_new, df_valid = random_split(df_train_, [0.7, 0.3], generator=generator)
    train_dataloader = DataLoader(df_train_new, sampler=RandomSampler(df_train_new, generator=generator), batch_size=16)
    valid_dataloader = DataLoader(df_valid, sampler=RandomSampler(df_valid, generator=generator), batch_size=16)
    test_dataloader = DataLoader(df_test_, sampler=RandomSampler(df_test_, generator=generator), batch_size=16)

    #### trainer ####
    ### load HF BERT Classifier
    loader = BertClassifierLoader(ClassifierType.BERT_CLASSIFIER, "bert-base-uncased", 22, 0.1, load_tokenizer=True)

    datamodeler = {DataLoaderType.TRAINING: train_dataloader,DataLoaderType.VALIDATION: valid_dataloader,DataLoaderType.TESTING:test_dataloader}
    my_loss_func = get_loss_functions(LossFuncType.CROSS_ENTROPY)
    
    ##### no lora ####
    optimizer = AdamNLP.newNLPAdam(loader.model, {'embeddings':False, 'encoder': None})
    model = optimizer.get_model_transformed()
    ##### lora #####
    # lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,target_modules=["query", "key", "value"], r=1, lora_alpha=1, lora_dropout=0.1)
    # optimizer = AdamNLP.newNLPAdam_LORA(loader.model, lora_config)
    # model = optimizer.get_model_transformed()
    trainer = Trainer(model, datamodeler, my_loss_func, optimizer)
    trainer.train(1, schedule_type = SchedulerType.INVERSE_SQRT, save_model_freq=1)