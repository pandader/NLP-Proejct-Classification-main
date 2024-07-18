import os
import pandas as pd
# torch
from torch import Generator
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader, RandomSampler
# transformer
from transformers.optimization import SchedulerType
# native
from NlpAnalytics import *

PATH = 'NlpAnalytics/data/dummy_data'
GENERATOR = Generator().manual_seed(42)
USE_LORA = False

if __name__ == '__main__':

    ###  Data Loader
    # get tokenizer
    tokenizer = BertLoader(load_tokenizer=True).tokenizer
    # create DatasetNLP
    df_sup_train = pd.read_csv(os.path.join(PATH, "sup_train.csv"))
    train_sup_ds = DatasetNLP(input_df=df_sup_train, tokenizer=tokenizer, cols_label=['label'])
    df_unsup_train = pd.read_csv(os.path.join(PATH, "unsup_train.csv"))[:16]
    train_unsup_ds = DatasetNLP(input_df=df_unsup_train, tokenizer=tokenizer, cols_to_tokenize=['orig_text', 'aug_text'])
    df_sup_test = pd.read_csv(os.path.join(PATH, "sup_test.csv"))[:16]
    test_ds = DatasetNLP(input_df=df_sup_test, tokenizer=tokenizer, cols_to_tokenize=['text'], cols_label=['label'])
    # assemble data loader
    datamodeler = \
    {
        DataLoaderType.TRAINING: DataLoader(
            train_sup_ds, sampler=RandomSampler(train_sup_ds, generator=GENERATOR), batch_size=8),
        DataLoaderType.TRAINING_UNLABELED: DataLoader(
            train_unsup_ds, sampler=RandomSampler(train_unsup_ds, generator=GENERATOR), batch_size=8),
        DataLoaderType.VALIDATION: DataLoader(
            test_ds, sampler=RandomSampler(test_ds, generator=GENERATOR), batch_size=8)
    }

    ### Model & Optimization
    loader = BertClassifierLoader(ClassifierType.BERT_CLASSIFIER_HF, "bert-base-uncased", 2, 0.1, load_tokenizer=True)
    loss_dict = {
        'sup': get_loss_functions(LossFuncType.CROSS_ENTROPY, reduce='none'), 
        'unsup': get_loss_functions(LossFuncType.KL_DIV, reduce='none')
    }
    if not USE_LORA:
        optimizer = AdamNLP.newNLPAdam(loader.model, {'embeddings':True, 'encoder': 9}, lr=2e-4)
        model = optimizer.get_model_transformed()
    else:
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=["query", "key", "value"], r=1, lora_alpha=1, lora_dropout=0.1)
        optimizer = AdamNLP.newNLPAdam_LORA(loader.model, lora_config)
        model = optimizer.get_model_transformed()

    ### Training
    trainer = TrainerUDA(model, datamodeler, loss_dict, optimizer)
    trainer.train(2, schedule_type=SchedulerType.CONSTANT, save_model_freq=1)

        