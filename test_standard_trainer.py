import os
import pandas as pd
from torch import Generator
from peft import LoraConfig, TaskType
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split, RandomSampler
# transformer
from transformers.optimization import SchedulerType
# native
from NlpAnalytics import *

PATH = 'NlpAnalytics/data/dummy_data'
GENERATOR = Generator().manual_seed(42)
USE_LORA = False

if __name__ == '__main__':

    ###  Data Loader    
    # load intent dataset [remove the slot filling column]
    df_train = pd.read_csv(os.path.join(PATH, 'intent_train.csv'),index_col=0).drop('slot filling', axis = 1)
    df_test = pd.read_csv(os.path.join(PATH, 'intent_test.csv'),index_col=0).drop('slot filling', axis = 1)
    encoder = LabelEncoder()
    df_train['intent_new'] = encoder.fit_transform(df_train['intent'])
    tokenizer = BertLoader(load_tokenizer=True).tokenizer
    df_train_ = DatasetNLP(input_df=df_train, tokenizer=tokenizer, cols_to_tokenize=['query'],  cols_label=['intent_new'] )
    df_test_ = DatasetNLP(input_df=df_test, tokenizer=tokenizer, cols_label=['intent'] )
    df_train_new, df_valid = random_split(df_train_, [0.7, 0.3], generator=GENERATOR)
    train_dataloader = DataLoader(df_train_new, sampler=RandomSampler(df_train_new, generator=GENERATOR), batch_size=16)
    valid_dataloader = DataLoader(df_valid, sampler=RandomSampler(df_valid, generator=GENERATOR), batch_size=16)
    test_dataloader = DataLoader(df_test_, sampler=RandomSampler(df_test_, generator=GENERATOR), batch_size=16)
    datamodeler = {
        DataLoaderType.TRAINING: train_dataloader,
        DataLoaderType.VALIDATION: valid_dataloader,
        DataLoaderType.TESTING:test_dataloader
    }

    ### Model & Optimizer
    loader = BertClassifierLoader(ClassifierType.BERT_CLASSIFIER, "bert-base-uncased", 22, 0.1, load_tokenizer=True)
    my_loss_func = get_loss_functions(LossFuncType.CROSS_ENTROPY)    
    if not USE_LORA:
        optimizer = AdamNLP.newNLPAdam(loader.model, {'embeddings':False, 'encoder': None})
        model = optimizer.get_model_transformed()
    else:
        # lora ?
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,target_modules=["query", "key", "value"], r=1, lora_alpha=1, lora_dropout=0.1)
        optimizer = AdamNLP.newNLPAdam_LORA(loader.model, lora_config)
        model = optimizer.get_model_transformed()

    #### Training
    trainer = Trainer(model, datamodeler, my_loss_func, optimizer)
    trainer.train(1, schedule_type = SchedulerType.CONSTANT)