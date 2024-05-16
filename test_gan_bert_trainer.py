import os
import math
import pandas as pd
import numpy as np
import torch
from torch import Generator
from peft import LoraConfig, TaskType
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, RandomSampler
# transformer
from transformers.optimization import AdamW, get_scheduler, SchedulerType
# native
from NlpAnalytics import *

### some utilities just for this stupid data set
def get_qc_examples(input_file):
  """Creates examples for the training and dev sets."""
  examples = []

  with open(input_file, 'r') as f:
      contents = f.read()
      file_as_list = contents.splitlines()
      for line in file_as_list[1:]:
          split = line.split(" ")
          question = ' '.join(split[1:])

          text_a = question
          inn_split = split[0].split(":")
          label = inn_split[0] + "_" + inn_split[1]
          examples.append((text_a, label))
      f.close()

  return examples

def stich_train_data(df_labeled, df_unlabeled):
    examples = []
    train_label_mask = np.ones(len(df_labeled), dtype=bool)
    train_unlabel_masks = np.zeros(len(df_unlabeled), dtype=bool)
    train_label_masks = np.concatenate([train_label_mask, train_unlabel_masks])
    df_all = pd.concat([df_labeled, df_unlabeled], axis=0)
    df_all.reset_index(drop=True, inplace=True)
    label_mask_rate = len(df_labeled) / len(df_all)
    balance = int(1 / label_mask_rate)
    balance = int(math.log(balance, 2)) # not sure why
    for index, row in df_all.iterrows(): 
        if label_mask_rate == 1:
            examples.append([row.text, row.label, train_label_masks[index]])
        else:
            if train_label_masks[index]:
                if balance < 1:
                    balance = 1
                for b in range(0, int(balance)):
                    examples.append([row.text, row.label, train_label_masks[index]])
            else:
                examples.append([row.text, row.label, train_label_masks[index]])
    return pd.DataFrame(examples, columns = ['text', 'label', 'mask'])


if __name__ == '__main__':

    ### global var
    file_path = os.path.join(get_root_path(), "data\gan_bert_data")
    labeled_file = "labeled.tsv"
    unlabeled_file = "unlabeled.tsv"
    test_filename = "test.tsv"
    batch_size = 32
    num_epochs = 3
    schedule_type = SchedulerType.CONSTANT

    ### Load the examples
    df_labeled = pd.DataFrame(get_qc_examples(os.path.join(file_path, labeled_file)), columns=['text', 'label'])
    df_unlabeled = pd.DataFrame(get_qc_examples(os.path.join(file_path, unlabeled_file)), columns=['text', 'label'])
    df_test = pd.DataFrame(get_qc_examples(os.path.join(file_path, test_filename)), columns=['text', 'label'])
    # exclude labels that have not been seen in labeled examples
    label_space = df_labeled.label.unique().tolist() + ['UNK_UNK']
    df_test = df_test[df_test.label.apply(lambda x: x in label_space)]
    # piece together labeled and unlabeled data
    df_train = stich_train_data(df_labeled, df_unlabeled)
    # label conversion
    encoder_map = {each : i for i, each in enumerate(label_space)}
    df_train['label_new'] = df_train['label'].apply(lambda x: encoder_map[x])
    df_test['label_new'] = df_test['label'].apply(lambda x: encoder_map[x])
    
    ### create data loader
    generator = Generator().manual_seed(42)
    tokenizer = BertLoader(load_tokenizer=True).tokenizer
    ds_train = DatasetNLP(input_df=df_train, tokenizer=tokenizer, cols_to_tokenize=['text'], cols_label=['label_new'], bool_col=['mask'])
    ds_test = DatasetNLP(input_df=df_test, tokenizer=tokenizer,  cols_to_tokenize=['text'], cols_label=['label_new'] )
    train_dataloader = DataLoader(ds_train, sampler=RandomSampler(ds_train, generator=generator), batch_size=batch_size)
    valid_dataloader = DataLoader(ds_test, batch_size=batch_size)
    data_modeler = {
        DataLoaderType.TRAINING: train_dataloader, 
        DataLoaderType.VALIDATION: valid_dataloader}

    #### model / optimizer
    loader = BertClassifierLoader(ClassifierType.BERT_CLASSIFIER, "bert-base-uncased", num_labels=len(label_space), dropout=0.1)
    optimizer = AdamNLP.newNLPAdam(loader.model, {'embeddings':False, 'encoder': None})
    model = optimizer.get_model_transformed()
    # extra package for GAN
    gen_pckage = GANPackage(num_labels=len(label_space), gen_lr=1e-4, disc_lr=1e-4)

    ### TRainer
    trainer = TrainerDA(model, gen_pckage, data_modeler, optimizer)
    trainer.train(num_epochs, schedule_type = schedule_type)