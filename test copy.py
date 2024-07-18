import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

MY_DEVICE = torch.device('mps')

if __name__ == '__main__':

    path = '/Users/lunli/Dropbox/Amex Project/'
    df_train = pd.read_csv(path+'intent_train.csv',index_col=0)
    text = df_train['query'].values.tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2").to(MY_DEVICE)
    v = model.encode(text)
