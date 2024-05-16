import os
import json
import platform
import torch

def get_device():
    if platform.system().upper() == "DARWIN":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif platform.system().upper() == "WINDOWS":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def get_root_path():
    if platform.system().upper() == "DARWIN":
        return "/Users/lunli/Library/CloudStorage/GoogleDrive-yaojn19880525@gmail.com/My Drive/Colab Notebooks/"
    elif platform.system().upper() == "WINDOWS":
        return "G:/My Drive/Colab Notebooks/"
