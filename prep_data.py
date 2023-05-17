import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def data_container():
    df = pd.read_csv('resources/flikr_8k/Flickr8k_text/Flickr8k.lemma.token.txt', sep='\t', header=None)
    caption_col = df[1]

    # Contruct vocabulary
    words = caption_col.str.split().explode()
    words = words.str.lower().unique()
    V = len(words)

    word2ix = {word: i for i, word in enumerate(words)}
    ix2word = {i: word for i, word in enumerate(words)}

    return df, V, word2ix, ix2word