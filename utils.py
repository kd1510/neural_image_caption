import torch
from PIL import Image


def get_image_and_caption(raw_df, ix):
    image_path = 'resources/flikr_8k/Flickr8k_Dataset/Flicker8k_Dataset/'
    img_col = raw_df[0]
    caption_col = raw_df[1]
    
    caption = caption_col.iloc[ix]
    image = Image.open(image_path + img_col.iloc[ix].split('#')[0])
    return image, caption 
