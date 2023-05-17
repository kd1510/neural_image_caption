import torch
import pickle

from torchvision.models import efficientnet_b0


def test_efficientnet_on_single_image(img):
    with open('imagenet_labels.pkl', 'rb') as f:
        label_dict = pickle.load(f)

    convnet = efficientnet_b0(pretrained=True)
    convnet.eval()

    with torch.no_grad():
        for i in convnet(img.unsqueeze(0)).topk(10).indices[0]:
            print(label_dict[i.item()])