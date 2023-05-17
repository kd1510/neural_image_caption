import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from encoder_decoder import Encoder, Decoder
from prep_data import data_container
from utils import get_image_and_caption


df, V, word2ix, ix2word = data_container()

enc = Encoder(100)
dec = Decoder(V, 100, 20)

enc.to('cuda')
dec.to('cuda')

optimizer = torch.optim.SGD(list(enc.params) + list(dec.parameters()), lr=0.01)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def caption2index(caption):
    caption = caption.lower().split()
    return torch.tensor([word2ix[word] for word in caption])

def fit_example(img, cap, train=True, log=False):
    img = transform(img).to('cuda')
    cap = caption2index(cap).to('cuda')

    feature_vec = enc(img.unsqueeze(0))
    pred = dec(feature_vec)

    for word in cap:
        pred = dec(word.view(-1, 1))
        loss = F.cross_entropy(pred.view(-1, V), word.view(-1))
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if log:
            return ix2word[word.item()], ix2word[pred.argmax().item()], loss.item()


if __name__ == '__main__':

    train_x = [get_image_and_caption(df, ix) for ix in range(0, 500)]
    val_x = [get_image_and_caption(df, ix) for ix in range(500, 600)]

    training_loss = []
    val_loss = []

    epochs = 10
    for epoch in range(epochs):
        epoch_losses = []

        for img, cap in train_x:
            _, _, loss = fit_example(img, cap, log=True)
            epoch_losses.append(loss)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        training_loss.append(avg_loss)

        with torch.no_grad():
            val_losses = []
            for img, cap in val_x:
                _, _, loss = fit_example(img, cap, train=False, log=True)
                val_losses.append(loss)
            avg_val_loss = sum(val_losses) / len(val_losses)
            val_loss.append(avg_val_loss)


plt.plot(training_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

plt.plot(val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.show()
