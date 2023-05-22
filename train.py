import torch
import torch.nn.functional as F

import numpy as np

def caption2index(caption, word2ix):
    caption = caption.lower().split()
    return torch.tensor([word2ix[word] for word in caption])

def fit(img, cap, transform, optimizer, encoder, decoder, vocab_size, word2ix, ix2word, train=True):
    img = transform(img).to('cuda')
    cap = caption2index(cap, word2ix).to('cuda')

    feature_vec = encoder(img.unsqueeze(0))
    pred = decoder(feature_vec)

    losses = []
    predicted_cap = []
    for word in cap:
        pred = decoder(word.view(-1, 1))
        loss = F.cross_entropy(pred.view(-1, vocab_size), word.view(-1))  
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        else:
            pred_nxt = ix2word[pred.argmax().item()]
            predicted_cap.append(pred_nxt)

        losses.append(loss.item())
    
    if not train:
        return predicted_cap
    
    return np.array(losses).mean()
