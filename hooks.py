import torch
import torch.nn as nn

import matplotlib.pyplot as plt

encoder_act_means = {}
decoder_act_means = {}

def act_means(store):
    def hook(model, input, output):
        if len(output) > 1:
            # In the case of lstm, activations are given for every token (hidden state).
            # We can visualise every hidden activation but for now just store the final one.
            store[model].append(output[0].mean().cpu().item())
        else:
            store[model].append(output.mean().cpu().item())
    return hook

def plot_activations(store):
    for layer, activations in store.items():
        plt.plot(activations)
        plt.xlabel('example #')
        plt.ylabel('activation')
        plt.title(layer)
        plt.show()

def register_average_activations_hook(modules: list[nn.Module], store: dict):
    '''
        This is useful for seeing the layer activations but does not allow us to view dead units.
        To view dead neurons we need a hook that does not perform an average over the entire output.
    '''
    for m in modules:
        m.register_forward_hook(act_means(store))
        store[m] = []

