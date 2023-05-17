
- We can use a CNN to encode an image into a vector. 
- This state is then used as the initial hidden state of an RNN to generate text description from the vector.


general method steps:
    1. Train a CNN to perform well on a classification task
    2. remove classification layer, use the final hidden layer as input to the RNN decoder.

- embeddings
    - 512 dimensional
    - randomly initialized weights (paper says pretraining embeddings didn't help)

- LSTM for sentence generator
    - special stop and start words included for each training example
    - loss = sum of nll of correct word at each step.

- Try EfficientNet-B0 for the CNN module
    - image is only input once, providing at every time step leads to worse results and can overfit to noise in the image.
    - use pretrained frozen weights from imagenet
