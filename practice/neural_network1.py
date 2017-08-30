'''
Neural Network Implementation:
1. Define Neural Network architecture
2. Transfer data to your model
3. Under the hood, the data is first divided into batches, so that it can be ingested. The batches are first preprocessed, augmented and then fed into Neural Network for training.
4. The model then gets trained incrementally.
5. Display the accuracy for a specific number of timesteps.
6. After training save the model for prediction.
7. Test the model on new data and find out the accuracy.
'''
import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf

# To stop potential randomness we use seed as starting point
seed = 128
rng = np.random.RandomState(seed)
