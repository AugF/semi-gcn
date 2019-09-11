# https://segmentfault.com/a/1190000014285048

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers

# keras vs tf.keras: keras本身就是已经成熟的接口,不过tf进行了进一步地集成
# https://datascience.stackexchange.com/questions/47759/keras-vs-tf-keras

from keras.datasets import imdb
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)


print("Categories:", np.unique(targets))
print("Number of unique words:", len(np.unique(np.hstack(data))))

# Categories: [0 1]
# Number of unique words: 9998

length = [len(i) for i in data]
print("Average Review length:", np.mean(length))
print("Standard Deviation:", round(np.std(length)))

# Average Review length: 234.75892
# Standard Deviation: 173.0