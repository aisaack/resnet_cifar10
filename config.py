from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

class BaseConfig:
  '''DO NOT MODIFY THIS FILE'''
  epochs = 60
  batch_size = 128
  lr = 0.005
  decay_rate = 1e-4
  momentum = 0.9
  input_shape = (32, 32, 3)
  n_filters = [[16, 16], [32, 32], [64, 64]]
  n_blocks = 6
  val_ratio = 0.2
  optimizer = optimizers.SGD(learning_rate=lr, momentum=momentum)
  loss = losses.SparseCategoricalCrossentropy()
  metric = metrics.SparseCategoricalAccuracy()
  init = initializers.HeNormal()
  decay = regularizers.L2(decay_rate)