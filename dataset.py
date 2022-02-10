from tensorflow.keras.datasets import cifar10
from numpy.random import RandomState

def split_validation(data:set, ratio:float=0.2):
  '''Spliting dataset into 2 parts i.e. train and validation set.

  Args
    data        (set): a set of data and label.
    ratio     (float): floating point between 0 and 1.
                       This argument decides ratio of validation set
                       default=0.2
  Return
    a tensor set of 4 i.e. train_data, train_label, validation_data, validation_label 
  '''
  assert 0 < ratio < 1
  train, label = data
  RandomState(234).shuffle(train)
  RandomState(234).shuffle(label)
  len_val = int(len(train) * (1-ratio))
  train_data = train[:len_val]
  val_data = train[len_val:]
  train_label = label[:len_val]
  val_label = label[len_val:]
  print('Splitting dataset is complete')
  return train_data, train_label, val_data,  val_label

def load_cifar10():
  '''Downloading cifar10 dataset from
     tf.keras.datasets
  '''
  return cifar10.load_data()