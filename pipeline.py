import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def load_cifar10():
  '''Downloading cifar10 dataset from
     tf.keras.datasets
  '''
  return cifar10.load_data()

def input_pipeline(feature, label,  batch_size, augmentations:list=None, split_ratio=0.2):
  '''Build input pipeline
  
  Args
    feature     (np.ndarray): x_train data
    label       (np.ndarray): y_train data
    augmentations     (list): list of augmentation methods
                              {"up_down", "left_right", "contrast", "bright"}
    split_ratio      (float): ratio of validation data between 0 and 1
                              if it is 0 data splitting is not applied

  Return
    tf.data.Dataset object
  '''
  seed = 14234
  ds = tf.data.Dataset.from_tensor_slices((feature, label))
  
  # nomalization
  ds = ds.map(lambda x, y: (tf.cast(x, tf.float32)/255., y))

  # standardization
  ds = ds.map(lambda x, y: (x - tf.math.reduce_mean(x) / tf.math.reduce_std(x), y))
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.cache()
  
  assert 0 <= val_ratio < 1
  if split_ratio:
    data_len = len(x_train)
    train_len = int(data_len * (1 - val_ratio))
    train_ds = ds.take(train_len)
    val_ds = ds.skip(train_len)
    print('Splitting is complete')
  else:
    train_ds = ds

  if augmentations:
    for augment in augmentation:
      x = train_ds.take(int(train_len * 0.7))
      x = x.shuffle(buffer_size=1000)
      if augment == 'random_crop':
        x = x.map(lambda x, y:(tf.image.resize(x, (40, 40)), y))
        x = x.map(lambda x, y:(tf.image.random_crop(x, size=(32, 32, 3), seed=seed), y))
      elif augment == 'flip_left_right':
        x = x.map(lambda x, y:(tf.image.random_flip_left_right(x, seed=seed), y))
      elif augment == 'flip_up_down':
        x = x.map(lambda x, y:(tf.image.random_flip_up_down(x, seed=seed), y))
      else:
        raise ValueError()
      train_ds = train_ds.concatenate(x)
      train_ds = train_ds.shuffle(buffer_size=1000)
      train_ds = train_ds.cache()
    train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)
    print('Augmentation is complete')

  if split_ratio:
    mini_batch = train_ds.cardinality().numpy()
    val_len = int(data_len - train_len)
    val_ds = val_ds.batch(batch_size=int(val_len / mini_batch), drop_remainder=False)
    val_ds = val_ds.prefetch(buffer_size=1)
    train_ds = train_ds.prefetch(buffer_size=1)
    return train_ds, val_ds
  train_ds = train_ds.prefetch(buffer_size=1)
  return train_ds
