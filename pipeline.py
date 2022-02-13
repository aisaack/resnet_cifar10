import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def load_cifar10():
  '''Downloading cifar10 dataset from
     tf.keras.datasets
  '''
  return cifar10.load_data()

def load_data(feature, label,  batch_size, augmentations:list=None, split_ratio=0.2):
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

  assert 0 <= split_ratio < 1
  ds = tf.data.Dataset.from_tensor_slices((feature, label))
  # normalize
  ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))

  # standardize
  ds = ds.map(lambda x, y: ((x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x), y))
  ds = ds.shuffle(buffer_size=1000)

  # split trian validation
  if split_ratio:
    feature_len = len(feature)
    train_len = int(feature_len * (1 - split_ratio))
    train_ds = ds.take(train_len)
    val_ds = ds.skip(train_len)

    mini_batch = int(feature_len / batch_size)
    val_len = int(feature_len - train_len)
    val_ds = val_ds.batch(batch_size=int(val_len / mini_batch), drop_remainder=True)
    val_ds = val_ds.prefetch(buffer_size=1)
  trian_ds = ds

  # augmentation
  if augmentations:
    aug = [train_ds]
    for augment in augmentations:
      if augment == 'up_down':
        x = train_ds.map(lambda x, y: (tf.image.flip_up_down(x), y))
      elif augment == 'left_right':
        x = train_ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
      elif augment == 'contrast':
        x = train_ds.map(lambda x, y: (tf.image.random_contrast(x, 0.1, 0.3), y))
      elif augment == 'bright':
        x = train_ds.map(lambda x, y: (tf.image.random_brightness(x, 0.1), y))
      else:
        raise ValueError(f'{augment} is not defined. Use "up_down, left_right, contrast, bright"')
      aug.append(x)
    train_ds = tf.data.Dataset.sample_from_datasets(aug, weights=[1/len(aug)] * len(aug))

  train_ds = train_ds.shuffle(buffer_size=1000)
  train_ds = train_ds.cache()
  train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)
  train_ds = train_ds.prefetch(buffer_size=1)
  if split_ratio:
    return train_ds, val_ds
  return train_ds