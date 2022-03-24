import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def load_cifar10():
  '''Downloading cifar10 dataset from
     tf.keras.datasets
  '''
  return cifar10.load_data()

<<<<<<< HEAD
def load_data(feature, label,  batch_size, validation_split:float=0.2, pad:int=4):
  '''Build input pipeline
  
  Args
    feature          (np.ndarray): x_train data
    label            (np.ndarray): y_train data
    batch_size              (int): batch_size
    validation_split      (float): ratio of validation data between 0 and 1
                                   if it is 0 data splitting is not applied
    pad                      (int): size of 0 pad at each side
=======
def input_pipeline(feature, label,  batch_size, augmentations:list=None, split_ratio=0.2):
  '''Build input pipeline
  
  Args
    feature     (np.ndarray): x_train data
    label       (np.ndarray): y_train data
    augmentations     (list): list of augmentation methods
                              {"random_crop", "flip_up_down", "flip_left_right"}
    split_ratio      (float): ratio of validation data between 0 and 1
                              if it is 0 data splitting is not applied
>>>>>>> a670c13ac64221b5feebbdc4e007053c6503c2a5

  Return
    tf.data.Dataset object
  '''
<<<<<<< HEAD

  assert 0 <= validation_split < 1
  ds = tf.data.Dataset.from_tensor_slices((feature, label))
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
  ds = ds.map(lambda x, y: (tf.divide(x, 255.), y))
  ds = ds.map(lambda x, y:(tf.divide(tf.subtract(x, tf.math.reduce_mean(x)), tf.math.reduce_std(x)), y))
  ds = ds.cache()
  N, H, W, C = feature.shape

  # Splitting validation dataset from train dataset
  if validation_split:
    n_val = int(N * validation_split)
    n_train = int(N - n_val)
    train_ds = ds.take(n_train)
    val_ds = ds.skip(n_train)
  

  # Data augmentation
  # Padding 4 pixels each side
  crop_ds = train_ds.map(lambda x, y: (tf.image.resize_with_crop_or_pad(x, H+pad*2, W+pad*2), y))

  # Random crop
  crop_ds = crop_ds.map(lambda x, y: (tf.image.random_crop(x, (H, W, C)), y))

  # Horizonal flip from the result of random crop
  hfilp_ds = crop_ds.map(lambda x, y: (tf.image.random_flip_up_down(x), y))

  # Putting these together
  train_ds = train_ds.concatenate(crop_ds)
  train_ds = train_ds.concatenate(hfilp_ds)

  train_ds = train_ds.shuffle(buffer_size=1000)
  train_ds = train_ds.cache()

  train_ds = train_ds.batch(batch_size=batch_size, drop_remainder=True)
  train_iter = train_ds.cardinality().numpy()
  val_ds = val_ds.batch(batch_size=batch_size, drop_remainder=True)
  val_iter = val_ds.cardinality().numpy()
  train_ds.prefetch(1)
  val_ds.prefetch(1)
  print(f'train iteration: {train_iter}, val iteration: {val_iter}')
  return train_ds, val_ds
=======
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
>>>>>>> a670c13ac64221b5feebbdc4e007053c6503c2a5
