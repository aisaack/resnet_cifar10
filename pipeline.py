import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def load_cifar10():
  '''Downloading cifar10 dataset from
     tf.keras.datasets
  '''
  return cifar10.load_data()

def load_data(feature, label,  batch_size, validation_split:float=0.2, pad:int=4):
  '''Build input pipeline
  
  Args
    feature          (np.ndarray): x_train data
    label            (np.ndarray): y_train data
    batch_size              (int): batch_size
    validation_split      (float): ratio of validation data between 0 and 1
                                   if it is 0 data splitting is not applied
    pad                      (int): size of 0 pad at each side

  Return
    tf.data.Dataset object
  '''

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