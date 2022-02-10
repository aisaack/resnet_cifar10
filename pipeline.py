import tensorflow as tf

def input_pipeline(train, label, args, dataset:str='train'):
  '''Concatenating both np.ndarray type data and change 
     into tf.data.Dataset object. Data augmentation can be added
     in the pipeline.

  Args
    train         (np.ndarray): image tensor (N, H, W, C)
    label         (np.adarray): lable vector (N, 1)
    args               (class): configuring class
    dataset              (str): flag describing incoming datasets are
                                train or validation dataset
  Return
    tf.data.Dataset  
  '''
  train_dataset = tf.data.Dataset.from_tensor_slices(train)
  train_dataset = train_dataset.map(lambda x: tf.cast(x, tf.float32) / 255)
  train_dataset = train_dataset.map(lambda x: (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x))

  label_dataset = tf.data.Dataset.from_tensor_slices(label)
  
  out = tf.data.Dataset.zip((train_dataset, label_dataset))
  out = out.shuffle(len(train))
  out = out.cache()
  if dataset == 'train':
    out = out.batch(batch_size=args.batch_size, drop_remainder=True)
  elif dataset == 'val':
    origin_len = int(len(train) * 100 * args.val_ratio)
    train_len = int(origin_len * (1 - args.val_ratio))
    mini_batch = int(train_len / args.batch_size)
    out = out.batch(batch_size=int(len(train) / mini_batch), drop_remainder=True)
  out = out.batch(batch_size=args.batch_size, drop_remainder=True)
  out = out.prefetch(tf.data.AUTOTUNE)
  return out