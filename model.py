import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
'''
In the study researchers designed 3 stage model that 
shrinks feature map size at each stage i.e. {32, 16, 8}.

When modifying the model's architecture or hyperparemeters,
please look into conifg.py file.
'''
def resblock_no_bottleneck(X, n_filter, idx, n_block, init, decay):
  '''
  Args
    X                   (tf.Tensor): input data
    n_filter                 (list): output feature map channel of the layer
    idx                       (int): block number
    n_block                   (int): it decides how many times repeat 
                                     whole resblock
    init    (tf.keras.initializers): layer initializer. default=HeNormal
    decay   (tf.keras.regularizers): weight regularizer. default=L2
  Return
    tf.Tensor:  feature map (batch, H, W, C)
  '''
  name_base = f'{idx+1}_{n_block+1}'
  f1, f2 = n_filter
  x = layers.BatchNormalization(name=name_base+'_BN1')(X)
  x = layers.ReLU(name=name_base+'_ReLU1')(x)
  x = layers.Conv2D(filters=f1,
                    kernel_size=(3, 3),
                    strides=(1, 1) if n_block > 0 or idx == 0 else (2, 2),
                    padding='same',
                    kernel_initializer=init,
                    kernel_regularizer=decay,
                    name=name_base+'_Conv1')(x)

  x = layers.BatchNormalization(name=name_base+'_BN2')(x)
  x = layers.ReLU(name=name_base+'_ReLU2')(x)
  x = layers.Conv2D(filters=f2,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=init,
                    kernel_regularizer=decay,
                    name=name_base+'_Conv2')(x)

  if n_block > 0 or idx == 0:
    x = layers.Add(name=name_base+'_ID_Add')([x, X])
    print(f'{idx+1}-{n_block+1} ResBlock has been built')
    return x
  else:
    X = layers.Conv2D(filters=f2,
                      kernel_size=(1, 1),
                      strides = (2, 2),
                      padding='valid',
                      kernel_initializer=init,
                      kernel_regularizer=decay,
                      name=name_base+'_ID_Conv')(X)
    x = layers.Add(name=name_base+'_ID_Add')([x, X])
    print(f'{idx+1}-{n_block+1} ResBlock has been built')

    return x
  
def resnet_cifar10(X, args):
  '''
  Args
    X     (tf.keras.layers.Input): input layer
    args                  (class): configuring class
  Return
    tf.tensor:  class imbeding (batch, 10)
  '''
  # stem cell
  x = layers.Conv2D(filters=16,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=args.init,
                    kernel_regularizer=args.decay,
                    name='Stem_cell')(X)
  print('Stem cell has been built')

  # ResBlocks
  for idx, filters in enumerate(args.n_filters):
    for n_block in range(args.n_blocks):
      # x = resblock(x, filters, idx, n_block, args.init, args.decay)
      x = resblock_no_bottleneck(x, filters, idx, n_block, args.init, args.decay)

  # network head
  x = layers.GlobalAveragePooling2D(name='network_head_GAP')(x)
  x = layers.Dense(units=10,
                   activation='softmax',
                   name='network_head_Dense')(x)
  print('Network head has been built')
  
  return x

def build_model(args):
  '''Building a model.

  Args:
    args    (class): configuring class
  Return:
    tf.keras.model
  '''
  inputs = layers.Input(shape=args.input_shape, name='Input')
  x = resnet_cifar10(inputs, args)
  model = models.Model(inputs=inputs, outputs=x)
  print('Model building complite')
  return model

def train(model, args, data, callbacks:list):
  '''Initiating training

  Args:
    model     (tf.keras.model): model
    args               (class): configuring class
    data                 (set): set of tf.data.Dataset object containing 
                                training, validation data. np.adarray
                                also available
    callbacks           (list): a list of callback class
  Return:
    None  
  '''
  if model is None:
    raise ValueError('Model is not defined')
    
  model.compile(optimizer=args.optimizer,
                loss=args.loss,
                metrics=args.metric)
  x_train, y_train = data
  if type(x_train) == np.ndarray or type(x_train) == tf.Tensor:
    print(f'Training is initiated with {type(x_train)} object')
    model.fit(x_train,
              y_train,
              epochs=args.epochs,
              batch_size = args.batch_size,
              validation_split=0.2,
              callbacks=callbacks)
    print('Training terminated')
    
  else:
    print(f'Training is initiated with tf.data.Dataset objdect')
    model.fit(x=x_train,
              validation_data=y_train,
              epochs=args.epochs,
              callbacks=callbacks)
    print('Training terminated')
    
