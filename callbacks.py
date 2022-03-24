from tensorflow.keras import callbacks
from tensorflow.keras import backend
from tensorflow.keras import optimizers

class LRdecayHelper(callbacks.Callback):
  '''LRdecayHelper implements LR decay at 30, 45, and 55 epoch.

  Args:
    init_lr     (float): Initial learning rate
    decay_rate  (float): Decay rate
  '''
  def __init__(self, init_lr, decay_rate=0.1):
    self.init_lr = init_lr
    self.decay_rate = decay_rate
    self.best_weight = None
    self.best_acc = None
    self.best_epoch = 0
  
  def decay(self, decay_rate):
    curr_lr = backend.get_value(self.model.optimizer.lr)
    next_lr = curr_lr * decay_rate
    backend.set_value(self.model.optimizer.lr, next_lr)
    print(f'\n LR is updated{next_lr}')

  def on_train_begin(self, logs=None):
    if not hasattr(self.model.optimizer.lr, 'lr'):
      try:
        backend.set_value(self.model.optimizer.lr, self.init_lr)
      except:
        raise ValueError('LR is missing')
    
  def on_epoch_end(self, epoch, logs=None):    
    if epoch == 29:
      self.decay(self.decay_rate)
    elif epoch == 44:
      self.decay(self.decay_rate)
    elif epoch == 54:
      self.decay(self.decay_rate)
    
    current = logs.get('val_sparse_categorical_accuracy')
    if self.best_acc is None:
      self.best_acc = current
    if self.best_acc < current:
      self.best_acc = current
      self.best_weight = self.model.get_weights()
      self.best_epoch = epoch
      print('\n', 'Best acc updated') 

  def on_train_end(self, logs=None):
    self.model.set_weights(self.best_weight)
    print(f'Best accuracy is {self.best_acc} at {self.best_epoch}th epoch')


class DecoupleHelper(callbacks.Callback):
  '''This helper implements Cosine Annealing LR decay.
     Once regularizer and weight_decay is given,
     the optimizer decouples L2 regularizer.

  Args:
    max_lr                    (float):
    regularizer (tf.keras.optimizers): 
    weight_decay              (float):
    decay_step                  (int):
  '''
  def __init__(self, max_lr, regularizer=None, weight_decay=None, decay_step=2):
    self.regularizer = regularizer
    self.weight_decay = weight_decay
    self.max_lr = max_lr
    self.decay_step = decay_step
    self.mini_batch = None
    self.epochs = None
    self.iterations = None
    self.iter_counter = 0
    self.best_acc = None
    self.best_weights = None
    self.best_epoch = 0

  def on_train_begin(self, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      try: 
        backend.set_value(self.model.optimizer.lr, self.max_lr)
      except:
        raise ValueError('Learning rate is missing')
    self.mini_batch = self.params.get('steps')
    self.epochs = self.params.get('epochs')
    self.iterations = self.mini_batch * self.epochs
    print(f'Initial LR: {self.max_lr}          Initial WD: {self.weight_decay}')

  def on_epoch_begin(self, batch, logs=None):
    self.iter_counter += 1

  def on_epoch_end(self, epoch, logs=None):
    new_lr = optimizers.schedules.CosineDecayRestarts(initial_learning_rate=self.max_lr,
                                                      first_decay_steps=self.decay_step)(self.iter_counter).numpy()
    backend.set_value(self.model.optimizer.lr, new_lr)
    logs['lr '] = new_lr
    if self.weight_decay and self.regularizer:
      new_wd = optimizers.schedules.CosineDecayRestarts(initial_learning_rate=self.weight_decay,
                                                        first_decay_steps=self.decay_step)(self.iter_counter).numpy()
      self.regularizer.l2 = new_wd
      logs['wd'] = new_wd
      
    curr_acc = logs.get('val_sparse_categorical_accuracy')
    if self.best_acc is None:
      self.best_acc = curr_acc
    if self.best_acc < curr_acc:
      self.best_acc = curr_acc
      self.best_weights = self.model.get_weights()
      self.best_epoch = epoch
      logs['Weight Updated'] = True

  def on_train_end(self, logs=None):
    if self.best_weights:
      self.model.set_weights(self.best_weights)
    print(f'Training terminated with best accuracy {self.best_acc} at epoch {self.best_epoch}')