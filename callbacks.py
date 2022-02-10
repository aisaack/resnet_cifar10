from tensorflow.keras import callbacks
from tensorflow.keras import backend
import numpy as np

import matplotlib.pyplot as plt

class LRSchedule(callbacks.Callback):
  def __init__(self, lr):
    self.one_forth = None
    self.half = None
    self.three_forth = None
    self.curr_lr = lr

  def on_train_begin(self, logs=None):
    if self.model.optimizer.learning_rate is None:
      backend.set_value(self.model.optimizer.learning_rate, self.curr_lr)
    epoch = self.params.get('epochs')
    self.one_forth = int(epoch * 0.25)
    self.half = int(epoch * 0.5)
    self.three_forth = int(epoch * 0.75)
    self.curr_lr = backend.get_value(self.model.optimizer.learning_rate)

  def on_epoch_end(self, epoch, logs=None):
    if epoch == self.half or epoch == self.one_forth or epoch == self.three_forth:
      updated_lr = self.curr_lr * 0.5
      backend.set_value(self.model.optimizer.learning_rate, updated_lr)
      self.curr_lr = updated_lr
      print(f'Learning rate has been updated by {updated_lr}')
      
      
class Plotting(callbacks.Callback):
  def __init__(self, loss, metric, val_data):
    self.train_loss = []
    self.train_acc = []
    self.val_loss = []
    self.val_acc = []
    self.lr = []
    self.loss = loss
    self.metric = metric
    self.val_data = val_data
    self.data = list(val_data.as_numpy_iterator())

  def on_batch_end(self, batch, logs={}):
    self.train_loss.append(logs.get('loss'))
    self.train_acc.append(logs.get('sparse_categorical_accuracy'))

    y_pred = self.model.predict(self.data[batch][0])
    val_loss = self.loss(self.data[batch][1], y_pred).numpy()
    val_acc = self.metric.update_state(self.data[batch][1], y_pred)
    val_acc = self.metric.result().numpy()    
    self.val_loss.append(val_loss)
    self.val_acc.append(val_acc)
    logs['validation_loss'] = val_loss
    logs['validation_acc'] = val_acc
    self.lr.append(backend.get_value(self.model.optimizer.learning_rate))

  def on_train_end(self, logs=None):
    self.plot_loss()
    self.plot_acc()
    self.plot_lr()

  def plot_loss(self):
    x = np.arange(0, len(self.train_loss))
    plt.figure(figsize=(9, 5))
    plt.plot(x, self.train_loss, label='train_loss')
    plt.plot(x, self.val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

  def plot_acc(self):
    plt.figure(figsize=(9, 5))
    plt.plot(x, self.train_acc, label='train_acc')
    plt.plot(x, self.val_acc, label='val_acc')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

  def plot_lr(self):
    plt.figure(figsize=(9, 5))
    plt.plot(x, self.lr, label='learning rate')
    plt.xlabel('Iteration')
    plt.ylabel('LR')


class LRrangeTest(callbacks.Callback):
  def __init__(self, min_lr=1e-9, max_lr=1e+1, policy: str='linear'):
    self.min_lr = min_lr
    self.max_lr = max_lr
    self.policy = policy
    self.epochs = None
    self.steps = None
    self.num_iter = None
    self.iter_counter = 0
    self.learning_rates = None
    self.losses = []
    self.lowest_loss = None
    self.accuracies = []

  def on_train_begin(self, logs=None):
    self.epochs = self.params.get('epochs')
    self.steps = self.params.get('steps')
    self.num_iter = self.epochs * self.steps
    if self.policy == 'linear':
      self.learning_rates = np.linspace(start=self.min_lr, stop=self.max_lr, num=self.num_iter+1)
    elif self.policy == 'exponential':
      self.learning_rates = np.geomspace(start=self.min_lr, stop=self.max_lr, num=self.num_iter+1)
    print(f'Learning rates are initiated from {self.learning_rates[0]} to {self.learning_rates[-1]}')
    print(f'Learning rates will increase during {self.num_iter} iterations')
    if self.min_lr:
      backend.set_value(self.model.optimizer.learning_rate, self.min_lr)
    else:
      backend.set_value(self.model.optimizer.learning_rate, self.learning_rates[self.iter_counter])
   
  def on_batch_end(self, batch, logs=None):
    lr = backend.get_value(self.model.optimizer.learning_rate)
    logs['lr'] = lr

    loss = logs.get('loss')
    if self.iter_counter == 0 or loss < self.lowest_loss:
      self.lowest_loss = loss
      logs['best_loss'] = self.lowest_loss
    
    if loss > self.lowest_loss * 2:
      self.model.stop_training = True
    self.losses.append(loss)
    accuracy = logs.get('sparse_categorical_accuracy')
    self.accuracies.append(accuracy)

    self.iter_counter += 1
    next_lr = self.learning_rates[self.iter_counter]
    # print(f' - next_lr: {next_lr}')
    backend.set_value(self.model.optimizer.learning_rate, next_lr)

  def plot_loss(self):
    plt.figure(figsize=(12, 6))
    plt.plot(self.learning_rates[:len(self.losses)], self.losses)
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.xscale('log')
    plt.show()

  def plot_accuracy(self):
    plt.figure(figsize=(12, 6))
    plt.plot(self.learning_rates[:len(self.accuracies)], self.accuracies)
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.show()

  def on_train_end(self, logs=None):
    self.plot_loss()
    self.plot_accuracy()


class CLR(callbacks.Callback):
  '''
  This callback class allows learning rate to go back and forth
  between minimum learning rate and maximum learning rate so that
  a model can learn effectively.

  Args: 
    min_lr (float): minimum bound learning rate
    max_lr (float): maximum bound learning rate
    step_size (int): It decides length of step size. Typical step size is 2-10 times mini_batch
    gamma (float): 
    policy (string): ('triangular', 'triangular2', 'exp_range')
  '''
  def __init__(self, min_lr, max_lr, gamma=1, step_size:int=2, policy:str='triangular') -> None:
    self.min_lr = min_lr
    self.max_lr = max_lr
    self.gamma = gamma
    self.step_size = step_size
    self.policy = policy
    self.step = None
    self.iter_counter = 1
    self.log = []      

  def triangular2(self, cycle):
    return 2**(1-cycle)

  def exp_range(self, gamma, cycle):
    return gamma**(cycle)

  def clr(self, gamma:float)->float:
    cycle = np.floor(1 + self.iter_counter / (2 * self.step))
    x = np.abs(self.iter_counter / self.step - 2 * cycle + 1)
    if self.policy == 'triangular':
      return self.min_lr + (self.max_lr - self.min_lr) * np.maximum(0, (1 - x))
    elif self.policy == 'triangular2':
      return self.min_lr + (self.max_lr - self.min_lr) * np.maximum(0, (1 - x)) * self.triangular2(cycle)
    elif self.policy == 'exp_range':
      return self.min_lr + (self.max_lr - self.min_lr) * np.maximum(0, (1 - x)) * self.exp_range(self.gamma, cycle)
    else:
      raise ValueError("Undefined policy. Choose one of triangular, triangular2 and exp_range or Define a new one")   

  def on_train_begin(self, batch, logs=None):
    iteration = self.params.get('steps')
    self.step = self.step_size * iteration
    if self.min_lr:
      backend.set_value(self.model.optimizer.learning_rate, self.min_lr)

  def on_batch_end(self, batch, logs=None):
    self.iter_counter += 1
    lr = self.clr(self.gamma)
    self.log.append(lr)
    backend.set_value(self.model.optimizer.learning_rate, lr)

  def on_train_end(self, logs=None):
    plt.figure(figsize=(12, 6))
    plt.plot(self.log)
    plt.xlabel('Iteration')
    plt.ylabel('LR')