from config import BaseConfig
from callbacks import *
from model import build_model
from model import train
from pipeline import load_cifar10
from pipeline import load_data

######################
##  TODO: argparse  ##
######################

class Config(BaseConfig):
  '''Inherent BaseConfig class from config.py'''
  lr = 1e-1
  epochs = 30
  callbacklist = [DecoupleHelper(max_lr=lr,
                                regularizer=BaseConfig.decay,
                                weight_decay=BaseConfig.decay_rate)]
  # callbacklist = [LRdecayHelper(init_lr=BaseConfig.lr)]

def main(args, callback:list):
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print('CIFAR10 dataset has been loaded')

<<<<<<< HEAD
    dataset = load_data(x_train, y_train, Config.batch_size, validation_split=0.1)
    print(f'{type(dataset)} is ready')
=======
    train_ds, val_ds = input_pipeline(x_train, y_train, Config.batch_size, augmentations=['left_right', 'contrast'], split_ratio=0.2)
    print(f'{type(train_ds)} is ready')
>>>>>>> a670c13ac64221b5feebbdc4e007053c6503c2a5

    model = build_model(args)
    train(model, args, (dataset), callback)


if __name__ == '__main__':
<<<<<<< HEAD
    main(Config, Config.callbacklist)
=======
    main(Config, [LRSchedule(Config.lr), Plotting()])
>>>>>>> a670c13ac64221b5feebbdc4e007053c6503c2a5
