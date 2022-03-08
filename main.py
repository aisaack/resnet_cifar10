from config import BaseConfig
from callbacks import LRSchedule
from callbacks import Plotting
from model import build_model
from model import train
from pipeline import load_cifar10
from pipeline import load_data

######################
##  TODO: argparse  ##
######################

class Config(BaseConfig):
  '''Inherent BaseConfig class from config.py'''
  # epochs = 40
  # batch_size = 64
  # lr = 0.002

def main(args, callback:list):
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print('CIFAR10 dataset has been loaded')

    train_ds, val_ds = input_pipeline(x_train, y_train, Config.batch_size, augmentations=['left_right', 'contrast'], split_ratio=0.2)
    print(f'{type(train_ds)} is ready')

    model = build_model(args)
    train(model, args, (train_ds, val_ds), callback)


if __name__ == '__main__':
    main(Config, [LRSchedule(Config.lr), Plotting()])
