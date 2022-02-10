from config import BaseConfig
from callbacks import LRSchedule
from callbacks import Plotting
from model import build_model
from model import train
from pipeline import input_pipeline
from dataset import load_cifar10
from dataset import split_validation

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

    train_img, train_label, val_img, val_label = split_validation((x_train, y_train), args.train_validation_ratio)
    train_data = input_pipeline(train_img, train_label, args, 'train')
    val_data = input_pipeline(val_img, val_label, args, 'val')

    model = build_model(args)
    train(model, args, (train_data, val_data), callback)


if __name__ == '__main__':
    main(Config, [LRSchedule(Config.lr), Plotting()])