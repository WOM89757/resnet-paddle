
from train import *
from utils import *


if __name__ == '__main__':
    logger = init_log_config()
    init_train_parameters()
    train(logger)