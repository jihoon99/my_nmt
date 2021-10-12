import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn
import torch_optimizer as custom_optim

from data_loader import DataLoader
import data_loader as data_loader

from trainer import SingleTrainer
from trainer import MaximumLikelihoodEstimationEngine

from model.transformer import Transformer


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help = 'Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help = 'Model file name to save.'
    )
    p.add_argument(
        '--train',
        required=not is_continue,
        help = 'Training set file name(path) except the extention.'
    )
    p.add_argument(
        '--valid',
        required=not is_continue,
        help = 'Validation set file name(path) except the extention.'
    )
    p.add_argument(
        '--lang',
        required=not is_continue,
        help = 'extentions or translation (enko) means english to korean'
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=20,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )
    p.add_argument(
        '--lr',
        type=float,
        default=1.,
        help='Initial learning rate. Default=%(default)s',
    )

    p.add_argument(
        '--lr_step',
        type=int,
        default=1,
        help='Number of epochs for each learning rate decay. Default=%(default)s',
    )
    p.add_argument(
        '--lr_gamma',
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_decay_start',
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s',
    )

    p.add_argument(
        '--use_adam',
        action='store_true',
        help='Use Adam as optimizer instead of SGD. Other lr arguments should be changed.',
    )
    p.add_argument(
        '--use_radam',
        action='store_true',
        help='Use rectified Adam as optimizer. Other lr arguments should be changed.',
    )
    p.add_argument(
        '--use_transformer',
        action='store_true',
        help='Set model architecture as Transformer.',
    )
    p.add_argument(
        '--n_splits',
        type=int,
        default=8,
        help='Number of heads in multi-head attention in Transformer. Default=%(default)s',
    )

    config = p.parse_args()

    return config

def get_model(input_size, output_size, config):
    if config.use_transformer:
        model = Transformer(input_size = input_size,
                            hidden_size = config.hidden_size,
                            output_size = output_size,
                            n_splits = config.n_splits,
                            num_enc_layer = config.n_layers,
                            num_dec_layer = config.n_layers,
                            dropout_p = config.dropout)

    return model

def get_crit(output_size, pad_index):
    tmp = torch.ones(output_size)
    tmp[pad_index] = 0

    return nn.NLLLoss(
            weight=tmp,
            reduction = 'sum'        
            )

def get_optim(model, config):
    if config.use_adam:
        if config.use_transformer:
            optimizer = optim.Adam(model.parameters(), lr = config.lr, betas=(.9,.98))

    elif config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr = config.lr)
    
    else:
        optimizer = optim.SGD(model.parameters(), lr = config.lr)
    
    return optimizer

def get_scheduler(optimizer, config):
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i for i in range(
                max(0, config.lr_decay_start -1),
                (config.init_epoch -1) + config.n_epochs,
                config.lr_step,
            )],
            gamma = config.lr_gamma,
            last_epoch = config.init_epoch - 1 if config.init_epoch > 1 else -1,
        )
    else:
        lr_scheduler = None
    return lr_scheduler


def main(config, model_weight=None, opt_weight=None):

    def print_config(config):
        pp = pprint.PrettyPrinter(indent = 4)
        pp.pprint(vars(config))
    
    print_config(config)

    # 1. build data_loader.py
    loader = DataLoader(
                    train_fn=config.train,
                    valid_fn=config.valid,
                    exts=(config.lang[:2], config.lang[-2:]),
                    batch_size = config.batch_size,
                    device = -1,
                    max_length = config.max_length,
                    )

    input_size, output_size = len(loader.src_field.vocab), len(loader.tgt_field.vocab)

    
    # 2. load model, criterion, optimizer
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, data_loader.PAD)

    if model_weight is not None:
        model.load_state_dict(model_weight)
    
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)    

    optimizer = get_optim(model, config)
    
    if opt_weight is not None and (config.use_adma or config.use_radma):
        optimizer.load_state_dict(opt_weight)
    lr_scheduler = get_scheduler(optim, config)

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)


    # 3. training start
    MLE_trainer = SingleTrainer(target_engine_class=MaximumLikelihoodEstimationEngine, config = config)
    MLE_trainer.train(
        model = model, 
        crit = crit, 
        optimizer = optimizer,
        train_loader = loader.train_iter,
        valid_loader = loader.valid_iter,
        src_vocab = loader.src_field.vocab,
        tgt_vocab = loader.tgt_field.vocab,
        n_epochs = config.n_epochs,
        lr_scheduler=lr_scheduler,
    )
    print('train_done')



if __name__ == '__main__':
    config = define_argparser()
    main(config)
