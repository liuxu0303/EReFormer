import os
from os.path import join
import numpy as np
import json
import logging
import argparse
import tqdm
import torch
from model.model import *
from model.loss import *
# from model.metric import *
from torch.utils.data import DataLoader, ConcatDataset
# from data_loader.dataset import *
from data_loader.dataset_mvsec import SequenceMVSEC
from data_loader.data_loaders import HDF5DataLoader
from trainer.transformer_trainer import TransformerTrainer
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from torchsummary import summary

logging.basicConfig(level=logging.INFO, format='')


def main(config, resume, initial_checkpoint=None):
    train_logger = None

    L = config['trainer']['sequence_length']
    assert (L > 0)

    dataset_type, base_folder = {}, {}
    
    step_size = {}
    
    clip_distance = {}
    


    use_phased_arch = config['use_phased_arch']

    for split in ['train', 'validation']:
        dataset_type[split] = config['data_loader'][split]['type']
        base_folder[split] = join(config['data_loader'][split]['base_folder'])
        
        try:
            step_size[split] = config['data_loader'][split]['step_size']
        except KeyError:
            step_size[split] = 1

        try:
            clip_distance[split] = config['data_loader'][split]['clip_distance']
        except KeyError:
            clip_distance[split] = 100.0

        
    np.random.seed(0)   

    # loss_composition = config['trainer']['loss_composition']
    # loss_weights = config['trainer']['loss_weights']
    normalize = config['data_loader'].get('normalize', True)
    
    data_loader = eval(dataset_type['train'])(data_file=base_folder['train'],
                                              batch_size=config['data_loader']['batch_size'],
                                              shuffle=config['data_loader']['shuffle'],
                                              num_workers=config['data_loader']['num_workers'],
                                              pin_memory=config['data_loader']['pin_memory'],
                                              sequence_kwargs={'sequence_length': L,
                                                                # 'transform': Compose([RandomRotationFlip(0.0, 0.5, 0.0)]),
                                                                'transform': Compose([RandomRotationFlip(0.0, 0.5, 0.0),
                                                                      RandomCrop(size=(224, 224))]),
                                                               'step_size': step_size['train'],
                                                               'clip_distance': clip_distance['train'],
                                                               'normalize': normalize})
    
    valid_data_loader = eval(dataset_type['validation'])(data_file=base_folder['validation'],
                                              batch_size=config['data_loader']['batch_size'],
                                              shuffle=False,
                                              num_workers=config['data_loader']['num_workers'],
                                              pin_memory=config['data_loader']['pin_memory'],
                                              sequence_kwargs={'sequence_length': L, 
                                                               'transform': None, 
                                                               'step_size': step_size['validation'],
                                                               'clip_distance': clip_distance['validation'],
                                                               'normalize': normalize})


    config['model']['gpu'] = config['gpu']
    

    torch.manual_seed(0)
    model = eval(config['arch'])(config['model'])
    # Initialize weight randomly
    # model.init_weights()

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        # model.load_state_dict(checkpoint['state_dict'])
        if use_phased_arch:
            C, (H, W) = config["model"]["num_bins_events"], config["model"]["spatial_resolution"]
            dummy_input = torch.Tensor(1, C, H, W)
            times = torch.Tensor(1)
            _ = model.forward(dummy_input, times=times, prev_states=None)  # tag="events"
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.init_weights()

    '''print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())'''

    # model.summary()
    summary(model, input_size=[(3, 224, 224)], batch_size=1, device='cpu')

    loss = eval(config['loss']['type'])
    loss_params = config['loss']['config'] if 'config' in config['loss'] else None
    print("Using %s with config %s" % (config['loss']['type'], config['loss']['config']))
    # metrics = [eval(metric) for metric in config['metrics']]

    trainer = TransformerTrainer(model, loss, loss_params,
                                 resume=resume,
                                 config=config,
                                 data_loader=data_loader,
                                 valid_data_loader=valid_data_loader,
                                 train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Monocular Depth Prediction')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-i', '--initial_checkpoint', \
        default=None, \
        type=str, help='path to the checkpoint with which to initialize the model weights (default: None)')
    parser.add_argument('-g', '--gpu_id', default=None, type=int,
                        help='path to the checkpoint with which to initialize the model weights (default: None)')
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        if args.initial_checkpoint is not None:
            logger.warning(
                'Warning: --initial_checkpoint overriden by --resume')
        config = torch.load(args.resume)['config']
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if args.resume is None:
            assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    main(config, args.resume, args.initial_checkpoint)
