import os
import json
import logging
import argparse
import torch
from model.model import *
from model.loss import *
from metric import *
from torch.utils.data import DataLoader, ConcatDataset
from data_loader.dataset_mvsec import SequenceMVSEC
from data_loader.data_loaders import HDF5DataLoader
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from os.path import join
import numpy as np
# import time

logging.basicConfig(level=logging.INFO, format='')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval_metrics(output, target):
    metrics = [mse, abs_rel_diff, scale_invariant_error, median_error, mean_error, rms_linear]
    acc_metrics = np.zeros(len(metrics))
    output = output.cpu().data.numpy()
    target = target.cpu().data.numpy()
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics

def make_colormap(img, color_mapper):
    #color_map = np.nan_to_num(img[0])
    #print("max min color map: ", np.max(color_map), np.min(color_map))
    img = np.nan_to_num(img, nan=1)
    color_map_inv = np.ones_like(img[0]) * np.amax(img[0]) - img[0]
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    color_map_inv = color_mapper.to_rgba(color_map_inv)
    color_map_inv[:, :, 0:3] = color_map_inv[:, :, 0:3][..., ::-1]
    return color_map_inv

def main(config, initial_checkpoint):
    train_logger = None
    
    L = 1

    dataset_type, base_folder = {}, {}
    
    clip_distance = {}
    

    # this will raise an exception is the env variable is not set
    # preprocessed_datasets_folder = os.environ['PREPROCESSED_DATASETS_FOLDER']

    use_phased_arch = config['use_phased_arch']

    for split in ['train', 'validation']:
        dataset_type[split] = config['data_loader'][split]['type']
        base_folder[split] = join(config['data_loader'][split]['base_folder'])

        try:
            clip_distance[split] = config['data_loader'][split]['clip_distance']
        except KeyError:
            clip_distance[split] = 100.0

    normalize = config['data_loader'].get('normalize', True)

    test_dataset = SequenceMVSEC(base_folder='../ETMD-net/DENSE/test_sequence_00_town10',
                                 sequence_length=L,
                                 transform=None,
                                 step_size=L,
                                 clip_distance=clip_distance['validation'],
                                 normalize=normalize,
                                 )                            

    config['model']['gpu'] = config['gpu']

    model = eval(config['arch'])(config['model'])

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        if use_phased_arch:
            C, (H, W) = config["model"]["num_bins_events"], config["model"]["spatial_resolution"]
            dummy_input = torch.Tensor(1, C, H, W)
            times = torch.Tensor(1)
            _ = model.forward(dummy_input, times=times, prev_states=None)
        model.load_state_dict(checkpoint['state_dict'])

    # model.summary()

    gpu = torch.device('cuda:' + str(config['gpu']))
    model.to(gpu)

    model.eval()

    N = len(test_dataset)
    
    print('N: '+ str(N))


    with torch.no_grad():
        idx = 0
        
        model.reset_states()
        while idx < N:
            item = test_dataset[idx]

            #print(len(item))
            
            events = item[0]['event'].unsqueeze(dim=0)
            
            target = item[0]['depth'].cpu().numpy()
            # print(target.shape)
            
            events = events.float().to(gpu)
            
            pred_dict = model(events)
            pred_depth = pred_dict['pred_depth']
            #print(pred_depth.shape)
            if len(pred_depth.shape) > 3:
                
                pred_depth = pred_depth.squeeze(dim=0).cpu().numpy()
            # print(pred_depth.shape)
            np.save(join('pred_depth', '{:010d}.npy'.format(idx)), pred_depth)
            
            np.save(join('gt', '{:010d}.npy'.format(idx)), target)
            print(idx)
            
            idx += 1


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Image Reconstruction')
    parser.add_argument('--path_to_model', type=str,
                        help='path to the model weights',
                        default='')
    parser.add_argument('--output_path', type=str,
                        help='path to folder for saving outputs',
                        default='')
    parser.add_argument('--config', type=str,
                        help='path to config. If not specified, config from model folder is taken',
                        default=None)
    parser.add_argument('--data_folder', type=str,
                        help='path to folder of data to be tested',
                        default=None)

    args = parser.parse_args()

    if args.config is None:
        head_tail = os.path.split(args.path_to_model)
        config = json.load(open(os.path.join(head_tail[0], 'config.json')))
    else:
        config = json.load(open(args.config))
    
    path_to_model = 'e2depth_checkpoints/EReFormer_dense_epoch200_000032/model_best.pth.tar'

    main(config, path_to_model)