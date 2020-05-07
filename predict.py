#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
#from data.mm import MovingMNIST
from loader import MovingMNIST
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from datetime import datetime
import json


parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--is-train',
                    default = False,
                    help='train or not',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help='mini-batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('--frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('--frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('--epochs', default=500, type=int, help='sum of epochs')
parser.add_argument('--timestamp', default='NA', type=str, help='timestamp of model you want to restore')
parser.add_argument('--checkpoint', default='NA', type=str, help='checkpoint of model you want to restore')
args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False







def test():
    '''
    main function to run the training
    '''
    testFolder = MovingMNIST(is_train=False,
                              root='../data/npy-064/',
                              mode ='test',
                              n_frames_input=args.frames_input,
                              n_frames_output=args.frames_output,
                              num_objects=[3])
    testLoader = torch.utils.data.DataLoader(testFolder,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    if args.convlstm:
        encoder_params = convlstm_encoder_params
        decoder_params = convlstm_decoder_params
    if args.convgru:
        encoder_params = convgru_encoder_params
        decoder_params = convgru_decoder_params
    else:
        encoder_params = convgru_encoder_params
        decoder_params = convgru_decoder_params

    #TIMESTAMP = args.timestamp
    # restore args

    CHECKPOINT = args.checkpoint    
    TIMESTAMP = args.timestamp
    save_dir = './save_model/' + TIMESTAMP

    args_path = os.path.join(save_dir, 'cmd_args.txt')
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            args.__dict__ = json.load(f)
            args.is_train = False
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1], args.frames_output).cuda()
    net = ED(encoder, decoder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(save_dir):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(CHECKPOINT)
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
    else:
        print('there is no such checkpoint in', save_dir)
        exit()
    lossfunction = nn.MSELoss().cuda()
    # to track the testation loss as the model trains
    test_losses = []
    # to track the average training loss per epoch as the model trains
    avg_test_losses = []
    # mini_val_loss = np.inf

    preds = [] 
    ######################
    # testate the model #
    ######################
    with torch.no_grad():
        net.eval()
        t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            if i == 3000:
                break
            inputs = inputVar.to(device)
            #label = targetVar.to(device)
            pred = net(inputs)
            #loss = lossfunction(pred, label)
            preds.append(pred)
            #loss_aver = loss.item() / args.batch_size
            # record testation loss
            #test_losses.append(loss_aver)

    torch.cuda.empty_cache()
    # print training/testation statistics
    # calculate average loss over an epoch
    #test_loss = np.average(test_losses)
    #avg_test_losses.append(test_loss)
    
    #print_msg = (f'test_loss: {test_loss:.6f}')
    #print(print_msg)

    import pickle
    with open("preds.pkl", "wb") as fp:
        pickle.dump(preds, fp)

if __name__ == "__main__":
    print('testing...')
    test()
