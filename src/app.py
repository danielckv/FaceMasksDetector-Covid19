#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import sys

from utils.env import _prepare_environment

# arguemnts:
# --type: train, test, infer
# --model: model name
# --dataset: dataset name

parser = argparse.ArgumentParser(description='FaceMaskDetector_CkvLabs')
parser.add_argument('--type', default='train', type=str, help='type of the run')
parser.add_argument('--model', default='resnet18', type=str, help='model name')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')

if __name__ == "__main__":
    _prepare_environment()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if args.type == 'train':
        print('Training...')
        train(args.model, args.dataset)
    elif args.type == 'test':
        print('Testing...')
        test(args.model, args.dataset)
    elif args.type == 'infer':
        print('Infering...')
        infer(args.model, args.dataset)
    