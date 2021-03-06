'''
Some baselines require a specific directory tree to work properly. This script prepares the directory tree and copies
the input files to their respective positions.

Author: Riccardo Cappuzzo
'''

import os
import os.path
import sys

import argparse


def dirtree_misf():
    os.makedirs('variants/misf/data/clean', exist_ok=True)
    os.makedirs('variants/misf/data/dirty', exist_ok=True)


def dirtree_grimp():
    os.makedirs('variants/grimp/data/clean', exist_ok=True)
    os.makedirs('variants/grimp/data/dirty', exist_ok=True)
    os.makedirs('variants/grimp/data/pretrained-emb', exist_ok=True)


def dirtree_holoclean():
    os.makedirs('variants/holoclean/testdata/raw', exist_ok=True)
    os.makedirs('variants/holoclean/dump', exist_ok=True)
    os.makedirs('variants/holoclean/meta_data', exist_ok=True)


