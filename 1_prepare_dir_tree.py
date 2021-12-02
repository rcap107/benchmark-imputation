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
    pass

def dirtree_grimp():
    pass

def dirtree_holoclean():
    os.makedirs('variants/holoclean/testdata/raw', exist_ok=True)
    os.makedirs('variants/holoclean/dump')


if __name__ == '__main__':
    dirtree_holoclean()
