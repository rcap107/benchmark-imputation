'''
This script converts a dataset with Holoclean's long format into a dataset in wide format.
It infers columns and
'''

import argparse
import pandas as pd
import os.path as osp
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dataset')
    # parser.add_argument()

    args = parser.parse_args()

    return args

def reconstruct_hc(df_path):
    df = pd.read_csv(df_path)
    columns = df['attribute'].unique()
    df_reconstructed = pd.DataFrame(columns=columns)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        df_reconstructed.loc[row['tid'],row['attribute']] = row['inferred_val']

    return df_reconstructed


if __name__ == '__main__':
    args = parse_args()

    base, ext = osp.splitext(args.input_dataset)
    df_reconstructed = reconstruct_hc(args.input_dataset)
    df_reconstructed.to_csv(base + '-rec' + ext, index=False)
