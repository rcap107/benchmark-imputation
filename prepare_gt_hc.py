'''
This script is used to convert a given dirty dataset rep[resented as a
'''

import pandas as pd
import os.path as osp
import os
import shutil
from tqdm import tqdm


def unravel(df, gt_path):
    with open(gt_path, 'w') as fp:
        header = 'tid,attribute,correct_val\n'
        fp.write(header)
        for rid, row in tqdm(df.iterrows(), total=len(df)):
            for col in df.columns:
                s = f'{rid},{col},{df.loc[rid, col]}\n'
                fp.write(s)


if __name__ == '__main__':
    orig_clean_dir = osp.normpath(f'data/new-hard')
    orig_dirty_dir = osp.normpath(f'data/new-hard_dirty')
    for df_file in os.listdir(orig_clean_dir):
        df_clean_path = osp.join(orig_clean_dir, df_file)
        df_clean = pd.read_csv(df_clean_path)

        df_name, ext = osp.splitext(df_file)
        hc_dir = osp.join(osp.join('data', 'variants/holoclean'), df_name)
        os.makedirs(hc_dir, exist_ok=True)

        shutil.copy(df_clean_path, osp.join(hc_dir, df_name + '.csv'))

        for df_dirty_path in os.listdir(orig_dirty_dir):
            if df_dirty_path.startswith(f'{df_name}_all_columns'):
                shutil.copy(osp.join(orig_dirty_dir, df_dirty_path), osp.join(hc_dir, df_dirty_path))
                df_dirty = pd.read_csv(osp.join(hc_dir, df_dirty_path))
                gt_path = osp.join(hc_dir, f'{df_name}_clean.csv')
                unravel(df_clean, gt_path)
