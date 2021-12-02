'''
This script converts datasets in wide form and converts them to the formats required by some baselines.

Author: Riccardo Cappuzzo
'''

import pandas as pd
import os.path as osp
import os
from tqdm import tqdm
from bi_utils import *

def convert_dirty_to_hivae():
    pass

def convert_clean_to_hivae():
    pass

def convert_dirty_to_holoclean():
    pass


def convert_clean_to_holoclean(df_path):
    basename = get_name(df_path)
    gt_path = osp.join(HOLOCLEAN_FOLDER, f'testdata/raw/{basename}')
    os.makedirs(gt_path, exist_ok=True)
    gt_path = osp.join(gt_path, f'{basename}_clean.csv')

    df = pd.read_csv(df_path)
    with open(gt_path, 'w') as fp:
        header = 'tid,attribute,correct_val\n'
        fp.write(header)
        for rid, row in tqdm(df.iterrows(), total=len(df)):
            for col in df.columns:
                s = f'{rid},{col},{df.loc[rid, col]}\n'
                fp.write(s)


def convert_to_number(df):
    df_copy = df.copy()
    n_uniques = {}
    for idx, col in enumerate(df.columns):
        if df[col].dtype == 'O':
            uniques = df_copy[col].unique().tolist()
            dict_uniques = {uniques[_i]: _i for _i in range(len(uniques))}
            df_copy[col] = df_copy[col].apply(lambda x: dict_uniques[x])
            n_uniques[idx] = len(uniques)
    return n_uniques



def write_hivae_version(df_dirty, df_base, args):
    # HI-VAE requires a specific error format to work take datasets as input. This function prepares all files required
    # by HI-VAE in the proper format.

    print('Writing additional files for HI-VAE.')
    input_file_name = args.input_file

    clean_name, ext = osp.splitext(input_file_name)
    basename = osp.basename(clean_name)
    # Input dataset is equivalent to the base dataframe, but all values are converted to (numeric) categorical values.
    input_dataset_path = f'data/{basename}_hivae{ext}'
    output_dataset_path = f'data/dirty_datasets/{basename}_{"_".join(args.target_columns)}_hivae'

    if args.tag:
        output_dataset_path += f'_{args.tag}{ext}'
    else:
        output_dataset_path += f'{ext}'

    for df, fpath in zip([df_base, df_dirty], [input_dataset_path, output_dataset_path]):
        n_uniques = convert_to_number(df)
        # convert_to_hivae_format(df_base, df_orig, fpath)

    # All errors (missing values) are listed in a text file that contains the coordinates of the missing values (row, col).
    basename_out, ext = path.splitext(output_dataset_path)
    error_file = f'{basename_out}_errors' + ext
    with open(error_file, 'w') as fp:
        for col_idx, col in enumerate(df_dirty.columns):
            error_idx = df_dirty.loc[df_dirty[col].isna()].index.tolist()
            for e in error_idx:
                s = f'{e + 1},{col_idx + 1}\n'
                fp.write(s)

    prepare_dtype_file(df_base, n_uniques)

    dd = {_: list(df_base.columns)[_] for _ in range(len(df_base.columns))}
    for idx, dt in enumerate(df_base.dtypes):
        # col = df_orig.columns[idx]
        if dt == 'object':
            dd[idx] = ['cat', str(n_uniques[idx]), str(n_uniques[idx])]
        else:
            dd[idx] = ['real', '1', '']

    dtype_fname = f'data/{basename}_hivae_data_types.csv'
    with open(dtype_fname, 'w') as fp:
        print(f'Datatypes saved in file {dtype_fname}')
        header = 'type,dim,nclass'
        fp.write(header)
        for k, v in dd.items():
            s = '\n' + ','.join(v)
            fp.write(s)


    return output_dataset_path


def get_name(df_path):
    basename, ext = osp.splitext(osp.basename(df_path))
    return basename

if __name__ == '__main__':
    # TODO: add parameters
    clean_base_dir = 'data/clean'
    dirty_base_dir = 'data/dirty'

    for clean_df_path in os.listdir(clean_base_dir):
        print(f'Working on dataset {clean_df_path}')
        convert_clean_to_holoclean(osp.join(clean_base_dir, clean_df_path))

    for dirty_df in os.listdir(dirty_base_dir):
        pass

