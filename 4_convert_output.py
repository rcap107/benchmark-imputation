'''
This script reads the inputed datasets coming from different sources and refactors them into a wide-form dataset to study.

Author: Riccardo Cappuzzo

'''

import pandas as pd
import os.path as osp
import os
from bi_utils import *
from tqdm import tqdm


def reconstruct_hc(df_path):
    df = pd.read_csv(df_path)
    columns = df['attribute'].unique()
    df_reconstructed = pd.DataFrame(columns=columns)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        df_reconstructed.loc[row['tid'],row['attribute']] = row['inferred_val']

    return df_reconstructed



if __name__ == '__main__':
    # TODO: generalize this
    df_name = 'bikes-dekho_all_columns_20_imputed_holo.csv'
    df_to_fix = osp.join(IMPUTED_DS_FOLDER, df_name)

    df_reconstructed = reconstruct_hc(df_to_fix)

    base, ext = osp.splitext(df_to_fix)
    out_name = base.rsplit('_', maxsplit=1)[0]
    df_reconstructed.to_csv(out_name + '_holo' + ext, index=False)
