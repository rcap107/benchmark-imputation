import os
from tqdm import tqdm

import pandas as pd
from src.utils import *
from utils_dset_study import Dataset, DatasetCase, ImputedDatasetCase, DirtyDatasetCase, measure_imputation_accuracy


def reconstruct(df_dirty, df_pred, dfname=None):
    df_rec = df_dirty.copy()
    for idx,r  in tqdm(df_pred.iterrows(), total=len(df_pred), desc=f'{dfname:<20}'):
        row, col = r[['tid', 'attribute']]
        df_rec.loc[row,col] = r['inferred_val']
    return df_rec


def reconstruct_imputed_dataset_holo(orig_dsname):
    orig_basename, ext = osp.splitext(orig_dsname)
    for imp_dir in os.listdir(HOLOCLEAN_RAW_RESULTS_FOLDER):
        this_dataset_name = imp_dir.split('_',maxsplit=1)[0]
        if this_dataset_name == orig_basename:
            # Only consider variants of the given orig_dsname
            result_path = osp.join(HOLOCLEAN_RAW_RESULTS_FOLDER, imp_dir)
            df_dirty_name = imp_dir + '.csv'
            df_dirty = pd.read_csv(osp.join(DIRTY_DS_FOLDER, df_dirty_name) )
            for f in os.listdir(result_path):
                fname, ext = osp.splitext(f)
                if ext == '.csv':
                    # print(f'Reconstructing dataset {imp_dir}_imputed_holo.csv')
                    df_rec_fname = osp.join(IMPUTED_DS_FOLDER, f'{imp_dir}_imputed_holo.csv')
                    df_pred = pd.read_csv(osp.join(result_path,f))
                    df_rec = reconstruct(df_dirty, df_pred, imp_dir)
                    df_rec.to_csv(df_rec_fname, index=False)

if __name__ == '__main__':

    # for clean_dsname in os.listdir(CLEAN_DS_FOLDER):
    #     if not osp.isdir(clean_dsname):
    #         reconstruct_imputed_dataset_holo(clean_dsname)
    #
    #
    # raise Exception

    name_format=[
        'dsname',
        'columns',
        'error',
        'case',
        'method'
    ]

    if osp.exists(RESULTS_PATH):
        f_res = open(RESULTS_PATH, 'a')
    else:
        f_res = open(RESULTS_PATH, 'w')
        header = 'dsname,imputation_method,error_fraction,avg_acc,avg_rmse' + ','.join([f'imp_col_{_}' for _ in range(1,21)]) + '\n'
        f_res.write(header)


    # tgt_datasets = []

    for orig_dataset in tqdm(sorted(os.listdir(CLEAN_DS_FOLDER)), total=len(os.listdir(CLEAN_DS_FOLDER))):
        if osp.isdir(osp.join(CLEAN_DS_FOLDER,orig_dataset)):
            continue
        orig_basename = osp.basename(orig_dataset)
        dsname, ext = osp.splitext(orig_basename)
        if dsname != 'mammogram':
            continue

        ds = Dataset(dsname)

        ds_orig = DatasetCase(orig_basename, CLEAN_DS_FOLDER)

        for imp_dataset in os.listdir(IMPUTED_DS_FOLDER):
            imp_basename = osp.basename(imp_dataset)
            name, _ = osp.splitext(imp_basename)
            info_imputed = dict(zip(name_format, name.split('_')))

            if info_imputed['dsname'] == dsname:
                dirty_basename = f'{dsname}_{info_imputed["columns"]}_{info_imputed["error"]}.csv'
                ds_dirty = DirtyDatasetCase(dirty_basename, DIRTY_DS_FOLDER)
                ds_imp = ImputedDatasetCase(imp_dataset, IMPUTED_DS_FOLDER)
                res_dict = measure_imputation_accuracy(ds_orig, ds_dirty, ds_imp)
                s  = f'{dsname},'
                s += f'{res_dict["imputation_method"]},'
                s += f'{res_dict["frac_missing"]},{res_dict["avg_acc"]},{res_dict["avg_rmse"]},'
                s = s + ','.join([str(_) for _ in res_dict['acc_dict'].values()]) + '\n'
                f_res.write(s)
            else:
                continue

    f_res.close()
