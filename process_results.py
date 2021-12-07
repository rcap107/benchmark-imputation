import pandas as pd
from src.utils import *
from utils_dset_study import Dataset, DatasetCase, ImputedDatasetCase, DirtyDatasetCase, measure_imputation_accuracy

if __name__ == '__main__':

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
        header = 'dsname,imputation_method,error_fraction,avg_acc,' + ','.join([f'imp_col_{_}' for _ in range(1,21)]) + '\n'
        f_res.write(header)

    for orig_dataset in os.listdir(CLEAN_DS_FOLDER):
        orig_basename = osp.basename(orig_dataset)
        dsname, ext = osp.splitext(orig_basename)

        ds = Dataset(dsname)

        ds_orig = DatasetCase(orig_basename, CLEAN_DS_FOLDER)

        for imp_dataset in os.listdir(IMPUTED_DS_FOLDER):
            imp_basename = osp.basename(imp_dataset)
            name, _ = osp.splitext(imp_basename)
            info = dict(zip(name_format, name.split('_')))

            if info['dsname'] == dsname:
                dirty_basename = f'{dsname}_{info["columns"]}_{info["error"]}.csv'
                ds_dirty = DirtyDatasetCase(dirty_basename, DIRTY_DS_FOLDER)
                ds_imp = ImputedDatasetCase(imp_dataset, IMPUTED_DS_FOLDER)
                res_dict = measure_imputation_accuracy(ds_orig, ds_dirty, ds_imp)
                s = f'{dsname},{res_dict["imputation_method"]},{res_dict["frac_missing"]},{res_dict["avg_acc"]},'
                s = s + ','.join([str(_) for _ in res_dict['acc_dict'].values()]) + '\n'
                f_res.write(s)
            else:
                continue

    f_res.close()
