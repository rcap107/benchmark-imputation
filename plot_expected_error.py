import os
from tqdm import tqdm

import pandas as pd
from src.utils import *
from utils_dset_study import Dataset, DatasetCase, ImputedDatasetCase, DirtyDatasetCase, measure_imputation_accuracy



orig_dataset = f'{CLEAN_DS_FOLDER}/thoracic.csv'

orig_basename = osp.basename(orig_dataset)
dsname, ext = osp.splitext(orig_basename)
dirty_basename = f'{dsname}_allcolumns_20.csv'

ds = Dataset(dsname)
ds_orig = DatasetCase(orig_basename, CLEAN_DS_FOLDER)
ds.add_dataset('orig', orig_basename, CLEAN_DS_FOLDER)
ds.add_dataset('dirty', dirty_basename, DIRTY_DS_FOLDER)


name_format=[
    'dsname',
    'columns',
    'error',
    'case',
    'method'
]

for imp_dataset in os.listdir('data/imputed_plot'):
    imp_basename = osp.basename(imp_dataset)
    name, _ = osp.splitext(imp_basename)
    info_imputed = dict(zip(name_format, name.split('_')))

    if info_imputed['dsname'] == dsname:
        ds_imp = ds.add_dataset('imp', imp_dataset, 'data/imputed_plot')
        # ds.measure_imputation_accuracy(ds_imp.df)
    else:
        continue

ds.extract_imputations()
