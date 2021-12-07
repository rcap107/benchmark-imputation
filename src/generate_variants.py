'''
This script converts datasets in wide form and converts them to the formats required by some baselines.

Author: Riccardo Cappuzzo
'''

import pandas as pd
from tqdm import tqdm
from src.utils import *
import json
from shutil import copyfile
import fasttext
import numpy as np

# HOLOCLEAN
def prepare_holoclean():
    os.makedirs('variants/holoclean/testdata/raw', exist_ok=True)
    os.makedirs('variants/holoclean/dump', exist_ok=True)
    os.makedirs('variants/holoclean/meta_data', exist_ok=True)

    for f in os.listdir(CLEAN_DS_FOLDER):
        basename, ext = osp.splitext(f)
        os.makedirs(osp.join(HOLOCLEAN_RAW_FOLDER, basename), exist_ok=True)
        tgt_dir = osp.join(CLEAN_DS_FOLDER,f)
        prepare_clean_holoclean(tgt_dir)

    for f in os.listdir(DIRTY_DS_FOLDER):
        basename = get_name(f)
        orig_dataset = basename.split('_', maxsplit=1)[0]
        # Create a dir for every dirty dataset
        os.makedirs(osp.join(HOLOCLEAN_RAW_FOLDER, basename), exist_ok=True)

        # Copy the dataset in the dir
        src_file = osp.join(DIRTY_DS_FOLDER, f)
        dst_file = osp.join(HOLOCLEAN_RAW_FOLDER, basename, f)
        copyfile(src_file, dst_file)

        # Prepare metadata
        prepare_metadata_holoclean(src_file)

        # Copy the ground truth file
        src_file = osp.join(HOLOCLEAN_RAW_FOLDER, orig_dataset, orig_dataset + '_clean.csv')
        dst_file = osp.join(HOLOCLEAN_RAW_FOLDER, basename, orig_dataset + '_clean.csv')
        copyfile(src_file, dst_file)

def prepare_clean_holoclean(df_path):
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

def prepare_metadata_holoclean(df_raw_path, numerical_columns=None):
    '''
    The run_holoclean.py script requires info to be present in the meta_data.py script.
    Metadata are hardcoded as dictionaries in the script.

    This function outputs a json file that should be copied in the metadaa file.
    :return:
    '''

    if numerical_columns is None:
        numerical_columns = []

    df_dirty = pd.read_csv(df_raw_path)
    df_raw_name, ext = osp.splitext(osp.basename(df_raw_path))

    if not numerical_columns:
        num_attrs = df_dirty.select_dtypes(include='number').columns.tolist()
    else:
        num_attrs = numerical_columns

    target_dict = {
        'target_attrs': df_dirty.columns.tolist(),
        'num_attrs': num_attrs,
        'num_attr_groups': [[_] for _ in  num_attrs],
        'data_dir': f'testdata/raw/{df_raw_name}',
        'raw_prefix': df_raw_name,
        'clean_prefix': df_raw_name.split('_', maxsplit=1)[0],
        'dc_file': None,
        'hc_batch': 32,
        'multiple_correct': False,
    }
    json.dump(target_dict, open(osp.join('variants/holoclean/meta_data/', f'{df_raw_name}.json'), 'w'), indent=4)


# MISSFOREST
def prepare_misf():
    os.makedirs('variants/misf/data/clean', exist_ok=True)
    os.makedirs('variants/misf/data/dirty', exist_ok=True)

    for f in os.listdir(CLEAN_DS_FOLDER):
        # basename, ext = osp.splitext(f)
        src_file = osp.join(CLEAN_DS_FOLDER, f)
        dst_dir = osp.join(MISF_FOLDER, 'data/clean')
        dst_file = osp.join(dst_dir, f)
        copyfile(src_file, dst_file)

    for f in os.listdir(DIRTY_DS_FOLDER):
        # basename = get_name(f)
        src_file = osp.join(DIRTY_DS_FOLDER, f)
        dst_dir = osp.join(MISF_FOLDER, 'data/dirty')
        dst_file = osp.join(dst_dir, f)
        copyfile(src_file, dst_file)

# GRIMP
def prepare_grimp():
    os.makedirs('variants/grimp/data/clean', exist_ok=True)
    os.makedirs('variants/grimp/data/dirty', exist_ok=True)

    print('Loading fasttext model...')
    fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    for f in os.listdir(CLEAN_DS_FOLDER):
        # basename, ext = osp.splitext(f)
        src_file = osp.join(CLEAN_DS_FOLDER, f)
        dst_dir = osp.join(GRIMP_FOLDER, 'data/clean')
        dst_file = osp.join(dst_dir, f)
        copyfile(src_file, dst_file)

    for f in os.listdir(DIRTY_DS_FOLDER):
        basename = get_name(f)
        src_file = osp.join(DIRTY_DS_FOLDER, f)
        dst_dir = osp.join(GRIMP_FOLDER, 'data/dirty')
        dst_file = osp.join(dst_dir, f)
        copyfile(src_file, dst_file)
        df_dirty = pd.read_csv(src_file)
        generated_emb_file = osp.join(GRIMP_PRETRAINED_EMB_FOLDER, f'{basename}_ft.emb')
        generate_pretrained_embeddings_grimp(df_dirty, generated_emb_file, model=fasttext_model)


def generate_pretrained_embeddings_grimp(df, generated_file, model, n_dim=300):
    # Replace '-' with spaces for sentence emb generation
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('-', ' ')

    # Add to each value in the dataset the column they belong to.
    for idx, col in enumerate(df.columns):
        df[col] = df[col].apply(lambda x: f'c{idx}_{x}')

    # Extract all unique values
    unique_values = [str(_) for _ in set(df.values.ravel())]

    # Generate sentence vectors for all unique values
    # print('Generating token embeddings. ')
    val_vectors = []
    missing_vals = []
    for idx, val in enumerate(unique_values):
        # Remove the prefix and generate the vector.
        prefix, true_val = val.split('_', maxsplit=1)
        # Ignore null values.
        if true_val == 'nan' or true_val != true_val:
            vector = np.zeros(n_dim)
        else:
            vector = model.get_sentence_vector(str(true_val))
        val_vectors.append(vector)

    # Prepare dict with unique value + sentence vector for that value
    vector_dict = dict(zip(unique_values, val_vectors))
    if 'nan' in vector_dict:
        vector_dict.pop('nan')

    vector_dict['nan'] = np.zeros(n_dim)

    # print('Generating row embeddings.')
    tot_rows = len(vector_dict) + df.shape[0] + df.shape[1]
    row_vectors = dict()

    for idx, row in df.iterrows():
        tmp_vec = np.zeros(shape=(df.shape[1], n_dim))
        for ri, w in enumerate(row):
            w = str(w)
            vector = vector_dict[w]
            tmp_vec[ri] = vector
        row_vectors[idx] = np.mean(tmp_vec, 0)

    # print('Generating column embeddings.')
    col_vectors = dict()

    for idx, col in enumerate(df.columns):
        tmp_vec = np.zeros(shape=(df.shape[0], n_dim))
        for ri, w in enumerate(df[col]):
            vector = vector_dict[str(w)]
            tmp_vec[ri] = vector
        col_vectors[col] = np.mean(tmp_vec, 0)

    # print('Writing embeddings on file. ')
    tot_rows = len(row_vectors) + len(col_vectors) + len(vector_dict)
    t = tqdm(total=tot_rows)
    with open(generated_file, 'w') as fp:
        fp.write(f'{tot_rows} {n_dim}\n')
        # for k, vec in tqdm(row_vectors.items(), total=len(row_vectors)):
        for k, vec in row_vectors.items():
            s = f'idx__{k} ' + ' '.join([str(_).strip() for _ in vec]) + '\n'
            fp.write(s)
            t.update(1)
            t.refresh()
        # for k, vec in tqdm(col_vectors.items(), total=len(col_vectors)):
        for k, vec in col_vectors.items():
            s = f'cid__{k.replace(" ", "-")} ' + ' '.join([str(_).strip() for _ in vec]) + '\n'
            fp.write(s)
            t.update(1)
            t.refresh()
        # for k, vec in tqdm(vector_dict.items(), total=len(vector_dict)):
        for k, vec in vector_dict.items():
            if k == 'nan':
                continue
            s = f'tt__{k.replace(" ", "-")} ' + ' '.join([str(_).strip() for _ in vec]) + '\n'
            fp.write(s)
            t.update(1)
            t.refresh()
    t.close()


def write_hivae_version(df_dirty, df_base, args):
    # HI-VAE requires a specific error format to work take datasets as input. This function prepares all files required
    # by HI-VAE in the proper format.

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
    # prepare_holoclean()
    # prepare_misf()
    prepare_grimp()
