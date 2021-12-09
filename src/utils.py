import os.path as osp
import os

CLEAN_DS_FOLDER=osp.normpath('data/clean')
DIRTY_DS_FOLDER=osp.normpath('data/dirty')
IMPUTED_DS_FOLDER=osp.normpath('data/imputed')
SUMMARIES_FOLDER=osp.normpath('data/summaries')
RESULTS_PATH=osp.normpath('data/results.csv')
FASTTEXT_MODEL_PATH=osp.normpath('data/cc.en.300.bin')


HOLOCLEAN_FOLDER=osp.normpath('variants/holoclean')
HOLOCLEAN_RAW_FOLDER= osp.normpath('variants/holoclean/testdata/raw')
GRIMP_FOLDER=osp.normpath('variants/grimp')
GRIMP_PRETRAINED_EMB_FOLDER=osp.normpath('variants/grimp/data/pretrained-emb')
HIVAE_FOLDER=osp.normpath('variants/hivae')
MISF_FOLDER=osp.normpath('variants/misf')
VARIANTS_FOLDER=osp.normpath('variants')