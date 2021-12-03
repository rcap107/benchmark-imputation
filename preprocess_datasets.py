import os

from src.utils import *
import src.prepare_dir_tree as dirtree
import src.generate_variants as generate_variants
import src.error_injection as error_injection

if __name__ == '__main__':
    # 1. Prepare directory tree for all datasets
    # 2. Generate dirty datasets for all cases required
    # 3. Generate all variants and auxiliary files

    # Pepare dirtree
    dirtree.dirtree_grimp()
    dirtree.dirtree_holoclean()
    dirtree.dirtree_misf()
    error_cases = [0.02, 0.1, 0.2, 0.4, 0.6]

    # Prepare dirty datasets
    for clean_dataset in os.listdir(CLEAN_DS_FOLDER):
        ds_name, ext = osp.splitext(clean_dataset)
        ds_path = osp.join(CLEAN_DS_FOLDER, clean_dataset)

        for ef in error_cases:
            ei = error_injection.ErrorInjector(ds_name, ds_path, error_fraction=ef, target_all_columns=True, seed=None)
            ei.inject_errors()
            ei.write_dirty_dataset_on_file()

    # Prepare variants
    generate_variants.prepare_holoclean()
    generate_variants.prepare_misf()
    generate_variants.prepare_grimp()
