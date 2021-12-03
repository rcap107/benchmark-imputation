

import argparse
import os
import os.path as path
import random

import numpy as np
from numpy.random import default_rng
import pandas as pd

from bi_utils import *

class ErrorInjector:
    def __init__(self, df_name, df_clean_path,
                 error_fraction=None,
                 error_fraction_list=None,
                 target_all_columns=True,
                 target_columns=None,
                 convert_columns=None,
                 blackout=False,
                 missing_value_flag=np.nan,
                 conversion_prefix='O_',
                 na_value=np.nan,
                 seed=1234
                 ):
        self.random_state = default_rng(seed).bit_generator
        self.df_name = df_name
        self.df_clean = pd.read_csv(df_clean_path, na_values=missing_value_flag)

        self.conversion_prefix = conversion_prefix
        self.blackout = blackout
        self.na_value = na_value

        self.error_fraction = error_fraction
        self.error_fraction_list = error_fraction_list


        # Consolidate target columns.
        if target_columns is None:
            if not target_all_columns:
                raise ValueError('If target_all_columns==False, target_columns must not be empty.')
            self.target_columns = self.df_clean.columns.tolist()
            self.target_all_columns = True
        else:
            self.target_columns = target_columns
            self.target_all_columns = False

        self._check_error_fraction()
        # If the error fraction is provided, the error_fraction_list contains copies of the same value
        if self.error_fraction is not None and self.error_fraction_list is None:
            self.error_fraction_list = [self.error_fraction for _ in self.target_columns]
            self.simple = True
        else:
            self.simple = False

        if convert_columns is None:
            self.convert_columns = []
        else:
            self.convert_columns = convert_columns

        self._input_columns_validity_check()
        self._convert_columns()

    def _check_error_fraction(self):
        # Check if either error_fraction or error_fraction_list contain something.
        if self.error_fraction is None and self.error_fraction_list is None:
            raise ValueError('No error fraction provided.')
        # Check if both are provided. There can be only one.
        if self.error_fraction is not None and self.error_fraction_list is not None:
            raise ValueError('Only one between error_fraction and error_fraction_list can be provided.')

        # Ensure that each target column has a proper error fraction.
        if self.error_fraction_list is not None:
            if len(self.error_fraction_list) != len(self.target_columns):
                raise ValueError(f'For detailed error injection, there should be a value of error fraction'
                                 f' for each target column.\nProvided {len(self.error_fraction_list)} values for {len(self.target_columns)} columns. ')
            for ef in self.error_fraction_list:
                if not 0<=ef<=1:
                    raise ValueError(f'Error fraction {ef} is not in [0,1].')

        if self.error_fraction_list is not None and self.blackout:
            raise ValueError('For blackout cases, only one error fraction value is supported. ')


    def _input_columns_validity_check(self):
        # Check that all column names supplied by the user are found in the dataframe.
        for col in set(self.target_columns + self.convert_columns):
            if col not in self.df_clean.columns:
                raise (ValueError(f'Column {col} was not found in the dataset header.'))

    def _convert_columns(self):
        for col in self.convert_columns:
            self.df_clean[col] = self.df_clean[col].apply(lambda x: self.conversion_prefix + str(x))

    def _column_error_injection(self, column: pd.Series, mask, na_value=np.nan):
        column_dirty = column.copy(deep=True)
        column_dirty.loc[mask] = na_value
        count_injected_values = sum((column_dirty.isna()))
        return column_dirty, count_injected_values

    def inject_errors(self):
        _tmp_df = self.df_clean.copy(deep=True)
        error_count = {column: 0 for column in _tmp_df.columns}
        # All errors are focused on the same lines
        # This is a very hard case to be used as a stress test.
        if self.blackout:
            mask = self._get_mask(_tmp_df, self.error_fraction)
        else:
            mask = None

        eff = dict(zip(self.target_columns, self.error_fraction_list))

        for col, ef in eff.items():
            if mask is not None:
                m = mask
            else:
                m = self._get_mask(_tmp_df, ef)
            _tmp_df[col], injected = self._column_error_injection(_tmp_df[col], m, self.na_value)
            error_count[col] = injected
            print(f'Column {col}: {injected} errors in {len(_tmp_df[col])} values.')

        self.df_dirty = _tmp_df
        return _tmp_df

    def _get_mask(self, df, error_fraction):
        return df.sample(frac=error_fraction, random_state=self.random_state).index


    def write_dirty_dataset_on_file(self, tag=None, tag_error_frac=True):
        if self.target_all_columns:
            output_name = f'{self.df_name}_all_columns'
        else:
            output_name = f'{self.df_name}_{"_".join(self.target_columns)}'
        s_tag = ''
        # Add error fraction in simple case
        if tag_error_frac and self.simple:
            s_tag += f'_{self.error_fraction * 100:02.0f}'
        # Add all error fractions.
        else:
            s_tag += f'_{"_".join(self.error_fraction_list) * 100:g}'
        if self.blackout:
            s_tag += '_blackout'
        if tag:
            s_tag += f'_{tag}'
        output_name += f'{s_tag}.csv'

        output_path = osp.join(DIRTY_DS_FOLDER, output_name)

        print(f'Output file: {output_path}')
        self.df_dirty.to_csv(output_path, index=False)
        return output_name

    def get_dirty_dataset(self):
        return self.df_dirty

    def get_clean_dataset(self):
        return self.df_clean


if __name__ == '__main__':
    df_name = 'bikesdekho'
    df_clean_path = osp.join(CLEAN_DS_FOLDER, df_name+'.csv')

    ei = ErrorInjector(df_name, df_clean_path, error_fraction=0.02)
    ei.inject_errors()
    ei.write_dirty_dataset_on_file()