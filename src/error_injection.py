

import argparse
import os
import os.path as path
import random

import numpy as np
from numpy.random import default_rng
import pandas as pd

from src.utils import *

class ErrorInjector:
    def __init__(self, df_name, df_clean_path,
                 error_fraction=None,
                 error_fraction_list=None,
                 target_all_columns=True,
                 target_columns=None,
                 convert_columns=None,
                 strategy='simple',
                 missing_value_flag=np.nan,
                 conversion_prefix='O_',
                 na_value=np.nan,
                 seed=1234
                 ):
        self.rng = default_rng(seed)
        self.df_name = df_name
        self.df_clean = pd.read_csv(df_clean_path, na_values=missing_value_flag, engine='python')
        self.df_dirty = None

        self.conversion_prefix = conversion_prefix
        self.strategy = strategy
        self.na_value = na_value

        self.error_fraction = error_fraction
        self.error_fraction_dict = None
        self.error_fraction_list = error_fraction_list


        # Consolidate target columns.
        if target_columns is None:
            # Check that flags are consistent
            if not target_all_columns:
                raise ValueError('If target_all_columns==False, target_columns must not be empty.')
            # Set target columns to all
            self.target_columns = self.df_clean.columns.tolist()
            self.target_all_columns = True
        else:
            # Set target columns to selected
            self.target_columns = target_columns
            self.target_all_columns = False

        self._check_error_fraction()
        # If the error fraction is provided, the error_fraction_list contains copies of the same value
        if self.error_fraction is not None and error_fraction_list is None:
            self.error_fraction_dict = {col: self.error_fraction for col in self.target_columns}
            self.simple = True
        elif self.error_fraction is None:
            self.error_fraction_dict = {col: self.error_fraction_list for col in self.target_columns}
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

        if self.error_fraction_list is not None and self.strategy == 'blackout':
            raise ValueError('For blackout cases, only one error fraction value is supported. Use strategy "partial" ')

    def _input_columns_validity_check(self):
        # Check that all column names supplied by the user are found in the dataframe.
        for col in set(self.target_columns + self.convert_columns):
            if col not in self.df_clean.columns:
                raise (ValueError(f'Column {col} was not found in the dataset header.'))

    def _convert_columns(self):
        '''
        Force columns to categorical dtype by adding a string prefix to the value.
        :return:
        '''
        for col in self.convert_columns:
            self.df_clean[col] = self.df_clean[col].apply(lambda x: self.conversion_prefix + str(x))

    def _column_error_injection(self, column: pd.Series, mask, na_value=np.nan):
        '''
        Inject errors in the given column according to the given mask.
        :param column: Target dataset column.
        :param mask: np array that contains the indices of the values to mask
        :param na_value: Value to use as null
        :return: masked column, number of injected missing values
        '''
        column_dirty = column.copy(deep=True)
        column_dirty.loc[mask.astype(bool)] = na_value
        count_injected_values = sum((column_dirty.isna()))
        return column_dirty, count_injected_values

    def _get_mask_simple(self):
        mask = {}
        for col in self.target_columns:
            num_errors = np.ceil(self.error_fraction_dict[col]*len(self.df_clean))
            mask[col] = self.rng.choice(self.df_clean[col].index, num_errors, replace=False,)

        return mask

    def _get_mask_blackout(self):
        '''
        # All errors are focused on the same lines
        # This is a very hard case to be used as a stress test.
        :return: mask dict
        '''

        num_errors = np.ceil(self.error_fraction * len(self.df_clean))
        single_mask = np.random.choice(self.df_clean.index, num_errors, replace=False)

        mask = {col: single_mask for col in self.target_columns}

        return mask


    def _get_mask_burst(self):
        '''


        :return: mask dict
        '''

        mask = {}

        for col in self.target_columns:
            num_errors = np.ceil(self.error_fraction_dict[col]*len(self.df_clean))
            num_bursts = int(len(self.df_clean)//num_errors)
            burst_length = int(num_errors//num_bursts)
            burst_start = self.rng.choice(range(len(self.df_clean)), num_bursts, replace=False)
            burst_start.sort()
            mm = np.zeros(len(self.df_clean))
            for burst in burst_start:
                mm[burst:min(len(self.df_clean), burst+burst_length)] = 1
            mask[col] = mm

        return mask

    def _get_mask_high_freq_bias(self):
        frequencies_by_col = {}
        quantile50 = {}
        for col in self.target_columns:
            frequencies_by_col[col] = self.df_clean[col].value_counts()
            quantile50[col] = frequencies_by_col[col].quantile(0.50)

        freq_values = {col: [] for col in frequencies_by_col}
        pos_frequents = {}
        for col in frequencies_by_col:
            for val, freq in frequencies_by_col[col].iteritems():
                if freq > quantile50[col]:
                    freq_values[col].append(val)
            pos_frequents[col] = self.df_clean.loc[self.df_clean[col].isin(freq_values[col])].index

        mask = {}
        for col in self.target_columns:
            num_errors = np.ceil(self.error_fraction_dict[col]*len(self.df_clean))
            mm = np.zeros(len(self.df_clean))
            if len(pos_frequents[col]>0):
                mm[self.rng.choice(pos_frequents[col], min(int(num_errors), len(pos_frequents[col])), replace=False)]=1
            mask[col] = mm
        return mask


    def _get_mask_low_freq_bias(self):
        frequencies_by_col = {}
        quantile50 = {}
        for col in self.target_columns:
            frequencies_by_col[col] = self.df_clean[col].value_counts()
            quantile50[col] = frequencies_by_col[col].quantile(0.50)

        freq_values = {col: [] for col in frequencies_by_col}
        pos_viable = {}
        for col in frequencies_by_col:
            for val, freq in frequencies_by_col[col].iteritems():
                if quantile50[col] > freq > 1:
                    freq_values[col].append(val)
            pos_viable[col] = self.df_clean.loc[self.df_clean[col].isin(freq_values[col])].index

        mask = {}
        for col in self.target_columns:
            num_errors = np.ceil(self.error_fraction_dict[col]*len(self.df_clean))
            mm = np.zeros(len(self.df_clean))
            if len(pos_viable[col]>0):
                mm[self.rng.choice(pos_viable[col], max(0, min(int(num_errors), len(pos_viable[col])) - 1), replace=False)]=1
            mask[col] = mm
        return mask

    def inject_errors(self):
        _tmp_df = self.df_clean.copy(deep=True)
        error_count = {column: 0 for column in _tmp_df.columns}
        # eff = dict(zip(self.target_columns, self.error_fraction_list))

        if self.strategy == 'simple':
            mask = self._get_mask_simple()
        elif self.strategy == 'blackout':
            mask = self._get_mask_blackout()
        elif self.strategy == 'burst':
            mask = self._get_mask_burst()
        elif self.strategy == 'high_freq':
            mask = self._get_mask_high_freq_bias()
        elif self.strategy == 'low_freq':
            mask = self._get_mask_low_freq_bias()
        else:
            raise ValueError(f'Unknown strategy {self.strategy}.')

        for col in self.target_columns:
            m = mask[col]
            _tmp_df[col], injected = self._column_error_injection(_tmp_df[col], m, self.na_value)
            error_count[col] = injected
            # print(f'Column {col}: {injected} errors in {len(_tmp_df[col])} values.')

        print(f'Dataset: {self.df_name}')
        print(f'Total injected errors: {_tmp_df.isna().sum().sum()}')
        print(f'Error fraction: {_tmp_df.isna().sum().sum()/len(_tmp_df.values.ravel())*100:.2f} %')

        self.df_dirty = _tmp_df
        return _tmp_df

    def write_dirty_dataset_on_file(self, tag=None, tag_error_frac=True):
        if self.target_all_columns:
            output_name = f'{self.df_name}_allcolumns'
        else:
            output_name = f'{self.df_name}_{"_".join(self.target_columns)}'
        s_tag = ''
        # Add error fraction in simple case
        if tag_error_frac and self.simple:
            s_tag += f'_{self.error_fraction * 100:02.0f}'
        # Add all error fractions.
        else:
            s_tag += f'_{"_".join(self.error_fraction_list) * 100:g}'
        if self.strategy == 'blackout':
            s_tag += '_blackout'
        elif self.strategy == 'burst':
            s_tag += '_burst'
        elif self.strategy == 'high_freq':
            s_tag += '_highfreq'
        elif self.strategy == 'low_freq':
            s_tag += '_lowfreq'
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
    df_name = 'mammogram_debug'
    # df_clean_path = osp.join(osp.realpath(CLEAN_DS_FOLDER), df_name+'.csv')
    df_clean_path = osp.join(f'../{CLEAN_DS_FOLDER}', df_name+'.csv')

    ei = ErrorInjector(df_name, df_clean_path, error_fraction=0.20, strategy='low_freq')
    ei.inject_errors()
    ei.write_dirty_dataset_on_file()