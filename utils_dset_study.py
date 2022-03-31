import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from collections import Counter
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from copy import deepcopy


class DatasetCase:
    @staticmethod
    def _clean_str(val):
        if val != val:
            # value is nan
            return np.nan
        else:
            try:
                return str(val).split('_', maxsplit=1)[1]
            except IndexError:
                return str(val)

    def __init__(self, path, tgt_dir, convert_columns=None, df=None, verbose=False):
        self.path = path
        self.name, _ = os.path.splitext(path)
        self.imputation_method = 'ground-truth'
        self.computed_fraction_missing = 0.0
        if verbose:
            print(f'Adding dataset {self.name}')
        # Threshold for null hypotesis in the chi2 test.
        self.null_hyp = 0.05
        # Quantile to define "rare" and "frequent" values.
        # Values with frequency <= 1-self.q_rare are considered as "rare" values.
        # Values with frequency > self.q_rare are considered as "frequent" values.
        self.q_rare = .9
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(osp.join(tgt_dir, self.path), engine='python')
            for i, col in enumerate(self.df.columns):
                self.df[col] = self.df[col].apply(self._clean_str)
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')

        self.columns = self.df.columns
        self.numerical_columns = self.df.select_dtypes(include='number').columns.to_list()
        self.categorical_columns = [col for col in self.columns if col not in self.numerical_columns]
        if convert_columns is not None:
            for col in convert_columns:
                self.numerical_columns.remove(col)
                self.categorical_columns.append(col)
                self.df[col] = self.df[col].astype(str)
                self.df[col] = self.df[col].replace('nan', np.nan)
        for col in self.categorical_columns:
            self.df[col] = self.df[col].str.lower()
        self.num_rows, self.num_columns = self.df.shape

        self.frequencies_by_col = {col: None for col in self.columns}
        self.redundacy_by_col = {col: None for col in self.columns}
        self.unique_by_col = {col: None for col in self.columns}
        self.quantile50 = {col: None for col in self.columns}

        self.avg_redundancy = 0
        self.all_frequencies_by_value = None

        self.stat_by_col()
        self.overall_stats()
        self.find_rare_values()

    def stat_by_col(self):
        for col in self.columns:
            self.frequencies_by_col[col] = self.df[col].value_counts()
            self.unique_by_col[col] = self.df[col].nunique()
            self.redundacy_by_col[col] = self.df[col].value_counts().mean()
            self.quantile50[col] = self.frequencies_by_col[col].quantile(0.50)

    def overall_stats(self):
        counts_all = Counter(self.df.values.ravel())
        counts_cat = Counter(self.df[self.categorical_columns].values.ravel())
        self.all_frequencies_by_value = pd.Series(counts_all)
        self.cat_frequencies_by_value = pd.Series(counts_cat)
        self.avg_redundancy = self.all_frequencies_by_value.mean()
        self.normalized_redundancy = self.avg_redundancy/len(self.all_frequencies_by_value)

    def find_rare_values(self, plot=True):
        self.q_rare = 0.9
        self.frequent_values = dict()
        self.rare_values = dict()
        self.count_frequent_values = {col: 0 for col in self.categorical_columns}
        self.count_rare_values = {col: 0 for col in self.categorical_columns}
        for col in self.categorical_columns:
            k, v = col, self.frequencies_by_col[col]
            # print(f'Column {k} unique values: {len(v):>8}')
            vc = self.frequencies_by_col[col]
            # Upper quantile. Values with freq>up_qle are "frequent"
            up_qle = vc.quantile(self.q_rare)
            # Lower quantile. Values with freq>up_qle are "rare"
            low_qle = vc.quantile(1 - self.q_rare)

            self.frequent_values[col] = vc[vc > up_qle].index.tolist()
            self.rare_values[col] = vc[vc < low_qle].index.tolist()
            self.count_frequent_values[col] = sum(vc[vc > up_qle])
            self.count_rare_values[col] = sum(vc[vc < low_qle])

            # print(f'{self.q_rare*100}% Quantile: {up_qle:.0f}')
            # print(vc[vc>up_qle])
            # Out of all unique values in the column, what fraction is taken by the values with frequency > up_qle
            # print(f'Fraction of frequent values: {sum(vc[vc>up_qle])/sum(vc):.2f}')
        # if plot:
        #     self.plot_freq_histogram()

    def print_stats(self, by_col=False):
        # Overall stats
        print(f'Dataset name:{self.name}')
        print(f'Rows: {self.num_rows} - Columns: {self.num_columns}')

        print('Quantiles. x% of values have frequency lower than the x-th quantile.')
        print(f'20% quantile: {self.all_frequencies_by_value.quantile(.2):.1f}')
        print(f'50% quantile: {self.all_frequencies_by_value.quantile(.5):.1f}')
        print(f'80% quantile: {self.all_frequencies_by_value.quantile(.8):.1f}')
        print(f'90% quantile: {self.all_frequencies_by_value.quantile(.9):.1f}')
        print(f'Average redundancy over the full dataset: {self.avg_redundancy:.2f}')

        # Stats by column
        if by_col:
            print(f'{"Column":<20}{"Uniques":>8}{"Avg freq":>10}{"Q50":>10}')

            for col in self.columns:
                k, v = col, self.frequencies_by_col[col]
                print(f'{k:<20}{len(v):>8}{self.redundacy_by_col[col]:>10.2f}{self.quantile50[col]:>10.2f}')

    def get_stat_by_col(self, col, stat):
        if stat == 'frequency':
            return self.frequencies_by_col[col]

    def compute_chi2_contingency(self, plot_heatmap=True):
        # Extract all the pairs of columns to compute chi2 over. They will be summarized by a heatmap.
        combs = [(x, y) for x in self.categorical_columns for y in self.categorical_columns]

        # Prepare a square dataframe to hold the p term in the chi2 contingency.
        chi2_map = pd.DataFrame(index=self.categorical_columns, columns=self.categorical_columns)

        for comb in combs:
            x, y = comb
            # Compute the frequency with which each value in the column x appears together with each value in the column y.
            contingency = pd.crosstab(self.df[x], self.df[y])
            from scipy.stats import chi2_contingency
            c, p, dof, exp = chi2_contingency(contingency)
            chi2_map.loc[x, y] = p
            # chi2_map.loc[y,x] = p

        self.chi2_map = chi2_map.astype(float)

        # Select only the pairs of columns that do not satisfy the chi-square null hypothesis => they are not independent of each other
        self.pairs_of_interest = {tuple(sorted([x, y])) for x in self.chi2_map.index for y in self.chi2_map.columns if
                                  chi2_map.loc[x, y] > self.null_hyp}

        if plot_heatmap:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.chi2_map, vmin=self.null_hyp)
            plt.title(f'Chi-Square test on dataset {self.name}')

            plt.show()

    def plot_freq_histogram(self):
        plt.figure(figsize=(10, 9))
        summary = pd.DataFrame({'frequent': self.count_frequent_values, 'rare': self.count_rare_values})
        summary['frequent'] = summary['frequent'] / len(self.df)
        summary['rare'] = summary['rare'] / len(self.df)
        summary = summary.reset_index().melt(id_vars='index', value_vars=['frequent', 'rare'])
        summary.columns = ['index', 'case', 'count']
        g = sns.catplot(data=summary, x='index', y='count', hue='case', kind='bar')
        plt.xticks(rotation=90)
        plt.title(f'Distribution of rare/frequent values. Dataset: {self.imputation_method}')
        # g.tight_layout()
        plt.show()


class DirtyDatasetCase(DatasetCase):
    def __init__(self, name, tgt_dir, convert_columns=None):
        super(DirtyDatasetCase, self).__init__(name, tgt_dir, convert_columns)
        self.imputation_method = 'dirty'
        self.null_values = pd.isna(self.df)
        self.target_columns = self.df.columns[self.null_values.sum(axis=0) > 0]
        self.computed_fraction_missing = pd.isna(self.df).sum().sum() / (self.df.shape[0] * self.df.shape[1])


class ImputedDatasetCase(DatasetCase):
    def __init__(self, name, tgt_dir, convert_columns=None):
        base, ext = osp.splitext(name)
        super(ImputedDatasetCase, self).__init__(name, tgt_dir, convert_columns)
        self.imputation_method = base.rsplit('_', maxsplit=1)[1]

    def reconstruct_hc(self, df_path):
        df = pd.read_csv(df_path)
        columns = df['attribute'].unique()
        df_reconstructed = pd.DataFrame(columns=columns)

        for idx, row in df.iterrows():
            df_reconstructed.loc[row['tid'], row['attribute']] = row['inferred_val']

        return df_reconstructed


class Dataset:

    def __init__(self, dataset_name, convert_columns=None):
        self.dset_name = dataset_name
        self.datasets = {'orig': [],'dirty': [], 'imp': []}
        self.correct_imputations_by_col = {}
        self.total_imputations_by_col = {}
        self.null_values = None
        self.convert_columns = convert_columns

    def add_dataset(self, case, dataset_name, tgt_dir):
        new_ds = None
        if case == 'dirty':
            if len(self.datasets[case])>0:
                print('Only 1 dirty dataset allowed per run. Skipping insertion.')
                # raise  ValueError('Only 1 dirty dataset allowed per run.')
            else:
                new_ds = DirtyDatasetCase(dataset_name, tgt_dir, self.convert_columns)
                self.target_columns = new_ds.target_columns
                self.null_values = new_ds.null_values
        elif case == 'imp':
            new_ds = ImputedDatasetCase(dataset_name, tgt_dir, self.convert_columns)
        else:
            if len(self.datasets[case])>0:
                raise  ValueError('Only 1 clean dataset allowed per run.')
            new_ds = DatasetCase(dataset_name, tgt_dir, self.convert_columns)
        self.datasets[case].append(new_ds)
        self.categorical_columns = self.datasets['orig'][0].categorical_columns
        return new_ds


    def get_all_datasets(self):
        ds_list = []
        for case, clist in self.datasets.items():
            for ds in clist:
                ds_list.append(ds)
        return ds_list

    def print_all_stats(self):
        for ds in self.get_all_datasets():
            ds.print_stats()

    def get_dataset(self, case, idx=0) -> DatasetCase:
        '''

        :param case:
        :return: DatasetCase
        '''
        return self.datasets[case][idx]

    def compare_clean_dirty(self, df_orig=None, df_dirty=None):
        '''
        This function checks
        :return:
        '''
        assert 'orig' in self.datasets
        assert 'dirty' in self.datasets

        assert self.get_dataset('orig').df.shape == self.get_dataset('dirty').df.shape

        df_orig = self.get_dataset('orig')
        df_dirty = self.get_dataset('dirty')

        print('Unique values by column')
        _tmp = {
            'clean':    df_orig.unique_by_col,
            'dirty':    df_dirty.unique_by_col
        }

        counts_df = pd.DataFrame(data=_tmp, columns=['clean', 'dirty'])
        counts_df['diff'] = counts_df['dirty'] - counts_df['clean']
        print(counts_df)

        impossible = {k: None for k in self.target_columns}
        affected_rows = {k: 0 for k in self.target_columns}
        for col in self.target_columns:
            unq_dirty = df_dirty.df[col].unique()
            unq_clean = df_orig.df[col].unique()
            impossible[col] = [_ for _ in unq_clean if _ not in unq_dirty]
            affected_rows[col] = len(df_orig.df[col].loc[df_orig.df[col].isin(impossible[col])])
            print(f'Column {col} has affected rows = {affected_rows[col]}')

    def measure_imputation_accuracy(self, df_imp):
        assert 'orig' in self.datasets
        assert 'dirty' in self.datasets
        assert 'imp' in self.datasets

        df_orig = self.get_dataset('orig').df
        df_dirty = self.get_dataset('dirty').df

        self.total_imputations_by_col = {col:0 for col in self.target_columns}
        self.correct_imputations_by_col = {col:0 for col in self.target_columns}

        for idx, row in self.null_values.iterrows():
            for col in self.target_columns:
                if pd.isna(df_dirty.loc[idx, col]):
                    true_value = df_orig.loc[idx, col]
                    imputed_value = df_imp.loc[idx, col]
                    if true_value == imputed_value:
                        self.correct_imputations_by_col[col] += 1
                    self.total_imputations_by_col[col] += 1

        acc_dict = {col: self.correct_imputations_by_col[col]/self.total_imputations_by_col[col] for col in self.target_columns}

        header = f'{"Column":^30}{"Score":^16}{"Correct":^8}{"Tot miss":^8}'
        print(header)
        for col in acc_dict:
            s = f'{col:^30}{acc_dict[col]:>16.4}{self.correct_imputations_by_col[col]:^8}{self.total_imputations_by_col[col]:^8}'
            print(s)
        acc = sum(list(self.correct_imputations_by_col.values()))/sum(list(self.total_imputations_by_col.values()))
        print(f'Imputation accuracy: {acc:.4f}')

    def compare_distribution_of_values(self, plot=True):
        full_summary = pd.DataFrame(columns=['column', 'case', 'count', 'dataset'])
        for idx, dataset in enumerate(self.get_all_datasets()):
            summary = pd.DataFrame({'frequent': dataset.count_frequent_values, 'rare': dataset.count_rare_values})
            summary['frequent'] = summary['frequent']/len(dataset.df)
            summary['rare'] = summary['rare']/len(dataset.df)
            summary = summary.reset_index().melt(id_vars='index', value_vars=['frequent', 'rare'])
            summary.columns = ['column', 'case', 'count']
            summary['dataset'] = dataset.imputation_method
            full_summary = full_summary.append(summary)
        if plot:
            plt.figure(figsize=(16,8))
            g = sns.catplot(kind='bar',data=full_summary, x='column', y='count', hue='dataset', col='case', sharey=False)
            ax1=g.axes_dict['frequent']
            ax2=g.axes_dict['rare']
            ax1.tick_params(labelrotation=90)
            ax2.tick_params(labelrotation=90)
            g.tight_layout()

            plt.show()

    def plot_all_freq_histograms(self):
        for dataset in self.get_all_datasets():
            dataset.plot_freq_histogram()


    def extract_imputations(self, plot=True):
        correct_imputations = {k:None for k in self.target_columns}
        incorrect_imputations = {k:None for k in self.target_columns}
        df_orig = self.datasets['orig'][0].df
        ds_imputed = self.datasets['imp']

        summary_all_columns = pd.DataFrame()
        selected_columns = ['Risk1Yr', 'PRE8', 'PRE30']
        # selected_columns = self.categorical_columns[:]
        for col in selected_columns:
            full_summary = pd.DataFrame(columns=['total', 'correct', 'wrong', 'ratio','frequency', '1-frequency', 'error2'])
            true_imputations = df_orig[col].loc[self.null_values[col]]
            true_imputations_by_value = df_orig.loc[self.null_values[col], col].value_counts()
            freqs = pd.DataFrame(columns=['frequency', 'ratio', 'dataset'])
            freqs['frequency'] = true_imputations_by_value/len(true_imputations)
            freqs['ratio'] = 1-freqs['frequency']
            freqs['dataset'] = freqs['dataset'].fillna('expected')

            for ds_imp in ds_imputed:
                df_imp = ds_imp.df
                imputations = df_imp[col].loc[self.null_values[col]]
                v_correct = (df_imp[col].loc[self.null_values[col]] == df_orig.loc[self.null_values[col], col])
                v_incorrect = (df_imp[col].loc[self.null_values[col]] != df_orig.loc[self.null_values[col], col])

                correct_imputations[col] = imputations[v_correct].index
                incorrect_imputations[col] = imputations[v_incorrect].index
                correct_imputations_by_value = df_orig.loc[self.null_values[col], col].loc[v_correct].value_counts()
                incorrect_imputations_by_value = df_orig.loc[self.null_values[col], col].loc[v_incorrect].value_counts()
                # incorrect_imputations_by_value = df_orig.iloc[incorrect_imputations[col]].value_counts(col)
                frac_wrong = incorrect_imputations_by_value/true_imputations_by_value
                dd = pd.DataFrame(columns=['dataset','total', 'correct', 'wrong', 'ratio', 'ratio_correct', 'frequency', '1-frequency', 'error2'])
                dd['total'] = true_imputations_by_value
                dd['correct'] = correct_imputations_by_value
                dd['wrong'] = incorrect_imputations_by_value
                dd['ratio'] = frac_wrong
                dd['ratio_correct'] = 1-frac_wrong
                dd = dd.fillna(0)
                dd['frequency'] = true_imputations_by_value/len(true_imputations)
                dd['1-frequency'] = 1-dd['frequency']
                dd['error2'] = (dd['ratio'] - dd['1-frequency'])**2
                dd['dataset'] = ds_imp.imputation_method
                # print(dd)
                full_summary = full_summary.append(dd)

            if len(full_summary) < 20 and plot:
                freqs_melted = pd.melt(freqs.reset_index(), id_vars=['index', 'dataset'], value_vars=['ratio'])
                full_summary_melted = pd.melt(full_summary.reset_index(), id_vars=['index', 'dataset'], value_vars=['ratio'])
                concat = pd.concat([freqs_melted, full_summary_melted])
                concat['column'] = col
                with sns.plotting_context('talk', font_scale=1.5):
                    fig = plt.figure(figsize=(10,8))
                    ax = fig.gca()
                    sns.barplot(ax=ax,data=concat, x='index', y='value', hue='dataset').set(
                        ylabel='% of incorrect imputations', xlabel='Attribute value',title=col
                    )
                    plt.tight_layout()
                    plt.show()
                # ax  = sns.catplot(kind='bar',data=concat, x='index', y='value', hue='dataset')
                # plt.xticks(rotation=45)
                # # ax.legend(facecolor='white')
                # plt.show()
                summary_all_columns = pd.concat([summary_all_columns, concat])
            print(full_summary)

        print()
        with sns.plotting_context('talk', font_scale=1.5):

            g = sns.catplot(data=summary_all_columns, x='index', y='value', hue='dataset', col='column',
                            kind='bar', palette=sns.color_palette('tab10'), height=7)
            plt.tight_layout()
            plt.show()



def measure_imputation_accuracy(ds_orig: DatasetCase, ds_dirty: DirtyDatasetCase, ds_imp: ImputedDatasetCase, verbose=False):
    target_columns = ds_dirty.target_columns

    total_imputations_by_col = {col:0 for col in target_columns}
    correct_imputations_by_col = {col:0 for col in target_columns}

    df_orig = ds_orig.df
    df_dirty = ds_dirty.df
    df_imp = ds_imp.df

    num_imp_true = {col:[] for col in ds_orig.numerical_columns}
    num_imp_pred = {col:[] for col in ds_orig.numerical_columns}

    did_not_predict = []

    for idx, row in ds_dirty.null_values.iterrows():
        for col in ds_orig.categorical_columns:
            if pd.isna(df_dirty.loc[idx, col]):
                true_value = df_orig.loc[idx, col]
                imputed_value = df_imp.loc[idx, col]
                if true_value == imputed_value:
                    correct_imputations_by_col[col] += 1
                total_imputations_by_col[col] += 1
        for col in ds_orig.numerical_columns:
            if pd.isna(df_dirty.loc[idx, col]):
                if pd.isna(df_imp.loc[idx, col]):
                    did_not_predict.append([idx,col])
                    continue
                num_imp_true[col].append(df_orig.loc[idx, col])
                num_imp_pred[col].append(df_imp.loc[idx, col])

    acc_dict = {}
    for col in ds_orig.categorical_columns:
        if col in target_columns:
            acc_dict[col] = correct_imputations_by_col[col]/total_imputations_by_col[col]
        else:
            acc_dict[col] = 0
    for col in ds_orig.numerical_columns:
        if col in target_columns:
            acc_dict[col] = mean_squared_error(num_imp_true[col], num_imp_pred[col], )
        else:
            acc_dict[col] = 0

    if len(ds_orig.categorical_columns) > 0:
        acc = sum(list(correct_imputations_by_col.values()))/sum(list(total_imputations_by_col.values()))
    else:
        acc = 0
    if len(ds_orig.numerical_columns) > 0:
        average_rmse = sum([acc_dict[col] for col in ds_orig.numerical_columns]) / len(ds_orig.numerical_columns)
    else:
        average_rmse = 0
    res_dict = {
        'ds_name': ds_dirty.name,
        'imputation_method': ds_imp.imputation_method,
        'frac_missing': ds_dirty.computed_fraction_missing,
        'avg_acc': acc,
        'avg_rmse': average_rmse,
        'acc_dict': acc_dict
    }
    if verbose:
        header = f'{"Column":^30}{"Score":^16}{"Correct":^8}{"Tot miss":^8}'
        print(header)
        for col in acc_dict:
            if col in ds_orig.categorical_columns:
                s = f'{col:^30}{acc_dict[col]:>16.4}{correct_imputations_by_col[col]:^8}{total_imputations_by_col[col]:^8}'
            else:
                s = f'{col:^30}{acc_dict[col]:>16.4}{"0":^8}{total_imputations_by_col[col]:^8}'
            print(s)
        print(f'Imputation accuracy: {acc:.4f}')

    return res_dict


if __name__ == '__main__':
    dataset_name = 'adultsample10'
    clean_dir = 'data/clean'
    dirty_dir = 'data/dirty/'
    imp_dir = 'data/imputed'

    datasets = ['adultsample10',
            'australian',
            'contraceptive',
            'credit',
            'flare',
            # 'fodorszagats',
            'imdb',
            'mammogram',
            'tax5000trimmed',
            'thoracic',
            # 'tictactoe'
            ]

    labels = ['Adult',
                'Australian',
                'Contraceptive',
                'Credit',
                'Flare',
                # 'fodorszagats',
                'IMDB',
                'Mammogram',
                'Tax',
                'Thoracic',
                # 'tictactoe'
                ]

    # ds = Dataset(dataset_name)
    # ds.add_dataset('orig', f'{dataset_name}.csv', clean_dir)
    # ds.add_dataset('dirty', f'{dataset_name}_allcolumns_50.csv', dirty_dir)

    # ds.print_all_stats()
    fig = plt.figure(figsize=(8, 6))

    ax = plt.gca()
    dset_info = {
           'name': [],
           'num_rows': [],
           'tot_columns': [],
           'cat_columns': [],
           'num_columns': [],
           'dist_values': [],
           'avg_redundancy': [],
           'norm_redundancy': [],
       }

    clean_datasets = {}
    for _, dataset_name in enumerate(datasets):
        dset_clean = DatasetCase(dataset_name+'.csv', tgt_dir=clean_dir)
        clean_datasets[dataset_name] = dset_clean
        dset_info['name'].append(dataset_name)
        dset_info['num_rows'].append(dset_clean.num_rows)
        dset_info['tot_columns'].append(dset_clean.num_columns)
        dset_info['cat_columns'].append(len(dset_clean.categorical_columns))
        dset_info['num_columns'].append(len(dset_clean.numerical_columns))
        dset_info['dist_values'].append(len(dset_clean.all_frequencies_by_value))
        dset_info['avg_redundancy'].append(dset_clean.avg_redundancy)
        dset_info['norm_redundancy'].append(dset_clean.normalized_redundancy)

    df_info = pd.DataFrame().from_dict(dset_info)

    df_info.to_csv('dfinfo.csv', index=False)

    for didx, dataset_name in enumerate(datasets):
        print(dataset_name)
        dset_clean = DatasetCase(dataset_name+'.csv', tgt_dir=clean_dir)
        dirty_dsets = {}
        dirty_dsets['00'] = dset_clean
        for frac in ['05', '20', '50']:
            dirty_dsets[frac] = DirtyDatasetCase(f'{dataset_name}_allcolumns_{frac}.csv', tgt_dir=dirty_dir)

        dict_avg_redundancy =  {}
        dict_norm_redundancy =  {}
        for key, dset in dirty_dsets.items():
            dict_avg_redundancy[key] = dset.avg_redundancy
            dict_norm_redundancy[key] = dset.normalized_redundancy
            print(key, dset.avg_redundancy, dset.computed_fraction_missing)

        # plt.plot(list(dict_norm_redundancy.values()), label=dataset_name)
        ax.plot(list(dict_avg_redundancy.values()), label=labels[didx])
        # ax.xticks(rotation=90)
        plt.xticks(range(len(dict_avg_redundancy.keys())), [int(x) for x in dict_avg_redundancy.keys()])
        # ax.set_xticks(range(len(dict_avg_redundancy.keys())),list(dict_avg_redundancy.keys()))
        # plt.show()
    plt.legend()
    plt.xlabel('Error fraction')
    plt.ylabel('Average redundancy')
    plt.title('Average redundancy as function of the error fraction')
    plt.tight_layout()
    fig.savefig('avg_redundancy.png')
    # plt.show()



print()