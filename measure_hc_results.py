import pandas as pd
import os.path as osp

def measure_accuracy(df_true: pd.DataFrame, df_pred):
    correct_imputations =0
    total_imputations = 0

    correct_by_col = {col: 0 for col in df_true.columns}
    total_by_col = {col: 0 for col in df_true.columns}

    numerical_columns = df_true.select_dtypes(include='number')

    for idx, row in df_pred.iterrows():
        ridx = row['tid']
        col = row['attribute']
        df_true = df_true.convert_dtypes(convert_string=True)
        if df_true[col].dtype not in numerical_columns:
            try:
                true_value = df_true.loc[ridx, col].lower()
                pred_value = row['inferred_val']
            except AttributeError:
                continue
                true_value = df_true.loc[ridx, col]
                pred_value = row['inferred_val']
        if pred_value == true_value:
            correct_imputations += 1
            correct_by_col[col] += 1
        total_imputations += 1
        total_by_col[col] += 1

    for col in df_true.columns:
        if total_by_col[col]>0:
            print(f'Column {col}: {correct_by_col[col]} correct over {total_by_col[col]} imputations. => {correct_by_col[col]/total_by_col[col]}')
    print(f'Imputation precision: {correct_imputations/total_imputations:.4f} over {total_imputations} imputations.')
    print(f'Correct categorical imputations: {correct_imputations}.')
    print(f'Total categorical imputations: {total_imputations}.')

def reconstruct(df_dirty, df_pred):
    df_rec = df_dirty.copy()
    for idx,r  in df_pred.iterrows():
        row, col = r[['tid', 'attribute']]
        df_rec.loc[row,col] = r['inferred_val']
    return df_rec

if __name__ == '__main__':
    df_name = 'bikes-dekho'

    df_true_path = f'data/holoclean/{df_name}/{df_name}.csv'
    df_pred_path = f'data/holoclean/{df_name}/{df_name}_20_pred.csv'
    df_dirty_path = f'data/holoclean/{df_name}/{df_name}_all_columns_20.csv'

    print(df_pred_path)

    df_true = pd.read_csv(df_true_path)
    df_pred = pd.read_csv(df_pred_path)
    df_dirty = pd.read_csv(df_dirty_path)

    all_missing_values = set()

    for idx, row in df_dirty.iterrows():
        for col in df_dirty.columns:
            if pd.isna(df_dirty.loc[idx, col]):
                all_missing_values.add((idx, col))

    pred_values = df_pred[['tid', 'attribute']].values.tolist()
    pred_values = set([tuple(_) for _ in pred_values])
    intersection = list(all_missing_values.intersection(pred_values))

    # df_rec = reconstruct(df_dirty, df_pred)
    df_pred_reduced = df_pred.set_index(['tid', 'attribute']).loc[intersection].reset_index()
    measure_accuracy(df_true,df_pred_reduced)
    # df_rec_cat = df_rec.convert_dtypes(convert_integer=False).select_dtypes(exclude='object', include='string')
