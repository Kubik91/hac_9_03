import os
import glob
import re
import subprocess
from multiprocessing import Process
from typing import Tuple, Optional

import pandas as pd
from tqdm import tqdm

import gensim
from gensim import corpora
from pprint import pprint
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score
from sklearn.metrics import accuracy_score
from catboost.utils import eval_metric


def my_full_cvs(path, save_filename):
    if os.path.isfile(os.path.join(path, save_filename)):
        combined = pd.read_csv(os.path.join(path, save_filename), low_memory=False)
    else:
        dataframes = []
        extension = 'csv'
        all_filenames = [i for i in glob.glob(f'{path}/*.{extension}')]
        for f in sorted(all_filenames):
            df = pd.read_csv(f, low_memory=False)
            year = re.search(rf'\d{{4}}(?=\.{extension})', f)
            if year:
                df['year'] = year.group(0)
                dataframes.append(df)
        combined = pd.concat(dataframes, ignore_index=True)
        combined.to_csv(os.path.join(path, save_filename), index=False)
    return combined


def get_coords(coords: str) -> Tuple[Optional[float], Optional[float]]:
    '''
    Разбирает строковое представление координат

    :param coords: сроковое представление координат
    :return tuple кортеж из полученных координат
    '''
    regex = re.search('(?P<lang>\d+\.?\d*)\s(?P<long>\d+\.?\d*)', coords)
    return regex.group('lang'), regex.group('long')


def get_train_full():
    print('----------------------------')
    df_train_full = my_full_cvs("dvhb_data/train", "train_full.csv")
    df_train_full_new_names = ['CODE_CULT', 'CODE_GROUP', 'CENTROID', 'YEAR']
    df_train_full.columns = df_train_full_new_names
    print('------------1---------------')
    df_train_grouped_full = df_train_full.groupby('CENTROID', dropna=False).agg(list)
    columns = [f'{column}_{year}' for column in df_train_full.columns.drop(['CENTROID', 'YEAR']) for year in
               df_train_full['YEAR'].unique()] + ['LATITUDE', 'LONGTITUDE']
    del df_train_full
    print('------------2---------------')
    new_df_train = pd.DataFrame(columns=columns)
    new_df_train.to_csv("dvhb_data/train/grouped_train_full.csv")

    for row in tqdm(df_train_grouped_full.iterrows(), total=len(df_train_grouped_full)):
        if len(new_df_train) == 1000:
            new_df_train.to_csv("dvhb_data/train/grouped_train_full.csv", mode='a', header=False, columns=columns)
            del new_df_train
            new_df_train = pd.DataFrame(columns=columns)
        regex = re.search('(?P<lat>\d+\.?\d*)\s(?P<long>\d+\.?\d*)', row[0])
        if regex:
            new_df_train = new_df_train.append(
                pd.Series(
                    {**{f'{column}_{year}': row[1][column][i] for column in row[1].keys().drop('YEAR') for i, year in
                        enumerate(row[1]['YEAR'])},
                     **{'LATITUDE': regex.group('lat'), 'LONGTITUDE': regex.group('long')}},
                    name=row[0])
            )
    else:
        new_df_train.to_csv("dvhb_data/train/grouped_train_full.csv", mode='a', header=False, columns=columns)
    print('train_full complete')


def get_test(year):
    print('----------------------------')
    df_train_full = my_full_cvs(f"dvhb_data/test/test {year}", "test_full.csv")
    df_train_full_new_names = ['CODE_CULT', 'CODE_GROUP', 'CENTROID', 'YEAR']
    df_train_full.columns = df_train_full_new_names
    print('------------1---------------')
    df_train_grouped_full = df_train_full.groupby('CENTROID', dropna=False).agg(list)
    columns = [f'{column}_{year}' for column in df_train_full.columns.drop(['CENTROID', 'YEAR']) for year in
               df_train_full['YEAR'].unique()] + ['LATITUDE', 'LONGTITUDE']
    del df_train_full
    print('------------2---------------')
    new_df_train = pd.DataFrame(columns=columns)
    new_df_train.to_csv(f"dvhb_data/test/test {year}/grouped_full.csv")

    for row in tqdm(df_train_grouped_full.iterrows(), total=len(df_train_grouped_full)):
        if len(new_df_train) == 1000:
            new_df_train.to_csv(f"dvhb_data/test/test {year}/grouped_full.csv", mode='a', header=False, columns=columns)
            del new_df_train
            new_df_train = pd.DataFrame(columns=columns)
        regex = re.search('(?P<lat>\d+\.?\d*)\s(?P<long>\d+\.?\d*)', row[0])
        if regex:
            new_df_train = new_df_train.append(
                pd.Series(
                    {**{f'{column}_{year}': row[1][column][i] for column in row[1].keys().drop('YEAR') for i, year in
                        enumerate(row[1]['YEAR'])},
                     **{'LATITUDE': regex.group('lat'), 'LONGTITUDE': regex.group('long')}},
                    name=row[0])
            )
    else:
        new_df_train.to_csv(f"dvhb_data/test/test {year}/grouped_full.csv", mode='a', header=False, columns=columns)
    print('train_full complete')


def get_train_cult_full():
    print('============================')
    df_train_cult_full_new_names = ['CODE_GROUP', 'NUM_ILOT', 'CENTROID', 'YEAR']
    df_train_cult_full = my_full_cvs("dvhb_data/raw_data", "train_cult_full.csv").iloc[:, : 4]
    df_train_cult_full.columns = df_train_cult_full_new_names
    df_train_cult_full = df_train_cult_full[['CODE_GROUP', 'CENTROID', 'YEAR']]
    df_train_cult_full.to_csv('train_cult_full_finish.csv')

    print('============1===============')
    df_train_grouped_full = df_train_cult_full.groupby('CENTROID', dropna=False).agg(list)
    columns = [f'{column}_{year}' for column in df_train_cult_full.columns.drop(['CENTROID', 'YEAR']) for year in
               df_train_cult_full['YEAR'].unique()] + ['LATITUDE', 'LONGTITUDE']
    del df_train_cult_full
    print('============2===============')
    new_df_train = pd.DataFrame(columns=columns)
    new_df_train.to_csv("dvhb_data/train/grouped_train_cult_full.csv")

    for row in tqdm(df_train_grouped_full.iterrows(), total=len(df_train_grouped_full)):
        if len(new_df_train) == 1000:
            new_df_train.to_csv("dvhb_data/train/grouped_train_cult_full.csv", mode='a', header=False, columns=columns)
            del new_df_train
            new_df_train = pd.DataFrame(columns=columns)
        regex = re.search('(?P<lat>\d+\.?\d*)\s(?P<long>\d+\.?\d*)', row[0])
        if regex:
            new_df_train = new_df_train.append(
                pd.Series(
                    {**{f'{column}_{year}': row[1][column][i] for column in row[1].keys().drop('YEAR') for i, year in
                        enumerate(row[1]['YEAR'])},
                     **{'LATITUDE': regex.group('lat'), 'LONGTITUDE': regex.group('long')}},
                    name=row[0])
            )
    else:
        new_df_train.to_csv("dvhb_data/train/grouped_train_cult_full.csv", mode='a', header=False, columns=columns)

    print('train_cult_full complete')


def fit_model():
    # кодирую слова векторами
    if os.path.isfile('cult_token.txtdic'):
        dictionary = corpora.Dictionary.load('cult_token.txtdic')
    else:
        df_train_full = my_full_cvs("dvhb_data/train", "train_full.csv")
        df_train_full_new_names = ['CODE_CULT', 'CODE_GROUP', 'CENTROID', 'YEAR']
        df_train_full.columns = df_train_full_new_names
        text = [df_train_full['CODE_CULT'].tolist()]

        dictionary = corpora.Dictionary(text)
        dictionary.save('cult_token.txtdic')

    grouped_df_train = pd.read_csv("dvhb_data/train/grouped_train_full.csv", index_col=0)
    df_permanent = grouped_df_train[(grouped_df_train['CODE_CULT_2015'] == grouped_df_train['CODE_CULT_2016']) & (
                grouped_df_train['CODE_CULT_2015'] == grouped_df_train['CODE_CULT_2017']) & (
                                                grouped_df_train['CODE_CULT_2015'] == grouped_df_train[
                                            'CODE_CULT_2018']) & (
                                                grouped_df_train['CODE_CULT_2015'] == grouped_df_train[
                                            'CODE_CULT_2019'])]
    df_two_year = grouped_df_train[(grouped_df_train['CODE_CULT_2015'] == grouped_df_train['CODE_CULT_2016']) & (
                grouped_df_train['CODE_CULT_2017'] == grouped_df_train['CODE_CULT_2018']) & (
                                               grouped_df_train['CODE_CULT_2015'] != grouped_df_train[
                                           'CODE_CULT_2018']) & (grouped_df_train['CODE_CULT_2019'] != grouped_df_train[
        'CODE_CULT_2018']) & ~grouped_df_train.index.isin(df_permanent.index)]
    # Пашин датасет, оставлю только растения и место
    df_test_swetlana = grouped_df_train[~(grouped_df_train.index.isin(df_permanent.index) | (grouped_df_train.index.isin(df_two_year.index)))][
        ['CODE_CULT_2015', 'CODE_CULT_2016', 'CODE_CULT_2017', 'CODE_CULT_2018', 'CODE_CULT_2019', 'LATITUDE',
         'LONGTITUDE']].copy()
    df_test_swetlana = df_test_swetlana.sample(n=int(len(df_test_swetlana)*0.40))
    print(len(df_test_swetlana))

    # заменяем значения в столбце object_name_n на данные из словаря, а ключи берем из столбца object_type_number
    df_test_swetlana['CODE_CULT_2019'] = df_test_swetlana['CODE_CULT_2019'].map(dictionary.token2id)
    df_test_swetlana['CODE_CULT_2018'] = df_test_swetlana['CODE_CULT_2018'].map(dictionary.token2id)
    df_test_swetlana['CODE_CULT_2017'] = df_test_swetlana['CODE_CULT_2017'].map(dictionary.token2id)
    df_test_swetlana['CODE_CULT_2016'] = df_test_swetlana['CODE_CULT_2016'].map(dictionary.token2id)
    df_test_swetlana['CODE_CULT_2015'] = df_test_swetlana['CODE_CULT_2015'].map(dictionary.token2id)

    df_test_swetlana.rename(columns={f'CODE_CULT_{2015+i}': f'{i+1}' for i in range(5)}, inplace=True)
    # делим
    X = df_test_swetlana.drop(['5'], axis=1)
    y = df_test_swetlana['5']

    features_train, features_valid, target_train, target_valid = train_test_split(
        X, y, test_size=0.25, random_state=12345)
    print('-------1-------')

    model = CatBoostClassifier(verbose=100,
                               learning_rate=0.7,
                               early_stopping_rounds=200,
                               eval_metric='AUC',
                               task_type="GPU",
                               )

    # count = len(features_train)
    # step = 10000
    # for i in range(0, count, step):
    #     print(f'{i=}')
    #     model.fit(features_train.iloc[i:min(i+step, count)], target_train.iloc[i:min(i+step, count)])
    #     break
    model.fit(features_train, target_train)
    model.save_model('catboostmodel')
    predictions_valid = model.predict(features_valid)

    print(accuracy_score(target_valid, predictions_valid))
    print('------')


def get_predict_2019():
    df_data = pd.read_csv("dvhb_data/test/test 2019/grouped_full.csv", index_col=0)

    # кодирую слова векторами
    if os.path.isfile('cult_token.txtdic'):
        dictionary = corpora.Dictionary.load('cult_token.txtdic')
    else:
        df_train_full = my_full_cvs("dvhb_data/train", "train_full.csv")
        df_train_full_new_names = ['CODE_CULT', 'CODE_GROUP', 'CENTROID', 'YEAR']
        df_train_full.columns = df_train_full_new_names
        text = [df_train_full['CODE_CULT'].tolist()]

        dictionary = corpora.Dictionary(text)
        dictionary.save('cult_token.txtdic')

    # заменяем значения в столбце object_name_n на данные из словаря, а ключи берем из столбца object_type_number
    # df_data['CODE_CULT_2019'] = df_data['CODE_CULT_2019'].map(dictionary.token2id)
    df_data['CODE_CULT_2018'] = df_data['CODE_CULT_2018'].map(dictionary.token2id)
    df_data['CODE_CULT_2017'] = df_data['CODE_CULT_2017'].map(dictionary.token2id)
    df_data['CODE_CULT_2016'] = df_data['CODE_CULT_2016'].map(dictionary.token2id)
    df_data['CODE_CULT_2015'] = df_data['CODE_CULT_2015'].map(dictionary.token2id)

    df_data.rename(columns={f'CODE_CULT_{2015 + i}': f'{i + 1}' for i in range(5)}, inplace=True)

    model = CatBoostClassifier()
    model.load_model("catboostmodel")
    predictions_valid = model.predict(df_data[['1', '2', '3', '4', 'LATITUDE', 'LONGTITUDE']])
    df_data = df_data.assign(CODE_CULT_2019=predictions_valid)

    df_data.rename(columns={f'{i + 1}': f'CODE_CULT_{2015 + i}' for i in range(5)}, inplace=True)
    for row in df_data[(df_data['CODE_CULT_2015'] == df_data['CODE_CULT_2016'])
                       & (df_data['CODE_CULT_2015'] == df_data['CODE_CULT_2017'])
                       & (df_data['CODE_CULT_2015'] == df_data['CODE_CULT_2018'])].iterrows():
        df_data.loc[row[0]]['CODE_CULT_2019'] = row[1]['CODE_CULT_2015']

    df_data['CODE_CULT_2019'] = df_data['CODE_CULT_2019'].map(dictionary.get)
    df_data['CODE_CULT_2018'] = df_data['CODE_CULT_2018'].map(dictionary.get)
    df_data['CODE_CULT_2017'] = df_data['CODE_CULT_2017'].map(dictionary.get)
    df_data['CODE_CULT_2016'] = df_data['CODE_CULT_2016'].map(dictionary.get)
    df_data['CODE_CULT_2015'] = df_data['CODE_CULT_2015'].map(dictionary.get)

    df_data[['CODE_CULT_2015', 'CODE_CULT_2016', 'CODE_CULT_2017', 'CODE_CULT_2018', 'CODE_CULT_2019', 'LATITUDE',
             'LONGTITUDE']].to_csv('predict_2019_full.csv', index=True)
    df_data[['CODE_CULT_2019']].to_csv('predict_2019.csv', index=True)


def get_predict_2020():
    df_data = pd.read_csv("dvhb_data/test/test 2020/grouped_full.csv", index_col=0)

    # кодирую слова векторами
    if os.path.isfile('cult_token.txtdic'):
        dictionary = corpora.Dictionary.load('cult_token.txtdic')
    else:
        df_train_full = my_full_cvs("dvhb_data/train", "train_full.csv")
        df_train_full_new_names = ['CODE_CULT', 'CODE_GROUP', 'CENTROID', 'YEAR']
        df_train_full.columns = df_train_full_new_names
        text = [df_train_full['CODE_CULT'].tolist()]

        dictionary = corpora.Dictionary(text)
        dictionary.save('cult_token.txtdic')

    # заменяем значения в столбце object_name_n на данные из словаря, а ключи берем из столбца object_type_number
    df_data['CODE_CULT_2019'] = df_data['CODE_CULT_2019'].map(dictionary.token2id)
    df_data['CODE_CULT_2018'] = df_data['CODE_CULT_2018'].map(dictionary.token2id)
    df_data['CODE_CULT_2017'] = df_data['CODE_CULT_2017'].map(dictionary.token2id)
    df_data['CODE_CULT_2016'] = df_data['CODE_CULT_2016'].map(dictionary.token2id)
    df_data['CODE_CULT_2015'] = df_data['CODE_CULT_2015'].map(dictionary.token2id)

    df_data.rename(columns={f'CODE_CULT_{2015 + i}': f'{i + 1}' for i in range(6)}, inplace=True)

    model = CatBoostClassifier()
    model.load_model("catboostmodel")
    predictions_valid = model.predict(
        df_data[['2', '3', '4', '5', 'LATITUDE', 'LONGTITUDE']].rename(columns={'2': '1', '3': '2', '4': '3', '5': '4'})
    )
    df_data = df_data.assign(CODE_CULT_2020=predictions_valid)

    df_data.rename(columns={f'{i + 1}': f'CODE_CULT_{2015 + i}' for i in range(6)}, inplace=True)
    df_permanent = df_data[
        (df_data['CODE_CULT_2015'] == df_data['CODE_CULT_2016'])
        & (df_data['CODE_CULT_2015'] == df_data['CODE_CULT_2017'])
        & (df_data['CODE_CULT_2015'] == df_data['CODE_CULT_2018'])
        & (df_data['CODE_CULT_2015'] == df_data['CODE_CULT_2019'])]
    df_two_year = df_data[
        (df_data['CODE_CULT_2015'] == df_data['CODE_CULT_2016'])
        & (df_data['CODE_CULT_2017'] == df_data['CODE_CULT_2018'])
        & (df_data['CODE_CULT_2015'] != df_data['CODE_CULT_2018'])
        & (df_data['CODE_CULT_2019'] != df_data['CODE_CULT_2018'])
        & ~df_data.index.isin(df_permanent.index)]

    for row in df_permanent.iterrows():
        df_data.loc[row[0]]['CODE_CULT_2020'] = row[1]['CODE_CULT_2015']

    for row in df_two_year.iterrows():
        df_data.loc[row[0]]['CODE_CULT_2020'] = row[1]['CODE_CULT_2019']

    df_data['CODE_CULT_2020'] = df_data['CODE_CULT_2020'].map(dictionary.get)
    df_data['CODE_CULT_2019'] = df_data['CODE_CULT_2019'].map(dictionary.get)
    df_data['CODE_CULT_2018'] = df_data['CODE_CULT_2018'].map(dictionary.get)
    df_data['CODE_CULT_2017'] = df_data['CODE_CULT_2017'].map(dictionary.get)
    df_data['CODE_CULT_2016'] = df_data['CODE_CULT_2016'].map(dictionary.get)
    df_data['CODE_CULT_2015'] = df_data['CODE_CULT_2015'].map(dictionary.get)

    df_data[['CODE_CULT_2015', 'CODE_CULT_2016', 'CODE_CULT_2017', 'CODE_CULT_2018', 'CODE_CULT_2019', 'CODE_CULT_2020',
             'LATITUDE', 'LONGTITUDE']].to_csv('predict_2020_full.csv', index=True)
    df_data['CODE_CULT_2020'].to_csv('predict_2020.csv', index=True)


if __name__ == '__main__':
    print('start\r')
    # commands = [
    #     ['mkdir -p dvhb_data dvhb_data/raw_data'],
    #     ['wget -q -O dvhb_data/archive.zip "http://35.156.82.253:8000/archive.zip"'],
    #     ['unzip -o -qq dvhb_data/archive.zip -d dvhb_data'],
    #     ['wget -q -O dvhb_data/raw_data.zip "http://35.156.82.253:8000/raw_archive.zip"'],
    #     ['unzip -o -qq dvhb_data/raw_data.zip -d dvhb_data/raw_data'],
    #     ['echo "Dataset unpack done!"']
    # ]
    # for command in commands:
    #     p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    #     p.communicate()
    print('data getting\r')
    # processes = [Process(target=get_train_full), Process(target=get_train_cult_full),
    #              Process(target=get_test, args=(2019,)), Process(target=get_test, args=(2020,))]
    # for proc in processes:
    #     proc.start()
    # for proc in processes:
    #     proc.join()
    # get_test(2019)
    # get_test(2020)
    # fit_model()
    get_predict_2019()
    get_predict_2020()
    print('complete')
    # get_predict_2019()
