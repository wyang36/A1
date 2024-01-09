import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_and_preprocess_data(encode=True, norm=True, test_size=0.1):
    dataset = pd.read_csv('./dataset/Myers Briggs Table_S1.csv')

    dataset['POSTURE'].value_counts().plot(kind='bar', rot=0)
    plt.title('Postures')
    plt.savefig('./images/mbti_data.png')
    plt.clf()
    # preprocessing
    # categorical data encoding
    le = preprocessing.LabelEncoder()
    le.fit(dataset['POSTURE'])
    y = le.transform(dataset['POSTURE'])
    dataset = dataset.drop(columns=['POSTURE', 'S No'])

    if encode:
        dataset = pd.get_dummies(dataset,
                                 columns=['SEX', 'ACTIVITY LEVEL', 'MBTI'],
                                 drop_first=False)
    # print(dataset.head())
    # normalization
    if norm:
        scaler = preprocessing.MinMaxScaler()
        dataset = scaler.fit_transform(dataset)
        # print(dataset)

    # train/test split
    # X, y = dataset[:, :-1], dataset[:, -1]
    kwargs = dict(test_size=test_size, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(dataset, y, **kwargs)
    return X_train, X_test, y_train, y_test
