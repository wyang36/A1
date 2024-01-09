import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_and_preprocess_data(encode=True, norm=True, test_size=0.2):
    dataset = pd.read_csv('./dataset/heart.csv')
    # print(dataset.head())

    dataset['HeartDisease'].value_counts().plot(kind='bar', rot=0)
    plt.title('Heart Disease Outcomes')
    plt.savefig('./images/heart_data.png')
    plt.clf()
    # preprocessing
    # categorical data encoding
    y = dataset['HeartDisease']
    dataset = dataset.drop(columns='HeartDisease')
    if encode:
        dataset = pd.get_dummies(dataset,
                                 columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
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
