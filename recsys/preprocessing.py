from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
import os

def get_examples(dataframe, labels_column="rating"):
    examples = dataframe[['userid', 'itemid']].values
    labels = dataframe[f'{labels_column}'].values
    return (examples, labels)

def split_data(examples, labels, test_size=0.1, verbose=0):
    """
    :param
        - ratings : dataframe of observed rating
    """
    if verbose:
        print("Train/Test split ")
        print(100-test_size*100, "% of training data")
        print(test_size*100, "% of testing data")    

    # split data into train and test sets
    train_examples, test_examples, train_labels, test_labels = train_test_split(
        examples, 
        labels, 
        test_size=0.1, 
        random_state=42, 
        shuffle=True
    )
    if verbose:
        print()
        print('number of training examples : ', train_examples.shape)
        print('number of training labels : ', train_labels.shape)
        print('number of test examples : ', test_examples.shape)
        print('number of test labels : ', test_labels.shape)

    return (train_examples, test_examples), (train_labels, test_labels)

def mean_ratings(dataframe):
    means = dataframe.groupby(by='userid', as_index=False)['rating'].mean()
    return means

def normalized_ratings(dataframe, norm_column="norm_rating"):
    """
    Subscribe users mean ratings from each rating 
    """
    mean = mean_ratings(dataframe=dataframe)
    norm = pd.merge(dataframe, mean, suffixes=('','_mean'), on='userid')
    norm[f'{norm_column}'] = norm['rating'] - norm['rating_mean']

    return norm

def rating_matrix(dataframe, column):
    R = csr_matrix(pd.crosstab(
        dataframe.userid, 
        dataframe.itemid, 
        dataframe[f'{column}'], 
        aggfunc=sum).fillna(0).values)
    return R

def scale_ratings(dataframe, scaled_column="scaled_rating"):
    """
    """
    dataframe[f"{scaled_column}"] = dataframe.rating / 5.0
    return dataframe

def encode_data(ratings, examples, labels):
    """
    """
    users = ratings['userid'].unique()
    items = ratings['itemid'].unique()

    # create users and items encoders
    uencoder = LabelEncoder()
    iencoder = LabelEncoder()

    # fit users and items ids to the corresponding encoder
    uencoder.fit(users)
    iencoder.fit(items)

    train_examples, test_examples = examples
    train_labels, test_labels = labels

    # transform train and test examples to their corresponding one-hot representations
    train_users = train_examples[:,0]
    test_users = test_examples[:,0]

    train_items = train_examples[:,1]
    test_items = test_examples[:,1]

    train_users = uencoder.transform(train_users)
    test_users = uencoder.transform(test_users)

    train_items = iencoder.transform(train_items)
    test_items = iencoder.transform(test_items)

    # Final training and test set
    X_train = np.array(list(zip(train_users, train_items)))
    X_test = np.array(list(zip(test_users, test_items)))

    y_train = train_labels
    y_test = test_labels

    return (X_train, X_test), (y_train, y_test), (uencoder, iencoder)
