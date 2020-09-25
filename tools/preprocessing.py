from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
import os


def split_data(ratings, test_size=0.1):
    """
    :param
        - ratings : dataframe of observed rating
    """
    print("[INFO] Train/Test split ")
    print("  - ", 100-test_size*100, "% of training data")
    print("  - ", test_size*100, "% of testing data")

    # get all observed pairs (u,i) with their corresponding labels
    examples = ratings[['userid', 'itemid']].values
    labels = ratings.rating.values

    # split data into train and test sets
    train_examples, test_examples, train_labels, test_labels = train_test_split(
        examples, 
        labels, 
        test_size=0.1, 
        random_state=42, 
        shuffle=True
    )
    print()
    print('number of training examples : ', train_examples.shape)
    print('number of training labels : ', train_labels.shape)
    print('number of test examples : ', test_examples.shape)
    print('number of test labels : ', test_labels.shape)

    return (train_examples, test_examples), (train_labels, test_labels)

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
    X_train = [train_users, train_items]
    X_test = [test_users, test_items]

    y_train = train_labels
    y_test = test_labels

    return (X_train, X_test), (y_train, y_test), (uencoder, iencoder)
