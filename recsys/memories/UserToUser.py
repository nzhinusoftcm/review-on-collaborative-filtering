#!/usr/bin/env python
# coding: utf-8

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from datetime import datetime
from glob import glob

import pandas as pd
import numpy as np
import zipfile
import os

class UserToUser:

    def __init__(self, ratings, movies, k=21,
        predictions_dir='predictions/user2user', 
        metric='cosine'):

        self.means, self.ratings = self.prepare_ratings(ratings)
        self.ratings_matrix = self.create_ratings_matrix(ratings)
        self.k = k
        self.metric = metric
        self.model = self.init_similarity_model()
        self.predictions_dir = predictions_dir
        self.neighbors, self.similarities = self.compute_nearest_neighbors()
        self.movies = movies

        os.makedirs(self.predictions_dir, exist_ok=True)

        print('User to user recommendation model created with success ...')

    def create_ratings_matrix(self, ratings):
        ratings_matrix = csr_matrix(pd.crosstab(
                                ratings.userid,
                                ratings.itemid,
                                ratings.rating,
                                aggfunc=sum).fillna(0).values)
        return ratings_matrix

    def init_similarity_model(self):
        print('Create the similarity model ...')
        model = NearestNeighbors(
                                metric=self.metric,
                                n_neighbors=self.k,
                                algorithm='brute')
        # fit the model with users's ratings
        model.fit(self.ratings_matrix)
        return model

    def prepare_ratings(self, ratings):
        """
        Add to the rating dataframe :
            - mean_ratings : mean rating for all users
            - norm_ratings : normalized ratings for each (user,item) pair
        """
        print('Normalize users ratings ...')
        means = ratings.groupby(
            by='userid',
            as_index=False)['rating'].mean()
        means_ratings = pd.merge(
            ratings,
            means,
            suffixes=('','_mean'), on='userid')
        means_ratings['norm_rating'] = means_ratings['rating'] - \
            means_ratings['rating_mean']

        return means, means_ratings

    def get_user_nearest_neighbors(self, userid):
        similarities, neighbors = \
                        self.model.kneighbors(self.ratings_matrix[userid-1,:])
        similarities = 1 - np.squeeze(similarities)
        neighbors = np.squeeze(neighbors) + 1

        return similarities[1:].tolist(), neighbors[1:].tolist()

    def compute_nearest_neighbors(self):
        print('Compute nearest neighbors ...')
        similarities, neighbors = self.model.kneighbors(self.ratings_matrix)
        similarities = 1 - similarities
        neighbors = neighbors + 1
        return neighbors, similarities

    def user_rated_items(self, userid):
        return self.ratings.loc[
            self.ratings.userid==userid].itemid.to_list()

    def find_user_candidate_items(self, userid, neighbors, n=50):
        user_rated_items = self.user_rated_items(userid)

        neighbors_rated_items = self.ratings.loc[
            self.ratings.userid.isin(neighbors)]

        # sort items in decreasing order of frequency
        items_frequencies = neighbors_rated_items.groupby('itemid')\
            ['rating'].count().reset_index(name='count').\
            sort_values(['count'],ascending=False)

        neighbors_rated_items_sorted_by_frequency = items_frequencies.itemid
        candidates_items = np.setdiff1d(
            neighbors_rated_items_sorted_by_frequency,
            user_rated_items,
            assume_unique=True)

        return candidates_items[:n]

    def similar_users_who_rated_this_item(self, itemid, neighbors):
        users_who_rated_this_item = self.ratings[self.ratings.itemid==itemid]
        similar_users_who_rated_this_item = users_who_rated_this_item.\
            loc[users_who_rated_this_item.userid.isin(neighbors)]

        return users_who_rated_this_item, similar_users_who_rated_this_item

    def predict(self, userid, itemid, similarities, neighbors):
        """
        predict what score userid would have given to itemid.
        :param userid : user id for which we want to make prediction
        :param itemid : item id on which we want to make prediction
        :return r_hat : predicted rating of user userid on item itemid
        """
        user_mean = self.means[self.means.userid==userid].rating.values[0]
        weighted_sum = 0
        W = 0

        _, similar_users_who_rated_this_item = self.similar_users_who_rated_this_item(
            itemid=itemid, neighbors=neighbors)

        if len(similar_users_who_rated_this_item) == 0:
            return 0

        for uid in similar_users_who_rated_this_item.userid:
            w = similarities[neighbors.index(uid)]
            norm_score = self.ratings[(self.ratings.userid==uid) & \
                (self.ratings.itemid==itemid)].norm_rating.values[0]
            weighted_score = norm_score * w
            W = W + abs(w)
            weighted_sum = weighted_sum + weighted_score
        r_hat = user_mean + weighted_sum / W
        return r_hat

    def evaluate(self, x_test, y_test):
        print('Evaluate the model on {} test data'.format(x_test.shape[0]))
        preds = list()
        for (u,i) in x_test :
            similarities, neighbors = self.get_user_nearest_neighbors(u)
            r = self.predict(u,i,similarities,neighbors)
            preds.append(r)
        mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
        print()
        print('MAE :', mae)
        return mae

    def user_predictions(self, userid, predictions_file):
        """
        Make rating prediction for the active user on each candidate item and save in file prediction.csv

        :param
            - userid : id of the active user
            - predictions_file : where to save predictions
        """
        similarities, neighbors = self.get_user_nearest_neighbors(userid)

        # find candidate items for the active user
        candidates = self.find_user_candidate_items(userid, neighbors)[:50]

        # loop over candidates items to make predictions
        for itemid in candidates:

            # prediction for userid on itemid
            r_hat = self.predict(userid, itemid, similarities, neighbors)

            # save predictions
            with open(predictions_file, 'a+') as file:
                line = '{},{},{}\n'.format(userid, itemid, r_hat)
                file.write(line)


    def all_predictions(self):
        """
        Make predictions for each user in the database.

        """
        # get list of users in the database
        users = self.ratings.userid.unique()

        now = str(datetime.now()).replace(' ','-').split('.')[0]
        file_name = f'prediction.{now}.csv'
        predictions_file = os.path.join(self.predictions_dir, file_name)

        for userid in users:

            # make rating predictions for the current user
            self.user_predictions(userid, predictions_file)


    def make_recommendations(self, userid):
        """
        """
        predictions_files = glob(f'{self.predictions_dir}/*.csv')
        last_predictions = sorted(
            predictions_files, 
            key=lambda file:datetime.fromtimestamp(os.path.getmtime(file)),
            reverse=True
        )[0]

        predictions = pd.read_csv(
            last_predictions, sep=',', 
            names=['userid', 'itemid', 'predicted_rating']
        )
        predictions = predictions[predictions.userid==userid]
        recommendation_list = predictions.sort_values(
            by=['predicted_rating'], 
            ascending=False
        )
        recommendation_list = pd.merge(
            recommendation_list, 
            self.movies, 
            on='itemid', 
            how='inner'
        )

        return recommendation_list