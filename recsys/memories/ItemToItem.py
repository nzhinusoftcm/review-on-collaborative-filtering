"""
@author : carmel wenga.
"""
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from .. preprocessing import ids_encoder
from sys import stdout

import pandas as pd
import numpy as np

import os


class ItemToItem:

    def __init__(self, ratings, movies, k=20, metric='adjusted_cosine', dataset_name='ml100k'):

        if metric not in ['cosine', 'euclidean', 'adjusted_cosine']:
            raise Exception('UnknownSimilarityMetric : The similarity metric must be selected among'
                            'the followings : cosine, euclidean, adjusted_cosine. You choosed {}'.format(metric))

        if k > 50:
            raise Exception('Choose a value of k that is smaller than 50. You choosed {}'.format(k))

        self.predictions_dir = 'recsys/predictions/item2item'
        self.weights_dir = 'recsys/weights/item2item'
        self.dataset_name = dataset_name

        self.ratings, self.uencoder, self.iencoder = ids_encoder(ratings)
        self.means, self.ratings = self.normalize()
        self.np_ratings = self.ratings.to_numpy()
        self.ratings_matrix = self.create_ratings_matrix()

        self.k = k
        self.nb_items = self.ratings.itemid.nunique()

        # compute similarities between items

        self.metric = metric
        if metric == 'adjusted_cosine':
            print('Using the Adjusted Cosine Similarity Metric : ', end=' ')
            similarities_file = os.path.join(self.weights_dir, self.dataset_name, 'similarities.npy')
            if os.path.exists(similarities_file):
                print('Load Similarities ...')
                self.similarities, self.neighbors = self.load_similarities()
            else:
                print('Compute Similarities ...')
                self.similarities, self.neighbors = self.adjusted_cosine()
        else:
            # the metric is either 'cosine' or 'euclidian'
            self.model = self.init_knn_model()
            self.similarities, self.neighbors = self.knn()

        self.movies = movies

        os.makedirs(self.predictions_dir, exist_ok=True)

        print('Item to item recommendation model created with success ...')

    def normalize(self):
        print('Normalize ratings ...')
        means = self.ratings.groupby(by='userid', as_index=False)['rating'].mean()
        means_ratings = pd.merge(self.ratings, means, suffixes=('', '_mean'), on='userid')
        means_ratings['norm_rating'] = means_ratings['rating'] - means_ratings['rating_mean']

        return means.to_numpy()[:, 1], means_ratings

    def create_ratings_matrix(self):
        crosstab = pd.crosstab(self.ratings.itemid, self.ratings.userid, self.ratings.norm_rating, aggfunc=sum)
        matrix = csr_matrix(crosstab.fillna(0).values)
        return matrix

    def init_knn_model(self):
        print('Create the similarity model ...')
        model = NearestNeighbors(metric=self.metric, n_neighbors=self.k+1, algorithm='brute')
        # fit the model with users's ratings
        model.fit(self.ratings_matrix)
        return model

    def knn(self):
        print('Compute nearest neighbors ...')
        similarities, neighbors = self.model.kneighbors(self.ratings_matrix)
        return similarities[:, :1], neighbors[:, :1]

    def save_similarities(self, similarities, neighbors):
        save_dir = os.path.join(self.weights_dir, self.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        similarities_file_name = os.path.join(save_dir, 'similarities.npy')
        neighbors_file_name = os.path.join(save_dir, 'neighbors.npy')
        try:
            np.save(similarities_file_name, similarities)
            np.save(neighbors_file_name, neighbors)
        except ValueError as error:
            print(f"An error occured when saving similarities, due to : \n ValueError : {error}")
        print('Save similarities : SUCCESS')

    def load_similarities(self):
        save_dir = os.path.join(self.weights_dir, self.dataset_name)
        similiraties_file = os.path.join(save_dir, 'similarities.npy')
        neighbors_file = os.path.join(save_dir, 'neighbors.npy')
        similarities = np.load(similiraties_file)
        neighbors = np.load(neighbors_file)
        return similarities[:, :self.k], neighbors[:, :self.k]

    @staticmethod
    def cosine(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def adjusted_cosine(self):
        similarities = np.zeros(shape=(self.nb_items, self.nb_items))
        similarities.fill(-1)

        def _progress(progress):
            stdout.write('\rComputing similarities. Progress status : %.1f%%' % (float(progress/self.nb_items)*100.0))
            stdout.flush()

        items = sorted(self.ratings.itemid.unique())
        for i in items[:-1]:
            for j in items[i + 1:]:
                scores = self.np_ratings[(self.np_ratings[:, 1] == i) | (self.np_ratings[:, 1] == j), :]
                vals, count = np.unique(scores[:, 0], return_counts=True)
                scores = scores[np.isin(scores[:, 0], vals[count > 1]), :]

                if scores.shape[0] > 2:
                    x = scores[scores[:, 1].astype('int') == i, 4]
                    y = scores[scores[:, 1].astype('int') == j, 4]
                    w = self.cosine(x, y)

                    similarities[i, j] = w
                    similarities[j, i] = w
            _progress(i)
        _progress(self.nb_items)

        # get neighbors by their neighbors in decreasing order of similarities
        neighbors = np.flip(np.argsort(similarities), axis=1)

        # sort similarities in decreasing order
        similarities = np.flip(np.sort(similarities), axis=1)

        # save similarities to disk
        # we save only the top 50 neighbors with their similarities.
        self.save_similarities(similarities[:, :50], neighbors[:, :50])

        return similarities[:, :self.k], neighbors[:, :self.k]

    def candidate_items(self, userid):
        """
        :param userid : user id for which we wish to find candidate items        
        :return I_u : list of items already purchased by userid
        :return candidates : list of candidate items
        """
        user_rated_items = self.np_ratings[self.np_ratings[:, 0] == userid]
        c = set()
        for iid in user_rated_items:        
            c.update(self.neighbors[iid])
        c = list(c)
        
        # exclude from the set C all items in I_u.
        candidates = np.setdiff1d(c, user_rated_items, assume_unique=True)
        return user_rated_items, candidates

    def similarity_with_i_u(self, c, user_rated_items):
        """
        compute similarity between an item c and a set of items Iu. 
        For each item i in Iu, get similarity between i and c, if 
        c exists in the set of items similar to itemid    
        :param c : a candidate itemid
        :param user_rated_items : set of items already purchased by a given user
        :return w : similarity between c and Iu
        """
        w = 0    
        for iid in user_rated_items:
            # get similarity between item id and c, if c is one of the k nearest neighbors of item id
            if c in self.neighbors[iid]:
                w = w + self.similarities[iid, self.neighbors[iid] == c][0]
        return w

    def rank_candidates(self, candidates, user_rated_items):
        """
        rank candidate items according to their similarities with Iu
        
        :param candidates : list of candidate items
        :param user_rated_items : list of items purchased by the user
        :return ranked_candidates : dataframe of candidate items, ranked in descending order of similarities with Iu
        """
        sims = [self.similarity_with_i_u(c, user_rated_items) for c in candidates]
        candidates = self.iencoder.inverse_transform(candidates)
        mapping = list(zip(candidates, sims))
        ranked_candidates = sorted(mapping, key=lambda couple: couple[1], reverse=True)
        return ranked_candidates

    def topn_recommendation(self, userid, n=30):
        """
        Produce top-N recommendation for a given user        
        :param userid : user for which we produce top-N recommendation
        :param n : length of the top-N recommendation list
        :return topN
        """
        user_rated_items, candidates = self.candidate_items(userid)
        ranked_candidates = self.rank_candidates(candidates, user_rated_items)
        topn = pd.DataFrame(ranked_candidates[:n], columns=['itemid', 'similarity_with_Iu'])
        topn = pd.merge(topn, self.movies, on='itemid', how='inner')        
        return topn

    def predict(self, userid, itemid):
        """
        Make rating prediction for userid on itemid       
        :param userid : id of the active user
        :param itemid : id of the item for which we are making prediction            
        :return r_hat : predicted rating
        """

        # get ratings of user with id userid
        user_ratings = self.np_ratings[self.np_ratings[:, 0].astype('int') == userid]

        # similar items to itemid rated by this user (siru)
        siru = user_ratings[np.isin(user_ratings[:, 1], self.neighbors[itemid])]
        scores = siru[:, 2]
        indexes = [np.where(self.neighbors[itemid] == iid)[0][0] for iid in siru[:, 1].astype('int')]
        sims = self.similarities[itemid, indexes]

        numerator = np.dot(scores, sims)
        denominator = np.sum(np.abs(sims))

        if denominator == 0:
            return self.means[userid]

        r_hat = numerator / denominator
        return r_hat

    def topn_prediction(self, userid):
        """
        :param userid : id of the active user        
        :return topN_list : initial topN recommendations returned by the function item2item_topN
        :return topN_predict : topN recommendations reordered according to rating predictions
        """
        topn = self.topn_recommendation(userid)
        itemids = topn.itemid.to_list()        
        predictions = zip(itemids, [self.predict(userid, itemid) for itemid in itemids])        
        predictions = pd.DataFrame(predictions, columns=['itemid', 'prediction'])
        
        # merge the predictions to topn and rearrange the list according to predictions
        topn_predict = pd.merge(topn, predictions, on='itemid', how='inner')
        topn_predict = topn_predict.sort_values(by=['prediction'], ascending=False)
        
        return topn_predict

    def evaluate(self, x_test, y_test):
        print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
        preds = list(self.predict(u, i) for (u, i) in x_test)
        mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
        print()
        print('MAE :', mae)
        return mae
