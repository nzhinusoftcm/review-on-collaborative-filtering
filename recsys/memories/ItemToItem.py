"""
@author : carmel wenga.
"""
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from datetime import datetime
from glob import glob

import pandas as pd
import numpy as np
import zipfile
import os

class ItemToItem:

    def __init__(self, ratings, movies, k=30,
        predictions_dir='predictions/item2item',
        metric='cosine'):

        self.means, self.ratings = self.normalize(ratings)
        self.ratings_matrix = self.create_ratings_matrix(self.ratings)

        itemids = sorted(ratings['itemid'].unique())
        self.itemids_to_idx = {itemid:idx for (itemid,idx) in zip(itemids, range(len(itemids)))}
        self.idx_to_itemids = {idx:itemid for (idx,itemid) in zip(range(len(itemids)),itemids)}

        self.k = k
        self.metric = metric
        self.model = self.init_similarity_model()
        self.predictions_dir = predictions_dir
        self.neighbors, self.similarities = self.compute_nearest_neighbors()
        self.movies = movies

        os.makedirs(self.predictions_dir, exist_ok=True)

        print('Item to item recommendation model created with success ...')

    def normalize(self, ratings):
        print('Normalize ratings ...')
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

    def create_ratings_matrix(self, ratings):
        ratings_matrix = csr_matrix(pd.crosstab(
                                ratings.itemid,
                                ratings.userid,
                                ratings.norm_rating,
                                aggfunc=sum).fillna(0).values)
        return ratings_matrix

    def init_similarity_model(self):
        print('Create the similarity model ...')
        model = NearestNeighbors(
                                metric=self.metric,
                                n_neighbors=self.k+1,
                                algorithm='brute')
        # fit the model with users's ratings
        model.fit(self.ratings_matrix)
        return model

    def compute_nearest_neighbors(self):
        print('Compute nearest neighbors ...')
        similarities, neighbors = self.model.kneighbors(self.ratings_matrix)
        similarities = 1 - similarities
        similarities[:,0] = 0
        return neighbors, similarities

    def candidate_items(self, userid):
        """
        :param userid : user id for which we wish to find candidate items        
        :return I_u : list of items already purchased by userid
        :return candidates : list of candidate items
        """
        user_rated_items = self.ratings.loc[self.ratings.userid == userid].itemid.to_list()
        C = set()        
        for iid in user_rated_items:        
            idx = self.itemids_to_idx[iid]            
            C.update([self.idx_to_itemids[ix] for ix in self.neighbors[idx]])            
        C = list(C)
        
        # exclude from the set C all items in I_u.
        candidates = np.setdiff1d(C, user_rated_items, assume_unique=True)        
        return user_rated_items, candidates

    def similarity_with_Iu(self, c, Iu):
        """
        compute similarity between an item c and a set of items Iu. 
        For each item i in Iu, get similarity between i and c, if 
        c exists in the set of items similar to itemid    
        :param c : a candidate itemid
        :param Iu : set of items already purchased by a given user    
        :return w : similarity between c and Iu
        """
        w = 0    
        for i in Iu :        
            idx = self.itemids_to_idx[i]        
            # get similarity between itemid and c, if c is one of the k nearest neighbors of itemid
            if c in self.neighbors[idx] :
                i_similarities = self.similarities[idx]
                w = w + i_similarities[list(self.neighbors[idx]).index(c)]    
        return w

    def rank_candidates(self, candidates, Iu):
        """
        rank candidate items according to their similarities with Iu
        
        :param candidates : list of candidate items
        :param Iu : list of items purchased by the user    
        :return ranked_candidates : dataframe of candidate items, ranked in descending order of similarities with Iu
        """
        mapping = zip(candidates, [self.similarity_with_Iu(c, Iu) for c in candidates])
        ranked_candidates = pd.DataFrame(mapping, columns=['itemid','similarity_with_Iu'])
        ranked_candidates = ranked_candidates.sort_values(by=['similarity_with_Iu'], ascending=False)    
        return ranked_candidates

    def topN(self, userid, N=30):
        """
        Produce top-N recommendation for a given user        
        :param userid : user for which we produce top-N recommendation
        :param N : length of the top-N recommendation list        
        :return topN
        """
        Iu, candidates = self.candidate_items(userid)        
        ranked_candidates = self.rank_candidates(candidates, Iu)        
        topn = ranked_candidates.iloc[:N]        
        topn = pd.merge(topn, self.movies, on='itemid', how='inner')        
        return topn

    def predict(self, userid, itemid):
        """
        Make rating prediction for userid on itemid       
        :param userid : id of the active user
        :param itemid : id of the item for which we are making prediction            
        :return r_hat : predicted rating
        """        
        idx = self.itemids_to_idx[itemid]        
        i_neighbors = self.neighbors[idx,1:]
        i_similarities = self.similarities[idx,1:]        
        weighted_sum, W = 0, 0
        for iid in i_neighbors:
            if ((self.ratings.userid==userid) & (self.ratings.itemid==iid)).any():
                r = self.ratings[(self.ratings.userid==userid) & (self.ratings.itemid==iid)].rating.values[0]                
                w = i_similarities[list(i_neighbors).index(iid)]                
                W = W + abs(w)                
                weighted_sum = weighted_sum + r * w                
        if weighted_sum == 0:
            r_hat = self.means[self.means.userid==userid].rating.values[0]
        else:    
            r_hat = weighted_sum / W        
        return r_hat

    def predict_topN(self, userid):
        """
        :param userid : id of the active user        
        :return topN_list : initial topN recommendations returned by the function item2item_topN
        :return topN_predict : topN recommendations reordered according to rating predictions
        """
        topn = self.topN(userid)
        itemids = topn.itemid.to_list()        
        predictions = zip(itemids, [self.predict(userid, itemid) for itemid in itemids])        
        predictions = pd.DataFrame(predictions, columns=['itemid','prediction'])
        
        # merge the predictions to topn and rearrange the list according to predictions
        topn_predict = pd.merge(topn, predictions, on='itemid', how='inner')
        topn_predict = topn_predict.sort_values(by=['prediction'], ascending=False)
        
        return topn_predict

    def evaluate(self, x_test, y_test):
        print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
        preds = list(self.predict(u,i) for (u,i) in x_test)
        mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
        print()
        print('MAE :', mae)
        return mae


