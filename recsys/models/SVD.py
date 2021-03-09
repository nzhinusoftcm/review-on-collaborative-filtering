"""
@author carmel wenga
"""
import numpy as np

class SVD:

    def __init__(self, umean, uencoder, iencoder):

        self.umean = umean.to_numpy()
        self.uencoder = uencoder
        self.iencoder = iencoder
        
        # init svd resultant matrices
        self.P = np.array([])
        self.S = np.array([])
        self.Qh = np.array([])
        
        # init users and items latent factors
        self.u_factors = np.array([])
        self.i_factors = np.array([])
    
    def fit(self, R):
        """
        Fit the SVD model with rating matrix R
        """
        P, s, Qh = np.linalg.svd(R, full_matrices=False)
        
        self.P = P
        self.S = np.diag(s)
        self.Qh = Qh
        
        # latent factors of users (u_factors) and items (i_factors)
        self.u_factors = np.dot(self.P, np.sqrt(self.S))
        self.i_factors = np.dot(np.sqrt(self.S), self.Qh)
    
    def predict(self, userid, itemid, add_mean=False):
        """
        Make rating prediction for a given user on an item
        
        :param userid : user's id
        :param itemid : item's id            
        :return r_hat : predicted rating
        """
        # encode user and item ids
        u = self.uencoder.transform([userid])[0]
        i = self.iencoder.transform([itemid])[0]        
        r_hat = np.dot(self.u_factors[u,:], self.i_factors[:,i])
        if add_mean:
            r_hat += self.umean[u]        
        return r_hat
        
    
    def recommend(self, userid):
        """
        :param userid : user's id
        """
        # encode user
        u = self.uencoder.transform([userid])[0]
        
        # the dot product between the uth row of u_factors and i_factors returns
        # the predicted value for user u on all items        
        predictions = np.dot(self.u_factors[u,:], self.i_factors) + self.umean[u]
        
        # sort item ids in decreasing order of predictions
        top_idx = np.flip(np.argsort(predictions))

        # decode indices to get their corresponding itemids
        top_items = self.iencoder.inverse_transform(top_idx)
        
        # sorted predictions
        preds = predictions[top_idx]
        
        return top_items, preds

    def evaluate(self, x_test, y_test):
        print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
        preds = list(self.predict(u,i) for (u,i) in x_test)
        mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
        print()
        print('MAE :', mae)
        return mae
        