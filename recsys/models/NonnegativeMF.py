import numpy as np


class NMF:
    
    def __init__(self, ratings, m, n, uencoder, iencoder, K=10, lambda_P=0.01, lambda_Q=0.01):
        
        np.random.seed(32)
        
        # initialize the latent factor matrices P and Q (of shapes (m,k) and (n,k) respectively) that will be learnt
        self.ratings = ratings
        self.K = K
        self.P = np.random.rand(m, K)
        self.Q = np.random.rand(n, K)
        
        # hyper parameter initialization
        self.lambda_P = lambda_P
        self.lambda_Q = lambda_Q

        # initialize encoders
        self.uencoder = uencoder
        self.iencoder = iencoder
        
        # training history
        self.history = {
            "epochs": [],
            "loss": [],
            "val_loss": [],
        }
    
    def print_training_parameters(self):
        print('Training Matrix Factorization Model ...')
        print(f'k={self.K}')
        
    def mae(self,  x_train, y_train):
        """
        returns the Mean Absolute Error
        """
        # number of training examples
        M = x_train.shape[0]
        error = 0
        for pair, r in zip(x_train, y_train):
            u, i = pair
            error += abs(r - np.dot(self.P[u], self.Q[i]))
        return error/M
    
    def ratings_by_this_user(self, userid):
        return self.ratings.loc[self.ratings.userid == userid]
    
    def ratings_on_this_items(self, itemid):
        return self.ratings.loc[self.ratings.itemid == itemid]
    
    def update_rule(self, u, i, error):
        I = self.ratings_by_this_user(u)
        U = self.ratings_on_this_items(i)    
        
        for k in range(self.K):
            num_uk = self.P[u, k] * np.sum(np.multiply(self.Q[I.itemid.to_list(), k], I.rating.to_list()))
            dem_uk = np.sum(np.multiply(self.Q[I.itemid.to_list(), k], np.dot(self.P[u], self.Q[I.itemid.to_list()].T))
                            ) + self.lambda_P * len(I) * self.P[u, k]
            self.P[u, k] = num_uk / dem_uk
                
            num_ik = self.Q[i, k] * np.sum(np.multiply(self.P[U.userid.to_list(), k], U.rating.to_list()))
            dem_ik = np.sum(np.multiply(self.P[U.userid.to_list(), k], np.dot(self.P[U.userid.to_list()], self.Q[i].T))
                            ) + self.lambda_Q * len(U) * self.Q[i, k]
            self.Q[i, k] = num_ik / dem_ik

    @staticmethod
    def print_training_progress(epoch, epochs, error, val_error, steps=5):
        if epoch == 1 or epoch % steps == 0:
                print("epoch {}/{} - loss : {} - val_loss : {}".format(
                    epoch, epochs, round(error, 3), round(val_error, 3)))
                
    def fit(self, x_train, y_train, validation_data, epochs=10):

        self.print_training_parameters()
        x_test, y_test = validation_data
        for epoch in range(1, epochs+1):
            for pair, r in zip(x_train, y_train):
                u, i = pair
                r_hat = np.dot(self.P[u], self.Q[i])
                e = abs(r - r_hat)
                self.update_rule(u, i, e)                
            # training and validation error  after this epochs
            error = self.mae(x_train, y_train)
            val_error = self.mae(x_test, y_test)
            self.update_history(epoch, error, val_error)
            self.print_training_progress(epoch, epochs, error, val_error, steps=1)
        
        return self.history
    
    def update_history(self, epoch, error, val_error):
        self.history['epochs'].append(epoch)
        self.history['loss'].append(error)
        self.history['val_loss'].append(val_error)
    
    def evaluate(self, x_test, y_test):        
        error = self.mae(x_test, y_test)
        print(f"validation error : {round(error,3)}")
        print('MAE : ', error)        
        return error
      
    def predict(self, userid, itemid):
        u = self.uencoder.transform([userid])[0]
        i = self.iencoder.transform([itemid])[0]
        r = np.dot(self.P[u], self.Q[i])
        return r

    def recommend(self, userid, N=30):
        # encode the userid
        u = self.uencoder.transform([userid])[0]

        # predictions for users userid on all product
        predictions = np.dot(self.P[u], self.Q.T)

        # get the indices of the top N predictions
        top_idx = np.flip(np.argsort(predictions))[:N]

        # decode indices to get their corresponding itemids
        top_items = self.iencoder.inverse_transform(top_idx)

        # take corresponding predictions for top N indices
        preds = predictions[top_idx]

        return top_items, preds
