import numpy as np


class EMF:
    
    def __init__(self, m, n, W, alpha=0.001, beta=0.01, lamb=0.1, k=10):
        """
            - m : number of users
            - n :  number of items
            - W : Explainability Weights of shape (m,n)
            - k : number of latent factors
            - beta : L2 regularization parameter
            - lamb : explainability regularization coefficient
            - theta : threshold above which an item is explainable for a user
        """
        self.W = W
        self.m = m
        self.n = n
        
        np.random.seed(64)
        
        # initialize the latent factor matrices P and Q (of shapes (m,k) and (n,k) respectively) that will be learnt
        self.k = k
        self.P = np.random.normal(size=(self.m,k))
        self.Q = np.random.normal(size=(self.n,k))
        
        # hyperparameter initialization
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        
        # training history
        self.history = {
            "epochs":[],
            "loss":[],
            "val_loss":[],
        }
        
    def print_training_parameters(self):
        print('Training EMF')
        print(f'k={self.k} \t alpha={self.alpha} \t beta={self.beta} \t lambda={self.lamb}')
        
    def update_rule(self, u, i, error):
        self.P[u] = self.P[u] + \
            self.alpha * (2 * error * self.Q[i] - self.beta * self.P[u] - self.lamb * (self.P[u] - self.Q[i]) * self.W[u,i])
        
        self.Q[i] = self.Q[i] + \
            self.alpha * (2 * error * self.P[u] - self.beta * self.Q[i] + self.lamb * (self.P[u] - self.Q[i]) * self.W[u,i])
        
    def mae(self,  x_train, y_train):
        """
        returns the Mean Absolute Error
        """
        # number of training exemples
        M = x_train.shape[0]
        error = 0
        for pair, r in zip(x_train, y_train):
            u, i = pair
            error += np.absolute(r - np.dot(self.P[u], self.Q[i]))
        return error/M
    
    def print_training_progress(self, epoch, epochs, error, val_error, steps=5):
        if epoch == 1 or epoch % steps == 0 :
                print("epoch {}/{} - loss : {} - val_loss : {}".format(epoch, epochs, round(error,3), round(val_error,3)))
                
    def learning_rate_schedule(self, epoch, target_epochs = 20):
        if (epoch >= target_epochs) and (epoch % target_epochs == 0):
                factor = epoch // target_epochs
                self.alpha = self.alpha * (1 / (factor * 20))
                print("\nLearning Rate : {}\n".format(self.alpha))
        
    def fit(self, x_train, y_train, validation_data, epochs=10):      
        self.print_training_parameters()
        x_test, y_test = validation_data        
        for epoch in range(1, epochs+1):
            for pair, r in zip(x_train, y_train):                
                u,i = pair                
                r_hat = np.dot(self.P[u], self.Q[i])                
                e = r - r_hat
                self.update_rule(u, i, error=e)                
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
        print(f"MAE : {round(error,3)}")
        return error


def explainable_score(u2uModel, users, items, theta=0):
    # initialize explainable score to zeros
    print('Compute explainable scores ...')
    W = np.zeros((len(users), len(items)))    
    for u in users:
        candidate_items = u2uModel.find_user_candidate_items(u)
        for i in candidate_items:
            user_who_rated_i, similar_user_who_rated_i = u2uModel.similar_users_who_rated_this_item(u, i)
            if len(user_who_rated_i) == 0:
                w = 0.0
            else:
                w = len(similar_user_who_rated_i) / len(user_who_rated_i)            
            W[u, i] = w if w > theta else 0.0
    return W
