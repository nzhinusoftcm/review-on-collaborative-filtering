[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nzhinusoftcm/review-on-collaborative-filtering/]


This repository presents a comprehensive implementation of collaborative filtering recommender systems, from memory-based collaborative filtering to more advanced machine learning algorithms. It starts by implementing basics collaborative filtering algorithms such as <b>user-based</b> collaborative filering also known as user-to-user collaborative filtering and <b>item-based</b> collaborative filtering (item-to-item collaborative filtering).

# Requirements

1. numpy==1.18.1
2. pandas==1.0.5
3. python==3.6.10
4. scikit-learn==0.23.1
5. tensorflow==2.3.0

# Content

The topics covered in this repository are as follows : We first explore the movielen data

1. [Data exploration](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/1.%20Download%20and%20explore%20movielen%20lasted%20small.ipynb) (```Download and explore movielen lasted small.ipynb```) : this notebook explore the <a href="https://grouplens.org/datasets/movielens/">movielen lasted small</a> dataset. This dataset is used throughout this repository to build collaborative filtering recommender systems.

Then the model we implemented are the followings

## 1. Memory-based Collaborative Filtering

Two main algorithms :

2. [User-based (or user to user) Collaborative Filtering](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/2.%20user-based%20collaborative%20filtering.ipynb) (```user-based collaborative filtering.ipynb```) : implements user-based collaborative filtering.

3. [Item-based (or item to item) Collaborative Filtering](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/3.%20item-based%20collaborative%20filtering.ipynb) (```item-based collaborative filtering.ipynb```) : implements item-based collaborative filtering.

## 2. Model-based Collaborative Filtering

### 2.1 Dimensionality reduction

Here the explored models are :

4. [Singular Value Decomposition (SVD)](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/4.%20Singular%20Value%20Decomposition.ipynb) (```Singular Value Decomposition.ipynb```) : implements dimensionality reduction with Singular Value Decomposition for collaborative filtering recommender systems

5. [Regulariezd SVD (Matrix Factorization)](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/5.%20matrix%20factorization%20based%20collaborative%20filtering.ipynb) (```matrix factorization based collaborative filtering.ipynb```): builds and trains a Matrix Factorization based recommender system.

### 2.2 Neural Networks based collaborative filtering

6. [Neural Network Matrix Factorization](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/6.%20neural%20networks%20matrix%20factorization.ipynb), (```neural networks matrix factorization.ipynb```): applies neural networks to matrix factorization.


# References

1. Daniel Billsus  and  Michael J. Pazzani (1998). [Learning Collaborative Information Filters](https://www.ics.uci.edu/~pazzani/Publications/MLC98.pdf)
2. Herlocker et al. (1999)<a href="https://dl.acm.org/doi/10.1145/3130348.3130372"> An Algorithmic Framework for Performing Collaborative Filtering</a>
3. Sarwar et al. (2000). [Application of Dimensionality Reduction in Recommender System -- A Case Study](http://files.grouplens.org/papers/webKDD00.pdf)
4. George Karypis (2001)<a href="https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.554.1671&rep=rep1&type=pdf">Evaluation of Item-Based Top-N Recommendation Algorithms</a>
5. Sarwar et al. (2001) <a href="https://dl.acm.org/doi/10.1145/371920.372071"> Item-based collaborative filtering recommendation algorithms</a>
6. Greg Linden, Brent Smith, and Jeremy York (2003) <a href="https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf">Amazon.com Recommendations : Item-to-Item Collaborative Filtering</a>
7. Yehuda Koren et al. (2009). <a href='https://ieeexplore.ieee.org/document/5197422'>Matrix Factorization Techniques for Recommender Systems</a>
8. Michael D. Ekstrand, et al. (2011). <a href="https://dl.acm.org/doi/10.1561/1100000009"> Collaborative Filtering Recommender Systems</a>
9. J. Bobadilla et al. (2013)<a href="https://romisatriawahono.net/lecture/rm/survey/information%20retrieval/Bobadilla%20-%20Recommender%20Systems%20-%202013.pdf"> Recommender systems survey</a>
10. Dziugaite and Roy (2015), [Neural Network Matrix Factorization](https://arxiv.org/abs/1511.06443)
11. Xiangnan He et al. (2017), [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)


# Author

[Carmel WENGA](https://www.linkedin.com/in/carmel-wenga-871876178/), Applied Machine Learning Research Engineer | [ShoppingList](https://shoppinglist.cm), Nzhinusoft
