[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nzhinusoftcm/review-on-collaborative-filtering/)


This repository presents a comprehensive implementation of collaborative filtering recommender systems, from memory-based collaborative filtering to more advanced machine learning algorithms. It starts by implementing basics collaborative filtering algorithms such as <b>user-based</b> collaborative filering also known as user-to-user collaborative filtering and <b>item-based</b> collaborative filtering (item-to-item collaborative filtering). This repository also goes through dimensionality reduction based recommendation system. It presents models such as **Singular Value Decomposition** (SVD), **Matrix Factorization** (MF), **Non Negative Matrix Factorization** (NMF) and **Explainable Matrix Factorization** (EMF).

# Requirements

- matplotlib==3.2.2
- numpy==1.19.2
- pandas==1.0.5
- python==3.7
- scikit-learn==0.24.1
- scikit-surprise==1.1.1
- scipy==1.6.2

# Content

The topics covered in this repository are as follows : We first explore the movielen data

1. [Data exploration](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/1.DownloadAndExploreMovielenLatestSmall.ipynb) : this notebook explore the <a href="https://grouplens.org/datasets/movielens/">movielen lasted small</a> dataset. This dataset is used throughout this repository to build collaborative filtering recommender systems.

Then the model we implemented are the followings

## 1. Memory-based Collaborative Filtering

Two main algorithms :

2. [User-based (or user to user) Collaborative Filtering](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/2.User-basedCollaborativeFiltering.ipynb)  : implements user-based collaborative filtering.

3. [Item-based (or item to item) Collaborative Filtering](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/3.Item-basedCollaborativeFiltering.ipynb) : implements item-based collaborative filtering.

## 2. Dimensionality reduction

Here the explored models are :

4. [Singular Value Decomposition (SVD)](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/4.SingularValueDecomposition.ipynb) : implements dimensionality reduction with Singular Value Decomposition for collaborative filtering recommender systems

5. [Matrix Factorization](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/5.MatrixFactorization.ipynb) : builds and trains a Matrix Factorization based recommender system.

6. [Non Negative Matrix Factorization](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/6.NonNegativeMatrixFactorization.ipynb): applying non negativity to the learnt factors of matrix factorization.

7. [Explainable Matrix Factorization](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/7.ExplainableMatrixFactorization.ipynb): add explainability to matrix factorization factors in order to improve recommendation performances.

## 3. Performances comparison

8. [Performances comparison](https://github.com/nzhinusoftcm/review-on-collaborative-filtering/blob/master/8.PerformancesMeasure.ipynb): this notebook presents an overall performance comparaison of all the models listed before.


# References

1. Daniel Billsus  and  Michael J. Pazzani (1998). [Learning Collaborative Information Filters](https://www.ics.uci.edu/~pazzani/Publications/MLC98.pdf)
2. Herlocker et al. (1999)<a href="https://dl.acm.org/doi/10.1145/3130348.3130372"> An Algorithmic Framework for Performing Collaborative Filtering</a>
3. Daniel D. Lee & H. Sebastian Seung (1999). [Learning the parts of objects by non-negative matrix factorization](https://www.nature.com/articles/44565)
4. Sarwar et al. (2000). [Application of Dimensionality Reduction in Recommender System -- A Case Study](http://files.grouplens.org/papers/webKDD00.pdf)
5. George Karypis (2001)<a href="https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.554.1671&rep=rep1&type=pdf">Evaluation of Item-Based Top-N Recommendation Algorithms</a>
6. Sarwar et al. (2001) <a href="https://dl.acm.org/doi/10.1145/371920.372071"> Item-based collaborative filtering recommendation algorithms</a>
7. Greg Linden, Brent Smith, and Jeremy York (2003) <a href="https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf">Amazon.com Recommendations : Item-to-Item Collaborative Filtering</a>
8. Deng Cai et al. (2008). [Non-negative Matrix Factorization on Manifold](https://ieeexplore.ieee.org/document/4781101)
9. Yehuda Koren et al. (2009). <a href='https://ieeexplore.ieee.org/document/5197422'>Matrix Factorization Techniques for Recommender Systems</a>
10. Michael D. Ekstrand, et al. (2011). <a href="https://dl.acm.org/doi/10.1561/1100000009"> Collaborative Filtering Recommender Systems</a>
11. Yu-Xiong Wang and Yu-Jin Zhang (2011). [Non-negative Matrix Factorization: a Comprehensive Review](https://ieeexplore.ieee.org/document/6165290)
12. J. Bobadilla et al. (2013)<a href="https://romisatriawahono.net/lecture/rm/survey/information%20retrieval/Bobadilla%20-%20Recommender%20Systems%20-%202013.pdf"> Recommender systems survey</a>
13. Nicolas Gillis (2014). [The Why and How of Nonnegative Matrix Factorization](https://arxiv.org/pdf/1401.5226.pdf)
14. Dziugaite and Roy (2015), [Neural Network Matrix Factorization](https://arxiv.org/abs/1511.06443)
15. Abdollahi and Nasraoui (2016). [Explainable Matrix Factorization for Collaborative Filtering](https://www.researchgate.net/publication/301616080_Explainable_Matrix_Factorization_for_Collaborative_Filtering)
16. Abdollahi and Nasraoui (2017). [Using Explainability for Constrained Matrix Factorization](https://dl.acm.org/doi/abs/10.1145/3109859.3109913)
17. Xiangnan He et al. (2017), [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
18. Shuo Wang et al, (2018). [Explainable Matrix Factorization with Constraints on Neighborhood in the Latent Space](https://dl.acm.org/doi/abs/10.1145/3109859.3109913)

# Author

[Carmel WENGA](https://www.linkedin.com/in/carmel-wenga-871876178/), <br>
PhD student at Université de la Polynésie Française, <br> 
Applied Machine Learning Research Engineer, <br>
[ShoppingList](https://shoppinglist.cm), NzhinuSoft.
