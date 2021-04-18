import pandas as pd
import zipfile
import urllib.request
import sys
import os

DOWNLOAD_DESTINATION_DIR = "dataset"


def unzip(name):
    path = os.path.join(DOWNLOAD_DESTINATION_DIR, name)
    print(f"Unzipping the {name} zip file ...")
        
    with zipfile.ZipFile(path, 'r') as data:
        data.extractall(DOWNLOAD_DESTINATION_DIR)


def _progress(count, block_size, total_size):
    sys.stdout.write('\rDownload data %.1f%%' % (float(count * block_size)/float(total_size) * 100.0))
    sys.stdout.flush()


def download(url, name):
    path = os.path.join(DOWNLOAD_DESTINATION_DIR, name)
    if not os.path.exists(path):        
        os.makedirs(DOWNLOAD_DESTINATION_DIR, exist_ok=True)
        fpath, _ = urllib.request.urlretrieve(url, filename=path, reporthook=_progress)
        
        print()
        statinfo = os.stat(fpath)
        print('Successfully downloaded', name, statinfo.st_size, 'bytes.')
        unzip(name)


class mlLatestSmall:

    @staticmethod
    def load():        
        url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        name = 'ml-latest-small'
        
        download(url, f"{name}.zip")
        
        ratings_path = os.path.join(DOWNLOAD_DESTINATION_DIR, name, 'ratings.csv')
        ratings = pd.read_csv(
            ratings_path,
            sep=',',
            names=["userid", "itemid", "rating", "timestamp"],
            skiprows=1
        )

        movies_path = os.path.join(DOWNLOAD_DESTINATION_DIR, name, 'movies.csv')
        movies = pd.read_csv(
            movies_path,
            sep=',',
            names=["itemid", "title", "genres"],
            encoding='latin-1',
            skiprows=1
        )
        
        return ratings, movies


class ml100k:

    @staticmethod
    def load():        
        url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
        name = 'ml-100k'
        
        download(url, f"{name}.zip")
        
        ratings_path = os.path.join(DOWNLOAD_DESTINATION_DIR, name, 'u.data')
        ratings = pd.read_csv(
            ratings_path,
            sep='\t',
            names=["userid", "itemid", "rating", "timestamp"],
        )
        ratings = ratings.sort_values(by=['userid', 'itemid']).reset_index(drop=True)
        ratings = ratings.drop(columns=['timestamp'])

        movies_columns = [
            'itemid', 'title', 'release date', 'video release date', 
            'IMDb URL ', 'unknown', 'Action', 'Adventure', 'Animation',
            "Children's", 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
            'Film-Noir', 'Horror', 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
            'Thriller' , 'War' , 'Western' ,
        ]

        movies_path = os.path.join(DOWNLOAD_DESTINATION_DIR, name, 'u.item')
        movies = pd.read_csv(
            movies_path,
            sep='|',
            names=movies_columns,
            encoding='latin-1',
        )
        # drop non necessary columns. From the third to the last column
        todrop = list(range(2, len(movies.columns)))
        movies = movies.drop(movies.columns[todrop], axis=1)
        
        return ratings, movies


class ml1m:

    @staticmethod
    def load():
        url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
        name = "ml-1m"

        download(url, f"{name}.zip")
        
        ratings_path = os.path.join(DOWNLOAD_DESTINATION_DIR, name, 'ratings.dat')
        ratings = pd.read_csv(
            ratings_path,
            sep='::',
            names=["userid", "itemid", "rating", "timestamp"],
            engine='python'
        )
        ratings = ratings.sort_values(by=['userid', 'itemid']).reset_index(drop=True)
        ratings = ratings.drop(columns=['timestamp'])

        movies_path = os.path.join(DOWNLOAD_DESTINATION_DIR, name, 'movies.dat')
        movies = pd.read_csv(
            movies_path,
            sep='::',
            names=["itemid", "title", "genres"],
            encoding='latin-1',
            engine='python'
        )
        
        return ratings, movies
