import pandas as pd
import urllib
import zipfile
import sys
import os


def download_data():
    
    url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    name = 'ml-lasted-small'
    
    save_path = os.path.join('tools','ml-latest-small.zip')

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r[INFO] Downloading ml-lasted-small %.1f%%' % (float(count * block_size)/float(total_size) * 100.0))
        sys.stdout.flush()

    fpath, _ = urllib.request.urlretrieve(url, filename=save_path, reporthook=_progress)
    
    print()
    statinfo = os.stat(fpath)
    print('[INFO] Successfully downloaded', name, statinfo.st_size, 'bytes.')
    print("[INFO] Unzipping the downloaded file ...")
    
    with zipfile.ZipFile(fpath, 'r') as data:
        data.extractall('tools')
    
    ratings_csv = os.path.join('tools', 'ml-latest-small', 'ratings.csv')
    movies_csv = os.path.join('tools', 'ml-latest-small', 'movies.csv')
    
    return ratings_csv, movies_csv
    
    
def load_ratings(ratings_csv):
    
    # load ratings
    ratings = pd.read_csv(
        ratings_csv,
        sep=',',
        names=["userid", "itemid", "rating", "timestamp"],
        skiprows=1
    )
    
    ratings = ratings.drop('timestamp', axis=1)
    
    return ratings


def load_movies(movies_csv):
    
    # load movies
    movies = pd.read_csv(
        movies_csv,
        names=["itemid", "title", "genres"],
        encoding='latin-1',
        skiprows=1
    )
    
    movies = movies.drop('genres', axis=1)
    
    return movies
