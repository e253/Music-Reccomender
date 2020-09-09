import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
import implicit
from implicit.nearest_neighbours import bm25_weight
from implicit.nearest_neighbours import BM25Recommender
import pickle
import time
from tqdm import tqdm
from tabulate import tabulate

t0 = time.time()
plays = pd.read_csv('train_triplets_csv.csv', header=None, index_col=[0])
t1 = time.time()
print("loaded triplets in {0} seconds".format(t1-t0))
plays.columns = ["Users", "Songs", "Plays"]
print('uti and sti...')
t0 = time.time()
user_to_index = {user: i for i, user in tqdm(enumerate(set(plays["Users"])))}
song_to_index = {song: i for i, song in tqdm(enumerate(set(plays["Songs"])))}

'''
print('Id to Metadata...')
id_to_metadata = {song: metadata.loc[song].to_dict() for ind, song in tqdm(plays['Songs'].iteritems())}
print('title to metadata...')
title_to_metadata = {metadata['title'].loc[song]: metadata.loc[song].to_dict() for ind, song in tqdm(plays['Songs'].iteritems())}
'''

n_users = len(user_to_index)
n_songs = len(song_to_index)

cols = np.array(plays["Users"].apply(lambda user: user_to_index[user]))
rows = np.array(plays["Songs"].apply(lambda song: song_to_index[song]))
values = np.array(plays["Plays"])
mat = coo_matrix((values, (rows, cols)), shape=(n_songs, n_users))
print('Fitting...')
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
model.fit(bm25_weight(mat))
song_vecs = model.item_factors

print('Dumping...')
np.save("song_vecs.npy", song_vecs)
t1 = time.time()
print('{0} seconds'.format(t1-t0))




