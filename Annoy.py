from annoy import AnnoyIndex
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle 
import utils

print('Loading Dictionaries/Arrays')
t0 = time.time()
data = pd.read_csv('/mnt/work/Music_Proj/metadata.csv', index_col=[0])
vecs = np.load('/mnt/work/Music_Proj/L2Norm_ALS_vecs.npy')
metadata = utils.create_metadata(data, vecs)
tags = pd.read_csv('lastfm_unique_tags.txt', delimiter='\t', nrows=5)
tags.columns = ['tags', 'frequency']
tags.drop(columns=['frequency'])
tags.to_csv('tags.csv')

for tag in ['rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 'None']:
    print('Making Index for tag: {}'.format(tag))
    time.sleep(2)
    if tag != 'None':
        metadata['has_{}'.format(tag)] = metadata['tags'].apply(lambda tags: '{}'.format(tag) in tags)
        song_vecs = metadata.loc[metadata['has_{}'.format(tag)], [str(i) for i in range(32)]].values
    else:
        song_vecs = vecs


    t1 = time.time()    
    print('{0} seconds'.format(t1-t0))

    trees = 50
    # Does NN Fit
    print('Starting NN fit.')
    t0 = time.time()
    annoy = AnnoyIndex(32, 'angular')
    for ind in tqdm(range(len(song_vecs))):
        annoy.add_item(ind, song_vecs[ind])
    annoy.build(trees)
    t1 = time.time()
    print('Finished NN fit\n{0} seconds'.format(t1-t0))
    print('Saving...')
    annoy.save('L2Norm-ALS_{}.ann'.format(tag))
    time.sleep(2)
    print('Done.\n\n')
