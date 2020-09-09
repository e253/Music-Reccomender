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
vecs = np.load('/mnt/work/Music_Proj/ALS-L2Norm/L2Norm-ALS_vecs.npy')
metadata = utils.create_metadata(data, vecs)

for tag in ['pop', 'rock', 'alternative', 'indie', 'electronic', 'female vocalists', 'None']:
    print('Making Index for tag: {}'.format(tag))
    time.sleep(2)
    if tag != 'None':
        metadata['has_{}'.format(tag)] = metadata['tags'].apply(lambda tags: '{}'.format(tag) in tags)
        song_vecs = metadata.loc[metadata['has_{}'.format(tag)], [i for i in range(32)]].values
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
    print('Finished NN fit in {0} seconds'.format(t1-t0))
    print('Saving...')
    annoy.save('L2Norm-ALS_{}-test.ann'.format(tag))
    time.sleep(2)
    test = AnnoyIndex(32, 'dot')
    try: 
        test.load('L2Norm-ALS_{}-test.ann'.format(tag))
        print('{0} a success!'.format(tag))
    except OSError:
        print('{0} failed'.format(tag))
    print('Done.\n')
