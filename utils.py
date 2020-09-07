import pandas as pd
import requests
import os.path
from annoy import AnnoyIndex
import time
from tqdm import tqdm

def setup():
    metadata = get_data()
    create_indicies(metadata)
    

def get_data():
    URL = 'https://www.dropbox.com/s/i9y97ikmizgs1sf/meta_with_vecs.csv?dl=0'
    if os.path.exists('meta_with_vecs.csv'):
        return pd.read_csv('meta_with_vecs.csv')
    else:
        print('Downloading Data')
        fileObj = requests.get(URL)
        open('meta_with_vecs.csv', 'wb').write(fileObj.content)
        print('Done')
        return pd.read_csv('meta_with_vecs.csv')

def create_indicies(metadata, tags=['alternative', 'rock', 'electronic', 'indie', 'female vocalists', 'pop', 'None']):
    '''
    Pass your own list of tags if you want! 
    They are all listed in the file "lastfm_unqiue_tags.csv" that can be found at this url ==> http://millionsongdataset.com/sites/default/files/lastfm/lastfm_unique_tags.txt
    '''
    for tag in tags:
        if os.path.exists('{}.ann'.format(tag)):
            continue
        if tag == 'None':
            song_vecs = metadata[[str(i) for i in range(32)]].values
        else:
            print('Making Index for tag: {}'.format(tag))
            metadata['has_{}'.format(tag)] = metadata['tags'].apply(lambda tags: '{}'.format(tag) in tags)
            song_vecs = metadata.loc[metadata['has_{}'.format(tag)], [str(i) for i in range(32)]].values
        
        # Does NN Fit
        trees = 50
        print('Starting NN fit.')
        annoy = AnnoyIndex(32, 'angular')
        for ind in tqdm(range(len(song_vecs))):
            annoy.add_item(ind, song_vecs[ind])
        annoy.build(trees)
        print('Saving...')
        annoy.save('{}.ann'.format(tag))
        print('Done.\n\n')