import numpy as np
import time
import pandas as pd
from tabulate import tabulate
from annoy import AnnoyIndex
import pickle as pkl
import utils

'''Helpers'''
def get_vec_from_title(metadata, title):
    return metadata.loc[metadata['title']==title, [str(i) for i in range(32)]].values[0]
def get_vector_from_song_id(metadata, song_id):
    return metadata.loc[metadata['song_id']==song_id, [str(i) for i in range(32)]].values[0]
def get_vec_index_from_song_id(df, song_id):
    return df['vec_ind'].loc[df['song_id'] == song_id].iat[0]
def get_title_and_artist_from_index(df, index):
        return df['title'].iloc[index], df['artist'].iloc[index]
def get_song_id_from_input_artist(metadata, artist):
    song_id_df = metadata[['song_id', 'title', 'artist']].loc[metadata['artist']==artist]
    song_id_df.reset_index(drop=True, inplace=True)
    print(tabulate(song_id_df, headers='keys', tablefmt='github', showindex=True))
    inp = input('Type index of desired song\n')
    song_id = song_id_df['song_id'].iloc[int(inp)]
    return song_id
def get_songs(metadata, artist):
    return tabulate(metadata[['title', 'artist', 'tags']].loc[metadata['artist']==artist], headers='keys', tablefmt='github')
'''Helpers'''




'''Loads Stuff'''

if __name__ == '__main__':
    print('Loading Vectors/Metadata/Model')
t0 = time.time()
data = pd.read_csv('/mnt/work/Music_Proj/metadata.csv', index_col=[0])
vecs = np.load('L2Norm_ALS_vecs.npy')
metadata = utils.create_metadata(data, vecs)
metric = 'dot'
reg = AnnoyIndex(32, metric)
reg.load('/mnt/work/Music_Proj/annoy.ann')
rock = AnnoyIndex(32, metric)  
rock.load('/mnt/work/Music_Proj/rock.ann')
pop = AnnoyIndex(32, metric)
pop.load('/mnt/work/Music_Proj/pop.ann')
indie = AnnoyIndex(32, metric)
indie.load('/mnt/work/Music_Proj/indie.ann')
female_vocalists = AnnoyIndex(32, metric)
female_vocalists.load('/mnt/work/Music_Proj/female vocalists.ann')
electronic = AnnoyIndex(32, metric)
electronic.load('/mnt/work/Music_Proj/electronic.ann')
alternative = AnnoyIndex(32, metric)
alternative.load('/mnt/work/Music_Proj/alternative.ann')

t1 = time.time()
if __name__ == '__main__':
    print('{0} seconds'.format(t1-t0))
'''Loads Stuff'''


def query_by_title(annoy, title, filt=None, match_num=0):
    # Gets Vector for Song Name
    vec = get_vec_from_title(metadata, title)
    # Gets Nearest Neighbors
    similar_vecs_index = annoy.get_nns_by_vector(vec, match_num+1)[match_num]
    # Gets Names of Nearest Neighbors
    
    if filt:
        title, name = get_title_and_artist_from_index(metadata.loc[metadata['tags'].apply(lambda tags: filt in tags)].reset_index(), int(similar_vecs_index))
    else:
        title, name = get_title_and_artist_from_index(metadata, int(similar_vecs_index))
    return '{0} - {1}'.format(title, name)


if __name__ == '__main__':
    def query_by_artist(annoy, artist, filt, num_similars=5):
        print('Initiating Query')
        # Gets Vector for Song Name
        song_id = get_song_id_from_input_artist(metadata, artist)
        title = metadata['title'].loc[metadata['song_id']==song_id].iat[0]
        print('Getting Neighbors for {0} by {1}'.format(title, metadata['artist'].loc[metadata['song_id']==song_id].iat[0]))


        # Gets Nearest Neighbors
        similar_songs = query_by_title(annoy, title, filt, num_similars)
        table = {'Title':[], 'Name':[]}
        for title, name in similar_songs:
            table['Title'].append(title)
            table['Name'].append(name)
        
        return tabulate(similar_songs, headers='keys', tablefmt='github', showindex=False)


    while True:
        try:
            inpt = input("Type Artist Name\n")
            print(query_by_artist(rock, 'rock', inpt))
        except (KeyError, ValueError):
            print("Name not found, try again")

