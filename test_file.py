import CondKNN
from CondKNN import get_songs
import pandas as pd
import numpy as np
from collections import defaultdict

metadata = pd.read_csv('meta_with_vecs.csv', index_col=[0])

filters = ['rock', 'electronic', 'female vocalists', 'indie', 'pop', 'alternative', None]
oldies_songs = ['Scenes From An Italian Restaurant', 'Jump (Album Version)', 'Something Happened On The Way To Heaven', 'Thriller', 'Raspberry Beret', 'Landlord', "What's Going On", 'All Summer Long', 'Original Rags', 'Stairway To Heaven (2007 Remastered LP Version)']
newer_songs = ['Lose Yourself', 'FACK', 'A Reminder', 'Amsterdam', 'Ring The Alarm', 'Sk8er Boi', 'Stronger', 'Lying From You (Album Version)', 'Birthday (Explicit Album Version)', 'Since U Been Gone']
songs_rand = [metadata['title'].iat[ind] for ind in np.random.randint(0, high=metadata.shape[0]+1, size=50)]

popular = defaultdict(list)
rand = defaultdict(list)

for song in oldies_songs:
    popular['Song'].append(song)
    popular['Rock Similars'].append(CondKNN.query_by_title(CondKNN.rock, song, 'rock'))
    popular['Elec Similars'].append(CondKNN.query_by_title(CondKNN.electronic, song, 'electronic'))
    popular['Fem Vocal Similars'].append(CondKNN.query_by_title(CondKNN.female_vocalists, song, 'female vocalists'))
    popular['Indie Similars'].append(CondKNN.query_by_title(CondKNN.indie, song, 'indie'))
    popular['Pop Similars'].append(CondKNN.query_by_title(CondKNN.pop, song, 'pop'))
    popular['Alternative Similars'].append(CondKNN.query_by_title(CondKNN.alternative, song, 'alternative'))
    #popular['General Similars'].append(CondKNN.query_by_title(CondKNN.reg, song))
for song in newer_songs:
    popular['Song'].append(song)
    popular['Rock Similars'].append(CondKNN.query_by_title(CondKNN.rock, song, 'rock'))
    popular['Elec Similars'].append(CondKNN.query_by_title(CondKNN.electronic, song, 'electronic'))
    popular['Fem Vocal Similars'].append(CondKNN.query_by_title(CondKNN.female_vocalists, song, 'female vocalists'))
    popular['Indie Similars'].append(CondKNN.query_by_title(CondKNN.indie, song, 'indie'))
    popular['Pop Similars'].append(CondKNN.query_by_title(CondKNN.pop, song, 'pop'))
    popular['Alternative Similars'].append(CondKNN.query_by_title(CondKNN.alternative, song, 'alternative'))
    #popular['General Similars'].append(CondKNN.query_by_title(CondKNN.reg, song))

for song in songs_rand:
    rand['Song'].append(song)
    rand['Rock Similars'].append(CondKNN.query_by_title(CondKNN.rock, song, 'rock'))
    rand['Elec Similars'].append(CondKNN.query_by_title(CondKNN.electronic, song, 'electronic'))
    rand['Fem Vocal Similars'].append(CondKNN.query_by_title(CondKNN.female_vocalists, song, 'female vocalists'))
    rand['Indie Similars'].append(CondKNN.query_by_title(CondKNN.indie, song, 'indie'))
    rand['Pop Similars'].append(CondKNN.query_by_title(CondKNN.pop, song, 'pop'))
    rand['Alternative Similars'].append(CondKNN.query_by_title(CondKNN.alternative, song, 'alternative'))
    #rand['General Similars'].append(CondKNN.query_by_title(CondKNN.reg, song))


popular = pd.DataFrame(popular)
print(popular)

#popular.to_csv('popular+.csv')
popular.to_html('popular+.html')
rand = pd.DataFrame(rand)
print(rand)
#rand.to_csv('rand1.csv')
rand.to_html('rand1.html')
'''


while True:
    inpt = input('Artist: ')
    print(get_songs(metadata, inpt))
    '''