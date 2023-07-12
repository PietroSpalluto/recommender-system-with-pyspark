import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# complete_df = pd.read_csv('data/complete.csv')


games_df = pd.read_csv('data/games.csv')
mechanics_df = pd.read_csv('data/mechanics.csv')
artists_df = pd.read_csv('data/artists_reduced.csv')
designers_df = pd.read_csv('data/designers_reduced.csv')
publishers_df = pd.read_csv('data/publishers_reduced.csv')
themes_df = pd.read_csv('data/themes.csv')
subcategories_df = pd.read_csv('data/subcategories.csv')
rating_distributions_df = pd.read_csv('data/ratings_distribution.csv')
user_ratings = pd.read_csv('data/user_ratings.csv')

# other user ratings
'''
bgg_15m_reviews = pd.read_csv('data/bgg-15m-reviews.csv')
bgg_19m_reviews = pd.read_csv('data/bgg-19m-reviews.csv')
games_info = pd.read_csv('data/games_detailed_info.csv')
games1 = pd.read_csv('data/2020-08-19.csv')
games2 = pd.read_csv('data/2022-01-08.csv')
'''

game_ratings = games_df.set_index('BGGId').join(rating_distributions_df.set_index('BGGId'), on='BGGId', how='left')
game_ratings = game_ratings.join(mechanics_df.set_index('BGGId'), on='BGGId', how='left')
game_ratings = game_ratings.join(artists_df.set_index('BGGId'), on='BGGId', how='left')
game_ratings = game_ratings.join(designers_df.set_index('BGGId'), on='BGGId', how='left', lsuffix='_artist', rsuffix='_designer')
game_ratings = game_ratings.join(publishers_df.set_index('BGGId'), on='BGGId', how='left')
game_ratings = game_ratings.join(themes_df.set_index('BGGId'), on='BGGId', how='left')
game_ratings = game_ratings.join(subcategories_df.set_index('BGGId'), on='BGGId', how='left')
game_ratings.sort_values(by='total_ratings', inplace=True, ascending=False)

n_games = len(games_df)
plt.plot(list(range(0, n_games)), game_ratings['total_ratings'])
plt.ylabel('Total ratings')
plt.xlabel('Games')
plt.title('Rating frequency for all games')
plt.grid()
plt.show()

users = user_ratings.groupby('Username')
users_dict = {}
for group in users.groups:
    users_dict[group] = len(users.groups[group])

users_df = pd.DataFrame({'Username': users_dict.keys(), 'n_reviews': users_dict.values()})
n_users = len(users_df)

users_df.sort_values(by='n_reviews', inplace=True, ascending=False)
plt.figure()
plt.plot(list(range(0, n_users)), users_df['n_reviews'])
plt.ylabel('Total ratings')
plt.xlabel('Users')
plt.title('Rating frequency for all users')
plt.grid()
plt.show()

ratings = rating_distributions_df.columns[1:-1]
n_ratings = rating_distributions_df[ratings].sum()
ratings = list(map(float, ratings))
plt.figure()
plt.bar(ratings, n_ratings, width=0.05)
plt.ylabel('Total ratings')
plt.xlabel('Ratings')
plt.title('Rating frequency')
plt.grid(axis='y')
plt.show()

print('end')
