# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

credits = pd.read_csv("tmdb_5000_credits.csv")


movies_df = pd.read_csv("tmdb_5000_movies.csv")
credits_column_renamed = credits.rename(index=str,columns={"movie_id":"id"})
movies_df_merge = movies_df.merge(credits_column_renamed,on='id')
movies_cleaned_df = movies_df_merge.drop(columns=['homepage','title_x','title_y','status','production_companies'])

r = movies_cleaned_df['vote_average']
v = movies_cleaned_df['vote_count']
m = movies_cleaned_df['vote_count'].quantile(0.70)
c = movies_cleaned_df['vote_average'].mean()

movies_cleaned_df['weighted_average'] = ((r*v)+(c*m))/(v+m)

movies_ranking = movies_cleaned_df.sort_values('weighted_average',ascending=False)
#print(movies_ranking[['original_title','vote_count','vote_average','weighted_average','popularity']].head(20))

scaling = MinMaxScaler()
movie_scaled_df = scaling.fit_transform(movies_cleaned_df[['weighted_average','popularity']])
movie_normalised_df = pd.DataFrame(movie_scaled_df,columns=['weighted_average','popularity'])

movies_cleaned_df[['normalized_weight_average','normalized_popularity']] = movie_normalised_df
movies_cleaned_df['score'] = movies_cleaned_df['normalized_weight_average'] * 0.35 + movies_cleaned_df['normalized_popularity']*0.65
movies_scored_df = movies_cleaned_df.sort_values(['score'],ascending=False)
print(movies_scored_df[['original_title','normalized_weight_average','normalized_popularity','score']].head())