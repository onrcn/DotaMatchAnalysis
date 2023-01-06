#!/usr/bin/env python3

import pandas as pd

df = pd.read_csv('./datasets/matches.csv')
df = df.reset_index(drop=True)
df = df.drop_duplicates()
df = df.dropna()
df = df.drop(['Unnamed: 0'], axis=1)

hero_columns = [col for col in df.columns if col != 'Result']

hero_stats = {hero: {'wins': 0, 'loses': 0} for hero in hero_columns}

for _, row in df.iterrows():
    result = row['Result']
    heroes = row[hero_columns]

    for hero, value in heroes.items():
        if value == 1 and result == 1:
            hero_stats[hero]['wins'] += 1
        elif value == 0 and result == 0:
            hero_stats[hero]['wins'] += 1
        elif value != -1:
            hero_stats[hero]['loses'] += 1

hero_win_rates = {hero: stats['wins'] / (stats['wins'] + stats['loses']) for hero, stats in hero_stats.items()}

sorted_heroes = sorted(hero_win_rates, key=hero_win_rates.get, reverse=True)

import matplotlib.pyplot as plt

top_heroes = sorted_heroes[:10]
bottom_heroes = sorted_heroes[-10:]

plt.bar(top_heroes, [hero_win_rates[hero] for hero in top_heroes], color='g')
plt.bar(bottom_heroes, [hero_win_rates[hero] for hero in bottom_heroes], color='r')

plt.xlabel('Hero Name')
plt.ylabel('Win Rate')
plt.title('Top and Bottom Performing Heroes')

plt.xticks(rotation=45)

plt.show()
