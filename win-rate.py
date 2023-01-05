#!/usr/bin/env python3

import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('./datasets/matches.csv')

# Create a list of the hero columns in the DataFrame
hero_columns = [col for col in df.columns if col != 'Result']

# Initialize a dictionary to store the number of wins and losses for each hero
hero_stats = {hero: {'wins': 0, 'losses': 0} for hero in hero_columns}

# Iterate through the rows of the DataFrame
for _, row in df.iterrows():
    # Get the result of the match and the heroes played in the match
    result = row['Result']
    heroes = row[hero_columns]

    # Increment the number of wins or losses for each hero played in the match
    for hero, value in heroes.items():
        if value == 1 and result == 1:  # Hero played on Dire team and Dire team won
            hero_stats[hero]['wins'] += 1
        elif value == 0 and result == 0:  # Hero played on Radiant team and Radiant team won
            hero_stats[hero]['wins'] += 1
        elif value != -1:  # Hero played in the match and the opposite team won
            hero_stats[hero]['losses'] += 1

# Calculate the win rate for each hero
hero_win_rates = {hero: stats['wins'] / (stats['wins'] + stats['losses']) for hero, stats in hero_stats.items()}

# Sort the heroes by win rate in descending order
sorted_heroes = sorted(hero_win_rates, key=hero_win_rates.get, reverse=True)

# Print the top 10 heroes by win rate
for hero in sorted_heroes[:10]:
    print(f"{hero}: {hero_win_rates[hero]:.2f}")
