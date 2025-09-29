import pandas as pd
import numpy as np
import random
from collections import defaultdict

#testing changes

"""
sunny: no/little precipitation, TAVG >= 50th percentile && PRCP <= 10th percentile
cloudy: no/little precipitation, TAVG < 50th percentile && PRCP <= 10th percentile
rainy: significant precipitation, PRCP > 10th percentile
"""

with open('seatac_weather.csv', 'r') as file:
    weather_data = pd.read_csv(file)
    weather_data['DATE'] = pd.to_datetime(weather_data['DATE'], format='%m/%d/%Y')

    total_days = len(weather_data)
    print(f"Total days in dataset: {total_days}")

    TAVG = weather_data['TAVG'].to_numpy() # average temperature
    PRCP = weather_data['PRCP'].to_numpy() # precipitation

PRCP_YES = PRCP[~np.isnan(PRCP) & (PRCP != 0)] # filter out NaN and zero values for PRCP

PRCP_10 = np.percentile(PRCP_YES, 10) # 10th percentile of non-zero PRCP
TAVG_50 = np.percentile(TAVG, 50) # 50th percentile of TAVG
print(f"10th percentile of PRCP: {PRCP_10}")
print(f"50th percentile of TAVG: {TAVG_50}")

sunny_mask = ((PRCP <= PRCP_10) | np.isnan(PRCP)) & (TAVG >= TAVG_50) # sunny condition
        # sunny: little/no precipitation, average temperature >= 50th percentile
cloudy_mask = ((PRCP <= PRCP_10) | np.isnan(PRCP)) & (TAVG < TAVG_50) # cloudy condition
        # cloudy: little/no precipitation, average temperature < 50th percentile
rainy_mask = PRCP > PRCP_10 # rainy condition
        # rainy: significant precipitation

sunny_days = np.sum(sunny_mask)
cloudy_days = np.sum(cloudy_mask)
rainy_days = np.sum(rainy_mask)
print(f"Sunny days: {sunny_days} ({(sunny_days/total_days)*100:.2f}%)")
print(f"Cloudy days: {cloudy_days} ({(cloudy_days/total_days)*100:.2f}%)")
print(f"Rainy days: {rainy_days} ({(rainy_days/total_days)*100:.2f}%)")

weather_sequence = np.empty(total_days, dtype='<U6') # array to hold weather states
        # '<U6' means string of max length 6, 
        # don't need more because 'cloudy' is the longest word
weather_sequence[sunny_mask] = 'sunny'
weather_sequence[cloudy_mask] = 'cloudy'
weather_sequence[rainy_mask] = 'rainy'

# This part calculates the transition matrix, which is the counts of transitions from one state to another

transition_counts = defaultdict(lambda: defaultdict(int))
for (current_state, next_state) in zip(weather_sequence[:-1], weather_sequence[1:]):
    transition_counts[current_state][next_state] += 1

# Converts the transition counts to probabilities
transition_matirx = {}
for state, next_states in transition_counts.items():
    total_transitions = sum(next_states.values())
    transition_matirx[state] = {next_state: count / total_transitions for next_state, count in next_states.items()}

"""
The last two sections made the transition matrix, which is the Markov chain part of this model.
Each entry of the matrix, P_ij, represents the probability of transitioning from state i to state j.
For example, P_sunny_rainy is the probability of transitioning from a sunny day to a rainy day.

This transition matrix is stochastic, meaning that the sum of probabilities from any given state equals 1.
In other words, the values in each row of the matrix sum to 1.

Repeatedly applying the transition matrix allows us to simulate weather over multiple days 
or compute the state distribution after a certain number of days/long-term.
"""



def simulate_weather(days, start_state):
    """
    Behavior: Simulates weather over a specified number of days starting from a given state.
    Parameters:
    - days: int, Number of days to simulate. 
    - start_state: string, Initial weather state ('sunny', 'cloudy', or 'rainy').
    Returns:
    - List of strings representing the weather state for each day in the simulation.
    Throws:
    - ValueError: If start_state is not one of the recognized states.
    """

    if start_state not in transition_matirx:
        raise ValueError(f"Invalid start state: {start_state}. Must be 'sunny', 'cloudy', or 'rainy'.")
    current_state = start_state
    weather_simulation = [current_state]

    for _ in range(days - 1):
        next_states = list(transition_matirx[current_state].keys())
        probabilities = list(transition_matirx[current_state].values())
        current_state = random.choices(next_states, probabilities)[0]
        weather_simulation.append(str(current_state))

    return weather_simulation  

print("weather simulation, 30 days, starting sunny:")
print(simulate_weather(30, 'sunny'))

def state_distribution(steps, initial_distribution):
    """
    Behavior: Computes the state distribution 
              after a specified number of steps given an initial distribution.
    Parameters:
    - steps: int, Number of steps to simulate.
    - initial_distribution: dict, Initial state distribution with keys 'sunny', 'cloudy', 'rainy' 
                                                                and their corresponding probabilities.
    Returns:
    - dict representing the state distribution after the specified number of steps.
    Throws:
    - ValueError: If the initial distribution does not sum to 1 or contains invalid states.
    """

    if not np.isclose(sum(initial_distribution.values()), 1.0):
        raise ValueError("Initial distribution probabilities must sum to 1.")
    for state in initial_distribution.keys():
        if state not in transition_matirx:
            raise ValueError(f"Invalid state in initial distribution: {state}. \
                             Must be 'sunny', 'cloudy', or 'rainy'.")

    current_distribution = initial_distribution.copy()

    for _ in range(steps):
        next_distribution = {state: 0.0 for state in transition_matirx.keys()}
        for state, prob in current_distribution.items():
            for next_state, transition_prob in transition_matirx[state].items():
                next_distribution[next_state] += prob * transition_prob
        current_distribution = next_distribution

    return current_distribution
