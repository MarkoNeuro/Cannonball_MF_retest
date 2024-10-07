import numpy as np
import json

num_trials = 90
# Pre-determined counts based on probabilities
left_pink_count = int(num_trials * 0.7)
left_purple_count = num_trials - left_pink_count

# For the left cannon: 70% pink, 30% purple
# Generate an array with 1s (pink) and 0s (purple) in the desired proportion
left_cannon_colors = np.array([1] * left_pink_count + [0] * left_purple_count)
np.random.shuffle(left_cannon_colors)  # Shuffle to randomize the sequence

# For the right cannon: 30% pink, 70% purple (inverse of the left cannon)
right_cannon_colors = 1 - left_cannon_colors  # Inverse the colors for the right cannon

trial_info = {}
for trial in range(num_trials):
    trial_info[str(trial)] = {
        "trial": trial,
        "pinkOption1": int(left_cannon_colors[trial]),  # 1 for pink, 0 for purple from the left cannon
        "purpleOption1": 1 - int(left_cannon_colors[trial]),  # Inverse of pinkOption1
        "pinkOption2": int(right_cannon_colors[trial]),  # 1 for pink, 0 for purple from the right cannon
        "purpleOption2": 1 - int(right_cannon_colors[trial]),  # Inverse of pinkOption2
        "pinkExplode": 1,# Assuming pink never explodes
        "purpleExplode": 0, # Assuming purple always explodes
        # Note: The explode chances are fixed
        "pinkExplodeChance": 0,  # Placeholder, as pink never explodes
        "purpleExplodeChance": 1,  #Purple always explodes
        "confidence": 0, 
        "confidenceScaling": 0, 
        "blockedSide": -1
    }

# Convert trial_info to JSON
trial_info_json = json.dumps(trial_info, indent=4)

# Save this JSON to a file
file_path = 'C:/Users/marko/OneDrive/Desktop/PhD/Code/trial_info_model-free.json'
with open(file_path, 'w') as file:
    file.write(trial_info_json)