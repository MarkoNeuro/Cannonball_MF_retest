import numpy as np
import json


# Set task parameters
num_trials = 180
switch_every = 20
stable_block_start, stable_block_end = 0, 90
volatile_block_start, volatile_block_end = 90, 180
total_segments = ((num_trials//2) // switch_every) + 1

volatile_block_is_first = False # If True, the volatile block comes first (stable block comes first by default) 
right_cannon_wins = False # If True, the right cannon wins more often (left wins by default)  

# Probabilities for pink balls for the left and right cannons in the stable block
left_pink_prob = 0.7      
right_pink_prob = 0.3

if right_cannon_wins == True:
    left_pink_prob, right_pink_prob = right_pink_prob, left_pink_prob
            
left_purple_prob = 1 - left_pink_prob
right_purple_prob = 1 - right_pink_prob



# Initialize the random number generator
rng = np.random.RandomState(100)
left_pink_count = round((num_trials/2) * left_pink_prob)
left_purple_count = round((num_trials/2) - left_pink_count)
    

# Generate an array with 1s (pink) and 0s (purple) in the desired proportions
# For the left cannon: 70% pink, 30% purple
left_cannon_colors = np.array([1]*left_pink_count + [0]*left_purple_count)
rng.shuffle(left_cannon_colors)

# For the right cannon: 30% pink, 70% purple (inverse of the left cannon)
right_cannon_colors = 1 - left_cannon_colors

# Check the proportion of pink balls fired by each cannon in the stable block
left_pink_proportion_stable = np.mean(left_cannon_colors)
right_pink_proportion_stable = np.mean(right_cannon_colors)
print("Stable block left cannon: ", left_pink_proportion_stable, "Stable block right cannon: ",right_pink_proportion_stable)
print(left_cannon_colors, right_cannon_colors)
# Probabilities for pink balls for the left and right cannons in the volatile block
i=0
 
left_pink_prob, right_pink_prob = right_pink_prob, left_pink_prob
for segment in range(total_segments):
    # Determine the number of pink and purple balls for each segment based on the probabilities
    if segment == 4: # The last segment has 10 trials less
          
        left_pink_count = int((switch_every-10) * left_pink_prob)
        left_purple_count = (switch_every -10) - left_pink_count
        right_pink_count = int((switch_every - 10) * right_pink_prob)
        right_purple_count = (switch_every - 10) - right_pink_count
    else:

        left_pink_count = int(switch_every * left_pink_prob)
        left_purple_count = switch_every - left_pink_count
        right_pink_count = int(switch_every * right_pink_prob)
        right_purple_count = switch_every - right_pink_count

    # Generate the sequence of balls for this segment
    left_segment_colors = [1] * left_pink_count + [0] * left_purple_count
    right_segment_colors = [1] * right_pink_count + [0] * right_purple_count
            
    # Shuffle the sequences to randomize the order of firing in each block separately
    rng.shuffle(left_segment_colors)
    rng.shuffle(right_segment_colors)
    print (left_segment_colors)
    
    print (len(left_segment_colors), len (right_segment_colors))    
    
    #i = ((segment+1) * switch_every) - 1    

    # Append the volatile block to the stable block
    left_cannon_colors = np.concatenate((left_cannon_colors, left_segment_colors))
    right_cannon_colors = np.concatenate((right_cannon_colors, right_segment_colors))  

    # Switch the probabilities every 20 trials after the first 90
    #if segment >= 4:  # Switching starts after the 5th segment (0-based index)
    left_pink_prob, right_pink_prob = right_pink_prob, left_pink_prob

# Generating trial info
if volatile_block_is_first == True:
    temp = np.copy(left_cannon_colors[stable_block_start:stable_block_end])
    left_cannon_colors[stable_block_start:stable_block_end] = left_cannon_colors[volatile_block_start:volatile_block_end]
    left_cannon_colors[volatile_block_start:volatile_block_end] = temp
    temp = np.copy(right_cannon_colors[stable_block_start:stable_block_end])
    right_cannon_colors[stable_block_start:stable_block_end] = right_cannon_colors[volatile_block_start:volatile_block_end]
    right_cannon_colors[volatile_block_start:volatile_block_end] = temp

#a = np.mean(left_cannon_colors[0:90])
#b = np.mean(right_cannon_colors[0:90])
#print("Left cannon: ", a, "Right cannon: ", b)

trial_info = {
    str(trial): {
        "trial": trial,
        "pinkOption1": int(left_cannon_colors[trial]),
        "purpleOption1": 1 - int(left_cannon_colors[trial]),
        "pinkOption2": int(right_cannon_colors[trial]),
        "purpleOption2": 1 - int(right_cannon_colors[trial]),
        "pinkExplode": 1,   # Assuming pink never explodes
        "purpleExplode": 0, # Assuming purple always explodes
        "pinkExplodeChance": 1,  # Placeholder, as pink never explodes
        "purpleExplodeChance": 0,  # Purple always explodes
        "confidence": 0, 
        "confidenceScaling": 0, 
        "blockedSide": -1
    } for trial in range(num_trials)
}

# Convert trial_info to JSON
trial_info_json = json.dumps(trial_info, indent=4)

# Save this JSON to a file
#file_path = 'C:/Users/marko/OneDrive/Desktop/PhD/Code/trial_info_model-free_LW_VF.json'
#with open(file_path, 'w') as file:
#    file.write(trial_info_json)

# Extracting the first 90 trials for the sanity check
#if volatile_block_is_first == True:
 #   left_cannon_pink_stable = left_cannon_colors[90:]
  #  right_cannon_pink_stable = right_cannon_colors[90:]
#else:
#    left_cannon_pink_stable = left_cannon_colors[:90]
 #   right_cannon_pink_stable = right_cannon_colors[:90]



# Check proportions for segments in volatile block
# Starting point of the volatile 90 trials
if volatile_block_is_first == True:
    start_volatile = 0
    
else:
    start_volatile = num_trials // 2

# Iterate over each segment in the last 90 trials
for segment_volatile_start in range(start_volatile, start_volatile + 90, switch_every):    
    segment_volatile_end = segment_volatile_start + switch_every
    if segment_volatile_start == 80:
        segment_volatile_end = segment_volatile_end - 10
    # Calculate the proportion of pink balls for the left and right cannons in this segment
    left_pink_proportion = np.mean(left_cannon_colors[segment_volatile_start:segment_volatile_end])
    right_pink_proportion = np.mean(right_cannon_colors[segment_volatile_start:segment_volatile_end])
    
    # Print the calculated proportions
    print(f"Segment {segment_volatile_start // switch_every + 1}: Left Cannon Pink Proportion = {left_pink_proportion:.2f}, Right Cannon Pink Proportion = {right_pink_proportion:.2f}")

