from matplotlib import pyplot as plt
import numpy as np
import random
random.seed(45)

num_coins = 100

def toss(num_trials):
    global num_coins
    results = []
    for i_ in range(num_trials):
        num_heads = 0
        for i_ in range(num_coins):
            if random.choice(['Heads', 'Tails']) == 'Heads':
                num_heads += 1
        results.append(num_heads)
    return np.array(results)
    
def plot_hist(trial, num_trials, y_scale=100.0):
    for i, num_trials in enumerate(num_trials_list):
        results = toss(num_trials)
        plt.subplot(len(num_trials_list), 1, i + 1)
        
        # Get the counts for the histogram
        counts, bins, patches = plt.hist(results, bins=range(32, 69), edgecolor='black', alpha=0.7, width=0.2, label=f'NUM_TRIALS = {num_trials}')        
        # Titles and labels
        plt.title(f'Number of Heads Distribution for {num_trials} Trials')
        plt.xlabel('Coins')
        plt.ylabel('Number of Heads')
        
        # Add the legend to each subplot
        plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_trials_list = [10,100,1000,10000, 100000]
    y_scale = 1 # Example scaling factor
    for num_trials in num_trials_list:
        heads_array = toss(num_trials)
        plot_hist(heads_array, num_trials, y_scale)
