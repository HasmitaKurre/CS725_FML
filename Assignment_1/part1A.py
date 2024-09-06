from matplotlib import pyplot as plt
import numpy as np
import random
random.seed(45)

num_coins = 100
def toss(num_trials):
    '''
    num_trials: number of trials to be performed.
    
    return a numpy array of size num_trials with each entry representing the number of heads found in each trial

    Use for loops to generate the numpy array and 'random.choice()' to simulate a coin toss
    
    NOTE: Do not use predefined functions to directly get the numpy array. 
    '''
    global num_coins
    results = []
    
    for _ in range(num_trials):
        num_heads = 0
        for _ in range(num_coins):
            if random.choice([0, 1]) == 1:  # 1 for heads, 0 for tails
                num_heads += 1
        results.append(num_heads)
    
    return results


def plot_hist(trials):
    '''
    trials: vector of values for a particular trial.

    plot the histogram for each trial.
    Use 'axs' from plt.subplots() function to create histograms. 

    Save the images in a folder named "histograms" in the current working directory.  
    ''' 
    fig, axs = plt.subplots(figsize=(10, 7), tight_layout=True)
    
    # Define bins covering the range from 0 to 100
    bins = list(range(num_coins + 2))  # Creates bins from 0 to 100
    
    # Plot histogram with the full range of bins
    counts, bin_edges, patches = axs.hist(trials, bins=bins, density=False, alpha=1, label=f'NUM_TRIALS= {len(trials)}')
    
    # Customize the bar widths to create gaps
    for patch in patches:
        patch.set_width(0.5 * (bin_edges[1] - bin_edges[0]))  # Adjust width to create gaps

    # Set plot title and labels
    ##axs.set_title(f'Histogram of Number of Heads ({len(trials)} Trials)')
    axs.set_xlabel('Coins')
    axs.set_ylabel('Number of heads')
    
    # Set x-axis limits to focus on the range from 30 to 70
    axs.set_xlim(30, 70)
    
    # Set x-axis ticks and labels to specific values
    specific_ticks = [35, 40, 45, 50, 55, 60, 65]
    axs.set_xticks(specific_ticks)  # Set x-ticks to specific values
    axs.set_xticklabels(specific_ticks)  # Label x-ticks with the same values

    # Add the legend to display the number of trials
    axs.legend(loc='upper right')  # Use 'upper right' to place the legend in the top-right corner

    # Save the histogram
    num_trials = len(trials)
    plt.savefig(f'histograms/hist_{num_trials}.png')
    plt.show()
    plt.close()
   

if __name__ == "__main__":
    num_trials_list = [10, 100, 1000, 10000, 100000]
    for num_trials in num_trials_list:
        heads_array = toss(num_trials)
        plot_hist(heads_array)
