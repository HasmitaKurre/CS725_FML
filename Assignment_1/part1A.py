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
    
    ## Write your code here
    for i_ in range(num_trials):
        num_heads = 0
        for i_ in range(num_coins):
            if random.choice(['Heads', 'Tails']) == 'Heads':
                num_heads += 1
        results.append(num_heads) 

    return np.array(results)


def plot_hist(trial):
    '''
    trial: vector of values for a particular trial.

    plot the histogram for each trial.
    Use 'axs' from plt.subplots() function to create histograms. You can search about how to use it to plot histograms.

    Save the images in a folder named "histograms" in the current working directory.  
    '''
    fig, axs = plt.subplots(figsize =(10, 7), tight_layout=True)
    
    ## Write your code here
    plt.figure(figsize=(15, 10))
    for i, num_trials in enumerate(num_trials_list):
        results = toss(num_trials)
        plt.subplot(len(num_trials_list), 1, i + 1)
        plt.hist(results, bins=range(num_coins + 2), edgecolor='black', alpha=0.7)
        plt.title(f'Number of Heads Distribution for {num_trials} Trials')
        plt.xlabel('Number of Heads')
        plt.ylabel('Frequency')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_trials_list = [10,100,1000,10000,100000]
    for num_trials in num_trials_list:
        heads_array = toss(num_trials)
        plot_hist(heads_array)
