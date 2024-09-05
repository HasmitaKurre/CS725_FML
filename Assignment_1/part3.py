import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from closedForm import LinearRegressionClosedForm

np.random.seed(2024)
    
def read_dataset(filepath):
    '''
    This function reads the dataset and creates train and test splits.
    
    n = 500
    n' = 0.9*n

    Args:
      filename: string containing the path of the csv file

    Returns:
      X_train: 2D numpy array of input values for training. Dimensions (n' x 1)
      y_train: 2D numpy array of target values for training. Dimensions (n' x 1)
      
      X_test: 2D numpy array of input values for testing. Dimensions ((n-n') x 1)
      y_test: 2D numpy array of target values for testing. Dimensions ((n-n') x 1)
      
    '''
    # Write your code here
    data = pd.read_csv(filepath)
 
    # Shuffle the data to ensure random distribution
    data = data.sample(frac=1, random_state=2024).reset_index(drop=True)

    # Separate features and target
    X = data.drop(columns=['ID', 'score']).values  # Drop 'ID' and 'score' columns to get features ,# Assuming 'score' is the target and 'ID' is to be ignored
    y = data['score'].values.reshape(-1, 1)  # 'score' column is the target

    
    # Add a bias term (column of ones) to X
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Define the split ratio
    test_ratio = 0.1
    test_size = int(len(data) * test_ratio)

    # Split the data into training and testing sets
    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    """print("X_train shape-", X_train.shape)
    print("y_train shape-", y_train.shape)
    print("X_test shape-", X_test.shape)
    print("y_test shape-", y_test.shape)"""

    return X_train, y_train, X_test, y_test

############################################
#####        Helper functions          #####
############################################

# Terminal text coloring
RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'

if __name__ == '__main__':
    
    print(RED + "##### Closed form solution for linear regression #####")
    
    print(RESET +  "Loading dataset: ",end="")
    try:
        X_train, y_train, X_test, y_test = read_dataset('train.csv')
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()

    print(RESET + "Calculating closed form solution: ", end="")
    try:
        linear_reg = LinearRegressionClosedForm()
        linear_reg.fit(X_train,y_train)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()
    
    print(RESET + "Predicting for test split: ", end="")
    try:
        y_hat = linear_reg.predict(X_test)
        print(GREEN + "done")
    except Exception as e:
        print(RED + "failed")
        print(e)
        exit()

    # Printing some example predictions vs actual values
    print(RESET+ "Actual vs Predicted: ", end="\n")
    for actual, predicted in zip(y_test[:10], y_hat[:10]):
        print(f"Actual: {actual[0]}, Predicted: {predicted[0]:.2f}")

    # Assuming the test set is available
    test = pd.read_csv("test.csv")

    # Prepare test data for prediction
    test_features = test.drop(columns=['ID']).values
    test_features = np.hstack([np.ones((test_features.shape[0], 1)), test_features])

    y_hat_test = linear_reg.predict(test_features)

    # Save predictions
    results = pd.DataFrame({
        'ID': test['ID'],
        'score': y_hat_test.flatten()
    })
    results.to_csv('kaggle.csv', index=False)
