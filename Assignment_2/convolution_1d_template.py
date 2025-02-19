import numpy as np

def Convolution_1D(
    array1: np.array, array2: np.array, padding: str, stride: int = 1
) -> np.array:
    """
    Compute the convolution array1 * array2.
    
    Args:
    array1: np.array, input array
    array2: np.array, kernel
    padding: str, padding type ('full' or 'valid')
    stride: int, specifies how much we move the kernel at each step
    
    Carefully look at the formula for convolution in the problem statement, specifically the g[n-m] term.
    What does it indicate?
    Also note the constraints on sizes:
    - For padding='full', the sizes of array1 and array2 can be anything
    - For padding='valid', the size of array1 must be greater than or equal to the size of array2.
    
    Returns:
    np.array, output of the convolution
    """
    
    Lf = len(array1)
    Lg = len(array2)

    # Determine padding size based on the chosen mode
    if padding == 'full':
        # For 'full' padding, we add Lg - 1 zeros on both sides of the input array
        pad_size = Lg - 1
        array1_padded = np.pad(array1, (pad_size, pad_size), mode='constant')
    elif padding == 'valid':
        # For 'valid' padding, no padding is applied
        array1_padded = array1
    else:
        raise ValueError("Padding must be either 'full' or 'valid'.")

    # Length of the padded input array
    padded_length = len(array1_padded)
    
    # Compute the output size based on stride
    output_length = (padded_length - Lg) // stride + 1

    # Initialize the result array
    result = np.zeros(output_length)

    # Perform the convolution
    for i in range(output_length):
        start = i * stride
        end = start + Lg
        result[i] = np.sum(array1_padded[start:end] * array2)
    
    return result

def probability_sum_of_faces(p_A: np.array, p_B:np.array) -> np.array:
    """
    Compute the probability of the sum of faces of two unfair dice rolled together.

    Args:
        p_A (np.array): Probabilities of the faces of die A, from 1 to number of faces of die A
        p_B (np.array): Probabilities of the faces of die B, from 1 to number of faces of die B

    Returns:
        np.array: Probabilities of the sum of faces of die A and die B.
    
    Note that the sum of the faces cannot be 1, and hence the probability array should start from 2.
    For example, given two fair dice with 2 faces each,
    p_A = [0.5, 0.5] and p_B = [0.5, 0.5], the probabilities of sum of faces are

    Sum = 1 => Probability = 0
    Sum = 2 => Probability = 0.25 (1_A, 1_B)
    Sum = 3 => Probability = 0.5 (1_A, 2_B), (2_A, 1_B)
    Sum = 4 => Probability = 0.25 (2_A, 2_B)

    The output should start with the probability of 2, and hence the expected output is [0.25, 0.5, 0.25].
    """
    result = np.convolve(p_A, p_B)

    # Return the result as a numpy array, where the sum starts from 2 (index 0 corresponds to sum = 2)
    return result
