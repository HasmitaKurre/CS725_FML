import numpy as np
from helper import *

def movePatchOverImg(image, filter_size, apply_filter_to_patch):
    #ADD CODE HERE
    if len(image.shape) == 3:  # Check if the image is RGB
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        image = 0.2989 * r + 0.5870 * g + 0.1141 * b
        #save_image("gray.png",image)

    # Determine padding needed
    pad = filter_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Prepare output image of same size as original
    output_image = np.zeros_like(image)

    # Slide over every pixel (patch)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the patch
            patch = padded_image[i:i+filter_size, j:j+filter_size]
            # Apply the filter to the patch and assign the output value
            output_image[i, j] = apply_filter_to_patch(patch)

    return output_image

def detect_horizontal_edge(image_patch):
    #ADD CODE HERE
    kernel = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])
    
    outputval= np.sum(kernel * image_patch)
    return outputval

def detect_vertical_edge(image_patch):
    #ADD CODE HERE
    kernel = np.array([[-1,  0,  1],
                       [-2,  0,  2],
                       [-1,  0,  1]])
    
    outputval= np.sum(kernel * image_patch)
    return outputval

def detect_all_edges(image_patch):
    #ADD CODE HERE
    kernel_horizontal = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])
    
    kernel_vertical = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]])

    # Apply the Sobel filters
    horizontal_edge = np.sum(kernel_horizontal * image_patch)
    vertical_edge = np.sum(kernel_vertical * image_patch)

    # Compute the magnitude of the gradient (all edges)
    outputval = np.sqrt(horizontal_edge**2 + vertical_edge**2)

    # Normalize to 255 scale
    outputval = np.clip(outputval, 0, 255)
    return outputval

def remove_noise(image_patch):
    #ADD CODE HERE
    flattened_patch = image_patch.flatten()
    outval = np.median(flattened_patch)
    return outval

def create_gaussian_kernel(size, sigma):
    #ADD CODE HERE
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    output_kernel= kernel / np.sum(kernel)  # Normalize the kernel to sum to 1
    return output_kernel

def gaussian_blur(image_patch):
    #ADD CODE HERE
    kernel = create_gaussian_kernel(image_patch.shape[0], sigma=1.0)
    outputval= np.sum(kernel * image_patch)
    return outputval

def unsharp_masking(image, scale):
    #ADD CODE HERE
    # Convert the original image to grayscale
    if len(image.shape) == 3:  # Check if the image is RGB
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        image = 0.2989 * r + 0.5870 * g + 0.1141 * b

    # Apply Gaussian blur
    blurred_image = movePatchOverImg(image, 25, gaussian_blur)

    # Calculate the mask (difference between original and blurred image)
    mask = image - blurred_image

    # Add the scaled mask to the original image to sharpen it
    sharp_image = image + scale * mask
    
    # Clip the values to stay in the valid range [0, 255]
    sharp_image = np.clip(sharp_image, 0, 255)
    out= sharp_image.astype(np.uint8)
    return out

#TASK 1  
img=load_image("cutebird.png")
filter_size=3 #You may change this to any appropriate odd number
hori_edges = movePatchOverImg(img, filter_size, detect_horizontal_edge)
save_image("hori.png",hori_edges)
filter_size=3 #You may change this to any appropriate odd number
vert_edges = movePatchOverImg(img, filter_size, detect_vertical_edge)
save_image("vert.png",vert_edges)
filter_size=3 #You may change this to any appropriate odd number
all_edges = movePatchOverImg(img, filter_size, detect_all_edges)
save_image("alledge.png",all_edges)

#TASK 2
noisyimg=load_image("noisycutebird.png")
filter_size=3 #You may change this to any appropriate odd number
denoised = movePatchOverImg(noisyimg, filter_size, remove_noise)
save_image("denoised.png",denoised)

#TASK 3
scale= 1.5#You may use any appropriate positive number (ideally between 1 and 3)
save_image("unsharpmask.png",unsharp_masking(img,scale))
