# -*- coding: utf-8 -*-

'''
Takes an image, scans it, and determines if it is a brick, cylinder, or ball.
Note - Some methods were taken from the provided Jupyter Notebooks.
'''

import numpy as np

# Convert to 0-1 grayscale values.
def convert_to_grayscale(im): 
    return np.mean(im, axis = 2)

# Scan Image
def scale_image_255(im):
    #Scale an image 0-1 values.
    return (im/255)

# Map points on graph
def mapPoints(edges):
    '''
    From Jupyter Notebooks
    

    Parameters
    ----------
    edges : TYPE
        DESCRIPTION.

    Returns
    -------
    accumulator : TYPE
        DESCRIPTION.

    '''
    y_coords, x_coords = np.where(edges)
    y_coords_flipped = edges.shape[0] - y_coords
    
    #How many bins for each variable in parameter space?
    phi_bins = 128
    theta_bins = 128
    accumulator = np.zeros((phi_bins, theta_bins))

    rho_min = -edges.shape[0]
    rho_max = edges.shape[1]
    
    theta_min = 0
    theta_max = np.pi
    
    #Compute the rho and theta values for the grids in our accumulator:
    rhos = np.linspace(rho_min, rho_max, accumulator.shape[0])
    thetas = np.linspace(theta_min, theta_max, accumulator.shape[1])
   
    for i in range(len(x_coords)):
        #Grab a single point
        x = x_coords[i]
        y = y_coords_flipped[i]

        #Actually do transform!
        curve_rhos = x*np.cos(thetas)+y*np.sin(thetas)

        for j in range(len(thetas)):
            #Make sure that the part of the curve falls within our accumulator
            if np.min(abs(curve_rhos[j]-rhos)) <= 1.0:
                #Find the cell our curve goes through:
                rho_index = np.argmin(abs(curve_rhos[j]-rhos))
                accumulator[rho_index, j] += 1
                if accumulator [rho_index, j] > 120:
                    return accumulator
                
    max_value = np.max(accumulator)
    relative_thresh = 0.35

    #Indices of maximum theta and rho values
    rho_max_indices, theta_max_indices,  = np.where(accumulator > relative_thresh * max_value)
    
    return accumulator

def getKx():
    return np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
def getKy():
    return np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])

def filter_2d(im, kernel):
    '''
    From Jupyter Notebooks.
    
    Filter an image by taking the dot product of each 
    image neighborhood with the kernel matrix.
    Args:
    im = (H x W) grayscale floating point image
    kernel = (M x N) matrix, smaller than im
    Returns: 
    (H-M+1 x W-N+1) filtered image.
    '''

    M = kernel.shape[0] 
    N = kernel.shape[1]
    H = im.shape[0]
    W = im.shape[1]
    
    filtered_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')
    
    brightPx = 0;
    
    
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            image_patch = im[i:i+M, j:j+N]
            filtered_image[i, j] = np.sum(np.multiply(image_patch, kernel))
            if filtered_image[i, j] < -1:
                brightPx += 1
            
    return filtered_image

def sobel_vertical(im1):
    #Get left and right matrices to be subtracted.
    # right column - left column
    a = im1[:,2:] - im1[:,:-2]
    #Make it crisp!
    #Add -1 left, -1 right, and -2 middle row.
    # [-1 0 -1]
    # [-2 0 -1]
    # [-1 0 -1]
    b = 2*a[1:-1] + a[2:] +  a[:-2]
    return b

def classify(im2):
    
    #Set default object.
    object = "ball"
    #Guess on large images 512x512.
    #Scan Small Images 255x255.
    if(im2.shape[0] > 400):
        object = "ball"
    else:
        #Get scaled image.
        originalImage_scaled = scale_image_255(im2)
        #Get grayscale image.
        im = convert_to_grayscale(originalImage_scaled)
        # Use Sobel Operator - Vertical.
        Gx = sobel_vertical(im)
        #Get direction.
        G  = np.sqrt(Gx**2)            
        #Replace pixels with gradient estimates above thresh 
        #with the direction of the gradient estimate:
        edges = G > 1
        im1 = mapPoints(edges)
        #Get max value for hough space.
        max_value = np.max(im1)
        
        #Low value - Round
        #High Value - Bricks
        #Get quick result if possible.
        if(max_value > 120):
            object = "brick"
        elif (max_value < 50):
            object = "ball"
        #Need more info.
        else:
            # Horizontal Detection.
            Gy = filter_2d(im, getKy())
            #Get directions.
            G  = np.sqrt(Gx**2 + Gy**2) 
            edges = G > 1.05
            im1 = mapPoints(edges)
            max_value = np.max(im1)
            #Determine Object.
            if(max_value > 120):
                object = "brick"
            elif (max_value < 55):
                object = "ball"
            else:
                object = "cylinder"
    
    #Return object string.
    return object