# -*- coding: utf-8 -*-

# Robert's Cross - Intro
import numpy as np
#Import matplot libraries for using images.
import matplotlib.image as mpimg #Map images.
import matplotlib.pyplot as plt  #Plot images.

def roberts_cross(x):
    '''Compute Robert's Cross of input image x.
       Args: x (nxm) grayscale floating point image
       Returns: (n-1) x (m-1) edge image.'''
        
    edges = np.zeros((x.shape[0]-1,x.shape[1]-1)) 
    #Our output will image will be one pixel smaller than our image

    
    maxLinePixels = 0 
    linePixels = 0
    
    for i in range(x.shape[0]-1):
        for j in range(x.shape[1]-1):
            #Grab Appropriate (2x2) image patch
            image_patch = x[i:i+2, j:j+2]
            # Compute Robert's Cross for image patch
            edges[i, j] = np.sqrt((image_patch[0,0] - image_patch[1, 1])**2 + 
                                   (image_patch[1, 0] - image_patch[0, 1])**2)
            # Threshold values.
            if edges[i, j] > .4:
                linePixels = linePixels + 1
            elif edges[i, j] <= 1 and linePixels >= 0:
                linePixels = linePixels - 1
            else:
                linePixels = linePixels
            
            if maxLinePixels < linePixels:
                    maxLinePixels = linePixels
        
        if maxLinePixels > 20:
            objectTypeFound = "brick"
        elif maxLinePixels > 1.5:
            objectTypeFound = "cylinder"
        else:
            objectTypeFound = "ball"
    
    print(maxLinePixels);
    return edges, objectTypeFound;

def convert_to_grayscale(im):
        '''
        Convert color image to grayscale.
        Args: im = (nxmx3) floating point color image scaled between 0 and 1
        Returns: (nxm) floating point grayscale image scaled between 0 and 1
        '''
        return np.mean(im, axis = 2)
    
def scanImage():
    # Read Images
    originalImage = mpimg.imread('data/easy/brick/brick_5.jpg') 
    #originalImage = mpimg.imread('data/easy/ball/ball_1.jpg') 
    #originalImage = mpimg.imread('data/easy/cylinder/cylinder_1.jpg') 

        
    # Output Images 
    plt.imshow(originalImage)
    
    originalImage_scaled = originalImage/255
       
    #We'll use Robert's notation, and call our grayscale image x
    im = convert_to_grayscale(originalImage_scaled)
    
    im2, objectTypeFound = roberts_cross(im)
    
    #Plot image
    plt.imshow(im2)
    
    classify(objectTypeFound)

def classify(objectTypeFound):
    print ("Scan Determined: " + objectTypeFound);

scanImage()

def classifyChance(newImage):
    '''    
    Args: im (nxmx3) unsigned 8-bit color image 
    Returns: One of three strings: 'brick', 'ball', or 'cylinder'
    '''
    
    #Holds the object types to be scanned for in the images.
    objectTypes = ['brick', 'ball', 'cylinder']
    
    # 33% chance of getting correct object type.
    random_integer = np.random.randint(low = 0, high = 3)
    
    # Return objectType that was determined by the function.
    return objectTypes[random_integer]

# Call classify function.
# Store determined objectType.
type = classifyChance(1)

# Print the determined objectType.
print("Random Chance: " + type)

