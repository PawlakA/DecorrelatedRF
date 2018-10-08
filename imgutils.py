import numpy as np

NAIVE_SPLIT_POINT = False
CHOICE_NUM_REGIONS = np.array([1,2,4], dtype=np.uint8)

def sampleRandomRegion(shape, size, outshape):
    '''
    Sample random regions from image data.
    Return:     ndarray of row indices, ndarray of columns indices
    '''
    rows = np.zeros(outshape, dtype=np.uint16)
    cols = np.zeros(outshape, dtype=np.uint16)
    # Sample randomly number of regions to be used
    if NAIVE_SPLIT_POINT:
        numberOfRegions = np.array(outshape[0]*[2])
    else:
        numberOfRegions = np.random.choice(CHOICE_NUM_REGIONS, size=outshape[0])
    for i,numberOfRegions_i in enumerate(numberOfRegions):
        # Sample starting points
        rows[i, :numberOfRegions_i] = np.random.randint(shape[0] - size, size=numberOfRegions_i,
                                                        dtype=rows.dtype)
        cols[i, :numberOfRegions_i] = np.random.randint(shape[1] - size, size=numberOfRegions_i,
                                                        dtype=cols.dtype)
    return rows,cols,numberOfRegions

def calculateFeatureValue(images, startingPoint, size):
    '''
    Extract patches from the image and calculate value of the feature using median operator.
    Return:     ndarray with shape (N), where N is the number of images
    '''
    # Flatten the patches of all images in the data set
    patches = images[:, startingPoint[0]:startingPoint[0] + size,
                        startingPoint[1]:startingPoint[1] + size]\
        .reshape(images.shape[0], -1)
    return np.median(patches, axis=1)

def calculateFeatureValue_new(images, startingPoint, size, func_id=0):
    '''
    Extract patches from the image and calculate value of the feature.
    Return:     ndarray with shape (N), where N is the number of images
    '''
    # Initialize the function list
    #funcs = [np.median, np.max, np.min, np.mean]
    # Flatten the patches of all images in the data set
    patches = images[:, startingPoint[0]:startingPoint[0] + size,\
                        startingPoint[1]:startingPoint[1] + size]\
        .reshape(images.shape[0], -1)
    if func_id == 0:
        return np.median(patches, axis=1)
    if func_id == 1:
        return np.max(patches, axis=1)
    if func_id == 2:
        return np.min(patches, axis=1)
    else:
        return np.mean(patches, axis=1)
#return funcs[func_id](patches, axis=1)









