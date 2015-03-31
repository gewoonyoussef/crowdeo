import numpy as np
import sys
from scipy.cluster.vq import kmeans, whiten
import scipy.misc
from sklearn.feature_extraction.image import extract_patches_2d
sys.path.append("pySift")
from pySift import sift, matching

def cluster_data(features, k, nr_iter=25):
    clusters = kmeans(features,k, iter=nr_iter)
    # Return the clusters
    return clusters

def euclidean_distance(x, y):
    # First, make sure x and t are of equal length.
    assert(len(x) == len(y))
    # Initialize the distance.
    d = 0
    D = 0
    # Find out how many dimensions there are.
    nr_dimensions = len(x)
    
    # Compute the distance value.
    # Go over all the dimensions of x and y.
    for point in xrange(0,nr_dimensions):
        D+=(x[point]-y[point])**2
        d=D**0.5
    # Return the proper distance value.
    return d

def cluster_assignment(samples, clusters):
    # Determine the number of samples.
    nr_samples = samples.shape[0]
    
    # Determine the number of clusters.
    nr_clusters = clusters.shape[0]
    
    # FILL THIS ARRAY WITH THE CORRECT VALUES!
    assignments  = np.zeros(nr_samples, dtype=int)
    
    # For each data sample, compute the distance to each cluster.
    # Assign each sample to the cluster with the smallest distance.
    # So, if for sample 3, the closest cluster is cluster 0, then
    # assignments[0] = 3
    # To access the features of sample 5, type: samples[5,:].!
    for i,sample in enumerate(samples):
        distance = []
        for cluster in clusters:
            distance.append(euclidean_distance(sample,cluster))
        assignments[i] = distance.index(min(distance))
        
    return assignments

def create_histogram(samples, clusters):
    # Perform the assignments first.
    assignments = cluster_assignment(samples,clusters)
    #print assignments
    
    # Initialize the histogram.
    histogram   = np.zeros(clusters.shape[0], dtype=np.float)
    
    # Go over all the assignments and place them in the correct histogram bin.
    for bar in xrange(0, clusters.shape[0]):
       # histogram[bar] = list(assignments).count(bar)
        histogram[bar] = sum(assignments==bar)
        
        
        
    # Normalize the histogram such that the sum of the bins is equal to 1.
    histogram /= sum(histogram)
    
    return histogram

def patch_vectors(img,patch_nr,patch_size=(3,3)):
    #We define some useful values
    patch_h,patch_w = patch_size
    
    #We define the output array with the expected size
    output = np.zeros((patch_nr,patch_h*patch_w*3))
    
    #For now, we set the random seed when calling extract_patches_2d
    #This should help your results be comparable
    patches = extract_patches_2d(img,patch_size,patch_nr,random_state=0)
    output[:,:] = np.reshape(patches,(patch_nr,np.prod(np.shape(patches)[1:])))[:,:]
    
    return output

def make_histogram_from_image(imagepath,cl):
    img = scipy.misc.imread(imagepath)
    test_patches = extract_patches_2d(img,(3,3),max_patches=500)
    test_patches = whiten(np.reshape(test_patches,(np.shape(test_patches)[0],np.prod(np.shape(test_patches)[1:]))))
    return create_histogram(test_patches,cl)

def makeSiftPoints(imagelist): 
    trainpoints = []
    for i,image in enumerate(imagelist):
        # Extract point locations from the image using your selected point method and parameters.
   
        # Example for dense sampling.
        #densepoints = sift.densePoints(image, stride=25)
        # Example for hessian interest points.
        sigma       = 1.0
        print "processing image: ",image," number: ",i
        hespoints   = sift.computeHes(image, sigma, magThreshold=15, hesThreshold=10, NMSneighborhood=10)
        # Example for harris interest points.
        harpoints   = sift.computeHar(image, sigma, magThreshold=5, NMSneighborhood=10)
        #allpoints = np.concatenate((densepoints, hespoints, harpoints))
        allpoints = np.concatenate((hespoints, harpoints))
        point2, sift2 = sift.computeSIFTofPoints(image, allpoints, sigma, nrOrientBins=8, nrSpatBins=4, nrPixPerBin=4)
        # Add the point locations to 'trainpoints'.
	trainpoints.append(sift2)
  
    return trainpoints

def makeHistogram(trainpoints,clusters):
    histogramList = []
    for i,image in enumerate(trainpoints):
        print "Making bag-of-words histogram for image number: ",i
        histogramList.append(create_histogram(image, clusters))
    return histogramList
  
