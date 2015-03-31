#started at 22:37
import os
from scipy.cluster.vq import kmeans
from cluster_functions import makeSiftPoints, makeHistogram, cluster_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

starttime = time.time()
imageslist = os.listdir("test") #set path to frames
imageslist.sort()

for i,image in enumerate(imageslist):
  imageslist[i] = "test/"+imageslist[i] #make absolute path 

allSiftPointsinLists = makeSiftPoints(imageslist)
allSiftPointsConcatenated = np.concatenate(allSiftPointsinLists)

nr_clusters = 5
visualFeatureClusters,distortion = cluster_data(allSiftPointsConcatenated, nr_clusters)
histogramListOfAllImages = makeHistogram(allSiftPointsinLists,np.array(visualFeatureClusters))

print "This process took: ",time.time()-starttime, "seconds"

#np.save("histogramListOfAllImages", histogramListOfAllImages)

# Plot the points!
#image = mpimg.imread(imagename)
#plt.imshow(image)
#plt.plot(densepoints[:,0], densepoints[:,1], 'go', ms=4)
#plt.plot(hespoints[:,0], hespoints[:,1], 'bo', ms=4)
#plt.plot(harpoints[:,0], harpoints[:,1], 'ro', ms=4)
#plt.tight_layout()
#plt.xlim([image.shape[1],0])
#plt.ylim([image.shape[0],0])
#plt.show()