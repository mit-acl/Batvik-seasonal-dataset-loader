import numpy as np
import rasterio

class OrthoImageLoader:
    def __init__(self,pathToTiffFile,forcedExtent=None):
        # forcedExtent: tuple given in order left, right, top, bottom.
        #  - left is the longitude corresponding with x pixel coordinate 0
        #  - right is the longitude corresponding with x pixel coordinate <maximum>
        #  - top is the latitude corresponding with y pixel coordinate 0
        #  - bottom is the longitude corresponding with y pixel coordinate <maximum>

        self.dataset = rasterio.open(pathToTiffFile,'r')

        if (forcedExtent is not None):
            self.left, self.right, self.top, self.bottom = forcedExtent
        else:           
            self.left = self.dataset.bounds.left
            self.right = self.dataset.bounds.right
            self.top = self.dataset.bounds.top
            self.bottom = self.dataset.bounds.bottom

        self.imagedata = self.dataset.read().transpose(1,2,0)
    
    def plotMap(self,ax,showOnlyRgb=True):
        if (showOnlyRgb):
            imageToShow = self.imagedata[:,:,0:3]
        else:
            imageToShow = self.imagedata

        ax.imshow(imageToShow,extent=[self.left,self.right,self.bottom,self.top],origin='upper')