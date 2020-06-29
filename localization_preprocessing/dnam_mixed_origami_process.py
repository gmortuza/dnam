from matplotlib import pyplot as plt

import numpy as np
import os.path as _ospath

from scipy import spatial
from scipy.optimize import minimize

from multiprocessing import Pool, cpu_count
import csv

import picasso_utils as picasso

import argparse

import numba as _numba
import gc
import sys

import xml.etree.ElementTree as ET

#picasso hdf5 format (with averaging): ['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy', 'group']
#Column Name       |	Description                                                                                                                      |	C Data Type
#frame	            |The frame in which the localization occurred, starting with zero for the first frame.	                                                |unsigned long
#x                |The subpixel x coordinate in camera pixels	                                                                                          |float
#y	              |The subpixel y coordinate in camera pixels	                                                                                          |float
#photons	       |The total number of detected photons from this event, not including background or camera offset	                                      |float
#sx	             |The Point Spread Function width in camera pixels                                                                                       |	float
#sy	             |The Point Spread Function height in camera pixels                                                                                      |	float
#bg	             |The number of background photons per pixel, not including the camera offset                                                            |	float
#lpx	         |The localization precision in x direction, in camera pixels, as estimated by the Cramer-Rao Lower Bound of the Maximum Likelihood fit.  |	float
#lpy	         |The localization precision in y direction, in camera pixels, as estimated by the Cramer-Rao Lower Bound of the Maximum Likelihood fit.  |	float


class LocalizationCluster:
    def __init__(self,grid,localizations,globalDeltaX,recenter):
        """Class container for localization and grid points, fitting functions"""
        #grid NX2 array of x,y localizations NX3 array of x,y,prec
        self.grid = np.copy(grid)
        self.gridTran = np.copy(grid)
        self.localizations = np.copy(localizations)
        self.globalDeltaX = globalDeltaX
        #self.weight = 1/np.sqrt(localizations[:,2]**2 + self.globalDeltaX**2)
        self.nnTree = []
        self.nn = []
        self.dx = 0
        self.dy = 0
        self.dt = 0
        
        
        #Center grid and localizations on 0,0
        xGAve = 0
        yGAve = 0
        gCount = 0
        self.xLAve = 0
        self.yLAve = 0
        lCount = 0
        self.uAve=0
        for row in range(self.grid.shape[0]):
            xGAve+=self.grid[row,0]
            yGAve+=self.grid[row,1]
            gCount+=1
            
        for row in range(self.localizations.shape[0]):
            self.xLAve+=self.localizations[row,0]
            self.yLAve+=self.localizations[row,1]
            self.uAve += self.localizations[row,2]
            lCount+=1
        
        xGAve/=gCount
        yGAve/=gCount
        
        self.xLAve/=lCount
        self.yLAve/=lCount
        self.uAve/=lCount
        self.locCount = lCount
        
        self.xCAve = self.xLAve
        self.yCAve = self.yLAve
        
        if recenter:
            for row in range(self.grid.shape[0]):
                self.grid[row,0]-=xGAve
                self.grid[row,1]-=yGAve
                
            for row in range(self.localizations.shape[0]):
                self.localizations[row,0]-=self.xLAve
                self.localizations[row,1]-=self.yLAve
                
            self.xCAve = 0
            self.yCAve = 0
                
            #self.roughClock()
            
        self.weightDistMatrix = np.zeros((self.localizations.shape[0],self.gridTran.shape[0]))
        self.wdmComputed = False
        self.objectiveValue = 0
        self.hist = []
        self.signal = []
        self.hist2 = np.ones((self.gridTran.shape[0]+1))
        self.rec = []
        self.kernel=[]
        self.sigma = 0
        self.cHull = spatial.ConvexHull(self.localizations[:,0:2])
        self.area = self.cHull.volume #For 2d points, the internal area is the volume reported by ConvexHull
        
    def transformGrid(self,tm):
        self.dx = tm[0]
        self.dy = tm[1]
        self.dt = tm[2]
        rotMat = np.array([[np.cos(self.dt),-np.sin(self.dt)],[np.sin(self.dt),np.cos(self.dt)]])
        self.gridTran = np.dot(self.grid,rotMat)
        self.gridTran[:,0]+=self.dx
        self.gridTran[:,1]+=self.dy
        self.nnTree = spatial.cKDTree(self.gridTran)
        self.nn = self.nnTree.query(self.localizations[:,0:2])
        self.wdmComputed = False
    
    def squareNNDist(self,tm):
        """Weighted mean square nearest neighbor distantce, computed by translating grid"""
        self.transformGrid(tm)
        weight = 1/np.sqrt(self.localizations[:,2]**2 + self.globalDeltaX**2)
        self.objectiveValue = float(sum(np.multiply(self.nn[0],weight)**2))/self.locCount
        
        return self.objectiveValue
    
    def squareNNDistFixed(self,tm):
        """Weighted mean square nearest neighbor distance, computed by translating localizations. Faster for large grids"""
        self.dx = tm[0]
        self.dy = tm[1]
        self.dt = tm[2]
        
        if self.nnTree == []:
            self.nnTree = spatial.cKDTree(self.gridTran)
            
        rotMat = np.array([[np.cos(self.dt),np.sin(self.dt)],[-np.sin(self.dt),np.cos(self.dt)]])
        localizationsTran = np.dot(self.localizations[:,0:2],rotMat)
        localizationsTran[:,0]-=self.dx
        localizationsTran[:,1]-=self.dy
        weight = 1/np.sqrt(self.localizations[:,2]**2 + self.globalDeltaX**2)
        self.nn = self.nnTree.query(localizationsTran)
        self.wdmComputed = False
        self.objectiveValue = float(sum(np.multiply(self.nn[0],weight)**2))/self.locCount
        
        return self.objectiveValue
    
    
    def squareNNDistUnweighted(self,tm,gridsize):
        """unweighted mean square nearest neighbor distantce, computed by translating grid"""
        self.dx = tm[0]
        self.dy = tm[1]
        self.dt = tm[2]
        rotMat = np.array([[np.cos(self.dt),-np.sin(self.dt)],[np.sin(self.dt),np.cos(self.dt)]])
        self.gridTran = np.dot(self.grid,rotMat)
        self.gridTran[:,0]+=self.dx
        self.gridTran[:,1]+=self.dy
        self.nnTree = spatial.cKDTree(self.gridTran)
        self.nn = self.nnTree.query(self.localizations[:,0:2])
        self.wdmComputed = False
        return float(sum((2*self.nn[0]/gridsize)**2))/self.locCount
    
    
    def rmsDist(self):
        """RMS distance from origin"""
        return self._rmsDistComp(self.localizations)
    
    @staticmethod
    @_numba.jit(nopython = True)
    def _rmsDistComp(localizations):
        locCount = localizations.shape[0]
        sqDist = 0
        
        for row in range(locCount):
            sqDist += localizations[row,0]**2 + localizations[row,1]**2
            
        sqDist/=locCount
        return sqDist
    
    #@_numba.jit(nopython = True, nogil = True)
    def roughClock(self):
        """Find approximate clocking"""
        minObj = self.squareNNDist([self.dx,self.dy,0])
        minTheta = 0
        for thetaId in range(360):
            theta = thetaId*np.pi/180
            obj = self.squareNNDist([self.dx,self.dy,theta])
            if obj<minObj:
                minObj = obj
                minTheta=theta
        self.dt = minTheta
        self.wdmComputed = False
        return self.squareNNDist([self.dx,self.dy,self.dt])
    
    def computeWeightDistMatrix(self):
        """Compute weighted distance likelihood value, used for likelihood score"""
        #for locId in range(self.localizations.shape[0]):
        #    for gridId in range(self.gridTran.shape[0]):
        #        self.weightDistMatrix[locId,gridId] = np.exp(-((self.localizations[locId,0]-self.gridTran[gridId,0])**2+(self.localizations[locId,1]-self.gridTran[gridId,1])**2)/(2.0*(self.localizations[locId,2]**2+self.globalDeltaX**2)))
        self.weightDistMatrix = self._computeWeightDistMatrixComp(self.localizations[:,0:3],self.gridTran,self.localizations.shape[0],self.gridTran.shape[0],self.globalDeltaX,self.area)
        self.wdmComputed=True
    
    @staticmethod
    @_numba.jit(nopython = True)
    def _computeWeightDistMatrixComp(localizations,grid,locCount,gridCount,globalDeltaX,area):
        weightDistMatrix = np.zeros((locCount,gridCount+1))

        for locId in range(locCount):
            for gridId in range(gridCount):
                sigma2 = (localizations[locId,2]**2+globalDeltaX**2)
                weightDistMatrix[locId,gridId] = (1.0/2.0/np.pi/sigma2)*np.exp(-((localizations[locId,0]-grid[gridId,0])**2+(localizations[locId,1]-grid[gridId,1])**2)/(2.0*sigma2))
            weightDistMatrix[locId,gridCount] = 1/area
        return weightDistMatrix

    def likelihoodScore(self,image): 
        """log likelihood function for localization positions given intensity values at grid points"""
        if not self.wdmComputed:
            self.computeWeightDistMatrix()
        
        score = self._likelihoodScoreComp(self.weightDistMatrix,image,self.locCount)
        
        return score
    
    
    @staticmethod
    @_numba.jit(nopython = True)
    def _likelihoodScoreComp(weightDistMatrix,image,locCount):

        score = -(np.sum(np.log(np.dot(weightDistMatrix,image))))
        
        imageSum = np.sum(image)
        if imageSum <= 0:
            imageSum = .000001
        score -= imageSum*np.log(locCount/imageSum) + imageSum
        
        return score
    
    def pixelate(self,gridShape,distThreshold):
        """Produce maximum likelihood pixelated image"""
        if self.nnTree == []:
            self.nnTree = spatial.cKDTree(self.gridTran)
            self.nn = self.nnTree.query(self.localizations[:,0:2])
            
        self.localizations = self.localizations[self.nn[0]<distThreshold,:]
        self.locCount = self.localizations.shape[0]
        
        self.cHull = spatial.ConvexHull(self.localizations[:,0:2])
        self.area = self.cHull.volume #Ugh, scipy
            
        self.hist = np.zeros((self.gridTran.shape[0]+1))
        
        for idx in self.nn[1]:
            self.hist[idx] += 1
      
        bounds = [(0,1e10) for item in self.hist]
        
        out = minimize(self.likelihoodScore,self.hist,bounds=bounds, method="L-BFGS-B",options={'eps':1e-10})
        self.hist2 = out.x
        #print(self.hist2[-1])
        
        Iscale = max(self.hist)*2     
        self.signal = (self.hist[:-1].reshape(gridShape))/Iscale
        self.rec = self.hist2[:-1].reshape(gridShape)
        self.rec *= sum(sum(self.signal))/sum(sum(self.rec))*Iscale
        
        return self.likelihoodScore(self.hist2)
    
    def fitAndPixelate(self,gridShape,distThreshold,maxIter,toler):
        """Fit grid and produce maximum likelihood pixelated image"""
            
        self.hist = np.zeros((self.gridTran.shape[0]+1))
        
        
      
        
        #bounds = [(-.5,.5),(-.5,.5),(-np.pi,3*np.pi)]
        bounds =[(0,1e10) for item in self.hist]
        
        self.roughClock()
        self.hist2 = np.ones((self.gridTran.shape[0]+1))
        lastFVal = self.likelihoodScore(self.hist2)
        #print(lastFVal)
        it=0
        for _ in range(maxIter):
            it+=1
            out1 = minimize(self.gridLikelihoodScore,[self.dx,self.dy,self.dt],method='Nelder-Mead')
            self.transformGrid(out1.x)
            
            #for idx in self.nn[1]:
            #    self.hist[idx] += 1
            
            out2 = minimize(self.likelihoodScore,self.hist2,bounds=bounds, method="L-BFGS-B", options={'eps':1e-10})
            self.hist2 = out2.x
            
            out3 = minimize(self.gdxLikelihoodScore,[self.globalDeltaX],bounds=[(0,1)],method="L-BFGS-B")
            self.globalDeltaX = out3.x[0]
            FVal = out3.fun
            #print(FVal)
            #print(2.0*np.abs(lastFVal-FVal)/(lastFVal+FVal), toler)
            
            if it==3:
                self.localizations = self.localizations[self.nn[0]<distThreshold,:]
                self.locCount = self.localizations.shape[0]
                self.cHull = spatial.ConvexHull(self.localizations[:,0:2])
                self.area = self.cHull.volume #Ugh, scipy
                #print(self.area,self.cHull.volume)
            
            if 2.0*np.abs((lastFVal-FVal)/(lastFVal+FVal)) < toler and it > 4:
                break
            lastFVal = FVal
        #print(self.hist2[-1])
        
        self.transformGrid([self.dx,self.dy,self.dt])
        for idx in self.nn[1]:
            self.hist[idx] += 1
        
        Iscale = max(self.hist)*2     
        self.signal = (self.hist[:-1].reshape(gridShape))/Iscale
        self.rec = self.hist2[:-1].reshape(gridShape)
        self.rec *= sum(sum(self.signal))/sum(sum(self.rec))*Iscale
        
        return out3.fun,it
    
    def gridLikelihoodScore(self,tm):
        """Likelihood score for current image (hist2) given grid coordinate"""
        self.transformGrid(tm)
        image = np.copy(self.hist2)
        image *= self.locCount/sum(image)
        return self.likelihoodScore(image)
    
    def gdxLikelihoodScore(self,gdx):
        """Likelihood score for current image (hist2) given global delta x"""
        self.globalDeltaX = gdx[0]
        self.wdmComputed = False
        image = np.copy(self.hist2)
        image *= self.locCount/sum(image)
        return self.likelihoodScore(image)
    
# =============================================================================
        #This doesn't seem to converge
#     def fullLikelihoodScore(self,tm):
#         self.transformGrid(tm[0:3])
#         image = tm[3:]
#         return self.likelihoodScore(image)
# =============================================================================
        
        
class RandomClusters:   
    def __init__(self,grid,locs,radius,minCountThreshold,maxCountThreshold,isoRadius,isoRatio,globalDeltaX):
        """Find random clusters fitting certain properties from localizations. Uses LocalizationCluster class"""
        self.locs = locs
        self.nnTree = []
        self.radius = radius
        self.minCountThreshold = minCountThreshold
        self.maxCountThreshold = maxCountThreshold
        self.isoRadius = isoRadius #Additional radius to search to check for isolation of cluster
        self.isoRatio = isoRatio #Ratio of localization that must be iwthin inner radius
        
        self.sublocalizations=[]
        self.grid = grid
        self.clusterLocations = []
        self.localizations = []
        self.updateLocalizations()
        self.locCount = self.localizations.shape[0]
        self.globalDeltaX = globalDeltaX
        
    def updateLocalizations(self):
        """Updates localizations and tree if locs has changed through another process"""
        x = self.locs['x']
        y = self.locs['y']
        lpx = self.locs['lpx']
    
        self.localizations = np.array([x,y,lpx,[i for i in range(len(x))]]).T
        self.nnTree = spatial.cKDTree(self.localizations[:,0:2])
        
    def findClusters(self,samples,maxAttempts):
        """Attempts to find random clusters, only tries maxAttempts times"""
        localizationClusters = []
        self.clusterLocations = []
        
        clusterLocationTree = []
        
        attempts = 0
        
        while len(localizationClusters) < samples and attempts < maxAttempts:
            attempts+=1
            randIndex = int(np.random.sample()*self.locCount)
            
            if len(self.clusterLocations) > 0:
                clusterNN = clusterLocationTree.query(self.localizations[randIndex,0:2])
                
                if clusterNN[0] < self.isoRadius:
                    continue
            
            sublocs_id = self.nnTree.query_ball_point(self.localizations[randIndex,0:2],2 * self.radius)
            
            firstCounts = len(sublocs_id)
            
            if firstCounts < self.minCountThreshold:
                continue
            self.sublocalizations = self.localizations[sublocs_id,:]
            lc = LocalizationCluster(self.grid,self.sublocalizations,self.globalDeltaX,True)
            
            subTree = spatial.cKDTree(lc.localizations[:,0:2])
            sublocs_id = subTree.query_ball_point(np.array([0,0]),self.radius)
            isoCounts = len(subTree.query_ball_point(np.array([0,0]),self.isoRadius))
            
            secondCounts = len(sublocs_id)

            if secondCounts < self.minCountThreshold or secondCounts > self.maxCountThreshold or secondCounts < self.isoRatio * isoCounts:
                continue
            
            
            self.sublocalizations = self.sublocalizations[sublocs_id,:]
            lc = LocalizationCluster(self.grid,self.sublocalizations,self.globalDeltaX,True)
            
            if lc.rmsDist() > 3 * self.radius / 4:
                continue
            
            self.clusterLocations.append([lc.xLAve,lc.yLAve])
            clusterLocationTree = spatial.cKDTree(np.array(self.clusterLocations))
            
            localizationClusters.append(lc)
            
        return localizationClusters
            

def clusterFit(lc,index,gridShape,distThreshold):
    """Multiprocess pool friendly cluster fitting algorithm"""
    print("Starting fit for cluster",index)
    #lc.roughClock()
    #method = "Nelder-Mead"
    #out = minimize(lc.squareNNDist,[lc.dx,lc.dy,lc.dt],method=method)
    #lc.squareNNDist(out.x)
    #minimizer_kwargs = {"method":method}
    #out = basinhopping(lc.squareNNDist,x0=out.x,T=.5,niter=200,minimizer_kwargs=minimizer_kwargs,stepsize = 0.25)
    #out = minimize(lc.squareNNDist,out.x,method=method)
    #bounds = np.array([[-.225+out.x[0],.225+out.x[0]],[-.225+out.x[1],.225+out.x[1]],[-np.pi+out.x[2],np.pi+out.x[2]]])    
    #out = dual_annealing(lc.squareNNDist,bounds,x0=out.x,local_search_options={'method':method},maxiter=2000)#,inital_temp=5e4,restart_temp_ratio=2e-6)
    out,it = lc.fitAndPixelate(gridShape,distThreshold,50,1e-6)
    obj = lc.squareNNDist([lc.dx,lc.dy,lc.dt])
    
    print("Fit complete for cluster",index,"Objective:",obj,"FVal:",out,"Iterations:",it,"bg:",lc.hist2[-1])
    
    return [index,lc.dx,lc.dy,lc.dt,obj,out,lc.globalDeltaX,lc.hist2]

def validateFilters(filters,DTYPE):
    """Checks if filter object derived from xmltree.root is valid"""
    dtype_dict = dict(DTYPE)
    for child in filters:
        if not child.tag in dtype_dict:
            return -1
        if (not 'type' in child.attrib) or (not 'low' in child.attrib) or (not 'high' in child.attrib):
            return -2
        if (not child.attrib['type'] == 'absolute') and (not child.attrib['type'] == 'percentile'):
            return -3
        try:
            low = float(child.attrib['low'])
        except ValueError:
            return -4
        try: 
            high = float(child.attrib['high'])
        except ValueError:
            return -5
        if (child.attrib['type'] == 'percentile') and (low < 0 or low > 1 or high < 0 or high >1):
            return -6
        if low >= high:
            return -7
        
    return 0


if __name__ == '__main__':       
    # parsing the arguments
    parser = argparse.ArgumentParser(description="dNAM origami process script")
    parser.add_argument("-f", "--file", help="File name", default="")
    parser.add_argument("-v", "--verbose", help="Print details of execution on console", action="store_true")
    parser.add_argument("-N", "--number-clusters", help="Number of clusters to analyze", type=int,default=0)
    parser.add_argument("-s", "--skip-clusters", help="Number of clusters to skip", type=int,default=0)
    parser.add_argument("-d", "--drift-correct-size", help="Final size of drift correct slice (0 is no drift correction))", type=int,default=0)
    parser.add_argument("-ps", "--pixel-size", help="Pixel size. Needed if reading Thunderstorm csv",type=float,default=0)
    parser.add_argument("-x", "--xml", help="XML config file for 3d-daostorm analyis (default = default_config.xml)",default="default_config.xml")
    parser.add_argument("-ft", "--filter-file", help="XML filter file for post-drift correct filtering (default is no filters)", default="")
    parser.add_argument("-gf", "--grid-file",help="CSV file containing x,y coordinates of grid points (default is DNAM average grid)", default="")
    parser.add_argument("-gr", "--grid-shape-rows",help="Rows in the grid", type=int, default=6)
    parser.add_argument("-gc", "--grid-shape-cols",help="Columns in the grid", type=int, default=8)
    parser.add_argument("-md", "--min-drift-clusters",help="Min number of cluster to attempt fine drift correction (default 5000)", type=int, default=5000)
    parser.add_argument("-gdx", "--global-delta-x",help="Starting guess for global localization precision due to drift correct, etc", type=float, default=.025)
    parser.add_argument("-st","--scaled-threshold",help="Threshold for binary counts, as a fraction of the average of the 10 brightest points",type=float,default=.25)
    parser.add_argument("-rf","--redo-fitting",help="Redo grid fitting, even if fitted grid data exists (has no effect on data without fits)",action="store_true")
    #parser.add_argument("-rd", "--redrift-correct",help="Apply secondary drift correction, post drift (must have >500 clusters)",action="store_true")

    args = parser.parse_args()
    
    gridShape = (args.grid_shape_rows,args.grid_shape_cols)
    
    #set group number
    groupNumbers = [i for i in range(0,args.number_clusters,1)]
    
    distThreshold = 5
    
    globalDeltaX = args.global_delta_x
    
    scaled_threshold = args.scaled_threshold
    
    clusterRadius = 0.55
    minCountThreshold = 1000
    maxCountThreshold = 20000
    objectiveThreshold = 1.5
    distThreshold = .1
    
    minDriftClusters = args.min_drift_clusters
    
    samples = args.number_clusters
    skip = args.skip_clusters
    #redrift = args.redrift_correct
    redrift = False
    
    np.random.seed(2)
    
    display = False
    display2 = True
    verbose = args.verbose
    
    pixelsize = args.pixel_size
    
    outputClusters = True
    
    if samples > 20:
        display2=False
    
    reuse_file = False
    try:
        lastFilePath = filePath
    except NameError:
        lastFilePath = ""
    
    filePath = args.file
    
    driftCorrectSize = args.drift_correct_size
    
    filterFile = args.filter_file
    
    
    
    LOCS_DTYPE = [
                    ("frame", "u4"),
                    ("x", "f4"),
                    ("y", "f4"),
                    ("photons", "f4"),
                    ("sx", "f4"),
                    ("sy", "f4"),
                    ("bg", "f4"),
                    ("lpx", "f4"),
                    ("lpy", "f4"),
                    ("group", "i4"),
                                ]
    
    if not filterFile == "":
        filters = ET.parse(filterFile).getroot()
        chk =  validateFilters (filters,LOCS_DTYPE)
        if chk < 0:
            print("Error: invalid filter XML file. Error code:",chk)
            sys.exit("Invalid XML")
        
    #driftCorrectSize = 0
    
    #set filepath
    #filtered by photons averaged dataset
    #filePath = r"20190913_All-Matrices_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM_13_42_03_fixed_locs_render_DRIFT_3_filter_picked_manual_avg.hdf5"
    #filePath = r"20190913_george_test2_conv_locs_render_render_picked_filter.hdf5"
    #filePath = r"20190913_All-Matrices_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM_13_42_03_fixed_locs_render_DRIFT_3_filter_picked_automatic.hdf5"
    #filePath = r"20190930_Matrix-3_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM_11_18_45_fixed_locs_render_DRIFT_4_filter_picked_manual.hdf5"
    #filePath = r"20190913_All-Matrices_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM_10_38_52_substack_fixed_locs_render_DRIFT_3_filter_picked_avg.hdf5"
    #filePath = r"20190905_Matrix2_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM_14_16_10_fixed_locs_picked_avg_filter.hdf5"
    
    if filePath == '':
        #filePath = r"20190913_george_test2_conv_locs_render_render_filter.hdf5"
        filePath = r"20190913_All-Matrices_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM_10_38_52_substack_fixed_locs_render_DRIFT_3_filter.hdf5"
    #filePath = r"20191002_Matrix_7_Triangles_fixed_locs_DRIFT_3.hdf5"
    #filePath = r"20190909_Matrix14_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM_11_39_47_fixed_locs_render_DRIFT_3.hdf5"
    #filePath = r"20190905_Matrix2_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM_14_16_10_fixed_locs_render_DRIFT_3_filter.hdf5"
    #filePath = r"20190830_Matrix5_syn2_pure_Triangles_300msExp_Mid-9nt-3nM_MgCl2_18mM_PCA_12mM_PCD_TROLOX_1mM_10_44_32_fixed_locs_render_DRIFT_3_filter.hdf5"
    
    #filePath = "/data/williamclay/"+filePath
    
    if not filePath == lastFilePath:
        reuse_file = False
        
    #baseFilePath = filePath[0:-5]
    baseFilePath, extension = _ospath.splitext(filePath)
        
    
    
    if not reuse_file:
        if 'hdf5' in extension:
            #try to open picasso hdf5 file as np.recarray
            try:
                locs, info = picasso.load_locs(filePath)
            except KeyError:
                #If picasso load failed, try Zhuanglab SA load
                print("Converting hdf5 file")
                locs, info = picasso.load_salocs(filePath)
                baseFilePath += "_locs"
                picasso.save_locs(baseFilePath+".hdf5",locs,info)
            
        elif 'csv' in extension:
            #Load ThunderStorm format csv
            while pixelsize <= 0:
                print("Please enter pixelsize (use -ps pixelsize to avoid this message):")
                pixeltext = input()
                try:
                    pixelsize = float(pixeltext)
                except ValueError:
                    pixelsize = -1
            locs, info = picasso.csv2locs(filePath,pixelsize)
            baseFilePath += "_locs"
            picasso.save_locs(baseFilePath+".hdf5",locs,info)
        elif 'txt' in extension:
            #Load 3d_daostorm format txt
            locs, info = picasso.txt2locs(filePath)
            picasso.save_locs(baseFilePath+"_locs.hdf5",locs,info)
        elif 'nd2' in extension or 'spe' in extension or 'tif' in extension or 'dax' in extension:
            #load movie
            import mufit_analysis_nd as mufit
            while driftCorrectSize <= 0:
                print("Drift correction required for analysis of movie file. Please enter frame segment size. Recommend 100-200 (use -d segment_size to avoid this message):")
                drifttext = input()
                try:
                    driftCorrectSize = float(drifttext)
                except ValueError:
                    driftCorrectSize = -1
            print("Starting localization analysis. Config file: "+args.xml)
            baseFilePath += "_sa"
            saFilePath = baseFilePath+".hdf5"
            mufit.analyze(filePath,saFilePath,args.xml)
            print("Converting hdf5 file")
            locs, info = picasso.load_salocs(saFilePath)
            baseFilePath += "_locs"
            picasso.save_locs(baseFilePath+".hdf5",locs,info)
        else:
            print("Unrecognized file extension (support hdf5 (picasso of 3d_dastorm format), csv (ThunderStorm), txt (3d_daostorm), nd2, spe, tif)")
            sys.exit("Unknown file type")
            
        #check header
        headers = locs.dtype.names
        if verbose:
            print(headers)
            print(locs.shape)
        
        if 'group' in headers:
            picked = True
        else:
            picked = False
            groupNumbers = [i for i in range(samples)]
        
        #guassian blur
        x = locs['x']
        y = locs['y']
        nBins = 750
        
        fitting = True
        
        if "fitpicks" in baseFilePath and not args.redo_fitting:
            gridLocs,gridInfo = picasso.load_locs(baseFilePath+"_grid.hdf5")
            fitting = False
            picked = True

        if not fitting and driftCorrectSize > 0:
            driftCorrectSize = 0 
            print("Warning: Cannot do drift correction when reusing fits. Drift correction will not be done")
        
        if (not fitting) and (not filterFile == ""):
            filterFile = ""
            print("Warning: Cannot do filtering when reusing fits. Filters will not be applied")
    
        if picked:
            outputClusters = False #Not yet implemented for picked clusters
            NN = len(groupNumbers)
            
            if not driftCorrectSize == 0:
                driftCorrectSize = 0
                print("Warning: drift correction not implemented for picked localizations")
                
            if redrift:
                redrift = False
                print("Warning: drift correction not implemented for picked localizations")
        
        
        
        if args.grid_file == "":
            centeroids = np.array([
                    
                       [255.65026699, 256.2699313 ],
                       [255.65522196, 255.9542909 ],
                       [255.65336587, 256.16112367],
                       [255.66110178, 255.74295807],
                       [255.74678208, 256.05605212],
                       [255.74547016, 256.16300094],
                       [255.75104081, 256.2720415 ],
                       [255.75228561, 255.84415798],
                       [255.75351785, 255.95273798],
                       [255.75479689, 255.73969906],
                       [255.84866381, 256.05934078],
                       [255.84685329, 256.16708376],
                       [255.85103522, 256.27369471],
                       [255.85160587, 255.74273925],
                       [255.85740968, 255.85026562],
                       [255.94392418, 256.27673285],
                       [255.94391768, 256.05966314],
                       [255.94611554, 256.17014591],
                       [255.95094813, 255.95014021],
                       [255.95158306, 255.85177168],
                       [255.95400055, 255.74037337],
                       [256.03677785, 256.2720229 ],
                       [256.03969133, 256.06114094],
                       [256.04283514, 256.16603679],
                       [256.04661846, 255.85896198],
                       [256.04490558, 255.96022842],
                       [256.05086907, 255.7461775 ],
                       [256.14012688, 256.1753788 ],
                       [256.14023585, 256.0697952 ],
                       [256.140229  , 256.2836567 ],
                       [256.14513271, 255.75414015],
                       [256.14299781, 255.96175843],
                       [256.2314431 , 256.17345542],
                       [256.23032086, 256.28052939],
                       [256.23801515, 256.06644361],
                       [256.23637598, 255.96237802],
                       [256.2397271 , 255.75018973],
                       [256.23966435, 255.85825364],
                       [256.32819487, 256.17121161],
                       [256.33097833, 255.75340018],
                       [256.32980017, 256.06243964],
                       [256.32933863, 255.96060716],
                       [256.33441629, 255.85686116],
                       [256.144     , 255.865     ],
                       [255.851     , 255.954     ],
                       [255.652     , 256.057     ],
                       [255.659     , 255.846     ],
                       [256.327     , 256.281     ]]
            
            )
            
            
            
            remap={  33:2,
                     29:3,
                     21:4,
                     15:5,
                     12:6,
                     6:7,
                     0:8,
                     38:9,
                     32:10,
                     27:11,
                     23:12,
                     17:13,
                     11:14,
                     5:15,
                     2:16,
                     40:17,
                     34:18,
                     28:19,
                     22:20,
                     16:21,
                     10:22,
                     4:23,
                     41:25,
                     35:26,
                     31:27,
                     25:28,
                     18:29,
                     44:30,
                     8:31,
                     1:32,
                     42:33,
                     37:34,
                     43:35,
                     24:36,
                     19:37,
                     14:38,
                     7:39,
                     39:41,
                     36:42,
                     30:43,
                     26:44,
                     20:45,
                     13:46,
                     9:47,
                     3:48,
                     47:1,
                     45:24,
                     46:40                                                                         
                     }
            
            centeroids = centeroids.tolist()
            
            
            
            sortedList=[]
            
            def getKey(item):
                return item[1]    
            
            for elem in sorted(remap.items(),key=getKey) :
                sortedList.append(centeroids[elem[0]])
            
            
            centeroids = np.array(sortedList)
        else:
            centeroids = []
            with open(args.grid_file,"r",newline="") as fup:
                csr = csv.reader(fup,quoting=csv.QUOTE_NONNUMERIC)
                for row in csr:
                    centeroids.append(row)
                centeroids = np.array(centeroids)
            if not gridShape[0]*gridShape[1] == centeroids.shape[0]:
                gridShape = (centeroids.shape[0],1)
    
    nShifts = 0
    #filter for one origami
    
    localizationClusterList = []
    localizationClusterListFull = []
    
    nOrigamis = 15

    
    groupTables = [np.recarray((0,len(LOCS_DTYPE)),dtype = LOCS_DTYPE) for i in range(nOrigamis)]
    groupCount = [0 for i in range(nOrigamis)]

    #driftCorrectSize = 10
    if driftCorrectSize > 0:
        maxFrame = max(locs['frame'])
        totalDriftX = np.zeros(maxFrame+1)
        totalDriftY = np.zeros(maxFrame+1)
        totalDrift = np.rec.array((totalDriftX,totalDriftY),dtype=[("x", "f"), ("y", "f")])
        
        sliceSizes = np.array([7500,1000,500,driftCorrectSize])
        sliceSizes = sliceSizes[sliceSizes*2 < maxFrame]
        
        for sliceSize in sliceSizes:
            print('Applying drift correction segment = ',sliceSize)
            drift, locs = picasso.undrift(locs,info,sliceSize)
            totalDrift['x'] += drift['x']
            totalDrift['y'] += drift['y']
            
        if skip+samples < minDriftClusters:
            skip = minDriftClusters-samples
            print('Warning: need at least {} clusters for drift correction, adding additional skipped clusters'.format(minDriftClusters))
            
            
    #Apply filters
    if not filterFile == "":
        print("Applying filters")
        for filt in filters:
            colName = filt.tag
            low = float(filt.attrib['low'])
            high = float(filt.attrib['high'])
            if filt.attrib['type'] == 'percentile':
                locs.sort(kind="mergesort", order=colName)
                NLocs = len(locs)
                locs = locs[int(NLocs*low):int(NLocs*high)]
            else:
                locs = locs[(locs[colName] >= low) & (locs[colName] < high)]
                
        locs.sort(kind="mergesort", order="frame")
        baseFilePath += "_filt"
        picasso.save_locs(baseFilePath+".hdf5",locs,info)
    

    if samples+skip > 0:
        if picked:
            ids = np.array(range(len(locs)))
            for groupNumber in groupNumbers:
                
                sublocs = locs[locs['group']==groupNumber]
                subIds = ids[locs['group']==groupNumber]
                x = sublocs['x']
                y = sublocs['y']
                #photons=sublocs['photons']
                lpx = sublocs['lpx']
                #lpy = sublocs['lpy']
                
                localizations = np.array([x,y,lpx,subIds]).T
                if localizations.shape[0]==0:
                    continue;
                print(groupNumber,"read")
     
                if fitting:
                    lc = LocalizationCluster(centeroids,localizations,globalDeltaX,True)
                else:
                    subgrid = gridLocs[gridLocs['group']==groupNumber]
                    gx = subgrid['x']
                    gy = subgrid['y']
                    grid = np.array([gx,gy]).T
                    gdx = subgrid['sx'][0]
                    if not grid.shape[0] == gridShape[0]*gridShape[1]:
                        continue
                    lc = LocalizationCluster(grid,localizations,gdx,False)
                        
                    lc.hist2 = subgrid['photons'].astype(float).T
                    lc.hist = np.zeros(lc.hist2.shape)
                    lc.transformGrid([0,0,0])
                    for idx in lc.nn[1]:
                        lc.hist[idx]+=1
                        
                    lc.signal = np.copy(lc.hist).reshape(gridShape)
                    lc.rec = np.copy(lc.hist2).reshape(gridShape)
                    lc.squareNNDist([0,0,0])
                    
                localizationClusterList.append(lc)
        
        
        else:
            print("Finding",samples,"clusters...")
            
            if not reuse_file:
                rc = RandomClusters(centeroids,locs,clusterRadius,minCountThreshold,maxCountThreshold,2*clusterRadius,0.85,globalDeltaX)
            else:
                rc.clusterLocations = []
    
            localizationClusterListFull = rc.findClusters(samples+skip,50000)
            if len(localizationClusterListFull) <= samples:
                localizationClusterList = localizationClusterListFull[:]
            else:
                if samples > 0:
                    localizationClusterList = localizationClusterListFull[-samples:]
                else:
                    localizationClusterList = []
            print("Found",len(localizationClusterList))
            NN = len(localizationClusterList)
            
            if driftCorrectSize > 0:
                print('Applying fine drift correction')
                #locIds = []
                
                pickedLocs = []
                if len(localizationClusterListFull) < minDriftClusters:
                    print("Too few clusters to drift correct, skipping cluster drift correction")
                else:
                    pickedLocs = []
                    for i,lc in enumerate(localizationClusterListFull):
                        locIds = lc.localizations[:,3].astype(int).tolist()
                        pickedLocs.append(locs[locIds])
                    drift, locs = picasso.undrift_from_picked(locs,pickedLocs,info)
                    totalDrift['x'] += drift['x']
                    totalDrift['y'] += drift['y']
                
                print("Finding drift clusters")
                driftRc = RandomClusters(np.array([[0.0,0.0]]),locs,.05,50,600,.07,.65,globalDeltaX)
                driftClusterList = driftRc.findClusters(50000-len(localizationClusterListFull),200000)
                print("Found",len(driftClusterList),"drift clusters")
                if len(driftClusterList) < minDriftClusters:
                    print("Too few clusters to drift correct, skipping fine drift correction")
                else:
                    for lc in driftClusterList:
                        locIds = lc.localizations[:,3].astype(int).tolist()
                        pickedLocs.append(locs[locIds])
                    drift, locs = picasso.undrift_from_picked(locs,pickedLocs,info)
                    totalDrift['x'] += drift['x']
                    totalDrift['y'] += drift['y']
        
                baseFilePath += "_adrift"
                picasso.save_locs(baseFilePath+".hdf5",locs,info)
                
                for i,lc in enumerate(localizationClusterList):
                    locIds = lc.localizations[:,3].astype(int).tolist()
                    subLocs = locs[locIds]
                    
                    x = subLocs['x']
                    y = subLocs['y']
                    lpx = subLocs['lpx']
                    
                    localizationClusterList[i] = LocalizationCluster(lc.grid,np.array([x,y,lpx,locIds]).T,globalDeltaX,True)
                        
                del localizationClusterListFull
                del driftClusterList
                gc.collect()

    if len(localizationClusterList) == 0:
        print("No clusters to analyze, exiting")
        sys.exit()
        
    #Create empty files
    with open(baseFilePath+'_match_info_'+str(NN)+'.csv','w') as fup:
        fup.write("\"Binary String\",\"ID\",\"Match Score\",\"Exact String\",\"False Negatives\",\"False Positives\",\"Row Shifts\",\"Column Shifts\",\"Fit Quality\",\"X Position\",\"Y Position\",\"Group Number\"\n")
    with open(baseFilePath+'_bin_info_'+str(NN)+'.csv','w') as fup:
        fup.write("\"Binary String\",\"Row Shifts\",\"Column Shifts\",\"Fit Quality\",\"X Position\",\"Y Position\"\n")
    open(baseFilePath+'_bin_'+str(NN),'w').close()
    if display2:
        open(baseFilePath+'_fig_data.csv',"w").close()
    
    processes = []
    parentConnections = []
    
    print("starting fits")
    

    useMultiProcessing = True
    
    print(len(localizationClusterList))

    
    if fitting:
        if useMultiProcessing:
    
            pool = Pool(cpu_count()-1)
    
            results = []
            for index in range(len(localizationClusterList)):
                results.append(pool.apply_async(clusterFit,(localizationClusterList[index],index,gridShape,distThreshold)))
            
            for res in results:
                msg = res.get()
                print(msg[0],'fitting done obj:',msg[4],"FVal:",msg[5],"GlobalDeltaX:",msg[6])
                lc = localizationClusterList[msg[0]]
                lc.globalDeltaX = msg[6]
                lc.hist2 = msg[7]
                lc.squareNNDist(msg[1:4])
                lc.hist = np.zeros(lc.hist2.shape)
                for idx in lc.nn[1]:
                    lc.hist[idx]+=1
                        
                lc.signal = np.copy(lc.hist[:-1]).reshape(gridShape)
                lc.rec = np.copy(lc.hist2[:-1]).reshape(gridShape)
            
        else:
            for index in range(len(localizationClusterList)):
                msg = clusterFit(localizationClusterList[index],index,gridShape,distThreshold)
                print(msg[0],'fitting done obj:',msg[4],"FVal:",msg[5],"GlobalDeltaX:",msg[6])
                #localizationClusterList[msg[0]].globalDeltaX = msg[6]
                #localizationClusterList[msg[0]].squareNNDist(msg[1:4])
            
    
    rejectedCount = 0
    
    
    
    fnErrorMaps = [np.zeros(gridShape) for i in range(nOrigamis+1)]
    fpErrorMaps = [np.zeros(gridShape) for i in range(nOrigamis+1)]
    errNormMap = np.zeros(gridShape)
    errorCounts = [0 for i in range(nOrigamis)]
    
    perfectHists = []
    perfectImages = []
    #with open('perfect_data.txt','r') as fup:
    #    perfectStrings = fup.readlines()
    perfectStrings = ['001101101111011100111010001101111101111000101010',
                      '001001101010100110110100000111011111110000000001',
                      '010111111101100100011100101111001110010000010111',
                      '001000011000010110011110100100011110001000011010',
                      '000110101111101100111110000100001101100010010010',
                      '010111111101010110111000001010011100001010110100',
                      '000100001110101100011010111000111000011010100101',
                      '010101001010000111100000110110001101100010000100',
                      '001000001011111101000100010100111110010001100101',
                      '001000001101101110101100000100111101000001111101',
                      '011011101101010100000110110111101111110001101000',
                      '000100011110011110011010100000001010111001010100',
                      '010011101100110101000100000000011100010011011000',
                      '010100101001111111001100010100111100111011011011',
                      '000010101101010101101100100010111001010011101111']

    for row in perfectStrings:
        perfectHists.append([])
        for char in row:
            if char == '1':
                perfectHists[-1].append(1)
            elif char =='0':
                perfectHists[-1].append(-1)
        perfectImages.append(np.array(perfectHists[-1]).reshape(gridShape))
        
# =============================================================================
#     if redrift:
#         print("Redrift correcting...")
#         combinedGrid = []
#         clusterLocIds = []
#         for lc in localizationClusterList:
#             grid = np.copy(lc.gridTran)
#             grid[:,0]+=lc.xLAve
#             grid[:,1]+=lc.yLAve
#             combinedGrid.extend(grid.tolist())
#             clusterLocIds.extend(lc.localizations[:,3].astype(int).tolist())
#             
#         if driftCorrectSize == 0:
#             maxFrame = max(locs['frame'])
#         clusterLocs = locs[clusterLocIds]
#         clusterLocs.sort(kind="mergesort", order="frame")
#         #frameLocs = []
#         drift = np.rec.array(np.zeros(maxFrame+1), dtype=[("x", "f"), ("y", "f")])
#         for frameId in range(maxFrame):
#             frameLocs = clusterLocs[clusterLocs['frame']==frameId]
#             if frameLocs.shape[0] == 0:
#                 continue
#             x = frameLocs['x']
#             y = frameLocs['y']
#             lpx = frameLocs['lpx']
#             
#             localizations = np.array([x,y,lpx]).T
#             frameCluster = LocalizationCluster(combinedGrid,localizations,globalDeltaX,False)
#             bounds = [(-.1,.1),(-.1,.1),(0,0)]
#             out = minimize(frameCluster.squareNNDist,[0,0,0],bounds=bounds,method='L-BFGS-B')
#             drift[frameId]['x'] = out.x[0]
#             drift[frameId]['y'] = out.x[1]
#             
#         locs.x -= drift.x[locs.frame]
#         locs.y -= drift.y[locs.frame]
#             #frameCluster
#         
#         for i,lc in enumerate(localizationClusterList):
#             locIds = lc.localizations[:,3].astype(int).tolist()
#             subLocs = locs[locIds]
#             
#             x = subLocs['x']
#             y = subLocs['y']
#             lpx = subLocs['lpx']
#             
#             localizations = np.array([x,y,lpx,locIds]).T
#             localizations[:,0] -= lc.xLAve
#             localizations[:,1] -= lc.yLAve
#             
#             localizationClusterList[i] = LocalizationCluster(lc.grid,localizations,globalDeltaX,True)
#             
#         picasso.save_locs(baseFilePath+"_Redrift.hdf5",locs,info)
# =============================================================================
        
    aveLPX2 = np.average(locs['lpx']**2)
    aveObj = np.average(np.array([lc.objectiveValue for lc in localizationClusterList]))
    aveGDX = np.average(np.array([lc.globalDeltaX for lc in localizationClusterList]))
    #globalDeltaX = np.sqrt(aveLPX2*(aveObj-1))
    
    print("Average lpx**2:",aveLPX2,"average obj:",aveObj,"Average Global Delta X",aveGDX)
    
    for lcInd in range(len(localizationClusterList)):
        lc = localizationClusterList[lcInd]
        if fitting:
            #lc.pixelate(gridShape,distThreshold)
            print("bg:",lc.hist2[-1])
        fitQuality = lc.squareNNDistUnweighted([lc.dx,lc.dy,lc.dt],.0937)
        if verbose:                
            print(lcInd,fitQuality)
        if fitQuality > objectiveThreshold:
            rejectedCount+=1
            continue
        
        gridPoints = np.copy(lc.gridTran).reshape(gridShape[0],gridShape[1],2)
        
        rx=0
        ry=0
        cx=0
        cy=0
        
        
        for rid in range(1,gridShape[0]):
            for cid in range(0,gridShape[1]):
                rx += gridPoints[rid,cid,0]-gridPoints[rid-1,cid,0]
                ry += gridPoints[rid,cid,1]-gridPoints[rid-1,cid,1]
        
        for rid in range(0,gridShape[0]):
            for cid in range(1,gridShape[1]):
                cx += gridPoints[rid,cid,0]-gridPoints[rid,cid-1,0]
                cy += gridPoints[rid,cid,1]-gridPoints[rid,cid-1,1]
        
        if gridShape[0] > 1:
            rx/=(gridShape[0]-1)*gridShape[1]
            ry/=(gridShape[0]-1)*gridShape[1]
        if gridShape[1] > 1:
            cx/=gridShape[0]*(gridShape[1]-1)
            cy/=gridShape[0]*(gridShape[1]-1)
        
        minDeltaR = 0
        minDeltaC = 0
        
        dx = lc.dx
        dy = lc.dy
        dt = lc.dt
        
        if fitting:
            minObj = lc.gridLikelihoodScore([dx,dy,dt])
            for deltaR in range(-2,3):
                for deltaC in range(-2,3):
                    obj3 = lc.gridLikelihoodScore([dx+deltaR*rx+deltaC*cx,dy+deltaR*ry+deltaC*cy,dt])
                    if obj3 < minObj:
                        minObj = obj3
                        minDeltaR = deltaR
                        minDeltaC = deltaC
            
            obj3 = lc.squareNNDist([dx+minDeltaR*rx+minDeltaC*cx,dy+minDeltaR*ry+minDeltaC*cy,dt])
            gridPoints = np.copy(lc.gridTran).reshape(gridShape[0],gridShape[1],2)
        

# =============================================================================
#         superGridPoints = np.zeros((gridShape[0]+2,gridShape[1]+2,2))
#         
#         for rid in range(0,gridShape[0]):
#             for cid in range(0,gridShape[1]):
#                 superGridPoints[rid+1,cid+1,:] = gridPoints[rid,cid,:]
#         
#         for rid in range(gridShape[0]+2):
#             superGridPoints[rid,0,0] = gridPoints[0,0,0] - rx - cx + rid * rx
#             superGridPoints[rid,0,1] = gridPoints[0,0,1] - ry - cy + rid * ry
#             superGridPoints[rid,-1,0] = gridPoints[-1,-1,0] + rx + cx - rid * rx
#             superGridPoints[rid,-1,1] = gridPoints[-1,-1,1] + ry + cy - rid * ry
#         
#         for cid in range(1,gridShape[1]+1):
#             superGridPoints[0,cid,0] = gridPoints[0,0,0] - rx - cx + cid * cx
#             superGridPoints[0,cid,1] = gridPoints[0,0,1] - ry - cy + cid * cy
#             superGridPoints[-1,cid,0] = gridPoints[-1,-1,0] + rx + cx - cid * cx
#             superGridPoints[-1,cid,1] = gridPoints[-1,-1,1] + ry + cy - cid * cy
#         
#         superGrid = np.copy(superGridPoints).reshape((superGridPoints.shape[0]*superGridPoints.shape[1],superGridPoints.shape[2]))
#         
#         lc2 = LocalizationCluster(superGrid,lc.localizations,lc.globalDeltaX,False)
#  
#         obj2 = lc2.squareNNDist([0,0,0])
#         if verbose:
#             print(lcInd,lc.objectiveValue,obj2)
#         
# =============================================================================
        
        #lc.pixelate(gridShape)
        lc2 = lc
        
        
        if display2:
            fig = plt.figure()
            plt.scatter(-lc2.localizations[:,0],lc2.localizations[:,1],s=3,color='blue')
            plt.scatter(-lc2.gridTran[:,0],lc2.gridTran[:,1],color='orange')
            fig.show()
            
            fig, axt = plt.subplots(3,2)
            axt[0,0].bar([i for i in range(len(lc2.hist))],lc2.hist.tolist())

            axt[1,0].imshow(lc2.signal)
            axt[1,1].imshow(lc2.rec)
           
            axt[0,1].bar([i for i in range(len(lc2.hist2))],lc2.hist2.tolist())
            
            with open(baseFilePath+'_fig_data.csv',"a",newline='') as ffup:
                csr = csv.writer(ffup)
                csr.writerow(["Cluster {} data:".format(lcInd)])
                csr.writerow(["Raw pixelation"])
                csr.writerows(lc2.signal.tolist())
                csr.writerow(["Optimal pixelation"])
                csr.writerows(lc2.rec.tolist())
                csr.writerow(["Grid Points"])
                csr.writerows(lc2.gridTran.tolist())
                csr.writerow(["Localization data"])
                csr.writerows(lc2.localizations.tolist())
                
        
        sortHist = np.sort(lc2.hist2)
        aveBrightBit = np.average(sortHist[-10:])
        aveOnBit = np.average(lc2.hist2[lc2.hist2>1])
        threshold = scaled_threshold*aveBrightBit
        print(aveBrightBit,scaled_threshold,threshold)
        
        
        rawImage = np.copy(lc2.rec)
        binaryImage = np.copy(lc2.rec)
        for rid in range(binaryImage.shape[0]):
            for cid in range(binaryImage.shape[1]):
                if rawImage[rid,cid] > threshold:
                    binaryImage[rid,cid] = 1
                else:
                    binaryImage[rid,cid] = 0
        
        if display2:        
            axt[2,1].imshow(binaryImage)
            
# =============================================================================
#             superRes = np.zeros((60,60))
#             
#             for rid in range(lc2.localizations.shape[0]):
#                 x = int(-lc2.localizations[rid,1]*pixelsize/2+superRes.shape[0]/2)
#                 y = int(-lc2.localizations[rid,0]*pixelsize/2+superRes.shape[1]/2)
#                 if x > 0 and x<superRes.shape[0] and y > 0 and y < superRes.shape[1]:
#                     superRes[x,y] += 1
# =============================================================================
            if picked:
                tmp, superRes = picasso.render_gaussian(locs[locs["group"]==lcInd], 160, lc.yLAve-.6,lc.xLAve-.6, lc.yLAve+.6, lc.xLAve+.6, .015)
            else:
                tmp, superRes = picasso.render_gaussian(locs[lc.localizations[:,3].astype(int)], 160, lc.yLAve-.6,lc.xLAve-.6, lc.yLAve+.6, lc.xLAve+.6, .015)
            axt[2,0].imshow(superRes)
            
            fig.show()
            
        if sum(sum(binaryImage)) == 0:
            rejectedCount += 1
            continue
        #Testing code, remove
        #binaryImage = np.array([[0,0,1,0,1,0,1,0],[0,1,0,1,0,0,0,0],[0,1,1,1,0,0,0,0],[0,1,1,0,1,1,0,0],[0,1,1,0,1,1,0,0],[0,1,0,0,0,0,1,0]])
        #binaryImage = np.array([[1,1,1,0,1,0,1,1],[0,0,0,1,1,0,1,0],[1,1,1,0,0,0,1,1],[1,0,0,0,0,1,1,0],[1,0,1,0,0,1,0,1],[0,0,0,0,0,0,0,0]])
        #rawImage = np.copy(binaryImage)*1000
        #end testing code
        
        rShifts = [0]
        cShifts = [0]
        
        rSums = np.sum(binaryImage,axis=1)
        cSums = np.sum(binaryImage,axis=0)
        
        if rSums[0] == 0:
            rShifts.append(-1)
        if rSums[-1] == 0:
            rShifts.append(1)
        if cSums[0] == 0:
            cShifts.append(-1)
        if cSums[-1] == 0:
            cShifts.append(1) 
            
        
        maxSScore = 0
        maxId = 0
        
        fPos = 0
        fNeg = 0
        maxRShift = 0
        maxCShift = 0
        
        rotation = 0
        
        for rShift in rShifts:
            for cShift in cShifts:
                binaryImageShifted = np.roll(binaryImage,(rShift,cShift),(0,1))
                rawImageShifted = np.roll(rawImage,(rShift,cShift),(0,1))
                
                Id = 0
                for pImage in perfectImages:
                    sScore = np.sum(pImage*(rawImageShifted-aveOnBit))
                    if sScore > maxSScore:
                        maxId = Id
                        maxSScore = sScore
                        fPos = np.sum(binaryImageShifted * pImage ==  -1)
                        fNeg = np.sum((binaryImageShifted-1) * pImage ==  -1)
                        maxRShift = rShift
                        maxCShift = cShift
                        rotation = np.pi
                    sScore = np.sum(np.flip(pImage,(0,1))*(rawImageShifted-aveOnBit))
                    if sScore > maxSScore:
                        maxId = Id
                        maxSScore = sScore
                        
                        fPos = np.sum(binaryImageShifted * np.flip(pImage,(0,1)) == -1)
                        fNeg = np.sum((binaryImageShifted-1) * np.flip(pImage,(0,1)) == -1)
                        maxRShift = rShift
                        maxCShift = cShift
                        
                        rotation = 0
                    Id+=1
                    
                
                binaryData = binaryImageShifted.reshape((binaryImage.shape[0]*binaryImage.shape[1]))
                binaryString=""
                
                if not (rShift == 0 and cShift == 0):
                    nShifts += 1
                    if display2:
                        plt.figure()
                        plt.imshow(binaryImageShifted)
                
                for bit in binaryData:
                    if bit==0:
                        binaryString+='0'
                    else:
                        binaryString+='1'
                
                with open(baseFilePath+'_bin_'+str(NN),'a') as fup:
                    fup.write(binaryString+'\n')
                with open(baseFilePath+'_bin_info_'+str(NN)+'.csv','a') as fup:
                    fup.write("\""+binaryString+"\","+str(rShift)+","+str(cShift)+","+str(fitQuality)+","+str(lc.xLAve)+","+str(lc.yLAve)+"\n")
        
        if verbose:                
            print(maxId,maxSScore)
                    
        binaryImageShifted = np.roll(binaryImage,(maxRShift,maxCShift),(0,1))
        
        
        if fPos + fNeg < 16:
            pImage = perfectImages[maxId]
            binaryImageShiftedFlipped = np.copy(binaryImageShifted)
            if rotation == 0:
                binaryImageShiftedFlipped = np.flip(binaryImageShiftedFlipped,(0,1))
                
            fpErrorMaps[-1] += (binaryImageShiftedFlipped * pImage == -1)
            fpErrorMaps[maxId] += (binaryImageShiftedFlipped * pImage == -1)
            fnErrorMaps[-1] += ((binaryImageShiftedFlipped-1) * pImage == -1)
            fnErrorMaps[maxId] += ((binaryImageShiftedFlipped-1) * pImage == -1)
            errorCounts[maxId]+=1
            errNormMap += (pImage>0)
        
        binaryData = binaryImageShifted.reshape((binaryImage.shape[0]*binaryImage.shape[1]))
        binaryString=""
        for bit in binaryData:
            if bit==0:
                binaryString+='0'
            else:
                binaryString+='1'
                
        print("\"b"+binaryString+"\","+str(maxId)+","+str(maxSScore)+",\"b"+perfectStrings[maxId][0:-1]+"\","+str(fNeg)+","+str(fPos)+","+str(maxRShift)+","+str(maxCShift)+","+str(fitQuality)+","+str(lc.xLAve)+","+str(lc.yLAve)+","+str(groupCount[maxId])+"\n")
        with open(baseFilePath+'_match_info_'+str(NN)+'.csv','a') as fup:
            fup.write("\"b"+binaryString+"\","+str(maxId)+","+str(maxSScore)+",\"b"+perfectStrings[maxId][0:-1]+"\","+str(fNeg)+","+str(fPos)+","+str(maxRShift)+","+str(maxCShift)+","+str(fitQuality)+","+str(lc.xLAve)+","+str(lc.yLAve)+","+str(groupCount[maxId])+"\n")
        
            
            
        cw = np.sqrt(cx**2+cy**2)
        rw = np.sqrt(rx**2+ry**2)
        if outputClusters and fPos + fNeg < 11:
            locTable = locs[lc.localizations[:,3].astype(int)]
            locCount = locTable.shape[0]
            groupNumber = groupCount[maxId]
            groupTable = np.rec.array((locTable['frame'], locTable['x'], locTable['y'], locTable['photons'], locTable['sx'], locTable['sy'],locTable['bg'],locTable['lpx'],locTable['lpy'],groupNumber*np.ones(locTable['frame'].shape)),dtype=LOCS_DTYPE,)

            
            xArr = np.array(groupTable['x'])
            yArr = np.array(groupTable['y'])

            
            deltaX = -lc.dx-lc.xLAve
            deltaY = -lc.dy-lc.yLAve
            
            for ind in range(xArr.shape[0]):

                xTemp = xArr[ind] + deltaX
                yTemp = yArr[ind] + deltaY
                xArr[ind] = (np.cos(lc.dt+rotation)*xTemp - np.sin(lc.dt+rotation)*yTemp)+256.0+maxCShift*cw
                yArr[ind] = (np.cos(lc.dt+rotation)*yTemp + np.sin(lc.dt+rotation)*xTemp)+256.0-maxRShift*rw

            
            groupTable['x'] = xArr
            groupTable['y'] = yArr
            
            
            
            groupCount[maxId]+=1

            groupTables[maxId].resize((groupTables[maxId].shape[0]+locCount),refcheck=False)
            groupTables[maxId][-locCount:] = groupTable
        
        
    if fitting:
        
        nLocs = 0
        nGridPoints = 0
        for lc in localizationClusterList:
            nLocs+=lc.localizations.shape[0]
            nGridPoints+=lc.gridTran.shape[0]
            
        
        zc = np.zeros((nLocs,))
        groupLocs = np.rec.array(
            (zc,zc,zc,zc,zc,zc,zc,zc,zc,zc),
            dtype=LOCS_DTYPE,
        )
        
        zc = np.zeros((nGridPoints,))
        gridLocs = np.rec.array(
            (zc,zc,zc,zc,zc,zc,zc,zc,zc,zc),
            dtype=LOCS_DTYPE,
        )
        
        locPtr = 0
        gridPtr = 0
        
        for lcInd,lc in enumerate(localizationClusterList):
            locTable = locs[lc.localizations[:,3].astype(int)]
            locCount = locTable.shape[0]
            groupTable = np.rec.array((locTable['frame'], locTable['x'], locTable['y'], locTable['photons'], locTable['sx'], locTable['sy'],locTable['bg'],locTable['lpx'],locTable['lpy'],lcInd*np.ones(locTable['frame'].shape)),dtype=LOCS_DTYPE,)
            groupLocs[locPtr:(locPtr+locCount)] = groupTable
            locPtr+=locCount
            
            gridCount = lc.gridTran.shape[0]
            zc = np.zeros((gridCount,))
            oc = np.ones((gridCount,))
            gridTable = np.rec.array((zc,lc.gridTran[:,0]+lc.xLAve,lc.gridTran[:,1]+lc.yLAve,lc.hist2[:-1],oc*lc.globalDeltaX,oc*lc.globalDeltaX,oc,oc*.003,oc*.003,lcInd*oc))
            gridLocs[gridPtr:(gridPtr+gridCount)] = gridTable
            gridPtr+=gridCount
        
        if "fitpicks" in baseFilePath:
            picasso.save_locs(baseFilePath+".hdf5",groupLocs,info)
            picasso.save_locs(baseFilePath+"_grid.hdf5",gridLocs,info)
        else:
            picasso.save_locs(baseFilePath+"_fitpicks_"+str(NN)+".hdf5",groupLocs,info)
            picasso.save_locs(baseFilePath+"_fitpicks_"+str(NN)+"_grid.hdf5",gridLocs,info)
            
    
    if outputClusters:
        for tableId in range(len(groupTables)):
            if groupTables[tableId].shape[0] > 0:
                table = groupTables[tableId]

                prefix = baseFilePath+'_Matrix_{0:02d}'.format(tableId)

                table.sort(kind="mergesort", order="frame")

                picasso.save_locs(prefix+'.hdf5', table, info)
                

                                                                  
    
    with open(baseFilePath+'_errmap_'+str(NN)+'.csv','w',newline='') as fup:
        csr = csv.writer(fup,quoting=csv.QUOTE_NONNUMERIC)
        csr.writerow(['False Negatives'])
        csr.writerow(['All Matrices'])
        temp = np.copy(errNormMap)
        temp[temp==0] = 1
        tempMap = fnErrorMaps[-1]/temp
        tempMap[errNormMap==0] = 0
        csr.writerows(tempMap)
        for idx in range(nOrigamis):
            csr.writerow(['Matrix {0:02d}'.format(idx)])
            
            if not errorCounts[idx] == 0:
                csr.writerows(fnErrorMaps[idx]/errorCounts[idx])
            else:
                csr.writerows(np.zeros(fnErrorMaps[idx].shape))
        csr.writerow(['False Positives'])
        csr.writerow(['All Matrices'])
        temp = -errNormMap+sum(errorCounts)
        temp[temp==0] = 1
        tempMap = fpErrorMaps[-1]/temp
        tempMap[-errNormMap+sum(errorCounts)==0] = 0
        csr.writerows(tempMap)
        for idx in range(nOrigamis):
            csr.writerow(['Matrix {0:02d}'.format(idx)])
            
            if not errorCounts[idx] == 0:
                csr.writerows(fpErrorMaps[idx]/errorCounts[idx])
            else:
                csr.writerows(np.zeros(fnErrorMaps[idx].shape))
        
    
    if picked:                 
        print(len(groupNumbers),rejectedCount,nShifts,aveObj,aveGDX)
    else:
        print(len(localizationClusterList),rejectedCount,nShifts,aveObj,aveGDX)
        
    if display2:        
        input()
