import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
#default feature function
def feature(fiducial_pt_list,index,info):
    def pairwise_dist(mat):
        mat.insert(2,2, [0]*mat.shape[0],allow_duplicates = True)
        vec1 = pdist(mat.iloc[:,[0,2]])    
        vec2 = pdist(mat.iloc[:,[1,2]])
        return np.r_[vec1,vec2]
    
    pairwise_dist_result = pd.DataFrame(map(pairwise_dist,[fiducial_pt_list[i] for i in index]))
    pairwise_dist_result.insert(len(pairwise_dist_result.columns),'emotion_idx', value = list(info['emotion_idx'][index]))
    
    pairwise_dist_result.columns = ['feature%s' % i for i in list(range(1,len(pairwise_dist_result.columns)))] + ['emotion_idx']
    return pairwise_dist_result



def feature_slope(mat_list, nfidu = 78):
    def pairwise_dist(vec):
        a = pairwise_distances(vec.reshape(nfidu,1))
        return(a[np.triu_indices(nfidu, k = 1)])
    
    def pairwise_dist_result(mat):
        a = np.apply_along_axis(pairwise_dist, 0, mat)
        return(np.rad2deg(np.arctan2(a[:,1], a[:,0]))) 
     
    feature_mat = [pairwise_dist_result(mat) for mat in mat_list]   
    return(np.vstack(feature_mat))

def feature_dist_slope(fiducial_pt_list,info):
    def pairwise_dist_slope(mat):
        mat = pd.DataFrame(mat)
        #distance between points
        dist_result =  pdist(mat)
        
        #slope between points
        mat.insert(2,2, [0]*mat.shape[0],allow_duplicates = True)
        vec1 = pdist(mat.iloc[:,[0,2]])    
        vec2 = pdist(mat.iloc[:,[1,2]])
        a = np.column_stack((vec1,vec2))
        slope_result = np.rad2deg(np.arctan2(a[:,1], a[:,0]))

        return np.r_[dist_result,slope_result]
    
    
    result = pd.DataFrame(map(pairwise_dist_slope,[fiducial_pt_list[i] for i in range(0,len(fiducial_pt_list))]))
    result.insert(len(result.columns),'emotion_idx', value = list(info['emotion_idx'][list(range(0,len(fiducial_pt_list)))]))
    
    result.columns = ['feature%s' % i for i in list(range(1,len(result.columns)))] + ['emotion_idx']
    return(result)
    
def feature_distance(mat_list, nfidu = 78):
    def pairwise_dist(vec):

