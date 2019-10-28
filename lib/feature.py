import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
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


#
#index = [1,3,5]    
#
#info[info.index[index],'emotion_idx']
#a = fiducial_pt_list[0].copy(deep = True)
#b[0].insert(2,2, [0]*b[0].shape[0],allow_duplicates = True)
#
#a[[0,3]]
#
#info.iloc['emotion_idx']
#
#b = pdist(a.iloc[:,[0,2]])
#
#df = pd.DataFrame({'a': ['1', '2'], 
#                       'b': ['45.0', '73.0'],
#                       'c': [10.0, 3.0]})
# = df.apply(pd.to_numeric)
#
#c = b.apply(pd.DataFrame.astype,dtype = 'int64')