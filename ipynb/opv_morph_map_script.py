import os
import sys
#from decimal import Decimal

import numpy as np
#import pandas as pd
#import scipy.stats as stats
import matplotlib.pyplot as plt


module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from m2py.utils import config
#from m2py.utils import utils
from m2py.utils import pre_processing as pre
from m2py.utils import post_processing as post
from m2py.utils import seg_label_utils as slu
from m2py.segmentation import segmentation_gmm as seg_gmm
from m2py.segmentation import segmentation_watershed as seg_water

opv_file_path = '/Volumes/Tatum_SSD-1/Grad_School/m2py/OPV_AFM/combined_npy_new/'
opv_morph_map_file_path = '/Volumes/Tatum_SSD-1/Grad_School/m2py/Morpholgy_labels/opv_morph_maps/'
files = os.listdir(opv_file_path)


#test_file_path = '/Users/wesleytatum//Desktop/2019-10-11/npy/'
#test_morph_map_file_path = '/Users/wesleytatum/Desktop/2019-10-11/test_morph_maps/'
#files = os.listdir(test_file_path)

files.sort()
print (len(files))
print (files)

ims = []

for i,fl in enumerate(files):
    ims.append(np.load(opv_file_path+fl))

# The files will all be read in and processed, saving tresults to opv_morph_map_file_path
def m2py_pipeline(dataframe, heightless, outlier_threshold, n_components, padding, embedding_dim, thresh, nonlinear, normalize, zscale, data_type, data_subtype, input_cmap):
    """
    Wrapper function for m2py tools. Allows to include or exclude m2py tools in the order shown in the code.
    
    Args:
    dataframe - np.array(). 3D array of SPM data
    heightless - bool. if 'True', height channel is removed, according to channel labels in config.py
    outlier_threshold
    n_components - int. number specifying the number of Gaussian phase distributions to be identified in dataframe
    padding - int. number specifying the degree of closeness of neighbors to be included in GMM labelling
    embedding_dims - int. number specifying the number of principle components to use in PCA before GMM labelling
    thresh
    nonlinear - bool. if 'True', nonlinear properties are generated prior to analysis. Includes x^2, x^3, abs(x),
                and 1/x
    normalize
    zscale
    data_type
    data_subtype
    input_cmap
                
    Returns:
    outliers - np.array(). 2D array of outliers
    seg1_labels - np.array(). 2D array of GMM labels
    seg2_labels - np.array(). 2D array of clustering labels
    """
  
    fill_zeros_flag = True
    remove_outliers_flag = True
    input_cmap = input_cmap
    
    # Apply frequency removal
    data = pre.apply_frequency_removal(dataframe, data_type)
    
    data_properties = config.data_info[data_type]['properties']

    # Extract outliers
#     outliers = pre.extract_outliers(data, data_type, prop = 'Z-scale', threshold = outlier_threshold, chip_size = 256, stride = 256)
    
    h, w, c = data.shape
    outliers = np.zeros((h, w))
    for prop in data_properties:
        temp_outliers = pre.extract_outliers(data, data_type, prop, 2.5, h, w)
        pre.show_outliers(data, data_type, prop, temp_outliers)
    
        outliers = np.logical_or(outliers, temp_outliers)
    
    no_outliers_data = pre.smooth_outliers_from_data(data, outliers)
    
    plt.imshow(outliers)
    plt.show()

    # Show a-priori distributions
    pre.show_property_distributions(data, data_type, outliers)

    c = data.shape[2]
    num_pca_components = min(embedding_dim, c)
    
    # Run GMM segmentation
    seg1 = seg_gmm.SegmenterGMM(n_components = n_components, embedding_dim = num_pca_components, padding = padding,
                                nonlinear = nonlinear, normalize=normalize, zscale=zscale)

######################################## Heightless == True ##########################################

    if heightless == True:
        # Remove height property
        height_id = data_properties.index("Height")
        
        no_height_data = np.delete(no_outliers_data, height_id, axis=2)

        seg1_labels = seg1.fit_transform(no_height_data, outliers)
        
        if fill_zeros_flag:
            seg1_labels = slu.fill_out_zeros(seg1_labels, outliers)

        elif remove_outliers_flag:
            seg1_labels = np.ma.masked_where(outliers == 1, seg1_labels)

        post.show_classification(seg1_labels, no_height_data, data_type)
        post.show_classification_correlation(seg1_labels, no_height_data, data_type)
        post.show_distributions_together(seg1_labels, no_height_data, data_type, input_cmap = 'jet')
        post.show_grain_area_distribution(seg1_labels, data_type, data_subtype)

        # Overlay distributions on pixels
        probs = seg1.get_probabilities(no_height_data)
        post.show_overlaid_distribution(probs, no_height_data, data_type)
        
        if embedding_dim != None:
            h, w, c = no_height_data.shape

            pca_components = seg1.get_pca_components(no_height_data)
            pca_components = pca_components.reshape(h, w, num_pca_components)
            post.show_classification_correlation(seg1_labels, pca_components, data_type, title_flag=False)

            post.show_classification_distributions(seg1_labels, pca_components, data_type, title_flag=False)

        else:
            pass
            

## Persistence Watershed Segmentation clustering
    
        if thresh != None:
        
            seg2 = seg_water.SegmenterWatershed()
            height_id = data_properties.index("Height")
            height_data = data[:, :, height_id]
            thresh = thresh

            seg2_labels = seg2.fit_transform(height_data, outliers, thresh)
            
                                                                        ## This is incorporates GMM labels into PWS
            
            seg2_labels[seg1_labels == 1] = np.max(seg1_labels) + 1 # map background label of gmm to next pws label
            for l in np.unique(seg2_labels): # clean up small pws labels
                if np.sum(seg2_labels == l) < 100:
                    seg2_labels = slu.fill_out_zeros(seg2_labels, seg2_labels == l)

            seg2_labels = slu.get_significant_labels(seg2_labels, bg_contrast_flag=True)
            post.show_classification(seg2_labels, no_height_data, data_type)

## Conected-components clustering
        else:
            post_labels = seg1.get_grains(seg1_labels)
            seg2_labels = slu.get_significant_labels(post_labels, bg_contrast_flag=True)

        post.show_classification(seg2_labels, no_height_data, data_type, input_cmap = input_cmap)
        post.show_grain_area_distribution(seg2_labels, data_type, data_subtype)

######################################## Heightless == False ##########################################
        
    elif heightless == False:
        
        seg1_labels = seg1.fit_transform(no_outliers_data, outliers)
        
        if fill_zeros_flag:
            seg1_labels = slu.fill_out_zeros(seg1_labels, outliers)

        elif remove_outliers_flag:
            seg1_labels = np.ma.masked_where(outliers == 1, seg1_labels)

        post.show_classification(seg1_labels, no_outliers_data, data_type)
        post.show_distributions_together(seg1_labels, no_outliers_data, data_type, input_cmap = input_cmap)

        post.show_classification(seg1_labels, no_outliers_data, data_type)
        post.show_classification_correlation(seg1_labels, no_outliers_data, data_type)
        post.show_distributions_together(seg1_labels, no_outliers_data, data_type, input_cmap = input_cmap)

        # Overlay distributions on pixels
        probs = seg1.get_probabilities(no_outliers_data)
        post.show_overlaid_distribution(probs, no_outliers_data, data_type)
        post.show_grain_area_distribution(seg1_labels, data_type, data_subtype)
        
        
        if embedding_dim != None:
            h, w, c = no_outliers_data.shape

            pca_components = seg1.get_pca_components(no_outliers_data)
            pca_components = pca_components.reshape(h, w, num_pca_components)
            post.show_classification_correlation(seg1_labels, pca_components, data_type, title_flag=False)

            post.show_classification_distributions(seg1_labels, pca_components, data_type, title_flag=False)

            
        else:
            pass

        
## Persistence Watershed Segmentation clustering
    
        if thresh != None:
        
            seg2 = seg_water.SegmenterWatershed()           
            height_id = data_properties.index("Height")
            height_data = data[:, :, height_id]
            thresh = thresh
            
            watershed_channel = no_outliers_data[:,:,0]
            
            plt.imshow(watershed_channel, cmap = 'gist_ncar')
            plt.show()

            seg2_labels = seg2.fit_transform(watershed_channel, outliers, thresh)
            seg2_labels = np.ma.masked_where(seg2_labels == 0, seg2_labels)

            ## This is incorporates GMM labels into PWS
            
            seg2_labels[seg1_labels == 2] = np.max(seg2_labels) + 1 # map background label of gmm to next pws label
            for l in np.unique(seg2_labels): # clean up small pws labels
                if np.sum(seg2_labels == l) < 500:
                    seg2_labels = slu.fill_out_zeros(seg2_labels, seg2_labels == l)

            seg2_labels = slu.get_significant_labels(seg2_labels, bg_contrast_flag=True)
            post.show_classification(seg2_labels, no_outliers_data, data_type)

## Conected-components clustering
        else:
            post_labels = seg1.get_grains(seg1_labels)
            seg2_labels = slu.get_significant_labels(post_labels, bg_contrast_flag=True)
        
        post.show_classification(seg2_labels, no_outliers_data, data_type, input_cmap = input_cmap)
        post.show_grain_area_distribution(seg2_labels, data_type, data_subtype)


    else:
        print ('Error: Heightless flag incorrect')
    
    return outliers, seg1_labels, seg2_labels

print ("ims has",len(ims), "items")

# Global parameters for OPV workflow. Includes all variables from above defined function, except 'dataframe = im'

n_components = 3
heightless = False
outlier_threshold = 2.5
padding = 0
embedding_dim = 4
thresh = None     ## Persistence Watershed Threshold
nonlinear = True
normalize = True
zscale = False
data_type = 'Full_QNM'
data_subtype = 'P3HT:PCBM_OPV'
input_cmap = 'jet'


for h, im in enumerate(ims):
    print (f'-----------',files[h],'---------------------------------')
    print (f'--------------------',files[h],'------------------------')
    print (f'-----------------------------',files[h],'---------------')
    outliers, seg1_labels, seg2_labels = m2py_pipeline(im, heightless = heightless,
                                                          n_components = n_components,
                                                          outlier_threshold = outlier_threshold,
                                                          padding = padding,
                                                          embedding_dim = embedding_dim,
                                                          thresh = thresh,
                                                          nonlinear = nonlinear,
                                                          normalize = normalize,
                                                          zscale = zscale,
                                                          data_type = data_type,
                                                          data_subtype = data_subtype,
                                                          input_cmap = input_cmap)
    
    