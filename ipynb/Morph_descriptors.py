import os
import sys
#from decimal import Decimal

import numpy as np
import pandas as pd
#import scipy.stats as stats
#import matplotlib.pyplot as plt
#%matplotlib inline

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
#from m2py.utils import config
#from m2py.utils import pre_processing as pre
#from m2py.utils import post_processing as post
from m2py.utils import seg_label_utils as slu
#from m2py.segmentation import segmentation_gmm as seg_gmm
#from m2py.segmentation import segmentation_watershed as seg_water

map_file_path = '/Volumes/Tatum_SSD-1/Grad_School/m2py/Morphology_labels/OFET_morph_maps/Default_params/'
files = os.listdir(map_file_path)
print (len(files))
print (files)

seg1_fl_list = []
seg2_fl_list = []

seg1_dict = {}
seg2_dict = {}

for fl in files:
    if fl[-5] == '1':
        seg1_fl_list.append(fl)
    elif fl[-5] == '2':
        seg2_fl_list.append(fl)
    else:
        print(fl[-5], ' is messed up')

for k, fl in enumerate(seg1_fl_list):
    seg1_dict[k] = np.load(map_file_path+fl)
    
for k, fl in enumerate(seg2_fl_list):
    seg2_dict[k] = np.load(map_file_path+fl)
    
for i in range(len(seg1_dict)):
    domain_labels = slu.relable(seg2_dict[i])
    phase_labels = slu.relable(seg1_dict[i])
    
    domain_props = slu.all_domain_properties(phase_labels, domain_labels)
    outfile = map_file_path+seg1_fl_list[i][:-8]+'domain_metrics.csv'
    domain_props.to_csv(outfile)
    
    print ('finished file # ', i)