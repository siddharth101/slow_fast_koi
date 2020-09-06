from gravityspy.utils import utils
import numpy as np
import pandas as pd

def make_scans(gpstimelist,path):
    
    for i in gpstimelist:
       spec,q_value = utils.make_q_scans(i,channel_name='L1:GDS-CALIB_STRAIN')
       utils.save_q_scans(path,spec,[0,25.5],[0.5,1.0,2.0,4.0],'L1',i)
       
    return
