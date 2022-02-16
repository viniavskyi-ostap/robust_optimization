from __future__ import division
import numpy as np
from bundle_io import *
from clipped_irls import adjust_structure_and_motion, NLSQ_ParamDesc
#from saddle_free_irls import adjust_structure_and_motion, NLSQ_ParamDesc


paramDesc = NLSQ_ParamDesc(10)
FILE_NAME = "data/problem-49-7776-pre.txt.bz2"
#FILE_NAME = "data/problem-16-22106-pre.txt.bz2"

#Read BAL data from file
cams, Xs, correspondingView, correspondingPoint, measurements2d = readBundleData(FILE_NAME)


#Start BA
adjust_structure_and_motion(cams, Xs, measurements2d, correspondingView, correspondingPoint, 1.0, \
                            clip_IRLS_Weights=False, log_file='logs/normal_IRLS.txt')



