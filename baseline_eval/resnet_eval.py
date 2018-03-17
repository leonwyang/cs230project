import matplotlib.pyplot as plt
import numpy as np
from evaluations import *
from os import  listdir
import os
from os.path import isfile, join
import shutil
import scipy.misc
for ff in os.listdir("/home/ubuntu/cs230project/SRGAN-tensorflow/result/Models"):
    if os.path.isdir(join("/home/ubuntu/cs230project/SRGAN-tensorflow/result/Models",ff)) is not True: continue
    if ff.startswith("Interpolated"): continue
    main_dir = join("/home/ubuntu/cs230project/SRGAN-tensorflow/result/Models",ff)

    f_stats = open(join(main_dir , 'stats.txt'), 'w')
    evals = [0.,0.,0.,]
    cnt = 0 
    for f in listdir(join(main_dir,'images')):
        if f.endswith('outputs.png'):
            cnt += 1
            resnet_output =  scipy.misc.imread(join(main_dir,'images',f))
            resnet_target = scipy.misc.imread(join(main_dir,'images',f.split('-')[0]+"-targets.png"))
            w = min(resnet_output.shape[0], resnet_target.shape[0])
            h = min(resnet_output.shape[1], resnet_target.shape[1])
            resnet_output = resnet_output[:w,:h]
            resnet_target = resnet_target[:w,:h]
        
            mse = MSE(resnet_output, resnet_target)
            pnsr=PSNR(resnet_output, resnet_target)
            ssim = SSIM_Array(resnet_output, resnet_target)
            f_stats.write(f+" MSE: %f PSNR: %f SSIM: %f \n" % (mse,pnsr,ssim))
            evals[0] += mse
            evals[1] += pnsr
            evals[2] += ssim
    evals = [e  * 1.0 / cnt for e in evals]
    f_stats.write(str(evals))
    f_stats.close()
    print (ff)
