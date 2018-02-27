import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from evaluations import *
from os import  listdir
import os
from os.path import isfile, join
import shutil

# Change it as it you need
main_dir = "/Users/zhangqixiang/Desktop/tmp"


A = -0.5 # HyperParam for sampling formula
def bicubicWeight(x):
    x_abs = abs(x)
    if x_abs <= 1 :
        res = 1 - (A+3)*(x_abs**2)+(A+2)*(x_abs**3)
    elif x_abs<=2 :
        res = -4*A+8*A*x_abs-5*A*(x_abs**2)+A*(x_abs**3)
    else : 
        raise ValueError("bicubicWeight input over 2", x)
    return res 

def interpolate(lr_image, hr_shape, method='bilinear'):
    ratio_vertical = hr_shape[0]/lr_image.shape[0]
    ratio_horizontal = hr_shape[1]/lr_image.shape[1]
    num_channels = lr_image.shape[2]
#     assert ratio_vertical is 2 and ratio_horizontal is 2 and  num_channels is 1
    interpolated_image = np.zeros(hr_shape,dtype=np.uint8)
    for i in range(lr_image.shape[0]):
        for j in range(lr_image.shape[1]):
            i_next = i+1 if i < lr_image.shape[0]-1 else i
            j_next = j+1 if j < lr_image.shape[1]-1 else j 
            
            for c in range(num_channels):
                if method == 'bilinear':
                    upperLeft =  lr_image[i][j][c]
                    upperRight = lr_image[i][j_next][c] 
                    lowerLeft = lr_image[i_next][j][c]  
                    lowerRight = lr_image[i_next][j_next][c]
                if method == 'bicubic':
                    i_nbrs =  list( min(lr_image.shape[0]-1,  max(0,x))  for x in  [i-1,i,i+1,i+2])
                    j_nbrs =  list( min(lr_image.shape[1]-1,  max(0,x)) for x in  [j-1,j,j+1,j+2])
                    B = np.zeros((4,4))
                    for ii, i_nbr in enumerate(i_nbrs):
                        for jj, j_nbr in enumerate(j_nbrs):
                            B[ii][jj] = lr_image[i_nbr][j_nbr][c]
                            
            
                for di in range(ratio_vertical):
                    for dj in range(ratio_horizontal):
                        v = di*1.0/ratio_vertical
                        h = dj*1.0/ratio_horizontal
                        
                        if method == 'bilinear':
                            value = np.array([1-h,h]).dot(np.array([[upperLeft,lowerLeft],[upperRight,lowerRight]])).dot(np.array([1-v,v]).reshape(2,1))
                        elif method == 'bicubic' :
                            vs = np.array([bicubicWeight(x) for x in [v+1,v,1-v,2-v]])
                            us = np.array([bicubicWeight(x) for x in [h+1,h,1-h,2-h]]).T
                            value = vs.dot(B).dot(us)   
                        value = min(255,max(value, 0))
                        interpolated_image[i*ratio_vertical+di][j*ratio_horizontal+dj][c] = (np.uint8) (value)
                       
    return interpolated_image 
if __name__ == "__main__":
    # Establish new directories for bicubic and bilinear interpolation
    if (os.path.exists(join(main_dir,'Interpolated_bilinear'))): shutil.rmtree(join(main_dir,'Interpolated_bilinear'))
    if (os.path.exists(join(main_dir,'Interpolated_bicubic'))): shutil.rmtree(join(main_dir,'Interpolated_bicubic'))
    os.mkdir(join(main_dir,'Interpolated_bicubic'))
    os.mkdir(join(main_dir,'Interpolated_bilinear'))
    

    f_stats = open(join(main_dir , 'stats.txt'), 'w')
    bicubic = [0.,0.,0.]
    bilinear = [0.,0.,0.]
    cnt = 0
    for f in listdir(join(main_dir,'HR_test')):
        if f.endswith('png'): 
            cnt += 1
            lr_image = scipy.misc.imread(join(main_dir,'LR_test',f))
            hr_image = scipy.misc.imread(join(main_dir,'HR_test',f))
            interpolated_image_bilinear =  interpolate(lr_image, hr_image.shape, 'bilinear')
            interpolated_image_bicubic =  interpolate(lr_image, hr_image.shape, 'bicubic')
            scipy.misc.imsave(join(main_dir,'Interpolated_bicubic',f), interpolated_image_bicubic)
            scipy.misc.imsave(join(main_dir,'Interpolated_bilinear',f), interpolated_image_bilinear)
            
            bicubic [0] += MSE(hr_image,interpolated_image_bicubic)
            bicubic[1] += PSNR(hr_image, interpolated_image_bicubic)
            bicubic[2] += SSIM_Array(hr_image,interpolated_image_bicubic )
            bilinear [0] += MSE(hr_image,interpolated_image_bilinear)
            bilinear[1] += PSNR(hr_image, interpolated_image_bilinear)
            bilinear[2] += SSIM_Array(hr_image,interpolated_image_bilinear )
            
            msg1 = (f+"\n")
            msg2 = ("Interpolated_bicubic MSE: %f PSNR: %f SSIM: %f \n" % (MSE(hr_image,interpolated_image_bicubic), PSNR(hr_image, interpolated_image_bicubic),SSIM_Array(hr_image,interpolated_image_bicubic )))
            msg3 = ("Interpolated_bilinear MSE: %f PSNR: %f SSIM: %f \n" % (MSE(hr_image,interpolated_image_bilinear), PSNR(hr_image, interpolated_image_bilinear),SSIM_Array(hr_image,interpolated_image_bilinear )))
            f_stats.write(msg1+msg2+msg3)
            print msg1,msg2,msg3
    
    bicubic = [ e*1.0/cnt for e in bicubic ]
    bilinear = [e*1.0/cnt for e in bilinear ]
    print bicubic, bilinear
    f_stats.write(" overall bicubic: %f \n" %( str(bicubic) ))
    f_stats.write(" overall bilinear: %f \n" %(str(bilinear) ))
    f_stats.close()
    
#    
# lr_image = scipy.misc.imread(imagePath1)
# hr_image = scipy.misc.imread(imagePath2)
#  
# interpolated_bilinear_real = scipy.misc.imresize(lr_image, hr_image.shape, interp='bilinear', mode=None)
# interpolated_bicubic_real = scipy.misc.imresize(lr_image, hr_image.shape,interp='bicubic', mode=None)
# 
# interpolated_mine_bilinear = interpolate(lr_image,hr_image.shape,method ='bilinear' )
# interpolated_mine_bicubic = interpolate(lr_image,hr_image.shape,method ='bicubic' )
# print np.mean( (interpolated_mine_bicubic-interpolated_bicubic_real) **2)
# scipy.misc.imsave(dest_dir+'sample_bilinear.png', interpolated_mine_bilinear)
# scipy.misc.imsave(dest_dir+'sample_bicubic.png', interpolated_mine_bicubic)
# 
#   
# plt.subplot(211)
# plt.imshow(interpolated_mine_bilinear)
#   
# plt.subplot(212)
# plt.imshow(interpolated_mine_bicubic)
# 
# print SSIM_Array(interpolated_mine_bilinear, interpolated_bilinear_real)
 

#  
# plt.subplot(331)
# plt.imshow(hr_image)
# plt.show()

