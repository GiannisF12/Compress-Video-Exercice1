import yuvio #https://pypi.org/project/yuvio/
import cv2 #opencv
import scipy.fftpack
import numpy as np

def function_4x4(arr, rows, cols):
    height, width = arr.shape
    return (arr.reshape(height//rows, rows, -1, cols).swapaxes(1,2).reshape(-1, rows, cols))


#Read and Write File with python package Yuvio

yuv_frame = yuvio.imread("1920x1080.yuv", 1920, 1080, "yuv420p")
yuvio.imwrite("1920x1080.yuv", yuv_frame)

y = yuv_frame.y # we take only Y data from the YUV Frame

print('loading.....')

array_dct = function_4x4(y,4,4) #4x4 DCT
array_dst= function_4x4(y,4,4) #4x4 DST


#define arrays from numpy
average_energy_dct = np.zeros([4,4])
average_energy_dst = np.zeros([4,4])  
average_energy_before = np.zeros([4,4]) 

for i in range(len(array_dct)):
    average_energy_before=average_energy_before + pow(array_dct[i][0][0],2) #for the c) task (einai idia kai gia to dct kai gia to dst)
    array_dct[i] = scipy.fftpack.dct(array_dct[i])
    array_dst[i] = scipy.fftpack.dst(array_dst[i])
    for j in range(4):
        for z in range(4):
            average_energy_dct[j][z] = average_energy_dct[j][z] + pow(array_dct[i][j][z],2)
            average_energy_dst[j][z] = average_energy_dst[j][z] + pow(array_dst[i][j][z],2)


for i in range(4):
    for j in range(4):
        average_energy_dct[i][j] = average_energy_dct[i][j]/len(array_dct)
        average_energy_dst[i][j] = average_energy_dst[i][j]/len(array_dst)

        average_energy_dct[i][j] = average_energy_dct[i][j]/128
        average_energy_dst[i][j] = average_energy_dst[i][j]/128



#DCT ASKISI1
average_energy_before = average_energy_before/len(array_dct)
average_energy_before = average_energy_before/128

print('Average Energy DCT: ',average_energy_dct)
print('Average Energy DCT Before: ',average_energy_before)
print('\n')

#DST ASKISI2
print('Average Energy DST: ',average_energy_dst)
print('Average Energy DST Before: ',average_energy_before)


