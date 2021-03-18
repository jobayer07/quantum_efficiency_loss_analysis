import numpy as np
import scipy.ndimage  #used for image enlargement
x = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 11, 12, 13]], np.int32)
#print ('Original Image:', x)
#im = plt.imshow(x,cmap = 'inferno')
#cbar = plt.colorbar()

#Clockwise 90 degree shifting
n,m=x.shape
for i in range(m):
    b=x[:,i]
    b = b[::-1]
    x[:,i]=b
x=x.transpose()
plt.figure(1)
im = plt.imshow(x,cmap = 'inferno')
cbar = plt.colorbar()
#print ('Clockwise 90 degree Shifted Image:', x)

#Image Enlargement Now
#order=3 means cubic spline interpolation,  order=2 means bilinear interpolation
#Resampled by a factor of u
u=5.5   #5.5 enlarge 5.5 times
y=(scipy.ndimage.zoom(x, u, order=3))   
#print('enlarged image=', y)
plt.figure(2)
im2 = plt.imshow(y,cmap = 'inferno')
cbar = plt.colorbar()