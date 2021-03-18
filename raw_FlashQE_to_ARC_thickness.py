"""
@author: Mohammad Jobayer Hossain
"""

#--------------------------Imports--------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import ndimage

#Setting
masking=1
dim=960

i_st=2        #index_wavelength_start
i_en=15       #index_wavelength_end

#---------------------------------Functions-------------------------------------------

def wavelength_interpol_ARC(x,y, xnew, ynew): 
    #tck is the spline being constructed
    tck = interpolate.splrep(x, y, s=6)     #s=0 -> No smoothing/regression required
    xnew = np.arange(x[i_st], x[i_en], 1)        #from x[2] to x[15] with 1 increment
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew

def matrix_rotation(image_in, theta):
    #clockwise rotation: theta=-1,  anti-clockwise rotation: theta=+1
    n,m=image_in.shape  #row, column
    for i in range(m):
        b=image_in[:,i]
        b = b[::theta]
        image_in[:,i]=b
    image_out=image_in.transpose()
    return image_out

def matrix_enlargement(matrix_in,u):
    #order=3 means cubic spline interpolation,  order=2 means bilinear interpolation
    #Resampled by a factor of u
    #u=how many times.   u=5.5 means enlarge 5.5 times
    matrix_out=(ndimage.zoom(matrix_in, u, order=1))  
    return matrix_out


def mask_creation(mas):
    mas=np.zeros([dim,dim], dtype=float)+1
    i1=133
    i2=180
    i3=462
    i4=506
    i5=783
    i6=829
    for i in range(i1, i2):
        for j in range(dim):
            mas[i,j]=0
    for i in range(i3, i4):
        for j in range(dim):
            mas[i,j]=0
    for i in range(i5, i6):
        for j in range(dim):
            mas[i,j]=0
    return mas
mask=np.zeros([dim,dim], dtype=float)
default_mask=np.zeros([dim,dim], dtype=float)+1
if masking==1:
    mask=mask_creation(mask)
elif masking==0:
    mask=default_mask
    
def ARC_calc(df1, w_len):
    df1=df1[(df1['Param'])=='RS']    #create a dataframe df1 with 'RS' in 'Param' 
    df1=df1.drop(df1.ix[:,'Site':'Temp_C'].head(0).columns, axis=1) #drop some columns
    R=df1.as_matrix()             #R=reflection; no more header, even wavelengths are lost
    w_len=w_len[0]                  #taking only wavelengths
    
    #wavelength interpolation
    p=len(R)                       #9409
    R_interpol=np.zeros([p,315])
    w_len_interpol=np.zeros(315)
    w_len_min=np.zeros(p)
    for i in range (0, p):
        w_len_interpol, R_interpol[i,:]=wavelength_interpol_ARC(w_len,R[i,:], w_len_interpol, R_interpol[i,:])    
    
    #Wavelength at which minimum Reflection occurs
    min_key=np.argmin(R_interpol, axis=1)  #axis=1 means maximum along row, axis=0 -> along column 
    for j in range (0, p):
        w_len_min[j] = w_len_interpol[min_key[j]]
        
    print('value='+ str(np.min(w_len_min)))
    
    #Rearrange the wavelength values into 2D image
    w_len_min_xy=np.zeros([97,97],dtype=float)
    z=0
    for k in range(0,97):
        w_len_min_xy[k,:]=w_len_min[z:z+97]     #selecting column 0; 
        z=z+97
    ARC_xy=w_len_min_xy/(4*2.04)
    return w_len, R, w_len_interpol, R_interpol, w_len_min_xy,ARC_xy
#---------------------------------Main Code--------------------------------------
#raw_FlashQE_B6_105_new
df1 = pd.read_table('raw_FlashQE_B6_105_new.txt', skiprows=11) #skipping first 11 rows
w_len=pd.read_csv('AM_1.5G_edited.txt', header=None,delimiter = '\t')

w_len, R, w_len_interpol, R_interpol, w_len_min_xy,ARC_xy=ARC_calc(df1, w_len)

#--------------------------------------Plotting--------------------------

plt.figure(1)
plt.imshow(w_len_min_xy, vmin=520,  vmax=600, cmap = 'inferno')
cbar = plt.colorbar()
plt.title('Wavelengths (nm) of Minimum Reflection')


#rotation
ARC_xy_rotated=matrix_rotation(ARC_xy, 1)
#interpolation
ARC_xy_interpolated=matrix_enlargement(ARC_xy_rotated,dim/97)

plt.figure(2)
plt.imshow(ARC_xy_interpolated*mask, vmin=65,  vmax=73, cmap = 'inferno')
cbar = plt.colorbar()
plt.axis('off')
plt.gcf().savefig('SiN Thickness.tiff', dpi=300, bbox_inches='tight')


plt.figure(3)
orig=plt.scatter(w_len, R[100, :], color='r')
new=plt.scatter(w_len_interpol, R_interpol[100, :], s=0.05, color='g')
plt.figure(3)
#orig=plt.scatter(w_len, R[100, :], marker='.', color='r')
#new=plt.scatter(w_fit, R_fit[100, :], marker='.', color='g')
plt.legend((orig, new),('Original', 'New'),fontsize=8)