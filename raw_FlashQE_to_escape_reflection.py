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
f_eff=0.05    #effective shading fraction
masking=1
dim=960
dim_cell=97
X=201      #For which pixel do you want the front reflectance, escape reflectance and total reflectance graph                                                                                                      
#---------------------------------Functions------------------------------------------------------------------
'''
def wavelength_interpolation(x,y, xnew, ynew): 
    tck = interpolate.splrep(x, y, s=0)     #s=0 -> No smoothing/regression required
    xnew = np.arange(x[0], x[40], 1)        #from x[0] to x[40] with 1 increment
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew
'''
def wavelength_interpolation(x,y, xnew, ynew): 
    xnew = np.arange(x[0], x[len(x)-1], 1)        #from x[0] to x[40] with 1 increment
    ynew = np.interp(xnew, x, y)
    return xnew, ynew

'''
def wavelength_extrapolation(x,y, xnew, ynew): 
    tck = interpolate.splrep(x[0:30], y[0:30], s=2)     #s=0 -> No smoothing/regression required
    xnew = np.arange(x[0], x[40], 1)        #from x[0] to x[40] with 1 increment
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew
'''
def wavelength_extrapolation(x,y, xnew, ynew): 
    x_dummy=x
    y_dummy=y
    m,c=np.polyfit(x_dummy[11:29],y_dummy[11:29], deg=1)
    y_dummy[30:41]=m*x_dummy[30:41]+c
    xnew = np.arange(x_dummy[0], x_dummy[40], 1)        #from x[0] to x[40] with 1 increment
    ynew = np.interp(xnew, x_dummy, y_dummy)
    return xnew, ynew

def matrix_rotation(image_in, theta):
    #clockwise rotation: theta=-1,  anti-clockwise rotation: theta=+1
    n,m=image_in.shape
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
    mas=np.zeros([dim_cell,dim_cell], dtype=float)+1
    i1=14
    i2=19
    i3=46
    i4=51
    i5=79
    i6=84
    for i in range(i1, i2):
        for j in range(dim_cell):
            mas[i,j]=0
    for i in range(i3, i4):
        for j in range(dim_cell):
            mas[i,j]=0
    for i in range(i5, i6):
        for j in range(dim_cell):
            mas[i,j]=0
    return mas
mask=np.zeros([dim_cell,dim_cell], dtype=float)
default_mask=np.zeros([dim_cell,dim_cell], dtype=float)+1
if masking==1:
    mask=mask_creation(mask)
elif masking==0:
    mask=default_mask
    

#-------------------- Dataframe: Panda World Starts------------------------------------------
df1 = pd.read_table('raw_FlashQE_B6_105_new.txt', skiprows=11) #skipping first 11 rows
df_rs=df1[(df1['Param'])=='RS']    #create a dataframe df_rs with 'RS' in 'Param' 
df_rs=df_rs.drop(df_rs.ix[:,'Site':'Temp_C'].head(0).columns, axis=1) #drop some columns
R=df_rs.as_matrix()/100             #R=reflection; no more header, even wavelengths are lost

AM_15G=pd.read_csv('AM_1.5G_edited.txt', header=None,delimiter = '\t')
w_len=AM_15G[0]                  #taking only wavelengths

#----------------------Main Code------------------------------------------------------------
#Total reflectance from wavelength interpolation
p=len(R)                       #9409
R_total=np.zeros([p,915])
w_len_interpol=np.zeros(915)
for i in range (0, p):
    w_len_interpol, R_total[i,:]=wavelength_interpolation(w_len,R[i,:], w_len_interpol, R_total[i,:])    
plt.figure(1)
imx=plt.plot(w_len_interpol, R_total[X,:], label='Total Reflectance')
#plt.title('Total Reflection')


#---------------------Escape reflection----------------------------------------------------------------------
#Front surface reflectance R_front
R_front=np.zeros([p,915])
w_len_extrapol=np.zeros(915)
for j in range (0, p):  #p=locations
    w_len_extrapol, R_front[j,:]=wavelength_extrapolation(w_len,R[j,:], w_len_extrapol, R_total[j,:])   

#plt.figure(2)
imy=plt.plot(w_len_extrapol, R_front[X,:], label='Front Reflectance')
#plt.title('Front Reflection')

#escape reflectance. Ref: McIntosh, 'Light Trapping in Sunpower's A-300 Solar Cells'
R_escape=(R_total-R_front)/(1-R_front)

#plt.figure(3)
imz=plt.plot(w_len_extrapol, R_escape[X,:], label='Escape Reflectance')
#plt.title('Escape Reflection')
plt.xlabel('wavelength(nm)')
plt.ylabel('Reflectance')

#plt.gcf().savefig('Reflectance_spectrum.tiff', dpi=200, bbox_inches='tight')
#-----------------------Take EQE Now: First Step towards Jsc---------------------------------------------------------------

p_d=AM_15G[1]        #second column=power density w/m^2/s

df_eqe=df1[(df1['Param'])=='EQE']
df_eqe=df_eqe.drop(df_eqe.ix[:,'Site':'Temp_C'].head(0).columns, axis=1)
FlashQE=df_eqe.as_matrix()/100 #deviding by 100 to make it the actual value

#41 LED to 915 wavelengths interpolation for QE
p=len(FlashQE)                       
EQE=np.zeros([p,915])
w_len_interpol=np.zeros(915)
for i in range (0, p):
    w_len_interpol, EQE[i,:]=wavelength_interpolation(w_len, FlashQE[i,:], w_len_interpol, EQE[i,:])    

#41 LED to 915 wavelengths interpolation for Power Density
p_density=np.zeros(915)
w_len_interpol, p_density=wavelength_interpolation(w_len, p_d, w_len_interpol, p_density)


#-------------------------------------------Current Density------------------------------------------
def Jsc_calculation(l, EQE, p_density):
    #constants
    hc=6.62607004e-34*3e8       
    e=1.6e-19
    #variable initiation
    
    n,m=np.shape(EQE)       #n=row=9409, m=column=915
    y=np.zeros([n,m],dtype=float)
    for i in range(n):
        y[i,:]=EQE[i,:]*p_density*l/hc*1e-9         #y=EQE*Phi; phi=p_density/(hc/l)  ; 9409*41 matrix
    #dl calculation
    dl=1
    #multiplication with dl
    dJs=np.zeros([n,m],dtype=float)
    for i in range(n):                              # n=row of y : 9409
        dJs[i,:]=y[i,:]*dl                          #first column of every row=0, because dl[0]=0
    # Integration output
    Js=np.zeros([n,1],dtype=float)
    for i in range(n):
        Js[i,:]=np.sum(dJs[i,:])                    # Integration(EQE*phi*dl)
    #Js_xy[97, 97] from Js[0:9409]
    Js_xy=np.zeros([97,97],dtype=float)
    z=0
    for k in range(97):
        Js_xy[k,:]=Js[z:z+97,0]                     #selecting column 0; in fact Js has only one column
        z=z+97
    Js_xy=e*Js_xy
    return Js_xy



#IQE
IQE=EQE/(1-f_eff-R_total)

Jsc_loss_total_reflect=Jsc_calculation(w_len_interpol, IQE*R_total, p_density)
Jsc_loss_front=Jsc_calculation(w_len_interpol, IQE*R_front, p_density)
Jsc_loss_escape=Jsc_calculation(w_len_interpol, IQE*R_escape, p_density)

#loss calculation
total_reflect_loss=Jsc_loss_total_reflect*0.1    #converted from A/m^2 to mA/cm^2
front_loss=Jsc_loss_front*0.1 #converted from A/m^2 to mA/cm^2
escape_loss=Jsc_loss_escape*0.1 #converted from A/m^2 to mA/cm^2


#--------------------------------image formating-----------------------------
#rotation
total_reflect_loss_rotated=matrix_rotation(total_reflect_loss, 1)
front_loss_rotated=matrix_rotation(front_loss, 1)
escape_loss_rotated=matrix_rotation(escape_loss, 1)
#interpolation
total_loss_interpolated=matrix_enlargement(total_reflect_loss_rotated,dim/97)
front_loss_interpolated=matrix_enlargement(front_loss_rotated,dim/97)
escape_loss_interpolated=matrix_enlargement(escape_loss_rotated,dim/97)

plt.figure(4)
plt.imshow(total_reflect_loss_rotated*mask, vmin=2.6, vmax=3.35, cmap = 'inferno')
#plt.title('Total Reflection loss Map (mA/cm^2)')
cbar = plt.colorbar()
plt.axis('off')
#plt.gcf().savefig('Reflection_Loss.tiff', dpi=300, bbox_inches='tight')

plt.figure(5)
plt.imshow(front_loss_rotated*mask, vmin=2.4, vmax=3.25,  cmap = 'inferno')
#plt.title('Front loss Map (mA/cm^2)')
cbar = plt.colorbar()
plt.axis('off')
#plt.gcf().savefig('Front_reflection.tiff', dpi=300, bbox_inches='tight')

plt.figure(6)
plt.imshow(escape_loss_rotated*mask, vmin=0.04, vmax=0.15, cmap = 'inferno')
#plt.title('Escape loss Map (mA/cm^2)')
cbar = plt.colorbar()
plt.axis('off')
#plt.gcf().savefig('Escape_reflection.tiff', dpi=300, bbox_inches='tight')
                   