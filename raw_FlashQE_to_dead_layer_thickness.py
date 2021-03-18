"""
@author: Mohammad Jobayer Hossain
"""

#--------------------------Imports---------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import ndimage
sample_id=str('B6-105-new')

#Setting
f_eff=0.05                  #Shading fraction
#Wavelength range where QE is kind of unchanged (needed in eqn 4, Eric's paper)
lambda1=490                 #490 nm
lambda2=890                 #890 nm
angle = 54                  #angle in degree

#initial guess
Leff=0.000400
N=10 #number of iterations

masking=1
dim=960
dim_cell=97


#-------------------- Dataframe: Panda World Starts-----------------------------------
df1 = pd.read_table('raw_FlashQE_B6_105_new.txt', skiprows=11) #skipping first 11 rows
R=df1[(df1['Param'])=='RS']    #create a dataframe df1 with 'RS' in 'Param' 
R=R.drop(df1.ix[:,'Site':'Temp_C'].head(0).columns, axis=1) #drop some columns
R=R.as_matrix()/100             #R=reflection; no more header, even wavelengths are lost

EQE=df1[(df1['Param'])=='EQE']    
EQE=EQE.drop(df1.ix[:,'Site':'Temp_C'].head(0).columns, axis=1) #drop some columns
EQE=EQE.as_matrix()/100             


AM_15G=pd.read_csv('AM_1.5G_edited.txt', header=None,delimiter = '\t')
w_len=AM_15G[0]                   #taking only wavelengths
w_len=w_len.as_matrix()

p_d=AM_15G[1]                     #second column=power density w/m^2/s

OpticalPropertiesOfSilicon=pd.read_csv('OpticalPropertiesOfSilicon.txt', skiprows=1, header=None,delimiter = '\t')
lambda_si=OpticalPropertiesOfSilicon[0]        #wavelengths, nm
La_si=OpticalPropertiesOfSilicon[4]            #absorption length, cm          


#----------------------Functions-------------------------------------------------------
def wavelength_interpolation(x,y): 
    #tck is the spline being constructed
    tck = interpolate.splrep(x, y, s=0)            #s=0 -> No smoothing/regression required
    xnew = np.arange(x[0], x[len(x)-1], 1)        #from x[0] to x[40] with 1 increment
    ynew = interpolate.splev(xnew, tck, der=0)
    return xnew, ynew

def find_index(function, target_value):
    for i in range (0,len(function)):
        if target_value==function[i]:
            index=i
    return index

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

#small mask
def small_mask_creation():
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
small_mask=np.zeros([dim_cell,dim_cell], dtype=float)
default_mask=np.zeros([dim_cell,dim_cell], dtype=float)+1
if masking==1:
    small_mask=small_mask_creation()
elif masking==0:
    small_mask=default_mask

#big mask
def big_mask_creation():
    mas=np.zeros([dim,dim], dtype=float)+1
    i1=133
    i2=179
    i3=463
    i4=505
    i5=783
    i6=827
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
big_mask=np.zeros([dim,dim], dtype=float)
default_mask=np.zeros([dim,dim], dtype=float)+1
if masking==1:
    big_mask=big_mask_creation()
elif masking==0:
    big_mask=default_mask

    
#1D to 2D conversion
def convert_1d_to_2d(x1):
    x2=np.zeros([97,97],dtype=float)
    z=0
    for p in range(97):
        x2[p,:]=x1[z:z+97]     #selecting column 0; in fact Js has only one column
        z=z+97
    return x2

#41 LED to 915 wavelengths interpolation for Power Density
p_density=np.zeros(915)
w_len_interpol, p_density=wavelength_interpolation(w_len, p_d)

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
    Js_xy=np.zeros([dim_cell,dim_cell],dtype=float)
    z=0
    for k in range(dim_cell):
        Js_xy[k,:]=Js[z:z+dim_cell,0]                     #selecting column 0; in fact Js has only one column
        z=z+dim_cell
    Js_xy=e*Js_xy
    return Js_xy


#----------------------Main Code-------------------------------------------------------
theta = angle*3.1416/180                    #angle in radian

IQE=EQE/(1-f_eff-R)
IQE[IQE<0]=0

#wavelength interpolation and slicing for Si                                #m=row=9409=number of pixels, n=column=number of wavelengths=41
lambda_si_new, La_si_new=wavelength_interpolation(lambda_si,La_si)        #silicon data has 121 wavelengths, now increaseing it with 1 nm interval
La=La_si_new[find_index(lambda_si_new, lambda1):find_index(lambda_si_new, lambda2)] #now La is from 490 to 890 nm , Initial array to enter into the loop

#slicing for IQE
m,n=np.shape(IQE)
Leff_all=np.zeros(m)
Wd_all=np.zeros(m) 
for i in range (0,m-1):
    IQE_local=IQE[i]                        #IQE(lambda) in a pixel, 1D
    IQE_sliced=IQE_local[find_index(w_len, lambda1):find_index(w_len, lambda2)+1] #IQE_sliced is from 490 nm to 890 nm now
    wlen_sliced=w_len[find_index(w_len, lambda1):find_index(w_len, lambda2)+1]
        
    wlen_interpolated, IQE_interpolated=wavelength_interpolation(wlen_sliced,IQE_sliced) #IQE_sliced is now in 1m interval
    k1=1
    k2=1
    for j in range(0,N):
        A=1/(La*np.cos(theta))
        B=np.log(k1*IQE_interpolated*(1+La*np.cos(theta)/Leff))  # y=m*x+c
        m1,c1=np.polyfit(A,B, deg=1)              #deg=degree of the fitting polynomial
        Wd=-m1
        k1=np.exp(-c1)
        
        C=La*np.cos(theta)
        D=(1/(k2*IQE_interpolated))*np.exp(Wd/(La*np.cos(theta)))
        m2,c2=np.polyfit(C,D, deg=1)
        Leff=c2/m2
        k2=c2
    Leff_all[i]=Leff            #Difusion length of all the pixels, [9409, 1]
    Wd_all[i]=Wd                #Dead layer thickness of all the pixels, [9409, 1]

#1D to 2D conversion
Leff_xy=np.zeros([dim_cell, dim_cell],dtype=float)
Wd_xy=np.zeros([dim_cell,dim_cell],dtype=float)
z=0
for p in range(dim_cell):
    Leff_xy[p,:]=Leff_all[z:z+dim_cell]     #selecting column 0; in fact Js has only one column
    Wd_xy[p,:]=Wd_all[z:z+dim_cell]
    z=z+dim_cell
    
#rotation
Leff_xy_rotated=matrix_rotation(Leff_xy, 1)
Wd_xy_rotated=matrix_rotation(Wd_xy, 1)


#----------------------------------Emitter and Base loss---------------------------

# lambda_si_new is the interpolated wavelength for La(Si) curve
# La_si_new is the interpolated form of La(Si)
# so both of them are [1200, 1] size vector from 250 nm to 1449 nm
m,n=np.shape(IQE)         #m=9409
N_l=len(lambda_si_new)    #N_l=1200
IQE_emi_loss=np.zeros([m,915],dtype=float)

#***********Emitter loss IQE2 ***********
IQE_emi_loss2=np.zeros([m,N_l],dtype=float)
for j in range (0,m-1):
    IQE_emi_loss2[j]=1-np.exp(-Wd_all[j]/La_si_new)           #function of wavelength, d(lambda_si_new)=1
    IQE_emi_loss[j, 136:915]= IQE_emi_loss2[j, 251:1030]                   #integration, because d(lambda)=1
    
#***********Emitter loss IQE1 ***********
IQE_emi_loss1=np.zeros([m,915],dtype=float)                 #m=9409

#Wavelength interpolation of IQE to make 1 nm interval
IQE_wave_interpol=np.zeros([m,915],dtype=float)
w_len_interpol=np.zeros(915)
for j in range (0,m-1):                                 #m=9409
    w_len_interpol, IQE_wave_interpol[j]=wavelength_interpolation(w_len,IQE[j,:])

#find index of La to match it with IQE i.e. from 365 nm to 1280 nm
s1=find_index(lambda_si_new, 365)
s2=find_index(lambda_si_new, 500)
s3=find_index(lambda_si_new, 1280)
La_si_base=La_si_new[s1:s3]

#loss2 calculation
for j in range (0,m-1):
    IQE_emi_loss1[j]=1-IQE_wave_interpol[j, :]*(1-La_si_base/Leff_all[j])           #function of wavelength, d(lambda_si_new)=1
    IQE_emi_loss[j, 0:s2]= IQE_emi_loss1[j, 0:s2] 
           
#***********Total Emitter loss Calculation ***********
Jsc_loss_emi_total_xy=Jsc_calculation(w_len_interpol, IQE_emi_loss, p_density)      #output is already in 2D, with 1D inputs, haha !

 
#***********Base loss calculation*********************
#key equation: IQE_base_loss=IQE-IQE*R-IQE_emi_loss = (1-R)IQE-IQE_emi_loss
#first of all interpolating 41 wavelengths to 915
#Wavelength interpolation of IQE to make 1 nm interval
p=len(R)                       #9409
R_total=np.zeros([p,915])
EQE_wave_interpol=np.zeros([p,915])
w_len_interpol=np.zeros(915)
for i in range (0, p):
    w_len_interpol, R_total[i,:]=wavelength_interpolation(w_len,R[i,:])    
    w_len_interpol, EQE_wave_interpol[i,:]=wavelength_interpolation(w_len, EQE[i,:]) 

IQE_ideal=np.ones([p,915])                             #If there was no loss. All the elements are 1 
Jsc_ideal=Jsc_calculation(w_len_interpol[0:829], IQE_ideal[:, 0:829], p_density[0:829])             #output is already in 2D, with 1D inputs, haha !
Jsc_practical=Jsc_calculation(w_len_interpol, EQE_wave_interpol, p_density) #The Jsc we get
Jsc_loss_reflect_shading=Jsc_calculation(w_len_interpol, IQE_wave_interpol*(R_total+f_eff), p_density)

Jsc_loss_bulk_rear_xy=Jsc_ideal-Jsc_practical-Jsc_loss_reflect_shading

#rotation
Jsc_loss_emi_total_xy_rotated=matrix_rotation(Jsc_loss_emi_total_xy, 1)
Jsc_loss_bulk_rear_xy_rotated=matrix_rotation(Jsc_loss_bulk_rear_xy, 1)

#unit conversion
Leff_xy_rotated=1e3*Leff_xy_rotated
Wd_xy_rotated=1e7*Wd_xy_rotated
Jsc_loss_emi_total_xy_rotated=Jsc_loss_emi_total_xy_rotated*0.1                     #converted from A/m^2 to mA/cm^2
Jsc_loss_bulk_rear_xy_rotated=Jsc_loss_bulk_rear_xy_rotated*0.1


#Spatial interpolation
Leff_interpolated=matrix_enlargement(Leff_xy_rotated,dim/dim_cell)
Wd_interpolated=matrix_enlargement(Wd_xy_rotated,dim/dim_cell)
base_loss_interpolated=matrix_enlargement(Jsc_loss_bulk_rear_xy_rotated,dim/dim_cell)
emi_loss_interpolated=matrix_enlargement(Jsc_loss_emi_total_xy_rotated,dim/dim_cell)


#------------------------------------Plotting Starts--------------------------------


plt.figure(1)
plt.imshow(Leff_interpolated*big_mask, vmin = 10, vmax = 150, cmap = 'inferno')
cbar = plt.colorbar()
#plt.title('Leff_xy Map (um)')
plt.axis('off')
plt.gcf().savefig(r'D:\STUDY_CREOL\Research with Davis\pv-characterization-analysis\2-working\pl_qe_paper\B6-105-new\Generated_Images_QE\TIF\Leff_xy.tiff', dpi=300, bbox_inches='tight')
im1t=np.savetxt(r'D:\STUDY_CREOL\Research with Davis\pv-characterization-analysis\2-working\pl_qe_paper\B6-105-new\Generated_Images_QE\TEXT\Leff_xy.txt', Leff_interpolated*big_mask)

plt.figure(2)
plt.imshow(Wd_interpolated*big_mask, vmin = 0.4, vmax = 10, cmap = 'inferno')
cbar = plt.colorbar()
#plt.title('Wd_xy Map (nm)')
plt.axis('off')
plt.gcf().savefig(r'D:\STUDY_CREOL\Research with Davis\pv-characterization-analysis\2-working\pl_qe_paper\B6-105-new\Generated_Images_QE\TIF\Wd_xy.tiff', dpi=300, bbox_inches='tight')
im2t=np.savetxt(r'D:\STUDY_CREOL\Research with Davis\pv-characterization-analysis\2-working\pl_qe_paper\B6-105-new\Generated_Images_QE\TEXT\Wd_xy.txt', Wd_interpolated*big_mask)


plt.figure(3)
plt.imshow(emi_loss_interpolated*big_mask, vmin = 0.3, vmax = 1.7, cmap = 'inferno')
cbar = plt.colorbar()
#plt.title('Emitter loss Map (mA/$\mathregular{cm^2}}$)')
plt.axis('off')
plt.gcf().savefig(r'D:\STUDY_CREOL\Research with Davis\pv-characterization-analysis\2-working\pl_qe_paper\B6-105-new\Generated_Images_QE\TIF\Emitter_Loss.tiff', dpi=300, bbox_inches='tight')
im3t=np.savetxt(r'D:\STUDY_CREOL\Research with Davis\pv-characterization-analysis\2-working\pl_qe_paper\B6-105-new\Generated_Images_QE\TEXT\Emitter_Loss.txt', emi_loss_interpolated*big_mask)

plt.figure(4)
plt.imshow(base_loss_interpolated*big_mask, vmin = 6.5, vmax = 9.5, cmap = 'inferno')
cbar = plt.colorbar()
#plt.title('Bulk and Rear loss Map (mA/$\mathregular{cm^2}}$)')
plt.axis('off')
plt.gcf().savefig(r'D:\STUDY_CREOL\Research with Davis\pv-characterization-analysis\2-working\pl_qe_paper\B6-105-new\Generated_Images_QE\TIF\Base_Loss.tiff', dpi=300, bbox_inches='tight')
im4t=np.savetxt(r'D:\STUDY_CREOL\Research with Davis\pv-characterization-analysis\2-working\pl_qe_paper\B6-105-new\Generated_Images_QE\TEXT\Base_Loss.txt', base_loss_interpolated*big_mask)



#*********************Post Processing*************************************

'''
base_loss_1d=(base_loss_interpolated*big_mask).ravel()
emi_loss_1d=(emi_loss_interpolated*big_mask).ravel()

Base_loss_J0_correlation=pd.read_csv('Base_loss_J0_correlation.txt', skiprows=1, header=None,delimiter = '\t')
base_loss=Base_loss_J0_correlation[0]                  #taking only wavelengths
J0_Jsc=Base_loss_J0_correlation[1]
J0_Jsc_xy=Base_loss_J0_correlation[2]                    
  
plt.figure(13)
#base_loss[base_loss>15]=0
plt.scatter(base_loss, J0_Jsc_xy, color='red', marker='.')
plt.xlim(8, 25)
plt.ylim(0.5e-10, 7e-10)
plt.title('Scatterplot base_loss vs J0')

plt.figure(14)
plt.scatter(base_loss, J0_Jsc_xy, color='red', marker='.')
plt.xlim(0.1, 30)
plt.ylim(0.5e-10, 7e-10)
plt.title('J0(Jsc_xy) vs base loss ')


plt.figure(14)
z = np.polyfit(base_loss, J0_Jsc_xy, 1)
p = np.poly1d(z)
plt.plot(base_loss,p(base_loss),"g--")
# the line equation:
#print ('J0_Jsc_xy=%.6fbase_loss+(%.6f)'%(z[0],z[1]))
#plt.gcf().savefig(str(sample_id)+r'\Voc_Jsc_correlation.tiff', dpi=600)
'''

