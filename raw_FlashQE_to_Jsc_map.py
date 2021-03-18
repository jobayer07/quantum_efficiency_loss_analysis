"""
@author: Mohammad Jobayer Hossain
"""

#--------------------------Imports--------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-------------------- Dataframe: Panda World Starts-----------------------------
df1 = pd.read_table('raw_FlashQE.txt', skiprows=11) #skipping first 11 rows
df2=pd.read_table ('AM_1.5G_edited.txt',header=None)

df1=df1[(df1['Param'])=='EQE']    #create a dataframe df1 with 'EQE' in 'Param'
#df1=df1.drop('Site',1,inplace=True) 
df1=df1.drop(df1.ix[:,'X_mm':'Temp_C'].head(0).columns, axis=1)
#print(df1)

#--------------Conversion to Matrix: Numpy World Starts-----------------------------
#Conversion to matrices
AM_15G = df2.as_matrix()    #1st column=wavelengths (nm), 2nd column Illumination W.m^2.nm^-1
FlashQE=df1.as_matrix()/100 #Wow ! got rid of the header as well; deviding by 100 to make it the actual value

hc=6.62607004e-34*3e8       #planck's constant*speed of light
e=1.6e-19                   #charge of an electron


#----------------Manin Code Starts----------------------------------
#Photon flux=Power density/(hc/lambda)
#Replacig 2nd column of AM_15G by EQE*Photon_flux/ #changing the second column => s^-1 m^-2; Now, 1st column=wavelengths (nm)

p_density=AM_15G[:,1]        #second column=power density w/m^2/s
l=AM_15G[:, 0]              #first column=wavelength(nm)

y=np.zeros([9409,41],dtype=float)
n,m=np.shape(FlashQE)       #n=row, m=column
#E and fi
for i in range(m-1):
    y[:,i]=FlashQE[:,i+1]   #now y=FlashQE with first column cxcluded=EQE(l)
n,m=np.shape(y)
for i in range(n):
    y[i,:]=y[i,:]*p_density*l/hc*1e-9         #y=EQE * Phi; phi=p_density/(hc/l)  ; 9409*41 matrix

#dl calculation
p,q=np.shape(AM_15G)          #p=no. of rows, q=no.of columns
dl=np.zeros(41, dtype=float)  
for i in range(p-1):
    dl[i+1]=l[i+1]-l[i]       #because we need dl[0]=0, now dl has 41 elements 

#multiplication with dl
dJs=np.zeros([9409,41],dtype=float)
for i in range(n):            # n=row of y : 9409
    dJs[i,:]=y[i,:]*dl         #first column of every row=0, because dl[0]=0

# Integration output
Js=np.zeros([9409,1],dtype=float)
for i in range(9409):
    Js[i,:]=np.sum(dJs[i,:])   # Integration(EQE*phi*dl)

#Js_xy[97, 97] from Js[0:9409]
Js_xy=np.zeros([97,97],dtype=float)
z=0
for k in range(97):
    Js_xy[k,:]=Js[z:z+97,0]     #selecting column 0; in fact Js has only one column
    z=z+97
Js_xy=e*Js_xy
plt.imshow(Js_xy, vmin = 330, vmax = 360, cmap = 'inferno')
cbar = plt.colorbar()
plt.title('Jsc_xy Map(A/m^2)')

