from setup import read_filters
import jax.numpy as jnp
import numpy as np
from jax import jit
import os

alf_home = os.environ.get('ALF_HOME')
jalf_home = os.environ.get('JALF_HOME')

def getmass(msto,imf1,imf2,imflo=0.08,imfup=2.35):
    m2=0.5 #division point between imf1, imf2
    m3=1.0 #division point between imf2, imfup
    imfhi=100.0
    bhlim=40.0
    nslim=8.5

    #Normalize weights so 1Msun formed at t=0
    imfnorm = (m2**(-imf1+2)-imflo**(-imf1+2))/(-imf1+2) + \
            m2**(-imf1+imf2)*(m3**(-imf2+2)-m2**(-imf2+2))/(-imf2+2) + \
            m2**(-imf1+imf2)*(imfhi**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)
    
    #stars still alive
    mass = (m2**(-imf1+2)-imflo**(-imf1+2))/(-imf1+2)
    if msto < m3:
        mass = mass + m2**(-imf1+imf2)*(msto**(-imf2+2)-m2**(-imf2+2))/(-imf2+2)
    else:
        mass = mass + m2**(-imf1+imf2)*(m3**(-imf2+2)-m2**(-imf2+2))/(-imf2+2) + \
                            m2**(-imf1+imf2)*(msto**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)
    mass = mass/imfnorm

    #BH remnants
    #40<M<imf_up leave behind a 0.5*M BH
    mass = mass + \
          0.5*m2**(-imf1+imf2)*(imfhi**(-imfup+2)-bhlim**(-imfup+2))/(-imfup+2)/imfnorm

    #WD remnants
    #M<8.5 leave behind 0.077*M+0.48 WD
    if msto < m3:
        mass = mass + \
             0.48*m2**(-imf1+imf2)*(nslim**(-imfup+1)-m3**(-imfup+1))/(-imfup+1)/imfnorm
        mass = mass + \
             0.48*m2**(-imf1+imf2)*(m3**(-imf2+1)-msto**(-imf2+1))/(-imf2+1)/imfnorm
        mass = mass + \
             0.077*m2**(-imf1+imf2)*(nslim**(-imfup+2)-m3**(-imfup+2))/(-imfup+2)/imfnorm
        mass = mass + \
             0.077*m2**(-imf1+imf2)*(m3**(-imf2+2)-msto**(-imf2+2))/(-imf2+2)/imfnorm
    else:
        mass = mass + \
             0.48*m2**(-imf1+imf2)*(nslim**(-imfup+1)-msto**(-imfup+1))/(-imfup+1)/imfnorm
        mass = mass + \
             0.077*m2**(-imf1+imf2)*(nslim**(-imfup+2)-msto**(-imfup+2))/(-imfup+2)/imfnorm

    return mass, imfnorm

def getm2l(lam,spec,logage,zh,imf1,imf2):
    #values taken from alf_vars.f90
    msto_t0=0.33250847
    msto_t1=-0.29560944
    msto_z0=0.95402521
    msto_z1=0.21944863
    msto_z2=0.070565820
    magsun = (4.64,4.52,5.14)
    
    msto = 10**(msto_t0+msto_t1*logage) * \
       ( msto_z0 + msto_z1*zh + msto_z2*zh**2 )


    aspec  = spec#we're going to normalize anyway, so I don't think this is necisary? *lsun/1E6/1E8/4/mypi/pc2cm**2

    mass,_ = getmass(msto,imf1,imf2)
    lam_filters, r_response, i_response, k_response = read_filters(alf_home+'infiles/')
    r_resp = np.interp(lam,lam_filters,r_response)
    i_resp = np.interp(lam,lam_filters,i_response)
    k_resp = np.interp(lam,lam_filters,k_response)

    m2l = []

    for i,filter_resp in enumerate([r_resp,i_resp,k_resp]):
          mag = np.sum(spec*filter_resp/lam)
          mag = -2.5*np.log10(mag) - 48.60
          m2l.append(mass/10**(2.0/5*(magsun[i]-mag)))
    return np.array(m2l)
            
          
    

