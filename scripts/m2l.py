from setup import read_filters
import jax.numpy as jnp
import numpy as np
from jax import jit
import os
import copy
import xarray as xr

alf_home = os.environ.get('ALF_HOME')
jalf_home = os.environ.get('JALF_HOME')

def getmass(msto,imf1,imf2,imflo=0.08,imfup=2.3):
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
    
    #NS remnants
    mass = mass + \
            1.4* m2**(-imf1+imf2) * (bhlim**(-imfup+1) - nslim**(-imfup+1)) / (-imfup+1) / imfnorm


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
    lsun = 3.839E33
    clight = 2.9979E10
    mypi = np.pi
    pc2cm=3.08568E18
    
    msto = 10**(msto_t0+msto_t1*logage) * \
       ( msto_z0 + msto_z1*zh + msto_z2*zh**2 )


    aspec  = spec*lsun/1E6*lam**2/clight/1E8/4/mypi/pc2cm**2

    mass,_ = getmass(msto,imf1,imf2)
    lam_filters, r_response, i_response, k_response = read_filters(alf_home+'infiles/')
    r_resp = np.interp(lam,lam_filters,r_response)
    i_resp = np.interp(lam,lam_filters,i_response)
    k_resp = np.interp(lam,lam_filters,k_response)

    m2l = []

    for i,filter_resp in enumerate([r_resp,i_resp,k_resp]):
          mag = np.sum(aspec*filter_resp/lam)
          mag = -2.5*np.log10(mag) - 48.60
          m2l.append(mass/10**(2.0/5*(magsun[i]-mag)))
    return np.array(m2l)

def get_alpha(param_set,mo):

    lam, flux = mo.model_flux_total(param_set)
    m2l_fit = getm2l(lam,flux,param_set[0],param_set[1],param_set[2],param_set[3])

    kroupa_params = copy.copy(param_set)
    kroupa_params[2] = 1.3
    kroupa_params[3] = 2.3
    lam, flux_kroupa = mo.model_flux_total(kroupa_params)
    m2l_kroupa = getm2l(lam,flux_kroupa,kroupa_params[0],kroupa_params[1],kroupa_params[2],kroupa_params[3])
    alpha_array = m2l_fit/m2l_kroupa
    return alpha_array[0]

def get_alpha_posterior(idata,mo):
    param_list = ['age','Z','imf1','imf2','velz','sigma','nah','cah','feh','ch','nh',
              'ah','tih','mgh','sih','mnh','bah','nih','coh','euh','srh','kh','vh','cuh','teff',
              'loghot','hotteff','logm7g',
              'age_young','log_frac_young',
              'velz2','sigma2','logemline_h','logemline_oiii','logemline_oii','logemline_nii','logemline_ni','logemline_sii',
              'h3','h4']
    default_values = {
        'age':np.log10(10),'Z':0.0,'imf1':1.3,'imf2':2.3,'velz':0.0,'sigma':300.0,'nah':0.0,'cah':0.0,'feh':0.0,'ch':0.0,'nh':0.0,
        'ah':0.0,'tih':0.0,'mgh':0.0,'sih':0.0,'mnh':0.0,'bah':0.0,'nih':0.0,'coh':0.0,'euh':0.0,'srh':0.0,'kh':0.0,'vh':0.0,'cuh':0.0,'teff':0.0,
        'loghot':-10,'hotteff':10,'logm7g':-10,
        'age_young':np.log10(2),'log_frac_young':-10,
        'velz2':0.0,'sigma2':300,'logemline_h':-10,'logemline_oiii':-10,'logemline_oii':-10,'logemline_nii':-10,'logemline_ni':-10,'logemline_sii':-10,
        'h3':0.0,'h4':0.0
    }
    posterior_samples = idata.posterior

    chains, draws = posterior_samples[list(posterior_samples.keys())[0]].shape[:2]

    sampled_params = []

    for pname in param_list:
        if pname in posterior_samples:
            arr = posterior_samples[pname]
            
            if pname == 'age':
                arr = np.log10(arr)
            elif pname in ['velz', 'sigma', 'teff']:
                arr = arr * 100
            
            sampled_params.append(arr)
            
        elif (pname == 'imf2') & ('imf1' in posterior_samples):
            # If 'imf2' is missing, duplicate 'imf1'
            print('duplicating imf1=imf2')
            arr = copy.copy(posterior_samples['imf1'])
            sampled_params.append(arr)
        else:
            # Default values
            shape = (chains, draws)
            default = default_values[pname]
            arr = np.full(shape, default)
            sampled_params.append(arr)

    params_array = np.stack([np.array(p) for p in sampled_params], axis=-1)
    flat_params = params_array.reshape(-1, params_array.shape[-1])
    alpha_vals = np.array([get_alpha(p,mo) for p in flat_params])
    alpha_vals = alpha_vals.reshape((chains, draws))

    alpha_da = xr.DataArray(
        alpha_vals,
        coords=posterior_samples.coords,
        dims=posterior_samples['age'].dims
    )

    idata.posterior['alpha'] = alpha_da

    return idata
            
          
    

