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

def get_alpha_posterior(idata, mo):
    posterior = idata.posterior

    # dims & coords just for ('chain','draw')
    dims2 = posterior['age'].dims              # usually ('chain','draw')
    coord_subset = {d: posterior.coords[d] for d in dims2}
    C = posterior.sizes[dims2[0]]
    D = posterior.sizes[dims2[1]]

    # Use scalar deterministics for kinematics of group 0
    param_list = [
        'age','Z','imf1','imf2','velz_0','sigma_0',
        'nah','cah','feh','ch','nh','ah','tih','mgh','sih','mnh','bah','nih','coh','euh','srh','kh','vh','cuh','teff',
        'loghot','hotteff','logm7g',
        'age_young','log_frac_young',
        'velz2','sigma2','logemline_h','logemline_oiii','logemline_oii','logemline_nii','logemline_ni','logemline_sii',
        'h3','h4'
    ]

    # defaults (2D: chain x draw)
    default_values = {
        'age': np.log10(10), 'Z': 0.0, 'imf1': 1.3, 'imf2': 2.3,
        'velz_0': 0.0, 'sigma_0': 300.0,
        'nah': 0.0,'cah': 0.0,'feh': 0.0,'ch': 0.0,'nh': 0.0,
        'ah': 0.0,'tih': 0.0,'mgh': 0.0,'sih': 0.0,'mnh': 0.0,
        'bah': 0.0,'nih': 0.0,'coh': 0.0,'euh': 0.0,'srh': 0.0,'kh': 0.0,'vh': 0.0,'cuh': 0.0,'teff': 0.0,
        'loghot': -10.0,'hotteff': 10.0,'logm7g': -10.0,
        'age_young': np.log10(2.0),'log_frac_young': -10.0,
        'velz2': 0.0,'sigma2': 300.0,
        'logemline_h': -10.0,'logemline_oiii': -10.0,'logemline_oii': -10.0,
        'logemline_nii': -10.0,'logemline_ni': -10.0,'logemline_sii': -10.0,
        'h3': 0.0,'h4': 0.0
    }

    sampled_params = []
    for pname in param_list:
        if pname in posterior:
            arr = posterior[pname].values  # (chain, draw)
            if pname == 'age':
                arr = np.log10(arr)
            elif pname in ('velz_0','sigma_0','teff'):
                arr = arr * 100.0
            sampled_params.append(arr)
        elif pname == 'imf2' and 'imf1' in posterior:
            # backwards compat if imf2 isn’t sampled
            sampled_params.append(posterior['imf1'].values)
        elif pname in ('velz_0','sigma_0'):
            # backwards compat if old runs only had 'velz'/'sigma'
            key_fallback = 'velz' if pname == 'velz_0' else 'sigma'
            if key_fallback in posterior:
                arr = posterior[key_fallback].values
                if key_fallback in ('velz','sigma'):
                    arr = arr * 100.0
                sampled_params.append(arr)
            else:
                sampled_params.append(np.full((C, D), default_values[pname]))
        else:
            sampled_params.append(np.full((C, D), default_values[pname]))

    params_array = np.stack(sampled_params, axis=-1)  # (chain, draw, P)
    flat_params = params_array.reshape(-1, params_array.shape[-1])

    # Compute alpha on the flattened samples
    alpha_vals = np.empty(flat_params.shape[0], dtype=float)
    for i, p in enumerate(flat_params):
        alpha_vals[i] = get_alpha(p, mo)
    alpha_vals = alpha_vals.reshape(C, D)

    alpha_da = xr.DataArray(alpha_vals, dims=dims2, coords=coord_subset)
    idata.posterior = idata.posterior.assign(alpha=alpha_da)
    return idata

            
          
    

