import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import jit
import pandas as pd

NGC1407_KCWI_df = pd.read_csv('../infiles/NGC1407_KCWI_param_summary.csv')
NGC1407_2017_df = pd.read_csv('../infiles/NGC1407_2017_param_summary.csv')
NGC2695_KCWI_df = pd.read_csv('../infiles/NGC2695_KCWI_param_summary.csv')
NGC2695_2012_df = pd.read_csv('../infiles/NGC2695_2012_param_summary.csv')
NGC2695_2017_df = pd.read_csv('../infiles/NGC2695_2017_param_summary.csv')
NGC1600_2017_df = pd.read_csv('../infiles/NGC2695_2017_param_summary.csv')

pwm = 1.0 #multiplier on ABUNDANCES ONLY since you might want to relax this
error_scale_dist = dist.LogNormal(jnp.log(5.0),2.0)

def fixed_imf_priors(velz_mean,sigma_mean,df_name,n_groups=1):
    if df_name == 'fixed_imf_NGC1407_2017':
        df = NGC1407_2017_df
    elif df_name == 'fixed_imf_NGC1600_2017':
        df = NGC1600_2017_df
    elif df_name == 'fixed_imf_NGC2695_2017':
        df = NGC2695_2017_df
    else:
        print('Problem finding distribution function, check priors')
        df = NGC2695_2017_df
    age = numpyro.sample("age", dist.TruncatedNormal(df['age'][0],df['age'][1],low=10.0,high=14.0))
    logage = jnp.log10(age)
    Z = numpyro.sample('Z', dist.Uniform(-1.8,0.3))
    imf1 = numpyro.sample("imf1", dist.TruncatedNormal(df['imf1'][0],df['imf1'][1]*pwm,low=0.9,high=3.5))
    imf2 = numpyro.sample("imf2", dist.TruncatedNormal(df['imf2'][0],df['imf2'][1]*pwm,low=0.9,high=3.5))
    velz = numpyro.sample('velz', dist.Normal(velz_mean,0.5))
    velz = velz * 100
    sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean,0.5,low=0.1,high=6.0))
    sigma = sigma * 100

    nah = numpyro.sample('nah',dist.Uniform(-0.3,1))
    cah = numpyro.sample('cah',dist.Uniform(-0.3,0.5))
    feh = numpyro.sample('feh',dist.Uniform(-0.3,0.5))

    ch = numpyro.sample('ch',dist.Uniform(-0.3,0.5))
    nh = numpyro.sample('nh',dist.Uniform(-0.3,1))
    ah = numpyro.sample('ah',dist.Uniform(-0.3,0.5))
    tih = numpyro.sample('tih',dist.Uniform(-0.3,0.5))
    mgh = numpyro.sample('mgh',dist.Uniform(-0.3,0.5))
    sih = numpyro.sample('sih',dist.Uniform(-0.3,0.5))
    mnh = numpyro.sample('mnh',dist.Uniform(-0.3,0.5))
    bah = numpyro.sample('bah',dist.Uniform(-0.6,0.5))
    nih = numpyro.sample('nih',dist.Uniform(-0.3,0.5))
    coh = numpyro.sample('coh',dist.Uniform(-0.3,0.5))
    euh = numpyro.sample('euh',dist.Uniform(-0.6,0.5))
    srh = numpyro.sample('srh',dist.Uniform(-0.3,0.5))
    kh = numpyro.sample('kh',dist.Uniform(-0.3,0.5))
    vh = numpyro.sample('vh',dist.Uniform(-0.3,0.5))
    cuh = numpyro.sample('cuh',dist.Uniform(-0.3,0.5))

    teff = 0#numpyro.sample('teff',dist.Uniform(-0.5,0.5))
    teff = teff*100

    loghot = numpyro.sample('loghot',dist.Uniform(-10.0,-1.0))
    hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
    logm7g = numpyro.sample('logm7g',dist.Uniform(-10.0,-1.0))

    age_young = numpyro.sample('age_young',dist.Uniform(1,8))
    logage_young = jnp.log10(age_young)
    log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-10,-1))

    velz2 = numpyro.sample('velz2',dist.Normal(0,2))
    velz2 = velz2 * 100

    sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(sigma_mean,1,low=0.1))
    sigma2 = sigma2 * 100
    logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-10.0,-1.0))
    logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-10.0,-1.0))
    logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-10.0,-1.0))
    logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-10.0,-1.0))
    logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-10.0,-1.0))
    logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-10.0,-1.0))

    h3 = numpyro.sample('h3',dist.Normal(0.0,0.1))
    h4 = numpyro.sample('h4',dist.Normal(0.0,0.1))

    #df = numpyro.sample("df", dist.Exponential(1/df_median))
    error_scale = numpyro.sample("error_scale",error_scale_dist)

    params = (logage,Z,imf1,imf2,velz,sigma,\
                nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                loghot,hotteff,logm7g,\
                logage_young,log_frac_young,\
                velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
                h3,h4)
    return params, error_scale

def infile_priors(velz_mean,sigma_mean,df_name,n_groups=1):
    if df_name == 'fixed_imf_NGC1407_2017':
        df = NGC1407_2017_df
    elif df_name == 'fixed_imf_NGC1600_2017':
        df = NGC1600_2017_df
    elif df_name == 'fixed_imf_NGC2695_2017':
        df = NGC2695_2017_df
    elif df_name == 'fixed_imf_NGC2695_KCWI':
        df = NGC2695_KCWI_df
    elif df_name == 'fixed_imf_NGC1407_KCWI':
        df = NGC1407_KCWI_df
    else:
        print('Problem finding distribution function, check priors')
        df = NGC2695_2017_df
    age = numpyro.sample("age", dist.TruncatedNormal(df['age'][0],df['age'][1],low=10.0,high=14.0))
    logage = jnp.log10(age)
    Z = numpyro.sample('Z', dist.TruncatedNormal(df['Z'][0],df['Z'][1]*pwm,low=-1.8,high=0.3))
    imf1 = numpyro.sample('imf1', dist.Uniform(0.9,3.5))
    imf2 = numpyro.sample('imf2', dist.Uniform(0.9,3.5))
    #velz = numpyro.sample('velz', dist.Normal(velz_mean,0.5))
    #velz = velz * 100
    #sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean,0.5,low=0.1,high=6.0))
    #sigma = sigma * 100

    nah = numpyro.sample('nah',dist.TruncatedNormal(df['nah'][0],df['nah'][1]*pwm,low=-0.3,high=1.0))
    cah = numpyro.sample('cah',dist.TruncatedNormal(df['cah'][0],df['cah'][1]*pwm,low=-0.3,high=0.5))
    feh = numpyro.sample('feh',dist.TruncatedNormal(df['feh'][0],df['feh'][1]*pwm,low=-0.3,high=0.5))

    ch = numpyro.sample('ch',dist.TruncatedNormal(df['ch'][0],df['ch'][1]*pwm,low=-0.3,high=0.5))
    nh = numpyro.sample('nh',dist.TruncatedNormal(df['nh'][0],df['nh'][1]*pwm,low=-0.3,high=1.0))
    ah = numpyro.sample('ah',dist.TruncatedNormal(df['ah'][0],df['ah'][1]*pwm,low=-0.3,high=0.5))
    tih = numpyro.sample('tih',dist.TruncatedNormal(df['tih'][0],df['tih'][1]*pwm,low=-0.3,high=0.5))
    mgh = numpyro.sample('mgh',dist.TruncatedNormal(df['mgh'][0],df['mgh'][1]*pwm,low=-0.3,high=0.5))
    sih = numpyro.sample('sih',dist.TruncatedNormal(df['sih'][0],df['sih'][1]*pwm,low=-0.3,high=0.5))
    mnh = numpyro.sample('mnh',dist.TruncatedNormal(df['mnh'][0],df['mnh'][1]*pwm,low=-0.3,high=0.5))
    bah = numpyro.sample('bah',dist.TruncatedNormal(df['bah'][0],df['bah'][1]*pwm,low=-0.6,high=0.5))
    nih = numpyro.sample('nih',dist.TruncatedNormal(df['nih'][0],df['nih'][1]*pwm,low=-0.3,high=0.5))
    coh = numpyro.sample('coh',dist.TruncatedNormal(df['coh'][0],df['coh'][1]*pwm,low=-0.3,high=0.5))
    euh = numpyro.sample('euh',dist.TruncatedNormal(df['euh'][0],df['euh'][1]*pwm,low=-0.6,high=0.5))
    srh = numpyro.sample('srh',dist.TruncatedNormal(df['srh'][0],df['srh'][1]*pwm,low=-0.3,high=0.5))
    kh = numpyro.sample('kh',dist.TruncatedNormal(df['kh'][0],df['kh'][1]*pwm,low=-0.3,high=0.5))
    vh = numpyro.sample('vh',dist.TruncatedNormal(df['vh'][0],df['vh'][1]*pwm,low=-0.3,high=0.5))
    cuh = numpyro.sample('cuh',dist.TruncatedNormal(df['cuh'][0],df['cuh'][1]*pwm,low=-0.3,high=0.5))

    teff = 0#numpyro.sample('teff',dist.Normal(df['teff'][0],df['teff'][1]))
    teff = teff*100

    loghot = numpyro.sample('loghot',dist.Uniform(-10.0,-1.0))
    hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
    logm7g = numpyro.sample('logm7g',dist.Uniform(-10.0,-1.0))

    age_young = numpyro.sample('age_young',dist.Uniform(1,8))
    logage_young = jnp.log10(age_young)
    log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-10,-1))

    velz2 = numpyro.sample('velz2',dist.Normal(0,2))
    velz2 = velz2 * 100

    sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(sigma_mean,1,low=0.1))
    sigma2 = sigma2 * 100
    logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-10.0,-1.0))
    logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-10.0,-1.0))
    logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-10.0,-1.0))
    logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-10.0,-1.0))
    logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-10.0,-1.0))
    logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-10.0,-1.0))

    h3 = numpyro.sample('h3',dist.Normal(0.0,0.1))
    h4 = numpyro.sample('h4',dist.Normal(0.0,0.1))

    #df = numpyro.sample("df", dist.Exponential(1/df_median))
    #error_scale = numpyro.sample("error_scale",error_scale_dist)
    velz_g  = numpyro.sample('velz_g',  dist.Normal(velz_mean, 0.5).expand([n_groups])) * 100
    sigma_g = numpyro.sample('sigma_g', dist.TruncatedNormal(sigma_mean, 0.5, low=0.1, high=6.0)
                                         .expand([n_groups])) * 100
    error_scale_g = numpyro.sample('error_scale_g', error_scale_dist.expand([n_groups]))
    numpyro.deterministic('velz',  velz_g[0])
    numpyro.deterministic('sigma', sigma_g[0])
    numpyro.deterministic('error_scale', error_scale_g[0])

    params = (logage,Z,imf1,imf2,0.0,200.0,\
                nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                loghot,hotteff,logm7g,\
                logage_young,log_frac_young,\
                velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
                h3,h4,
                velz_g, sigma_g)
    return params, error_scale_g

def MWimf_priors(velz_mean,sigma_mean,n_groups=1):
    age = numpyro.sample("age", dist.TruncatedNormal(13.5,0.5,low=10.0,high=14.0))
    logage = jnp.log10(age)
    Z = numpyro.sample('Z', dist.Uniform(-1.8,0.3))
    imf1 = 1.3
    imf2 = 2.3
    #velz = numpyro.sample('velz', dist.Normal(velz_mean,0.5))
    #velz = velz * 100
    #sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean,0.5,low=0.1,high=6.0))
    #sigma = sigma * 100

    nah = numpyro.sample('nah',dist.Uniform(-0.3,1))
    cah = numpyro.sample('cah',dist.Uniform(-0.3,0.5))
    feh = numpyro.sample('feh',dist.Uniform(-0.3,0.5))

    ch = numpyro.sample('ch',dist.Uniform(-0.3,0.5))
    nh = numpyro.sample('nh',dist.Uniform(-0.3,1))
    ah = numpyro.sample('ah',dist.Uniform(-0.3,0.5))
    tih = numpyro.sample('tih',dist.Uniform(-0.3,0.5))
    mgh = numpyro.sample('mgh',dist.Uniform(-0.3,0.5))
    sih = numpyro.sample('sih',dist.Uniform(-0.3,0.5))
    mnh = numpyro.sample('mnh',dist.Uniform(-0.3,0.5))
    bah = numpyro.sample('bah',dist.Uniform(-0.6,0.5))
    nih = numpyro.sample('nih',dist.Uniform(-0.3,0.5))
    coh = numpyro.sample('coh',dist.Uniform(-0.3,0.5))
    euh = numpyro.sample('euh',dist.Uniform(-0.6,0.5))
    srh = numpyro.sample('srh',dist.Uniform(-0.3,0.5))
    kh = numpyro.sample('kh',dist.Uniform(-0.3,0.5))
    vh = numpyro.sample('vh',dist.Uniform(-0.3,0.5))
    cuh = numpyro.sample('cuh',dist.Uniform(-0.3,0.5))

    teff = 0#numpyro.sample('teff',dist.Uniform(-0.5,0.5))
    teff = teff*100

    loghot = numpyro.sample('loghot',dist.Uniform(-10.0,-1.0))
    hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
    logm7g = numpyro.sample('logm7g',dist.Uniform(-10.0,-1.0))

    age_young = numpyro.sample('age_young',dist.Uniform(1,8))
    logage_young = jnp.log10(age_young)
    log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-10,-1))

    velz2 = numpyro.sample('velz2',dist.Normal(0,2))
    velz2 = velz2 * 100

    sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(sigma_mean,1,low=0.1))
    sigma2 = sigma2 * 100
    logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-10.0,-1.0))
    logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-10.0,-1.0))
    logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-10.0,-1.0))
    logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-10.0,-1.0))
    logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-10.0,-1.0))
    logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-10.0,-1.0))

    h3 = numpyro.sample('h3',dist.Normal(0.0,0.1))
    h4 = numpyro.sample('h4',dist.Normal(0.0,0.1))

    #df = numpyro.sample("df", dist.Exponential(1/df_median))
    #error_scale = numpyro.sample("error_scale",error_scale_dist)
    velz_g  = numpyro.sample('velz_g',  dist.Normal(velz_mean, 0.5).expand([n_groups])) * 100
    sigma_g = numpyro.sample('sigma_g', dist.TruncatedNormal(sigma_mean, 0.5, low=0.1, high=6.0)
                                         .expand([n_groups])) * 100
    error_scale_g = numpyro.sample('error_scale_g', error_scale_dist.expand([n_groups]))

    numpyro.deterministic('velz',  velz_g[0])
    numpyro.deterministic('sigma', sigma_g[0])
    numpyro.deterministic('error_scale', error_scale_g[0])

    params = (logage,Z,imf1,imf2,0.0,200.0,\
                nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                loghot,hotteff,logm7g,\
                logage_young,log_frac_young,\
                velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
                h3,h4,
                velz_g, sigma_g)
    return params, error_scale_g


def default_priors(velz_mean,sigma_mean,n_groups=1):
    age = numpyro.sample("age", dist.TruncatedNormal(12.0,1.0,low=10.0,high=14.0))
    logage = jnp.log10(age)
    Z = numpyro.sample('Z', dist.Uniform(-1.8,0.3))
    imf1 = numpyro.sample('imf1', dist.Uniform(0.9,3.5))
    imf2 = numpyro.sample('imf2', dist.Uniform(0.9,3.5))
    #velz = numpyro.sample('velz', dist.Normal(velz_mean,0.5))
    #velz = velz * 100
    #sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean,0.5,low=0.1,high=6.0))
    #sigma = sigma * 100

    velz_g = numpyro.sample('velz_g',
                dist.Normal(velz_mean, 0.5).expand([n_groups])).astype(jnp.float32) * 100
    sigma_g = numpyro.sample('sigma_g',
                dist.TruncatedNormal(sigma_mean, 0.5, low=0.1, high=6.0).expand([n_groups])).astype(jnp.float32) * 100

    nah = numpyro.sample('nah',dist.Uniform(-0.3,1))
    cah = numpyro.sample('cah',dist.Uniform(-0.3,0.5))
    feh = numpyro.sample('feh',dist.Uniform(-0.3,0.5))

    ch = numpyro.sample('ch',dist.Uniform(-0.3,0.5))
    nh = numpyro.sample('nh',dist.Uniform(-0.3,1))
    ah = numpyro.sample('ah',dist.Uniform(-0.3,0.5))
    tih = numpyro.sample('tih',dist.Uniform(-0.3,0.5))
    mgh = numpyro.sample('mgh',dist.Uniform(-0.3,0.5))
    sih = numpyro.sample('sih',dist.Uniform(-0.3,0.5))
    mnh = numpyro.sample('mnh',dist.Uniform(-0.3,0.5))
    bah = numpyro.sample('bah',dist.Uniform(-0.6,0.5))
    nih = numpyro.sample('nih',dist.Uniform(-0.3,0.5))
    coh = numpyro.sample('coh',dist.Uniform(-0.3,0.5))
    euh = numpyro.sample('euh',dist.Uniform(-0.6,0.5))
    srh = numpyro.sample('srh',dist.Uniform(-0.3,0.5))
    kh = numpyro.sample('kh',dist.Uniform(-0.3,0.5))
    vh = numpyro.sample('vh',dist.Uniform(-0.3,0.5))
    cuh = numpyro.sample('cuh',dist.Uniform(-0.3,0.5))

    teff = 0#numpyro.sample('teff',dist.Uniform(-0.5,0.5))
    teff = teff*100

    loghot = numpyro.sample('loghot',dist.Uniform(-10.0,-1.0))
    hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
    logm7g = numpyro.sample('logm7g',dist.Uniform(-10.0,-1.0))

    age_young = numpyro.sample('age_young',dist.Uniform(1,8))
    logage_young = jnp.log10(age_young)
    log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-10,-1))

    velz2 = numpyro.sample('velz2',dist.Normal(0,2))
    velz2 = velz2 * 100

    sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(2,2,low=0.1))
    sigma2 = sigma2 * 100
    logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-10.0,-1.0))
    logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-10.0,-1.0))
    logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-10.0,-1.0))
    logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-10.0,-1.0))
    logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-10.0,-1.0))
    logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-10.0,-1.0))

    h3 = numpyro.sample('h3',dist.Normal(0.0,0.1))
    h4 = numpyro.sample('h4',dist.Normal(0.0,0.1))

    #error_scale = numpyro.sample("error_scale",error_scale_dist)
    error_scale_g = numpyro.sample("error_scale_g", error_scale_dist.expand([n_groups]))

    '''numpyro.deterministic('velz_0',  velz_g[0])
    numpyro.deterministic('sigma_0', sigma_g[0])
    numpyro.deterministic('error_scale_0', error_scale_g[0])'''

    params = (logage,Z,imf1,imf2,velz_g,sigma_g,\
                nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                loghot,hotteff,logm7g,\
                logage_young,log_frac_young,\
                velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
                h3,h4)
    return params, error_scale_g

def NGC1277center_priors(velz_mean,sigma_mean):
    age = numpyro.sample("age", dist.TruncatedNormal(11.5,0.2,low=10.0,high=14.0))
    logage = jnp.log10(age)
    Z = numpyro.sample('Z', dist.TruncatedNormal(0.3,0.02,low=-1.8,high=0.3))
    #should be mean of 0.41 but doesn't go that high
    imf1 = numpyro.sample('imf1', dist.Uniform(0.9,3.5))
    imf2 = numpyro.sample('imf2', dist.Uniform(0.9,3.5))
    velz = numpyro.sample('velz', dist.Normal(velz_mean,0.5))
    velz = velz * 100
    sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean,0.5,low=0.1,high=6.0))
    sigma = sigma * 100

    nah = numpyro.sample('nah',dist.Uniform(-0.3,1))
    cah = numpyro.sample('cah',dist.TruncatedNormal(0.35,0.15,low=-0.3,high=0.5))
    feh = numpyro.sample('feh',dist.Uniform(-0.3,0.5))

    ch = numpyro.sample('ch',dist.Uniform(-0.3,0.5))
    nh = numpyro.sample('nh',dist.Uniform(-0.3,1))
    ah = numpyro.sample('ah',dist.Uniform(-0.3,0.5))
    tih = numpyro.sample('tih',dist.TruncatedNormal(0.35,0.15,low=-0.3,high=0.5))
    mgh = numpyro.sample('mgh',dist.TruncatedNormal(0.35,0.15,low=-0.3,high=0.5))
    sih = numpyro.sample('sih',dist.TruncatedNormal(0.35,0.15,low=-0.3,high=0.5))
    mnh = numpyro.sample('mnh',dist.Uniform(-0.3,0.5))
    bah = numpyro.sample('bah',dist.Uniform(-0.6,0.5))
    nih = numpyro.sample('nih',dist.Uniform(-0.3,0.5))
    coh = numpyro.sample('coh',dist.Uniform(-0.3,0.5))
    euh = numpyro.sample('euh',dist.Uniform(-0.6,0.5))
    srh = numpyro.sample('srh',dist.Uniform(-0.3,0.5))
    kh = numpyro.sample('kh',dist.Uniform(-0.3,0.5))
    vh = numpyro.sample('vh',dist.Uniform(-0.3,0.5))
    cuh = numpyro.sample('cuh',dist.Uniform(-0.3,0.5))

    teff = 0#numpyro.sample('teff',dist.Uniform(-0.5,0.5))
    teff = teff*100

    loghot = numpyro.sample('loghot',dist.Uniform(-10.0,-1.0))
    hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
    logm7g = numpyro.sample('logm7g',dist.Uniform(-10.0,-1.0))

    age_young = numpyro.sample('age_young',dist.Uniform(1,8))
    logage_young = jnp.log10(age_young)
    log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-10,-1))

    velz2 = numpyro.sample('velz2',dist.Normal(0,2))
    velz2 = velz2 * 100

    sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(sigma_mean,1,low=0.1))
    sigma2 = sigma2 * 100
    logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-10.0,-1.0))
    logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-10.0,-1.0))
    logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-10.0,-1.0))
    logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-10.0,-1.0))
    logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-10.0,-1.0))
    logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-10.0,-1.0))

    h3 = numpyro.sample('h3',dist.Normal(0.0,0.1))
    h4 = numpyro.sample('h4',dist.Normal(0.0,0.1))

    #df = numpyro.sample("df", dist.Exponential(1/df_median))
    error_scale = numpyro.sample("error_scale",error_scale_dist)

    params = (logage,Z,imf1,imf2,velz,sigma,\
                nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                loghot,hotteff,logm7g,\
                logage_young,log_frac_young,\
                velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
                h3,h4)
    return params, error_scale

def NGC1277outer_priors(velz_mean,sigma_mean):
    age = numpyro.sample("age", dist.TruncatedNormal(13.5,1.0,low=10.0,high=14.0))
    logage = jnp.log10(age)
    Z = numpyro.sample('Z', dist.TruncatedNormal(0.2,0.1,low=-1.8,high=0.3))
    imf1 = numpyro.sample('imf1', dist.Uniform(0.9,3.5))
    imf2 = numpyro.sample('imf2', dist.Uniform(0.9,3.5))
    velz = numpyro.sample('velz', dist.Normal(velz_mean,0.5))
    velz = velz * 100
    sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean,0.5,low=0.1,high=6.0))
    sigma = sigma * 100

    nah = numpyro.sample('nah',dist.Uniform(-0.3,1))
    cah = numpyro.sample('cah',dist.TruncatedNormal(0.35,0.15,low=-0.3,high=0.5))
    feh = numpyro.sample('feh',dist.Uniform(-0.3,0.5))

    ch = numpyro.sample('ch',dist.Uniform(-0.3,0.5))
    nh = numpyro.sample('nh',dist.Uniform(-0.3,1))
    ah = numpyro.sample('ah',dist.Uniform(-0.3,0.5))
    tih = numpyro.sample('tih',dist.TruncatedNormal(0.35,0.15,low=-0.3,high=0.5))
    mgh = numpyro.sample('mgh',dist.TruncatedNormal(0.35,0.15,low=-0.3,high=0.5))
    sih = numpyro.sample('sih',dist.TruncatedNormal(0.35,0.15,low=-0.3,high=0.5))
    mnh = numpyro.sample('mnh',dist.Uniform(-0.3,0.5))
    bah = numpyro.sample('bah',dist.Uniform(-0.6,0.5))
    nih = numpyro.sample('nih',dist.Uniform(-0.3,0.5))
    coh = numpyro.sample('coh',dist.Uniform(-0.3,0.5))
    euh = numpyro.sample('euh',dist.Uniform(-0.6,0.5))
    srh = numpyro.sample('srh',dist.Uniform(-0.3,0.5))
    kh = numpyro.sample('kh',dist.Uniform(-0.3,0.5))
    vh = numpyro.sample('vh',dist.Uniform(-0.3,0.5))
    cuh = numpyro.sample('cuh',dist.Uniform(-0.3,0.5))

    teff = 0#numpyro.sample('teff',dist.Uniform(-0.5,0.5))
    teff = teff*100

    loghot = numpyro.sample('loghot',dist.Uniform(-10.0,-1.0))
    hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
    logm7g = numpyro.sample('logm7g',dist.Uniform(-10.0,-1.0))

    age_young = numpyro.sample('age_young',dist.Uniform(1,8))
    logage_young = jnp.log10(age_young)
    log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-10,-1))

    velz2 = numpyro.sample('velz2',dist.Normal(0,2))
    velz2 = velz2 * 100

    sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(sigma_mean,1,low=0.1))
    sigma2 = sigma2 * 100
    logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-10.0,-1.0))
    logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-10.0,-1.0))
    logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-10.0,-1.0))
    logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-10.0,-1.0))
    logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-10.0,-1.0))
    logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-10.0,-1.0))

    h3 = numpyro.sample('h3',dist.Normal(0.0,0.1))
    h4 = numpyro.sample('h4',dist.Normal(0.0,0.1))

    #df = numpyro.sample("df", dist.Exponential(1/df_median))
    error_scale = numpyro.sample("error_scale",error_scale_dist)

    params = (logage,Z,imf1,imf2,velz,sigma,\
                nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                loghot,hotteff,logm7g,\
                logage_young,log_frac_young,\
                velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
                h3,h4)
    return params, error_scale

def NGC1407_priors(velz_mean,sigma_mean):
    age = numpyro.sample("age", dist.Normal(12.5,0.5))
    logage = jnp.log10(age)
    Z = numpyro.sample('Z', dist.Normal(0.27,0.05))
    imf1 = numpyro.sample('imf1', dist.Uniform(0.9,3.5))
    imf2 = numpyro.sample('imf2', dist.Uniform(0.9,3.5))
    velz = numpyro.sample('velz', dist.Normal(velz_mean,0.5))
    velz = velz * 100
    sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean,0.5,low=0.1,high=6.0))
    sigma = sigma * 100

    nah = numpyro.sample('nah',dist.Uniform(-0.3,1))
    cah = numpyro.sample('cah',dist.Normal(0.35,0.05))
    feh = numpyro.sample('feh',dist.Uniform(-0.3,0.5))

    ch = numpyro.sample('ch',dist.Uniform(-0.3,0.5))
    nh = numpyro.sample('nh',dist.Uniform(-0.3,1))
    ah = numpyro.sample('ah',dist.Uniform(-0.3,0.5))
    tih = numpyro.sample('tih',dist.Normal(0.35,0.05))
    mgh = numpyro.sample('mgh',dist.Normal(0.35,0.05))
    sih = numpyro.sample('sih',dist.Normal(0.35,0.05))
    mnh = numpyro.sample('mnh',dist.Uniform(-0.3,0.5))
    bah = numpyro.sample('bah',dist.Uniform(-0.6,0.5))
    nih = numpyro.sample('nih',dist.Uniform(-0.3,0.5))
    coh = numpyro.sample('coh',dist.Uniform(-0.3,0.5))
    euh = numpyro.sample('euh',dist.Uniform(-0.6,0.5))
    srh = numpyro.sample('srh',dist.Uniform(-0.3,0.5))
    kh = numpyro.sample('kh',dist.Uniform(-0.3,0.5))
    vh = numpyro.sample('vh',dist.Uniform(-0.3,0.5))
    cuh = numpyro.sample('cuh',dist.Uniform(-0.3,0.5))

    teff = 0#numpyro.sample('teff',dist.Uniform(-0.5,0.5))
    teff = teff*100

    loghot = numpyro.sample('loghot',dist.Uniform(-10.0,-1.0))
    hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
    logm7g = numpyro.sample('logm7g',dist.Uniform(-10.0,-1.0))

    age_young = numpyro.sample('age_young',dist.Uniform(1,8))
    logage_young = jnp.log10(age_young)
    log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-10,-1))

    velz2 = numpyro.sample('velz2',dist.Normal(0,2))
    velz2 = velz2 * 100

    sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(sigma_mean,1,low=0.1))
    sigma2 = sigma2 * 100
    logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-10.0,-1.0))
    logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-10.0,-1.0))
    logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-10.0,-1.0))
    logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-10.0,-1.0))
    logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-10.0,-1.0))
    logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-10.0,-1.0))

    h3 = numpyro.sample('h3',dist.Normal(0.0,0.1))
    h4 = numpyro.sample('h4',dist.Normal(0.0,0.1))

    #df = numpyro.sample("df", dist.Exponential(1/df_median))
    error_scale = numpyro.sample("error_scale",error_scale_dist)

    params = (logage,Z,imf1,imf2,velz,sigma,\
                nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                loghot,hotteff,logm7g,\
                logage_young,log_frac_young,\
                velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
                h3,h4)
    return params, error_scale

def NGC1600_priors(velz_mean,sigma_mean):
    age = numpyro.sample("age", dist.TruncatedNormal(13.2,0.5,low=10,high=14.0))
    logage = jnp.log10(age)
    Z = numpyro.sample('Z', dist.Normal(0.10,0.02))
    imf1 = numpyro.sample('imf1', dist.Uniform(0.9,3.5))
    imf2 = numpyro.sample('imf2', dist.Uniform(0.9,3.5))
    velz = numpyro.sample('velz', dist.Normal(velz_mean,0.5))
    velz = velz * 100
    sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean,0.5,low=0.1,high=6.0))
    sigma = sigma * 100

    nah = numpyro.sample('nah',dist.Normal(0.46,0.2))
    cah = numpyro.sample('cah',dist.Normal(0.31,0.05))
    feh = numpyro.sample('feh',dist.Uniform(-0.3,0.5))

    ch = numpyro.sample('ch',dist.Uniform(-0.3,0.5))
    nh = numpyro.sample('nh',dist.Uniform(-0.3,1))
    ah = numpyro.sample('ah',dist.Uniform(-0.3,0.5))
    tih = numpyro.sample('tih',dist.Normal(0.31,0.05))
    mgh = numpyro.sample('mgh',dist.Normal(0.31,0.01))
    sih = numpyro.sample('sih',dist.Normal(0.31,0.05))
    mnh = numpyro.sample('mnh',dist.Uniform(-0.3,0.5))
    bah = numpyro.sample('bah',dist.Uniform(-0.6,0.5))
    nih = numpyro.sample('nih',dist.Uniform(-0.3,0.5))
    coh = numpyro.sample('coh',dist.Uniform(-0.3,0.5))
    euh = numpyro.sample('euh',dist.Uniform(-0.6,0.5))
    srh = numpyro.sample('srh',dist.Uniform(-0.3,0.5))
    kh = numpyro.sample('kh',dist.Uniform(-0.3,0.5))
    vh = numpyro.sample('vh',dist.Uniform(-0.3,0.5))
    cuh = numpyro.sample('cuh',dist.Uniform(-0.3,0.5))

    teff = 0#numpyro.sample('teff',dist.Uniform(-0.5,0.5))
    teff = teff*100

    loghot = numpyro.sample('loghot',dist.Uniform(-10.0,-1.0))
    hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
    logm7g = numpyro.sample('logm7g',dist.Uniform(-10.0,-1.0))

    age_young = numpyro.sample('age_young',dist.Uniform(1,8))
    logage_young = jnp.log10(age_young)
    log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-10,-1))

    velz2 = numpyro.sample('velz2',dist.Normal(0,2))
    velz2 = velz2 * 100

    sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(sigma_mean,1,low=0.1))
    sigma2 = sigma2 * 100
    logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-10.0,-1.0))
    logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-10.0,-1.0))
    logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-10.0,-1.0))
    logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-10.0,-1.0))
    logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-10.0,-1.0))
    logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-10.0,-1.0))

    h3 = numpyro.sample('h3',dist.Normal(0.0,0.1))
    h4 = numpyro.sample('h4',dist.Normal(0.0,0.1))

    #df = numpyro.sample("df", dist.Exponential(1/df_median))
    error_scale = numpyro.sample("error_scale",error_scale_dist)

    params = (logage,Z,imf1,imf2,velz,sigma,\
                nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                loghot,hotteff,logm7g,\
                logage_young,log_frac_young,\
                velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
                h3,h4)
    return params, error_scale


def NGC2695_priors(velz_mean,sigma_mean):
    age = numpyro.sample("age", dist.Normal(12.75,0.54))
    logage = jnp.log10(age)
    Z = numpyro.sample('Z', dist.Normal(-0.13,0.02))
    imf1 = numpyro.sample('imf1', dist.Uniform(0.9,3.5))
    imf2 = numpyro.sample('imf2', dist.Uniform(0.9,3.5))
    velz = numpyro.sample('velz', dist.Normal(velz_mean,0.5))
    velz = velz * 100
    sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean,0.5,low=0.1,high=6.0))
    sigma = sigma * 100

    nah = numpyro.sample('nah',dist.Uniform(-0.3,1))
    cah = numpyro.sample('cah',dist.Normal(0.34,0.1))
    feh = numpyro.sample('feh',dist.Uniform(-0.3,0.5))

    ch = numpyro.sample('ch',dist.Uniform(-0.3,0.5))
    nh = numpyro.sample('nh',dist.Uniform(-0.3,1))
    ah = numpyro.sample('ah',dist.Uniform(-0.3,0.5))
    tih = numpyro.sample('tih',dist.Normal(0.34,0.1))
    mgh = numpyro.sample('mgh',dist.Normal(0.34,0.02))
    sih = numpyro.sample('sih',dist.Normal(0.34,0.1))
    mnh = numpyro.sample('mnh',dist.Uniform(-0.3,0.5))
    bah = numpyro.sample('bah',dist.Uniform(-0.6,0.5))
    nih = numpyro.sample('nih',dist.Uniform(-0.3,0.5))
    coh = numpyro.sample('coh',dist.Uniform(-0.3,0.5))
    euh = numpyro.sample('euh',dist.Uniform(-0.6,0.5))
    srh = numpyro.sample('srh',dist.Uniform(-0.3,0.5))
    kh = numpyro.sample('kh',dist.Uniform(-0.3,0.5))
    vh = numpyro.sample('vh',dist.Uniform(-0.3,0.5))
    cuh = numpyro.sample('cuh',dist.Uniform(-0.3,0.5))

    teff = 0#numpyro.sample('teff',dist.Uniform(-0.5,0.5))
    teff = teff*100

    loghot = numpyro.sample('loghot',dist.Uniform(-10.0,-1.0))
    hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
    logm7g = numpyro.sample('logm7g',dist.Uniform(-10.0,-1.0))

    age_young = numpyro.sample('age_young',dist.Uniform(1,8))
    logage_young = jnp.log10(age_young)
    log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-10,-1))

    velz2 = numpyro.sample('velz2',dist.Normal(0,2))
    velz2 = velz2 * 100

    sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(sigma_mean,1,low=0.1))
    sigma2 = sigma2 * 100
    logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-10.0,-1.0))
    logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-10.0,-1.0))
    logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-10.0,-1.0))
    logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-10.0,-1.0))
    logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-10.0,-1.0))
    logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-10.0,-1.0))

    h3 = numpyro.sample('h3',dist.Normal(0.0,0.1))
    h4 = numpyro.sample('h4',dist.Normal(0.0,0.1))

    #df = numpyro.sample("df", dist.Exponential(1/df_median))
    error_scale = numpyro.sample("error_scale",error_scale_dist)

    params = (logage,Z,imf1,imf2,velz,sigma,\
                nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                loghot,hotteff,logm7g,\
                logage_young,log_frac_young,\
                velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
                h3,h4)
    return params, error_scale