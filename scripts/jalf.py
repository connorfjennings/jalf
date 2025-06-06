import os, sys, numpy as np
import jax.numpy as jnp
from jax import random, lax, vmap, jit
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
import arviz as az
from copy import deepcopy

from interpolate import interpolate_nd_jax
from smoothing import batch_smooth_scan, velbroad, fast_smooth4_variable_sigma
import setup
from model import model

def jalf(filename, tag):
    #--------PARAMETERS TO EDIT---------#

    burn_in_length = 500 #numpyro calls this "warmup"
    samples_length = 1500

    ang_per_poly_degree = 100

    young_old_cutoff_age = 8.0#Gyr, defines priors

    #infiles
    ssp_type = 'VCJ_v9'
    chem_type='atlas'
    atlas_imf='krpa'
    weights_type = 'H2O_weights'
    #weights here work differently from how they do in alf: they are applied to the final velocity
    #shifted values, and don't move with the input spectra. To-deweight suspicious lines (i.e. water)

    #todo, set grange bassed on assumed max velocity dispersion, grange=50 should be good for sigma<500km/s
    grange=50

    progress_bar_bool = True #turn this off if running using slurm

    #todo, define the priors up here and reference later (priors are defined around line 220)

    use_multiple_cpus = False

    if use_multiple_cpus:
        num_cpus = min([os.cpu_count(),int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))])
        numpyro.set_host_device_count(num_cpus)
        print(f'Using {num_cpus} cpus')



    alf_home = os.environ.get('ALF_HOME')
    jalf_home = os.environ.get('JALF_HOME')

    indata_file = jalf_home+'indata/'+filename
    weights_file = jalf_home+'infiles/'+weights_type+'.dat'

    #setup the model
    mo = model(indata_file,
               ssp_type = ssp_type,chem_type=chem_type,atlas_imf=atlas_imf,
               ang_per_poly_degree = ang_per_poly_degree,grange=grange,weights_file=weights_file)

    #get data from model
    params = (jnp.log10(8.0),0.0,1.3,2.3,0.0,100.0,\
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
                -6.0,10.0,-6.0,\
                jnp.log10(2.0),-6.0,\
                0.0,100.0,-6.0,-6.0,-6.0,-6.0,-6.0,-6.0)

    wl_d_region, flux_d_region, dflux_d_region, flux_m_region, flux_mn_region = mo.model_flux_regions(params)
    

    #-------estimate radial velocity and dispersion-------#
    print('Getting initial velocity estimates...')
    def vel_fit():
        #fit just velz and sigma
        velz = numpyro.sample('velz', dist.Normal(0.0,5))
        velz = velz * 100
        sigma = numpyro.sample('sigma', dist.Uniform(0.2,5))
        sigma = sigma * 100
        df = numpyro.sample("df", dist.Exponential(1.0))  

        params = (jnp.log10(10.0),0.0,1.3,2.3,velz,sigma,\
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
                -6.0,10.0,-6.0,\
                jnp.log10(2.0),-6.0,\
                0.0,100.0,-6.0,-6.0,-6.0,-6.0,-6.0,-6.0)
        _, _, dflux_d_region, flux_m_region, flux_mn_region = mo.model_flux_regions(params)
        for i in range(mo.n_regions):
            numpyro.sample(mo.region_name_list[i],dist.StudentT(df,flux_mn_region[i],dflux_d_region[i]),obs=flux_d_region[i])

    rng_key = random.PRNGKey(42)
    kernel = NUTS(vel_fit)
    mcmc = MCMC(
        kernel,
        num_warmup=500,
        num_samples=500,
        progress_bar = progress_bar_bool,
    )
    mcmc.run(rng_key)
    mcmc.print_summary()
    vel_posterior_samples = mcmc.get_samples()
    velz_mean_est = jnp.mean(vel_posterior_samples['velz'])
    velz_std_est = jnp.std(vel_posterior_samples['velz'])
    sigma_mean_est = jnp.mean(vel_posterior_samples['sigma'])
    sigma_std_est = jnp.std(vel_posterior_samples['sigma'])
    df_median = jnp.median(vel_posterior_samples['df'])

    #--------DEFINE PRIORS AND DO THE ACTUAL FIT---------#
    print('Starting main run...')
    def model_fit():
        #define priors here
        age = numpyro.sample("age", dist.Uniform(8,14))
        logage = jnp.log10(age)
        Z = numpyro.sample('Z', dist.Uniform(-1.8,0.3))
        imf1 = numpyro.sample('imf1', dist.Uniform(0.5,3.5))
        imf2 = imf1#numpyro.sample('imf2', dist.Uniform(0.5,3.5))
        velz = numpyro.sample('velz', dist.Normal(velz_mean_est,0.5))
        velz = velz * 100
        sigma = numpyro.sample('sigma', dist.TruncatedNormal(sigma_mean_est,1,low=0.1))
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

        teff = numpyro.sample('teff',dist.Uniform(-0.5,0.5))
        teff = teff*100

        loghot = numpyro.sample('loghot',dist.Uniform(-6.0,-2.0))
        hotteff = numpyro.sample('hotteff',dist.Uniform(8.0,30.0))
        logm7g = numpyro.sample('logm7g',dist.Uniform(-6.0,-2.0))

        age_young = numpyro.sample('age_young',dist.Uniform(1,8))
        logage_young = jnp.log10(age_young)
        log_frac_young = numpyro.sample('log_frac_young',dist.Uniform(-6,-1))

        velz2 = numpyro.sample('velz2',dist.Normal(0,2))
        velz2 = velz2 * 100

        sigma2 = numpyro.sample('sigma2', dist.TruncatedNormal(sigma_mean_est,1,low=0.1))
        sigma2 = sigma2 * 100
        logemline_h     = numpyro.sample('logemline_h', dist.Uniform(-6.0,-2.0))
        logemline_oiii  = numpyro.sample('logemline_oiii', dist.Uniform(-6.0,-2.0))
        logemline_oii   = numpyro.sample('logemline_oii', dist.Uniform(-6.0,-2.0))
        logemline_nii   = numpyro.sample('logemline_nii', dist.Uniform(-6.0,-2.0))
        logemline_ni    = numpyro.sample('logemline_ni', dist.Uniform(-6.0,-2.0))
        logemline_sii   = numpyro.sample('logemline_sii', dist.Uniform(-6.0,-2.0))


        df = numpyro.sample("df", dist.Exponential(1/df_median))  

        params = (logage,Z,imf1,imf2,velz,sigma,\
                    nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
                    loghot,hotteff,logm7g,\
                    logage_young,log_frac_young,\
                    velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii)
        _, _, dflux_d_region, flux_m_region, flux_mn_region = mo.model_flux_regions(params)
        for i in range(mo.n_regions):
            numpyro.sample(mo.region_name_list[i],dist.StudentT(df,flux_mn_region[i],dflux_d_region[i]),obs=flux_d_region[i])

    rng_key = random.PRNGKey(42)
    kernel = NUTS(model_fit)
    if use_multiple_cpus:
        mcmc = MCMC(
            kernel,
            num_warmup=burn_in_length,
            num_samples=samples_length,
            num_chains=num_cpus,
            chain_method="parallel",
            progress_bar=progress_bar_bool
        )
    else:
        mcmc = MCMC(
            kernel,
            num_warmup=burn_in_length,
            num_samples=samples_length,
            progress_bar=progress_bar_bool
        )

    #set initial parameters for a faster fit
    param_info, potential_fn, transform_fn, _ = initialize_model(rng_key, model_fit)
    mutable_constrained = deepcopy(param_info[0])
    mutable_constrained['velz'] = jnp.array(velz_mean_est)
    mutable_constrained['sigma'] = jnp.array(sigma_mean_est)
    init_unconstrained = transform_fn(mutable_constrained)

    mcmc.run(rng_key,init_params=init_unconstrained)
    
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    print('Run Finished!')

    if tag == '':
        output_name_base = filename
    else:
        output_name_base = filename+'_'+tag
    outdir = jalf_home+'results/'

    idata = az.from_numpyro(mcmc)
    idata.to_netcdf(outdir+output_name_base+".nc")

    np.savez(outdir+output_name_base+'.npz', **posterior_samples)
    print('Output saved, all done!')


if __name__ == "__main__":
    argv_l = sys.argv
    n_argv = len(argv_l)
    filename = argv_l[1]
    tag = argv_l[2] if n_argv >= 3 else ''
    #should implement some sort of multiprocessing
    jalf(filename, tag)