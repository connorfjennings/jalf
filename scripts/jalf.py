import os, sys, numpy as np
import jax.numpy as jnp
from jax import random, lax, vmap, jit
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import initialize_model
import arviz as az
from copy import deepcopy
import xarray as xr

import setup, m2l
from model import model
import priors

def jalf(filename, priorname, tag):
    #--------PARAMETERS TO EDIT---------#

    burn_in_length = 500 #numpyro calls this "warmup"
    samples_length = 1500

    ang_per_poly_degree = 100


    calc_alpha = True #adds alpha=(M/L)/(M/L)MW to the final paramater chain

    #infiles
    ssp_type = 'VCJ_v9'
    chem_type='atlas'
    atlas_imf='krpa'
    weights_type = 'default'#'H2O_weights'
    #weights here work differently from how they do in alf: they are applied to the final velocity
    #shifted values, and don't move with the input spectra. To-deweight suspicious lines (i.e. water)

    #define the prior function
    if priorname == 'NGC1277_center':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC1277center_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC1277_outer':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC1277outer_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC1407':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC1407_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC1600':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC1600_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC2695':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC2695_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC1407_KCWI':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC1407_KCWI_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC2695_KCWI':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC2695_KCWI_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC1407_2017':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC1407_2017_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC2695_2012':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC2695_2012_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC2695_2017':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC2695_2017_priors(velz_mean_est,sigma_mean_est)
    elif priorname == 'NGC1600_2017':
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.NGC1600_2017_priors(velz_mean_est,sigma_mean_est)
    else:
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.default_priors(velz_mean_est,sigma_mean_est)
        print('using default priors')

    #todo, set grange bassed on assumed max velocity dispersion, grange=50 should be good for sigma<500km/s
    grange=50

    progress_bar_bool = True #turn this off if running using slurm

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
                0.0,100.0,-6.0,-6.0,-6.0,-6.0,-6.0,-6.0,\
                0.0,0.0)

    wl_d_region, flux_d_region, dflux_d_region, flux_m_region, flux_mn_region = mo.model_flux_regions(params)
    

    #-------estimate radial velocity and dispersion-------#
    print('Getting initial velocity estimates...')
    def vel_fit():
        #fit just velz and sigma
        velz = numpyro.sample('velz', dist.Normal(0.0,5))
        velz = velz * 100
        sigma = numpyro.sample('sigma', dist.Uniform(0.2,8))
        sigma = sigma * 100
        #df = numpyro.sample("df", dist.Exponential(1.0))
        error_scale = numpyro.sample("error_scale",dist.LogNormal(jnp.log10(2.0),1.0))

        params = (jnp.log10(10.0),0.0,1.3,2.3,velz,sigma,\
                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,\
                -6.0,10.0,-6.0,\
                jnp.log10(2.0),-6.0,\
                0.0,100.0,-6.0,-6.0,-6.0,-6.0,-6.0,-6.0,\
                0.0,0.0)
        _, _, dflux_d_region, flux_m_region, flux_mn_region = mo.model_flux_regions(params)
        for i in range(mo.n_regions):
            numpyro.sample(mo.region_name_list[i],dist.StudentT(1,flux_mn_region[i],dflux_d_region[i]*error_scale),obs=flux_d_region[i])
            #numpyro.sample(mo.region_name_list[i],dist.Normal(flux_mn_region[i],dflux_d_region[i]*error_scale),obs=flux_d_region[i])

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

    #--------DEFINE PRIORS AND DO THE ACTUAL FIT---------#
    print('Starting main run...')
    def model_fit():
        
        params, error_scale = get_priors(velz_mean_est,sigma_mean_est)
        
        _, _, dflux_d_region, flux_m_region, flux_mn_region = mo.model_flux_regions(params)
        for i in range(mo.n_regions):
            #numpyro.sample(mo.region_name_list[i],dist.StudentT(df,flux_mn_region[i],dflux_d_region[i]),obs=flux_d_region[i])
            numpyro.sample(mo.region_name_list[i],dist.Normal(flux_mn_region[i],dflux_d_region[i]*error_scale),obs=flux_d_region[i])


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
        output_name_base = filename+'_'+priorname
    else:
        output_name_base = filename+'_'+priorname+'_'+tag
    outdir = jalf_home+'results/'

    idata = az.from_numpyro(mcmc)

    if calc_alpha:
        print('Calculating alpha values...')
        mo = model(indata_file,
               ssp_type = 'VCJ_v8',chem_type=chem_type,atlas_imf=atlas_imf,
               ang_per_poly_degree = ang_per_poly_degree,grange=grange,weights_file=weights_file)
        idata = m2l.get_alpha_posterior(idata,mo)


    idata.to_netcdf(outdir+output_name_base+".nc")

    np.savez(outdir+output_name_base+'.npz', **posterior_samples)
    print('Output saved, all done!')


if __name__ == "__main__":
    argv_l = sys.argv
    n_argv = len(argv_l)
    filename = argv_l[1]
    priorname = argv_l[2] if n_argv >=3 else 'default'
    tag = argv_l[3] if n_argv >= 4 else ''
    #should implement some sort of multiprocessing
    jalf(filename, priorname, tag)