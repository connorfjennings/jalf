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
import pickle

import setup, m2l
from model import model
import priors
import smoothing

def jalf(filename, priorname, tag):
    #--------PARAMETERS TO EDIT---------#

    burn_in_length = 500 #numpyro calls this "warmup"
    samples_length = 2000

    ang_per_poly_degree = 100
    ang_per_poly_degree_15000_mult = 1.5 #multiplier for bins greater than 15000 ang
    poly_degree_13300 = 1 #degree if bin stradles 13300 ang (H2O onset)

    StudentT_dof = 10 #make very large to approach normal errors

    calc_alpha = True #adds alpha=(M/L)/(M/L)MW to the final paramater chain
    make_spectra_summary = True #makes a .pkl file to help make plots fast
    target_sigma = 500 #km/s, one of the summary outputs will have spectra smoothed to this dispersion

    #infiles
    ssp_type = 'VCJ_v9'
    chem_type='atlas'
    atlas_imf='krpa'
    weights_type = 'none'#'H2O_weights'
    #weights here work differently from how they do in alf: they are applied to the final velocity
    #shifted values, and don't move with the input spectra. To-deweight suspicious lines (i.e. water)

    #define the prior function
    adjust_pname = False
    if priorname[-3:] == '.nc':
        get_priors = lambda velz_mean_est,sigma_mean_est, priorname: priors.prior_from_file(velz_mean_est,sigma_mean_est, priorname)
        adjust_pname = True
    elif priorname == 'NGC1277_center':
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
    elif priorname == 'MWimf':
        calc_alpha = False
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.MWimf_priors(velz_mean_est,sigma_mean_est)
    elif (priorname == 'fixed_imf_NGC1407_2017') or (priorname == 'fixed_imf_NGC1600_2017') or (priorname == 'fixed_imf_NGC2695_2017'):
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.fixed_imf_priors(velz_mean_est,sigma_mean_est,df_name=priorname)
    else:
        get_priors = lambda velz_mean_est,sigma_mean_est: priors.default_priors(velz_mean_est,sigma_mean_est)
        print('using default priors')

    #todo, set grange bassed on assumed max velocity dispersion, grange=50 should be good for sigma<500km/s
    grange=70

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
               ang_per_poly_degree = ang_per_poly_degree,grange=grange,weights_file=weights_file,
               ang_per_poly_degree_15000_mult=ang_per_poly_degree_15000_mult,
               poly_degree_13300=poly_degree_13300)

    #get data from model
    params = (jnp.log10(12.0),0.0,1.3,2.3,0.0,100.0,\
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
            numpyro.sample(mo.region_name_list[i],dist.StudentT(StudentT_dof,flux_mn_region[i],dflux_d_region[i]*error_scale),obs=flux_d_region[i])
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
            numpyro.sample(mo.region_name_list[i],dist.StudentT(10,flux_mn_region[i],dflux_d_region[i]*error_scale),obs=flux_d_region[i])
            #numpyro.sample(mo.region_name_list[i],dist.Normal(flux_mn_region[i],dflux_d_region[i]*error_scale),obs=flux_d_region[i])


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

    if adjust_pname:
        priorname = priorname[:-3]
    if tag == '':
        output_name_base = filename+'_'+priorname
    else:
        output_name_base = filename+'_'+priorname+'_'+tag
    outdir = jalf_home+'results/'

    idata = az.from_numpyro(mcmc)

    if make_spectra_summary:
        print('Pickling summary file...')
        clight = 299792.46
        #this is all stuff you typically want to do any time you visualize results
        param_list = ['age','Z','imf1','imf2','velz','sigma','nah','cah','feh','ch','nh',
              'ah','tih','mgh','sih','mnh','bah','nih','coh','euh','srh','kh','vh','cuh','teff',
              'loghot','hotteff','logm7g',
              'age_young','log_frac_young',
              'velz2','sigma2','logemline_h','logemline_oiii','logemline_oii','logemline_nii','logemline_ni','logemline_sii',
              'h3','h4']
        default_values = {
            'age':np.log10(13.5),'Z':0.0,'imf1':1.3,'imf2':2.3,'velz':0.0,'sigma':300.0,'nah':0.0,'cah':0.0,'feh':0.0,'ch':0.0,'nh':0.0,
            'ah':0.0,'tih':0.0,'mgh':0.0,'sih':0.0,'mnh':0.0,'bah':0.0,'nih':0.0,'coh':0.0,'euh':0.0,'srh':0.0,'kh':0.0,'vh':0.0,'cuh':0.0,'teff':0.0,
            'loghot':-10,'hotteff':10,'logm7g':-10,
            'age_young':np.log10(2),'log_frac_young':-10,
            'velz2':0.0,'sigma2':300,'logemline_h':-10,'logemline_oiii':-10,'logemline_oii':-10,'logemline_nii':-10,'logemline_ni':-10,'logemline_sii':-10,
            'h3':0.0,'h4':0.0
        }
        samples = mcmc.get_samples(group_by_chain=True)
        ef = mcmc.get_extra_fields(group_by_chain=True)
        pe = ef["potential_energy"]                     # shape [n_chains, n_draws]
        ind = jnp.argmin(pe)                            
        chain_ind = jnp.int32(ind // pe.shape[1])
        draw_ind  = jnp.int32(ind %  pe.shape[1])

        map_params = {k: v[chain_ind, draw_ind] for k, v in samples.items()}

        best_params_for_model = [] #fed into mo.model_flux_regions() etc to get spectra
        best_params_true = {} #the physical values of the parameters, with standard errors

        chains, draws = posterior_samples[list(posterior_samples.keys())[0]].shape[:2]

        for pname in param_list:
            if pname in posterior_samples:
                arr = posterior_samples[pname]
                map_v = map_params[pname]
                
                if pname == 'age':
                    arr = np.log10(arr)
                    map_v = np.log10(map_v)
                elif pname in ['velz', 'sigma', 'teff']:
                    arr = arr * 100
                    map_v = map_v * 100
            elif (pname == 'imf2') & ('imf1' in posterior_samples):
                arr = posterior_samples['imf1']
                map_v = map_params['imf1']
            else:
                # Default values
                shape = (chains, draws)
                default = default_values[pname]
                arr = np.full(shape, default)
                map_v = default
            best_params_for_model.append(map_v)

            if pname == 'sigma':
                arr = np.sqrt(arr**2+100**2) #THIS ONLY WORKS FOR THE VCJ MODELS!!!!
                map_v = np.sqrt(map_v**2+100**2)
            if (pname[-1]=='h') & (pname != 'logemline_h'):
                arr = arr + posterior_samples['Z']
                map_v = map_v + map_params['Z']
            param_err = np.std(arr)
            best_params_true[pname] = [map_v,param_err]

        region_spectra = {}

        wl_d_region, flux_d_region, dflux_d_region, flux_m_region, flux_mn_region = mo.model_flux_regions(best_params_for_model)
        wl_m_total, flux_m_total = mo.model_flux_total(best_params_for_model)
        region_spectra['wl_d_region']   = wl_d_region
        region_spectra['flux_d_region'] = flux_d_region
        region_spectra['dflux_d_region'] = dflux_d_region
        region_spectra['flux_m_region']  = flux_m_region
        region_spectra['flux_mn_region'] = flux_mn_region

        region_spectra['wl_m_total']   = wl_m_total
        region_spectra['flux_m_total'] = flux_m_total

        data = np.loadtxt(indata_file+'.dat',unpack=True)
        lam_data, flux_data, dflux_data, weights_data, ires_data = jnp.array(data)

        region_spectra['wl_d_total'] = lam_data
        region_spectra['flux_d_total'] = flux_data
        region_spectra['dflux_d_total'] = dflux_data

        #also shift and disperse the data
        velz_mean = float(np.mean(posterior_samples['velz']) * 100)
        sigma_mean = np.sqrt(float(np.mean(posterior_samples['sigma']) * 100)**2 + 100**2)
        wl = lam_data * (velz_mean/clight + 1)
        smooth_to = float(np.sqrt(target_sigma**2 - sigma_mean**2))
        flux = smoothing.fast_smooth4(wl,flux_data,smooth_to)
        dflux = smoothing.fast_smooth4(wl,dflux_data,smooth_to)
        region_spectra['wl_d_adjust'] = wl
        region_spectra['flux_d_adjust'] = flux
        region_spectra['dflux_d_adjust'] = dflux

        summary_dict = {'spec':region_spectra,
                        'params':best_params_true}
        with open(outdir+output_name_base+'.pkl', "wb") as f:
                pickle.dump(summary_dict, f)
            

    if calc_alpha:
        print('Calculating alpha values...')
        mo = model(indata_file,
               ssp_type = 'VCJ_v8',chem_type=chem_type,atlas_imf=atlas_imf,
               ang_per_poly_degree = ang_per_poly_degree,grange=grange,weights_file=weights_file,
               ang_per_poly_degree_15000_mult=ang_per_poly_degree_15000_mult)
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