import os, sys, numpy as np
import jax.numpy as jnp
from jax import random, lax, vmap, jit
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import os

from interpolate import interpolate_nd_jax
from smoothing import batch_smooth_scan, velbroad, fast_smooth4_variable_sigma, fast_smooth4_gausshermite
import setup
import m2l

class model:
    def __init__(self, indata_file, loud = True,
                  ssp_type = 'VCJ_v9',chem_type='atlas',atlas_imf='krpa',
                  ang_per_poly_degree = 100,grange=50,weights_file='NA',
                  fit_two_ages=True,fit_emlines=True,fit_h3h4=True,
                  ):
        #--------KNOW FIT OPTIONS----------#
        self.fit_two_ages=fit_two_ages
        self.fit_emlines =fit_emlines
        self.fit_h3h4 = fit_h3h4


        #--------SET UP DATA AND GRIDS----------#
        alf_home = os.environ.get('ALF_HOME')
        jalf_home = os.environ.get('JALF_HOME')

        self.grange = grange

        #read in data to fit
        if os.path.exists(indata_file+'.dat'):
            fit_regions, lam_data, flux_data, dflux_data, weights_data, ires_data = setup.read_data(indata_file)
            self.fit_regions = fit_regions
            self.lam_data = lam_data
            self.flux_data = flux_data
            self.dflux_data = dflux_data
            self.weights_data = weights_data
            self.ires_data = ires_data

            if loud: print('Indata loaded')
            indata_exists = True
            if np.max(ires_data) <= 1:
                if loud: print('No instrument resolution given')
                smooth_to_ires = False
            else:
                smooth_to_ires = True
        else:
            if loud: print('No indata detected')
            smooth_to_ires = False
            indata_exists = False

        if os.path.exists(weights_file):
            self.lam_weights, self.value_weights = np.loadtxt(weights_file,unpack=True)
            if loud: print('Weights loaded')
            self.use_weights = True
        else:
            if loud: print('Not using weights')
            self.use_weights = False

        #read in ssp grid
        print('Loading SSP grid')
        infiles = alf_home+'infiles/'
        lam_ssp, ssp_value_grid, flux_ssp_grid = setup.read_ssp_models(infiles,ssp_type=ssp_type)
        if (ssp_type != 'VCJ_v9') and (ssp_type != 'VCJ_v8'):
            #should do a real compatability check here, but this works for now
            if loud: print('chem and hotstar specs will interp to ssp when loaded')
            self.interp_wl = True
            wl_to_interp = lam_ssp
        else:
            self.interp_wl = False
            wl_to_interp = None

        if smooth_to_ires:
            if loud: print('Smoothing ssp grid to ires...')
            ires_ssp_model = jnp.interp(lam_ssp,lam_data,ires_data)
            flux_ssp_grid_smooth = batch_smooth_scan(lam_ssp,flux_ssp_grid,ires_ssp_model,batch_size=256)
            flux_ssp_grid_smooth.block_until_ready()
        else:
            if loud: print('Skipping ires smoothing')
            flux_ssp_grid_smooth = flux_ssp_grid
        
        if loud: print('Done!')

        #read in abundance files
        if loud: print('Loading abundance grid')
        lam_chem, chem_dict, chem_names = setup.read_chem_models(infiles,chem_type=chem_type,atlas_imf=atlas_imf,
                                                                 wl_to_interp=wl_to_interp)

        if smooth_to_ires:
            if loud: print('Smoothing abundance grid to ires...')
            ires_chem_model = jnp.interp(lam_chem,lam_data,ires_data)
            chem_dict_smooth = {}
            for cname in chem_names:
                #print('starting '+cname)
                chem_value_grid, flux_chem_grid = chem_dict[cname]
                
                flux_chem_grid_smooth = batch_smooth_scan(lam_chem,flux_chem_grid,ires_chem_model,batch_size=64)
                flux_chem_grid_smooth.block_until_ready()
                chem_dict_smooth[cname] = (chem_value_grid, flux_chem_grid_smooth)
        else:
            if loud: print('Skipping ires smoothing')
            chem_dict_smooth = chem_dict
        if loud: print('Done')

        if loud: print('Loading hot star grid')
        lam_hotspec, hotspec_value_grid, flux_hotspec_grid = setup.read_hotspec_models(infiles,
                                                                                       wl_to_interp=wl_to_interp)

        #normalize to 13Gyr at 1um
        ssp_1um_ind = np.argmin(np.abs(10000 - lam_ssp))
        ssp_1um_flux = interpolate_nd_jax((jnp.log10(13),0,1.3,2.3),ssp_value_grid,flux_ssp_grid,n_dims=4)[ssp_1um_ind]
        hotspec_1um_ind = np.argmin(np.abs(10000 - lam_hotspec))
        hotspec_1um_flux = flux_hotspec_grid[:,:,hotspec_1um_ind]
        flux_hotspec_grid = flux_hotspec_grid * ssp_1um_flux / hotspec_1um_flux[:,:,None]
        if smooth_to_ires:
            print('Smoothing hot star to ires...')
            ires_hotspec_model = jnp.interp(lam_hotspec,lam_data,ires_data)
            flux_hotspec_grid_smooth = batch_smooth_scan(lam_hotspec,flux_hotspec_grid,ires_hotspec_model)
        else:
            if loud: print('Skipping ires smoothing')
            flux_hotspec_grid_smooth = flux_hotspec_grid

        if loud: print('Loading M7III star')
        M7_filename = f'{infiles}M7III.spec.s100'
        lam_M7, flux_M7 = np.loadtxt(M7_filename,unpack=True)
        lam_M7 = jnp.array(lam_M7)
        flux_M7 = jnp.array(flux_M7)
        if smooth_to_ires:
            if loud: print('Smoothing hot star to ires...')
            ires_M7_model = jnp.interp(lam_M7,lam_data,ires_data)
            flux_M7_smooth = fast_smooth4_variable_sigma(lam_M7,flux_M7,ires_M7_model)
        else:
            if loud: print('Skipping ires smoothing')
            flux_M7_smooth = flux_M7
        if self.interp_wl:
            self.flux_M7 = jnp.interp(lam_ssp,lam_M7,flux_M7_smooth)
        else:
            self.flux_M7 = flux_M7_smooth

        self.chem_dict = chem_dict_smooth
        self.flux_ssp_grid = flux_ssp_grid_smooth
        self.log_flux_ssp_grid = jnp.log10(self.flux_ssp_grid)
        self.ssp_value_grid = ssp_value_grid
        self.flux_hotspec_grid = flux_hotspec_grid_smooth
        self.hotspec_value_grid = hotspec_value_grid
        self.lam_model = lam_ssp

        #get emission line centers
        self.emlines = jnp.array(setup.define_emlines())

        if indata_exists:
            #--------SETUP JIT FUNCTIONS------------#
            self.fit_regions_model_ind = []
            self.fit_regions_data_ind = []
            self.fit_regions_poly_deg = []
            for i in range(fit_regions.shape[0]):
                i_start_model = int(np.searchsorted(self.lam_model,fit_regions[i,0]))
                i_stop_model = int(np.searchsorted(self.lam_model,fit_regions[i,1]))
                self.fit_regions_model_ind.append(tuple((i_start_model,i_stop_model)))
                i_start_data = int(np.searchsorted(self.lam_data,fit_regions[i,0]))
                i_stop_data = int(np.searchsorted(self.lam_data,fit_regions[i,1]))
                self.fit_regions_data_ind.append(tuple((i_start_data,i_stop_data)))

                region_size = fit_regions[i,1] - fit_regions[i,0]
                poly_deg = int(np.floor(region_size/ang_per_poly_degree))
                self.fit_regions_poly_deg.append(poly_deg)
                if loud: print(f'Region {i} has size {region_size:.2f} ang, normalized with poly degree {poly_deg}')

            self.n_regions = len(self.fit_regions_model_ind)
            self.region_name_list = []
            for i in range(self.n_regions):
                self.region_name_list.append(f"Region {i+1}")

    def get_response(self,chem_name,logt,z,abund,flux_solar):
        value_grid, flux_grid = self.chem_dict[chem_name]
        flux_chem = interpolate_nd_jax((logt,z,abund),value_grid,flux_grid,n_dims=3)
        response = flux_chem/flux_solar
        return response
    get_response = jit(get_response,static_argnames=('self','chem_name'))

    def ssp_interp(self,age,Z,imf1,imf2):
        logssp = interpolate_nd_jax((age,Z,imf1,imf2),self.ssp_value_grid,self.log_flux_ssp_grid,n_dims=4)
        return 10**(logssp)
    ssp_interp = jit(ssp_interp,static_argnames=('self'))

    def hotspec_interp(self,Z,hotteff):
        hotflux = interpolate_nd_jax((Z,hotteff),self.hotspec_value_grid,self.flux_hotspec_grid,n_dims=2)
        return hotflux
    hotspec_interp = jit(hotspec_interp,static_argnames=('self'))

    def get_smoothed_region(self,wl,flux,sigma,wl_range_ind,h3,h4):
        # Clip indices to valid range
        i_start_pad = wl_range_ind[0]-self.grange#jnp.maximum(wl_range_ind[0] - grange, 0)
        i_stop_pad = wl_range_ind[1]+self.grange#jnp.minimum(wl_range_ind[1] + grange, wl.shape[0])

        size = i_stop_pad - i_start_pad
        wl_1 = lax.dynamic_slice(wl, (i_start_pad,), (size,))
        flux_1 = lax.dynamic_slice(flux, (i_start_pad,), (size,))

        flux_1 = velbroad(wl_1,flux_1,sigma,gausshermite=self.fit_h3h4,h3=h3,h4=h4)
        wl_2 = wl_1[self.grange:-self.grange]
        flux_2 = flux_1[self.grange:-self.grange]
        return wl_2, flux_2
    get_smoothed_region = jit(get_smoothed_region,static_argnames=('self','wl_range_ind'))

    def get_region(self,wl,flux,wl_range_ind):
        # Clip indices to valid range
        i_start = wl_range_ind[0]
        i_stop = wl_range_ind[1]

        size = i_stop - i_start
        wl_1 = lax.dynamic_slice(wl, (i_start,), (size,))
        flux_1 = lax.dynamic_slice(flux, (i_start,), (size,))
        return wl_1, flux_1
    get_region = jit(get_region,static_argnames=('self','wl_range_ind'))


    def model_flux(self,params):
        #does everything except the final velocity broadening and emission lines
        clight = 299792.46
        age,Z,imf1,imf2,velz,sigma,\
        nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
        loghot,hotteff,logm7g,\
        age_young, log_frac_young,\
        velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
        h3,h4 = params
        #keep in mind "age" here is actually log10(age)
        
        flux = self.ssp_interp(age,Z,imf1,imf2)
        wl = self.lam_model / (velz/clight + 1)

        if self.fit_two_ages:
            flux_young = self.ssp_interp(age_young,Z,imf1,imf2)
            fy = 10**log_frac_young
            flux = flux_young*fy + flux*(1-fy)

        flux_solar = interpolate_nd_jax((age,Z),self.chem_dict['solar'][0],self.chem_dict['solar'][1],n_dims=2)
        flux = flux * self.get_response('na',age,Z,nah,flux_solar)
        flux = flux * self.get_response('ca',age,Z,cah,flux_solar)
        flux = flux * self.get_response('fe',age,Z,feh,flux_solar)
        flux = flux * self.get_response('c',age,Z,ch,flux_solar)
        flux = flux * self.get_response('n',age,Z,nh,flux_solar)
        flux = flux * self.get_response('a',age,Z,ah,flux_solar)
        flux = flux * self.get_response('ti',age,Z,tih,flux_solar)
        flux = flux * self.get_response('mg',age,Z,mgh,flux_solar)
        flux = flux * self.get_response('si',age,Z,sih,flux_solar)
        flux = flux * self.get_response('mn',age,Z,mnh,flux_solar)
        flux = flux * self.get_response('ba',age,Z,bah,flux_solar)
        flux = flux * self.get_response('ni',age,Z,nih,flux_solar)
        flux = flux * self.get_response('co',age,Z,coh,flux_solar)
        flux = flux * self.get_response('eu',age,Z,euh,flux_solar)
        flux = flux * self.get_response('sr',age,Z,srh,flux_solar)
        flux = flux * self.get_response('k',age,Z,kh,flux_solar)
        flux = flux * self.get_response('v',age,Z,vh,flux_solar)
        flux = flux * self.get_response('cu',age,Z,cuh,flux_solar)

        #special case for teff, force use of 13gyr model
        flux = flux * self.get_response('teff',jnp.log10(13),Z,teff,flux_solar)

        #hotstars
        flux = flux + (10**loghot)*self.hotspec_interp(Z,hotteff)

        #M7 star
        fy = (10**logm7g)
        flux = (1-fy)*flux + fy*self.flux_M7 #not sure why this is treated differently than the hotstar, but same as in alf

        return wl, flux
    model_flux = jit(model_flux,static_argnames=('self'))

    def model_emission_lines(self,wl,params):
        clight = 299792.46
        age,Z,imf1,imf2,velz,sigma,\
        nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
        loghot,hotteff,logm7g,\
        age_young, log_frac_young,\
        velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
        h3,h4 = params

        base_h    = 10**logemline_h
        base_oiii = 10**logemline_oiii
        base_nii  = 10**logemline_nii
        base_sii  = 10**logemline_sii
        base_oii  = 10**logemline_oii
        base_ni   = 10**logemline_ni

        base_array = jnp.array([
                        base_h, base_h, base_h,         # Hy, Hd, Hb
                        base_oiii, base_oiii,           # [OIII], [OIII]
                        base_ni,                        # [NI]
                        base_nii, base_h, base_nii,     # [NII], Ha, [NII]
                        base_sii, base_sii,             # [SII], [SII]
                        base_oii, base_oii,             # [OII], [OII]
                        base_h, base_h, base_h,         # H12, H11, H10
                        base_h, base_h, base_h,         # H9, H8, H7
                        base_h, base_h, base_h, base_h, # Pa
                        base_h, base_h, base_h          # Br
                    ])
        #Pa and Br are from pyneb, all else from Nell Byler's Cloudy lookup table (from alf)
        scaling_factors = jnp.array([
                        1/11.21, 1/6.16, 1/2.87,        # Hy, Hd, Hb
                        1/3.0, 1.0,                     # [OIII]
                        1.0,                            # [NI]
                        1/2.95, 1.0, 1.0,               # [NII], Ha, [NII]
                        1.0, 0.77,                      # [SII]
                        1.0, 1.35,                      # [OII]
                        1/65.0, 1/55.0, 1/45.0,         # H12, H11, H10
                        1/35.0, 1/25.0, 1/18.0,         # H9, H8, H7
                        0.118269, 0.057014, 0.031589, 0.019369,  # Pa
                        0.009714, 0.006377, 0.004420    # Br
                    ])
        emnormal = base_array * scaling_factors

        ve = self.emlines / (velz2/clight + 1)
        lsig = ve*sigma2/clight
        emline_spec = emnormal[jnp.newaxis,:]*jnp.exp(-0.5 * (wl[:,jnp.newaxis]-ve[jnp.newaxis,:])**2 / (lsig[jnp.newaxis,:]**2))
        emline_spec = jnp.sum(emline_spec,axis=1)

        return emline_spec
    model_emission_lines = jit(model_emission_lines,static_argnames=('self'))

    def model_flux_total(self,params):
        clight = 299792.46
        age,Z,imf1,imf2,velz,sigma,\
        nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
        loghot,hotteff,logm7g,\
        age_young, log_frac_young,\
        velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
        h3,h4 = params
        
        wl, flux = self.model_flux(params)

        flux = velbroad(wl,flux,sigma,gausshermite=self.fit_h3h4,h3=h3,h4=h4)

        #emission lines
        #I am handling this slightly differently than alf. There, sigma2 is used to disperse the em lines BEFORE sigma is used to
        #broaden the entire spectra, but this seemed weird to me so I'm doing it after (so sigma2 is the actual dispersion of the em lines)
        if self.fit_emlines:
            emline_spec = self.model_emission_lines(wl,params)
            flux = flux + emline_spec

        return wl, flux

    def model_flux_regions(self,params):
        clight = 299792.46
        age,Z,imf1,imf2,velz,sigma,\
        nah,cah,feh,ch,nh,ah,tih,mgh,sih,mnh,bah,nih,coh,euh,srh,kh,vh,cuh,teff,\
        loghot,hotteff,logm7g,\
        age_young, log_frac_young,\
        velz2,sigma2,logemline_h,logemline_oiii,logemline_oii,logemline_nii,logemline_ni,logemline_sii,\
        h3,h4 = params
        #keep in mind "age" here is actually log10(age)
        
        wl, flux = self.model_flux(params)

        wl_d_region = []
        flux_d_region = []
        dflux_d_region = []
        flux_m_region = []
        flux_mn_region = []

        for i in range(self.n_regions):
            wl_m,flux_m = self.get_smoothed_region(wl,flux,sigma,wl_range_ind=self.fit_regions_model_ind[i],h3=h3,h4=h4)

            #emission lines
            if self.fit_emlines:
                emline_spec = self.model_emission_lines(wl_m,params)
                flux_m = flux_m + emline_spec

            wl_d,flux_d = self.get_region(self.lam_data,self.flux_data,wl_range_ind=self.fit_regions_data_ind[i])
            _, dflux_d = self.get_region(self.lam_data,self.dflux_data,wl_range_ind=self.fit_regions_data_ind[i])

            flux_m_interp = jnp.interp(wl_d,wl_m,flux_m)
            wl_d_zeroed = wl_d - wl_d[0]
            p = jnp.polyfit(wl_d_zeroed,flux_d/flux_m_interp,self.fit_regions_poly_deg[i])
            flux_m_norm = flux_m_interp * jnp.polyval(p,wl_d_zeroed)

            if self.use_weights:
                weights_interp = jnp.interp(wl_d,self.lam_weights,self.value_weights,left=1.0,right=1.0)
                dflux_d = dflux_d/weights_interp

            wl_d_region.append(wl_d)
            flux_d_region.append(flux_d)
            dflux_d_region.append(dflux_d)
            flux_m_region.append(flux_m_interp)
            flux_mn_region.append(flux_m_norm)

        return wl_d_region, flux_d_region, dflux_d_region, flux_m_region, flux_mn_region
    model_flux_regions = jit(model_flux_regions,static_argnames=('self'))