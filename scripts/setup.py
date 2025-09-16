import numpy as np
import jax.numpy as jnp

from smoothing import batch_smooth_scan

def get_list_indicies(list,args):
    inds = []
    for arg in args:
        inds.append(list.index(arg))
    return np.array(inds)

def read_hotspec_models(infiles,wl_to_interp = None):
    z_strings  = ['-1.50','-1.00','-0.50','+0.00','+0.25']
    z_values = np.array([-1.5,-1.0,-0.5,0.0,0.25])
    hotteff_values = np.array([8.0,10.,12.,14.,16.,18.,20.,22.,24.,26.,28.,30.])

    hotspec_value_grid = np.meshgrid(z_values,hotteff_values,indexing='ij')

    filename = f'{infiles}hotteff_feh{z_strings[0]}.dat'
    f22 = np.loadtxt(filename,unpack=True)
    lam_hotspec = f22[0]

    if type(wl_to_interp) == type(None):
        flux_hotspec_grid = np.zeros(np.append(hotspec_value_grid[0].shape,lam_hotspec.shape))
    else:
        flux_hotspec_grid = np.zeros(np.append(hotspec_value_grid[0].shape,wl_to_interp.shape))

    for z, _ in enumerate(z_values):
        filename = f'{infiles}hotteff_feh{z_strings[z]}.dat'
        f22 = np.loadtxt(filename, unpack=True)

        column_index = 1
        for i, _ in enumerate(hotteff_values):
            if type(wl_to_interp) == type(None):
                flux_hotspec_grid[z,i] = f22[column_index]
            else:
                flux_hotspec_grid[z,i] = np.interp(wl_to_interp,lam_hotspec,f22[column_index])
            column_index+=1
    lam_hotspec = jnp.array(lam_hotspec)
    flux_hotspec_grid = jnp.array(flux_hotspec_grid)
    z_values = jnp.array(z_values)
    hotteff_values = jnp.array(hotteff_values)

    if type(wl_to_interp) != type(None):
        lam_hotspec = wl_to_interp
    
    return lam_hotspec, (z_values,hotteff_values), flux_hotspec_grid

def read_chem_models(infiles,chem_type='atlas',atlas_imf='krpa',wl_to_interp = None):
    chem_type = 'atlas'
    atlas_imf = 'krpa'

    z_strings  = ['m1.5','m1.0','m0.5','p0.0','p0.2']
    z_values = np.array([-1.5,-1.0,-0.5,0.0,0.2])
    t_strings  = ['01','03','05','09','13']
    t_values = np.array([1.0,3.0,5.0,9.0,13.0])
    logt_values = np.log10(t_values)

    #get wavelengths from chem files
    filename = f'{infiles}{chem_type}_ssp_t{t_strings[0]}_Z{z_strings[0]}.abund.{atlas_imf}.s100'
    f22 = np.loadtxt(filename,unpack=True)
    lam_chem = f22[0]

    chem_col_names = ['lam','solar','nap','nam','cap','cam','fep','fem',
                            'cp','cm','d1','np','nm','ap','tip','tim','mgp','mgm',
                            'sip','sim','teffp','teffm','crp','mnp','bap','bam',
                            'nip','cop','eup','srp','kp','vp','cup','nap6','nap9']
    #need to sort into (lam,logage,Zmet,p/n) shape (n_lam,n_logage,n_zmet,2)

    chem_names = ['solar','na','ca','fe','c','n','a','ti','mg','si','teff','cr','mn','ba','ni','co','eu','sr','k','v','cu']
    chem_dict = {}
    
    if type(wl_to_interp) == type(None):
        flux_solar_grid = np.zeros((len(logt_values),len(z_values),len(lam_chem)))
    else:
        flux_solar_grid = np.zeros((len(logt_values),len(z_values),len(wl_to_interp)))
    #solar_value_grid = np.meshgrid(logt_values,z_values,indexing='ij')
    for t,_ in enumerate(logt_values):
        for z,_ in enumerate(z_values):
            filename = f'{infiles}{chem_type}_ssp_t{t_strings[t]}_Z{z_strings[z]}.abund.{atlas_imf}.s100'
            f22 = np.loadtxt(filename,unpack=True)
            if type(wl_to_interp) == type(None):
                flux_solar_grid[t,z] = f22[1]
            else:
                flux_solar_grid[t,z] = np.interp(wl_to_interp,lam_chem,f22[1])
    solar_value_grids = (jnp.array(logt_values),jnp.array(z_values))
    chem_dict['solar'] = (solar_value_grids,jnp.array(flux_solar_grid))


    abund_ind = get_list_indicies(chem_col_names,['nam','solar','nap','nap6','nap9'])
    abund_values = np.array([-0.3,0.0,0.3,0.6,0.9])
    if type(wl_to_interp) == type(None):
        flux_chem_grid = np.zeros((len(logt_values),len(z_values),len(abund_values),len(lam_chem)))
    else:
        flux_chem_grid = np.zeros((len(logt_values),len(z_values),len(abund_values),len(wl_to_interp)))
    #chem_value_grid = np.meshgrid(logt_values,z_values,abund_values,indexing='ij')

    for t,_ in enumerate(logt_values):
        for z,_ in enumerate(z_values):
            filename = f'{infiles}{chem_type}_ssp_t{t_strings[t]}_Z{z_strings[z]}.abund.{atlas_imf}.s100'
            f22 = np.loadtxt(filename,unpack=True)
            for i,_ in enumerate(abund_ind):
                if type(wl_to_interp) == type(None):
                    flux_chem_grid[t,z,i] = f22[abund_ind[i]]
                else:
                    flux_chem_grid[t,z,i] = np.interp(wl_to_interp,lam_chem,f22[abund_ind[i]])
    chem_value_grids = (jnp.array(logt_values),jnp.array(z_values),jnp.array(abund_values))
    chem_dict['na'] = (chem_value_grids,jnp.array(flux_chem_grid))


    for chem_str in ['ca','fe','c','n','ti','mg','si','ba','teff']:
        abund_ind = get_list_indicies(chem_col_names,[chem_str+'m','solar',chem_str+'p'])
        if chem_str == 'c':
            abund_values = np.array([-0.15,0.0,0.15])
        elif chem_str == 'teff':
            abund_values = np.array([-50.0,0.0,50.0])
        else:
            abund_values = np.array([-0.3,0.0,0.3])
        if type(wl_to_interp) == type(None):
            flux_chem_grid = np.zeros((len(logt_values),len(z_values),len(abund_values),len(lam_chem)))
        else:
            flux_chem_grid = np.zeros((len(logt_values),len(z_values),len(abund_values),len(wl_to_interp)))
        #chem_value_grid = np.meshgrid(logt_values,z_values,abund_values,indexing='ij')

        for t,_ in enumerate(logt_values):
            for z,_ in enumerate(z_values):
                filename = f'{infiles}{chem_type}_ssp_t{t_strings[t]}_Z{z_strings[z]}.abund.{atlas_imf}.s100'
                f22 = np.loadtxt(filename,unpack=True)
                for i,_ in enumerate(abund_ind):
                    if type(wl_to_interp) == type(None):
                        flux_chem_grid[t,z,i] = f22[abund_ind[i]]
                    else:
                        flux_chem_grid[t,z,i] = np.interp(wl_to_interp,lam_chem,f22[abund_ind[i]])
        chem_value_grids = (jnp.array(logt_values),jnp.array(z_values),jnp.array(abund_values))
        chem_dict[chem_str] = (chem_value_grids,jnp.array(flux_chem_grid))

    for chem_str in ['a','cr','mn','ni','co','eu','sr','k','v','cu']:
        abund_ind = get_list_indicies(chem_col_names,['solar',chem_str+'p'])
        abund_values = np.array([0.0,0.3])
        if type(wl_to_interp) == type(None):
            flux_chem_grid = np.zeros((len(logt_values),len(z_values),len(abund_values),len(lam_chem)))
        else:
            flux_chem_grid = np.zeros((len(logt_values),len(z_values),len(abund_values),len(wl_to_interp)))
        #chem_value_grid = np.meshgrid(logt_values,z_values,abund_values,indexing='ij')

        for t,_ in enumerate(logt_values):
            for z,_ in enumerate(z_values):
                filename = f'{infiles}{chem_type}_ssp_t{t_strings[t]}_Z{z_strings[z]}.abund.{atlas_imf}.s100'
                f22 = np.loadtxt(filename,unpack=True)
                for i,_ in enumerate(abund_ind):
                    if type(wl_to_interp) == type(None):
                        flux_chem_grid[t,z,i] = f22[abund_ind[i]]
                    else:
                        flux_chem_grid[t,z,i] = np.interp(wl_to_interp,lam_chem,f22[abund_ind[i]])
        chem_value_grids = (jnp.array(logt_values),jnp.array(z_values),jnp.array(abund_values))
        chem_dict[chem_str] = (chem_value_grids,jnp.array(flux_chem_grid))

    if type(wl_to_interp) != type(None):
        lam_chem = wl_to_interp

    return lam_chem, chem_dict, chem_names


def read_ssp_models(infiles,ssp_type='VCJ_v9'):
    z_strings  = ['m1.5','m1.0','m0.5','p0.0','p0.2']
    z_values = np.array([-1.5,-1.0,-0.5,0.0,0.2])
    t_strings  = ['01.0','03.0','05.0','07.0','09.0','11.0','13.5']
    t_values = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])
    logt_values = np.log10(t_values)
    imf1_values = np.arange(0.5,3.7,0.2)
    imf2_values = np.copy(imf1_values)

    ssp_value_grid = np.meshgrid(logt_values,z_values,imf1_values,imf2_values,indexing='ij')
    ssp_value_flat = np.stack(ssp_value_grid, axis=-1).reshape(-1, 4)

    #get wavelengths from ssp files
    filename = f'{infiles}{ssp_type}_mcut0.08_t{t_strings[0]}_Z{z_strings[0]}.ssp.imf_varydoublex.s100'
    f22 = np.loadtxt(filename,unpack=True)
    lam_ssp = f22[0]

    #get flux from ssp file grid
    flux_ssp_grid = np.zeros(np.append(ssp_value_grid[0].shape,lam_ssp.shape))

    for (t,z), _ in np.ndenumerate(ssp_value_grid[0][:,:,0,0]):
        filename = f'{infiles}{ssp_type}_mcut0.08_t{t_strings[t]}_Z{z_strings[z]}.ssp.imf_varydoublex.s100'
        #f22 = np.array(pd.read_csv(filename, delim_whitespace=True, header=None, comment='#'))
        f22 = np.loadtxt(filename, unpack=True)

        column_index = 1

        for i1, imf1 in enumerate(imf1_values):
            for i2, imf2 in enumerate(imf2_values):
                flux_ssp_grid[t,z,i1,i2] = f22[column_index]
                column_index += 1

    flux_ssp_flat = flux_ssp_grid.reshape(-1, flux_ssp_grid.shape[-1])

    #make everything work with jax
    ssp_value_flat = jnp.array(ssp_value_flat)
    ssp_value_grid = jnp.array(ssp_value_grid)
    t_values = jnp.array(t_values)
    logt_values = jnp.array(logt_values)
    z_values = jnp.array(z_values)
    imf1_values = jnp.array(imf1_values)
    imf2_values = jnp.array(imf2_values)
    flux_ssp_flat = jnp.array(flux_ssp_flat)
    flux_ssp_grid = jnp.array(flux_ssp_grid)
    
    return lam_ssp, (logt_values,z_values,imf1_values,imf2_values), flux_ssp_grid

def smooth_ssp_models(data,lam_ssp,flux_ssp_grid):
    lam_data,_,_,_,ires_data = data
    ires_model = jnp.interp(lam_ssp,lam_data,ires_data)
    return batch_smooth_scan(lam_ssp,flux_ssp_grid,ires_model,batch_size=256)

def read_data(datfile):
    fit_regions = get_alf_header(datfile+'.dat')
    fit_regions[:,:2] *= 10000 #convert to angstroms
    data = np.loadtxt(datfile+'.dat',unpack=True)
    lam_data, flux_data, dflux_data, weights_data, ires_data = jnp.array(data)
    return fit_regions, lam_data, flux_data, dflux_data, weights_data, ires_data

def read_filters(infiles):
    filename = f'{infiles}filters.dat'
    lam_filters, r_response, i_response, k_response = np.loadtxt(filename,unpack=True)
    return lam_filters, r_response, i_response, k_response

def define_emlines():
    #define central wavelengths of emission lines (in vacuum)
    emlines = np.zeros(26)
    #these wavelengths come from NIST
    emlines[0]  = 4102.89  # Hd
    emlines[1]  = 4341.69  # Hy
    emlines[2]  = 4862.71  # Hb
    emlines[3]  = 4960.30  # [OIII]
    emlines[4]  = 5008.24  # [OIII]
    emlines[5]  = 5203.05  # [NI]
    emlines[6]  = 6549.86  # [NII]
    emlines[7]  = 6564.61  # Ha
    emlines[8]  = 6585.27  # [NII]
    emlines[9] = 6718.29  # [SII]
    emlines[10] = 6732.67  # [SII]
    emlines[11] = 3727.10  # [OII]
    emlines[12] = 3729.86  # [OII]
    emlines[13] = 3751.22  # Balmer
    emlines[14] = 3771.70  # Balmer
    emlines[15] = 3798.99  # Balmer
    emlines[16] = 3836.49  # Balmer
    emlines[17] = 3890.17  # Balmer
    emlines[18] = 3971.20  # Balmer
    #Paschen and Bracket come from Gemini
    emlines[19] = 1.87561 * 10000 #Paschen 4-3
    emlines[20] = 1.28216 * 10000 #Paschen 5-3
    emlines[21] = 1.09411 * 10000 #Paschen 6-3
    emlines[22] = 1.00521 * 10000 #Paschen 7-3
    emlines[23] = 2.16612 * 10000 #Bracket 7-4
    emlines[24] = 1.94509 * 10000 #Bracket 8-4
    emlines[25] = 1.81791 * 10000 #Bracket 9-4

    return emlines


def get_alf_header(infile):
    #from alfpy
    char = '#'
    header = []
    with open(infile, "r") as myfile:
        for line in myfile:
            if line.startswith(char):
                header.append([float(item) for item in line[1:].split()])
    return np.array(header)