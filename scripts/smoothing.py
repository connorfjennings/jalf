import jax.numpy as jnp
from jax import jit, vmap, lax
import math, numpy as np
import jax


def interp_1d(xs, xp, fp):
    #not sure this in necisarry. ChatGPT thinks that this will work better with numpy differentiation
    #my tests say that jnp.interp works just fine, so using that instead
    """
    Vectorized, differentiable linear interpolation for JAX.
    Interpolates values at `xs` given knots `xp` and `fp`, all 1D arrays.
    """
    idxs = jnp.clip(jnp.searchsorted(xp, xs, side="right") - 1, 0, xp.shape[0] - 2)

    x0 = xp[idxs]
    x1 = xp[idxs + 1]
    y0 = fp[idxs]
    y1 = fp[idxs + 1]

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (xs - x0)

@jit
def fast_smooth1(lam, inspec,sigmal):
    #from alfpy, modified to work with jit
    #sigmal is an jarray with the same shape as lam in km/s

    wl_grid1, wl_grid2 = jnp.meshgrid(lam,lam,indexing='ij')

    vel = (wl_grid1/wl_grid2 - 1)*299792.46
    temr = vel/sigmal[jnp.newaxis,:]
    sq_temr = temr*temr
    func = 1/jnp.sqrt(2*jnp.pi)/sigmal[jnp.newaxis,:] * jnp.exp(-sq_temr/2)
    norm = jnp.trapezoid(y=func,x=vel,axis=1)
    func = func / norm
    outspec = jnp.trapezoid(y=func*inspec[jnp.newaxis,:],x=vel,axis=1)
    return outspec

@jit
def fast_smooth5_variable_sigma(lam,inspec,sigmal):
    #hopefully the most accurate version with a fixed kernal
    #not currently working
    clight = 299792.46
    grange = 50
    kernel_size = 2 * grange + 1

    n2 = lam.shape[0]
    i_range = jnp.arange(n2 - 2*grange)
    outspec = jnp.copy(inspec)

    def smooth_one(sspec, i):
        sig = lax.dynamic_slice(sigmal, (i+grange,), (1,))[0]
        x = lax.dynamic_slice(lam, (i,), (kernel_size,))
        x_center = x[grange]
        vel = (x/x_center - 1)*clight
        y = lax.dynamic_slice(inspec, (i,), (kernel_size,))
        
        psf = jnp.exp(-(vel/sig)**2 / 2) #*1/sigl
        psf = psf/jnp.sum(psf)
        weighted_y = y*psf + lax.dynamic_slice(sspec,(i,), (kernel_size,))
        sspec = lax.dynamic_update_slice(sspec, weighted_y, (i,))

        return sspec, None
    outspec_mid, _ = lax.scan(smooth_one,jnp.zeros(n2),i_range)
    outspec_mid = outspec_mid[0]
    outspec = outspec.at[grange:n2 - grange].set(outspec_mid.at[grange:n2 - grange])
    return outspec

@jit
def fast_smooth4_gausshermite(lam,inspec,sigma,h3,h4):
    clight = 299792.46
    grange = 50
    kernel_size = 2 * grange + 1

    n2 = lam.shape[0]
    i_range = jnp.arange(n2 - 2*grange)
    outspec = jnp.copy(inspec)

    def smooth_one(carry, i):
        x = lax.dynamic_slice(lam, (i,), (kernel_size,))
        y = lax.dynamic_slice(inspec, (i,), (kernel_size,))
        x_center = lax.dynamic_slice(lam, (i+grange,), (1,))[0]
        vel = (x_center/x - 1)*clight
        diff = vel/sigma

        H3_coef = (1/jnp.sqrt(6))*(2*diff**3 - 3*diff)
        H4_coef = (1/jnp.sqrt(24))*(4*diff**4 - 12*diff**2 +3)
        
        psf = jnp.exp(-(diff)**2 / 2) * (1 + h3*H3_coef + h4*H4_coef)
        psf = psf/jnp.sum(psf)
        smoothed_val = jnp.sum(y*psf)

        return carry, smoothed_val
    _, outspec_mid = lax.scan(smooth_one,None,i_range)
    outspec = outspec.at[grange:n2 - grange].set(outspec_mid)
    return outspec


@jit
def fast_smooth4_variable_sigma(lam,inspec,sigmal):
    #optimistic combination of fast_smooth1 and 2
    clight = 299792.46
    grange = 50
    kernel_size = 2 * grange + 1

    n2 = lam.shape[0]
    i_range = jnp.arange(n2 - 2*grange)
    outspec = jnp.copy(inspec)

    def smooth_one(carry, i):
        sigl = lax.dynamic_slice(sigmal, (i,), (kernel_size,))
        x = lax.dynamic_slice(lam, (i,), (kernel_size,))
        x_center = lax.dynamic_slice(lam, (i+grange,), (1,))[0]
        vel = (x_center/x - 1)*clight
        y = lax.dynamic_slice(inspec, (i,), (kernel_size,))
        
        psf = jnp.exp(-(vel/sigl)**2 / 2)
        psf = psf/jnp.sum(psf)
        smoothed_val = jnp.sum(y*psf)

        return carry, smoothed_val
    _, outspec_mid = lax.scan(smooth_one,None,i_range)
    outspec = outspec.at[grange:n2 - grange].set(outspec_mid)
    return outspec

@jit
def fast_smooth4(lam,inspec,sigma):
    #optimistic combination of fast_smooth1 and 2
    clight = 299792.46
    grange = 50
    kernel_size = 2 * grange + 1

    n2 = lam.shape[0]
    i_range = jnp.arange(n2 - 2*grange)
    outspec = jnp.copy(inspec)

    def smooth_one(carry, i):
        x = lax.dynamic_slice(lam, (i,), (kernel_size,))
        y = lax.dynamic_slice(inspec, (i,), (kernel_size,))
        x_center = lax.dynamic_slice(lam, (i+grange,), (1,))[0]
        vel = (x_center/x - 1)*clight
        
        psf = jnp.exp(-(vel/sigma)**2 / 2)
        psf = psf/jnp.sum(psf)
        smoothed_val = jnp.sum(y*psf)

        return carry, smoothed_val
    _, outspec_mid = lax.scan(smooth_one,None,i_range)
    outspec = outspec.at[grange:n2 - grange].set(outspec_mid)
    return outspec

@jit
def fast_smooth3_variable_sigma(lam,inspec,sigmal):
    #optimistic combination of fast_smooth1 and 2
    clight = 299792.46
    grange = 50
    kernel_size = 2 * grange + 1

    n2 = lam.shape[0]
    i_range = jnp.arange(n2 - 2*grange)
    outspec = jnp.copy(inspec)

    def smooth_one(carry, i):
        sig = lax.dynamic_slice(sigmal, (i+grange,), (1,))
        x = lax.dynamic_slice(lam, (i,), (kernel_size,))
        x_center = x[grange]
        vel = (x/x_center - 1)*clight
        y = lax.dynamic_slice(inspec, (i,), (kernel_size,))
        
        psf = jnp.exp(-(vel/sig)**2 / 2)
        psf = psf/jnp.sum(psf)
        smoothed_val = jnp.sum(y*psf)

        return carry, smoothed_val
    _, outspec_mid = lax.scan(smooth_one,None,i_range)
    outspec = outspec.at[grange:n2 - grange].set(outspec_mid)
    return outspec

@jit
def fast_smooth3(lam,inspec,sigma):
    #optimistic combination of fast_smooth1 and 2
    clight = 299792.46
    grange = 50
    kernel_size = 2 * grange + 1

    n2 = lam.shape[0]
    i_range = jnp.arange(n2 - 2*grange)
    outspec = jnp.copy(inspec)

    def smooth_one(carry, i):
        x = lax.dynamic_slice(lam, (i,), (kernel_size,))
        x_center = x[grange]
        vel = (x/x_center - 1)*clight
        y = lax.dynamic_slice(inspec, (i,), (kernel_size,))
        
        psf = jnp.exp(-(vel/sigma)**2 / 2)
        psf = psf/jnp.sum(psf)
        smoothed_val = jnp.sum(y*psf)

        return carry, smoothed_val
    _, outspec_mid = lax.scan(smooth_one,None,i_range)
    outspec = outspec.at[grange:n2 - grange].set(outspec_mid)
    return outspec
    

@jit
def fast_smooth2(lam, inspec, sigma):
    #from alfpy
    #accounts for edge effects, but uses interp to get around changes in the wavelength bins?
    clight = 299792.46 #scipy.constants.speed_of_light*100  # speed of light (km/s)
    grange=50 #this will cover a spread of about 500km/s
    psf_index = jnp.arange(2*grange+1)

    n2 = lam.shape[0]
    dlstep = (jnp.log(lam[-1])-jnp.log(lam[0]))/n2
    lnlam = jnp.arange(n2)*dlstep+jnp.log(lam[0])
    outspec = jnp.copy(inspec)
    
    psig = sigma*2.35482/clight/dlstep/2./(-2.0*jnp.log(0.5))**0.5 #! equivalent sigma for kernel

    tspec = jnp.interp(x=lnlam, xp=jnp.log(lam), fp= outspec)
    #tspec = interp_1d(lnlam, jnp.log(lam), outspec)

    nspec = jnp.copy(tspec)

    psf = jnp.exp(-((psf_index-grange)/psig)**2 / 2)
    psf = psf/jnp.sum(psf,axis=0)

    nspec_mid = sliding_conv_scan(psf, tspec, grange)
    nspec = nspec.at[grange:n2 - grange].set(nspec_mid)

    outspec = jnp.interp(x=lam, xp=jnp.exp(lnlam), fp=nspec)
    #outspec = interp_1d(lam, jnp.exp(lnlam), nspec)
            
    return outspec

def fast_smooth2_variable_sigma(lam, inspec, sigmal):
    clight = 299792.46
    grange = 50
    kernel_size = 2 * grange + 1
    psf_index = jnp.arange(kernel_size)

    n2 = lam.shape[0]
    dlstep = (jnp.log(lam[-1]) - jnp.log(lam[0])) / (n2-1)
    lnlam = jnp.arange(n2) * dlstep + jnp.log(lam[0])
    outspec = jnp.copy(inspec)

    tspec = jnp.interp(lnlam, jnp.log(lam), outspec)
    #tspec = interp_1d(lnlam, jnp.log(lam), outspec)
    nspec = jnp.copy(tspec)
    i_range = jnp.arange(n2 - 2*grange)
    psigl = sigmal * 2.35482 / clight / dlstep / 2. / jnp.sqrt(-2.0 * jnp.log(0.5))

    def smooth_one(carry, i):
        psig, tspec = carry

        psig = lax.dynamic_slice(psigl, (i,), (1,))

        psf = jnp.exp(-((psf_index - grange) / psig) ** 2 / 2)
        psf = psf / jnp.sum(psf)

        window = lax.dynamic_slice(tspec, (i,), (kernel_size,))
        smoothed_val = jnp.sum(window * psf)

        return carry, smoothed_val

    # initial carry with constants we need
    carry = (psigl, tspec)
    _, nspec_mid = lax.scan(smooth_one, carry, i_range)

    nspec = nspec.at[grange:n2 - grange].set(nspec_mid)
    outspec = jnp.interp(lam, jnp.exp(lnlam), nspec)
    #outspec = interp_1d(lam, jnp.exp(lnlam), nspec)
    return outspec


def sliding_conv_scan(psf, tspec, grange):
    kernel_size = 2 * grange + 1
    n2 = tspec.shape[0]
    n_out = n2 - 2 * grange

    def body(carry, i):
        window = lax.dynamic_slice(tspec, (i,), (kernel_size,))
        val = jnp.sum(window * psf)
        return carry, val

    _, out = lax.scan(body, None, jnp.arange(n_out))
    return out


def batch_smooth(lam, inspec, sigmal, batch_size=256):
    orig_shape = inspec.shape[:-1]
    n_lambda = inspec.shape[-1]
    flattened = inspec.reshape(-1, n_lambda)  # shape (N, n_lambda)
    N = flattened.shape[0]

    smoother = vmap(fast_smooth1, in_axes=(None, 0, None))

    print(N/batch_size)
    results = []
    for i in range(0, N, batch_size):
        print(i/batch_size)
        batch = flattened[i:i+batch_size]
        smoothed_batch = smoother(lam, batch, sigmal)  # shape (B, n_lambda)
        results.append(smoothed_batch)

    smoothed_flat = jnp.concatenate(results, axis=0)
    return smoothed_flat.reshape(*orig_shape, n_lambda)

def batch_smooth_scan(lam, inspec, sigmal, batch_size=64):
    orig_shape = inspec.shape[:-1]
    n_lambda = inspec.shape[-1]
    flattened = inspec.reshape(-1, n_lambda)
    N = flattened.shape[0]
    num_batches = (N + batch_size - 1) // batch_size

    #should really be using fast_smooth1 here for full accuracy, but this is almost the same and 1000 times faster
    def smoother(lam, inspec_batch, sigmal):
        return vmap(fast_smooth4_variable_sigma, in_axes=(None, 0, None))(lam, inspec_batch, sigmal)

    # Pre-pad the flattened array to handle any partial final batch
    pad_len = num_batches * batch_size - N
    padded = jnp.pad(flattened, ((0, pad_len), (0, 0)))

    def scan_fn(carry, i):
        start = i * batch_size
        batch = lax.dynamic_slice(padded, (start, 0), (batch_size, n_lambda))
        smoothed = smoother(lam, batch, sigmal)
        carry = lax.dynamic_update_slice(carry, smoothed, (start, 0))
        return carry, None

    output = jnp.zeros_like(padded)
    output, _ = lax.scan(scan_fn, output, jnp.arange(num_batches))

    return output[:N].reshape(*orig_shape, n_lambda)


def velbroad(lam,inspec,sigma,gausshermite=False,h3=0,h4=0):
    #for the typical case where sigma is the same everywhere (km/s)
    #I *think* fast_smooth1 is the most accurate, followed by fast_smooth4 which is significantly faster
    #fast_smooth2 is fastest, but does some log + interp stuff I'm not sure I understand
    if gausshermite:
        return fast_smooth4_gausshermite(lam,inspec,sigma,h3,h4)
    else:
        return fast_smooth4(lam,inspec,sigma)
velbroad = jit(velbroad,static_argnames=('gausshermite'))