import numpy as np
from classy import Class
Tcmb = 2.7255 

def compute_fiducial_cl(delta_T_arcmin=10.0, theta_fwhm_arcmin=1.4, lmax=int(5e4)):
    """Compute the fiducial power spectra for a given noise level and beam FWHM."""
    params = {
        'h': 0.6736,
        'omega_b': 0.02237,
        'omega_cdm': 0.1200,
        'tau_reio': 0.0544,
        'A_s': 2.1e-9,
        'n_s': 0.9649,
        'lensing': 'yes',
        'output': 'tCl lCl',
        'l_max_scalars': lmax
    }
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(lmax)
    arcmin_to_rad = np.pi / (180.0 * 60.0)
    delta_T_rad = delta_T_arcmin * arcmin_to_rad * 1e-6  # K-radian
    theta_rad = theta_fwhm_arcmin * arcmin_to_rad  # radians
    N_ell = (delta_T_rad)**2*np.exp(cls['ell']*(cls['ell']+1)*theta_rad**2/(8.0*np.log(2)))
    l    = cls['ell'][2:] 
    ctt  = Tcmb**2*1e12*cls['tt'][2:]
    ctot = Tcmb**2*1e12*(cls['tt'][2:]+N_ell[2:])
    return l, ctt, ctot

# Linear responses
def fk(l,L,theta,fC0):
    """linear response to kappa"""
    Lml = (L**2+l**2-2*L*l*np.cos(theta))**0.5
    res = L*l*np.cos(theta)*fC0(l)
    res+= L*(L-l*np.cos(theta))*fC0(Lml)
    return np.nan_to_num(2*res/L**2)
def ft(l,L,theta,fC0):
    """linear response to patchy screening"""
    Lml = (L**2+l**2-2*L*l*np.cos(theta))**0.5
    res = fC0(l) + fC0(Lml)
    return -1.*res
def rxy(L,fx,fy,fC0,fCinvtot,lmin=1,lmax=10e3,ntheta=300,fwhm=1.4):
    """compute rxy = int fx fy B B / 2 Ctot Ctot"""
    theta = np.linspace(0,2*np.pi,ntheta)
    l     = np.linspace(lmin,lmax,int(lmax-lmin+1),endpoint=True)
    res   = np.zeros_like(theta)
    # Convert FWHM in arcmin to sigma in radians
    fwhm_rad = fwhm*np.pi/(180*60)
    sigma = fwhm_rad/(np.sqrt(16.*np.log(2))) 
    beam  = lambda l: np.exp(-0.5*(l*sigma)**2)
    for i in range(ntheta):
        Lml    = (L**2+l**2-2*L*l*np.cos(theta[i]))**0.5
        intg   = fx(l,L,theta[i],fC0)*fy(l,L,theta[i],fC0)*fCinvtot(l)*fCinvtot(Lml)
        intg  *= (Lml>=lmin)*(Lml<=lmax)*beam(l)*beam(Lml)*l/(2*np.pi)**2/2
        res[i] = np.trapz(intg,l)
    return np.trapz(res,theta)
# adding this function here since macs are stupid...
def compute_all_rxy(L, fC0, fCinvtot, lmin, lmax, ntheta, fwhm):
    return [
        rxy(L, ft, ft, fC0, fCinvtot, lmin=lmin, lmax=lmax, ntheta=ntheta, fwhm=fwhm),
        rxy(L, fk, fk, fC0, fCinvtot, lmin=lmin, lmax=lmax, ntheta=ntheta, fwhm=fwhm),
        rxy(L, ft, fk, fC0, fCinvtot, lmin=lmin, lmax=lmax, ntheta=ntheta, fwhm=fwhm),
    ]

if __name__ == "__main__":
    import os, sys, multiprocessing as mp, json
    from scipy.interpolate import interp1d
    from functools import partial
    ncpu = mp.cpu_count()
    if len(sys.argv) != 5:
        print("Usage: python prepare_filters.py <noise [muK-arcmin]> <fwhm [arcmin]> <lmin> <lmax>")
        sys.exit(1)
    noise= float(sys.argv[1])
    fwhm = float(sys.argv[2])
    lmin = int(sys.argv[3])
    lmax = int(sys.argv[4])
    ntheta = 500
    nL = 5000
    fname = f"data/full_sky_est/filters_noise={noise}muKarcmin_fwhm={fwhm}arcmin_lmin={lmin}_lmax={lmax}.json"
    if os.path.exists(fname):
        print(f'already computed {fname}, skipping...')
        sys.exit()
    l, ctt, ctot = compute_fiducial_cl(noise, fwhm, lmax)
    fC0 = interp1d(l,ctt,bounds_error=False,fill_value=0.)
    fCinvtot = interp1d(l[np.where(ctot>0)],1/ctot[np.where(ctot>0)],bounds_error=False,fill_value=0.)
    L = np.linspace(2,lmax,nL)
    worker = partial(compute_all_rxy, fC0=fC0, fCinvtot=fCinvtot, lmin=lmin, lmax=lmax, ntheta=ntheta, fwhm=fwhm)
    filters = {'l':l,'ctt_l':ctt,'ctot_l':ctot,'wf_l':(l>=lmin)*(l<=lmax)*fC0(l)*fCinvtot(l),
               'ivar_l':(l>=lmin)*(l<=lmax)*fCinvtot(l),'L':L}
    pool = mp.Pool(processes=(ncpu-1))
    filters['rtt_L'],filters['rkk_L'],filters['rtk_L'] = np.array(pool.map(worker,L)).T
    pool.close()
    pfac = filters['rtt_L'] - filters['rtk_L']**2/filters['rkk_L']
    filters['A_wf_ivar_L'] = (-1-filters['rtk_L']/filters['rkk_L'])/pfac
    filters['A_laplace_wf_ivar_L'] = -filters['rtk_L']/filters['rkk_L']/pfac/L**2
    for key in filters.keys(): filters[key] = filters[key].tolist()
    json.dump(filters, open(fname, "w"), indent=2)