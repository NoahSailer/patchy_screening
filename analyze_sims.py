import os, sys, healpy as hp, numpy as np
from pixell import enmap, curvedsky
from astropy.table import Table
from est import lens_hardened_full_sky_est_LM, stacked_est

nstack = int(1e6)
cutout_size = 20. # arcmin
simulated_maps = ['tcmb','tcmb_lensed','tcmb_patchy','tcmb_patchy_lensed']
def get_tau_hat_fname(map_in,noise,fwhm,lmin,lmax,Lmin,Lmax):
    fn = f"data/full_sky_est/tau_hat_lens_hardened_{map_in}_noise={noise}muKarcmin_"
    fn+= f"fwhm={fwhm}arcmin_lmin={lmin}_lmax={lmax}_Lmin={Lmin}_Lmax={Lmax}.fits"
    return fn

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python analyze_sims.py <noise [muK-arcmin]> <fwhm [arcmin]> <lmin> <lmax> <Lmin> <Lmax>")
        sys.exit(1)
    noise,fwhm = np.array(sys.argv[1:3]).astype(float)
    lmin,lmax,Lmin,Lmax =np.array(sys.argv[3:]).astype(int)

    print('Working on full-sky tau estimates')
    mr = hp.read_map('data/sims/mask_hp.fits') # binary mask
    tau_true = enmap.read_map('data/sims/tau_true.fits')
    for map_in in simulated_maps:
        input_map_fname = f'data/sims/{map_in}_fwhm=1.4arcmin_hp.fits'
        tau_hat_fname = get_tau_hat_fname(map_in,noise,fwhm,lmin,lmax,Lmin,Lmax)
        print(f"Reconstructing tau from t = {input_map_fname}")
        if os.path.exists(tau_hat_fname): continue
        tau_hat_LM = lens_hardened_full_sky_est_LM(hp.read_map(input_map_fname)*mr,noise,fwhm,lmin,lmax,Lmin,Lmax,verbose=True)
        tau_hat = curvedsky.alm2map(tau_hat_LM,tau_true,copy=True)
        enmap.write_map(tau_hat_fname,tau_hat)
        print(f"saved enmap tau estimate as {tau_hat_fname}")
    print('Finished full-sky tau estimates')

    print('Stacking on mock galaxies')
    gcat = Table.read('data/from_boryana/Extended_LRG_zerr2.0_AbacusSummit_huge_c000_ph201_masked.fits')
    tau_hat_fns = [get_tau_hat_fname(map_in,noise,fwhm,lmin,lmax,Lmin,Lmax) for map_in in simulated_maps]
    stack_fns = [fn.replace("data/full_sky_est", "data/stacked_est").replace(".fits",".npy") for fn in tau_hat_fns]
    tau_hat_fns.append('data/sims/tau_true.fits')
    stack_fns.append('data/stacked_est/tau_true.npy')
    for tau_hat_fn,stack_fn in zip(tau_hat_fns,stack_fns):
        if os.path.exists(stack_fn): continue
        print(f'Working on {tau_hat_fn}')
        tau_hat = enmap.read_map(tau_hat_fn)
        stack = stacked_est(tau_hat,gcat,nstack=nstack,size=cutout_size,res=0.25,verbose=True)
        np.save(stack_fn,stack)
    print('Done!')