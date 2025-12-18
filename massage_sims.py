from pixell import enmap, curvedsky
import numpy as np, healpy as hp
# For now I'm using healpy in the estimator, but the data products are in pixell format.
# This script apodizes the (implicit) mask, constructs healpy maps with/without patchy
# screening and lensing, and convolves them with a 1.4arcmin beam.   
fwhm_beam = 1.4*np.pi/180/60  # 1.4 arcmin in radians
fwhm_mask = 1*np.pi/180       # 1 degrees in radians (see below)
nside = 8192    
lmax = int(2*np.pi/fwhm_beam)
ogddir = 'data/from_boryana'
tcmb_fname = 'unlensed_map_8192.fits'
tcmb_lensed_fname = 'lensed_map_8192_ph201.fits'
tau_fname = 'map_tau_8192_ph201_MTNG.fits'
newddir = 'data/sims'

def pixell_to_healpix_hifi(m): return hp.alm2map(curvedsky.map2alm(m, lmax=lmax).astype(np.complex128), nside=nside, lmax=lmax)
def healpix_to_pixell_hifi(m,target_enmap): return curvedsky.alm2map(hp.map2alm(m,lmax=lmax).astype(np.complex128),target_enmap,copy=True)

# load boryana's sims, remove means, make patchy
print('loading data from boryana, subtracting means and making patchy...')
tcmb,tcmb_lensed,tau = [enmap.read_map(f'{ogddir}/{fname}') for fname in [tcmb_fname,tcmb_lensed_fname,tau_fname]]
tcmb -= np.mean(tcmb) ; tcmb_lensed -= np.mean(tcmb_lensed); tau -= np.mean(tau)
tcmb_patchy = tcmb*np.exp(-tau) ; tcmb_patchy_lensed = tcmb_lensed*np.exp(-tau)
enmap.write_map(f'{newddir}/tau_true.fits',tau)
print(f'converting tau_true to healpix, saving as {newddir}/tau_true_hp.fits')
hp.write_map(f'{newddir}/tau_true_hp.fits',pixell_to_healpix_hifi(tau))
del tau

# convolve mask with Gaussian and multiply by original binary mask, repeat (hacky apodization, should improve...)
print('converting mask to healpix and apodizing...')
mlm = curvedsky.map2alm(tcmb*0.+1., lmax=lmax).astype(np.complex128)  # raw (binary) mask alm
mr = hp.alm2map(mlm, nside=nside, lmax=lmax); mr[np.where(mr < 0.5)] = 0. ; mr[np.where(mr >= 0.5)] = 1. # raw (binary) mask
m = np.clip(hp.alm2map(hp.almxfl(mlm,hp.gauss_beam(fwhm_mask, lmax=lmax)), nside=nside, lmax=lmax)*mr,0.,1.) # apodized mask
# iterate, in the limit N->infinity the mask smoothly -> 0 near the edges
N=4
for i in range(N):
    mlm = hp.map2alm(m, lmax=lmax)
    m = np.clip(hp.alm2map(hp.almxfl(mlm,hp.gauss_beam(fwhm_mask, lmax=lmax)), nside=nside, lmax=lmax)*mr,0.,1.)
hp.write_map(f'{newddir}/mask_hp.fits', mr)
hp.write_map(f'{newddir}/mask_apodized_hp.fits', m)
del mlm,mr
print('saving apodized mask in pixell format')
enmap.write_map(f'{newddir}/mask_apodized.fits',healpix_to_pixell_hifi(m,tcmb))

# convert to healpix, multiply by apodized mask, convolve with beam
pixell_maps = [tcmb,tcmb_lensed,tcmb_patchy,tcmb_patchy_lensed]
names = ['tcmb','tcmb_lensed','tcmb_patchy','tcmb_patchy_lensed']
for pixell_map,name in zip(pixell_maps,names):
    print(f"converting {name} to healpix format, applying apodized mask and beam")
    healpix_map = pixell_to_healpix_hifi(pixell_map)
    healpix_map = hp.alm2map(hp.almxfl(hp.map2alm(healpix_map*m,lmax=lmax).astype(np.complex128), hp.gauss_beam(fwhm_beam, lmax=lmax)), nside=8192, lmax=lmax)
    hp.write_map(f'{newddir}/{name}_fwhm=1.4arcmin_hp.fits',healpix_map)
print('Done!')