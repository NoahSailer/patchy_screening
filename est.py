import json, healpy as hp, numpy as np
from scipy.interpolate import interp1d
from pixell import enmap, utils

def full_sky_est_LM(input_map,noise,fwhm,lmin,lmax,Lmin,Lmax,verbose=False):
    # SHOULD CHANGE THIS SO THAT FILTERS IS AN INPUT, OR FILTERS IS AUTOMATICALLY COMPUTED IF NOT FOUND
    """compute tau_hat_lm from an input_map"""
    safe_interp = lambda x,y,xp: interp1d(x,y,bounds_error=False,fill_value=0.)(xp)
    filters_fn = f"data/full_sky_est/filters_noise={noise}muKarcmin_fwhm={fwhm}arcmin_lmin={lmin}_lmax={lmax}.json"
    filters = json.load(open(filters_fn))
    for key in filters.keys(): filters[key] = np.array(filters[key])
    if verbose: print(f"Loaded filters from {filters_fn}")
    # interpolate 
    l_hp = np.arange(lmax+1)
    wf_hp = safe_interp(filters['l'],filters['wf_l'],l_hp) 
    ivar_hp = safe_interp(filters['l'],filters['ivar_l'],l_hp)
    laplace_wf_hp = safe_interp(filters['l'],-1.*filters['l']**2*filters['wf_l'],l_hp)
    laplace_ivar_hp = safe_interp(filters['l'],-1.*filters['l']**2*filters['ivar_l'],l_hp)
    L_hp = np.arange(Lmax+1)
    A_wf_ivar_hp = (L_hp>=Lmin)*(L_hp<=Lmax)*safe_interp(filters['L'],filters['A_wf_ivar_L'],L_hp)
    A_laplace_wf_ivar_hp = (L_hp>=Lmin)*(L_hp<=Lmax)*safe_interp(filters['L'],filters['A_laplace_wf_ivar_L'],L_hp)
    rtt_nonzero = np.where(filters['rtt_L']!=0)
    rtt_inv_hp = (L_hp>=Lmin)*(L_hp<=Lmax)*safe_interp(filters['L'][rtt_nonzero],1./filters['rtt_L'][rtt_nonzero],L_hp)
    # compute filtered maps
    nside = hp.get_nside(input_map)
    t_lm = hp.map2alm(input_map,lmax=lmax)
    if verbose: print("Computed t_lm")
    t_wf = hp.alm2map(hp.almxfl(t_lm,wf_hp),nside=nside,lmax=lmax)
    if verbose: print("Computed t_wf")
    t_ivar = hp.alm2map(hp.almxfl(t_lm,ivar_hp),nside=nside,lmax=lmax)
    if verbose: print("Computed t_ivar")
    t_laplace_wf = hp.alm2map(hp.almxfl(t_lm,laplace_wf_hp),nside=nside,lmax=lmax)
    if verbose: print("Computed t_laplace_wf")
    t_laplace_ivar = hp.alm2map(hp.almxfl(t_lm,laplace_ivar_hp),nside=nside,lmax=lmax)
    if verbose: print("Computed t_laplace_ivar")
    del input_map, t_lm
    wf_ivar_map = t_wf*t_ivar
    laplace_wf_ivar_map = t_wf*t_laplace_ivar-t_laplace_wf*t_ivar
    del t_wf, t_ivar, t_laplace_ivar, t_laplace_wf
    if verbose: print("Computed wf_ivar_map and laplace_wf_ivar_map")
	# compute minimum-variance and bias-hardened full sky estimates
    tau_hat_mv_LM = hp.almxfl(hp.map2alm(wf_ivar_map,lmax=Lmax),-1.*rtt_inv_hp)
    if verbose: print("Computed tau_hat_mv_LM")
    tau_hat_bh_LM = hp.almxfl(hp.map2alm(wf_ivar_map,lmax=Lmax),A_wf_ivar_hp)
    if verbose: print("Computed tau_hat_bh_LM (term 1)")
    tau_hat_bh_LM+= hp.almxfl(hp.map2alm(laplace_wf_ivar_map,lmax=Lmax),A_laplace_wf_ivar_hp)
    if verbose: print("Computed tau_hat_bh_LM (term 2)")
    return tau_hat_mv_LM, tau_hat_bh_LM 

def stacked_est(tau_hat,gcat,nstack=None,size=20.,res=0.25,verbose=False):
    """
    tau_hat: enmap
    gcat: galaxy catalog
    nstack: int
    size: float [arcmin]
    res: float [arcmin]
    """
    if nstack is None: nstack = len(gcat)
    if verbose: print(f"stacking on {nstack} out of {len(gcat)} galaxies")
    ii = np.random.choice(len(gcat), nstack, replace=False) if nstack<len(gcat) else range(len(gcat))
    cutout_example = make_cutout(tau_hat, (gcat['RA'][0], gcat['DEC'][0]), size, cutout_res=res)
    tau_stack = np.zeros_like(cutout_example)
    counter = 0.
    for i in ii:
        try:
            tau_est_loc = make_cutout(tau_hat,(gcat['RA'][i], gcat['DEC'][i]),size,cutout_res=res)
            tau_stack += tau_est_loc
            counter +=1.
        except:
            continue
    return tau_stack/counter

# helper functions taken from thumbstack repo

def moveaxis(a, o, n):
	if o < 0: o = o+a.ndim
	if n < 0: n = n+a.ndim
	if n <= o: return np.rollaxis(a, o, n)
	else: return np.rollaxis(a, o, n+1)

def rotmatrix(ang, raxis, axis=0):
	"""Construct a 3d rotation matrix representing a rotation of
	ang degrees around the specified rotation axis raxis, which can be "x", "y", "z"
	or 0, 1, 2. If ang is a scalar, the result will be [3,3]. Otherwise,
	it will be ang.shape + (3,3)."""
	ang  = np.asarray(ang)
	raxis = raxis.lower()
	c, s = np.cos(ang), np.sin(ang)
	R = np.zeros(ang.shape + (3,3))
	if   raxis == 0 or raxis == "x": R[...,0,0]=1;R[...,1,1]= c;R[...,1,2]=-s;R[...,2,1]= s;R[...,2,2]=c
	elif raxis == 1 or raxis == "y": R[...,0,0]=c;R[...,0,2]= s;R[...,1,1]= 1;R[...,2,0]=-s;R[...,2,2]=c
	elif raxis == 2 or raxis == "z": R[...,0,0]=c;R[...,0,1]=-s;R[...,1,0]= s;R[...,1,1]= c;R[...,2,2]=1
	else: raise ValueError("Rotation axis %s not recognized" % raxis)
	return moveaxis(R, 0, axis)

def ang2rect(angs, zenith=True, axis=0):
	"""Convert a set of angles [{phi,theta},...] to cartesian
	coordinates [{x,y,z},...]. If zenith is True (the default),
	the theta angle will be taken to go from 0 to pi, and measure
	the angle from the z axis. If zenith is False, then theta
	goes from -pi/2 to pi/2, and measures the angle up from the xy plane."""
	phi, theta = moveaxis(angs, axis, 0)
	ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
	if zenith: res = np.array([st*cp,st*sp,ct])
	else:      res = np.array([ct*cp,ct*sp,st])
	return moveaxis(res, 0, axis)

def rect2ang(rect, zenith=True, axis=0):
	"""The inverse of ang2rect."""
	x,y,z = moveaxis(rect, axis, 0)
	r     = (x**2+y**2)**0.5
	phi   = np.arctan2(y,x)
	if zenith: theta = np.arctan2(r,z)
	else:      theta = np.arctan2(z,r)
	return moveaxis(np.array([phi,theta]), 0, axis)

def euler_mat(euler_angles, kind="zyz"):
	"""Defines the rotation matrix M for a ABC euler rotation,
	such that M = A(alpha)B(beta)C(gamma), where euler_angles =
	[alpha,beta,gamma]. The default kind is ABC=ZYZ."""
	alpha, beta, gamma = euler_angles
	R1 = rotmatrix(gamma, kind[2])
	R2 = rotmatrix(beta,  kind[1])
	R3 = rotmatrix(alpha, kind[0])
	return np.einsum("...ij,...jk->...ik",np.einsum("...ij,...jk->...ik",R3,R2),R1)

def euler_rot(euler_angles, coords, kind="zyz"):
	coords = np.asarray(coords)
	co     = coords.reshape(2,-1)
	M      = euler_mat(euler_angles, kind)
	rect   = ang2rect(co, False)
	rect   = np.einsum("...ij,j...->i...",M,rect)
	co     = rect2ang(rect, False)
	return co.reshape(coords.shape)

def recenter(angs, center):
	"""recenter(angs, [from_ra, from_dec]) or recenter(angs, [from_ra, from_dec, to_ra, to_dec]).
	In the first form, performs a coordinate rotation such that a point at (form_ra, from_to)
	ends up at the north pole. In the second form, that point is instead put at (to_ra, to_dec)."""
	if len(center) == 4: ra0, dec0, ra1, dec1 = center
	elif len(center) == 2: ra0, dec0, ra1, dec1 = center[0], center[1], 0, np.pi/2
	return euler_rot([ra1,dec0-dec1,-ra0], angs, kind="zyz")
    
def make_cutout(input_map, location, cutout_size=10, cutout_res=0.25):
    """
    input_map: pixell object
    location: (RA,DEC) in degrees
    cutout_size: side length of square cutout (arcmin)
    """
    size = (2*np.floor((cutout_size/cutout_res-1)/2)+1)*cutout_res/60 #deg
    box  = 0.5*np.array([[-size,-size],[size,size]])*utils.degree
    shape,wcs = enmap.geometry(box,res=cutout_res*utils.arcmin, proj='cea')
    cutout = enmap.zeros(shape, wcs)
    opos = cutout.posmap()
    ipos = recenter(opos[::-1], [0, 0, location[0]*utils.degree, location[1]*utils.degree])[::-1]
    cutout[:, :] = input_map.at(ipos, order=1)
    return cutout