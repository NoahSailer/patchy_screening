To create a `patchy` enviornment on mac use
```
chmod +x conda_setup_local.sh
./conda_setup_local.sh
conda activate patchy
```
After copying Boryana's sims to `data/from_boryana` run `python massage_sims.py` to prepare the sims for tau reconstruction. The massage script converts to `healpy` format, apodizes the mask, adds patchiness to the mocks, and convolves with a 1.4arcmin beam. See `data/sims/README.md` for the list of files that will be produced. Files ending with `_hp.fits` are in `healpy` format, and are otherwise in `pixell` format.


Here's an example of (trying to) reconstruct tau from these sims
```
noise=1.0
fwhm=1.4
lmin=300
lmax=16000
Lmin=300
Lmax=16000

python prepare_filters.py $noise $fwhm $lmin $lmax
python analyze_sims.py $noise $fwhm $lmin $lmax $Lmin $Lmax
```