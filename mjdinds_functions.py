import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd

# NOTE: THIS IS AN OUTDATED VERSION OF RUBIN_SIM -- USE PACKAGE IMPORTS FROM GUIDE IN RUNNOE GROUP GOOGLE DOC! Everything about the function should work, though

import rubin_sim.maf as maf
import rubin_scheduler.utils as rsUtils
from rubin_scheduler.data.data_sets import get_baseline

# <----------------- ESPECIALLY CONVENIENT FUNCTIONS: -----------------> 
# Especially useful functions are denoted with *****

def radec_to_mjd(ra, dec, bands=['g', 'r'], vet_lim=25.): #*****
    '''
    Converts a single (ra, dec) point into simulated MJDs for different bands.

    Parameters:
    -----------
    `ra`: Right-ascension coordinate
    `dec`: Declination coordinate
    `bands`: List. Band of desired simulated MJDs
    `vet_lim`: Minimum visit exposure time desired. Very few observations will have vet = 15, which will have much lower 5-sigma depths, so vet_lim=25. is desirable

    Future work:
    ------------
    Generate a dictionary of multiband MJDs for DIFFERENT (ra, dec) coordinates. If `ra` and `dec` are equal-length lists, the slicer will propagate these different poitns into the bundle_list and thus bundle_list[0].metric_values[i] will correspond to the simulated MJDs for the i'th (ra, dec) pair!
    
    '''
    
    baseline_file = get_baseline()
    name = os.path.basename(baseline_file).replace('.db', '')
    out_dir = 'temp'
    results_db = maf.db.ResultsDb(out_dir=out_dir)

    bundle_list = []
    metric = maf.metrics.PassMetric(cols=['filter', 'observationStartMJD', 'fiveSigmaDepth', 'visitExposureTime'])
    sql = ''
    slicer = maf.slicers.UserPointsSlicer(ra=ra, dec=dec)
    bundle_list.append(maf.MetricBundle(metric, slicer, sql, run_name=name))

    bd = maf.metricBundles.make_bundles_dict_from_list(bundle_list)
    bg = maf.metricBundles.MetricBundleGroup(bd, baseline_file, out_dir=out_dir, results_db=results_db)
    bg.run_all()

    data_slice = bundle_list[0].metric_values[0]
    data_slice = data_slice[data_slice['visitExposureTime'] > vet_lim] # Imposing visit exposure time minimum

    if type(bands) is list:
        dict_ = dict(zip(bands, [[[], []] for i in range(len(bands))])) # Make the second argument of zip() into three arrays if including vet

        for band in bands:
            filt = np.where(data_slice['filter'] == band)
            mjd = data_slice['observationStartMJD'][filt]
            sig5 = data_slice['fiveSigmaDepth'][filt]
            # vet = data_slice['visitExposureTime'][filt]

            dict_[band][0], dict_[band][1] = mjd, sig5#, vet
            pass
        q = dict_
        pass
    else:
        filt = np.where(data_slice['filter'] == bands)
        mjd = data_slice['observationStartMJD'][filt]
        sig5 = data_slice['fiveSigmaDepth'][filt]
        # vet = data_slice['visitExposureTime'][filt]
        
        q = [mjd, sig5]#, vet]
        pass
    
    return q


def MJD_to_inds_magfunc(mjd_sig5_dict, bands=['g', 'r'], gauss=True, nondets=True): #*****
    '''
    Best function here. Converts dictionary of MJDs and 5-sigma depths (keys corresponding to bands) into 1) masking index arrays 
    for each band and 2) a function `magfunc()` which processes magnitudes into photometric uncertainties and accounts for 
    nondetections.
    '''
    # Define dataframe with relevant photometric uncertainty parameters from Ivezic et al. 2009
    cols = ['m sky', 'theta', 'theta eff', 'gamma', 'k', 'C', 'm5', 'dC', 'dC2', 'dm5']
    ivezic_params = pd.DataFrame([[22.99, 0.81, 0.92, 0.38, 0.491, 23.09, 23.78, 0.62, 0.23, 0.21],
                   [22.26, 0.77, 0.87, 0.039, 0.213, 24.42, 24.81, 0.18, 0.08, 0.16],
                  [21.20, 0.73, 0.83, 0.039, 0.126, 24.44, 24.35, 0.10, 0.05, 0.14],
                  [20.48, 0.71, 0.80, 0.039, 0.096, 24.32, 23.92, 0.07, 0.03, 0.13]],
                 columns=cols, index=['u', 'g', 'r', 'i'])
    

    MJDs = [mjd_sig5_dict[band][0] for band in bands]
    sig5s = [mjd_sig5_dict[band][1] for band in bands]
    
    mjd, sig5s, mask_arrs = mjd_to_inds2(MJDs=MJDs, sig5s=sig5s, gauss=gauss)
    sig5_dict = dict(zip(bands, sig5s))

    funclist = []
    for band in bands:
        def magfunc(mag, sig5=sig5_dict[band]):
            mag, magerr = photunc(mag=mag, sig5=sig5, band=band, nondets=nondets)
            return mag, magerr
        funclist.append(magfunc)
        pass
    
    dict_ = dict(zip(bands, [[[mask], [func]] for i, (mask, func) in enumerate(zip(mask_arrs, funclist))]))
    
    return mjd, dict_


# <----------------- Sub-functions for more holistic functions listed above -----------------> #

def mjd_to_inds(MJD, gauss=True):
    '''
    Converts LSST-generated `observationStartMJD` array, here labeled `MJD`, to a masking index array, 'inds'. 
    `inds` must be applied to a daily-cadenced MJD array which spans at least the domain of the input `MJD` array.

    Parameters:
    -----------
    `MJD`: maf-generated LSST observation MJDs
    `gauss`: Boolean to determine if normal uncertainties are added to daily-cadenced output array
    
    '''
    MJD = np.sort(np.array(MJD))
    MJD = MJD[MJD > 0]

    MJD_con = []
    diff = np.diff(MJD)
    for i in range(len(diff)):
        if diff[i] > 1.:
        # This conditional locates points in the MJD array where the gap in observations is greater than 1 day. 
        # MJD[i] is thus the last observation date BEFORE said >1 day gap.
            a = np.where(MJD.round(0) == MJD[i].round(0))[0]
            # a is the array of indices whose ROUNDED observation date equals the ROUNDED observation date, MJD[i]
            MJD_con.append(np.mean(MJD[a]))
            # Observations 'MJD[a]' encode the array of observations that had a <1 day gap BEFORE MJD[i]. These
            # observations are bin-averaged and added to the condensed MJD array, 'MJD_con'
            pass
        pass
    MJD_con = np.array(MJD_con)
    
    MJD_con_r = [mjd.round(0) for mjd in MJD_con] # rounding condensed or bin-average MJD array (this inherently
    # sacrifices the precision of MJDs generated by maf_sims, but on the order of <1 day)
    MJD_r = list(MJD.round(0)) # rounding input MJD array
    mjd = np.arange(MJD_con_r[0], MJD_con_r[-1]+1, 1) # daily-cadenced array to generate masking indices

    inds = []
    for i in range(len(mjd)):
        if mjd[i] in MJD_con_r: # Identifies observations in bin-averaged maf MJDs that are present in the daily-
        # cadenced 'mjd' array to create masking array
            inds.append(i)
            pass
        pass

    if gauss:
        mjd_new = [] # Same as the 'mjd' array, but with gaussian uncertainty added to the daily cadence.
        for m in range(len(mjd)):
            err = np.random.normal(0, 1)
            mjd_new.append(mjd[m]+err) # 
            pass
        mjd = mjd_new
        pass

    return np.array(mjd), np.array(inds)

def mjd_to_inds2(MJDs, sig5s, gauss=True):
    '''
    Converts MULTIPLE MJD arrays into ONE underlying (constant-cadence) mjd array and MULTIPLE masking index arrays.
    '''
    MJDs_binavg_rounded = []
    sig5s_binavg = []
    for MJD, sig5 in zip(MJDs, sig5s):
        MJD = np.array(MJD)
        sig5 = sig5[np.argsort(MJD)]
        MJD = np.sort(MJD)
        
        MJD_binavg = []
        sig5_binavg = []
        diff = np.diff(MJD)
        for i in range(len(diff)):
            if diff[i] > 1.:
                a = np.where(MJD.round(0) == MJD[i].round(0))[0]
                MJD_binavg.append(np.mean(MJD[a]))
                sig5_binavg.append(np.mean(sig5[a]))
                pass
            pass
            
        MJD_binavg = np.array(MJD_binavg)
        sig5_binavg = np.array(sig5_binavg)
        MJD_binavg_rounded = np.round(MJD_binavg, 0)
        
        MJDs_binavg_rounded.append(MJD_binavg_rounded)
        sig5s_binavg.append(sig5_binavg)
        pass
    
    mjd_start = min([t[0] for t in MJDs_binavg_rounded])
    mjd_end = max([t[-1] for t in MJDs_binavg_rounded])
    mjd = np.arange(mjd_start, mjd_end+1, 1)

    mask_arrs = []

    for i, MJD_binavg_rounded in enumerate(MJDs_binavg_rounded):
        mask = []
        for idx in range(len(mjd)):
            if mjd[idx] in MJD_binavg_rounded:
                mask.append(idx)
                pass
            pass
        mask_arrs.append(np.array(mask))
        pass
    
    if gauss:
        mjd += np.random.normal(0, 1, len(mjd))
        pass
        
    return np.array(mjd), sig5s_binavg, mask_arrs


# First step: Make a function with inputs (ra, dec), bands, and vet --> MJD masking indices and associated 5-sigma depth. 
# This will require that the underlying (currently daily-cadenced) array is masked to EACH of the band's MJDs. Thus, there should be ONE 
# underlying mjd array, a masking index array for each band and associated 5-sigma depths.
# Second step: Make a function which maps magnitudes to magerrs using an array of 5-sigma depths (should be arbitrary, not band-dependent)
# This should be pretty straightforward - use ivezic function from Jessie research notebooks
# Third step: Incorporate this second function into the first to make one intermediate function which:
# Maps inputs (ra, dec), bands, and vet --> outputs MJD masking indices and a FUNCTION which takes in a magnitude array (assumed to be
# simulated with the same underlying mjd returned by the first aforementioned function) and converts them to uncertainties using ivezic
# function and 5-sigma depths (in this way, there's no need to return the 5-sigma depths unless for plotting reasons). ACTUALLY, 
# this magnitude function should have an option to CUTOFF magnitudes with mag > 5-sigma depth and convert them into observation points EQUAL
# to 5-sigma depth with associated uncertainty... I've done this before...

def radec_to_inds_5sig(ra, dec, bands=['g', 'r'], vet_lim=25., gauss=True):
    '''
    Converts (ra, dec) to masking indices AND 5-sigma depths. Good if you want to simulate many LSST light curves in parallel, but 
    might be difficult to keep track of masking indices and 5-sigma depths - BETTER TO JUST USE radec_to_mjds in parallel and then
    convert mjds/sig5s into masking indices and a photometric uncertainty calculator function (See `MJD_to_inds_magfunc()`)
    '''
    q = radec_to_mjd(ra=ra, dec=dec, bands=bands, vet=vet)

    if type(bands) is list:
        MJDs = [q[band][0] for band in bands]
        sig5s = [q[band][1] for band in bands]
        mjd, sig5s, mask_arrs = mjd_to_inds2(MJDs=MJDs, sig5s=sig5s, gauss=gauss)
        sig5s = [q[band][1][mask_arrs[i]] for i, band in enumerate(bands)]
        dict_ = dict(zip(bands, [[MJDs[i], sig5s[i]] for i, band in enumerate(bands)]))
        pass
    else:
        MJD = q[0]
        mjd, mask_arr = mjd_to_inds(MJD=MJD, gauss=gauss)
        sig5 = q[1][mask_arr]
        dict_ = dict(zip(bands, [MJD, sig5]))
        pass
    
    return mjd, dict_

def MJD_to_inds_5sig(mjd_sig5_dict, bands=['g', 'r'], gauss=True):
    '''
    Alternative to 'radec_to_inds_5sig' which converts a dictionary of MJDs and 5-sigma depths in different bands to a constant-cadence mjd
    array and masking index arrays and 5-sigma depths for each band. THIS IS CONVENIENT if you use a separate file to grab many LSST MJDs from
    a wide array of ra, dec. The 'radec_to_inds_5sig()' function requires you to run maf on the local computer, significantly slowing the 
    simulation process.

    Parameters:
    -----------
    'mjd_sig5_dict': Dictionary whose keys correspond to different bands. The dictionary slots should be 2-element lists where the first 
    element is an MJD array (or list) and whose second element is an array/list of corresponding 5-sigma depths.
    'bands': (MUST BE A LIST) of bands you wish to simulate
    'gauss': Boolean which determines if normal uncertainties are added to constant-cadenced mjd array

    '''
    MJDs = [mjd_sig5_dict[band][0] for band in bands]
    sig5s = [mjd_sig5_dict[band][1] for band in bands]
    mjd, sig5s, mask_arrs = mjd_to_inds2(MJDs=MJDs, sig5s=sig5s, gauss=gauss)
    dict_ = dict(zip(bands, [[MJDs[i], sig5s[i]] for i, band in enumerate(bands)]))

    return mjd, dict_

def photunc(mag, sig5, band, a=-10, sc=1.5, sig_sys=.005, nondets=True):
    '''
    Calculates photometric uncertainty for given `mag` array using parameters from Ivezic et al. 2009.
    Note, also includes option to count magnitudes which exceed 5-sigma depth as nondetections.
    '''
    # Define dataframe with relevant photometric uncertainty parameters from Ivezic et al. 2009
    cols = ['m sky', 'theta', 'theta eff', 'gamma', 'k', 'C', 'm5', 'dC', 'dC2', 'dm5']
    df = pd.DataFrame([[22.99, 0.81, 0.92, 0.38, 0.491, 23.09, 23.78, 0.62, 0.23, 0.21],
                   [22.26, 0.77, 0.87, 0.039, 0.213, 24.42, 24.81, 0.18, 0.08, 0.16],
                  [21.20, 0.73, 0.83, 0.039, 0.126, 24.44, 24.35, 0.10, 0.05, 0.14],
                  [20.48, 0.71, 0.80, 0.039, 0.096, 24.32, 23.92, 0.07, 0.03, 0.13]],
                 columns=cols, index=['u', 'g', 'r', 'i'])

    if nondets:
        nondets = np.where(mag >= sig5)
        mag[nondets] = sig5[nondets]
        pass
    
    gamma = df['gamma'].loc[band]
    x = 10**(0.4*(mag - sig5))
    sig_rand = np.sqrt((0.04 - gamma)*x + gamma*x**2)
    sigma_phot = np.sqrt(sig_sys**2 + sig_rand**2)

    return mag, sigma_phot