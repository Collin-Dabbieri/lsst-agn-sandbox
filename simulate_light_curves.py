import numpy as np
from astroML.time_series import generate_damped_RW
from matplotlib import pyplot as plt
import time
from astropy.io import fits
from astropy.cosmology import Planck18,z_at_value
from astropy import units as u
from astropy.modeling.powerlaws import BrokenPowerLaw1D as bpl
from rubin_sim.phot_utils import Bandpass, Sed
import rubin_sim.maf as maf
import os
import math
from matplotlib import pyplot as plt
from astropy.time import Time
import multiprocessing
import pandas as pd
from datasets import load_dataset, load_from_disk
from datasets import Dataset

# read section 9.3 at https://docs.python.org/3/tutorial/classes.html to get a good description of python classes

# The idea for this will be every quasar gets its own class 
# and all of the important properties (including the light curve) will be loaded into instance variables inside the class
# functions in the class generally won't return the variables they calculate but will instead save them as instance variables
# At the end of the day each quasar will have one class object that contains all of the relevant information you could ever want

# For our first pass we will just be generating normal quasar DRW light curves, we will add sinusoids next

# This version of the simulation with have independent damped DRW's for each band in order to give us a quick starting point


######################## USER SPECIFIED VALUES ###################################################################################

datadir='/Users/dabbiecm/DataProducts/'

np_savepath=datadir+'1000_singleband_lcs_lsst_sampling_w_labels.npy'
ds_savepath=datadir+'1000_singleband_lcs_lsst_sampling_w_labels.hf'

run_id = 'lc_sim'
out_dir = '/Users/dabbiecm/DataProducts/rubin_results/'+run_id
baseline_file='/Users/dabbiecm/rubin_sim_data/sim_baseline/baseline_v3.4_10yrs.db'
name = os.path.basename(baseline_file).replace('.db','')
results_db = maf.db.ResultsDb(out_dir=out_dir+'_'+name)



######################## Load Data ###############################################################################################

# Load in data for class
# load in fits file containing SDSS quasar properties
hdu = fits.open(datadir+'dr16q_prop_May01_2024.fits')
dr16q = hdu[1].data

# find directory for LSST throughput curves
fdir = os.getenv('RUBIN_SIM_DATA_DIR')
if fdir is None:  #environment variable not set
    fdir = os.path.join(os.getenv('HOME'), 'rubin_sim_data')
fdir = os.path.join(fdir, 'throughputs', 'baseline')

# load in throughput curves
filterlist = ['u', 'g', 'r', 'i', 'z', 'y']
lsst = {}
for f in filterlist:
    lsst[f] = Bandpass()
    lsst[f].read_throughput(os.path.join(fdir, f'total_{f}.dat'))
    # TODO add ZTF filters for comparison with real ZTF light curves
    # Under 'Palomar -> ZTF' on filter website



######################## CLASS DEFINITION ########################################################################################

class lightcurvesim():
    """A class for generating simulated quasar light curves"""
    def __init__(self,fiducial=False):
        # The __init__ function will automatically be called whenever you initialize the class
        # You can use this function to pre-load instance variables that will be useful to the user
        self.filterlist=['u', 'g', 'r', 'i', 'z', 'y']
        self.generate_quasar_properties(fiducial)
        

    def generate_quasar_properties(self,fiducial):
        '''
        Randomly generates a mass, redshift, and i-band absolute magnitude for the quasar, run on class initialization.
        These values are drawn from real SDSS quasars.
        This function also calculates mean magnitude values for each band by assuming a power law SED and convolving with the LSST throughput curves
        
        requires:
            None
        creates:
            logM - Mass of the black hole (log_10(solar masses))
            Mi - i-band absolute magnitude (magnitude)
            z - redshift
            mean_mag - mean magnitude value in each LSST band
            ra,dec - location on sky
            deep_field - Boolean whether the object is in an LSST deep field (much higher sampling)
            label - 'binary' or 'single'
            lambda_eff - dictionary w effective wavelength of each filter in observer frame
            lambda_rest - same but rest frame
            
        '''

        num_quasars=len(dr16q)
        idx_all=np.arange(num_quasars)
        zlow=0.1
        zhigh=1
        use = np.where((dr16q['Z_DR16Q']>(zlow)) & (dr16q['Z_DR16Q']<(zhigh)) & (dr16q['LOGMBH']>6))[0].tolist()
        idx_choose=np.random.choice(use)
        #idx_choose=394515
        
        self.z=dr16q['Z_DR16Q'][idx_choose]
        self.logLbol=dr16q['LOGLBOL'][idx_choose]
        self.Mi = 90-2.5*self.logLbol #absolute i band magnitude K corrected to z=2
        self.logM=dr16q['LOGMBH'][idx_choose] #primary mass

        # because vera rubin is down in chile, we're going to randomly choose southern sky locations instead of using the actual RA and DEC of sloan quasars
        if np.random.random()>0.5:
            # 50% chance to randomly select a location
            self.dec=np.random.uniform(-65,5) #https://www.lsst.org/science/simulations/opsim/skycoverage (further trimmed to ensure well-sampled observations)
            self.ra=np.random.uniform(0,360)
            self.deep_field=False
        else:
            # 50% chance to select a deep drilling field
            # there are 4 deep drilling fields
            self.deep_field=True
            rand_field=np.random.choice([1,2,3,4])
            if rand_field==1:
                self.ra=9.45
                self.dec=-44.0
            elif rand_field==2:
                self.ra=35.71
                self.dec=-4.75
            elif rand_field==3:
                self.ra=53.12
                self.dec=-28.10
            elif rand_field==4:
                self.ra=150.10
                self.dec=2.18

        # get the 5100 A flux normalization from the bolometric luminosity
        dl        = Planck18.luminosity_distance(self.z).to(u.cm)
        L5100     = (self.logLbol-4.891)/0.912 # runnoe+12
        f_510     = ((u.erg/u.s)*10**(L5100))/(4.*np.pi*dl*dl)
        flam510   = f_510/(510.*u.nm)
        norm_5100 = flam510

        # the quasar SED is a power law with different slopes in different wavelength regions
        # vanden berk+ (2001)
        # for flamda
        a_uv = -1.54  # lambda < 4200
        a_opt = -0.42 # lambda > 4200
        
        wave = np.linspace(150.,2000.,2500) # 150nm to 2um in 0.75nm steps
        flux = bpl.evaluate(wave,amplitude=1,x_break=4200,alpha_1=-a_uv,alpha_2=-a_opt)
        idx = self.closest(wave,5100.)
        A = norm_5100/flux[idx]
        flux = A*flux
        wave_obs = wave*(1+self.z)
        flux_obs = flux.value # the flux array has already been scaled to the derived flux at 5100 A
                              # which is already in the observer frame
        # set the SED
        agn = Sed(wavelen=wave_obs,flambda=flux_obs)

        self.mags = {}
        for f in self.filterlist:
            self.mags[f] = agn.calc_mag(lsst[f])

        self.lambda_eff={}
        self.lambda_eff['u']=3671
        self.lambda_eff['g']=4827
        self.lambda_eff['r']=6223
        self.lambda_eff['i']=7546
        self.lambda_eff['z']=8691
        self.lambda_eff['y']=9712
        
        self.lambda_rest={}
        for f in self.filterlist:
            self.lambda_rest[f]=self.lambda_eff[f]/(1+self.z)

        # 50% chance of being a binary
        if np.random.random()>0.5:
            self.label='binary'

            self.q=np.random.uniform(low=0.1,high=1) #sample a mass ratio to use to calculate the velocity of the secondary
            mass_secondary=(10**self.logM)*self.q
            self.logM_secondary=np.log10(mass_secondary)

            # TODO given the mass of the primary and the secondary, sample the period in a reasonable way
            # the binary will spend a much longer time at long separations compared to short separations
            # so uniformly sampling the period is not very sensible
            # you would want a sampling that is heavily weighted toward longer periods
            # alternatively, maybe we want to make sure the model can handle shorter periods
            # so we uniformly sample the period, not because it's physical, but because we want the model to learn all possible periods
            self.P_rest=(3*365-10)*np.random.random()+10 # uniformly sample from 10 days to 3 years
            self.P_obs=self.P_rest*(1+self.z)

            
            # in order to get a random viewing angle, uniformly sample over the cos of the inclination angle
            # face-on viewing angle (less likely) corresponds to i of 0
            cos_inc=np.random.uniform(0,1)
            self.inc=math.acos(cos_inc) #radians
            
        else:
            self.label='single'


        # if True, use Charisi et al. 2022 fiducial values
        if fiducial:
            self.label='binary'
            self.logM=8.9 #primary mass
            self.q=0.25
            inc_degrees=60
            self.inc=inc_degrees*(2*np.pi/360)
            self.P_obs=365
            self.P_rest=self.P_obs/(1+self.z)
            mass_secondary=(10**self.logM)*self.q
            self.logM_secondary=np.log10(mass_secondary)
            self.deep_field=True
            self.ra=9.45
            self.dec=-44.0

    def closest(self, array, value):     
        array = np.asarray(array)    
        idx = (np.abs(array - value)).argmin()   
        return idx

    def get_mag_err(self,fltr,m,m5):
        # from table 2 of Ivezic et al. (2019)
        # https://ui.adsabs.harvard.edu/abs/2019ApJ...873..111I/abstract
        gammas = {'u':0.038,'g':0.039,'r':0.039,'i':0.039,'z':0.039,'y':0.039}
        gamma = gammas[fltr]
        
        # systematic errors are <0.005mag by design
        sig_sys2 = (0.005)**2
    
        # random errors
        x = 10**(0.4*(m-m5))
        sig_rand2 = (0.04-gamma)*x+gamma*x**2
    
        # photometric error in a single visit
        sig1 = np.sqrt(sig_sys2+sig_rand2)
    
        mag_err = np.random.randn(m.size)*sig1
        return sig1,mag_err
        
    def get_DRW_params(self):
        '''
        Uses Burke et al. 2023 Eq. 9 and 10 to get parameters for generating DRW light curves.
        
        requires:
            logM - Mass of the black hole (log_10(solar masses))
            Mi - i-band absolute magnitude (magnitude)
            z - redshift
        returns:
            SF_infinity - structure function evaluated at infinity (magnitude)
            tau - damping timescale (days)
        '''
        if self.label=='binary':
            logM=self.logM_secondary
        else:
            logM=self.logM

        A1, B1, C1, D1 = -0.51, -0.479, 0.131, 0.18
        SF_infinity={}
        for f in self.filterlist:
            SF_infinity[f] = 10 ** (A1 + B1*math.log(self.lambda_rest[f]/4000,10) + C1*(self.Mi+23) + D1*(logM-9))
        
    
        A2, B2, C2, D2 = 2.4, 0.17, 0.03, 0.21
        tau={}
        for f in self.filterlist:
            tau[f] = 10 ** (A2 + B2*math.log(self.lambda_rest[f]/4000,10) + C2*(self.Mi+23) + D2*(logM-9))

        return SF_infinity, tau

    def get_DRW_params_macleod_2010(self):
        '''
        logf = A + B log (lambda_rf/4000A) + C(Mi+23) + Dlog (Mbh/10^9) + Elog(1+z)
        f=SF_inf -> A=-0.57+-0.01, B=-0.479+=0.005, C=0.117+-0.009, D=0.11+-0.02, E=0.07+-0.05
        f=tau -> A=2.4+-0.2, B=0.17+-0.02, C=-0.05+-0.03, D=0.12+-0.04, E=-0.7+-0.5
        '''
        if self.label=='binary':
            logM=self.logM_secondary
        else:
            logM=self.logM

        A1,B1,C1,D1,E1=-0.57,-0.479,0.117,0.11,0.07

        SF_infinity={}
        for f in self.filterlist:
            SF_infinity[f] = 10 ** (A1 + B1*math.log(self.lambda_rest[f]/4000,10) + C1*(self.Mi+23) + D1*(logM-9) + E1*math.log(1+self.z,10) )

        A2,B2,C2,D2,E2=2.4,0.17,-0.05,0.12,-0.7
        tau={}
        for f in self.filterlist:
            tau[f] = 10 ** (A2 + B2*math.log(self.lambda_rest[f]/4000,10) + C2*(self.Mi+23) + D2*(logM-9) + E1*math.log(1+self.z,10) )

        return SF_infinity,tau
        

    def get_mjd_sampling(self):
        '''
        uses rubin_sim suite to get realistic observation times in each band

        requires:
            ra
            dec
        returns:
            mjd[fltr] - Modified Julian Date (MJD) of LSST observations in each filter in the observer frame
            mjd_rest[fltr] - MJD of LSST observations in each filter in the rest frame
            mjd_all[fltr] - high resolution sampling of the MJD in the observer frame
            mjd_rest_all[fltr] -high resolution sampling of the MJD in the rest frame
        '''
        # choose metrics
        metric = maf.metrics.PassMetric(cols=['filter', 'observationStartMJD', 'fiveSigmaDepth'])
        sql = ''
        # slice on single sky position
        slicer = maf.slicers.UserPointsSlicer(ra=self.ra, dec=self.dec)

        # bundle the constraints, slicer, and metrics
        bundle_list = []
        bundle_list.append(maf.MetricBundle(metric, slicer, sql, run_name=name))

        # run the MAF
        bd = maf.metricBundles.make_bundles_dict_from_list(bundle_list)
        bg = maf.metricBundles.MetricBundleGroup(bd, baseline_file, out_dir=out_dir, results_db=results_db)
        bg.run_all()

        # the visits overlapping a single sky position are a "slice" of data
        # this nameing convention is related to how the MAF works
        data_slice = bundle_list[0].metric_values[0]

        # find the unique filters
        filters = np.unique(data_slice['filter'])

        # expand the data_slice recarray to include columns for simulated quantities
        # do this just so that it is easy to save the information for plotting
        new_dt = np.dtype(data_slice.dtype.descr + [('obs_mag', '<f8'),('magerr','<f8'),('sigm','<f8')])
        sim_slice = np.zeros(data_slice.shape, dtype=new_dt)
        for col in data_slice.dtype.names: sim_slice[col] = data_slice[col]

        # sort by MJD and set a new time array
        # this is useful for the DRW generator
        sim_slice.sort(order='observationStartMJD')
        min_mjd = np.min(sim_slice['observationStartMJD'])

        nt = Time.now()
        nsteps = 11*365
        mjd_now = np.floor(nt.mjd)
        mjd_drw = np.linspace(mjd_now,mjd_now+nsteps,nsteps+1)
        mjd_min_drw = np.min(mjd_drw)

        mjd_all = np.concatenate([mjd_drw, sim_slice['observationStartMJD']])
        mjd_all.sort()
        mjd_rest_all=mjd_all/(1+self.z)
        time_all = mjd_all - np.min(mjd_all)
        time_all_rest=time_all/(1+self.z)

        idx_drw = np.searchsorted(mjd_all,mjd_drw)
        mjd={}
        mjd_rest={}
        for fltr in self.filterlist:
            idx = np.searchsorted(mjd_all, sim_slice['observationStartMJD'][sim_slice['filter'] == fltr])
            mjd[fltr] = mjd_all[idx]
            mjd_rest[fltr]=mjd_all[idx]/(1+self.z)

        return mjd,mjd_rest,mjd_all,mjd_rest_all
        
    def get_lag(self):
        '''
        Calculates the reverberation lag time for each LSST filter 
        tau_obs=tau_0*(1+z)**(1-beta)*[(lambda_i/9000A)**beta-(lambda_g/9000A)**beta]
        tau_0=5.38+0.43-0.34
        beta=1.28+0.41-0.39

        returns:
            tau_obs - dictionary with lag times from the g band to every other band (tau_obs['gu'] will be negative)
        '''
        #flip a coin, if heads, sample from positive, if tails, sample from negative
        if np.random.rand()>0.5:
            dev=np.absolute(np.random.normal(loc=0,scale=0.43))
            tau_0=5.38+dev
        else:
            dev=np.absolute(np.random.normal(loc=0,scale=0.34))
            tau_0=5.38-dev
        if np.random.rand()>0.5:
            dev=np.absolute(np.random.normal(loc=0,scale=0.41))
            beta=1.28+dev
        else:
            dev=np.absolute(np.random.normal(loc=0,scale=0.39))
            beta=1.28-dev
            
        tau_obs={}
        tau_obs['gu']=tau_0*(1+self.z)**(1-beta)*((self.lambda_rest['u']/9000)**beta-(self.lambda_rest['g']/9000)**beta)
        tau_obs['gr']=tau_0*(1+self.z)**(1-beta)*((self.lambda_rest['r']/9000)**beta-(self.lambda_rest['g']/9000)**beta)
        tau_obs['gi']=tau_0*(1+self.z)**(1-beta)*((self.lambda_rest['i']/9000)**beta-(self.lambda_rest['g']/9000)**beta)
        tau_obs['gz']=tau_0*(1+self.z)**(1-beta)*((self.lambda_rest['z']/9000)**beta-(self.lambda_rest['g']/9000)**beta)
        tau_obs['gy']=tau_0*(1+self.z)**(1-beta)*((self.lambda_rest['y']/9000)**beta-(self.lambda_rest['g']/9000)**beta)

        return tau_obs

    def damped_random_walk(self,t_rest,tau,SFinf,m_true):
    
        dt=np.diff(t_rest) #the n-th discrete difference
        #sigma = np.sqrt(2 * SFinf/tau)
        #sigma=SFinf/(np.sqrt(tau/2))
        #SF_inf=sqrt(2)sigma
        sigma=SFinf/np.sqrt(2)
        n = len(t_rest)
        x = np.zeros(n)
        x[0] = m_true
        v = np.random.normal(0, 1, size=n-1)
    
        for i in range(1,n):
            x[i]=(x[i-1]) - ((x[i-1]-m_true)*dt[i-1]/tau) + sigma*v[i-1]*np.sqrt(2*dt[i-1]/tau)
    
        return x

    def get_binary_orbital_velocity(self,P_rest):
        '''
        Calculates the velocity of the secondary black hole in the binary. We're assuming the secondary black hole is the only actively accreting black hole
        P^2=(4pi^2/GM)a^3
        a*2pi=P*v_2
        ...
        v_2^3=2piGM/P

        returns:
            v_2 velocity of the secondary black hole in km/s
        '''
        P_seconds=P_rest*86400 #s
        G=6.67e-11 #kg m s^-2 m^2 kg^-2 = m^3 s^-2 kg^-1
        # GM/P = m^3 s^-2 kg^-1 kg s^-1 = m^3 s^-3
        M_kg=(10**(self.logM_secondary))*1.989e30 #kg

        v_2=((2*np.pi*G*M_kg/P_seconds)**(1/3))/1000 #km/s
        return v_2

    def doppler_boost_variability(self, alpha_nu, v_2, inc, omega, phi_0, mjd_rest_all):
        """
        Constants:
        c : speed of light
    
        Requires:
        F_sec_ν_0 : stationary luminosity of secondary mini-disc
        α_ν : spectral index
        v_2 : orbital velocity of secondary SMBH
        i : inclination / angle between orbital plane and plane of sky
        ω : orbital angular frequency 
        t : time - array of days
        Φ_0 : phase of sinusoid
    
        returns:
        sinusoid[fltr] - flux factor of sinusoid
        """

        # sample a high res sinusoid in luminosity units
        c = 299792 #km/s
        F_sec_nu_0=1 # this will allow the sinusoid equation to return a flux factor
        sinusoid = F_sec_nu_0*(1 + (3 - alpha_nu)*v_2*np.sin(inc)*np.sin(omega*mjd_rest_all + phi_0)/c)

        return sinusoid
        
    def generate_light_curve(self):
        '''
        
        returns:
            mjd[fltr] - Modified Julian Date (MJD) of LSST observations in each filter in the observer frame
            mjd_all[fltr] - high resolution sampling of the MJD in the observer frame
            drw[fltr] - damped random walk at actual observation sampling for each filter in mag units
            drw_all[fltr] - damped random walk at high resoltion sampling for each filter in mag units
            sinusoid[fltr] - binary orbital variability in mag units at LSST sampling for each filter
            (There's no sinusoid_all because the sinusoid in mags is not calculated at high resolution sampling to save computation time)
            (If you're interested in the sinusoid at high res sampling, grab self.F_nu which is in luminosity units)
            light_curve[fltr] - final light curve for each band, if label=='binary' this is the sum of the sinusoid and drw
        '''
        #sample according to LSST cadence
        mjd,mjd_rest,mjd_all,mjd_rest_all=self.get_mjd_sampling()

        # first, generate a drw light curve in the g-band at the high res sampling
        SF_infinity,tau=self.get_DRW_params()
        drw_all={}
        drw_all['g']=self.damped_random_walk(mjd_rest_all, tau['g'], SF_infinity['g'], self.mags['g'])

        # inject the difference in magnitude for each band
        drw_all['u']=drw_all['g']+(self.mags['u']-self.mags['g'])
        drw_all['r']=drw_all['g']+(self.mags['r']-self.mags['g'])
        drw_all['i']=drw_all['g']+(self.mags['i']-self.mags['g'])
        drw_all['z']=drw_all['g']+(self.mags['z']-self.mags['g'])
        drw_all['y']=drw_all['g']+(self.mags['y']-self.mags['g'])

        # Now sample each filter at LSST observations
        drw={}
        for fltr in self.filterlist:
            idx = np.searchsorted(mjd_all, mjd[fltr]) # find the indexes in our high res sampling that correspond to actual observations in this filter
            #mjd_rest_all and mjd_all have the same indexing so we can use them interchangeably in this case
            drw[fltr] = drw_all[fltr][idx]
        
        # Now, use Homayouni et al. 2019 relationship to get lags for other bands
        tau_obs=self.get_lag()
        
        # shift your obs frame time array according to the injected lag
        mjd['u']=mjd['u']+tau_obs['gu']
        mjd['r']=mjd['r']+tau_obs['gr']
        mjd['i']=mjd['i']+tau_obs['gi']
        mjd['z']=mjd['z']+tau_obs['gz']
        mjd['y']=mjd['y']+tau_obs['gy']
        # also update rest frame time arrays
        mjd_rest['u']=mjd['u']/(1+self.z)
        mjd_rest['r']=mjd['r']/(1+self.z)
        mjd_rest['i']=mjd['i']/(1+self.z)
        mjd_rest['z']=mjd['z']/(1+self.z)
        mjd_rest['y']=mjd['y']/(1+self.z)

        # if this object is a binary, add the appropriate sinusoid
        if self.label=='binary':
            alpha_nu=-0.44 #charisi et al. 2022, from vanden berk 2001
            # TODO make sure you're properly using rest-frame vs observed frame period
            self.v_2=self.get_binary_orbital_velocity(self.P_rest) #km/s
            omega=2*np.pi/self.P_obs # sinusoids are in observed frame
            phi_0=np.random.uniform(0,2*np.pi)
            sinusoid=self.doppler_boost_variability(alpha_nu, self.v_2, self.inc, omega, phi_0, mjd_rest_all) # this is a flux factor
        else:
            sinusoid=None
            self.P_rest=None
            self.P_obs=None

        light_curve={}
        for fltr in self.filterlist:
            if self.label=='binary':
                num_obs=len(drw[fltr])
                light_curve[fltr]=np.empty(num_obs)
                # find the index of the high res sinusoid corresponding to the observations in this filter
                idx = np.searchsorted(mjd_all, mjd[fltr]) # find the indexes in our high res sampling that correspond to actual observations in this filter
                #mjd_rest_all and mjd_all have the same indexing so we can use them interchangeably in this case
                sinusoid_obs=sinusoid[idx]
                for i in range(num_obs):               
                    #m_f-m_i=-2.5log_10(F_f/F_i)
                    # where sinusoid==>alpha, F_f=alpha*F_i because sinusoid is a flux factor
                    #m_f=-2.5log_10(alpha)+m_i
                    #or in natural logs
                    #m_f=-(2.5/ln(10))*ln(alpha)+m_i
                    m_i=drw[fltr][i]
                    alpha=sinusoid_obs[i]
                    m_f=-(2.5/np.log(10))*np.log(alpha)+m_i
                    
                    light_curve[fltr][i]=m_f
            else:
                light_curve[fltr]=np.array(drw[fltr])
            
        # TODO inject realistic variance as well as bluer-when-brighter relationship into the light curves for each band

        return mjd,mjd_all,drw,drw_all,sinusoid,light_curve

def generate_single_band_light_curve(index):
    # the index here is not actually used but is required for multiprocessing
    quasar=lightcurvesim()
    mjd,mjd_all,drw,drw_all,sinusoid,light_curve=quasar.generate_light_curve()
    F_nu=quasar.F_nu # high res sunusoid in luminosity units
    return [mjd['g'],light_curve['g'],quasar.label,quasar.P]
    

######################## MAIN LOOP ########################################################################################


if __name__=='__main__': 

    # We're going to be generating many light curves for many simulated quasars
    # because simulating one quasar light curve is independent of all of the other quasars' light curves, this is easily parallelizable
    # use multiprocessing.Pool to parallelize the generation of 1000 quasar light curves

    time0=time.time()
    num_gen=1000
    tasks=np.arange(num_gen)
    results=[]
    with multiprocessing.Pool(4) as pool:
        for result in pool.imap_unordered(generate_single_band_light_curve,tasks):
            results.append(result)
    time1=time.time()
    print('Time to make {:.0f} light curves: {:.1f} minutes'.format(num_gen,(time1-time0)/60))


    #time0=time.time()
    #results=[]
    #for task in tasks:
    #    result=generate_single_band_light_curve(task)
    #    results.append(result)
    #time1=time.time()
    #print('Time to make {:.0f} light curves without Pool: {:.1f} minutes'.format(num_gen,(time1-time0)/60))
    
    savefile={}
    savefile['mjd']=[]
    savefile['light_curve']=[]
    savefile['label']=[]
    savefile['period']=[]
    for result in results:
        savefile['mjd'].append(result[0])
        savefile['light_curve'].append(result[1])
        savefile['label'].append(result[2])
        savefile['period'].append(result[3])
    np.save(datadir+'1000_singleband_lcs_lsst_sampling_w_labels.npy',savefile)


    # convert to huggingface-like data format
    # convert from essentially long to wide format
    data=np.load(np_savepath,allow_pickle=True)
    mjds=data.item().get('mjd')
    light_curves=data.item().get('light_curve')
    labels=data.item().get('label')
    num_light_curves=len(mjds)
    
    dataset_data = []
    for i in range(num_light_curves):
        if labels[i]=='single':
            label_int=0
        elif labels[i]=='binary':
            label_int=1
        else:
            raise ValueError('Wrong Label')
        
        # combine into a single dictionary
        quasar_data = {'mjd':mjds[i], 'light_curve':light_curves[i], 'label': label_int, 'label_name': labels[i]}
        # add to dataset
        dataset_data.append(quasar_data)
    
    # convert into a pandas dataframe
    df = pd.DataFrame(dataset_data)
    # save as datasets format
    dataset = Dataset.from_pandas(df, preserve_index=False)
    # save to disk
    dataset.save_to_disk(ds_savepath)
    # see if we can load in the data
    lightwave_ds = load_from_disk(ds_savepath)
    # look at the first 5 rows
    print(lightwave_ds)


    