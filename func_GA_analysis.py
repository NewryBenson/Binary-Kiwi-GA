# Functions for GA analysis script part of Kiwi-GA
# Created by Sarah Brands @ 29 July 2022

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
from scipy import stats
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import fastwind_wrapper as fw
import magnitude_to_radius as m2r

RMSEA_THRESHOLD = 1.5

def get_luminosity(Teff, radius):
    '''Calculate L in terms of log(L/Lsun), given Teff (K)
    and the radius in solar radii'''

    sigmaSB = 5.67051e-5
    Lsun = 3.9e33
    Rsun = 6.96e10

    radius_cm = radius * Rsun
    luminosity_cgs  = 4*math.pi * sigmaSB * Teff**4 * radius_cm**2
    luminosity = np.log10(luminosity_cgs / Lsun)

    return luminosity

def get_mass(logg, Rstar):
    ''' Gives Mstar in solar units, given logg and Rstar in solar units'''

    Msun = 1.99e33
    Rsun = 6.96e10
    Gcgs = 6.67259e-8

    g = 10**logg
    Rstar = Rstar*Rsun

    Mstar = g * Rstar**2 / Gcgs

    return Mstar / Msun

# def get_fx(mdot, vinf):
#     """ Estimates fx based on the Mdot and vinf, based on the
#     power law of Kudritzki, Palsa, Feldmeier et al. (1996). This power law
#     is extrapolated also outside where Kudritzki+96 have data points.
#     """
#
#     mdot = 10**mdot / 10**(-6)
#     logmdotvinf = np.log10(mdot/vinf)
#
#     # Relation from Kudritzki, Palsa, Feldmeier et al. (1996)
#     logfx = -5.45 - 1.05*logmdotvinf
#     # fx = 10**(logfx)
#
#     return logfx

def get_Gamma_Edd(Lum, Mass, kappa_e=0.344):
    """
    * Lum = luminosity in solor luminosity (no log)
    * Mass = stellar mass in solar mass

    kappa_e default from Bestenlehner (2020) page 3942
    """

    Lsun = 3.9e33
    Msun = 1.99e33
    ccgs = 2.99792458*10**10 #cm/s
    Gcgs = 6.67259e-8

    Lum = Lum*Lsun
    Mass = Mass*Msun

    GammE = Lum * kappa_e / (4.*np.pi*ccgs*Gcgs*Mass)

    return GammE

def get_vesc_eff(mass, radius, GammE):
    Rsun = 6.96e10
    Msun = 1.99e33
    Gcgs = 6.67259e-8

    mass_eff = mass*(1-GammE)

    neg_idx = np.less_equal(mass_eff, 0)
    mass_eff[neg_idx] = 0

    vesc_cms = np.sqrt((2*Gcgs*mass_eff*Msun)/(radius*Rsun))
    vesc_kms = vesc_cms*1e-5

    return vesc_kms

def get_fw_fluxcont(fwdir):

    rsun = 6.957e10 #cm
    fluxcont = fwdir + 'FLUXCONT'
    indat = fwdir + 'INDAT.DAT'

    Rmax_fw_model = float(open(indat, 'r').readlines()[4].strip().split()[0])
    rstar = float(open(indat, 'r').readlines()[3].strip().split()[-1])
    mdot = float(open(indat, 'r').readlines()[5].strip().split()[0])

    stellar_surface = 4*np.pi*(rsun*rstar)**2

    # Look up the number of useful lines in the FLUXCONT
    lcount = -2
    for aline in open(fluxcont, 'r').readlines():
        lcount = lcount+1
        if len(aline.split()) == 1:
            break

    # Get FASTWIND spectrum
    lam, logFnu = np.genfromtxt(fluxcont, max_rows=lcount,
        skip_header=1, delimiter='').T[1:3]
    fnu = 10**logFnu # ergs/s/cm^2/Hz / RMAX^2
    flam = 3.00* 1e18 * fnu / (lam**2) # ergs/s/cm^2/A /RMAX^2
    flam = flam * stellar_surface # ergs/s/A /RMAX^2
    flam = flam * Rmax_fw_model**2 # ergs/s/A

    sorting = lam.argsort()
    lam = lam[sorting]
    flam = flam[sorting]

    return lam, flam

def magnitude_to_radius_SED(sed_wave, sed_flam, band, obsmag, zp_system,
    filterdir='filter_transmissions/'):

    '''Compute the radius of the star based on a fastwind model
    and observed (dereddened) absolute magnitude.

    Input:
     - band: name of the photometric band (string), see section
       'Available photometric bands' at the start of of this
       functions for which ones are included, and the
       description below on how to add more.
     - obsmag: the observed absolute magnitude in the given
       band (float)
     - zp_system: choose from 'vega', 'AB', 'ST' (string)
     - filterdir: specify (relative) path to the directory
       where the filter information is stored (string)

    Output:
     - Stellar radius in solar units (float)

    NOTE ON ADDING NEW FILTERS

    A useful resource for filter information is the
    SVO Filter Profile Service:
       http://svo2.cab.inta-csic.es/theory/fps/

    When adding a new filter, please do the following:
    1. Place an asci file with wavelengths and transmissions in
       the filter directory (specified in the parameter
       'filterdir'. In this file, lines with columns names or
       headers should start with a '#'. Wavelengths can be in
       Angstrom or nm (see next point).
    2. Add an 'elif' statement in the code below under 'Available
       photometric bands', in which you give the filter a clear
       and descriptive name, and point to the transmission file.
       Wavelength units in the data file can be either nanometers
       or Angstrom, specify which one is used in the file in the
       parameter 'waveunit' in the elif statement.
    3. Add zero points to the file 'zero_points.dat' in the
       filterdirectory. In the first column give the name of
       the filter: use the same name as in point 2.

    '''

    ##########################################################
    ###            Available photometric bands             ###
    ##########################################################

    if band == 'SPHERE_Ks':
        filterfile = 'SPHERE_IRDIS_B_Ks.dat'
        waveunit = 'nm'
    elif band == 'HST_555w':
        filterfile = 'HST_ACS_HRC.F555W.dat'
        waveunit = 'angstrom'
    elif band == '2MASS_Ks':
        filterfile = '2MASS_Ks.dat'
        waveunit = 'angstrom'
    elif band == 'VISTA_Ks':
        filterfile = 'Paranal_VISTA.Ks.dat'
        waveunit = 'angstrom'
    elif band == 'Johnson_V':
        filterfile = 'GCPD_Johnson.V.dat'
        waveunit = 'angstrom'
    elif band == "Johnson_J":
        filterfile = "Generic_Johnson.J.dat"
        waveunit = 'angstrom'
    else:
        print('Unknown value for <band>, exiting')
        sys.exit()

    ##########################################################
    ###             Computation starts here                ###
    ##########################################################

    # Read transmission profile and convert units if necessary
    filterfile = filterdir + filterfile
    wave, trans = np.genfromtxt(filterfile, comments='#').T

    if waveunit == 'nm':
        nm_to_Angstrom = 10
        wave = wave * nm_to_Angstrom
    elif waveunit == 'angstrom':
        pass
    else:
        print('Unknown value for <waveunit>, exiting')

    # Get filter zero point
    zpfile = filterdir + 'zero_points.dat'
    zp_values = np.genfromtxt(zpfile, comments='#', dtype=str)
    the_zero_point = ''
    for afilter in zp_values:
        if afilter[0] == band:
            if zp_system == 'vega':
                the_zero_point = float(afilter[1])
            elif zp_system == 'AB':
                the_zero_point = float(afilter[2])
            elif zp_system == 'ST':
                the_zero_point = float(afilter[3])
            else:
                print('Unknown value for <zp_system>, exiting')
                sys.exit()
    if the_zero_point == '':
        print('Zero point for band ' + band + ' not found, exiting')
        sys.exit()

    sed_ip = interp1d(sed_wave, sed_flam)
    F_lambda = sed_ip(wave)

    rsun = 6.96e10
    parsec_cm = 3.08567758e18
    radius_ratio = 10*parsec_cm / rsun

    filtered_flux = np.trapz(trans*F_lambda, wave)/np.trapz(trans, wave)
    obsflux = m2r.magnitude_to_flux(obsmag, the_zero_point)
    bolflux_10pc = obsflux/filtered_flux
    luminosity = bolflux_10pc * (10*parsec_cm / rsun)**2
    radius_rsun = luminosity**0.5

    return radius_rsun

def more_parameters(df, param_names, fix_names, fix_vals):
    fix_dict = dict(zip(fix_names, fix_vals))

    # List of parameters that are plotted in the extra parameter fitness plot
    # Always plot Q-parameters, luminosity, radius and spectroscopic mass, Gamma
    plist = ['logL', 'radius', 'Mspec', 'Gamma_Edd', 'vesc_eff',
        'logq0', 'logQ0', 'logq1', 'logQ1', 'logq2', 'logQ2']

    for par in ['logq0', 'logQ0', 'logq1', 'logQ1', 'logq2', 'logQ2']:
        if par not in df.columns:
            plist.remove(par)

    # Always get the luminosity and spectroscopic mass and Eddington factor
    if 'teff' in df.columns:
        df['logL'] = get_luminosity(df['teff'], df['radius'])
    else:
        df['logL'] = get_luminosity(fix_dict['teff'], df['radius'])

    if 'logg' in df.columns:
        df['Mspec'] = get_mass(df['logg'], df['radius'])
    else:
        df['Mspec'] = get_mass(fix_dict['logg'], df['radius'])

    df['Gamma_Edd'] = get_Gamma_Edd(10**df['logL'], df['Mspec'])
    df['vesc_eff'] = get_vesc_eff(df['Mspec'], df['radius'],df['Gamma_Edd'])

    # If X-rays are given and variable then include them
    if 'xlum' in df.columns:
        the_xlum = df['xlum'].values
        if np.min(the_xlum) != np.max(the_xlum):
            nan_logx = np.less_equal(the_xlum,0)
            the_xlum[nan_logx] = 1e-20
            the_logxlum = np.log10(the_xlum)
            the_logxlum[nan_logx] = math.nan
            df['logxlum'] = the_logxlum
            plist.append('logxlum')

    # Retreive fx in case it was estimated to get a fixed logxlum
    if 'fx' not in df.columns and 'fx' in fix_names:
        if ('logfx' not in df.columns) and (np.abs(fix_dict['fx']) > 1000.0):
            if 'mdot' in df.columns:
                the_mdot = df['mdot'].values
            else:
                the_mdot = fix_dict['mdot']*np.ones(len(df))
            if 'vinf' in df.columns:
                the_vinf = df['vinf'].values
            elif 'vinf' in fix_dict.keys():
                the_vinf = fix_dict['vinf']*np.ones(len(df))
            else:
                the_vinf = 2.6*df['vesc_eff'].values
                print('WARNING: assuming vinf = 2.6 vesc for all Teffs')
            the_radius = df['radius'].values
            the_logfxlist = []
            for i in range(len(df)):
                fxdct = {'radius':the_radius[i],
                         'vinf':the_vinf[i],
                         'mdot':10**the_mdot[i]}
                if float(fix_dict['fx']) > 1000.0:
                    the_fx = np.log10(float(fw.get_fx_obs(fxdct)['fx']))
                    the_logfxlist.append(the_fx)
                elif float(fix_dict['fx']) < -1000.0:
                    the_fx = np.log10(float(fw.get_fx_theory(fxdct)['fx']))
                    the_logfxlist.append(the_fx)
            df['logfx'] = the_logfxlist
            plist.append('logfx')

    # Other derived parameters are only computed when relevant.
    if 'vinf' in df.columns:
        df['vinf_vesc'] = df['vinf']/df['vesc_eff']
        plist.append('vinf_vesc')

    if 'windturb' in df.columns and 'vinf' in df.columns:
        df['windturb_kms'] = df['windturb'] * df['vinf']
        plist.append('windturb_kms')
    if 'mdot' in df.columns and 'fclump' in df.columns:
        df['mdot_fclump'] = np.log10(10**df['mdot'] * np.sqrt(df['fclump']))
        plist.append('mdot_fclump')
    elif 'mdot' in df.columns:
        df['mdot_fclump'] = np.log10(10**df['mdot'] * np.sqrt(fix_dict['fclump']))
        plist.append('mdot_fclump')
    elif 'fclump' in df.columns:
        df['mdot_fclump'] = np.log10(10**fix_dict['mdot'] * np.sqrt(df['fclump']))
        plist.append('mdot_fclump')

    return df, plist

def calculateP(chi2, degreesFreedom, normalize):
    """
    Based on the chi2 value of a model, compute the P-value
    Before this is done, all chi2 values are normalised by the lowest
    chi2 value of the run.
    """
    if normalize:
        scaling = np.min(chi2)
    else:
        scaling = degreesFreedom

    # In principle, don't use this correction factor (keep set to 1.0)
    # Can be used to make error bars artificially larger
    correction_factor = 1.0
    if correction_factor != 1.0:
        print("!"*70)
        print("\n\n\n       WARNING!!!!!!!! chi2 correction\n\n\n")
        print("!"*70)
        print("chi2 of all models artificially lowered in order to enlarge")
        print("uncertainties\n\n\n")

    chi2 = correction_factor * (chi2 * degreesFreedom) / scaling
    probs = np.zeros_like(chi2)
    try:
        for i in range(len(chi2)):
            probs[i] = stats.chi2.sf(chi2[i], degreesFreedom)
    except:
        chi2 = chi2.values
        for i in range(len(chi2)):
            probs[i] = stats.chi2.sf(chi2[i], degreesFreedom)
    return probs

def calculateP_noncent(chi2, degreesFreedom, lambda_nc):
    """
    Based on the chi2 value of a model, compute the P-value assuming a
    non-central chi2 distribution
    """

    # scaling = np.min(chi2)
    # chi2 = chi2 /scaling * degreesFreedom

    probs = np.zeros_like(chi2)
    try:
        for i in range(len(chi2)):
            probs[i] = stats.ncx2.sf(chi2[i], degreesFreedom, lambda_nc)
    except:
        chi2 = chi2.values
        for i in range(len(chi2)):
            probs[i] = stats.ncx2.sf(chi2[i], degreesFreedom, lambda_nc)
    return probs

def update_magnitude(m_name_orig, m_value_orig, m_system_orig,
    the_runname):
    ''' Look up runname to check if a different magniude should be adopted'''

    fname_muptdate = "lum_anchor_update.dat"
    if os.path.isfile(fname_muptdate):
        the_rnames = np.genfromtxt(fname_muptdate, dtype='str', usecols=[0]).T
        if not the_runname in the_rnames:
            return m_name_orig, m_value_orig, m_system_orig
        updatefile = open(fname_muptdate, 'r')
        allines_mupdate = updatefile.readlines()
        for magline in allines_mupdate[1:]:
            maglinelist = magline.strip().split()
            if len(maglinelist) == 4:
                runname0 = maglinelist[0]
                if runname0 == the_runname:
                    m_name_new = maglinelist[1]
                    m_value_new = float(maglinelist[2])
                    m_system_new = maglinelist[3]
                    print('Adopting new luminosity anchor')
                    return m_name_new, m_value_new, m_system_new
            # Extra option to include an uncertainty on the magnitude
            elif len(maglinelist) == 5:
                runname0 = maglinelist[0]
                if runname0 == the_runname:
                    m_name_new = maglinelist[1]
                    m_value_new = float(maglinelist[2])
                    m_value_error = float(maglinelist[3])
                    m_system_new = maglinelist[4]
                    print('Adopting new luminosity anchor with uncertainty')
                    return m_name_new, m_value_new,\
                           m_system_new, m_value_error
            else:
                errstr = "ERROR IN " + fname_muptdate + '!'
                errstr = errstr + '\n press enter to use UNCHANGED values.'
                input(errstr)
    else:
        return m_name_orig, m_value_orig, m_system_orig


def radius_correction(df, fw_path, runname, thecontrolfile, theradiusfile,
    datapath, outpath, comp_fw):

    radcorrfile = outpath + 'radius_correction.txt'
    if os.path.isfile(radcorrfile):
        os.system('rm ' + radcorrfile)

    xbest = pd.Series.idxmin(df['rchi2'])
    best_model_name = df['run_id'][xbest]
    print('Best model:', best_model_name)
    best_gen_name = best_model_name.split('_')[0]
    bestmod_fw = fw_path + runname + '_' + best_model_name + '/'
    savemoddir = datapath + 'saved/' + best_gen_name + '/'
    the_best_indat = savemoddir + best_model_name + '/INDAT.DAT'
    if not os.path.isfile(bestmod_fw + 'FLUXCONT'):
        if comp_fw:
            os.system('mkdir -p ' + bestmod_fw)
            moddir = savemoddir + best_model_name + '/'
            modtar = savemoddir + best_model_name + '.tar.gz'
            if not os.path.isdir(moddir):
                os.system('mkdir -p ' + moddir)
                os.system('tar -xzf ' + modtar + ' -C ' + moddir + '/.')
            os.system('cp ' + the_best_indat + ' ' + fw_path + '.')
            fwindat = fw_path + 'INDAT.DAT'
            pnlte_logfile = (fw_path + runname + '_' + best_model_name
                + '.pnltelog')

            with open(fwindat) as f:
                lines = f.readlines()
            lines[0] = "'" + runname + '_' + best_model_name + "'\n"
            with open(fwindat, "w") as f:
                f.writelines(lines)

            for acontrl in np.genfromtxt(thecontrolfile,dtype='str'):
                if acontrl[0] == 'modelatom':
                    modelatom = acontrl[1]
                    break
            currentdir = os.getcwd()
            os.chdir(fw_path)
            print('Start FASTWIND Computation ...', 3*'\n...')
            print('    ... pnlte output here: ' + pnlte_logfile)
            runpnlte = './pnlte_' + modelatom + '.eo > ' + pnlte_logfile
            os.system(runpnlte)
            os.chdir(currentdir)
            if os.path.isfile(bestmod_fw + 'FLUXCONT'):
                corr_ready = True
                os.system('rm ' + pnlte_logfile)

            else:
                print('\n\n\nERROR! fw model could not compute, check!\n\n\n')
                corr_ready = False
        else:
            corr_ready = False
    else:
        corr_ready = True

    if corr_ready:
        lam, flam = get_fw_fluxcont(bestmod_fw)

        fwindat = fw_path + 'INDAT.DAT'

        rsun = 6.96e10 # cm
        tefffact = 0.9
        mod_rstar = float(open(fwindat, 'r').readlines()[3].strip().split()[-1])
        stellar_surface = 4*np.pi*(rsun*mod_rstar)**2

        m_name = np.genfromtxt(theradiusfile, dtype='str')[0]
        m_value = np.genfromtxt(theradiusfile)[1]
        m_system = np.genfromtxt(theradiusfile, dtype='str')[2]

        m_name, m_value, m_system = update_magnitude(m_name, m_value, m_system,
            runname)[:3]

        new_rad = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value, m_system,
            filterdir='filter_transmissions/')

        radius_ratio = new_rad/mod_rstar

        df['Q_radius_old'] = (10**df['mdot'])/(df['radius'])**(3./2.)

        # Correct all radii with the perc. correction from the best fit model.
        df['radius'] = df['radius']*radius_ratio

        # Correct mass loss rates by assuming a fixed Q value (Puls+96)
        df['mdot'] = np.log10(df['Q_radius_old']*(df['radius'])**(3./2.))

        df['q0'] = 10**df['logq0']
        df['logQ0'] = np.log10(df['q0']*4*np.pi*(rsun*df['radius'])**2)
        df['q1'] = 10**df['logq1']
        df['logQ1'] = np.log10(df['q1']*4*np.pi*(rsun*df['radius'])**2)
        df['q2'] = 10**df['logq2']
        df['logQ2'] = np.log10(df['q2']*4*np.pi*(rsun*df['radius'])**2)

        with open(radcorrfile, 'w') as f:
            f.write('# Radius corretion (= corrected radius/estimated radius\n')
            f.write(str(radius_ratio) + '\n')

    else:
        print("WARNING! No radius correction done. ")

    return df, the_best_indat

def get_uncertainties(df, dof_tot, npspec, param_names, param_space,
    deriv_pars, incl_deriv=True):

    if np.min(df['rchi2']) > RMSEA_THRESHOLD:
        which_statistic = 'RMSEA' # 'Pval_chi2' or 'Pval_ncchi2' or 'RMSEA'
    else:
        which_statistic = 'Pval_chi2'

    # Assign P-vaues and compute inverse reduced chi2
    df['invrchi2'] = 1./df['rchi2']
    df['norm_rchi2'] = df['rchi2']/np.min(df['rchi2'])

    df['RMSEA'] = np.sqrt((df['chi2']-dof_tot)/(dof_tot*(npspec-1)))
    minRMSEA = np.min(df['RMSEA'])
    # closefit_RMSEA = minRMSEA + 0.005
    closefit_RMSEA = minRMSEA
    lambda_nc = (closefit_RMSEA)**2 * dof_tot*(npspec-1)

    if which_statistic == 'Pval_ncchi2':
        df['P-value'] = calculateP_noncent(df['chi2'], dof_tot, lambda_nc)
    else:
        # ORIGINAL P-VALUE
        df['P-value'] = calculateP(df['chi2'], dof_tot, normalize=True)

    # Store the best fit parameters and 1 and 2 sig uncertainties in a dict
    params_error_1sig = {}
    params_error_2sig = {}

    xbest = pd.Series.idxmin(df['rchi2'])
    best_model_name = df['run_id'][xbest]
    best_gen_name = best_model_name.split('_')[0]

    if which_statistic in ('Pval_ncchi2', 'Pval_chi2'):
        min_p_1sig = 0.317
        min_p_2sig = 0.0455
        ind_1sig = df['P-value'] >= min_p_1sig
        ind_2sig = df['P-value'] >= min_p_2sig
    elif which_statistic == 'RMSEA':
        min_p_1sig = minRMSEA*1.04
        min_p_2sig = minRMSEA*1.09
        ind_1sig = df['RMSEA'] <= min_p_1sig
        ind_2sig = df['RMSEA'] <= min_p_2sig

    for i, aspace in zip(param_names, param_space):
        the_step_size = aspace[2]
        params_error_1sig[i] = [min(df[i][ind_1sig])-the_step_size,
            max(df[i][ind_1sig])+the_step_size, df[i][xbest]]
        params_error_2sig[i] = [min(df[i][ind_2sig])-the_step_size,
            max(df[i][ind_2sig])+the_step_size, df[i][xbest]]
    if incl_deriv:
        deriv_params_error_1sig = {}
        deriv_params_error_2sig = {}
        for i in (deriv_pars):
            deriv_params_error_1sig[i] = [min(df[i][ind_1sig]),
                max(df[i][ind_1sig]), df[i][xbest]]
            deriv_params_error_2sig[i] = [min(df[i][ind_2sig]),
                max(df[i][ind_2sig]), df[i][xbest]]

    # Read best model names (for plotting of line profiles)
    bestfamily_name = df['run_id'][ind_2sig].values

    if incl_deriv:
        best_uncertainty = (best_model_name, bestfamily_name, params_error_1sig,
            params_error_2sig, deriv_params_error_1sig, deriv_params_error_2sig,
            which_statistic)
    else:
        best_uncertainty = (best_model_name, bestfamily_name, params_error_1sig,
            params_error_2sig, which_statistic)

    return df,best_uncertainty


def propagate_uncertainty(value_dict, param_name, radius, delta_radius, power,
                          log=False):
    """
    Propagates the uncertainty on the radius to other parameters.
    Assumes the additional error term is normally distributed and small enough
    to be considered symmetric in powerlaws.
    value_dict assumes for each key the shape (lower_limit, upper_limit, best)
    param_name indicates the parameter to propagate error to.
    radius and delta_radius are the radius and extra uncertainty on it. Note
    that the delta_radius has to correspond to the desired nsigma.
    power is the power dependence on the radius for the parameter.
    """
    if log:
        best = 10**value_dict[param_name][2]
        low = best - 10**value_dict[param_name][0]
        up = 10**value_dict[param_name][1] - best
    else:
        best = value_dict[param_name][2]
        low = best - value_dict[param_name][0]
        up = best - value_dict[param_name][1]

    # The additional uncertainty term:
    extra_err = best * (delta_radius / radius) * power

    new_low = (extra_err**2 + low**2)**0.5
    new_up = (extra_err**2 + up**2)**0.5

    if log:
        value_dict[param_name][0] = np.log10(best - new_low)
        value_dict[param_name][1] = np.log10(best + new_up)
    else:
        value_dict[param_name][0] = best - new_low
        value_dict[param_name][1] = best + new_up

    return value_dict


def add_anchor_magnitude_uncertainty(df, runname, best_uncertainty,
                                     fw_path, theradiusfile):
    """
    Adds the uncertainty from the error in the absolute magnitude. The error is
    taken from the lum_anchor_update file. Performs error propagation to all
    relevant parameters.
    """
    # Get the original magnitude first
    m_name = np.genfromtxt(theradiusfile, dtype='str')[0]
    m_value = np.genfromtxt(theradiusfile)[1]
    m_system = np.genfromtxt(theradiusfile, dtype='str')[2]
    # try to update
    new_mag = update_magnitude(m_name, m_value, m_system, runname)
    if len(new_mag) == 4:
        m_name, m_value, m_system, m_error = new_mag
    else:
        print("Not adding anchor magnitude uncertainties!")
        return best_uncertainty

    # Check if a SED has been calculated for this run. Use that if available.
    # Otherwise use the default approximation for the radius.
    xbest = pd.Series.idxmin(df['rchi2'])
    best_model_name = df['run_id'][xbest]
    bestmod_fw = fw_path + runname + '_' + best_model_name + '/'
    if os.path.isfile(bestmod_fw + 'FLUXCONT'):
        lam, flam = get_fw_fluxcont(bestmod_fw)
        fwindat = fw_path + 'INDAT.DAT'

        rsun = 6.96e10 # cm
        mod_rstar = float(open(fwindat, 'r').readlines()[3].strip().split()[-1])
        stellar_surface = 4*np.pi*(rsun*mod_rstar)**2

        new_rad = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value, m_system)
        max_rad1 = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value - m_error, m_system)
        max_rad2 = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value - 2 * m_error, m_system)
        min_rad1 = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value + m_error, m_system)
        min_rad2 = magnitude_to_radius_SED(lam, flam/stellar_surface,
            m_name, m_value + 2 * m_error, m_system)

    else:
        print("No SED for radius correction found, using approximation")
        new_rad = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value, m_system)
        max_rad1 = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value - m_error, m_system)
        max_rad2 = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value - 2*m_error, m_system)
        min_rad1 = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value + m_error, m_system)
        min_rad2 = m2r.magnitude_to_radius(df['teff'][xbest],
                                          m_name, m_value + 2*m_error, m_system)

    delta_radius1 = (max_rad1 - min_rad1) * 0.5
    delta_radius2 = (max_rad2 - min_rad2) * 0.5

    # Unpack uncertainties
    best_model_name, bestfamily_name, pars_err1, pars_err2, d_pars_err1,\
        d_pars_err2, which_statistic = best_uncertainty

    # All the parameters that need to have their uncertainty updated based on
    # the changed radius uncertainty. The number are the power of the dependence
    # on the radius. The bool indicates if the parameter is used in log
    all_derived_parameters = (("radius", 1.0, False),
                              ("logL", 2.0, True),
                              ("logQ0", 2.0, True),
                              ("logQ1", 2.0, True),
                              ("logQ2", 2.0, True),
                              ("Mspec", 2.0, False),
                              ("vesc_eff", 0.5, False),
                              ("vinf_vesc", 0.5, False),
                              ("mdot_fclump", 1.5, True))

    for dpar, power, log in all_derived_parameters:
        if dpar in d_pars_err1:
            d_pars_err1 = propagate_uncertainty(d_pars_err1, dpar, new_rad,
                                                delta_radius1, power, log=log)
            d_pars_err2 = propagate_uncertainty(d_pars_err2, dpar, new_rad,
                                                delta_radius2, power, log=log)

    # mdot is done separately, because it is not a derived parameter.
    if "mdot" in pars_err1:
        pars_err1 = propagate_uncertainty(pars_err1, "mdot", new_rad,
                                          delta_radius1, 1.5, log=True)
        pars_err2 = propagate_uncertainty(pars_err2, "mdot", new_rad,
                                          delta_radius2, 1.5, log=True)

    # returns the same best_uncertainty tuple, but with updated values
    return best_model_name, bestfamily_name, pars_err1, pars_err2, d_pars_err1,\
        d_pars_err2, which_statistic

def titlepage(df, runname, params_error_1sig, params_error_2sig,
    the_pdf, param_names, maxgen, nind, linedct, which_sigma,
    deriv_params_error_1sig, deriv_params_error_2sig, deriv_pars):
    """
    Make a page with best fit parameters and errors
    """

    ncrash = len(df.copy()[df['chi2'] == 999999999])
    ntot = len(df)
    perccrash = round(100.0*ncrash/ntot,1)
    minrchi2 = round(np.min(df['rchi2']),2)
    nlines = len(linedct['name'])

    fig, ax = plt.subplots(2,2,figsize=(12.5, 12.5),
        gridspec_kw={'height_ratios': [0.5, 3], 'width_ratios': [2, 8]})

    # Not catch all, but catch most solution
    path_to_ga = sys.argv[0].strip("GA_analysis.py")
    if os.path.isfile(path_to_ga + 'kiwi.jpg'):
        ax[0,0].imshow(mpimg.imread(path_to_ga + 'kiwi.jpg'))

    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')

    boldtext = {'ha':'left', 'va':'top', 'weight':'bold'}
    normtext = {'ha':'left', 'va':'top'}
    offs = 0.12
    yvalmax = 0.9
    ax[0,1].text(0.0, yvalmax, 'Run name', **boldtext)
    ax[0,1].text(0.25, yvalmax, runname, **normtext)
    ax[0,1].text(0.0, yvalmax-1*offs, 'Best rchi2', **boldtext)
    ax[0,1].text(0.25, yvalmax-1*offs, str(minrchi2), **normtext)
    ax[0,1].text(0.0, yvalmax-2*offs, 'Generations', **boldtext)
    ax[0,1].text(0.25, yvalmax-2*offs, str(maxgen), **normtext)
    ax[0,1].text(0.0, yvalmax-3*offs, 'Individuals per gen', **boldtext)
    ax[0,1].text(0.25, yvalmax-3*offs, str(nind), **normtext)
    ax[0,1].text(0.0, yvalmax-4*offs, 'Total # models', **boldtext)
    ax[0,1].text(0.25, yvalmax-4*offs, str(ntot), **normtext)
    ax[0,1].text(0.0, yvalmax-5*offs, 'Crashed models', **boldtext)
    ax[0,1].text(0.25, yvalmax-5*offs, str(perccrash) + '%', **normtext)
    ax[0,1].text(0.0, yvalmax-6*offs, 'Number of lines', **boldtext)
    ax[0,1].text(0.25, yvalmax-6*offs, str(nlines), **normtext)

    if which_sigma == 2:
        psig = params_error_2sig
        deriv_psig = deriv_params_error_2sig
    else:
        psig = params_error_1sig
        deriv_psig = deriv_params_error_1sig

    offs = 0.02
    yvalmax = 1.0
    secndcol = 0.15
    ax[1,1].text(0.0, yvalmax, 'Parameter', weight='bold')
    ax[1,1].text(secndcol, yvalmax, 'Best', weight='bold')
    ax[1,1].text(secndcol*2, yvalmax, '-' + str(which_sigma)
        + r'$\mathbf{\sigma}$', weight='bold')
    ax[1,1].text(secndcol*3, yvalmax, '+' + str(which_sigma)
        + r'$\mathbf{\sigma}$', weight='bold')
    ax[1,1].text(secndcol*4, yvalmax,
        r'Min (' + str(which_sigma) + r'$\mathbf{\sigma}$)', weight='bold')
    ax[1,1].text(secndcol*5, yvalmax,
        r'Max (' + str(which_sigma) + r'$\mathbf{\sigma}$)', weight='bold')
    for paramname in param_names:
        yvalmax = yvalmax - offs
        ax[1,1].text(0.0, yvalmax, paramname)
        ax[1,1].text(secndcol, yvalmax,
            round(psig[paramname][2],3))
        ax[1,1].text(secndcol*2, yvalmax,
            round(psig[paramname][2]-psig[paramname][0],3))
        ax[1,1].text(secndcol*3, yvalmax,
            round(psig[paramname][1]-psig[paramname][2],3))
        ax[1,1].text(secndcol*4, yvalmax,
            round(psig[paramname][0],3))
        ax[1,1].text(secndcol*5, yvalmax,
            round(psig[paramname][1],3))

    yvalmax = yvalmax - offs
    for paramname in deriv_pars:
        yvalmax = yvalmax - offs
        ax[1,1].text(0.0, yvalmax, paramname)
        ax[1,1].text(secndcol, yvalmax,
            round(deriv_psig[paramname][2],3))
        ax[1,1].text(secndcol*2, yvalmax,
            round(deriv_psig[paramname][2]-deriv_psig[paramname][0],3))
        ax[1,1].text(secndcol*3, yvalmax,
            round(deriv_psig[paramname][1]-deriv_psig[paramname][2],3))
        ax[1,1].text(secndcol*4, yvalmax,
            round(deriv_psig[paramname][0],3))
        ax[1,1].text(secndcol*5, yvalmax,
            round(deriv_psig[paramname][1],3))

    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    return the_pdf


def titlepage_latex(df, runname, params_error_1sig, params_error_2sig,
    the_pdf, param_names, maxgen, nind, linedct, which_sigma,
    deriv_params_error_1sig, deriv_params_error_2sig, deriv_pars):
    """
    Make a page with best fit parameters and errors
    """
    plt.rcParams['text.usetex'] = True

    ncrash = len(df.copy()[df['chi2'] == 999999999])
    ntot = len(df)
    perccrash = round(100.0*ncrash/ntot,1)
    minrchi2 = round(np.min(df['rchi2']),2)
    nlines = len(linedct['name'])

    fig, ax = plt.subplots(2,2,figsize=(12.5, 12.5),
        gridspec_kw={'height_ratios': [0.5, 3], 'width_ratios': [2, 8]})

    # Not catch all, but catch most solution
    path_to_ga = sys.argv[0].strip("GA_analysis.py")
    if os.path.isfile(path_to_ga + 'kiwi.jpg'):
        ax[0,0].imshow(mpimg.imread(path_to_ga + 'kiwi.jpg'))

    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')

    boldtext = {'ha':'left', 'va':'top', 'weight':'bold'}
    normtext = {'ha':'left', 'va':'top'}
    offs = 0.12
    yvalmax = 0.9
    ax[0,1].text(0.0, yvalmax, r'{\bf Run name}', **boldtext)
    ax[0,1].text(0.25, yvalmax, runname, **normtext)
    ax[0,1].text(0.0, yvalmax-1*offs, r'{\bf Best rchi2}', **boldtext)
    ax[0,1].text(0.25, yvalmax-1*offs, str(minrchi2), **normtext)
    ax[0,1].text(0.0, yvalmax-2*offs, r'{\bf Generations}', **boldtext)
    ax[0,1].text(0.25, yvalmax-2*offs, str(maxgen), **normtext)
    ax[0,1].text(0.0, yvalmax-3*offs, r'{\bf Individuals per gen}', **boldtext)
    ax[0,1].text(0.25, yvalmax-3*offs, str(nind), **normtext)
    ax[0,1].text(0.0, yvalmax-4*offs,r'{\bf Total number of models}',**boldtext)
    ax[0,1].text(0.25, yvalmax-4*offs, str(ntot), **normtext)
    ax[0,1].text(0.0, yvalmax-5*offs, r'{\bf Crashed models}', **boldtext)
    ax[0,1].text(0.25, yvalmax-5*offs, str(perccrash) + '\%', **normtext)
    ax[0,1].text(0.0, yvalmax-6*offs, r'{\bf Number of lines}', **boldtext)
    ax[0,1].text(0.25, yvalmax-6*offs, str(nlines), **normtext)

    if which_sigma == 2:
        psig = params_error_2sig
        deriv_psig = deriv_params_error_2sig
    else:
        psig = params_error_1sig
        deriv_psig = deriv_params_error_1sig

    table_text = r"\begin{tabular}{l|r|rr} "
    table_text += r"{\bf Parameter} & {\bf Value} & {\bf min %i$\sigma$} &" \
        r" {\bf max %i$\sigma$} \rule{0pt}{2.6ex} \\ \hline " % (which_sigma,
                                                                 which_sigma)
    for paramname in param_names:
        table_text += r"%s & $%s_{-%s}^{+%s}$ & $%s$ & $%s$ \rule{0pt}{2.6ex}" \
                      r" \\ " % (
            paramname,
            np.format_float_positional(psig[paramname][2],
                                       trim="-", precision=2),
            np.format_float_positional(psig[paramname][2] - psig[paramname][0],
                                       trim="-", precision=2),
            np.format_float_positional(psig[paramname][1] - psig[paramname][2],
                                       trim="-", precision=2),
            np.format_float_positional(psig[paramname][0],
                                       trim="-", precision=2),
            np.format_float_positional(psig[paramname][1],
                                       trim="-", precision=2))
    table_text += r"\hline "
    for paramname in deriv_pars:
        table_text += r"%s & $%s_{-%s}^{+%s}$ & $%s$ & $%s$ \rule{0pt}{2.6ex}" \
                      r" \\ " % (
fix_latex(paramname),
np.format_float_positional(deriv_psig[paramname][2], trim="-", precision=2),
np.format_float_positional(deriv_psig[paramname][2] - deriv_psig[paramname][0],
                           trim="-", precision=2),
np.format_float_positional(deriv_psig[paramname][1] - deriv_psig[paramname][2],
                           trim="-", precision=2),
np.format_float_positional(deriv_psig[paramname][0], trim="-", precision=2),
np.format_float_positional(deriv_psig[paramname][1], trim="-", precision=2))

    table_text += r"\end{tabular}"
    ax[1,1].text(0, 0.5, table_text, ha="left", va="center")

    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    # Stop using latex rendering to not mess with any other plots
    plt.rcParams['text.usetex'] = False
    return the_pdf


def fix_latex(string):
    """
    removes underscores
    """
    string = string.replace("_", " ")
    return string


def fitnessplot(df, yval, params_error_1sig, params_error_2sig,
    the_pdf, param_names, param_space, maxgen,
    which_cmap=plt.cm.viridis, save_jpg=False, df_tot=[]):

    """
    Plot the fitness as a function of each free parameter. This function
    can be used for plotting the P-value, 1/rchi2 of all lines combined,
    or for the fitness of individual lines (1/rchi2)
    """

    # Only consider models that have not crashed
    df = df[df['chi2'] < 999999999]

    # Prepare colorbar
    cmap = which_cmap
    bounds = np.linspace(0, maxgen+1, maxgen+2)
    norm = matplotlib.colors.BoundaryNorm(bounds, int(cmap.N*0.8))

    # Set up figure dimensions and subplots
    ncols = 5
    # nrows len(param_names)+1 to ensure space for the colorbar
    nrows =int(math.ceil(1.0*(len(param_names)+1)/ncols))
    nrows =max(nrows, 2)
    ccol = ncols - 1
    crow = -1
    figsizefact = 2.5
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, figsizefact*nrows),
        sharey=True)

    # Loop through parameters
    for i in range(ncols*nrows):

        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1

        if i >= len(param_names):
            ax[crow,ccol].axis('off')
            continue

        # Make actual plots
        ax[crow,ccol].set_title(param_names[i])
        if len(param_space) > 0:
            ax[crow,ccol].set_xlim(param_space[i][0], param_space[i][1])
        elif param_names[i] == 'Gamma_Edd':
            ax[crow,ccol].set_xlim(0,1.0)
        elif param_names[i] == 'vinf_vesc':
            ax[crow,ccol].set_xlim(0,10.0)
        scat0 = ax[crow,ccol].scatter(df[param_names[i]], df[yval],
            c=df['gen'], cmap=cmap, norm=norm, s=10)

        min1sig = params_error_1sig[param_names[i]][0]
        max1sig = params_error_1sig[param_names[i]][1]
        min2sig = params_error_2sig[param_names[i]][0]
        max2sig = params_error_2sig[param_names[i]][1]
        bestfit = params_error_2sig[param_names[i]][2]
        # if not save_jpg:
        ax[crow,ccol].axvline(bestfit, color='orangered', lw=1.5)
        ax[crow,ccol].axvspan(min1sig, max1sig, color='gold',
            alpha=0.70, zorder=0)
        ax[crow,ccol].axvspan(min2sig, max2sig, color='gold',
            alpha=0.25, zorder=0)
        ax[crow, ccol].set_rasterized(True)

        # Set y-labels
        if ccol == 0:
            if yval == 'P-value':
                ax[crow,ccol].set_ylabel('P-value')
            elif yval == 'fitness':
                ax[crow,ccol].set_ylabel('Fitness')
            else:
                ax[crow,ccol].set_ylabel(r'1/$\chi^2_{\rm red}$')

        if len(df_tot) > 0:
            ax[crow,ccol].set_ylim(-0.05*np.max(df_tot[yval]),
                np.max(df_tot[yval])*1.10)
        else:
            ax[crow,ccol].set_ylim(-0.05*np.max(df[yval]),
                np.max(df[yval])*1.10)

    # Colorbar
    cbar = plt.colorbar(scat0, orientation='horizontal', ax=ax[-1,-1])
    cbar.ax.set_title('Generation')

    # Set title
    if yval in ('invrchi2', 'P-value'):
        if len(param_space) > 0:
            plt.suptitle('All lines')
        else:
            plt.suptitle('All lines (derived parameters)')
    else:
        if len(param_space) > 0:
            plt.suptitle(yval)
        else:
            plt.suptitle(yval + ' (derived parameters)')

    # Tight layout and save plot
    plt.tight_layout()
    if nrows == 2:
        plt.subplots_adjust(0.07, 0.07, 0.93, 0.85)
    else:
        plt.subplots_adjust(0.07, 0.07, 0.93, 0.90)
    if not save_jpg:
        if yval in ('invrchi2', 'P-value'):
            the_pdf.savefig(dpi=150)
        else:
            the_pdf.savefig(dpi=100)
        plt.close()

        return the_pdf
    else:
        return fig, ax


def fitnessplot_per_parameter(df, xval, params_error_1sig, params_error_2sig,
    the_pdf, line_names, param_space, maxgen,
    which_cmap=plt.cm.viridis, save_jpg=False, df_tot=[]):

    """
    Plot the fitness as a function for each line for a given free parameter.
    This function can be used for plotting the P-value, 1/rchi2 of all lines
    combined, or for the fitness of individual lines (1/rchi2)
    Also plots the fitness of the given parameter for all lines,
    and the sample density.
    """

    # Only consider models that have not crashed
    df_crash = df[df['chi2'] == 999999999]
    df = df[df['chi2'] < 999999999]

    # Prepare colorbar
    cmap = which_cmap
    bounds = np.linspace(0, maxgen+1, maxgen+2)
    norm = matplotlib.colors.BoundaryNorm(bounds, int(cmap.N*0.8))

    # Determine the colors of the stacked histogram.
    scalarmap = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    scalarmap._A = []
    colors = [scalarmap.to_rgba(gen) for gen in range(maxgen)]

    line_names = ["invrchi2"] + list(line_names) + [xval]
    # Set up figure dimensions and subplots
    ncols = 5
    # nrows len(param_names)+1 to ensure space for the colorbar
    nrows =int(math.ceil(1.0*(len(line_names)+1)/ncols))
    nrows =max(nrows, 2)
    ccol = ncols - 1
    crow = -1
    figsizefact = 2.5
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, figsizefact*nrows), sharey=False)

    # Loop through lines
    for i in range(ncols*nrows):

        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1

        if i >= (len(line_names)):
            ax[crow,ccol].axis('off')
            continue

        # Make actual plots
        if line_names[i] == "invrchi2":
            ax[crow,ccol].set_title("All lines")
        elif line_names[i] == xval:
            ax[crow,ccol].set_title("Sample density")
        else:
            ax[crow,ccol].set_title(line_names[i])
        if len(param_space) > 0:
            ax[crow,ccol].set_xlim(param_space[0], param_space[1])

        if xval != line_names[i]:
            scat0 = ax[crow,ccol].scatter(df[xval], df[line_names[i]],
                c=df['gen'], cmap=cmap, norm=norm, s=10)
            df_tmp_best = df.copy().sort_values(by=[line_names[i]],
                ascending=False)
            thebestval = df_tmp_best[xval].iloc[0]
            ax[crow,ccol].text(0.95, 0.95,thebestval,
                ha='right', va='top', transform=ax[crow,ccol].transAxes)
        else:
            bins = np.arange(param_space[0] - 0.5 * param_space[2],
                             param_space[1] + 0.5 * param_space[2],
                             param_space[2])
            hist_data = [df[xval][df['gen'] == i] for i in range(maxgen)]
            ax[crow,ccol].hist(hist_data, bins=bins, density=True, color=colors,
                               stacked=True)
            ax[crow,ccol].hist(df_crash[xval], histtype="step", color="red",
                               density=True, bins=bins)

        min1sig = params_error_1sig[xval][0]
        max1sig = params_error_1sig[xval][1]
        min2sig = params_error_2sig[xval][0]
        max2sig = params_error_2sig[xval][1]
        bestfit = params_error_2sig[xval][2]

        ax[crow,ccol].axvline(bestfit, color='orangered', lw=1.5)
        ax[crow,ccol].axvspan(min1sig, max1sig, color='gold',
            alpha=0.70, zorder=0)
        ax[crow,ccol].axvspan(min2sig, max2sig, color='gold',
            alpha=0.25, zorder=0)

        # Set y-labels
        if ccol == 0:
            ax[crow,ccol].set_ylabel(r'1/$\chi^2_{\rm red}$')

        if xval != line_names[i]:
            if len(df_tot) > 0:
                ax[crow,ccol].set_ylim(-0.05*np.max(df_tot[line_names[i]]),
                    np.max(df_tot[line_names[i]])*1.10)
            else:
                ax[crow,ccol].set_ylim(-0.05*np.max(df[line_names[i]]),
                    np.max(df[line_names[i]])*1.10)
        ax[crow, ccol].set_rasterized(True)


    # Colorbar
    cbar = plt.colorbar(scat0, orientation='horizontal', ax=ax[-1,-1])
    cbar.ax.set_title('Generation')

    # Set title
    if xval in ('invrchi2', 'P-value'):
        if len(param_space) > 0:
            plt.suptitle('All lines')
        else:
            plt.suptitle('All lines (derived parameters)')
    else:
        if len(param_space) > 0:
            plt.suptitle(xval)
        else:
            plt.suptitle(xval + ' (derived parameters)')

    # Tight layout and save plot
    plt.tight_layout()
    if nrows == 2:
        plt.subplots_adjust(0.07, 0.07, 0.93, 0.85)
    else:
        plt.subplots_adjust(0.07, 0.07, 0.93, 0.90)

    if not save_jpg:
        the_pdf.savefig(dpi=100)
        plt.close("all")

        return the_pdf
    else:
        return fig, ax

def lineprofiles(df, spectdct, linedct, savedmoddir,
    best_model_name, bestfamily_name, the_pdf, plotlineprofdir,
    extra_fwmod, extra_mod, save_jpg=False):
    """
    Create plot with line profiles of best fitting models.
    In the background, plot the data.
    """

    nlines = len(linedct['name'])

    # Untar best fitting models.
    plotmoddirlist = []
    for amod in bestfamily_name:
        moddir = savedmoddir + amod.split('_')[0] + '/' + amod + '/'
        modtar = savedmoddir + amod.split('_')[0] + '/' + amod + '.tar.gz'
        if not os.path.isdir(moddir):
            os.system('mkdir -p ' + moddir)
            os.system('tar -xzf ' + modtar + ' -C ' + moddir + '/.')
        plotmoddirlist.append(moddir)

    bestmoddir = (savedmoddir + best_model_name.split('_')[0] +
        '/' + best_model_name +  '/')

    # Set up figure dimensions and subplots
    ncols = 5
    nrows =int(math.ceil(1.0*nlines/ncols))
    nrows =max(nrows, 2)
    ccol = ncols - 1
    crow = -1
    figsizefact = 2.5
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, 0.7*figsizefact*nrows))

    # Loop through parameters
    for i in range(ncols*nrows):

        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1

        if i >= nlines:
            ax[crow,ccol].axis('off')
            continue

        # Read data per line
        keep_idx = [(spectdct['wave'] > linedct['left'][i]) &
            (spectdct['wave'] < linedct['right'][i])]
        wave_tmp = spectdct['wave'][tuple(keep_idx)]
        flux_tmp = spectdct['flux'][tuple(keep_idx)]
        err_tmp = spectdct['err'][tuple(keep_idx)]

        # Read profile of best fitting model
        best_prof_file = bestmoddir + linedct['name'][i] + '.prof.fin'
        bestmodwave, bestmodflux = np.genfromtxt(best_prof_file).T

        # Read profiles of best fitting family of models
        lineflux_min = np.copy(bestmodflux)
        lineflux_max = np.copy(bestmodflux)
        if len(plotmoddirlist) > 0:
            for asig_mod in plotmoddirlist:
                fam_prof_file = asig_mod + linedct['name'][i] + '.prof.fin'
                smwave, smflux = np.genfromtxt(fam_prof_file).T

                lineflux_min = np.min(np.array([lineflux_min, smflux]), axis=0)
                lineflux_max = np.max(np.array([lineflux_max, smflux]), axis=0)

        lineprof_arr = np.array([bestmodwave, bestmodflux, lineflux_min,
            lineflux_max]).T
        lineprof_arr_head = 'wave bestflux minflux maxflux'
        np.savetxt(plotlineprofdir + linedct['name'][i] + '.txt', lineprof_arr,
            header=lineprof_arr_head)

        # Make actual plots
        ax[crow,ccol].set_title(linedct['name'][i])
        ax[crow,ccol].axhline(1.0, color='black', lw=0.8)
        ax[crow,ccol].errorbar(wave_tmp, flux_tmp, yerr=err_tmp,
            fmt='o', color='black', ms=0)
        ax[crow,ccol].fill_between(bestmodwave, lineflux_min, lineflux_max,
            color='#8cd98c', alpha=0.7)
        ax[crow,ccol].plot(bestmodwave, bestmodflux, color='#1ca641', lw=2.4,
            alpha=1.0)
        ax[crow,ccol].set_xlim(linedct['left'][i], linedct['right'][i])
        ax[crow,ccol].set_ylim(*ax[crow,ccol].get_ylim())

        # plot an extra fastwind model
        if not extra_fwmod == '/':
            extramfile = extra_fwmod + linedct['name'][i] + '.prof'
            if os.path.isfile(extramfile):
                em_wave, em_flux = np.genfromtxt(extramfile).T
                ax[crow,ccol].plot(em_wave, em_flux, color='dodgerblue', lw=2.4)
                ax[crow,ccol].plot(bestmodwave, bestmodflux, color='red',
                    lw=2.4,alpha=1.0)
            else:
                print(extramfile, 'not found')
                ax[crow,ccol].plot(bestmodwave, bestmodflux, color='orangered',
                    lw=2.4, alpha=1.0)

        # plot another
        if not extra_mod == '':
            if os.path.isfile(extra_mod):
                em_wave, em_flux = np.genfromtxt(extra_mod).T
                ax[crow,ccol].plot(em_wave, em_flux, color='blue', lw=2.4)
                ax[crow,ccol].plot(bestmodwave, bestmodflux, color='orangered',
                    lw=2.4, alpha=1.0)
            else:
                print(extra_mod, 'not found')

    # Tight layout and save plot
    plt.tight_layout()

    if not save_jpg:
        the_pdf.savefig(dpi=150)
        plt.close()
        return the_pdf
    else:
        return fig, ax

def correlationplot(the_pdf, df, corrpars):
    """
    Create a correlation plot of the parameters in the list corrpars.
    """

    orig_corrpar = corrpars.copy()

    for par in orig_corrpar:
        if par not in df.columns:
            corrpars.remove(par)

    dfs = df.sort_values(by=['invrchi2'])

    # Set up figure dimensions and subplots
    ncols = len(corrpars)
    nrows = ncols
    hratios = 30*np.ones(ncols)
    wratios = 30*np.ones(ncols)
    hratios[0] = 1.0
    wratios[-1] = 1.0
    figsizefact = 2.0
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, figsizefact*nrows),
            sharex='col', sharey='row',
            gridspec_kw={'height_ratios': hratios, 'width_ratios': wratios})


    if ncols == 1:
        ax = np.array([[ax]])
    elif ncols == 0:
        plt.close()
        return the_pdf

    # Loop through parameters to create correlation plot
    pairlist = []
    for ccol in range(ncols):
        for crow in range(nrows):
            pc1 = corrpars[ccol]
            pc2 = corrpars[crow]
            pair = [pc1, pc2]
            if (pc1 == pc2) or (pair in pairlist):
                ax[crow,ccol].axis('off')
            else:
                ax[crow,ccol].scatter(dfs[pc1], dfs[pc2],
                    c=dfs['invrchi2'],s=10, rasterized=True)
                pairlist.append(pair)
                pairlist.append(pair[::-1])

            ax[crow,ccol].set_xlim(np.min(dfs[pc1]), np.max(dfs[pc1]))
            ax[crow,ccol].set_ylim(np.min(dfs[pc2]), np.max(dfs[pc2]))
    # Label axes
    for i in range(0, ncols-1):
        ax[-1,i].set_xlabel(corrpars[i])
        ax[i+1, 0].set_ylabel(corrpars[i+1])

    # Tight layout and save plot
    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    return the_pdf

def get_fwmaxtime(controlfile):
    dct = fw.read_control_pars(controlfile)
    timeoutstr = dct['fw_timeout']
    if timeoutstr.endswith('m'):
        timeout = float(timeoutstr[:-1])*60
    else:
        print('Timeout string not given in minutes, exiting')
        sys.exit()
    return timeout

def fw_performance(the_pdf, df, controlfile):
    """
    Show maximum interations, convergence and run time of FW models.
    """

    # Pick up fastwind timeout to assign a number to the runs that ran to max
    fw_timeout = get_fwmaxtime(controlfile)
    fw_timeout_min = 1.0*fw_timeout/60.0
    df.loc[(df['cputime'] == 99999.9), 'cputime'] = fw_timeout

    # Only consider models that can generate line profiles
    df = df[df['chi2'] < 999999999]
    df = df[df['maxcorr'] > 0.0]
    df['cputime_min'] = 1.0*df['cputime'].values/60.0

    nb = 101
    bins_maxit = np.linspace(0, 100, nb)
    bins_maxco = np.linspace(-3, 1.5, nb)
    bins_ticpu = np.linspace(0, fw_timeout_min, nb)

    fig, ax = plt.subplots(2,3, figsize=(12,6.5))
    ax[0,0].hist(df['maxit'], bins_maxit,
        color='#2b0066', alpha=0.7)
    ax[0,1].hist(np.log10(df['maxcorr']), bins_maxco,
        color='#009c60', alpha=0.7)
    ax[0,2].hist(df['cputime_min'], bins_ticpu,
        color='#b5f700', alpha=0.7)

    ax[0,0].set_xlabel('Maximum iteration')
    ax[0,1].set_xlabel('log(Maximum correction)')
    ax[0,2].set_xlabel('CPU-time (minutes)')
    ax[0,0].set_ylabel('Count')
    ax[0,1].set_ylabel('Count')
    ax[0,2].set_ylabel('Count')

    sct1 = ax[1,0].scatter(np.log10(df['maxcorr']), df['maxit'],
        s=6, c=df['cputime']/60.0, rasterized=True)
    ax[1,0].set_xlabel('log(Maximum correction)')
    ax[1,0].set_ylabel('Maximum iteration')
    cbar1 = plt.colorbar(sct1, ax=ax[1,0])
    cbar1.ax.set_title(r'CPU-time (min)', fontsize=9)

    sct2 = ax[1,1].scatter(np.log10(df['maxcorr']), df['cputime']/60.0,
        s=6, c=df['maxit'], rasterized=True)
    ax[1,1].set_xlabel('log(Maximum correction)')
    ax[1,1].set_ylabel('CPU-time (minutes)')
    cbar2 = plt.colorbar(sct2, ax=ax[1,1])
    cbar2.ax.set_title(r'Max. iteration', fontsize=9)

    sct3 = ax[1,2].scatter(df['cputime']/60.0, df['maxit'],
        s=4, c=np.log10(df['maxcorr']), rasterized=True)
    ax[1,2].set_xlabel('CPU-time (minutes)')
    ax[1,2].set_ylabel('Maximum iteration')
    cbar3 = plt.colorbar(sct3, ax=ax[1,2])
    cbar3.ax.set_title(r'log(Max. corr.)', fontsize=9)

    # Tight layout and save plot
    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    return the_pdf

def convergence(the_pdf, df_orig, dof_tot, npspec, param_names, param_space,
    deriv_pars, maxgen, runname, fw_path, thecontrolfile,
    theradiusfile, datapath):

    evol_list_best = []
    evol_list_1sig_up = []
    evol_list_1sig_down = []
    evol_list_2sig_up = []
    evol_list_2sig_down = []
    for apar in param_names:
        evol_list_best.append([])
        evol_list_1sig_up.append([])
        evol_list_1sig_down.append([])
        evol_list_2sig_up.append([])
        evol_list_2sig_down.append([])

    for the_max in range(1,maxgen):
        df_tmp = df_orig.copy()[df_orig['gen'] < the_max]

        # Compute uncertainties
        df_tmp, best_uncertainty = get_uncertainties(df_tmp, dof_tot,
            npspec, param_names, param_space, deriv_pars, incl_deriv=False)

        # Unpack all computed values
        best_model_name, bestfamily_name, params_error_1sig, \
            params_error_2sig, which_statistic = best_uncertainty

        for ipar in range(len(param_names)):
            pname = param_names[ipar]
            evol_list_best[ipar].append(params_error_1sig[pname][2])
            evol_list_1sig_up[ipar].append(params_error_1sig[pname][1])
            evol_list_1sig_down[ipar].append(params_error_1sig[pname][0])
            evol_list_2sig_up[ipar].append(params_error_2sig[pname][1])
            evol_list_2sig_down[ipar].append(params_error_2sig[pname][0])

    x_gen = range(len(evol_list_best[0]))
    # fig, ax = plt.subplots(1, len(param_names))

    # Set up figure dimensions and subplots
    ncols = 3
    nrows =int(math.ceil(1.0*(len(param_names))/ncols))
    nrows =max(nrows, 2)
    ccol = ncols - 1
    crow = -1
    figsizefact = 4.0
    fig, ax = plt.subplots(nrows, ncols,
        figsize=(figsizefact*ncols, 0.3*figsizefact*nrows), sharex=True)

    # Loop through parameters
    for i in range(ncols*nrows):

        if ccol == ncols - 1:
            ccol = 0
            crow = crow + 1
        else:
            ccol = ccol + 1
        if crow == nrows -1:
            ax[crow,ccol].set_xlabel('Generation')

        if i >= len(param_names):
            ax[crow,ccol].axis('off')
            continue

        ax[crow,ccol].plot(x_gen, evol_list_best[i], color='red')
        ax[crow,ccol].fill_between(x_gen, evol_list_1sig_down[i],
            evol_list_1sig_up[i], color='gold', alpha=0.70)
        ax[crow,ccol].fill_between(x_gen, evol_list_2sig_down[i],
            evol_list_2sig_up[i], color='gold', alpha=0.25)
        ax[crow,ccol].set_ylim(param_space[i][0], param_space[i][1])
        ax[crow,ccol].set_ylabel(param_names[i])

    plt.tight_layout()
    the_pdf.savefig(dpi=150)
    plt.close()

    return the_pdf

def save_bestvals(param_names, deriv_pars, params_error_1sig, params_error_2sig,
    deriv_params_error_1sig, deriv_params_error_2sig, savebest_txt):
    """
    Save best fit parameters and errors to text file
    """

    if os.path.isfile(savebest_txt):
        os.system('rm ' + savebest_txt)

    write_lines = []
    rv = 4
    lj = 10
    lj0 = 15
    for apar in param_names:
        bestfit = params_error_2sig[apar][2]
        low1sig = str(round(bestfit - params_error_1sig[apar][0], rv))
        up1sig = str(round(params_error_1sig[apar][1] - bestfit, rv))
        low2sig = str(round(bestfit - params_error_2sig[apar][0], rv))
        up2sig = str(round(params_error_2sig[apar][1] - bestfit, rv))
        bestfit = str(round(bestfit,rv))
        savestr = (apar.ljust(lj0) + ' ' + bestfit.ljust(lj) + ' '
            + low1sig.ljust(lj) + ' ' + up1sig.ljust(lj) + ' '
            + low2sig.ljust(lj) + ' ' + up2sig.ljust(lj))
        write_lines.append(savestr)

    for apar in deriv_pars:
        bestfit = deriv_params_error_2sig[apar][2]
        low1sig = str(round(bestfit - deriv_params_error_1sig[apar][0], rv))
        up1sig = str(round(deriv_params_error_1sig[apar][1] - bestfit, rv))
        low2sig = str(round(bestfit - deriv_params_error_2sig[apar][0], rv))
        up2sig = str(round(deriv_params_error_2sig[apar][1] - bestfit, rv))
        bestfit = str(round(bestfit,rv))
        savestr = (apar.ljust(lj0) + ' ' + bestfit.ljust(lj) + ' '
            + low1sig.ljust(lj) + ' ' + up1sig.ljust(lj) + ' '
            + low2sig.ljust(lj) + ' ' + up2sig.ljust(lj))
        write_lines.append(savestr)

    with open(savebest_txt, 'a') as myfile:
        myfile.write('# Parameter'.ljust(lj0) + ' best'.ljust(lj+1) +
            ' low1sig'.ljust(lj+1) + ' up1sig'.ljust(lj+1) +
            ' low2sig'.ljust(lj+1) + ' up2sig' + '\n')
        for aline in write_lines:
            myfile.write(aline)
            myfile.write('\n')

















#
