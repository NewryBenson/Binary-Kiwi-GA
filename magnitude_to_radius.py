# Estimate the stellar radius given an effective temperature
# and magnitude. Requires filter transmissions and zero points.
# Created by Sarah Brands on 20 Dec 2019 (s.a.brands@uva.nl)

import __future__
import numpy as np
import sys
from scipy.interpolate import interp1d


def magnitude_to_radius(teff, band, obsmag, zp_system, Tfrac=0.9,
    filterdir='filter_transmissions/'):

    '''Estimate the radius of the star given a temperature,
    photometric filter and observed (dereddened) absolute
    magnitude.

    Input:
     - teff: model effective temperature in K (float)
     - band: name of the photometric band (string), see section
       'Available photometric bands' at the start of of this
       functions for which ones are included, and the
       description below on how to add more.
     - obsmag: the observed absolute magnitude in the given
       band (float)
     - Tfrac: fraction of the effective temperature that used
       used for calculating the 'theoretical SED' aka black
       body curve (float)
     - zp_system: choose from 'vega', 'AB', 'ST' (string)
     - filterdir: specify (relative) path to the directory
       where the filter information is stored (string)

    Output:
     - Estimated stellar radius in solar units (float)

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

    NOTE ON THE 'THEORETICAL SED'

    The "theoretical SED" on which the radius estimate is based
    is a Planck function. The temperature used for this can be
    scaled with Tfrac, is now set default to 0.9, this is as done in
    Mokiem 2005, who follows Markova 2004.

    #FIXME it would be interesting to check whether this
    chosen value for Tfrac from Markova 2004 gives the best
    approximation by comparing for calculated models the
    real SEDs with the 0.9*teff black body spectrum, and see
    how those radii compare

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
    elif band == 'Johnson_V':
        filterfile = 'GCPD_Johnson.V.dat'
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

    tBB = teff * Tfrac

    # Integration over angles results in the factor of pi
    F_lambda = np.pi*planck_wavelength(wave, tBB)

    rsun = 6.96e10
    parsec_cm = 3.08567758e18
    radius_ratio = 10*parsec_cm / rsun

    filtered_flux = np.trapz(trans*F_lambda, wave)/np.trapz(trans, wave)
    obsflux = magnitude_to_flux(obsmag, the_zero_point)
    bolflux_10pc = obsflux/filtered_flux
    luminosity = bolflux_10pc * (10*parsec_cm / rsun)**2
    radius_rsun = luminosity**0.5

    return radius_rsun

def planck_wavelength(wave_angstrom, temp):
    ''' Calculate the Planck function as function of temperature,
    and wavelengt (in Angstrom, output is then also in Angstrom).
    '''

    angstrom_to_cm = 1e-8
    wave = wave_angstrom * angstrom_to_cm

    # All units in cgs
    hh = 6.6260755e-27 #Planck constant;
    cc = 2.99792458e10 #speed of light in a vacuum;
    kk = 1.380658e-16 #Boltzmann constant;

    prefactor = 2.0 * hh * cc**2 / (wave**5)
    exponent = (hh * cc / kk) / (wave * temp)
    Blambda = prefactor * (1.0 / (np.exp(exponent)-1))

    #Blambda from per cm to per angstrom
    Blambda = Blambda * angstrom_to_cm

    return Blambda

def magnitude_to_flux(magnitude, zpflux):
    ''' Calculate observed flux from magnitude and zeropoint flux'''
    obsflux = zpflux * 10**(-magnitude/2.5)
    return obsflux

def magnitude_to_radius_SED(sed_wave, sed_flam, band, obsmag, zp_system,
    filterdir='filter_transmissions/'):

    '''Estimate the radius of the star given a temperature,
    photometric filter and observed (dereddened) absolute
    magnitude.

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
     - Estimated stellar radius in solar units (float)

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
    elif band == 'Johnson_V':
        filterfile = 'GCPD_Johnson.V.dat'
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
    obsflux = magnitude_to_flux(obsmag, the_zero_point)
    bolflux_10pc = obsflux/filtered_flux
    luminosity = bolflux_10pc * (10*parsec_cm / rsun)**2
    radius_rsun = luminosity**0.5

    return radius_rsun
