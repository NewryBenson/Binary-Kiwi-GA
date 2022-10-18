# Sarah Brands s.a.brands@uva.nl
# This script is part of Kiwi-GA: https://github.com/sarahbrands/Kiwi-GA
# Connects the evolutionary algorithm to the FASTWIND stellar atmosphere
# code, i.e. manages and creates FASTWIND input and output

import os
import sys
import numpy as np
import math
import glob
import collections
import magnitude_to_radius as m2r
from scipy import interpolate
import broaden as br

def mkdir(path):
    """Create a directory"""
    if not os.path.isdir(path):
        os.system('mkdir -p ' + path)

def rmfile(path):
    """Remove a file"""
    if os.path.isfile(path):
        os.system('rm ' + path)

def read_paramspace(param_source):
    """ Read the parameter space from text file"""

    dtypes = (str, float, float, float)
    pspace = np.genfromtxt(param_source, dtype=str).T

    ndecimals = []
    for x in pspace[3]:
        if '.' in x:
            ndecimals.append(len(x.split('.')[-1]))
        else:
            ndecimals.append(0)
    ndecimals = np.array(ndecimals)

    names = pspace[0]
    values = pspace[1:].astype(float).T

    variable_names = []
    variable_vals = []
    fixed_names = []
    fixed_vals = []

    for apname, pval, ndec in zip(names, values, ndecimals):
        if pval[0] == pval[1]:
            fixed_names.append(apname)
            fixed_vals.append(pval[0])
        else:
            new_pval = np.concatenate((np.array(pval), np.array([int(ndec)])))
            variable_names.append(apname)
            variable_vals.append(new_pval)

    return variable_names, variable_vals, fixed_names, fixed_vals

def read_control_pars(control_source):
    """ Read the control parameters from text file """

    # Read values from file and put into dictionary
    keys, vals = np.genfromtxt(control_source, dtype=str, comments='#').T
    ctrldct = dict(zip(keys, vals))

    # Convert the numeric files to integers or floats
    ctrldct["nind"] = int(ctrldct["nind"])
    ctrldct["ngen"] = int(ctrldct["ngen"])
    ctrldct["f_gen1"] = float(ctrldct["f_gen1"])
    ctrldct["ratio_po"] = float(ctrldct["ratio_po"])
    ctrldct["f_parent"] = float(ctrldct["f_parent"])
    ctrldct["p_value"] = float(ctrldct["p_value"])
    ctrldct["clone_fraction"] = float(ctrldct["clone_fraction"])
    ctrldct["w_gauss_br"] = float(ctrldct["w_gauss_br"])
    ctrldct["w_gauss_na"] = float(ctrldct["w_gauss_na"])
    ctrldct["b_gauss_br"] = float(ctrldct["b_gauss_br"])
    ctrldct["b_gauss_na"] = float(ctrldct["b_gauss_na"])
    ctrldct["mut_rate_na"] = float(ctrldct["mut_rate_na"])
    ctrldct["mut_rate_init"] = float(ctrldct["mut_rate_init"])
    ctrldct["mut_rate_min"] = float(ctrldct["mut_rate_min"])
    ctrldct["mut_rate_max"] = float(ctrldct["mut_rate_max"])
    ctrldct["mut_rate_factor"] = float(ctrldct["mut_rate_factor"])
    ctrldct["pure_reinsert_min"] = float(ctrldct["pure_reinsert_min"])
    ctrldct["fit_cutoff_min_charb"] = float(ctrldct["fit_cutoff_min_charb"])
    ctrldct["fit_cutoff_max_charb"] = float(ctrldct["fit_cutoff_max_charb"])
    ctrldct["cutoff_increase_genv"] = float(ctrldct["cutoff_increase_genv"])
    ctrldct["cutoff_decrease_genv"] = float(ctrldct["cutoff_decrease_genv"])

    n_parent = ctrldct["nind"] * ctrldct["ratio_po"]
    ctrldct["n_keep_parent"] = math.ceil(n_parent * ctrldct["f_parent"])
    f_keep_offspring = ctrldct["f_parent"] * ctrldct["ratio_po"]
    ctrldct["n_keep_offspring"] = n_parent - ctrldct["n_keep_parent"]

    return ctrldct

def get_defvals(the_filename, freenames, fixednames):
    """Load the default parameters and their names into arrays
    with removal of the parameter values that are specified in
    the parameter file.
    """

    # Read file with defaults
    defnames, defvals = np.genfromtxt(the_filename, dtype='str').T

    # Check for parameters in the parameter space file
    specified_parnames = np.concatenate((freenames, fixednames))
    rmdefault = []
    for i in range(len(defnames)):
        if defnames[i] in specified_parnames:
            rmdefault.append(False)
        else:
            rmdefault.append(True)

    # Remove the duplicates from the default list
    defnames = defnames[rmdefault]
    defvals = defvals[rmdefault]

    return defnames, defvals

def add2indat(alist, dct, values, element=''):
    """Given a listof parameters (strings), create a line for
    the fastwind INDAT.DAT file.

    Each parameter is separated with a space from the others,
    and each line ends with a newline mark.

    Lines for abundanes are created in a different way
    and need the extra argument 'element' to be specified.
    """

    if element == '':
        line = dct[values[0]]
        if len(values) > 1:
            for value in values[1:]:
                line = line + ' ' + dct[value]
    else:
        if float(dct[values[0]]) == -1.0:
            return
        else:
            line = element + ' ' + dct[values[0]]

    line = line + '\n'
    alist.append(line)

    return alist

def create_dict(freevals, freenames, fixvals, fixnames, defvals, defnames):
    """Store all parameters, free, fixed and defaults, in a
    dictionary. All parameter values are converted to strings."""

    allvals = np.concatenate((freevals, fixvals, defvals))
    allkeys = np.concatenate((freenames, fixnames, defnames))
    allvals = allvals.astype(str)

    dct = dict(zip(allkeys, allvals))

    return dct

def get_radius(dct, radinfo):
    """Check whether a fixed radius or luminosity anchor is used
    Obtain radius directly or from calculation, respectively.
    """

    if radinfo[0] == 'fixed_radius':
        dct['radius'] = radinfo[1]
    else:
        band, obsmag, zpsyst = radinfo
        obsmag = float(obsmag)
        teffrad = float(dct['teff'])
        dct['radius'] = str(round(m2r.magnitude_to_radius(teffrad, band,
            obsmag, zpsyst),2))

    return dct

def get_vinf(dct):
    """If the vinf is not a fixed or free (i.e. if its value is set
    at vinf = -1 in the input, then adapt a value of x times the escape
    velocity where x depends on temperature. Otherwise leave unchanged.
    """

    logg = float(dct['logg'])
    g_cgs = 10**logg

    Rsun = 6.96e10 # cm
    radius_rsun = float(dct['radius'])
    radius_cgs = radius_rsun * Rsun

    cms_to_kms = 1.0e-5
    vesc_cgs = np.sqrt(2*g_cgs*radius_cgs)
    vesc_kms = vesc_cgs * cms_to_kms

    # Scaling factor changes with temperature, numbers from
    # Lamers & Cassinelli 1999, fig 2.20 (page 49)
    teff = float(dct['teff'])
    if teff > 21000.0:
        scale_factor = 2.6
    elif teff <= 21000.0 and teff > 10000.0:
        scale_factor = 1.3
    else:
        scale_factor = 0.7

    if float(dct['vinf']) == -1:
        vinf_approx = vesc_kms*scale_factor
        dct['vinf'] = str(round(vinf_approx,0))

    return dct

def convert_vclmax_scale(vclstart, vclextend):
    """ If vclmax is a free parameter, then in case that it is
    lower than vstart, get another vclmax value by scaling vclstart
    """
    maxrange = 1.0 - vclstart
    vclmax = vclstart + (vclextend * maxrange)
    vclmax = str(round(vclmax,4))

    return vclmax

def convert_vclmax_multi(vclstart, vclextend, factor):
    """ If vclmax is a free parameter, then in case that it is
    lower than vstart, get another vclmax value by multiplying vclstart
    """
    vclmax = min(round(factor * vclstart, 4), 1.0)
    vclmax = str(vclmax)

    return vclmax

def convert_vclmax_add(vclstart, vclextend, addition):
    """ If vclmax is a free parameter, then in case that it is
    lower than vstart, get another vclmax value by adding a value
    to vclstart.
    """
    vclmax = min(round(vclstart + addition, 4), 1.0)
    vclmax = str(vclmax)

    return vclmax

def get_vclmax(dct):
    """ If the value for vclmax is -1, we adopt 2*vclstart
        If it can vary freely (between e.g. 0 and 1.0) then it is
        checked that the chosen value is not lower than vstart. If
        it is, then it is adapted to a physical value.
    """

    vclstart = float(dct['vclstart'])
    vclmax = float(dct['vclmax'])

    if vclmax == -1.0:
        dct['vclmax'] = convert_vclmax_multi(vclstart, vclmax, 2.0)
    elif vclmax == -2.0:
        dct['vclmax'] = vclstart + 0.10
    # Varying vclmax is somewhat experimental
    elif vclmax >= 0.0 and vclmax <= 1.0:
        if vclmax < vclstart:
            # If vclmax is too low (i.e. lower than vclstart), the value
            # is adapted by choosing vclmax = vclstart + 0.05
            dct['vclmax'] = convert_vclmax_add(vclstart, vclmax, 0.05)

            # Alternative ways of adapting the vclmax value:
            #   dct['vclmax'] = convert_vclmax_scale(vclstart, vclmax)
            #   dct['vclmax'] = convert_vclmax_multi(vclstart, vclmax, 2.0)
            # I chose for the _add option, as in this case, when reproduc-
            # tion leads to a too low vclmax value, the value that is
            # eventually used, is closer to the original value than in the
            # case of scaling or multiplication, where the new value is
            # closer to random.

        else:
            # If it isn't smaller than vclstart, then vclmax is kept
            # to the value picked originally by GA
            pass

    return dct

def check_windturb(dct):
    """ Check that the wind turbulence (maximum turbulence at vinf)
        is not higher than the atmospheric turbulence.
        The wind turbulence can be given in units of vinf (when it
        is between 0.0 and 1.0) or in km/s.
    """

    windturb = float(dct['windturb'])
    vinf = float(dct['vinf'])
    micro = float(dct['micro'])

    if windturb > 1.0:
        windturb_abs = vinf*windturb
    else:
        windturb_abs = windturb

    if windturb_abs < micro:
        windturb = micro
    else:
        windturb = windturb_abs

    dct['windturb'] = float(windturb)

    return dct

def calculate_mdot(dct, significant_digits=6):
    """ Given a mass loss rate in logspace, get the mass loss
    rate in Msun/year and round this to a certain amount of
    significant digits.

    #FIXME #LOWPRIORITY The amount of sign digits should
    actually correlate with the stepsize given the parameter
    space. However we just round it with a high amount of
    significant_digits, the rounding is only done to improve
    readability of the INDAT.DAT file anyway so it does not
    really matter as long as the rounding does not cause a
    loss of information.
    """

    logmdot = float(dct['mdot'])
    realmdot = 10**logmdot

    # Round. Use ceil instead of floor because the log10(mdot)
    # values are always negative.
    sdcor = math.ceil(logmdot)
    roundmdot = round(realmdot, int(-sdcor + significant_digits))

    dct['mdot'] = str(roundmdot)

    return dct

def get_fx_obs(dct):
    """ Estimates fx based on the Mdot and vinf, based on the
    power law of Kudritzki, Palsa, Feldmeier et al. (1996). This power law
    is extrapolated also outside where Kudritzki+96 have data points.

    Input:
        - Dictionary with parameter values in the form of strings
    Output:
        - Dictionary with parameter values, updated with fx
    """

    mdot = float(dct['mdot'])
    vinf = float(dct['vinf'])

    mdot = mdot / 10**(-6)
    logmdotvinf = np.log10(mdot/vinf)

    # Relation from Kudritzki, Palsa, Feldmeier et al. (1996)
    logfx = -5.45 - 1.05*logmdotvinf
    fx = round(10**(logfx),8)

    dct['fx'] = str(fx)

    return dct

def get_fx_theory(dct):
    """ Estimates fx based on Mdot, vinf, and radius. This relation is
        based on the analysis of FW model output.

    Input:
        - Dictionary with parameter values in the form of strings
    Output:
        - Dictionary with parameter values, updated with fx
    """

    msun = 1.989e33
    rsun = 6.955e10
    year = 365*24*60*60
    mdot = 10**float(dct['mdot'])*msun/year # to g/s
    vinf = float(dct['vinf'])*1e5 # to cm
    radius = float(dct['radius'])*rsun # to cm

    # Compute log10 of wind density in cgs units
    logWD = np.log10(mdot/(4*np.pi * radius**2 *vinf))

    # Relation to get fx that gives approx Lx = 10**-7 Lstar by Brands
    logfx = -0.5541 + 1.2442*logWD + 0.0851*logWD**2
    fx = round(10**(logfx),8)

    dct['fx'] = str(fx)

    return dct

def clumping_type(ficval):
    """
    Determine the type of clumping. If 'fic' is specified, this
    means that optically thick clumping is taken into account
    when creating INDAT.DAT.
    """

    if float(ficval) >= 999:
        clumptype = 'thin'
    else:
        clumptype = 'thick'

    return clumptype

def fcl_rep_hillier(the_dct):
    """ Get the maximum clumping parameter of the Hillier exponential
        clumping law, assuming an outer radius of 120.0 Rsun.
        Lower that slightly to be safe; this is the represetative clumping
        parameter.
    """
    fcl_out = float(the_dct['fclump'])
    beta = float(the_dct['beta'])
    vcl = float(the_dct['vcl'])
    vinf = float(the_dct['vinf'])

    r_in = 1.004
    radius = np.linspace(r_in, 120.0, 1000)
    velocity_r = vinf * (1-r_in/radius)**beta
    fclump_rad = 1.*fcl_out + (1.-1.*fcl_out)*np.exp(-velocity_r/vcl)
    max_fcl = np.max(fclump_rad)
    fcl_rep = round(max_fcl*0.90,5)
    if fcl_rep < 1.0:
        fcl_rep = 1.0
    return fcl_rep

def create_indat(freevals, modname, moddir, freenames, fixvals, fixnames,
    defvals, defnames, radinfo, indat_file='INDAT.DAT', formal_in='formal.in',
    broad_in='broad.in'):
    """
    Given a set of parameters and defaults, create an INDAT.DAT
    file to be read by fastwind.

    Furthermore, input files for pformal and broaden.py are created.

    """

    # Load all parameters in a dictionary
    dct = create_dict(freevals, freenames, fixvals, fixnames,
            defvals, defnames)
    # For approximating vinf with the vesc we need the correct
    # value for the radius to be already present in the dct.
    if isinstance(radinfo, float):
        dct['radius'] = str(radinfo)
    else:
        dct = get_radius(dct, radinfo)
    dct = get_vinf(dct)
    dct = calculate_mdot(dct)

    # Optically thin or optically thick clumping?
    clumptype = clumping_type(dct['fic'])

    # If vclmax = -1, get a value based on vclstart
    # If vclmax is free, then adapt the range so that
    # it matched vclstart.
    dct = get_vclmax(dct)

    # Creating lines to write to the inifile here.
    inl = ["'" + modname + "'\n"]

    # If logfclump < 3., use logfclump instead of fclump
    if float(dct['logfclump']) <= np.log10(1000.0):
        dct['fclump'] = str(round(10**(float(dct['logfclump'])),7))

    # Stuff needed in every FW model.
    add2indat(inl, dct, ['optne_update', 'he_one', 'it_start', 'itmore'])
    add2indat(inl, dct, ['optmixed'])
    add2indat(inl, dct, ['teff', 'logg', 'radius'])
    add2indat(inl, dct, ['rmax', 'tmin'])
    add2indat(inl, dct, ['mdot', 'vmin_start', 'vinf', 'beta', 'vdiv'])
    add2indat(inl, dct, ['yhe', 'ihe_start'])
    add2indat(inl, dct, ['optmod', 'opttlucy', 'megas', 'accel', 'optcmf'])
    add2indat(inl, dct, ['micro', 'metallicity', 'lines', 'lines_in_model'])
    add2indat(inl, dct, ['enat_cor', 'expansion', 'set_first', 'set_step'])

    # representative clumping parameter only required for exponential law.
    # for the linear law the clumping factor can be used directly
    # for the Hillier law this gives problems as fclout as given as input
    # is higher than the representative value
    #dct['fclump_rep'] = str(round(float(dct['fclump'])*0.8,2)) # rough way
    dct['fclump_rep'] = str(fcl_rep_hillier(dct)) # precise way

    # Wind clumping and porosity etc, optically thin or thick
    if clumptype == 'thin':

        # If vcl > 0, use Hilliers exponential clumping law
        if float(dct['vcl']) > 0:
            add2indat(inl, dct, ['fclump_rep', 'fclump', 'vcl', 'vcldummy'])
        # Else, use the linear step function law
        else:
            add2indat(inl, dct, ['fclump', 'vclstart', 'vclmax'])
    else:
        # Assume fic was given in log scale if fic < 0
        if float(dct['fic']) < 0.0:
            dct['fic'] = str(round(10**(float(dct['fic'])),6))

        inl.append('THICK\n')

        # If vcl > 0, use Hilliers exponential clumping law
        if float(dct['vcl']) > 0:
            add2indat(inl, dct, ['fclump_rep', 'fclump', 'vcl', 'vcldummy'])
            add2indat(inl, dct, ['fic', 'fic', 'vcl', 'vcldummy'])
            add2indat(inl, dct, ['fvel', 'fvel', 'vcl', 'vcldummy'])
            add2indat(inl, dct, ['hclump', 'hclump', 'vcl', 'vcldummy'])
        # Else, use the linear step function law
        else:
            add2indat(inl, dct, ['fclump', 'vclstart', 'vclmax'])
            add2indat(inl, dct, ['fic', 'vclstart', 'vclmax'])
            add2indat(inl, dct, ['fvel', 'vclstart', 'vclmax'])
            add2indat(inl, dct, ['hclump', 'vclstart', 'vclmax'])

    # Abundances (will only be addded if not set to -1)
    # Mind the allcaps spelling that has to be written
    # to the INDAT file for the multi-letter abbreviations!
    add2indat(inl, dct, ['C'], 'C')
    add2indat(inl, dct, ['N'], 'N')
    add2indat(inl, dct, ['O'], 'O')
    add2indat(inl, dct, ['Mg'], 'MG')
    add2indat(inl, dct, ['Si'], 'SI')
    add2indat(inl, dct, ['P'], 'P')
    add2indat(inl, dct, ['S'], 'S')
    add2indat(inl, dct, ['Fe'], 'FE')
    add2indat(inl, dct, ['Na'], 'NA')
    add2indat(inl, dct, ['Ca'], 'CA')

    # XRAYS - the parameter 'xpow' determines which X-ray prescription is used
    #  -  if xpow <= -1000, the one of  Carneiro+16 is used. In this case 'fx'
    #     is the X-ray volume filling fraction and 'xpow' has no meaning.
    #  -  if xpow > -1000 (but in practice > 0) the prescription of Puls+20
    #     is used. In this case 'fx' is n0, a normalisation of the power law,
    #     (see n_so in Puls+20 paper for details), and xpow the PL exponent.

    # Include X-rays if the volume filling fraction fx > 0.0
    # (fx = 0 means no volume filled with X-rays, so exclude X-rays)
    # When fx > 1000, estimate it based on mdot and vinf:
    if float(dct['xpow']) <= -1000:
        # Use the Carneiro+16 prescription
        if float(dct['fx']) > 1000:
            dct = get_fx_obs(dct) # # Kudritzki relation to get 10**-7
        if float(dct['fx']) < -1000:
            dct = get_fx_theory(dct) # Theoretical relation to get 10**-7
        # Use logscale fx value if that is in a valid range
        #  (only when it has set to such value in defaults, or in para-
        #  meter space, it will)
        if float(dct['logfx']) <= np.log10(16.0):
            dct['fx'] = str(round(10**(float(dct['logfx'])),7))
        # Add X-rays if the volume filling fraction > 0 *and* if teff is high
        # enough X-rays are in FW not allowed for Teff<25000 (model will crash)
        if (float(dct['fx']) > 0.0 and float(dct['teff']) >= 25000.0):
            inl.append('XRAYS ' + dct['fx'] + '\n')
            add2indat(inl, dct, ['gamx', 'mx', 'Rinx', 'uinfx', 'xpow'])
    else:
        # Use the Puls+20 prescription

        if (float(dct['fx']) > 0.0 and float(dct['teff']) >= 25000.0):
            inl.append('XRAYS ' + dct['fx'] + '\n')
            add2indat(inl, dct, ['gamx', 'mx', 'Rinx', 'uinfx', 'xpow'])

    # Write indat file
    with open(moddir + indat_file, 'w') as f:
        for indatline in inl:
            f.write(indatline)

    # Write input file for pformal
    with open(moddir + formal_in, 'w') as f:
        f.write(modname + '\n')
        windt = float(dct['windturb'])
        if windt > 0.0 and windt < 1.0:
            turbstring = dct['micro'] + ' ' + dct['windturb'] + '\n'
        else:
            turbstring = dct['micro'] + '\n'
        f.write(turbstring)
        f.write(dct['do_iescat'] + '\n')

    # Write input file for broaden.py
    with open(moddir + broad_in, 'w') as f:
        f.write(dct['vrot'] + '\n')
        f.write(dct['macro'] + '\n')

    return dct['radius'], dct['rmax']

def read_linelist(theflinelist):
    """Read the line_list file and output numpy arrays"""
    llist_names = np.genfromtxt(theflinelist, dtype='str').T[0]
    llist_params = np.genfromtxt(theflinelist, dtype='float').T[1:]
    return llist_names, llist_params

def parallelcrop(list1, list2, list3, start_list1, stop_list1):
    """ Based on values in list1, crop the same arguments of
    list2 and 3. Used for cropping spectra based on wavelength
    boundaries. If list3 equals [], only 2 lists are cropped"""

    newlist1 = list1[(list1 > start_list1) & (list1 < stop_list1)]
    newlist2 = list2[(list1 > start_list1) & (list1 < stop_list1)]
    if list3 == []:
        return newlist1, newlist2
    else:
        newlist3 = list3[(list1 > start_list1) & (list1 < stop_list1)]
        return newlist1, newlist2, newlist3

def renorm(wave, flux, lx, ly, rx, ry):
    """Rernormalise the data with a linear fit through the points
    (lx, ly) and (rx, ry) on the left and right of the line.

    Input:
    - Arrays with avelength and (more or less) normalised flux.
    - lx, rx: wavelengths of the renormalise anchor points
    - ly, ry: difference with continuum at lx and rx.

    Output:
    - Array with renormalised flux
    """

    slope = (ry - ly) / (rx - lx)
    offset = 1 + ly - slope * lx
    linfit = offset + slope * wave
    renorm = flux / linfit

    return renorm

def rvshift(lamb0, vr):
    """
    Apply a radial velocity (RV) shift to a wavelength (range).

    Input parameters:
        - lamb0: wavelength (float or np array) in angstrom
        - vr: radial velocity in km/s (float)
    Output:
        - Wavelength or wavelength array with RV shift applied.
    """

    # Constants
    c = 2.99792458*10**10 #cm/s
    angstrom = 1.0*10**-8 # multiply to go from Angstrom to cm
    kms = 10**-5 # multiply to go from cm/s to km/s

    lamb0 = lamb0 * angstrom
    vr = vr / kms
    deltalamb = (vr/c) * lamb0
    lamb0 = lamb0 - deltalamb
    lamb0 = lamb0 / angstrom

    return lamb0

def read_data(flinelist, fnorm):
    """Based on the line_list info, select the right data
    from the spectrum, and output data, line names and resolution.
    Each line in renormalised with the values found in the linelist

    Input:
    - linelist file (line names should be as in FORMAL_INPUT)
    - file with normalised spectrum

    Output:
    - array with line names
    - array with resolution of each line
    - array with per line a an array of arrays containing per line
        - wavelength
        - (renomalised) flux
        - errors on the flux
    """

    names, llp = read_linelist(flinelist)
    res, lbound, rbound, rv, normlx, normly, normrx, normry, lw, ang = llp

    data_per_line = []
    wave, norm, error = np.genfromtxt(fnorm).T
    for i in range(len(names)):
        # Per line select data, renormalise
        wv, nm, er = parallelcrop(wave, norm, error, lbound[i], rbound[i])
        if normly[i] != 0.0 or normry[i] != 0.0:
            nm = renorm(wv, nm, normlx[i], normly[i], normrx[i], normry[i])
        if rv[i] != 0.0:
            wv = rvshift(wv, rv[i])
        data_per_line.append([wv, nm, er])
    data_per_line = np.array(data_per_line)

    return names, res, data_per_line, lw

def gen_genname(gen, zfillen=4):
    """Generate generation name by filling in zeros"""
    genname = str(gen).zfill(zfillen)
    return genname

def gen_modnames(gen, lengen, zfillen=4):
    """Generate model names of the format xxxx_xxxx, e.g.
    for generation 23 and individual 147 this is 0023_0147.
    """
    genname = gen_genname(gen, zfillen)
    indnames = np.arange(lengen)

    modnames = []
    for ind in indnames:
        modnames.append(genname + '_' + str(ind).zfill(zfillen))

    return modnames

def execute_fastwind(atom, fwtimeout, moddir):
    """Execute pnlte and pformalsol for a certain model.
    Navigation to the model is crucial because of hardcoded
    paths in FASTWIND. At the end go back to the main
    directory.
    """
    # Go to the model directory
    os.chdir(moddir)


    pnlte_eo = './pnlte_' + atom + '.eo '

    # timeout based on cpu time
    fwtimeout = str(int(''.join(filter(str.isdigit, fwtimeout)))*60)
    timeout = 'ulimit -t ' + fwtimeout + ' ; '

    # timeout based on actual time
    #timeout = 'timeout ' + fwtimeout + ' '

    # ----------------------------------------------------
    # This is a workaround for testing on OS X,
    # where the timeout command is not available, only
    # gtimeout is. Can be removed when using on linux
    # but is in principle harmless
    if sys.platform == "darwin":
        timeout = 'gtimeout ' + fwtimeout + ' '
    # ----------------------------------------------------
    write_output = ' > pnlte.log'
    do_pnlte = timeout + pnlte_eo + write_output

    print('Start formalsol ' + moddir)
    os.system('ls -lhtr pformalsol_' + atom + '.eo ')
    os.system('pwd')
    pformal_eo = 'timeout 15m ./pformalsol_' + atom + '.eo '
    read_input = '< formal.in '
    write_output = ' > pformal.log'
    do_pformal = pformal_eo + read_input + write_output

    os.system(do_pnlte)
    os.system(do_pformal)

    # Uncomment if you want to save the FW log files
    #name_pnlte = 'pnlte_' + moddir.strip('/').split('/')[-2] + '.log'
    #name_pform = 'pformal_' + moddir.strip('/').split('/')[-2] + '.log'
    #os.system('mkdir -p ../../../pnlte/')
    #os.system('mkdir -p ../../../pformal/')
    #os.system('cp pnlte.log ../../../pnlte/' + name_pnlte)
    #os.system('cp pformal.log ../../../pformal/' + name_pform)

    # Return to the main directory
    # This is a weak point: if paths are changed
    # elsewhere then we run into errors here. #FIXME
    os.chdir('../../../../')

def read_fwline(OUT_file):
    '''Get wavelength and normflux from OUT.-file
       Treat CMF parts of the spectrum different from v10-
       like lines'''
    if not OUT_file.split('OUT.')[-1].startswith('UV_'):
        tmp_matrix = np.genfromtxt(OUT_file, max_rows=161).T
        if tmp_matrix.size == 0:
            wave, flux = [0], [0]
        else:
            wave, flux = tmp_matrix[2], tmp_matrix[4]
    else:
        tmp_matrix = np.genfromtxt(OUT_file).T
        if tmp_matrix.size == 0:
            wave, flux = [0], [0]
        else:
            wave, flux = tmp_matrix[1], tmp_matrix[3]
    return wave, flux

def prep_broad(linename, line_file, moddir):
    """Read in fastwind output of the 'OUT.' format and
    convert this to a file that can be read by broaden.py.
    """
    out_clean = moddir + 'profiles/' + linename + '.prof'
    wave, flux = read_fwline(line_file)
    if len(wave) == 1:
        out_clean = 'skip'
    else:
        np.savetxt(out_clean, np.array([wave, flux]).T)
    return out_clean, wave, flux

def apply_broadening(mname, moddir, linenames, lineres):
    """Broaden the fastwind output with the instrumental profile,
    rotational broadening and macro broadening. The values for
    the instrumental broadening can differ per line and are
    given to the function, the values for the rotational and
    macroturbulence are read from a file generated in the
    function create_indat.
    """

    inicalcdir = moddir
    moddir = moddir + mname + '/'

    mkdir(moddir + 'profiles')

    # Look up all FW line output and exit if there is none.
    linefiles = glob.glob(moddir + 'OUT.*')
    if len(linefiles) == 0:
        return 0

    # These likely have a different order than the filenames
    # that are read in from the linefile.
    linenames_fromfile = []
    for line in linefiles:
        the_line_name = line.rpartition('_')[0][4:]
        the_line_name = the_line_name.rpartition('OUT.')[-1]
        if the_line_name.startswith('UV_'):
            lsplit = the_line_name.split('_')
            the_line_name = 'UV_' + lsplit[1] + '_' + lsplit[2]
        else:
            the_line_name = the_line_name.rpartition('OUT.')[-1]
        linenames_fromfile.append(the_line_name)

    # Read in the broadening properties for the model.
    vrot, vmacro = np.genfromtxt(inicalcdir + 'broad.in')

    # Create a dictionary for lookup of resolving power per line
    resdct = dict(zip(linenames, lineres))

    # Loop through the OUT. files and apply broadening
    for linename, linefle in zip(linenames_fromfile, linefiles):
        # Convert to readable format
        finput, wave, flux = prep_broad(linename, linefle, moddir)
        if finput == 'skip':
            return 0
        # Lookup resolving power
        res = resdct[linename]
        # Apply broadening
        new_wave, new_flux = br.broaden_fwline(wave, flux, vrot, res, vmacro)
        np.savetxt(finput + ".fin", np.array([new_wave, new_flux]).T)

    return 1

def run_fw(modatom, moddir, modname, fwtime, lineinfo):
    """Run a fastwind model. This involves executing the files
    that calculate the NLTE and the formal solution, and apply
    the broadening to the output files. """

    linenames, lineres = lineinfo[:2]

    # Execute pnlte, pformalsol
    execute_fastwind(modatom, fwtime, moddir)

    # Apply instrumental, rotational and macroturbulent
    # broadening to the fastwind OUT. files.
    try:
        return apply_broadening(modname, moddir, linenames, lineres)
    except:
        return 0

def interp_modflux(wave_data, wave_mod, flux_mod):
    """Interpolate the flux of the model lines so that they are
    mapped to the same wavelengths as the data
    """
    fmod = interpolate.interp1d(wave_mod, flux_mod)
    flux_interp = fmod(wave_data)
    return flux_interp

def calc_chi2_line(resdct, nme, linefile, lenfp, maxlen=150):
    """Calculate the chi2 value of a line, and, in case
    the model spectrum is saved in high resolution, create and
    save a degraded version of the model spectrum to prevent massive
    output files.
    (Note: a file with 150 lines is 7.3K, for a run of 20 lines,
    180 generations, and 240 individuals, the total output is then
    about 6GB).
    """
    wave_data, flux_data, error_data = resdct[nme]
    wave_mod, flux_mod_orig = np.genfromtxt(linefile).T

    # If the wavelength range of the model is smaller than that of the
    # data, then add continuum at either side, so that an interpolation
    # can be done around the formally calculated line (the model predicts
    # continuum, though this is 'cut off' when saving the line)
    # (cropping the data to match the model range can give problems,
    # when the amount of points that is left is less than the degrees
    # of freedom. Furthermore, when doing this, effectively you throw out
    # part of the data when fitting).

    if min(wave_data) < min(wave_mod) or max(wave_data) > max(wave_mod):
        # Points to be added: continuum at 1.0 at both sides
        # Store values in array to be concatenated
        delta_wave = wave_mod[1] - wave_mod[0]
        addwave_left = wave_mod[0] - delta_wave
        addwave_right =  wave_mod[-1] + delta_wave
        wave_ext1 = np.array([min(wave_mod) - 90., addwave_left])
        wave_ext2 = np.array([addwave_right, max(wave_mod) + 90.])
        cont_ext = np.array([1.0, 1.0])
        # Add the points to the existing arrays
        wave_mod = np.concatenate((wave_ext1, wave_mod))
        wave_mod = np.concatenate((wave_mod, wave_ext2))
        flux_mod_orig = np.concatenate((cont_ext, flux_mod_orig))
        flux_mod_orig = np.concatenate((flux_mod_orig, cont_ext))

    flux_mod = interp_modflux(wave_data, wave_mod, flux_mod_orig)

    chi2_line = np.sum(((flux_data - flux_mod) / (error_data))**2)
    np_line = len(flux_data)
    dof_line = np_line - lenfp
    rchi2_line = chi2_line / dof_line

    if len(wave_mod) > maxlen:
        # Adjust maxlen in case of a CMF line
        if nme.startswith('UV_'):
            delta_wave_data = wave_data[1] - wave_data[0]
            # Save x times higher resolution than the data
            delta_save = delta_wave_data / 5.0
            maxlen = (max(wave_data)-min(wave_data))/delta_save

        # Replace the model output by a low resolution spectrum
        save_wave = np.linspace(min(wave_data), max(wave_data), maxlen)
        flux_mod_save = interp_modflux(save_wave, wave_mod, flux_mod_orig)
        np.savetxt(linefile, np.array([save_wave,flux_mod_save]).T,
            fmt='%10.5f')

    # #FIXME an unconvolved low res copy could be saved here

    return chi2_line, rchi2_line, np_line

def calc_fitness(rchi2s, weights):
    """ Calculate the fitness, given reduced chi2 values and
    weights for each spectral line.
    """
    weights = np.array(weights)
    rchi2s = np.array(rchi2s)
    fitness = 1./(np.sum(weights * rchi2s)/np.sum(weights))
    return fitness

def failed_model(linenames):
    """Returns the fitness values of a crashed model"""
    chi2_tot = 999999999
    rchi2_tot = 999999999
    dof_tot = -1
    fitness = 0.0
    fitnesses_lines = np.zeros(len(linenames))
    fitm = 999999999
    return (fitm, fitness, chi2_tot, rchi2_tot, dof_tot,
        linenames, fitnesses_lines)

def assess_fitness(moddir, modname, lineinfo, lenfree, fitmeasure):
    """Given the fastwind output of a model (broadened), assess
    the fitness of a model by comparing it to the data.
    Output are several fitness measures that will be written
    to an output file, as well as 'fitm', this is the measure
    that will be used for producing the new generation.
    """

    linenames, lineres, linedata, lineweight = lineinfo
    moddir = moddir + modname + '/profiles/'
    linefiles = []
    for lname in linenames:
        linefiles.append(moddir + lname + '.prof.fin')

    try:
        # Create a dictionary for looking up the data of a line
        resdct = dict(zip(linenames, linedata))

        chi2_tot = 0
        dof_tot = 0
        chi2_lines = []
        rchi2_lines = []
        weight_lines = []

        for i in range(len(linefiles)):

            chi2info = calc_chi2_line(resdct, linenames[i], linefiles[i],
                lenfree)
            chi2_line, rchi2_line, np_line = chi2info

            chi2_lines.append(chi2_line)
            rchi2_lines.append(rchi2_line)
            weight_lines.append(lineweight[i])

            chi2_tot = chi2_tot + chi2_line
            dof_tot = dof_tot + np_line

        dof_tot = dof_tot - lenfree
        rchi2_tot = chi2_tot / dof_tot
        fitness = calc_fitness(rchi2_lines, weight_lines)

        fitnesses_lines = 1./np.array(rchi2_lines)

    except:
        return failed_model(linenames)

    ####################### FITNESS MEASURE #######################
    # The reproduction code assumes higher value for the fitness
    # measure = fitter model therefore we inverse the fitness here.
    # For chi2 as a measure this is already the case. Because only
    # the _order_ order the fitness of the models is relevant for
    # reproduction, and not the absolute fitness, the way of
    # 'changing the scale' is not important.
    if fitmeasure == 'chi2':
        fitm = chi2_tot
    else:
        if fitness != 0.0:
            fitm = 1./fitness
        else:
            fitm = 999999999

    return (fitm, fitness, chi2_tot, rchi2_tot, dof_tot,
        linenames, fitnesses_lines)

def store_model(txtfile, genes, fitinfo, runinfo, paramnames, modname, rad,
        xlum, ionfluxinfo):
    """ Write the paramters and fitness of an individual to the
    chi2-textfile.
    """

    (fitmeasure, fitness, chi2_tot, rchi2_tot, dof_tot,
        linenames, linefitns) = fitinfo

    write_lines = []

    if not os.path.isfile(txtfile):
        hstring = '#run_id gen chi2 rchi2 dof fitness maxcorr maxit cputime '
        for pname in paramnames:
            hstring = hstring + pname + ' '
        hstring = hstring + 'radius '
        hstring = hstring + 'xlum '
        hstring = hstring + 'logq0 logQ0 logq1 logQ1 logq2 logQ2 '
        for linenm in linenames:
            hstring = hstring + linenm + ' '
        hstring = hstring + '\n'
        write_lines.append(hstring)

    gen = modname.split('_')[0]
    fitstr = (str(chi2_tot) + ' ' + str(rchi2_tot) + ' ' +  str(dof_tot) +
        ' ' + str(fitness) + ' ')
    metastr = runinfo[0] + ' ' + runinfo[1] + ' ' + runinfo[2] + ' '
    istr = modname + ' ' + gen + ' ' + fitstr + metastr
    for param in genes:
        istr = istr + str(param) + ' '
    istr = istr + str(rad) + ' '
    istr = istr + str(xlum) + ' '
    for aflux in ionfluxinfo:
        istr = istr + str(aflux) + ' '
    for lfit in linefitns:
        istr = istr + str(lfit) + ' '
    istr = istr + '\n'
    write_lines.append(istr)

    with open(txtfile, 'a') as the_file:
        for aline in write_lines:
            the_file.write(aline)

def clean_run(moddir, modname, savedir, outflag):
    """Copy output files to the savedir, and remove the model
    from the rundir, after the model has completed and the
    fitness been assessed.
    """

    gendir = savedir + modname.split('_')[0] + '/'
    mod_outdir = gendir + modname + '/'
    mkdir(gendir)
    mkdir(mod_outdir)

    # Copy the line profiles, if any, to the savedir
    if outflag == 1:
        profdir = moddir + modname + '/profiles/'
        copycommand = 'cp ' + profdir + '*.prof.fin ' + mod_outdir
        os.system(copycommand)

    # Copy the files that describe the model to the savedir
    os.system('cp ' + moddir + 'INDAT.DAT ' + mod_outdir + 'INDAT.DAT')
    os.system('cp ' + moddir + 'broad.in ' + mod_outdir + 'broad.in')
    os.system('cp ' + moddir + 'formal.in ' + mod_outdir + 'formal.in')

    # Compress saved dir to tar.gz file and remove the directory
    tarfilename = gendir + modname + '.tar.gz'
    os.system('tar -czf ' + tarfilename + ' -C ' + mod_outdir + ' .')
    os.system('rm -r ' + mod_outdir)

    # Remove all the files from 'run'
    rmdircommand = 'rm -r ' + moddir[:-8]
    os.system(rmdircommand)

def grep_pnlte(moddir, search, outputfile, loc):
    """Function to search the pnlte-log file"""
    pnltelog = moddir + 'pnlte.log'
    tmp = moddir + outputfile + '.tmp'
    txt = moddir + outputfile + '.txt'
    grep = 'grep "' + search + '" ' + pnltelog + ' >> ' + tmp + ' ; '
    tail = 'tail -1 ' + tmp + ' > ' + txt + ' ; '
    rm = 'rm ' + tmp
    os.system(grep + tail + rm)
    if os.path.getsize(txt) > 0:
        value = np.genfromtxt(txt)[loc]
    else:
        value = ""
    return str(value)

def get_runinfo(moddir):
    """Function that looks up the number of NLTE-iterations that
    fastwind has done, the maximum correction of the last iteration,
    and, if the model has finished, the total CPU time.
    This is done by using grep on the pnlte.log file.
    (this file will later be removed)
    """
    if os.path.exists(moddir + 'pnlte.log'):
        try:
            maxcor = grep_pnlte(moddir, "CORR. MAX:", 'corr_max', -1)
            if maxcor == '':
                maxcor = '0.0'
        except:
            maxcor = '0.0'
        try:
            maxit = grep_pnlte(moddir, "+  ITERATION NO", 'it_max', -2)
            if maxit == '':
                maxit = '0'
        except:
            maxit = '0'
        try:
            cputime = grep_pnlte(moddir, "CPU time", 'cpu', -1)
            if cputime == '':
                cputime = '99999.9'
        except:
            cputime = '99999.9'
    else:
        maxcor = '0.0'
        maxit = '0'
        cputime = '99999.9'

    return [maxcor, maxit, cputime]

def get_xlum_out(moddir, mname):
    """ Get the X-ray luminosity from the FW output
        Input: path to model directory (string)
        Output: Lx/L (string)
    """

    xlumitfile = moddir + mname + '/XLUM_ITERATION'

    if os.path.isfile(xlumitfile):
        with open(xlumitfile) as f:
            content = f.readlines()
            if len(content) > 0:
                xlumline = content[-1].strip().split()
                xlum = str(xlumline[2])
            else:
                xlum = '-1'
    else:
        xlum = '-1'

    return xlum

def ionizing_fluxes(lam, fnu, radius):
    c = 2.99792458e10
    h = 6.6260755e-27
    rsun = 6.957e10

    nu = c/(lam * 1e-8)# Hz [per second]
    photon_energy_nu = nu*h # Hz * ergs s^-1 = ergs [units of energy]

    integrand = fnu/photon_energy_nu # ph s^-1 cm^-2 Hz^-1
    sorting = nu.argsort()
    nu = nu[sorting]
    integrand = integrand[sorting]
    nulow_HI = c/(912.0e-8)
    nulow_HeI = c/(504.0e-8)
    nulow_HeII = c/(228.0e-8)

    nip = 1000000
    the_ip = interpolate.interp1d(nu, integrand)
    nu = np.linspace(min(nu), max(nu), nip)
    integrand = the_ip(nu)

    nuHI, integrandHI = parallelcrop(nu, integrand, [], nulow_HI, 1e100)
    nuHeI, integrandHeI = parallelcrop(nu, integrand, [], nulow_HeI, 1e100)
    nuHeII, integrandHeII = parallelcrop(nu, integrand, [], nulow_HeII, 1e100)

    # Integrate the integrand [ph s^-1 cm^-2 Hz^-1] over frequency: Hz
    # ph s^-1 cm^-2  [number of photons per surface area per second]
    q0 = np.trapz(integrandHI, nuHI)
    Q0 = q0 * 4*np.pi * (radius*rsun)**2 # ph s^-1 [integrate over surface]
    logq0 = round(np.log10(q0),3)
    logQ0 = round(np.log10(Q0),3)
    q1 = np.trapz(integrandHeI, nuHeI)
    Q1 = q1 * 4*np.pi * (radius*rsun)**2 # ph s^-1 [integrate over surface]
    logq1 = round(np.log10(q1),3)
    logQ1 = round(np.log10(Q1),3)
    q2 = np.trapz(integrandHeII, nuHeII)
    Q2 = q2 * 4*np.pi * (radius*rsun)**2 # ph s^-1 [integrate over surface]
    logq2 = round(np.log10(q2),3)
    logQ2 = round(np.log10(Q2),3)

    return logq0, logQ0, logq1, logQ1, logq2, logQ2

def read_fluxcont(moddir, mname, rstar, rmax_fw):
    """Check if FLUXCONT is there and if so, read it to get out
       the ionising fluxes.
       Input: path to model directory and model name,
           maximum radius of FW model, stellar radius (both in rsun).
       Output: q0, Q0, q1, Q1, q2, Q2
    """

    fluxcont = moddir + mname + '/FLUXCONT'

    if os.path.isfile(fluxcont):

        # Look up the number of useful lines in the FLUXCONT
        lcount = -2
        for aline in open(fluxcont, 'r').readlines():
            lcount = lcount+1
            if len(aline.split()) == 1:
                break

        rsun = 6.96e10 # cm
        rstar = float(rstar)
        rmax_fw = float(rmax_fw)
        stellar_surface = 4*np.pi*(rsun*rstar)**2

        # Only read non empty files, FLUXCONT has typically about
        # 1700-1800 lines containing flux information
        if lcount > 500:
            # Get FASTWIND spectrum
            lam, logFnu = np.genfromtxt(fluxcont, max_rows=lcount,
                skip_header=1, delimiter='').T[1:3]
            fnu = 10**logFnu # ergs/s/cm^2/Hz / RMAX^2
            fnu = fnu * rmax_fw**2 # ergs/s/A

            q0, Q0, q1, Q1, q2, Q2 = ionizing_fluxes(lam, fnu, rstar)

            return [q0, Q0, q1, Q1, q2, Q2]

    return [-888, -888, -888, -888, -888, -888]


def evaluate_fitness(inicalcdir, rundir, savedir, all_pars, modelatom,
    fw_timeout, lineinfo, dof, fitmeasure, chi2file, paramnames, name_n_genes):
    """Evaluate the fitness of an individual. This step is
    responsible for the bulk of the computation time. It does the
    following:
    - Create an inicalc directory for the model
    - Based on the genome and default/fixed parameters, create
      an INDAT.FILE, plus a file that is required for computing
      the formal solution, and a file with vrot and vmacro.
    - Run fastwind (pnlte, pformal, apply broadening)
    - Assess the fitness of the model
    - Save output and clean the run directory.
    """

    mname, genes = name_n_genes

    moddir = init_mod_dir(inicalcdir, rundir, mname)
    radius, rmax = create_indat(genes, mname, moddir, *all_pars)
    out = run_fw(modelatom, moddir, mname, fw_timeout, lineinfo)
    if out == 0:
        fitinfo = failed_model(lineinfo[0])
    else:
        fitinfo = assess_fitness(moddir, mname, lineinfo, dof, fitmeasure)

    runinfo = get_runinfo(moddir)
    xlum = get_xlum_out(moddir, mname)
    ionfluxinfo = read_fluxcont(moddir, mname, radius, rmax)
    clean_run(moddir, mname, savedir, out)
    store_model(chi2file, genes, fitinfo, runinfo, paramnames, mname,
        radius, xlum, ionfluxinfo)

    return fitinfo[0], fitinfo[3]

def add_to_dict(dictname, entryname, entryval):
    """Add an entry to a dictionary"""
    dictname[entryname] = entryval
    return dictname

def make_file_dict(indir, outdir):
    """Make a dictionary of meta files."""

    # File names of input files
    linelistfile = 'line_list.txt'
    paramspacefile = 'parameter_space.txt'
    radinfofile = 'radius_info.txt'
    defvalfile = 'defaults_fastwind.txt'
    normspecfile = 'spectrum.norm'
    ctrlfile = 'control.txt'

    # File names of output files
    chi2file = 'chi2.txt'
    duplfile = 'check_duplicates.txt'
    mutationfile = 'mutation_by_gen.txt'
    charblimfile = 'charbonneau_limits.txt'
    bestchi2file = 'best_chi2.txt'
    paramspacefile_out = 'parameter_space.txt'
    genvarfile_out = 'genetic_variety.txt'

    # File names of files for run continuation
    # These are copies that contain only fully completed generations
    chi2_contfile = 'chi2_cont.txt'
    dupl_contfile = 'dupl_cont.txt'
    generation_contfile = 'savegen_cont.txt'
    fitnesses_contfile = 'savefitness_cont.txt'
    redchi2_contfile = 'redchi2s_cont.txt'

    dct = {}

    dct = add_to_dict(dct, "linelist_in", indir + linelistfile)
    dct = add_to_dict(dct, "paramspace_in", indir + paramspacefile)
    dct = add_to_dict(dct, "radinfo_in", indir + radinfofile)
    dct = add_to_dict(dct, "defvals_in", indir + defvalfile)
    dct = add_to_dict(dct, "normspec_in", indir + normspecfile)
    dct = add_to_dict(dct, "control_in", indir + ctrlfile)

    dct = add_to_dict(dct, "chi2_out", outdir + chi2file)
    dct = add_to_dict(dct, "dupl_out", outdir + duplfile)
    dct = add_to_dict(dct, "mutation_out", outdir + mutationfile)
    dct = add_to_dict(dct, "charblim_out", outdir + charblimfile)
    dct = add_to_dict(dct, "bestchi2_out", outdir + bestchi2file)
    dct = add_to_dict(dct, "paramspace_out", outdir + paramspacefile_out)
    dct = add_to_dict(dct, "genvar_out", outdir + genvarfile_out)

    dct = add_to_dict(dct, "chi2_cont", outdir + chi2_contfile)
    dct = add_to_dict(dct, "dupl_cont", outdir + dupl_contfile)
    dct = add_to_dict(dct, "gen_cont", outdir + generation_contfile)
    dct = add_to_dict(dct, "fit_cont", outdir + fitnesses_contfile)
    dct = add_to_dict(dct, "redchi_cont", outdir + redchi2_contfile)

    return dct

def init_setup(theoutdir):
    """ Setup the structure of the output directory
    in the subdirectory that is used for the run, and
    return the paths to the created directories.
    """

    rundir = theoutdir + 'run/'
    savedir = theoutdir + 'saved/'
    indir = theoutdir + 'input_copy/'
    for adir in (theoutdir, rundir, savedir, indir):
        mkdir(adir)
    return theoutdir, rundir, savedir, indir

def check_indir(indir):
    """Check if the input directory exits"""
    if not os.path.isdir(indir):
        print("No input directory found!")
        print("I was searching here: " + indir)
        print("Exiting")
        return False
    return True

def copy_input(adict, theindir):
    """Copy the input directory to the output directory.
    This is not strictly necessary, but when looking at the
    output it is nice to have the input also at hand.
    """
    for key in adict:
        if "_in" in key:
            os.system("cp " + adict[key] + " " + theindir)

def prepare_output_files(adict, cont_tf):
    """Move the _cont files to chi2 and duplicate files if the run is
    continuing another run, otherwise remove the old output files,
    if any are present.
    """

    if cont_tf:
        os.system("cp " + adict["chi2_cont"] + " " + adict["chi2_out"])
        os.system("cp " + adict["dupl_cont"] + " " + adict["dupl_out"])
    else:
        for key in adict:
            if "_out" in key and os.path.isfile(adict[key]):
                os.system("rm " + adict[key])

def init_mod_dir(inidir, therundir, modname):
    """Copy the inicalc directory to a directory for a specific
    model. We need separate inicalc dirs for each model because
    FW reuses files. Note that it is necessary that this
    directory is called 'inicalc', because FW has that hardcoded
    somewhere, it seemed.

    #FIXME #LOWPRIORITY inicalc is now copied every time,
    while in principle it could be just moved after the
    first generation. This costs about 0.6 seconds per
    model.This comes down to 0.02% of the total computation
    time so I leave it at this for now.

    """
    moddir = therundir + modname + '/'
    mkdir(moddir)
    moddir = moddir + 'inicalc/'
    os.system("cp -r " + inidir + ' ' + moddir)
    mkdir(moddir + modname)
    return moddir

def create_FORMAL_INPUT(inidir, line_subset, lfile, create=True):
    """Create a FORMAL_INPUT file that contains only the lines that
    will be fitted. Based on the linelist we go through the
    FORMAL_INPUT "master" file and only copy those that we need to
    the FORMAL_INPUT that we will use.
    Furthermore, the function checks whether all lines that are in
    the line_subset (the diagnositc lines) are present in the
    FORMAL_INPUT_master file, if not, it exits the run.
    If the parameter 'create' is false, only a check is done,
    and the FORMAL input file is not really created.
    """

    # Read all line-info, this is needed for the UV_ v11 lines
    thelnames, llp = read_linelist(lfile)
    res, lbound, rbound, rv, normlx, normly, normrx, normry, lw, ang = llp

    # Navigate to the main inicalc directory, the one that will
    # be copied all the time.
    os.chdir(inidir)

    # Read all lines of the 'master' FORMAL_INPUT file
    if not os.path.isfile('FORMAL_INPUT_master'):
        os.system('cp FORMAL_INPUT FORMAL_INPUT_master')
    with open('FORMAL_INPUT_master') as f:
        lines = f.readlines()

    # Loop through all lines that will be needed and copy them
    # to a new FORMAL_INPUT file.
    # The not so pretty if-statements below are there to take into
    # account that the information about a line with multiple
    # transitions can be spread out over several lines.
    formal_new = [':T VSINI\n', '0.\n']
    continueline = False
    list_formal_lines = []
    for line in lines:
        splitline = line.strip().split()
        if len(splitline) > 2:
            if splitline[0] in line_subset or continueline:
                if splitline[0] in line_subset:
                    list_formal_lines.append(splitline[0])
                    nsubslines = int(splitline[1])
                    lenforminput = nsubslines*4 + 2
                    ninputs = len(splitline)
                if continueline:
                    ninputs = ninputs + len(splitline)
                if ninputs < lenforminput:
                    continueline = True
                else:
                    continueline = False
                formal_new.append(line)

    for ic in range(len(thelnames)):
        theUVline = thelnames[ic]
        if theUVline.startswith('UV_'):
            thelb = str(int(theUVline.split('_')[1]))
            therb = str(int(theUVline.split('_')[2]))
            theang = str(ang[ic])
            newformalin = 'UV ' + thelb + ' ' + therb + ' ' + theang + '\n'
            formal_new.append(newformalin)

    if create:
        # Write the collected lines to the new FORMAL_INPUT file.
        # This is the file that will be used during the run.
        with open('FORMAL_INPUT', 'w') as f:
            for aline in formal_new:
                f.write("%s" % aline)
            f.write("\n")

    # Navigate back to the main directory
    os.chdir('..')

    missing_lines = []
    for aline in line_subset:
        if not aline.startswith('UV_'):
            if aline not in list_formal_lines:
                missing_lines.append(aline)

    if missing_lines != []:
        print('ERROR! Some diagnostic lines are not ' +
            'found in FORMAL_INPUT_master!')
        for missing in missing_lines:
            print(missing + ' not found')
        if not create:
            return False
        else:
            print('Exiting Kiwi-GA... :-(')
            sys.exit()
    if not create:
        print('All lines are present in FORMAL_INPUT_master')
        return True

def read_mut_gen(mut_gen_file):
    """ Function for restarting the run. Reads mutation rate and
    generation number of last generation
    """
    mutgenlines = np.genfromtxt(mut_gen_file)
    lenmut  = len(np.array(mutgenlines.shape))

    # If multiple generations have been computed already, take
    # the last line of the file only.
    if lenmut == 2:
        mutgenlines = mutgenlines[-1]

    gen = int(mutgenlines[0])
    mutrate = mutgenlines[1]

    return gen, mutrate
