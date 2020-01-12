import os
import sys
import numpy as np
import math
import glob
import collections
import magnitude_to_radius as m2r
from scipy import interpolate

def mkdir(path):
    """Create a directory"""
    if not os.path.isdir(path):
        os.system('mkdir ' + path)

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

    keys, vals = np.genfromtxt(control_source, dtype=str, comments='#').T
    ctrldct = dict(zip(keys, vals))
    ctrldct["nind"] = int(ctrldct["nind"])
    ctrldct["ngen"] = int(ctrldct["ngen"])
    ctrldct["f_gen1"] = int(ctrldct["f_gen1"])
    ctrldct["clone_fraction"] = float(ctrldct["clone_fraction"])
    ctrldct["mut_rate_init"] = float(ctrldct["mut_rate_init"])
    ctrldct["doerr_factor"] = float(ctrldct["doerr_factor"])
    ctrldct["mut_rate_min"] = float(ctrldct["mut_rate_min"])
    ctrldct["mut_rate_max"] = float(ctrldct["mut_rate_max"])
    ctrldct["mut_rate_factor"] = float(ctrldct["mut_rate_factor"])
    ctrldct["fit_cutoff_min_carb"] = float(ctrldct["fit_cutoff_min_carb"])
    ctrldct["fit_cutoff_max_carb"] = float(ctrldct["fit_cutoff_max_carb"])
    ctrldct["cutoff_increase_genv"] = float(ctrldct["cutoff_increase_genv"])
    ctrldct["cutoff_decrease_genv"] = float(ctrldct["cutoff_decrease_genv"])

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

def add2indat(list, dct, values, element=''):
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
    list.append(line)

    return list

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
    """If the vinf is not a fixed or free, adapt a value of
    x times the escape velocity. Otherwise leave unchanged.
    #FIXME #HIGHPRIORITY
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

def clumping_type(ficval):
    """
    Determine the type of clumping. If 'fic' is specified, this
    means that optically thick clumping is taken into account
    when creating INDAT.DAT.
    """

    if float(ficval) == -1:
        clumptype = 'thin'
    else:
        clumptype = 'thick'

    return clumptype

def create_indat(freevals, modname, moddir, freenames, fixvals, fixnames,
    defvals, defnames, radinfo, indat_file='INDAT.DAT', formal_in='formal.in',
    broad_in='broad.in'):
    """
    Given a set of parameters and defaults, create an INDAT.DAT
    file to be read by fastwind.

    Furthermore, input files for pformal and broaden.py are created.

    #FIXME X-rays are missing (FW syntax unknown, see below)
    """

    # Load all parameters in a dictionary
    dct = create_dict(freevals, freenames, fixvals, fixnames,
            defvals, defnames)
    dct = get_radius(dct, radinfo)
    # For approximating vinf with the vesc we need the correct
    # value for the radius to be already present in the dct.
    dct = get_vinf(dct)
    dct = calculate_mdot(dct)

    clumptype = clumping_type(dct['fic'])

    # Creating lines to write to the inifile here.
    inl = ["'" + modname + "'\n"]

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

    # Wind clumping and porosity etc, optically thin or thick
    if clumptype == 'thin':
        add2indat(inl, dct, ['fclump', 'vclstart', 'vclmax'])
    else:
        inl.append('THICK\n')
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
    add2indat(inl, dct, ['Si'], 'SI')
    add2indat(inl, dct, ['Mg'], 'MG')
    add2indat(inl, dct, ['P'], 'P')

    # **********************************************************
    # #FIXME: X-rays could be added here, but I don't know the
    # exact syntax for that in FW. Should be tested first.
    # In principle "adding X-rays to GA" should boil down to
    #   -  Add the indat part here (with some if-statement)
    #   -  Put the indat parameters to the defaults_fastwind.txt
    #      so that you don't get an error if you don't specify
    #      the X-ray parameters in the param_space.txt file
    # **********************************************************

    # Write indat file
    with open(moddir + indat_file, 'w') as f:
        for indatline in inl:
            f.write(indatline)

    # Write input file for pformal
    with open(moddir + formal_in, 'w') as f:
        f.write(modname + '\n')
        if dct['windturb'] > 0.0 and dct['windturb'] < 1.0:
            turbstring = dct['vturb'] + ' ' + dct['windturb'] + '\n'
        else:
            turbstring = dct['vturb'] + '\n'
        f.write(turbstring)
        f.write(dct['do_iescat'] + '\n')

    # Write input file for broaden.py
    with open(moddir + broad_in, 'w') as f:
        f.write(dct['vrot'] + '\n')
        f.write(dct['macro'] + '\n')

def check_parameters():
    """ Placeholder
    #FIXME! do some basic parameter checks on the input before the run
    starts. Ideally this is part of the "submission to cluster" script,
    so that you don't waste time if there is an error in your input.
    Check in param_file.txt:
    - step size: should be always greater than zero if a param is free.
    - check for duplicates in param_space file
    - if fic != -1, then are all clumping values specified outside default?
    - check whether all lines in line_info are in FORMAL_INPUT.
    """

def read_linelist(theflinelist):
    """Read the line_list file and output numpy arrays"""
    llist_names = np.genfromtxt(theflinelist, dtype='str').T[0]
    llist_params = np.genfromtxt(theflinelist, dtype='float').T[1:]
    return llist_names, llist_params

def parallelcrop(list1, list2, list3, start_list1, stop_list1):

    minarg = np.argmin(np.abs(list1-start_list1))
    maxarg = np.argmin(np.abs(list1-stop_list1))

    newlist1 = list1[minarg:maxarg]
    newlist2 = list2[minarg:maxarg]
    newlist3 = list3[minarg:maxarg]

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
    res, lbound, rbound, rv, normlx, normly, normrx, normry, lw = llp

    data_per_line = []
    wave, norm, error = np.genfromtxt(fnorm).T
    for i in range(len(names)):

        # Per line select data, renormalise
        wv, nm, er = parallelcrop(wave, norm, error, lbound[i], rbound[i])
        nm = renorm(wv, nm, normlx[i], normly[i], normrx[i], normry[i])
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
    timeout = 'timeout ' + fwtimeout + ' '
    # ----------------------------------------------------
    # #FIXME This is a workaround for testing on OS X,
    # where the timeout command is not available, only
    # gtimeout is. Can be removed when using on linux
    if sys.platform == "darwin":
        timeout = 'gtimeout ' + fwtimeout + ' '
    # ----------------------------------------------------
    write_output = ' > pnlte.log'
    do_pnlte = timeout + pnlte_eo + write_output

    pformal_eo = './pformalsol_' + atom + '.eo '
    read_input = '< formal.in '
    write_output = ' > pformal.log'
    do_pformal = pformal_eo + read_input + write_output

    os.system(do_pnlte)
    os.system(do_pformal)

    # Return to the main directory
    os.chdir('../../../../../')

def read_fwline(OUT_file):
    '''Get wavelength and normflux from OUT. file'''
    tmp_matrix = np.genfromtxt(OUT_file, max_rows=161).T
    wave, flux = tmp_matrix[2], tmp_matrix[4]
    return wave, flux

def prep_broad(linename, line_file, moddir):
    """Read in fastwind output of the 'OUT.' format and
    convert this to a file that can be read by broaden.py.
    """
    out_clean = moddir + 'profiles/' + linename + '.prof'
    wave, flux = read_fwline(line_file)
    np.savetxt(out_clean, np.array([wave, flux]).T)
    return out_clean

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
        os.chdir('..')
        return

    # These likely have a different order than the filenames
    # that are read in from the linefile.
    linenames_fromfile = []
    for line in linefiles:
        the_line_name = line.rpartition('_')[0][4:]
        the_line_name = the_line_name.rpartition('OUT.')[-1]
        linenames_fromfile.append(the_line_name)

    # If this command takes longer than 5 min than certainly
    # something is wrong
    do_broaden = 'timeout 5m python broaden.py -f '
    # ----------------------------------------------------
    # #FIXME This is a workaround for testing on OS X,
    # where the timeout command is not available, only
    # gtimeout is. Can be removed when using on linux
    if sys.platform == "darwin":
        do_broaden = 'gtimeout 5m python broaden.py -f '
    # ----------------------------------------------------

    # Read in the broadening properties for the model.
    vrot, vmacro = np.genfromtxt(inicalcdir + 'broad.in')
    if vmacro > 0:
        do_macro = ' -m ' + str(vmacro)
    else:
        do_macro = ''
    if vrot > 0:
        do_vrot = ' -v ' + str(vrot)
    else:
        do_vrot = ''

    # Create a dictionary for lookup of resolving power per line
    resdct = dict(zip(linenames, lineres))

    # Loop through the OUT. files and apply broadening
    for linename, linefle in zip(linenames_fromfile, linefiles):
        # Convert to readable format
        input = prep_broad(linename, linefle, moddir)
        # Lookup resolving power
        do_res = ' -r ' + str(resdct[linename])
        # Apply broadening
        broadcommand = do_broaden + input + do_res + do_vrot + do_macro
        os.system(broadcommand)

    return

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
        apply_broadening(modname, moddir, linenames, lineres)
        return 1
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
    flux_mod = interp_modflux(wave_data, wave_mod, flux_mod_orig)

    chi2_line = np.sum(((flux_data - flux_mod) / (error_data))**2)
    np_line = len(flux_data)
    dof_line = lenfp + np_line
    rchi2_line = chi2_line / dof_line

    if len(wave_mod) > maxlen:
        # Replace the model output by a low resolution spectrum
        save_wave = np.linspace(min(wave_data), max(wave_data), maxlen)
        flux_mod_save = interp_modflux(save_wave, wave_mod, flux_mod_orig)
        np.savetxt(linefile, np.array([save_wave,flux_mod_save]).T)

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
    fitness = 0.0
    fitnesses_lines = np.zeros(len(linenames))
    fitm = 999999999
    return fitm, fitness, chi2_tot, rchi2_tot, linenames, fitnesses_lines

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
        dof_tot = lenfree
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

    return fitm, fitness, chi2_tot, rchi2_tot, linenames, fitnesses_lines

def store_model(txtfile, genes, fitinfo, runinfo, paramnames, modname):
    """ Write the paramters and fitness of an individual to the
    chi2-textfile.
    """

    fitmeasure, fitness, chi2_tot, rchi2_tot, linenames, linefitns = fitinfo

    write_lines = []

    if not os.path.isfile(txtfile):
        headerstring = '#run_id gen chi2 rchi2 fitness maxit maxcorr cputime'
        for pname in paramnames:
            headerstring = headerstring + pname + ' '
        for linenm in linenames:
            headerstring = headerstring + linenm + ' '
        headerstring = headerstring + '\n'
        write_lines.append(headerstring)

    gen = modname.split('_')[0]
    fitstr = str(chi2_tot) + ' ' + str(rchi2_tot) + ' ' +  str(fitness) + ' '
    metastr = runinfo[0] + ' ' + runinfo[1] + ' ' + runinfo[2] + ' '
    istr = modname + ' ' + gen + ' ' + fitstr + metastr
    for param in genes:
        istr = istr + str(param) + ' '
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

    # Remove all the files from 'run'
    rmdircommand = 'rm -r ' + moddir[:-8]
    os.system(rmdircommand)

def grep_pnlte(moddir, search, outputfile, loc):
    pnltelog = moddir + 'pnlte.log'
    tmp = moddir + outputfile + '.tmp'
    txt = moddir + outputfile + '.txt'
    grep = 'grep "' + search + '" ' + pnltelog + ' >> ' + tmp + ' ; '
    tail = 'tail -1 ' + tmp + ' > ' + txt + ' ; '
    rm = 'rm ' + tmp
    os.system(grep + tail + rm)
    value = np.genfromtxt(txt)[loc]
    return str(value)

def get_runinfo(moddir):
    if os.path.exists(moddir + 'pntle.log'):
        try:
            maxcor = grep_pnlte(moddir, "CORR. MAX:", 'corr_max', -1)
            maxit = grep_pnlte(moddir, "+  ITERATION NO", 'it_max', -2)
            cputime = grep_pnlte(moddir, "CPU time", 'cpu', -1)
        except:
            maxcor = '0.0'
            maxit = '0'
            cputime = '99999.9'
    else:
        maxcor = '0.0'
        maxit = '0'
        cputime = '99999.9'

    return [maxcor, maxit, cputime]

def evaluate_fitness(inicalcdir, rundir, savedir, all_pars, modelatom,
    fw_timeout, lineinfo, dof, fitmeasure, chi2file, paramnames, mname, genes):
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

    moddir = init_mod_dir(inicalcdir, rundir, mname)
    create_indat(genes, mname, moddir, *all_pars)
    out = run_fw(modelatom, moddir, mname, fw_timeout, lineinfo)
    out = 1

    if out == 0:
        fitinfo = failed_model(lineinfo[0])
    else:
        fitinfo = assess_fitness(moddir, mname, lineinfo, dof, fitmeasure)

    runinfo = get_runinfo(moddir)
    clean_run(moddir, mname, savedir, out)
    store_model(chi2file, genes, fitinfo, runinfo, paramnames, mname)

    return fitinfo[0]

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
    bestchi2file = 'best_chi2.txt'
    paramspacefile_out = 'parameter_space.txt'

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
    dct = add_to_dict(dct, "bestchi2_out", outdir + bestchi2file)
    dct = add_to_dict(dct, "paramspace_out", outdir + paramspacefile_out)

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
        sys.exit()

def copy_input(dict, theindir):
    """Copy the input directory to the output directory.
    This is not strictly necessary, but when looking at the
    output it is nice to have the input also at hand.
    """
    for key in dict:
        if "_in" in key:
            os.system("cp " + dict[key] + " " + theindir)

def remove_old_output(dict):
    """Remove old output files that might be present"""
    for key in dict:
        if "_out" in key and os.path.isfile(dict[key]):
            os.system("rm " + dict[key])

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

def create_FORMAL_INPUT(inidir, line_subset):
    """Create a FORMAL_INPUT file that contains only the lines that
    will be fitted. Based on the linelist we go through the
    FORMAL_INPUT "master" file and only copy those that we need to
    the FORMAL_INPUT that we will use.
    """

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
    for line in lines:
        splitline = line.strip().split()
        if len(splitline) > 2:
            if splitline[0] in line_subset or continueline:
                if splitline[0] in line_subset:
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

    # Write the collected lines to the new FORMAL_INPUT file.
    # This is the file that will be used during the run.
    with open('FORMAL_INPUT', 'w') as f:
        for aline in formal_new:
            f.write("%s" % aline)
        f.write("\n")

    # Navigate back to the main directory
    os.chdir('..')

#
