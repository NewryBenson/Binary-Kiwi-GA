# PyGA analysis script
# Sarah Brands
# s.a.brands@uva.nl
# Created on: 19-11-2019
# Latest change: 25-05-2021
# Probabilty function and read in of parameters from script of Calum Hawcroft
#    and Michael Abdul-Masih
# Tested with python 2.7 on Mac

import __future__
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('agg') # to prevent some problems when running through ssh
                        # however this disables the seaborn plots
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from collections import OrderedDict
from scipy import stats
from scipy.optimize import curve_fit
import warnings

from PyPDF2 import PdfFileMerger, PdfFileReader
import cv2 #only required if make_paramspace_avi=True
import seaborn as sns
import img2pdf # for saving the scatter plots (fitness vs parameter) in
                      # PNG and then transforming them to pdf (otherwise the
                      # pdfs load ridiculously slow because of the many
                      # points in each scatter plot)

from astropy.convolution import convolve, Box1DKernel
import cmocean
import scipy
import fastwind_wrapper as fwr
import paths as ppp

''' ----------------------------------------------------------------------'''
''' ----------------------------------------------------------------------'''
'''                               INPUT                                   '''
''' ----------------------------------------------------------------------'''
''' ----------------------------------------------------------------------'''

''' ------------------------------------------'''
'''             Specify directories           '''
''' ------------------------------------------'''

# The output (pdf files, avi files) is written to outpath, in a subdirectory
# with the name of the run (will be made if it doesn't exits yet)
# The data should be stored in datapath/runname There it should have the
# structure as in 'Outputs' of pyGA (so with a dir Output inside)
# The line profiles per model should either be in the 0000_0000.tar.gz archive,
# or alternatively if already unpacked, the line profiles should be in a folder
# names 0000_0000 in the parent directory 0000. (example numbers)
outpath = ppp.outpath_analysis 
datapath = ppp.datapath_analysis 

''' ------------------------------------------'''
'''      which plots do you want to make?     '''
''' ------------------------------------------'''
# The pdf of a plot is saved and will be included in the full report once
# it is made, so setting a plot on 'False' here, means only that it is
# not generated _again_, not that it is not included in the full report.

# Specify which plots you want to be generated.
# full_short_manual can have the following values:
# 'full' =  all possible plots (will take a while)
# 'short' = does not generate correlation and fitness plots per line
# 'manual' = only selected plots, to be specified below
full_short_manual = 'full'

# Verbosity: if True, print progress, if False, print only errors/warnings
be_verbose = True

# If plot_individual_lines is set to True, then all models in the 2 sigma
# range will be plotted --> when this are many models, this can lead to
# segmentation faults. Advised is to set this parameter to False, in that case
# the min and max flux value of all profiles will be checked, and the
# plot will be made with fill_between between those values
plot_individual_lines = False

if full_short_manual in ('manual'):
    # If full_short_manual is set to 'manual'
    make_run_summary = True
    make_convergence_analysis_plot = False
    make_fitnessdistribution_plot = False
    make_fitnessdistribution_plot_P = False
    make_fitnessdistribution_plot_fitness = False # False when fitmeasure = chi2
    make_fitnessdistribution_per_line_plot = False
    make_derived_fitness_plots = False
    make_chi2pgen_plot = False
    make_correlation_plot = False # will always be done when make_correlation_per_line_plot = True
    make_correlation_per_line_plot = False
    make_param_dist_plot = False # violin plots
elif full_short_manual in ('short'):
    make_run_summary = True
    make_convergence_analysis_plot = True
    make_fitnessdistribution_plot = True
    make_fitnessdistribution_plot_P = True
    make_fitnessdistribution_plot_fitness = True  # is False when fitmeasure = chi2
    make_fitnessdistribution_per_line_plot = False
    make_derived_fitness_plots = True
    make_chi2pgen_plot = True
    make_correlation_plot = False # will always be done when make_correlation_per_line_plot = True
    make_correlation_per_line_plot = False
    make_param_dist_plot = True # violin plots
elif full_short_manual in ('full'):
    make_run_summary = True
    make_convergence_analysis_plot = True
    make_fitnessdistribution_plot = True
    make_fitnessdistribution_plot_P = True
    make_fitnessdistribution_plot_fitness = True  # is False when fitmeasure = chi2
    make_fitnessdistribution_per_line_plot = True
    make_derived_fitness_plots = True
    make_chi2pgen_plot = True
    make_correlation_plot = True # will always be done when make_correlation_per_line_plot = True
    make_correlation_per_line_plot = False  # This takes a lot of time
    make_param_dist_plot = True # violin plots
else:
    print('Unknown value for "full_short_manual" ')
    sys.exit()

plotprofs = sys.argv[2]
if plotprofs in ('plotprofs', 'plot_profs', 'pp', 'prof'):
    # Will take some time depending on how many models are included.
    make_lineprofiles_plot = True
else:
    # Will take some time depending on how many models are included.
    make_lineprofiles_plot = False

# Under construction, does not really work yet
make_fitness_fit_analysis_plot = False
make_param_dist_plot_detail = False

# If you want to only plot the first x (given by newmaxgen) generations
adapt_ngen = False
newmaxgen = 35

# Videos
make_paramspace_avi = False

''' ------------------------------------------'''
'''          Extra settings for plots         '''
''' ------------------------------------------'''

''' Specify parameter pairs '''
# Only used if make_paramspace_avi = True
# For each pair, an .avi is made that shows where in the parameter space
# models are calculated per generation
param_pair_list_avi = [  ['teff', 'logg']
                        ,['teff', 'He']
                        ,['vrot', 'macro']
                        ,['Si', 'C']
                        ,['Si', 'teff']
                        ]

''' Adding an extra (model)spectrum to the lineprofile plots ** Optional '''
    # Set include_extra_spectra to True to plot extra spectra or models over
    # the GA output and the .norm data.
    # Specify paths pointing to your spectra

#If making lineprofiles plot, include an extra spectrum for comparison
include_extra_spectrum = False

# Add up to 2 extra spectra or models to the plots
# If you only want to add one, set the other path equal to ''
# Currently no error bars are supported as I only loaded cmfgen models
extra_spectrum_path1 = ''
extra_spectrum_path2 = ''

''' Specify how to select the models in the lineprofile plots'''
# Choose between:
# - best_models_cutoff_method = 'P':
#    model selection based on proper chi square based probabilities
# - best_models_cutoff_method = 1.05 or other float:
#    model selection based on the reduced chi squared value (the float)
#    !!! no proper statistics done !!! Just to artificially show more models)
# Should be set to 'P' if you want it to correspond with the error bars
#   pyGA gives on parameters.
best_models_cutoff_method = 'P'

''' Set number of plots per page '''

nrows_lineprofileplot = 6
ncols_lineprofileplot = 3

nrows_fitnessparamplot = 4
ncols_fitnessparamplot = 3

''' ------------------------------------------------------------'''
''' ------------------------------------------------------------'''
'''      Script starts reading data and making plots here       '''
''' ------------------------------------------------------------'''
''' ------------------------------------------------------------'''

def do_print(astring, be_vb):
    if be_vb:
        print(astring)

''' ------------------------------------ '''
'''    Reading files and other things    '''
'''  that are useful for several plots   '''
''' -------------------------------------'''

if make_correlation_per_line_plot:
    make_correlation_plot = True

''' File names of output pdfs '''
# Sub output pdfs. Numbering determines the order in which they will come
# in the final report.

do_print("Loading stuff...", be_verbose)
name_runsummary = "0_runsummary_short_GAreport.pdf"
name_fitnessdistribution_plot = "1_fitnessdistribution_short_GAreport.pdf"
name_fitnessdistributionP_plot = "1_fitnessdistributionP_short_GAreport.pdf"
name_fitnessdistributionfitness_plot = "1_fitnessdistributionfit_short_GAreport.pdf"
name_derivedparP_plot = "1b_derivedparamsP_short_GAreport.pdf"
name_derivedparfitness_plot = "1b_derivedparamsfit_short_GAreport.pdf"
name_derivedpar_plot = "1b_derivedparams_short_GAreport.pdf"
name_convergence_plot = "2_convergence_short_GAreport.pdf"
name_lineprofiles_plot = "3_lineprofiles_short_GAreport.pdf"
name_correlation_plot = "4_correlation_short_GAreport.pdf"
name_chi2pergen_plot = "5_chi2pergen_short_GAreport.pdf"
name_param_dist_plot = "6_param_dist_short_GAreport.pdf"
name_fitness_per_line_plot = "7_fitness_param_perline_GAreport.pdf"
name_correlation_per_line_plot = "8_correlation_perline_GAreport.pdf"
name_param_dist_zoom_plot = "9_param_dist_zoom_GAreport.pdf"

name_best_fit_params_file = "best_fit_parameters.txt"


def calculateP(params, nnormspec, chi2, normalize, be_verbose=False):
    degreesFreedom = nnormspec - len(params)
    if be_verbose:
        do_print('No. of specpoints     : ' + str(nnormspec), be_verbose)
        do_print('No. of free parameters: ' + str(len(params)), be_verbose)
        do_print('Degrees of freedom    : ' + str(degreesFreedom), be_verbose)
    if normalize:
        scaling = np.min(chi2)
        if be_verbose:
            do_print('Scaling Chi2, dividing by: ' + str(round(scaling, 3)),
                be_verbose)
    else:
        scaling = degreesFreedom
        if be_verbose:
            do_print('Not scaling Chi2, dividing by: ' + str(round(scaling, 3)),
                be_verbose)
    chi2 = (chi2 * degreesFreedom) / scaling
    probs = np.zeros_like(chi2)
    try:
        for i in range(len(chi2)):
            probs[i] = stats.chi2.sf(chi2[i], degreesFreedom)
    except:
        chi2 = chi2.values
        for i in range(len(chi2)):
            probs[i] = stats.chi2.sf(chi2[i], degreesFreedom)
    return probs

def parallelcrop(list1, list2, start_list1, stop_list1,
    list3=[], list4=[]):

    list1 = np.array(list1)
    list2 = np.array(list2)

    minarg = np.argmin(np.abs(list1-start_list1))
    maxarg = np.argmin(np.abs(list1-stop_list1))

    newlist1 = list1[minarg:maxarg]
    newlist2 = list2[minarg:maxarg]

    if list3 != []:
        newlist3 = list3[minarg:maxarg]
        if list4 != []:
            newlist4 = list4[minarg:maxarg]
            return newlist1, newlist2, newlist3, newlist4
        else:
            return newlist1, newlist2, newlist3

    return newlist1, newlist2

def cm_rgba(x):
    # IF YOU CHANGE THIS, MAKE SURE TO CHANGE ALSO THE LEGEND
    setcmap = cm.jet
    setnorm = colors.Normalize(0.0, 1.0)

    setscalarMap = cm.ScalarMappable(norm=setnorm, cmap=setcmap)
    return setscalarMap.to_rgba(x)

'''
=================== Defining paths ===================
Includes hard copy directory and  and file names less
likely to be changed.
'''

run = sys.argv[1]
if run[-1] != '/':
    run +='/'
runname = run[:-1]

print("Generating report for << " + runname + " >>")

if not outpath.endswith('/'):
    outpath = outpath + '/'
if not datapath.endswith('/'):
    datapath = datapath + '/'

datapath = datapath + run
plotpath = outpath + run
plotlineprofdir = outpath + run + 'lineprofs/'
os.system("mkdir -p " + plotpath)
os.system("mkdir -p " + plotlineprofdir)

inputcopydir = datapath + 'input_copy/'
savedmoddir = datapath + 'saved/'

thechi2file = datapath + 'chi2.txt'
thebestchi2file = datapath + 'best_chi2.txt'
themutgenfile = datapath + 'mutation_by_gen.txt'
thecontrolfile = inputcopydir + 'control.txt'
thelinefile = inputcopydir + 'line_list.txt'
theparamfile = inputcopydir + 'parameter_space.txt'
thespectrumfile = inputcopydir + 'spectrum.norm'
theradiusfile = inputcopydir + 'radius_info.txt'
thefwdefaultfile = inputcopydir + 'defaults_fastwind.txt'

name_fullreport_pdf = plotpath + 'full_report_' + run[:-1] + ".pdf"
name_shortreport_pdf = plotpath + 'short_report_' + run[:-1] + ".pdf"

best_fit_pars_txt = plotpath + name_best_fit_params_file

'''
=============== Reading output files ===============
'''

# with open(thechi2file) as f:
#     content = f.readlines()
# for line in input_file:
#     if line.endswith(" \n"):
#        line = line.replace(' \n', '\n')

parout = fwr.read_paramspace(theparamfile)
variable_names, variable_vals, fixed_names, fixed_vals = parout
defnames, defvals = fwr.get_defvals(thefwdefaultfile,
    variable_names, fixed_names)

var_dct = dict(zip(variable_names, variable_vals))
fix_dct = dict(zip(np.concatenate((fixed_names, defnames)),
    np.concatenate((fixed_vals, defvals))))

x = pd.read_csv(thechi2file, sep=' ')
for acolname in x.columns:
    if acolname.startswith('#'):
        x = x.rename(columns = {acolname : acolname[1:]})
    if acolname.startswith('Unnamed: '):
        x = x.drop(acolname, 1)

if len(x) < 10:
    print('Something is wrong with the chi2.txt file')
    print('Sometimes the header is printed at the second line, or multiple')
    print('   times. Check the top lines of the chi2.txt file')
    print('   File can be found here: ' + thechi2file)

# For compatibility with older versions of the script where
# radius was not included. In that case skip 'derived parameters'
if not 'radius' in x.columns:
    no_radius_provided = True
    if be_verbose:
        print('No radius provided: skipping derived parameter plot')
else:
    no_radius_provided = False
    if be_verbose:
        print('Radius provided: computing derived parameters e.g. luminosity')
if not 'xlum' in x.columns:
    no_xlum_provided = True
    if be_verbose:
        print('No xlum provided: skipping derived parameter plot')
else:
    no_xlum_provided = False


outcontrol = np.genfromtxt(thecontrolfile,dtype='str').T
control_dct = dict(zip(outcontrol[0], outcontrol[1]))

if control_dct["fitmeasure"] == 'chi2':
    make_fitnessdistribution_plot_fitness = False

maxindid = float(control_dct["nind"])
# maxgen = np.max(x['gen'].values)
try: # make sure the script also works if there is only one generation
    maxgen = np.max(np.genfromtxt(thebestchi2file)[:,0])
except:
    maxgen = np.max(np.genfromtxt(thebestchi2file)[0])

# If a generation was not completed, drop those models.
x = x.drop(x[x.gen > maxgen].index)

if adapt_ngen:
    # Only plot the first maxgen generations
    maxgen = newmaxgen
    x = x.drop(x[x.gen > maxgen].index)


if 'vclmax' in x.columns and 'vclstart' in x.columns:
    real_vclmax = []
    for vclstartval, vclmaxval in zip(x['vclstart'].values, x['vclmax'].values):
        if vclmaxval <= vclstartval:
            real_vclmax.append(vclstartval + 0.05)
        else:
            real_vclmax.append(vclmaxval)
    real_vclmax = np.array(real_vclmax)
    x['vclmax'] = real_vclmax

elif 'vclmax' in x.columns:
    do_print('Only vclmax appears to be a free parameter.', be_verbose)
    raw_input()
maxgenid = str(int(maxgen)).zfill(4)
do_print('Last generation: ' + str(maxgenid), be_verbose)
x['gen_id'] = map(lambda lam: str(int(lam)).zfill(4), x['gen'])

unique_gen = np.unique(x['gen_id'].values)
unique_genid = np.unique(x['gen_id'].values)

''' Evaluate chi2 per generation '''
median_chi2_per_gen = []
mean_chi2_per_gen = []
lowest_chi2_per_gen = []
for a_gen_id in unique_genid:
    x_1gen = x[x['run_id'].str.contains(a_gen_id + '_')]
    x_1gen = x_1gen[x_1gen.rchi2 < 1000]
    chi2val = x_1gen['chi2'].values
    chi2val = chi2val[chi2val < 999999999]
    if len(chi2val) > 1:
        median_chi2_per_gen.append(np.median(chi2val))
        mean_chi2_per_gen.append(np.mean(chi2val))
        lowest_chi2_per_gen.append(np.min(chi2val))
    else:
        # In case a generation has only crashed models
        median_chi2_per_gen.append(np.nan)
        mean_chi2_per_gen.append(np.nan)
        lowest_chi2_per_gen.append(np.nan)
        print('Empty generation: ' + str(a_gen_id))
if maxgen>1:
    ''' Read in mutation per generation '''
    mutationpergen = np.genfromtxt(themutgenfile)
    if len(mutationpergen) != len(median_chi2_per_gen):
        difflen = np.abs((len(mutationpergen)-len(median_chi2_per_gen)))
        mutationpergen = mutationpergen[:-difflen]
    generation, mutation = mutationpergen.T

''' Get linelist '''

linenames = np.genfromtxt(thelinefile, dtype='str').T[0]
lineinfo = np.genfromtxt(thelinefile, dtype='float').T[1:]

linestarts = lineinfo[1]
linestops = lineinfo[2]
line_radialvelocity = lineinfo[3]
line_norm_starts = lineinfo[4]
line_norm_leftval = lineinfo[5]
line_norm_stops = lineinfo[6]
line_norm_rightval = lineinfo[7]

numlines = len(linenames)

do_print('Reading line file: ', be_verbose)
do_print('Number of diagnostic lines: ' + str(numlines), be_verbose)

''' Normfile read in, parameter file read in, P value calculation'''
normspectrum = np.loadtxt(thespectrumfile)

params = []
with open(theparamfile) as f:
    content = f.readlines()
# Only use in the parameters that have been varied
for aline in content:
    if aline.split()[0] in x.columns:
        params.append(aline.split())

min_redchi2_value = min(x['chi2'])
do_print('Min chi2: ' + str(min_redchi2_value), be_verbose)

normspec_wave, normspec_flux, normspec_err = normspectrum.T

specpoints = 0
for i in range(len(line_norm_starts)):
    npoints_line = len(parallelcrop(normspec_wave, normspec_flux,
        line_norm_starts[i], line_norm_stops[i], normspec_err)[0])
    specpoints = specpoints + npoints_line

do_print('best fitness : ' + str(max(x['fitness'])), be_verbose)
do_print('best chi2    : ' + str(min(x['chi2'])), be_verbose)
do_print('best red_chi2: ' + str(min(x['chi2'])/(specpoints - len(params))),
    be_verbose)
probabilities = calculateP(params, specpoints, x['chi2'], True)
x = x.assign(P=probabilities)

params_dic = OrderedDict()
params_error_1sig = OrderedDict()
params_error_2sig = OrderedDict()

# The p-value is the probability of observing a test statistic
# at least as extreme in a chi-square distribution.
min_p_1sig = 0.317
min_p_2sig = 0.0455
best = pd.Series.idxmax(x['P'])
ind_1sig = x['P'] > min_p_1sig
ind_2sig = x['P'] > min_p_2sig

for i in params:
    params_dic[i[0]] = [float(i[1]), float(i[2])]
    params_error_1sig[i[0]] = [min(x[i[0]][ind_1sig]),
        max(x[i[0]][ind_1sig]), x[i[0]][best]]
    params_error_2sig[i[0]] = [min(x[i[0]][ind_2sig]),
        max(x[i[0]][ind_2sig]), x[i[0]][best]]

params_error = params_error_2sig

param_keys = params_dic.keys()

''' Mokiem errors '''

if best_models_cutoff_method == 'Mokiem':
    def _skew_gauss_vec(x, A, x0, w, b, c):
        """vectorised version of skewed Gaussian, called by skew_gauss"""
        ndeps = np.finfo(x.dtype.type).eps
        lim0 = 2.*np.sqrt(ndeps)
        # Through experimentation I found 2*sqrt(machine_epsilon) to be
        # a good safe threshold for switching to the b=0 limit
        # at lower thresholds, numerical rounding errors appear
        if (abs(b) <= lim0):
            sg = 1 + c + A * np.exp(-4*np.log(2)*(x-x0)**2/w**2)
        else:
            lnterm = 1.0 + ((2*b*(x-x0))/w)
            sg = np.zeros_like(lnterm)  + 1 + c
            sg[lnterm>0] =\
                1 + c + A * np.exp(-np.log(2)*(np.log(lnterm[lnterm>0])/b)**2)
        return sg

    def skew_gauss(x, A, x0, w, b, c):
        """Fraser-Suzuki skewed Gaussian.

        A: peak height, x0: peak position,
        w: width, b: skewness"""
        if type(x)==np.ndarray:
            sg = _skew_gauss_vec(x, A, x0, w, b, c)
        else:
            x = float(x)
            ndeps = np.finfo(type(x)).eps
            lim0 = 2.*np.sqrt(ndeps)
            if (abs(b) <= lim0):
                sg = 1 + c + A * np.exp(-4*np.log(2)*(x-x0)**2/w**2)
            else:
                lnterm = 1.0 + ((2*b*(x-x0))/w)
                if (lnterm>0):
                    sg = 1 + c + A * np.exp(-np.log(2)*(np.log(lnterm)/b)**2)
                else:
                    sg = 0
        return sg

    start = 0.00
    stop = 1.0
    step = 0.005
    nsteps = (stop - start)/step + 1
    histbins = np.linspace(start, stop, nsteps)

    x['fitness_norm'] = x['fitness']/np.max(x['fitness'])

    histmids = 0.5*(histbins[:-1]+ histbins[1:])

    plt.figure()
    histvals = plt.hist(x['fitness_norm'], histbins, label='fitness')
    plt.close()

    histvals = histvals[0]
    upper_histmids = histmids[int(len(histmids)/2):]
    upper_histvals = histvals[int(len(histvals)/2):]
    popt, pcov = curve_fit(skew_gauss, upper_histmids, upper_histvals,
        p0 = [200, 0.85, 0.1, 0.1 ,0])

    histmids_hires = np.linspace(min(upper_histmids), max(upper_histmids), 1000)
    skewgaussfit_histvals = skew_gauss(histmids_hires, *popt)
    peak_val = histmids_hires[np.argmax(skewgaussfit_histvals)]

    print('Mokiem errors: Peak val = ' + str(round(peak_val,3)) +
        ' (maxgen = ' + str(maxgen) + ')')

    fig, ax = plt.subplots(1, 1, figsize=(15, 2.5))
    ax.set_title('Fitness distribution (#models)')
    histvals = ax.hist(x['fitness_norm'], histbins, label='fitness')
    ax.axvline(peak_val, color='C1')
    ax.plot(histmids_hires, skewgaussfit_histvals)
    ax.legend()
    plt.tight_layout()
    plt.close()

    params_error_Mokiem = OrderedDict()

    # The p-value is the probability of observing a test statistic
    # at least as extreme in a chi-square distribution.
    best_mokiem = pd.Series.idxmax(x['fitness_norm'])
    ind_mokiem = x['fitness_norm'] > peak_val

    for i in params:
        params_error_Mokiem[i[0]] = [min(x[i[0]][ind_mokiem]),
            max(x[i[0]][ind_mokiem]), x[i[0]][best_mokiem]]

    params_error = params_error_Mokiem

''' Loading extra spectra '''

if extra_spectrum_path1 == '' and extra_spectrum_path1 == '':
    include_extra_spectra = False

if make_lineprofiles_plot and include_extra_spectrum:

    do_print('Reading extra spectra or models...', be_verbose)
    if extra_spectrum_path1 == '':
        pass
    elif os.path.isfile(extra_spectrum_path1):
        try:
            cmfgenwave1, cmfgenflux1 = np.genfromtxt(extra_spectrum_path1).T
        except:
            cmfout = np.genfromtxt(extra_spectrum_path1).T
            cmfgenwave1, cmfgenflux1, cmfgenerror1 = cmfout
    else:
        do_print("Cannot plot extra spectra, file not found: " +
            extra_spectrum_path1, True)

    if extra_spectrum_path2 == '':
        pass
    elif os.path.isfile(extra_spectrum_path2):
        try:
            cmfgenwave2, cmfgenflux2 = np.genfromtxt(extra_spectrum_path2).T
        except:
             cmfout2 = np.genfromtxt(extra_spectrum_path2).T
             cmfgenwave2, cmfgenflux2, cmfgenerror2 = cmfout2
    else:
        do_print("Cannot plot extra spectra, file not found: " +
            extra_spectrum_path2, True)

''' Absolutely ridiculous workaround to get a legend'''
# Namely here I create a pdf that contains the colorbar,
# and then later I import the image of that colorbar into the figure

colorbar_jet_legend = "colorbar_generations_jet_" + runname + ".jpg"

a = np.array([[0,1]])
plt.figure(figsize=(9, 3.5))
img = plt.imshow(a, cmap="jet")
plt.gca().set_visible(False)
cax = plt.axes([0.05, 0.3, 0.9, 0.1])#[0.3, 0.3, 0.8, 0.8])
cbar = plt.colorbar(cax=cax, orientation='horizontal', ticks=np.linspace(0,1,5))
cbar.ax.set_xticklabels(['0', str(int(0.25*float(maxgenid))),
    str(int(0.5*float(maxgenid))), str(int(0.75*float(maxgenid))),
    str(int(float(maxgenid)))], fontsize=30)
cbar.ax.set_xlabel('Generation',fontsize=30)
plt.savefig(colorbar_jet_legend, dpi=300)
plt.close()


''' ----------------------------------------------------------------------'''
'''                          Make pdfs with plots                         '''
''' ----------------------------------------------------------------------'''

''' ------------------------------------------'''
'''            Derived quantities             '''
''' ------------------------------------------'''

if (make_derived_fitness_plots or make_run_summary) and not no_radius_provided:
    def get_luminosity(Teff, radius):
        '''Calculate L in terms of log(L/Lsun), given Teff (K)
        and the radius in solar radii'''

        sigmaSB = 5.67051e-5
        Lsun = 3.9e33
        Rsun = 6.96e10

        radius_cm = radius * Rsun

        if isinstance(Teff, str):
            Teff = float(Teff)
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

    def get_vesc(mass, radius):
        Rsun = 6.96e10
        Msun = 1.99e33
        Gcgs = 6.67259e-8

        vesc_cms = np.sqrt((2*Gcgs*mass*Msun)/(radius*Rsun))
        vesc_kms = vesc_cms*1e-5

        return vesc_kms

    def get_logmodwindmom(mdot, radius, vinf, clumping=1):
        ''' Give mdot in log space, radius is solar radii, and vinf in km/s'''
        Msun = 1.99e33
        year = 365*24*3600

        vinf_cms = vinf*1e5
        mdot_cgs = 10**mdot * Msun / year
        dmom = (mdot_cgs/np.sqrt(clumping)) * vinf_cms * np.sqrt(radius)
        logdmom = np.log10(dmom)

        return logdmom

    def get_fx(mdot, vinf):
        """ Estimates fx based on the Mdot and vinf, based on the
        power law of Kudritzki, Palsa, Feldmeier et al. (1996). This power law
        is extrapolated also outside where Kudritzki+96 have data points.
        """

        mdot = 10**mdot / 10**(-6)
        logmdotvinf = np.log10(mdot/vinf)

        # Relation from Kudritzki, Palsa, Feldmeier et al. (1996)
        logfx = -5.45 - 1.05*logmdotvinf
        # fx = 10**(logfx)

        return logfx

    # Some derived parameters are always included
    derived_parameters = ['radius'
                            ,'luminosity'
                            ,'specmass'
                            ,'vesc']

    if 'teff' in var_dct.keys():
        x['luminosity'] = get_luminosity(x['teff'], x['radius'])
    else:
        x['luminosity'] = get_luminosity(fix_dct['teff'], x['radius'])

    if 'logg' in var_dct.keys():
        x['specmass'] =  get_mass(x['logg'], x['radius'])
    else:
        x['specmass'] =  get_mass(float(fix_dct['logg']), x['radius'])

    x['vesc'] = get_vesc(x['specmass'], x['radius'])

    # Other derived parameters are only computed when relevant.
    if 'vinf' in var_dct.keys():
        x['inf_esc'] = x['vinf']/x['vesc']
        derived_parameters.append('inf_esc')

    if 'windturb' in var_dct.keys() and 'vinf' in var_dct.keys():
        x['windturb_kms'] = x['windturb'] * x['vinf']
        derived_parameters.append('windturb_kms')

    if 'mdot' in var_dct.keys() and 'vinf' in var_dct.keys():
        x['dmom'] = get_logmodwindmom(x['mdot'], x['radius'], x['vinf'])
        derived_parameters.append('dmom')
        if 'fclump' in var_dct.keys():
            x['dmom_clump'] = get_logmodwindmom(x['mdot'], x['radius'],
                x['vinf'], x['fclump'])
            derived_parameters.append('dmom_clump')

    if not no_xlum_provided:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("ignore", RuntimeWarning)
            if 'fx' in fix_dct.keys():
                x['logxlum'] = np.log10(x['xlum'])
                derived_parameters.append('logxlum')
                if float(fix_dct['fx']) > 1000.0:
                    if 'mdot' in var_dct.keys():
                        derived_parameters.append('logfx')
                        if 'vinf' in var_dct.keys():
                            x['logfx'] = get_fx(x['mdot'], x['vinf'])
                        elif 'vinf' in fix_dct.keys():
                            x['logfx'] = get_fx(x['mdot'],
                                np.ones(len(x['mdot']))*float(fix_dct['vinf']))
                        else:
                            x['logfx'] = get_fx(x['mdot'], x['vesc']*2.6)
            else:
                x['logfx'] = float(fix_dct['fx'])*np.ones(len(x))
    if 'logfclump' in var_dct.keys():
        derived_parameters.append('fclump')
        x['fclump'] = 10**x['logfclump']


    # Get the best values and error margins on the derived parameters
    derivedparams_dic = OrderedDict()
    derivedpar_error1sig_dic = OrderedDict()
    derivedpar_error2sig_dic = OrderedDict()
    for i in derived_parameters:
        derivedparams_dic[i] = [np.nanmin(x[derived_parameters]),
            np.nanmax(x[derived_parameters])]
        derivedpar_error1sig_dic[i] = [np.nanmin(x[i][ind_1sig]),
            np.nanmax(x[i][ind_1sig]), x[i][best]]
        derivedpar_error2sig_dic[i] = [np.nanmin(x[i][ind_2sig]),
            np.nanmax(x[i][ind_2sig]), x[i][best]]

    x['invchi']= 1./x['rchi2']


if make_derived_fitness_plots and not no_radius_provided:
    # Only if the number of derived parameters will exceed 12 it's
    # needed to rewrite the script so that it can include multiple pages.
    nrows_ppage = 4
    ncols_ppage = 3

    if control_dct["fitmeasure"] == 'fitness':
        fitmeasurelist = ['P', 'invchi' , 'fitness']
    else:
        fitmeasurelist = ['P', 'invchi']

    for fitmeasure in fitmeasurelist:
        fig, ax = plt.subplots(nrows_ppage,ncols_ppage, figsize=(12,12*1.41))
        lp = -1 # paramater counter
        for arow in xrange(nrows_ppage):
            for acol in xrange(ncols_ppage):
                lp = lp + 1 # Next parameter is plotted
                if lp < len(derived_parameters):
                    derpar = derived_parameters[lp]
                    ax[arow,acol].scatter(x[derpar], x[fitmeasure],
                        s=12, edgecolors='none', color='darkblue', alpha=1.0)
                    ax[arow,acol].axvspan(derivedpar_error1sig_dic[derpar][0],
                        derivedpar_error1sig_dic[derpar][1], color='gold',
                        alpha=0.6, zorder=0)
                    ax[arow,acol].axvspan(derivedpar_error2sig_dic[derpar][0],
                        derivedpar_error2sig_dic[derpar][1], color='gold',
                        alpha=0.25, zorder=0)
                    ax[arow,acol].axvline(derivedpar_error2sig_dic[derpar][2],
                        color='black', zorder=0)
                    ax[arow,acol].set_xlabel(derpar)
                    ax[arow,acol].set_ylabel(fitmeasure)
                    if derpar == 'fclump':
                        ax[arow,acol].set_xlim(1.0, 51.0)
                else:
                    ax[arow,acol].axis('off')

        fig.suptitle('Derived parameters (' + str(max(x['gen']))
            + ' generations) ')
        fig.tight_layout(rect=[0, 0.01, 1, 0.97])
        derivedpar_jpgname = 'derived_tmp' + runname + '.jpg'
        plt.savefig(derivedpar_jpgname, dpi=400)
        plt.close()

        if fitmeasure == 'P':
            with open(plotpath + name_derivedparP_plot, "wb") as out_file:
                out_file.write(img2pdf.convert(derivedpar_jpgname))
                os.system('rm ' + derivedpar_jpgname)
        elif fitmeasure == 'invchi':
            with open(plotpath + name_derivedpar_plot, "wb") as out_file:
                out_file.write(img2pdf.convert(derivedpar_jpgname))
                os.system('rm ' + derivedpar_jpgname)
        elif fitmeasure == 'fitness':
            with open(plotpath + name_derivedparfitness_plot, "wb") as out_file:
                out_file.write(img2pdf.convert(derivedpar_jpgname))
                os.system('rm ' + derivedpar_jpgname)

    # plt.show()
    # sys.exit()


''' ------------------------------------------'''
'''    Write run summary to pdf               '''
''' ------------------------------------------'''

if make_run_summary:

    if os.path.isfile(best_fit_pars_txt):
        os.system('rm ' + best_fit_pars_txt)

    fig, ax = plt.subplots(1,2,figsize=(12,12*1.41))

    # Set spacing depending on number of free parameters.
    maxfreepar = 20.
    if len(params_error) > 30:
        raw_input('\n\n\nWARNING --> Too many free parameters to print\n\n\n')
    elif len(params_error) > 20:
        maxfreepar = len(params_error)

    # add extra whitespace for readability
    ew = 1.3

    # Font sizes for values + error values
    fz = 12
    fz_err = 10

    numround = 3

    pls = 1./maxfreepar
    base = 0.5 * pls
    base2 = 0.6 * pls

    # column width in textfile
    cwtxt = 15

    for lp in range(len(param_keys)):
        strpname = param_keys[lp]
        strbestp = str(round(params_error[strpname][2],numround))
        strdiffmin = str(round(params_error[strpname][2] -
            params_error[strpname][0],numround))
        strdiffplus =  str(round(params_error[strpname][1] -
            params_error[strpname][2],numround))
        strdiffmin_1sig = str(round(params_error_1sig[strpname][2] -
            params_error_1sig[strpname][0],numround))
        strdiffplus_1sig =  str(round(params_error_1sig[strpname][1] -
            params_error_1sig[strpname][2],numround))
        strrange = ('[' + str(params_error[strpname][0]) + ', '
            +  str(params_error[strpname][1]) + ']')
        strrange_1sig = ('[' + str(params_error_1sig[strpname][0]) + ', '
            +  str(params_error_1sig[strpname][1]) + ']')

        ax[0].text(0.05, 1.-base-lp*ew*pls, strpname,
            transform=ax[0].transAxes, fontsize=fz, va='top')
        ax[0].text(0.35, 1.-base-lp*ew*pls, strbestp,
            transform=ax[0].transAxes, fontsize=fz, va='top', ha='right')

        hpls = pls*0.25*0.5
        ax[0].text(0.37, (1.-base-lp*ew*pls)+hpls, '+',
            transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.37, (1.-base-lp*ew*pls)-hpls, '-',
            transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.40, (1.-base-lp*ew*pls)+hpls, strdiffplus_1sig,
            transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.40, (1.-base-lp*ew*pls)-hpls, strdiffmin_1sig,
            transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.60, 1.-base-lp*ew*pls, strrange_1sig,
            transform=ax[0].transAxes, fontsize=fz_err, va='top')

        ax[0].text(0.37, (1.-base2-lp*ew*pls)+hpls - 0.5*pls, '+',
            transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.37, (1.-base2-lp*ew*pls)-hpls - 0.5*pls, '-',
            transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.40, (1.-base2-lp*ew*pls)+hpls - 0.5*pls, strdiffplus,
            transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.40, (1.-base2-lp*ew*pls)-hpls - 0.5*pls, strdiffmin,
            transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.60, 1.-base2-lp*ew*pls - 0.5*pls, strrange,
            transform=ax[0].transAxes, fontsize=fz_err, va='top')

        bestparstr = (strpname.ljust(cwtxt) + strbestp.ljust(cwtxt) +
            strdiffplus_1sig.ljust(cwtxt) + strdiffmin_1sig.ljust(cwtxt) +
            strdiffplus.ljust(cwtxt) + strdiffmin + '\n')
        with open(best_fit_pars_txt, 'a') as myfile:
            myfile.write(str(bestparstr))

    maxlenrunname = 15
    if len(runname) > maxlenrunname:
        printrunname = runname[:maxlenrunname] + '\n' + runname[maxlenrunname:]
    else:
        printrunname = runname

    run_info_labels = []
    run_info_data = []
    run_info_labels.append('Run name')
    run_info_data.append(printrunname)
    run_info_labels.append('# Generations')
    run_info_data.append(str(int(max(x['gen_id']))))
    run_info_labels.append('# Population size')
    run_info_data.append(str(int(maxindid)))
    run_info_labels.append('# Free parameters')
    run_info_data.append(len(param_keys))
    # run_info_labels.append('Fitness best model')
    # run_info_data.append(str(round(max(x['fitness']),5)))
    run_info_labels.append('Chi2 best model')
    run_info_data.append(str(round(min(x['rchi2']),5)))

    for i in xrange(len(run_info_data)):
        ax[1].text(0.05, 1.-pls-i*ew*pls*0.5+base, run_info_labels[i],
            transform=ax[1].transAxes, fontsize=fz, va='top')
        ax[1].text(0.55, 1.-pls-i*ew*pls*0.5+base, run_info_data[i],
            transform=ax[1].transAxes, fontsize=fz, va='top')

    if not no_radius_provided:

        numround = 2

        for lp in range(len(derivedpar_error1sig_dic)):
            strpname = derivedpar_error1sig_dic.keys()[lp]
            if strpname == 'vesc':
                numround_tmp = 0
            else:
                numround_tmp = numround
            strbestp = str(round(derivedpar_error2sig_dic[strpname][2],
                numround_tmp))
            strdiffmin = str(round(derivedpar_error2sig_dic[strpname][2] -
                derivedpar_error2sig_dic[strpname][0],numround_tmp))
            strdiffplus =  str(round(derivedpar_error2sig_dic[strpname][1] -
                derivedpar_error2sig_dic[strpname][2],numround_tmp))
            strdiffmin_1sig = str(round(derivedpar_error1sig_dic[strpname][2] -
                derivedpar_error1sig_dic[strpname][0],numround_tmp))
            strdiffplus_1sig =  str(round(derivedpar_error1sig_dic[strpname][1] -
                derivedpar_error1sig_dic[strpname][2],numround_tmp))
            strrange = ('[' + str(round(derivedpar_error2sig_dic[strpname][0],
                numround_tmp)) + ', '
                +  str(round(derivedpar_error2sig_dic[strpname][1],
                    numround_tmp)) + ']')
            strrange_1sig = ('[' + str(round(derivedpar_error1sig_dic[strpname][0],
                numround_tmp)) + ', '
                +  str(round(derivedpar_error1sig_dic[strpname][1],
                    numround_tmp)) + ']')

            ax[1].text(0.05, 0.71-base-lp*ew*pls, strpname,
                transform=ax[1].transAxes, fontsize=fz, va='top')
            ax[1].text(0.50, 0.71-base-lp*ew*pls, strbestp,
                transform=ax[1].transAxes, fontsize=fz, va='top', ha='right')

            hpls = pls*0.25*0.5
            ax[1].text(0.52, (0.71-base-lp*ew*pls)+hpls, '+',
                transform=ax[1].transAxes, fontsize=fz_err, va='top')
            ax[1].text(0.52, (0.71-base-lp*ew*pls)-hpls, '-',
                transform=ax[1].transAxes, fontsize=fz_err, va='top')
            ax[1].text(0.55, (0.71-base-lp*ew*pls)+hpls, strdiffplus_1sig,
                transform=ax[1].transAxes, fontsize=fz_err, va='top')
            ax[1].text(0.55, (0.71-base-lp*ew*pls)-hpls, strdiffmin_1sig,
                transform=ax[1].transAxes, fontsize=fz_err, va='top')
            ax[1].text(0.70, 0.71-base-lp*ew*pls, strrange_1sig,
                transform=ax[1].transAxes, fontsize=fz_err, va='top')

            ax[1].text(0.52, (0.71-base2-lp*ew*pls)+hpls - 0.5*pls, '+',
                transform=ax[1].transAxes, fontsize=fz_err, va='top')
            ax[1].text(0.52, (0.71-base2-lp*ew*pls)-hpls - 0.5*pls, '-',
                transform=ax[1].transAxes, fontsize=fz_err, va='top')
            ax[1].text(0.55, (0.71-base2-lp*ew*pls)+hpls - 0.5*pls, strdiffplus,
                transform=ax[1].transAxes, fontsize=fz_err, va='top')
            ax[1].text(0.55, (0.71-base2-lp*ew*pls)-hpls - 0.5*pls, strdiffmin,
                transform=ax[1].transAxes, fontsize=fz_err, va='top')
            ax[1].text(0.70, 0.71-base2-lp*ew*pls - 0.5*pls, strrange,
                transform=ax[1].transAxes, fontsize=fz_err, va='top')

            bestparstr = (strpname.ljust(cwtxt) + strbestp.ljust(cwtxt) +
                strdiffplus_1sig.ljust(cwtxt) + strdiffmin_1sig.ljust(cwtxt) +
                strdiffplus.ljust(cwtxt) + strdiffmin + '\n')
            with open(best_fit_pars_txt, 'a') as myfile:
                myfile.write(str(bestparstr))

    ax[0].set_title('Parameter fit summary', fontsize=fz+2)
    ax[1].set_title('Meta info + derived parameters', fontsize=fz+2)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])

    plt.savefig(plotpath + name_runsummary)
    plt.close()


''' --------------------------------------------------------------------'''
'''    Convergence: best parameters and their errors over time           '''
''' --------------------------------------------------------------------'''

if make_convergence_analysis_plot:

    save_convergence = []
    save_conv_parnames = []

    do_print('Computing convergence, this can take a minute ... ', be_verbose)

    # -----------
    # COMPUTATION
    # -----------

    error_vals_1s_low = []
    error_vals_1s_up = []
    error_vals_2s_low = []
    error_vals_2s_up = []
    error_vals_best = []
    for i in range(len(params)):
        error_vals_1s_low.append([])
        error_vals_1s_up.append([])
        error_vals_2s_low.append([])
        error_vals_2s_up.append([])
        error_vals_best.append([])

    specpoints_utg = -1
    ct = -1
    while specpoints_utg == -1:
        specpoints_utg = float(x['dof'].iloc[ct])
        ct = ct-1
    specpoints_utg = specpoints_utg + len(params)
    min_p_1sig_utg = 0.317
    min_p_2sig_utg = 0.0455

    xgen_utg = np.unique(x['gen'])#[:20]

    for agen in xgen_utg:

        params_dic_utg = OrderedDict()
        params_error_1sig_utg = OrderedDict()
        params_error_2sig_utg = OrderedDict()

        x_utg = x.copy(deep=True)
        x_utg = x_utg.loc[x_utg['gen'] <= agen]

        probabilities = calculateP(params, specpoints_utg, x_utg['chi2'], True)
        x_utg = x_utg.assign(P=probabilities)

        best_utg = pd.Series.idxmax(x_utg['P'])
        ind_1sig_utg = x_utg['P'] > min_p_1sig_utg
        ind_2sig_utg = x_utg['P'] > min_p_2sig_utg

        for p, i in zip(params, range(len(params))):
            error_vals_1s_low[i].append(min(x_utg[p[0]][ind_1sig_utg]))
            error_vals_1s_up[i].append(max(x_utg[p[0]][ind_1sig_utg]))
            error_vals_2s_low[i].append(min(x_utg[p[0]][ind_2sig_utg]))
            error_vals_2s_up[i].append(max(x_utg[p[0]][ind_2sig_utg]))
            error_vals_best[i].append(x_utg[p[0]][best_utg])
        if agen % 25 == 0:
            do_print('Calculating convergence ' + str(agen) + '/'
                + str(len(xgen_utg)), be_verbose)

    # --------
    # PLOTTING
    # --------

    plotscalefactor = 12.0
    v_param_pp = 6
    maxgenperline = 25

    tickspacing = 5

    x['gen_id_num'] = x['gen_id'].astype(float)
    latest_gen = max(x['gen_id_num'])
    nsubpages = int(math.ceil(float(latest_gen)/maxgenperline))
    if latest_gen == 0:
        gens_pp = 1
    else:
        gens_pp = latest_gen/nsubpages

    # If the 'all generations plot' is already zoomed because
    # there are not so many generations, never make the zoom.
    if latest_gen <= maxgenperline:
        make_param_dist_plot_detail = False

    nparams = len(param_keys)
    npages = int(math.ceil(1.0*nparams/v_param_pp))

    rowwidth = plotscalefactor*1.41
    colwidth = plotscalefactor

    line_pdf_names = []
    paramcount = 0
    for npage in range(npages):
        fig, ax = plt.subplots(v_param_pp,1, figsize=(colwidth,rowwidth),
            sharex=False)
        make_legend = True
        for i in range(v_param_pp):

            nticks = math.floor(int(maxgenid)/tickspacing)
            genticks = np.linspace(0, nticks*tickspacing, nticks+1)

            paramcount = paramcount + 1
            if paramcount <= nparams:
                p = params[paramcount-1]
                save_convergence.append(np.array([xgen_utg,
                    error_vals_2s_low[paramcount-1],
                    error_vals_1s_low[paramcount-1],
                    error_vals_best[paramcount-1],
                    error_vals_1s_up[paramcount-1],
                    error_vals_2s_up[paramcount-1]]).T)
                save_conv_parnames.append(p[0])
                if make_legend:
                    ax[i].fill_between(xgen_utg, error_vals_2s_low[paramcount-1],
                        error_vals_2s_up[paramcount-1], color='#fed976',
                        alpha=0.8, label=r'$2\sigma$')
                    ax[i].fill_between(xgen_utg, error_vals_1s_low[paramcount-1],
                        error_vals_1s_up[paramcount-1], color='#fc4e2a',
                        alpha=0.5, label=r'$1\sigma$')
                    ax[i].plot(xgen_utg, error_vals_best[paramcount-1],
                        color='#bd0026', lw=2, label='best fit')
                    make_legend = False
                else:
                    ax[i].fill_between(xgen_utg, error_vals_2s_low[paramcount-1],
                        error_vals_2s_up[paramcount-1], color='#fed976',
                        alpha=0.8)
                    ax[i].fill_between(xgen_utg, error_vals_1s_low[paramcount-1],
                        error_vals_1s_up[paramcount-1], color='#fc4e2a',
                        alpha=0.5)
                    ax[i].plot(xgen_utg, error_vals_best[paramcount-1],
                        color='#bd0026', lw=2)
                ax[i].set_ylim(float(p[1]), float(p[2]))
                ax[i].set_ylabel(p[0])
            else:
                ax[i].axis('off')

            if i == v_param_pp - 1 or paramcount == nparams:
                # FIXME for some reason if there are empty axes
                # on the last page, then ngen is not indicated
                ax[i].set_xticks(genticks)
                ax[i].set_xticklabels(genticks.astype('int'))
            else:
                ax[i].set_xticks([])
                ax[i].set_xlabel("")
                ax[i].set_xticklabels([])

        fig.suptitle('Convergence')
        fig.legend(loc='upper right',ncol=3)
        fig.tight_layout(rect=[0, 0.01, 1, 0.97])
        fig.subplots_adjust(wspace=0, hspace=0)

        # Saving page in pdf. Pdfs will be sticked later.
        pdfname_tmp = plotpath + 'convergence_p' + str(int(npage)) + '.pdf'
        line_pdf_names.append(pdfname_tmp)
        plt.savefig(pdfname_tmp)
        plt.close()
        do_print("Convergence plots: saved page " + str(int(npage+1))
            + " out of " + str(int(npages)) + " pages.", be_verbose)

    # Merge all line profile plot pages into one document.
    if os.path.isfile(plotpath + name_convergence_plot):
        os.system("rm " + plotpath + name_convergence_plot)
    merger = PdfFileMerger()
    for filename in line_pdf_names:
        merger.append(PdfFileReader(file(filename, 'rb')))
    merger.write(plotpath + name_convergence_plot)

    # Remove the individual pages after the file has been merged.
    for filename in line_pdf_names:
        os.system("rm " + filename)

    # Save the convergence for each parameter to a text file
    for i in range(len(save_conv_parnames)):
        np.savetxt(datapath + 'convergence_' + save_conv_parnames[i] +
            '.txt', save_convergence[i])

''' ---------------------------------------'''
'''    Fit fitness distribution            '''
''' ---------------------------------------'''

def fitpoly(x, c1, c2, c3, c4, c5, n0, n1, n2, n3, n4, n5):
    y = (n0 + n1*(x-c1) + n2*(x-c2)**2 + n3*(x-c3)**3 +
        n4*(x-c4)**4 + n5*(x-c5)**5)
    return y

def fitpoly_simple(x, c1, c2, c3, c4, c5, n0, n1, n2, n3, n4, n5):
    y = (n0 + n1*(x-c1) + n2*(x-c2)**2 + n3*(x-c3)**3 +
        n4*(x-c4)**4 + n5*(x-c5)**5)
    return y

def fit_top_distr(all_x, x_data, y_data, mp, f_err=0.70, dlim=0.001,
    be_verbose=False):
    """Given a set of chi2 values as a function of parameter,
    return the best fit to the *top* of that distribution, so that
    no points lie below the fit.

    Input:
      -  all_x: array of all possible values of x
      -  x_data: parameter values of the data
      -  y_data: chi2 values of the data

    Returns:
      -  polybestfit: the best fitting polynomial as a function
         of all_x.
    """
    param_x = []
    param_maxy = []
    for abin in all_x:
        if abin in x_data:
            param_x.append(abin)
            bargs = np.argwhere(x_data == abin)
            param_maxy.append(np.max(y_data[bargs]))

    maxdiff = 10.0
    count_loops = 0
    error_noise = np.ones(len(param_maxy))
    while maxdiff > dlim:
        popt, pcov = scipy.optimize.curve_fit(fitpoly_simple, param_x,
            param_maxy, sigma=error_noise,p0=[mp, mp, mp, mp, mp,
                0, 0, 0, 0, 0, 0])
        currentbestfit = fitpoly_simple(param_x, *popt)
        diff = param_maxy - currentbestfit
        hargs = np.argwhere(diff > 0)
        error_noise[hargs] = error_noise[hargs]*f_err
        plotybestfit = fitpoly_simple(all_x, *popt)

        if len(diff[hargs]) > 0:
            maxdiff = np.max(diff[hargs])
        else:
            maxdiff = 0.0
        count_loops = count_loops + 1

    if be_verbose:
        print('Iterations: ' + str(count_loops))
        print(popt)

    return plotybestfit

if make_fitness_fit_analysis_plot:

    error_vals_1s_low = []
    error_vals_1s_up = []
    error_vals_2s_low = []
    error_vals_2s_up = []
    error_vals_best = []
    for i in range(len(params)):
        error_vals_1s_low.append([])
        error_vals_1s_up.append([])
        error_vals_2s_low.append([])
        error_vals_2s_up.append([])
        error_vals_best.append([])

    specpoints_utg = x['dof'].iloc[-1] - len(params)
    min_p_1sig_utg = 0.317
    min_p_2sig_utg = 0.0455

    xgen_utg = np.unique(x['gen'])#[:20]

    for agen in xgen_utg:

        params_dic_utg = OrderedDict()
        params_error_1sig_utg = OrderedDict()
        params_error_2sig_utg = OrderedDict()

        x_utg = x.copy(deep=True)
        x_utg = x_utg.loc[x_utg['gen'] <= agen]
        probabilities = calculateP(params, specpoints_utg, x_utg['chi2'], True)
        x_utg = x_utg.assign(P=probabilities)
        x_utg = x_utg.loc[x_utg['P'] > 0.0005]

        best_utg = pd.Series.idxmax(x_utg['P'])
        ind_1sig_utg = x_utg['P'] > min_p_1sig_utg
        ind_2sig_utg = x_utg['P'] > min_p_2sig_utg

        if agen > 30:
            pc = 0
            bestval = x_utg[params[pc][0]][best_utg]
            par_name = params[pc][0]
            par_max = max(x_utg[par_name])
            par_min = min(x_utg[par_name])
            par_step = float(params[pc][3])
            par_nbins = 1 + (par_max - par_min)/par_step
            par_xrange = np.linspace(par_min, par_max, par_nbins)

            par_bestfit = fit_top_distr(par_xrange, x_utg[par_name].values,
                x_utg['P'].values, bestval, f_err=0.95, dlim=0.001,
                    be_verbose=True)
            fig, ax = plt.subplots()
            ax.plot(par_xrange, par_bestfit)
            ax.plot(x_utg[par_name], x_utg['P'], ls='', marker='o')
            ax.set_title(str(agen))
            plt.show()


        for p, i in zip(params, range(len(params))):
            error_vals_1s_low[i].append(min(x_utg[p[0]][ind_1sig_utg]))
            error_vals_1s_up[i].append(max(x_utg[p[0]][ind_1sig_utg]))
            error_vals_2s_low[i].append(min(x_utg[p[0]][ind_2sig_utg]))
            error_vals_2s_up[i].append(max(x_utg[p[0]][ind_2sig_utg]))
            error_vals_best[i].append(x_utg[p[0]][best_utg])
        if agen % 25 == 0:
            do_print('Calculating convergence ' + str(agen) + '/' +
                str(len(xgen_utg)), be_verbose)

''' ---------------------------------------------------------------'''
'''     Make distribution violin plots as function of generation   '''
''' ---------------------------------------------------------------'''

if make_param_dist_plot or make_param_dist_plot_detail:

    plotscalefactor = 12.0
    v_param_pp = 6
    maxgenperline = 25

    tickspacing = 5

    x['gen_id_num'] = x['gen_id'].astype(float)
    latest_gen = max(x['gen_id_num'])
    nsubpages = int(math.ceil(float(latest_gen)/maxgenperline))
    if latest_gen == 0:
        latest_gen = 1
        gens_pp = 1
    else:
        gens_pp = latest_gen/nsubpages

    # If the 'all generations plot' is already zoomed because
    # there are not so many generations, never make the zoom.
    if latest_gen <= maxgenperline:
        make_param_dist_plot_detail = False

    nparams = len(param_keys)
    npages = int(math.ceil(1.0*(nparams+1)/v_param_pp))

    rowwidth = plotscalefactor*1.41
    colwidth = plotscalefactor

    do_print('Start making the violin parameter distribution plots ...',
        be_verbose)

    line_pdf_names = []
    if make_param_dist_plot:
        paramcount = 0
        for npage in range(npages):
            fig, ax = plt.subplots(v_param_pp,1, figsize=(colwidth,rowwidth),
                sharex=False)

            for i in range(v_param_pp):

                nticks = math.floor(int(maxgenid)/tickspacing)
                genticks = np.linspace(0, nticks*tickspacing, nticks+1)

                paramcount = paramcount + 1
                if paramcount <= nparams:
                    sns.violinplot( x=x['gen_id'],
                    y=x[param_keys[v_param_pp*npage+i]],
                    linewidth=0.1, scale='width', ax=ax[i], cut=0,
                    palette="Blues")
                elif paramcount == nparams + 1:
                    thechi2min = np.nanmin(x['chi2'].values)
                    chi2bound = thechi2min*10
                    chi2crop =  x.loc[x['chi2'] < 999999999]
                    chi2crop =  chi2crop['chi2'].loc[chi2crop['rchi2'] < 1000.0].values
                    sns.violinplot( x=x['gen_id'], y=chi2crop,
                    linewidth=0.1, scale='width', ax=ax[i],
                        cut=0, palette="husl")
                    ax[i].set_ylabel('chi2')
                    ax[i].set_ylim(thechi2min, chi2bound)
                else:
                    ax[i].axis('off')

                if i == v_param_pp - 1 or paramcount == nparams + 1:
                    # FIXME for some reason if there are empty axes
                    # on the last page, then ngen is not indicated
                    ax[i].set_xticks(genticks)
                    ax[i].set_xticklabels(genticks.astype('int'))
                else:
                    ax[i].set_xticks([])
                    ax[i].set_xlabel("")
                    ax[i].set_xticklabels([])

            plt.tight_layout()
            fig.subplots_adjust(wspace=0, hspace=0)

            # Saving page in pdf. Pdfs will be sticked later.
            pdfname_tmp = plotpath + 'param_dist_p' + str(int(npage)) + '.pdf'
            line_pdf_names.append(pdfname_tmp)
            plt.savefig(pdfname_tmp)
            plt.close()
            do_print("Parameter distribution (violinplot): saved page "
                + str(int(npage+1)) + " out of " + str(int(npages)) + " pages.",
                be_verbose)

    # Merge all line profile plot pages into one document.
    if os.path.isfile(plotpath + name_param_dist_plot):
        os.system("rm " + plotpath + name_param_dist_plot)
    merger = PdfFileMerger()
    for filename in line_pdf_names:
        merger.append(PdfFileReader(file(filename, 'rb')))
    merger.write(plotpath + name_param_dist_plot)

    # Remove the individual pages after the file has been merged.
    for filename in line_pdf_names:
        os.system("rm " + filename)

    line_pdf_names = []
    if make_param_dist_plot_detail:

        for nsubpage in  range(nsubpages):
            x_subpage = x.loc[(x['gen_id_num'] >= gens_pp*(nsubpage)) & (x['gen_id_num'] < gens_pp*(nsubpage+1)) ]
            new_mingenid = min(x_subpage['gen_id_num'])
            new_maxgenid = max(x_subpage['gen_id_num'])
            new_ngenid = new_maxgenid - new_mingenid

            paramcount = 0
            for npage in range(npages):
                fig, ax = plt.subplots(v_param_pp,1, figsize=(colwidth,rowwidth),
                    sharex=True)

                for i in range(v_param_pp):
                    paramcount = paramcount + 1
                    if paramcount <= nparams:
                        sns.violinplot( x=x_subpage['gen_id'],
                        y=x_subpage[param_keys[v_param_pp*npage+i]],
                        linewidth=0.1, scale='width', ax=ax[i], cut=0)
                    else:
                        ax[i].axis('off')

                    genticks = []
                    for agenid in range(int(new_mingenid), int(new_maxgenid)):
                        if agenid % tickspacing == 0:
                            genticks.append(agenid)

                    genticks = np.array(genticks)
                    if i == v_param_pp -1 or paramcount == nparams:
                        ax[i].set_xticks(genticks-new_mingenid)
                        ax[i].set_xticklabels(genticks.astype('int'))
                    else:
                        ax[i].set_xticks([])
                        ax[i].set_xlabel("")
                        ax[i].set_xticklabels([])

                    # ax[i].grid('on', linestyle='--')
                plt.tight_layout()
                fig.subplots_adjust(wspace=0, hspace=0)

                # Saving page in pdf. Pdfs will be sticked later.
                pdfname_tmp = (plotpath + 'param_dist_zoom_p' + str(int(npage))
                    + '_' + str(int(nsubpage)) + '.pdf')
                line_pdf_names.append(pdfname_tmp)
                plt.savefig(pdfname_tmp)
                plt.close()
                do_print("Parameter distribution (violinplot): saved page " +
                    str(int(npage+1)) + " subpage " + str(int(nsubpage+1)) +
                    " out of " + str(int(1.0*npages*nsubpages)) + " pages.",
                    be_verbose)

    # Merge all line profile plot pages into one document.
    if os.path.isfile(plotpath + name_param_dist_zoom_plot):
        os.system("rm " + plotpath + name_param_dist_zoom_plot)
    merger = PdfFileMerger()
    for filename in line_pdf_names:
        merger.append(PdfFileReader(file(filename, 'rb')))
    merger.write(plotpath + name_param_dist_zoom_plot)

    # Remove the individual pages after the file has been merged.
    for filename in line_pdf_names:
        os.system("rm " + filename)

if maxgen > 1:
    if make_chi2pgen_plot:
        do_print("Making chi2 as a function of generation plots ...",
            be_verbose)

        pdfs_chi2pgen = []

        mingen = 3500 # very large number means: don't make a zoomed plot
        last_X_generations = int(float(maxgenid)-mingen)
        generation_crop, mutation_crop = parallelcrop(generation, mutation,
            float(mingen), float(maxgenid))

        plotscalefactor = 12.0
        if last_X_generations > 0 and maxgenid > last_X_generations + 10:
            fig, ax = plt.subplots(6,1,
                figsize=(plotscalefactor*1,plotscalefactor*1.41))
            make_zoom = True
        else:
            fig, ax = plt.subplots(3,1,
                figsize=(plotscalefactor*1,plotscalefactor*1.41))
            make_zoom = False

        C0blue = (31./255, 119./255, 180./255)
        C1orange = (255./255, 127./255, 14./255)
        ax[0].plot(median_chi2_per_gen, label=r'median $\chi^2$', color=C0blue)
        ax[1].plot(mean_chi2_per_gen, label=r'mean $\chi^2$', color=C0blue)
        ax[2].plot(lowest_chi2_per_gen, label=r'lowest $\chi^2$', color=C0blue)
        ax_0 = ax[0].twinx()
        ax_0.plot(generation, mutation, color=C1orange, label='mutation rate')
        ax_1 = ax[1].twinx()
        ax_1.plot(generation, mutation, color=C1orange, label='mutation rate')
        ax_2 = ax[2].twinx()
        ax_2.plot(generation, mutation, color=C1orange, label='mutation rate')

        # After 35 generations we expect the solution to be converged
        # so if there are more than 45 generations carried out, we also make
        # a zoomed plot of the chi2 behavior
        if make_zoom:
            xaxisvalue = np.linspace(mingen+1, float(maxgenid),
                (float(maxgenid)-mingen))
            ax[3].plot(xaxisvalue, median_chi2_per_gen[-last_X_generations:],
                label=r'median $\chi^2$', color=C0blue)
            ax[4].plot(xaxisvalue, mean_chi2_per_gen[-last_X_generations:],
                label=r'mean $\chi^2$', color=C0blue)
            ax[5].plot(xaxisvalue, lowest_chi2_per_gen[-last_X_generations:],
                label=r'lowest $\chi^2$', color=C0blue)
            vspanvalues0_y = ax[3].get_ylim()
            vspanvalues1_y = ax[4].get_ylim()
            vspanvalues2_y = ax[5].get_ylim()
            ax_3 = ax[3].twinx()
            ax_3.plot(generation_crop, mutation_crop, color=C1orange,
                label='mutation rate')
            ax_4 = ax[4].twinx()
            ax_4.plot(generation_crop, mutation_crop, color=C1orange,
                label='mutation rate')
            ax_5 = ax[5].twinx()
            ax_5.plot(generation_crop, mutation_crop, color=C1orange,
                label='mutation rate')
            ax[3].set_ylabel(r'median $\chi^2$')
            ax[4].set_ylabel(r'mean $\chi^2$ p')
            ax[5].set_ylabel(r'lowest $\chi^2$')
            ax_3.set_ylabel('mutation rate')
            ax_4.set_ylabel('mutation rate')
            ax_5.set_ylabel('mutation rate')
            ax[3].axvspan(mingen, maxgenid, alpha=0.2, color='yellow')
            ax[4].axvspan(mingen, maxgenid, alpha=0.2, color='yellow')
            ax[5].axvspan(mingen, maxgenid, alpha=0.2, color='yellow')
            ax[3].set_title(r'median $\chi^2$ per generation (last ' +
                str(int(last_X_generations)) + ' generations)')
            ax[4].set_title(r'mean $\chi^2$ per generation (last ' +
                str(int(last_X_generations)) + ' generations)')
            ax[5].set_title(r'lowest $\chi^2$ per generation (last ' +
                str(int(last_X_generations)) + ' generations)')
            ax[3].set_xlabel('Generation')
            ax[4].set_xlabel('Generation')
            ax[5].set_xlabel('Generation')
            ax[3].legend(loc='upper left')
            ax[4].legend(loc='upper left')
            ax[5].legend(loc='upper left')
            ax_3.legend(loc='upper right')
            ax_4.legend(loc='upper right')
            ax_5.legend(loc='upper right')

            # rect adds extra whitespace so the size of these plots
            # is more balanced with respect to the lineprofile plots etc.
            plt.tight_layout(rect=[0.0,0.5,.7,1.0])


        ax[0].set_ylabel(r'median $\chi^2$')
        ax[1].set_ylabel(r'mean $\chi^2$ p')
        ax[2].set_ylabel(r'lowest $\chi^2$')
        ax_0.set_ylabel('mutation rate')
        ax_1.set_ylabel('mutation rate')
        ax_2.set_ylabel('mutation rate')
        ax[0].set_title(r'median $\chi^2$ per generation')
        ax[1].set_title(r'mean $\chi^2$ per generation')
        ax[2].set_title(r'lowest $\chi^2$ per generation')
        ax[0].set_xlabel('Generation')
        ax[1].set_xlabel('Generation')
        ax[2].set_xlabel('Generation')
        ax[0].legend(loc='upper left')
        ax[1].legend(loc='upper left')
        ax[2].legend(loc='upper left')
        ax_0.legend(loc='upper right')
        ax_1.legend(loc='upper right')
        ax_2.legend(loc='upper right')

        if make_zoom:
            ax[0].axvspan(mingen, maxgenid, alpha=0.2, color='yellow')
            ax[1].axvspan(mingen, maxgenid, alpha=0.2, color='yellow')
            ax[2].axvspan(mingen, maxgenid, alpha=0.2, color='yellow')

        if not make_zoom:
            # rect adds extra whitespace so the size of these plots
            # is more balanced with respect to the lineprofile plots etc.
            plt.tight_layout(rect=[0.0,0.5,.7,1.0])
            plt.savefig(plotpath + name_chi2pergen_plot)
            plt.close()

        if make_zoom:
            plt.savefig(plotpath + name_chi2pergen_plot)
            plt.close()

''' ------------------------------------------'''
'''   Plot fitness as function of parameter   '''
''' ------------------------------------------'''
if make_fitnessdistribution_plot:

    if (make_fitnessdistribution_plot and make_fitnessdistribution_plot_P
        and make_fitnessdistribution_plot_fitness):
        plotvalues = ['invchi', 'P', 'fitness']
    elif (make_fitnessdistribution_plot_P and
        make_fitnessdistribution_plot_fitness):
        plotvalues = ['P', 'fitness']
    elif (make_fitnessdistribution_plot and
        make_fitnessdistribution_plot_P):
        plotvalues = ['invchi', 'P']
    elif (make_fitnessdistribution_plot and
        make_fitnessdistribution_plot_fitness):
        plotvalues = ['invchi', 'fitness']
    elif make_fitnessdistribution_plot:
        plotvalues = ['invchi']
    elif make_fitnessdistribution_plot_P:
        plotvalues = ['P']
    elif make_fitnessdistribution_plot_fitness:
        plotvalues = ['fitness']

    for pv in plotvalues:
        do_print("Making fitness vs parameter plots...", be_verbose)

        # Can be changed, all lines will be plotted, the number of
        # number of pages will be adapted accordingly.
        nrows_ppage = nrows_fitnessparamplot

        ncols_ppage = ncols_fitnessparamplot # Same holds for the columns.

        # This sets the 'absolute size' of the line profile
        # plots. The plots are always on A4 format, but the
        # size of the labels are relative to the size of the
        # plotting canvas. Therefore a higher number here
        # means smaller labels on the plots.
        plotscalefactor = 12.0

        plots_ppage = nrows_ppage * ncols_ppage
        npages = int(math.ceil(1.0*len(param_keys) / plots_ppage))
        do_print("Plotting fitness vs parameter on " + str(int(npages)) +
            " page(s).", be_verbose)

        gen_id = map(lambda q: float(q[:4]), x['run_id'])
        if max(gen_id) < 1:
            gen_id_scaled = 1.0
        else:
            gen_id_scaled = np.array(gen_id) / max(gen_id)
        scatter_colors = cm_rgba(gen_id_scaled)

        fitparam_jpg_names = []
        lp = -1 # paramater counter
        for apage in xrange(npages):
            fig, ax = plt.subplots(nrows_ppage, ncols_ppage,
                figsize=(plotscalefactor*1., plotscalefactor*1.41))
            for arow in xrange(nrows_ppage):
                for acol in xrange(ncols_ppage):
                    lp = lp + 1 # Next parameter is plotted
                    if lp < len(param_keys):
                        if pv == 'P':
                            plot_this = x['P'].values
                            the_ylabel = 'P-value'
                        elif pv == 'invchi':
                            plot_this = 1./x['rchi2'].values
                            the_ylabel = r'1/$\chi^2_{\rm red}$'
                        elif pv == 'fitness':
                            plot_this = x['fitness'].values
                            the_ylabel = 'fitness'
                        im1 = ax[arow,acol].scatter(x[param_keys[lp]].values,
                        plot_this,
                            s=10.0, c=scatter_colors)
                        ax[arow,acol].set_ylim(0, 1.1*np.nanmax(plot_this))
                        ax[arow,acol].set_xlim(params_dic[param_keys[lp]][0],
                        params_dic[param_keys[lp]][1])
                        ax[arow,acol].set_title(param_keys[lp])#, fontsize=14)
                        if best_models_cutoff_method == 'Mokiem':
                            ax[arow,acol].axvspan(params_error_Mokiem[param_keys[lp]][0],
                                params_error_Mokiem[param_keys[lp]][1],
                                alpha=0.6, color='orange', zorder=0)
                        else:
                            ax[arow,acol].axvspan(params_error_1sig[param_keys[lp]][0],
                                params_error_1sig[param_keys[lp]][1],
                                alpha=0.6, color='orange', zorder=0)
                            ax[arow,acol].axvspan(params_error_2sig[param_keys[lp]][0],
                                params_error[param_keys[lp]][1],
                                alpha=0.3, color='yellow', zorder=0)
                        ax[arow,acol].axvline(params_error_2sig[param_keys[lp]][2],
                        alpha=1.0, color='black',lw=1.5)#, zorder=0)
                        ax[arow,acol].set_ylabel(the_ylabel)

                    else:
                        ax[arow,acol].axis('off')

            plt.suptitle('Fitness vs. parameter (all lines)', fontsize=16)
            plt.tight_layout(rect=[0, 0.00, 1.0, 0.95])

            # Load the earlier produced colorbar/legend.
            # Sorry for this very desparate workaround
            imlegend = plt.imread(colorbar_jet_legend)
            newax = fig.add_axes([0.72, 0.885, 0.25, 0.2], anchor='C')
            newax.imshow(imlegend)
            newax.axis('off')

            fit_param_pagename = plotpath + 'overview_' + str(int(apage)) + '.jpg'
            fitparam_jpg_names.append(fit_param_pagename)
            plt.savefig(fit_param_pagename, dpi=400)
            plt.close()

        if pv == 'P':
            with open(plotpath + name_fitnessdistributionP_plot, "wb") as out_file:
                out_file.write(img2pdf.convert(fitparam_jpg_names))
        elif pv == 'invchi':
            with open(plotpath + name_fitnessdistribution_plot, "wb") as out_file:
                out_file.write(img2pdf.convert(fitparam_jpg_names))
        elif pv == 'fitness':
            with open(plotpath + name_fitnessdistributionfitness_plot, "wb") as out_file:
                out_file.write(img2pdf.convert(fitparam_jpg_names))

        for ajpg in fitparam_jpg_names:
            os.system("rm " + ajpg)

        # # Merge all line plot pages into one document.
        # if os.path.isfile(plotpath + name_fitnessdistribution_plot):
        #     os.system("rm " + plotpath + name_fitnessdistribution_plot)
        # merger = PdfFileMerger()
        # for filename in fitparam_pdf_names:
        #     merger.append(PdfFileReader(file(filename, 'rb')))
        # merger.write(plotpath + name_fitnessdistribution_plot)
        #
        #
        # # Remove the individual pages after the file has been merged.
        # for filename in fitparam_pdf_names:
        #     os.system("rm " + filename)

''' ------------------------------------------'''
'''               Plot line profiles          '''
''' ------------------------------------------'''

lineprofiles_incomplete = False

def vrshift(lamb0, vr):
    '''
    Apply a radial velocity (RV) shift to a wavelength (range).

    Input parameters:
        - lamb0: wavelength (float or np array) in angstrom
        - vr: radial velocity in km/s (float)
    Output:
        - Wavelength or wavelength array with RV shift applied.

    '''

    # Constants
    c = 2.99792458*10**10 #cm/s
    angstrom = 1.0*10**-8 # multiply to go from Angstrom to cm
    kms = 10**-5 # multiply to go from cm/s to km/s

    lamb0 = lamb0 * angstrom
    vr = vr / kms
    deltalamb = (vr/c) * lamb0
    lamb0 = lamb0 + deltalamb
    lamb0 = lamb0 / angstrom

    return lamb0

if make_lineprofiles_plot:

    do_print("Making lineprofile plots...", be_verbose)
    prof_jpg_names = []

    def renormalize_line(wave, flux, l1, l2, l3, l4):
        ''' Renormalises the line with a straight line, given the values in
            the ini file. (As in pyGA).

            Input:
            - wave = wavelength array
            - flux = flux array (already more or less normalised)
            - l1 = left wavelength point
            - l2 = left continuum offset at l1
            - l3 = right wavelength point
            - l4 = right continuum offset at l3

            Output:
            - renormflux = renormalised flux
        '''

        m = (l4 - l2) / (l3 - l1)
        b = l2 - m * l3 + 1
        y = m * wave + b
        renormflux = flux/y

        return renormflux

    ''' Selecting which models to plot and unpacking them '''

    # 1. Selection and identification of best fitting model
    # #FIXME see comments on 'best_models_cutoff_method' parameter at the top

    scaled_redchi2 = x['chi2']/min_redchi2_value
    x = x.assign(scaledRedChi2=scaled_redchi2)
    x['scaledRedChi2'] = scaled_redchi2
    if best_models_cutoff_method == 'P':
        x_best = x.loc[x['P'] > min_p_2sig]
        x_best_1sig = x.loc[x['P'] > min_p_1sig]
    elif best_models_cutoff_method == 'Mokiem':
        x_best = x.loc[x['fitness_norm'] > peak_val]
        x_best_1sig = x.loc[x['fitness_norm'] > peak_val]
    else:
        x_best = x.loc[x['scaledRedChi2'] < best_models_cutoff_method]
        x_best_1sig = x.loc[x['scaledRedChi2'] < best_models_cutoff_method]

    #FIXME should instead just select lowest chi2
    the_best_model = x.loc[x['scaledRedChi2'] == 1.0]
    do_print("Amount of models with lowest chi2 is: " +
        str(int(len(the_best_model))), be_verbose)
    the_best_model = the_best_model.iloc[0]
    the_best_model = the_best_model['run_id']
    do_print("The best model: " + the_best_model, be_verbose)
    bestmodeldir = savedmoddir +  the_best_model[:4] + '/' + the_best_model

    # 2. Saving directories of the to be plotted models.
    best_models = x_best['run_id'].values
    best_models_1sig = x_best_1sig['run_id'].values
    best_models_dirs = []
    best_models_tar = []
    for bm in best_models:
        best_models_dirs.append(savedmoddir +  bm[:4] + '/' + bm)
        # best_models_tar.append(datapath_output + bm[:4] + '/' + bm + '.tar.gz')

    # # 3. Unzipping the profile tars of the to be plotted models.
    # # Note that it doesn't explicitly check whether all the profiles are
    # # there, but only whether there is already a folder with the name of
    # # the model. If there is not, it assumes there is a unzipped tar.gz
    # # file and it starts unpacking (and removes the tar.gz afterwards)
    # do_print("Loading " + str(len(x_best)) + " best fitting models...", be_verbose)
    # for bmtar, bmdir in zip(best_models_tar, best_models_dirs):
    #     if not os.path.isdir(bmdir):
    #         mkdircommand = 'mkdir ' + bmdir
    #         untarcommand = 'tar -C ' + bmdir + ' -xzf ' + bmtar
    #         rmtarcommand = 'rm ' + bmtar
    #         os.system(mkdircommand)
    #         os.system(untarcommand)
    #         os.system(rmtarcommand)

    ''' Setup page layout '''
    # Can be changed, all lines will be plotted, the number of
    # number of pages will be adapted accordingly.
    nrows_ppage = nrows_lineprofileplot
    ncols_ppage = ncols_lineprofileplot # Same holds for the columns.

    plotscalefactor = 12.0
    plots_ppage = nrows_ppage * ncols_ppage
    npages = int(math.ceil(1.0*numlines / plots_ppage))
    do_print("Plotting the line profiles on " + str(int(npages)) + " pages.",
        be_verbose)

    ''' Reading the spectrum '''
    wave, flux, error = np.genfromtxt(thespectrumfile).T

    lc = -1 # line counter
    line_pdf_names = []

    # For each page we set up a grid of axes, each plotting the models and
    # data of one diagnostic line.
    for apage in xrange(npages):
        fig, ax = plt.subplots(nrows_ppage, ncols_ppage,
        figsize=(plotscalefactor*1., plotscalefactor*1.41))
        for arow in xrange(nrows_ppage):
            for acol in xrange(ncols_ppage):
                lc = lc + 1 # Next diagnostic line is plotted

                # if lc < 7:#numlines: # As long as not all the lines are plotted.
                if lc < numlines: # As long as not all the lines are plotted.
                    do_print("Plotting " + linenames[lc], be_verbose)

                    if not plot_individual_lines:
                        bmdir_base = best_models_dirs[0]
                        os.system('mkdir -p ' + bmdir_base)
                        name_bm = bmdir_base.split('/')[-1]
                        os.system('tar -xzf ' + bmdir_base + '.tar.gz -C '
                            + bmdir_base + '/.')
                        linefile_tmp = (bmdir_base + '/' + linenames[lc]
                            + '.prof.fin')
                        linewave_tmp, lineflux_tmp = np.genfromtxt(linefile_tmp).T

                        lineflux_min = np.copy(lineflux_tmp)
                        lineflux_max = np.copy(lineflux_tmp)

                    ''' Plot the models '''
                    for bmdir in best_models_dirs:
                        try:
                            # if not os.path.isdir(bmdir):
                            #     os.system('mkdir -p ' + bmdir)
                            #     name_bm = bmdir.split('/')[-1]
                            #     os.system('tar -xzf ' + bmdir + '.tar.gz -C ' + bmdir + '/.')
                            # Uncomment the next 4 lines (and comment the above 4)
                            # if you want to untar *all* tars, in case you have created
                            # empty folders by accident.
                            if lc == 0:
                                os.system('mkdir -p ' + bmdir)
                                name_bm = bmdir.split('/')[-1]
                                os.system('tar -xzf ' + bmdir + '.tar.gz -C ' + bmdir + '/.')
                            linefile_tmp = bmdir + '/' + linenames[lc] + '.prof.fin'
                            linewave_tmp, lineflux_tmp = np.genfromtxt(linefile_tmp).T
                            linewave_tmp = vrshift(linewave_tmp,
                                line_radialvelocity[lc])

                            lineflux_min = np.min(np.array([lineflux_min,
                                lineflux_tmp]), axis=0)
                            lineflux_max = np.max(np.array([lineflux_max,
                                lineflux_tmp]), axis=0)

                            if bmdir == bestmodeldir:
                                # The best fitting model in dargreen, note the
                                # high zorder so that it is on top.
                                # +20 in zorder makes sure that other things
                                # line axhlines are taken into account
                                ax[arow, acol].plot(linewave_tmp, lineflux_tmp,
                                    color='#2a7f2a', # darkgreen!
                                    zorder=len(best_models_dirs)+20, lw=3.0)
                                np.savetxt(plotlineprofdir + linenames[lc]
                                    + '.best',
                                    np.array([linewave_tmp, lineflux_tmp]).T)
                            elif plot_individual_lines:
                                # The x best models in lightgreen
                                ax[arow, acol].plot(linewave_tmp, lineflux_tmp,
                                    color='#8cd98c') # lightgreen

                        except:
                            do_print('Could not find lineprofiles of ' +
                                bmdir.split('/')[-1], be_verbose)
                            lineprofiles_incomplete = True

                    ax[arow, acol].fill_between(linewave_tmp, lineflux_min,
                        lineflux_max, color='#8cd98c')
                    np.savetxt(plotlineprofdir + linenames[lc] + '.minmax',
                        np.array([linewave_tmp, lineflux_min, lineflux_max]).T)

                    ''' Plot the data '''
                    # The spectra are cropped because otherwise plotting
                    # will take up a lot of memory. (The set_xlim might therefore
                    # be a bit unnecessary)
                    wave_tmp, flux_tmp = parallelcrop(wave, flux,
                        linestarts[lc], linestops[lc])
                    wave_tmp, error_tmp = parallelcrop(wave, error,
                        linestarts[lc], linestops[lc])
                    ax[arow, acol].axhline(1.0, color='grey', lw=1.0)
                    if linenames[lc] == 'HEI4026':
                        ax[arow, acol].set_title('HE4026')
                    else:
                        ax[arow, acol].set_title(linenames[lc])

                    ax[arow, acol].set_xlim(linestarts[lc], linestops[lc])
                    flux_renorm_tmp = renormalize_line(wave_tmp, flux_tmp,
                        line_norm_starts[lc], line_norm_leftval[lc],
                        line_norm_stops[lc], line_norm_rightval[lc])

                    if len(wave_tmp) > 150:
                        box = 2
                        if len(wave_tmp) > 200:
                            box = 3
                        do_print(linenames[lc] +
                            ' plotted with less points, box = ' + str(int(box)),
                            be_verbose)
                        ax[arow, acol].errorbar(wave_tmp[box:-box:box],
                            flux_renorm_tmp[box:-box:box],
                            yerr=error_tmp[box:-box:box], marker='o',
                            markersize=0.1, linestyle='None', color='black',
                            zorder=10e10, alpha=0.6)
                    else:
                        ax[arow, acol].errorbar(wave_tmp, flux_renorm_tmp,
                            yerr=error_tmp, marker='o', markersize=0.1,
                            linestyle='None', color='black', zorder=10e10,
                            alpha=0.6)


                    ''' Plot extra spectrum if wanted '''
                    if include_extra_spectrum:
                        try:
                            cmfout3 = parallelcrop(cmfgenwave1, cmfgenflux1,
                                linestarts[lc], linestops[lc])
                            cmfgenwave_tmp, cmfgenflux_tmp = cmfout3
                            ax[arow, acol].plot(cmfgenwave_tmp, cmfgenflux_tmp,
                                color='blue', lw=2.0,
                                zorder=len(best_models_dirs)+20)
                        except:
                            if extra_spectrum_path1 != "":
                                do_print(">> Something might have gone wrong " +
                                "with plotting the extra spectra ", True)
                        try:
                            cmfout4 = parallelcrop(cmfgenwave2, cmfgenflux2,
                                linestarts[lc], linestops[lc])
                            cmfgenwave_tmp, cmfgenflux_tmp = cmfout4
                            ax[arow, acol].plot(cmfgenwave_tmp, cmfgenflux_tmp,
                                color='orange', lw=2.0,
                                zorder=len(best_models_dirs)+21)
                        except:
                            if extra_spectrum_path2 != "":
                                do_print(">> Something might have gone wrong " +
                                "with plotting the extra spectra ", True)
                else:
                    # We loop through all axes (rows, cols) that are set up,
                    # but if there are not enough lines to fill a page axes
                    # are removed so that the space is truly empty.
                    ax[arow, acol].axis('off')

        plt.tight_layout()

        prof_param_pagename = plotpath + 'profiles_' + str(int(apage)) + '.jpg'
        prof_jpg_names.append(prof_param_pagename)
        plt.savefig(prof_param_pagename, dpi=400)
        plt.close()
        # plt.show()
        do_print("Lineprofiles: saved page " + str(int(apage+1)) +
            " out of " + str(int(npages)) + " pages.", be_verbose)

    with open(plotpath + name_lineprofiles_plot, "wb") as out_file:
        out_file.write(img2pdf.convert(prof_jpg_names))

    for ajpg in prof_jpg_names:
        os.system("rm " + ajpg)

''' ------------------------------------------'''
'''   Videos of parameter space exploration   '''
''' ------------------------------------------'''
if make_paramspace_avi:

    for param_pair in param_pair_list_avi:

        xparam = param_pair[0]
        yparam = param_pair[1]
        do_print("Making video of paramater space exploration of " +
            xparam + ", " + yparam, be_verbose)

        imnames = xparam + '_' + yparam + '_exploration'

        for the_genid in unique_genid:

            x_specific_gen = x.loc[x['gen_id'] == the_genid]
            p1_tmp = x_specific_gen[xparam].values
            p2_tmp = x_specific_gen[yparam].values
            chi_tmp = x_specific_gen['chi2'].values

            chi_tmp_colorscale = chi_tmp/(np.median(chi_tmp)*10)
            color_array = cm_rgba(chi_tmp_colorscale)

            themutationrate = mutation[int(the_genid)]
            themutationrate = '{0:.4f}'.format(themutationrate)
            fig, ax = plt.subplots()
            ax.set_title("Gen = " + the_genid + ", Mutation rate = " +
                themutationrate)
            ax.scatter(p1_tmp, p2_tmp, c=color_array)
            ax.set_xlim(*params_dic[xparam])
            ax.set_ylim(*params_dic[yparam])
            ax.set_xlabel(xparam)
            ax.set_ylabel(yparam)
            plt.savefig(plotpath + imnames + str(the_genid) + '.png')
            plt.close()

        image_folder = plotpath
        video_name = plotpath + '00_' + imnames + '.avi'

        images = [img for img in os.listdir(image_folder) if (img.startswith(imnames) and img.endswith('.png'))]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 10, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
            os.system("rm " + os.path.join(image_folder, image))

        cv2.destroyAllWindows()
        video.release()



''' ------------------------------------------'''
'''   Plots of correlation between parameters '''
''' ------------------------------------------'''

if make_correlation_plot:
    do_print("Making plots of correlation between parameters", be_verbose)

    paramlist = params_dic.keys()

    nbins1 = 10
    nbins2 = nbins1

    colorbar_chi2 = 'viridis'
    colorbar_other = 'viridis_r'

    plotscalefactor = 12.0

    correlation_plot_list = ['chi2']
    if make_correlation_per_line_plot:

        for diagnostic in linenames:
            correlation_plot_list.append(diagnostic)
        all_jpg_names = []

    make_legend = True
    for which_plot in correlation_plot_list:

        fig, ax = plt.subplots(len(paramlist)-1,len(paramlist)-1,
            figsize=(plotscalefactor*1.0, plotscalefactor*1.0))
        if which_plot == 'chi2':
            the_colormap = colorbar_chi2
        else:
            the_colormap = colorbar_other

        if which_plot == 'chi2':
            do_print("Making correlation plot for all lines together ",
                be_verbose)
        else:
            do_print("Making correlation plot for " + which_plot, be_verbose)

        param_pairs_correlation = []
        for ii in xrange(len(paramlist)):
            for jj in xrange(len(paramlist)):
                xparam = paramlist[ii]
                yparam = paramlist[jj]
                if (xparam != yparam) and ([yparam, xparam] not in param_pairs_correlation):

                    param_pairs_correlation.append([xparam, yparam])

                    p1_tmp = x[xparam].values
                    p2_tmp = x[yparam].values
                    chi_tmp = x[which_plot].values

                    p1min, p1max = params_dic[xparam]
                    p2min, p2max = params_dic[yparam]

                    p1_space = np.linspace(p1min, p1max, nbins1)
                    p2_space = np.linspace(p2min, p2max, nbins2)
                    p1_lowbound = p1_space[:-1]
                    p1_upbound = p1_space[1:]
                    p2_lowbound = p2_space[:-1]
                    p2_upbound = p2_space[1:]

                    chi2_matrix = []

                    for nb in xrange(nbins1-1):
                        matrix_axis = []
                        for nb in xrange(nbins2-1):
                            matrix_axis.append([])
                        chi2_matrix.append(matrix_axis)

                    for ap1, ap2, achi in zip(p1_tmp, p2_tmp, chi_tmp):
                        i1 = 0
                        for p1low, p1up in zip(p1_lowbound, p1_upbound):
                            i2 = 0
                            for p2low, p2up in zip(p2_lowbound, p2_upbound):
                                if ((p1low < ap1) and (p1up > ap1) and
                                    (p2low < ap2) and (p2up > ap2)):
                                    if achi != 999999999:
                                        chi2_matrix[i1][i2].append(float(achi))
                                i2 = i2 + 1
                            i1 = i1 + 1

                    for i1 in xrange(nbins1-1):
                        for i2 in xrange(nbins2-1):
                            if len(chi2_matrix[i1][i2]) > 0:
                                median_tmp = np.median(chi2_matrix[i1][i2])
                            else:
                                median_tmp = float('nan')
                            chi2_matrix[i1][i2] = median_tmp

                    if which_plot == 'chi2':
                        chi2_matrix = np.log10(chi2_matrix)
                    else:
                        chi2_matrix = np.array(chi2_matrix)

                    # so that x and y correspond x and y definition.
                    chi2_matrix = chi2_matrix.T
                    ax[jj-1][ii].imshow(chi2_matrix, cmap=the_colormap)
                    ax[jj-1][ii].set_yticks([])
                    ax[jj-1][ii].set_xticks([])
                    if jj == len(paramlist)-1:
                        ax[jj-1][ii].set_xlabel(xparam)
                    if ii == 0:
                        ax[jj-1][ii].set_ylabel(yparam)

                else:
                    if jj != 0 and ii < len(paramlist)-1:
                        ax[jj-1][ii].axis('off')
                        if jj == 1 and ii == len(paramlist)-2:
                            if which_plot == 'chi2':
                                ax[jj-1][ii].text(0.5, 0.8,
                                    'Correlation\n(all lines)',
                                    ha='center', va='center',
                                    transform=ax[jj-1][ii].transAxes)
                            else:
                                ax[jj-1][ii].text(0.5, 0.8,
                                    'Correlation\n(' + which_plot + ')',
                                    ha='center', va='center',
                                    transform=ax[jj-1][ii].transAxes)
                        if jj == 1 and ii == len(paramlist)-2:
                            if which_plot == 'chi2':
                                if the_colormap.endswith('_r'):
                                    the_colormap = the_colormap[-2]
                                else:
                                    the_colormap = the_colormap + '_r'
                            else:
                                doreverse = False
                            img = sns.heatmap(np.array([[0,1]]),
                                ax=ax[jj-1][ii],
                                mask=np.array([[True,True]]),
                            cbar_kws={"orientation": "horizontal"},
                                cmap=the_colormap)
                            cbar = img.collections[0].colorbar
                            cbar.set_ticks([.2, .8])
                            cbar.set_ticklabels(['<-- worse fit',
                                '\nbetter fit -->'])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.02, wspace=0.05)
        if which_plot == 'chi2':
            # FIXME I would like to have this figure too, on A4 page format...
            # However editing canvas size or rect in tight_layout spreads the plots over
            # the full A4, I want them an the top of middle, but in a square, not spread out :(
            plt.savefig(plotpath + name_correlation_plot)
            plt.close()
        else:
            the_jpg_name = plotpath + 'correlation_' + which_plot + '.jpg'
            all_jpg_names.append(the_jpg_name)
            plt.savefig(the_jpg_name)
            plt.close()

    if make_correlation_per_line_plot:
        with open(plotpath + name_correlation_per_line_plot, "wb") as out_file:
            out_file.write(img2pdf.convert(all_jpg_names))

        for ajpg in all_jpg_names:
            os.system("rm " + ajpg)


''' ------------------------------------------'''
'''  Plot fitn per line as function of param  '''
''' ------------------------------------------'''

if make_fitnessdistribution_per_line_plot:

    do_print("Making fitness per parameter plots per diagnostic line...",
        be_verbose)

    fitparam_jpg_names = []

    for diagnostic in linenames:

        nrows_ppage = nrows_fitnessparamplot
        ncols_ppage = ncols_fitnessparamplot
        plotscalefactor = 12.0
        plots_ppage = nrows_ppage * ncols_ppage
        npages = int(math.ceil(1.0*len(param_keys) / plots_ppage))
        do_print("Plotting fitness vs parameter for " + diagnostic +
            " on " + str(int(npages)) + " page(s).", be_verbose)

        gen_id = map(lambda q: float(q[:4]), x['run_id'])
        if max(gen_id) < 1:
            gen_id_scaled = 1.0
        else:
            gen_id_scaled = np.array(gen_id) / max(gen_id)
        scatter_colors = cm_rgba(gen_id_scaled)

        lp = -1 # paramater counter
        for apage in xrange(npages):
            fig, ax = plt.subplots(nrows_ppage, ncols_ppage,
                figsize=(plotscalefactor*1., plotscalefactor*1.41))
            for arow in xrange(nrows_ppage):
                for acol in xrange(ncols_ppage):
                    lp = lp + 1 # Next parameter is plotted
                    if lp < len(param_keys):
                        line_fitness = x[diagnostic].values
                        ax[arow,acol].scatter(x[param_keys[lp]].values, line_fitness,
                            s=10.0, c=scatter_colors)
                        ax[arow,acol].set_xlim(params_dic[param_keys[lp]][0],
                            params_dic[param_keys[lp]][1])
                        ax[arow,acol].set_title(param_keys[lp])
                        ax[arow,acol].axvspan(params_error_1sig[param_keys[lp]][0],
                            params_error_1sig[param_keys[lp]][1], alpha=0.6,
                                color='orange', zorder=0)
                        ax[arow,acol].axvspan(params_error_2sig[param_keys[lp]][0],
                            params_error[param_keys[lp]][1], alpha=0.3,
                                color='yellow', zorder=0)
                        ax[arow,acol].axvline(params_error_2sig[param_keys[lp]][2],
                            alpha=1.0, color='black',lw=1.5)

                    else:
                        ax[arow,acol].axis('off')


            plt.suptitle('Fitness vs. parameter (' + diagnostic + ')',
                fontsize=16)
            plt.tight_layout(rect=[0, 0.00, 1.0, 0.95])

            # Load the earlier produced colorbar/legend.
            # Sorry for this very desparate workaround
            imlegend = plt.imread(colorbar_jet_legend)
            newax = fig.add_axes([0.72, 0.885, 0.25, 0.2], anchor='C')
            newax.imshow(imlegend)
            newax.axis('off')

            fit_param_pagename_jpg = (plotpath + 'overview_' + diagnostic
                + '_' + str(int(apage)) + '.jpg')
            fitparam_jpg_names.append(fit_param_pagename_jpg)
            plt.savefig(fit_param_pagename_jpg, dpi=200)
            plt.close()

    with open(plotpath + name_fitness_per_line_plot, "wb") as out_file:
        out_file.write(img2pdf.convert(fitparam_jpg_names))

    for ajpg in fitparam_jpg_names:
        os.system("rm " + ajpg)



''' --------------- OUTPUT -------------------'''
'''   Merge all available pdfs into rerport   '''
''' ------------------------------------------'''

os.system("rm " + colorbar_jet_legend)

''' Short report '''
# Does not contain the correlation and fitness plots per line.
# In the end, merge all line profile plot pages into one document.
osld = os.listdir(plotpath)
report_pdfs = []
for afile in osld:
    if afile.endswith('_GAreport.pdf') and 'short' in afile:
        report_pdfs.append(plotpath + afile)

report_pdfs.sort()

if adapt_ngen:
    do_print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', be_verbose)
    do_print('!!                                               !!', be_verbose)
    do_print('!!                                               !!', be_verbose)
    do_print('!!                     WARNING                   !!', be_verbose)
    do_print('!!      ONLY PLOTTED SUBSET OF GENERATIONS!!!    !!', True)
    do_print('!!                                               !!', be_verbose)
    do_print('!!                                               !!', be_verbose)
    do_print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', be_verbose)

if os.path.isfile(name_shortreport_pdf):
    os.system("rm " + name_shortreport_pdf)
merger = PdfFileMerger()
for filename in report_pdfs:
    merger.append(PdfFileReader(file(filename, 'rb')))
merger.write(name_shortreport_pdf)
#do_print("Short report saved to:", be_verbose)
#do_print(name_shortreport_pdf, be_verbose)

''' Full report '''
# Containing all plots
# In the end, merge all line profile plot pages into one document.
osld = os.listdir(plotpath)
report_pdfs = []
for afile in osld:
    if afile.endswith('_GAreport.pdf'):
        report_pdfs.append(plotpath + afile)

report_pdfs.sort()

if lineprofiles_incomplete:
    do_print('WARNING - plotted lineprofes lineprofiles_incomplete', True)
if os.path.isfile(name_fullreport_pdf):
    os.system("rm " + name_fullreport_pdf)
merger = PdfFileMerger()
for filename in report_pdfs:
    merger.append(PdfFileReader(file(filename, 'rb')))
merger.write(name_fullreport_pdf)
do_print("Full report saved to:", True)
do_print(name_fullreport_pdf, True)
