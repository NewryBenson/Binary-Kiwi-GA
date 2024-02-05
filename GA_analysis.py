# Script for visualising the run output of Kiwi-GA
# Created by Sarah Brands @ 29 July 2022
# Edited by Frank Backs & Sarah Brands

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
import argparse
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import func_GA_analysis as fga
import fastwind_wrapper as fw
import paths as ppp
from scipy import stats

###############################################################################
#  Parse arguments (e.g. runname)
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('runname', help='Specify a run name')
parser.add_argument('-full', help='Create all plots possible',
    action='store_true', default=False)
parser.add_argument('-prof', help='Plot line profiles',
    action='store_true', default=False)
parser.add_argument('-fast', help='Only make fitness plot and title page',
    action='store_true', default=False)
parser.add_argument('-open', help='After making the report, open it.',
    action='store_true', default=False)
parser.add_argument('-param', help='Makes fitness per line plots grouped by' +
    'parameter (not part of -full)',
    action='store_true', default=False)
parser.add_argument('-fitness', help='Add fitness plot with fitness ' +
    '(in addition to 1/chi2)',
    action='store_true', default=False)
parser.add_argument('-radius', help='Correct radius (compute FW model)',
    action='store_true', default=False)
parser.add_argument('-em', help='Add an extra FW model to the line profile'  +
    ' plots. Supply here the path to the spectrum, which should be an asci ' +
    'file containing two columns (wavelength (angstrom) and normalised flux).',
    default='')
parser.add_argument('-efm', help='Add an extra FW model to the line  ' +
    'profile plots. Supply here the path to the directory. The directory ' +
    'should contain (broadened) line profiles with names as in the GA run ' +
    '+ ending on .prof, e.g. "HALPHA.prof". The files schould be asci ' +
    'format and have two columns: wavelength (angstrom) and normalised flux. ',
    default='')
args = parser.parse_args()

runname = args.runname
print("Generating report for << " + runname + " >>")

###############################################################################
#  Definition of paths & files
###############################################################################

datapath = ppp.datapath_analysis
outpath = ppp.outpath_analysis + runname + '/'
fastwind_local = ppp.fastwind_local

pdfname = outpath + runname + '.pdf'
if args.full:
    pdfname = outpath + runname + '_full.pdf'
if args.fast:
    pdfname = outpath + runname + '_fast.pdf'
if args.param:
    pdfname = outpath + runname + '_param.pdf'

datapath = datapath + runname + '/'

plotlineprofdir = outpath + 'lineprofs/'
os.system("mkdir -p " + outpath)
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
savebestfile = outpath + runname + '_bestvals.txt'
save_dffile = outpath + 'df_' + runname + '.csv'
savebestindatfile = outpath + 'INDAT.DAT'

extra_fwmod = args.efm
if not extra_fwmod.endswith('/'):
    extra_fwmod = extra_fwmod + '/'
extra_mod = args.em

###############################################################################
#  Read GA output files
###############################################################################

# Read chi2.txt into pandas dataframe
chi2data = np.genfromtxt(thechi2file)
with open(thechi2file, "r") as file:
    head_chi2file = file.readline()
if not head_chi2file.startswith('#'):
    print('Header not found in ' + thechi2file)
colnames = head_chi2file[1:].split()
df = pd.DataFrame(chi2data, columns=colnames)

# Finish dataframe with proper units for run_id's and generation numbers
runids = np.genfromtxt(thechi2file, dtype='str').T[0]
df['run_id'] = runids
df['gen'] = df['gen'].astype(int)
maxgen = np.max(df['gen']) + 1
df_orig = df.copy()

# Read spectrum
obswave, obsflux, obserr = np.genfromtxt(thespectrumfile).T
spectdct = {'wave':obswave, 'flux':obsflux, 'err':obserr}

# Read linelist
linenames = np.genfromtxt(thelinefile, dtype='str').T[0]
lineboundleft = np.genfromtxt(thelinefile).T[2]
lineboundright = np.genfromtxt(thelinefile).T[3]
linedct = {'name':linenames, 'left':lineboundleft, 'right':lineboundright}

# Read parameter_space
pspacedata = fw.read_paramspace(theparamfile)
param_names, param_space, fix_names, fix_vals = pspacedata
nfree = len(param_names)

# Get dof: count number of points in spectrum and subtract nfree.
npspec = 0
for lb, rb, in zip(lineboundleft, lineboundright):
    obswave_cnt = obswave[(obswave > lb) & (obswave < rb)]
    npspec = npspec + len(obswave_cnt)
dof_tot = npspec - nfree

# Read number of individuals
nind = int(np.genfromtxt(thecontrolfile, dtype='str')[0,1])

# Do radius correction. This is always done if a FW model is present
df, best_indat = fga.radius_correction(df, fastwind_local, runname,
    thecontrolfile, theradiusfile, datapath, outpath, comp_fw=args.radius)

#  Compute derived parameters
df, deriv_pars = fga.more_parameters(df, param_names, fix_names, fix_vals)

###############################################################################
#  Calculate P-value and best fit parameters
###############################################################################

# Compute uncertainties
df, best_uncertainty, n1sig, n2sig = fga.get_uncertainties(df, dof_tot,
    npspec, param_names, param_space, deriv_pars, incl_deriv=True)
np.savetxt(outpath + 'n_1sig_2sig.txt', np.array([n1sig, n2sig]))

# Unpack all computed values
best_model_name, bestfamily_name, params_error_1sig, \
    params_error_2sig, deriv_params_error_1sig, deriv_params_error_2sig, \
    which_statistic = best_uncertainty

###############################################################################
#  Create plots
###############################################################################

with PdfPages(pdfname) as the_pdf:
    #  Create a title page with best fit parameters
    the_pdf = fga.titlepage(df, runname, params_error_1sig, params_error_2sig,
        the_pdf, param_names, maxgen, nind, linedct, 1,
        deriv_params_error_1sig, deriv_params_error_2sig, deriv_pars)

    # Create overview fitness plot (1/rchi2)
    the_pdf = fga.fitnessplot(df, 'invrchi2', params_error_1sig,
        params_error_2sig, the_pdf, param_names, param_space,maxgen)
    the_pdf = fga.fitnessplot(df, 'invrchi2', deriv_params_error_1sig,
        deriv_params_error_2sig, the_pdf, deriv_pars,[],maxgen)
    if args.fitness:
        the_pdf = fga.fitnessplot(df, 'fitness', params_error_1sig,
            params_error_2sig, the_pdf, param_names, param_space,maxgen)

    if args.prof:
        #  Create line profile plots
        the_pdf = fga.lineprofiles(df, spectdct, linedct, savedmoddir,
            best_model_name, bestfamily_name, the_pdf,
            plotlineprofdir, extra_fwmod, extra_mod)

    if not args.fast:
        #  Create correlation plots
        corr_vars = ['teff', 'logg', 'yhe', 'vrot', 'micro']
        params_corr = [value for value in param_names if value in corr_vars]
        the_pdf = fga.correlationplot(the_pdf, df,params_corr)

        corr_vars = ['mdot', 'beta', 'fclump', 'fic', 'fvel', 'vcl']
        params_corr = [value for value in param_names if value in corr_vars]
        the_pdf = fga.correlationplot(the_pdf, df,params_corr)

        # Convergence plot
        the_pdf = fga.convergence(the_pdf, df_orig, dof_tot, npspec,
            param_names,param_space, deriv_pars, maxgen, runname,
            fastwind_local, thecontrolfile, theradiusfile, datapath)

        # Fastwind performance plot
        the_pdf = fga.fw_performance(the_pdf, df, thecontrolfile)

        # P-value plot
        if which_statistic in ('Pval_ncchi2', 'Pval_chi2'):
            #  Create overview fitness plot (P-value)
            the_pdf = fga.fitnessplot(df, 'P-value', params_error_1sig,
                params_error_2sig, the_pdf, param_names, param_space,maxgen)

    if args.full:

        #  Create fitness plots per line
        for yval in linenames:
            the_pdf = fga.fitnessplot(df, yval, params_error_1sig,
                params_error_2sig, the_pdf, param_names, param_space,maxgen)

    if args.param or args.full:
        for i, param in enumerate(param_names):
            the_pdf = fga.fitnessplot_per_parameter(df, param, params_error_1sig,
                params_error_2sig, the_pdf, linenames, param_space[i], maxgen)

fga.save_bestvals(param_names, deriv_pars, params_error_1sig, params_error_2sig,
    deriv_params_error_1sig, deriv_params_error_2sig, savebestfile)

os.system('cp ' + best_indat + ' ' + savebestindatfile)
df.to_csv(save_dffile)
print('Report saved to ' + pdfname)

if args.open:
    os.system('open ' + pdfname)
