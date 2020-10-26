# Sarah Brands s.a.brands@uva.nl 25-02-2020
# Script with some basic checks on pyEA input.
#
# Should be executed before starting a run:
#  - Checks several aspects of the run setup
#      * Prints report where errors are pointed out
#      * Plots spectrum, line boundaries and parameter space
#        (saved to pdf in input directory of the run)
#  - Creates a job file
#
# Usage: 
#   python pre_run_check.py <runname>
# Or if you want to both print and save the output of the script:
#   python pre_run_check.py <runname> | tee -a "input/<runname>/prerun.log"
# Saves log into run input directory (also the pdf report goes here).

import os
import sys 
import math
import numpy as np
from matplotlib import pyplot as plt
from PyPDF2 import PdfFileMerger

import fastwind_wrapper as fw
import population as pop

jobscriptfile = 'run_pyEA.job'

run_name = sys.argv[1]
if run_name.endswith('/'):
    run_name = run_name[:-1]
if len(sys.argv) > 2:
    if sys.argv[2] == 'restart':
        do_restart='yes'
    else:
        do_restart='no'
else:
    do_restart='no'

inputmain = 'input/'
inputdir = inputmain + run_name + '/'
controlname = inputdir + 'control.txt'
paramspacename = inputdir + 'parameter_space.txt'
radinfoname = inputdir + 'radius_info.txt'
linesetname = inputdir + 'line_list.txt'
defaultsname = inputdir + 'defaults_fastwind.txt'
spectrumname = inputdir + 'spectrum.norm'

param_fig = inputdir + 'parameter_input.pdf'
spectrum_fig = inputdir + 'spectrum_input.pdf'

pdfs = [param_fig, spectrum_fig]
merged_report = inputdir + "pre_run_report.pdf"

report = inputdir + "pre_run_report.txt"

if not os.path.isdir(inputdir):
    print("Could not find direcotry " + inputdir)
    print("    Typos happen to the best")
    sys.exit()

def print_write(the_string, the_file):
    with open(the_file, "a") as text_file:
        text_file.write(the_string + '\n')

def printsection(title):
    print('\n____' + title + '____')

def printsumline(entry, comment, nd):
    printstring = '##     >  ' + entry + ': ' + comment
    spaces = (nd - len(printstring)-2)*' ' + '##'
    printstring = printstring + spaces
    print(printstring)

###################################
#        Check parameters         #
###################################

# Start of script, Set up dictionary
nd = 79
intro = '  Starting pre-run check for: ' + run_name + '  '
halfpt = int((nd - len(intro)-2)/2)
otherhalfpt = int((nd - len(intro) - halfpt))
print('\n' + halfpt*'#'  + intro + otherhalfpt*'#')
checkdict = {}

# Read in files
ctrldct = fw.read_control_pars(controlname)
lineinfo = fw.read_data(linesetname, spectrumname)
the_paramspace = fw.read_paramspace(paramspacename)
param_names, param_space, fixed_names, fixed_pars = the_paramspace
defnames, defvals = fw.get_defvals(defaultsname, param_names, fixed_names)

# Test if inicalc as specified in control.txt is present
test_inidir = ctrldct['inicalcdir']
if test_inidir[-1] != '/':
    print("ERRROR! inicalcdir name as specified in control.txt" + 
        " does not end with '/'. Aborting pre_run_check.")
    sys.exit()
test_inidir = test_inidir[:-1]
if test_inidir not in next(os.walk('.'))[1]:
    print("ERROR! fastwind directory '" + ctrldct['inicalcdir'] +
        "' not found. Aborting pre_run_check.")
    sys.exit()

# Check n_ind
n_cpu_core = 24.0
if (ctrldct["nind"]+1)/n_cpu_core % 1.0  > 0.0:
    print("ERROR! Not using all " + str(n_cpu_core) + " cpu's per" 
        " core")
    sys.exit()

print("\nnind = " + str(ctrldct["nind"]))
print("ngen = " + str(ctrldct["ngen"]))

# Check mutation rate parameters
printsection('Mutation rate')
checkdict["Mutation"] = True
mut_adjust_type = ctrldct["mut_adjust_type"]
if not mut_adjust_type in ('contstant', 'charbonneau', 'autocharb'):
    print('ERROR: mut_adjust_type unknown: ' + mut_adjust_type)
    checkdict["Mutation"] = False

fit_cutoff_min_charb = ctrldct["fit_cutoff_min_charb"]
fit_cutoff_max_charb = ctrldct["fit_cutoff_max_charb"]
w_gauss_na = ctrldct['w_gauss_na']
w_gauss_br = ctrldct['w_gauss_br']
b_gauss_na = ctrldct['b_gauss_na']
b_gauss_br = ctrldct['b_gauss_br']
type_br = ctrldct["broad_type"]
type_na = ctrldct["narrow_type"]

print("Narrow type: " + type_na)
print("Broad type : " + type_br)
if w_gauss_br < w_gauss_na and type_br == type_na:
    print('ERROR: narrow gaussian is broader than wide gaussian')
    checkdict["Mutation"] = False
    print('   - Narrow gauss width: ' + str(w_gauss_na))
    print('   - Broad gauss width : ' + str(w_gauss_br))
if w_gauss_br > 0.50 and type_br == 'frac':
    print('WARNING: high value for w_gauss_br')
    checkdict["Mutation"] = False
    print('   - Broad gauss width : ' + str(w_gauss_br))
if w_gauss_na > 2.00 and type_na == 'step':
    print('WARNING: high value for w_gauss_na')
    checkdict["Mutation"] = False
    print('   - Narrow gauss width: ' + str(w_gauss_na))
if w_gauss_na > 0.05 and type_na == 'frac':
    print('WARNING: high value for w_gauss_na')
    checkdict["Mutation"] = False
    print('   - Narrow gauss width: ' + str(w_gauss_na))
if b_gauss_br > 0.15:
    print('WARNING: high value for b_gauss_br')
    checkdict["Mutation"] = False
    print('   - Braod gauss base  : ' + str(b_gauss_br))
if b_gauss_na != 0.0:
    print('WARNING: b_gauss_na is nonzero.')
    print('  This means it will cause mutations far away from current value.')
    print('   - Narrow gauss base : ' + str(b_gauss_br))
    checkdict["Mutation"] = False
if checkdict["Mutation"] == True:
    print('No suspicious things found in mutation parameters.')

mut_adj_type = ctrldct["mut_adjust_type"]
if mut_adj_type == 'autocharb':
    printsection('Auto adaption charbonneau limits')
    ac_fit_a = ctrldct["ac_fit_a"]
    ac_fit_b = ctrldct["ac_fit_b"]
    ac_max_factor = ctrldct["ac_max_factor"]
    ac_maxgen = ctrldct["ac_maxgen"]
    ac_maxgen_min = ctrldct["ac_maxgen_min"]
    ac_maxgen_max = ctrldct["ac_maxgen_max"]
    ac_lowerlim = ctrldct["ac_lowerlim"]
    ac_upperlim = ctrldct["ac_upperlim"]
    print('ac_fit_a        ' + ac_fit_a)
    print('ac_fit_b        ' + ac_fit_b)
    print('ac_max_factor   ' + ac_max_factor)
    print('ac_maxgen       ' + ac_maxgen)
    print('ac_maxgen_min   ' + ac_maxgen_min)
    print('ac_maxgen_max   ' + ac_maxgen_max)
    print('ac_lowerlim     ' + ac_lowerlim)
    print('ac_upperlim     ' + ac_upperlim)

# Checks on parameter space
printsection('Parameter space')
print('Number of free parameters: ' + str(len(param_names)))
for pn in param_names:
    print('   - ' + pn)
checkdict["Parameter space"] = True
print('\nFixed values read in from parameter space file: ')
for fp, fv in zip(fixed_names, fixed_pars):
    print('   - ' + fp + ': ' + str(fv))

with open(paramspacename) as f:
    plines = f.readlines()

if 'vturb' in param_names or 'vturb' in fixed_names or 'vturb' in defnames:
    checkdict["Parameter space"] = False
    print("ERROR: 'vturb' is not a parameter, use instead: \n  'micro' (for " +
        "micro turbulence), 'macro' (for macro turbulence) or \n  'windturb'" +
        " (for wind turbulence)")


printsection("X-rays")

all_names = np.concatenate((np.concatenate((param_names,fixed_names)),defnames))
nonfree_names = np.concatenate((fixed_names,defnames))
nonfree_vals = np.concatenate((fixed_pars,defvals))
allp_dict = dict(zip(nonfree_names, nonfree_vals))

if not 'fx' in all_names:
    checkdict["Parameter space"] = False 
    print("ERROR! X-rays nowhere specified. Add to params or defaults file")
else:
    need_xraydetails = False
    if 'fx' in param_names:
        need_xraydetails = True
        print("X-rays included")
        print("   - fx is a free parameter")
    elif float(allp_dict['fx']) > 0.0:
        need_xraydetails = True
        print("X-rays included")
        print("   - fx is fixed at " + allp_dict['fx'])
    
    if need_xraydetails:   
        if not ('gamx' in all_names and 'Rinx' in all_names and 'mx' in all_names 
            and 'uinfx' in all_names):
            checkdict["Parameter space"] = False
            print("ERROR: One or more X-ray parameters are missing")
        else:
            print("Other parameters:")
            for apar in ('uinfx', 'mx', 'gamx', 'Rinx'):
                if apar in param_names:
                    print("   - " + apar + " is a free parameter")
                else:
                    print("   - " + apar + " = " + allp_dict[apar])
    else:
        print("X-rays not included")
        for apar in ('uinfx', 'mx', 'gamx', 'Rinx'):
            if apar in param_names:
                checkdict["Parameter space"] = False
                print("ERROR: " + apar + " is a free parameteter but " + 
                    "fx is 0.0!") 
 
p_line_names = []
for aline in plines:
    if not aline.startswith('#'):
        p_line_names.append(aline.split()[0])
if len(p_line_names) != len(np.unique(p_line_names)):
    print('ERROR: duplicates found in parameter space')
    pduplicates = []
    for ap1 in range(len(p_line_names)):
        for ap2 in range(len(p_line_names)):
            if ap1 != ap2 and p_line_names[ap1] == p_line_names[ap2]:
                pduplicates.append(p_line_names[ap1])
    print('Duplicate parameters: ')
    for dup in np.unique(pduplicates):
        print('  - ' + dup)
    checkdict["Parameter space"] = False
if 'fic' not in param_names:
    if 'fic' in fixed_names:
        for i in range(len(fixed_names)):
            if fixed_names[i] == 'fic':
                if fixed_pars[i] == -1:
                    tparams = ''
                    tlen = 0
                else:
                    tparams = ''
                    tlen = -10
    else:
        tparams = ''
        tlen = 0
    for aparam in param_names:
        if aparam in ('vclstart', 'vclmax', 'fvel'):
            tparams = tparams + '"' + aparam + '" ' 
            tlen = tlen + 1
    if tlen == 1:
        tparams = tparams + 'is'
    else:
        tparams = tparams + 'are'
    if tlen > 0:
        print('ERROR: no thick clumping, but ' + tparams + ' varied!')
        checkdict["Parameter space"] = False
            
if checkdict["Parameter space"] == True:
    print('\nParameter space ok.')

# Check stepsizes in parameter space
warning_na = 3.0
printsection('Step sizes')
checkdict['Step size'] = True
for i in range(len(param_space)):
    stepsize = param_space[i][2]
    pwidth = np.abs(param_space[i][1] - param_space[i][0])
    if np.abs(pwidth/stepsize - np.round(pwidth/stepsize,0)) > 0.0001:
        print('   - ' + param_names[i] + ': ERROR: step size cannot divide'
            ' parameter range into equal parts')
        print('     start: ' + str(param_space[i][0]) + ', stop: ' + 
             str(param_space[i][1]) + ', step: ',  str(param_space[i][2]) )
        print(np.abs(pwidth/stepsize))
        print(int(pwidth/stepsize))
        checkdict['Step size'] = False
    stepfrac = 1.0*stepsize/pwidth
    if stepsize < 0:
        checkdict['Step size'] = False
        print('  - ' + param_names[i] + ': ERROR: negative stepsize')
    if stepsize == 0.0:
        checkdict['Step size'] = False
        print('  - ' + param_names[i] + ': ERROR: stepsize = 0.0')
    if stepfrac*warning_na > w_gauss_na and type_na == 'frac':
        print('  - ' + param_names[i] + ': WARNING: stepsize is large ' +
            'compared to narrow mutation width')
        print('    ' + 'ratio is ' + str(round(w_gauss_na/stepfrac,2)) +
             ' while it is preferrably ~>' + str(warning_na) +
             '\n    Current step size: ' + str(stepsize) + 
             '\n    --> consider to decrease step size or increase w_gauss_na')
        checkdict['Step size'] = False
if checkdict['Step size'] == True:
    print('All stepsizes ok.')
         
# Check wheter all lines in the line list are present in the 
# FORMAL_INPUT master file 
printsection('Formal Input')
tf_formal = fw.create_FORMAL_INPUT(ctrldct["inicalcdir"], lineinfo[0], 
    create=False)
checkdict["FORMAL_INPUT"] = tf_formal

# Check properties reinsertion scheme
printsection('Reinsertion scheme')
ratio_po = ctrldct['ratio_po']
f_parent = ctrldct['f_parent']
if ratio_po == 1.0 and f_parent == 0.0:
    print('You will be using pure reinsertion')
else:
    print('You will be using elitist/fitness based reinsertion scheme')
if f_parent >= 1.0 - (1.0 / ratio_po):
    print('Ratios of parents/offspring are ok.')
    checkdict['Reinsertion'] = True
else:   
    print('ERROR in reinsertion ratios!')
    print('  ratio_po = ' + str(ratio_po))
    print('  f_parent = ' + str(f_parent))
    print('Must satisfy: f_parent >= 1-1/ratio_po')
    checkdict['Reinsertion'] = False

# Print summary of the results
sumstr = '  Summary of pre-run checks:'
sumstr = sumstr + (nd - len(sumstr) - 5)*' ' + '##'
print('\n' + nd*'#') 
print("##" + (nd-4)*' ' + '##')
print('## ' + sumstr)
print("##" + (nd-4)*' ' + '##')
for i in checkdict:
    if checkdict[i]:
        printsumline(i, 'all ok.', nd)
    else:
        printsumline(i, 'irregularities found --> CHECK INPUT!', nd)
print("##" + (nd-4)*' ' + '##')
print(nd*'#')

###################################
#        Plot param space         #
###################################
print('\nMaking plots...')
ncols = 3
nrows =int(math.ceil(1.0*len(param_space)/ncols))

w_gauss_na = ctrldct["w_gauss_na"]
w_gauss_br = ctrldct["w_gauss_br"]
b_gauss_na = ctrldct["b_gauss_na"]
b_gauss_br = ctrldct["b_gauss_br"]
do_doublebroad = ctrldct["doublebroad"]
nbins=[]
ccol = ncols - 1
crow = -1
props = dict(facecolor='white', alpha=1.0)
fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
for i in range(ncols*nrows):
    if ccol == ncols - 1:
        ccol = 0
        crow = crow + 1
    else:
        ccol = ccol + 1

    if i >= len(param_space):
        ax[crow,ccol].axis('off')
        continue
    
    start = param_space[i][0]
    stop = param_space[i][1]
    step = param_space[i][2]
    width = stop - start
    nbin = ((width)/step) + 1
    nbins.append(nbin)
    vlines = np.linspace(start, stop, nbin)

    if ctrldct["narrow_type"] == 'frac':
        sig_na = w_gauss_na * width
    elif ctrldct["narrow_type"] == 'step':
        sig_na = w_gauss_na * step
    sig_br = w_gauss_br * width
    b_na = b_gauss_na
    b_br = b_gauss_br
    xgauss = np.linspace(start, stop, 1000)
    na_gauss = pop.gauss(xgauss, b_na, 1.0, start+int(nbin/2)*step, sig_na)
    if do_doublebroad == 'yes':
        br_gauss = pop.double_gauss(xgauss, b_br, 1.0, start+int(nbin/2)*step, 
            sig_br)
    else:
        br_gauss = pop.gauss(xgauss, b_br, 1.0, start+int(nbin/2)*step, sig_br)
    br_gauss = br_gauss / max(br_gauss)
    na_gauss = na_gauss / max(na_gauss)

    for vline in vlines:
        ax[crow,ccol].axvline(vline, lw=1.0, alpha=0.5)
    ax[crow,ccol].plot(xgauss, na_gauss, 'red', lw=2)
    ax[crow,ccol].plot(xgauss, br_gauss, 'orange', lw=2)
    ax[crow,ccol].set_xlim(start-3*step, stop+3*step)
    ax[crow,ccol].set_title(param_names[i])
    ax[crow,ccol].set_yticks([])
    boxtext = 'Step size: ' + str(step) + ',\n nsteps: ' + str(len(vlines))
    ax[crow,ccol].text(0.05, 0.88, boxtext,
        transform=ax[crow,ccol].transAxes, fontsize=9, bbox=props)

fig.suptitle('Parameter space check: ' + run_name, fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig(param_fig)  

print('\n       >  Plotted parameter space and mutation distributions') 

###################################
#          Plot spectrum          #
###################################

lnames, line_params = fw.read_linelist(linesetname)
names, res, data_per_line, lweight = lineinfo

left_bounds = line_params[1]
right_bounds = line_params[2]

ncols = 4
nrows =int(math.ceil(1.0*len(names)/ncols))

ccol = ncols - 1
crow = -1
props = dict(facecolor='white', alpha=1.0)
errprops = dict(facecolor='red', alpha=1.0)
fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

extra_AA = 2.0
specwave, specflux, specerr = np.genfromtxt(spectrumname).T
specmin, specmax = np.min(specflux), np.max(specflux)
for i in range(ncols*nrows):
    if ccol == ncols - 1:
        ccol = 0
        crow = crow + 1
    else:
        ccol = ccol + 1

    if i >= len(names):
        ax[crow,ccol].axis('off')
        continue 

    for j in range(len(lnames)):
        if lnames[j] == names[i]:
            lbound = left_bounds[j]
            rbound = right_bounds[j]
    wave, flux, err = data_per_line[i]
    resolution = res[i]
    linename = names[i]
    weight = lweight[i]

    swave, sflux, serr = fw.parallelcrop(specwave, specflux, specerr,
        lbound-extra_AA, rbound+extra_AA)

    npoint_error = False
    if not len(wave) > len(param_names) + 1:
        print('\nERROR! Too few data points for this line compared to' 
            'the amount of free parameters.')
        print('Line: ' + linename)
        print('Data points: ' + str(len(wave)))
        print('Free params: ' + str(len(param_names))+'\n')
        npoint_error = True
    
    if npoint_error:
        ax[crow,ccol].axvspan(lbound,rbound, color='red', alpha=0.6)
    elif weight != 1.0:
        ax[crow,ccol].axvspan(lbound,rbound, color='green', alpha=0.3)
    else:
        ax[crow,ccol].axvspan(lbound,rbound, color='gold', alpha=0.3)
    ax[crow,ccol].errorbar(swave, sflux, yerr=serr, ls='', fmt='o',
        color='C0', alpha=0.5, markersize=3.0)
    ax[crow,ccol].errorbar(wave, flux, yerr=err, ls='', fmt='o',
        color='darkblue', markersize=3.0)
    ax[crow,ccol].set_ylim(specmin, specmax)
    ax[crow,ccol].set_title(linename)
    ax[crow,ccol].axhline(1.0, lw=1.0, color='grey')
    ax[crow,ccol].axvline(lbound, lw=1.0, color='red')
    ax[crow,ccol].axvline(rbound, lw=1.0, color='red')
    boxtext = 'R = ' + str(resolution) + ',\n weight = ' + str(weight)
    ax[crow,ccol].text(0.05, 0.05, boxtext,
        transform=ax[crow,ccol].transAxes, fontsize=8, bbox=props)
    if npoint_error:
        errortext = 'Not enough data points\n compared to #free parameters'
        ax[crow,ccol].text(0.5, 0.5, errortext, ha='center',
            transform=ax[crow,ccol].transAxes, fontsize=8, bbox=errprops,
            color='white')

fig.suptitle('Spectrum check: ' + run_name, fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig(spectrum_fig)


print('       >  Plotted spectrum and line selection') 

merger = PdfFileMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write(merged_report)
merger.close()

for pdf in pdfs:
    os.system("rm " + pdf)

jobscriptlines = ['#!/bin/bash',
'#SBATCH --job-name=' + run_name,
'#SBATCH -t 120:00:00',
'#SBATCH -N ' + str(int((ctrldct["nind"]+1)/n_cpu_core)),
'#SBACTH -n ' + str(ctrldct["nind"]+1),
'#SBATCH --no-requeue',
'',
'runname=' + run_name,
'do_restart=' + do_restart,
'ncpu=' + str(ctrldct["nind"]+1),
'inidir=FW_inicalc',
'',
'echo Run ${runname}',
'echo Using $ncpu CPUs',
'echo Do restart? $do_restart',
'',
'# Load modules',
'module load 2019',
'module load Python/3.6.6-intel-2018b',
'',
'# Define paths',
'scratch=/scratch-shared/sbrands/${runname}/',
'homedir=/home/sbrands/pyEA/',
'',
'echo Copying files',
'',
'# Create and copy directories and files',
'mkdir -p $scratch',
'cp -r ${homedir}*.py $scratch',
'cp -r ${homedir}filter_transmissions $scratch',
'mkdir -p ${scratch}input/',
'mkdir -p ${scratch}input/${runname}/',
'cp -r ${homedir}input/${runname}/* ${scratch}input/${runname}/.',
'cp -r ${homedir}${inidir} $scratch',
'',
'# Navigate to computation directory',
'cd $scratch',
'',
'echo Starting run!',
'date',
'',
'# Start run',
'if [ "$do_restart" == "yes" ]',
'then',
'    echo ...restarting run',
'    srun -n $ncpu python3 pyEA.py ${runname} -c',
'else',
'    echo ... creating output dir',
'    mkdir -p output',
'    echo ... starting run',
'    srun -n $ncpu python3 pyEA.py ${runname}',
'fi',
'',
'date',
'echo ... Run ENDED!']

with open(jobscriptfile,'w') as f:
    for aline in jobscriptlines:
        f.write(aline + '\n')

if do_restart == 'no':
    print('\nCreated ' + jobscriptfile + ' --- NO restart')
if do_restart == 'yes':
    print('\nCreated ' + jobscriptfile + ' --- WITH RESTART!')
print('\nEnd of pre-run check.')




