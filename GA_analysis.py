# PyGA analysis script
# Sarah Brands
# s.a.brands@uva.nl
# Created on: 19-11-2019
# Latest change: 16-01-2020
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
from PyPDF2 import PdfFileMerger, PdfFileReader
import cv2 #only required if make_paramspace_avi=True
import seaborn as sns
import img2pdf # for saving the scatter plots (fitness vs parameter) in
                      # PNG and then transforming them to pdf (otherwise they load
                      # ridiculously slow)

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
# The data should be stored in datapath/run_name There it should have the
# structure as in 'Outputs' of pyGA (so with a dir Output inside)
# The line profiles per model should either be in the 0000_0000.tar.gz archive,
# or alternatively if already unpacked, the line profiles should be in a folder
# names 0000_0000 in the parent directory 0000. (example numbers)
outpath = '/Users/sarah/Dropbox/projects/pyEA_plots/'
datapath = '/Users/sarah/pyEA/'

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
full_short_manual = 'manual'

# If full_short_manual is set to 'manual'
make_run_summary = True
make_fitnessdistribution_plot = True
make_fitnessdistribution_per_line_plot = False
make_chi2pgen_plot = True
make_lineprofiles_plot = True # Will take a long time depending on how many models are included.
make_correlation_plot = True # will always be done when make_correlation_per_line_plot = True
make_correlation_per_line_plot = False # This takes a lot of time
make_param_dist_plot = True
make_param_dist_plot_detail = False

# Videos (in full or manual only)
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

include_extra_spectrum = False #If making lineprofiles plot, include an extra spectrum for comparison?

# Add up to 2 extra spectra or models to the plots
# If you only want to add one, set the other path equal to ''
# Currently no error bars are supported as I only loaded cmfgen models
extra_spectrum_path1 = '/Absolute/path/to/other/spectrum/here/spectrum-file-without-errors.spec'
extra_spectrum_path2 = '/Absolute/path/to/other/spectrum/here/spectrum-file-without-errors.spec'

''' Specify how to select the models in the lineprofile plots'''
# Should be set to 'P' if you want it to correspond with the error bars pyGA gives on parameters.

# Choose between:
# - best_models_cutoff_method = 'P':
#    model selection based on proper chi square based probabilities
# - best_models_cutoff_method = 1.05 or other float:
#    model selection based on the reduced chi squared value (the float)
#    !!! no proper statistics done !!! Just to artificially show more models #FIXME)
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

''' ------------------------------------ '''
'''    Reading files and other things    '''
'''  that are useful for several plots   '''
''' -------------------------------------'''

if make_correlation_per_line_plot:
    make_correlation_plot = True

''' File names of output pdfs '''
# Sub output pdfs. Numbering determines the order in which they will come
# in the final report.

print("Loading stuff...")
name_runsummary = "0_runsummary_short_GAreport.pdf"
name_fitnessdistribution_plot = "1_fitnessdistribution_short_GAreport.pdf"
name_lineprofiles_plot = "2_lineprofiles_short_GAreport.pdf"
name_correlation_plot = "3_correlation_short_GAreport.pdf"
name_chi2pergen_plot = "4_chi2pergen_short_GAreport.pdf"
name_param_dist_plot = "5_param_dist_short_GAreport.pdf"
name_fitness_per_line_plot = "6_fitness_param_perline_GAreport.pdf"
name_correlation_per_line_plot = "7_correlation_perline_GAreport.pdf"
name_param_dist_zoom_plot = "8_param_dist_zoom_GAreport.pdf"


def calculateP(params, lc, chi2, normalize = False):
    degreesFreedom = len(lc) - len(params)
    if normalize:
        scaling = np.min(chi2)
    else: scaling = 1
    chi2 = (chi2 * degreesFreedom) / scaling
    probs = np.zeros_like(chi2)
    for i in range(len(chi2)):
        probs[i] = stats.chi2.sf(chi2[i], degreesFreedom)
    return probs

def parallelcrop(list1, list2, start_list1, stop_list1,
    list3='', list4=''):

    list1 = np.array(list1)
    list2 = np.array(list2)

    minarg = np.argmin(np.abs(list1-start_list1))
    maxarg = np.argmin(np.abs(list1-stop_list1))

    newlist1 = list1[minarg:maxarg]
    newlist2 = list2[minarg:maxarg]

    if list3 != '':
        newlist3 = list3[minarg:maxarg]
        if list4 != '':
            newlist4 = list4[minarg:maxarg]
            return newlist1, newlist2, newlist3, newlist4
        else:
            return newlist1, newlist2, newlist3

    return newlist1, newlist2

def cm_rgba(x):
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

if not outpath.endswith('/'):
    outpath = outpath + '/'
if not datapath.endswith('/'):
    datapath = datapath + '/'

datapath = datapath + run
plotpath = outpath + run
os.system("mkdir -p " + plotpath)

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

'''
=============== Reading output files ===============
'''

# with open(thechi2file) as f:
#     content = f.readlines()
# for line in input_file:
#     if line.endswith(" \n"):
#        line = line.replace(' \n', '\n')

x = pd.read_csv(thechi2file, sep=' ')
for acolname in x.columns:
    if acolname.startswith('#'):
        x = x.rename(columns = {acolname : acolname[1:]})
    if acolname.startswith('Unnamed: '):
        x = x.drop(acolname, 1)

outcontrol = np.genfromtxt(thecontrolfile,dtype='str').T
control_dct = dict(zip(outcontrol[0], outcontrol[1]))

maxindid = float(control_dct["nind"])
# maxgen = np.max(x['gen'].values)
maxgen = np.max(np.genfromtxt(thebestchi2file)[:,0])

# If a generation was not completed, drop those models. 
x = x.drop(x[x.gen > maxgen].index)

maxgenid = str(int(maxgen)).zfill(4)
print('Last generation: ' + str(maxgenid))
x['gen_id'] = map(lambda lam: str(int(lam)).zfill(4), x['gen'])

unique_gen = np.unique(x['gen_id'].values)
unique_genid = np.unique(x['gen_id'].values)

''' Evaluate chi2 per generation '''
median_chi2_per_gen = []
mean_chi2_per_gen = []
lowest_chi2_per_gen = []
for a_gen_id in unique_genid:
    x_1gen = x[x['run_id'].str.contains(a_gen_id + '_')]
    chi2val = x_1gen['chi2'].values
    # if len(chi2val) == maxind
    chi2val = chi2val[chi2val <= 1e6]
    median_chi2_per_gen.append(np.median(chi2val))
    mean_chi2_per_gen.append(np.mean(chi2val))
    lowest_chi2_per_gen.append(np.min(chi2val))

''' Read in mutation per generation '''
mutationpergen = np.genfromtxt(themutgenfile)
if len(mutationpergen) != len(median_chi2_per_gen):
    difflen = np.abs((len(mutationpergen)-len(median_chi2_per_gen)))
    mutationpergen = mutationpergen[:-difflen]
generation, mutation = mutationpergen.T

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
print('Min chi2: ' + str(min_redchi2_value))

probabilities = calculateP(params, normspectrum, x['chi2'], True)
x = x.assign(P=probabilities)

params_dic = OrderedDict()
params_error = OrderedDict()

min_p = 0.05
best = pd.Series.idxmax(x['P'])
ind = x['P'] > min_p

for i in params:
    params_dic[i[0]] = [float(i[1]), float(i[2])]
    params_error[i[0]] = [min(x[i[0]][ind]), max(x[i[0]][ind]), x[i[0]][best]]

param_keys = params_dic.keys()

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

print('Reading line file: ')
print('Number of diagnostic lines: ' + str(numlines))

''' Loading extra spectra '''

if extra_spectrum_path1 == '' and extra_spectrum_path1 == '':
    include_extra_spectra = False

if make_lineprofiles_plot and include_extra_spectrum:

    print('Reading extra spectra or models...')
    if extra_spectrum_path1 == '':
        pass
    elif os.path.exists(extra_spectrum_path1):
        cmfgenwave1, cmfgenflux1 = np.genfromtxt(extra_spectrum_path1).T
    else:
        print("Cannot plot extra spectra, file not found: " + extra_spectrum_path1)

    if extra_spectrum_path2 == '':
        pass
    elif os.path.exists(extra_spectrum_path2):
        cmfgenwave2, cmfgenflux2 = np.genfromtxt(extra_spectrum_path2).T
    else:
        print("Cannot plot extra spectra, file not found: " + extra_spectrum_path2)

''' Absolutely ridiculous workaround to get a legend on the param vs fitness plots '''
# Namely here I create a pdf that contains the colorbar,
# and then later I import the image of that colorbar into the figure

colorbar_jet_legend = "colorbar_generations_jet.jpg"

a = np.array([[0,1]])
plt.figure(figsize=(9, 3.5))
img = plt.imshow(a, cmap="jet")
plt.gca().set_visible(False)
cax = plt.axes([0.05, 0.3, 0.9, 0.1])#[0.3, 0.3, 0.8, 0.8])
cbar = plt.colorbar(cax=cax, orientation='horizontal', ticks=np.linspace(0,1,5))
cbar.ax.set_xticklabels(['0', str(int(0.25*float(maxgenid))), str(int(0.5*float(maxgenid))), str(int(0.75*float(maxgenid))), str(int(float(maxgenid)))],
    fontsize=30)
cbar.ax.set_xlabel('Generation',fontsize=30)
plt.savefig(colorbar_jet_legend, dpi=300)
plt.close()

''' ----------------------------------------------------------------------'''
'''                          Make pdfs with plots                         '''
''' ----------------------------------------------------------------------'''


''' ------------------------------------------'''
'''    Write run summary to pdf               '''
''' ------------------------------------------'''

if make_run_summary or full_short_manual in ('short', 'full'):
    fig, ax = plt.subplots(1,2,figsize=(12,12*1.41))
    maxfreepar = 40.

    # add extra whitespace for readability
    ew = 1.3

    # Font sizes for values + error values
    fz = 12
    fz_err = 10

    numround = 3

    pls = 1./maxfreepar
    for lp in range(len(param_keys)):
        strpname = param_keys[lp]
        strbestp = str(round(params_error[param_keys[lp]][2],numround))
        strdiffmin = str(round(params_error[param_keys[lp]][2] - params_error[param_keys[lp]][0],numround))
        strdiffplus =  str(round(params_error[param_keys[lp]][1] - params_error[param_keys[lp]][2],numround))
        strrange = ('[' + str(params_error[param_keys[lp]][0]) + ', '
            +  str(params_error[param_keys[lp]][1]) + ']')

        ax[0].text(0.05, 1.-pls-lp*ew*pls, strpname, transform=ax[0].transAxes, fontsize=fz, va='top')
        ax[0].text(0.35, 1.-pls-lp*ew*pls, strbestp, transform=ax[0].transAxes, fontsize=fz, va='top', ha='right')

        hpls = pls*0.25
        ax[0].text(0.37, (1.-pls-lp*ew*pls)+hpls, '+', transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.37, (1.-pls-lp*ew*pls)-hpls, '-', transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.40, (1.-pls-lp*ew*pls)+hpls, strdiffmin, transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.40, (1.-pls-lp*ew*pls)-hpls, strdiffplus, transform=ax[0].transAxes, fontsize=fz_err, va='top')
        ax[0].text(0.60, 1.-pls-lp*ew*pls, strrange, transform=ax[0].transAxes, fontsize=fz_err, va='top')

    run_info_labels = []
    run_info_data = []
    run_info_labels.append('Run name')
    run_info_data.append(runname)
    run_info_labels.append('# Generations')
    run_info_data.append(str(int(max(x['gen_id']))))
    run_info_labels.append('# Population size')
    run_info_data.append(str(int(maxindid)))
    run_info_labels.append('# Free parameters')
    run_info_data.append(len(param_keys))
    run_info_labels.append('Fitness best model')
    run_info_data.append(str(round(max(x['fitness']),5)))
    run_info_labels.append('Chi2 best model')
    run_info_data.append(str(round(min(x['chi2']),3)))

    for i in xrange(len(run_info_data)):
        ax[1].text(0.05, 1.-pls-i*ew*pls, run_info_labels[i], transform=ax[1].transAxes, fontsize=fz, va='top')
        ax[1].text(0.55, 1.-pls-i*ew*pls, run_info_data[i], transform=ax[1].transAxes, fontsize=fz, va='top')


    ax[0].set_title('Parameter fit summary', fontsize=fz+2)
    ax[1].set_title('Run meta information', fontsize=fz+2)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])

    plt.savefig(plotpath + name_runsummary)
    # print 'Saved', plotpath + name_runsummary
    plt.close()


''' ------------------------------------------'''
'''     Plot chi2 as function of generation   '''
''' ------------------------------------------'''

if (make_param_dist_plot or make_param_dist_plot_detail or
        full_short_manual in ('short', 'full')):

    plotscalefactor = 12.0
    v_param_pp = 6
    maxgenperline = 25

    tickspacing = 5

    x['gen_id_num'] = x['gen_id'].astype(float)
    latest_gen = max(x['gen_id_num'])
    nsubpages = int(math.ceil(float(latest_gen)/maxgenperline))
    gens_pp = latest_gen/nsubpages

    # If the 'all generations plot' is already zoomed because
    # there are not so many generations, never make the zoom.
    if latest_gen <= maxgenperline:
        make_param_dist_plot_detail = False

    nparams = len(param_keys)
    npages = int(math.ceil(1.0*nparams/v_param_pp))

    rowwidth = plotscalefactor*1.41
    colwidth = plotscalefactor

    print 'Starting in seaborn...'

    line_pdf_names = []
    if make_param_dist_plot:
        paramcount = 0
        for npage in range(npages):
            fig, ax = plt.subplots(v_param_pp,1, figsize=(colwidth,rowwidth), sharex=True)

            for i in range(v_param_pp):
                paramcount = paramcount + 1
                if paramcount <= nparams:
                    sns.violinplot( x=x['gen_id'], y=x[param_keys[v_param_pp*npage+i]],
                    linewidth=0.1, scale='width', ax=ax[i])
                else:
                    ax[i].axis('off')

                nticks = math.floor(int(maxgenid)/tickspacing)
                genticks = np.linspace(0, nticks*tickspacing, nticks+1)

                if i == v_param_pp -1 or paramcount == nparams:
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
            print("Parameter distribution (violinplot): saved page " + str(int(npage+1)) + " out of " + str(int(npages)) + " pages.")

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
                fig, ax = plt.subplots(v_param_pp,1, figsize=(colwidth,rowwidth), sharex=True)

                for i in range(v_param_pp):
                    paramcount = paramcount + 1
                    if paramcount <= nparams:
                        sns.violinplot( x=x_subpage['gen_id'], y=x_subpage[param_keys[v_param_pp*npage+i]],
                        linewidth=0.1, scale='width', ax=ax[i])
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
                pdfname_tmp = plotpath + 'param_dist_zoom_p' + str(int(npage)) + '_' + str(int(nsubpage)) + '.pdf'
                line_pdf_names.append(pdfname_tmp)
                plt.savefig(pdfname_tmp)
                plt.close()
                print("Parameter distribution (violinplot): saved page " + str(int(npage+1)) + " subpage " + str(int(nsubpage+1)) + " out of " + str(int(1.0*npages*nsubpages)) + " pages.")

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

if make_chi2pgen_plot or full_short_manual in ('short', 'full'):
    print("Making chi2 as a function of generation plots ...")

    pdfs_chi2pgen = []

    mingen = 35
    last_X_generations = int(float(maxgenid)-mingen)
    generation_crop, mutation_crop = parallelcrop(generation, mutation, float(mingen), float(maxgenid))

    plotscalefactor = 12.0
    if last_X_generations > 0 and maxgenid > last_X_generations + 10:
        fig, ax = plt.subplots(6,1, figsize=(plotscalefactor*1,plotscalefactor*1.41))
        make_zoom = True
    else:
        fig, ax = plt.subplots(3,1, figsize=(plotscalefactor*1,plotscalefactor*1.41))
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

    # after 35 generations we expect the solution to have more or less converged
    # so if there are more than 45 generations carried out, we also make a zoomed
    # plot of the chi2 behavior
    if make_zoom:
        xaxisvalue = np.linspace(mingen+1, float(maxgenid), (float(maxgenid)-mingen))
        ax[3].plot(xaxisvalue, median_chi2_per_gen[-last_X_generations:], label=r'median $\chi^2$', color=C0blue)
        ax[4].plot(xaxisvalue, mean_chi2_per_gen[-last_X_generations:], label=r'mean $\chi^2$', color=C0blue)
        ax[5].plot(xaxisvalue, lowest_chi2_per_gen[-last_X_generations:], label=r'lowest $\chi^2$', color=C0blue)
        vspanvalues0_y = ax[3].get_ylim()
        vspanvalues1_y = ax[4].get_ylim()
        vspanvalues2_y = ax[5].get_ylim()
        ax_3 = ax[3].twinx()
        ax_3.plot(generation_crop, mutation_crop, color=C1orange, label='mutation rate')
        ax_4 = ax[4].twinx()
        ax_4.plot(generation_crop, mutation_crop, color=C1orange, label='mutation rate')
        ax_5 = ax[5].twinx()
        ax_5.plot(generation_crop, mutation_crop, color=C1orange, label='mutation rate')
        ax[3].set_ylabel(r'median $\chi^2$')
        ax[4].set_ylabel(r'mean $\chi^2$ p')
        ax[5].set_ylabel(r'lowest $\chi^2$')
        ax_3.set_ylabel('mutation rate')
        ax_4.set_ylabel('mutation rate')
        ax_5.set_ylabel('mutation rate')
        ax[3].axvspan(mingen, maxgenid, alpha=0.2, color='yellow')
        ax[4].axvspan(mingen, maxgenid, alpha=0.2, color='yellow')
        ax[5].axvspan(mingen, maxgenid, alpha=0.2, color='yellow')
        ax[3].set_title(r'median $\chi^2$ per generation (last ' + str(int(last_X_generations)) + ' generations)')
        ax[4].set_title(r'mean $\chi^2$ per generation (last ' + str(int(last_X_generations)) + ' generations)')
        ax[5].set_title(r'lowest $\chi^2$ per generation (last ' + str(int(last_X_generations)) + ' generations)')
        ax[3].set_xlabel('Generation')
        ax[4].set_xlabel('Generation')
        ax[5].set_xlabel('Generation')
        ax[3].legend(loc='upper left')
        ax[4].legend(loc='upper left')
        ax[5].legend(loc='upper left')
        ax_3.legend(loc='upper right')
        ax_4.legend(loc='upper right')
        ax_5.legend(loc='upper right')
        plt.tight_layout(rect=[0.0,0.5,.7,1.0]) # rect adds extra whitespace so the size of these plots
                                                # is more balanced with respect to the lineprofile plots etc.

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
        plt.tight_layout(rect=[0.0,0.5,.7,1.0])# rect adds extra whitespace so the size of these plots
                                                # is more balanced with respect to the lineprofile plots etc.
        plt.savefig(plotpath + name_chi2pergen_plot)
        plt.close()

    if make_zoom:
        plt.tight_layout(rect=[0.0,0.0,0.7,1.0])
        plt.savefig(plotpath + name_chi2pergen_plot)
        plt.close()

''' ------------------------------------------'''
'''   Plot fitness as function of parameter   '''
''' ------------------------------------------'''
if make_fitnessdistribution_plot or full_short_manual in ('short', 'full'):

    print("Making fitness vs parameter plots...")

    nrows_ppage = nrows_fitnessparamplot # Can be changed, all lines will be plotted, the number of
                    # number of pages will be adapted accordingly.
    ncols_ppage = ncols_fitnessparamplot # Same holds for the columns.
    plotscalefactor = 12.0 # This sets the 'absolute size' of the line profile
                             # plots. The plots are always on A4 format, but the
                             # size of the labels are relative to the size of the
                             # plotting canvas. Therefore a higher number here
                             # means smaller labels on the plots.
    plots_ppage = nrows_ppage * ncols_ppage
    npages = int(math.ceil(1.0*len(param_keys) / plots_ppage))
    print("Plotting fitness vs parameter on " + str(int(npages)) + " page(s).")

    gen_id = map(lambda q: float(q[:4]), x['run_id'])
    gen_id_scaled = np.array(gen_id) / max(gen_id)
    scatter_colors = cm_rgba(gen_id_scaled)

    fitparam_jpg_names = []
    lp = -1 # paramater counter
    for apage in xrange(npages):
        fig, ax = plt.subplots(nrows_ppage, ncols_ppage, figsize=(plotscalefactor*1., plotscalefactor*1.41))
        for arow in xrange(nrows_ppage):
            for acol in xrange(ncols_ppage):
                lp = lp + 1 # Next parameter is plotted
                if lp < len(param_keys):
                    im1 = ax[arow,acol].scatter(x[param_keys[lp]].values, x['fitness'].values,
                        s=10.0, c=scatter_colors)
                    ax[arow,acol].set_ylim(0, 1.1*np.max(x['fitness'].values))
                    ax[arow,acol].axvspan(params_error[param_keys[lp]][0], params_error[param_keys[lp]][1], alpha=0.3, color='red')
                    ax[arow,acol].set_xlim(params_dic[param_keys[lp]][0], params_dic[param_keys[lp]][1])
                    ax[arow,acol].set_title(param_keys[lp])#, fontsize=14)
                    ax[arow,acol].axvspan(params_error[param_keys[lp]][0],
                        params_error[param_keys[lp]][1], alpha=0.3, color='red')

                    print(param_keys[lp] + ' - ' + str(params_error[param_keys[lp]][2]) + '    [' + str(params_error[param_keys[lp]][0]) + ', ' +  str(params_error[param_keys[lp]][1]) + ']')
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

    with open(plotpath + name_fitnessdistribution_plot, "wb") as out_file:
        out_file.write(img2pdf.convert(fitparam_jpg_names))

    for ajpg in fitparam_jpg_names:
        os.system("rm " + ajpg)

    # # Merge all line profile plot pages into one document.
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

if make_lineprofiles_plot or full_short_manual in ('short', 'full'):

    print("Making lineprofile plots...")

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
        x_best = x.loc[x['P'] > 0.05]
    else:
        x_best = x.loc[x['scaledRedChi2'] < best_models_cutoff_method]

    the_best_model = x.loc[x['scaledRedChi2'] == 1.0] #FIXME should instead just select lowest chi2
    print("Amount of models with lowest chi2 is: " + str(int(len(the_best_model))))
    the_best_model = the_best_model.iloc[0]
    the_best_model = the_best_model['run_id']
    print("The best model: " + the_best_model)
    bestmodeldir = savedmoddir +  the_best_model[:4] + '/' + the_best_model

    # 2. Saving directories of the to be plotted models.
    best_models = x_best['run_id'].values
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
    # print("Loading " + str(len(x_best)) + " best fitting models...")
    # for bmtar, bmdir in zip(best_models_tar, best_models_dirs):
    #     if not os.path.isdir(bmdir):
    #         mkdircommand = 'mkdir ' + bmdir
    #         untarcommand = 'tar -C ' + bmdir + ' -xzf ' + bmtar
    #         rmtarcommand = 'rm ' + bmtar
    #         os.system(mkdircommand)
    #         os.system(untarcommand)
    #         os.system(rmtarcommand)

    ''' Setup page layout '''
    nrows_ppage = nrows_lineprofileplot # Can be changed, all lines will be plotted, the number of
                    # number of pages will be adapted accordingly.
    ncols_ppage = ncols_lineprofileplot # Same holds for the columns.
    plotscalefactor = 12.0 # This sets the 'absolute size' of the line profile
                             # plots. The plots are always on A4 format, but the
                             # size of the labels are relative to the size of the
                             # plotting canvas. Therefore a higher number here
                             # means smaller labels on the plots.
    plots_ppage = nrows_ppage * ncols_ppage
    npages = int(math.ceil(1.0*numlines / plots_ppage))
    print("Plotting the line profiles on " + str(int(npages)) + " pages.")

    ''' Reading the spectrum '''
    wave, flux, error = np.genfromtxt(thespectrumfile).T

    lc = -1 # line counter
    line_pdf_names = []

    # For each page we set up a grid of axes, each plotting the models and
    # data of one diagnostic line.
    for apage in xrange(npages):
        fig, ax = plt.subplots(nrows_ppage, ncols_ppage, figsize=(plotscalefactor*1., plotscalefactor*1.41))
        for arow in xrange(nrows_ppage):
            for acol in xrange(ncols_ppage):
                lc = lc + 1 # Next diagnostic line is plotted

                if lc < numlines: # As long as not all the lines are plotted.
                    print("Plotting " + linenames[lc])

                    ''' Plot the models '''
                    for bmdir in best_models_dirs:
                        linefile_tmp = bmdir + '/' + linenames[lc] + '.prof.fin'
                        linewave_tmp, lineflux_tmp = np.genfromtxt(linefile_tmp).T
                        linewave_tmp = vrshift(linewave_tmp, line_radialvelocity[lc])
                        if bmdir != bestmodeldir:
                            # The x best models in green
                            ax[arow, acol].plot(linewave_tmp, lineflux_tmp, color='green')
                        else:
                            # The best fitting model in red, note the high zorder so that it is on top.
                            # +20 in zorder is a bit random, but I thought, I don't now how many
                            # other things line axhlines, have orders higher than the amount of lines.
                            ax[arow, acol].plot(linewave_tmp, lineflux_tmp, color='red', zorder=len(best_models_dirs)+20, lw=2.0)

                    ''' Plot the data '''
                    # The spectra are cropped because otherwise plotting
                    # will take up a lot of memory. (The set_xlim might therefore
                    # be a bit unnecessary)
                    wave_tmp, flux_tmp = parallelcrop(wave, flux, linestarts[lc], linestops[lc])
                    wave_tmp, error_tmp = parallelcrop(wave, error, linestarts[lc], linestops[lc])
                    ax[arow, acol].axhline(1.0, color='grey', lw=1.0)
                    ax[arow, acol].set_title(linenames[lc])
                    ax[arow, acol].set_xlim(linestarts[lc], linestops[lc])
                    flux_renorm_tmp = renormalize_line(wave_tmp, flux_tmp, line_norm_starts[lc], line_norm_leftval[lc], line_norm_stops[lc], line_norm_rightval[lc])
                    ax[arow, acol].errorbar(wave_tmp, flux_renorm_tmp, yerr=error_tmp, marker='o', markersize=0.1, linestyle='None', color='black', zorder=10e10)

                    ''' Plot extra spectrum if wanted '''
                    if include_extra_spectrum:
                        try:
                            cmfgenwave_tmp, cmfgenflux_tmp = parallelcrop(cmfgenwave1, cmfgenflux1, linestarts[lc], linestops[lc])
                            ax[arow, acol].plot(cmfgenwave_tmp, cmfgenflux_tmp, color='blue', lw=2.0, zorder=len(best_models_dirs)+20)
                        except:
                            print(">> Something might have gone wrong with plotting the extra spectra ")
                        try:
                            cmfgenwave_tmp, cmfgenflux_tmp = parallelcrop(cmfgenwave2, cmfgenflux2, linestarts[lc], linestops[lc])
                            ax[arow, acol].plot(cmfgenwave_tmp, cmfgenflux_tmp, color='orange', lw=2.0, zorder=len(best_models_dirs)+21)
                        except:
                            print(">> Something might have gone wrong with plotting the extra spectra ")
                else:
                    # We loop through all axes (rows, cols) that are set up,
                    # but if there are not enough lines to fill a page axes
                    # are removed so that the space is truly empty.
                    ax[arow, acol].axis('off')

        plt.tight_layout()

        # Saving page in pdf. Pdfs will be sticked later.
        pdfname_tmp = plotpath + 'line_profiles_p' + str(int(apage)) + '.pdf'
        line_pdf_names.append(pdfname_tmp)
        plt.savefig(pdfname_tmp)
        plt.close()
        print("Lineprofiles: saved page " + str(int(apage+1)) + " out of " + str(int(npages)) + " pages.")

    # Merge all line profile plot pages into one document.
    if os.path.isfile(plotpath + name_lineprofiles_plot):
        os.system("rm " + plotpath + name_lineprofiles_plot)
    merger = PdfFileMerger()
    for filename in line_pdf_names:
        merger.append(PdfFileReader(file(filename, 'rb')))
    merger.write(plotpath + name_lineprofiles_plot)

    # Remove the individual pages after the file has been merged.
    for filename in line_pdf_names:
        os.system("rm " + filename)

''' ------------------------------------------'''
'''   Videos of parameter space exploration   '''
''' ------------------------------------------'''
if (full_short_manual == 'manual' and make_paramspace_avi) or full_short_manual in ('full'):

    for param_pair in param_pair_list_avi:

        xparam = param_pair[0]
        yparam = param_pair[1]
        print("Making video of paramater space exploration of " + xparam + ", " + yparam)

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
            ax.set_title("Gen = " + the_genid + ", Mutation rate = " + themutationrate)
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

if make_correlation_plot or full_short_manual in ('short', 'full'):
    print("Making plots of correlation between parameters")

    paramlist = params_dic.keys()

    nbins1 = 10
    nbins2 = nbins1

    colorbar_chi2 = 'viridis'
    colorbar_other = 'viridis_r'

    plotscalefactor = 12.0

    correlation_plot_list = ['chi2']
    if make_correlation_per_line_plot or full_short_manual in ('full'):

        for diagnostic in linenames:
            correlation_plot_list.append(diagnostic)
        all_jpg_names = []

    make_legend = True
    for which_plot in correlation_plot_list:

        fig, ax = plt.subplots(len(paramlist)-1,len(paramlist)-1, figsize=(plotscalefactor*1.0, plotscalefactor*1.0))
        if which_plot == 'chi2':
            the_colormap = colorbar_chi2
        else:
            the_colormap = colorbar_other

        if which_plot == 'chi2':
            print("Making correlation plot for all lines together ")
        else:
            print("Making correlation plot for " + which_plot)

        param_pairs_correlation = []
        for ii in xrange(len(paramlist)):
            for jj in xrange(len(paramlist)):
                xparam = paramlist[ii]
                yparam = paramlist[jj]
                if (xparam != yparam) and ([yparam, xparam] not in param_pairs_correlation):

                    #print("Assessing correlation between " + xparam + " and " + yparam )

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
                                if (p1low < ap1) and (p1up > ap1) and (p2low < ap2) and (p2up > ap2):
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

                    if which_plot == 'chi2' and (np.log10(np.nanmax(chi2_matrix)) - np.log10(np.nanmin(chi2_matrix)) > 2.0):
                        # Fidle with scale of correlation plot
                        # if the differences between the chi2s are very large, it may be
                        # better to use a log scale
                        chi2_matrix = np.log10(chi2_matrix)
                        #print("Using log scale for correlation plot ")
                    else:
                        chi2_matrix = np.array(chi2_matrix)

                    chi2_matrix = chi2_matrix.T # so that x and y correspond with how we normally define x and y.
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
                                ax[jj-1][ii].text(0.5, 0.8, 'Correlation\n(all lines)', ha='center', va='center', transform=ax[jj-1][ii].transAxes)
                            else:
                                ax[jj-1][ii].text(0.5, 0.8, 'Correlation\n(' + which_plot + ')', ha='center', va='center',transform=ax[jj-1][ii].transAxes)
                        if jj == 1 and ii == len(paramlist)-2:
                            if which_plot == 'chi2':
                                if the_colormap.endswith('_r'):
                                    the_colormap = the_colormap[-2]
                                else:
                                    the_colormap = the_colormap + '_r'
                            else:
                                doreverse = False
                            img = sns.heatmap(np.array([[0,1]]), ax=ax[jj-1][ii], mask=np.array([[True,True]]),
                            cbar_kws={"orientation": "horizontal"}, cmap=the_colormap)
                            cbar = img.collections[0].colorbar
                            cbar.set_ticks([.2, .8])
                            cbar.set_ticklabels(['<-- worse fit', '\nbetter fit -->'])

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.02, wspace=0.05)
        if which_plot == 'chi2':
            plt.savefig(plotpath + name_correlation_plot) #FIXME I would like to have this figure too, on A4 page format...
                                                      # However editing canvas size or rect in tight_layout spreads the plots over
                                                      # the full A4, I want them an the top of middle, but in a square, not spread out :(
            plt.close()
        else:
            the_jpg_name = plotpath + 'correlation_' + which_plot + '.jpg'
            all_jpg_names.append(the_jpg_name)
            plt.savefig(the_jpg_name)
            plt.close()

    if make_correlation_per_line_plot or full_short_manual in ('full'):
        with open(plotpath + name_correlation_per_line_plot, "wb") as out_file:
            out_file.write(img2pdf.convert(all_jpg_names))

        for ajpg in all_jpg_names:
            os.system("rm " + ajpg)


''' ------------------------------------------'''
'''  Plot fitn per line as function of param  '''
''' ------------------------------------------'''

if (full_short_manual == 'manual' and make_fitnessdistribution_per_line_plot) or full_short_manual in ('full'):

    print("Making fitness per parameter plots per diagnostic line...")

    fitparam_jpg_names = []

    for diagnostic in linenames:

        nrows_ppage = nrows_fitnessparamplot # Can be changed, all lines will be plotted, the number of
                        # number of pages will be adapted accordingly.
        ncols_ppage = ncols_fitnessparamplot # Same holds for the columns.
        plotscalefactor = 12.0 # This sets the 'absolute size' of the line profile
                                 # plots. The plots are always on A4 format, but the
                                 # size of the labels are relative to the size of the
                                 # plotting canvas. Therefore a higher number here
                                 # means smaller labels on the plots.
        plots_ppage = nrows_ppage * ncols_ppage
        npages = int(math.ceil(1.0*len(param_keys) / plots_ppage))
        print("Plotting fitness vs parameter for " + diagnostic + " on " + str(int(npages)) + " page(s).")

        gen_id = map(lambda q: float(q[:4]), x['run_id'])
        gen_id_scaled = np.array(gen_id) / max(gen_id)
        scatter_colors = cm_rgba(gen_id_scaled)

        lp = -1 # paramater counter
        for apage in xrange(npages):
            fig, ax = plt.subplots(nrows_ppage, ncols_ppage, figsize=(plotscalefactor*1., plotscalefactor*1.41))
            for arow in xrange(nrows_ppage):
                for acol in xrange(ncols_ppage):
                    lp = lp + 1 # Next parameter is plotted
                    if lp < len(param_keys):
                        line_fitness = x[diagnostic].values
                        ax[arow,acol].scatter(x[param_keys[lp]].values, line_fitness,
                            s=10.0, c=scatter_colors)
                        ax[arow,acol].set_ylim(0, 1.1*np.max(line_fitness))
                        ax[arow,acol].axvspan(params_error[param_keys[lp]][0], params_error[param_keys[lp]][1], alpha=0.3, color='red')
                        ax[arow,acol].set_xlim(params_dic[param_keys[lp]][0], params_dic[param_keys[lp]][1])
                        ax[arow,acol].set_title(param_keys[lp])#, fontsize=14)
                        ax[arow,acol].axvspan(params_error[param_keys[lp]][0],
                            params_error[param_keys[lp]][1], alpha=0.3, color='red')
                    else:
                        ax[arow,acol].axis('off')


            plt.suptitle('Fitness vs. parameter (' + diagnostic + ')', fontsize=16)
            plt.tight_layout(rect=[0, 0.00, 1.0, 0.95])

            # Load the earlier produced colorbar/legend.
            # Sorry for this very desparate workaround
            imlegend = plt.imread(colorbar_jet_legend)
            newax = fig.add_axes([0.72, 0.885, 0.25, 0.2], anchor='C')
            newax.imshow(imlegend)
            newax.axis('off')

            fit_param_pagename_jpg = plotpath + 'overview_' + diagnostic + '_' + str(int(apage)) + '.jpg'
            fitparam_jpg_names.append(fit_param_pagename_jpg)
            plt.savefig(fit_param_pagename_jpg)#, dpi=100)
            plt.close()

    with open(plotpath + name_fitness_per_line_plot, "wb") as out_file:
        out_file.write(img2pdf.convert(fitparam_jpg_names))

    for ajpg in fitparam_jpg_names:
        os.system("rm " + ajpg)



''' --------------- OUTPUT -------------------'''
'''   Merge all available pdfs into rerport   '''
''' ------------------------------------------'''

''' Short report '''
# Does not contain the correlation and fitness plots per line.
# In the end, merge all line profile plot pages into one document.
osld = os.listdir(plotpath)
report_pdfs = []
for afile in osld:
    if afile.endswith('_GAreport.pdf') and 'short' in afile:
        report_pdfs.append(plotpath + afile)

if os.path.isfile(name_shortreport_pdf):
    os.system("rm " + name_shortreport_pdf)
merger = PdfFileMerger()
for filename in report_pdfs:
    merger.append(PdfFileReader(file(filename, 'rb')))
merger.write(name_shortreport_pdf)
print("Short report saved to:")
print(name_shortreport_pdf)

''' Full report '''
# Containing all plots
# In the end, merge all line profile plot pages into one document.
osld = os.listdir(plotpath)
report_pdfs = []
for afile in osld:
    if afile.endswith('_GAreport.pdf'):
        report_pdfs.append(plotpath + afile)

if os.path.isfile(name_fullreport_pdf):
    os.system("rm " + name_fullreport_pdf)
merger = PdfFileMerger()
for filename in report_pdfs:
    merger.append(PdfFileReader(file(filename, 'rb')))
merger.write(name_fullreport_pdf)
print("Full report saved to:")
print(name_fullreport_pdf)
