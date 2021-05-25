# Sarah Brands s.a.brands@uva.nl
# This script is part of Kiwi-GA: https://github.com/sarahbrands/Kiwi-GA
# This is the main script for the python Evolutionary Algorithm. 
# It prepares the MPI, reads input files, then either initiates a 
# run or restarts one. The bulk of the computation is then done by
# iterating through x generations. 

import __future__
import os
import sys
import numpy as np
import collections
import argparse
import functools
from schwimmbad import MPIPool

import paths as paths
import population as pop
import fastwind_wrapper as fw

"""
***************************** #FIXME *****************************
# Rewrite functions in such a way that the parameters that are
  passed are full dictionaries, instead of separate entries.
  It functions as it is now, but it's messy. 

******************************************************************
"""

''' INITIALIZE / SET UP '''

# Start MPIPool to control the distrubution of models over CPUs
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

# Read command line arguments and exit if no input is found.
parser = argparse.ArgumentParser(description='Run pika2')
parser.add_argument('runname', help='Specify run name')
parser.add_argument('-c', action='store_true', help='Continue run')
args = parser.parse_args()
inputdir = paths.inputdir + args.runname + '/'
if not fw.check_indir(inputdir):
    pool.close()
    sys.exit()

# Initial setup of directories and file paths
# Note that if you want to make a subdirectory, you have to take 
# this into account when going back to the main directory after 
# fastwind has run (in the function execute_fastwind)
outputdir = paths.outputdir
fd = fw.make_file_dict(inputdir, outputdir)
fw.mkdir(paths.outputdir)
fw.mkdir(outputdir)
outdir, rundir, savedir, indir = fw.init_setup(outputdir)
fw.copy_input(fd,indir)
# The control file is from now on read from the input_copy dir
# So if the user wants to change control files, this has to 
# be done there. Changes in the original input dir have no effect.
fd["control_in"] = indir + fd["control_in"].split('/')[-1]

# Read control parameters
cdict = fw.read_control_pars(fd["control_in"])

# Remove (new run) or replace (continued run) old output files. 
fw.prepare_output_files(fd, args.c)

''' READ INPUT PARAMETERS AND DATA '''

# Read input files and data
the_paramspace = fw.read_paramspace(fd["paramspace_in"])
param_names, param_space, fixed_names, fixed_pars = the_paramspace
radinfo = np.genfromtxt(fd["radinfo_in"], comments='#', dtype='str')
defnames, defvals = fw.get_defvals(fd["defvals_in"], param_names, fixed_names)
all_pars = [param_names, fixed_pars, fixed_names, defvals, defnames, radinfo]
dof = len(param_names)
lineinfo = fw.read_data(fd["linelist_in"], fd["normspec_in"])

''' PREPARE FASTWIND '''

# Create a FORMAL_INPUT file containing the relevant lines.
fw.create_FORMAL_INPUT(cdict['inicalcdir'], lineinfo[0], fd["linelist_in"])

# Initialise the fitness function with parameters that are
# the same for every model.
eval_fitness = functools.partial(fw.evaluate_fitness, cdict["inicalcdir"],
    rundir, savedir, all_pars, cdict["modelatom"], cdict["fw_timeout"],
    lineinfo, dof, cdict["fitmeasure"], fd["chi2_out"], param_names)

''' THE GENETIC ALGORITHM STARTS HERE '''

# When starting from scratch, the first generation is calculated
if not args.c:
    gencount = 0
    
    # Pick first generation of models. The amount of individuals can
    # be more than a typical generation.
    nind_first_gen = int(cdict["f_gen1"]*cdict["nind"])
    generation = pop.init_pop(nind_first_gen, param_space, fd["dupl_out"])
    modnames = fw.gen_modnames(gencount, nind_first_gen)
    
    # Reorder input for eval_fitness function and assess fitness.
    names_genes = []
    for mname, gene in zip(modnames, generation):
        names_genes.append([mname, gene])               
    parallelout = list(pool.map(eval_fitness, names_genes))
    fitmeasures, red_chi2s = np.transpose(parallelout)
    
    # If the first generation is larger than the typical generation,
    # The top nind fittest individuals of this generation are selected.
    if cdict["f_gen1"] > 1:
        topfit = pop.get_top_x_fittest(generation, fitmeasures, cdict["nind"])
        generation, fitmeasures = topfit
    
    # The fittest individual is selected
    genbest, best_fitness = pop.get_fittest(generation, fitmeasures)
    lowest_redchi2 = np.min(red_chi2s)
    pop.store_lowestchi2(fd["bestchi2_out"], lowest_redchi2, 0)
    pop.print_report(gencount, best_fitness, np.median(fitmeasures),
        cdict["be_verbose"])

    mutation_rate = cdict["mut_rate_init"] # initial mutation rate
    gen_variety = pop.assess_variation(generation, param_space, genbest)

    pop.store_mutation(fd["mutation_out"], mutation_rate, gencount)
    pop.store_charbonneaulimits(fd["charblim_out"], cdict, gencount)
    pop.store_genvar(fd["genvar_out"], gencount, gen_variety, fitmeasures)
    os.system('cp ' + fd["chi2_out"] + ' ' + fd["chi2_cont"])
    os.system('cp ' + fd["dupl_out"] + ' ' + fd["dupl_cont"]) 
    np.savetxt(fd["gen_cont"], generation)
    np.savetxt(fd["fit_cont"], fitmeasures)
    np.savetxt(fd["redchi_cont"], red_chi2s)

# When continuing an old run, simply pick up the gencount, mutation
# rate and the fitmeasures and parameters of the last generation. 
else:
    gencount, mutation_rate = fw.read_mut_gen(fd["mutation_out"])
    generation = np.genfromtxt(fd["gen_cont"])
    fitmeasures = np.genfromtxt(fd["fit_cont"])
    red_chi2s = np.genfromtxt(fd["redchi_cont"])
    genbest, best_fitness = pop.get_fittest(generation, fitmeasures)

while gencount <= cdict["ngen"]:
    
    gencount = gencount + 1

    # Read control parameters: the user can change these duuring the run.
    # !!! The control file is read from *input_copy* directory, 
    #     so changing values in the input directory has no effect !
    cdict = fw.read_control_pars(fd["control_in"])
  
    if gencount > cdict["ngen"]:
        break
 
    # Re-initialise the fitness function with parameters that are
    # the same for every model. The control parameters, especially 
    # the fw_timeout, might be changed by the user during the run. 
    eval_fitness = functools.partial(fw.evaluate_fitness, cdict["inicalcdir"],
        rundir, savedir, all_pars, cdict["modelatom"], cdict["fw_timeout"],
        lineinfo, dof, cdict["fitmeasure"], fd["chi2_out"], param_names)
    
    # Reproduce and asses fitness
    generation_o = pop.reproduce(generation, fitmeasures, mutation_rate,
        cdict["clone_fraction"], param_space, fd["dupl_out"],
        cdict["w_gauss_na"], cdict["w_gauss_br"], cdict["b_gauss_na"],
        cdict["b_gauss_br"], cdict["mut_rate_na"], cdict["nind"],
        cdict["narrow_type"], cdict["broad_type"], cdict["doublebroad"],
        cdict["use_string"], cdict["sigs_string"], 
        cdict["fracdouble_string"])
    modnames = fw.gen_modnames(gencount, cdict["nind"])

    names_genes = []
    for mname, gene in zip(modnames, generation_o):
        names_genes.append([mname, gene])
    parallelout = list(pool.map(eval_fitness, names_genes))
    fitmeasures_o, red_chi2s_o = np.transpose(parallelout)

    # The parent population (generation, fitmeasures), is created 
    # based on the offpsring pop. (generation_o, fitmeasures_o)
    if cdict["ratio_po"] == 1.0 and cdict["f_parent"] == 0.0:
        # Case of pure reinsertion: offspring pop = parent pop.,
        # but the fittest individual of the run always survives
        # (This only has to be done explictly if the pure reinsertion
        # scheme is used, otherwise this is the case automatically.)
        generation, fitmeasures = pop.reincarnate(generation_o, fitmeasures_o,
            genbest, best_fitness)
        red_chi2s = red_chi2s_o
    else:
        # In the other cases, i.e. when the reinsertion schemes of 
        # elitist and fitness-based are combined, the best inidividuals
        # of the parent population and the offspring are combined. 
        generation_o, fitmeasures_o = pop.get_top_x_fittest(generation_o, 
            fitmeasures_o, cdict["n_keep_offspring"])
        generation, fitmeasures = pop.get_top_x_fittest(generation, fitmeasures, 
            cdict["n_keep_parent"])
        generation = np.concatenate((generation, generation_o))
        fitmeasures  = np.concatenate((fitmeasures, fitmeasures_o))
        red_chi2s = np.concatenate((red_chi2s, red_chi2s_o))
    
    genbest, best_fitness = pop.get_fittest(generation, fitmeasures)
    best_rchi2 = np.min(red_chi2s)

    gen_variety = pop.assess_variation(generation, param_space, genbest)
    mean_gen_variety = np.mean(gen_variety)
    pop.store_genvar(fd["genvar_out"], gencount, gen_variety, fitmeasures)
    pop.store_lowestchi2(fd["bestchi2_out"], best_rchi2, gencount)

    # Before adjusting the mutation rate, set the charbonneau limits, 
    # if 'autocharb' is chosen. This is done every generation so that you
    # can change the mutation type during the run, if wanted. 
    if cdict['mut_adjust_type'] == 'autocharb':
        cdict = pop.autoadjust_charbonneau(cdict, fd, gencount)
 
    # Depending on the scheme chosen, adjust the mutation rate.
    # If the chosen scheme is 'constant', no adaption is made. 
    if cdict["mut_adjust_type"] in ('charbonneau', 'autocharb'):
        mutation_rate = pop.adjust_mutation_rate_charbonneau(mutation_rate,
            fitmeasures, cdict["mut_rate_factor"], cdict["mut_rate_min"],
            cdict["mut_rate_max"], cdict["fit_cutoff_min_charb"],
            cdict["fit_cutoff_min_charb"])
    
    elif cdict["mut_adjust_type"] == 'genvariety':
        mutation_rate = pop.adjust_mutation_genvariety(mutation_rate,
            cdict["cutoff_decrease_genv"], cdict["cutoff_increase_genv"],
            cdict["mut_rate_factor"], cdict["mut_rate_min"],
            cdict["mut_rate_max"], mean_gen_variety, param_space)

    # Store mutation rate and files for run continuation
    # Copies of the chi2 file and dupl file are certain to only 
    # contain the output of a fully completed generation. 
    pop.store_mutation(fd["mutation_out"], mutation_rate, gencount)
    pop.store_charbonneaulimits(fd["charblim_out"], cdict, gencount)
    os.system('cp ' + fd["chi2_out"] + ' ' + fd["chi2_cont"])
    os.system('cp ' + fd["dupl_out"] + ' ' + fd["dupl_cont"]) 
    np.savetxt(fd["gen_cont"], generation)
    np.savetxt(fd["fit_cont"], fitmeasures)
    np.savetxt(fd["redchi_cont"], red_chi2s)

    pop.print_report(gencount, best_fitness, np.median(fitmeasures),
        cdict["be_verbose"])

pool.close()




#
