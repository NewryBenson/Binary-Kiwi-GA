# Sarah Brands s.a.brands@uva.nl
# This script is part of Kiwi-GA: https://github.com/sarahbrands/Kiwi-GA
# This script contains functions for controlling the evolutionary
# algorithm, i.e. population control: reproduction, mutation rate etc.
# For using the algorithm this has to be combined with a wrapper for
# the model/function to be optimised.

import os
import random
import numpy as np
import string

import subprocess
import time

######################################################################
# Functions that control the parameters of the population
######################################################################

def gauss(x,baseline,height,center,sigma):
    """
    Gaussian with continuum at 0.
    """
    y = baseline + height * np.exp(-(x-center)**2 / (2*sigma**2))
    return y

def double_gauss(x,baseline,height,center,sigma):
    """
    Gaussian with continuum at 0.
    """
    center1 = center-0.75*sigma
    center2 = center+0.75*sigma

    sigma = 0.4*sigma

    y = (baseline + height * np.exp(-(x-center1)**2 / (2*sigma**2)) +
        height * np.exp(-(x-center2)**2 / (2*sigma**2)))

    return y


def rmfile(filename):
    """ Remove a file """
    if os.path.isfile(filename):
        os.system("rm " + filename)

def store_models(duplfile, individual):
    """ Write the paramters of each individual in the population into
    a textfile (to check for duplictity later, so without run_id).
    """
    write_lines = []

    param_string = ''
    for param in individual:
        param_string = param_string + str(param) + ' '
    istr = param_string + '\n'
    write_lines.append(istr)

    with open(duplfile, 'a') as the_file:
        for aline in write_lines:
            the_file.write(aline)

def charbonneau_ratio(the_fitn):
    """ Measure for the fitness spread of the population"""
    best_mod = np.min(the_fitn)
    median_mod = np.median(the_fitn)

    charbratio = np.abs(best_mod - median_mod) / (best_mod + median_mod)

    return best_mod, median_mod, charbratio

def store_lowestchi2(txtfile, lowestchi2, gcount):
    """ Write the paramters and fitness of each individual in the
    population into a textfile.
    """
    write_lines = []

    gcount = str(int(gcount)).zfill(4)

    if not os.path.isfile(txtfile):
        headerstring = '#Generation Lowest_Chi2 \n'
        write_lines.append(headerstring)

    chi2line = gcount + ' ' + str(lowestchi2) + '\n'
    write_lines.append(chi2line)

    with open(txtfile, 'a') as the_file:
        for aline in write_lines:
            the_file.write(aline)

def store_mutation(txtfile, mutrate, gcount):
    """ Write the mutation rate of the current generation into a
    textfile.
    """
    write_lines = []

    # gcount = str(int(gcount)).zfill(4)

    if not os.path.isfile(txtfile):
        headerstring = '#Generation Mutation_rate \n'
        write_lines.append(headerstring)

    mutline = str(gcount) + ' ' + str(mutrate) + '\n'
    write_lines.append(mutline)

    with open(txtfile, 'a') as the_file:
        for aline in write_lines:
            the_file.write(aline)

def store_charbonneaulimits(txtfile, thedct, gcount):
    """ Write the charbonneau limits used for adapting the mutation
    rate of the current generation into a textfile.
    """
    write_lines = []

    if not os.path.isfile(txtfile):
        headerstring = '#Generation charblim_min charblim_max \n'
        write_lines.append(headerstring)

    charbmin = thedct["fit_cutoff_min_charb"]
    charbmax = thedct["fit_cutoff_max_charb"]

    carline = str(gcount) + ' ' + str(charbmin) + ' ' + str(charbmax) + '\n'
    write_lines.append(carline)

    with open(txtfile, 'a') as the_file:
        for aline in write_lines:
            the_file.write(aline)

def store_genvar(txtfile, gcount, genvar, fitnesses):
    """ Write the genetic variety and charbonneau ratio of the current
    genertaion into a textfile.
    """
    write_lines = []

    if not os.path.isfile(txtfile):
        headerstring = ('#Generation median_genvariety charbonneau_ratio '
                        'median_fitness best_fitness \n')
        write_lines.append(headerstring)

    cbest, cmed, cratio = charbonneau_ratio(fitnesses)
    charbstring = str(cratio) + ' ' + str(cmed) + ' ' + str(cbest)

    med_genvar = str(np.median(genvar))
    genvarline = str(gcount) + ' ' + med_genvar + ' ' + charbstring + '\n'
    write_lines.append(genvarline)

    with open(txtfile, 'a') as the_file:
        for aline in write_lines:
            the_file.write(aline)

def Gamma_Edd_check(model, param_names):
    """
    Check whether the Eddinton limit is exceeded.
    We do not compute mR (mean atomic mass) in detail, but assume a
    conservative value (giving the lowest Gamma_Edd) of 2.2
    For Helium we use standard a low value, as FW computes GAMMA 
    not based on the input HE but based on the starting model HE
    If model computation is allowed, return True, else, return False
    """

    # If 'teff' and 'logg' are the free parameters, compute gamma
    if ('teff' in param_names) and ('logg' in param_names):
        teff_idx = param_names.index('teff')
        logg_idx = param_names.index('logg')
        teff = float(model[teff_idx])
        logg = float(model[logg_idx])
        yhe = 0.08

    # If not, compute the model
    else:
        return True

    sigmaB = 5.6704e-5
    speed_light = 2.997925e10
    amh = 1.67352e-24
    sigmae = 6.65e-25/amh
    ggrav = 10**(logg)

    ihe_start = 1.0 # Lowest value for OB stars (O stars = 2, B stars = 1)
    mu = 2.2 # Mean atomic mass. Use a rather high value to be safe

    c2 = (1.0 + ihe_start*yhe)/mu
    sigem = sigmae * c2

    gamma = sigem * (sigmaB/speed_light) * teff**4 /ggrav

    print('gamma = ', gamma, 'teff =', teff, 'logg = ', logg, 'yhe =', yhe)

    gamma_cutoff = 1.00
    # Compute models that are possibly meeting the FW eddington criterion
    if gamma < gamma_cutoff:
        return True
    # Do not compute models that are certainly not meeting that criterion
    else:
        return False


def identify_duplicate(dupfile, individual):
    """
    Look into the text file with all models to check whether a model
    was already calculated.

    #FIXME #LOWPRIORITY Can reading and searching the file be done
    faster? (low priority because it is now really "immediately"
    compared to the speed at which a FW model is computed...)
    Tested: using grep called by python. This was way slower.
    """

    param_string = ''
    for param in individual:
        param_string = param_string + str(param) + ' '
    param_string = param_string[:-1]

    f = open(dupfile)
    allmods = f.read()

    if param_string in allmods:
        duplicate = True
    else:
        duplicate = False

    return duplicate

def find_nearest(array, value):
    """ Give value of array that is closest to the input value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def rand_from_range(themin, themax, thestep, rounding):
    """ Generate a random number between themin and themax,
    with a stepsize thestep """

    paramarray = np.arange(themin, themax, thestep)
    randval = random.uniform(themin,themax)
    randparam = find_nearest(paramarray, randval)
    randparam = round(randparam, int(rounding))

    return randparam

def init_pop(nindiv, params, param_names, dupfile):
    """ Generate the parameters for the initial population

    Input:
    - nindiv: number of individuals
    - params: array with bounds for each parameter.
      [minimum value, maximum value, step size]

    Output is a list of nindiv sets of model parameters
    """

    os.system("touch " + dupfile)

    the_init_pop = []
    while len(the_init_pop) < nindiv:
        params_onemod = []
        for pb in params:
            paramval = rand_from_range(*pb)
            params_onemod.append(paramval)
        if ((not identify_duplicate(dupfile, params_onemod)) and
            Gamma_Edd_check(params_onemod, param_names)):
            the_init_pop.append(params_onemod)
            store_models(dupfile, params_onemod)
        
    return np.array(the_init_pop)

def crossover(mother_genes, father_genes, clone_fraction):
    """Generate new indiviuals based on two sets of genes.
    """

    if random.random() < clone_fraction:
        babygirl_genes = mother_genes
        babyboy_genes = father_genes
    else:
        babygirl_genes = []
        babyboy_genes = []
        for gene_mama, gene_papa in zip(mother_genes, father_genes):
            if np.random.choice(2, 1)[0] == 1:
                babygirl_genes.append(gene_mama)
                babyboy_genes.append(gene_papa)
            else:
                babygirl_genes.append(gene_papa)
                babyboy_genes.append(gene_mama)


    return babygirl_genes, babyboy_genes

def gaussian_mutation(baby_genes, paramspace, mutation_rate, gwidth,
        gbase, gtype, double_yn):
    """ Changes (with a certain probability) the value of parameters,
    hereby following a gaussian distribution around the current value
    of the parameter that will mutate.

    Input are the parameters of an individual, then each
    parameter has a chance of mutation_rate to mutate, with a
    gaussian with a certain width. The width is specified either in
    terms of a fraction of the parameter space width (then determined
    for each parameter), or in terms of steps, so depending on the grid
    of each parameter ('gtype').

    Output is the mutated genome (parameters of the individual).
    """

    mutated_genes = []

    # Loop through all genes (parameters) of the model
    for i in range(len(paramspace)):

        # A mutation only occurs in a fraction (mutation_rate) of
        # the genes.
        if random.random() < mutation_rate:
            the_p_min = paramspace[i][0]
            the_p_max = paramspace[i][1]
            the_p_step = paramspace[i][2]
            the_p_rounding = int(paramspace[i][3])

            nsteps = int(round((the_p_max-the_p_min)/the_p_step+1,0))
            param_space = np.linspace(the_p_min, the_p_max, nsteps)
            param_space = param_space[param_space != baby_genes[i]]
            if gtype == 'frac':
                gauss_width = (the_p_max-the_p_min)*gwidth
            else:
                # If not 'frac', this means: gtype == 'step'
                gauss_width = the_p_step*gwidth
            if double_yn == 'yes':
                props = double_gauss(param_space, gbase, 1.,
                    baby_genes[i], gauss_width)
            else:
                props = gauss(param_space, gbase, 1., baby_genes[i],
                    gauss_width)
            props = props / np.sum(props)

            mutated_gene = np.random.choice(param_space, 1, p=props)[0]

            # The rounding is mainly done for the mass loss, which
            # has a different input format in FW than it has in the
            # parameter space (there it is in 10log)
            mutated_gene = round(mutated_gene, the_p_rounding)

            # This is a check, but actually the code should never
            # go here and always append the mutated genes?!
            # I added a print statement to track whether this happens,
            # if not then this part can be removed.  #FIXME
            if not (mutated_gene < the_p_min or mutated_gene > the_p_max):
                mutated_genes.append(mutated_gene)
            else:
                print('ERROR in gaussian_mutataion: outside range')
                mutated_genes.append(baby_genes[i])
        else:
            mutated_genes.append(baby_genes[i])

    return mutated_genes

def genes2str(the_genes, the_pars, add_sig):

    genestring = ''
    for i in range(len(the_pars)):
        # Read input from parameter space
        pstart = the_pars[i][0]
        pstop = the_pars[i][1]
        pstep = the_pars[i][2]

        # Assess significant digits.
        sig_digits = np.ceil(np.log10(abs(float(pstop) - (float(pstart)))) -
            np.floor(np.log10(float(pstep))))
        sig_digits = int(sig_digits) + add_sig

        # Convert parameter value to string value
        gstr = str(int((np.abs(the_genes[i] - pstart)/np.abs(pstop - pstart))*(10**sig_digits - 1))).zfill(sig_digits)

        # Append to gene string of this individual
        genestring = genestring + gstr

    return genestring

def str2genes(the_str, the_pars, add_sig):

    the_genes = []

    for i in range(len(the_pars)):

        # Read input from parameter space
        pstart = the_pars[i][0]
        pstop = the_pars[i][1]
        pstep = the_pars[i][2]
        pround = the_pars[i][3]

        # Assess significant digits.
        sig_digits = np.ceil(np.log10(abs(float(pstop) - (float(pstart)))) -
            np.floor(np.log10(float(pstep))))
        sig_digits = int(sig_digits) + add_sig

        # Crop the relevant number of digits from the string
        print('the_str (i=' + str(i) + ')' , the_str)
        print('sig_digits ' , sig_digits)
        par_string = the_str[:sig_digits]
        the_str = the_str[sig_digits:]

        # Convert to parameter value and round acc to stepsize in input
        real_val = (float(par_string)/(10**sig_digits - 1))*np.abs(pstop -
            pstart) + pstart
        paramarray = np.arange(pstart, pstop, pstep)
        round_real = find_nearest(paramarray, real_val)
        round_real = round(round_real, int(pround))

        # Append to array of parameters for this individual
        the_genes.append(round_real)

    return the_genes

def crossover_strings(mother_str, father_str, clonefrac, pfrac):

    strlen = len(mother_str)

    if random.random() < clonefrac:
        babygirl = mother_str
        babyboy = father_str
    else:
        if random.random() > pfrac:
            cutidx = np.random.choice(strlen)
            babygirl = mother_str[:cutidx] + father_str[cutidx:]
            babyboy = father_str[:cutidx] + mother_str[cutidx:]
        else:
            randidx1, randidx2 =  np.random.choice(strlen, 2, replace=False)
            cutidx1 = min(randidx1, randidx2)
            cutidx2 = max(randidx1, randidx2)
            babygirl = (mother_str[:cutidx1] + father_str[cutidx1:cutidx2] +
                mother_str[cutidx2:])
            babyboy = (father_str[:cutidx1] + mother_str[cutidx1:cutidx2] +
                father_str[cutidx2:])
    return babygirl, babyboy

def mutation_random_string(gene_string, mutrate):

    strlen = len(gene_string)
    mutidx = np.where(np.random.random(strlen) < mutrate)[0]
    split_genestring = np.array(list(gene_string))
    split_genestring[mutidx] = np.random.choice(10, len(mutidx)).astype(str)
    mutated_genestring = ''.join(str(x) for x in split_genestring)

    return mutated_genestring

def mutation_creep_string(gene_string, mutrate):

    strlen = len(gene_string)
    mutidx = np.where(np.random.random(strlen) < mutrate)[0]
    split_genestring = np.array(list(gene_string))
    creep = (split_genestring[mutidx].astype(int) +
        np.random.choice([-1, 1], len(mutidx))) % 10
    split_genestring[mutidx] = creep.astype(str)
    mutated_genestring = ''.join(str(x) for x in split_genestring)

    return mutated_genestring

def reproduce(pop_orig, fitm, mutation_rate, clone_fraction, paramspace,
    param_names, dupfile, gauss_w_na, gauss_w_br,
    gauss_b_na, gauss_b_br, mut_rate_na, n_ind, na_type, br_type, dgauss,
    use_string, add_sigs, frac_double):
    """Given a population of individuals and a measure for their
    fitness, generate a new generation of individuals.

    IMPORTANT: models are ranked based on their so-called fitness
    measrue <fitm>. This value can be the total chi2 or some other
    measure, but:
       >> make sure that a lower value corresponds to a fitter model!
    With the current implementation (models are ranked and then
    weighted according to their ranking, but not weighted directly
    by their fitness measure) the absolute difference between the
    values of the fitness measure does not matter, but in an
    approach that uses the fitness directly for weight, it will.

    """

    add_sigs = int(add_sigs)
    frac_double = float(frac_double)

    # Rank the individuals according to their fitness
    order = np.argsort(fitm)
    rank = np.argsort(order)

    # Assign a reproduction probability to each indiv. using their rank
    pop_len = len(pop_orig)
    repro_prop = pop_len - rank
    repro_prop = 1.0*repro_prop / np.sum(repro_prop)

    pop_new = []

    dupcount = 0
    while len(pop_new) < n_ind:

        # Pick two random parents and look up their genes
        mother_idx = np.random.choice(pop_len, 1, p=repro_prop)[0]
        father_idx = np.random.choice(pop_len, 1, p=repro_prop)[0]
        mother_genes = pop_orig[mother_idx]
        father_genes = pop_orig[father_idx]

        # Option to use crossover and reproduction as in Charbonneau+95,
        # Using strings of numbers representing the parameters.
        if use_string in ('yes', 'y', 'Yes', 'True', True):
            # Convert genes to strings
            mother_str = genes2str(mother_genes, paramspace, add_sigs)
            father_str = genes2str(father_genes, paramspace, add_sigs)

            # Parent genomes produce two baby genomes
            baby_genes1, baby_genes2 = crossover_strings(mother_str, father_str,
                clone_fraction, frac_double)

            # Mutation
            baby_genes1 = mutation_random_string(baby_genes1, mutation_rate)
            baby_genes2 = mutation_random_string(baby_genes2, mutation_rate)

            baby_genes1 = mutation_creep_string(baby_genes1, mutation_rate)
            baby_genes2 = mutation_creep_string(baby_genes2, mutation_rate)

            # Convert strings back to genes
            baby_genes1 = str2genes(baby_genes1, paramspace, add_sigs)
            baby_genes2 = str2genes(baby_genes2, paramspace, add_sigs)

        # Recombination and mutation as described in Brands+in prep.
        else:
            # Parent genomes produce two baby genomes
            baby_genes1, baby_genes2 = crossover(mother_genes, father_genes,
                clone_fraction)

            # Mutate the baby genomes. There are two modes of mutation.
            # Load values defining the distributions for the two types.
            gauss_w_na = float(gauss_w_na)
            gauss_w_br = float(gauss_w_br)
            gauss_b_na = float(gauss_b_na)
            gauss_b_br = float(gauss_b_br)
            mut_rate_na = float(mut_rate_na)

            # Narrow mutation: close to original value, high mutation
            # rate that is in principle fixed
            baby_genes1 = gaussian_mutation(baby_genes1, paramspace,
                mut_rate_na, gauss_w_na, gauss_b_na, na_type, double_yn='no')
            baby_genes2 = gaussian_mutation(baby_genes2, paramspace,
                mut_rate_na, gauss_w_na, gauss_b_na, na_type, double_yn='no')

            # Broad mutation: further away from original value, lower
            # mutation rate that is variable
            baby_genes1 = gaussian_mutation(baby_genes1, paramspace,
                mutation_rate, gauss_w_br, gauss_b_br, br_type, double_yn=dgauss)
            baby_genes2 = gaussian_mutation(baby_genes2, paramspace,
                mutation_rate, gauss_w_br, gauss_b_br, br_type, double_yn=dgauss)

        dup_tf = identify_duplicate(dupfile, baby_genes1)
        if (not dup_tf) and Gamma_Edd_check(baby_genes1, param_names):
            pop_new.append(baby_genes1)
            store_models(dupfile, baby_genes1)

            # In addition to the while statement:
            # Loops breaks half way if the pop len is already
            # reached after one out of two individuals is appended.
            if len(pop_new) == n_ind:
                break
        else:
            dupcount = dupcount + 1

        dup_tf = identify_duplicate(dupfile, baby_genes2)
        if (not dup_tf) and Gamma_Edd_check(baby_genes2, param_names):
            pop_new.append(baby_genes2)
            store_models(dupfile, baby_genes2)
        else:
            dupcount = dupcount + 1

    print("DUPLICATE COUNT GEN: " + str(dupcount))

    return pop_new

def reincarnate(population, chi_pop, previous_best, chi2_prevbest):
    """ Replace worst fitting individual from generation with the best
    fitting individual of the previous generation.

    (This should happen before the reproduction takes place.)

    Principle: the overall fittest individual of the run should always
    be present in each generation before it reproduces.
    """

    # The population is only altered when the current population does
    # not contain a fitter individual than the previous one.
    if chi2_prevbest < min(chi_pop):

        # Rank the individuals according to their fitness
        order = np.argsort(chi_pop)
        rank = np.argsort(order)

        # Select least fit model, to be kicked out of population
        least_fit_idx = np.argmax(rank)

        # Replace least fit model by fittest model of prev. generation
        population[least_fit_idx] = previous_best
        chi_pop[least_fit_idx] = chi2_prevbest

    return population, chi_pop

def get_fittest(population, chi_pop):
    """Find the fittest individual in the population."""

    # Rank the individuals according to their fitness
    order = np.argsort(chi_pop)
    rank = np.argsort(order)
    least_fit_idx = np.argmin(rank)

    # Select fittest model
    best_params = population[least_fit_idx]
    best_chi = chi_pop[least_fit_idx]

    return best_params, best_chi

def get_top_x_fittest(population, chi_pop, topx):
    """Of a population, return the topx fittest individuals.
    Used if one works with a larger first generation compared to
    the rest of the generations.
    """

    # Rank the individuals according to their fitness
    order = np.argsort(chi_pop)
    rank = np.argsort(order)
    chi_pop = np.array(chi_pop)
    population = np.array(population)

    # Select fittest topx models
    best_chi2s = chi_pop[rank < topx]
    best_population = population[rank < topx]

    return best_population, best_chi2s

def adjust_mutation_rate_charbonneau(old_rate, chi2, mut_rate_factor,
    mut_rate_min, mut_rate_max, fit_cutoff_min, fit_cutoff_max):
    """Adjust the mutation rate based on the typical fitness
    in a population of individuals, as is suggested in
    Charbonneau (1995)"""

    # A large ratio means that the difference between the 'typical
    # model' in a generation, and the fittest model is large. In this
    # case the mutation rate is decreased, so that the genome of the
    # fitter models is better preserved, so that the fit can improve.
    # If the ratio is low, a (local) minimum has apparently been
    # explored well, and no improvements are there to be reached. The
    # mutation rate is then increased so that less well explored parts
    # of the parameter space will be probed.

    cbest, cmod, ratio = charbonneau_ratio(chi2)

    if ratio <= fit_cutoff_min:
        new_rate = min(mut_rate_max, old_rate*mut_rate_factor)
    elif ratio >= fit_cutoff_max:
        new_rate = max(mut_rate_min, old_rate/mut_rate_factor)
    else:
        new_rate = old_rate

    return new_rate

def auto_charbonneau_limits(dct, charbini):
    ''' Given the initial Charbonneau ratio of the run, compute
        reasonable values for the final Charbonnau ratio '''

    ac_fit_a = float(dct['ac_fit_a'])
    ac_fit_b = float(dct['ac_fit_b'])
    ac_lowerlim = float(dct['ac_lowerlim'])
    ac_upperlim = float(dct['ac_upperlim'])
    ac_max_factor = float(dct['ac_max_factor'])

    charblim_min = ac_fit_a + charbini*ac_fit_b

    if charblim_min < ac_lowerlim:
        charblim_min = ac_lowerlim
    elif charblim_min > ac_upperlim:
        charblim_min = ac_upperlim

    charblim_max = ac_max_factor*charblim_min

    charblim_min = round(charblim_min,3)
    charblim_max = round(charblim_max,3)

    dct['fit_cutoff_min_charb'] = charblim_min
    dct['fit_cutoff_max_charb'] = charblim_max

    return dct

def autoadjust_charbonneau(dct_ctrl, dct_files, gencount):
    ''' This function adjusts the charbonneau limits (so overrides
    the ones given in the control file) after the first generation
    by using the charbonneau ratio of the first generation as an
    indication for reasonable limits. This is because while ideally
    the ratio reflects only the convergence of the run, in practice
    it also depends on the spectrum that you fit. This has to be
    taken into account when using the limits.
    If after ac_maxgen generations the charbonneau lower limit is
    still not reached, then automatically increase mutation rate.
    '''
    with open(dct_files["genvar_out"]) as f:
        content = f.readlines()

    charbini_run = float(content[1].strip().split()[2])

    dct_ctrl = auto_charbonneau_limits(dct_ctrl, charbini_run)

    if gencount > float(dct_ctrl['ac_maxgen']):
        charb_list = np.genfromtxt(dct_files["genvar_out"]).T[2]
        min_charb_run = np.min(charb_list)
        last_charb_run = charb_list[-1]

        if min_charb_run > float(dct_ctrl['fit_cutoff_min_charb']):
            dct_ctrl['fit_cutoff_min_charb'] = round(last_charb_run +
                float(dct_ctrl['ac_maxgen_min']),3)
            dct_ctrl['fit_cutoff_max_charb'] = round(last_charb_run +
                float(dct_ctrl['ac_maxgen_max']),3)

    return dct_ctrl

def assess_variation(fullgeneration, paramspace, fittest_ind):
    """Look at how many 'steps' each parameter differs from the
    best fitting model. This is a measure for genetic variety
    of the generation

    ***** Function no longer in use *****
    The problem with this method was that the genetic variety is depending
    strongly on the stepsize that you allow for the different parameters.
    There are solutions to this, but they are not implemented.
    """

    fullgeneration = np.array(fullgeneration)
    fittest_ind = np.array(fittest_ind)

    param_steps = np.array(paramspace).T[2]
    diffs_per_gene = np.abs(fullgeneration - fittest_ind) / param_steps
    diff_per_ind = np.sum(diffs_per_gene, axis=1)

    return diff_per_ind

def adjust_mutation_genvariety(mutrate, cutdecr, cutincr, mutfactor,
    mutmin, mutmax, mean_gen_var, parspace):
    """ Adjust the mutation rate, depending on genetic genetic
    variety of the generation.

    ***** Function no longer in use *****
    The problem with this method was that the genetic variety is depending
    strongly on the stepsize that you allow for the different parameters.
    There are solutions to this, but they are not implemented.
    """

    if mean_gen_var > cutdecr *len(parspace):
        mutrate = mutrate/mutfactor
    elif mean_gen_var < cutincr*len(parspace):
        mutrate = mutrate*mutfactor

    if mutrate < mutmin:
        mutrate = mutmin
    if mutrate > mutmax:
        mutrate = mutmax

    return mutrate

def print_report(gennumber, bestfitness, medianfitness, verbose):
    if verbose:
        print('================================================')
        print('Generation       ' + str(gennumber))
        print('Best fitness     ' + str(bestfitness))
        print('Median fitness   ' + str(medianfitness))
