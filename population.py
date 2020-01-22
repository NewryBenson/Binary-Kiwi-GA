import os
import random
import numpy as np

import subprocess
import time

######################################################################
# Functions that control the parameters of the population
######################################################################

def gauss(x,a,x0,sigma):
    """
    Gaussian with continuum at 0.
    """
    y = a * np.exp(-(x-x0)**2 / (2*sigma**2))
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
    """ Write the paramters and fitness of each individual in the
    population into a textfile.
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

def init_pop(nindiv, params, dupfile):
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
        if not identify_duplicate(dupfile, params_onemod):
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

def stepsize_mutation(baby_genes, paramspace, mutation_rate):
    """ Creep mutation changes a gene by one stepsize """

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

            # Either increase or decrease the parameter by one step,
            # if this does not bring the model outside of the
            # parameter space
            sign = np.random.choice([-1,1], 1)[0]
            mutated_gene = baby_genes[i] + sign*the_p_step
            mutated_gene = round(mutated_gene, the_p_rounding)

            if not (mutated_gene < the_p_min or mutated_gene > the_p_max):
                mutated_genes.append(mutated_gene)
            else:
                mutated_genes.append(baby_genes[i])
        else:
            mutated_genes.append(baby_genes[i])

    return mutated_genes

def gaussian_mutation(baby_genes, paramspace, mutation_rate, gwidth_steps=3.0):
    """ Mutation by x step sizes, where the chance is weighted
    by 1./x, so that larger changes are less likely. """

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

            nsteps = (the_p_max-the_p_min)/the_p_step+1
            param_space = np.linspace(the_p_min, the_p_max, nsteps)
            param_space = param_space[param_space != baby_genes[i]]
            gauss_width = the_p_step*gwidth_steps
            props = gauss(param_space, 1, baby_genes[i], gauss_width)
            props = props / np.sum(props)

            mutated_gene = np.random.choice(param_space, 1, p=props)[0]
            mutated_gene = round(mutated_gene, the_p_rounding)

            if not (mutated_gene < the_p_min or mutated_gene > the_p_max):
                mutated_genes.append(mutated_gene)
            else:
                mutated_genes.append(baby_genes[i])
        else:
            mutated_genes.append(baby_genes[i])

    return mutated_genes

def uniform_mutation(baby_genes, paramspace, mutation_rate):
    """Mutates a fraction of the genes uniformly, i.e. a genes value is
    replaced by a random other value in the parameter range
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

            new_gene = rand_from_range(the_p_min, the_p_max, the_p_step,
                the_p_rounding)
            mutated_genes.append(new_gene)
        else:
            mutated_genes.append(baby_genes[i])

    return mutated_genes

def reproduce(pop_orig, fitm, mutation_rate, clone_fraction, paramspace,
    dupfile):
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

    # Rank the individuals according to their fitness
    order = np.argsort(fitm)
    rank = np.argsort(order)

    # Assign a reproduction probability to each indiv. using their rank
    pop_len = len(pop_orig)
    repro_prop = pop_len - rank
    repro_prop = 1.0*repro_prop / np.sum(repro_prop)

    pop_new = []

    dupcount = 0
    while len(pop_new) < pop_len:

        # Pick two random parents and look up their genes
        mother_idx = np.random.choice(pop_len, 1, p=repro_prop)[0]
        father_idx = np.random.choice(pop_len, 1, p=repro_prop)[0]
        mother_genes = pop_orig[mother_idx]
        father_genes = pop_orig[father_idx]

        # Parent genomes produce two baby genomes
        baby_genes1, baby_genes2 = crossover(mother_genes, father_genes,
            clone_fraction)

        # Mutate the baby genomes. There are two modes of mutation,
        # creep mutation and uniform mutataion, each genome is affected
        # by both. The probability that each gene mutates is equal to
        # the mutation_rate, this hold for both modes.
        # The probability that one none of the genes of a individual
        # mutate if it is affected by both these thus equal to
        # (1-mutation_rate)^(2*number of genes).

        # #FIXME 'uniform_factor' should be a variable in control
        # 'uniform_factor' is the factor with which the mutation rate
        # is lowered for so called uniform mutation. This kind of
        # mutation is quite agressive therefore you don't want it
        # to happen to often. (maybe not at all?)
        uniform_factor = 0.3
        mutation_rate_u = mutation_rate * uniform_factor

        baby_genes1 = uniform_mutation(baby_genes1, paramspace, mutation_rate_u)
        baby_genes2 = uniform_mutation(baby_genes2, paramspace, mutation_rate_u)

        baby_genes1 = stepsize_mutation(baby_genes1, paramspace, mutation_rate)
        baby_genes2 = stepsize_mutation(baby_genes2, paramspace, mutation_rate)

        baby_genes1 = gaussian_mutation(baby_genes1, paramspace, mutation_rate)
        baby_genes2 = gaussian_mutation(baby_genes2, paramspace, mutation_rate)

        dup_tf = identify_duplicate(dupfile, baby_genes1)
        if not dup_tf:
            pop_new.append(baby_genes1)
            store_models(dupfile, baby_genes1)

            # In addition to the while statement:
            # Loops breaks half way if the pop len is already
            # reached after one out of two individuals is appended.
            if len(pop_new) == pop_len:
                break
        else:
            dupcount = dupcount + 1

        dup_tf = identify_duplicate(dupfile, baby_genes2)
        if not dup_tf:
            pop_new.append(baby_genes2)
            store_models(dupfile, baby_genes2)
        else:
            dupcount = dupcount + 1

    be_verbose = False
    if be_verbose:
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

def adjust_mutation_rate_carbonneau(old_rate, chi2, mut_rate_factor,
    mut_rate_min, mut_rate_max, fit_cutoff_min, fit_cutoff_max):
    """Adjust the mutation rate based on the typical fitness
    in a population of individuals, as is suggested in
    Carbonneau."""

    best_mod = np.min(chi2)
    median_mod = np.median(chi2)

    # A large ratio means that the difference between the 'typical
    # model' in a generation, and the fittest model is large. In this
    # case the mutation rate is decreased, so that the genome of the
    # fitter models is better preserved, so that the fit can improve.
    # If the ratio is low, a (local) minimum has apparently been
    # explored well, and no improvements are there to be reached. The
    # mutation rate is then increased so that less well explored parts
    # of the parameter space will be probed.

    ratio = np.abs(best_mod - median_mod) / (best_mod + median_mod)

    if ratio <= fit_cutoff_min:
        new_rate = min(mut_rate_max, old_rate*mut_rate_factor)
    elif ratio >= fit_cutoff_max:
        new_rate = max(mut_rate_min, old_rate/mut_rate_factor)
    else:
        new_rate = old_rate

    return new_rate

def assess_variation(fullgeneration, paramspace, fittest_ind):
    """Look at how many 'steps' each parameter differs from the
    best fitting model. This is a measure for genetic variety
    of the generation """

    fullgeneration = np.array(fullgeneration)
    fittest_ind = np.array(fittest_ind)

    param_steps = np.array(paramspace).T[2]
    diffs_per_gene = np.abs(fullgeneration - fittest_ind) / param_steps
    diff_per_ind = np.sum(diffs_per_gene, axis=1)

    return diff_per_ind

def do_reproduce_doerr(mrate, doerfact, thegen, thefit, clfrac,
    pspace, theduplfile):
    """Split the genetarion into two pools, and apply on each
    of those pools different mutation rates, according to
    Doerr+2019."""

    num_ind = len(thegen)
    mutation_rate_a = mrate/doerfact
    mutation_rate_b = mrate*doerfact
    generation_a = reproduce(thegen, thefit, mutation_rate_a,
        clfrac, pspace, theduplfile)[:num_ind/2]
    generation_b = reproduce(thegen, thefit, mutation_rate_b,
        clfrac, pspace, theduplfile)[:num_ind-(num_ind/2)]

    return generation_a, generation_b, mutation_rate_a, mutation_rate_b

def adjust_mutation_doerr(fit_a, fit_b, mrate_a, mrate_b, minmut, maxmut):
    """ Adjust the mutation rate, depending on the fitness of the
    individuals of the two Doerr groups.
    Currently has no implementation in the main code.
    """

    if np.median(fit_a) < np.median(fit_b):
        mrate = mrate_a
        if mrate < minmut:
            mrate = minmut
    elif np.median(fit_a) > np.median(fit_b):
        mrate = mrate_b
        if mrate > maxmut:
            mrate = maxmut
    return mrate

def concat_gen_doerr(gen_a, gen_b, fit_a, fit_b):
    """Merge the subgenerations of the Doerr mutation rate
    into one."""

    thegen = np.concatenate((gen_a, gen_b))
    thefit = np.concatenate((fit_a, fit_b))

    return thegen, thefit


def adjust_mutation_genvariety(mutrate, cutdecr, cutincr, mutfactor,
    mutmin, mutmax, mean_gen_var, parspace):
    """ Adjust the mutation rate, depending on genetic genetic
    variety of the generation."""

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









#
