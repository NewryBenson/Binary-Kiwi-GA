"""
=====================   Meta parameters of GA   ======================

Parameters that control population size and generations
-   nind: number of individuals per generation (int)
-   ngen: number of generations (int)
-   f_gen1: factor with which nind is increased for the 1st gen
    (int and >= 1)

Parameters that control cross over behavior
-   clone_fraction: fraction of the models that is cloned during the
    reproduction (before they undergo mutation)
    (0.0 => x >= 1.0)

Parameters that control mutation behavior
-   mut_rate_init: initial mutation rate
    (mut_rate_min => x >= mut_rate_max)
-   mut_rate_factor: factor by which mutation rate is multiplied if
    certain conditions are met (0.0 => x >= 1.0)
-   mut_rate_min: minimum allowed mutation rate (0.0 => x >= 1.0)
-   mut_rate_max: maximum allowed mutation rate (0.0 => x >= 1.0)
-   fit_ratio_cutoff: criterion for when the mutation rate is
    increased or decreased. (0.0 => x >= 1.0) A higher value will
    result in mutation to be increased with lower convergence.
    This can be useful when the code gets stuck in a local minimum.
    Decreasing the value can be useful if one wants more accurate
    evaluation of models around the current minimum. The optimal
    value is a balance between the two and depends on the function
    that is fitted (i.e. chance of getting stuck in local minimum)

# #FIXME This file could be made into an input file rather than
# a module that is imported (and then made into a dictionary?)

======================================================================
"""

be_verbose = True

# Specify whether the fitness of a model is based on the total
# chi squared value, or on the fitness.
# - 'chi2'
# - 'fitness'
# When using 'chi2', lines with more data points carry more weight,
# this is not the case with the option 'fitness': all lines will
# then carry equal weight unless
fitmeasure = 'fitness'

nind = 1 #100
ngen = 1
f_gen1 = 1

#######################################################################
######################  Directories & file paths ######################
#######################################################################

# Paths to input and output parent directory
inputdir = 'input/'
outputdir = 'output/'

# Paths to FW inicalc and output directory
inicalcdir = 'inicalc/'
modelatom = 'A10HHeNCOPSi'

# Maximum time that a FW model can take
# #FIXME: can non-converged FW models really not be used by pformal?
fw_timeout = '40m'

if not inputdir.endswith('/'):
    inputdir = inputdir + '/'
if not outputdir.endswith('/'):
    outputdir = outputdir + '/'

#####################################################################
######################   MUTATION PARAMETERS   ######################
#####################################################################

clone_fraction = 0.00
mut_rate_init = 0.05

# 'constant' - mutation rate  stays like mut_rat_init the whole run
# 'doerr' - Use a highered and lowered mutation rate
# 'carbonneau' - mutation rate adapted based on fitness variety
# 'genvariety' - mutation rate adapted based on genetic variety
mut_adjust_type = 'genvariety'

# Extra mutation parameters for 'doerr'
doerr_factor = 1.5

# Extra mutation parameters for 'carbonneau' and 'genvariety'
mut_rate_min = 0.001
mut_rate_max = 0.25
mut_rate_factor = 1.5

# Extra mutation parameters for 'carbonneau'
fit_cuttof_min_carb = 0.60
fit_cuttof_max_carb = 0.90

# Extra mutation parameters for 'genvariety'
cuttof_increase_genv = 1.0
cuttof_decrease_genv = 3.0
