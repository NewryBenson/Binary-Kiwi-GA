# Paths to input and output parent directory during a GA run
# Do not change
inputdir = 'input/'
outputdir = 'output/'

# Paths to input (datapath) and output (outpath) neccesary for
# running the GA_analysis.py script.
# - Your run output should be in a directory inside datapath_analysis
# - Output of the script will be put in a direcotry insdie
#   outpath_analysis, that will be made by the script
outpath_analysis = '/Dir/where/analysis/pdfs/will/be/stored/'
datapath_analysis = '/Dir/where/run/output/is/stored/'
fastwind_local = 'Dir/from/which/to/run/fastwind/'

if not inputdir.endswith('/'):
    inputdir = inputdir + '/'
if not outputdir.endswith('/'):
     outputdir = outputdir + '/'
if not outpath_analysis.endswith('/'):
     outpath_analysis = outpath_analysis + '/'
if not datapath_analysis.endswith('/'):
     datapath_analysis = datapath_analysis + '/'
if not fastwind_local.endswith('/'):
     fastwind_local = fastwind_local + '/'
