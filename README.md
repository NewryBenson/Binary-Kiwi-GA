# pyEA
Genetic/Evolutionary Algorithm in python for fitting Fastwind models

== Very brief instructions for usage ==

Directory FW_inicalc:
- Tested with FASTWIND version: V10.3.1dev (not in repository). 
- Compile FW elsewhere, then copy everything in the inicalc directory 
   to FW_inicalc in the folder pyEA.

Running pyEA:
- The input should be put in a directory 'input/runname/' 
- There are 6 input files, examples are in the directory example_input. 
  Most are self explanatory, except control.txt that contains all the 
  meta parameters - a description/manual will follow.
- After preparing the input files, you can do a check by running 
  “python pre_run_check.py runname" a report will then be printed in text, 
  and a plot with pdfs will be in the input folder of the source (also a 
  txt file that contains the printed text). Running this right before you 
  start the run prevents the run goes bad by silly mistakes in the input 
  files (visually check the output pdf!).
- The pre_run_check also generates a job script, but you should tailor 
this ot the cluster on which you are running the code. A jobscript for 
cartesius.surfsara can be found in the main directory. 
- The only "input" that should be set in the job script rather than one 
of the input files, is whether you start a new run, or restart an old one. 

A more detailed description of the code will follow in Brands in prep.: 
"The R136 star cluster dissected with Hubble Space Telescope/STIS. 
  The wind properties of the most massive stars in the core of R136"

The code contains elements of the Pikaia algorithm of Charbonneau+ 1995, 
but in the basis is different.
