MODULE version_nlte
!-----------------------------------------------------------------------
!
!     program history
!
!     version  0.0  june 1995
!                   original version by quique
!
!     version  1.0  may 30th 1996
!                   new hydro, cmf, number of bugs corrected
!
!     version: 1.1  july 10th 1996
!                   modified for mixed cmf/sa ,
!                   outer boundary (sa-lines),
!                   density inversion, maximum corrections
!
!     version: 1.2  oct  11th 1996
!                   modified for dq'/dtau'-term in hydro
!                   and hopf-parameter input
!
!     version: 1.3  nov  13th 1996
!                   hydrogen collision rates new
!
!     version: 1.4  dec  4th 1996
!                   consistent treatment of p vs. rho including mu(r)
!                   storage of pressure in model-file
!                   check of consistencies of tnew and tact
!                   ground1 (iteration scheme)
!
!     version: 1.5  march 11th 1997
!                   changed location of 2nd radius point (wrs!)
!                   now allowed to start from converged hydro-models
!                   if itlast.ne.0
!
!                   !!! note that due to a slightly different updating
!                       of xlevel = (ni/nk)*, minor differences in
!                       the occupation numbers will arise if one
!                       compares the results of a calculation with
!                       and without an interupt!!!
!
!     version: 1.6  march 17th 1997
!                   z-integration for optically thick lines in cmf
!                   changed (z=0 point at p = rmax included).
!                   most important!!! consistent alo at r = rmax
!                   for optically thick lines developed and included.
!
!     version: 1.7  march 21st 1997
!                   first version of mixed sa/cmf approach according
!                   to the requirements specified in subroutine
!                   cmfprep. this possibility is used if the parameter
!                   optmixed in the input (the former parameter df,
!                   which has no meaning any longer) is set to unity.
!                   if all lines are to be treated in the cmf, optmixed
!                   must be zero.
!
!     version: 2.0  may 23rd 1997
!                   completely revised, can calculate now all atoms
!                   provided by detail.
!                   most important changes refer to photo-integrals,
!                   which are calculated by a completely new algorithm,
!                   and frequency meshes, which are now two!
!
!     version: 2.0.1 june 4th 1997
!
!                   qualifiers for problematic levels introduced
!                   occupation numbers and ionization set to zero
!                   where outside ilow, imax
!
!     version: 2.0.2 june 5th 1997
!
!                   rybicki scheme in continuum transport only after
!                   major changes of occ.no., else pure iteration of
!                   eddington factors
!
!
!     version: 2.0.3 june 17th 1997
!
!                   checked for conistency with old version.
!                   found two additional errors in old one:
!                   update of electron density for ni* (in netmat,
!                   coll. terms) forgotten. inconsistent treatment of
!                   gaunt-factor in opacity (all freq.) and photo-
!                   integrals (only at edge) leads to inappropriate alo.
!
!                   correction of bug in intermepho (wrong end condition
!                   in some cases)
!
!     version: 2.0.4 dec 4th 1997
!
!                   density inversions now allowed!
!                   only warning on output!
!
!
!     version: 2.0.5 jan 9th 1998
!                   copy of indat.dat into directory
!
!     version: 2.0.6 feb 10th 1998
!                   check for photospheric mass
!                   problems for too fast convergence removed
!
!     version: 2.1.0 feb 13th 1998
!                   polished, i4 compilation now possible
!
!     version: 2.2.0 march 9th 1998
!                   accuracy for matrix solution improved
!                   left out line is now that with largest occ.num.
!                   in sobo: inversion treated differently   
!
!     version: 2.2.1 nov 18th 1998
!                   yhe has no longer to be updated in the atomic
!                   data input file. abund(he), if existent, is
!                   overwritten by yhein(from indat.dat) in "start"
!
!     version: 3.0.0 jan 15th 1999  conversion to f90
!
!     version: 3.0.1 jan 20th 1999  inclusion of work by christian wiethaus
!                   micro-turbulence included
!                   mg i,ii (simple version) can be treated now
!
!     version: 3.0.2 feb 2nd 1999
!                   correction of  (very old) error in ray(=> cmfset)
!
!     version: 3.1 march 4th 1999  almost real f90 now
!
!     version: 3.2 may 6th 1999  line quantities for inverted levels
!                  resulting in total negative opacity (line + cont)   
!                  approximated in a VERY poor way to avoid difficulties
!                  in convergence
!
!     version: 4.0 feb 2000 including nlte_approx
!                  (Note: xmet added to input) 
!
!                  bug in control of gff(-> i) fixed.
!                  (note: no effect for all models calculated so far, 
!                   due to same charge of HII and HeI)                   
!
!     version: 4.1 may 2000 final version including approx. metal
!                  background. Valid for O/B stars.
!                  to rund old version, simply set xmet = 0.
!      
!     version: 4.2 june 7th 2000 catalog name (modnam) extended up
!                  to 25 characters
!
!     version: 4.2.1 oct 18th 2000 improvement of modnam extension (30 char)
!
!     version: 4.3 jan 24th 2001 minimum freq. in frescal/fresfin
!                  automatically calculated.
!                  Sobo_tranport modified with respect to calc. of bcIc  
!                  in cases of taus ne O(1)
!                  Test version of SA transport for inverted levels 
!
!     version: 5.0 feb 7th 2001 internal beta-Iteration in SA-Transport
!                  abolished
!                  All tables for U and U2 functions moved to DATA
!
!     version: 5.1 feb 20th 2001 Sobo_transport for inverted levels finalized
!                  dilfac1 modified 
!
!     version: 5.1.1 march 1st 2001 : Limits for FASTBCIC in TRATRAD changed!
!                  If convergence fails for some models with pure SOBO- &
!                  transport, check limits again.
!
!     version: 6.0 minor modifications to allow for inclusion of averaged
!                  metal line background  
!
!     version: 6.1 april 2nd 2001: finalized version for metallic line
!                  backgroud: input in INDAT.DAT modified, requires now
!                  also input of LINES and LINES_IN_MODEL (cf. docu.txt)  
!
!     version: 6.2 april 10th 2001: cmf-treatment of inverted levels
!                  now "exact", smaller corrections to allow for -nan comp.
!    
!     version: 6.2.1 april 23rd 2001: taumax for inversion changed,
!                  CMF-lines until 150000 A to include H5->6 transition
!
!     version: 6.3 april 24th 2001: inclusion of H- opacity, higher resolution
!              in photosphere possible if dimes.f90 changed (nd->67), 
!              meaning of first logical variable in INDAT.DAT changed:
!              OPTNEUPDATE = .TRUE., as usual
!              OPTNEUPDATE = .FALSE., use always LTE electron density (e.g.,
!                                     for high gravity A-stars)
!              Rayleigh scattering can be included if lines including
!              "rayleigh" are uncommented
!
!     version: 6.4 may 2nd 2001: can be used now to use detailled
!              H/He atomic model A10HHe: all required formulas implemented
!
!     version: 6.5 may 14th 2001: error in line transfer (update of
!               cont. source function) found and corrected
!
!     version: 6.6 may 18th 2001: freq. grid changed to resolve line-blocked
!               regions  
!
!     version: 6.7 june 21st 2001: Two Hopfparameter files, in dependence of xmet
!
!     version: 7.0 july 25th 2001: line overlap included/checked
!
!     version: 7.1  nov  7th 2001: log q in Hopfpara-files assumed
!
!     version: 7.1.1 april 18th 2002: subroutine hopfpara changed
!                   change of subroutine prepov, in order to allow
!                   for Sobolev lines inside overlap complex 
!
!     version: 7.1.2 april 26th 2002: INDAT.DAT used from cat
!
!     version: 7.2  july 3rd 2002: vdiv may adapt now in MODVL to
!                                  prevent inversion
!                   additional file FASTWIND.LOG
!
!     version: 7.2.1 march 2003, Tcorr almost finalized
!                    more metals (see below)
!
!     version: 7.2.2 april 2003, Tcorr finalized
!                    including line-blocking (approx.treatment)tcorrvar
!
!     version: 7.3   alternate path: modification for new nlte_approx
!                    october 10th 2002: xxk (2nd moment) calculated
!                    always (req. for eddf)
!                    different treatment for LINES_IN_MODEL = .FALSE.
!                    model calculated without background-lines, however
!                    final temperature with lines (allows for a comparsion
!                    with WM-basic); Restart capabilities not tested so far  
!                    used only by Jo.
!
!     version: 7.4   june 2003: Vers. 7.2.2 and 7.3 coupled
!     version: 8.0   july 1st 2003: most important metals now "exactly treated, &
!                    T-correction from corresponding transitions 
!                    needs nlte_approx Vers. 5.0, 
!                    (-> new atomic data file-path: ATOMDAT_NEW!)

!     version: 8.1   july 30th: small changes after testing for Bstars;
!                    most important changes: check for TAURLIM and increase
!                    if strong winds; imin, imax for Si.
!                    check also for TMIN1 in module tcorr_var!
!
!     version: 8.2   inclusion of approximate treatment of clumping
!                    different layout of INDAT.DAT required
!                    use nlte_approx version 5.1 and higher
!  
!                    NOTE1: ne, nh and occup. numbers include enhancement-factor,
!                          opacities are corrected
!                          (except FFOPA, which is needed only for T-structure)
!                          to account for spatial averages in RT calculations. 
!                          PAY ATTENTION if line processes are considered, &
!                          particularly in SA, and
!                          whenever local opacities are needed
!                    WARNING: since the energy balance is calculated inside
!                          the clumps (alternative argument: enhancement-factor
!                          cancels out), pay attention if adiabatic cooling is
!                          included, since then the rates MUST be corrected!
!                    NOTE2: not rigorously tested so far. Use at own risk!
!  
!     version: 8.3   changes in INTERMEPHO (calculate heating/cooling rates
!                    without ALO)
!                    XNE -> XNELTE whenever LTE values are used
!                    (to clarify programming)
!                    heating/cooling rates of explicit elements calculated
!                    with NEW occupation numbers (which is more consistent)
!                    START of Sobolev Treatment from it = 4 on!
!
!     version: 8.3.1 improvement of T-convergence, hundreds of tests
!
!     version: 8.3.2 May 2004 
!                    inclusion of DELPARA in calculation of hydro-structure
!                    to account for deviations from power-law in case of
!                    cool stars (Teff < 12000). Now the actual and used
!                    opacities should agree perfectly.  
!                    Note that in this case vdiv is NOT updated, &
!                    thus density inversions are possible. 
!                    Bug in logic concerning OPTNEUPDATE=.FALSE. and T-corr
!                    removed. Ne is now updated to actual LTE density.  
!                    New hack in the calculation of T(hydro). If Tact
!                    is still Tmin, dT is set to zero. This improves the
!                    consistency between Tact and Thydro significantly.
!                    The deviation between both (-> FASTWIND.LOG) should be
!                    now of order of 1%.  
!
!     version: 8.4   August 2004                    
!                    inclusion of formula 46 plus function prhcol
!                    (n, np, z, e), as referred to by
!                      Przybilla and Butler, A&A 2004.
!                    New formula 113 (old H-coll, cf. Burke et al. 1967), 
!                      corresponds to the data given in A10HHe if the
!                      modification as outlined by formula 13 (see below)
!                      should NOT be applied.
!                    This version is applicable to both the optical and the
!                    infrared provided that the correct DETAIL DATASET
!                       is taken; i.e., either
!                    A10HHe (old H-coll. data from Giovanardi et al., &
!                            subr. hcol, formula 13)
!                       or
!                    A11HHe (new H-coll. data from Butler, see above)  
!
!     version: 8.5   Nov. 2004
!                    change in calculation of dqdt partials for explicit
!                    elements to be consistent with nlte_approx.
!                    smaller changes in treatment of inverted levels
!
!                    changes in frescal and fresfin to resolve H and HeI
!                    resonance lines  
!
!     version: 8.5.1 April 2005: Bug in Tratcol (formula 113) identified
!
!     version: 8.5.2 Sept. 2005: Rel diff in cooling changed: at first
!                    differences > 1e-3 are checked, then differences > 1e-4 
!                    abundance changes for explicit and background elements
!                    now from INDAT.DAT (see comments around line 870 and docu.txt
!                    NEW solar abundances from file abundances_solar
!
!     version: 8.6   Oct. 2005: Careful tests concerning clumping performed;
!                    no programming error in original approach from
!                    version 8.2 detected.
!                    Input of clumping-parameters generalized:
!                    Variable number of parameters (up to ten) possible
!                    (e.g., if user changes stratification in subroutine clumping)
!                     parameters now contained in allocated field clf_field) 
!
!                    IMPORTANT: first parameter has to be REPRESENTATIVE
!                    clumping factor which will be ONLY used to find
!                    appropriate HOPF-parameters for calculating the
!                    T-structure in the start-model.
!
!                Correspondance with old approach:  
!
!                clf_field(1)=clf, clf_field(2)=vclstart, clf_field(3)=vclmax 
!
!     version: 8.6.1 Oct. 2005: distribution of radial points changed, 
!                    to better sample the transonic region (for IR!)
!                    xnh precision in GROUND1 changed, to allow for restart
!                    with different (explicit) metal abundances
!
!     version: 8.6.2 Feb. 2006: formula 17 - CBF (needed, e.g., for ground
!                    state ionization of NIII) changed. Note that
!                    there was a bug in the old version!
!
!                    precision of inversion of pure continuum rate eq.
!                    increased, in analogy to line case  
!
!                    pure continuum iteration only for H/He
!
!
!     version: 8.7 March 2006: inclusion of Miguel's version 8.5.2_alpha
!                    treatment of the OP rbf data improved: a new
!		     subroutine (op_setup) takes care of the initial
!		     set up of the OP data. Minor modifications to account for
!		     changes in the frequency grid (see LTE subroutine).
!                    It also checks for differences in the ionization energies 
!                    defined in the input DETAIL file and the OP energies, for
!                    consistency. 
!                    Bug detected and corrected in TRATCOL formula (26), so far 
!                    used only by CIII model atom.
!                    Bug in INTERMECOL formula (17) corrected (Achim),
!                    only affects NIII model atom, so far
!                    Integration limit DEX (subroutine INTERMEPHO) changed to
!                    allow for very detailed model atoms (mainly Norbert's neutral
!                    ions) that have levels with fairly similar energies.
!
!     version: 8.7.1 March 2006: smoothing of OP-DATA RBF cross-sections 
!
!     version: 9.0 March 2006: REQUIRES NEW nlte_type.f90
!                    new formualae for Adi's data set 
!                    completely new approach to calculate ILOW, IMAX
!                    new routines ilowimax_lte and ilowimax_nlte
!                    in case, update ONLY ILOWIMAX_NLTE (contained in nlte_approx.f90) 
!                    The old treatment (which is used also for z=0) is
!                    kept in routine ILOWIMAX_OLD (requires "old" detail files)
!                    Now, we make a division between
!                    ILOW, IMAX, IMIA, IMAA
!                    which refer to LTE values and define, e.g., the
!                    frequency grid, and
!                    ILOWNL, IMAXNL, IMIANL, IMAANL
!                    which refer to NLTE values and are used to solve the 
!                    rate equations. For H and He, both quantities are
!                    identical and calculated in OCCLTE.
!                    Dimensions of xl, partit and occnum corrected.
!
!                    Note: z=0 treatment not checked so far
!
!                    default value for nd (id_ndepth) is now 51. One point
!                    close to photosphere (just below) included, to allow
!                    for appropriate thermalization of background elements
!
!                    iterative improvement of solution of rate-equations
!                    calculated with residuum in qp precision 
!                    (possible with recent versions of intel compiler)
!
!     version: 9.1 August 2006: subr. ILOWIMAX_NLTE moved to nlte_approx in
!                    order to allow compilation from scratch (uses nlte_app) 
!                    VDIV allowed to INCREASE only (old "bug") 
!
!     version: 9.2 August 2006: subr. ILOWIMAX_NLTE moved to nlte_approx
!
!     version: 9.2.1 Nov. 2006 ensure convergence of oscillating cool/high gravity models.
!                    Call to TLUCY in MODVL (with e-scat only) only for first
!                    Hydro-Interation
!
!     version: 9.2.2 Dec. 2006: old bug in optsing = .false.
!                    (multi-line-approach) fixed
!
!     version: 9.2.3 Jan. 2007: improved calc. of dvdr
!
!     version: 9.2.4 March 2007: crossbf formula 101: read tabulated
!                    (OP-) photocross-sections provided by Norbert P.
!                    and convolve with Gaussian
!                    small changes in 'smooth' to account for this
!
!     version: 9.2.5 July 2007: photospheric grad written to GRAD.OUT
!
!     version: 9.2.6 October 2007: width of zones with resonances increased
!                    from 10 to 20 times emin (in subroutine smooth)
!                    to allow for correct treatment of SiII bf cross-sections
!
!     version: 9.3 Sept 2008: detail/kurucz models can be read in if
!                    OPTMOD = TRUE in the first iteration.
!                    Filename has to be Modelname+'.struct'
!                    New module photstruc  
!
!     version: 9.4 Feb 2009: tlusty models can be read in, in the same
!                    spirit as above. Here, we calculate P from rho instead
!                    rho from P. 
!                    Filename has to be Modelname+'.tlusty'
!                    module photstruc updated  
! 
!                    !!! NOTE: SO FAR, EXTERNAL PHOTOSPHERIC STRUCTURES
!                        ONLY FOR UNCLUMPED MODELS !!!!
!                    Warning: if you get the message
!                              ERROR IN R(TAU-LUCY=2/3), 
!                             then most probably Rstar is too small!
!
!     version: 9.5 June 2009: lots of smaller changes to allow a consistent
!                    treatment for explicit elements with MORE than three ions 
!                    in particular, all occupation numbers outside the local
!                    values of ILOWNL, IMAXNL are set to zero. Only at restart
!                    or Temp-update, these numbers are set to LTE*1.D-10, to
!                    allow for calculation of meaningful bf/ff opacities. 
!
!                    NOTE: NEW parameter in subroutine ILOWIMAX_NLTE (package
!                    nlte_approx), to control globally constant or locally
!                    consistent values of ILOWNL, IMAXNL. Default is global.
!
!     version: 9.6 July 2009: inclusion of dielectronic recombination for
!                    explicit ions following the method by Mihalas, Hummer etc. 
!                    RBF formula 21: PhotoData described by smooth cross-sections
!                    in Seaton parameterization, resonances described as
!                    stabilizing lines from the double excited configuration 
!                    (fromulated with respect to the ground-state of the next higher ion)
!
!     version: 9.7 July 2009: new formulas 60...65 in tratcol (for NIII)
!                    interpolation of OP-data where zero cross-section  
!                    Hack in subroutine THERMAL BALANCE (to avoid stop because
!                    of inaccurate flux-integration - done in Sept. 2009) 
!
!     version: 9.8 deprecated
!
!     version: 9.9 Sept. 2009: all global options put into module nlte_opt
!                    needs nlte_type v1.2
!
!                    inclusion from work done in March 2009:
!                    small changes required to use
!                    nlte_approx v7.0 and higher (photospheric line transfer)
!                    weighting scheme routine tempcorr slightly changed, 
!                    towards larger influence of dT(flux-correction)
!
!     version: 10.0 Oct. 2009: update of photo-structure
!                    new module RUN_ONCE with new variables
!                    (replacing START and FIRST in the routines
!                    WEIGH11, CMFSING, CMFMULTI (nlte.f90) and
!                    jbar_metals, cmf_simple (nlte_approx,f90) introduced
!                    calling sequence of WEIGH11 changed
!
!                    number of smaller topics changed
!                    GLOBAL (controlling ILOWNL, IMAXNL) put to nlte_opt
!                    EXPIN: argument zero explictly treated 
!                    check in ILOWIMAX_LTE: if blevel (HE) is zero, &
!                                           update to a small number
!
!     version : 10.1 Dec. 2009: formal solution (jbar and alo) of RT
!                    for photospheric lines changed (nlte_approx, V7.1)
!                    new method uses differential formulation
!                    switch between new (differential) and old (integral)
!                    formulation with OPTPHOTLINES_DIFF = TRUE/FALSE
!                    in module nlte_opt.  
!
!                    subroutine TCORR: calculation of dFerr/dtau improved
!                    for closely separated grid-points
!
!                    subroutine MODVL and MODVL_UPDATE: check that point
!                    just below SRNOM lies not too close to previous point.
!                    in case, cure this problem by skipping the previous point.  
!
!                    new file CONVERG_METALS: controls convergence of metals
!
!                    if UPDATE_STRUCT = TRUE,
!                    convergence criterium for first hydro-iterations changed:
!                    if ITHYD > 20 and abs(corr) > 0.97, model considered as
!                    converged (usually, < 10 iterations required). 
!                    Should not matter since model updated anyway.
! 
!                    test of photospheric clumping with rho_phot = rho_phot/fcl
!                    (constant clumping factor) enabled if OPT_PHOTCLUMP:
!                    two models (unclumped model 0 and clumped model 1 with
!                    FCL > 1 but constant) should be identical with respect
!                    to model structure, ionization fraction etc
!                    (plotted as a function of M or TAUR) and profiles
!                    if Mdot1 = fcl Mdot0 and Rstar1 = fcl Rstar0
!
!                    in MODVL, accuracy w.r.t. location of v < vs2 changed
!
!                    new formula 66 in tratcol (for NIV) 
!                    old bug in FRESFIN corrected
!                    for restarted models with different metallicity, 
!                    metallic background now recalculated (see nlte_approx) 
!
!                    no update of phot. structure performed for dense winds
!                    where radacc(start) > radacc(actual)
!                    at the outer photosphere (wind acceleration becomes of importance)
!
!                    subroutine ILOWIMAX_OLD updated w.r.t. new N atom
!
!     version : 10.1.1 April 2010: Also cmfgen input can be used (with ending
!                    .tlusty), see subroutine MODVL (extrapolation of
!                    input structure now allowed.)  
!
!     version : 10.1.2 Jan 2011: three changes of precision (for intel 11.0)
!
!     version : 10.1.3 March 2012: can use also Michel Cure's input.
!                    NO update of phot. structure
!                    OPTMOD = TRUE in the first iteration.
!                    Filename has to be Modelname+'.michel'
!                    NOTE: The input value of beta has to be representative
!                    for Michel's v-field
!                    Bug in MODVL_UPDATE fixed: SIGMA_THOMSON needs still
!                    to be calculated from XNE(LTE) to be consistent with TAUR
!                    (problems occured for cool stars).
!                    New philosophy in MODVL and MODVL_UPDATE to cure problems
!                    with point close to SRNOM (2nd try with safety-factor 1.2)  
!     version : 10.1.4 April 2012: inclusion of Paco's clumping law with
!                    4 parameters (CL1 ... CL4)
!                    small bug in FRESCAL removed, regarding LAM_LY 
!  
!     version : 10.1.5 April 2013: Improved treatment of freq. spacing between
!                    911 and 925 A, to ensure a maximum separation of 3 A
!                    (minimum box size for nsubinter = 120  is 3.6 A, can lead
!                    to problems with nlambox_ly when many ions included)
!                    -> changes in FRESCAL (also in dimes.f90)
!
!     version : 10.1.6 July 2013: Update of routines that convolve
!                    OP-cross-sections with a Gaussian (mainly SMOOTH and
!                    CONVOLVE, but also OP_RBFSETUP, used by formula 20), &
!                    to allow for a correct treatment of Miguel's data
!                    (e.g., MgI and MgII)  
!
!     version : 10.2.0 October 2014: inclusion of wind-embedded X-rays
!                    new module nlte_xrays, new package lambda_xray.f90,
!                    new data file k_shell_Auger_data  
!                    X-ray emission following Raymond & Krolik,
!                    shock cooling zones following Feldmeier et al. (1998)
!                    requires additional input with keyword XRAYs (see docu.txt)
!                    K-shell Auger ionization if OPTAUGER = .TRUE. (default)
!                    L/M-shell Auger ionization for higher elements missing ..,  
!                    L_x/L_bol calculated and saved (file XLUM),
!                    in range 100 eV to 2.5 keV
!                    Tested: explicit and bg elements (if consistent) give very
!                            similar results for ionization fractions
!                    Tested: HHe and other models give very similar results
!
!     version : 10.3.0 October 2014: inclusion of optically thick clumping
!                    (porosity and vorosity), programmed by Jon Sundqvist
!                    included into version 10.2.0  
!
!     version : 10.3.1 missing in standard path. See nlte_v10/standard_10.3
!
!     version : 10.3.2 Feb. 2015: improved freq. grid to allow for
!                    a sufficient resolution between 911 and 1600 A
!                    (particularly Ly_beta) corresponding to v 10.1.7/10.1.7.1
!
!     version : 10.4  July 2016: inclusion of 10.1.7.2
!                     Frequency grid (FRESCAL, FRESFIN) + compiler issue
!
!                     rechanged COPYOCC (e.g. CV/CVI during first iterations)
!
!                     inclusion of new parameterization of X-ray filling
!                     factor, as resulting from the work by Koushik Sen/Jo
!                     when combining the results from Feldmeier et al. 1997
!                     and Owocki et al. 2013.
!                     The present input allows to use the old parameterization
!                     (rho^2 dependence of emissivity in radiative and
!                     adiabatic shocks), and the new one, where the
!                     emissivity in radiative shocks is rho-dependent.
!                     When the parameter PX is set to -1000 or lower, the old
!                     parameterization is used. If p > -1000, the new one is
!                     used, with PX the exponent describing the distribution
!                     of shocks (see Owocki+). For the old description,
!                     FX corresponds to the volume-filling factor, for the
!                     new one to the normalization of the shock-number, n_0.  
!                     The combined, radius-dependent filling factor is now
!                     called FXL
!                     Also updated: electron density for X-ray calculations;
!                     since this density refers to the hot plasma, the
!                     electron density has to refer to completely ionized
!                     Helium (error in Feldmeier et al., but also used in
!                     previous x-ray versions).
!                     Also updated: the input value Rmin refers to units
!                     of Rstar and not to the lowermost radius.  
!
!                     new formula 101 for CBB-transitions implemented,
!                     to allow for an effective reading for tabulated collision-strengths  
!
!     version : 10.4.1 July 2016: inclusion of modifications from CMF-path:
!                     subr. MODVL -- region between 2.0 to 0.2 vs2
!                                        changed to 2.0 to 0.3 vs2
!                                 -- smoothing of v-field at transition point 
!                     subr. CONT  -- flux-error
!                     subr. MODVL_UPDATE: write XNELTE instead XNE to model file
!                     ... and few minor issues
!                     including one bug found by Sarah Brands later on in v10.3.1
!
!     version: 10.4.2 Aug 2016: updated routine CLUMPING from Jon
!                     optically thick clumping generalized,
!                     fvel normalized,
!                     input generalized.
!                     still missing: stop statement to prevent clumping below *sonic* point
!
!     version: 10.4.3 Sept 2016: routine OP_RBFSETUP/CONVOL modified:
!                     improved OP-cross sections when ionization to excited levels
!                     STOP for RBF-formula 101 enforced
!                     related changes in FRESFIN (INDFRQ1), NETMAT and OPACITC.  
!                     new routine INTERMEPHO1.
!                     in brief: split of photo-integrals. For FRE ge FRECIN,
!                     rates towards excited (and, if FRECIN = FRECFN, towards
!                     ground state). Calculated in INTERMEPHO.
!                     For FRECFN1 le FREC le FRECIN, corresponding photo-rates
!                     towards ground-state. Calculated in INTERMEPHO1
!                     Corresponds to explicit DR-approch,
!                     rates are called RBF1 in megasalida.
!                     Approach should be a good approximation. It assumes
!                     that largest part of cross-sections (ge FRECIN) relates to
!                     excited or ground-state, whilst lower frequency part
!                     (if present)  corresponds to resonances and relates
!                     to ground-state.
!                     Somewhat wrong if ionization to higher than 2nd stage.
!
!     version: 10.4.4 Dec. 2016: CBF rates for ionization towards excited
!                     states improved. requires updated princesa.f90  
!
!     version: 10.4.5 March 2017: Inclusion at one additional freq. point
!                     at HeII 303.797 (corresponding to A10HHe-energies),
!                     to enforce identical solutions for different elements
!                     (different freq. grids). Driven by a first discrepancy
!                     when calculating HHe, HHeC and HHeN and HHeNC models
!                     (in the HHeC models, HeII did not recombine for d40,
!                      contrasted to the other models)
!                     NOTE: change wavelength when HeII energies are changed
!                     in data-file  
!
!     version: 10.4.6 July 2017: subr. OPACITC modified; calculate
!                     only those bf-opacities below ILOW where ENIONND NE 0 
!                     (otherwise problems in RESTART for OPTXRAY (oxygen!)
!     version: 10.4.7 Oct 2017: subr. MODVL_UPDATE modified; allow for
!                     somewhat larger inconsistencies, but particularly
!                     re-define xm0 exactly.  
!
!     version: 10.5.0 Oct 2020: compatible with gfortran, old bug by Jon
!                     for optthick and npara=3 removed  
!
!     version: 10.5.1 May 2021: small inconsistencies in WM-Basic Data Base
!                     regarding DR data for OII (re-) detected. For those
!                     transitions, stop statement in subr. intermedr de-activated
!
!     version: 10.5.1.1 Nov 2021: old bug from Jon in subr. clumping for
!                      optthick=T and npara=9 (Puls 2006) regarding
!                      fvol finally removed. Never used until now.
!
!     version: 10.6.0 July 2021 (new version number from Sept. 2021 on,
!                     but test calculations already in July) 
!                     change in FRESCAL AND FRESFIN, to allow for a consistent
!                     frequency grid around Ly_gam and Ly_delta, in the same
!                     spirit as previously for Ly_alpha and Ly_beta (important
!                     for B-stars). 
!                     Specific problem in detail input detected:
!                     for high frequencies (ga 100 A), at least RBF formula 12
!                     might give erroneous values, when out of region where
!                     fit is valid (detected in OIII data from Norbert/Miguel).
!                     In future, range of validity needs to be provided.
!                     Until then, RBF formula-12-cross-sections for lambda < 100
!                     are approximated via 
!                     sigma(nu) = sigma_old(100A)*(nu(100A)/nu)^2 
!                     whenever sigma_old(nu) > sigma_old(100A)  
!
!                     Sept. 2021: symmetric extension of Stark-broadening
!                     to neigbouring "boxes" in sumopal (lte and nlte).  
!                     Addition (in FRESCAL) of freq. points for most important
!                     Stark-broadened transitions (defined in sumopal_nlte)
!                     saved in subr. SAVE_ADD_FREQ_POINTS (new file
!                     ADD_FREQUENCIES), read in (or deleted, if starting
!                     with iteration "0") after INDAT.DAT has been read.  
!                     FRESCAL updated (new variable NEWFREQ, which indicates
!                     that freq. grid has changed even if IFRE did not change.
!                     (can happen). Important to reset OP_FLAG, OP_FLAG2.
!                     MEANERR <= -4.5 as additional convergence criterium
!
!                     Oct. 2021: He ionization in ILOWIMAX_LTE changed, to
!                     allow for HeII also in cooler stars, and
!                     to be consistent with v11  
  
!                     This version requires nlte_approx.f90 v7.5.2 and
!                     dimes.f90 v1.3.2  
!
!!! WARNING !!!!
! for future calculations, remember that opac_nolines is mean, not
! effective opacity
!!! WARNING !!!!
!
CHARACTER*10:: VER_NLTE='10.6.0'
!
END MODULE version_nlte  
!
!----------------------------------------------------------------------
!
MODULE nlte_opt
!
! control options
!
IMPLICIT NONE
! to be consistent with older versions (before V9.6), set the next 4 options to .false.

!  Enables update of photo-struct. by using flux-mean around iteration 10
!  Only one update is performed
!  
!  default .true.
LOGICAL :: UPDATE_STRUCT=.TRUE.

!  Exact treatment of photospheric lines from selected background elements
!  default .true.
LOGICAL :: OPTPHOTLINES=.TRUE.

! Stark-broadening for (quasi-) resonance lines of metals and H at Lyman-jump 
!  default .true.
LOGICAL :: OPTSTARK=.TRUE.

! controls whether DR for background elements
!  default .true.
LOGICAL, PARAMETER :: OPTDR=.TRUE.

!----------------------------------------------------------------------------
! the following options should be only changed if you know what you are doing

! new parameter, from V6.1 on
! if set to .true., ILOWNL, IMAXNL will be constant throughout complete atmosphere, &
! controlled by the average in between NS-5 to NS+5
! if set to .false., ILOWNL, IMAXNL will be locally consistent (but with
! disadvantages regarding the CMF lines, see below)
LOGICAL, PARAMETER :: GLOBAL = .TRUE.

!----------------------------------------------------------------------------
!  treatment of photospheric lines via differential or integral method
!  (integral method needs additional tables (k2atable, l2atable, l2table, l3atable) in DATA) 
!  default .true.
LOGICAL :: OPTPHOTLINES_DIFF=.TRUE.

! enhanced output for control of temperature correction (additional files!)
!  default .false.
LOGICAL, PARAMETER :: OUTTHB = .FALSE.

! enhanced output for optically thick clumping (additional files!)
!  default .false.
LOGICAL, PARAMETER :: OUTTHICK = .FALSE.

! lot of enhanced output for opacity data cross-sections and convolution (additional files!)
!  default .false.
LOGICAL, PARAMETER :: OPDEBUG = .FALSE.

! use Kurucz abundances for background elements
!  default .false.
LOGICAL :: OPTKUR = .FALSE.

! simple temperature correction (out-dated)
!  default .false.
LOGICAL :: OPTTCOR_SIMPLE=.FALSE.

! test of photospheric clumping with rho_phot=rho_phot/fcl 
!  default .false.
LOGICAL :: OPT_PHOTCLUMP=.FALSE.

! to force single line treatment for all lines, set OPTSING = .TRUE.
!  default .true.
LOGICAL, PARAMETER ::  OPTSING=.TRUE.

! to forbid treatment of wind induced overlap (=>tests), set IOV_ONLY = .TRUE.
!  default .false.
LOGICAL, PARAMETER ::  IOV_ONLY=.FALSE.

END MODULE nlte_opt
!
!----------------------------------------------------------------------
!
MODULE nlte_xrays
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE

! enable xray-treatment; controlled by content of INDAT.DAT
LOGICAL ::  OPTXRAY
! enable Auger (K-shell) ionization; Auger L and M shell ionization for heavy
! elements still missing
LOGICAL ::  OPTAUGER=.TRUE.

REAL(DP), ALLOCATABLE, DIMENSION(:) :: ENERX, FXL
REAL(DP), ALLOCATABLE, DIMENSION(:,:) :: LAMBDAX,LAMBDANU,OPA_KSHELL

INTEGER(I4B), PARAMETER :: N_KEDGES = 59 ! in ATOMDAT_NEW/k_shell_Auger_data
INTEGER(I4B), PARAMETER :: K_NMIN = 3, K_NMAX = 8  ! min/max ion. stage to be included in freq. grid

REAL(DP) :: FX ! VOLUME FILLING FACTOR or SHOCK NUMBER (corresponding to n_0)
!                FOR SHOCK EMISSION
REAL(DP) :: PX ! PARAMETER FOR dN/dlnr = n_0*(RMIN/r)^p
REAL(DP) :: RMINX_EFF ! MINIMUM EMISSION RADIUS (in units of SR)
                      ! (slightly different from input RMINX (in units of Rstar))

INTEGER(I4B) :: LXMIN=1 ! INDEX FOR MINIMUM EMISSION RADIUS


REAL(DP), DIMENSION(N_KEDGES) :: ETH,SIGMA,S,ZEFF

REAL(DP), DIMENSION(N_KEDGES,6) :: AUG_1_6

INTEGER(I4B), DIMENSION(N_KEDGES) :: Z,N

character*2, dimension(30) :: name 

! for info and checks
data name /'H ','HE','LI','BE','B ','C ', &
&            'N ','O ','F ','NE','NA','MG', &
&            'AL','SI','P ','S ','CL','AR',&
&            'K ','CA','SC','TI','V ','CR',&
&            'MN','FE','CO','NI','CU','ZN'/


END MODULE nlte_xrays
!
!----------------------------------------------------------------------
!
MODULE run_once
!
! global options
!
IMPLICIT NONE
LOGICAL :: START_WEIGH11=.TRUE.
LOGICAL :: START_CMFSING=.TRUE.
LOGICAL :: START_CMFMULTI=.TRUE.
LOGICAL :: START_JBAR_METALS=.TRUE.
LOGICAL :: START_CMF_SIMPLE=.TRUE.
!
END MODULE run_once
!
!----------------------------------------------------------------------
!
MODULE nlte_var
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!for U and U2 functions 
!
character*(*), parameter :: fpath='../inicalc/DATA/'
!
!----------------------------------------------------------------------
! wavelength (A) for Hydrogen Lyman-jump (red side)
!
REAL(DP) :: LAM_LY
!----------------------------------------------------------------------
! transition point for selected elements
!
INTEGER(I4B) :: LWION1
!
!----------------------------------------------------------------------
! correction for lowermost input flux dB/dtau
!
REAL(DP) :: CORRFC
!
!----------------------------------------------------------------------
! various velocities
!
REAL(DP) :: VMAX,VSOUND,VDIV
INTEGER(I4B) :: NDIV=0, NDIV_CALC !(OUTERMOST POINT OF QUASI-HYDROSTATIC REGIME
!
!----------------------------------------------------------------------
! variables for clumping parameters
!
INTEGER(I4B) :: NPARA_CLF = 11 !maximum number of parameters,
                               !will be overwritten 
REAL(DP), DIMENSION(11) :: PARA_CLF_INPUT=-1.D99
REAL(DP), ALLOCATABLE, DIMENSION(:) :: CLF_FIELD
!
!----------------------------------------------------------------------
! variables for selected Stark-broadened transitions in subr. sumopal
!
INTEGER(I4B) :: NSTARK
INTEGER(I4B), PARAMETER :: NSTARKMAX = 500
INTEGER(I4B), DIMENSION(NSTARKMAX) :: IDSTARK
REAL(DP), DIMENSION(NSTARKMAX) :: WAVESTARK
!
!----------------------------------------------------------------------
! variables for ff gaunt-factors
!
! local parameter IONMAX
!
INTEGER(I4B), PARAMETER :: IONMAX = 9 
INTEGER(I4B) :: IZMIN, IZMAX 

REAL(DP), DIMENSION(IONMAX,ID_NDEPT,ID_FREC1) :: GFFNLTE
!
!----------------------------------------------------------------------
!alphafs 
!
!
REAL(DP), DIMENSION(ID_RBFTR,ID_FREC1) :: ALPHAFS
!
!----------------------------------------------------------------------
!blocf
!
REAL(DP), DIMENSION(ID_FREC1) :: AS,A2S,A3S,BS,B2S,B3S  
!----------------------------------------------------------------------
!
!ccom1 (without alevel)
!
REAL(DP), DIMENSION(ID_NDEPT) :: XNELTE
INTEGER(I4B), DIMENSION(ID_ATOMS,ID_NDEPT) :: IONIS
INTEGER(I4B), DIMENSION(ID_ATOMS) :: IONISST  
!----------------------------------------------------------------------
!ccom4 dimension corrected (from kis to kis+1)
!
REAL(DP), DIMENSION(ID_ATOMS,ID_KISAT+1) :: XL
REAL(DP), DIMENSION(ID_ATOMS,ID_KISAT+1,ID_NDEPT) :: PARTIT
!----------------------------------------------------------------------
!ccomoc
!
REAL(DP), DIMENSION(ID_ATOMS,ID_KISAT+1,ID_NDEPT) :: OCCNUM
INTEGER(I4B), DIMENSION(ID_NDEPT,ID_ATOMS) :: MAINION
!----------------------------------------------------------------------
!cffine
!
REAL(DP), DIMENSION(ID_FREC2) :: FREF,WFREF
INTEGER(I4B), DIMENSION(ID_FREC2) :: INDFIN
INTEGER(I4B), DIMENSION(ID_RBFTR) :: INDFRQ, INDFRQ1
INTEGER(I4B) :: IFREF
!----------------------------------------------------------------------
!cfgrid
!
REAL(DP), DIMENSION(ID_FREC1) :: FRE,WFRE,HC
INTEGER(I4B), DIMENSION(ID_FREC1) :: LTHIN
INTEGER(I4B) :: IFRE = 0 ! initialize
!----------------------------------------------------------------------
!clevel
!
REAL(DP), DIMENSION(ID_LLEVS) :: ALEVEL, BLEVEL, NIST, XNKK
REAL(DP), DIMENSION(ID_ATOMS,ID_KISAT+1,ID_NDEPT) :: ENIONND,ENIONND_LTE
!----------------------------------------------------------------------
!cmms
!
INTEGER(I4B), DIMENSION(ID_RCBBT) :: MMLOW,MMUP
INTEGER(I4B), DIMENSION(ID_RBFTR) :: MLOW,MUP
INTEGER(I4B), DIMENSION(ID_CBSFT) :: MCLOW,MCUP
!----------------------------------------------------------------------
!comcmf1,comcmf2,comcfm3,comcmfs
!
INTEGER(I4B), DIMENSION(ID_NTTRD) :: INDXLAMC
CHARACTER*6, DIMENSION(ID_NTTRD) ::  LABLINELO,LABLINEUP

REAL(DP), DIMENSION(ID_NTTRD) :: XLAMCMF,VDOPCMF,BLUCMF,BULCMF,AULCMF, &
&                                XMAXDOP
REAL(DP), DIMENSION(ID_NDEPT,ID_NPOIN) :: TAUZ1,TAU1,PP1=0.D0
REAL(DP), DIMENSION(ID_NFCMF,ID_NDEPT) :: AK1,AZ1,SLINEK
!----------------------------------------------------------------------
!comcmfa,comcmfb,comtemp
!
LOGICAL :: OPTNEUPDATE, OPTCMF, OPTMODEL, OPTLUCY, OPTMET, OPTLINES=.FALSE.,&
&          LINES,LINES_IN_MODEL, &
&          METALS_CONVERGED = .FALSE.,LTE_UPDATE=.FALSE.,CONCON=.FALSE.,&
&          ALMOST_CONVERGED = .FALSE.

REAL(DP) :: OPTMIXED
!
!----------------------------------------------------------------------
!comcont
!
REAL(DP), DIMENSION(ID_NDEPT,ID_FREC1) :: OPAC,STRUE,XJ,XXK,ALO, &
&                                         STRUE_M_OLD,STRUE_M_NEW, &
&                                         OPAT_M_OLD,OPAT_M_NEW, &
&                                         OPAC_NOLINES, OPAT_M_NOLINES, &
&                                         THOMSON_LINES
!                                       , RAYLEIGH = 0.D0
REAL(DP), DIMENSION(ID_NDEPT,ID_FREC1) :: TRADJ,TRADJ1
!
!----------------------------------------------------------------------
!
!comiact
INTEGER(I4B), DIMENSION(ID_ATOMS) :: IMIA,IMAA,IMIANL,IMAANL
!----------------------------------------------------------------------
!
!comopac
REAL(DP), DIMENSION(ID_NTTRD,ID_NDEPT) :: OPACLIN
!----------------------------------------------------------------------
!compz
!
REAL(DP), DIMENSION(ID_NPOIN) :: P
REAL(DP), DIMENSION(ID_NDEPT,ID_NPOIN) :: Z
!----------------------------------------------------------------------
!comtlucy
!
INTEGER(I4B) :: NSDIV
REAL(DP) :: SR,SRVMIN,SRNOM,CONSTT
!----------------------------------------------------------------------
!comtrad, comsa
!
INTEGER(I4B), DIMENSION(ID_RCBBT) :: INDEXRBB
INTEGER(I4B), DIMENSION(ID_NTTRD) :: INDEXCMF,INDEXSA
REAL(DP), DIMENSION(ID_NTTRD,ID_NDEPT) :: TLUMAT,TULMAT,OPACM,SCONTM, &
&                                         BETOLD
!----------------------------------------------------------------------
!comvelo
!
REAL(DP) :: H1
!----------------------------------------------------------------------
!comweigh
!
REAL(DP) :: W0LAST
REAL(DP), DIMENSION(ID_NDEPT,ID_NPOIN-1) :: VP,VP2
!----------------------------------------------------------------------
!der,der1
!
INTEGER(I4B) :: IDUM,ITMIN=1
REAL(DP) ::  GGRAV,TEFF,SIGEM,CKAPPA,XT,A2TE
REAL(DP), DIMENSION(ID_NDEPT) :: DELSIG=1.D0,DQDTAU=0.D0,DELMU=1.D0, &
&                                DELPARA=1.D0

!----------------------------------------------------------------------
!cqual
!
INTEGER(I4B), DIMENSION(ID_LLEVS,ID_NDEPT) :: IQUAL
!----------------------------------------------------------------------
!hopf,mic_turb,precision
!
REAL(DP) :: QINF,Q0,GAMHOPF,VTURB,PRECIS
!----------------------------------------------------------------------
!ithist
!
REAL(DP), DIMENSION(30) :: EXITH,XKAPITH,CORRITH
!----------------------------------------------------------------------
!modnam
!
CHARACTER*50 :: MODNAM
!----------------------------------------------------------------------
!otra
!
INTEGER(I4B) :: INDR,INDC  
!----------------------------------------------------------------------
!output
!
INTEGER(I4B) :: ICONVER  
!----------------------------------------------------------------------
!path (ode)
!
INTEGER(I4B) :: KMAX,KOUNT
REAL(DP) :: DXSAV, XP(200),YP(10,200)
!----------------------------------------------------------------------
!soluna
LOGICAL :: UNASOL=.FALSE., MEGAS
!----------------------------------------------------------------------
!store
!
REAL(DP), DIMENSION(ID_FREC2) :: BN
INTEGER(I4B), DIMENSION(ID_FREC2) :: INDEX
!----------------------------------------------------------------------
!ufuncg,ufunck
!
REAL(DP) ::  UFUNG(9,17),UFUNK(21,76)  
!----------------------------------------------------------------------
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: N1_B = 21, N2_B = 6  
INTEGER(I4B), PARAMETER :: N1_L1 = 61, N2_L1 = 71  
INTEGER(I4B), PARAMETER :: N1_L2 = 12, N2_L2 = 18  
INTEGER(I4B), PARAMETER :: N1_S = 192, N2_S = 200  
!     ..
! U2-values from tables
REAL(DP), DIMENSION(N1_B,N2_B) :: U2B
REAL(DP), DIMENSION(N1_L1,N2_L1) :: U2L1
REAL(DP), DIMENSION(N1_L2,N2_L2) :: U2L2
REAL(DP), DIMENSION(N1_S,N2_S) :: U2S
!----------------------------------------------------------------------
!vthermal, abundnd, comyhe
!
REAL(DP) :: YHEIN
REAL(DP), DIMENSION(ID_ATOMS) :: VTHER
REAL(DP), DIMENSION(ID_ATOMS, ID_NDEPT) :: ABUNDND

! prespecified elements
INTEGER(I4B) :: NAT_TOT, NAT_BG
REAL(DP), DIMENSION(30) :: ABCHANGED,ABUNDBG
CHARACTER*6, DIMENSION(30) :: NAMES_CHANGED,NAMES_BG
LOGICAL :: ABUND_HAVE_CHANGED = .FALSE.
!----------------------------------------------------------------------
!weights
!
REAL(DP) :: W5(5), W9(9)
!----------------------------------------------------------------------
!Explicit RBF cross sections (Opacity Project Data etc.)

character*(*), parameter :: oppath='../inicalc/OP_DATA_NEW/' 
REAL(DP), DIMENSION(ID_RBFTR,ID_FREC1) :: OP_DATA
!FRECFN1 analogous to FRECFN (ground-state ionization), but for OP-data
!(rbf-formula 20)
!only until minimum energy where sigma ne 0 (in between FRECIN AND FRECFN)
!if sigma ne 0 for energies lower than FRECFN, then FRECFN1=FRECFN
!calculated in routine RBF_SETUP. This is a certain approximation
!(sometimes, sig ne 0 even below ground-state edge), but used
!for consistency with the remaining approach.
REAL(DP), DIMENSION(ID_RBFTR) :: EOPDIFF, FRECFN1
LOGICAL, DIMENSION(ID_RBFTR) :: OP_FLAG, OP_FLAG2
!
!----------------------------------------------------------------------
!
END MODULE nlte_var
!
!***********************************************************************
!
MODULE cmf_multi
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE

TYPE infotable
  INTEGER(I4B) :: INDEX
  INTEGER(I4B) :: INDBW
  REAL(DP) :: XOV
  REAL(DP) :: IF
END TYPE

! controls velocity range where overlap is accounted for
REAL(DP), PARAMETER :: TAUSCRIT=0.01D0

INTEGER(I4B), PARAMETER :: LTO = 2*ID_NDEPT-1

INTEGER(I4B), DIMENSION(ID_NTTRD) :: INDOV

INTEGER(I4B), DIMENSION(:), ALLOCATABLE :: INDOVERLAP_ORDER

TYPE(INFOTABLE), DIMENSION(:), ALLOCATABLE :: INFOTAB

REAL(DP), DIMENSION(:,:,:), ALLOCATABLE :: USAVE,VSAVE
REAL(DP), DIMENSION(:,:),   ALLOCATABLE :: IPLUS,IMINUS

INTEGER(I4B), DIMENSION(ID_NPOIN-1) :: LMAXJP
REAL(DP), DIMENSION(ID_NPOIN-1) :: FMIN,FMAX
REAL(DP), DIMENSION(ID_NPOIN-1) :: IMINUS_CONT
REAL(DP), DIMENSION(ID_NPOIN-ID_NDEPT+1) :: IPLUS_CONT

REAL(DP), DIMENSION(LTO,ID_NPOIN-1) :: UV,ZRAY

END MODULE cmf_multi
!
!***********************************************************************
!
MODULE tcorr_var
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE

! Limit for the Flux-Tcorr
REAL(DP) :: TAURLIM=.5D0 ! standard value, is adapted in case of thick winds
REAL(DP) :: TMIN1 = .4D0 ! minimum value for T(r) in units of Teff
                         ! from comparison with Adi's models, a value of .4 to .5
                         ! seems to be reasonable.
REAL(DP) :: TMINABS=6000.! absolute minimum temp. which can be handeled

REAL(DP) :: EMAXTC

! new variable: convergence control of metals
REAL(DP) :: ERROR_MET
! .
! Free-free quantities
REAL(DP), DIMENSION(ID_NDEPT,ID_FREC1) :: FFOPA,DTFFOPA, &
&          FFOPA_M=0.,DTFFOPA_M=0.

REAL(DP), DIMENSION(ID_NDEPT) :: QFFH,QFFC,DTQFFH,DTQFFC
! .
! Continuum quantities
REAL(DP), DIMENSION(ID_NDEPT) :: QCBFR,QCBFI,QRBFR,QRBFI, &
& DTQRBFR,DTQRBFI,DTQCBFH,DTQCBFC

REAL(DP), DIMENSION(ID_NDEPT) :: QCBBU,QCBBD,DTQCBBU,DTQCBBD

! Quantities for metal-background (no coll. ionization, since not present
!                                  in our approx.)
REAL(DP), DIMENSION(ID_NDEPT) :: QCBFR_M=0.,QCBFI_M=0., &
&                                DTQCBFR_M=0.,DTQCBFI_M=0., &
&                                QRBFR_M=0.,QRBFI_M=0., &
&                                DTQRBFR_M=0.,DTQRBFI_M=0., &
&                                QCBBU_M=0.,QCBBD_M=0., &
&                                DTQCBBU_M=0.,DTQCBBD_M=0. 

!
! Logical controls
LOGICAL :: ENATCOR,OPTTCOR,TEMP_CONVERGED=.FALSE.
! .
!
END MODULE tcorr_var
!
!***********************************************************************
!
MODULE photstruc

USE nlte_type
USE nlte_dim
IMPLICIT NONE


INTEGER(I4B) :: NDMOD
REAL(DP) :: XT, CONST, EXPO, SIGEMIN, CLF_CONST=1.
REAL(DP), DIMENSION(:), ALLOCATABLE::  XMEXT,TEXT
REAL(DP), DIMENSION(:), ALLOCATABLE::  PKUR,XNEKUR,GRADKUR
REAL(DP), DIMENSION(:), ALLOCATABLE::  RHOTLU,XNETLU
REAL(DP), DIMENSION(:), ALLOCATABLE::  RMICH,RMICH1,VMICH,RHOMICH
REAL(DP), DIMENSION(ID_NDEPT) ::  XM, CHIBAR_H, DELCHI
CHARACTER(6) :: MODTYPE
END MODULE photstruc
!
!***********************************************************************
!
MODULE nlte_porvor

USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT,ID_FREC1
IMPLICIT NONE

REAL(DP), PARAMETER :: EPSI= 1.D-12 ! FOR PRECISION OF OPA_EFF_RAT IN OPTICALLY THIN APPROACH
REAL(DP), PARAMETER :: EPSI1=1.D0+1.D-12 ! FOR PRECISON OF OPA_EFF_RAT

LOGICAL :: OPTTHICK

REAL(DP) :: W_BG_RED,W_BG_BLUE ! will be set to WAVBLUE and WAVRED in nlte_approx
REAL(DP) :: CONV_EFF_OPA_RAT=1.0D0  

REAL(DP), ALLOCATABLE, DIMENSION(:) :: FIC_FIELD,FVEL_FIELD,HPOR_FIELD

REAL(DP), DIMENSION(ID_NDEPT) :: FIC,TCL_FAC_LINE,TCL_FAC_CONT,FVEL,FVOL,HPOR

REAL(DP), DIMENSION(ID_NDEPT,ID_FREC1) :: OPA_EFF_RAT=1.D0, OPA_EFF_RAT_OLD=1.D0
!these are the crucial quantities which correct the mean opacities
!<opa> = opa/clf to effecitve opacities opa_eff = <opa> * opa_eff_rat with
!opa_eff_rat = (1 + fic * tau_cl)/(1 + tau_cl)
!in the optically thin case, opa_eff_rat has to be unity at all frequency
!points, by forcing tau_cl to 0.

END MODULE nlte_porvor
!
!***********************************************************************
!
! main program
!
!***********************************************************************
!
PROGRAM FASTWIND
!
!-----this program will - hopefully - solve the nlte line-formation
!-----problem for unified atmospheres in an optimum  way
!
!
!     program can be used now for calculation of hydrogen and helium,
!     silicon, magnesium, C, N, O (DETAIL versions) 
!
!     and for all elements with Adi's data (if you have a corresponding
!     input file). Note, however, that using elements with a large
!     line number is inconsistent with present assumptions
!
!     developing/programming: j.puls, e. santolaya-rey (version 0.0), &
!                             miguel a. urbaneja (tcorr and more ions, &
!                             version 7.2.1)   
!
!     You can include own clumping laws by manipulating subroutine
!     clumping (max. number of parameters: 10)  
!
!     note: if new atoms are implemented, provide new formulae in
!           subroutines crossbf (photo cross sections),
!                       intermecol (coll. cross sections (ionisation))
!                   and tratcol (coll. excitation)
!           if necessary!!!
!           also required is update of ilow, imax in routine ilowimax 
!           (_old, _lte, _nlte (latter in nlte_approx.f90))
!
!
!     missing points at present (mostly solved in v11 -- full cmf)
!
!   i)calculating taur(nlte)
!  ii)updating of hydro-structure with respect to tau'(nlte)
! iii)(overlapping) lines included in continuum and ionization integrals 
!     maybe important at edges
!  iv) external photospheric structures only for unclumped models (generally)
!   v) RBF-formula 101 (currently not allowed) needs to be updated and checked.
!
!
!-----------------------------------------------------------------------
!
! XNH is hydrogen density (total one, HI+HII), independent of LTE/NLTE
!
!-----this program uses three values for the stellar radius:
!--   1) the input value rstar, which is the nominal radius where
!--      tau(-lucy) = 2/3; note that log g  and teff are
!--      also defined with respect to this radius. srnom = rstar * rsun.
!--   2) the lowest value of r(nd) = sr < srnom which is the scaling
!--      radius of the poblem.
!--   3) the radius where the beta-field and the photosphere merge
!--      ("transition point"): srvmin
!--
!--   rtau23 is the ratio of srnom/sr
!--
!
! WARNING!!! program nlte uses

!  abund = n(k)/n(H), i.e. normalized to hyd.

!            program nlte_approx uses   
!  abund = n(k)/Sum(n(k)), i.e. normalized to ntot
!
! Thus, also the value of summass = Sum(abund(i)) in both cases differ!!!
!  
! END WARNING!!!


!!! WARNING !!!!
! for future calculations, remember that opac_nolines is mean, not
! effective opacity
!!! WARNING !!!!

!
USE version_nlte
USE nlte_type
USE nlte_dim
USE fund_const, ONLY: amh,akb

USE run_once ! all start variables

USE nlte_opt, ONLY: UPDATE_STRUCT, OUTTHB, OPT_PHOTCLUMP, OUTTHICK, OPTSTARK

USE nlte_xrays, ONLY: OPTXRAY,FX,LXMIN

USE princesa_var, ONLY: ABUND,LABAT,LABL,LKL,LABX,LKX,LABLS,KLS,NAT,WEIGHT

USE nlte_var, ONLY: BLEVEL,ENIONND,OPTNEUPDATE, &
& OPTCMF,OPTMET,OPTMIXED,OPTMODEL,OPTLUCY,LINES,LINES_IN_MODEL, &
& METALS_CONVERGED, ALMOST_CONVERGED, &
& IQUAL, &
& QINF,Q0,GAMHOPF,VTURB,PRECIS, &
& XKAPITH,EXITH,CORRITH, &
& MODNAM, &
& ICONVER, &
& UNASOL,MEGAS, LTE_UPDATE, NAT_TOT, NAMES_CHANGED, ABCHANGED, &
& NPARA_CLF, PARA_CLF_INPUT, CLF_FIELD, &
& IMIA,IMAA,IMIANL,IMAANL,IONIS,CONCON,LWION1,NDIV,NDIV_CALC,FRE,IFRE, &
& NSTARKMAX,NSTARK,WAVESTARK

Use nlte_var, ONLY: CORRFC,SR,VMAX,VSOUND,VDIV

Use nlte_var, ONLY: XNELTE_COMMON => XNELTE

USE tcorr_var, ONLY : ENATCOR,OPTTCOR,TAURLIM,TEMP_CONVERGED,EMAXTC,ERROR_MET

USE photstruc, ONLY : XT

USE nlte_porvor, ONLY: OPTTHICK, &
&                      FIC,TCL_FAC_LINE,TCL_FAC_CONT,FVEL,FVOL,HPOR, &
&                      FIC_FIELD,FVEL_FIELD,HPOR_FIELD, &  
&                      OPA_EFF_RAT,OPA_EFF_RAT_OLD,CONV_EFF_OPA_RAT

IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,NP1=ID_NPOIN  
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
!     ..
!     .. local scalars ..
!
CHARACTER*256 :: CLF_PARA_STR
CHARACTER*3 :: NCNAME
REAL(DP) ::  BETA,CORVM,EMAX,EXKAP,GGRAV,HEI,RELEX, &
&            RELHYD,RELKAP,RMAX,RTAU23,SRNOM,TEFF,TMIN, &
&            VMIN,VMIN1,XMET,XKAP,XMLOSS,XMMAX,XMU,YHE,YHEIN,DUM,MEANERR

REAL(DP) :: GAMX,MX,RMINX,UINFX,PX

INTEGER(I4B) ::  I,IEND,IITCMF,IITSOBO,IITUPDATE,IN,IQ,ISTART,ITHYD, &
&        ITHYDS,ITLAST,IIT,ITMORE,K,J,L,NREC,NTEMP,NTEMP1,ND,NP,&
&        ITTCOR,NCOR,SET_STEP,SET_FIRST,LWION

LOGICAL ACCEL,FASTSOBO,FRENEW,GREY,HELONE,HOPFSELF,OPTCMF1, &
&       OPTCV,OPTLTE,OPTMOD,RYBICKI, &
&       EXPANSION,RESTART_MET_LTE,LASTLTE,FIRST_METALS,UPDATED,UPDATE_DONE,EXI, &
&       START_XRAY
!     ..
!     .. local arrays ..
REAL(DP) ::  DILFAC(ND1),DILFAC1(ND1),DVDR(ND1),ERR(ND1),ERROLD(ND1),R(ND1), &
&            RHO(ND1),PRESSURE(ND1),RTEMP(ND1),SPECMAT(5,ND1), &
&            TAUE(ND1),TAUR(ND1), &
&            TEMP(ND1),VELO(ND1),XNE(ND1),XNH(ND1),DTFCORR(ND1),FLUXERR(ND1),CLFAC(ND1), &
&            XNELTE(ND1),XNELTE_TAUR(ND1),TAUR_TAUR(ND1),TSHO(ND1)

REAL(DP) :: CO_DUM, TIME_CPU  

INTEGER(I4B) ::  ILOW(ND1,KEL),IMAX(ND1,KEL), &
&                ILOWNL(ND1,KEL),IMAXNL(ND1,KEL),INDEX(ND1)  
!     ..
!     .. external subroutines ..
EXTERNAL CHECKMASS,CONT,COPYOCC,FORMAL,FRESFIN,HOPFPARA,LTE,MODEL, &
&         OPACITC,PZGRID,RATEEQ,RATELI,START,ULECTURE
!     ..
!     .. external function ..
REAL(DP) :: XINTFRE
EXTERNAL XINTFRE
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,MAX,MIN  
!     ..
!     .. data statements ..
DATA GREY/.TRUE./  
DATA OPTLTE/.TRUE./
DATA FIRST_METALS/.TRUE./
DATA START_XRAY/.TRUE./
!     ..
NREC = ID_LLEVS  
ND=ND1
NP=NP1


! the following quantities might be changed during iteration
IITSOBO = 4 !start of line-treatment with SOBO 
IITCMF = 21 !typical start of cmf 
IITUPDATE = 11 !hydro-update if .not. OPTCMF_FULL

FASTSOBO = .FALSE.  
!
PRECIS = 1.D-6
!
! INITIALIZE ENIONND
ENIONND=0.D0
!
!     now it begins...
!
!     copy of indat.dat into catalogue
!
OPEN (1,FILE='INDAT.DAT',STATUS='OLD')  
REWIND 1  

READ (1,FMT=*) MODNAM
READ (1,FMT=*) OPTNEUPDATE,HELONE,ITLAST,ITMORE  
READ (1,FMT=*) OPTMIXED  
READ (1,FMT=*) TEFF,GGRAV,SRNOM  
READ (1,FMT=*) RMAX,TMIN  
READ (1,FMT=*) XMLOSS,VMIN,VMAX,BETA,VDIV  
READ (1,FMT=*) YHEIN,HEI  
READ (1,FMT=*) OPTMOD,OPTLUCY,MEGAS,ACCEL,OPTCMF1
READ (1,FMT=*) VTURB, XMET,LINES,LINES_IN_MODEL
READ (1,FMT=*) ENATCOR,EXPANSION,SET_FIRST,SET_STEP

! no update of phot-struct if external model
IF (OPTMOD) UPDATE_STRUCT=.FALSE.
IF (OPTMOD .AND. OPT_PHOTCLUMP) &
& STOP ' external photospheric model .AND. photospheric clumping not implemented'

!JO changed Dec. 2015
! no update if no temp. correction 
IF (.NOT.ENATCOR) UPDATE_STRUCT=.FALSE.

IF (OPTMOD .AND. ITLAST.EQ.0) THEN
!  IF (ENATCOR) STOP ' Kurucz/Detail model to be used, but ENATCOR=.TRUE.'
  INQUIRE(FILE=TRIM(MODNAM)//'.michel', EXIST=EXI)
  IF (.NOT. OPTLUCY .AND. .NOT. EXI) STOP ' external photospheric model to be used, but OPTLUCY=.FALSE.'
ENDIF

READ (1,FMT='(A)') CLF_PARA_STR

IF(TRIM(ADJUSTL(CLF_PARA_STR)).EQ.'THICK') THEN
  OPTTHICK=.TRUE.
  READ (1,FMT='(A)') CLF_PARA_STR
! read cluming factor
ELSE
  OPTTHICK=.FALSE.
! clumping factor already read, no further action required
ENDIF  

! trick to read unknown number of variables (including description)
OPEN (2,STATUS='SCRATCH',FORM='FORMATTED')
WRITE(2,*) CLF_PARA_STR
REWIND(2)
READ (2,FMT=*,END=1,ERR=1) (PARA_CLF_INPUT(I),I=1,NPARA_CLF)
CLOSE(2)
!
1 DO I=1,NPARA_CLF
  IF(PARA_CLF_INPUT(I).EQ.-1.D99) EXIT
ENDDO

NPARA_CLF=I-1 !(one for the exit)

ALLOCATE(CLF_FIELD(NPARA_CLF))

DO I=1,NPARA_CLF
  CLF_FIELD(I)=PARA_CLF_INPUT(I)
ENDDO

! check for TYPICAL clumping factor
IF (CLF_FIELD(1).LT.1.D0) STOP ' FIRST CLUMPING PARAMETER < 1!'

!Below parameters controlling if optically thick clumping should 
!be considered, or if old assumption of only optically thin clumps
!should be used    
ALLOCATE(FIC_FIELD(NPARA_CLF))
ALLOCATE(FVEL_FIELD(NPARA_CLF))
ALLOCATE(HPOR_FIELD(NPARA_CLF))
IF (OPTTHICK) THEN 
   READ(1,FMT=*) FIC_FIELD
   READ(1,FMT=*) FVEL_FIELD
   READ(1,FMT=*) HPOR_FIELD
ELSE
   FIC_FIELD = 0.0 
!JO changed Aug 2016, since now normalized
   FVEL_FIELD = 1.0
   HPOR_FIELD = 0.0
ENDIF

!for tests
!DO I=1,NPARA_CLF
!   PRINT*,FIC_FIELD(I),FVEL_FIELD(I),HPOR_FIELD(I)
!ENDDO
!STOP ' JS-TESTING!'


OPTMET=.FALSE.
IF(XMET.NE.0.) OPTMET=.TRUE.

! check for consistency
IF(.NOT.OPTMET.AND.LINES) STOP ' XMET = 0., BUT LINE-BLOCKING REQUIRED!'
IF(.NOT.LINES.AND.LINES_IN_MODEL) &
& STOP ' NO LINE-BLOCKING, BUT REQUIRED IN MODEL!'

IF(ENATCOR) THEN
  IF(LINES) THEN
    IF(SET_STEP.LT.2) &
&     STOP ' METAL LINE BACKGROUND AND TCORR: MINIMUM VALUE FOR SET_STEP = 2!'
  ENDIF
  IF(LINES.AND..NOT.LINES_IN_MODEL) &
&   STOP ' TCORR AND LINES: SET LINES_IN_MODEL TO .TRUE.!'
ENDIF  

PRINT*,TRIM(MODNAM)
PRINT* 
PRINT*,' NUMBER OF PARAMETERS FOR CLUMPING FACTOR:', NPARA_CLF
PRINT*,' CORRESPONDING PARAMETERS:'
PRINT*,CLF_FIELD
PRINT*
IF(OPTTHICK) THEN
  PRINT*,' OPTICALLY THICK CLUMPING'
  PRINT*
ELSE
  PRINT*,' OPTICALLY THIN CLUMPING'
  PRINT*
ENDIF

OPEN (999,FILE=TRIM(MODNAM)//'/FASTWIND.LOG',STATUS='UNKNOWN', &
&                                            POSITION='APPEND')  
WRITE(999,*) ' FASTWIND VERSION ',VER_NLTE,    ' (nlte)'

OPEN  (2,FILE=TRIM(MODNAM)//'/INDAT.DAT',STATUS='UNKNOWN')  
REWIND 2  
WRITE (2,FMT=*) "'",MODNAM,"'"  
WRITE (2,FMT=*) OPTNEUPDATE,HELONE,ITLAST,ITMORE  
WRITE (2,FMT=*) OPTMIXED  
WRITE (2,FMT=*) TEFF,GGRAV,SRNOM  
WRITE (2,FMT=*) RMAX,TMIN  
WRITE (2,FMT='(E12.6,4(F10.4))') XMLOSS,VMIN,VMAX,BETA,VDIV  
WRITE (2,FMT=*) YHEIN,HEI  
WRITE (2,FMT=*) OPTMOD,OPTLUCY,MEGAS,ACCEL,OPTCMF1
WRITE (2,FMT=*) VTURB, XMET,LINES,LINES_IN_MODEL
WRITE (2,FMT=*) ENATCOR,EXPANSION,SET_FIRST,SET_STEP
IF(OPTTHICK) WRITE(2,FMT=*) 'THICK'
WRITE (2,FMT='(10(G10.4,2x))') (CLF_FIELD(I),I=1,NPARA_CLF)

IF (OPTTHICK) THEN 
   WRITE (2,FMT='(10(G10.4,2x))') (FIC_FIELD(I),I=1,NPARA_CLF)
   WRITE (2,FMT='(10(G10.4,2x))') (FVEL_FIELD(I),I=1,NPARA_CLF)
   WRITE (2,FMT='(10(G10.4,2x))') (HPOR_FIELD(I),I=1,NPARA_CLF)
ENDIF
!-----------------

! setup of TC
ITTCOR=ITLAST+SET_FIRST
OPTTCOR=.FALSE.

IF(ENATCOR) THEN
! TOTAL NUMBER OF T-CORRECTIONS
NCOR=-1
EMAXTC=1.D6
  IF(ITLAST .EQ. 0) THEN
  OPEN (777,FILE=TRIM(MODNAM)//'/MAXTCORR.dat',STATUS='REPLACE')
  ELSE
  OPEN (777,FILE=TRIM(MODNAM)//'/MAXTCORR.dat',STATUS='UNKNOWN')
  REWIND 777
  DO
! IF RESTART FROM A MODEL WITH T-CORR, READ T-CONVERGENCE FROM PREVIOUS RUN
    READ (777,*,END=10) NCOR,EMAXTC
  ENDDO
ENDIF

! position file just before EOF
10 BACKSPACE 777
  NCOR=NCOR+1 !

! CHECK WHETHER PREVIOUS RUN HAS BEEN CONVERGED
IF(EMAXTC .LT. 3.D-3) THEN
     ENATCOR=.FALSE.
     TEMP_CONVERGED=.TRUE.
   ENDIF
ENDIF

PRINT *
PRINT *, ' >>> VTURB =', VTURB,' km/s'
PRINT *      
!
!----vturb in cm/s
!
VTURB = VTURB*1.0D5 
!
!     definition of hopfparameters
!
IF (.NOT.OPTLUCY) THEN  
     READ (1,FMT=*) HOPFSELF  
     WRITE (2,FMT=*) HOPFSELF  
     IF (HOPFSELF) THEN  
          CALL HOPFPARA(TEFF,GGRAV,YHEIN,XMET,XMLOSS,VMAX,SRNOM,CLF_FIELD(1)) 
     ELSE  
          READ (1,FMT=*) QINF,Q0,GAMHOPF  
          WRITE (2,FMT=*) QINF,Q0,GAMHOPF  
     END IF  
END IF  
!
! prespecified abundances (explicit and background elements), EXCEPT FOR He
! if explicit element, abundance from DETAIL input file will be overwritten
! if background element, rescaled solar abundance will be overwritten
!
! input RELATIVE TO HYDROGEN and ALWAYS LOGARITHMICALLY
! either with respect to H=12 (positive number)
! or directly, log10(abund/H), i.e., negative
!
! if log10(abund/H) gt 0 (e.g., for Y > 1), use only first possibility
! (with respect to H=12), since positive numbers always interpreted in this way
!
! NOTE THAT THESE NUMBERS WILL BEUSED DIRECTLY, i.e.,
! NOT RENORMALIZED IN ROUTINE ABUNDAN
!
OPTXRAY=.FALSE.
NAT_TOT=0
DO I=1,31
  READ(1,FMT=*,END=15) NAMES_CHANGED(I),ABCHANGED(I)
! TAKE CARE: if all 30 elements would be changed, the line containing
! the x-ray parameters would not be read
  IF(NAMES_CHANGED(I).EQ.'XRAYS') THEN
    OPTXRAY=.TRUE.
    FX=ABCHANGED(I)
    GOTO 15
  ENDIF  
  NAT_TOT=NAT_TOT+1
ENDDO  

15 CONTINUE

IF (NAT_TOT.GT.0) THEN
PRINT*
PRINT*,'NUMBER OF ELEMENTS WITH SPECIFIED ABUNDANCES = ',NAT_TOT
PRINT*
DO I=1,NAT_TOT
  WRITE(2,FMT=*) NAMES_CHANGED(I),ABCHANGED(I)
  IF(ABCHANGED(I).GT.0.) ABCHANGED(I)=ABCHANGED(I)-12.D0
  ABCHANGED(I)=10.D0**ABCHANGED(I)
  PRINT*,'ELEMENT = ',NAMES_CHANGED(I),&
  'prespec. abundance (relative to H) = ',ABCHANGED(I)
ENDDO  
PRINT*
ENDIF

OPEN (3,FILE=TRIM(MODNAM)//'/INXRAY.DAT',STATUS='UNKNOWN')  
IF(OPTXRAY) THEN 
   IF(.NOT.OPTMET) STOP ' XMET = 0., BUT XRAY-TREATMENT'
   IF(TEFF.LT.25000.) STOP ' TOO LOW TEFF FOR XRAY-TREATMENT'   
  !UINFX IN KM/S
   READ (1,FMT=*) GAMX,MX,RMINX,UINFX,PX  
   REWIND 3  
   WRITE (3,FMT=*) OPTXRAY,FX
   WRITE (3,FMT='(5(G10.4,2x))') GAMX,MX,RMINX,UINFX,PX  
   WRITE (2,FMT=*) 'XRAYS',FX 
   WRITE (2,FMT='(5(G10.4,2x))') GAMX,MX,RMINX,UINFX,PX  
ELSE
   WRITE (3,FMT=*) 'NO XRAY TREATMENT'
ENDIF 
  
CLOSE (1)  
CLOSE (2)
CLOSE (3)

IF (OPTXRAY) CALL READ_KSHELL_DATA

IF (OPTCMF1 .AND. OPTMIXED.NE.1. .AND. &
&    OPTMIXED.NE.0.) STOP 'ERROR IN OPTMIXED'
!
!***  check for consistency: file indexcmf has to present under certain
!***  conditions
!
IF (OPTMIXED.EQ.1 .AND. ITLAST.GE.20) THEN  
     OPEN (1,ERR=310,FILE=TRIM(MODNAM)//'/INDEXCMF',STATUS='OLD')  
     CLOSE (1)  
END IF  

OPEN (8,FILE=TRIM(MODNAM)//'/CONVERG',STATUS='UNKNOWN',FORM='FORMATTED')  

IF(OPTXRAY) OPEN (9,FILE=TRIM(MODNAM)//'/XLUM_ITERATION',STATUS='UNKNOWN',FORM='FORMATTED')  
OPEN (776,FILE=TRIM(MODNAM)//'/CONVERG_METALS',STATUS='UNKNOWN',FORM='FORMATTED')  
!
!**** initializing of iqual (for non-treated levels)
!
IQUAL=0

CALL START(NREC,TEFF,YHEIN)

!JO Sept. 2021
!check whether file ADD_FREQUENCIES exist, and delete if starting with iteration 0
NSTARK=0 !initialize
IF(OPTSTARK) THEN

INQUIRE(FILE=TRIM(MODNAM)//'/ADD_FREQUENCIES', EXIST=EXI)
IF(EXI) THEN
  IF(ITLAST.EQ.0) THEN
    OPEN(1,FILE=TRIM(MODNAM)//'/ADD_FREQUENCIES', STATUS='OLD')
    CLOSE(1,STATUS='DELETE')
    PRINT*
    PRINT*,'old file ADD_FREQUENCIES deleted'
  ELSE
    OPEN(1,FILE=TRIM(MODNAM)//'/ADD_FREQUENCIES', STATUS='OLD')
    READ(1,*) NSTARK
    IF(NSTARK.GT.NSTARKMAX) STOP ' NSTARK > NSTARKMAX (after input)'
    DO I=1,NSTARK
      READ(1,*) WAVESTARK(I)
    ENDDO
    CLOSE(1)
    PRINT*
    PRINT*,NSTARK,' additional freq. points (Stark) read'
  ENDIF  
ELSE
  IF(ITLAST.NE.0) STOP ' RESTART, BUT FILE ADD_FREQUENCIES MISSING'
ENDIF

ENDIF
!
!.....loop for establishing approximate photospheric structure
!
VMIN1 = 0.D0  
XMMAX = 200.D0  

IIT = ITLAST + 1 ! new NLTE-cycle

1111 continue

IF (ITLAST.EQ.0 .AND. .NOT. OPTTCOR) THEN  
     ICONVER = 0  
     ITHYDS = 1  
     OPTMODEL=.FALSE.
     IF (OPTMOD) THEN
       PRINT*,' External photospheric structure will be used'
       OPTMODEL=.TRUE.
       OPTMOD=.FALSE.
     ENDIF  
ELSE  
!
!.....start from previous model!  ... or T-corr!
!.....check whether file 'temp' was previously written
!.....to allow restart in old philosophy
!
     REWIND 21  
     READ (21,*,ERR=40) RTEMP(ND)  
     ICONVER = 1  
     ITHYDS = 29  
     OPTMOD = .TRUE.  
     GO TO 50
!     
   40      CONTINUE  
    PRINT *, &
&      ' HYDRO MODEL STARTED FROM SCRATCH, SINCE NO FILE "TEMP"'
     ICONVER = 0  
     ITHYDS = 1  
     OPTMODEL=.FALSE.
     IF (OPTMOD) THEN
       PRINT*,' External photopsheric structure will be used'
       OPTMODEL=.TRUE.
       OPTMOD=.FALSE.
     ENDIF  
   50      CONTINUE  
END IF  


ITHYDRO: DO ITHYD = ITHYDS,30  

     IF (ICONVER.EQ.1) THEN  
          IF (ITLAST.EQ.0.AND. .NOT. OPTTCOR) STOP ' ERROR IN START MODEL'  
          GO TO 70  
     END IF  

     RELKAP = 1.D0  
     RELEX = 1.D0  
     IF (ITHYD.GT.3) THEN  
          RELKAP = ABS(1.D0-XKAPITH(ITHYD-1)/XKAPITH(ITHYD-2))  
          RELEX = ABS(1.D0-EXITH(ITHYD-1)/EXITH(ITHYD-2))  
          RELHYD = MAX(RELKAP,RELEX)  
          IF (RELHYD.LT.0.01D0) ICONVER = 1  
! this hack forces convergence if H-recombination oscillates
          IF (TEFF.LT.12000..AND.ITHYD.GE.20) ICONVER = 1
!
          IF (UPDATE_STRUCT .AND. ITHYD .GE. 20) THEN
! this hack forces convergence if oscillations of different origin present
! (but only if correlation high enough)
! limit of 20 and corr > 0.97 from inspection of ob_grid
! should not matter since structure updated later on anyway
            IF (ABS(CORRITH(ITHYD-1)) .GE. 0.97) ICONVER = 1
          ENDIF  
     END IF  
!
!---- corvm added to correct (if neccessary) vmin to get proper
!---- tau-ross maximum (e. santolaya, 11/2/94)
     OPTCV = .FALSE.  
   60      CONTINUE  
   
     PRINT *  
     PRINT *,' -------------- START VALUES FOR NE,TEMP,TAU-E-------------'
     PRINT *,' ---------------ITHYD = ',ITHYD 
     PRINT *  
!
!     input/output: ACTUAL XNE (lte), temp values (ne,nh corrected for clumping)
!     this routine is called only at iteraton 0!

     CALL MODEL(TAUE,VMIN1,.TRUE.,ITHYD,XKAP,EXKAP,TEMP,XMMAX, &
      RTAU23,XNH,XNELTE,VDIV,NDIV)
!---- here's where parameters get defined (incl. clfac) 
     PRINT *  
     PRINT *,' -------------------------------------------------------------'
     PRINT *  

     IF (ICONVER.EQ.1) THEN
        PRINT *  
        PRINT *,' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        PRINT *,'            FINAL ITERATION NO . ',ITHYD,' FOR HYDRO MODEL'
        PRINT *,' +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        PRINT *  
     END IF  
!
!---- file 'model' is read
!
   70      CONTINUE  
     REWIND 20  
!
!           XNE(lte),nh corrected for clumping
!           last entry is now NDIV
! 

     PRINT*
     PRINT*,' MODEL FILE READ'
!xnelte dummy except for first iteration(s) 
     READ (20, ERR=75) TEFF,GGRAV,SR,YHE,XMU,VMAX,XMLOSS,BETA, &
&      (R(I),I=1,ND), (VELO(I),I=1,ND), (DVDR(I),I=1,ND), (RHO(I),I=1,ND), &
&      (XNELTE(I),I=1,ND), (XNH(I),I=1,ND), (CLFAC(I),I=1,ND), &
&      (INDEX(I),I=1,ND), (PRESSURE(I),I=1,ND), NDIV_CALC, UPDATED, XT
!
     GOTO 76
!    OLD VERSION OF MODEL FILE, SET UPDATE_STRUCTURE TO FALSE 
75   UPDATE_STRUCT=.FALSE.
     PRINT*,' WARNING WARNING WARNING!'
     PRINT*,' OLD VERSION OF MODEL'
     PRINT*,' UPDATE-STRUCT SET TO FALSE!'
     PRINT* 
!---- comment upper lines and uncomment following line if UPDATE_STRUCT for models
!---- with old version of MODEL-file should be performed
!75   UPDATED=.FALSE.  

76   CONTINUE

!---- in case, read in clumping params
!---- standardwise, not required, since quantities stored in  module nlte_porvor
!     Note that when skipping this read, the solution will be slightly different,
!     since CLUMPING_OUTPUT is written with a lower precision than present in the module
!     reading in the parameters is required after restart!
   
     IF(OPTTHICK.AND.ITLAST.NE.0.AND.IIT.EQ.ITLAST+1) THEN

        OPEN(24, FILE=TRIM(MODNAM)//'/CLUMPING_OUTPUT', STATUS='UNKNOWN')
        READ(24,*) !header
        DO L=1,ND1 
          READ(24,FMT=*) I,CO_DUM,CO_DUM,CO_DUM,CO_DUM, &
            FIC(L),FVEL(L),HPOR(L),FVOL(L),TCL_FAC_LINE(L),TCL_FAC_CONT(L)
        ENDDO
        CLOSE(24) 
        PRINT* 
        PRINT*,' OPTTHICK AND RESTART: CLUMPING PARAMETERS READ' 
     ENDIF
        
     PRINT*
     VSOUND=SQRT(TEFF*AKB/(XMU*AMH))
     PRINT*,' VSOUND = ',VSOUND/1.D5,' KM/S'

!
!---- set up of p-z-grid
!
     CALL PZGRID(ND,NP,R)  
!
!---- set up of electron temperature
!
!---- file 'temp' is read
!---- Temp updated in LTE (hydro-cycle) or in TEMPCORR_MAIN
!---- xnelte updated in LTE
!     
     REWIND 21  
     READ (21,*) (RTEMP(I),I=ND,1,-1), (TEMP(I),I=ND,1,-1)  
     DO L = 1,ND    
        IF (ABS(RTEMP(L)-R(INDEX(L))).GT. &
&            1.D-10) STOP ' RTEMP NOT COMPATIBLE WITH R'
     END DO
!
!.....input of XNE(lte) from converged hydro-model (in case of No T-correction)
!     (actually, value from next to last iteration, to ensure restart with identical numbers)
!     and rtau23.
!
!     in the case of T-correction, xnelte has been updated in TEMPCORR_MAIN
!     and refers to the last T-iteration 
     IF (ICONVER.EQ.1 .AND. OPTMOD) THEN  
          READ (21,*) (XNELTE(I),I=ND,1,-1)  !includes clumping
          READ (21,*) RTAU23  
     END IF  

!
!-----update of photo-stratification 
!
     IF (LTE_UPDATE.AND.UPDATE_STRUCT) THEN
       IF(.NOT.UPDATED.AND.IIT.GE.IITUPDATE.AND.EMAXTC.LT.0.03) THEN
!-----here, SRNOM is Rstar (in Rsun), and GGRAV is 10^logg
!-----needs to be transfered into module nlte_var inside routine
         CALL MODVL_UPDATE(ND,R,VELO,DVDR,RHO,XNE,XNELTE,XNH,TEMP,TAUE,TAUR,CLFAC, &
&          TEFF,GGRAV,SRNOM,XMLOSS,BETA,NDIV_CALC,YHEIN,YHE,HEI,RTAU23,UPDATE_DONE)

!-----update p-z-grid
         IF(UPDATE_DONE) THEN
           CALL PZGRID(ND,NP,R)  
!-----relax for 4 additional iterations
!     immediate update destroys lower boundary
           ITTCOR=ITTCOR+4  
!-----reset all start variables
           START_WEIGH11=.TRUE.
           START_CMFSING=.TRUE.
           START_CMFMULTI=.TRUE.
           START_JBAR_METALS=.TRUE.
           START_CMF_SIMPLE=.TRUE.
! comment: recalculate lambda
         ENDIF
       ENDIF
     ENDIF  

!
!-----set up of lte numbers and spherical grey temperature (if grey =
!-----.true.)
!
     GREY = .NOT. OPTMOD  

!    to obtain consistent temperature stratification
!    this is the new statement from ver 7.3 (restart capab. not tested)
     IF(ICONVER.EQ.1.AND.LINES) LINES_IN_MODEL=.TRUE.

     RESTART_MET_LTE=.FALSE.
!    if METAL-RESTART, to obtain consistent metal-opacities
     IF(ITLAST.NE.0 .AND. IIT.EQ.ITLAST+1 .AND. OPTMET) RESTART_MET_LTE=.TRUE.
!    xnelte is updated and transfered to "common block" in module nlte_var and
!    written to TAU_ROS
     CALL LTE(TEFF,SR,TAUR,TAUE,R,VELO,DVDR,RHO,XNELTE,XNH,TEMP,DILFAC,CLFAC, &
&      ILOW,IMAX,ND,GREY,CORRFC,NTEMP,CORVM,OPTCV,ITHYD,XKAP, EXKAP, &
&      XMMAX,RTAU23,XMET,RESTART_MET_LTE,LTE_UPDATE)
!
!---- from here on, temp = actual non-lte temperature
!
     IF (OPTCV .AND. ITHYD.EQ.1) THEN  
          VMIN1 = VELO(ND)*CORVM*VMAX*1.D-5  
          PRINT *,' '  
          PRINT *,'CORRECTION TO VMIN; NEW VMIN (KM/S)=',VMIN1  
          PRINT *,' '  
          GO TO 60  
     END IF  
!
!    check whether all operations concerning XNELTE where successfull
!
     DO I=1,ND
       IF(XNELTE(I).NE.XNELTE_COMMON(I)) STOP ' XNELTE .NE. XNELTE (common)'
     ENDDO
     OPEN (1,FILE=TRIM(MODNAM)//'/TAU_ROS',STATUS='UNKNOWN',FORM='FORMATTED')  
     REWIND 1
     DO I=1,ND
       READ(1,*) TAUR_TAUR(I),XNELTE_TAUR(I)
       IF(TAUR(I).NE.0.D0.AND.ABS(1.-TAUR(I)/TAUR_TAUR(I)).GT.1.D-13) &
&        STOP ' TAUR .NE. TAUR IN TAU_ROS'
       IF(ABS(1.-XNELTE(I)/XNELTE_TAUR(I)).GT.1.D-13) &
&        STOP ' XNELTE .NE. XNELTE IN TAU_ROS' !not exact, since formattedly written/read 
     ENDDO
     CLOSE (1)
!
     IF (ICONVER.EQ.1) GO TO 110  
END DO ITHYDRO  

PRINT*, ' ITERATION HISTORY OF KRAMER OPACITY REGR. COEFF.'  
PRINT*  
DO L = 1,ITHYD - 1  
     WRITE(*,105) L,XKAPITH(L),EXITH(L),CORRITH(L)  
     PRINT *  
END DO  

WRITE(999,*) ' ITERATION HISTORY OF KRAMER OPACITY REGR. COEFF.'  
WRITE(999,*)
DO L = 1,ITHYD - 1  
     WRITE(999,*)
     WRITE(999,105) L,XKAPITH(L),EXITH(L),CORRITH(L)  
END DO  
105 FORMAT(' IT.NO. ',I2,' XKAP = ',E12.6,' EX = ',F10.5,' CORR = ',F10.5)

STOP ' WARNING!!!! HYDRO NOT CONVERGED IN 30 ITERATIONS'  
!
!-----radiative transfer to obtain start values for j
!
  110 CONTINUE  
!
!     final check for photospheric mass < fraction of total mass
!
CALL CHECKMASS(SR,GGRAV,RTAU23,R,RHO,ND,NDIV,NDIV_CALC)  

FRENEW = .TRUE.  
!
!---- u-function tabular values are read and stored (only once)
!
CALL ULECTURE  
!
!---- u2-function tabular values are read and stored (only once)
!
CALL U2LECTURE

ERROLD = PRECIS  

SPECMAT = 1.D0  
!
!     calculation of fine frequency grid
!
CALL FRESFIN(ND,XNH,TEMP(ND),.TRUE.,TEFF)  
!
!     redefinition of dilfac to ensure thermalization in nlte_approx
!
IF(OPTMET) THEN
  WHERE(DILFAC.LT.0.5)
    DILFAC1=DILFAC
  ELSEWHERE
    DILFAC1=1.
  ENDWHERE
  DO L=1,ND
    IF(DILFAC1(L).EQ.1.) THEN
      DILFAC1(L)=0.75 !smooth transition
      EXIT
    ENDIF
  ENDDO
ENDIF  

LWION=L+1
! check whether TAURLIM is outside metal-lte range or too small
IF (ENATCOR .AND. OPTMET) THEN
    IF(TAURLIM.LT.TAUR(NDIV_CALC+1)) TAURLIM=TAUR(NDIV_CALC+1) !STRONG WIND
    IF(TAURLIM.GT.2.) TAURLIM=2. !MAXIMUM VALUE EVEN FOR STRONG WINDS
    IF(TAURLIM.GT.TAUR(LWION)) TAURLIM=TAUR(LWION) 
    PRINT*,' TAURLIM        = ',TAURLIM,'   (DIVISION OF T-CORR SCHEMES)'
ENDIF
PRINT*,' TAUROSS(LWION) = ',TAUR(LWION),' (DIVISION OF LTE/NLTE for non-seclected METALS)'

! specify transition point for selected ions
DO L=LWION,ND
  IF(TAUR(L).GT.2.D0) EXIT 
!  IF(TAUR(L).GT.10.D0) EXIT  ! for tests
ENDDO

LWION1 = L
PRINT*,' TAUROSS(LWION1) = ',TAUR(LWION1),' (DIVISION OF LTE/NLTE for seclected METALS)'
PRINT*,'FROM NLTE: LWION = ',LWION
PRINT*,'FROM NLTE: LWION1 = ',LWION1

IF (ITLAST.NE.0.OR.OPTTCOR) GO TO 160  

PRINT *,'               ++++++++++++++++++++++++++'  
PRINT *,'               +     ITERATION NO 0     +'  
PRINT *,'               ++++++++++++++++++++++++++'  
PRINT *  

! XNE is input as LTE 
IF (OPTTCOR) STOP ' SOMETHING WRONG WITH OPTTCOR'
LASTLTE=.TRUE.
RYBICKI = .TRUE.

! JO Oct. 2021: here was a bug (IIT missing)
CALL CONT(ND,NP,R,RHO,XNELTE,TEMP,CLFAC,TEFF,SR,CORRFC,OPTLTE,CONCON,TAUR, &
&         RTAU23,RYBICKI,DTFCORR,FLUXERR,NDIV_CALC+1,IIT)

IF (OPTMET) THEN
     PRINT*,' BEGIN OF APPROXIMATE NLTE FOR METALS (It 0, xne(LTE))'
     CALL TRAD(ND,TEMP,DILFAC1,XNELTE,CLFAC,IIT,OPTCMF1,EMAX,LASTLTE)

     
     CALL NLTE_APPROX(TEFF,YHE,XMET,RHO,XNELTE,DILFAC1,TEMP, &
&       DVDR*VMAX/SR,VELO/R*VMAX/SR,VELO*VMAX,CLFAC,ND, &
&       FRENEW,.FALSE.,1,.TRUE.)

     
     CALL OPACITM(ND,XNELTE,TEMP,CLFAC,.FALSE.,1,.TRUE.,FRENEW,.TRUE.)

     PRINT*,' END OF APPROXIMATE NLTE FOR METALS'
     PRINT* 
ENDIF

! INPUT: LTE, OUTPUT NLTE (updated in UPDATE!)
XNE=XNELTE
!print*,'xnecheck: iteration zero'
!print*,xnelte
!print*,xne
!
! here, we do not need ilow and imax for the metals, since they
! are calculated only from the line-iteration on (CONCON = T).
! ilow and imax (H/He) from ILOWIMAX_LTE
!
CALL RATEEQ(XNH,XNE,TEMP,CLFAC,ILOW,IMAX,ND,ERR,CONCON,R,VELO,DVDR,SR, &
&            VMAX,ERROLD,SPECMAT,ACCEL,OPTMET)

FRENEW = .FALSE.  

ERROLD = ERR  

160 CONTINUE  

IF (ITLAST.NE.0 .OR. OPTTCOR) THEN
!
!   if start from existing model, then update of electron density
!   and (ni/nk)*, which are normally updated in UPDATE, but at this
!   stage defined by the lte-electron density.
!   input XNE(lte), output XNE(nlte)
!
  PRINT*,' RESTART FROM PREVIOUS ITERATION'
  XNE=XNELTE
! ilow and imax only dummy arguments here, to preserve RESTART
! locate new levels for .not.(concon.and.optmet)  
  CALL RESTART(ND,XNE,DILFAC1,OPTMET,ILOW,IMAX)
! here was the bug; update nlte electron density by actual lte e-density
! could be done also in subroute copyocc, but here it's more clear
  IF(.NOT.OPTNEUPDATE) XNE=XNELTE
!  print*,xnelte
!  print*,xne
!
! to obtain MET_IMIN and MET_IMAX
! NOTE: Here, in contrast to standard approach, subr. select works
! works on mixed ionization fractions:
! ilow to imax from 'exact', rest from approx. method.
! This might lead to certain inconsistencies after restart. 
  IF (OPTXRAY) CALL SELECT(.FALSE.)
 
ENDIF

OPTLTE = .FALSE.  
NTEMP1 = 1  
EMAX = 1000.D0  

! Begin of the NLTE-cycle iteration  
11111 IF (IIT.EQ.ITLAST+ITMORE+1) GOTO 22222
! from here on, only NLTE values for XNE as input

    PRINT *,'               ++++++++++++++++++++++++++'  
    PRINT *,'               +  ITERATION NO ',IIT,'  +'  
    PRINT *,'               ++++++++++++++++++++++++++'  
    PRINT *  


     OPTCMF = OPTCMF1  
     IF (IIT.LE.20 .AND. .NOT.FASTSOBO) THEN  
          OPTCMF = .FALSE.  
!
!     if fast convergence of sobo
!
          IF (CONCON .AND. OPTCMF1 .AND. EMAX.LE..1 .AND.  &
&             (.NOT. UPDATE_STRUCT .OR. UPDATED) ) THEN  
               FASTSOBO = .TRUE.  
               OPTCMF = .TRUE.  
               IITCMF = IIT  
          END IF  
     END IF  

     IF (IIT.GT.3) CONCON = .TRUE.  
!
!     if fast convergence of continuum
!
     IF (IIT.LE.3 .AND. EMAX.LE..1) THEN  
          CONCON = .TRUE.  
          IITSOBO = IIT  
     END IF  
!
!---- complete output for the last iteration
!
     IF (IIT.EQ.ITLAST+ITMORE) UNASOL = .TRUE.  
!
!---- file for complete output
!
     IF (UNASOL .AND. MEGAS) OPEN (22,FILE=TRIM(MODNAM)//'/megasalida', &
      STATUS='UNKNOWN', FORM='FORMATTED')

!     print*,'xnecheck: normal iterations'
!     print*,xne

     IF(.NOT.CONCON) THEN
!    only H/He values for ilownl/imaxnl are needed, 
!    as well as LTE values for ilow/imax.
!    These have been calculated already in ILOWIMAX_LTE
        ILOWNL=ILOW
        IMAXNL=IMAX
     ENDIF
!    if CONCON, metals are switched on; thus we need ILOWNL and IMAXNL
!    for all routines, including OPACITC.
!    Usually, we obtain them from our approximate NLTE solution
!    (met_imin, met_imax, previous iteration).
!    Only inn case z=0, we have already calculated them in ILOWIMAX_OLD.
!    ILOWNL, IMAXNL must be updated only when the temperature has changed, 
!    and we are not close to convergence. THUS

!JO changed Sept. 2014, to allow for better guess
!     IF (CONCON .AND. (FIRST_METALS.OR. (LTE_UPDATE .AND. EMAXTC.GE.0.05))) THEN
     IF (CONCON .AND. (FIRST_METALS.OR. (LTE_UPDATE .AND. EMAXTC.GE.0.03))) THEN
         IF (FIRST_METALS) FIRST_METALS=.FALSE.
         IF (OPTMET) THEN 
!--- this is the subroutine to fiddle around if you are not satisfied with
!--- ILOW and/or IMAX !!! (in nlte_approx.f90)
           CALL ILOWIMAX_NLTE(ILOWNL,IMAXNL,ILOW,IMAX, &
&            TEMP,TEFF,ND,OPTCMF1,NDIV_CALC)

         ELSE   
           ILOWNL=ILOW
           IMAXNL=IMAX
           IMIANL=IMIA
           IMAANL=IMAA
         ENDIF
     ELSE
! Usually, ILOWIMAX_NLTE ensures that IMAX(NLTE) always LE IONIS(LTE).
! Under very conspicous circumstances, however, IONIS might decrease
! even if the the temperature correction becomes low
! (remember the first CNOP test).
! In this case then, IMAXNL would be too large since no LTE info would
! be present any longer (leads to NISTAR = infinity).
! Thus have to perform the following hack
     DO K=1,NAT
       DO L=1,ND
         IF(IMAXNL(L,K).GT.IONIS(K,L)) THEN
           IMAXNL(L,K)=IONIS(K,L)
           IF(IMAXNL(L,K).LT.ILOWNL(L,K)) STOP ' IMAXNL < ILOWNL'
           PRINT*,'CORRECTION OF IMAX(NLTE) REQUIRED!!!'
           PRINT*,K,L,' IMAX(NLTE) = ',IMAXNL(L,K),' (ZEFF CORRECTED)'
         ENDIF  
       ENDDO
     ENDDO  
     ENDIF

! calculate run of shock-temperatures
     IF(OPTXRAY) THEN
        IF(START_XRAY) THEN
! needs only to be iterated when the change in ne should be accounted for
! in this case, LXMIN might change as well
! JO: comment changed July 2016
! BUT: ne is not changing, since it refers to the hot plasma;
! and the total wind-structure is NOT changing even after hydro-update,
! since only photosphere affected by this update.
! THUS: Indeed, only one call of TSHOCK and LAMBDA_XRAY required 
! inconsistently, we use XMU from the cool wind. Does not matter! 
          CALL TSHOCK(TSHO,VELO,R,RTAU23,XMU,ND)
! calculate X-ray cooling functions accounting for cooling zones, both for
! radiative and adiabatic shocks, following Feldmeier et al. 1997
! same situation as for TSHOCK
          CALL LAMBDA_XRAY(XMU,TSHO,R,VELO,RHO,XNELTE,XNH,CLFAC,ND)
          CALL INTERPOL_XRAY !usually, in the first call frenew = false
          START_XRAY=.FALSE. 
        ENDIF
!
! this needs to be reevaluated when frequency grid changes 
        IF(FRENEW) CALL INTERPOL_XRAY
     ENDIF
     
     CALL OPACITC(ND,XNE,XNH,TEMP,CLFAC,ILOWNL,IMAXNL,OPTLTE,NTEMP1,.TRUE., &
     FRENEW,OPTMET)
!     STOP 'TESTING-JS!'
! THICK: HERE OPAC GETS UPDATED; it is the critical part of the effective opacity inclusion! 
!
!     rybicki solution only if larger changes in continuum to be ex-
!     pected, i.e. after first nlte-solution of sobolev and cmf case
!
     RYBICKI = .FALSE.  
     IF (IIT.EQ.ITLAST+1) RYBICKI = .TRUE.  
     IF (IIT.EQ.IITSOBO+1) RYBICKI = .TRUE.  
     IF (IIT.EQ.IITCMF+1) RYBICKI = .TRUE.  
     IF (OPTTCOR) RYBICKI = .TRUE.  

     PRINT *  
     IF (RYBICKI) THEN  
          PRINT *,' CONTINUUM WITH RYBICKI !!!! '  
     ELSE  
          PRINT *,' CONTINUUM WITHOUT RYBICKI '  
     END IF  
     PRINT *  

     CALL CONT(ND,NP,R,RHO,XNE,TEMP,CLFAC,TEFF,SR,CORRFC,OPTLTE,CONCON,TAUR, &
&              RTAU23,RYBICKI,DTFCORR,FLUXERR,NDIV_CALC+1,IIT)
                           
     LASTLTE=.FALSE.
     IF(OPTTCOR) LASTLTE=.TRUE. !previous LTE-update 

     OPTTCOR = .FALSE.
! no T-correction in final iteration (to obtain consistent temp. and LTE-occ.num.)
     IF (ENATCOR .AND. IIT.EQ.ITTCOR .AND. .NOT.UNASOL) THEN
       OPTTCOR = .TRUE.     ! enables the update of the LTE numbers and
                            ! calculation of partial derivatives
       CALL TEMPCORR_START  ! starting up all the ThB quantities	
     ENDIF	

     IF (OPTMET.AND..NOT.METALS_CONVERGED) THEN
       PRINT*,' BEGIN OF APPROXIMATE NLTE FOR METALS'
       CALL TRAD(ND,TEMP,DILFAC1,XNE,CLFAC,IIT,OPTCMF1,EMAX,LASTLTE)

       CALL NLTE_APPROX(TEFF,YHE,XMET,RHO,XNE,DILFAC1,TEMP, &
&       DVDR*VMAX/SR,VELO/R*VMAX/SR,VELO*VMAX,CLFAC,ND, &
&       FRENEW,OPTLTE,NTEMP1,.TRUE.)
                           
       CALL OPACITM(ND,XNE,TEMP,CLFAC,OPTLTE,NTEMP1,.TRUE.,FRENEW,.FALSE.)
       PRINT*       
     ENDIF
!
!---  calculation of rbb-rates (explict elements) from old occup.numbers
!---  for all depth points (cmf or sobolev)
!
     IF (CONCON) THEN  
          CALL RATELI(XNE,TEMP,CLFAC,ILOWNL,IMAXNL,ND,CONCON,R,VELO,DVDR,SR, &
           VMAX,TEFF,CORRFC)
          PRINT *  
          PRINT *,' RBB-RATES FOR EXPLICIT ELEMENTS PREPARED (RATELI)'  
          PRINT *  
     END IF       
!
!--- now, full rateeq including actual values ilownl, imaxnl
! 
     CALL RATEEQ(XNH,XNE,TEMP,CLFAC,ILOWNL,IMAXNL,ND,ERR,CONCON,R,VELO,DVDR, &
      SR,VMAX,ERROLD,SPECMAT,ACCEL,OPTMET)

     IF (ENATCOR .AND. IIT .EQ. ITTCOR .AND. .NOT. UNASOL) THEN    
	CALL TEMPCORR_MAIN(TEMP,XNE,TAUR,DTFCORR,FLUXERR,TEFF,NCOR,EMAXTC, &
&		EXPANSION,OUTTHB)
!        WRITE (777,FMT='(1X,I3,1X,G12.5)') NCOR,EMAXTC 
        WRITE (777,FMT='(1X,I3,1X,G12.5,1X,I3)') NCOR,EMAXTC,IIT 
        IF(EMAXTC.LT.3.D-3 .AND. IIT.GT.21) THEN ! prevent convergence in Sobo cycle
          PRINT*,' !!!!!!!!!!!!!!!!!!!!!'
          PRINT*,' TEMPERATURE CONVERGED'
          PRINT*,' !!!!!!!!!!!!!!!!!!!!!'
          ENATCOR=.FALSE. !no more temperature-corrections required
          TEMP_CONVERGED=.TRUE.
        ENDIF
	ITTCOR=ITTCOR+SET_STEP ! iteration for the next T-Corr
	NCOR=NCOR+1            ! number of performed T-Corr 
! At this point, T has change and the TEMP file has been also written 	
	print*,' '
     ENDIF	
	
     IF (CONCON) THEN  
          DO J = 1,4  
             SPECMAT(J,:) = SPECMAT(J+1,:)  
          END DO

          DO L = 1,ND  
               IF (ERR(L).GT.3.D-3.and.ERROLD(L).NE.0.) THEN
                    SPECMAT(5,L) = ERR(L)/ERROLD(L)  
               ELSE  
                    SPECMAT(5,L) = 0.D0  
               END IF  
          END DO 
     END IF  

     ERROLD = ERR  

     FRENEW = .FALSE.  

     EMAX = 0.D0  

     DO IN = 1,ND  
          EMAX = MAX(EMAX,ERR(IN))  
     END DO

     PRINT *,' CORR. MAX: ',EMAX  
     PRECIS = MIN(PRECIS,EMAX/10.D0)  

     MEANERR=SUM(LOG10(ERR(1:LWION)))/LWION
     
! UPDATED AND NOT OPTCMF_FULL (FAST SOLUTION)
! JO changed Jan 2016 from -2 to -2.5 
     IF(MEANERR.LE.-2.5D0) THEN
         ALMOST_CONVERGED=.TRUE.
         PRINT*,' LOG MEANERR(1:LWION) = ',MEANERR,': ALMOST_CONVERGED'
         PRINT*
     ELSE
         ALMOST_CONVERGED=.FALSE.                        
         PRINT*,' LOG MEANERR(1:LWION) = ',MEANERR
         PRINT*
     ENDIF

     WRITE (8,  FMT=*) IIT,EMAX,' ',MEANERR  

!---- convergence output also for 
!NOTE: This tests the convergence *ONLY* within the range set by 
!wavblue wavred. *If* we want to test complete convergence, we might 
!include this in OPACITC, just before setting opa_eff_rat_old=opa_eff_rat  
!------------------------------------------------------------------
     CONV_EFF_OPA_RAT = MAXVAL(ABS((OPA_EFF_RAT-OPA_EFF_RAT_OLD)/OPA_EFF_RAT))
     PRINT *,' CORR. MAX FOR OPA_EFF_RAT: ',CONV_EFF_OPA_RAT 
     PRINT *
!NOTE: 

     WRITE (776,FMT=*) IIT,ERROR_MET,CONV_EFF_OPA_RAT    
     
     IF (UNASOL) GO TO 22222  

     IF(.NOT.ENATCOR.AND..NOT.OPTTCOR) THEN
!       IF (EMAX.LT.3.D-3) UNASOL = .TRUE.  
!JO Sept 2021
       IF (EMAX.LT.3.D-3 .OR. MEANERR.LE.-4.5) UNASOL = .TRUE.  
     ENDIF

     IIT=IIT+1  
     LTE_UPDATE=.FALSE.

!---- output OPA_EFF_RAT (for the total pseudo-continuum) 
!---- output created also for optically thin clumping, to check consistency
!---- analogous to OUTTHB, for additional printouts when testing 
     IF(OUTTHICK) THEN
        CALL NCOR_NAME(IIT,NCNAME)
     ELSE
        NCNAME='LA'
     ENDIF
     OPEN (1,FILE=TRIM(MODNAM)//'/OPA_EFF_RAT_'//TRIM(NCNAME), STATUS='UNKNOWN')
     WRITE(1,FMT=*) ND, IFRE
     DO I = 1,ND
        DO J = 1, IFRE
              WRITE(1,333) OPA_EFF_RAT(I,J),1.D8/FRE(J),R(I),TAUR(I),I
        ENDDO
     ENDDO
     CLOSE(1) 
333  FORMAT(f15.5,e20.10,f15.5,e20.10,i10)

!    UPDATE LTE-NUMBERS
     IF (OPTTCOR) then
       IF(OPTMET) CALL SAVE_METAL_INFO
       LTE_UPDATE=.TRUE. 
       GOTO 1111
     ENDIF



!    NEW CYCLE
     GOTO 11111


22222 CONTINUE  

! PRINT *,' CONVERG. NOT ACHIEVED IN',ITLAST + ITMORE,' IT. BY ',EMAX
!
!---- output for formal solution
!
CALL FORMAL(XNE,TEMP,CLFAC)  

IF(OPTMET) THEN
  CALL SAVE_METAL_INFO
! JO Sept. 2021
  IF(NSTARK.GT.0) CALL SAVE_ADD_FREQ_POINTS
ENDIF  
!
!     output of problematic levels
!
IQ = 0  
DO L = 1,ND  
   DO I = 1,ID_LLEVS
          IQ = MAX(IQ,IQUAL(I,L))  
   END DO
END DO

IF (CONV_EFF_OPA_RAT.GT.1.0D-3) THEN 
   PRINT *, ' WARNING!!! -- Maximum last relative correction in OPA_EFF_RAT > 1.d-3!' 
   PRINT *, ' View by 2nd column in CONVERG_METALS'
   PRINT *
ENDIF

IF (ENATCOR.OR.TEMP_CONVERGED) THEN
     PRINT *,' TOTAL NUMBER OF APPLIED T CORRECTION(S) ',NCOR
     PRINT *,' Maximum relative correction in the last TC ',EMAXTC
     PRINT *
     CLOSE(777)
ENDIF	      

!---- printing also cpu time, for testings 
CALL CPU_TIME(TIME_CPU) 
PRINT*
PRINT* , ' CPU time: ',TIME_CPU 

IF (IQ.EQ.0) THEN  
     PRINT *
     PRINT *,' ALL LEVELS OK!!!'       
     STOP '! ESTO ES EL ACABOSE !'  
END IF  

PRINT *  

DO 300 I = 1,ID_LLEVS  
     DO L = 1,ND  
          IF (IQUAL(I,L).NE.0) THEN  
               ISTART = L  
               GO TO 270  
          END IF  
     END DO 
     GO TO 300  

  270      CONTINUE  
     DO L = ND,1,-1  
          IF (IQUAL(I,L).NE.0) THEN  
               IEND = L  
               GO TO 290  
          END IF  
     END DO 

     STOP ' WRONG END CONDITION'  
  290      CONTINUE  
     WRITE (*,FMT=9000) LABL(I),I,ISTART,IEND,R(IEND)  

  300 END DO  
!
!    finally...
!  
CLOSE (8)   !CONVERG
IF (OPTXRAY) CLOSE (9) !XLUM_ITERATION
CLOSE (15)  !LTE_POT
CLOSE (20)  !MODEL  
CLOSE (21)  !TEMP
CLOSE (27)  !NLTE_POP
CLOSE (99)  !MAX_CHANGE
CLOSE (17)  !SCRATCH (actual pop)
CLOSE (40)  !SCRATCH (jnue,nf)
CLOSE (41)  !SCRATCH (hnue,nf)
CLOSE (42)  !SCRATCH (knue,nf)
CLOSE (43)  !SCRATCH (nnue,nf)
CLOSE (50)  !SCRATCH (jnue,nd)
CLOSE (51)  !SCRATCH (hnue,nd)
CLOSE (52)  !SCRATCH (knue,nd)
CLOSE (53)  !SCRATCH (nnue,nd)
CLOSE (60)  !CHIBAR_H
CLOSE (776) !CONVERG_METALS
! 777 MAXTCORR ALREADY CLOSED
CLOSE (999) !ERROR.LOG

PRINT *  
STOP '! ESTO ES EL ACABOSE !'  


  310 CONTINUE  

STOP ' OPTMIXED, ITLAST > 19 AND NO FILE INDEXCMF'  
 9000 FORMAT (' PROBLEMS WITH LEVEL ',A6,' (NO.=',I3,') FROM L = ',I2, &
&       ' TO ',I2,'(R = ',F10.5,')')

END
!
!***********************************************************************
!
! subroutines: simple ones
!
!***********************************************************************
!
SUBROUTINE CHECKMASS(SR,GGRAV,RTAU23,R,RHO,ND,NDIV,NS)  
! calculates photospheric mass and provides ns 
! new approach, NS saved in MODEL_FILE

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     note: sr is lowest radius in cgs, stellar radius is sr*rtau23
!
!     .. parameters ..
REAL(DP), PARAMETER :: GCONST=6.673D-8,XMSUN=1.989D33  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  GGRAV,RTAU23,SR  
INTEGER(I4B) ::  ND  
INTEGER(I4B), INTENT(IN) ::  NDIV !TRANSITION POINT (NE 0 IF START FROM SCRATCH) 
INTEGER(I4B), INTENT(OUT) ::  NS  !TRANSITION POINT (READ IN FROM MODEL_FILE)


!     ..
!     .. array arguments ..
REAL(DP) ::  R(ND),RHO(ND),DVDR(ND),XM(ID_NDEPT)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DR,FRAC,PI,STM1,STMASS,SUMM,XMPHOT,DM,DMOLD  
INTEGER(I4B) ::  I  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ACOS  
!     ..
PI = ACOS(-1.D0)  
STMASS = GGRAV* (SR*RTAU23)**2/GCONST  
STM1 = STMASS/XMSUN  


SUMM = 0.D0  
DO I = 1,ND - 1  
     DR = R(I) - R(I+1)  
     SUMM = SUMM + .5D0* (RHO(I)*R(I)*R(I)+RHO(I+1)*R(I+1)*R(I+1))*DR
END DO  

XM(1) = 0.D0  
DO I = 2,ND  
     XM(I) = XM(I-1) + .5D0* (RHO(I-1)+RHO(I))* (R(I-1)-R(I))  
! for tests
!     print*,i,rho(i),r(i),log10(xm(i))
END DO  

! FROM VERSION 9.0 ON, THE OLD METHOD IS NO LONGER WORKING
! (since point close to photosphere might be close to sonic point)
! Thus: NS read from file

! RECALCULATE TRANSITION POINT
! first region with constant dlog_m
!DMOLD=LOG10(XM(ND-1)/XM(ND))
!DO I=ND-1,2,-1 
!   DM=LOG10(XM(I-1)/XM(I))
!   print*,i,dmold,dm
!   IF(ABS(1.-DMOLD/DM).GT.0.15) EXIT
!   DMOLD=DM
!ENDDO
! 2nd region with constant dlog_m/3 (2 times) and dlog_m/6 (2 times)
!NS=I-4 ! cf. SUBROUTINE MODVL

IF(NDIV.NE.0) THEN
!  CHECK CONSISTENCY IN CASE OF START FROM SCRATCH
  IF(NS.NE.NDIV) THEN
    PRINT*,NDIV,' ', NS    
    STOP ' SOMETHING WRONG WITH TRANSITION POINT (CHECKMASS)!'
  ENDIF
ENDIF

PRINT* 
PRINT*,' TRANSITION POINT (WIND/PHOTOSPHERE) AT L = ',NS
!
!     note: r in units sr
!
XMPHOT = 4.D0*PI*SUMM*SR**3  
FRAC = XMPHOT/STMASS  

PRINT *  
PRINT *,' STELLAR MASS = ',STM1,' SOLAR MASSES'  
PRINT *,' PHOTOSPH. MASS = ',FRAC,' STELLAR MASSES'  
PRINT *  
PRINT *  
PRINT *,'-------------------------------------------------------------'
PRINT *  

IF (FRAC.GE..01) STOP ' TOO MUCH MASS IN PHOTOSPHERE'  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE COPYOCC(XNE,ND,ILOW,IMAX)  

USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: LI,LE
USE nlte_var, ONLY: XNELTE,BLEVEL,ENIONND,MODNAM,ALEVEL,OPTMET,CONCON

IMPLICIT NONE
!
!     update of ne-lte to ne-nlte, copying old nlte-numbers to new ones
!     changed from version 9.0 on
!
!     ilow and imax only dummy arguments, no longer used
!
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT, KEL=ID_ATOMS  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND  
!     ..
!     .. array arguments ..
REAL(DP) ::  XNE(ND1)
INTEGER(I4B) :: ILOW(ND1,KEL), IMAX(ND1,KEL)  

!     .. local scalars ..
INTEGER(I4B) ::  I,LL,NREC
!     ..
!     .. local arrays ..
REAL(DP) ::  XNENLTE(ND1)  
!     ..

NREC = ID_LLEVS  

OPEN (1,FILE=TRIM(MODNAM)//'/ENION',FORM='UNFORMATTED',STATUS='OLD')  

REWIND 1  
READ (1) ENIONND,XNENLTE
CLOSE (1)  

DO I=1,NREC
IF(BLEVEL(I).NE.ALEVEL(I)) STOP ' SOMETHING WRONG IN COPYOCC-START'
ENDDO

PRINT*
DO LL = 1,ND  
     READ (27,REC=LL) (BLEVEL(I),I=1,NREC)  

     IF(.NOT.OPTMET.OR..NOT.CONCON) THEN ! check for "new levels"
       READ (15,REC=LL) (ALEVEL(I),I=1,NREC) ! forgotten in versions earlier 
                                             ! than 9.0
! NOT REQUIRED FROM VER 9.5 ON
! above statement wrong. still required as long as ILOWIMAX_NLTE not
! called, i.e., in very first iterations (example: CV/CVI) 
!
       DO I=1,NREC
         IF(BLEVEL(I).EQ.0.) THEN
! VERY SMALL OCCUP. NUMBER
           BLEVEL(I)=ALEVEL(I)*1.D-10
           IF (ALEVEL(I).NE.0.) PRINT*,'CONCON = .FALSE., new level',ll,i
         ENDIF
       ENDDO  

     ENDIF  ! always: copy nlte numbers from last nlte iteration to local ones
     WRITE (17,REC=LL) (BLEVEL(I),I=1,NREC)  
     XNELTE(LL) = XNE(LL)  
     XNE(LL) = XNENLTE(LL)  
END DO  

!print*,'used in copyocc,'
!do ll=1,nd
!	print*,xnelte(ll), xne(ll)
!enddo  
!print*,'xnechek: ne read in copyocc'
!print*,xnelte
!print*,xne     

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FORMAL(XNE,TEMP,CLF)  

! UPDATED FOR CLUMPING, THOMSON_LINES ALREADY CORRECTED
! OPA_EFF_RAT added to output, since needed in formalsol

USE nlte_type
USE nlte_dim
USE fund_const, ONLY: SIGMAE, XMH=>AMH
USE nlte_var, ONLY: FRE,WFRE,HC,IFRE, &
& OPAC,STRUE,XJ,MODNAM,OPTLINES,THOMSON_LINES
!,RAYLEIGH
USE nlte_xrays, ONLY: OPTXRAY, LAMBDANU, FX, LXMIN, OPA_KSHELL

USE nlte_porvor, ONLY: OPA_EFF_RAT_OLD
  
IMPLICIT NONE
!
!
!---- it writes the file used in formal solution
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND=ID_NDEPT  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
!     ..
!     .. array arguments ..
REAL(DP) ::  TEMP(ND),XNE(ND),CLF(ND)

!     ..
INTEGER(I4B) ::  I,J  
!     ..
!     .. local arrays ..
REAL(DP) ::  SCONT(ND,IFRETOT),THOMSON(ND,IFRETOT)  
!     ..
! Just the Format of OPACITY_LP
LOGICAL :: OUT_LP = .TRUE.
CHARACTER(LEN=30) :: Forma 


OPEN (1,FILE=TRIM(MODNAM)//'/CONT_FORMAL',STATUS='UNKNOWN',FORM='UNFORMATTED')

REWIND 1  


IF(OPTLINES) THEN
  DO I = 1,ND  
     DO J = 1,IFRE  
!---- OPAC is effective opacity, thus ratio needs to be corrected 
!---- Since OPAC is one iteration behind, we use OPA_EFF_RAT_OLD here    
!---- (As long as we're converged, it of course makes no difference) 
       THOMSON(I,J) = &
!        (XNE(I)*SIGMAE*XMH/CLF(I)+THOMSON_LINES(I,J)+RAYLEIGH(I,J)/CLF(I))/OPAC(I,J)  
&        (XNE(I)*SIGMAE*XMH/CLF(I)+THOMSON_LINES(I,J))/OPAC(I,J) * OPA_EFF_RAT_OLD(I,J)
     END DO
  END DO  
ELSE
  DO I = 1,ND  
     DO J = 1,IFRE  
!----      As above 
!          THOMSON(I,J) = (XNE(I)*SIGMAE*XMH+RAYLEIGH(I,J))/CLF(I)/OPAC(I,J)  
          THOMSON(I,J) = (XNE(I)*SIGMAE*XMH)/CLF(I)/OPAC(I,J) * OPA_EFF_RAT_OLD(I,J)
     END DO
  END DO  
ENDIF

DO I = 1,ND  
     DO J = 1,IFRE  
          SCONT(I,J) = STRUE(I,J) + THOMSON(I,J)*XJ(I,J)
     END DO
END DO  

WRITE (1) IFRE, (FRE(I),I=1,IFRETOT),((SCONT(I,J),I=1,ND),J=1,IFRETOT), &
&  ((OPAC(I,J),I=1,ND),J=1,IFRETOT), (TEMP(I),I=1,ND), &
&  ((OPA_EFF_RAT_OLD(I,J),I=1,ND),J=1,IFRETOT)

CLOSE (1)  


OPEN (1,FILE=TRIM(MODNAM)//'/CONT_FORMAL_ALL',STATUS='UNKNOWN', &
& FORM='UNFORMATTED')

REWIND 1  
!
!   now, all continuum quantities are stored in such a way that
!   scont = strue + thomson * xj
!

WRITE (1) IFRE, (FRE(I),I=1,IFRETOT),((OPAC(I,J),I=1,ND),J=1,IFRETOT), &
&  ((THOMSON(I,J),I=1,ND),J=1,IFRETOT),((STRUE(I,J),I=1,ND),J=1,IFRETOT), &
&  ((XJ(I,J),I=1,ND),J=1,IFRETOT), (TEMP(I),I=1,ND), &
&  ((OPA_EFF_RAT_OLD(I,J),I=1,ND),J=1,IFRETOT)

CLOSE (1)  

!from Luiz
IF(OPTXRAY.AND.OUT_LP) THEN
OPEN (1,FILE=TRIM(MODNAM)//'/OPACITY_LP',STATUS='UNKNOWN')  
Forma =  "(E13.6,F11.6,I7.3,E14.6,E14.6)"
WRITE(1,*) "  OPACITY      LAMBDA      dp  OPACITY-OPA_KSHELL   OPA_KSHELL"
!WRITE (1,FMT=*) ((OPAC(I,J),I=1,ND),J=1,IFRETOT)
do i=1,nd
   do j=1,ifre
      if (1.D8/FRE(J) .gt.0.  .and. 1.D8/FRE(J) .lt. 50.) then
         write(1,Forma) OPAC(I,J), 1.D8/FRE(J), i, (OPAC(I,J)-OPA_KSHELL(I,J)), OPA_KSHELL(I,J)
      end if 
   enddo
enddo
!WRITE (1,FMT=*) ((OPAC(I,J),I=1,ND),J=1,IFRETOT)
CLOSE (1)  
ENDIF


RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FRESFIN(ND,XNH,TMAX,OPTOUT,TEFF)  

USE nlte_type
USE nlte_dim
USE fund_const
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&       ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
& LABL4,LABL1,LABU1,LABL3,IPARE5, &
& TL,ANCDO,WLC,DL,FRECIN,FRECFN, &
& LABL2,PAREN2,PAREN3,PAREN4,LABU2  

USE nlte_var, ONLY: OCCNUM,MAINION, &
& FREF,WFREF,INDFIN,INDFRQ,INDFRQ1,IFREF, &
& FRE,IFRE, & 
& BLEVEL,ENIONND, &
& IMIA,IMAA, &
& ICONVER, &
& BN,INDEX,OP_FLAG,FRECFN1

USE nlte_xrays, ONLY: OPTXRAY, N_KEDGES, K_NMIN, K_NMAX, Z, NIONK=>N, ETH, NAME

IMPLICIT NONE
!
!-------this subroutine calculates the fine frequency grid
!       accounting for all edges. otherwise, this routine
!       has to be consistent with frescal
!
!       parameter las frecuencias del fichero de datos atomicos
!
!       hay que definir tambien las matrices de /atomdat/
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
INTEGER(I4B), PARAMETER :: NFMIN=300
!INTEGER(I4B), PARAMETER :: LEVMIN1=ID_LEVM1  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
INTEGER(I4B), PARAMETER :: IFREFIN=ID_FREC2  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  

!JO Dec. 2015
! FREQUENCIES FOR Lya and Lyb which should be used for consistency
! precision 1.d-9
REAL(DP), PARAMETER :: LYA_LO=1.D8/1217.00043262610D0, LYA_HI=1.D8/1212.85710645286D0, &
                       LYB_LO=1.D8/1027.46136261866D0, LYB_HI=1.D8/1024.50655646359D0, &
                       LYG_LO=1.D8/973.196165978500D0, LYG_HI=1.D8/972.310779289066D0, &
                       LYD_LO=1.D8/949.843120035809D0, LYD_HI=1.D8/948.999697008538D0
!     ..
!     .. scalar arguments ..
REAL(DP) ::  TMAX, TEFF  
INTEGER(I4B) ::  ND  
LOGICAL OPTOUT  
!     ..
!     .. array arguments ..
REAL(DP) ::  XNH(ND1)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AAA,DM,DMIN,DMIN1,DMIN2,DNEW,DNEW1,EDGE,EMAX,EPS, &
&            ERRP,FRENEW,PF,PFE,SUM1,SUM2,X,X1,XM1,XM2,XNH1,FREMIN,DFL
INTEGER(I4B) ::  I,IF1,IFORM,IGENIO,II,IJ,IL,ILAB,INEW,INEWD,INTEG, &
&        IO,IOLD,IZ,J,K,K1,K2,L,LEVP,LEVPORG,LEVTOT,LL,LLI,M, &
&        M1,M2,MM1,MM2,N,NC,NDAT,NDATOS,NOPA,NPER,NK,ISTART,IEND,NSKIP
CHARACTER LAB*6,PARE*6  
!     ..
!     .. local arrays ..
REAL(DP) ::  DATA(ID_NDATA),FLAUX(ID_RBFTR),FREEDGE(ID_RBFTR)  
INTEGER(I4B) ::  IOPA(KIS*KEL)  
!     ..
!     .. external functions ..
REAL(DP) ::  BNUE,XINTFRE  
EXTERNAL BNUE,XINTFRE  
!     ..
!     .. external subroutines ..
EXTERNAL SORT  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,DBLE,INT,LOG,LOG10,MAX,MIN  
!     ..
!     .. statement functions ..

REAL(DP) ::  EPSLON  
!     ..
!     .. statement function definitions ..
!
EPSLON(EDGE) = 5.D0*10.D0** (DBLE(INT(LOG10(EDGE)))-6.D0)  
!     ..


IF (ICONVER.NE.1) STOP ' INCORRECT CALL OF FRESFIN'  
IF (NL.GT.999) STOP ' TOO MANY LEVELS, CODING OF INDEX AFFECTED'  
!
!       most abundant element k1
!
K1 = 1  
X = ABUND(1)  
DO K = 2,NAT  
     X1 = MAX(X,ABUND(K))  
     IF (X1.EQ.X) CYCLE  
     X = X1  
     K1 = K  
END DO  
!
!       2nd abundant element if present k2
!
IF (NAT.GT.1) THEN  
     K2 = 1  
     IF (K1.EQ.1) K2 = 2  
     X = ABUND(K2)  
     DO K = 1,NAT  
          IF (K.EQ.K1) CYCLE  
          X1 = MAX(X,ABUND(K))  
          IF (X1.EQ.X) CYCLE  
          X = X1  
          K2 = K  
     END DO
!
END IF  
!
!       dominant ionisation stages of k1, k2 (if present)
!
SUM1 = 0.D0  
SUM2 = 0.D0  

DO L = 1,ND  
     SUM1 = SUM1 + MAINION(L,K1)  
     IF (NAT.GT.1) SUM2 = SUM2 + MAINION(L,K2)  
END DO

XM1 = SUM1/ND  
XM2 = SUM2/ND  
M1 = INT(XM1)  
M2 = INT(XM2)  
IF (XM1-M1.GT..5D0) M1 = M1 + 1  
IF (XM2-M2.GT..5D0) M2 = M2 + 1  

IOPA(1) = K1*100 + M1  

IF (NAT.GT.1) THEN  
     IOPA(2) = K2*100 + M2  
ELSE  
     K2 = 0  
     IF (M2.NE.0) STOP ' ERROR IN M2'  
     IOPA(2) = 0  
END IF  

IF (IMAA(K1).GT.1) THEN  
!
!       2nd important ionization stage of k1 (if present)
!
     IF (IMAA(K1).EQ.2) THEN  
!
!       e.g. helium
!
          MM1 = 3 - M1  
          IOPA(3) = K1*100 + MM1  
     ELSE  
!
          IF (M1.EQ.IMAA(K1)) THEN  
               MM1 = M1 - 1  
               GO TO 50  
          END IF  
!
!       above or below m1?
!
          SUM1 = 0.D0  
          SUM2 = 0.D0  
          DO L = 1,ND  
               XNH1 = 1.D0/XNH(L)  
               SUM1 = SUM1 + ENIONND(K1,M1+1,L)*XNH1  
               SUM2 = SUM2 + ENIONND(K1,M1-1,L)*XNH1  
          END DO
          SUM1 = SUM1/DBLE(ND)  
          SUM2 = SUM2/DBLE(ND)  

          IF (SUM1.GT.SUM2) THEN  
               MM1 = M1 + 1  
          ELSE  
               MM1 = M1 - 1  
          END IF  
   50           CONTINUE  
          IOPA(3) = K1*100 + MM1  

     END IF  

     NOPA = 3  
ELSE  
!
!       most abundant elememt has only one ionization stage
!
     MM1 = M1  
     NOPA = 2  
END IF  

IF (NAT.GT.1) THEN  
!
!       2nd important ionization stage of k2, as for k1
!
     IF (IMAA(K2).GT.1) THEN  
!
          IF (IMAA(K2).EQ.2) THEN  
               MM2 = 3 - M2  
               IOPA(NOPA+1) = K2*100 + MM2  
          ELSE  
               IF (M2.EQ.IMAA(K2)) THEN  
                    MM2 = M2 - 1  
                    GO TO 70  
               END IF  
               SUM1 = 0.D0  
               SUM2 = 0.D0  

               DO L = 1,ND  
                    XNH1 = 1.D0/XNH(L)  
                    SUM1 = SUM1 + ENIONND(K2,M2+1,L)*XNH1  
                    SUM2 = SUM2 + ENIONND(K2,M2-1,L)*XNH1  
               END DO

               SUM1 = SUM1/DBLE(ND)  
               SUM2 = SUM2/DBLE(ND)  
               IF (SUM1.GT.SUM2) THEN  
                    MM2 = M2 + 1  
               ELSE  
                    MM2 = M2 - 1  
               END IF  
   70                CONTINUE  
               IOPA(NOPA+1) = K2*100 + MM2  

          END IF  

          NOPA = NOPA + 1  
     ELSE  
          MM2 = M2  
!
!       may happen if only one ion per element present
!           if(nopa.eq.2) stop ' error in nopa'
!
     END IF  

END IF  
!
!       total number of important ions (with levmax,levmin1)
!
DO 90 K = 1,NAT  
     IF (ABUND(K)/ABUND(K1).LE.1.D-7) GO TO 90  
     DO M = IMIA(K),IMAA(K)  
          IF (K.EQ.K1 .AND. (M.EQ.M1.OR.M.EQ.MM1)) cycle
          IF (K.EQ.K2 .AND. (M.EQ.M2.OR.M.EQ.MM2)) cycle  
          NOPA = NOPA + 1  
          IOPA(NOPA) = K*100 + M  
     END DO
   90 END DO  

IF (NOPA.GT.KIS*KEL) STOP ' ERROR IN NOPA'  

IFREF = 0  
LEVTOT = 0  

DO 140 N = 1,NOPA  

     IF (IOPA(N).EQ.0) GO TO 140  
     K = IOPA(N)/100  
     I = IOPA(N) - 100*K  

     IF (I.EQ.NIONS(K)) STOP ' ERROR IN NIONS OR ISI'  
     IF (I.LE.0) STOP ' ERROR IN ION - FRESFIN '  

     LEVPORG = 0  
     DO II = 1,ID_RBFTR  
          ILAB = LABL4(II)  
          IF (LE(ILAB).EQ.K .AND. LI(ILAB).EQ.I) THEN  

!JO Dec. 2015: avoid edges around Lya and Lyb
!JO July 2021: and around Lyg and Lyd
               IF(FRECIN(II).GT.LYA_LO .AND. FRECIN(II).LT.LYA_HI) THEN
                  PRINT*,'WARNING!!!! WARNING!!!! WARNING!!!!'
                  PRINT*,'EDGE CLOSE TO LY-ALPHA NEGLECTED IN FREQ. GRID',K,' ',I
                  STOP ' WHEN YOU SEE THIS STATEMENT, PLEASE CONTACT J.P.'
! after tests, the stop statement can be exchanged with the cycle statement
!                  CYCLE
               ENDIF  

               IF(FRECIN(II).GT.LYB_LO .AND. FRECIN(II).LT.LYB_HI) THEN
                  PRINT*,'WARNING!!!! WARNING!!!! WARNING!!!!'
                  PRINT*,'EDGE CLOSE TO LY-BETA NEGLECTED IN FREQ. GRID',K,' ',I
                  STOP ' WHEN YOU SEE THIS STATEMENT, PLEASE CONTACT J.P.'
! after tests, the stop statement can be exchanged with the cycle statement
!                  CYCLE
               ENDIF

               IF(FRECIN(II).GT.LYG_LO .AND. FRECIN(II).LT.LYG_HI) THEN
                  PRINT*,'WARNING!!!! WARNING!!!! WARNING!!!!'
                  PRINT*,'EDGE CLOSE TO LY-GAMMA NEGLECTED IN FREQ. GRID',K,' ',I
                  STOP ' WHEN YOU SEE THIS STATEMENT, PLEASE CONTACT J.P.'
! after tests, the stop statement can be exchanged with the cycle statement
!                  CYCLE
               ENDIF  

               IF(FRECIN(II).GT.LYD_LO .AND. FRECIN(II).LT.LYD_HI) THEN
                  PRINT*,'WARNING!!!! WARNING!!!! WARNING!!!!'
                  PRINT*,'EDGE CLOSE TO LY-DELTA NEGLECTED IN FREQ. GRID',K,' ',I
                  STOP ' WHEN YOU SEE THIS STATEMENT, PLEASE CONTACT J.P.'
! after tests, the stop statement can be exchanged with the cycle statement
!                  CYCLE
               ENDIF  

               LEVPORG = LEVPORG + 1  
               FLAUX(LEVPORG) = FRECIN(II)  

          END IF  
     END DO
!
!     reordering of flaux since fl maybe not ordered by energy
!     example: hei
!
     LEVTOT = LEVTOT + LEVPORG  
     CALL SORT(LEVPORG,FLAUX)  
!
!---  all levels, up to the highest ones consistent with frescal
!
     LEVP = LEVPORG  
     IF (LEVP.EQ.0) STOP ' ERROR IN LEVP - FRESMIN '  
!     LEVP1 = MIN(LEVP,LEVMIN1)  

     DO 130 L = 1,LEVP  
          IFREF = IFREF + 1  
          LLI = LEVPORG + 1 - L  
          DO II = 1,ID_RBFTR  
               IF (FLAUX(LLI).EQ.FRECIN(II)) GO TO 120  
          END DO
          STOP ' FLAUX NOT FOUND IN FRECIN'  

  120           CONTINUE  

          INDEX(IFREF) = 100000*K + I*1000 + II  
!
!     note that flaux here is in the "wrong' order
!
          FREF(IFREF) = FLAUX(LLI)  

  130      CONTINUE  
  140 END DO  

IF (LEVTOT.GT.ID_RBFTR) STOP ' ERROR IN LEVTOT'  

!IF (IFREF.GT.ID_LLEVS) STOP  ! this was the old statement with one edge per level
IF (IFREF.GT.ID_RBFTR) STOP ' ERROR IN IFREF'  ! THIS IS THE NEW ONE


PRINT *,' NUMBER OF CONSIDERED EDGES = ',IFREF  
PRINT *  

DO I = 1,IFREF  
     EPS = EPSLON(FREF(I))  
     FREF(I+IFREF) = FREF(I) - EPS  
     FREEDGE(I) = FREF(I)  
END DO  

IF1 = IFREF  
IFREF = 2*IFREF  

!K-shell edges
NK=0

IF(OPTXRAY) THEN
! only edges from ionization stage III on
  DO K=1,N_KEDGES
    IF (NIONK(K).LT.K_NMIN) CYCLE
    IF (NIONK(K).GT.K_NMAX) CYCLE
    IF (ETH(K).GT.2.D7) CYCLE
     NK=NK+1
     IFREF=IFREF+1
     FREF(IFREF)=ETH(K)
     IFREF=IFREF+1
     EPS = EPSLON(ETH(K))  
     FREF(IFREF) = FREF(IFREF-1) - EPS
  ENDDO
PRINT *,' NUMBER OF CONSIDERED K-SHELL EDGES = ',NK  
PRINT *  
ENDIF  

CALL SORT(IFREF,FREF)  

FREMIN=FREF(1)
I=LOG10(FREMIN)-1
I=MAX(I,0)
I=10**I
FREMIN=FLOAT(INT(FREMIN/I)*I)

IF (FREF(1).LE.FREMIN) STOP ' ERROR IN FREMIN'  

!OLD VERSION, WITH FRE(1)=FREMIN
!DO I = IFREF + 1,2,-1  
!          FREF(I) = FREF(I-1)  
!END DO

!FREF(1) = FREMIN
!IFREF = IFREF + 1  

!NEW VERSION FOR mm-FLUXES
! 9 points before fremin, starting at 1mm = 10 Kayser

DFL=LOG10(FREMIN/10.)/9. !10 Kayser, 10 points = 9 intervals

DO I = IFREF + 10,11,-1  
          FREF(I) = FREF(I-10)  
END DO 

FREF(1)=10.
DO I=1,9
  FREF(I+1)=FREF(1)*10.**(I*DFL)
ENDDO  
  
IFREF=IFREF+10

IF(OPTXRAY) THEN
  AAA = 2.D7 !    (5 A)
  
ELSE
EMAX = 10.D0*TMAX/HKL  

IF(NAT.EQ.1.AND.K1.EQ.1.OR.TEFF.LT.10000.) THEN
  AAA=4.D5    ! (250 A)
ELSE IF(TEFF.LT.20000.) THEN
  AAA = 1.D8/180.D0  !  (180 A)
ELSE IF(TEFF.LT.35000.) THEN
  AAA = 1.D6  !  (100 A)
ELSE
  AAA = 5.D6  !  ( 20 A)
ENDIF

ENDIF

PRINT *
PRINT *, ' INTERMEDIATE FREF(IFREF): ', 1.0D8/FREF(IFREF)
PRINT *, ' MIN      : ', 1.0D8/AAA
PRINT *


EMAX = MAX(EMAX,AAA)  
EMAX = MAX(EMAX,FREF(IFREF)*1.2D0)  

IF(EMAX.GT.7.5D5) THEN ! in case, resolve HeII edge
  IFREF = IFREF + 1  
  FREF(IFREF) = 7.5D5 !(133 A)
ENDIF


IF(OPTXRAY) THEN
  IFREF = IFREF + 1  
  FREF(IFREF) = 1.2D6  !(86 A, half way between 133 A and first K-shell at 39A)
  IFREF = IFREF + 1  
  FREF(IFREF) = 5.D6  !( 20 A) 
ENDIF  

IFREF = IFREF + 1  
FREF(IFREF) = EMAX  
!
!here was a bug, missing sort statement until Oct. 2014
CALL SORT(IFREF,FREF)  
!
!      changed to obtain similar freq. grids in all cases
!      dmin=log(fre(ifre)/fre(1))/nfmin
!
DMIN = LOG(1.D6/1.D3)/NFMIN  
NPER = IFREF - 1  
DO 190 I = 1,NPER  

     IF (FREF(I+1).LT.1.D4) THEN ! >10000 A  
          DMIN1 = DMIN*3  
     ELSE IF (FREF(I+1).LT.5.D4) THEN ! >2000 A   
          DMIN1 = DMIN*2  
     ELSE IF (FREF(I+1).LT.1.D8/1600.) THEN ! >1600 A   
!changed Feb 2015, for better resol between 2000 and 1600 of non-HHe models
!          DMIN1 = DMIN  
          DMIN1 = DMIN/4.  
! here was the  bug (corrected from V10.1 on): fre instead of fref)
     ELSE IF (FREF(I+1).LT.1.D8/910.) THEN ! > 910 A   
          DMIN1 = DMIN/4.

!   minimum separation behind HeII edge should be approx. 10  A or 2 A (if X-rays)
!
     ELSE IF (FREF(I+1).GT.440528.D0) THEN  ! < 227 A 
          DMIN2 = 10.D-8* FREF(I+1) ! DELTA LAM / LAM  
          IF(OPTXRAY) DMIN2 = 2.D-8* FREF(I+1) ! DELTA LAM / LAM  
          IF(OPTXRAY.AND.FREF(I+1).GT.5.D6) DMIN2=0.3          
          DMIN1 = DMIN2      
     ELSE  ! 227 (or min) ... 910 A 
          DMIN1 = 2000./300000. !(MIN. RESOL = 2000 KM/S)  
     END IF  

     DM = FREF(I+1)/FREF(I) - 1.D0  
!
!   nice trick to obtain resolved edges in any case
!
     IF (DM.GT.DMIN1/4.D0) THEN  
          INEW = INT(DM/DMIN1) + 1  
          DNEW = (FREF(I+1)-FREF(I))/INEW  

          DO J = 1,INEW  
               DNEW1 = DNEW  
!
!   concentration towards edges for all transitions
!
               IF (J.EQ.1) THEN  
                    DNEW1 = DNEW/4.D0  
                    DO K = 1,3  
                         FRENEW = FREF(I) + K*DNEW1  
                         IFREF = IFREF + 1  
                         FREF(IFREF) = FRENEW  
                    END DO 

               ELSE IF (J.EQ.2) THEN  
                    DNEW1 = DNEW/2.D0  
                    FRENEW = FREF(IFREF) + DNEW1  
                    IFREF = IFREF + 1  
                    FREF(IFREF) = FRENEW  
               END IF  

               IF (J.NE.INEW) THEN  
                    FRENEW = FREF(I) + J*DNEW  
                    IFREF = IFREF + 1  
                    FREF(IFREF) = FRENEW  
               END IF  
          END DO

     END IF  
  190 END DO  

!JO Dec. 2015
! modify frequency points around Lya and Lyb so that consistency
! with pure HHe freq. grid (if HHe only, no action performed)
! points chosen in such a way that Lya and Lyb not completely centered,
! to allow for a compromise (otherwise self-shadowing too strong)  

! in case, add points
DO I=1,IFREF
  IF(ABS(FREF(I)-LYA_LO).LT.1.D-9) GOTO 226 
ENDDO
IFREF=IFREF+1
FREF(IFREF)=LYA_LO
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FREF(IFREF)

226 DO I=1,IFREF
  IF(ABS(FREF(I)-LYA_HI).LT.1.D-9) GOTO 227
ENDDO
IFREF=IFREF+1
FREF(IFREF)=LYA_HI
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FREF(IFREF)

227 DO I=1,IFREF
  IF(ABS(FREF(I)-LYB_LO).LT.1.D-9) GOTO 228 
ENDDO
IFREF=IFREF+1
FREF(IFREF)=LYB_LO
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FREF(IFREF)

228 DO I=1,IFREF
  IF(ABS(FREF(I)-LYB_HI).LT.1.D-9) GOTO 229 
ENDDO
IFREF=IFREF+1
FREF(IFREF)=LYB_HI
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FREF(IFREF)

229 CONTINUE

CALL SORT(IFREF,FREF)  

!-----------------------------------------------
!in case, remove points in between lya_lo,lya_hi

DO I=1,IFREF
  IF(ABS(FREF(I)-LYA_LO).LT.1.D-9) GOTO 326 
ENDDO
STOP ' LYA_LO NOT FOUND IN FRESFIN' 

326 ISTART=I
DO I=ISTART,IFREF
  IF(ABS(FREF(I)-LYA_HI).LT.1.D-9) GOTO 327 
ENDDO
STOP ' LYA_HI NOT FOUND IN FRESFIN' 

327 IEND=I

IF (OPTOUT .AND. ICONVER.EQ.1) THEN
  DO I=ISTART+1,IEND-1
    WRITE (*,FMT=9033) 1.D8/FREF(I)
  ENDDO
ENDIF

NSKIP=IEND-ISTART-1
IF(NSKIP.LT.0) STOP ' ERROR IN NSKIP (FRESFIN)'

IF(NSKIP.NE.0) THEN
  IFREF=IFREF-NSKIP
  DO I=ISTART+1,IFREF
    FREF(I)=FREF(I+NSKIP)
  ENDDO
ENDIF

!-----------------------------------------------
!in case, remove points in between lyb_lo,lyb_hi

DO I=1,IFREF
  IF(ABS(FREF(I)-LYB_LO).LT.1.D-9) GOTO 328 
ENDDO
STOP ' LYB_LO NOT FOUND IN FRESFIN' 

328 ISTART=I
DO I=ISTART,IFREF
  IF(ABS(FREF(I)-LYB_HI).LT.1.D-9) GOTO 329 
ENDDO
STOP ' LYB_HI NOT FOUND IN FRESFIN' 

329 IEND=I

IF (OPTOUT .AND. ICONVER.EQ.1) THEN
  DO I=ISTART+1,IEND-1
    WRITE (*,FMT=9033) 1.D8/FREF(I)
  ENDDO
ENDIF

NSKIP=IEND-ISTART-1
IF(NSKIP.LT.0) STOP ' ERROR IN NSKIP (FRESFIN)'

IF(NSKIP.NE.0) THEN
  IFREF=IFREF-NSKIP
  DO I=ISTART+1,IFREF
    FREF(I)=FREF(I+NSKIP)
  ENDDO
ENDIF

!    include high resolution points from frescal
!    (hydrogen resonance lines (in particular, Lyg and Lyd), hei singlet resonance line and heii 303)
!
OUTER: DO I = 1,IFRE  
   DO IJ = 1,IFREF  
      IF (ABS(1.-FREF(IJ)/FRE(I)).LT.1.D-14) CYCLE OUTER ! CAN BE EQUAL BY CHANCE 
   END DO
   IFREF=IFREF+1
   FREF(IFREF)=FRE(I)
   WRITE (*,FMT=9031) 1.D8/FREF(IFREF)
ENDDO OUTER


CALL SORT(IFREF,FREF)  

IF (IFREF.GT.IFREFIN) STOP ' TOO MANY FREQUENCIES IN FRESFIN!'  
PRINT *,' TOTAL NUMBER OF FINE GRID FREQUENCIES = ',IFREF  
PRINT *  
PRINT *  
PRINT *,' FINAL FREF(IFREF): ', 1.0D8/FREF(IFREF)
PRINT *


IF (OPTOUT .AND. ICONVER.EQ.1) THEN  
    PRINT *,' NO    WAVELENGTH(A)     FREQ(HZ)'  
     DO 240 I = 1,IFREF  
          DO J = 1,IF1  
               IF (FREF(I).EQ.FREEDGE(J)) GO TO 220  
          END DO
          
          DO IJ = 1,IFRE  
               IF (ABS(1.-FREF(I)/FRE(IJ)).LT.1.D-14) THEN  
                    INDFIN(I) = IJ  

                    IF(OPTXRAY) THEN
                    DO J = 1,N_KEDGES  
                       IF (FREF(I).EQ.ETH(J)) THEN
                         WRITE (*,FMT=9025) INDFIN(I),1.D8/FREF(I), &
                           CLIGHT*FREF(I),NAME(Z(J)),NIONK(J)-1  
                         GO TO 240  
                       ENDIF
                    ENDDO
                    ENDIF

                    WRITE (*,FMT=9000) INDFIN(I),1.D8/FREF(I), &
                     CLIGHT*FREF(I)
                    GO TO 240  
               END IF  
          END DO

          IF (I.EQ.1) STOP ' ERROR IN PHILOSOPHY - FRESFIN '  
          INDFIN(I) = -ABS(INDFIN(I-1))  
          WRITE (*,FMT=9010) INDFIN(I),1.D8/FREF(I),CLIGHT*FREF(I)  
          GO TO 240  

  220           CONTINUE  
          IL = INDEX(J)  
          K = IL/100000  
          IZ = INT(ZEFF(K))  
          IL = IL - 100000*K  
          IO = IL/1000  
          LL = IL - 1000*IO  
          LAB = LABL(LABL4(LL))  
          PARE = PAREN4(LL)  
          DO IJ = 1,IFRE  
               IF (FREF(I).EQ.FRE(IJ)) THEN  
                    INDFIN(I) = IJ  
                    WRITE (*,FMT=9020) INDFIN(I),1.D0/FREF(I)*1.D8, &
                     CLIGHT*FREF(I),LABAT(K),IO,IZ + IO - 1,LAB,PARE
                    GO TO 240  
               END IF  
          END DO

          IF (I.EQ.1) STOP ' ERROR IN PHILOSOPHY - FRESFIN '  

          INDFIN(I) = -ABS(INDFIN(I-1))  

          WRITE (*,FMT=9030) 1.D0/FREF(I)*1.D8,CLIGHT*FREF(I), &
           LABAT(K),IO,IZ + IO - 1,LAB,PARE

  240      CONTINUE  

     PRINT *  

END IF  
!
!---    check of indfin
!
IOLD = INDFIN(1)  

DO I = 2,IFREF  
     INEW = ABS(INDFIN(I))  
     INEWD = INEW - IOLD  
     IF (INEWD.NE.0 .AND. INEWD.NE.1) THEN
       PRINT*,IFREF,I,INEW,IOLD,INEWD
       STOP ' ERROR IN INDFIN'  
     ENDIF
     IOLD = INEW  
END DO  
!
!---   final check
!
IF(ABS(INDFIN(IFREF)).NE.IFRE) STOP ' ERROR IN INDFIN: NUMBER OF FREQ. POINTS'

!
!       index array with indices of edge frequencies
!       according to input (i.e., ground or excited states)
!
DO 270 II = 1,ID_RBFTR  

     DO IJ = 1,IFREF  
          IF (FREF(IJ).EQ.FRECIN(II)) THEN  
             INDFRQ(II) = IJ  
             GO TO 270  
          END IF  
     END DO
!
!       may happen if not all ions present
!       stop ' edge not found in fre'
!
  270 END DO  

!       index array with indices of edge frequencies for OP-data
!       (range between ground and excited states)

   
DO 280 II = 1,ID_RBFTR  
    
     IF(.NOT.OP_FLAG(II)) THEN
       INDFRQ1(II) = 0
       CYCLE
     ELSE  
       DO IJ = 1,IFREF  
!JO Sept. 2016: changed to account for FRECFN1 (OP-DATA) 
            IF(FRECFN1(II).NE.FRECFN(II)) THEN
              IF (FREF(IJ).EQ.FRECFN1(II)) THEN  
                 INDFRQ1(II) = IJ  
!                 PRINT*,'1',II,LABL(LABL4(II)),FRECFN1(II),FREF(IJ) 
                 GO TO 280  
              END IF  
            ELSE
              IF (FREF(IJ).GE.FRECFN1(II).AND.INDFIN(IJ).GT.0) THEN  
!HERE: GE, since in this case FRECFN1 (=FRECFN) not necessarily a frequency grid point;
!      (it is definitely a grid point if either FRECFN1 > FRECFN -- previous
!       condition or FRECFN1=FRECFN=FRECIN -- this condition)
!      INDFIN has to be positive, since edge needs to be on the coarse grid 
                 INDFRQ1(II) = IJ  
!                 PRINT*,'2',II,LABL(LABL4(II)),FRECFN1(II),FREF(IJ) 
                 GO TO 280  
              END IF  
            ENDIF  
       END DO
!       PRINT*,II,LABL(LABL4(II)),OP_FLAG(II)
!       STOP 'NOT FOUND'
!  
!       may happen if not all ions present
!       stop ' edge not found in fre'
!
     ENDIF

 280 END DO  

DO K = 1,IFREF  
     BN(K) = BNUE(1.D8/FREF(K),TMAX)  
END DO

PF = XINTFRE(BN,WFREF,1,IFREF,IFREF,FREF,4,'NEW')  
PFE = SIGSB/PI*TMAX**4  
ERRP = ABS(1.D0-PF/PFE)  
PRINT *,' ERROR IN PLANCK-FUNCTION WITH FINE FREQ. GRID = ',ERRP  
PRINT *  

RETURN  

 9000 FORMAT ('*',I4,2X,F12.3,6X,E12.6)  
 9010 FORMAT (I5,2X,F12.3,6X,E12.6)  
 9020 FORMAT ('*',I4,2X,F12.3,6X,E12.6,3X,A2,I2,' Z=',I2,'  FROM ',A6, &
&       ' TO ',A6)
 9025 FORMAT ('*',I4,2X,F12.3,6X,E12.6,3X,A2,' Z=',I2,' K-SHELL ABSORPTION')
 9030 FORMAT (I5,2X,F12.3,6X,E12.6,3X,A2,I2,' Z=',I2,'  FROM ',A6, &
&       ' TO ',A6)
 9031 FORMAT (' ADDIT. POINT FROM FRESCAL AT ',F12.3)
 9032 FORMAT (' ADDIT. POINT AT ',F12.3,' LY_ALPHA/LY_BETA')
 9033 FORMAT ('        POINT AT ',F12.3,' DROPPED (LY_ALPHA/LY_BETA)')
 END
!
!-----------------------------------------------------------------------
!
SUBROUTINE HOPFPARA(TEFF,GGRAV,YHE,XMET,XMLOSS,VMAX,SRNOM,CLF) 

USE nlte_type  
USE nlte_dim  
Use nlte_var, ONLY: QINF,Q0,GAM=>GAMHOPF  

IMPLICIT NONE  
!
!     assumes files 'hopfpara_all_HHe/met' in catalog above local directory
!
!     .. scalar arguments ..
REAL(DP) ::  GGRAV,TEFF,YHE,XMET,XMLOSS,VMAX,SRNOM,CLF
!     ..
!     .. local scalars ..
REAL(DP) ::  GIN,TIN,YIN,QQIN,QQQ  
!     ..
!     ..

! INCLUDING MAX. CLF
QQQ=LOG10(SQRT(CLF)*XMLOSS/(VMAX*SRNOM)**1.5) 
PRINT*,'log Q = (for Hopfpara-file)',QQQ 

IF(ABS(XMET).LT.0.2D0) THEN
  OPEN (3,FILE='../HOPFPARA_ALL_HHe',STATUS='OLD')  
ELSE
  OPEN (3,FILE='../HOPFPARA_ALL_met',STATUS='OLD')  
ENDIF
!
!     first line is description
!
READ (3,FMT=*)  

READFILE: DO

!READ (1,FMT=*,END=20) TIN,GIN,QQIN,YIN,QINF,Q0,GAM
!IF (TIN.EQ.TEFF .AND. GIN.EQ.GGRAV .AND. YIN.EQ.YHE .AND. &
!&    abs(QQIN-QQQ).lt.1.d-3) THEN

READ (3,FMT=*,END=20) TIN,GIN,YIN,QINF,Q0,GAM 
IF (TIN.EQ.TEFF .AND. GIN.EQ.GGRAV .AND. YIN.EQ.YHE ) THEN 
     PRINT * 
     PRINT *,' QINF = ',QINF,' Q0 = ',Q0,' GAMMA = ',GAM  
     PRINT *  
     CLOSE (3)
     RETURN
END IF

END DO READFILE

20 CLOSE (3)


IF(ABS(XMET).LT.0.2D0) THEN
  OPEN (3,FILE='../HOPFPARA_ALL_HHe',STATUS='OLD')  
ELSE
  OPEN (3,FILE='../HOPFPARA_ALL_met',STATUS='OLD')  
ENDIF

READ (3,FMT=*)  

QQQ=-14.

READFILE_1: DO

READ (3,FMT=*,END=30) TIN,GIN,QQIN,YIN,QINF,Q0,GAM

IF (TIN.EQ.TEFF .AND. GIN.EQ.GGRAV .AND. YIN.EQ.YHE .AND. QQIN.EQ.QQQ) THEN
     PRINT *
     PRINT *,' QINF = ',QINF,' Q0 = ',Q0,' GAMMA = ',GAM
     PRINT * 
     CLOSE (3)
     RETURN
END IF

END DO READFILE_1

   30 STOP ' HOPF-PARAMETERS NOT FOUND'  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE PZGRID(ND,NP,R)  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: P,Z
IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,NP1=ID_NPOIN  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND,NP  
!     ..
!     .. array arguments ..
REAL(DP) ::  R(ND1)  
!     ..
!     .. local scalars ..
REAL(DP) ::  Z2  
INTEGER(I4B) ::  J,L,LMAX,NC  
!     ..
!     .. intrinsic functions ..
!INTRINSIC DBLE,MIN0,SQRT  
!     ..

IF (ND.NE.ND1 .OR. NP.NE.NP1) STOP ' WRONG DIMENSIONS IN PZGRID'  

NC = NP - ND  
!
!     calculation of the p-grid
!
DO L = 1,NC  
     P(L) = .99D0*DBLE(L-1)/DBLE(NC-1)  
END DO  

DO L = NC + 1,NP  
     P(L) = R(NP+1-L)  
END DO  
!
!     calculation of the z-grid
!
DO L = 1,NP  
     LMAX = MIN0(NP+1-L,ND)  
     DO J = 1,LMAX  
          Z2 = R(J)*R(J) - P(L)*P(L)  
          IF (L.GT.NC .AND. J.EQ.LMAX) THEN  
               IF (Z2.GT.1.D-5) STOP 'ERROR IN Z=0'  
               Z2 = 0.  
          END IF  
          Z(J,L) = SQRT(Z2)  
     END DO
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE START(NREC,TEFF,YHEIN)  

USE nlte_type
USE nlte_dim
USE fund_const, ONLY : AKB,AMH
USE princesa_var, ONLY : NAT,LABAT,ABUND,WEIGHT,FRECFN
USE nlte_var, ONLY : VTURB,VTHER, &
&                    NAT_TOT,NAMES_CHANGED,ABCHANGED, &
&                    NAT_BG,NAMES_BG,ABUNDBG,FRECFN1

IMPLICIT NONE
!
!----- read atomic data
!
!     .. scalar arguments ..
REAL(DP) ::  TEFF,YHEIN  
INTEGER(I4B) ::  NREC  
!     ..
!     .. local scalars ..
REAL(DP) :: YHE
INTEGER(I4B) ::  I,ISI,NENE
!     ..
!     .. external subroutines ..
EXTERNAL OPENFS,PRINCE  
!     ..
!     .. intrinsic functions ..
!INTRINSIC SQRT  
!     ..

CALL PRINCE  
! check for identical rbf frequencies and modify them
CALL RBF_CHECK
!JO Sept 2016 initialize FRECFN1
FRECFN1=FRECFN


NENE=0
DO  ISI=1,NAT
     IF(LABAT(ISI).EQ.'HE') NENE=ISI
END DO

IF (NENE.NE.0) THEN
YHE=ABUND(NENE)
  IF (ABS(1.D0-YHE/YHEIN).GT.1.D-5) THEN
    PRINT* 
    PRINT*,' ********************************************'
    PRINT*,' ** WARNING: YHE AND YHEIN ARE DIFFERENT  ***'
    PRINT*,' ** MODEL ATOM HAS YHE= ',YHE,' INPUT IS ', YHEIN
    PRINT*,' ** ADOPTED : ', YHEIN
    PRINT*,' ********************************************'
    PRINT*
    WRITE(999,*) 
    WRITE(999,*) ' ********************************************'
    WRITE(999,*) ' ** WARNING: YHE AND YHEIN ARE DIFFERENT  ***'
    WRITE(999,*) ' ** MODEL ATOM HAS YHE= ',YHE,' INPUT IS ', YHEIN
    WRITE(999,*) ' ** ADOPTED : ', YHEIN
    WRITE(999,*) ' ********************************************'
    WRITE(999,*)
    ABUND(NENE)= YHEIN
  ENDIF
ENDIF
! prespecified abundances of explicit and background elements
NAT_BG=0

DO I=1,NAT_TOT
  IF(NAMES_CHANGED(I).EQ.'HE') STOP ' HE FOUND IN CHANGED ABUNDANCES'
  NENE=0
  DO  ISI=1,NAT
     IF(LABAT(ISI).EQ.NAMES_CHANGED(I)) NENE=ISI
  END DO

  IF (NENE.NE.0) THEN
    PRINT* 
    PRINT*,' ********************************************'
    PRINT*,' ** EXPLICIT ABUNDANCE CHANGED ***'
    PRINT*,' ** MODEL ATOM ',LABAT(NENE),' CHANGED TO',ABCHANGED(I)
    PRINT*,' ********************************************'
    PRINT*
    WRITE(999,*) 
    WRITE(999,*) ' ********************************************'
    WRITE(999,*) ' ** EXPLICIT ABUNDANCE CHANGED ***'
    WRITE(999,*) ' ** MODEL ATOM ',LABAT(NENE),' CHANGED TO',ABCHANGED(I)
    WRITE(999,*) ' ********************************************'
    WRITE(999,*)
    ABUND(NENE)= ABCHANGED(I)
  ELSE
! must be background element
    NAT_BG=NAT_BG+1
    ABUNDBG(NAT_BG)=ABCHANGED(I)
    NAMES_BG(NAT_BG)=NAMES_CHANGED(I)
    PRINT* 
    PRINT*,' ********************************************'
    PRINT*,' ** BACKGROUND ABUNDANCE CHANGED ***'
    PRINT*,' ** MODEL ATOM ',NAMES_BG(NAT_BG),' CHANGED TO',ABUNDBG(NAT_BG)
    PRINT*,' ********************************************'
    PRINT*
    WRITE(999,*) 
    WRITE(999,*) ' ********************************************'
    WRITE(999,*) ' ** BACKGROUND ABUNDANCE CHANGED ***'
    WRITE(999,*) ' ** MODEL ATOM ',NAMES_BG(NAT_BG),' CHANGED TO',ABUNDBG(NAT_BG)
    WRITE(999,*) ' ********************************************'
    WRITE(999,*)
  ENDIF
ENDDO

DO I = 1,NAT
!
! finally changed
!
     VTHER(I) = SQRT(2.D0*AKB*TEFF/WEIGHT(I)/AMH+VTURB**2)
END DO  
!
!-----open direct access-files
!
CALL OPENFS(NREC)

RETURN

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE OPENFS(NRECET)  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: MODNAM
IMPLICIT NONE
!
!------ this subroutine opens all direct access (and some other files.)
!       the files have a status of 'scratch', so they will be dismissed
!       when the program ends
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND=ID_NDEPT  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  NRECET  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  INFRE,NDE,NREC  
!     ..
NREC = NRECET  
NREC = NREC*8  
NDE = 8*ND  

INFRE = IFRETOT*8  
!
!       file for lte population (alevel(i))
!
OPEN (UNIT=15,FILE=TRIM(MODNAM)//'/LTE_POP',STATUS='UNKNOWN',RECL=NREC, &
& ACCESS='DIRECT')
!
!-----------------------------------------------------------------------
!       file 'model'
!
OPEN (UNIT=20,FILE=TRIM(MODNAM)//'/MODEL',STATUS='UNKNOWN',FORM='UNFORMATTED')
!
!-----------------------------------------------------------------------
!       file 'temp'
!
OPEN (UNIT=21,FILE=TRIM(MODNAM)//'/TEMP',STATUS='UNKNOWN',FORM='FORMATTED')
!
!-----------------------------------------------------------------------
!       file for occnlte_old (blevel(i))
!
OPEN (UNIT=27,FILE=TRIM(MODNAM)//'/NLTE_POP',STATUS='UNKNOWN',RECL=NREC, &
& ACCESS='DIRECT')
!
!-----------------------------------------------------------------------
! from V10 on: file 'CHIBAR_H': flux-weighted opacities
!
OPEN (UNIT=60,FILE=TRIM(MODNAM)//'/CHIBAR_H',STATUS='UNKNOWN',FORM='FORMATTED')
!
!-----------------------------------------------------------------------
!       info file for changes in occupation numbers and extrapolations
!       (acceleration)
!
OPEN (UNIT=99,FILE=TRIM(MODNAM)//'/MAX_CHANGE',STATUS='UNKNOWN')
!
!-----------------------------------------------------------------------
!       file for actual population (blevel(i))
!
OPEN (UNIT=17,STATUS='SCRATCH',RECL=NREC,ACCESS='DIRECT')  
!
!-----------------------------------------------------------------------
!        files used in sobolev+continuum approach (lines)
!        file for j_nue values at different depth points
!
OPEN (UNIT=50,STATUS='SCRATCH',RECL=NDE,ACCESS='DIRECT')  
!
!-----------------------------------------------------------------------
!        file for h_nue values at different depth points, no longer req.
!
OPEN (UNIT=51,STATUS='SCRATCH',RECL=NDE,ACCESS='DIRECT')  
!
!-----------------------------------------------------------------------
!        file for k_nue values at different depth points
!
OPEN (UNIT=52,STATUS='SCRATCH',RECL=NDE,ACCESS='DIRECT')  
!
!-----------------------------------------------------------------------
!        file for n_nue values at different depth points, no longer req.
!
OPEN (UNIT=53,STATUS='SCRATCH',RECL=NDE,ACCESS='DIRECT')  
!
!      same as before but at a given depth points, for different
!      nue values. 40: j; 41: h; 42: k; 43: n.
!
OPEN (UNIT=40,STATUS='SCRATCH',RECL=INFRE,ACCESS='DIRECT')  
OPEN (UNIT=42,STATUS='SCRATCH',RECL=INFRE,ACCESS='DIRECT')  

OPEN (UNIT=41,STATUS='SCRATCH',RECL=INFRE,ACCESS='DIRECT')  
OPEN (UNIT=43,STATUS='SCRATCH',RECL=INFRE,ACCESS='DIRECT')  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE ULECTURE  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: FPATH,UFUNG,UFUNK
IMPLICIT NONE
!
!---- it reads and stores u-function values from files utablak.dat and
!---- utablag.dat. values are stores into module
!
!     .. local scalars ..
REAL(DP) ::  BETA  
INTEGER(I4B) ::  IB,IT  
!     ..
!     .. local arrays ..
REAL(DP) ::  BETAG(17),BETAK(76)  
!     ..
!     .. intrinsic functions ..
!INTRINSIC LOG10  
!     ..
!     .. data statements ..
DATA BETAK/1.0D-10,2.8D-10,4.6D-10,6.4D-10,8.2D-10,1.0D-9,2.8E-9, &
&     4.6D-9,6.4D-9,8.2D-9,1.0D-8,2.8D-8,4.6D-8,6.4D-8,8.2D-8, &
&     1.0D-7,2.8D-7,4.6D-7,6.4D-7,8.2D-7,1.0D-6,2.8D-6,4.6D-6, &
&     6.4D-6,8.2D-6,1.0D-5,2.8D-5,4.6D-5,6.4D-5,8.2D-5,1.0D-4, &
&     2.8D-4,4.6D-4,6.4D-4,8.2D-4,1.0D-3,2.8D-3,4.6D-3,6.4D-3, &
&     8.2D-3,1.0D-2,2.8D-2,4.6D-2,6.4D-2,8.2D-2,1.0D-1,2.8D-1, &
&     4.6D-1,6.4D-1,8.2D-1,1.0D+0,2.8D+0,4.6D+0,6.4D+0,8.2D+0, &
&     1.0D+1,2.8D+1,4.6D+1,6.4D+1,8.2D+1,1.0D+2,2.8D+2,4.6D+2, &
&     6.4D+2,8.2D+2,1.0D+3,2.8D+3,4.6D+3,6.4D+3,8.2D+3,1.0D+4, &
&     2.8D+4,4.6D+4,6.4D+4,8.2D+4,1.0D+5/

DATA BETAG/1.D-10,1.D-9,1.D-8,1.D-7,1.D-6,1.D-5,1.D-4,1.D-3,1.D-2, &
&     1.D-1,5.D-1,1.D+0,1.D+1,1.D+2,1.D+3,1.D+4,1.D+5/
!     ..

OPEN (UNIT=10,FILE=fpath//'utablak.dat',STATUS='OLD',FORM='FORMATTED')  

OPEN (UNIT=11,FILE=fpath//'utablag.dat',STATUS='OLD',FORM='FORMATTED')  

DO IT = 1,9  
     DO IB = 1,17  
          READ (11,FMT=*) UFUNG(IT,IB)  
     END DO
END DO  

DO IT = 1,21  
     DO IB = 1,76  
          READ (10,FMT=*) UFUNK(IT,IB)  
     END DO
END DO  
!
!---- u-g function is changed into ug/beta if beta<1,
!
DO IT = 1,9  
     DO IB = 1,11  
          BETA = BETAG(IB)  
          UFUNG(IT,IB) = UFUNG(IT,IB)/BETA  
     END DO
END DO
!
!---- u-k function is changed into log10(uk) (log10(uk/beta) if beta<1)
!
DO IT = 1,21  
     DO IB = 1,50  
          BETA = BETAK(IB)  
          UFUNK(IT,IB) = LOG10(UFUNK(IT,IB)/BETA)  
     END DO
        
     DO IB = 51,76  
          UFUNK(IT,IB) = LOG10(UFUNK(IT,IB))  
     END DO

END DO  

CLOSE(10)
CLOSE(11)
RETURN

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE U2LECTURE  
use nlte_type
use nlte_var, ONLY: FPATH,U2B,U2S,U2L1,U2L2
implicit none
!
!
!     ..
!     .. local scalars ..
INTEGER(I4B) :: I, J, K  
!     ..
OPEN (UNIT = 30, FILE = fpath//'U2B.dat', STATUS = 'old')  
OPEN (UNIT = 31, FILE = fpath//'U2S.dat', STATUS = 'old')  
OPEN (UNIT = 32, FILE = fpath//'U2L1.dat', STATUS = 'old')  
OPEN (UNIT = 33, FILE = fpath//'U2L2.dat', STATUS = 'old')  
!
DO J = 1, 21  
   READ (30, FMT = * ) (U2B (J, K), K = 1, 3)  
   READ (30, FMT = * ) (U2B (J, K), K = 4, 6)  
END DO  
!
DO I = 1, 192  
   K = 0  
   DO J = 1, 66  
      READ (31, FMT = * ) U2S (I, J + K), U2S (I, J + K + 1), &
       U2S (I, J + K + 2)
      K = K + 2  
   END DO  
   READ (31, FMT = * ) U2S (I, 199), U2S (I, 200)  
END DO  
!
DO I = 1, 61  
   K = 0  
   DO J = 1, 23  
      READ (32, FMT = * ) U2L1 (I, J + K), U2L1 (I, J + K + 1), &
       U2L1 (I, J + K + 2)
      K = K + 2  
   END DO  
   READ (32, FMT = * ) U2L1 (I, 70), U2L1 (I, 71)
END DO  
!
DO I = 1, 12  
   K = 0  
   DO J = 1, 6  
      READ (33, FMT = * ) U2L2 (I, J + K), U2L2 (I, J + K + 1), &
       U2L2 (I, J + K + 2)
      K = K + 2  
    END DO
END DO  
!
CLOSE (30)  
CLOSE (31)  
CLOSE (32)  
CLOSE (33)  

RETURN  

END SUBROUTINE U2LECTURE
!
!***********************************************************************
!
! subroutines: complex ones
! cont and related
!
!***********************************************************************
!
SUBROUTINE CONT(ND,NP,R,RHO,XNE,TEMP,CLF,TEFF,SR,CORRFC,OPTLTE,CONCON, &
&               TAUR,RTAU23,RYBICKI,DTFCORR,FLTOT,NDIV,IIT)

! updated for clumping; all opacities correspond to average values!
! only change: redefinition of THOMSON

USE nlte_type
USE nlte_dim
USE fund_const
USE nlte_var, ONLY: FRE,WFRE,HC,IFRE,LTHIN,&
& OPAC,ST=>STRUE,XJ,ALO, &
& Z,P, &
& MODNAM, &
& PRECIS,UNASOL,MEGAS,OPTLINES,THOMSON_LINES
!,RAYLEIGH  

USE photstruc, ONLY: CHIBAR_H

USE nlte_xrays, ONLY: OPTXRAY

USE nlte_porvor, ONLY: FIC, TCL_FAC_LINE, TCL_FAC_CONT, OPA_EFF_RAT
                                                      
IMPLICIT NONE
!
!     .. parameters ..
! 
! range for X-ray luminosities (0.1, 0.15. 0.35 to 2.5 keV); Rosat and XMM
REAL(DP), PARAMETER :: CONST_EV=EV/(HH*CLIGHT)
REAL(DP), PARAMETER :: XRED1=CONST_EV*100.D0, XRED2=CONST_EV*150.D0, &
&                      XRED3=CONST_EV*350.D0, XBLU=CONST_EV*2450.D0

INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,NP1=ID_NPOIN  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORRFC,RTAU23,SR,TEFF  
INTEGER(I4B) ::  ND,NP,NDIV,IIT  
LOGICAL CONCON,OPTLTE,RYBICKI  
!     ..
!     .. array arguments ..
REAL(DP), intent(in) :: R(ND1),RHO(ND1),TAUR(ND1),TEMP(ND1),XNE(ND1),CLF(ND1)  
REAL(DP), intent(out) :: DTFCORR(ND1),FLTOT(ND1)

!     ..
!     .. local scalars ..
REAL(DP) ::  AIC,BNUECON,DBDR,DBDTF,DTDR,FERRI,FERRMAX,FERRO, &
&                 FLUXCIN,FLUXCON,FLUXDIF,OP,OPAND,RTHIN,SIG, &
&                 TRADIC,TRADIC1,WFR,XIC,XICIN,XICR,XICV1,XLAMBDA, &
&                 XHMIN, DESH, CONVLX, TAU, DT, R1

REAL(DP) :: TCL

INTEGER(I4B) ::  IMOD,K,L,IXR1,IXR2,IXR3,IXB

!     ..
!     .. local arrays ..
REAL(DP) ::       OPA(ND1),Q1(ND1-2),Q2(ND1-2), &
&                 THOMSON(ND1),WP(ND1+1,NP1-1),WP2(ND1+1,NP1-1), &
&                 XH(ND1+1),HTOT(ND1), &
&                 LUMX1(ND1+1), LUMX2(ND1+1), LUMX3(ND1+1), AUX(ND1+1), &
&                 R1K(IFRETOT)
!     ..
!     .. external functions ..
REAL(DP) ::  DBDT, bnue       
EXTERNAL DBDT, bnue       
!     ..
!     .. external subroutines ..
EXTERNAL CONT1,CONVERSION,DIFFUS  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,LOG,LOG10,MAX,MOD  
!     ..

SIG = AMH*SIGMAE  
FLUXCIN = 0.D0  
FLUXCON = 0.D0  
FLUXDIF = 0.D0  

FLTOT = 0.D0  
HTOT = 0.D0
CHIBAR_H=0.D0

!
!-----same argument as in subr. lte
!-----flux has to be corrected with respect to the lower radius according to
!     rstar^2*flux(teff) = rmin^2*flux(rmin)
!     -> flux(rmin)=(rstar/rmin)^2 flux(teff)=RTAU23^2 flux(teff)
!
BNUECON = SIGSB/PI*TEFF**4*RTAU23**2  ! corresponds to 4 H (astrophys. flux) * (rstar/sr)^2 =
                                      ! total flux AT LOWER BOUNDARY
IF (.NOT.OPTLTE) THEN  
     DTDR = (TEMP(ND)-TEMP(ND-1))/ (R(ND-1)-R(ND))  
     DO K = 1,IFRE  
          WFR = WFRE(K)  
          DBDTF = DBDT(1.D8/FRE(K),TEMP(ND))  
          OPAND = OPAC(ND,K)  
          FLUXDIF = FLUXDIF + WFR*DBDTF/OPAND  
      END DO 

     FLUXDIF = FLUXDIF*4.D0*DTDR/ (3.D0*SR)  
     CORRFC = BNUECON/FLUXDIF  
     PRINT *  
     PRINT *,' NEW CORRECTION FACTOR = ',CORRFC  
     PRINT *  
     FLUXDIF = FLUXDIF*CORRFC  
END IF  

! boundaries for X-ray treatment
IF(.NOT.OPTLTE.AND.OPTXRAY) THEN
 LUMX1=0.D0
 LUMX2=0.D0
 LUMX3=0.D0

 DO K=1,IFRE   
   IF(FRE(K).GT.XRED1) EXIT
 ENDDO 
 IXR1=K-1

 DO K=IXR1,IFRE   
   IF(FRE(K).GT.XRED2) EXIT
 ENDDO 
 IXR2=K-1

 DO K=IXR2,IFRE   
   IF(FRE(K).GT.XRED3) EXIT
 ENDDO 
 IXR3=K-1
 
 DO K=IXR3,IFRE   
   IF(FRE(K).GE.XBLU) EXIT
 ENDDO
 IXB=K

 IF(FRE(IXB).LE.XBLU) STOP ' CONT: IXB NOT FOUND IN FRE!' 
! missing RTAU23^2 corrected (Oct. 16, 2014)
 CONVLX= 4.D0*PI/(SIGSB*TEFF**4*RTAU23**2) !for conversion, see below
 IF(ABS(4.D0-CONVLX*BNUECON).GT.1.D-6) STOP ' ERROR IN CONVLX'
ENDIF 

IF (IFRE.LT.50) THEN  
     IMOD = 1  
ELSE  
     IMOD = IFRE/50  
END IF  

PRINT *  
PRINT *,' NOMINAL RADIUS = ',RTAU23  
PRINT *  
WRITE (*,FMT='(2A)') &
&      '   NO        LAMBDA       LOG F-NUE       TRAD  ', &
&      '     TRAD1      RTHIN   LTHIN  R(TAU=1)'
PRINT *  
!
!---- output file for continuum fluxes
!
IF (UNASOL) THEN  
  OPEN (23,FILE=TRIM(MODNAM)//'/FLUXCONT',STATUS='UNKNOWN', &
&   FORM= 'FORMATTED')
    WRITE (23,FMT='(2A)') &
&      '   NO        LAMBDA       LOG F-NUE       TRAD  ', &
&      '     TRAD1      RTHIN   LTHIN  R(TAU=1)'
END IF  

FRELOOP: DO K = 1,IFRE  

     XLAMBDA = 1.D8/FRE(K)  
     CALL DIFFUS(XLAMBDA,TEMP,R,ND,AIC,DBDR)  

     DO L = 1,ND  
          OP = OPAC(L,K)  
!----  OPAC already effective, from OPACITC 
!      thus just rescale, and we're fine with ratio 
!      (since thomson_lines etc. are mean quantities)
!----  NOTE: This routine called directly after OPACITC, 
!      so OK to scale with opa_eff_rat (rather than old values opa_eff_rat_old) 
          IF(OPTLINES) THEN
!            THOMSON(L) = (SIG*XNE(L)/CLF(L)+THOMSON_LINES(L,K)+RAYLEIGH(L,K)/CLF(L))/OP  
            THOMSON(L) = (SIG*XNE(L)/CLF(L)+THOMSON_LINES(L,K))/OP * OPA_EFF_RAT(L,K)   
          ELSE
!            THOMSON(L) = (SIG*XNE(L)+RAYLEIGH(L,K))/CFL(L)/OP
            THOMSON(L) = (SIG*XNE(L))/CLF(L)/OP * OPA_EFF_RAT(L,K)   
          ENDIF  
          IF (THOMSON(L).GE.1.D0) then
            PRINT *,'WARNING: ERROR IN THOMSON',L,THOMSON(L)
            PRINT*,K,XLAMBDA,OP,SIG,XNE(L)/CLF(L),THOMSON_LINES(L,K)
            THOMSON(L)=1. ! should happen only between interpolation of freq. grids
          ENDIF  
          OPA(L) = OP*SR  
     END DO

     IF (MOD(K,IMOD).EQ.0 .OR. UNASOL) THEN
! calculate r(tau_k=1) for output frequencies
     TAU=TAUR(2)/10.
     DO L = 2,ND
       DT=0.5*(OPA(L-1)+OPA(L))*(R(L-1)-R(L))
       IF(TAU+DT.GT.1.) GOTO 10
       TAU=TAU+DT
     ENDDO   
     PRINT*,' TAU = ',TAU,' (< 1! AT ',XLAMBDA
     STOP ' TAU(K) < 1 AT PHOTOSPHERE'

! assuming log tau linear in r 
10   R1=R(L-1)-LOG10(TAU)*(R(L)-R(L-1))/LOG10(1.D0+DT/TAU)
     IF(R1.GT.R(L-1).OR.R1.LT.R(L)) STOP ' SOMETHING WRONG WITH R(TAU=1)!'
     R1K(K)=R1
     ENDIF
!
!-----------------------------------------------------------------------
!
!     calculation of continuum radiation field
!
!     if(xlambda.lt.8.79) then
!       print*,xlambda,aic,dbdr
!       do l=1,nd
!         print*,l,opa(l),thomson(l),st(l,k)
!       enddo
!       print*
!     endif

!---- note: at the end of CONT1, XJ to XN (1,ND) are saved at each frequency,
!-----and further processed in subr. CONVERSION and FACTINC, to be used
!-----in TRATRAD for Sobo-lines 
     CALL CONT1(ND,NP,R,P,Z,ST(:,K),OPA,THOMSON,AIC,DBDR,CORRFC, &
      WP,WP2,ALO(:,K),XJ(:,K),XH,.TRUE.,XLAMBDA,Q1,Q2,K,CONCON,RYBICKI)
!
!---- note that xh is not the eddington flux but (h*r**2). to obtain
!---- emergent absolute fluxes it has to be multiplied by 4*pi/r**2
!
!     for tests
!     IF(XLAMBDA.GT.912..AND.XLAMBDA.LT.1230.) XH=XH*0.7 
!
     XIC = 4.D0*XH(1)  
     XICIN = 4.D0*XH(ND+1)  
     HC(K) = XIC*PI/R(1)/R(1)  
!
!    for test purposes     
!
!     alo(:,k)=0.
!     do 150 l=1,nd
!     if(k.ge.1118.and.k.le.1150) then
!       write(*,151) xlambda,l,opa(l),st(l,k),thomson(l),xj(l,k)
!       print*,xlambda,l,xj(l,k),st(l,k),alo(l,k),thomson(l),opa(l)
!     endif
!     aux=1./(1.-alo(l,k)*thomson(l))
!     alolk=alo(l,k)*aux
!     if(xj(l,k)/st(l,k).lt.alolk) stop ' alo > j/s'
!150  continue
!     print*
!151  format(f8.3,1x,i2,1x,4(e8.3,1x))
!-----------------------------------------------------------------------
!
! for testing LTE
!     do l=1,nd
!       xj(l,k)=bnue(xlambda,temp(l))
!       alo(l,k) =0.
!     enddo  

     DO L = 2,ND  
          XICR = 4.D0*XH(L)  
          IF (ABS(1.D0-XICR/XIC).GT..1D0) GO TO 50  
     END DO

   50      CONTINUE  

     RTHIN = R(L-1)  
     LTHIN(K) = L - 1  

     IF (XIC.GT.0.D0) THEN  
          TRADIC = 1.4388D8/XLAMBDA/ (LOG(3.97D8/XLAMBDA**3/XIC+1.D0))
          XICV1 = XIC/ (RTAU23**2)  
          TRADIC1 = 1.4388D8/XLAMBDA/ (LOG(3.97D8/XLAMBDA**3/XICV1+ 1.D0))
     ELSE  
          TRADIC = -1.4388D8/XLAMBDA/ (LOG(3.97D8/XLAMBDA**3/ABS(XIC)+1.D0))
          XICV1 = XIC/ (RTAU23**2)  
          TRADIC1 = -1.4388D8/XLAMBDA/ (LOG(3.97D8/XLAMBDA**3/ABS(XICV1)+1.D0))
     END IF  
!
!---- normal running output (reduced)
!
     IF (MOD(K,IMOD).EQ.0) WRITE (*,FMT=9000) K,XLAMBDA, LOG10( &
      ABS(HC(K))),TRADIC,TRADIC1,RTHIN,LTHIN(K),R1K(K)
!
!---- absolute fluxes at r=rmax are displayed
!
!
!---- converged output (full)
!
     IF (UNASOL) WRITE (23,FMT=9000) K,XLAMBDA,LOG10(ABS(HC(K))), &
      TRADIC,TRADIC1,RTHIN,LTHIN(K),R1K(K)

     WFR = WFRE(K)  
     FLUXCON = FLUXCON + WFR*XIC  
     FLUXCIN = FLUXCIN + WFR*XICIN  

!---- calculate X-ray luminosity in three bands (staggered grid)
     IF(.NOT.OPTLTE.AND.OPTXRAY.AND.K.GE.IXR1.AND.K.LE.IXB) THEN 
       AUX = WFR *XH
       LUMX1 = LUMX1 + AUX
       IF(K.GE.IXR2) LUMX2 = LUMX2 + AUX
       IF(K.GE.IXR3) LUMX3 = LUMX3 + AUX
     ENDIF

! here was a bug (see below):
! fltot defined with respect to 'regular' grid and not to staggered grid
!     DO L = 2, ND  
!          FLTOT(L) = FLTOT(L) + 4.D0*XH(L)*WFR  
!     END DO
     IF (OPTLTE) FLUXDIF = FLUXDIF + WFR*4.D0*DBDR*CORRFC/ (3.D0*OPA(ND))

!---- calculation of flux-weighted opacity

!---- at first, interpolate H onto radial grid
!---- note that xh here is h*r**2
     XHMIN = 0.D0  
     DO L = 2,ND  
     XHMIN = MIN(XHMIN,XH(L))  
     END DO

     DESH = 2.D0*ABS(XHMIN)  

     DO L = 2,ND  
       XH(L) = LOG10(XH(L)+DESH)  
     END DO

     DO L = 1,ND - 2  
     XH(L+1) = Q1(L)*XH(L+1) + Q2(L)*XH(L+2)  
     XH(L+1) = 10.D0**XH(L+1) - DESH  
     END DO  
     XH(ND) = XH(ND+1)  

!---- now, calculate chibar_h, corrected for clumping 
!     NOTE: dp/dm=4pi/(c rho) H chibar; dp=4pi/(c rho) H chibar rho dr  
!            = 4pi/c H chibar dr;
!     to obtain correct result, volume filling factor is needed
!     ALREADY INCLUDED IN OPAC
!JO check whether optically thick case consistent as well
!     Thus, no additional correction required if average density rho is used 
!---- new version: standard formulation with chi/rho


     DO L=1,ND
       OP=4.D0*XH(L)*WFR
       HTOT(L)=HTOT(L) + OP
       CHIBAR_H(L)=CHIBAR_H(L) + OP*OPAC(L,K)/RHO(L)
     ENDDO
! corrected in v10.4.1 (July 2016): fltot has to be calculated HERE
! (after XH defined on regular mesh);
! this was already corrected in v10.4 of CMF-path (Jan 2015) 
     DO L = 2, ND  
          FLTOT(L) = FLTOT(L) + 4.D0*XH(L)*WFR  
     END DO

END DO FRELOOP  

IF (UNASOL) WRITE (23,FMT=*) RTAU23  
!
!---- transformation between frequency stored binary files and depth
!---- ones
!
IF (CONCON) CALL CONVERSION  

IF (ABS(1.D0-FLUXDIF/BNUECON).GT.PRECIS) STOP ' ERROR IN DBDR'  

PRINT *  
PRINT *,' USED CORRECTION FACTOR = ',CORRFC  
PRINT *  
PRINT *,'  SIGSB/PI * TEFF**4 * RTAU23^2 = ',BNUECON  
PRINT *,' INTEGRAL(4*HNUE*DNUE)           = ',FLUXCIN,' (R_IN)'  
PRINT *,' INTEGRAL(4*HNUE*DNUE)* RMAX^2   = ',FLUXCON,' (R_OUT)'  

CHIBAR_H = CHIBAR_H/HTOT
!---- write actual CHIBAR_H (reg. grid) and grad to file CHIBAR_H
REWIND 60
DO L=1,ND
  WRITE(60,*) L,CHIBAR_H(L),CHIBAR_H(L)*SIGSB/CLIGHT*TEFF**4
ENDDO

! note: following quantities flux-errors;
! since H values actually (r/sr)^2*H and H_nom = sigsb/(4pi)*Teff^4 at r=rstar
! fltot/bnuecon=4*(r/sr)^2 * H_int(r)/(sigsb/pi*teff^4*(rstar/sr)^2 =
! (r/rstar)^2 * H_int(r) /(sigsb/(4pi)*teff^4) = 1 if no error!
FERRI = FLUXCIN/BNUECON - 1.D0  
FERRO = FLUXCON/BNUECON - 1.D0  

FLTOT(1)=FERRO
DO L=2,ND
  FLTOT(L)=FLTOT(L)/BNUECON-1.D0
ENDDO  
! new check
IF(FLTOT(ND).NE.FERRI) STOP ' ERROR IN FLTOT'

FERRMAX = MAXVAL(ABS(FLTOT(2:ND))) 


IF(.NOT.OPTLTE.AND.OPTXRAY) THEN
  LUMX1=LUMX1*CONVLX
  LUMX2=LUMX2*CONVLX
  LUMX3=LUMX3*CONVLX
  WRITE (9,9020) IIT, R(1),LUMX1(1), LUMX2(1), LUMX3(1)
  PRINT*
  WRITE (*,*) "**************************************************"
  WRITE (*,*) "radius (staggered)  L_x/L_bol"
  WRITE (*,*) "**************************************************"
  WRITE (*,*) '(range =',1.d8/fre(ixr1),' to',1.d8/fre(ixb),' A'
  WRITE (*,*) '       =',fre(ixr1)/const_ev,' to',fre(ixb)/const_ev,' eV'
  WRITE (*,*) ' 2nd and 3rd range until',fre(ixr2)/const_ev,' and',fre(ixr3)/const_ev,' eV)'
  L=1
  WRITE (*,9020) L,R(L),LUMX1(L),LUMX2(L),LUMX3(L)
  DO L=2,ND
    DESH=0.5D0*(R(L-1)+R(L))
    WRITE (*,9020) L,DESH,LUMX1(L),LUMX2(L),LUMX3(L)
  ENDDO
  L=ND+1
  WRITE (*,9020) L,R(L-1),LUMX1(L),LUMX2(L),LUMX3(L)
  PRINT*
ENDIF  
  
IF (UNASOL) THEN  
     WRITE (23,FMT=*)  
     WRITE (23,FMT=*) '  L        R        TAUR       FLUX-ERROR  '  
     L = 1  
     WRITE (23,FMT=9010) L,R(L),TAUR(L),FERRO  

     DO L = 2,ND - 1  
          WRITE (23,FMT=9010) L,R(L),LOG10(TAUR(L)), FLTOT(L)
     END DO

     L = ND  
     WRITE (23,FMT=9010) L,R(L),LOG10(TAUR(L)),FERRI 
     
     
     IF(.NOT.OPTLTE.AND.OPTXRAY) THEN
! CONVERT TO Lx/Lbol
! here was a (moderate) bug (corrected by Jo, Oct. 16, 2014)
! LX = 4 pi Rstar^2 * 4 pi (r/Rstar)^2 H = 4 pi Rstar^2 * 4 Pi (r/SR)^2 H * (SR/Rstar)^2 
! => Lx/Lbol = 4 pi Int(Hc)/(sigsb * Teff^4 * RTAU23^2)
! WITH RTAU23=RSTAR/SR       
        OPEN (UNIT=1,FILE= TRIM(MODNAM)//'/XLUM', STATUS = 'UNKNOWN')
        WRITE (1,*) "**************************************************"
        WRITE (1,*) "radius (staggered)  L_x/L_bol"
        WRITE (1,*) "**************************************************"
        L=1
        WRITE (1,9020) L,R(L),LUMX1(L),LUMX2(L),LUMX3(L)
        DO L=2,ND
          DESH=0.5D0*(R(L-1)+R(L))
          WRITE (1,9020) L,DESH,LUMX1(L),LUMX2(L),LUMX3(L)
        ENDDO
        L=ND+1
        WRITE (1,9020) L,R(L-1),LUMX1(L),LUMX2(L),LUMX3(L)
        CLOSE(1)
     ENDIF
END IF  

CLOSE (23)  

PRINT *  
PRINT *,' DIFFUSION APPROX. NOT VALID BY ',FERRI  
PRINT *,' FLUX (RMAX)  NOT  CONSERVED BY ',FERRO  
PRINT *,' MAXIMUM FLUX ERROR = ',FERRMAX  
PRINT *  

! TEMPERATURE CORRECTION WITH RESPECT TO FLUX CONSERVATION
CALL TCORR(TAUR,FLTOT,TEFF,TEMP,ND,DTFCORR,NDIV)

RETURN  

 9000 FORMAT (I5,2X,F15.5,2X,F10.5,2 (2X,F10.0),3X,F8.3,3X,I3,3X,F8.3)  
 9010 FORMAT (I3,3 (2X,F10.5))  
 9020 FORMAT (I4,'     ',F10.4,' ',3(E10.4,2x))

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE TCORR(TAUR,FERR,TEFF,TEMP,ND,DT,NDIV)
! temperature correction with respect to flux conservation

! to be used in lower part of atmosphere
  
! modified Lucy Unsoeld procedure (Mihalas page 174), &
! assuming all averaged opacities are equal.  
!
! modified in V10.1 (check for closely separated grid-points, &
! and improve calculation of df/dt)
USE nlte_type

IMPLICIT NONE
!
!
INTEGER(I4B), intent(in) :: ND,NDIV
  
REAL(DP), DIMENSION(ND), intent(in) ::  TAUR,FERR,TEMP
REAL(DP), intent(in) :: TEFF
REAL(DP), DIMENSION(ND), intent(out) :: DT

INTEGER(I4B) :: I , IS, II
REAL(DP), DIMENSION(ND) :: DF, DFDT
REAL(DP) :: TAULIM,AUX, DLOGM, DLOGP, TAURM, TAURP, FERRM, FERRP


TAULIM=TAUR(NDIV)
TAULIM=TAULIM*2. !safety factor, to be well away from the transition zone
IF(TAULIM.GT.0.1) TAULIM=0.1 !the old standard value
IF(TAULIM.LT.0.001) TAULIM=0.001

DO I=1,ND
  IF(TAUR(I).GE.TAULIM) EXIT
ENDDO  

IS=I

PRINT*,'TCORR'
DF(IS-1)=0.
DO I=IS,ND 

! integral f(tau) dtau, expressed with respect to dlntau
DF(I)=DF(I-1)+.5*(FERR(I-1)*TAUR(I-1)+FERR(I)*TAUR(I))* &
& (LOG(TAUR(I))-LOG(TAUR(I-1)))
IF (I.EQ.ND) THEN 
  DFDT(I)=(FERR(I-1)-FERR(I))/(TAUR(I-1)-TAUR(I))
ELSE 
! derivative df(tau)/dtau, expressed with respect to dlntau
! check for closely separated grid-points
DLOGM=LOG(TAUR(I-1))-LOG(TAUR(I))
DLOGP=LOG(TAUR(I+1))-LOG(TAUR(I))

IF(ABS(DLOGM).LT.0.1 .OR. ABS(DLOGP).LT.0.1) THEN ! factor 1.1
 TAURM=TAUR(I)/1.1
 TAURP=TAUR(I)*1.1
 DO II=I-1,1,-1
   IF(TAUR(II).LT.TAURM .AND. TAUR(II+1).GT.TAURM) EXIT
 ENDDO  
 FERRM=FERR(II)+(FERR(II+1)-FERR(II))/(TAUR(II+1)-TAUR(II))*(TAURM-TAUR(II))
!JO changed Jan 2016 (old version: loop till ND, no reset of II)
 DO II=I,ND-1
   IF(TAUR(II).LT.TAURP .AND. TAUR(II+1).GT.TAURP) GOTO 10
 ENDDO  
 II=ND-1  !in case no point is found 
10  FERRP=FERR(II)+(FERR(II+1)-FERR(II))/(TAUR(II+1)-TAUR(II))*(TAURP-TAUR(II))

 DFDT(I)=.5/TAUR(I)*((FERRM-FERR(I))/(LOG(TAURM)-LOG(TAUR(I)))+ &
&    (FERRP-FERR(I))/(LOG(TAURP)-LOG(TAUR(I))))
ELSE
 DFDT(I)=.5/TAUR(I)*((FERR(I-1)-FERR(I))/DLOGM + (FERR(I+1)-FERR(I))/DLOGP)
ENDIF

AUX=ABS(DFDT(I))
! to avoid too large corrections in case of very closely separated
! grid points; idea max df/dlntau from 0.03/(.33*ln10) (3 points per
! decade) plus safety factor 1.5  
  IF(AUX*TAUR(I).GT.0.06) THEN
!    PRINT*,'orig',I,DFDT(I)
    DFDT(I)=AUX/DFDT(I)*0.06/TAUR(I)
!    PRINT*,'corr',I,DFDT(I)
  ENDIF
ENDIF
ENDDO

DT(1:IS-1)=0.D0
DO I=IS,ND
   DT(I)=1.D0/16.D0*(TEFF/TEMP(I))**4*TEMP(I)*(-3.*DF(I)+DFDT(I))
   AUX=ABS(DT(I))
! restrict correction to 7.5%
   IF (AUX .GT. 0.075*TEMP(I)) DT(I)= 0.075* AUX/DT(I)*TEMP(I) !correct sign
! for tests
!   PRINT*,DF(I),DFDT(I),-3.*DF(I)+DFDT(I)
   WRITE (*,FMT='(1X,I2,4(1X,G14.7))') I,TAUR(I),FERR(I),TEMP(I),DT(I)
ENDDO

RETURN
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CONT1(ND,NP,R,P,Z,ST,OPA,THOMSON,XIC1,XIC2,CORR,WP,WP2, &
&                 ALO,XJ,XH,OPTFLUX,XLAM,Q1,Q2,K,CONCON,RYBICKI)
!
!.......................................................................
!
!     attention!!! subroutine cont1 changed for fast and consistent
!     continuum transfer in program nonplusultra. so far, only case
!     i) can be treated. in this new version, both j and h are taken
!     from the solution of the moments' equation( here, h corresponds
!     to r**2 * h-nue).
!     additionally, the alo is calculated consistent with j.
!
!
!
!     calculates the solution of the following equation applying the
!     rybicki algorithm (including the solution of moments' equation,
!                        if necessary , for case i) )
!
!     xj = lambda (st+thomson*xj) ,
!
!     or, explicitly written
!
!     1)     j = lamdda (s)
!     2) dj/dt = lambda (ds/dt)
!
!     where s is the usual nlte-source function including electron
!     scattering
!
!     upper boundary condition : valid to 3rd order including self-
!     consistent calculated i-minus  (dj/dtau neglected)
!
!     lower boundary condition: valid to 2nd order
!     diffusion approximation possible : xic1 = bnue
!                                        xic2 = dbnue/dr
!     rescaling of lower input flux possible : corr = h-true/h-actual
!
!     optflux = .true. : flux = xh is calculated, where
!     xh(1) , xh(nd+1) boundary values consistent
!                      with actual i-plus,i-minus
!     xh(i),i=2...nd   intermesh values at .5*(r(i-1)+r(i))
!
!
!     case 1) input:
!
!          st = (eta - opa-thomson*j)/opa
!         opa = total opacity * stellar radius
!     thomson = opa-thomson/opa
!        xic1 = i-plus at lower boundary (diff.approx. = bnue(t))
!        xic2 = 0. (diff. approx. = dbnue/dr)
!
!     output : xj...mean intensity
!              xh...eddington flux (optflux=.true.)
!
!
!     case 2) input:
!
!          st = d/dt((eta - opa-thomson*j)/opa) + d/dt(thomson) *j
!          opa , thomson as  above
!        xic1 = d/dt(i-plus)  (diff. approx. = d/dt(bnue(t))
!        xic2 = 0. (diff. approx. = d2/drdt (bnue(t)) )
!
!     output : xj = dj/dt
!
!
!.......................................................................
!
USE nlte_type
USE nlte_dim
USE run_once, ONLY: START_WEIGH11
USE nlte_var, ONLY: PRECIS,VP,VP2,W0LAST,XXK  

IMPLICIT NONE
!
!
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,NP1=ID_NPOIN  
INTEGER(I4B), PARAMETER :: IMAX=10  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORR,XIC1,XIC2,XLAM  
INTEGER(I4B) ::  K,ND,NP  
LOGICAL CONCON,OPTFLUX,RYBICKI  
!     ..
!     .. array arguments ..
REAL(DP) ::  ALO(ND1),OPA(ND1),P(NP1),Q1(ND1-2),Q2(ND1-2), &
&                 R(ND1),ST(ND1),THOMSON(ND1),WP(ND1+1,NP1-1), &
&                 WP2(ND1+1,NP1-1),XH(ND1+1),XJ(ND1),Z(ND1,NP1)
!     ..
!     .. local scalars ..
REAL(DP) ::  DTAUG,DXI,E0,E1,EDMUE,EPSMAX,HEDDI,HEDDO,HIN, &
&                 HOUT,PI2,UX,V,VI,XIPLUS
INTEGER(I4B) ::  III,IJ,JP,L,LL,LMAX,LSTART,LZ  
LOGICAL CONV,CORE  
!     ..
!     .. local arrays ..
REAL(DP) ::  AKP(ND1),AKPJP(ND1*NP1-1),AKSUM(ND1),F(ND1+3), &
&                 QQ(ND1),TA(ND1),TAJP(ND1*NP1-1),TAUM(NP1),TB(ND1), &
&                 TB1(ND1),TBJP(ND1*NP1-1),TC(ND1),TC1(ND1), &
&                 TCJP(ND1*NP1-1),TP(ND1,ND1),TSUM(ND1,ND1), &
&                 U(ND1,NP1-1),UP(ND1),UPJP(ND1*NP1-1),XHS(ND1+1), &
&                 XJOLD(ND1),XK(ND1),XNS(ND1+1)
!     ..
!     .. external subroutines ..
EXTERNAL DEV,INTERPOL,INV,INVTRI,INVTRI3,MADD,MDMV,MDV, &
&         MOMALO,MOMCONT,MVMD,MVV,SETUP1,VADD,VSUB,WEIGH11,WEIGH5, &
&         WEIGH9
!     ..
!     .. intrinsic functions ..
!INTRINSIC ACOS,ATAN,EXP,MIN,MIN0  
!     ..

PI2 = ACOS(0.D0)  
CONV = .FALSE.  
!
!     calculation of integration weights for xj, xh only once
!---- also calculation of interpolation weights for xh and xn
!
IF (START_WEIGH11) THEN
    CALL WEIGH11(ND,NP,R,Z,P,WP,WP2,Q1,Q2)  
    CALL WEIGH9  
    CALL WEIGH5  
    START_WEIGH11 = .FALSE.
END IF  

!
IF (RYBICKI) THEN  
AKSUM = .0D0  
TSUM = .0D0  
END IF  

DO 40 JP = 1,NP - 1  

     IF (JP.EQ.1) THEN  
          TAUM(1) = OPA(1)*R(1)* (1.D0/3.D0+2.D0*THOMSON(1)/3.D0)  
     ELSE  
          TAUM(JP) = OPA(1)*R(1)*R(1)* (THOMSON(1)/P(JP)* (PI2- &
           ATAN(Z(1,JP)/P(JP)))+ (1.D0-THOMSON(1))*R(1)*R(1)/2.D0/ &
           P(JP)**3* (PI2-ATAN(Z(1,JP)/P(JP))-Z(1, JP)*P(JP)/R(1)/ &
           R(1)))
     END IF  

     IF (TAUM(JP).LT.0.D0) STOP 'TAUM NEGATIVE!'  

     LMAX = MIN0(NP+1-JP,ND)  
     CORE = (LMAX.EQ.ND)  
!
!     calculation of tp,akp
!
     CALL SETUP1(LMAX,CORE,Z(:,JP),ST,OPA,THOMSON,XIC1,XIC2,CORR, &
      UP,AKP,TA,TB,TC,TAUM(JP))
     TA(1)=0.D0
     TC(LMAX)=0.D0
!
!---  storage for further use
!
!---  start index
!
     LSTART = ND1* (JP-1)  
     DO L = 1,LMAX  
          LL = LSTART + L  
          UPJP(LL) = UP(L)  
          AKPJP(LL) = AKP(L)  
          TAJP(LL) = TA(L)  
          TBJP(LL) = TB(L)  
          TCJP(LL) = TC(L)  
     END DO 
!
     IF (.NOT.RYBICKI) GO TO 40  
!
!     rybicki algorithm to obtain ajc
!
     CALL INVTRI3(TA,TB,TC,TP,LMAX,ND)  
     CALL MDMV(TP,VP(:,JP),LMAX,ND)  
     CALL MVV(QQ,TP,AKP,LMAX,LMAX,ND)  
     CALL VADD(AKSUM,QQ,LMAX)  
!
!     tp=:(vp*tp-1)*u
!
     CALL MVMD(TP,UP,LMAX,ND)  
     CALL MADD(TSUM,TP,LMAX,ND)  
   40 END DO  
!
!     addition of unity-matrix to tsum
!

IF (RYBICKI) THEN  
     DO L = 1,ND  
       TSUM(L,L) = TSUM(L,L) + 1.D0  
     END DO  
     CALL INV(ND,ND,TSUM)  
     CALL MVV(XJ,TSUM,AKSUM,ND,ND,ND)  
END IF  

QQ = XJ  
XJOLD = XJ  

IF (.NOT.OPTFLUX .AND. RYBICKI) RETURN  
!
!     backsubstitution to obtain u
!
III = 0  

   70 CONTINUE  

III = III + 1  

JPLOOP: DO JP = 1,NP - 1  

     LMAX = MIN0(NP+1-JP,ND)  
     CORE = (LMAX.EQ.ND)  
!
!---  start index
!
     LSTART = ND1* (JP-1)  
     DO L = 1,LMAX  
          LL = LSTART + L  
          UP(L) = UPJP(LL)  
          AKP(L) = AKPJP(LL)  
          TA(L) = -TAJP(LL)  
          TB(L) = TBJP(LL)  
          TC(L) = -TCJP(LL)  
     END DO
!
!     recalculation of total source-function
!
     CALL MDV(QQ,UP,LMAX)  
     CALL VSUB(AKP,UP,LMAX)  

     CALL INVTRI(TA,TB,TC,AKP,LMAX)  

     DO L = 1,LMAX  
          U(L,JP) = AKP(L)  
     END DO 

END DO JPLOOP  
!
!     recalculation of j and calculation of k
!
XJ = 0.D0  
XK = 0.D0  

DO JP = 1,NP - 1  
     LMAX = MIN0(NP+1-JP,ND)  
     DO L = 1,LMAX  
          UX = U(L,JP)  
          XJ(L) = XJ(L) + VP(L,JP)*UX  
          XK(L) = XK(L) + VP2(L,JP)*UX  
     END DO 
END DO  

DO L = 1,ND  
     F(L) = XK(L)/XJ(L)  
END DO  
!
!     calculation of inner and outer eddington factors respective to h
!
!---- this is the old form for boundary conditions (i.e., achim's form
!---- instead of gudrun's). i prefer it because the first one takes into
!---- account the new radiation field in the calculation of h-nue as the
!---- outer boundary condition.
!
HIN = 0.D0  
HOUT = 0.D0  
EDMUE = 0.D0  

DO JP = 1,NP - 1  

     LMAX = MIN0(NP+1-JP,ND)  
!
!     outer boundary
!
     IF (TAUM(JP).LT.100.D0) THEN  
          E0 = EXP(-TAUM(JP))  
          E1 = 1.D0 - E0  
     ELSE  
          E1 = 1.D0  
     END IF  

!     DXI = U(1,JP) - E1* (ST(1)+THOMSON(1)*XJ(1))  
!
!---- this line has been masked (21-10-92). it corresponds to case
!---- i_plus<i_minus (at r_max), and now there is no difference
!     if(dxi.lt.0.d0) e1=u(1,jp)/(st(1)+thomson(1)*xj(1))
!
     HOUT = HOUT + U(1,JP)*WP(1,JP)  

     EDMUE = EDMUE + E1*WP(1,JP)  
!
!---- with the last modification, edmue is always i_minus(1)/s_total(1)
!---- integrated over mue
!
!     inner boundary
!
     IF (LMAX.EQ.ND) THEN  
          HIN = HIN + U(ND,JP)*WP(ND+1,JP)  
     END IF  

END DO  

HEDDO = HOUT/XJ(1)  
HEDDI = HIN/XJ(ND)  
F(ND+1) = HEDDO  
F(ND+2) = HEDDI  
F(ND+3) = EDMUE  

CALL MOMCONT(ND,R,OPA,THOMSON,ST,XIC1,XIC2,CORR,QQ,XH,TA,TB,TC, &
&             TB1,TC1,AKP,ALO,XJ,F,.FALSE.)
!
!-----------------------------------------------------------------------
!

IF (CONV .OR. III.EQ.IMAX) THEN  

     DO L=1,ND
       XK(L)=F(L)*XJ(L)
       XXK(L,K)=XK(L)
     ENDDO

     IF (.NOT.CONCON) GO TO 190  
!
!      
!--------------------------------------------------------------------------
!---- here we start the specific treatment for lines. h_nue and n_nue
!     are calculated so that ic can be expressed as a combination of
!     blocking factors and mue's. from gudrun taresch's program.
!     a new iteration step has been made, so that u's values be more
!     accurate
     
     XNS = 0.D0
     XHS = 0.D0
!
     DO JP = 1,NP - 1  

          LMAX = MIN(NP+1-JP,ND)  
          LZ = LMAX - 1  
!
!---------- outer boundary for h,n
!
          E1 = 1.D0 - EXP(-TAUM(JP))  
          DXI = U(1,JP) - E1* (ST(1)+THOMSON(1)*XJ(1))  
          XNS(1) = XNS(1) + DXI*WP2(1,JP)  
          XHS(1) = XHS(1) + DXI*WP(1,JP)  
!
!---------- intermesh points of h,n
!
          DO LL = 1,LZ  
               DTAUG = 2.D0/ ((OPA(LL)+OPA(LL+1))* (Z(LL+1,JP)-Z( &
                LL,JP)))
               V = (U(LL,JP)-U(LL+1,JP))*DTAUG  
               XHS(LL+1) = XHS(LL+1) + V*WP(LL+1,JP)  
               XNS(LL+1) = XNS(LL+1) + V*WP2(LL+1,JP)  
          END DO
!
!---------- inner boundary for h, n
!
          IF (LMAX.EQ.ND) THEN  
               XIPLUS = XIC1 + Z(ND,JP)*XIC2/OPA(ND)*CORR  
               VI = XIPLUS - U(LMAX,JP)  
               XHS(ND+1) = XHS(ND+1) + VI*WP(ND+1,JP)  
               XNS(ND+1) = XNS(ND+1) + VI*WP2(ND+1,JP)  
          END IF  

     END DO
!
!------- interpolation of h and n to get points on the grid
!
     CALL INTERPOL(XHS,XNS,ND,Q1,Q2)  
!
!------- copy of momenta to a binary file
!
!
      WRITE (50,REC=K) XJ
      WRITE (52,REC=K) XK
      WRITE (51,REC=K) (XHS(IJ),IJ=1,ND)
      WRITE (53,REC=K) (XNS(IJ),IJ=1,ND)
!----------------------------------------------------------------------------

  190      CONTINUE  

     IF (III.EQ.IMAX.AND.EPSMAX.GT.1.D-5) PRINT *, &
&        ' ACHIEVED ACCURACY IN CONT. TRANSPORT AT ',XLAM,' = ',EPSMAX

     CALL MOMALO(ND,R,OPA,THOMSON,ST,TA,TB1,TC1,QQ,F,AKP,ALO,XH,XIC2,CORR)  

     RETURN  

END IF  
!
!-----------------------------------------------------------------------
!
CALL DEV(R,XJ,XJOLD,ND,EPSMAX,.FALSE.)  

IF (EPSMAX.LT.PRECIS .OR. III.EQ.IMAX) CONV = .TRUE.  

QQ=XJ
XJOLD=XJ

GO TO 70  

END
!
!----------------------------------------------------------------
!
SUBROUTINE DEV(R,A1,A,ND,EPSMAX,OPT)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     .. scalar arguments ..
REAL(DP) ::  EPSMAX  
INTEGER(I4B) ::  ND  
LOGICAL OPT  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(ND),A1(ND),R(ND)  
!     ..
!     .. local scalars ..
REAL(DP) ::  E1,E2,E3,E4,E5,EPS1,EPS2,EPS3,EPS4,EPS5  
INTEGER(I4B) ::  L  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,MAX  
!     ..

EPS1 = 0.D0  
EPS2 = 0.D0  
EPS3 = 0.D0  
EPS4 = 0.D0  
EPS5 = 0.D0  
DO L = 1,ND  
     IF (R(L).GE.50.D0) THEN  
          E1 = ABS(1.D0-A1(L)/A(L))  
          IF (E1.GT.EPS1) EPS1 = E1  

     ELSE IF (R(L).LT.50.D0 .AND. R(L).GE.10.D0) THEN  
          E2 = ABS(1.D0-A1(L)/A(L))  
          IF (E2.GT.EPS2) EPS2 = E2  

     ELSE IF (R(L).LT.10.D0 .AND. R(L).GE.5.D0) THEN  
          E3 = ABS(1.D0-A1(L)/A(L))  
          IF (E3.GT.EPS3) EPS3 = E3  

     ELSE IF (R(L).LT.5.D0 .AND. R(L).GE.2.D0) THEN  
          E4 = ABS(1.D0-A1(L)/A(L))  
          IF (E4.GT.EPS4) EPS4 = E4  

     ELSE IF (R(L).LT.2.D0) THEN  
          E5 = ABS(1.D0-A1(L)/A(L))  
          IF (E5.GT.EPS5) EPS5 = E5  
     END IF  
END DO

EPSMAX = MAX(EPS1,EPS2,EPS3,EPS4,EPS5)  

IF (OPT) THEN  
    PRINT *  
    PRINT *,' MAXIMUM DEVIATION ',EPSMAX  
    PRINT *  
    PRINT *,'  1. < R <   2.  : ',EPS5  
    PRINT *,'  2. < R <   5.  : ',EPS4  
    PRINT *,'  5. < R <  10.  : ',EPS3  
    PRINT *,' 10. < R <  50.  : ',EPS2  
    PRINT *,' 50. < R < 100.  : ',EPS1  
END IF  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE INTERPOL(XHS,XNS,ND,Q1,Q2)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!---- this subroutine interpolates h and n to get points on the r-grid
!---- changed (26/3/93) so that negative fluxes does not affect the
!---- logarithmic interpolation
!----
!---- comment from jo: this is an elegant, but not very exact way,
!---- since it is more appropriate for linear interpolation. however,
!---- since log (c*r^n + desh) = log(c'*r^n) with c' = c + desh*r^(-n),
!---- the real assumption underlying this approach is that c' instead
!---- of c is constant over the considered interval. this is not too
!---- bad and was checked by myself that it works fairly well.
!
!---- if somebody has doubt about the above statement or there are
!---- conspicous cases, uncomment the commented lines below and try!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND  
!     ..
!     .. array arguments ..
REAL(DP) ::  Q1(ND1-2),Q2(ND1-2),XHS(ND1+1),XNS(ND1+1)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DESH,DESN,XHMIN,XNMIN  
INTEGER(I4B) ::  L  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,LOG10,MIN  
!     ..

XHMIN = 0.D0  
XNMIN = 0.D0  
DO L = 2,ND  
     XHMIN = MIN(XHMIN,XHS(L))  
     XNMIN = MIN(XNMIN,XNS(L))  
END DO

DESH = 2.D0*ABS(XHMIN)  
DESN = 2.D0*ABS(XNMIN)  

DO L = 2,ND  
     XHS(L) = LOG10(XHS(L)+DESH)  
     XNS(L) = LOG10(XNS(L)+DESN)  
END DO

DO L = 1,ND - 2  
     XHS(L+1) = Q1(L)*XHS(L+1) + Q2(L)*XHS(L+2)  
     XNS(L+1) = Q1(L)*XNS(L+1) + Q2(L)*XNS(L+2)  
     XHS(L+1) = 10.D0**XHS(L+1) - DESH  
     XNS(L+1) = 10.D0**XNS(L+1) - DESN  
!         xhmin=min(xhold(l+1),xhs(l+1),xhold(l+2))
!         xhmax=max(xhold(l+1),xhs(l+1),xhold(l+2))
!         xnmin=min(xnold(l+1),xns(l+1),xnold(l+2))
!         xnmax=max(xnold(l+1),xns(l+1),xnold(l+2))
!         if(xhmin.eq.xhs(l+1)) stop ' error in interpolation'
!         if(xhmax.eq.xhs(l+1)) stop ' error in interpolation'
!         if(xnmin.eq.xns(l+1)) stop ' error in interpolation'
!         if(xnmax.eq.xns(l+1)) stop ' error in interpolation'
END DO  

XHS(ND) = XHS(ND+1)  
XNS(ND) = XNS(ND+1)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE MOMALO(ND,R,OPA,THOMSON,ST,TA,TB1,TC1,Q,F,AKP,ALO,XH,XIC2,CORR)  
!
! AKP on input is here r^2 *J
!  
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT 
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND  
REAL(DP) :: XIC2, CORR
!     ..
!     .. array arguments ..
REAL(DP) ::  AKP(ND1),ALO(ND1),F(ND1+3),OPA(ND1),Q(ND1), &
&                 R(ND1),ST(ND1),TA(ND1),TB1(ND1),TC1(ND1), &
&                 THOMSON(ND1),XH(ND1+1)
!     ..
!     .. local scalars ..
REAL(DP) ::  DTAU1,R2  
INTEGER(I4B) ::  L  
!     ..
!     .. external subroutines ..
EXTERNAL DIAG  
!     ..

CALL DIAG(TA,TB1,TC1,ALO,ND,1)  

R2 = R(1)*R(1)  
XH(1) = AKP(1)* (F(ND+1)-F(ND+3)*THOMSON(1)) - F(ND+3)*ST(1)*R2  

DO L = 2,ND  
     DTAU1 = 2.D0/ ((R(L-1)-R(L))* (OPA(L-1)+OPA(L)))  
     XH(L) = (F(L)*Q(L)*AKP(L)-F(L-1)*Q(L-1)*AKP(L-1))*DTAU1/ &
      (.5D0* (Q(L-1)+Q(L)))
! LEAVE THIS STATEMENT HERE; OTHERWISE AN OPTIMIZATION ERROR WITH THE
! RESULT XH = 0 MIGHT OCCUR!
! ALSO, THIS CAN HAPPEN INDEED (FOR X-RAYS, IN FIRST ITERATION)
! and also in iteration zero 
     IF(XH(L).EQ.0.D0) THEN
!       PRINT*,L,DTAU1,XH(L)
!       PRINT*,F(L),Q(L),AKP(L)
!       PRINT*,F(L-1),Q(L-1),AKP(L-1)
!       PRINT*,'WARNING!!!! XH = 0 IN MOMALO!!!!!'
       XH(L)=1.D-40
!       STOP ' XH = 0 IN MOMALO!'
     ENDIF
END DO  

XH(ND+1) = TC1(ND) - AKP(ND)*F(ND+2)  
!PRINT*,'test',XH(ND+1),TC1(ND),AKP(ND),F(ND+2)
!JO Jan 2016: might happen after update of hyd-structure
IF(XH(ND+1).LE.0.D0) THEN
  XH(ND+1)=XIC2*CORR/(3.D0*OPA(ND))
!  PRINT*,'CORR',XH(ND+1)
ENDIF

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE DIAG(A,B,C,DIA,N,IOPT)  
!
!   diagonal of inverse of tridiag matrix
!   (acc. to Rybicky & Hummer, 1991, A&A 245, 171)
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
INTEGER(I4B) ::  IOPT,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(N),B(N),C(N),DIA(N)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  L  
!     ..
!     .. local arrays ..
REAL(DP) ::  D(0:100),E(101)  
!     ..

IF (N.GT.100) STOP 'TOO MANY GRID POINTS IN DIAG'  

D(0) = 0.D0  

DO L = 1,N  
     D(L) = C(L)/ (B(L)-A(L)*D(L-1)) !trick: although a(1) undefined, d(0)=0.  
END DO  
E(N+1) = 0.D0  

DO L = N,1,-1  
     E(L) = A(L)/ (B(L)-C(L)*E(L+1)) !trick: although c(n) undefined, e(n+1)=0.
END DO  

IF (IOPT.EQ.0) THEN  
     DO L = 1,N  
          DIA(L) = 1.D0/ ((1.D0-D(L)*E(L+1))* (B(L)-A(L)*D(L-1)))  
     END DO

ELSE IF (IOPT.EQ.1) THEN  
     DO L = 1,N  
          DIA(L) = DIA(L)/ ((1.D0-D(L)*E(L+1))* (B(L)-A(L)*D(L-1)))
     END DO

ELSE  
     STOP 'WRONG OPTION IN DIAG'  

END IF  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE MOMCONT(ND,R,OPA,THOMSON,ST,XIC1,XIC2,CORR,Q,XH,TA,TB, &
&                   TC,TB1,TC1,AKP,ALO,XJ,F,OPTFLUX)

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!
!-----solves moments equation for continuum
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORR,XIC1,XIC2  
INTEGER(I4B) ::  ND  
LOGICAL OPTFLUX  
!     ..
!     .. array arguments ..
REAL(DP) ::  AKP(ND1),ALO(ND1),F(ND1+3),OPA(ND1),Q(ND1), &
&                 R(ND1),ST(ND1),TA(ND1),TB(ND1),TB1(ND1),TC(ND1), &
&                 TC1(ND1),THOMSON(ND1),XH(ND1+1),XJ(ND1)
!     ..
!     .. local scalars ..
REAL(DP) ::  DT0,DTM,DTP,EDMUE,FL,FLP,HI,HO,RL,RL2,RLP,RRQ  
INTEGER(I4B) ::  L  
!     ..
!     .. external subroutines ..
EXTERNAL INVTRI,MOMALO  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP  
!     ..

HO = F(ND+1)  
HI = F(ND+2)  
EDMUE = F(ND+3)  

Q(ND) = 1.D0  
RRQ = 1.D0  
FL = 3.D0 - 1.D0/F(ND)  

DO L = ND - 1,1,-1  
     RL = R(L)  
     RLP = R(L+1)  
     FLP = FL  
     FL = 3.D0 - 1.D0/F(L)  
     RRQ = RRQ*EXP(FL-FLP)* (RL/RLP)** ((FLP*RL-FL*RLP)/ (RL-RLP))  
     Q(L) = RRQ/RL/RL  
END DO  
!
!     feautrier scheme to solve moments equation :
!
!     (-ta,tb,-tc) * rr * xj = akp
!
!     outer boundary condition
!
DTP = 2.D0/ ((Q(1)*OPA(1)+Q(2)*OPA(2))* (R(1)-R(2)))  
TB(1) = (F(1)*Q(1)*DTP+HO-EDMUE*THOMSON(1))  
TB1(1) = TB(1) + EDMUE*THOMSON(1)  
TC(1) = F(2)*Q(2)*DTP  
TC1(1) = TC(1)  
AKP(1) = R(1)*R(1)*ST(1)*EDMUE  
ALO(1) = EDMUE  
!
!     non boundary points
!
DO L = 2,ND - 1  
     DTM = DTP  
     DTP = 2.D0/ ((Q(L)*OPA(L)+Q(L+1)*OPA(L+1))* (R(L)-R(L+1)))  
     DT0 = 2.D0/ (1.D0/DTP+1.D0/DTM)  
     TA(L) = F(L-1)*Q(L-1)*DT0*DTM  
     TC(L) = F(L+1)*Q(L+1)*DT0*DTP  
     TC1(L) = TC(L)  
     TB(L) = F(L)*Q(L)*DT0* (DTM+DTP) + (1.D0-THOMSON(L))/Q(L)  
     TB1(L) = TB(L) + THOMSON(L)/Q(L)  
     ALO(L) = 1.D0/Q(L)  
     AKP(L) = R(L)*R(L)*ST(L)/Q(L)  
END DO  
!
!     inner boundary condition
!
L = ND  
TA(L) = F(L-1)*Q(L-1)*DTP  
TB(L) = F(L)*Q(L)*DTP + HI  
TB1(L) = TB(L)  
ALO(L) = 0.D0  
AKP(L) = .5D0*XIC1 + XIC2*CORR/ (3.D0*OPA(L))  
!print*,'hplus',.5*xic1,xic2*corr/(3.d0*opa(l)),akp(l)
TC1(ND) = AKP(L)  

CALL INVTRI(TA,TB,TC,AKP,ND)  
! now, AKP is r^2 J

DO L = 1,ND  
     RL2 = R(L)*R(L)  
     XJ(L) = AKP(L)/RL2  
END DO  

IF (.NOT.OPTFLUX) RETURN  
!
!     calculation of alo and h
!
CALL MOMALO(ND,R,OPA,THOMSON,ST,TA,TB1,TC1,Q,F,AKP,ALO,XH,XIC2,CORR)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE SETUP1(LMAX,CORE,Z,ST,OPA,THOMSON,XIC1,XIC2,CORR,UP, &
&                  AKP,TA,TB,TC,TAUM)
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     sets up matrix elements for subroutine cont1
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORR,TAUM,XIC1,XIC2  
INTEGER(I4B) ::  LMAX  
LOGICAL CORE  
!     ..
!     .. array arguments ..
REAL(DP) ::  AKP(ND1),OPA(ND1),ST(ND1),TA(ND1),TB(ND1), &
&                 TC(ND1),THOMSON(ND1),UP(ND1),Z(ND1)
!     ..
!     .. local scalars ..
REAL(DP) ::  AK,AKDZ,BB,CC,DT0,DTM,DTP,DZ,E0,E1  
INTEGER(I4B) ::  L,LZ  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP  
!     ..

LZ = LMAX - 1  
!
!     outer boundary, 3rd order, corrected for i-minus
!
AK = .5D0* (OPA(1)+OPA(2))  
DZ = Z(1) - Z(2)  
AKDZ = .5D0*AK*DZ  
DTP = 1.D0/AK/DZ  
BB = 1.D0/DTP/3.D0  
CC = .5D0*BB  

IF (TAUM.LT.100.D0) THEN  
     E0 = EXP(-TAUM)  
     E1 = 1.D0 - E0  
ELSE  
     E1 = 1.D0  
END IF  

IF (AKDZ.LT..5D0) THEN  
     UP(1) = THOMSON(1)* (BB+E1) + THOMSON(2)*CC  
     AKP(1) = -ST(1)* (BB+E1) - ST(2)*CC  
     TB(1) = -1.D0 - DTP - BB  
     TC(1) = DTP - CC  
ELSE  
     UP(1) = THOMSON(1)*E1  
     AKP(1) = -ST(1)*E1  
     TB(1) = -1.D0 - DTP  
     TC(1) = DTP  
END IF  

IF (LZ.LT.2) GO TO 20  
!
!     non boundary points
!
DO L = 2,LZ  
     DTM = DTP  
     AK = .5D0* (OPA(L)+OPA(L+1))  
     DZ = Z(L) - Z(L+1)  
     DTP = 1.D0/AK/DZ  
     DT0 = 2.D0/ (1.D0/DTM+1.D0/DTP)  
     UP(L) = THOMSON(L)  
     AKP(L) = -ST(L)  
     TA(L) = DT0*DTM  
     TC(L) = DT0*DTP  
     TB(L) = -DT0* (DTM+DTP) - 1.D0  
END DO  

   20 CONTINUE  
L = LMAX  
!
!     inner boundary, 2nd order
!
IF (CORE) THEN  
     UP(L) = 0.D0  
     TA(L) = DTP  
     AKP(L) = -XIC1 - Z(L)*XIC2/OPA(L)*CORR  
     TB(L) = -DTP - 1.D0  
ELSE  
     AKP(L) = -ST(L)  
     UP(L) = THOMSON(L)  
     TA(L) = 2.D0*DTP*DTP  
     TB(L) = -2.D0*DTP*DTP - 1.D0  
END IF  

RETURN

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE WEIGH11(ND,NP,R,Z,P,W1,W3,Q1,Q2)  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: W0=>VP,W2=>VP2,W0LAST  


IMPLICIT NONE
!
!***  calculation of angular integration weights
!***  the weights w1  are calculated for intermesh points
!***  and interpolation weights q1,q2 for h,n
!
!***  changed from V10 on: everything calculated, &
!***  called after update of structure
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,NP1=ID_NPOIN  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND,NP  
!     ..
!     .. array arguments ..
REAL(DP) ::  P(NP1),Q1(ND1-2),Q2(ND1-2),R(ND1), &
&                 W1(ND1+1,NP1-1),W3(ND1+1,NP1-1),Z(ND1,NP1)
!     ..
!     .. local scalars ..
REAL(DP) ::  A,AA,B,BB,C,CC,RD1,RD2,RL,RL2,RRR12,RRRR20,RZ, &
&                 RZQ6,W1LZ,W3LZ,WW1,WW3,XLOG2
INTEGER(I4B) ::  JP,L,LMAX,LZ
!     ..
!     .. intrinsic functions ..
!INTRINSIC LOG10,MIN0  
!     ..

JPLOOP1: DO JP = 1,NP - 1  

          LMAX = MIN0(NP+1-JP,ND)  
          LZ = LMAX - 1  
!***
!***  0. and 2. moment. the integration is performed in the z variable
!***
LLOOP1:   DO L = 1,LMAX  
               RL = R(L)  
               RL2 = RL + RL  
               RRR12 = RL*RL*RL*12.D0  
!***
!***  first step if jp=1
!***
               IF (JP.EQ.1) THEN  
                    B = Z(L,1)  
                    A = Z(L,2)  
                    W0(L,JP) = (B-A)/RL2  
                    AA = A*A  
                    BB = B*B  
                    W2(L,JP) = (B* (3.D0*BB-AA)-A* (BB+AA))/RRR12  
               ELSE  
                    IF (L.NE.LMAX .OR. JP.LE. (NP-ND)) THEN  
!***
!***  intermediate step
!***
                         A = Z(L,JP+1)  
                         B = Z(L,JP)  
                         C = Z(L,JP-1)  
                         W0(L,JP) = (C-A)/RL2  
                         AA = A*A  
                         BB = B*B  
                         CC = C*C  
                         W2(L,JP) = (B* (CC-AA)+C* (CC+BB)-A* (BB+ &
                          AA))/ RRR12
                    ELSE  
!***
!***  last step, implying z(l,jmax)=0
!***
                         B = Z(L,JP-1)  
                         W0(L,JP) = B/RL2  
                         W2(L,JP) = B*B*B/RRR12  
                    END IF  
               END IF  

          END DO LLOOP1

END DO JPLOOP1
!***
!*** integration weight for p = rmax and z=0
!***
W0LAST = Z(1,NP-1)/ (R(1)+R(1))  

JPLOOP2: DO JP = 1,NP - 1  

     LMAX = MIN0(NP+1-JP,ND)  
     LZ = LMAX - 1  
!***
!*** 1.moment.
!*** first step
!***
     IF (JP.EQ.1) THEN  
          WW1 = P(2)*P(2)  
          WW3 = WW1*WW1  
     ELSE  
!***
!***  intermediate steps
!***
          A = P(JP-1)  
          B = P(JP)  
          C = P(JP+1)  
          WW1 = (A+B+C)* (C-A)  
          AA = A*A  
          BB = B*B  
          CC = C*C  
          WW3 = (B-A)* (AA* (A+2.D0*B)+BB* (3.D0*A+4.D0*B)) + &
           (C-B)* (CC* (C+2.D0*B)+BB* (3.D0*C+4.D0*B))
!***
!***  for the last interval (l=lz to the p axis), the next point is an
!***  intermesh-point in p
!***
          C = .5D0* (B+C)  
          W1LZ = (A+B+C)* (C-A)  
          CC = C*C  
          W3LZ = (B-A)* (AA* (A+2.D0*B)+BB* (3.D0*A+4.D0*B)) + &
           (C-B)* (CC* (C+2.D0*B)+BB* (3.D0*C+4.D0*B))
     END IF  
!***
!***  no weight for the z=0 point is calculated, as v=0 there for
!***  symmetry.
!***
!***  loop over depth index l
!***
LLOOP2: DO L = 1,LZ  
          RZ = .5D0* (R(L)+R(L+1))  
          RZQ6 = RZ*RZ*6.D0  
          RRRR20 = RZ*RZ*RZ*RZ*20.D0  
          IF (L.NE.LZ .OR. JP.LE. (NP-ND)) THEN  
               W1(L+1,JP) = WW1/RZQ6  
               W3(L+1,JP) = W1(L+1,JP) - WW3/RRRR20  
          ELSE  
               W1(L+1,JP) = W1LZ/RZQ6  
               W3(L+1,JP) = W1(L+1,JP) - W3LZ/RRRR20  

          END IF  
     END DO LLOOP2    
!***
!***  special weights at the outer boundary for h
!***
     RL = R(1)  
     W1(1,JP) = WW1/RL/RL/6.D0  
     W3(1,JP) = W1(1,JP) - WW3/RL/RL/RL/RL/20.D0  
!***
!***  special weights at the inner boundary
!***
     IF (LMAX.LT.ND) CYCLE  
     IF (JP.GT. (NP-ND)) THEN  
!***  core tangent ray, last step of integration
          WW1 = (B-A)* (2.D0*B+A)  
          WW3 = (B-A)* (AA* (A+2.D0*B)+BB* (3.D0*A+4.D0*B))  
     END IF  
     W1(ND+1,JP) = WW1/6.D0  
     W3(ND+1,JP) = W1(ND+1,JP) - WW3/20.D0  

END DO JPLOOP2

!***
!***  interpolation weights
!***

XLOG2 = LOG10(2.D0)  
! weights rechecked Feb. 2015, Ok. 
! Q2=(log r1'-log r)/(log r1' - log r2') = log(r1'/r)/log(r1'/r2') with
!    r  = r(l+1)
!    r1'=1/2(r(l)+r(l+1))
!    r2'=1/2(r(l+1)+r(l+2)
DO L = 1,ND - 2  
     RD1 = (R(L)+R(L+1))/R(L+1)  
     RD2 = (R(L)+R(L+1))/ (R(L+1)+R(L+2))  
     Q2(L) = (LOG10(RD1)-XLOG2)/LOG10(RD2)  
     Q1(L) = 1.D0 - Q2(L)  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE WEIGH5  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: W5

IMPLICIT NONE
!
!---- calculates integration weights for five points integration
!---- (simpson modified)
!
W5(1) = 14.D0/45.D0  
W5(2) = 64.D0/45.D0  
W5(3) = 24.D0/45.D0  
W5(4) = W5(2)  
W5(5) = W5(1)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE WEIGH9  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: W9

IMPLICIT NONE
!
!---- calculates integration weights for nine points integrations
!
W9(1) = 14.D0/45.D0  
W9(2) = 64.D0/45.D0  
W9(3) = 24.D0/45.D0  
W9(4) = W9(2)  
W9(5) = 28.D0/45.D0  
W9(6) = W9(2)  
W9(7) = W9(3)  
W9(8) = W9(2)  
W9(9) = W9(1)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CONVERSION  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: FRE,WFRE,HC,IFRE  

IMPLICIT NONE
!
!---- it converts binary files stored by frecuency into binary files
!---- stores by depth
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
INTEGER(I4B), PARAMETER :: ND=ID_NDEPT  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  IFR,IND,J  
!     ..
!     .. local arrays ..
REAL(DP) ::  A(IFRETOT,ND)  
!     ..
!     ..
DO IFR = 1,IFRE  
     READ (50,REC=IFR) (A(IFR,J),J=1,ND)  
END DO  

DO IND = 1,ND  
    WRITE (40,REC=IND) (A(J,IND),J=1,IFRE)  
END DO  


DO IFR = 1,IFRE  
     READ (52,REC=IFR) (A(IFR,J),J=1,ND)  
END DO  

DO IND = 1,ND  
     WRITE (42,REC=IND) (A(J,IND),J=1,IFRE)  
END DO  

DO IFR = 1,IFRE  
     READ (51,REC=IFR) (A(IFR,J),J=1,ND)  
END DO  

DO IND = 1,ND  
     WRITE (41,REC=IND) (A(J,IND),J=1,IFRE)  
END DO  


DO IFR = 1,IFRE  
     READ (53,REC=IFR) (A(IFR,J),J=1,ND)  
END DO  

DO IND = 1,ND  
     WRITE (43,REC=IND) (A(J,IND),J=1,IFRE)  
END DO  

RETURN  
END
!
!-----------------------------------------------------------------------
!
FUNCTION DBDT(X,T)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     .. parameters ..
REAL(DP), PARAMETER :: C1=1.4388354967334D8  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  T,X,DBDT  
!     ..
!     .. external functions ..
REAL(DP) ::  BNUE  
EXTERNAL BNUE  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP  
!     ..

DBDT = BNUE(X,T)*C1/X/T/T/ (1.D0-EXP(-C1/X/T))  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION BNUE(XLAMBDA,T)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     planck function,lambda in angstroem,t in kelvin
!     bnue in erg per (cm**2 * sec * hertz )
!
!     constanten: c1=h*c/k,c2=2*h*c
!
!
!     .. parameters ..
REAL(DP), PARAMETER :: C1=1.4388354967334D8,C2=3.972970127D8
!     ..
!     .. scalar arguments ..
REAL(DP) ::  T,XLAMBDA,BNUE  
!     ..

REAL(DP) :: X, EXPO
!     .. intrinsic functions ..
!INTRINSIC EXP  
!     ..
! MODIFIED TO ACCOUNT FOR LARGE EXPONENTS (X-RAY REGIME)

X=C1/XLAMBDA/T

IF(X.LT.200.D0) THEN 
  BNUE = C2/ (EXP(X)-1.D0)/XLAMBDA/XLAMBDA/XLAMBDA
ELSE
  EXPO=LOG(C2/XLAMBDA**3)-X
  BNUE=EXP(EXPO)
ENDIF           

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE DIFFUS(XLAMBDA,T,R,ND,AIC,DBDR)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     gives the planck function aic and its radius derivative dbdr
!     at the inner boundary from the given temperature-stratification.
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  AIC,DBDR,XLAMBDA  
INTEGER(I4B) ::  ND  
!     ..
!     .. array arguments ..
REAL(DP) ::  R(ND1),T(ND1)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DTDR  
!     ..
!     .. external functions ..
REAL(DP) ::  BNUE,DBDT  
EXTERNAL BNUE,DBDT  
!     ..

AIC = BNUE(XLAMBDA,T(ND))  
DTDR = (T(ND)-T(ND-1))/ (R(ND-1)-R(ND))  
DBDR = DBDT(XLAMBDA,T(ND))*DTDR  

RETURN  

END
!
!***********************************************************************
!
! subroutines: complex ones
! lte and related
!
!***********************************************************************
!
SUBROUTINE LTE(TEFF,SR,TAUR,TAUE,R,VELO,DVDR,RHO,XNE,XNH,TEMP,DILFAC,CLF, &
&               ILOW,IMAX,ND,GREY,CORRFC,NTEMP,CORVM,OPTCV,ITHYD, &
&               XKAP,EX,XMMAX,RTAU23,XMET,RESTART_MET_LTE,LTE_UPDATE)
! clf is clumping factor (overdensity in clumps)

USE nlte_type  
USE nlte_dim  
USE fund_const
USE nlte_var, ONLY: FRE,WFRE,HC,IFRE, &
& OPAC,ST=>STRUE,XJ,ALO, &
& IDUM,SIGEM,DELSIG,DELMU,DQDTAU, &
& XKAPITH,EXITH,CORRITH, &
& MODNAM, &
& ICONVER,OPTMET,YHEIN,LINES_IN_MODEL,ENIONND,DELPARA, & 
& OP_FLAG, OP_FLAG2, FRECFN1 	   
USE princesa_var, ONLY: NAT,ABUND,FRECFN

USE nlte_porvor, ONLY: FIC,TCL_FAC_LINE,TCL_FAC_CONT
                           
                           
IMPLICIT NONE  
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORRFC,CORVM,EX,RTAU23,SR,TEFF,XKAP,XMMAX  
INTEGER(I4B) ::  ITHYD,ND,NTEMP  
LOGICAL GREY,OPTCV,RESTART_MET_LTE,LTE_UPDATE  
!     ..
!     .. array arguments ..
REAL(DP) ::  DILFAC(ND1),DVDR(ND1),R(ND1),RHO(ND1),TAUE(ND1),TAUR(ND1), &
&                 TEMP(ND1),VELO(ND1),XNE(ND1),XNH(ND1),CLF(ND1)
                           
INTEGER(I4B) ::  ILOW(ND1,KEL),IMAX(ND1,KEL)  
!     ..
!     ..
!     .. local scalars ..
REAL(DP) ::  A,BT,CORR,DEL,DELTMAX,EPSMAX,FLUX,FLUXROSS,OP, &
&                 OPTH,RS,SIG,SRCONST,SUMM,TMIN,XLAM,XMET,X,X1,MAXI,TTAUR,TCL
INTEGER(I4B) ::  I,III,IIT,K,L,LL,LROSS,N,NT1,IFREOLD,K1  
LOGICAL CONV,OPTETA,OPTOUT1,OPTTEMP,WRITEOC,NEWFREQ
!     ..
!     .. local arrays ..
REAL(DP) ::  ARHO(ND1),BRHO(ND1),CRHO(ND1), &
&                 OPAROLD(ND1),OPAROSS(ND1),SUMOPAR(ND1), &
&                 TEMP1(ND1),XNESAV(ND1),DUMMY(ND1)
DATA IFREOLD/0/

!     ..
!     .. external functions ..
REAL(DP) ::  DBDT  
EXTERNAL DBDT  
!     ..
!     .. external subroutines ..
EXTERNAL DEV,FRESCAL,INTEGRA,LINREG,NISTAR,OCCLTE,OPACITC, &
&         PRMODEL,SPLINE,TLUCY
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,LOG10,MAX,SQRT  
!     ..

OPTETA = .FALSE.  
WRITEOC = .FALSE.  
OPTOUT1 = .FALSE.  

DO L = 1,ND  
     RS = RTAU23/R(L)  
     IF (RS.GT.1.D0) THEN  
          DILFAC(L) = .5D0  
     ELSE  
          DILFAC(L) = .5D0* (1.D0-SQRT(1.D0-RS**2))  
     END IF  
END DO

IF (GREY) THEN  
     CONV = .FALSE.  
     IIT = 20  
ELSE  
     CONV = .TRUE.  
     IIT = 1  
END IF  

DO L = 2,ND  
!     IF (TEMP(L).GT.TEMP(L-1)) EXIT  
!CHANGED
  IF (TEMP(L).NE.TEMP(L-1)) EXIT
END DO  

NTEMP = L - 1  

IF (NTEMP.EQ.1) THEN  
     OPTTEMP = .TRUE.  
     TMIN = 0.D0  
ELSE  
     OPTTEMP = .FALSE.  
     TMIN = TEMP(NTEMP)  
END IF  

PRINT *,' TMIN = ',TMIN  
PRINT *  
!
!     calculation of grey temperature
!
IF (GREY) THEN  
     PRINT *  
     PRINT *,' TEMPERATURE CORRECTION '  
     PRINT *  
END IF  

NTEMP = 1  
!
!===================================================================
!
ITTEMP: DO III = 1,IIT  

     IF (CONV) THEN  
          OPTETA = .TRUE.  
          WRITEOC = .FALSE.  
          OPTOUT1 = .TRUE.  
     END IF  

     IF (.NOT.OPTTEMP .AND. III.GE.2) THEN  
          DO L = 1,ND  
               IF (TEMP(L).GT.TMIN) THEN  
                    NTEMP = L - 1  
                    GO TO 50  
               END IF  
          END DO
     END IF  

   50      CONTINUE  

     IF (CONV) NTEMP = 1  

     CALL OCCLTE(RHO,XNE,XNH,CLF,TEMP,ILOW,IMAX,ND,NTEMP,LTE_UPDATE)  
! all occup. numbers include clf
! NOT changed in case of OPTTHICK, since occupation numbers remain 

     CALL NISTAR  

     IF(IFREOLD .NE.0) IFREOLD =IFRE
     CALL FRESCAL(ND,XNH,TEMP(ND),OPTOUT1,TEFF,NEWFREQ)  

!     IF(IFRE.NE.IFREOLD) THEN
! JO Sept. 2021: New condition
     IF(NEWFREQ) THEN
! IF FREQ. GRID HAS CHANGED, NTEMP = 1 REQUIRED TO UPDATE ALL DEPTH POINTS
       NTEMP=1
       TMIN=TEMP(1)
       IFREOLD=IFRE
! A change in the frequency grid requires to read again the OP rbf
! data file, to map the cross sections in the new frequency grid; miguel
! Sept. 2016: in this case, reset FRECFN1
       OP_FLAG(:) = .FALSE.	
       OP_FLAG2(:) = .FALSE.
       FRECFN1 = FRECFN
       PRINT*,' OP_FLAG AND FRECFN1 RESET'
     ENDIF

     IF (OPTMET) THEN
       PRINT*,' LTE FOR METALS'
!
! note: dvdr only dummy argument here
!       
       CALL NLTE_APPROX(TEFF,YHEIN,XMET,RHO,XNE,DILFAC,TEMP,DUMMY,DUMMY,DUMMY, &
&        CLF,ND,.TRUE.,.TRUE.,NTEMP,OPTOUT1)
       
       CALL OPACITM(ND,XNE,TEMP,CLF,.TRUE.,NTEMP,OPTETA, .TRUE.,.FALSE.)
       
     ENDIF

     CALL OPACITC(ND,XNE,XNH,TEMP,CLF,ILOW,IMAX,.TRUE.,NTEMP,OPTETA,.TRUE., &
                  OPTMET)
!    remember, that all opacities have been corrected for clumping
!    to allow for correctly taken spatial averages

!    if restart and very first LTE run, 2nd call in order
!    to provide correct OPAC_NOLINES
     
     IF(RESTART_MET_LTE.AND.LINES_IN_MODEL) THEN
        PRINT*, &
& ' RESTART AND LTE: 2ND CALL OF OPACITL TO OBTAIN CONSISTENT LTE-OPACITIES'
        CALL OPACITL(XNE,.TRUE.,NTEMP,DUMMY,DUMMY,DUMMY,CLF,.TRUE.)

        CALL OPACITM(ND,XNE,TEMP,CLF,.TRUE.,NTEMP,OPTETA, .TRUE.,.FALSE.)

        CALL OPACITC(ND,XNE,XNH,TEMP,CLF,ILOW,IMAX,.TRUE.,NTEMP,OPTETA,.TRUE., &
                     OPTMET) 
     ENDIF


     DO K = 1,IFRE  
          XLAM = 1.D8/FRE(K)  
          DO L = NTEMP,ND  
               OPAROSS(L) = DBDT(XLAM,TEMP(L))/OPAC(L,K)
!---- OPAC is effective, thus effective rosseland mean as well!
          END DO
          CALL INTEGRA(OPAROSS,SUMOPAR,ND,IFRE,WFRE(K),K)  
     END DO
!
!      print*,'renormalization for opaross '
!
     FLUXROSS = OPAROSS(ND)* (TEMP(ND)-TEMP(ND-1))/ (R(ND-1)-1.D0)/ SR/3.D0
!
!     only used if ntemp>1, i.e. if iii.gt.1
!
     IF (III.EQ.1 .AND. NTEMP.NE.1) STOP ' ERROR IN CHOSEN PHILOSOPHY -- OPAROLD'

     DO L = 1,NTEMP - 1  
          OPAROSS(L) = OPAROLD(L)  
     END DO

     DO L = NTEMP,ND  
          BT = 7.21863D-5*TEMP(L)**3  
          OPAROSS(L) = BT/OPAROSS(L)/R(L)/R(L)  
     END DO

     OPAROLD = OPAROSS  

! note that opaross is corrected for clumping
!
!---- calculation of tau-lucy and tau-ross
!---- arho: tau-lucy
!---- brho: tau-ross
!---- crho:(effective or mean) opa-ross (optthick = .true./false.) 
!
!-----x with respect to srnom, but r with respect to sr
!     hence: constant in front of integral srnom*rtau23=sr*rtau23^2
!
     SRCONST = SR*RTAU23**2  

     DO L = 1,ND  
          CRHO(L) = OPAROSS(L)*SRCONST    
     END DO

     SUMM = 0.D0  
     ARHO(1) = 0.D0  
     DO L = 1,ND - 1  
          DEL = R(L+1) - R(L)  
          SUMM = SUMM - DEL*5.D-1* (CRHO(L)+CRHO(L+1))  
          ARHO(L+1) = SUMM  
     END DO

     IF (CONV) GO TO 160  

     DO L = 1,ND  
       CRHO(L) = OPAROSS(L)*R(L)*R(L)*SR
     END DO
        
     SUMM = 0.D0  
     BRHO(1) = 0.D0  

     DO L = 1,ND - 1  
          DEL = R(L+1) - R(L)  
          SUMM = SUMM - DEL*5.D-1* (CRHO(L)+CRHO(L+1))  
          BRHO(L+1) = SUMM  
     END DO

     CALL TLUCY(R,ARHO,BRHO,TEMP1,ND,TMIN,TEFF,RTAU23,ITHYD)  
!     print*,'from hydro,tlucy',temp1 
     CALL DEV(R,TEMP1,TEMP,ND,EPSMAX,.FALSE.)  
     PRINT *  

     TEMP = TEMP1

     IF (EPSMAX.LT.0.01D0) THEN  
          CONV = .TRUE.  
!
!.....storage of electron density
!     (from next to last hydro-iteration, to ensure precise restart)
!
          XNESAV = XNE
     END IF  

END DO ITTEMP 
!
!===================================================================
!
!-----calculation of rosseland-optical depth
!
  160 CONTINUE  

SIG = SIGMAE*AMH  

DO L = 1,ND  
     SUMOPAR(L) = OPAROSS(L)*R(L)*R(L)*SR 
END DO  

SUMM = 0.D0  
TAUR(1) = 0.D0  
DO L = 1,ND - 1  
     DEL = R(L+1) - R(L)  
     SUMM = SUMM - DEL*5.D-1* (SUMOPAR(L)+SUMOPAR(L+1))  
     TAUR(L+1) = SUMM  
END DO  
!
!---- if needed, vmin is corrected to obtain a large enough value of
!---- tau-ross at the inner point (e. santolaya, 11/2/94)
!
IF (TAUR(ND).LT.50.D0) THEN  
     OPTCV = .TRUE.  
     CORVM = TAUR(ND)/60.D0  
ELSE  
     OPTCV = .FALSE.  
END IF  

OPTCV = OPTCV .AND. GREY  

IF (OPTCV .AND. ITHYD.EQ.1) RETURN  

IF (TAUR(ND).GT.80.D0) XMMAX = XMMAX*70.D0/TAUR(ND)  

IF (TAUR(ND).LT.60.D0) XMMAX = XMMAX*70.D0/TAUR(ND)  

DO L = 1,ND - 1  
     IF (TAUR(L).LT..66D0 .AND. TAUR(L+1).GE..66D0) GO TO 200  
END DO  

PRINT *,' WARNING!!!! TAUROSS = 2/3 NOT FOUND'  
PRINT *  

LROSS = 0  
GO TO 210  

  200 CONTINUE  

LROSS = L + 1  

  210 CONTINUE  

PRINT *  
PRINT *,' TAU-ROSS AT RSTAR = ',TAUR(ND)  
PRINT *  

IF (LROSS.NE.0) THEN  
     PRINT *,' TAU-ROSS = ',TAUR(LROSS),' AT R = ',R(LROSS)/ RTAU23, ' NOM. RADII'
     TTAUR=TEMP(LROSS)**4+(TEMP(LROSS)**4-TEMP(LROSS-1)**4)/ &
&      (TAUR(LROSS)-TAUR(LROSS-1))*(0.66-TAUR(LROSS))
     TTAUR=TTAUR**0.25
     PRINT *,' T(TAUROSS=2/3) = ',TTAUR  
     PRINT *  
END IF  

PRINT *  
PRINT *,' FOLLOWING TABLE WITH EFFECTIVE OPACITIES'
PRINT *
PRINT *,' NO    OPAROSS     OPAR/OPATH    OPTH/RHO      TAUROSS', &
&  '       TAUP         TEMP'
PRINT *  
DO L = 1,ND  
     OP = OPAROSS(L)*R(L)*R(L)  
     OPTH = SIG*XNE(L)/CLF(L) !since op has been corrected 
     TCL = OPTH * TCL_FAC_CONT(L) 
     OPTH = OPTH * (1.+FIC(L)*TCL)/(1.+TCL) 
!---- printing effective thomson opacity, as if the atmosphere
!     would have only this as opacity source. 
     WRITE (*,FMT=9000) L,LOG10(OP),OP/OPTH,OPTH/RHO(L),TAUR(L), &
      ARHO(L),TEMP(L)/TEFF
END DO  
!
!-----this is the Eddington flux with respect to the nom. radius rstar(teff)
!
FLUX = 4.511693D-6*TEFF**4  !sigma_B/(4pi)
!
!-----it has to be corrected with respect to the lower radius according to
!     rstar^2*flux(teff) = rmin^2*flux(rmin)
!     -> flux(rmin)=(rstar/rmin)^2 flux(teff)=RTAU23^2 flux(teff)
!
FLUX = FLUX*RTAU23**2  

CORRFC = FLUX/FLUXROSS

PRINT *  
PRINT *,' CORRFC FOR LOWER BOUNDARY  = ',CORRFC  
PRINT *  
IF (CORRFC.LT..8D0 .OR. CORRFC.GT.1.2D0) THEN  
     PRINT *,' WARNING!!! CORRFC EXTREMELY LARGE'  
     PRINT *  
END IF  

!
!     ne(lte) written to tau_ros for calculation of dep. coeffiecients in totout.f90
!
OPEN (1,FILE=TRIM(MODNAM)//'/TAU_ROS',STATUS='UNKNOWN',FORM='FORMATTED')  
REWIND 1
DO L = 1,ND  
     WRITE (1,FMT=*) TAUR(L),XNE(L)  !xne includes clf
END DO  
CLOSE (1)  
!print*,'xnecheck: tau_ros written'
!print*,taur
!print*,xne
!
!.....only for models which start from given hydro-structure
!     (.not.grey corresponds to optmod)
!
IF (.NOT.GREY) GO TO 320  

DO L = 1,ND  
     SUMOPAR(L) = SIG*XNE(L)*SR / CLF(L)  !corrected
!---- corrected here as above, assuming the atmosphere has only e- scattering 
     TCL = SUMOPAR(L) * TCL_FAC_CONT(L)/SR !SR already included  
     SUMOPAR(L) = SUMOPAR(L) * (1.+FIC(L)*TCL)/(1.+TCL) 
!---- effective opacity 
     DELSIG(L) = SUMOPAR(L)/ (SIGEM*RHO(L)*SR) 
END DO  
!
!     regression coefficients for kramer opacities used for photospheric
!     correction in wind model
!
DO L = 1,ND  
     IF (TAUR(L).GT..1D0) GO TO 260  
END DO  
STOP ' NT1 NOT FOUND'  

  260 CONTINUE  

! FROM HERE ON, WE LEAVE EVERYTHING AS IN THE UNCLUMPED CASE, &
! SINCE IT APPLIES ONLY TO THE PHOTOSPHERE  

NT1 = L  
IF (ITHYD.GE.15) NT1 = NT1 + 1  

! all quantities now refer to average density

! correction for delpara (if .ne. 1)
DO L = NT1,ND  
     LL = L - NT1 + 1  
     ARHO(LL) = LOG10((OPAROSS(L)*R(L)**2*SR/SUMOPAR(L)-1.D0)/ &
&                     (RHO(L)*DELPARA(L)))
     BRHO(LL) = LOG10(TEMP(L))  
END DO  

N = ND - NT1 + 1  

CALL LINREG(BRHO,ARHO,N,EX,A,CORR)  
XKAP = 10.D0**A  
EX = -EX  
XKAPITH(ITHYD) = XKAP  
EXITH(ITHYD) = EX  
CORRITH(ITHYD) = CORR
PRINT *  
PRINT *,'   LINEAR REGRESSION OF FLUX-MEAN OF TYPE :'  
PRINT *  
PRINT *,'   -- CHI(R) = OPA-TH(R)*(1.+ XKAP*DELPARA(L)*RHO(L)/T(L)**EX) -- '  
PRINT *  
PRINT *,'   FROM R = 1. UP TO  R = ',R(NT1)  
PRINT *  
PRINT *,'   YIELDS XKAP = ',XKAP,'   EX = ',EX  
PRINT *,'   CORR. COEFF = ',CORR  
PRINT *  


IF (ABS(CORR).LT..8D0) STOP ' CORR IN KRAMER OPACITIES TOO LARGE'  

! ACCOUNTING FOR DIFFERENCES BETWEEN PARAMETERIZED AND ACTUAL OPACITY
IF (TEFF.LT.12000..AND.ITHYD.GE.5.AND.ITHYD.LT.15) THEN
! if H-recom is oscillating, freeze DELPARA
 DO L=1,ND
  ARHO(L)=OPAROSS(L)*R(L)**2*SR/SUMOPAR(L)-1.D0
  BRHO(L)=XKAP*RHO(L)*TEMP(L)**(-EX)
  DELPARA(L)=ARHO(L)/BRHO(L)
  IF(DELPARA(L).LT.0.D0) DELPARA(L)=0.D0
 ENDDO
ENDIF

IF (ICONVER.NE.1) GO TO 340  

PRINT *,' ITERATION HISTORY OF KRAMER OPACITY REGR. COEFF.'  
PRINT *  

DO L = 1,ITHYD  
     WRITE(*,275) L,XKAPITH(L),EXITH(L),CORRITH(L)  
     PRINT*
END DO  
PRINT *  

WRITE(999,*) ' ITERATION HISTORY OF KRAMER OPACITY REGR. COEFF.'  
WRITE(999,*)

DO L = 1,ITHYD  
     WRITE(999,*)
     WRITE(999,275) L,XKAPITH(L),EXITH(L),CORRITH(L)  
END DO  
275 FORMAT(' IT.NO. ',I2,' XKAP = ',E12.6,' EX = ',F10.5,' CORR = ',F10.5)

WRITE(999,*)

REWIND 21  
READ (21,*) (BRHO(I),I=ND,1,-1), (TEMP1(I),I=ND,1,-1)  
DELTMAX = 0.D0  
DO L = 1,ND  
     DELTMAX = MAX(DELTMAX,ABS(1.D0-TEMP1(L)/TEMP(L)))  
END DO  
!print*,'from hydro,readfrom temp',temp1

REWIND 21  
WRITE (21,*) (BRHO(I),I=ND,1,-1), (TEMP(I),I=ND,1,-1)  
WRITE (21,*) (XNESAV(I),I=ND,1,-1)  
WRITE (21,*) RTAU23  
!print*,'from hydro,writeto temp',temp

DO L = 1,ND  
     ARHO(L) = SUMOPAR(L)/SR* (1.D0+XKAP*DELPARA(L)*RHO(L)*TEMP(L)** (-EX))  
END DO  

PRINT *  
PRINT *,'   CHECK VALUES , DELTA-MAX(T_HYDRO,T_ACT) = ',DELTMAX  
PRINT *
PRINT *,'   ALL OPACITIES CORRESPOND TO AVERAGE DENSITY' 
PRINT *
PRINT *,' NO   OPA(CALC)    OPA(APPROX)   T(HYDRO)       T(ACT)      DELPARA'  

DO L = 1,ND  
     WRITE (*,FMT=9010) L,LOG10(OPAROSS(L)*R(L)**2), LOG10(ARHO(L)), &
      TEMP1(L),TEMP(L),DELPARA(L)
END DO  

WRITE(999,*)  
WRITE(999,*) '   CHECK VALUES , DELTA-MAX(T_HYDRO,T_ACT) = ',DELTMAX  
WRITE(999,*)
WRITE(999,*) '   ALL OPACITIES CORRESPOND TO AVERAGE DENSITY' 
WRITE(999,*)
WRITE(999,*) ' NO   OPA(CALC)    OPA(APPROX)   T(HYDRO)   T(ACT)   DELPARA'  

DO L = 1,ND  
     WRITE (999,FMT=9010) L,LOG10(OPAROSS(L)*R(L)**2), LOG10(ARHO(L)), &
      TEMP1(L),TEMP(L),DELPARA(L)
END DO  

  320 CONTINUE  

!JO changed Jan. 2016
!recalc taue in micro_clumping approx.
!(above, calculated including optically thick clumping)
SUMOPAR = SIG * XNE * SR / CLF  !corrected
CALL SPLINE(R,SUMOPAR,ARHO,BRHO,CRHO,ND)  
SUMM = 0.D0  
TAUE(1) = 0.D0  

DO L = 1,ND - 1  
     DEL = R(L+1) - R(L)  
     SUMM = SUMM - DEL* (SUMOPAR(L)+DEL* (ARHO(L)/2.D0+DEL* (BRHO(L)/ &
&      3.D0+DEL*CRHO(L)/4.D0)))
     TAUE(L+1) = SUMM  
END DO

PRINT *  
PRINT *,' -------------- ACTUAL WIND MODEL--------------------------'
PRINT *  

!REMEMBER: TAUE=TAU-TH ONLY IN MICRO-CLUMPING APPROX.
CALL PRMODEL(R,VELO,DVDR,RHO,XNE,XNH,TEMP,TAUE,CLF,TEFF,ND)  

PRINT *  
PRINT *,' -------------------------------------------------------------'
PRINT *  

  340 CONTINUE  

DO L = 2,ND  
     IF (TEMP(L).GT.TEMP(L-1)) GO TO 360  
END DO  

  360 CONTINUE  

NTEMP = L - 1  

RETURN  

 9000 FORMAT (I3,7 (3X,F10.5))  
 9010 FORMAT (I3,3X,2 (F10.5,3X),2 (F10.0,3X),G12.6)  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FRESCAL(ND,XNH,TMAX,OPTOUT,TEFF,NEWFREQ)  
!
!-----  this subroutine calculates the frequency grid, nh corrected for clumping
!
!       present philosophy requires that EACH level of a DETAILed ion
!       is represented with two corresponding frequencies, namely at
!       the corresponding edge and at edge-epslon(edge)  
!   
!       levmax : maximum number of resolved edges for dominant ions of
!                the two most abundant atoms
!       levmin1: maximum number of resolved edges for 2nd dominant ions
!                of the two most abundant atoms
!       levmin2: maximum number of resolved edges for all other ions
!
!       NOTE: we always count from the highest transition frequency.
!             Two resolved edges means using the 2 highest frequencies.
!             This does not necessarily correspond to the two lowest levels,
!             since transitions to excited levels might lie at higher freqs.
!             Example: SiIII (usually 2nd ion). Highest transitions are
!             Si301->401 and Si304->402
!
!       resolved means here, that we use a sufficient number of freq.
!       points above the corresponding edges.  
!
USE nlte_type  
USE nlte_dim
USE fund_const

USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&       ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
& LABL4,LABL1,LABU1,LABL3,IPARE5, &
& TL,ANCDO,WLC,DL,FRECIN,FRECFN, &
& LABL2,PAREN2,PAREN3,PAREN4,LABU2  

USE nlte_var, ONLY: OCCNUM,MAINION, &
& FRE,WFRE,HC,IFRE, &
& BLEVEL,ENIONND, &
& IMIA,IMAA, &
& ICONVER, &
& BN,INDEX,LAM_LY,NSTARK, WAVESTARK


USE nlte_xrays, ONLY: OPTXRAY, N_KEDGES, K_NMIN, K_NMAX, Z, NIONK=>N, ETH, NAME

IMPLICIT NONE  
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT
INTEGER(I4B), PARAMETER :: NFMIN=300
INTEGER(I4B), PARAMETER :: LEVMAX=ID_LEVMA,LEVMIN1=ID_LEVM1,LEVMIN2=ID_LEVM2
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  

!JO Dec. 2015
! FREQUENCIES FOR Lya and Lyb which should be used for consistency
! precision 1.d-9
!JO July 2021
! extended for Lyg and Lyd to avoid problems for B-stars
REAL(DP), PARAMETER :: LYA_LO=1.D8/1217.00043262610D0, LYA_HI=1.D8/1212.85710645286D0, &
                       LYB_LO=1.D8/1027.46136261866D0, LYB_HI=1.D8/1024.50655646359D0, &
                       LYG_LO=1.D8/973.196165978500D0, LYG_HI=1.D8/972.310779289066D0, &
                       LYD_LO=1.D8/949.843120035809D0, LYD_HI=1.D8/948.999697008538D0, &
                       LAM_CIII=977.0200D0 ! CIII RESONANCE LINE (1-3)
!     ..
!     .. scalar arguments ..
REAL(DP) ::  TMAX, TEFF
INTEGER(I4B) ::  ND  
LOGICAL OPTOUT, NEWFREQ  
!     ..
!     .. array arguments ..
REAL(DP) ::  XNH(ND1)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AAA,DM,DMIN,DMIN1,DMIN2,DNEW,DNEW1,EDGE,EMAX,EPS, &
&            ERRP,FRENEW,PF,PFE,SUM1,SUM2,X,X1,XM1,XM2,XNH1,FREMIN, &
&            XMIN,XMAX,DFL,KAYSER
INTEGER(I4B) ::  I,IF1,IFORM,IGENIO,IHI,II,IL,ILAB,IMP, &
&        INEW,INTEG,IO,ISI,IZ,J,JI,K,K1,K2,L,LEVE,LEVE1,LEVE2,LEVP, &
&        LEVP1,LEVPORG,LEVTOT,LEVUSED,LL,LLI,M,M1,M2,MM1,MM2,N,NC, &
&        NDAT,NDATOS,NIMP,NOPA,NPER,NK,ISTART,IEND,NSKIP,IFREOLD
CHARACTER LAB*6,PARE*6  
!     ..
!     .. local arrays ..
REAL(DP) ::  DATA(ID_NDATA),FLAUX(ID_RBFTR),FREEDGE(ID_RBFTR),FREOLD(IFRETOT)  
INTEGER(I4B) ::  INDEDGE(ID_RBFTR),IOPA(KIS*KEL)
!     ..
!     .. external functions ..
REAL(DP) ::  BNUE,XINTFRE  
EXTERNAL BNUE,XINTFRE  
!     ..
!     .. external subroutines ..
EXTERNAL SORT  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,DBLE,INT,LOG,LOG10,MAX,MIN  
!     ..
!     .. statement functions ..
REAL(DP) ::  EPSLON  
!     ..
!     .. statement function definitions ..
!
EPSLON(EDGE) = 5.D0*10.D0** (DBLE(INT(LOG10(EDGE)))-6.D0)  
!     ..
!optout=.true.
!iconver=1

IF (ID_RBFTR.GT.999) STOP ' TOO MANY RBF-TRANSITIONS, CODING OF INDEX AFFECTED'  

!JO Sept. 2021
IFREOLD=IFRE
IF (IFREOLD.NE.0) THEN
  DO I=1,IFRE
    FREOLD(I)=FRE(I)
  ENDDO
ENDIF  
!
!       most abundant element k1
!
K1 = 1  
X = ABUND(1)  
DO K = 2,NAT  
     X1 = MAX(X,ABUND(K))  
     IF (X1.EQ.X) CYCLE  
     X = X1  
     K1 = K  
END DO  
!
!       2nd abundant element if present k2
!
IF (NAT.GT.1) THEN  
     K2 = 1  
     IF (K1.EQ.1) K2 = 2  
     X = ABUND(K2)  
     DO K = 1,NAT  
          IF (K.EQ.K1) CYCLE  
          X1 = MAX(X,ABUND(K))  
          IF (X1.EQ.X) CYCLE  
          X = X1  
          K2 = K  
     END DO
END IF  
!
!       dominant ionisation stages of k1, k2 (if present)
!
SUM1 = 0.D0  
SUM2 = 0.D0  

DO L = 1,ND  
     SUM1 = SUM1 + MAINION(L,K1)  
     IF (NAT.GT.1) SUM2 = SUM2 + MAINION(L,K2)
END DO  

XM1 = SUM1/ND  
XM2 = SUM2/ND  
M1 = INT(XM1)  
M2 = INT(XM2)  
IF (XM1-M1.GT..5D0) M1 = M1 + 1  
IF (XM2-M2.GT..5D0) M2 = M2 + 1  

IOPA(1) = K1*100 + M1  

IF (NAT.GT.1) THEN  
     IOPA(2) = K2*100 + M2  
ELSE  
     K2 = 0  
     IF (M2.NE.0) STOP ' ERROR IN M2'  
     IOPA(2) = 0  
END IF  

IF (IMAA(K1).GT.1) THEN  
!
!       2nd important ionization stage of k1 (if present)
!
     IF (IMAA(K1).EQ.2) THEN  
!
!       e.g. helium
!
          MM1 = 3 - M1  
          IOPA(3) = K1*100 + MM1  
     ELSE  
!
          IF (M1.EQ.IMAA(K1)) THEN  
               MM1 = M1 - 1  
               GO TO 50  
          END IF  
!
!       above or below m1?
!
          SUM1 = 0.D0  
          SUM2 = 0.D0  
          DO L = 1,ND  
               XNH1 = 1.D0/XNH(L)  
               SUM1 = SUM1 + ENIONND(K1,M1+1,L)*XNH1  
               SUM2 = SUM2 + ENIONND(K1,M1-1,L)*XNH1  
          END DO
          SUM1 = SUM1/DBLE(ND)  
          SUM2 = SUM2/DBLE(ND)  
          IF (SUM1.GT.SUM2) THEN  
               MM1 = M1 + 1  
          ELSE  
               MM1 = M1 - 1  
          END IF  

   50           CONTINUE  
          IOPA(3) = K1*100 + MM1  
     END IF  
     NOPA = 3  
ELSE  
!
!       most abundant elememt has only one ionization stage
!
     MM1 = M1  
     NOPA = 2  
END IF  

IF (NAT.GT.1) THEN  
!
!       2nd important ionization stage of k2, as for k1
!
     IF (IMAA(K2).GT.1) THEN  
          IF (IMAA(K2).EQ.2) THEN  
               MM2 = 3 - M2  
               IOPA(NOPA+1) = K2*100 + MM2  
          ELSE  
               IF (M2.EQ.IMAA(K2)) THEN  
                    MM2 = M2 - 1  
                    GO TO 70  
               END IF  
               SUM1 = 0.D0  
               SUM2 = 0.D0  
               DO L = 1,ND  
                    XNH1 = 1.D0/XNH(L)  
                    SUM1 = SUM1 + ENIONND(K2,M2+1,L)*XNH1  
                    SUM2 = SUM2 + ENIONND(K2,M2-1,L)*XNH1  
               END DO
               SUM1 = SUM1/DBLE(ND)  
               SUM2 = SUM2/DBLE(ND)  
               IF (SUM1.GT.SUM2) THEN  
                    MM2 = M2 + 1  
               ELSE  
                    MM2 = M2 - 1  
               END IF  

   70                CONTINUE  
               IOPA(NOPA+1) = K2*100 + MM2  
          END IF  
          NOPA = NOPA + 1  
     ELSE  
          MM2 = M2  
!
!       may happen if only one ion per element present
!           if(nopa.eq.2) stop ' error in nopa'
!
     END IF
END IF  
!
!       total number of important ions (with levmax,levmin1)
!
NIMP = NOPA  

DO 90 K = 1,NAT  
! commented out, since we might consider also depleted elements
!     IF (ABUND(K)/ABUND(K1).LE.1.D-7) GO TO 90  
     DO M = IMIA(K),IMAA(K)  
          IF (K.EQ.K1 .AND. (M.EQ.M1.OR.M.EQ.MM1)) cycle  
          IF (K.EQ.K2 .AND. (M.EQ.M2.OR.M.EQ.MM2)) cycle  
          NOPA = NOPA + 1  
          IOPA(NOPA) = K*100 + M  
     END DO
   90 END DO  

IF (NOPA.GT.KIS*KEL) STOP ' ERROR IN NOPA'  
PRINT *  
PRINT *,' USED IONS FOR FREQUENCY GRID'  
PRINT *  

DO N = 1,NOPA  
     IF (IOPA(N).EQ.0) CYCLE  
     IHI = IOPA(N)/100  
     IZ = INT(ZEFF(IHI))  
    ISI = IOPA(N) - 100*IHI  
     IF (NIONS(IHI).EQ.ISI) STOP ' ERROR IN NIONS OR ISI'  
     LEVUSED = LEVMAX  
     IF (N.GT.2) LEVUSED = LEVMIN1  
     IF (N.GT.NIMP) THEN   !NEW
       IF(IMAA(IHI)-IMIA(IHI).GE.2) THEN ! at least 3 ions present
         LEVUSED = LEVMIN2  
       ELSE
         LEVUSED = LEVMIN1  
       ENDIF
     ENDIF
     WRITE (*,FMT=9000) LABAT(IHI),ISI,IZ + ISI - 1,LEVUSED  
END DO  

PRINT *  

IFRE = 0  
LEVTOT = 0  
DO 150 N = 1,NOPA  

     IF (IOPA(N).EQ.0) GO TO 150  
     K = IOPA(N)/100  
     I = IOPA(N) - 100*K
     
     IF (I.EQ.NIONS(K)) STOP ' ERROR IN NIONS OR ISI'  
     IF (I.LE.0) STOP ' ERROR IN ION - FRESCAL '  

     LEVPORG = 0  
     DO II = 1,ID_RBFTR  
          ILAB = LABL4(II)  
          IF (LE(ILAB).EQ.K .AND. LI(ILAB).EQ.I) THEN  

!JO Dec. 2015: avoid edges around Lya and Lyb
!JO July 2021: and around Lyg and Lyd
               IF(FRECIN(II).GT.LYA_LO .AND. FRECIN(II).LT.LYA_HI) THEN
                  PRINT*,'WARNING!!!! WARNING!!!! WARNING!!!!'
                  PRINT*,'EDGE CLOSE TO LY-ALPHA NEGLECTED IN FREQ. GRID',K,' ',I
                  STOP ' WHEN YOU SEE THIS STATEMENT, PLEASE CONTACT J.P.'
! after tests, the stop statement can be exchanged with the cycle statement
!                  CYCLE
               ENDIF  

               IF(FRECIN(II).GT.LYB_LO .AND. FRECIN(II).LT.LYB_HI) THEN
                  PRINT*,'WARNING!!!! WARNING!!!! WARNING!!!!'
                  PRINT*,'EDGE CLOSE TO LY-BETA NEGLECTED IN FREQ. GRID',K,' ',I
                  STOP ' WHEN YOU SEE THIS STATEMENT, PLEASE CONTACT J.P.'
! after tests, the stop statement can be exchanged with the cycle statement
!                  CYCLE
               ENDIF  

               IF(FRECIN(II).GT.LYG_LO .AND. FRECIN(II).LT.LYG_HI) THEN
                  PRINT*,'WARNING!!!! WARNING!!!! WARNING!!!!'
                  PRINT*,'EDGE CLOSE TO LY-GAMMA NEGLECTED IN FREQ. GRID',K,' ',I
                  STOP ' WHEN YOU SEE THIS STATEMENT, PLEASE CONTACT J.P.'
! after tests, the stop statement can be exchanged with the cycle statement
!                  CYCLE
               ENDIF  

               IF(FRECIN(II).GT.LYD_LO .AND. FRECIN(II).LT.LYD_HI) THEN
                  PRINT*,'WARNING!!!! WARNING!!!! WARNING!!!!'
                  PRINT*,'EDGE CLOSE TO LY-DELTA NEGLECTED IN FREQ. GRID',K,' ',I
                  STOP ' WHEN YOU SEE THIS STATEMENT, PLEASE CONTACT J.P.'
! after tests, the stop statement can be exchanged with the cycle statement
!                  CYCLE
               ENDIF  

               LEVPORG = LEVPORG + 1  
               FLAUX(LEVPORG) = FRECIN(II)  

          END IF  
     END DO
!
!     reordering of flaux since fl maybe not ordered by energy
!     example: hei
!
     LEVTOT = LEVTOT + LEVPORG  
     CALL SORT(LEVPORG,FLAUX)  
!
!     find which edges shall be resolved
!
!     levmax levels for main ions. stage of elem. k1,k2
!
     LEVE = MIN(LEVPORG,LEVMAX)  
     LEVE1 = MIN(LEVE,LEVMIN1)  

     IF(IMAA(K)-IMIA(K).GE.2) THEN ! NEW, at least 3 ions present
       LEVE2 = MIN(LEVE,LEVMIN2)  
     ELSE
       LEVE2 = LEVE1
     ENDIF
    
!
!     levmin1 levels for 2nd important ions. stage of elem. k1,k2
!
     IF (N.GT.2) LEVE = LEVE1  
!
!     levmin2 levels for rest
!
     IF (N.GT.NIMP) LEVE = LEVE2  

     LEVP = LEVPORG
     IF (LEVP.EQ.0) STOP ' ERROR IN LEVP - FRESMIN '  
     LEVP1 = MIN(LEVMIN1,LEVP) ! not used in the following???

     DO L = 1,LEVP  
          IFRE = IFRE + 1  
          LLI = LEVPORG + 1 - L  
          DO II = 1,ID_RBFTR  
               IF (FLAUX(LLI).EQ.FRECIN(II)) GO TO 130  
          END DO
          STOP ' FLAUX NOT FOUND IN FRECIN'  

  130           CONTINUE  
! K number of element 
! I number of ion
! II number of transition
! all w.r.t. DATA file (MAINION and ENIONND w.r.t enumeration as in data file)
          INDEX(IFRE) = 100000*K + I*1000 + II  

! here L counts from the highest transition freq. on.
! If there is only ground state ionization, L corresponds to level (starting at 1).
! For ionization to excited states, L does *not* correspond to level
! NOTE: INDEDGE modified below, if required
          IF (L.LE.LEVE) THEN  
               INDEDGE(IFRE) = 1  
          ELSE  
               INDEDGE(IFRE) = 0  
          END IF  
!
!     note that flaux here is in the "wrong' order
!
          FRE(IFRE) = FLAUX(LLI)  
!
      END DO

  150 END DO  

IF (LEVTOT.GT.ID_RBFTR) STOP ' ERROR IN LEVTOT'  

!IF (IFRE.GT.ID_LLEVS) STOP  ! this was the old statement with one edge per level
IF (IFRE.GT.ID_RBFTR) STOP ' ERROR IN IFRE'  ! THIS IS THE NEW ONE

PRINT *,' NUMBER OF CONSIDERED EDGES = ',IFRE  
PRINT *  

LAM_LY=0.
DO I = 1,IFRE  
     EPS = EPSLON(FRE(I))  
     FRE(I+IFRE) = FRE(I) - EPS
! RED SIDE OF LYMAN-JUMP
!     IF(INDEX(I).EQ.101001) LAM_LY=1.D8/FRE(I+IFRE) 
! above statement not general, new version
     IL = INDEX(I)  
     K = IL/100000  
     IL = IL - 100000*K  
     IO = IL/1000  
     LL = IL - 1000*IO  
     LAB = LABL(LABL4(LL))  
     PARE = PAREN4(LL)  
     IF(LAB.EQ.'H11' .AND. PARE.EQ.'H21') LAM_LY=1.D8/FRE(I+IFRE) 
     FREEDGE(I) = FRE(I)  
END DO  

IF1 = IFRE  
IFRE = 2*IFRE  

!K-shell edges
NK=0

IF(OPTXRAY) THEN
! only edges from ionization stage III on
  DO K=1,N_KEDGES
    IF (NIONK(K).LT.K_NMIN) CYCLE
    IF (NIONK(K).GT.K_NMAX) CYCLE
    IF (ETH(K).GT.2.D7) CYCLE
     NK=NK+1
     IFRE=IFRE+1
     FRE(IFRE)=ETH(K)
     IFRE=IFRE+1
     EPS = EPSLON(ETH(K))  
     FRE(IFRE) = FRE(IFRE-1) - EPS
  ENDDO
PRINT *,' NUMBER OF CONSIDERED K-SHELL EDGES = ',NK  
PRINT *  
ENDIF  

CALL SORT(IFRE,FRE)  

FREMIN=FRE(1)
I=LOG10(FREMIN)-1
I=MAX(I,0)
I=10**I
FREMIN=FLOAT(INT(FREMIN/I)*I)

IF (FRE(1).LE.FREMIN) STOP ' ERROR IN FREMIN'  

!OLD VERSION, WITH FRE(1)=FREMIN
!DO I = IFRE + 1,2,-1  
!          FRE(I) = FRE(I-1)  
!END DO 

!FRE(1) = FREMIN
!IFRE = IFRE + 1  

!NEW VERSION FOR mm-FLUXES
! 9 points before fremin, starting at 1mm = 10 Kayser
DFL=LOG10(FREMIN/10.)/9. !10 Kayser, 10 points = 9 intervals

DO I = IFRE + 10,11,-1  
          FRE(I) = FRE(I-10)  
END DO 

FRE(1)=10.
DO I=1,9
  FRE(I+1)=FRE(1)*10.**(I*DFL)
ENDDO  
  
IFRE=IFRE+10

IF(OPTXRAY) THEN
  AAA = 2.D7 !    (5 A)
  
ELSE
EMAX = 10.D0*TMAX/HKL  

IF(NAT.EQ.1.AND.K1.EQ.1.OR.TEFF.LT.10000.) THEN
  AAA=4.D5    ! (250 A)
ELSE IF(TEFF.LT.20000.) THEN
  AAA = 1.D8/180.D0  !  (180 A)
ELSE IF(TEFF.LT.35000.) THEN
  AAA = 1.D6  !  (100 A)
ELSE
  AAA = 5.D6  !  ( 20 A)
ENDIF

ENDIF

PRINT *
PRINT *, ' INTERMEDIATE FRE(IFRE): ', 1.0D8/FRE(IFRE)
PRINT *, ' MIN      : ', 1.0D8/AAA
PRINT *

EMAX = MAX(EMAX,AAA)  
EMAX = MAX(EMAX,FRE(IFRE)*1.2D0)  

IF(EMAX.GT.7.5D5) THEN ! in case, resolve HeII edge
  IFRE = IFRE + 1  
  FRE(IFRE) = 7.5D5 ! (133 A)
ENDIF

IF(OPTXRAY) THEN
  IFRE = IFRE + 1  
  FRE(IFRE) = 1.2D6  !(86 A, half way between 133 A and first K-shell at 39A)
  IFRE = IFRE + 1  
  FRE(IFRE) = 5.D6  ! ( 20 A) 
ENDIF  

IFRE = IFRE + 1  
FRE(IFRE) = EMAX  
!
!here was a bug, missing sort statement until Oct. 2014
CALL SORT(IFRE,FRE)  
!
!      changed to obtain similar freq. grids in all cases
!      dmin=log(fre(ifre)/fre(1))/nfmin
!
DMIN = LOG(1.D6/1.D3)/NFMIN  
NPER = IFRE - 1  

DO 220 I = 1,NPER  

     IF (FRE(I+1).LT.1.D4) THEN ! >10000 A  
          DMIN1 = DMIN*3  
     ELSE IF (FRE(I+1).LT.5.D4) THEN ! >2000 A   
          DMIN1 = DMIN*2  
     ELSE IF (FRE(I+1).LT.1.D8/1600.) THEN ! >1600 A   
!changed Feb 2015, for better resol between 2000 and 1600 of non-HHe models
!          DMIN1 = DMIN*2  
          DMIN1 = DMIN/4.          
     ELSE IF (FRE(I+1).LT.1.D8/910.) THEN ! > 910 A   
          DMIN1 = DMIN/4.
!
!   minimum separation behind HeII edge should be approx. 10  A or 2 A (if X-rays)
!
     ELSE IF (FRE(I+1).GT.440528.D0) THEN  ! < 227 A
          DMIN2 = 10.D-8 * FRE(I+1)  !  DELTA LAM / LAM   
          IF(OPTXRAY) DMIN2 = 2.D-8 * FRE(I+1)  !  DELTA LAM / LAM   
          IF(OPTXRAY.AND.FRE(I+1).GT.5.D6)  DMIN2=0.3
          DMIN1 = DMIN2      
     ELSE  ! 227 (or min) ... 910 A 
          DMIN1 = 2000./300000. !(MIN. RESOL = 2000 KM/S)  
     END IF  

     DM = FRE(I+1)/FRE(I) - 1.D0  
!
!----      find out whether edge should be resolved
!
     IMP = 0  
     DO JI = 1,IF1  
          IF (FRE(I).EQ.FREEDGE(JI) .AND. INDEDGE(JI).EQ.1) IMP = 1
     END DO
!
!----      nice trick to obtain resolved edges in case
!
     IF (DM.GT.DMIN1/4.D0) THEN  
          INEW = INT(DM/DMIN1) + 1  
          DNEW = (FRE(I+1)-FRE(I))/INEW  

          DO 200 J = 1,INEW  
               DNEW1 = DNEW  
!
!   concentration towards edge for important transitioms
!
               IF (IMP.EQ.1 .AND. J.EQ.1) THEN  
                    DNEW1 = DNEW/4.D0  
                    DO K = 1,3  
                         FRENEW = FRE(I) + K*DNEW1  
                         IFRE = IFRE + 1  
                         FRE(IFRE) = FRENEW  
                         IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*, &
                          FMT=9030) 1.D8/FRE(IFRE),DNEW1, DNEW1/ &
                          FRE(IFRE)
                    END DO

               ELSE IF (IMP.EQ.1 .AND. J.EQ.2) THEN  
                    DNEW1 = DNEW/2.D0  
                    FRENEW = FRE(IFRE) + DNEW1  
                    IFRE = IFRE + 1  
                    FRE(IFRE) = FRENEW  
                    IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*, &
                     FMT=9030) 1.D8/FRE(IFRE),DNEW1, DNEW1/FRE( &
                     IFRE)
               END IF  

               IF (J.NE.INEW) THEN  
                    FRENEW = FRE(I) + J*DNEW  
                    IFRE = IFRE + 1  
                    FRE(IFRE) = FRENEW  
                    IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*, &
                     FMT=9030) 1.D8/FRE(IFRE),DNEW1, DNEW1/FRE( &
                     IFRE)

               END IF  

  200           CONTINUE  
     ELSE IF (IMP.EQ.1) THEN  
!
!          important edge, but due to near next edge not resolved
!          assume then that next edge is important and so on, until
!          final resolution
!
          DO JI = 1,IF1  
               IF (FRE(I+2).EQ.FREEDGE(JI)) INDEDGE(JI) = 1  
          END DO
     END IF  

  220 END DO  


!JO Dec. 2015
! modify frequency points around Lya and Lyb so that consistency
! with pure HHe freq. grid (if HHe only, no action performed)
! points chosen in such a way that Lya and Lyb not completely centered,
! to allow for a compromise (otherwise self-shadowing too strong)  

! in case, add points
DO I=1,IFRE
  IF(ABS(FRE(I)-LYA_LO).LT.1.D-9) GOTO 226 
ENDDO
IFRE=IFRE+1
FRE(IFRE)=LYA_LO
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FRE(IFRE)

226 DO I=1,IFRE
  IF(ABS(FRE(I)-LYA_HI).LT.1.D-9) GOTO 227
ENDDO
IFRE=IFRE+1
FRE(IFRE)=LYA_HI
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FRE(IFRE)

227 DO I=1,IFRE
  IF(ABS(FRE(I)-LYB_LO).LT.1.D-9) GOTO 228 
ENDDO
IFRE=IFRE+1
FRE(IFRE)=LYB_LO
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FRE(IFRE)

228 DO I=1,IFRE
  IF(ABS(FRE(I)-LYB_HI).LT.1.D-9) GOTO 229 
ENDDO
IFRE=IFRE+1
FRE(IFRE)=LYB_HI
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FRE(IFRE)

229 CONTINUE

CALL SORT(IFRE,FRE)  

!-----------------------------------------------
!in case, remove points in between lya_lo,lya_hi
DO I=1,IFRE
  IF(ABS(FRE(I)-LYA_LO).LT.1.D-9) GOTO 326 
ENDDO
STOP ' LYA_LO NOT FOUND IN FRESCAL' 

326 ISTART=I
DO I=ISTART,IFRE
  IF(ABS(FRE(I)-LYA_HI).LT.1.D-9) GOTO 327 
ENDDO
STOP ' LYA_HI NOT FOUND IN FRESCAL' 

327 IEND=I

IF (OPTOUT .AND. ICONVER.EQ.1) THEN
  DO I=ISTART+1,IEND-1
    WRITE (*,FMT=9033) 1.D8/FRE(I)
  ENDDO
ENDIF

NSKIP=IEND-ISTART-1
IF(NSKIP.LT.0) STOP ' ERROR IN NSKIP (FRESCAL)'

IF(NSKIP.NE.0) THEN
  IFRE=IFRE-NSKIP
  DO I=ISTART+1,IFRE
    FRE(I)=FRE(I+NSKIP)
  ENDDO
ENDIF

!-----------------------------------------------
!in case, remove points in between lyb_lo,lyb_hi

DO I=1,IFRE
  IF(ABS(FRE(I)-LYB_LO).LT.1.D-9) GOTO 328 
ENDDO
STOP ' LYB_LO NOT FOUND IN FRESCAL' 

328 ISTART=I
DO I=ISTART,IFRE
  IF(ABS(FRE(I)-LYB_HI).LT.1.D-9) GOTO 329 
ENDDO
STOP ' LYB_HI NOT FOUND IN FRESCAL' 

329 IEND=I

IF (OPTOUT .AND. ICONVER.EQ.1) THEN
  DO I=ISTART+1,IEND-1
    WRITE (*,FMT=9033) 1.D8/FRE(I)
  ENDDO
ENDIF

NSKIP=IEND-ISTART-1
IF(NSKIP.LT.0) STOP ' ERROR IN NSKIP (FRESCAL)'

IF(NSKIP.NE.0) THEN
  IFRE=IFRE-NSKIP
  DO I=ISTART+1,IFRE
    FRE(I)=FRE(I+NSKIP)
  ENDDO
ENDIF

!    check resolution for
!    hydrogen resonance lines and hei singlet resonance line
!

!XMIN=1.D0/1030.D-8
XMIN=1.D0/1025.D-8  !Lyman-beta
XMAX=1.D0/912.D-8

!following block changed Feb 2015 to improve resolution 
!and to include Lyman beta (important for Halpha)
!JO Dec. 2015: around Lya and Lyb, nothing should happen now
!(predefined points used for all atomic models (hopefully)
DO I = 1,IFRE-1  

  IF (FRE(I).LE.1.D0/1215.D-8 .AND. FRE(I+1).GT.1.D0/1215.D-8) THEN !LYMAN ALPHA
  DMIN1=0.005D0  !VERY HIGH RESOLUTION
! leave this value here;
! usually, not completely centered, to allow for a compromise (otherwise self-shadowing too strong)  
! if freq. points at Lya as given used, no additional point should be included
  DM = FRE(I+1)/FRE(I) - 1.D0  
  IF(DM .LT. DMIN1) CYCLE

! new treatment
  ELSE IF (FRE(I).LE.XMIN .AND. FRE(I+1).GT.XMIN) THEN !LYMAN BETA
! that was the condition in the 'erroneous' version 10.7-1
!  DMIN1=0.001D0  !EVEN HIGHER RESOLUTION
  DMIN1=0.005D0  !VERY HIGH RESOLUTION
! leave this value here;
! usually, not completely centered, to allow for a compromise (otherwise self-shadowing too strong)  
! if freq. points at Lyb as given used, no additional point should be included
  DM = FRE(I+1)/FRE(I) - 1.D0  
  IF(DM .LT. DMIN1) CYCLE
    
  ELSE IF (FRE(I).GT.XMIN .AND. FRE(I+1).LE.1.D0/925.D-8) THEN ! OTHER HYDROGEN RES. LINES
!  DMIN1=0.01
  DMIN1=0.001
  DM = FRE(I+1)/FRE(I) - 1.D0  
  IF(DM .LT. DMIN1) CYCLE

  ELSE IF (FRE(I).GT.XMIN .AND. FRE(I).LE.1.D0/925.D-8 .AND. FRE(I+1).LE.XMAX) THEN
  DMIN1=1.5/925. ! max resol = 1.5 A
  DM = FRE(I+1)/FRE(I) - 1.D0  
  IF(DM .LT. DMIN1) CYCLE
  
  ELSE IF (FRE(I).GT.1.D0/925.D-8 .AND. FRE(I+1).LE.XMAX) THEN
!  DMIN1=3./925. ! max resol = 3 A
  DMIN1=1.5/925. ! max resol = 1.5 A
  DM = FRE(I+1)/FRE(I) - 1.D0  
  IF(DM .LT. DMIN1) CYCLE

  ELSE IF (FRE(I).LE.XMAX .AND. FRE(I+1).GT.XMAX) THEN
!  DMIN1=3./912. ! max resol = 3 A
  DMIN1=1.5/912. ! max resol = 1.5 A
  DM = FRE(I+1)/FRE(I) - 1.D0  
  IF(DM .LT. DMIN1) CYCLE
! end new treatment

  ELSE IF (FRE(I).LE.1.D0/584.D-8 .AND. FRE(I+1).GT.1.D0/584.D-8) THEN !HEI RES. LINE
!  DMIN1=0.01
  DMIN1=0.001
  DM = FRE(I+1)/FRE(I) - 1.D0  
  IF(DM .LT. DMIN1) CYCLE

  ELSE ! NO PROBLEMS SO FAR, HEII LYMAN ALPHA RESOLVED ANYWAY
  CYCLE
  ENDIF

! INCLUDE ADDITONAL FREQ. POINTS
  INEW = INT(DM/DMIN1) + 1  
  DNEW = (FRE(I+1)-FRE(I))/INEW  

     DO J = 1,INEW-1  
       FRENEW = FRE(I) + J*DNEW  
       IFRE = IFRE + 1  
       FRE(IFRE) = FRENEW  
       IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9031) &
&        1.D8/FRE(IFRE),DNEW, DNEW/FRE(IFRE)
     END DO

ENDDO

!JO CHANGED March 2017
! ONE ADDITIONAL POINT EXACTLY AT HEII LY-ALPHA
IFRE = IFRE + 1
FRE(IFRE)=1.D8/303.797
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9034) 1.D8/FRE(IFRE)

CALL SORT(IFRE,FRE)  

!JO July 2021
! modify frequency points around Lyg and Lyd so that consistency
! with pure HHe freq. grid (if HHe only, no action performed)

DO I=1,IFRE
  IF(ABS(FRE(I)-LYG_LO).LT.1.D-9) GOTO 230 
ENDDO
IFRE=IFRE+1
FRE(IFRE)=LYG_LO
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FRE(IFRE)

230 DO I=1,IFRE
  IF(ABS(FRE(I)-LYG_HI).LT.1.D-9) GOTO 231 
ENDDO
IFRE=IFRE+1
FRE(IFRE)=LYG_HI
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FRE(IFRE)

231 DO I=1,IFRE
  IF(ABS(FRE(I)-LYD_LO).LT.1.D-9) GOTO 232 
ENDDO
IFRE=IFRE+1
FRE(IFRE)=LYD_LO
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FRE(IFRE)

232 DO I=1,IFRE
  IF(ABS(FRE(I)-LYD_HI).LT.1.D-9) GOTO 233 
ENDDO
IFRE=IFRE+1
FRE(IFRE)=LYD_HI
IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9032) 1.D8/FRE(IFRE)

233 CONTINUE

CALL SORT(IFRE,FRE)  

!-----------------------------------------------
!in case, remove points in between lyg_lo,lyg_hi
DO I=1,IFRE
  IF(ABS(FRE(I)-LYG_LO).LT.1.D-9) GOTO 330 
ENDDO
STOP ' LYG_LO NOT FOUND IN FRESCAL' 

330 ISTART=I
DO I=ISTART,IFRE
  IF(ABS(FRE(I)-LYG_HI).LT.1.D-9) GOTO 331 
ENDDO
STOP ' LYG_HI NOT FOUND IN FRESCAL' 

331 IEND=I

IF (OPTOUT .AND. ICONVER.EQ.1) THEN
  DO I=ISTART+1,IEND-1
    WRITE (*,FMT=9033) 1.D8/FRE(I)
  ENDDO
ENDIF

NSKIP=IEND-ISTART-1
IF(NSKIP.LT.0) STOP ' ERROR IN NSKIP (FRESCAL)'

IF(NSKIP.NE.0) THEN
  IFRE=IFRE-NSKIP
  DO I=ISTART+1,IFRE
    FRE(I)=FRE(I+NSKIP)
  ENDDO
ENDIF

!-----------------------------------------------
!in case, remove points in between lyd_lo,lyd_hi
DO I=1,IFRE
  IF(ABS(FRE(I)-LYD_LO).LT.1.D-9) GOTO 332 
ENDDO
STOP ' LYD_LO NOT FOUND IN FRESCAL' 

332 ISTART=I
DO I=ISTART,IFRE
  IF(ABS(FRE(I)-LYD_HI).LT.1.D-9) GOTO 333 
ENDDO
STOP ' LYG_HI NOT FOUND IN FRESCAL' 

333 IEND=I

IF (OPTOUT .AND. ICONVER.EQ.1) THEN
  DO I=ISTART+1,IEND-1
    WRITE (*,FMT=9033) 1.D8/FRE(I)
  ENDDO
ENDIF

NSKIP=IEND-ISTART-1
IF(NSKIP.LT.0) STOP ' ERROR IN NSKIP (FRESCAL)'

IF(NSKIP.NE.0) THEN
  IFRE=IFRE-NSKIP
  DO I=ISTART+1,IFRE
    FRE(I)=FRE(I+NSKIP)
  ENDDO
ENDIF

CALL SORT(IFRE,FRE)  

!-----------------------------------------------
!in case, add points at center of important Stark-broadened lines (k ge 6)
IF(NSTARK.NE.0) THEN

OUTER: DO II=1,NSTARK ! not executed for nstark=0

! DO NOT TOUCH INTERVALS FOR HYDROGEN LY-LINES  
  KAYSER=1.D8/WAVESTARK(II)
  IF((KAYSER.LE.LYA_HI.AND.KAYSER.GE.LYA_LO).OR. &
     (KAYSER.LE.LYB_HI.AND.KAYSER.GE.LYB_LO).OR. &
     (KAYSER.LE.LYG_HI.AND.KAYSER.GE.LYG_LO).OR. &
     (KAYSER.LE.LYD_HI.AND.KAYSER.GE.LYD_LO)) CYCLE OUTER

  DO I=1,IFRE
    IF(ABS(FRE(I)-KAYSER).LT.FRE(I)*0.5D-3 .AND. ABS(WAVESTARK(II)-LAM_CIII).GT.1.D-4) THEN ! corresponding to 0.5 A at 1000 A
!      PRINT*,'NEGLECTED',1.D8/FRE(I),WAVESTARK(II)
      CYCLE OUTER
    ENDIF
  ENDDO
  IFRE=IFRE+1
  IF (IFRE.GT.IFRETOT) STOP ' TOO MANY FREQUENCIES!'  
  FRE(IFRE)=KAYSER
  IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9035) 1.D8/FRE(IFRE)
ENDDO OUTER 

CALL SORT(IFRE,FRE)
ENDIF


PRINT *,' TOTAL NUMBER OF FREQUENCIES = ',IFRE  
PRINT *  
PRINT *,' FINAL FRE(IFRE): ', 1.0D8/FRE(IFRE)
PRINT *


IF (OPTOUT .AND. ICONVER.EQ.1) THEN  
    PRINT *,' WAVELENGTH(A)     FREQ(HZ)'  
     DO 250 I = 1,IFRE  

          DO J = 1,IF1  
               IF (FRE(I).EQ.FREEDGE(J)) GO TO 240
          END DO

          IF(OPTXRAY) THEN
          DO J = 1,N_KEDGES  
               IF (FRE(I).EQ.ETH(J)) THEN
                 WRITE (*,FMT=9025) 1.D8/FRE(I),CLIGHT*FRE(I), &
                 NAME(Z(J)),NIONK(J)-1  
               GO TO 250  
               ENDIF
          ENDDO
          ENDIF
   
          WRITE (*,FMT=9010) 1.D8/FRE(I),CLIGHT*FRE(I)  
          GO TO 250  

  240           CONTINUE  
          IL = INDEX(J)  
          K = IL/100000  
          IZ = INT(ZEFF(K))  
          IL = IL - 100000*K  
          IO = IL/1000  
          LL = IL - 1000*IO  
          LAB = LABL(LABL4(LL))  
          PARE = PAREN4(LL)  
          IF(INDEDGE(J).NE.1) THEN
            WRITE (*,FMT=9020) 1.D0/FRE(I)*1.D8,CLIGHT*FRE(I), &
             LABAT(K),IO,IZ + IO - 1,LAB,PARE
          ELSE
            WRITE (*,FMT=9021) 1.D0/FRE(I)*1.D8,CLIGHT*FRE(I), &
             LABAT(K),IO,IZ + IO - 1,LAB,PARE
          ENDIF

 250        CONTINUE  
     PRINT *  
END IF  

DO K = 1,IFRE  
     BN(K) = BNUE(1.D8/FRE(K),TMAX)  
END DO  

PF = XINTFRE(BN,WFRE,1,IFRE,IFRE,FRE,4,'NEW')  
PFE = SIGSB/PI*TMAX**4  
ERRP = ABS(1.D0-PF/PFE)  
PRINT *,' ERROR IN PLANCK-FUNCTION WITH ACTUAL FREQ. GRID = ',ERRP  
PRINT * 

!JO Sept. 2021
!check whether frequency grid has changed (points can be the same,
!but frequencies different). Important to reset OP_FLAG
NEWFREQ=.FALSE.

IF(IFREOLD.EQ.0) THEN
  NEWFREQ=.TRUE.
ELSE IF (IFRE.NE.IFREOLD) THEN
  NEWFREQ=.TRUE.
ELSE
  DO K=1,IFRE
    IF(ABS(1.D0-FRE(K)/FREOLD(K)).GT.1.D-15) THEN
      NEWFREQ=.TRUE.
      EXIT    
    ENDIF
  ENDDO  
ENDIF

PRINT*
IF(NEWFREQ) THEN
  PRINT*,' MESSAGE FROM FRESCAL: FREQ. GRID HAS CHANGED!'   
ELSE
  PRINT*,' MESSAGE FROM FRESCAL: FREQ. GRID HAS NOT CHANGED!'   
ENDIF
PRINT*

!stop
RETURN  

 9000 FORMAT (1X,A2,I2,' Z = ',I2,' MAX.NO.OF RESOLVED EDGES = ',I2)  
 9010 FORMAT (1X,F12.3,6X,E12.6)  
 9020 FORMAT (1X,F12.3,6X,E12.6,3X,A2,I2,' Z=',I2,'  FROM ',A6,' TO ', &
&       A6)
 9021 FORMAT (1X,F12.3,6X,E12.6,3X,A2,I2,' Z=',I2,'  FROM ',A6,' TO ', &
&       A6,' IMPORTANT')
 9025 FORMAT (1X,F12.3,6X,E12.6,3X,A2,' Z=',I2,' K-SHELL ABSORPTION')
 9030 FORMAT (' ADDIT. POINT AT ',F12.3,6X,F12.3,' DELTA E / E = ', &
&       F5.3)
 9031 FORMAT (' ADDIT. POINT AT ',F12.3,6X,F12.3,' DELTA E / E = ', &
&       F5.3,' (RESONANCE LINES!)')
 9032 FORMAT (' ADDIT. POINT AT ',F12.3,' LY_ALPHA/LY_BETA/LY_GAM/LY_DELTA')
 9033 FORMAT ('        POINT AT ',F12.3,' DROPPED (LY_ALPHA/LY_BETA/LY_GAM/LY_DELTA)')
 9034 FORMAT (' ADDIT. POINT AT ',F12.3,' HEII LY_ALPHA')
 9035 FORMAT (' ADDIT. POINT AT ',F12.3,' TRANSITION WITH STARK TREATMENT IN SUMOPAL')
END
!
!-----------------------------------------------------------------------
!
FUNCTION XINTFRE(QUAN,W,ILOW,IMAX,N,X,IOPT,CHAR)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     integrates quan over frequency from ilow to imax
!
!     char = 'new'..... new integration weights calculated in frewe
!     char = 'old'..... old integration weights used
!
!     x.... input for frewe as described there
!     opt.. input for frewe describing x
!
!
!     .. scalar arguments ..
INTEGER(I4B) ::  ILOW,IMAX,IOPT,N  
CHARACTER CHAR*3  
!     ..
!     .. array arguments ..
REAL(DP) ::  QUAN(N),W(N),X(N),XINTFRE  
!     ..
!     .. local scalars ..
REAL(DP) ::  SUMM  
INTEGER(I4B) :: I  
!     ..
!     .. external subroutines ..
EXTERNAL FREWE  
!     ..

IF (CHAR.EQ.'NEW') THEN  
     CALL FREWE(X,W,N,IOPT)  
ELSE IF (CHAR.NE.'OLD') THEN  
     STOP 'WRONG OPTION FOR CHAR IN INTFRE'  
END IF  

SUMM = 0.D0  
DO I = ILOW,IMAX  
     SUMM = SUMM + QUAN(I)*W(I)  
END DO  

XINTFRE = SUMM  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FREWE(X,W,N,IOPT)  

USE nlte_type  
USE nlte_dim  
USE fund_const, ONLY: CLIGHT
USE nlte_var, ONLY: ICONVER,PRECIS  

IMPLICIT NONE  
!
!     calculates integration weights: input x, dim(x) = n  < 2500
!
!     iopt=1 frequencies
!     iopt=2 wavelengths (a)
!     iopt=3 wavelengths (cm)
!     iopt=4 kayser (1/cm)
!
!     x can be ordered from high to low or vice versa
!
!     output w, frequency integration weights such that
!
!     integral(x-nue dnue) = sum over i(x-i*w-i)
!
!     .. scalar arguments ..
INTEGER(I4B) ::  IOPT,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  W(N),X(N)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DELNUE,DELNUE1,SUMM  
INTEGER(I4B) ::  I,II,K,KINT,KK,NINT,NINT2,NINT3,NINT4,NINT5, &
&  NINT6,NINT7,NINT8
LOGICAL OPTINV,OPTOUT  
!     ..
!     .. local arrays ..
REAL(DP) ::  DEL(4000),XNUE(4000)  
INTEGER(I4B) ::  INDEX(4000)  
!     ..
!     .. external subroutines ..
EXTERNAL FREWE2,FREWE3,FREWE4,FREWE5,FREWE6,FREWE7,FREWE8  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,MOD  
!     ..

IF (N.GT.4000) STOP ' N > 4000 IN FREWE'  

NINT2 = 0  
NINT3 = 0  
NINT4 = 0  
NINT5 = 0  
NINT6 = 0  
NINT7 = 0  
NINT8 = 0  

OPTOUT = .TRUE.  

OPTINV = .FALSE.  

IF (IOPT.EQ.1 .OR. IOPT.EQ.4) THEN  
     IF (X(1).LT.X(N)) THEN  
          OPTINV = .TRUE.  
          DO I = 1,N  
               XNUE(I) = X(N+1-I)  
          END DO
     ELSE  
          DO I = 1,N  
               XNUE(I) = X(I)  
          END DO
     END IF  
     IF (IOPT.EQ.4) THEN  
          DO I = 1,N  
               XNUE(I) = CLIGHT*XNUE(I)  
          END DO
     END IF  

ELSE IF (IOPT.EQ.2 .OR. IOPT.EQ.3) THEN  
     IF (X(1).GT.X(N)) THEN  
          OPTINV = .TRUE.  
          DO I = 1,N  
               XNUE(I) = X(N+1-I)  
          END DO
     ELSE  
          DO I = 1,N  
               XNUE(I) = X(I)  
          END DO
     END IF  
     IF (IOPT.EQ.2) THEN  
          DO I = 1,N  
               XNUE(I) = CLIGHT*1.D8/XNUE(I)  
          END DO
     ELSE  
          DO I = 1,N  
               XNUE(I) = CLIGHT/XNUE(I)  
          END DO
     END IF  

ELSE  
     STOP 'WRONG INPUT OPTION'  

END IF  


K = 1  
I = 1  
INDEX(1) = 1  

   80 CONTINUE  
DELNUE = XNUE(I) - XNUE(I+1)  
IF (I+1.EQ.N) THEN  
     DEL(K) = DELNUE  
     K = K + 1  
     INDEX(K) = N  
     GO TO 110  
END IF  

DO II = I + 1,N - 1  
     DELNUE1 = XNUE(II) - XNUE(II+1)  
     IF (ABS(1.D0-DELNUE/DELNUE1).GT.1.D-5) GO TO 100  
END DO  

DEL(K) = DELNUE  
K = K + 1  
INDEX(K) = N  
GO TO 110  

  100 CONTINUE  
DEL(K) = DELNUE  
K = K + 1  
INDEX(K) = II  
I = II  
GO TO 80  

  110 CONTINUE  
KINT = K  
IF (OPTOUT .AND. ICONVER.EQ.1) THEN  
     PRINT *  
     PRINT *,' NUMBER OF EQUIDISTANT SUBINTERVALS = ',KINT - 1  
     PRINT *  
END IF  

W(1) = 0.D0  
DO KK = 2,KINT  
     NINT = INDEX(KK) - INDEX(KK-1) + 1  

     IF (NINT.EQ.1) THEN  
          STOP ' WRONG NUMBER OF SUBINTERVALS'  

     ELSE IF (NINT.EQ.2) THEN  
          NINT2 = NINT2 + 1  
          CALL FREWE2(INDEX(KK-1),INDEX(KK),DEL(KK-1),W,N)  

     ELSE IF (NINT.EQ.3) THEN  
          NINT3 = NINT3 + 1  
          CALL FREWE3(INDEX(KK-1),INDEX(KK),DEL(KK-1),W,N)  

     ELSE IF (NINT.EQ.4) THEN  
          NINT4 = NINT4 + 1  
          CALL FREWE4(INDEX(KK-1),INDEX(KK),DEL(KK-1),W,N)  

     ELSE IF (MOD(NINT,4).EQ.1) THEN  
          NINT5 = NINT5 + 1  
          CALL FREWE5(INDEX(KK-1),INDEX(KK),DEL(KK-1),W,N)  

     ELSE IF (MOD(NINT,4).EQ.2) THEN  
          NINT6 = NINT6 + 1  
          CALL FREWE6(INDEX(KK-1),INDEX(KK),DEL(KK-1),W,N)  

     ELSE IF (MOD(NINT,4).EQ.3) THEN  
          NINT7 = NINT7 + 1  
          CALL FREWE7(INDEX(KK-1),INDEX(KK),DEL(KK-1),W,N)  

     ELSE IF (MOD(NINT,4).EQ.0) THEN  
          NINT8 = NINT8 + 1  
          CALL FREWE8(INDEX(KK-1),INDEX(KK),DEL(KK-1),W,N)  

     END IF  
END DO  

IF (OPTOUT .AND. ICONVER.EQ.1) THEN  
    PRINT *  
    PRINT *,' NUMBER OF              TRAPEZ INTEGRATIONS = ',NINT2  
    PRINT *,' NUMBER OF             SIMPSON INTEGRATIONS = ',NINT3  
    PRINT *,' NUMBER OF                 3/8 INTEGRATIONS = ',NINT4  
    PRINT *,' NUMBER OF        5-PT-SIMPSON INTEGRATIONS = ',NINT5  
    PRINT *,' NUMBER OF 5-PT-SIMPSON+TRAPEZ INTEGRATIONS = ',NINT6
    PRINT *,' NUMBER OF        7-PT-SIMPSON INTEGRATIONS = ',NINT7  
    PRINT *,' NUMBER OF 7-PT-SIMPSON+TRAPEZ INTEGRATIONS = ',NINT8
    PRINT *  
END IF  

SUMM = 0.D0  
DO K = 1,N  
     SUMM = SUMM + W(K)  
END DO  

DELNUE = XNUE(1) - XNUE(N)  
IF (ABS(1.D0-DELNUE/SUMM).GT.PRECIS) STOP 'ERROR IN FREWE!'  

IF (.NOT.OPTINV) RETURN  

DO K = 1,N  
     XNUE(K) = W(N+1-K)  
END DO  

DO K = 1,N  
     W(K) = XNUE(K)  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FREWE2(I1,I2,DEL,W,N)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!     trapezoidal rule
!
!
!     .. scalar arguments ..
REAL(DP) ::  DEL  
INTEGER(I4B) ::  I1,I2,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  W(N)  
!     ..

W(I1) = W(I1) + .5D0*DEL  
W(I2) = .5D0*DEL  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FREWE3(I1,I2,DEL,W,N)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!     simpson's rule
!
!
!     .. scalar arguments ..
REAL(DP) ::  DEL  
INTEGER(I4B) ::  I1,I2,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  W(N)  
!     ..

W(I1) = W(I1) + DEL/3.D0  
W(I1+1) = 4.D0*DEL/3.D0  
W(I2) = DEL/3.D0  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FREWE4(I1,I2,DEL,W,N)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!     3/8 rule
!
!
!     .. scalar arguments ..
REAL(DP) ::  DEL  
INTEGER(I4B) ::  I1,I2,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  W(N)  
!     ..

W(I1) = W(I1) + 3.D0*DEL/8.D0  
W(I1+1) = 9.D0*DEL/8.D0  
W(I1+2) = 9.D0*DEL/8.D0  
W(I2) = 3.D0*DEL/8.D0  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FREWE5(I1,I2,DEL,W,N)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!     modified simpson's rule
!
!
!     .. scalar arguments ..
REAL(DP) ::  DEL  
INTEGER(I4B) ::  I1,I2,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  W(N)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  K,NINT,NK  
!     ..
!     .. intrinsic functions ..
!INTRINSIC MOD  
!     ..

NINT = I2 - I1 + 1  
DO K = I1,I2  
     NK = K - I1 + 1  

     IF (NK.EQ.1) THEN  
          W(K) = W(K) + (1.D0-1.D0/15.D0)*DEL/3.D0  

     ELSE IF (NK.EQ.NINT) THEN  
          W(K) = (1.D0-1.D0/15.D0)*DEL/3.D0  

     ELSE IF (MOD(NK,2).EQ.0) THEN  
          W(K) = 4.D0* (1.D0+1.D0/15.D0)*DEL/3.D0  

     ELSE IF (MOD(NK,4).EQ.1) THEN  
          W(K) = 2.D0* (1.D0-1.D0/15.D0)*DEL/3.D0  

     ELSE  
          W(K) = 1.6D0*DEL/3.D0  

     END IF  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FREWE6(I1,I2,DEL,W,N)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!     modified simpson's rule + trapezoidal rule for last interval
!
!
!     .. scalar arguments ..
REAL(DP) ::  DEL  
INTEGER(I4B) ::  I1,I2,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  W(N)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  K,NINT,NK  
!     ..
!     .. intrinsic functions ..
!INTRINSIC MOD  
!     ..

NINT = I2 - I1 + 1  

DO K = I1,I2  
     NK = K - I1 + 1  

     IF (NK.EQ.1) THEN  
          W(K) = W(K) + (1.D0-1.D0/15.D0)*DEL/3.D0  

     ELSE IF (NK.EQ.NINT-1) THEN  
          W(K) = (1.D0-1.D0/15.D0)*DEL/3.D0 + DEL/2.D0  

     ELSE IF (NK.EQ.NINT) THEN  
          W(K) = DEL/2.D0  

     ELSE IF (MOD(NK,2).EQ.0) THEN  
          W(K) = 4.D0* (1.D0+1.D0/15.D0)*DEL/3.D0  

     ELSE IF (MOD(NK,4).EQ.1) THEN  
          W(K) = 2.D0* (1.D0-1.D0/15.D0)*DEL/3.D0  

     ELSE  
          W(K) = 1.6D0*DEL/3.D0  

     END IF  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FREWE7(I1,I2,DEL,W,N)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!     simpson's rule
!
!
!     .. scalar arguments ..
REAL(DP) ::  DEL  
INTEGER(I4B) ::  I1,I2,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  W(N)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  K,NINT,NK  
!     ..
!     .. intrinsic functions ..
!INTRINSIC MOD  
!     ..

NINT = I2 - I1 + 1  

DO K = I1,I2  
     NK = K - I1 + 1  
     IF (NK.EQ.1) THEN  
          W(K) = W(K) + DEL/3.D0  

     ELSE IF (NK.EQ.NINT) THEN  
          W(K) = DEL/3.D0  

     ELSE IF (MOD(NK,2).EQ.0) THEN  
          W(K) = 4.D0*DEL/3.D0  

     ELSE  
          W(K) = 2.D0*DEL/3.D0  

     END IF  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FREWE8(I1,I2,DEL,W,N)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!     simpson's rule + trapezoidal rule for last interval
!
!
!     .. scalar arguments ..
REAL(DP) ::  DEL  
INTEGER(I4B) ::  I1,I2,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  W(N)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  K,NINT,NK  
!     ..
!     .. intrinsic functions ..
!INTRINSIC MOD  
!     ..

NINT = I2 - I1 + 1  

DO K = I1,I2  
     NK = K - I1 + 1  

     IF (NK.EQ.1) THEN  
          W(K) = W(K) + DEL/3.D0  

     ELSE IF (NK.EQ.NINT-1) THEN  
          W(K) = DEL/3.D0 + DEL/2.D0  

     ELSE IF (NK.EQ.NINT) THEN  
          W(K) = DEL/2.D0  

     ELSE IF (MOD(NK,2).EQ.0) THEN  
          W(K) = 4.D0*DEL/3.D0  

     ELSE  
          W(K) = 2.D0*DEL/3.D0  

     END IF  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE NISTAR  

USE nlte_type  
USE nlte_dim  
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&       ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
& LABL4,LABL1,LABU1,LABL3,IPARE5, &
& LABL2,PAREN2,PAREN3,PAREN4,LABU2  

USE nlte_var, ONLY: MLOW,MUP,MMLOW,MMUP,MCLOW,MCUP, &
& INDR,INDC  

IMPLICIT NONE  
!
! ------this subroutine calculates (nlow/nup)*
!       hay que definir tambien las matrices de /atomdat/
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IFORM,IHI,ILO,INTEG,IUP,JJ,K,M, &
&        MCLOWA,MCUPA,MLOWA,MMLOWA,MUPA,NC,NDAT,NDATOS
LOGICAL QCL,QDB,QRT  
CHARACTER LUPP*6  
!     ..
!     .. local arrays ..
REAL(DP) ::  ABUND(ID_ATOMS),DATA(ID_NDATA),DL(ID_NLONG), &
&                 FL(ID_LLEVS),FLX(ID_XLEVS),FRECFN(ID_RBFTR), &
&                 FRECIN(ID_RBFTR),GL(ID_LLEVS),GLX(ID_XLEVS), &
&                 GS0(ID_SLEVS),GS1(ID_SLEVS),QD(ID_SLEVS), &
&                 RYD(ID_SLEVS),WEIGHT(ID_ATOMS),ZEFF(ID_ATOMS), &
&                 ZL(ID_LLEVS)
INTEGER(I4B) ::  IFPTR(ID_NTTRD),NFPTR(ID_NTTRD)  
!     ..

DO I = 1,INDEX4  

     LUPP = PAREN4(I)  
     ILO = LABL4(I)  
     MLOWA = 0  
     K = LE(ILO)  
     DO JJ = 1,K - 1  
          MLOWA = MLOWA + IXA1(JJ)  
     END DO

     MLOW(I) = MLOWA + ILO  
     IHI = KL(ILO)  

   20      CONTINUE  

     IF (LABL(IHI).EQ.LUPP) THEN  
          IUP = IHI  
          GO TO 30  
     END IF  

     IF (IHI.EQ.NL) STOP ' UPPER LEVEL NOT FOUND IN RBF '  
     IHI = IHI + 1  

     GO TO 20  

   30      CONTINUE  

     K = LE(IUP)  
     MUPA = 0  

     DO JJ = 1,K - 1  
          MUPA = MUPA + IXA1(JJ)  
     END DO

     MUP(I) = MUPA + IUP  

END DO  
!
!------ introduced (4-oct-92) for bound-bound transitions treatment
!
DO I = 1,INDEX1  

     ILO = LABL1(I)  
     MMLOWA = 0  
     K = LE(ILO)  

     DO JJ = 1,K - 1  
          MMLOWA = MMLOWA + IXA1(JJ)  
     END DO

     MMLOW(I) = MMLOWA + ILO  
     IUP = LABU1(I)  
     MMUP(I) = MMLOWA + IUP  
END DO  
!
!------ for cbf transitions
!
DO I = 1,INDEX3  

     ILO = LABL3(I)  
     IF (ILO.LT.0) THEN  
          STOP ' ILO < 0 IN LABL3'  
!              mclow(i)=0
!              mcup(i)=0
     ELSE  
          MCLOWA = 0  
          K = LE(ILO)  

          DO JJ = 1,K - 1  
               MCLOWA = MCLOWA + IXA1(JJ)  
          END DO

          MCLOW(I) = MCLOWA + ILO  
          LUPP = PAREN3(I)  
          IHI = KL(ILO)  

   90           CONTINUE  

          IF (LABL(IHI).EQ.LUPP) THEN  
               IUP = IHI  
               GO TO 100  
          END IF  

          IF (IHI.EQ.NL) STOP 'UPPER LEVEL NOT FOUND - CBF'  
          IHI = IHI + 1  
          GO TO 90  

  100           CONTINUE  

          K = LE(IUP)  
          MCUPA = 0  
          DO JJ = 1,K - 1  
               MCUPA = MCUPA + IXA1(JJ)  
          END DO

          MCUP(I) = MCUPA + IUP  

     END IF  

END DO  

INDR = INDEX4  
INDC = INDEX3  
IF (INDR.LT. (NL-NAT)) STOP ' LEVELS NOT CONSIDERED IN PHOTO. '  

RETURN  

END
!
!----------------------------------------------------------------------
!
SUBROUTINE OCCLTE(RHO,XNE,XNH,CLF,TE,ILOW,IMAX,ND,NTEMP,LTE_UPDATE)  
!---- unaffected by OPTTHICK 

USE nlte_type  
USE nlte_dim
USE fund_const
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
& ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS  
                               
USE nlte_var, ONLY: IONISST,IONIS,XNELTE, &
& OCCNUM,MAINION, &
& ALEVEL,BLEVEL,ENIONND, &
& IMIA,IMAA, &
& ABUNDND,YHEIN, &
& ICONVER, MODNAM, PARTIT,OPTMET

USE tcorr_var, ONLY: ENATCOR, EMAXTC, TEMP_CONVERGED
                             
IMPLICIT NONE  
!
!-------this subroutine calculates lte occupation numbers, &
!       accounting for clumping with enhancement-factor clf and
!       corrected values for xne and xnh
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND,NTEMP  
!     ..
!     .. array arguments ..
REAL(DP) ::  RHO(ND1),TE(ND1),XNE(ND1),XNH(ND1),CLF(ND1)  
INTEGER(I4B) ::  ILOW(ND1,KEL),IMAX(ND1,KEL)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DELTAE,DUMREAL,E0,EN,OCC,OPTMIXED,REL, &
&                 RSTNOM,TEFF,XLOGG
INTEGER(I4B) ::  I,I1,I2,IHI,IMA,IMI,IONI,ISI,J,K,K1,L,LEO,LEVL, &
&        LEVX,LL,M,MAIN,NENE,NNN,NREC,IMI1,IMA1
LOGICAL HELONE,OPTOUT,OPTNEUPDATE,LTE_UPDATE,DUMLOGIC,OPTCMF1  
CHARACTER DUMCHAR*30, AT*6  
!     ..
!     .. external subroutines ..
EXTERNAL GROUND1  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,EXP,MAX,MAX0,MIN,MIN0  
!     ..
OPTOUT=.TRUE.
                        
NREC = ID_LLEVS  

!check consistency
IF(LTE_UPDATE.AND..NOT.ENATCOR.AND.EMAXTC.GT.0.003D0) &
& STOP ' INCONSISTENCY IN OCCLTE'
                                      
OPEN  (1,FILE=TRIM(MODNAM)//'/INDAT.DAT',STATUS='OLD')  
REWIND 1  
READ (1,FMT=*) DUMCHAR
READ (1,FMT=*) OPTNEUPDATE,HELONE,NNN,NNN  
READ (1,FMT=*) OPTMIXED  
READ (1,FMT=*) TEFF,XLOGG,RSTNOM  
READ (1,FMT=*) DUMREAL,DUMREAL  
READ (1,FMT=*) DUMREAL,DUMREAL,DUMREAL  
READ (1,FMT=*) YHEIN,DUMREAL  
READ (1,FMT=*) DUMLOGIC,DUMLOGIC,DUMLOGIC,DUMLOGIC,OPTCMF1
CLOSE (1)

IF (OPTOUT .AND. ICONVER.EQ.1) THEN  
     WRITE (*,FMT=9000) TEFF,XLOGG,YHEIN  
     PRINT *  
END IF  
!
!-----------------------------------------------------------------------
!
NENE = 0  
DO ISI = 1,NAT  
     IF (LABAT(ISI).EQ.'HE') NENE = ISI  
     IF (IXA1(ISI).NE.0) STOP ' X-LEVELS NOT YET IMPLEMENTED'  
END DO  

CALL GROUND1(TE,RHO,XNE,XNH,CLF,ND,OPTOUT,NTEMP,YHEIN,NENE)  

! occmain maximum fraction of all ions EXCLUDING the highest one without
! transitions
!
! IF TEMP. ALMOST CONVERGED, DON'T CHANGE IMIN, IMAX ETC.
!IF(LTE_UPDATE.AND.EMAXTC.LT.0.05) THEN
! JO CHANGED: one final update after temperature has converged, to allow
! successful restart
IF(LTE_UPDATE .AND. EMAXTC.LT.0.03 .AND. .NOT. TEMP_CONVERGED) THEN
! at least, check for changes in IONIS
     DO K=1,NAT
       DO L=1,ND
         IF(IMAX(L,K).GT.IONIS(K,L)) THEN
           IMAX(L,K)=IONIS(K,L)
           IF(IMAX(L,K).LT.ILOW(L,K)) STOP ' IMAX < ILOW'
           PRINT*,'CORRECTION OF IMAX(LTE) REQUIRED!!!'
           PRINT*,K,L,' IMAX(NLTE) = ',IMAX(L,K),' (ZEFF CORRECTED)'
         ENDIF  
       ENDDO
     ENDDO  
     GOTO 100
ENDIF
!
!----------- determination of ilow and imax etc, for LTE calculations
!            and H/He (always) in the old spirit
!            Note the major change from version 9.0 on

IF (OPTMET) THEN
  CALL ILOWIMAX_LTE(ILOW,IMAX,TE,TEFF,ND,NTEMP,HELONE,OPTCMF1,OPTOUT,LTE_UPDATE)
ELSE
! this is the very old version kept for consistency
! in case optmet=.false., i.e., no background metals
! (which are used to predict the correct ionization
! equilibrium in the standard approach)  
  CALL ILOWIMAX_OLD(ILOW,IMAX,TE,TEFF,ND,NTEMP,HELONE,OPTCMF1,OPTOUT)
ENDIF  
!
!-------------calculation of n(i,j,k) = g_i *(n1/g1) * exp(-dE/kT)
!             n1/g1=occnum at this stage (see ground1)
!
100 CONTINUE
DEPTHLOOP: DO LL = NTEMP,ND  

     J = 0  
     M = 0  

KLOOP: DO K = 1,NAT  

       K1 = NIONS(K)  ! excitation for ALL levels
                      ! thus: excited levels of IONIS(K,L) populated
                      ! rest = 0.

IONLOOP:  DO I = 1,K1 - 1  
               J = J + 1  
               LEVL = IFIRSL(J+1) - IFIRSL(J)  
               LEVX = IFIRX(J+1) - IFIRX(J)  
!                levs=ifirs(j+1)-ifirs(j)
               E0 = FL(IFIRSL(J))  

               DO LEO = 0,LEVL - 1  
                    M = M + 1  
                    IHI = IFIRSL(J) + LEO  
                    DELTAE = ABS(E0-FL(IHI))  
                    OCC = OCCNUM(K,I,LL)*EXP(-1.D0*HK*DELTAE/TE(LL))
                    ALEVEL(M) = OCC*GL(IHI)  
                    BLEVEL(M) = ALEVEL(M)  
               END DO

               DO LEO = 0,LEVX - 1  
                    M = M + 1  
                    IHI = IFIRX(J) + LEO  
                    DELTAE = (E0-FLX(IHI))  
                    OCC = OCCNUM(K,I,LL)*EXP(-1.D0*HK*DELTAE/TE(LL))
                    ALEVEL(M) = OCC*GLX(IHI)  
                    BLEVEL(M) = ALEVEL(M)  
               END DO
!
!                do leo=o,levs
!                   ihi=ifirs(j)+leo
!                   do isi=ns0(ihi),ns1(ihi)
!                      m=m+1
!                      deltae=e0-ryd(ihi)/(isi-qd(ihi))**2
!                      occ=occnum(k,i,ll)*exp(-1.d0*deltae/akb/te(ll))
!                      alevel(m)=occ
!                      blevel(m)=alevel(m)
!                   end do
!                end do
!
          END DO IONLOOP
!
!-----------k level
!
          J = J + 1  
          M = M + 1  


! here was the bug (forgotten to multiply with partition function of uppermost ion)!

          IF (PARTIT(K,K1,LL) .NE. GL(M)) STOP ' SOMETHING WRONG WITH LAST G-VALUE!'
          OCC = OCCNUM(K,K1,LL)*GL(M)  
          ALEVEL(M) = OCC  
          BLEVEL(M) = ALEVEL(M)  

          END DO KLOOP

     WRITE (15,REC=LL) (ALEVEL(I),I=1,NREC)  
     WRITE (17,REC=LL) (BLEVEL(I),I=1,NREC)  

     XNELTE(LL) = XNE(LL)  

     IF (ICONVER.EQ.1) THEN  
        PRINT *,' L= ',LL,'   TOTAL NUMBER OF LEVELS= ',M  
!
!---    calculation of depth dependent abundance (including clumping)
!
          DO K = 1,KEL  
               ABUNDND(K,LL) = 0.D0  
               DO I = 1,KIS + 1  
                    ABUNDND(K,LL) = ABUNDND(K,LL) + ENIONND(K,I,LL)
               END DO
          END DO
     END IF  

END DO DEPTHLOOP 

PRINT *  
PRINT *,'  FILE OCCLTE WRITTEN '  
PRINT *  

RETURN  

 9000 FORMAT (10X,'TEFF= ',F7.0,'LOGG= ',F5.2,'YHE= ',F5.2)  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE GROUND1(TE,RHO,XNE,XNHHYD,CLF,ND,OPTOUT,NTEMP,YHEIN,NENE)  
!---- unaffected by OPTTHICK

USE nlte_type  
USE nlte_dim  
USE fund_const
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&    ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD  

USE nlte_var, ONLY: IONISST,IONIS,XNELTE, &
& PARTIT,XL, &
& OCCNUM,MAINION, &
& ALEVEL,BLEVEL,ENIONND,ENIONND_LTE, &
& ICONVER, PRECIS  

IMPLICIT NONE  
!
!       this subroutine calculates the lte population of ground level
!       (divided by g1), accounting for clumping with clumping factor clf
!       xne and xnh are corrected, rho is average (uncorrected) density
!
!
!------- calculation of partition function
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  YHEIN  
INTEGER(I4B) ::  ND,NENE,NTEMP  
LOGICAL OPTOUT  
!     ..
!     .. array arguments ..
REAL(DP) ::  RHO(ND1),TE(ND1),XNE(ND1),XNHHYD(ND1),CLF(ND1)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AG,AL,DEV,E1,EDENH,EL,G,OCCMAIN,SUMMASS,SUMXMUE, &
&                 XNEH,XNH,ZI,ZI1
INTEGER(I4B) ::  I,IHI,II,IIT,IJ,IK,IS,IZ,J,JJ,K,K1,L,L1,L2,LL,LX, &
&        N,NIONMAX
LOGICAL CONV,NON  
!     ..
!     .. local arrays ..
REAL(DP) ::  ENION(KEL),U(ND1)  
!     ..
!     .. external functions ..
INTEGER(I4B) ::  IGENIO  
EXTERNAL IGENIO  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,DBLE,EXP,INT,MAX  
!     ..

SUMMASS = 0.D0  
NIONMAX = 0  

DO I = 1,NAT  
     NIONMAX = MAX(NIONMAX,NIONS(I))  
     SUMMASS = SUMMASS + WEIGHT(I)*ABUND(I)  
END DO  

IF (NENE.EQ.0 .AND. YHEIN.NE.0.) THEN  
     SUMMASS = SUMMASS + 4.D0*YHEIN  
END IF  

IF (NIONMAX-1.GT.KIS) STOP ' KIS TOO SMALL, RE-DIMENSION IN DIMES!'

J = 0  

KLOOP1: DO K = 1,NAT  
     IK = NIONS(K)  
IONLOOP: DO JJ = 1,IK - 1  
          J = J + 1  
          DO LX = NTEMP,ND  
               U(LX) = 0.D0  
          END DO

          N = IFIRSL(J)  
          E1 = FL(N)  
          IHI = IFIRSL(J+1) - 1  

          DO LL = IFIRSL(J),IHI  
               EL = ABS(FL(LL)-E1)  
               DO LX = NTEMP,ND  
                    U(LX) = U(LX) + GL(LL)*EXP(-1.D0*HK*EL/TE(LX))  
               END DO
          END DO

          IHI = IFIRX(J+1) - 1  
          DO LL = IFIRX(J),IHI  
               EL = ABS(FLX(LL)-E1)  
               DO LX = NTEMP,ND  
                    U(LX) = U(LX) + GLX(LL)*EXP(-1.D0*HK*EL/TE(LX))
               END DO
          END DO

          IHI = IFIRS(J+1) - 1  

          DO LL = IFIRS(J),IHI  
               DO IS = NS0(LL),NS1(LL)  
                    EL = ABS(ABS(RYD(LL))/ (DBLE(IS)-QD(LL))**2) - E1*HH
                    G = GS0(LL) + GS1(LL)*DBLE(IS)**2  
                    DO LX = NTEMP,ND  
                         U(LX) = U(LX) + G*EXP(-1.D0*EL/AKB/TE(LX))
                    END DO
               END DO
          END DO

          PARTIT(K,JJ,NTEMP:ND) = U(NTEMP:ND)  

     END DO IONLOOP

     J = J + 1  
     IHI = IFIRSL(J)  

     DO LX = NTEMP,ND  
          PARTIT(K,IK,LX) = GL(IHI)  
     END DO

END DO KLOOP1
!
!-------calculation of n(j,k)/n(h) !
!
SUMXMUE = 0.D0  

IIT = 0  

DEPTHLOOP: DO LL = NTEMP,ND  

     CONV = .FALSE.  
     DEV = 1.D0  

  140      CONTINUE  

     IIT = IIT + 1  
!
!-------at first, we calculate of l(j,k)
!
     I = 0  
KLOOP2: DO K = 1,NAT  
          K1 = NIONS(K)  
          NON = .FALSE.  

          DO II = 1,K1 - 1  
               I = I + 1  
               N = IFIRSL(I)  
               AL = FL(N)  
               AG = AL*HK/TE(LL)  

               IF ((AG.GT.150D0) .AND. .NOT.NON) THEN  
                    IONIS(K,LL) = II - 1  
                    NON = .TRUE.  
               END IF  
! xne includes clumping
               XL(K,II) = .5D0*SAHA*PARTIT(K,II,LL)*XNE(LL)*EXP( &
                AG)/ PARTIT(K,II+1,LL)/TE(LL)**1.5D0
          END DO

          I = I + 1  
          XL(K,K1) = 1.D0  
          IF (.NOT.NON) IONIS(K,LL) = K1 - 1  
     END DO KLOOP2
!
!------calculation of n(j,k)/n(h) = occnum(k,i,ll)
!
     XNH = RHO(LL)*CLF(LL)/AMH/SUMMASS  ! corrected for clumping
!     IF (ABS(1.D0-XNHHYD(LL)/XNH).GT.PRECIS) STOP 'ERROR IN XNH'  
     IF (ABS(1.D0-XNHHYD(LL)/XNH).GT.0.01) STOP 'ERROR IN XNH' !changed  

KLOOP3: DO K = 1,NAT  
          K1 = IONIS(K,LL)  
          AL = 1.D0  

          DO L = 1,K1  
               AG = 1.D0  

               DO L1 = L,K1  
                    AG = AG*XL(K,L1)  
               END DO

               AL = AL + AG  
          END DO

          ENION(K) = ABUND(K)/AL  
          OCCNUM(K,K1,LL) = XL(K,K1)*ENION(K)  

          DO L2 = K1 - 1,1,-1  
               OCCNUM(K,L2,LL) = XL(K,L2)*OCCNUM(K,L2+1,LL)  
          END DO

     END DO KLOOP3
!
!------------- new electronic density, accounting for zeff
!
     EDENH = 0.D0  

KLOOP4: DO K = 1,NAT  
          K1 = IONIS(K,LL)  
          DO I = 1,K1  
               ZI = ZEFF(K) + I - 1  
!
!     check of charge
!
               IJ = IGENIO(K,I)  
               ZI1 = ZL(IFIRSL(IJ))  
               IF (ZI.NE.ZI1) STOP ' ERROR IN CHARGE'  
               EDENH = EDENH + ZI*OCCNUM(K,I,LL)  
          END DO

          ZI = ZEFF(K) + K1  
          IJ = IGENIO(K,K1+1)  
          ZI1 = ZL(IFIRSL(IJ))  
          IF (ZI.NE.ZI1) STOP ' ERROR IN CHARGE-HIGHEST ION'  
          EDENH = EDENH + ZI*ENION(K)  

     END DO KLOOP4

     IF (CONV) SUMXMUE = SUMXMUE + RHO(LL)*CLF(LL) / (EDENH*XNH)  
     XNEH = XNE(LL)/XNH  
     DEV = ABS(XNEH-EDENH)/XNEH  
     XNE(LL) = EDENH*XNH  
!
!--------------calculation of mainion and output
!
     IF (CONV) THEN  

          DO K = 1,NAT  
               OCCMAIN = 0.D0  
               K1 = IONIS(K,LL)  

               DO L = 1,K1  
                    ENIONND(K,L,LL) = OCCNUM(K,L,LL)*XNH  
                    OCCMAIN = MAX(OCCMAIN,OCCNUM(K,L,LL))  
                    IF (OCCMAIN.EQ.OCCNUM(K,L,LL)) MAINION(LL,K) = L
               END DO

               ENIONND(K,K1+1,LL) = ENION(K)*XNH  
               ENIONND(K,K1+2:KIS,LL) = 0.D0 ! in case ionization changes
          END DO
          ENIONND_LTE=ENIONND ! for metals in iterations before CONCON 


          IF (OPTOUT .AND. ICONVER.EQ.1) THEN  
               PRINT *  
               PRINT *,'-----------------------------------------'  
               PRINT *  
               PRINT *,'  L=',LL,'    TE=',TE(LL),'    NE/NH=',XNEH  
               PRINT *  
               PRINT *, &
&                ' NO(ELEM) NO(MAIN ION) Z(MAIN ION) // NO  Z  N(J,K)/N(H)  '
               DO K = 1,NAT  
                    K1 = IONIS(K,LL)  
                    IZ = INT(ZEFF(K))  
                    WRITE (*,FMT=9000) K,MAINION(LL,K),(IZ+ MAINION(LL,K)-1), &
&                     (L, (IZ+L-1), OCCNUM(K,L,LL),L=1,K1), &
&                     K1 + 1,IZ + K1,ENION(K)
               END DO
          END IF  

     END IF  
!
!---------------calculation of n(j,k)/u(j,k)=occnum(k,i,ll) = n1/g1
!
KLOOP5: DO K = 1,NAT  
          K1 = IONIS(K,LL)  

          DO I = 1,K1  
               OCCNUM(K,I,LL) = OCCNUM(K,I,LL)/PARTIT(K,I,LL)*XNH  
          END DO

          L1 = NIONS(K)  
          OCCNUM(K,K1+1,LL) = ENION(K)/PARTIT(K,K1+1,LL)*XNH  

          DO I = K1 + 2,L1  
               OCCNUM(K,I,LL) = 0.D0  
          END DO

     END DO KLOOP5

     IF (CONV) CYCLE  
     IF (DEV.LT.PRECIS) CONV = .TRUE.  
     GO TO 140  

END DO DEPTHLOOP 
!
!-----------------------------------------------------------------------
!
SUMXMUE = SUMXMUE/AMH/ (ND+1-NTEMP)  
IIT = IIT/ (ND+1-NTEMP)  
PRINT *  
PRINT *,' MUE-ELECTRON= ',SUMXMUE  
PRINT *,' AVERAGE NUMBER OF NE-ITERATIONS WAS ',IIT  
PRINT *  

RETURN  
 9000 FORMAT (/,I3,2 (3X,'/',I3),/,9 (I3,3X,I3,3X,E12.6,/))  

END
!
!-----------------------------------------------------------------------
!
FUNCTION IGENIO(KK,II)  

USE nlte_type  
USE nlte_dim  
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&     ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS

IMPLICIT NONE  
!
!------ this function returns the value of 'general ion' for atom kk and
!       ion ii
!
!     .. scalar arguments ..
INTEGER(I4B) ::  II,KK,IGENIO  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I  
!     ..
!     .. local arrays ..
REAL(DP) ::  ABUND(ID_ATOMS),FL(ID_LLEVS),FLX(ID_XLEVS), &
&                 GL(ID_LLEVS),GLX(ID_XLEVS),GS0(ID_SLEVS), &
&                 GS1(ID_SLEVS),QD(ID_SLEVS),RYD(ID_SLEVS), &
&                 WEIGHT(ID_ATOMS),ZEFF(ID_ATOMS),ZL(ID_LLEVS)
!     ..

IGENIO = 0  

DO I = 1,KK - 1  
     IGENIO = IGENIO + NIONS(I)  
END DO  

IGENIO = IGENIO + II  
IF (IGENIO.GT.IONG) STOP ' ERROR IN GENERAL ION - IGENIO '  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE OPACITC(ND,XNE,XNH,TEMP,CLF,ILOW,IMAX,LTEOPT,NTEMP,OPTETA, &
&                   FRENEW,OPTMET)
!
!
!-------this subroutine calculates the free-free and bound-free
!       opacity for every frequency at every depth point
!
!---    note: standard detail input provides only ff opacities
!       for h and he. e.g., if calculating wr's, take care for cno!!!
!
!       CHANGED FROM VER 9.5
!---    warning: opacities calculated for all transitions with
!                n_l ne 0 (bf) or ENIONND ne 0 (ff).
!                Occupation/ionic numbers with zero value should
!                occur only  in the range outside ILOW / IMAX  
!
!       ilow and imax can be lte or nlte values, dependent of call
!
!       THE FINAL OPACITIES, EMISSIVITES AND SOURCE-FUNCTIONS
!       HAVE BEEN CORRECTED FOR CLUMPING, EXCEPT FFOPA WHICH IS NEEDED
!       ONLY FOR THE CALCULATION OF THE T-STRUCTURE IN ORDER TO ENABLE
!       CORRECTLY EVALUATED SPATIAL AVERAGES
!
!       OPAC CHANGED FOR EFFECTS FROM OPTICALLY THICK CLUMPING, I.E.,
!       OPAC IS NOW EFFECTIVE OPACITY
!       ALL OTHER OPACITIES ETC. SHOULD REMAIN AS BEFORE (OPTICALLY THIN
!       CLUMPING), I.E., SHOULD BE MEAN OPACITIES  
!
!       includes emissivities from hot plasma (wind-embedded shocks)
  
USE nlte_type  
USE nlte_dim  
USE fund_const
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&       ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
& LABL,LABL4,LABL1,LABU1,LABL3,IPARE5, &
& TL,ANCDO,WLC,DL,FRECIN,FRECFN 

USE nlte_var, ONLY: ALPHAFS, &
& GFFNLTE, IONMAX, IZMIN, IZMAX, &
& IONISST,IONIS,XNELTE, &
& FRE,WFRE,HC,IFRE, &
& ALEVEL,BLEVEL,ENIONND, &
& MLOW,MUP,MMLOW,MMUP,MCLOW,MCUP, &
& OPAC,STRUE,XJ,ALO, &
& OPAT_M_OLD,OPAT_M_NEW,STRUE_M_OLD,STRUE_M_NEW, &
& OPTLINES,OPAC_NOLINES,OPAT_M_NOLINES,PRECIS, &
& OPTNEUPDATE, CONCON, YHE=>YHEIN, OP_FLAG, FRECFN1
!,RAYLEIGH  

USE nlte_xrays, ONLY: OPTXRAY, LAMBDANU, FXL, LXMIN, OPA_KSHELL

USE tcorr_var, ONLY : FFOPA, DTFFOPA 

USE nlte_porvor, ONLY: EPSI,EPSI1,OPTTHICK, &
&                      FIC,TCL_FAC_LINE,TCL_FAC_CONT, &
&                      OPA_EFF_RAT,OPA_EFF_RAT_OLD, &
&                      W_BG_RED,W_BG_BLUE  

IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  

!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND,NTEMP  
LOGICAL FRENEW,LTEOPT,OPTETA,OPTMET  
!     ..
!     .. array arguments ..
REAL(DP) ::  TEMP(ND1),XNE(ND1),XNH(ND1),CLF(ND1)  

INTEGER(I4B) ::  ILOW(ND1,KEL),IMAX(ND1,KEL)  
!     ..
!     .. local scalars ..
REAL(DP) ::  A,ALPHA,BN,BN1,CHIFF,CHIFFC,DEPART,EHNUEKT,ETAB, &
&                 ETABF,ETAFFC,FREQ,GF,HCFRE3,OPAB,OPABF,OPAFF, &
&                 OPAFF1,OPATH,SIG,XNERAT,XNI,XNISTAR,XNK,Z, &
&                 OPAHMINUS,EN,CONSTX_LL,X,EXPO, TCL,OPA_DUM, LAM, XNEL

REAL(DP) :: ZERO=0.

INTEGER(I4B) ::  I,IFORM,II,IJ,IZ, &
&        INATO,INTEG,KK,LISTIN,LL,M,NATO,NC,NDAT,NDATOS,NLAS,NM,NN, &
&        NREC, KMIN
LOGICAL QCL,QDB,QRT,START  
!     ..
!     .. local arrays ..
REAL(DP) ::  DATA(ID_NDATA)  
INTEGER(I4B) ::  IFPTR(ID_NTTRD),NFPTR(ID_NTTRD)  
!JO: Sept 2016: commented out, not needed
!CHARACTER KLS(ID_SLEVS)*6,LABAT(ID_ATOMS)*6,LABL(ID_LLEVS)*6, &
!&          LABL2(ID_RCBXT)*6,LABLS(ID_SLEVS)*6,LABU2(ID_RCBXT)*6, &
!&          LABX(ID_XLEVS)*6,LKL(ID_LLEVS)*6,LKX(ID_XLEVS)*6, &
!&          PAREN2(ID_RCBXT)*6,PAREN3(ID_CBSFT)*6,PAREN4(ID_RBFTR)*6
!     ..
!     .. external functions ..
REAL(DP) ::  GFF  
INTEGER(I4B) ::  IGENIO  
EXTERNAL GFF,IGENIO  
!     ..
!     .. external subroutines ..
EXTERNAL CROSSBF  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,EXP,SQRT  
!     ..

NREC = ID_LLEVS  
SIG = AMH*SIGMAE  

IF(.NOT.LTEOPT.AND.OPTXRAY) THEN
  CALL OPACIT_KSHELL(ND,XNE,ILOW,IMAX)
  KMIN=LBOUND(OPA_KSHELL,DIM=2)
  IF(UBOUND(OPA_KSHELL,DIM=2).NE.IFRE) STOP ' ERROR IN UBOUND OF OPA_KSHELL'
ENDIF
  
FFOPA =   0.D0 
DTFFOPA = 0.D0 

IF (FRENEW) THEN   ! IN LTE-ITERATION, FRENEW IS ALWAYS TRUE
IZMIN=100
IZMAX=0

! Find out all possible charges to be treated

          DO I = 1,INDEX5  
               IF (IFORM5(I).NE.1) THEN  
!------        has to be implemented if x-levels are present
                    PRINT *,'RFF NOT IMPLEMENTED; ',I,'!'  
                    STOP  
               END IF  

               II = IPARE5(I)  
               Z = ZL(II)  
               IZMIN=MIN(IZMIN,INT(Z+1.D-6))
               IZMAX=MAX(IZMAX,INT(Z+1.D-6))
          END DO
             
IF(IZMAX.GT.IONMAX) STOP ' IONMAX > 9, REDIMENSION GFFNLTE!'
!corresponds to NION in nlte_approx

! prepare ff gaunt-factors, for all possible charges, depth points and freq., &
! if not already done in opacitm

 IF(.NOT.OPTMET) THEN
  DO LL=NTEMP,ND
    DO KK=1,IFRE
      FREQ=FRE(KK)
      DO IZ=IZMIN,IZMAX
          Z=FLOAT(IZ)
          GFFNLTE(IZ,LL,KK) = GFF(FREQ*CLIGHT,TEMP(LL),Z,LL,KK,IZ)*Z*Z  
      ENDDO 
    ENDDO
  ENDDO
 ENDIF 

ENDIF

DEPTHLOOP: DO LL = NTEMP,ND  

     IF(.NOT.LTEOPT.AND.OPTXRAY) THEN
!JO modified July 2016
       XNEL=XNH(LL)*(1.+2.*YHE)
       CONSTX_LL=XNEL/CLF(LL)*XNH(LL)/CLF(LL)*FXL(LL)/(4.*PI)
!---- No changes for optically thick clumping, since emission from hot plasma
!---- with independent filling factor FXL (either constant or radius-dependent)
     ENDIF
       
     IF (LTEOPT) THEN  
        IF (XNE(LL).NE.XNELTE(LL)) STOP ' ERROR IN NE -- LTEOPT'  
          XNERAT = 1.D0  
     ELSE   
!        IF (OPTNEUPDATE.AND.XNE(LL).EQ.XNELTE(LL)) STOP ' ERROR IN NE -- NLTEOPT'  
          XNERAT = XNE(LL)/XNELTE(LL)  
     END IF  

     READ (17,REC=LL) (BLEVEL(I),I=1,NREC)  
     READ (15,REC=LL) (ALEVEL(I),I=1,NREC)  

! check for clumping if included (so far, assumed that RAYLEIGH itself not corrected)
!     IF(TEMP(LL).LT.12000.) THEN
!       CALL RAYLEIGH_SCAT(LL)
!     ELSE IF(RAYLEIGH(LL,1).NE.0.D0) THEN
!       RAYLEIGH(LL,:) = 0.D0
!     ENDIF  
       
KLOOP: DO KK = 1,IFRE  
          FREQ = FRE(KK)  
          IF (OPTETA) HCFRE3 = HC2*FREQ**3  
          EHNUEKT = EXP(-HKL*FREQ/TEMP(LL))  
          OPABF = 0.D0  
          ETABF = 0.D0  

ILOOP:    DO I = 1,INDEX4
!JO Sept. 2016: for N=20, lower frequencies allowed
               IF(IFORM4(I).NE.20) THEN
                  A = FRECIN(I)
               ELSE      
                  A = FRECFN1(I)
!JO can be skipped later on    
                  IF(A.LT.FRECFN(I)) STOP ' FRECFN1 LT FRECFN'
               ENDIF   
               IF (A.GT.FREQ) CYCLE ILOOP

               IF (FRENEW.AND.LL.EQ.NTEMP) THEN  !ONLY ONCE
                    CALL CROSSBF(I,ALPHA,FREQ)
                    IF(ALPHA.EQ.0.) THEN
                      IF(OP_FLAG(I) .AND. FREQ.LT.FRECIN(I)) THEN
!sigma = 0 in OP-data shortwards from FRECFN, no further problems
                         CYCLE ILOOP
                      ELSE
                         STOP ' ERROR IN ALPHA(1)'
                      ENDIF
!should not happen                        
                    ENDIF
                    ALPHAFS(I,KK) = ALPHA
                    
               ELSE  
                    ALPHA = ALPHAFS(I,KK)  
                    IF (ALPHA.EQ.0.) THEN
                      PRINT*,TRIM(LABL(LABL4(I))),LL
                      PRINT*,FREQ,I,OP_FLAG(I),FRECIN(I),FRECFN1(I)
                      STOP ' ERROR IN ALPHA(2)'
! something rotten, since FRECFN1 should be OK now (defined in previous call
! of OP_RBFSETUP
                    ENDIF
               END IF  

               NN = MLOW(I)  
               INATO = LI(NN)  
               NATO = LE(NN)  
               XNI = BLEVEL(NN)  

               IF (XNI.EQ.0.) THEN
! can happen when restart
!                 IF (INATO.GE.ILOW(LL,NATO).AND.INATO.LE.IMAX(LL,NATO)) THEN
!                   PRINT*,LL,NATO,INATO,ILOW(LL,NATO),IMAX(LL,NATO) 
!                   STOP ' INCONSISTENCY IN XNI=0 (OPACITC)'
!                 ENDIF
                 CYCLE ILOOP
               ENDIF

               IF (INATO.GT.IMAX(LL,NATO)) CYCLE ILOOP
!JO July 2017: hopefully cures problems in restart with X-rays,
!              when new levels with low occ (but large source-function) are included 
               IF (INATO.LT.ILOW(LL,NATO)) THEN
                 IF(ENIONND(NATO,INATO,LL).EQ.0.) CYCLE ILOOP
               ENDIF
               
! either to ground state (for OP_FLAG and low frequencies)  
               IF(OP_FLAG(I) .AND. FREQ.LT.FRECIN(I)) THEN
                 NM = KL(NN)
!                 IF(LL.EQ.1) PRINT*,LABL(LABL4(I)),FREQ,FRECIN(I),NM
! or to ground/excited state as defined in DETAIL input
               ELSE
                 NM = MUP(I)  
               ENDIF
!
!--- outside ilow, imax!
!

               IJ = IGENIO(NATO,IMAX(LL,NATO))+1  
               NLAS = IFIRSL(IJ)  
!
!--- ionization to excited level of imax+1, consistent with netmat
!    completely changed, see notes.
!    idea: assume excited levels to be in LTE to ground-state of nk = nk1
!    then: nistar = nki * (ni/nki)* = nk1 * (ni/nk1)* 
!
!    Note also that there was a bug here in former versions:
!    We used an alternative formulation, but only if xnk_i = 0, 
!    although xnk_i was at its (absolute) LTE value. Since the ground-state
!    was in NLTE, the ratio was wrong, and nistar was wrong as well. 

!
               IF (NM.GT.NLAS) THEN ! this is the condition
                 IF(OP_FLAG(I) .AND. FREQ.LT.FRECIN(I)) STOP ' NM > NLAS for low freq. OP-data!'
                 NM=NLAS
               ENDIF   
! THAT'S ALL!!! 
               XNK = BLEVEL(NM)  
               XNISTAR = XNK*ALEVEL(NN)/ALEVEL(NM)*XNERAT  
               IF (XNISTAR.EQ.0.) THEN
                  PRINT*,LL,NN,NM,XNK,ALEVEL(NN),ALEVEL(NM),XNERAT
                  PRINT*,NATO,ILOW(LL,NATO),IMAX(LL,NATO)
                  PRINT*,BLEVEL
                  STOP ' XNISTAR = 0!'
               ENDIF  
               DEPART = XNI/XNISTAR  

               IF (LTEOPT .AND. ABS(DEPART-1.D0).GT.PRECIS) &
&               STOP 'ERROR IN DEPART - LTE'

               OPABF = OPABF + XNISTAR*ALPHA* (DEPART-EHNUEKT)  
!               if(kk.ge.739.and.kk.le.743.and.mod(ll,6).eq.0 .and. &
!&                nato.eq.3.and.inato.ge.2) then
!            print*,ll,kk,XNISTAR*ALPHA* (DEPART-EHNUEKT),i,nato,inato,nm,nn,alpha,xni,xnk,xnistar,depart
!          endif

               IF (OPTETA) ETABF = ETABF + XNISTAR*ALPHA  
                
          ENDDO ILOOP
!
!-------- now we have calculated bf opacity for all transitions
!------             we start ff opacity
!
          OPAFF = 0.D0  
          CHIFF = 1.3695D-23*XNE(LL)/ (FREQ**3*SQRT(TEMP(LL)))  
          CHIFFC = CHIFF* (1.D0-EHNUEKT)  

          IF (OPTETA) ETAFFC = CHIFF  

          DO I = 1,INDEX5  
               II = IPARE5(I)
               NATO=LE(II)
               INATO=LI(II)
               EN=ENIONND(NATO,INATO,LL)
               IF(EN.EQ.0.) THEN
! can happen when restart
!                 IF (INATO.GE.ILOW(LL,NATO).AND.INATO.LE.IMAX(LL,NATO)+1) THEN
!                   PRINT*,LL,NATO,INATO,ILOW(LL,NATO),IMAX(LL,NATO) 
!                   STOP ' INCONSISTENCY IN ENIONND = 0 (OPACITC)'
!                 ENDIF
                 CYCLE
               ENDIF  
               Z = ZL(II)
               IZ=INT(Z+1.D-6)
               GF = GFFNLTE(IZ,LL,KK)
! FOR TESTS
!               IF(GF.LE.0.) STOP 'ERROR IN FF GAUNT-FACTORS'   
               OPAFF = OPAFF + GF*EN  
          END DO

          OPAFF1 = OPAFF*CHIFFC  
!  the following two quanties remain uncorrected for clumping, since
!  they are needed only for the T-structure and all other quantities
!  entering the energy balance remain uncorrected as well

	  FFOPA(LL,KK) = OPAFF * CHIFF ! maup
	  DTFFOPA(LL,KK) = (-1.d0)*FFOPA(LL,KK)/(2.D0*TEMP(LL))
!  considering that the gaunt fact. does not depend on temperature
	  
          OPATH = XNE(LL)*SIG / CLF(LL)  !xne includes cl =>  mean opacity

          IF(TEMP(LL).LE.10000. .AND. 1.D8/FREQ.GT. 200.D0) THEN
! opahminus yields uncorrected values
            CALL HMINUS(XNE(LL),TEMP(LL),1.D8/FREQ,EHNUEKT,LL,OPAHMINUS)
          ELSE
            OPAHMINUS=0.D0
          ENDIF  
          OPAB = OPABF + OPAFF1 + OPAHMINUS
! still uncorrected
          
! this is the only place where the k-shell opacities should appear
          IF(.NOT.LTEOPT.AND.OPTXRAY.AND.KK.GE.KMIN) OPAB = OPAB + OPA_KSHELL(LL,KK)
          
          OPAB = OPAB/CLF(LL)  ! now corrected => mean opacity

!          OPAC(LL,KK) = OPAB + OPATH + RAYLEIGH(LL,KK)/CLF(LL)
!          print*,ll,1.d8/freq,opab,rayleigh(ll,kk)

          OPAC(LL,KK) = OPAB + OPATH ! corrected => mean opacity

!---- Now, OPAC is changed below to its EFFECTIVE value 
          IF(OPTMET) THEN
!     opat_m_new etc corrected (mean opacities)
            
             OPAT_M_OLD(LL,KK) = OPAT_M_NEW(LL,KK) ! already corrected in OPACITM
             STRUE_M_OLD(LL,KK) = STRUE_M_NEW(LL,KK)
             OPAC(LL,KK) = OPAC(LL,KK) + OPAT_M_NEW(LL,KK) ! so far, mean 

!           if(mod(ll,6).eq.0.and..not.lteopt) &
!             print*,'old',kk,ll,1.d8/fre(kk), &
!&              (opat_m_new(ll,kk)-opat_m_nolines(ll,kk))/opac_nolines(ll,kk)
             OPAC_NOLINES(LL,KK) = OPAB + OPATH  + OPAT_M_NOLINES(LL,KK)
!---- Remains as mean (not effective) for next iteration 
!     NOTE: K-Shell absorption included in both (OPAC and OPAC_NO_LINES)

!           if(mod(ll,6).eq.0.and..not.lteopt) &
!             print*,'new',kk,ll,1.d8/fre(kk), &
!&              (opac(ll,kk)/opac_nolines(ll,kk)-1.)

!---- Inside or outside line-blocking range? 
             LAM = 1.D8/FRE(KK)
             IF (LAM.GT.W_BG_RED.OR.LAM.LT.W_BG_BLUE) THEN 
!JO test for transition effects (are all frequencies treated?)
!---- Just continuum bg opacity 
                TCL = OPAC_NOLINES(LL,KK) * TCL_FAC_CONT(LL) 
                OPA_EFF_RAT(LL,KK) = (1.+FIC(LL)*TCL)/(1.+TCL)
             ENDIF

          ELSE
!---- for models with only explicit elements (e.g., pure HHe), 
!---- OPAC consists of continuum only. THUS  
             TCL = OPAC(LL,KK) * TCL_FAC_CONT(LL) 
             OPA_EFF_RAT(LL,KK) = (1.+FIC(LL)*TCL)/(1.+TCL)
          ENDIF

          IF(.NOT.OPTTHICK) THEN
            IF(ABS(OPA_EFF_RAT(LL,KK)-1.D0).GT.EPSI) THEN
!---- this is a good test, since also in thin clumping opa_eff_grid is
!---- calculated in sumopal. Test proves appropriate averaging
               PRINT*,'opa',LL,KK,1.d8/freq,OPA_EFF_RAT(LL,KK)-1.D0
               STOP ' OPTICALLY THIN CLUMPING AND OPA_EFF_RAT NE 1 IN OPACITC'
            ENDIF
! FINALLY, SET TO UNITY
            OPA_EFF_RAT(LL,KK)=1.D0
          ENDIF  
          
          OPA_EFF_RAT_OLD(LL,KK) = OPA_EFF_RAT(LL,KK)

          IF (OPA_EFF_RAT(LL,KK).GT.EPSI1 .OR. OPA_EFF_RAT(LL,KK).LE.0.D0) THEN 
                PRINT*,OPA_EFF_RAT(LL,KK),LL,KK
                STOP ' OPA_EFF_RAT OUT OF BOUNDS IN OPACITC_1'             
          ENDIF
!---- This is the crucial correction!  
          OPAC(LL,KK) = OPAC(LL,KK) * OPA_EFF_RAT(LL,KK) 


          IF (OPTETA) THEN  

               IF(EHNUEKT.NE.0.D0) THEN
                 ETAB = EHNUEKT*HCFRE3* (ETABF+OPAFF*ETAFFC) !still uncorrected
               ELSE  
                 X=HKL*FREQ/TEMP(LL)
                 IF(X.LT.200.D0) STOP ' OPACITC: PROBLEM AT HIGH ENERGIES'
                 EXPO=LOG(HCFRE3 * (ETABF+OPAFF*ETAFFC))-X
                 ETAB=EXP(EXPO)  !still uncorrected
               ENDIF
               
               IF(TEMP(LL).LE.10000. .AND. 1.D8/FREQ.GT. 200.D0) &
&                  ETAB=ETAB+OPAHMINUS*HCFRE3/(1.D0/EHNUEKT-1.D0)  !still uncorrected

               ETAB = ETAB/CLF(LL) !now corrected => mean emissivity
               ETAB = ETAB * OPA_EFF_RAT(LL,KK) !now effective emissivity

!add x-ray emissivity
               IF(.NOT.LTEOPT.AND.OPTXRAY .AND. LL.LE.LXMIN) &
&                 ETAB = ETAB + CONSTX_LL*LAMBDANU(KK,LL) ! 2nd term corrected, since constx_ll includes fx

!for tests
!               IF(.NOT.LTEOPT.AND.OPTXRAY .AND. LL.LE.LXMIN) THEN
!                  IF(KK.GE.KMIN) THEN
!                  WRITE(*,5) LL,KK,OPA_KSHELL(LL,KK),CONSTX_LL,LAMBDANU(KK,LL)
!                  ELSE
!                  WRITE(*,5) LL,KK,ZERO,CONSTX_LL,LAMBDANU(KK,LL)
!                  ENDIF
!               ENDIF   
!5 FORMAT('XRCF',I3,2X,I5,2X,3(E10.4,2X))
                 
!---- we do NOT correct X-ray emission for optically thick effects
!     (assuming X-ray emission/absorption dispatched, see notes etc.)

               IF (.NOT.OPTMET) THEN
                 STRUE(LL,KK) = ETAB/OPAC(LL,KK) !correction cancels out anyway
               ELSE
!                 STRUE(LL,KK) = &
!&                (ETAB + OPAT_M_NEW(LL,KK)*STRUE_M_NEW(LL,KK)) / OPAC(LL,KK) 
!---- needed to make opat_m_new effective quantity here!
                 STRUE(LL,KK) = &
&                (ETAB + OPA_EFF_RAT(LL,KK)*OPAT_M_NEW(LL,KK)*STRUE_M_NEW(LL,KK)) / OPAC(LL,KK) 
               ENDIF

!              IF(LL.EQ.1.AND.OPTXRAY.AND..NOT.LTEOPT) THEN
!                PRINT*
!                IF(KK.GE.KMIN) THEN
!                  PRINT*,1.D8/FRE(KK),OPAC(LL,KK),OPA_KSHELL(LL,KK)
!                ELSE
!                  PRINT*,1.D8/FRE(KK),OPAC(LL,KK),'0.'
!                ENDIF
!                PRINT*,ETAB,CONSTX_LL*LAMBDANU(KK,LL),STRUE(LL,KK)
!                PRINT*,OPAT_M_NEW(KK,LL),STRUE_M_NEW(LL,KK)
!              ENDIF


!               print*,ll,kk,1.d8/fre(kk),strue(ll,kk)               
               IF (STRUE(LL,KK).EQ.0.) THEN
                 IF(1.D8/FREQ.GT.20.D0) STOP ' LAMBDA > 20 A and STRUE = 0!'  
               ENDIF

               IF (LTEOPT) THEN  

                    IF(EHNUEKT.NE.0.D0) THEN
                      BN = HCFRE3/ (1.D0/EHNUEKT-1.D0)  
                    ELSE  
                      X=HKL*FREQ/TEMP(LL)
                      EXPO=LOG(HCFRE3)-X
                      BN=EXP(EXPO)
                    ENDIF              
!---- since ETAB has been corrected, but not OPAB we need to adjust for that here
                    !BN1 = ETAB/OPAB  
                    BN1 = ETAB/(OPAB*OPA_EFF_RAT(LL,KK)) 
                    IF(1.D8/FREQ.GT.20.D0) THEN
                      IF (ABS(1.D0-BN1/BN).GT.PRECIS) THEN
                        PRINT*,'PROBLEMS IN LTE:',BN,BN1!!!
                        IF(BN.GT.1.D-100) STOP 'ERROR IN LTE!!!'
                      ENDIF
                    ENDIF
               END IF  
          END IF  

     END DO KLOOP

END DO DEPTHLOOP  

! for tests
!DO KK = 1,IFRE  
!          FREQ = 1.d8/FRE(KK)  
!          print*,freq,opac(54,kk),opac_nolines(54,kk)
!enddo


! finally, all opacities (and source-functions) have been corrected w.r.t. CLF
! i.e., are mean quantities, in particular OPAC_NOLINES,
! except for FFOPA (which remains uncorrected at all),
! and ETAB and OPAC(L,K), which have been corrected for macro-clumping. 

! SUMMARY: on output, OPAC is effective, whilst all other quantities remain
! as in the optically thin case.
RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CROSSBF(I,ALPHA,FREQ)  
! JO Sept 2021 new provisional treatment for formula 12
USE nlte_type  
USE nlte_dim  
USE fund_const
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&       ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS, &
& DATA,NDAT, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
& LABL4,LABL1,LABU1,LABL3,IPARE5, &
& TL,ANCDO,WLC,DL,FRECIN,FRECFN
USE nlte_var, ONLY : OPPATH,OP_FLAG,OP_DATA,IFRE,FRE, &
&       EOPDIFF,OP_FLAG2,MODNAM,FRECFN1

IMPLICIT NONE  
!
!------ this subroutine calculates photoionization cross-sections in the
!------ way described by 'detail'. the cross-section is return in
!------ 'alpha'
!
!------ used (and checked so far)
!
!------ formula 4: mg i, ii
!------ formula 5: h i, he i, he ii, si ii,iii,iv
!------ formula 6: he i
!------ formula 7: he i
!------ formula 2: he i
!------ formula 3: he i
!------ formala 12: si ii,iii,iv. attention: assuming here that only
!------             first three coefficients are different from zero!!!
!------             o ii, o iii: all coefficients might be different from zero
!------             provisional treatment for lambda < 100 A
!------ formula 13: o ii (rbf)
!------ formula 16: he i (rbf)
!------ formula 17: he i (rbf)
!------ formula 20: OPACITY PROJECT data, convolved with Gaussian of average
!                                   width E/Delta E = 100 
!------ formula 101: rbf data (new detail format), convolved with Gaussian
!                                   of average width E/Delta E = 100 
!
!------ formula 51: Seaton approximation 
!------ formula 21: Seaton approximation, followed by DR-data for the
!                   stabilizing transitions
!
!       hay que definir tambien las matrices de /atomdat/
!
!
!     .. scalar arguments ..

INTEGER(I4B), PARAMETER :: MMM = 1
REAL(DP), DIMENSION(MMM) :: ALPHAM, XNUM

REAL(DP) ::  ALPHA,FREQ  
INTEGER(I4B) ::  I  
!     ..
!     .. local scalars ..
REAL(DP) ::  ALP0,ALP1,ALP2,G,SUMM,X,XIP,XIQ,XLOG,XNU,Z,Z2,AW, &
&            ALP3,ALP4,ALP5,X100,ALPHA100  
INTEGER(I4B) ::  IFORM,II1,IN,INTEG,M,N,NATOMO,NC,NDATOS, &
&        NIVLL,NUMI,LL,NN,IL,IS,IZ
LOGICAL QCL,QDB,QRT  
CHARACTER KEY*6 
!
! for the use of the OP data
INTEGER(I4B) :: NDATOSMAX,NRECORD,ESTADO,J,K,L1,L2 
REAL(DP) :: FGS(2000),PICS(2000),XNAUX, RELDIFF, &
&           PEQ=0.D0, LOP1=0.D0, LOP2=0.D0 ! hawaii 

CHARACTER*(30) :: NAMEFILEOP
CHARACTER*(80) :: DUMMYCHAR
!     ..
!     .. local arrays ..
INTEGER(I4B) ::  IFPTR(ID_NTTRD),NFPTR(ID_NTTRD)  
CHARACTER LABL2(ID_RCBXT)*6,LABU2(ID_RCBXT)*6,PAREN2(ID_RCBXT)*6, &
&          PAREN3(ID_CBSFT)*6,PAREN4(ID_RBFTR)*6
!     ..
!     .. external functions ..
REAL(DP) ::  AGAU,GAUNT  
EXTERNAL AGAU,GAUNT  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP,INT,LOG  
!     ..
!     ..

N = IFORM4(I)  

!JO Sept. 2016: for N=20, frequencies down to ground-state ionization allowed
!NOTE: at first call, FRECFN1 is equal to FRECFN. After later calls
!(op_flag=.true.) FRECFN1 (with FRECFN LE FRECFN1 LE FRECIN) present  

IF(N.NE.20) THEN
  X = FRECIN(I)/FREQ  
ELSE
  X = FRECFN1(I)/FREQ
ENDIF

IF (X.GT.1.D0) STOP ' ERROR IN THRESHOLD '  

XNU = FREQ*CLIGHT  
II1 = INDAT4(I)  

IF (N.EQ.1) THEN  
     alp0=data(ii1)
     alpha=alp0*x**3
     return
!
ELSE IF (N.EQ.2) THEN  
     ALP0 = DATA(II1)  
     XIP = DATA(II1+1)  
     ALPHA = ALP0*X** (-XIP)  
     RETURN  

ELSE IF (N.EQ.3) THEN  
     ALP0 = DATA(II1)  
     XIP = DATA(II1+1)  
     ALP1 = DATA(II1+2)  
     XIQ = DATA(II1+3)  
     ALPHA = ALP0*X** (-XIP) + ALP1*X** (-XIQ)  
     RETURN  

ELSE IF (N.EQ.4) THEN  
     ALP0=DATA(II1)
     ALP1=DATA(II1+1)
     ALP2=DATA(II1+2)
     ALPHA=ALP0*(ALP1+(1.D0-ALP1)*X)*X**ALP2
     RETURN

ELSE IF (N.EQ.5) THEN  
     IN = INT(DATA(II1)+0.5D0)  
     NIVLL = LABL4(I)  
     Z = ZL(NIVLL) + 1.D0  
     Z2 = Z*Z  
     NATOMO = LE(NIVLL)  
     KEY = LABAT(NATOMO)  
     NUMI = LI(NIVLL)  
     IF (KEY.EQ.'H' .OR. KEY.EQ.'HE') THEN  
!
!          artemios gaunt faktoren
!
          G = AGAU(IN,XNU,KEY,NUMI)  
     ELSE  
!
!          mihalas gaunt faktoren
!
          G = GAUNT(IN,XNU/Z2)  
     END IF  
     ALPHA = 2.815D29*Z2*Z2/ ((IN**5)* (XNU**3))*G  
     RETURN  

ELSE IF (N.EQ.6) THEN  
     ALP0 = DATA(II1)  
     ALP1 = DATA(II1+1)  
     ALP2 = DATA(II1+2)  
     ALPHA = ALP0*EXP(ALP1+ALP2*XNU)  
     RETURN  

ELSE IF (N.EQ.7) THEN  
     ALP0 = DATA(II1)  
     ALP1 = DATA(II1+1)  
     ALP2 = DATA(II1+2)  
     ALPHA = EXP(ALP0+LOG(XNU)* (ALP1+ALP2*LOG(XNU)))  
     RETURN  

ELSE IF (N.EQ.12) THEN
! JO new provisional version from Sept. 2021
     ALP0 = DATA(II1)  
     ALP1 = DATA(II1+1)  
     ALP2 = DATA(II1+2)  
     ALP3=DATA(II1+3)
     ALP4=DATA(II1+4)
     ALP5=DATA(II1+5)
    
     XLOG = LOG(X)  
     IF(ALP3.EQ.0. .AND. ALP4.EQ.0. .AND. ALP5.EQ.0.) THEN
        SUMM = ALP0 + XLOG* (ALP1+XLOG*ALP2)  
     ELSE
        SUMM = ALP0 + XLOG*(ALP1+XLOG*(ALP2+XLOG*(ALP3+XLOG*(ALP4+XLOG*ALP5))))
     ENDIF
     ALPHA = EXP(SUMM)  

     IF(FREQ.LE.1.D6) RETURN !100 A  

     X100=FRECIN(I)/1.D6  
     XLOG = LOG(X100)  
     IF(ALP3.EQ.0. .AND. ALP4.EQ.0. .AND. ALP5.EQ.0.) THEN
        SUMM = ALP0 + XLOG* (ALP1+XLOG*ALP2)  
     ELSE
        SUMM = ALP0 + XLOG*(ALP1+XLOG*(ALP2+XLOG*(ALP3+XLOG*(ALP4+XLOG*ALP5))))
     ENDIF
     ALPHA100 = EXP(SUMM)  
     IF(ALPHA.GT.ALPHA100) THEN ! PRESUMABLY OUT OF VALIDITY RANGE
!        PRINT*,' INCREASING ALPHA: ',FREQ,I,ALPHA,ALPHA100
        ALPHA=ALPHA100*(1.D6/FREQ)**2
!        PRINT*,' new approximate value: ',ALPHA
     ENDIF
     RETURN  

ELSE IF (N.EQ.13) THEN		! not checked by J.Puls
     ALP0 = DATA(II1)
     ALP1 = DATA(II1+1)
     ALP2 = DATA(II1+2)
     ALP3 = DATA(II1+3)

     SUMM = ALP0*((ALP1+(1.D0-ALP1)*X)*X**ALP2*(1.D0-EXP(ALP3*(1.D0/X-1.D0))) &
&           + EXP(1.D0-1.D0/X))
     ALPHA = SUMM            
     RETURN

ELSE IF (N.EQ.16) THEN  
     NIVLL=LABL4(I)
     NATOMO=LE(NIVLL)
     AW=WEIGHT(NATOMO)
     NN=INT(DATA(II1)+0.5)
     LL=INT(DATA(II1+1)+0.5)
     Z=ZL(NIVLL)+1.D0
     XNUM(1)=XNU
     CALL PIX11(Z,AW,NN,LL,MMM,XNUM,ALPHAM)
     ALPHA=ALPHAM(1)
     RETURN

ELSE IF (N.EQ.17) THEN  
     NN=INT(DATA(II1)+0.5)
     IS=INT(DATA(II1+1)+0.5)
     IL=INT(DATA(II1+2)+0.5)
     CALL PIX21(IS,IL,NN,XNU,ALPHA)
     RETURN

ELSE IF (N.EQ.20) THEN ! OP data

!JO changed Sept. 2016  
	
     IF (.NOT. OP_FLAG(I)) THEN         ! this is only required once
        ALP0=INT(DATA(II1)+0.5d0)  	
        NDATOSMAX=INT(DATA(II1+1)+0.5D0)! maximum number of data points
        NRECORD=INT(DATA(II1+2)+0.5D0)	! record to be read
        NDATOS=INT(DATA(II1+3)+0.5D0)	! number of OP data points in this record
     	CALL OP_RBFSETUP(I,ALP0,NDATOSMAX,NRECORD,NDATOS,NAMEFILEOP)	
	OP_FLAG(I) = .TRUE.
        IF(FREQ.LT.FRECFN1(I)) THEN
! in first call(s) of CROSSBF (OP_FLAG=.FALSE.), FREQ can be located between
! FRECFN and FRECFN1, since FRECFN1 updated only in OP_RBFSETUP
          IF(FREQ.LT.FRECFN(I)) STOP ' ERROR IN FREQ (OP_RBFSETUP)'
          ALPHA = 0.
          RETURN
        ENDIF   
     ENDIF

! After the first time that is called, the smoothed cross sections
! are calculated by interpolation of the OP data stored in OP_DATA matrix, &
! one point per each of the frequencies in the (coarse) grid.
    
     DO K=2,IFRE
	IF (FREQ .LE. FRE(K)) GOTO 10
     ENDDO	
     STOP ' freq not found in fre(k) -- crossbf' 

!JO CHANGED SEPT 2016     
!10   PEQ = FREQ - FRECIN(I)
10   PEQ = FREQ - FRECFN1(I)

!    fre(k) > freq > fre(k-1)

     IF (ABS(PEQ) .LE. 1.D-5 .OR. (FRE(K)-FREQ) .LE. 1.D-5) THEN 
          ALPHA = OP_DATA(I,K) ! exactly at the edge OR freq. point
     ELSE
       LOP1=LOG10(FREQ/FRE(K))/LOG10(FRE(K-1)/FRE(K))
       LOP2=1.D0-LOP1
       ALPHA =  LOP1*LOG10(OP_DATA(I,K-1)) + LOP2*LOG10(OP_DATA(I,K)) 
       ALPHA =  10.D0**ALPHA
     ENDIF	

     IF (ALPHA .LE. 0.D0) THEN
         NRECORD=INT(DATA(II1+2)+0.5D0)	! record to be read
         PRINT*
         PRINT*,' ERROR in OP data file ',TRIM(NAMEFILEOP), &
&          ' cross-section ZERO OR NEGATIVE!!!!!',ALPHA
	 PRINT*,' Transition numer: ',I,' at frequency: ',FREQ
	 PRINT*,' OP data record: ',NRECORD,' corresponds to level: ',LABL(LABL4(I))
	 PRINT*," level's ionization edge : ",FRECFN(I)
	 PRINT*,' frequencies : ',FRE(K),' ',FRE(K+1),' ',OP_DATA(I,K),' ',OP_DATA(I,K+1)
	 PRINT*
	 DUMMYCHAR=trim(modnam)//trim('/crossbf-error-'//trim(labl(labl4(i)))//'.dat')
	 OPEN (77,FILE=DUMMYCHAR,STATUS='UNKNOWN')
	     DO K=1,IFRE
	        WRITE(77,*) FRE(K),OP_DATA(I,K)
	     ENDDO   
	 CLOSE(77)
	 STOP
     ENDIF     
   RETURN      

ELSE IF (N.EQ.21) THEN  
! first three data points for Seaton approx
     ALP0 = DATA(II1)  
     ALP1 = DATA(II1+1)  
     ALP2 = DATA(II1+2)  
     ALPHA = ALP0*(ALP1*X**ALP2 + (1.D0-ALP1)*X**(ALP2+1.D0))  
     IF (ALPHA.LT.0.) THEN
       IF (ALP1.GE.0.) STOP ' ALPHA NEGATIVE AND BETA POSITIV'
       ALPHA=1.D-40
     ENDIf  
     RETURN  

ELSE IF (N.EQ.33) THEN  
     PRINT *,' FORMULA NO. ',N,' NOT CHECKED YET'  
     STOP  
!          alpha0=data(ii1)*1.d-18
!          beta=data(ii1+1)
!          ese=data(ii1+2)
!          nivel=data(ii1+3)
!          nivll=labl4(i)
!          natomo=le(nivll)
!          key=labat(natomo)
!          numi=li(nivll)
!c          facgau=agau(nivel,xnu,key,numi)
!c          alpha0=alpha0*facgau
!          alpha=alpha0*(x**ese)*(beta+(1.d0-beta)*x)
!          return

ELSE IF (N.EQ.51) THEN  
     ALP0 = DATA(II1)  
     ALP1 = DATA(II1+1)  
     ALP2 = DATA(II1+2)  
     ALPHA = ALP0*(ALP1*X**ALP2 + (1.D0-ALP1)*X**(ALP2+1.D0))  
     IF (ALPHA.LT.0.) THEN
       IF (ALP1.GE.0.) STOP ' ALPHA NEGATIVE AND BETA POSITIV'
       ALPHA=1.D-40
     ENDIf  
     RETURN  

! comment JO Sept 2021: modify according to formula 20 
ELSE IF (N.EQ.101) THEN ! rbf cross-sections in new DETAIL format
     print*,' Cross sections according to FORMULA 101 need to be adapted/tested'
     print*,' whenever ionization to excited level occurs'
     print*,' (effective edge is ground-state edge due to dielectr. recomb.)'     
     print*,' Proceed in analogy to subr. OP_RBFSETUP' 
     STOP ' Cross sections according to FORMULA 101 need to be adapted/tested'
     NIVLL = LABL4(I)  
     NATOMO = LE(NIVLL)  
     KEY = LABAT(NATOMO)  
     Z = ZL(NIVLL) + 1.D0  
     IZ=INT(Z+0.5d0)
     IF (Z.NE.INT(DATA(II1))) STOP ' ERROR IN RBF CODING (FORMULA101)'
     NAMEFILEOP=TRIM(KEY)//ACHAR(48+IZ)//'PIC' ! assumes specific labelling of data-files
     NRECORD=INT(data(ii1+1)) !corresponds to label
     IF (.NOT. OP_FLAG(I)) THEN        ! this is only required once
     	CALL OP_RBFSETUP1(I,KEY,IZ,NRECORD,NAMEFILEOP)	
	OP_FLAG(I) = .TRUE.
     ENDIF

! After the first time that is called, the smoothed cross sections
! are calculated by interpolation of the OP data stored in OP_DATA matrix, &
! one point per each of the frequencies in the (coarse) grid.
    
     DO K=2,IFRE
	IF (FREQ .LE. FRE(K)) GOTO 15
     ENDDO	
     STOP ' freq not found in fre(k) -- crossbf' 

15   PEQ = FREQ - FRECIN(I)

!    fre(k) > freq > fre(k-1)

     IF (ABS(PEQ) .LE. 1.D-5 .OR. (FRE(K)-FREQ) .LE. 1.D-5) THEN 
          ALPHA = OP_DATA(I,K) ! exactly at the edge OR freq. point
     ELSE
       LOP1=LOG10(FREQ/FRE(K))/LOG10(FRE(K-1)/FRE(K))
       LOP2=1.D0-LOP1
       ALPHA =  LOP1*LOG10(OP_DATA(I,K-1)) + LOP2*LOG10(OP_DATA(I,K)) 
       ALPHA =  10.D0**ALPHA
     ENDIF	

     IF (ALPHA .LE. 0.D0) THEN
         PRINT*
         PRINT*,' ERROR in OP data file ',TRIM(NAMEFILEOP), &
&          ' cross-section ZERO OR NEGATIVE!!!!!',ALPHA
	 PRINT*,' Transition numer: ',I,' at frequency: ',FRECIN(I)
	 PRINT*,' OP data record: ',NRECORD,' corresponds to level: ',LABL(LABL4(I))
	 PRINT*," level's ionization edge : ",FRECIN(I)
	 PRINT*,' frequencies : ',FRE(K),' ',FRE(K+1),' ',OP_DATA(I,K),' ',OP_DATA(I,K+1)
	 PRINT*
	 DUMMYCHAR=trim(modnam)//trim('/crossbf-error-'//trim(labl(labl4(i)))//'.dat')
	 OPEN (77,FILE=DUMMYCHAR,STATUS='UNKNOWN')
	     DO K=1,IFRE
	        WRITE(77,*) FRE(K),OP_DATA(I,K)
	     ENDDO   
	 CLOSE(77)
	 STOP
     ENDIF     
     RETURN      

   
ELSE  
!          ihi=labl4(i)
!          call alphas(ihi,alpha,freq)
!          return
     PRINT *,' FORMULA NO. ',N,' NOT IMPLEMENTED YET'  

     STOP  
END IF  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE OP_CODEX(NUM,NAME,CHECKNAME,RYDBERG)
!
! It translates the codification of the file name for the OP photionization
! cross section data
! 
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
INTEGER(I4B) ::  NUM  
CHARACTER*30 ::  NAME,CHECKNAME
REAL(DP) :: RYDBERG
!     ..
!     .. local scalars ..
INTEGER(I4B) :: I,J,DIV  
!     ..
REAL(DP), DIMENSION(26) :: AUXRYDBERG
CHARACTER*6, DIMENSION(26) :: ATOMO
CHARACTER*5, DIMENSION(10) :: ESTADO
CHARACTER*20 :: AUXNAME
!
DATA ATOMO  /'H','He','X','X','X','C','N','O','F','Ne','Na','Mg', &
&            'Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V', &
             'Cr','Mn','Fe'/  
DATA ESTADO /'I','II','III','IV','V','VI','VII','VIII','IX','X'/  
DATA AUXRYDBERG /109677.62,109722.28,109728.64,109730.64,109731.75,109732.30,109733.02, &
&            109733.55,109734.15,109734.33,109734.70,109734.84, &
&            109735.08,109735.17,109735.37,109735.44,109735.62, &
&            109735.81,109735.78,109735.81,109735.98,109736.06, &
&            109736.13,109736.16,109736.22,109736.24/
!

DIV=1000
IF (NUM .LT. 1000) DIV=10
I=INT(NUM/DIV+0.5)
J=INT(NUM-I*DIV)

IF (J .GT. I) THEN
	PRINT *,'I: ',I,'  J: ',J
	STOP 'ERROR IN OP DATA CODEX'
ENDIF	
AUXNAME=TRIM(ATOMO(I))//TRIM(ESTADO(J))//'_rbf_dat'
NAME=TRIM(AUXNAME)
CHECKNAME=TRIM(ATOMO(I))//TRIM(ESTADO(J))//'_rbf_lis'
RYDBERG=AUXRYDBERG(I)
RETURN
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE PRMODEL(R,V,GRADV,RHO,XNE,XNH,TEMP,TAU,CLFAC,TEFF,N)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     druckt r in sternradien,v in einheiten von v-unendlich,
!     log rho, log ne, log nh, t-lucy/teff  und tau-thomson aus
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  TEFF  
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  GRADV(ND1),R(ND1),RHO(ND1),TAU(ND1),TEMP(ND1), &
&                 V(ND1),XNE(ND1),XNH(ND1),CLFAC(ND1)
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I  
!     ..
!     .. intrinsic functions ..
!INTRINSIC LOG10  
!     ..

WRITE (*,FMT=9000)  

DO I = 1,N  
     WRITE (*,FMT=9010) I,R(I),V(I),GRADV(I),LOG10(RHO(I)), &
      LOG10(XNH(I)),LOG10(XNE(I)),TEMP(I)/TEFF,TAU(I),CLFAC(I)
END DO  
!
!      do i=1,n
!      if(gradv(i).le.0.d0) then
!      print*,' negative vel. gradient -- density inversion! at n =',i
!      print*,' increase log g!!!!'
!      stop ' density inversion'
!      endif
!      END DO
!
!
RETURN  

 9000 FORMAT( &
& ' NO   RADIUS     VELOCITY   GRADIENT  DENSITY(AV) N-HYD(ENH) N-EL(ENH)   T/TEFF TAU-TH(MIC) ENH-FAC',/,/)
 9010 FORMAT (I3,2X,8 (F9.5,2X),F6.2)

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE TLUCY(R,TAUL,TAU,T,N,TMIN,TEFF,RTAU23,ITHYD)  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: OPT=>OPTLUCY, OPTMODEL, &
& SR,SRVMIN,SRNOM,CONSTT,NSDIV, &
!& IDUM,GGRAV,CKAPPA,XT,A2TE,DELSIG,DELMU,DQDTAU, &
! JO: fixed July 2015 (XT not needed here, overlap with XT from photstruc)
& IDUM,GGRAV,CKAPPA,A2TE,DELSIG,DELMU,DQDTAU, &
& QINF,Q0,GAM=>GAMHOPF  
USE photstruc

IMPLICIT NONE
!
!     berechnet die temperatur nach dem verfahren von
!     lucy bei vorgegebener dichte. (opt=.true.)
!
!     constt already with respect to rtau23 = srnom/sr
!
!     uses nlte hopf function (opt=.false.)
!
!     for optmodel=.true., input T-struct. is used
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  RTAU23,TEFF,TMIN  
INTEGER(I4B) ::  ITHYD,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  R(ND1),T(ND1),TAU(ND1),TAUL(ND1)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DTAUDR,Q,QH,QOLD,QPRIME,R23,RI,ROLD,RP,TAUOLD, &
&                 TAUP,TAUR,W,W1,X,DTDM,TMINKUR
INTEGER(I4B) ::  K,I  
LOGICAL OPTFOUND  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP,SQRT  
!     ..

OPTFOUND = .FALSE.  

IF (OPT) THEN

     TAUOLD = 0.D0  
     IF (.NOT.OPTMODEL .OR. ITHYD .EQ. 1) THEN

     W = .5D0* (1.D0-SQRT(1.D0- (RTAU23/R(1))**2))  
     T(1) = TEFF* (W+CONSTT)**.25D0  

     DO K = 1,N - 1  
          IF (T(K).LT.TMIN) T(K) = TMIN  
          W1 = 1.D0 - (RTAU23/R(K+1))**2  
          IF (W1.LT.0.D0) W1 = 0.D0  
          W = .5D0* (1.D0-SQRT(W1))  
          TAUP = TAUL(K+1)  

          IF (TAUP.GE..66D0 .AND. TAUOLD.LT..66D0) THEN  
               OPTFOUND = .TRUE.  
               ROLD = R(K)  
               RP = R(K+1)  
               DTAUDR = (TAUP-TAUOLD)/ (RP-ROLD)  
               R23 = ROLD + (.66D0-TAUOLD)/DTAUDR  
               IF (R23.GT.ROLD .OR. R23.LT.RP) STOP ' ERROR IN R(TAU-LUCY=2/3)'
               SRVMIN = SRNOM*R(NSDIV)/R23  
               PRINT *  
               PRINT *,' MESSAGE FROM TLUCY: SRVMIN/SRNOM = ',SRVMIN/SRNOM
               PRINT *  
          END IF  
          TAUOLD = TAUP  
          T(K+1) = TEFF* (W+.75D0*TAUP+CONSTT)**.25D0  
    END DO

    ELSE
! interpolation of input structure with respect to m
    T(1) = TMIN
    TMINKUR=0.
    
    DO K = 1,N - 1  

          TAUP = TAUL(K+1)  

          IF (TAUP.GE..66D0 .AND. TAUOLD.LT..66D0) THEN  
               OPTFOUND = .TRUE.  
               ROLD = R(K)  
               RP = R(K+1)  
               DTAUDR = (TAUP-TAUOLD)/ (RP-ROLD)  
               R23 = ROLD + (.66D0-TAUOLD)/DTAUDR  
               IF (R23.GT.ROLD .OR. R23.LT.RP) STOP ' ERROR IN R(TAU-LUCY=2/3)'
               SRVMIN = SRNOM*R(NSDIV)/R23  
               PRINT *  
               PRINT *,' MESSAGE FROM TLUCY: SRVMIN/SRNOM = ',SRVMIN/SRNOM
               PRINT *  
          END IF  
          TAUOLD = TAUP  

          X=XM(K+1)

          IF(X.EQ.0.) THEN
            T(K+1) = TMIN  
          ELSE  

            DO I=1,NDMOD-1
             IF (X .GT. XMEXT(I).AND. X .LT. XMEXT(I+1)) EXIT
            ENDDO
            IF (I .EQ. NDMOD) THEN
              IF(X.LT.XMEXT(1)) THEN
                I=1
              ELSE
                I=NDMOD-1
!                STOP ' xm not found in external phot. struct. (tlucy)'
              ENDIF
            ENDIF

            DTDM=LOG10(TEXT(I+1)/TEXT(I))/LOG10(XMEXT(I+1)/XMEXT(I))
            T(K+1) = LOG10(TEXT(I))+DTDM*LOG10(X/XMEXT(I))           
            T(K+1) = 10.**T(K+1)
!            IF (TMINKUR.EQ.0.) TMINKUR=T(K+1)
            IF (TMINKUR.EQ.0.) TMINKUR=AMAX1(TMIN,T(K+1))
          ENDIF
    END DO

    WHERE (T.EQ.TMIN) T=TMINKUR
      TMIN=TMINKUR
! until here, we have only the othermost points set to max(tmin_input,tmin(kur)
    WHERE (T.LT.TMIN) T=TMIN
! the latter statement ensures to have all temperatures above tmin7tminkur
! in case the t-struct. is not montonic, this feature will be removed
! comment out the line if you want to preserve this feature 
    ENDIF

    IF (ITHYD.GE.2 .AND. .NOT.OPTFOUND) STOP ' TAU-LUCY = 2/3 NOT FOUND'
    
RETURN  

END IF  
!
!---  preliminary version at tau = 0.
!---  should not matter since t(l) anyway tmin
!
RI = R(1)/RTAU23  
QH = Q0/ (RI**2)  
T(1) = TEFF* (.75D0*QH)**.25D0  
TAUOLD = 0.D0  
QOLD = 0.D0  
DQDTAU(1) = 0.D0  

DO K = 1,N - 1  
     IF (T(K).LT.TMIN) T(K) = TMIN  
     TAUP = TAUL(K+1)  
     TAUR = TAU(K+1)  

     IF (TAUP.GE..66D0 .AND. TAUOLD.LT..66D0) THEN  
          OPTFOUND = .TRUE.  
          ROLD = R(K)  
          RP = R(K+1)  
          DTAUDR = (TAUP-TAUOLD)/ (RP-ROLD)  
          R23 = ROLD + (.66D0-TAUOLD)/DTAUDR  
          IF (R23.GT.ROLD .OR. R23.LT.RP) &
&          STOP ' ERROR IN R(TAU-LUCY=2/3)'
          SRVMIN = SRNOM*R(NSDIV)/R23  
          PRINT *  
          PRINT *,' MESSAGE FROM TLUCY: SRVMIN/SRNOM = ', SRVMIN/ SRNOM
          PRINT *  
     END IF  

     Q = QINF - (QINF-Q0)*EXP(-TAUR*GAM)  
     QPRIME = TAUP/TAUR*Q  
     QH = TAUP/TAUR* (Q+TAUR)  
     DQDTAU(K+1) = (QOLD-QPRIME)/ (TAUOLD-TAUP)  
     T(K+1) = TEFF* (.75D0*QH)**.25D0  
     TAUOLD = TAUP  
     QOLD = QPRIME  
END DO  

IF (ITHYD.GE.2 .AND. .NOT.OPTFOUND) STOP ' TAU-LUCY = 2/3 NOT FOUND'

RETURN

END
!
!***********************************************************************
!
! subroutines: complex ones
! model and related
!
!***********************************************************************
!
SUBROUTINE MODEL(TAU,VMIN1,OPTOUT,ITHYD,XKAP,EX,TACT,XMMAX,RTAU23, &
&                 XNH,XNE,VDIV,NDIV)

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: ICONVER, MODNAM, OPTMODEL

IMPLICIT NONE
!
!
!     calculates atmospheric structure for analytical velocity laws
!     and approx. photospheric structure
!
!
!     analytical velocity law
!     calculated values of ne, nh, tau, t  assume pure h/he atmosphere
!
!          v(r) = vmax * (1.-b/r)**beta   ( r > srnom, i.e,
!                 v > vdiv * vsound)
!          v(r) = vmin * exp((r-sr/hl)    ( r < srnom )
!
!     clumping is now accounted for according to parameterization
!     provided by input parameters (clf_field) and parameterization
!     given in subroutine clumping
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!
!     input
!     local input files: indat
!
!     indat:
!     teff, log g, rstar/rsun(nominal value)
!     rmax,tmin
!     mloss,v0,vmax,beta,vdiv
!     hei
!     optmod
!     **
!     rmax.....in stellar radii
!     tmin.....minimum temperature for lucy's temp.stratification
!     mloss....mass loss in m-sun/year
!     v0,vmax..in km/s
!     optmod...if optmod=.true., Kurucz/detail structure will be used
!              to create structure (implies optmodel=.true.)
!
!********************************************************************
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  EX,RTAU23,VMIN1,XKAP,XMMAX,VDIV  
INTEGER(I4B) ::  ITHYD,NDIV  
LOGICAL OPTOUT  
!     ..
!     .. array arguments ..
REAL(DP) ::  TACT(ND),TAU(ND),XNE(ND),XNH(ND)  
!     ..
!     .. local scalars ..
REAL(DP) ::  BETA,HEI,OPTMIXED,RMAX,RSTARNOM,TEFF,TMIN,VDIVOLD, &
&                 VMAX,VMIN,XLOGG,XMLOSS,YHEIN
INTEGER(I4B) ::  ND1,ND2,ND3,ND4,NNN  
LOGICAL OPTMOD  
!     ..
!     .. local arrays ..
REAL(DP) ::  GRADV(ND),R(ND),R1(ND),RHO(ND),RHO1(ND),TD(ND), &
&            TEMP(ND),TEMP1(ND),V(ND),V1(ND),WNE(ND),CLFAC(ND), &
&            XNE1(ND), XNH1(ND)
!
!     ..
!     .. external subroutines ..
!
CHARACTER DUMCHAR*30
EXTERNAL MODVL,PRMODEL  
!     ..

IF(ND.LT.47) STOP ' ND TOO LOW'

OPEN (1,FILE=TRIM(MODNAM)//'/INDAT.DAT',STATUS='OLD')  
REWIND 1  
READ (1,FMT=*) DUMCHAR  
READ (1,FMT=*) OPTMOD,OPTMOD,NNN,NNN  
READ (1,FMT=*) OPTMIXED  
READ (1,FMT=*) TEFF,XLOGG,RSTARNOM  
READ (1,FMT=*) RMAX,TMIN  
READ (1,FMT=*) XMLOSS,VMIN,VMAX,BETA,VDIVOLD  
READ (1,FMT=*) YHEIN,HEI  
READ (1,FMT=*) OPTMOD
CLOSE (1)  

IF(OPTMOD.AND..NOT.OPTMODEL) STOP ' ERROR: OPTMOD.AND..NOT.OPTMODEL'

IF (TMIN.LT.2.D0) TMIN = TMIN*TEFF  
IF (VMIN1.NE.0.D0) VMIN = VMIN1  

ND1 = 10  
ND2 = 13  
ND3 = 16  !inclusive one extra point at .98 Rmax
          !(one less than in versions before 8.6.1)
!ND1 = 10  
!ND2 = 10  
!ND3 = 10  

! save 7 points for N5 + one extra point close to R=1)

IF(ND .LT. 51) THEN
  ND4 = (ND-(ND1+ND2+ND3+7)) ! old version, no particular resolution of transonic region
ELSE
! save additonal 4 points for transonic region (if vdiv not too large)
  ND4 = (ND-(ND1+ND2+ND3+11)) 
ENDIF

IF(ND4.LT.0) STOP ' ERROR IN ND4'

CALL MODVL(TEFF,XLOGG,RSTARNOM,RMAX,TMIN,XMLOSS,VMIN,VMAX,BETA, &
&           VDIV,YHEIN,HEI,OPTMOD,ND,ND1,ND2,ND3,ND4,R,V,GRADV,RHO, &
&           XNE,XNH,TEMP,TAU,TD,WNE,R1,V1,RHO1,XNE1,XNH1,TEMP1, &
&           ITHYD,XKAP,EX,TACT,XMMAX,RTAU23,NDIV,CLFAC)

!REMEMBER: TAU=TAU-TH ONLY IN MICRO-CLUMPING APPROX.
IF (OPTOUT .AND. ICONVER.EQ.1) CALL PRMODEL(R,V,GRADV,RHO,XNE,XNH, &
& TEMP,TAU,CLFAC,TEFF,ND)

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE MODVL(TEFF,XLOGG,RSTARNOM,RMAX,TMIN,XMLOSS,VMIN,VMAX, &
&                 BETA,VDIV,YHEIN,HEI,OPTMOD,ND,ND1,ND2,ND3,ND4,R, &
&                 V,GRADV,RHO,XNE,XNH,TEMP,TAU,TD,WNE,R1,V1,RHO1, &
&                 XNE1,XNH1,TEMP1,ITHYD,XKAP,EX,TACT,XMMAX,RTAU23,NS,CLFAC)

USE nlte_var, ONLY: ICONVER
USE nlte_type
USE nlte_dim
USE fund_const, ONLY: XMH=>AMH, AKB
USE nlte_opt, ONLY: OPT_PHOTCLUMP

USE princesa_var, ONLY: NL,NLX,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&    ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS  

USE nlte_var, ONLY: MODNAM,SR,SRVMIN,SRNOM,CONSTT,NSDIV,H1, &
& IDUM,ITMIN,GGRAV,TE=>TEFF, &
& SIGEM,CKAPPA,XT1=>XT,A2TE,DELSIG,DELMU,DQDTAU,DELPARA

USE nlte_porvor, ONLY: OPTTHICK

USE photstruc  ! includes XM

IMPLICIT NONE
!
!     calculation of atmospheric structures for velocity law
!     as described in Santolaya-Rey et al., 
!     accounting now for clumping (see below)
!
!     possibility to use Kurucz/Detail or Tlusty structure for photosphere
!
!     output: first iteration: ne, temp1=temp as start values
!             other iterations:    temp1=(tact,tnew),
!                                   connected at sonic point
!     input : in hydro iteration cycle: ne, tact
!
!     in hydro: tnew, p (from diff eq., with approximated opacity)
!               on output p is recalculated from tact to be
!               intrinsically consistent
!
!     after convergence, tnew has to be a good approximation of tact
!     this is checked in routine lte by "check"
!
!
!     !!!! recheck carefully, if more than h/he present!!!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: NNDE=ID_NDEPT

REAL(DP), PARAMETER :: RSU=6.96D10,XMSU=1.989D33  
REAL(DP), PARAMETER :: SIGMAE=6.65D-25  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  BETA,EX,HEI,RMAX,RSTARNOM,RTAU23,TEFF,TMIN,VDIV, &
&             VMAX,VMIN,XKAP,XLOGG,XMLOSS,XMMAX,YHEIN

INTEGER(I4B) ::  ITHYD,ND,ND1,ND2,ND3,ND4,NS  
LOGICAL OPTMOD, FOUND, FIRST
LOGICAL :: UPDATED=.FALSE.
!     ..
!     .. array arguments ..
REAL(DP) ::  GRADV(NNDE),R(NNDE),R1(NNDE),RHO(NNDE), &
&                 RHO1(NNDE),TACT(ND),TAU(NNDE),TD(NNDE), &
&                 TEMP(NNDE),TEMP1(NNDE),V(NNDE),V1(NNDE), &
&                 WNE(NNDE),XNE(NNDE),XNE1(NNDE),XNH(NNDE), &
&                 XNH1(NNDE),CLFAC(NNDE)
!     ..
!     .. local scalars ..
REAL(DP) ::  A2,AUXI,B,C1,C2,DEL,DEL1,DELTAM,DELTAM1,DELTAR, &
&                 DIFFV,DIVR,DIVRHO,DM,DPX,EPS1,GAMMA,H2,H3,HL,HS, &
&                 P0,PI,R0,RC,RDIV2,RHO0,RHOPHO,RNEW1,RPAR,RPER, &
&                 RRR,RVMAX,SIGE,SRCONST,SUMM,SUMATOM,SUMMASS,T0, &
&                 TAUMAX,TAUMIN,VDIV2,VDST,VS2,VSONIC,X1,X2,XI,XM0, &
&                 XMU,XMU1,YHE,TERM,GRAD, &
&                 PL,TL,RL,DMDR,DUMMY,RHOL, &
&                 VNS,RNS,SAFETY,INTEG,TAU_E, &
&                 RR1, RR2, DET, V1TEST, V2TEST, RR0, VSMOOTH, RHOSMOOTH, VNSOLD

INTEGER(I4B) ::  I,II,ISI,IVS2,J,K,L,LL,M,N5,N6,NBAD,NENE,NOK,VINV,ISTART,IEND,NDK,NR
!     ..
!     .. local arrays ..
REAL(DP) ::  ARHO(ID_NDEPT),BRHO(ID_NDEPT),CRHO(ID_NDEPT), &
&            P(ID_NDEPT),TNEW(ID_NDEPT),Y(3), &
&            RHOCL(ID_NDEPT), XNECL(ID_NDEPT),RADACC(ID_NDEPT), &
&            ZZ1(4),ZZ2(4),VMAT(4,4),YY(4),GRADVNEW(ID_NDEPT)

INTEGER(I4B) ::  INDEX(ID_NDEPT), INDLUD(4)  
!     ..
!     .. external functions ..
REAL(DP) ::  RHOFUNC,VFVL  
EXTERNAL RHOFUNC,VFVL  
!     ..
!     .. external subroutines ..
EXTERNAL DERIVS,DERIVS_KUR,DERIVS_TLU,ODEINT,RKQC,SPLINE,TLUCY,VINDEX  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,ACOS,DBLE,EXP,INT,LOG,LOG10,SQRT  
!     ..
!     .. data statements ..
!
!
DATA EPS1/.5D0/  
DATA FIRST/.TRUE./

!     in order to account for clumping, only the last part of this
!     routine is affected, 
!     a) since tau-scale of first part refers to
!        tau_e alone (which remains unaffected -- clf cancels out)
!
!        NOTE: THIS IS NO LONGER THE CASE, SINCE POSSIBILITY FOR POROSITY 
!        in e-scattering. Thus this scale might be modified accordingly.  
!        Not considered yet, since it is only a start-model -- and SO FAR
!        it has not been any problems at all.
!        Note by JO: tau_e only used as reference scale for setting up the grid
!        MIGHT NEED TO BE UPDATED IN FUTURE.
!
!     b) since photosphere is not affected by clumping
!
!---- variable optstw incorporated (e. santolaya, 6-11-94) to allow the
!---- input of a given atmosphere model
!

NENE = 0  

DO ISI = 1,NAT  
     IF (LABAT(ISI).EQ.'HE') NENE = ISI  
END DO  

IF (NENE.EQ.0) THEN  
     PRINT *,'HE NOT FOUND - MODVL13'  
     IF (HEI.NE.0.D0) THEN  
          PRINT *,' NO HELIUM BUT ELSCAT-CONTRIBUTION!!!'  
          PRINT *, ' EITHER SET HEI TO ZERO OR CALCULATE WITH HELIUM!!!'
! comment the following line out for specific tests
          STOP ' INCONSISTENT TREATMENT OF HELIUM'  
     END IF  
     YHE = YHEIN  
ELSE  
     YHE = ABUND(NENE)  
     IF (ABS(1.D0-YHE/YHEIN).GT.1.D-5) STOP ' ERROR IN YHE - MODVL13'
END IF  


WRITE (*,FMT=9000) TEFF,XLOGG,RSTARNOM,YHE

TAUMAX = 20.D0  
TE = TEFF  
GGRAV = 10.D0**XLOGG  
VDST = SQRT(2.D0*AKB*TEFF/XMH)  

SRNOM = RSTARNOM*RSU  
VMAX = VMAX*1.D5  
VMIN = VMIN*1.D5  
H1 = XMLOSS*XMSU/3.9657D8  

WRITE (*,FMT=9010) XMLOSS,VMAX/1.D5,VMIN/1.D5  

SUMATOM = 0.D0  
SUMMASS = 0.D0  

DO K = 1,NAT  
     SUMATOM = SUMATOM + ABUND(K)  
     SUMMASS = SUMMASS + WEIGHT(K)*ABUND(K)  
END DO  

IF (NENE.EQ.0 .AND. YHE.NE.0.) THEN  
     SUMATOM = SUMATOM + YHE  
     SUMMASS = SUMMASS + 4.D0*YHE  
END IF  

SUMMASS = SUMMASS*XMH  
C1 = SUMMASS  
!
!---- for sigem, only contribution of helium and hydrogen
!
C2 = (1.D0+HEI*YHE)/C1  
SIGEM = SIGMAE*C2  

TAUMIN = H1*SIGEM/ (VMAX*RMAX*SRNOM)  
CONSTT = TAUMIN/ (4.D0*RMAX**2)  


IF (.NOT. OPTMOD) THEN  

   MODTYPE='FASTWD'

ELSE !(i.e., if optmod)
! IF THIS FEATURE IS ENABLED, TAKE CARE CONCERNING CORRECT TREATMENT OF CLUMPING!!!
! to be considered later; so far assumed that no photospheric clumping 

!------- the model is read from structure file
   
   IF (FIRST) THEN
   OPEN (4,FILE=TRIM(MODNAM)//'.struct',STATUS='OLD',FORM='FORMATTED',ERR=5)  
   MODTYPE='KURUCZ'
   PRINT*
   PRINT*,' ** STRUCTURE = KURUCZ **'
   GOTO 6

5  OPEN (4,FILE=TRIM(MODNAM)//'.tlusty',STATUS='OLD',FORM='FORMATTED',ERR=7)  
   MODTYPE='TLUSTY' 
   PRINT*
   PRINT*,' ** STRUCTURE = TLUSTY **'
   GOTO 6 


7  OPEN (4,FILE=TRIM(MODNAM)//'.michel',STATUS='OLD',FORM='FORMATTED',ERR=9)  
   MODTYPE='MICHEL' 
   PRINT*
   PRINT*,' ** STRUCTURE = MICHEL **'
   GOTO 6 
9  STOP ' Model structure (*.struct/*.tlusty/*.michel) not present - please create!'

6  CONTINUE

   IF (MODTYPE.EQ.'KURUCZ') THEN
      READ(4,*)
      READ(4,*) NDMOD
      ALLOCATE(XMEXT(NDMOD),TEXT(NDMOD),PKUR(NDMOD),XNEKUR(NDMOD),GRADKUR(NDMOD))
      DO I=1,NDMOD
        READ(4,*) XMEXT(I),TEXT(I),PKUR(I),XNEKUR(I),GRADKUR(I)
      ENDDO

   ELSE IF (MODTYPE.EQ.'TLUSTY') THEN  !also for cmfgen input
      READ(4,*) NDMOD
      ALLOCATE(XMEXT(NDMOD),TEXT(NDMOD),RHOTLU(NDMOD),XNETLU(NDMOD))
      DO I=1,NDMOD
        READ(4,*) II, XMEXT(I),DUMMY,DUMMY,TEXT(I),XNETLU(I),RHOTLU(I)
      ENDDO
      IF (II.NE.NDMOD) STOP ' WRONG INPUT FILE - CHECK!' 

   ELSE IF (MODTYPE.EQ.'MICHEL') THEN
      READ(4,*) NDMOD
      ALLOCATE(RMICH(NDMOD),RMICH1(NDMOD),VMICH(NDMOD),RHOMICH(NDMOD))
      DO I=1,NDMOD
        READ(4,*) RMICH(I),DUMMY,VMICH(I),DUMMY,DUMMY,RHOMICH(I)
        VMICH(I)=VMICH(I)*1.D5 
!        PRINT*,I,RMICH(I),VMICH(I),RHOMICH(I)
! consistency check for Mdot/(4.*Pi)
        DUMMY=(SRNOM*RMICH(I))**2*VMICH(I)*RHOMICH(I)
        PRINT*,DUMMY,H1
! if this line is commented out, one can use mdots inconsistent
! with the density stratification
        IF (ABS(1.-DUMMY/H1).GT.0.01) STOP ' DENSITY(MICHEL) AND MDOT(INPUT) INCONSISTENT'
      ENDDO
! consistency check for VINF
      IF (ABS(1.-VMICH(NDMOD)/VMAX).GT.0.01) STOP ' VMAX(MICHEL) AND VINF(INPUT) INCONSISTENT'      
      VMAX=VMICH(NDMOD)


   ELSE   
      STOP ' WRONG MODEL-TYPE'
   ENDIF

   CLOSE(4)

   FIRST=.FALSE.
   ENDIF  

ENDIF
!
!------- the model is calculated
!
XMU = (1.D0+4.D0*YHE)/ (2.D0+YHE* (1.D0+HEI))  

IF (ITHYD.NE.1) THEN  
     DO L = 1,ND  
! clumping cancels out
          XMU1 = SUMMASS/ (SUMATOM+XNE(L)/XNH(L))/XMH  
!
!---  delmu defined in order to correct a2te (vs(iso)**2/t)
!---  for ionisation effects
!
          DELMU(L) = XMU/XMU1  
     END DO
END IF  

GAMMA = SIGEM*1.8913D-15*TE**4/GGRAV  

IF (GAMMA.GE.1.D0) STOP ' GAMMA .GE. 1'  
!---- DO NOT ALLOW POROSITY-MEDIATED MASS LOSS FROM PHOTOSPHERE YET 
!     thus keep this condition 

HL = AKB*TE/XMU/XMH/ (GGRAV* (1.D0-GAMMA))  

PRINT *  
PRINT *,' GAMMA = ',GAMMA  
PRINT *,' PRESS. SCALE HEIGHT(TEFF) = ',HL/SRNOM,' (IN NOM. RADII)'
PRINT *  
!
!      a=(log(vmax/vmin)+sr/hl)/beta
!-----approximate formula
!      yy=a-log(a)
!      rc=beta*hl*yy
!      r0=rc/(1.d0/yy+1.d0)
!
!-----division point at vdiv*vsound corresponding to srnom
!
VSONIC = SQRT(AKB*TE/XMH/XMU)  
VS2 = VDIV*VSONIC  
! check for too large Mdot which cannot handled by FASTWIND
B = 1.D0 - (VSONIC/VMAX)** (1.D0/BETA)  
!
! integral is int_1^inf dr/[r^2*(1-b/r)^beta]
!
IF (BETA.NE.1.D0) THEN
  INTEG=(1.-B)**(-BETA)*(B-1.+(1.-B)**BETA)
  INTEG=INTEG/(B*(1.-BETA))
ELSE
  INTEG=-LOG(1.-B)/B
ENDIF
IF(INTEG.LT.0.) STOP ' ERROR IN INTEG (MODVL)'
! TAU_E is optical depth in e-scat (micro) at sonic point
TAU_E=SIGEM*H1/(VMAX*SRNOM)*INTEG
PRINT*,' TAU_E (MICRO) AT VSOUND = ',TAU_E
PRINT*
IF(TAU_E.GT.2.) THEN
  PRINT*,' MASS LOSS TOO LARGE OR VINF TOO LOW TO BE HANDLED BY FASTWIND!!!'
  PRINT*,' REDUCE MDOT AT LEAST TO ',XMLOSS*2./TAU_E
  PRINT*,' TO BE ON THE SAFE SIDE, REDUCE AT LEAST TO ',XMLOSS/TAU_E
  PRINT* 
  PRINT*,' --------------------'
  PRINT*,' NO MODEL CALCULATED'  
  PRINT*,' --------------------'
  STOP
ENDIF  
  
IF (VS2.LT.VMIN) STOP ' VDIV < VMIN'  

IF (ITHYD.EQ.1) THEN  
     RC = SRNOM  
ELSE  
     RC = SRVMIN  
END IF  

SR = RC - HL*LOG(VS2/VMIN)  
R0 = RC* (1.D0- (VS2/VMAX)** (1.D0/BETA))  
!here and in the following R0 corrsponds to b, and RC -> R(NS) at VDIV
!in the 2nd part of this routine (photosphere), R0 corresponds to R(NS) at VDIV

PRINT *,' LOWER (SCALING RADIUS) AT ',SR/SRNOM,' NOM. RADII'  
PRINT *,' SRVMIN = ',RC/SRNOM,' NOM. RADII'  


RHOPHO = H1/SR**2/VMIN  
!
!     aufstellen des r-rasters ueber nd1 stuetzstellen,die
!     bezueglich log tau aequidistant verteilt sind
!
R(1) = RMAX*SRNOM  
V(1) = VFVL(R(1),RC,R0,SR,VMAX,VMIN,BETA,HL)  
RHO(1) = RHOFUNC(R(1),V(1))  

R(ND1) = SR  
V(ND1) = VMIN  
RHO(ND1) = RHOPHO  

   40 CONTINUE  

DEL1 = LOG10(TAUMAX/TAUMIN)/DBLE(ND1-1)  
H2 = 1.D0 - 10.D0** (-DEL1)  

DO K = 2,ND1 - 1  
     SIGE = SIGEM*RHO(K-1)  
     R(K) = R(K-1) - TAUMIN*10.D0** (DEL1* (K-1))*H2/SIGE  

     IF (R(K).LT.SR) THEN  
          TAUMAX = 4.D0*TAUMAX/5.D0  
          GO TO 40  
     END IF  

     V(K) = VFVL(R(K),RC,R0,SR,VMAX,VMIN,BETA,HL)  
     RHO(K) = RHOFUNC(R(K),V(K))  
END DO  
!
!     einschieben von nd2 punkten,so dass v "gleichmaessig
!     ueber das r-raster verteilt wird.
!
DO J = 0,ND2 - 1  
     DIFFV = (V(1) - V(2))  
     I = 2  

     DO K = 3,ND1 + J  
          H3 = (V(K-1) - V(K)) 
          IF (H3.GT.DIFFV) THEN  
               DIFFV = H3  
               I = K  
          END IF  
     END DO

     M = ND1 + J - I  
     CALL VINDEX(R,V,RHO,M,I)  
     R(I) = .5D0* (R(I)+R(I-1))  
     V(I) = VFVL(R(I),RC,R0,SR,VMAX,VMIN,BETA,HL)  
     RHO(I) = RHOFUNC(R(I),V(I))  
END DO  
!
!     einschieben von nd3-1 punkten,sodass rho "gleichmaessig"
!     ueber das r-gitter verteilt wird
!
DO J = 0,ND3 - 2  
     DIVRHO = RHO(2)/RHO(1)  
     I = 2  

     DO K = 3,ND1 + ND2 + J  
          H3 = RHO(K)/RHO(K-1)  
          IF (H3.GT.DIVRHO) THEN  
               DIVRHO = H3  
               I = K  
          END IF  
     END DO

     M = ND1 + ND2 + J - I  
     CALL VINDEX(R,V,RHO,M,I)  
     R(I) = .5D0* (R(I)+R(I-1))  
     V(I) = VFVL(R(I),RC,R0,SR,VMAX,VMIN,BETA,HL)  
     RHO(I) = RHOFUNC(R(I),V(I))  
END DO  
!
!---- einschieben von einem punkt bei .98*rmax
!
I = 2  
M = ND1 + ND2 + (ND3-1) - I  
CALL VINDEX(R,V,RHO,M,I)  
R(I) = .98D0*R(1)  

IF (R(3).GT.R(I)) STOP ' R(3) > R(2)'  

V(I) = VFVL(R(I),RC,R0,SR,VMAX,VMIN,BETA,HL)  
RHO(I) = RHOFUNC(R(I),V(I))  
!
!      ND4 points exclusively for the photosphere
!
IF(ND4.GT.0) THEN
RPAR=R(ND1+ND2+ND3-1)
RPER=R(ND1+ND2+ND3)
DELTAR = (RPAR-RPER)/DBLE(ND4+1)  
DO II = 1,ND4  
     RNEW1 = RPAR - DBLE(II)*DELTAR  

     DO J = 1,ND1 + ND2 + ND3 + II  
          IF (R(J).GT.RNEW1) K = J + 1  
     END DO

     M = ND1 + ND2 + ND3 + II - K  
     CALL VINDEX(R,V,RHO,M,K)  
     R(K) = RNEW1  
     V(K) = VFVL(R(K),RC,R0,SR,VMAX,VMIN,BETA,HL)  
     RHO(K) = RHOFUNC(R(K),V(K))  
END DO
ENDIF
!
!---- addition of points at photospheric division
!-old:(between 2.*vs2 and .2*vs2)
! changed March 2016; form 
!-----(between 2.*vs2 and .3*vs2)
!
IF (3.*VSONIC .GT. VMAX) STOP ' 3 VSOUND > VMAX! CHANGE MODVL!'

IF (ND.LT.51 .OR. (3.*VSONIC-2.*VS2).lt.VSONIC) THEN
  N6=0
  N5 = ND - (ND1+ND2+ND3+ND4+1) ! old version or large vdiv
  IF (ND.LT.51) THEN
    IF(N5 .NE. 6) STOP ' ERROR IN N5'
  ELSE
    IF(N5 .NE. 10) STOP ' ERROR IN N5'
  ENDIF
ELSE
  N6=4
  N5 = ND - (ND1+ND2+ND3+ND4+1) - N6 ! save 4 points for transonic region
  IF (N5 .NE. 6) STOP ' ERROR IN N5'
ENDIF

AUXI = (2.D0*VS2/VMAX)** (1.D0/BETA)  
RDIV2 = R0/ (1.D0-AUXI)  

IF (RDIV2.LE.RC) STOP 'ERROR IN RDIV2'  

VDIV2 = 0.3D0*VS2  

IF (VDIV2.LE.VMIN) STOP 'VMIN TOO LARGE - MODIFY INPUT (EITHER VMIN OR VDIV)'  
! modified March 2016 (nearest index not w.r.t. to v, but to r)
! otherwise, points become to close to each other
!
!     nearest index to rc (corresponding to vs2)
!
!XI = (2.D0-1.D0)* (N5-1)/ (2.D0-.2D0)  
!IVS2 = INT(XI)  
!IF (XI-IVS2.GT..5D0) IVS2 = IVS2 + 1  
!

RVMAX = SR + HL*LOG(VDIV2/VMIN)  

IF (RVMAX.GE.RC) STOP 'ERROR IN 0.3 * VS2'  

RPAR = RDIV2  
RPER = RVMAX  

! modified
XI = (RPAR-RC)* (N5-1)/ (RPAR-RPER)  
IVS2 = INT(XI)  
IF (XI-IVS2.GT..5D0) IVS2 = IVS2 + 1  

DELTAR = (RPAR-RPER)/DBLE(N5-1)  

!----------------------------------------------------------------------------
! MARCH 2016
! remove points between rpar+deltar and rc (will be filled up with N5/2 points below),
! and redistribute them w.r.t. velocity
ISTART=0
IEND=0

NBAD=ND1 + ND2 + ND3 + ND4
DO I=1,NBAD
  IF (R(I).LE.RPAR+DELTAR .AND. R(I).GE.RC) THEN
    ISTART=I
    EXIT
  ENDIF  
ENDDO

DO I=NBAD,1,-1
  IF (R(I).LE.RPAR+DELTAR .AND. R(I).GE.RC) THEN
    IEND=I
    EXIT
  ENDIF  
ENDDO

IF(ISTART.EQ.0 .AND. IEND.NE.0) STOP ' ERROR IN ISTART, IEND (REMOVE IN MODVL)'

IF(ISTART.NE.0) THEN 
  NR = IEND - ISTART + 1
  IF(NR.LT.1) STOP ' ERROR IN NR (REMOVE IN MODVL)'
! remove points
  DO J = IEND + 1,NBAD
    R(J - NR) = R(J)
    V(J - NR) = V(J)
    RHO(J- NR) = RHO(J)
  ENDDO
!re-distribute them w.r.t. velocity (in this case, in the outer regions)
  DO J = 0,NR - 1  
     DIFFV = (V(1) - V(2))  
     I = 2  

     DO K = 3,NBAD - NR + J  
          H3 = (V(K-1) - V(K)) 
          IF (H3.GT.DIFFV) THEN  
               DIFFV = H3  
               I = K  
          END IF  
     END DO
     
     M = NBAD - NR  + J - I  
     CALL VINDEX(R,V,RHO,M,I)  
     R(I) = .5D0* (R(I)+R(I-1))  
     V(I) = VFVL(R(I),RC,R0,SR,VMAX,VMIN,BETA,HL)  
     RHO(I) = RHOFUNC(R(I),V(I))  
  END DO  
ENDIF
!----------------------------------------------------------------------------

DO II = 0,N5 - 1  
     RNEW1 = RPAR - DBLE(II)*DELTAR  
!
!        one point exactly at rc
!
     IF (II.EQ.IVS2) RNEW1 = RC ! usually, at II=2 (for N5=6)

     DO J = 1,ND1 + ND2 + ND3 + ND4 + II  
          IF (R(J).GT.RNEW1) K = J + 1  
     END DO

     M = ND1 + ND2 + ND3 + ND4 + II - K  
     CALL VINDEX(R,V,RHO,M,K)  
     R(K) = RNEW1  
     V(K) = VFVL(R(K),RC,R0,SR,VMAX,VMIN,BETA,HL)  
     RHO(K) = RHOFUNC(R(K),V(K))  
END DO  

IF(N6.GT.0) THEN
! N6 points to resolve transonic region from 2*VDIV2 to 3*VSOUND
  
DO I=1,ND-1
  IF(V(I).LT.3.*VSONIC) EXIT
ENDDO

!DIFFV=ABS(V(I)-3.*VSONIC)

!IF(DIFFV.LT.ABS(V(I-1)-3.*VSONIC)) THEN
!  ISTART=I
!ELSE
  ISTART=I-1
!ENDIF

RPAR=R(ISTART)

DO I=ISTART,ND-1
! CHANGED IN V10.1 BECAUSE OF ACCURACY
  IF(V(I).LT.(2.D0-1.D-12)*VS2) EXIT
ENDDO

IEND=I-1
RPER=R(IEND)

IF (RPAR-RPER.LE.0.) STOP ' RPAR < RPER IN TRANSONIC POINTS'

DO J = 0,N6-1  

     DIFFV = LOG10(V(ISTART)/V(ISTART+1))  
     I = ISTART+1 
     DO K = ISTART+2,IEND+J 
          H3 = LOG10(V(K-1)/V(K)) 
          IF (H3.GT.DIFFV) THEN  
               DIFFV = H3  
               I = K  
          END IF  
     END DO

     M = ND1 + ND2 + ND3 + ND4 + N5 + J - I  
     CALL VINDEX(R,V,RHO,M,I)  
     R(I) = .5D0* (R(I)+R(I-1))  
     V(I) = VFVL(R(I),RC,R0,SR,VMAX,VMIN,BETA,HL)  
!     PRINT*,RPAR,RPER,V(istart)/1.e5,v(iend)/1.e5,R(I),V(I)/1.E5
     RHO(I) = RHOFUNC(R(I),V(I))  
END DO  

ENDIF
!
!     einschieben eines punktes nahe der photosphaere
!
CALL VINDEX(R,V,RHO,0,ND-1)  
R(ND-1) = R(ND) + EPS1* (R(ND-2)-R(ND))  
V(ND-1) = VFVL(R(ND-1),RC,R0,SR,VMAX,VMIN,BETA,HL)  
RHO(ND-1) = RHOFUNC(R(ND-1),V(ND-1))  

!DO K=1,ND
!  PRINT*,K,R(K)/SR,V(K)/1.E5
!ENDDO

DO K = 2,ND  
     IF (V(K).GT.V(K-1)) THEN  
          V(K) = V(K-1)  
          RHO(K) = RHOFUNC(R(K),V(K))  
     END IF  
END DO  

DO I = 1,ND  
     RRR = R(I)  
     IF (RRR.GE.RC) THEN  
          GRADV(I) = VMAX*BETA*R0/RRR/RRR* (1.D0-R0/RRR)** (BETA-1.D0)
     ELSE  
          GRADV(I) = VMIN/HL*EXP((RRR-SR)/HL)  
     END IF  
END DO  
!
!---- end of calculation of the analytical model
!
!     correction for photopheric density stratification
!     warning!!! warning!!! warning!!!
!     up to now neglected: advection term (minor)
!     approximation of sigma-thomson in vsound (minor)
!
!     So far, no optically thick clumping in photosphere allowed!

DO L = 1,ND  
     IF (R(L).EQ.RC) GO TO 170  
END DO  
STOP ' ERROR: RC NOT FOUND'  

  170 CONTINUE  

NS = L  
NSDIV = NS  

IF (ITHYD.EQ.1) THEN  
     RTAU23 = SRNOM/SR  
     RMAX = R(1)/SR  
ELSE  
     PRINT *  
     PRINT *,' PHOTOSPHERIC CORRECTION, M(RSTAR) = ',XMMAX  
     PRINT *  
     XT1 = EX ! for module nlte_var 
     XT = EX  ! for module photstruc
     XMLOSS = XMLOSS*1.989D33/3.1557D7  
     PI = ACOS(-1.D0)  
     A2 = VSONIC**2  
     A2TE = A2/TE  
     CKAPPA = SIGEM*XKAP/A2TE  
     VMIN = V(NS)  
     R0 = R(NS)  
     T0 = TACT(NS)  
     B = 1.D0 - (VMIN/VMAX)** (1.D0/BETA)  

     IF (BETA.EQ.1.D0) THEN  
          H2 = -LOG(1.D0-B)  
     ELSE  
          H2 = (1.D0- (VMIN/VMAX)** ((1.D0-BETA)/BETA))/ (1.D0-BETA)
     END IF  

     RHO0 = RHO(NS)  
     P0 = A2TE*DELMU(NS)*RHO0*T0  
     XM0 = XMLOSS/ (4.D0*PI*R0*VMAX)/B*H2  

     PRINT *  
     PRINT *,' DIVISION AT ',VMIN*1.D-5,' KM/S (L = ',NS,')'  
     PRINT *,' CORRESPONDING TO ',R0/SRNOM,' NOM. RADII'  
     PRINT *,' AND LG(M0) = ',LOG10(XM0)  
     PRINT *  

     DELTAM = LOG10(XMMAX/XM0)/ (ND-NS-5)  
     XM(NS) = LOG10(XM0)  

     RADACC=0.

     DO L = NS + 1,ND  
          DELTAM1 = DELTAM  
          IF (L.LE.NS+8) DELTAM1 = DELTAM/2.D0  
          IF (L.LE.NS+4) DELTAM1 = DELTAM/3.D0  
          IF (L.LE.NS+2) DELTAM1 = DELTAM/6.D0  
          XM(L) = XM(L-1) + DELTAM1  
     END DO

     DO L = NS,ND  
       XM(L) = 10.D0**XM(L)
     END DO
!JO Jan/April 2016: last point close to previous one, to improve diffusion approx.
     IF(ND.GE.61) XM(ND)=XM(ND-1)*1.1

     X1 = XM0  
     Y(1) = P0  
     IF(MODTYPE.EQ.'TLUSTY') Y(1) = RHO0  
     Y(2) = T0  
     Y(3) = R0  
     P(NS) = P0  
     TNEW(NS) = T0
! RHO(NS) ALREADY DEFINED

! THIS IS NEW, ACCOUNTING FOR TMIN 
     DO L = NS + 1,ND
       IF(TACT(L).EQ.T0) CYCLE
       EXIT
     ENDDO  
     ITMIN=L-1

!     PRINT*,'ITMIN=',ITMIN


! LOOP OVER ALL XM
! FROM V10.1 ON: WHILE LOOP, TO ALLOW FOR RESET OF L
!-------------------------------------------------------------------------
     L = NS +1

     DO WHILE (L < ND+1)

          PL=Y(1)
          IF(MODTYPE.EQ.'TLUSTY') RHOL=Y(1)  
          TL=Y(2)
          RL=Y(3)
          X2 = XM(L)  
          HS = (X2-X1)/20.D0  
          IDUM = L  

          IF (MODTYPE.EQ.'FASTWD' .OR. MODTYPE.EQ.'MICHEL') THEN
             CALL ODEINT(Y,3,X1,X2,1.D-5,HS,0.D0,NOK,NBAD,DERIVS,RKQC)
          ELSE IF (MODTYPE.EQ.'KURUCZ') THEN
             CALL ODEINT(Y,3,X1,X2,1.D-5,HS,0.D0,NOK,NBAD,DERIVS_KUR,RKQC)
          ELSE IF (MODTYPE.EQ.'TLUSTY') THEN
             CALL ODEINT(Y,3,X1,X2,1.D-5,HS,0.D0,NOK,NBAD,DERIVS_TLU,RKQC)
          ELSE
             STOP ' WRONG MODEL TYPE'
          ENDIF

          P(L) = Y(1)  
          TNEW(L) = Y(2)  
          R(L) = Y(3)  
          IF(MODTYPE.EQ.'TLUSTY') THEN
             RHO(L)=Y(1)
             P(L) = RHO(L)*(A2TE*DELMU(L)*TNEW(L))   
          ENDIF
!          PRINT*,'TLUSTY',L,log10(XM(L)),R(L)/SR,log10(RHO(L)),TNEW(L),P(L)

! one point just below SRNOM 
! always for standard calculation (OPT_PHOTCLUMP=.FALSE.) or from ITHYD > 3
! if photospheric clumping (otherwise, photosphere too close to division point)
          IF(R(L).LT.SRNOM.AND.R(L-1).GT.SRNOM .AND. &
&            (.NOT.OPT_PHOTCLUMP .OR. ITHYD .GT.3)) THEN
            IF(L.LE.NS+4) THEN
            PRINT*
            PRINT*,' WARNING! PHOTOSPHERE AT ',L,' VERY CLOSE TO DIVISION POINT AT',NS
            PRINT*
            ENDIF
! new try
            SAFETY=1.1D0
95          DMDR=(XM(L)-XM(L-1))/(R(L-1)-R(L))
            DELTAM=SAFETY*DMDR*(R(L-1)-SRNOM) !including safety factor
            IF (DELTAM.LT.0.) STOP ' ERROR IN DELTAM < 0'

! FROM V10.1 ON: AVOID TOO SMALL SEPARATIONS (except L - 1 = NS, i.e., 
!           last point = starting point)
            IF (DELTAM/XM(L-1) .LT. 0.1 .AND. L-1 .NE. NS) THEN ! FACTOR 1.1
! SKIP LAST POINT, OVERWRITE IT
              XM(L-1)=XM(L-1)+DELTAM
! RESET INDEX, INITIALIZE ODE
              L = L - 1
              X1 = XM(L-1)
              Y(1) = P(L-1) 
              IF(MODTYPE.EQ.'TLUSTY') Y(1)=RHO(L-1) 
              Y(2) = TNEW(L-1)  
              Y(3) = R(L-1)  
              IDUM = L
            ELSE
! ACCEPT LAST POINT, RECALCULATE NEW POINT
              XM(L)=XM(L-1)+DELTAM
              Y(1)=PL
              IF(MODTYPE.EQ.'TLUSTY') Y(1)=RHOL  
              Y(2)=TL
              Y(3)=RL  
            ENDIF  
            X2=XM(L)
            HS = (X2-X1)/20.D0  

            IF (MODTYPE.EQ.'FASTWD' .OR. MODTYPE.EQ.'MICHEL') THEN
               CALL ODEINT(Y,3,X1,X2,1.D-5,HS,0.D0,NOK,NBAD,DERIVS,RKQC)
            ELSE IF (MODTYPE.EQ.'KURUCZ') THEN
               CALL ODEINT(Y,3,X1,X2,1.D-5,HS,0.D0,NOK,NBAD,DERIVS_KUR,RKQC)
            ELSE IF (MODTYPE.EQ.'TLUSTY') THEN
               CALL ODEINT(Y,3,X1,X2,1.D-5,HS,0.D0,NOK,NBAD,DERIVS_TLU,RKQC)
            ELSE
               STOP ' WRONG MODEL TYPE'
            ENDIF

            P(L) = Y(1)  
            TNEW(L) = Y(2)  
            R(L) = Y(3)  
            IF(MODTYPE.EQ.'TLUSTY') THEN
              RHO(L)=Y(1)
              P(L) = RHO(L)*(A2TE*DELMU(L)*TNEW(L))   
            ENDIF
            IF(R(L)/SRNOM.GE.1.0D0) THEN
! NEW PHILOSOPHY FROM V10.1.3 ON
               IF(SAFETY.EQ.1.1D0) THEN
                  PRINT*,'WARNING!!!  PROBLEMS WITH POINT CLOSE TO PHOTOSPHERE!'
                  SAFETY=1.2
                  GOTO 95
               ENDIF 
  
               PRINT*,XM(L-2),XM(L-1),XM(L)
               PRINT*,R(L-2)/SRNOM,R(L-1)/SRNOM,R(L)/SRNOM
               PRINT*,R(L)/SRNOM
               STOP ' PROBLEMS WITH POINT CLOSE TO PHOTOSPHERE CANNOT BE CURED!'
            ENDIF
! define new xm grid (equidistant)
            DELTAM = LOG10(XMMAX/X2)/ (ND-L)  
            DELTAM=10.D0**DELTAM
            DO LL = L + 1,ND  
              XM(LL) = XM(LL-1)*DELTAM   
            END DO
!           
          ENDIF  
          X1 = X2  
! re-calculate R^2*GRAD (OK, checked!!!)
          TERM = SIGEM + CKAPPA*DELPARA(L)/DELMU(L)*P(L)*Y(2)** (-XT-1.D0)  
! note that ckappa includes sigmae
          TERM = TERM*DELSIG(L)					  
          GRAD = 5.67D-5/2.9979D10*TEFF**4*TERM
          RADACC(L)=GRAD

          L = L + 1
     END DO

!---- while loop finished

     DO L = NS + 1,ND  
          IF (P(L).LT.0.D0) STOP 'PRESSURE NEGATIVE!'  
          IF (MODTYPE .NE. 'TLUSTY') RHO(L) = P(L)/ (A2TE*DELMU(L)*TNEW(L))  
! for tests; note that differences in grad should scale with
! differences in chiross/rho, where chiross corresponds to TERM below        
!          TERM = SIGEM*(1.D0+XKAP*DELPARA(L)*RHO(L)*TNEW(L)**(-XT))
!          TERM = TERM*DELSIG(L)*RHO(L)
!          GRAD=5.67D-5*TEFF**4/2.9979D10*TERM/RHO(l)*(SRNOM/R(L))**2
!          print*,l,' ',p(l),' ',tnew(l),' ',tact(l),' ',delmu(l),' ', &
!&           rho(l),' ',LOG10(TERM)
!          print*,l,' ',grad,' ',radacc(l)
     END DO

     SR = R(ND)  
     RMAX = R(1)/SR  
     RTAU23 = SRNOM/SR  
     SRVMIN = R0  
     IF (1.D0-1.D0/RTAU23.GT..1D0) PRINT *,' WARNING!!! EXTENDED PHOTOSPHERE'

     PRINT *,' LOWER (PHOT.) RADIUS AT ',1.D0/RTAU23,' NOM. RADII'  
     PRINT *,' MAX. RADIUS (RMAX) = ',RMAX,' (IN SR)'  

     DO L = NS + 1,ND  
          V(L) = XMLOSS/ (4.D0*PI*R(L)**2*RHO(L))  
     END DO
!
!    modified to ensure smooth transition
!
! new version
     DO L = NS,ND - 1  
          DM = LOG(V(L)/V(L-1))/ (R(L)-R(L-1))  
          DPX =LOG(V(L)/V(L+1))/ (R(L)-R(L+1))  
          GRADV(L) = .5D0* (DM+DPX) * V(L)   
     END DO

     L = ND  
     GRADV(L) = LOG(V(L)/V(L-1))/ (R(L)-R(L-1))*V(L)  

END IF  

!-------------
IF (MODTYPE.EQ.'MICHEL') THEN

! OVERWRITE THE WIND-STRUCTURE WITH MICHEL'S VALUES, AFTER RENORMALIZING
 VNS=V(NS)
 DO L=1,NDMOD-1
   IF(VMICH(L).LT.VNS .AND. VMICH(L+1).GE.VNS) EXIT
 ENDDO
 RNS=RMICH(L)+(RMICH(L+1)-RMICH(L))/(VMICH(L+1)-VMICH(L))*(VNS-VMICH(L))
 PRINT*,VNS, RNS, R(NS), SRNOM
!THIS POINT NOW SHOULD CORRESPOND TO R(NS) = SRNOM in the first iteration,
!and to SRVMIN later on (until convergence) 

!print*,'renormalization!'
!print*,rns,rns/srnom,l

!RENORMALIZATION
 DO L=1,NDMOD
   RMICH1(L)=RMICH(L)/RNS*R(NS)
 ENDDO

 IF (RMICH1(NDMOD).LE.R(1)) STOP ' OUTER RADIUS (MICHEL) SMALLER THAN RMAX, DECREASE RMAX'

 DO K=1,NS-1
   DO L=1,NDMOD-1
     IF(RMICH1(L).LT.R(K).AND. RMICH1(L+1).GE.R(K)) EXIT
   ENDDO
   VNS=VMICH(L)+(VMICH(L+1)-VMICH(L))/(RMICH1(L+1)-RMICH1(L))*(R(K)-RMICH1(L))
   V(K)=VNS
   RHO(K)=H1/(R(K)**2*V(K))
   PRINT*,'michel',K,R(K)/SRNOM,V(K),RHO(K)
 ENDDO   
!DVDR
CALL DERIV_3P(GRADV,R,V,ND) 

ENDIF
!-------------

DO K = 1,ND  
     R(K) = R(K)/SR  
     V(K) = V(K)/VMAX  
     GRADV(K) = GRADV(K)*SR/VMAX  
     IF (ITHYD.GT.1) THEN
       P(K) = RHO(K)* (A2TE*DELMU(K)*TACT(K))  
     ENDIF    
END DO  

R(ND) = 1.D0  

!-------------
!JO Apri16: now we 'smooth' the velocity at the transition point,
! requiring a 3rd degree polynom and prescribed v, gradv above and below
! transition point (4 unknowns for the parabola, four conditions to fix them)

IF(ITHYD.GT.1) THEN
  RR1=R(NS-1)
  RR2=R(NS+1)

  ZZ1=(/RR1**3,RR1**2,RR1,1.D0/)
  ZZ2=(/RR2**3,RR2**2,RR2,1.D0/)
  VMAT(1,:)=ZZ1
  VMAT(2,:)=ZZ2
  VMAT(3,:)=(/3.D0*RR1**2,2.D0*RR1,1.D0,0.D0/)
  VMAT(4,:)=(/3.D0*RR2**2,2.D0*RR2,1.D0,0.D0/)

  YY=(/V(NS-1),V(NS+1),GRADV(NS-1),GRADV(NS+1)/)

  CALL LUDCMP(VMAT,4,4,INDLUD,DET)  
  CALL LUBKSB(VMAT,4,4,INDLUD,YY)  

!test precision
  V1TEST=DOT_PRODUCT(YY,ZZ1)
  IF(ABS(1.D0-V1TEST/V(NS-1)).GT.1.D-4) STOP ' PRECISION OF V1 NOT SUFFICIENT (ROUTINTE MODVL)'
  V2TEST=DOT_PRODUCT(YY,ZZ2)
  IF(ABS(1.D0-V2TEST/V(NS+1)).GT.1.D-4) STOP ' PRECISION OF V2 NOT SUFFICIENT (ROUTINTE MODVL)'

!calculate smoothed velocity at NS
  RR0=R(NS)
  ZZ1=(/RR0**3,RR0**2,RR0,1.D0/)
  VSMOOTH=DOT_PRODUCT(YY,ZZ1)
  PRINT* 
  PRINT*,'OLD/NEW V-VALUES AROUND NS:'
  WRITE(*,FMT='(3(F12.6,2X))') V(NS-1),V(NS),V(NS+1)
  WRITE(*,FMT='(3(F12.6,2X))') V(NS-1),VSMOOTH,V(NS+1)

!recalculate rho and p, assuming r^2*rho*v = const
  RHOSMOOTH=RHO(NS)*V(NS)/VSMOOTH
  VNSOLD=V(NS)
  V(NS)=VSMOOTH
  P(NS) = P(NS)/RHO(NS)*RHOSMOOTH
  RHO(NS)=RHOSMOOTH

  IF (MODTYPE.EQ.'MICHEL') THEN
     CALL DERIV_3P(GRADVNEW,R,V,ND) 
  ELSE
    GRADVNEW=GRADV
    DO L = NS-1,NS+1 
        DM = LOG(V(L)/V(L-1))/ (R(L)-R(L-1))  
        DPX =LOG(V(L)/V(L+1))/ (R(L)-R(L+1))  
        GRADVNEW(L) = .5D0* (DM+DPX) * V(L)   
    END DO
  ENDIF
! for tests
  PRINT*,'OLD/NEW DVDR-VALUES AROUND NS:'
  WRITE(*,FMT='(5F12.6,2X)') GRADV(NS-2:NS+2)
  WRITE(*,FMT='(5F12.6,2X)') GRADVNEW(NS-2:NS+2)

  GRADV=GRADVNEW
ENDIF

!----------------------------------------------------------------------------

!----from here on, we have to consider clumping
!    RHOCL is the overdense rho (in the average clump), whilst
!    RHO is the mean rho, consistent with Mdot
CALL CLUMPING(R,V,RHO,CLFAC,RHOCL,RTAU23,ND,NS,VMAX,GRADV)

DO L = 1,ND  
     IF (ITHYD.EQ.1) XNE(L) = C2*RHOCL(L) !corrected
     XNH(L) = RHOCL(L)/C1  !corrected
END DO  

! assumed that xne for ithyd ne 1 has been corrected

XNECL=XNE/CLFAC ! corrected for integration of optical depth
! in the following and until further evidence, we use the micro-clumped
! tau_e since this is presumably used for start-values only
!JO check

CALL SPLINE(R,XNECL,ARHO,BRHO,CRHO,ND)  
TAU(1) = TAUMIN  
SUMM = TAUMIN  

DO L = 1,ND - 1  
     DEL = R(L+1) - R(L)  
     SUMM = SUMM - SIGMAE*SR*DEL* (XNECL(L)+ DEL* (ARHO(L)/2.D0+DEL* &
&      (BRHO(L)/3.D0+ DEL*CRHO(L)/4.D0)))
     TAU(L+1) = SUMM  
END DO  

PRINT *  
PRINT *,' TAU-THOMSON (MICRO-CLUMPED) AT RSTAR = ',TAU(ND)  
PRINT *  

!
!---- arho: tau-lucy(electron)
!---- brho: tau-electron
!
SRCONST = SR*RTAU23**2  

DO L = 1,ND  
          CRHO(L) = SRCONST*SIGMAE*XNECL(L)/R(L)/R(L)  
END DO

SUMM = 0.D0  
ARHO(1) = 0.D0
DO L = 1,ND - 1  
          DEL = R(L+1) - R(L)  
          SUMM = SUMM - DEL*5.D-1* (CRHO(L)+CRHO(L+1))  
          ARHO(L+1) = SUMM  
END DO

DO L = 1,ND  
          CRHO(L) = SR*SIGMAE*XNECL(L)  
END DO

SUMM = 0.D0  
BRHO(1) = 0.D0  

DO L = 1,ND - 1  
          DEL = R(L+1) - R(L)  
          SUMM = SUMM - DEL*5.D-1* (CRHO(L)+CRHO(L+1))  
          BRHO(L+1) = SUMM  
END DO

! changed: here, tlucy uses tau_e only. Thus, rtau23 is no longer found
! for recombined models. Necessary only in first iteration anyway
IF(ITHYD.LT.2) CALL TLUCY(R,ARHO,BRHO,TEMP,ND,TMIN,TEFF,RTAU23,ITHYD)  


DO L = 1,ND  
     INDEX(L) = L  
END DO  

! in case, adapt vdiv to prevent density inversion
IF(TEFF .GE. 12000. .AND. ITHYD.GE.3 .AND. VDIV.LT.0.9) THEN

  VINV=0
! JO July 2016: this test has to be performed with unsmoothed v(ns)=VNSOLD
  V(NS)=VNSOLD  
  DO L = ND-1,1,-1
  IF(V(L).LT.V(L+1)) THEN
    VINV=L+1
    EXIT
  ENDIF
  ENDDO

  IF(VINV.NE.0) THEN
    DO L=VINV,1,-1  
    IF(V(L).GT.V(VINV)) THEN
      AUXI=VDIV !old value
      VDIV=1.2*V(L)*VMAX/VSONIC !including safety factor 1.2
      VDIV=MAX(VDIV,1.2*AUXI) ! this is the new version; ensure that VDIV
                              ! becomes always larger (not smaller)
      VDIV=MIN(VDIV,0.9D0)    ! not beyond vsound
      ICONVER=0
      EXIT
    ENDIF
    ENDDO
  PRINT*,' WARNING! WARNING! WARNING!'
  PRINT*,' VDIV ADAPTED, NEW VALUE = ',VDIV
  WRITE(999,*)
  WRITE(999,*) ' VDIV ADAPTED, NEW VALUE = ',VDIV
  PRINT*
  ENDIF
  V(NS)=VSMOOTH
ENDIF

FOUND=.FALSE.
DO L = 1,ND  
     IF (GRADV(L).LE.0.D0) THEN  
     FOUND=.TRUE.
       GRADV(L) = -GRADV(L)  
            PRINT *, &
&           ' NEGATIVE VEL. GRADIENT -- DENSITY INVERSION! AT N =',L
            PRINT *,' ACTUAL VDIV = ',VDIV
         IF(ICONVER.EQ.1) THEN
            PRINT *, &
&             ' NEGATIVE VEL. GRADIENT -- DENSITY INVERSION! AT N =',L
            WRITE(999,*) &
&             ' NEGATIVE VEL. GRADIENT -- DENSITY INVERSION! AT N =',L
         ENDIF
     END IF  
END DO  
IF (FOUND.AND.ICONVER.EQ.1) THEN
            PRINT *,' ACTUAL VDIV = ',VDIV
            PRINT *,' INCREASE LOG G???'  
            WRITE(999,*) ' ACTUAL VDIV = ',VDIV
            WRITE(999,*) ' INCREASE LOG G???'
            WRITE(999,*) ' PROCEED AT OWN RISK!!!!'
ENDIF

!
!---- file 'model' is written: XNE, XNH include CLFAC
!     (RHO average density without CLFAC)
!     from version 9.0 on: NS as last entry
!     from version 10.0 on: UPDATED and XT as last entries
!
REWIND 20  
! UPDATED always .false. here

WRITE (20) TEFF,GGRAV,SR,YHE,XMU,VMAX,XMLOSS,BETA, (R(I),I=1,ND), &
&   (V(I),I=1,ND), (GRADV(I),I=1,ND), (RHO(I),I=1,ND), &
&   (XNE(I),I=1,ND), (XNH(I),I=1,ND), (CLFAC(I),I=1,ND), &
&   (INDEX(I),I=1,ND), (P(I),I=1,ND), NS, UPDATED, XT 

! store photospheric rad. acceleration
OPEN(61,FILE=TRIM(MODNAM)//'/GRAD.OUT',STATUS='UNKNOWN')
DO L=1,ND
   WRITE(61,*),'RADACC ',L,' ',RADACC(L)
ENDDO
CLOSE(61)


IF (ITHYD.GT.1) THEN  

     DO L = 1,NS  
          TEMP(L) = TACT(L)  
     END DO

     DO L = NS + 1,ND  
          TEMP(L) = TNEW(L)  
     END DO

END IF  

DO LL = 1,ND  
     L = INDEX(ND+1-LL)  
     TEMP1(LL) = TEMP(L)  
     TD(LL) = 1.2D0*SIGMAE*XNH(L)*VDST/ (GRADV(L)*VMAX/SR)  
     WNE(LL) = XNE(L)/ (.5D0* (1.D0-SQRT(1.D0-1.D0/R(L)/R(L))))  
     R1(LL) = R(L)  
     V1(LL) = V(L)*VMAX*1.D-5  
     RHO1(LL) = RHO(L)  
     XNH1(LL) = XNH(L)  
     XNE1(LL) = XNE(L)/XNH(L)  
!     write(*,190) l,td(ll),wne(ll),r1(ll),v1(ll),rho1(ll),xne1(ll)
END DO  
!
!     open(1,file='stwithd',form='unformatted')
!     rewind 1
!     write(1) (td(i),i=1,nx),(wne(i),i=1,nx),(r1(i),i=1,nx),
!    *         (v1(i),i=1,nx),(rho1(i),i=1,nx)
!     close(1)
!
!---- file 'temp' is written
!
REWIND 21  
WRITE (21,*) R1, TEMP1  

RETURN  

 9000 FORMAT ('  TEFF = ',F10.1,'  LOG G = ',F5.2,'  RSTAR(NOMINAL)', &
&       ' = ',F5.1,' YHE = ',F6.2,/)
 9010 FORMAT ('  MLOSS/YEAR = ',E12.6,'  VMAX = ',F7.1,'  VMIN = ', &
&       F11.7)
 9020 FORMAT (I3,6 (2X,E10.5))  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CLUMPING(R,V,RHO,CLFAC,RHOCL,RTAU23,ND,NS,VMAX,DVDR)
!
!
! calculates clumping factor clfac, CAN BE MODIFIED BY USER
! parameters (up to ten) provided in array clf_field 
! number of parameters given by NPARA_CLF
!
! first parameter MUST be representative clumping factor, i.e. in any case
! 
!  MIN(CLFAC) <= CLF_FIELD(1) <= MAX(CLFAC)
!
! can be used, of course, do indicate that no clumping is present
! (this is NOT a must, however)
!
! User can change parameterization
! (as a function of V = v(r)/vinf or R = r/SR, with SR INNER radius)
! RTAU23 is ratio of SRNOM/SR, i.e., if you want to refer to
! the nominal radius, all values of r have to be divided by RTAU23
! (in this case then, r(nd) < 1 finally) => available in array RSCAL
!
! NS is the index of the transition point between wind and photosphere, can
! be used as well  
!  
! output values are clfac, rhocl=rho*clfac (overdensity in clumps)
!
! adapted for optically thick clumping by JS
! NOTE: variables HPOR, FVEL, FVOL *ONLY* needed for printing oputputs, 
! ALL calculations work on TCL_FAC_LINE, TCL_FAC_CONT, and FIC -- see notes 
! NOTE2: In Sundqvist+ 2016, TCL_FAC_LIN = G_l , TCL_FAC_CONT = G_c   

! bug in conversion from fcl/fic to fvol identified March 2018, but removed
! only in Nov. 2021  

USE nlte_type
USE nlte_var, ONLY: CLF_FIELD, NPARA_CLF, SR, MODNAM
USE nlte_opt, ONLY: OPT_PHOTCLUMP
USE photstruc, ONLY: CLF_CONST

USE nlte_porvor, ONLY: OPTTHICK, FIC_FIELD, FVEL_FIELD, HPOR_FIELD, &
&                      FIC,TCL_FAC_LINE,TCL_FAC_CONT, FVEL,FVOL,HPOR

IMPLICIT NONE

INTEGER(I4B), INTENT(IN) :: ND, NS
REAL(DP), INTENT(IN) :: RTAU23, VMAX
REAL(DP), DIMENSION(ND), INTENT(IN) :: R, V, RHO, DVDR
REAL(DP), DIMENSION(ND), INTENT(OUT) :: CLFAC,RHOCL
!
REAL(DP) :: CLF_REPRESENTATIVE      !always
REAL(DP) :: CLF,VCLSTART,VCLMAX, CLFMIN, CLFMAX     !for NPARA_CLF=3
REAL(DP) :: DUMMY                   !for NPARA_CLF=4
REAL(DP) :: CLFMID,RMID,CLFOUT,ROUT !for NPARA_CLF=5
REAL(DP) :: CL1,CL2,CL3,CL4 !for NPARA_CLF=6
REAL(DP) :: RIN, RFAR, CLFIN, CLFFAR! additionally for NPARA_CLF=9, 11
REAL(DP) :: RPHOT,CLFPHOT! additionally for NPARA_CL=11 
REAL(DP) :: FVOR

REAL(DP), DIMENSION(ND) :: RSCAL ! automatic array, in case we need it

INTEGER(I4B) :: I,IMAX,ISTART,IMID
INTEGER(I4B) :: IPHOT,IIN,IOUT ! additionally for NPARA_CLF=11

CLF_REPRESENTATIVE=CLF_FIELD(1)
RSCAL=R/RTAU23 !scaled to nominal radius Rstar

IF(CLF_REPRESENTATIVE.LT.1.D0) STOP ' represent. value of CLF < 1!'

!------------------------------------------------------------
! CHANGE IN CASE
IF(OPT_PHOTCLUMP .AND. NPARA_CLF.NE.3) &
& STOP ' photospheric clumping .AND. NPARA NE 3' 

IF(NPARA_CLF.EQ.3) THEN

CLF=CLF_REPRESENTATIVE
VCLSTART=CLF_FIELD(2)
VCLMAX=CLF_FIELD(3)
! "old" approach as used from version 8.2 on (linear increase from 1 to CLF
!  in between vclstart and vclmax
!  vor v < vclstart, clfac = 1, for v > vclmax, clfac = clf
!
!  clf = 1 gives unclumped model
!  clf = x and vclstart = vclmax = 0. gives constant clumping factor = x EVERYWHERE
!
IF(VCLSTART.GT.VCLMAX) STOP ' VCLSTART > VCLMAX!'

IF(CLF.EQ.1.D0) THEN

CLFAC=1.D0

ELSE

DO I=1,ND
  IF(V(I).LT.VCLMAX) EXIT
ENDDO

IMAX=MAX0(1,I-1)
DO I=IMAX,ND
  IF(V(I).LT.VCLSTART) EXIT
ENDDO
ISTART=I-1 ! works also for vclstart = 0. (completely clumped atmosphere)

IF(ISTART.LT.IMAX) THEN
  PRINT*,ISTART,IMAX
  STOP ' ISTART < IMAX IN CLUMPING'
ENDIF

! photospheric clumping now allowed except for tests (OPT_PHOTCLUMP)
IF(.NOT.OPT_PHOTCLUMP) THEN
  IF(ISTART.GE.NS)  THEN
    PRINT*,ISTART,NS,V(NS)
    STOP ' ISTART < NS IN CLUMPING'
  ENDIF
ENDIF

CLFAC(1:IMAX)=CLF

DO I=IMAX+1,ISTART
! linear increase
  CLFAC(I)=1.+(CLF-1.)/(VCLMAX-VCLSTART)*(V(I)-VCLSTART)
ENDDO  
  
CLFAC(ISTART+1:ND)=1.D0
ENDIF

! except for tests, no photospheric clumping
IF(.NOT.OPT_PHOTCLUMP) THEN
  CLFAC(NS:ND)=1.D0
ELSE
  CLFMIN=MINVAL(CLFAC)  
  CLFMAX=MAXVAL(CLFAC)  
  IF (CLFMIN.NE.CLFMAX) STOP ' photospheric clumping, but CLF not constant'
  CLF_CONST=CLFMIN
ENDIF  

! for tests
!PRINT* 
!PRINT*,' CLUMPING FACTOR'
!DO I=1,ND
!  PRINT*,I,V(I),CLFAC(I)
!ENDDO
!
!------------------------------------------------------------
ELSE IF (NPARA_CLF.EQ.4) THEN
DUMMY=CLF_FIELD(4)

CLFOUT=CLF_FIELD(2)
VCLSTART=CLF_FIELD(3) ! in km/s

IF (DUMMY .EQ. 0.) THEN 
! either Exponential law a la Hillier; input filling factor, vcl, dummy=0.
CLFAC=CLFOUT + (1.-CLFOUT)*EXP(-(V*VMAX*1.D-5/VCLSTART))

ELSE
! Paco's law returning to an unclumped outer wind
CLFAC=CLFOUT + (1.-CLFOUT)*EXP(-(V*VMAX*1.D-5/VCLSTART))+ &
&              (1.-CLFOUT)*EXP((V-V(1))*VMAX*1.D-5/DUMMY)
ENDIF

CLFAC=1./CLFAC
! no photospheric clumping
CLFAC(NS:ND)=1.D0

! for tests
PRINT* 
PRINT*,' CLUMPING FACTOR'
DO I=1,ND
  PRINT*,I,RSCAL(I),V(I)*VMAX*1.D-5,CLFAC(I)
ENDDO
!
!------------------------------------------------------------
ELSE IF (NPARA_CLF.EQ.5) THEN
! preliminary parameterization of clfac starting from unity (at NS)
! to clmid (at rmid), reaching clout (at rout).
! For r> rout, clfac is set to clout, in between linearly interpolated
! idea: maximum value (around 5 to 9 at clmid (roughly at Rstar = 2), 
! then declining, following the results by Jo, Nevy and Salvo from the
! combined IR/radio analyis  
!
! rmid and rout in units of nominal radius
  
CLFMID=CLF_FIELD(2)
RMID=CLF_FIELD(3)
CLFOUT=CLF_FIELD(4)
ROUT=CLF_FIELD(5)
IF (CLFMID.LT.1.D0.OR.CLFOUT.LT.1.D0) STOP ' CLFMID OR CLFOUT < 1'

IF(RMID.GT.ROUT) STOP ' RMID > ROUT!'

IF(CLFMID.EQ.1.D0 .AND. CLFOUT.EQ.1.D0) THEN

CLFAC=1.D0

ELSE

DO I=1,ND
  IF(RSCAL(I).LT.ROUT) EXIT
ENDDO
IMAX=MAX0(1,I-1)

DO I=IMAX,ND
  IF(R(I).LT.RMID) EXIT
ENDDO
IMID=I-1

ISTART=NS
! PHOTOSPHERIC CLUMPING NOT ALLOWED
IF(IMID.GE.ISTART) IMID=ISTART-1

IF(IMID.LE.IMAX) THEN
  PRINT*,IMID,IMAX
  STOP ' IMID LE IMAX IN CLUMPING'
ENDIF

CLFAC(1:IMAX)=CLFOUT

DO I=IMAX+1,IMID
! linear interpol.
  CLFAC(I)=CLFOUT+(CLFMID-CLFOUT)/(RMID-ROUT)*(RSCAL(I)-ROUT)
ENDDO  

DO I=IMID+1,ISTART-1
! linear increase
  CLFAC(I)=1.+(CLFMID-1.)/(RMID-RSCAL(NS))*(RSCAL(I)-RSCAL(NS))
ENDDO  

CLFAC(ISTART:ND)=1.D0
ENDIF
! for tests
!PRINT* 
!PRINT*,' CLUMPING FACTOR'
!DO I=1,ND
!  PRINT*,I,RSCAL(I),CLFAC(I)
!ENDDO

!------------------------------------------------------------
ELSE IF (NPARA_CLF.EQ.6) THEN

! Paco's law with CL1 to CL4; one more (dummy) parameter than
! actually needed, to allow for 6 parameters  


CL1=CLF_FIELD(2)
CL2=CLF_FIELD(3)
CL3=CLF_FIELD(4)
CL4=CLF_FIELD(5)
DUMMY=CLF_FIELD(6)


IF (DUMMY .NE. 0.) STOP ' CLF_FIELD(6) NE 0, MODIFY!'
CLFAC=CL1 + (1.-CL1)*EXP(-(V*VMAX*1.D-5/CL2))+ &
&              (CL4-CL1)*EXP((V-V(1))*VMAX*1.D-5/CL3)

CLFAC=1./CLFAC
! no photospheric clumping
CLFAC(NS:ND)=1.D0

! for tests
PRINT* 
PRINT*,' CLUMPING FACTOR'
DO I=1,ND
  PRINT*,I,RSCAL(I),V(I)*VMAX*1.D-5,CLFAC(I),1./CLFAC(I)
ENDDO
!
!------------------------------------------------------------
ELSE IF (NPARA_CLF.EQ.11) THEN
! preliminary parameterization of clfac starting from unity (at NS)
! over clin (at rin), clmid(rmid), clout(rout) reaching clfar (at rfar).

! this is the linear formulation with respect to the histogram solution
! provided by Puls et al. (2006) (see npara_clf = 9 version below)
!
! all radii in units of nominal radius
  
RPHOT=CLF_FIELD(2)

! PHOTOSPHERIC CLUMPING NOT ALLOWED
IF (RPHOT .LT. RSCAL(NS)) STOP ' PHOTOSPHERIC CLUMPING NOT ALLOWED(1)!!!'

RIN=CLF_FIELD(3)
RMID=CLF_FIELD(4)
ROUT=CLF_FIELD(5)
RFAR=CLF_FIELD(6)


CLFPHOT=CLF_FIELD(7)
IF (CLFPHOT .NE. 1.) STOP ' PHOTOSPHERIC CLUMPING NOT ALLOWED(2)!!!'

CLFIN=CLF_FIELD(8)
CLFMID=CLF_FIELD(9)
CLFOUT=CLF_FIELD(10)
CLFFAR=CLF_FIELD(11)

DO I=1,ND
  IF(RSCAL(I).LT.RFAR) EXIT
ENDDO
IMAX=MAX0(1,I-1)

DO I=IMAX,ND
  IF(RSCAL(I).LT.ROUT) EXIT
ENDDO
IOUT=I-1

DO I=IOUT,ND
  IF(RSCAL(I).LT.RMID) EXIT
ENDDO
IMID=I-1

DO I=IMID,ND
  IF(RSCAL(I).LT.RIN) EXIT
ENDDO
IIN=I-1

DO I=IIN,ND
  IF(RSCAL(I).LT.RPHOT) EXIT
ENDDO
IPHOT=I-1

ISTART=NS
! PHOTOSPHERIC CLUMPING NOT ALLOWED
IF(IPHOT.GE.ISTART) IIN=ISTART-1

CLFAC(1:IMAX)=CLFFAR

DO I=IMAX+1,IOUT
! linear interpol.
  CLFAC(I)=CLFFAR+(CLFOUT-CLFFAR)/(ROUT-RFAR)*(RSCAL(I)-RFAR)
ENDDO  

DO I=IOUT+1,IMID
! linear interpol.
  CLFAC(I)=CLFOUT+(CLFMID-CLFOUT)/(RMID-ROUT)*(RSCAL(I)-ROUT)
ENDDO  

DO I=IMID+1,IIN
! linear interpol.
  CLFAC(I)=CLFMID+(CLFIN-CLFMID)/(RIN-RMID)*(RSCAL(I)-RMID)
ENDDO  

DO I=IIN+1,IPHOT
! linear interpol.
  CLFAC(I)=CLFIN+(CLFPHOT-CLFIN)/(RPHOT-RIN)*(RSCAL(I)-RIN)
ENDDO  

DO I=IPHOT+1,ISTART-1
! linear increase in case clfphot ne 1
  CLFAC(I)=1.+(CLFPHOT-1.)/(RPHOT-RSCAL(NS))*(RSCAL(I)-RSCAL(NS))
ENDDO  

CLFAC(ISTART:ND)=1.D0

! for tests
PRINT* 
PRINT*,' CLUMPING FACTOR'
DO I=1,ND
  PRINT*,I,RSCAL(I),CLFAC(I)
ENDDO

!------------------------------------------------------------
ELSE IF (NPARA_CLF.EQ.9) THEN
! parameterization according to Puls et al. (2006)
! rin, rmid, rout and rfar in units of nominal radius
  
RIN =CLF_FIELD(2)

! PHOTOSPHERIC CLUMPING NOT ALLOWED
IF (RIN .LT. RSCAL(NS)) STOP ' PHOTOSPHERIC CLUMPING NOT ALLOWED!!!'

RMID=CLF_FIELD(3)
ROUT=CLF_FIELD(4)
RFAR=CLF_FIELD(5)

CLFIN =CLF_FIELD(6)
CLFMID=CLF_FIELD(7)
CLFOUT=CLF_FIELD(8)
CLFFAR=CLF_FIELD(9)

CLFAC=1.D0

DO I=1,ND

IF(RSCAL(I).GE.RFAR) THEN
  CLFAC(I)=CLFFAR

ELSE IF (RSCAL(I).LT.RFAR .AND. RSCAL(I).GE.ROUT) THEN
  CLFAC(I)=CLFOUT

ELSE IF (RSCAL(I).LT.ROUT .AND. RSCAL(I).GE.RMID) THEN
  CLFAC(I)=CLFMID

ELSE IF (RSCAL(I).LT.RMID .AND. RSCAL(I).GE.RIN) THEN
  CLFAC(I)=CLFIN
ENDIF

ENDDO

! for tests
!PRINT* 
!PRINT*,' CLUMPING FACTOR'
!DO I=1,ND
!  PRINT*,I,RSCAL(I),CLFAC(I)
!ENDDO

!------------------------------------------------------------
ELSE  
  Print*,' NUMBER OF CLUMPING PARAMETERS = ',NPARA_CLF
  STOP ' CORRESPONDING IMPLEMENTATION OF CLFAC NOT AVAILABLE'
ENDIF
!------------------------------------------------------------

IF (OPTTHICK) THEN 

   !POSSIBLE TO INCLUDE SEVERAL OPTION FOR THICK CLUMPING AS WELL 
   !NOTE: MUST BE CONSISTENT WITH CORRESPNDING OPTION FOR THIN CLUMPING! 
   !(I.E. CLFAC IS HERE *NOT* RECALCULATED) 

   IF (NPARA_CLF.EQ.9) THEN 
      !Puls+ 2006 paramterization 

      ! FIC first 
      RIN  = FIC_FIELD(2) 
      RMID = FIC_FIELD(3) 
      ROUT = FIC_FIELD(4) 
      RFAR = FIC_FIELD(5) 
      
      CLFIN =FIC_FIELD(6)
      CLFMID=FIC_FIELD(7)
      CLFOUT=FIC_FIELD(8)
      CLFFAR=FIC_FIELD(9)
      FIC = 1.0D0 
      FVOL=1.0D0
      DO I=1,ND
         
! bug created by JON, finally cured by Jo Nov. 2021 (from 10.5.1.1 on)
! (1-fic^2) -> (1-fic)^2
         IF (RSCAL(I).GE.RFAR) THEN
            FIC(I)=CLFFAR
            FVOL(I) = (1.-FIC(I))**2 / ( CLFAC(I) - 2.*FIC(I) + FIC(I)**2 )
         ELSE IF (RSCAL(I).LT.RFAR .AND. RSCAL(I).GE.ROUT) THEN
            FIC(I)=CLFOUT
            FVOL(I) = (1.-FIC(I))**2 / ( CLFAC(I) - 2.*FIC(I) + FIC(I)**2 )
         ELSE IF (RSCAL(I).LT.ROUT .AND. RSCAL(I).GE.RMID) THEN
            FIC(I)=CLFMID
            FVOL(I) = (1.-FIC(I))**2 / ( CLFAC(I) - 2.*FIC(I) + FIC(I)**2 )
         ELSE IF (RSCAL(I).LT.RMID .AND. RSCAL(I).GE.RIN) THEN
            FIC(I)=CLFIN
            FVOL(I) = (1.-FIC(I))**2 / ( CLFAC(I) - 2.*FIC(I) + FIC(I)**2 )
         ENDIF
         ! volume filling factor, just as output for viewing 
         IF (FVOL(I).GT.1.0) STOP ' FVOL > 1 -- Check FCL and FIC input'
      ENDDO
      
      ! TCL FOR lines (multiplication factor) next 
      RIN =  FVEL_FIELD(2) 
      RMID = FVEL_FIELD(3) 
      ROUT = FVEL_FIELD(4) 
      RFAR = FVEL_FIELD(5) 
      ! FVEL
      CLFIN =FVEL_FIELD(6)
      CLFMID=FVEL_FIELD(7)
      CLFOUT=FVEL_FIELD(8)
      CLFFAR=FVEL_FIELD(9)
      !NOTE: JS -- CHANGED INPUT TO NORMALIZED VELOCITY FILLING FRACTION, cf Sundqvist+ 2014 
      !   FVEL = 1.0D30
      FVEL = 1.0 
      TCL_FAC_LINE = 0.0 
      DO I=1,ND
         
         IF(RSCAL(I).GE.RFAR) THEN
            FVEL(I) = CLFFAR       
            IF (FVEL(I).GE.1.0D0) THEN
               FVEL(I)=1.0D0
               TCL_FAC_LINE(I) = 0.0D0 
            ELSE
               FVOR = FVEL(I)/(1.-FVEL(I))  
               TCL_FAC_LINE(I) = ( 1.-(1.-FVOL(I))*FIC(I) )/( ABS(DVDR(I))*FVOR ) * SR/VMAX            
            ENDIF
         ELSE IF (RSCAL(I).LT.RFAR .AND. RSCAL(I).GE.ROUT) THEN
            FVEL(I)=CLFOUT
            IF (FVEL(I).GE.1.0D0) THEN 
               FVEL(I)=1.0D0
               TCL_FAC_LINE(I) = 0.0D0 
            ELSE
               FVOR = FVEL(I)/(1.-FVEL(I))  
               TCL_FAC_LINE(I) = ( 1.-(1.-FVOL(I))*FIC(I) )/( ABS(DVDR(I))*FVOR ) * SR/VMAX            
            ENDIF
         ELSE IF (RSCAL(I).LT.ROUT .AND. RSCAL(I).GE.RMID) THEN
            FVEL(I)=CLFMID
            IF (FVEL(I).GE.1.0D0) THEN 
               FVEL(I)=1.0D0
               TCL_FAC_LINE(I) = 0.0D0 
            ELSE
               FVOR = FVEL(I)/(1.-FVEL(I))  
               TCL_FAC_LINE(I) = ( 1.-(1.-FVOL(I))*FIC(I) )/( ABS(DVDR(I))*FVOR ) * SR/VMAX            
         ENDIF
      ELSE IF (RSCAL(I).LT.RMID .AND. RSCAL(I).GE.RIN) THEN
         FVEL(I)=CLFIN
         IF (FVEL(I).GE.1.0D0) THEN 
            FVEL(I)=1.0D0
            TCL_FAC_LINE(I) = 0.0D0 
         ELSE
            FVOR = FVEL(I)/(1.-FVEL(I))  
            TCL_FAC_LINE(I) = ( 1.-(1.-FVOL(I))*FIC(I) )/( ABS(DVDR(I))*FVOR ) * SR/VMAX            
         ENDIF
      ENDIF
      

      ! Note: this includes a factor SR/VMAX! 
      ! Also we don't do it in one wash-up, since default does not become tcl=0. then
   ENDDO

! HPOR LAST 
   RIN =  HPOR_FIELD(2) 
   RMID = HPOR_FIELD(3) 
   ROUT = HPOR_FIELD(4) 
   RFAR = HPOR_FIELD(5) 
! HPOR
   CLFIN =HPOR_FIELD(6)
   CLFMID=HPOR_FIELD(7)
   CLFOUT=HPOR_FIELD(8)
   CLFFAR=HPOR_FIELD(9)
   HPOR = 0.0
   TCL_FAC_CONT = 0.0 
   DO I=1,ND

      IF(RSCAL(I).GE.RFAR) THEN
         HPOR(I)=CLFFAR
         
      ELSE IF (RSCAL(I).LT.RFAR .AND. RSCAL(I).GE.ROUT) THEN
         HPOR(I)=CLFOUT
         
      ELSE IF (RSCAL(I).LT.ROUT .AND. RSCAL(I).GE.RMID) THEN
         HPOR(I)=CLFMID
         
      ELSE IF (RSCAL(I).LT.RMID .AND. RSCAL(I).GE.RIN) THEN
         HPOR(I)=CLFIN
      ENDIF
      TCL_FAC_CONT(I) = HPOR(I) * V(I) * SR 
!Jo times SR or times SRNOM?
      TCL_FAC_CONT(I) = TCL_FAC_CONT(I) * ( 1.-(1.-FVOL(I))*FIC(I) )

!JS-CHANGE: Jun -16, now included non-void ic correction properly 
! Assume velocity stretch porosity law as default
! Include scale-factor SR (gives porosity-length h 'real' units)
! and also results in consistent tau_cl 
   ENDDO

ELSE IF (NPARA_CLF.EQ.3) THEN

      !'standard' simple option, linearly increase to max/min value from onset in v/vinf 
      !NOTE: Is good e.g. in B/A-supergiant cases where velocity very low beyond tau=2/3, 
      !and will also be useful for consideration of optically thick winds 
      
      !FIC FIRST----------------------------------------------- 
      CLF=FIC_FIELD(1) 
      VCLSTART=FIC_FIELD(2)
      VCLMAX=FIC_FIELD(3)
      IF(VCLSTART.GT.VCLMAX) STOP ' VCLSTART > VCLMAX!'      
      DO I=1,ND
         IF(V(I).LT.VCLMAX) EXIT
      ENDDO
      IMAX=MAX0(1,I-1)
      DO I=IMAX,ND
         IF(V(I).LT.VCLSTART) EXIT
      ENDDO
      ISTART=I-1 ! works also for vclstart = 0. (completely clumped atmosphere)      
      IF(ISTART.LT.IMAX) THEN
         PRINT*,ISTART,IMAX
         STOP ' ISTART < IMAX IN CLUMPING'
      ENDIF
      ! photospheric clumping not allowed except for tests (OPT_PHOTCLUMP)
      IF(.NOT.OPT_PHOTCLUMP) THEN
         IF(ISTART.GE.NS)  THEN
            PRINT*,ISTART,NS,V(NS)
            STOP ' ISTART < NS IN CLUMPING'
         ENDIF
      ENDIF
      
      !value reached 
      FIC(1:IMAX)=CLF
      ! linear increase
      DO I=IMAX+1,ISTART
         FIC(I)=1.+(CLF-1.)/(VCLMAX-VCLSTART)*(V(I)-VCLSTART)
      ENDDO
      !Unclumped inner atmosphere (FIC=1.0)  
      FIC(ISTART+1:ND)=1.D0
      !-----------------------------------------------------
     !Volume filling factor  
      DO I=1,ND 
         IF (CLFAC(I).LE.1.0 .OR. FIC(I).GE.1.0) THEN 
            FVOL(I) = 1.0 
         ELSE
! bug created by JON, cured Sept 2018 (but still present in 10.4.7 till 10.5.1)
            FVOL(I) = (1.-FIC(I))**2 / ( CLFAC(I) - 2.*FIC(I) + FIC(I)**2 )
            !print*,fic(i),clfac(i),fvol(i) 
            IF (FVOL(I).GT.1.0) STOP 'FVOL >1, subroutine CLUMPING'
         ENDIF
      ENDDO

      !TCL_FAC_LINE AND FVEL NEXT----------------------------------------------- 
      CLF=FVEL_FIELD(1) 
      VCLSTART=FVEL_FIELD(2)
      VCLMAX=FVEL_FIELD(3)
      IF(VCLSTART.GT.VCLMAX) STOP ' VCLSTART > VCLMAX!'      
      DO I=1,ND
         IF(V(I).LT.VCLMAX) EXIT
      ENDDO
      IMAX=MAX0(1,I-1)
      DO I=IMAX,ND
         IF(V(I).LT.VCLSTART) EXIT
      ENDDO
      ISTART=I-1 ! works also for vclstart = 0. (completely clumped atmosphere)      
      IF(ISTART.LT.IMAX) THEN
         PRINT*,ISTART,IMAX
         STOP ' ISTART < IMAX IN CLUMPING'
      ENDIF
      ! photospheric clumping not allowed except for tests (OPT_PHOTCLUMP)
      IF(.NOT.OPT_PHOTCLUMP) THEN
         IF(ISTART.GE.NS)  THEN
            PRINT*,ISTART,NS,V(NS)
            STOP ' ISTART < NS IN CLUMPING'
         ENDIF
      ENDIF
      !value reached 
      FVEL(1:IMAX)=CLF
      ! linear increase
      DO I=IMAX+1,ISTART
         FVEL(I)=1.+(CLF-1.)/(VCLMAX-VCLSTART)*(V(I)-VCLSTART)
      ENDDO
      !Unclumped inner atmosphere (FIC=1.0)  
      FVEL(ISTART+1:ND)=1.D0
      !
      DO I=1,ND 
         IF (CLFAC(I).LE.1.0 .OR. FIC(I).GE.1.0) THEN 
            TCL_FAC_LINE(I) = 0.0D0 
         ELSE
            FVOR = FVEL(I)/(1.-FVEL(I))  
            TCL_FAC_LINE(I) = ( 1.-(1.-FVOL(I))*FIC(I) )/( ABS(DVDR(I))*FVOR ) * SR/VMAX 
         ENDIF
      ENDDO
      !-----------------------------------------------------

      !HPOR and TCL_FAC_CONT last--------------------------- 
      CLF=HPOR_FIELD(1) 
      VCLSTART=HPOR_FIELD(2)
      VCLMAX=HPOR_FIELD(3)
      IF(VCLSTART.GT.VCLMAX) STOP ' VCLSTART > VCLMAX!'      
      DO I=1,ND
         IF(V(I).LT.VCLMAX) EXIT
      ENDDO
      IMAX=MAX0(1,I-1)
      DO I=IMAX,ND
         IF(V(I).LT.VCLSTART) EXIT
      ENDDO
      ISTART=I-1 ! works also for vclstart = 0. (completely clumped atmosphere)      
      IF(ISTART.LT.IMAX) THEN
         PRINT*,ISTART,IMAX
         STOP ' ISTART < IMAX IN CLUMPING'
      ENDIF
      ! photospheric clumping not allowed except for tests (OPT_PHOTCLUMP)
      IF(.NOT.OPT_PHOTCLUMP) THEN
         IF(ISTART.GE.NS)  THEN
            PRINT*,ISTART,NS,V(NS)
            STOP ' ISTART < NS IN CLUMPING'
         ENDIF
      ENDIF
      !value reached 
      HPOR(1:IMAX)=CLF
      !Unclumped inner atmosphere (FIC=1.0)  
      HPOR(ISTART+1:ND)=0.D0
      ! linear increase between 
      DO I=IMAX+1,ISTART
         HPOR(I)=HPOR(ISTART+1)+(HPOR(IMAX)- HPOR(ISTART+1))/(VCLMAX-VCLSTART)*(V(I)-VCLSTART)
      ENDDO
      !
      DO I=1,ND 
         IF (CLFAC(I).LE.1.0 .OR. FIC(I).GE.1.0) THEN 
            TCL_FAC_CONT(I) = 0.0D0 
         ELSE
!Jo times SR or times SRNOM?
            TCL_FAC_CONT(I) = HPOR(I) * V(I) * SR
            TCL_FAC_CONT(I) = TCL_FAC_CONT(I) * ( 1.-(1.-FVOL(I))*FIC(I) )
         ENDIF
      ENDDO
      !-----------------------------------------------------

   ELSE 
      STOP 'THIS OPTHICK CLUMPING-OPTION NOT ALLOWED!, subroutine clumping' 
   ENDIF

ELSE
   
   DO I=1,ND 
      FIC(I)  = 1.0D0 
      FVEL(I) = 1.0D0 
      HPOR(I) = 0.0D0 
      FVOL(I) = 1.0D0
      TCL_FAC_LINE(I) = 0.0D0
      TCL_FAC_CONT(I) = 0.0D0
      !Switch off by setting all clump optical depths to zero 
   ENDDO

ENDIF
!
OPEN(24, FILE=TRIM(MODNAM)//'/CLUMPING_OUTPUT', STATUS='UNKNOWN')
!WRITE(24,fmt='(A)') '            r/Rstar          velo      abs(dvdr)           f_cl       f_ic           f_vel              hpor           f_vol           tcl_fac_line             tcl_fac_cont'
WRITE(24,fmt='(A)') '            r/Rstar          velo         abs(dvdr)&
&         f_cl       f_ic           f_vel              hpor&
&           f_vol           tcl_fac_line             tcl_fac_cont'
DO I=1,ND
   WRITE(24,333) I,RSCAL(I),V(I),ABS(DVDR(I)),CLFAC(I),FIC(I),FVEL(I),HPOR(I),FVOL(I),TCL_FAC_LINE(I),TCL_FAC_CONT(I)
ENDDO
333 FORMAT(i3,2x,4f15.5,2e15.5,2f15.5,2e25.10)
CLOSE(24) 

!stop ' testing'
!ALWAYS!!!
! check for representative value of clf
IF(CLF_REPRESENTATIVE.LT.MINVAL(CLFAC) &
&   .OR. CLF_REPRESENTATIVE.GT.MAXVAL(CLFAC)) THEN
    PRINT*,' REPRESENTATIVE VALUE OF CLUMPING FACTOR OUTSIDE ACTUAL RUN'
    PRINT*,CLF_REPRESENTATIVE
    DO I=1,ND
    PRINT*,I,R(I),V(I),CLFAC(I)
    ENDDO
    STOP ' REPRESENTATIVE VALUE OF CLUMPING FACTOR OUTSIDE ACTUAL RUN'
ENDIF

! finally, calculate overdensity
RHOCL=RHO*CLFAC

RETURN

END SUBROUTINE CLUMPING
!
!-----------------------------------------------------------------------
!
SUBROUTINE DERIVS(X,Y,DY)  

USE nlte_type
USE nlte_dim
USE nlte_opt, ONLY: OPT_PHOTCLUMP
USE nlte_var, ONLY: SR,SRVMIN,SRNOM,CONSTT,NSDIV, &
& IDUM,GGRAV,TEFF,SIGMAE=>SIGEM,CKAPPA,XT,A2TE,DELSIG,DELMU,DQDTAU,DELPARA,ITMIN

USE photstruc, ONLY: CLF_CONST

IMPLICIT NONE
!
!     delsig corrects for constant sigmae
!     delmu  corrects for constant mue
!     deli   corrects for recombination
!
!     .. scalar arguments ..
REAL(DP) ::  X  
!     ..
!     .. array arguments ..
REAL(DP) ::  DY(3),Y(3)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DELM,DELS,DY1,DY2,TERM0,TERM1,TERM2,DELI  
!     ..

DELS = .5D0* (DELSIG(IDUM-1)+DELSIG(IDUM))  
DELM = .5D0* (DELMU(IDUM-1)+DELMU(IDUM))  
DELI = .5D0* (DELPARA(IDUM-1)+DELPARA(IDUM))  

TERM0 = SIGMAE + CKAPPA*DELI/DELM*Y(1)*Y(2)** (-XT-1.D0)  

! note that ckappa includes sigmae
TERM1 = TERM0*DELS  

DY1 = GGRAV - 5.67D-5/2.9979D10*TEFF**4*TERM1  
! this is the ONLY hack to ensure that rho_phot => rho_phot/fcl
IF (OPT_PHOTCLUMP) DY1 = DY1/CLF_CONST

!with this statement we can check for grad
!if(idum.ge.38) print*,idum,5.67D-5/2.9979D10*TEFF**4*TERM1
DY(1) = DY1* (SRNOM/Y(3))**2  

TERM2 = DELS/Y(2)**3*TERM0  

DY2 = 3.D0/16.D0*TEFF**4*TERM2* (1.D0+DQDTAU(IDUM))  
DY(2) = DY2* (SRNOM/Y(3))**2  

! THIS IS NEW
IF(IDUM.LE.ITMIN) DY(2)=0.D0

DY(3) = -Y(2)/Y(1)*A2TE*DELM  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE DERIVS_KUR(X,Y,DY)  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: IDUM,A2TE,DELMU,ITMIN
USE photstruc

IMPLICIT NONE
!
!     delmu  corrects for constant mue
!
!     .. scalar arguments ..
REAL(DP) ::  X  
!     ..
!     .. array arguments ..
REAL(DP) ::  DY(3),Y(3)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DELM, DPDM, DTDM 
!     ..
INTEGER(I4B) :: I

DO I=1,NDMOD-1
  IF (X .GT. XMEXT(I).AND. X .LT. XMEXT(I+1)) EXIT
ENDDO
IF (I .EQ. NDMOD) THEN
!  IF(X.LT.XMEXT(1)) THEN
!    I=1
!  ELSE  
!    PRINT*,X,XMEXT(1),XMEXT(NDMOD)
    IF(X.LT.XMEXT(1)) PRINT*,' INCREASE MDOT!!!'
    STOP ' xm not found in external phot. struct.'
!  ENDIF
ENDIF

DPDM=(PKUR(I+1)-PKUR(I))/(XMEXT(I+1)-XMEXT(I))
DTDM=(TEXT(I+1)-TEXT(I))/(XMEXT(I+1)-XMEXT(I))

DELM = .5D0* (DELMU(IDUM-1)+DELMU(IDUM))  


DY(1) = DPDM


DY(2) = DTDM
! THIS IS NEW
IF(IDUM.LE.ITMIN) DY(2)=0.D0

DY(3) = -Y(2)/Y(1)*A2TE*DELM  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE DERIVS_TLU(X,Y,DY)  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: IDUM,A2TE,DELMU,ITMIN
USE photstruc

IMPLICIT NONE
!
!     delmu  corrects for constant mue
!
!     .. scalar arguments ..
REAL(DP) ::  X  
!     ..
!     .. array arguments ..
REAL(DP) ::  DY(3),Y(3)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DELM, DRHODM, DTDM 
!     ..
INTEGER(I4B) :: I

DO I=1,NDMOD-1
  IF (X .GT. XMEXT(I).AND. X .LT. XMEXT(I+1)) EXIT
ENDDO
IF (I .EQ. NDMOD) THEN
!STOP ' xm not found in external phot. struct.'
! extraplolation at own risk
  IF (X.GT.XMEXT(NDMOD)) THEN
    I=NDMOD-1  
  ELSE IF (X.LT.XMEXT(1)) THEN
    I=1  
  ELSE
    STOP ' something wrong with xm in external phot. struct.'
  ENDIF
ENDIF

DRHODM=(RHOTLU(I+1)-RHOTLU(I))/(XMEXT(I+1)-XMEXT(I))
DTDM=(TEXT(I+1)-TEXT(I))/(XMEXT(I+1)-XMEXT(I))

DELM = .5D0* (DELMU(IDUM-1)+DELMU(IDUM))  


DY(1) = DRHODM


DY(2) = DTDM
! THIS IS NEW
IF(IDUM.LE.ITMIN) DY(2)=0.D0

DY(3) = -1.D0/Y(1)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION RHOFUNC(R,V)  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: H1
IMPLICIT NONE
!
!     berechnet die dichte aus mloss und v
!
!     .. scalar arguments ..
REAL(DP) ::  R,V,RHOFUNC  
!     ..

RHOFUNC = H1/ (R**2*V)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION VFVL(R,RC,R0,SR,VINF,VMIN,BETA,HL)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     calculates velocities for analytical vel.law (hamann, 1985)
!
!     .. scalar arguments ..
REAL(DP) ::  BETA,HL,R,R0,RC,SR,VINF,VMIN,VFVL
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP  
!     ..

IF (R.LT.RC) THEN  
     VFVL = VMIN*EXP((R-SR)/HL)  
ELSE  
     VFVL = VINF*(1.D0-R0/R)**BETA  
END IF  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE VINDEX(R,V,RHO,M,I)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     verschiebt m indices ab index i um 1 nach oben
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  I,M  
!     ..
!     .. array arguments ..
REAL(DP) ::  R(ND1),RHO(ND1),V(ND1)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  J,K  
!     ..

DO J = 0,M  
     K = M + I + 1 - J  
     R(K) = R(K-1)  
     V(K) = V(K-1)  
     RHO(K) = RHO(K-1)
END DO  

RETURN  

END
!
!***********************************************************************
!
! subroutines: complex ones
! rateeq and related
!
!***********************************************************************
!
SUBROUTINE RATEEQ(XNH,XNE,TEMP,CLF,ILOW,IMAX,ND,ERR,CONCON,R,V,GRADV, &
&                  SR,VMAX,ERROLD,SPECMAT,ACCEL,OPTMET)

! UPATED FOR CLUMPING
! ILOW AND IMAX ARE EITHER LTE OR NLTE VALUES, DEPENDENT ON CALL

USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&    ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS,LABAT

USE nlte_var, ONLY: IONISST,IONIS,XNELTE, &
& ALEVEL,BLEVEL,ENIONND,ENIONND_LTE, &
& IQUAL, &
& MODNAM  

USE tcorr_var, ONLY : QCBFR,QCBFI,QRBFR,QRBFI, &
&                     DTQRBFR,DTQRBFI,DTQCBFH,DTQCBFC, &
&                     QCBBU,QCBBD,DTQCBBU,DTQCBBD, &
&                     OPTTCOR

IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
INTEGER(I4B), PARAMETER :: NREC = ID_LLEVS  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  SR,VMAX  
INTEGER(I4B) ::  ND  
LOGICAL ACCEL,CONCON,OPTMET
!     ..
!     .. array arguments ..
REAL(DP) ::  ERR(ND1),ERROLD(ND1),GRADV(ND1),R(ND1), &
&                 SPECMAT(5,ND1),TEMP(ND1),V(ND1),XNE(ND1),XNH(ND1),CLF(ND1)

INTEGER(I4B) ::  ILOW(ND1,KEL),IMAX(ND1,KEL)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AMPL,BINF,EN,ERR1,ERRN,OCC,OCCOLD,OCCOLDMAX, &
&                 SPECMAX,SPECMIN,SPECNEW,VELR,XMUST,TCL
INTEGER(I4B) ::  I,I1,I2,IILO,IIMA,J,JERR,K,LL,NUMION,NATO
!     ..
!     .. local arrays ..
REAL(DP) ::  ABUND(ID_ATOMS),BLEV0(ID_LLEVS),FL(ID_LLEVS), &
&                 FLX(ID_XLEVS),GL(ID_LLEVS),GLX(ID_XLEVS), &
&                 GS0(ID_SLEVS),GS1(ID_SLEVS),QD(ID_SLEVS), &
&                 RYD(ID_SLEVS),WEIGHT(ID_ATOMS),ZEFF(ID_ATOMS), &
&                 ZL(ID_LLEVS)

REAL(DP), DIMENSION(NREC,2) :: QRION,QRREC,QCION,QCREC,QLU,QLD
!     ..
!     .. external functions ..
INTEGER(I4B) ::  IGENIO  
EXTERNAL IGENIO  
!     ..
!     .. external subroutines ..
EXTERNAL FACTINC,NETMAT,UPDATE  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,MAX,MIN,SQRT  
!     ..

DEPTHLOOP: DO LL = 1,ND  

     ERR1 = 0.D0  
     READ (17,REC=LL) (BLEVEL(I),I=1,NREC)  
     READ (15,REC=LL) (ALEVEL(I),I=1,NREC)  

     DO I = 1,NREC  
!
!       !!! value from previous iteration
          NATO  = LE(I)  
!
          BLEV0(I) = BLEVEL(I)  
!
!       !!! set to zero to obtain only actually calculated values in
!           dependence of ilow,imax
!       ONLY FOR H,He if pure continuum

          IF(.NOT.CONCON) THEN
            IF (LABAT(NATO).EQ.'H' .OR. LABAT(NATO).EQ.'HE') THEN
              BLEVEL(I) = 0.D0
            ELSE
!---- use UPDATED LTE value for metals!!!
!     otherwise, blevel and enionnd reamain at their LTE value from it. 0,
!     and the boundary condition is deteriorated (=> opacont must be consistent)
              BLEVEL(I) = ALEVEL(I)
            ENDIF  
          ENDIF
     END DO
!
!---- calculation of limb-darkening coefficients
!
     IF (CONCON) THEN
       CALL FACTINC(R(LL),LL)  
! THIS IS NEW FROM V9.5 ON
       BLEVEL= 0.D0
     ENDIF

     VELR = V(LL)/R(LL)  
     XMUST = SQRT(1.D0-1.D0/R(LL)/R(LL))  

     IF(OPTTCOR) THEN
       QRION=0.
       QRREC=0.
       QCION=0.
       QCREC=0.
       IF(CONCON) THEN
         QLU=0.
         QLD=0.
       ENDIF  
     ENDIF  

KLOOP: DO K = 1,NAT
          IILO = ILOW(LL,K)  
          IIMA = IMAX(LL,K)  
          IF (IIMA.LT.IILO)  STOP ' iima < iilo' 
          IF(.NOT.CONCON.AND.LABAT(K).NE.'H'.AND.LABAT(K).NE.'HE') THEN
!-----use UPDATED enionnd for metals (see above)
            ENIONND(K,:,:) = ENIONND_LTE(K,:,:)
            GOTO 40
          ENDIF
!
!-----calculation of net bound-free rates and derivatives
!-----setup of nlte equations and newton-raphson iteration
!	  
          CALL NETMAT(K,IILO,IIMA,LL,XNE(LL),TEMP(LL),CLF(LL),BLEV0, &
&            CONCON, GRADV(LL),VELR,SR,VMAX,XMUST,ERROLD(LL),OPTMET, &
&            QRION,QRREC,QCION,QCREC,QLU,QLD) 
!
!-----calculation of enionnd(k,i,ll)
!
ILOOP:    DO I = IILO,IIMA  
               NUMION = IGENIO(K,I)  
               I1 = IFIRSL(NUMION)  
               I2 = IFIRSL(NUMION+1) - 1  
               EN = 0.D0  
!
!---  maxmimum occ.numbers (old)
!
               OCCOLDMAX = 0.D0  

               DO J = I1,I2 + 1  
                    OCCOLDMAX = MAX(OCCOLDMAX,BLEV0(J))  
               END DO

               OCCOLDMAX = 1.D0/OCCOLDMAX  

JLOOP:         DO J = I1,I2  
!
!     check for negative occup. no.
!
                    IF (BLEVEL(J).LE.0.D0) THEN  
                    PRINT *,LL, &
&                      ' WARNING: NEGATIVE OCCUP. NO AT LEVEL ',J
                         BLEVEL(J) = BLEV0(J)  
                    END IF

                    OCC = BLEVEL(J)  
                    OCCOLD = BLEV0(J)  
!---
!---     only those errors are considered, which are not contaminated
!---     by insufficient numerics
!---
                    IF (OCCOLD*OCCOLDMAX.LT.1.D-11) GO TO 30  
                    IF (IQUAL(J,LL).NE.0) GO TO 30  
                    ERRN = MAX(ERR1,ABS(1.D0-OCC/OCCOLD))  
                    IF (ERRN.NE.ERR1) JERR = J  
                    ERR1 = ERRN  

   30                     CONTINUE  
                    
                    EN = EN + BLEVEL(J)  

               END DO JLOOP
               ENIONND(K,I,LL) = EN  
          END DO ILOOP

          I1 = IFIRSL(NUMION+1)  

          OCC = BLEVEL(I1)
          
          IF (BLEVEL(I1).LE.0.D0) THEN  
               PRINT *,LL,' WARNING: NEGATIVE OCCUP. NO AT IMAX ',I1
               BLEVEL(I1) = BLEV0(I1)  
          END IF  

          OCCOLD = BLEV0(I1)  

          IF (OCCOLD*OCCOLDMAX.GT.1.D-11 .AND. IQUAL(I1,LL).EQ.0) THEN
               ERRN = MAX(ERR1,ABS(1.D0-OCC/OCCOLD))  
               IF (ERRN.NE.ERR1) JERR = I1  
               ERR1 = ERRN  
          END IF  
          IF (JERR.NE.I1 .AND. ACCEL) THEN  
!
!     extrapolation of occupation number
!
               IF (ERROLD(LL).GT.6.D-3) THEN  
                    SPECNEW = ERR1/ERROLD(LL)  
               ELSE  
                    SPECNEW = 0.D0  
               END IF  

               SPECMAX = MAX(SPECMAT(1,LL),SPECMAT(2,LL), SPECMAT( &
                3,LL),SPECMAT(4,LL),SPECMAT(5,LL), SPECNEW)
               SPECMIN = MIN(SPECMAT(1,LL),SPECMAT(2,LL), SPECMAT( &
                3,LL),SPECMAT(4,LL),SPECMAT(5,LL), SPECNEW)
               IF (SPECMAX.LT.1.D0 .AND. SPECMIN.GT..95D0) THEN  
                    AMPL = 1.D0/ (1.D0-SPECMIN)  
                    PRINT *  
                    PRINT *,'EXTRAPOLATION OF LEVEL ',JERR  
                    PRINT *,'EXTRAPOLATION OF LEVEL ',JERR  
                    PRINT *,'EXTRAPOLATION OF LEVEL ',JERR  
                    PRINT *  
                    BINF = BLEV0(JERR) + (BLEVEL(JERR)-BLEV0(JERR))* AMPL

                    IF (BINF.LT..1D0*BLEVEL(JERR)) BINF = .1D0*BLEVEL(JERR)

                    IF (BINF.GT.10.D0*BLEVEL(JERR)) BINF = 10.D0*BLEVEL(JERR)

                    BLEVEL(JERR) = BINF  
                    WRITE (99,FMT=*) 'EXTRAPOLATED VALUE',BINF  
                    ERR1 = ABS(1.D0-BINF/BLEV0(JERR))  
               END IF  
          END IF  
!
!---  last ionization stage assumed always as single level
!
          ENIONND(K,IIMA+1,LL) = BLEVEL(I1)  
!
!     not calculated ions explicitly set to zero
!
40          DO I = 1,IILO - 1  
               ENIONND(K,I,LL) = 0.D0  
          END DO

          DO I = IIMA + 2,KIS + 1 
               ENIONND(K,I,LL) = 0.D0  
          END DO

  END DO KLOOP

! update heating/cooling rates: at this place, since consistent
! nlte numbers have to be used
!(rate-coefficients from actual lte numbers!) 

! NOTE: since only those coefficients are populated which have
! transitions within (imin,imax), we can multiply with TOTAL vector blevel 
     IF(OPTTCOR) THEN
         QRBFR(LL)=DOT_PRODUCT(QRREC(:,1),BLEVEL)
       DTQRBFR(LL)=DOT_PRODUCT(QRREC(:,2),BLEVEL)
         QRBFI(LL)=DOT_PRODUCT(QRION(:,1),BLEVEL)
       DTQRBFI(LL)=DOT_PRODUCT(QRION(:,2),BLEVEL)
         QCBFR(LL)=DOT_PRODUCT(QCREC(:,1),BLEVEL)
       DTQCBFH(LL)=DOT_PRODUCT(QCREC(:,2),BLEVEL)
         QCBFI(LL)=DOT_PRODUCT(QCION(:,1),BLEVEL)
       DTQCBFC(LL)=DOT_PRODUCT(QCION(:,2),BLEVEL)
       IF(CONCON) THEN
            QCBBU(LL)=DOT_PRODUCT(QLU(:,1),BLEVEL)
          DTQCBBU(LL)=DOT_PRODUCT(QLU(:,2),BLEVEL)
            QCBBD(LL)=DOT_PRODUCT(QLD(:,1),BLEVEL)
          DTQCBBD(LL)=DOT_PRODUCT(QLD(:,2),BLEVEL)
       ENDIF
     ENDIF
       
     CALL UPDATE(XNH,XNE,TEMP,ILOW,IMAX,LL,NAT)  

     ERR(LL) = ERR1  
!     nnn=ll/6
!     nnuu=nnn*6
!     if(ll.eq.1.or.ll.eq.47) nnuu=ll
!     if(nnuu.eq.ll) then

     PRINT *  
     PRINT *,' MAXIMUM CHANGE IN OCCUPATION NUMBERS = ',ERR(LL)  
     WRITE (99,FMT=*) LL,' MAXIMUM CHANGE IN OCCUPATION NUMBERS = ',ERR(LL)
     PRINT *  
     PRINT *  
!
!     endif
!
     WRITE (17,REC=LL) (BLEVEL(I),I=1,NREC)  
     WRITE (27,REC=LL) (BLEVEL(I),I=1,NREC)  

END DO DEPTHLOOP  

OPEN (1,FILE=TRIM(MODNAM)//'/ENION',FORM='UNFORMATTED',STATUS='UNKNOWN')  
REWIND 1  
WRITE (1) ENIONND,XNE  
CLOSE (1)  
!print*,'xnecheck: enion written'
!print*,xne
RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FACTINC(RR,L)  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: AS,A2S,A3S,BS,B2S,B3S, &
& FRE,WFRE,HC,IFRE  

IMPLICIT NONE
!
!     below, find the description of the old approach no longer supported
!
!---- calculation of limb darkening factors ('blocking factors') for
!---- i_inc using momenta j,k,h,n
!              i_inc = as +bs * mue    for mu_*<mu<1
!              i_inc = a2s+b2s* mue    for  0 <mu<mu_*
!              i_inc = a3s+b3s* mue    for  -1<mue<0
!---- it has been changed (e. santolaya, october 92) so that a new
!---- phylosophy is applied: j, h, k and n are now vectors wich elements
!---- contain the radiation field at the different frecuency points
!---- at a certain depth point.
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  RR  
INTEGER(I4B) ::  L  
!     ..
!     .. local scalars ..
REAL(DP) ::  AA,BB,CC,DD,SMUE,SMUED,SMUEQ,SMUEZ,XXHN  
INTEGER(I4B) ::  I,LLL  
!     ..
!     .. local arrays ..
REAL(DP) ::  XXH(IFRETOT),XXJ(IFRETOT),XXK(IFRETOT), &
&                 XXN(IFRETOT)
!     ..
!     .. intrinsic functions ..
!INTRINSIC SQRT  
!     ..

READ (40,REC=L) (XXJ(I),I=1,IFRE)  
READ (42,REC=L) (XXK(I),I=1,IFRE)  
READ (41,REC=L) (XXH(I),I=1,IFRE)  
READ (43,REC=L) (XXN(I),I=1,IFRE)  

! This was the old approach

SMUE = SQRT(1.D0-1.D0/RR/RR)  
!
!---- smueq=(1-(mu_*)**2)**2
!
SMUEQ = 1.D0/RR**4  
SMUED = 3.D0 - SMUE**2  
SMUEZ = SMUE**2 - 2.D0  

DO LLL = 1,IFRE  

     XXHN = 24.D0*XXH(LLL) - 40.D0*XXN(LLL)  

     IF (SMUEQ.NE.0.D0) THEN  
          BB = XXHN/SMUEQ  
     ELSE  
          BB = XXHN*RR**4  
     END IF  

     CC = 6.D0*XXH(LLL) - BB*.5D0*SMUED  
     DD = BB*SMUE*SMUEZ + 24.D0*XXK(LLL) - 8.D0*XXJ(LLL)  
     AA = 0.5D0*SMUE*BB*SMUED + 6.D0*XXJ(LLL) - 12.D0*XXK(LLL)  

     AS(LLL) = (AA+BB)/2.D0  
     A2S(LLL) = (AA-BB)/2.D0  
     A3S(LLL) = A2S(LLL)  
     BS(LLL) = (CC+DD)/2.D0  

     IF (SMUE.NE.0.D0) THEN  
          B2S(LLL) = (AS(LLL)-A2S(LLL))/SMUE + BS(LLL)  
     ELSE  
          B2S(LLL) = 0.D0  
     END IF  

     B3S(LLL) = (CC-DD)/2.D0  

END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE NETMAT(NATO,ILO,IMA,LL,XNE,TEMP,CLF,BLEV0,CONCON,VGRAD, &
&                 VELR,SR,VMAX,XMUST,ERROLD,OPTMET, &
&                 QRION,QRREC,QCION,QCREC,QLU,QLD)

! UPDATED FOR CLUMPING
! ADDITIONAL CYCLE FOR EXPLICIT DR TRANSITIONS INCLUDED:
! ASSUMES THAT PARENTAL LEVEL IS GROUNDSTATE OF NEXT ION (KL!!!)

USE nlte_type
USE nlte_dim
USE fund_const
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&     ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD,DREXPLICIT,LKL,LABL, &
& LABAT,LABL3,LABL4,FRECIN,FRECFN,FLCBF

USE nlte_var, ONLY: ABUNDND, &
& IONISST,IONIS,XNELTE, &
& ALEVEL,BLEVEL,ENIONND, &
& MLOW,MUP,MMLOW,MMUP,MCLOW,MCUP, &
& INDEX4=>INDR,INDEX3=>INDC, &
& PRECIS,UNASOL,MEGAS,NIST,XNKK, &
& FRE,IFRE,OP_FLAG,FRECFN1,INDFRQ1

USE NLTE_XRAYS, ONLY: OPTXRAY, OPTAUGER, NAME, &
&               N_KEDGES, K_NMIN, K_NMAX, ETH, Z_K=>Z, N_K=>N, AUG_1_6

USE tcorr_var, ONLY: OPTTCOR

IMPLICIT NONE
!
!-------this subroutine calculates the net rates, setups the rate
!-------equation and solves them iteratively in a newton-raphson scheme
!-------starting with the populations obtained by solving 'a pelo' the
!-------system.
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
INTEGER(I4B), PARAMETER :: NDIM=ID_LLEVS  
INTEGER(I4B), PARAMETER :: NNREC=ID_LLEVS  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  ERROLD,SR,TEMP,VELR,VGRAD,VMAX,XMUST,XNE,CLF  
INTEGER(I4B) ::  ILO,IMA,LL,NATO  
LOGICAL CONCON, OPTMET  
!     ..
!     .. array arguments ..
REAL(DP), DIMENSION(NNREC) :: BLEV0
REAL(DP), DIMENSION(NNREC,2) :: QRION,QRREC,QCION,QCREC,QLU,QLD  
!     ..
!     ..
!     .. local scalars ..
REAL(DP) ::  BMAX,CASTA1,CASTA2,CIK,DEPART,DET,ERRMAX,SDP, &
&                 X1,X2,XNERAT,XNI,XNISTAR,XNK,XXLEV, &
&                 X3,X4,DTCIK,X5,X6,HHNU,CIK1,PROB
INTEGER(I4B) ::  I,III,ITM,J,JMAX,ML,MN,MU,MU1,N1,N2,NFIR,NLAS,NUMION, &
&                K,L,ITRANS,JEFF,NFM,NLM,JJ,IJ,LO
!     ..
!     .. local arrays ..
REAL(DP) ::  BOLD(NDIM),DELTAN(NDIM),F(NDIM),FLUD(NDIM), &
&            RAT2(NDIM,NDIM),RATMAT(NDIM,NDIM)

REAL(QP) :: AUX(NDIM)

INTEGER(I4B) ::  INDLUD(NDIM)  
!     ..
!     .. external functions ..
INTEGER(I4B) ::  IGENIO  
EXTERNAL IGENIO  
!     ..
!     .. external subroutines ..
EXTERNAL INTERMECOL,INTERMEPHO,LINESOB,LUBKSB,LUDCMP  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,MAX  
!     ..

ITM = 10  

NIST=0
XNERAT = XNE/XNELTE(LL)  
NUMION = IGENIO(NATO,ILO)  
NFIR = IFIRSL(NUMION)  

NUMION = IGENIO(NATO,IMA) + 1  
IF (NUMION.GT.IONG) STOP 'ERROR IN NETMAT - NUMION'  
NLAS = IFIRSL(NUMION)  

MN = NLAS - NFIR + 1  
!
!------ initialitation of rate matrix and solution vector
!
F(1:MN)=0.D0
RATMAT(1:MN,1:MN)=0.D0
!
!------ radiative b-f transitions
!
RBFLOOP: DO I = 1,INDEX4  

! standard treatment, all rbf-data, integration in INTERMEPHO
! starting at FRECIN (gs or excited state)  
  
     ML = MLOW(I)  
     MU = MUP(I)
       
     IF (ML.LT.NFIR .OR. ML.GE.NLAS) CYCLE  

     XNI = BLEV0(ML)

     IF (MU.GT.NLAS) THEN
!          this path refers to ionization to excited levels of imax+1
!          upper level assumed to be in lte with respect to nlas (groundstate of imax+1)
!          new formulation, see notes
          MU = NLAS  
     END IF  

     XNK = BLEV0(MU)  
     XXLEV = ALEVEL(ML)/ALEVEL(MU)*XNERAT  

     XNISTAR = XNK*XXLEV ! n_u *(n_i/n_u)* 
     NIST(ML)=XXLEV! witout nk
     XNKK(ML)=XNK
     DEPART = XNI/XNISTAR  
!
!-------   photo-rates (integration starting at INDFRQ(I),
!          corresponding to FRECIN)
!
     CALL INTERMEPHO(I,DEPART,XNISTAR,LL,X1,X2,XNE,TEMP,CLF,OPTMET, &
&                    X3,X4,X5,X6)

! all rates entering the energy equation are with respect to overdense clumps
! and according occupation numbers (NOT AVERAGED!) 

     IF(OPTTCOR) THEN
       HHNU = HH*FL(ML) ! to be consistent with LTE    
       IF(HHNU .LE. 0.D0) STOP 'ERROR IN HNU - RBF'

!      dtrbfc(n1) = XNK*XXLEV*X6 !does presently not work at low temp. Why?
       QRREC(MU,1) = QRREC(MU,1) + X4*XXLEV ! (*n_k)
       QRION(ML,1) = QRION(ML,1) + X3       ! (*n_i)
       QRREC(MU,2) = QRREC(MU,2) + XXLEV*(X6 - X4*(1.5D0 + &
                 HHNU/AKB/TEMP)/TEMP)       ! (*n_k)
       QRION(ML,2) = QRION(ML,2) - X3*(1.5D0 + HHNU/AKB/TEMP)/TEMP ! (*n_i), minus!
     ENDIF

     N1 = ML - NFIR + 1  
     N2 = MU - NFIR + 1  

     IF (N2.GT.MN .OR. N1.LT.0) STOP 'ERROR IN NETMAT, N'  

!
!----------output (last iteration)
!
     IF (UNASOL .AND. MEGAS) THEN  
          CASTA1 = XNI*X1
          CASTA2 = XNK*X2*XXLEV  
          WRITE (22,FMT=9000) LL,'RBF',ML,MU,CASTA1,CASTA2, &
           CASTA2 - CASTA1
! NIST*RIK, NIST*RKI, NIST*RKI-DEP*NIST*RIK
!          WRITE (22,FMT=9000) LL,'RBF',ML,MU,CASTA1*NIST(ML)/XNI,CASTA2/XNK, &
!           (CASTA2 - CASTA1)/XNK
     END IF  
!
!----------set up of matrix
!
     IF(X1.EQ.0.D0.OR.X2*XXLEV.EQ.0.D0) STOP 'INTERMEPHO'
     
!     if(nato.eq.3)print*,ll,n1,n2,x1,x2,depart,xxlev
     RATMAT(N1,N1) = RATMAT(N1,N1) - X1  
     RATMAT(N1,N2) = RATMAT(N1,N2) + X2*XXLEV ! ok, since with respect to n_k (and not n_u) 
!
!---makes only sense for intermediate ions, couples lower to upper ones
!
     RATMAT(N2,N1) = RATMAT(N2,N1) + X1  
     RATMAT(N2,N2) = RATMAT(N2,N2) - X2*XXLEV  

     MU1=KL(ML)
     IF(.NOT.OP_FLAG(I) .OR. MU.EQ.MU1) CYCLE RBFLOOP

! additional treatment for OP-data with ionization to excited states
! (resonances between FRECFN and FRECIN) 

! consistency tests 
     IF(FRECFN(I).EQ.FRECIN(I)) THEN
       PRINT*,LABL(LABL4(I)),FRECIN(I),FRECFN(I),FRECFN1(I)
       STOP ' PROBLEM IN ADDITIONAL PATH OF RBF-TREATMENT (OP-DATA)'
     ENDIF
     
     IF(INDFRQ1(I).EQ.0) &
       STOP ' INDFRQ1 = 0 IN ADDITIONAL PATH OF RBF-TREATMENT (OP-DATA)'
     
! upper state now ground-state of next ion
! (this is an approximation, particularly  when excited state = 3 or higher
! -- in the latter case, upper level could be the 2nd state etc.)     
     MU = MU1
     IF (MU.GT.NLAS) STOP ' MU > NLAS IN ADDITIONAL PATH OF RBF-TREATMENT (OP-DATA)'

     XNK = BLEV0(MU)  
     XXLEV = ALEVEL(ML)/ALEVEL(MU)*XNERAT  

     XNISTAR = XNK*XXLEV ! n_u *(n_i/n_u)* 
     NIST(ML)=XXLEV! witout nk
     XNKK(ML)=XNK
     DEPART = XNI/XNISTAR  
!
!-------   additional photo-rates (integration starting at INDFRQ1(I),
!          corresponding to FRECFN1, until FRECIN; analogous to INTERMEDR)
!
!          no rates for heating and cooling (already treated above)
     CALL INTERMEPHO1(I,DEPART,XNISTAR,LL,X1,X2,XNE,TEMP,CLF,OPTMET)
     
     N2 = MU - NFIR + 1  

     IF (N2.GT.MN) STOP 'ERROR IN NETMAT, N2 (additional rbf-path)'  

!
!----------output (last iteration)
!
     IF (UNASOL .AND. MEGAS) THEN  
          CASTA1 = XNI*X1
          CASTA2 = XNK*X2*XXLEV  
          WRITE (22,FMT=9001) LL,'RBF1',ML,MU,CASTA1,CASTA2, &
           CASTA2 - CASTA1
! NIST*RIK, NIST*RKI, NIST*RKI-DEP*NIST*RIK
!          WRITE (22,FMT=9000) LL,'RBF',ML,MU,CASTA1*NIST(ML)/XNI,CASTA2/XNK, &
!           (CASTA2 - CASTA1)/XNK
     END IF  
!
!----------set up of matrix
!
     IF(X1.EQ.0.D0.OR.X2*XXLEV.EQ.0.D0) STOP 'INTERMEPHO (additional path)'
     
! now additional rates (de-) populating gs of next ion (analogous to INTERMEDR)
!     if(nato.eq.3)print*,ll,n1,n2,x1,x2,depart,xxlev
     RATMAT(N1,N1) = RATMAT(N1,N1) - X1  
     RATMAT(N1,N2) = RATMAT(N1,N2) + X2*XXLEV ! ok, since with respect to n_k (and not n_u) 
!
!---makes only sense for intermediate ions, couples lower to upper ones
!
     RATMAT(N2,N1) = RATMAT(N2,N1) + X1  
     RATMAT(N2,N2) = RATMAT(N2,N2) - X2*XXLEV       
     
END DO RBFLOOP  
!
!
!------ explicit DR-transitions
!
DRLOOP: DO I = 1,INDEX4  
     IF(DREXPLICIT(I).EQ.0) CYCLE
     ML = MLOW(I)  
     MU = KL(ML) ! w.r.t. IONIC GROUND-STATE!
     
     IF (ML.LT.NFIR .OR. ML.GE.NLAS) CYCLE  

     IF (MU.GT.NLAS) THEN
!SHOULD NEVER HAPPEN. THUS
         STOP ' UPPER STATE OF DR TRANSITION NOT GROUND-STATE'
     END IF  

     XXLEV = ALEVEL(ML)/ALEVEL(MU)*XNERAT  
!
!-------   DR-rates
!
     CALL INTERMEDR(I,LL,X1,X2,TEMP)
!
!    HEATING COOLING RATES NOT INCLUDED SO FAR
!
     N1 = ML - NFIR + 1  
     N2 = MU - NFIR + 1  

     IF (N2.GT.MN .OR. N1.LT.0) STOP 'ERROR IN NETMAT, N'  
!
!----------output (last iteration)
!
     IF (UNASOL .AND. MEGAS) THEN  
          XNI = BLEV0(ML)
          XNK = BLEV0(MU)  
          CASTA1 = XNI*X1  
          CASTA2 = XNK*X2*XXLEV  
          WRITE (22,FMT=9000) LL,'RDR',ML,MU,CASTA1,CASTA2, &
           CASTA2 - CASTA1
! NIST*RIK, NIST*RKI, NIST*RKI-DEP*NIST*RIK
!          WRITE (22,FMT=9000) LL,'RDR',ML,MU,CASTA1*XXLEV/XNI,CASTA2/XNK, &
!           (CASTA2 - CASTA1)/XNK
     END IF  
!
!----------set up of matrix
!
     IF(X1.EQ.0.D0.OR.X2*XXLEV.EQ.0.D0) STOP 'INTERMEDR'
     
     RATMAT(N1,N1) = RATMAT(N1,N1) - X1  
     RATMAT(N1,N2) = RATMAT(N1,N2) + X2*XXLEV ! ok, since with respect to n_k (and not n_u) 
!
!---makes only sense for intermediate ions, couples lower to upper ones
!
     RATMAT(N2,N1) = RATMAT(N2,N1) + X1  
     RATMAT(N2,N2) = RATMAT(N2,N2) - X2*XXLEV  
END DO DRLOOP

!---
!--- Auger-ionization
!--- in its present form (see table), only for K-shell ionization
!--- NOT only for two ejected electrons, but generalized

IF (OPTXRAY.AND.OPTAUGER) THEN
  DO K=1,30
    IF(NAME(K).EQ.LABAT(NATO)) GOTO 5
  ENDDO
  STOP ' LABAT(NATO) NOT FOUND IN NAME' 

  5 CONTINUE
  
  AUGERLOOP: DO J = ILO,IMA  
  JEFF=J+ZEFF(NATO)
    
  IF(JEFF.LT.K_NMIN .OR. JEFF.GT.K_NMAX) CYCLE AUGERLOOP
  
  DO L=1,N_KEDGES
    IF(Z_K(L).EQ.K .AND. N_K(L).EQ.JEFF) GOTO 10
  ENDDO
! ELEMENT (OR ION) NOT PRESENT IN K_SHELL_AUGER_DATA
  CYCLE AUGERLOOP

10 ITRANS=L
  IF(ETH(ITRANS).GE.FRE(IFRE)) CYCLE AUGERLOOP
  
  CALL INTERMEPHO_AUGER(X1,ITRANS,LL)   

  NUMION=IGENIO(NATO,J)
  NFM=IFIRSL(NUMION)
  NLM=IFIRSL(NUMION+1)-1
  
  
  DO ML=NFM,NLM

! LOOP OVER ALL PROBABILITIES (1,2,3 ETC. EJECTED ELECTRONS)    
PROBA: DO JJ=J+1,IMA+1    
       MU=IFIRSL(IGENIO(NATO,JJ))
       N1 = ML - NFIR + 1  
       N2 = MU - NFIR + 1  
! E.G., IF JJ=J+1, WE HAVE JJ-J = 1 EJECTED ELECTRONS, FOR
!          JJ=J+2  WE HAVE JJ-J = 2 EJECTED ELECTRONS (TYPICAL CASE), AND SO ON
       PROB=AUG_1_6(ITRANS,JJ-J)
       IF(PROB.EQ.0.D0) CYCLE PROBA
       
! FOR TESTS
!       IF(LL.EQ.1) PRINT*,K,J,ITRANS,JJ,ML,MU,N1,N2,PROB,X1
!
!----------output (last iteration)
!
       IF (UNASOL .AND. MEGAS) THEN  
          XNI = BLEV0(ML)
          CASTA1 = XNI*X1*PROB
          CASTA2 = 0.
          WRITE (22,FMT=9000) LL,'AUG',ML,MU,CASTA1,CASTA2, &
           CASTA2 - CASTA1
       END IF  
! only upwards rates
       RATMAT(N1,N1) = RATMAT(N1,N1) - X1*PROB  
       RATMAT(N2,N1) = RATMAT(N2,N1) + X1*PROB  
    ENDDO PROBA
  ENDDO  
ENDDO AUGERLOOP
ENDIF !AUGER

!
!------- collisional b-f transitions
!
CBFLOOP: DO I = 1,INDEX3  
     ML = MCLOW(I)  
     MU = MCUP(I)  

     IF (ML.LT.NFIR .OR. ML.GE.NLAS) CYCLE

     XNI = BLEV0(ML)  

     IF (MU.GT.NLAS) THEN  
!          this path refers to ionization to excited levels of imax+1
!          upper level assumed to be in lte with respect to nlas (groundstate of imax+1)
!          see above
          MU = NLAS  
     END IF  
     XNK = BLEV0(MU)  
     XXLEV = ALEVEL(ML)/ALEVEL(MU)*XNERAT  

! for tests
     IJ=IGENIO(LE(MU),LI(MU))       
     LO=IFIRSL(IJ)
     IF(LO.EQ.MU) THEN
! frequencies can be different if actual upper level > NLAS
! in this case, use actual frequency (FLCBF(I)),
! but add rate ground-state of imax+1 
       IF(FLCBF(I).NE.FL(ML) .AND. MU.EQ.MCUP(I)) THEN
         PRINT*,FLCBF(I),FL(ML),labl3(i),ml,mu,ilo,ima,ll
         STOP ' PROBLEMS WITH GROUND-STATE IONIZATION FREQUENCY (CBF)'
       ENDIF
     ENDIF  
     
     CALL INTERMECOL(I,TEMP,CIK,MCUP(I),FL,LE,ZL)  
     CIK = CIK*XNE  ! includes clumping

     IF(OPTTCOR) THEN
       CALL INTERMECOL(I,TEMP*1.01,CIK1,MCUP(I),FL,LE,ZL)  
! partial derivative
       DTCIK = (cik1*XNE-cik)/(0.01*temp) 
!JO CHANGED DEC. 2016
!       HHNU = HH*FL(ML) ! to be consistent with LTE            
       HHNU = HH*FLCBF(I)
       IF(HHNU .LE. 0.D0) STOP 'ERROR IN HNU - CBF'     

       QCREC(MU,1) = QCREC(MU,1) + HHNU*XXLEV*CIK ! (*n_k)
       QCION(ML,1) = QCION(ML,1) + HHNU*CIK       ! (*n_i)
!      DTQCBFH(LL,NATO) = DTQCBFH(LL,NATO) + HHNU*XNK*XLEVEL*DTCIK !does not work
       QCREC(MU,2) = QCREC(MU,2) + HHNU*XXLEV*CIK  &
&    	*(DTCIK/CIK - (3.D0/2.D0 + HHNU/AKB/TEMP)/TEMP) ! (*n_k)
!Here comes the question! Why does the first formulation give a better
!flux, however leads to a destabilization, whereas the 2nd one give a stable
!convergence?
!       DTQCBFC(LL,NATO) = DTQCBFC(LL,NATO) + HHNU*XNI*DTCIK
       QCION(ML,2) = QCION(ML,2) + HHNU*CIK  &
&    	*(DTCIK/CIK - (3.D0/2.D0 + HHNU/AKB/TEMP)/TEMP) ! (*n_i)!

     ENDIF
     
     N1 = ML - NFIR + 1  
     N2 = MU - NFIR + 1  
     IF (N2.GT.MN .OR. N1.LT.0) STOP 'ERROR IN NETMAT, N'  
!
!----------output (last iteration)
!
     IF (UNASOL .AND. MEGAS) THEN  
          CASTA1 = XNI*CIK  
          CASTA2 = XNK*CIK*XXLEV  
          WRITE (22,FMT=9000) LL,'CBF',ML,MU,CASTA1,CASTA2, &
           CASTA2 - CASTA1
! NIST*CIK, NIST*CIK, NIST*CIK-DEP*NIST*CIK, with CKI=(NI/NK)_star*CIK
!          WRITE (22,FMT=9000) LL,'CBF',ML,MU,CASTA1*NIST(ML)/XNI,CASTA2/XNK, &
!           (CASTA2 - CASTA1)/XNK
     END IF  
!
!----------set up of matrix
!
     if(cik.eq.0.d0.or.xxlev.eq.0.d0) stop ' intermecoll'
!     if(nato.eq.3)print*,ll,n1,n2,cik,depart,xxlev
     RATMAT(N1,N1) = RATMAT(N1,N1) - CIK  
     RATMAT(N1,N2) = RATMAT(N1,N2) + CIK*XXLEV  
!
!---makes only sense for intermediate ions, couples lower to upper ones
!
     RATMAT(N2,N1) = RATMAT(N2,N1) + CIK  
     RATMAT(N2,N2) = RATMAT(N2,N2) - CIK*XXLEV  
        
END DO CBFLOOP  
!
!-----  line treatment
!
IF(CONCON) GOTO 140
!
!       changed, improved approach (see linesob)
!------ last row of ratmat is explicitly overwritten (pure cont.) 
!
!DO I = 1,MN  
!     RATMAT(MN,I) = 1.D0  
!END DO  
!
!F(MN) = ABUNDND(NATO,LL)  
!
JMAX=MN
BMAX=0.

DO I=1,MN
     J=I+NFIR-1
     BOLD(I)=BLEV0(J)
     IF(BOLD(I).GT.BMAX) THEN
          BMAX=BOLD(I)
          JMAX=I
     ENDIF
END DO

DO I=1,MN
     RATMAT(JMAX,I)=1.D0
END DO
        
F(JMAX)=ABUNDND(NATO,LL)


FLUD(1:MN) = F(1:MN)
RAT2(1:MN,1:MN) = RATMAT(1:MN,1:MN) 

!
!-----solution of rate equations
!
CALL LUDCMP(RATMAT,MN,NDIM,INDLUD,DET)  
!
CALL LUBKSB(RATMAT,MN,NDIM,INDLUD,F)  
!
!-----------------------------------------------------------------
!--------newton raphson improvement-cycle-------------------------
!-----------------------------------------------------------------
!
ERRIT: DO III = 1,ITM  

! old version (if no qp precision available)
!     DO I = 1,MN  
!          SDP = -FLUD(I)  
!
!     high precision summation
!
!          DO J=1,MN
!               AUX(J)=RAT2(I,J)*F(J)
!          END DO
!
!          CALL SORT1(MN,AUX)     
!
!          DO J=1,MN
!               SDP=SDP+AUX(J)
!          END DO
!
!          DELTAN(I) = SDP  
!     END DO

!    new version (residuum calculated in qp precision
     AUX(1:MN)=MATMUL(REAL(RAT2(1:MN,1:MN),QP),REAL(F(1:MN),QP))-REAL(FLUD(1:MN),QP)
     DELTAN(1:MN)=AUX(1:MN)
     CALL LUBKSB(RATMAT,MN,NDIM,INDLUD,DELTAN)
     
     ERRMAX = 0.D0  

     DO I = 1,MN  
          F(I) = F(I) - DELTAN(I)  
          ERRMAX = MAX(ERRMAX,ABS(DELTAN(I)/F(I)))  
     END DO
!
!     print*,'errmax: ',ll,errmax
!
     IF (ERRMAX.LT.PRECIS) THEN  
          DO I = 1,MN  
               J = I + NFIR - 1  
               BLEVEL(J) = F(I)  
          END DO
          GO TO 140  
     END IF  

END DO ERRIT  

DO I = 1,MN  
     J = I + NFIR - 1  
     BLEVEL(J) = F(I)  
END DO  

IF(ERRMAX.GT.1.D-3) THEN
   PRINT *,'WARNING!! N-R CONVERGENCE NOT ACHIEVED IN ',III, &
&  ' ITERATIONS BY',ERRMAX,' (NETMAT)'
   PRINT *,'ATOM: ',NATO,'   DEPTH: ',LL  
! EMERGENCY
  IF(ERRMAX.GT.0.5) THEN
  DO I = 1,MN  
     J = I + NFIR - 1  
     BLEVEL(J) = BLEV0(I)
   ENDDO
  ENDIF 
ENDIF

  140 CONTINUE  

IF (CONCON) CALL LINESOB(LL,VELR,VGRAD,NFIR,NLAS,BLEV0,RATMAT,NDIM, &
&                         XNE,SR,VMAX,NATO,XMUST,TEMP,ERROLD,QLU,QLD)

RETURN  

 9000 FORMAT (1X,I2,3X,A3,2 (3X,I3),3 (2X,G16.6))  
 9001 FORMAT (1X,I2,3X,A4,2X,I3,3X,I3,3 (2X,G16.6))  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE INTERMECOL(NTR,TEMP,CIK,MMCUP,FL,LE,ZL)  

USE nlte_type
USE nlte_dim
USE fund_const
USE princesa_var, ONLY: DATA,NDAT, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
&       LABL4,LABL1,LABU1,LABL3,IPARE5,FLCBF

IMPLICIT NONE
!
!---- it calculates collisional bound-free coeffs.
!---- used (and checked so far)
!
!------ formula 1 : simple cbf coefficient, a a la Mihalas (for Adi's input)
!                   as formula 15, but with GQ as input
!------ formula 10: h i (all levels without level 2)
!------ formula 11: h i (level 2), he ii
!------ formula 12: he ii
!------ formula 14: he i
!------ formula 15: he ii, si ii,iii,iv
!------ formula 17: n iii  (Feb. 2006: bug found by JO) 
!------ formula 20: mg ii
!------ formula 21: he i
!
!JO: changed Dec. 2016, to account for ionization to excited levels
!
!     .. scalar arguments ..
REAL(DP) ::  CIK,TEMP  
INTEGER(I4B) ::  MMCUP,NTR  
!     ..
!     .. array arguments ..
REAL(DP) ::  FL(ID_LLEVS),ZL(ID_LLEVS)  
INTEGER(I4B) ::  LE(ID_LLEVS)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AA1,ANON,APAR,GAMMA,GQ,TLOG,USU1,USU2,USUB0,XNU0, &
&                 Z,XXX,E(10)
INTEGER(I4B) ::  IDA,IFORM,INTEG,IZ,M,NC,NDATOS,NNF,NNL,NN,N,I
LOGICAL QCL,QDB,QRT  
!     ..
!     .. local arrays ..
REAL(DP) ::  DL(ID_NLONG),FRECFN(ID_RBFTR),FRECIN(ID_RBFTR)  
INTEGER(I4B) ::  IFPTR(ID_NTTRD),NFPTR(ID_NTTRD)  
!     ..
!     .. external functions ..
REAL(DP) ::  EXPIN1  
EXTERNAL EXPIN1  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP,INT,LOG10,SQRT  
!     ..

AA1 = 5.465727564D-11  
NNL = LABL3(NTR)  
!XNU0 = FL(NNL)  
XNU0 = FLCBF(NTR)
NNF = IFORM3(NTR)  


! thus far, we only allow for ionization to excited states for formula 1 and 15
IF(NNF.NE.1 .AND. NNF.NE.15) THEN
  IF(XNU0.NE.FL(NNL)) STOP ' CBF FORMULA 17 AND IONIZATION TO EXCITED LEVEL'
ENDIF

USUB0 = HH*XNU0/AKB/TEMP  
IDA = INDAT3(NTR)  


Z = ZL(NNL) + 1.D0  

IZ = INT(Z+0.5D0)  

IF (NNF.EQ.1) THEN
     CIK=1.55D13*DATA(IDA)*DATA(IDA+1)*EXP(-USUB0)/(SQRT(TEMP)*USUB0)

ELSE IF (NNF.EQ.11) THEN  
     GAMMA = DATA(IDA)/ (TEMP**2) + DATA(IDA+1)/TEMP + DATA(IDA+2) &
      + DATA(IDA+3)*TEMP + DATA(IDA+4)*TEMP**2

     CIK = AA1*GAMMA*EXP(-USUB0)*SQRT(TEMP)  

ELSE IF (NNF.EQ.12) THEN  
     TLOG = LOG10(TEMP)  
     GAMMA = DATA(IDA)/ (TLOG**2) + DATA(IDA+1)/TLOG + DATA(IDA+2) &
      + DATA(IDA+3)*TLOG + DATA(IDA+4)*TLOG**2

     CIK = AA1*GAMMA*EXP(-USUB0)*SQRT(TEMP)  

ELSE IF (NNF.EQ.13) THEN  
     GAMMA = (DATA(IDA)/Z)**4  

     CIK = AA1*GAMMA*EXP(-USUB0)*SQRT(TEMP)  

ELSE IF (NNF.EQ.14) THEN  
     USU1 = USUB0 + 0.27D0  
     USU2 = USUB0 + 1.43D0  
     APAR = USUB0*EXPIN1(USUB0)*EXP(-USUB0) - 0.728D0*USUB0**2/ &
      USU1*EXPIN1(USU1)*EXP(-USU1)
     ANON = 0.189D0*USUB0**2*EXP(-USUB0)* (2.D0+USU2)/USU2**3  

     CIK = (APAR-ANON)*DATA(IDA)*AA1/SQRT(TEMP)  

ELSE IF (NNF.EQ.15) THEN  

     IF (IZ.EQ.1) THEN  
          GQ = .1D0  
     ELSE IF (IZ.EQ.2) THEN  
          GQ = .2D0  
     ELSE  
          GQ = .3D0  
     END IF  

     CIK = 1.55D13*EXP(-USUB0)*DATA(IDA)*GQ/USUB0/SQRT(TEMP)  

ELSE IF (NNF.EQ.17) THEN ! checked by JO, bug identified, changed
!                         there is still an inconsistency between
!                         the Detail program and the manual.
!                         original source by Moore 72 need to be checked  
     N=INT(DATA(IDA)+0.5D0)
     GQ=3.039652336D-16*XNU0                !  nu/nu_hyd, d.h. Ei in Ryd
     USU1=HH*CLIGHT/AKB/TEMP*1.097373177D5  ! = h/kt * nu_hyd = u_ion for hyd.
     E(1)=EXPIN1(USUB0)*EXP(-USUB0)
     DO I=2,N
     	E(I)=(EXP(-USUB0)-USUB0*E(I-1))/(FLOAT(I-1))     
     ENDDO
     GAMMA=0.D0
     DO I=1,N
        GAMMA=GAMMA+E(I)*DATA(IDA+2+I)
     ENDDO
     GAMMA=GAMMA*USUB0
     GAMMA=GAMMA+DATA(IDA+1)*E(1)+DATA(IDA+2)*EXP(-USUB0)
     GAMMA=GAMMA*USU1/GQ    ! usu1/gq = u_hyd/(nu/nu_hyd) = uh/ei 
     
     CIK=AA1*SQRT(TEMP)*GAMMA

ELSE IF (NNF.EQ.20) THEN ! taken from DETAIL, not checked by JO
     
     GAMMA = HH*CLIGHT/AKB/TEMP*1.097373177D5/USUB0
     USU1 = USUB0+DATA(IDA+2)
     GAMMA = GAMMA**2*(DATA(IDA)*USUB0*EXPIN1(USUB0)*EXP(-1.D0*USUB0) + &
&     	    DATA(IDA+1)*(USUB0/USU1)**2*(EXPIN1(USU1)*EXP(-1.D0*USU1) + &
&           EXP(-1.D0*USU1)))
     CIK = AA1*SQRT(TEMP)*GAMMA

ELSE IF (NNF.EQ.21) THEN  
         NN=INT(DATA(IDA)+5.D-1)
!------- gamma is calculated with electron density equals to unity
!------- because is scaled by it in netmat
         XXX=1.D0
         CALL HECION(TEMP,XXX,NN,GAMMA)
         CIK=GAMMA

ELSE  
     PRINT *,'TRANSITION CBF NOT IMPLEMENTED, NFOR=',NNF  
     STOP  

END IF  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE INTERMEPHO(NTR,DEPART,XNISTAR,LL,X1,X2,XNE,TEMP,CLF,OPTMET, &
&           X3,X4,X5,X6)

!----  several quantities 'back-corrected' here, since 
!----  OPAC variable EFFECTIVE opacity. See comments.  

!----  comment by JO (Jan. 2016):
!      the scattering component of ALO (pure Thomson so far)
!      might need to be updated regarding the line pseudo continuum
!      (note that this refers to the 'old' approach, i.e.,
!       it is an old problem)
!
USE nlte_type
USE nlte_dim
USE fund_const
USE princesa_var, ONLY: LABL, LABL4
USE nlte_var, ONLY: ALPHAFS, &
& FRE1=>FREF,WFRE1=>WFREF,INDFIN,INDFRQ,IFRE1=>IFREF, &
& FRE,WFRE,HC,IFRE, &
& OPAC,ST=>STRUE,XJ,ALO, &
& OPAT_M_OLD,STRUE_M_OLD, STRUE_M_NEW, METALS_CONVERGED,OP_FLAG

USE nlte_porvor, ONLY: OPA_EFF_RAT, OPA_EFF_RAT_OLD

IMPLICIT NONE
!
!---  new algorithm
!---  subroutine intemepho (see netmat). it calculates photo-integrals
!---  for some the formulae implemented in detail.
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
INTEGER(I4B), PARAMETER :: IFREFIN=ID_FREC2  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  DEPART,TEMP,X1,X2,XNE,X3,X4,X5,X6,EDGENU,XNISTAR,CLF 
INTEGER(I4B) ::  LL,NTR  
LOGICAL OPTMET,OPTMET1
!     ..
!     .. local scalars ..
REAL(DP) ::  ALOVTH,ALPHA,AUX1,AUX2,DEX,DFREQ,DFX,DR,EXK,F,F1, &
&                 F2,FACF,FACG,FGCONST,FREQ,FREQII,FREQII1,G,G1,G2, &
&                 HCFRE3,HCKT,HNUE1,OPA,PI4,RERR,SIG,SUM1,SUM2, &
&                 THOM,THOMALO,WF,XALO,XF,XF1,XF2,XFREQ,XFREQ1,XG, &
&                 XG1,XG2,XXJ,BETAR, &
&                 SUM3,SUM4,SUM5,SUM6, &
&                 XXJJ,AUX11,FF,GG,FF1,FF2,GG1,GG2, &
&                 XFF1,XGG1,XFF2,XGG2,FACFF,FACGG,XFF,XGG         
INTEGER(I4B) ::  II,II1,IIOLD,ISTART,KK
!     ..
!     .. external subroutines ..
EXTERNAL CROSSBF  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,EXP,LOG10,MAX  
!     ..

OPTMET1=OPTMET
IF(OPTMET1.AND.METALS_CONVERGED) OPTMET1=.FALSE.

HCKT = HKL/TEMP  
!
!       begin of integration of the four photo-integrals
!       (for definition, see the santolaya et al. paper)
!       note: integration over frequency, fre in kayser
!
IIOLD = 0  
RERR = 1.D-5  
SIG = AMH*SIGMAE  
ISTART = INDFRQ(NTR)  
IF (ISTART.LE.0) THEN
  PRINT*,'TRANSITION',NTR,' AT',LL 
  PRINT*,'INDFRQ NOT DEFINED: CHECK ILOW OR IMAX (LTE AND NLTE!!!)'
ENDIF

SUM1 = 0.D0
SUM2 = 0.D0  
FF = 0.D0 ! maup
!
!---    note: first integration weight has to be modified
!
KK = ISTART  
WF = WFRE1(KK) - .5D0*CLIGHT* (FRE1(KK)-FRE1(KK-1))  
FREQ = FRE1(KK)  
!
!---    ii index corresponding to coarse freq. grid
!
EDGENU = FREQ ! maup
!IF(LL.EQ.1. .AND. OP_FLAG(NTR)) PRINT*,LABL(LABL4(NTR)), EDGENU

II = INDFIN(KK)  

IF (II.GT.0.) THEN  
     ALPHA = ALPHAFS(NTR,II)  
ELSE  
    STOP ' II MUST NOT BE LT ZERO IN INDFIN AT FIRST FREQ. POINT'  
END IF  


!OPA = OPAC(LL,II)*CLF !corrected
OPA = OPAC(LL,II)*CLF/OPA_EFF_RAT_OLD(LL,II) !back-corrected
!---- since OPAC is already an effective quantity, back-corrected here
!     (and thus also below).
!NOTE: Use OLD values, since opat_m_old is used below
THOM = XNE*SIG/OPA
THOMALO = 1.D0 - ALO(LL,II)*THOM  
ALOVTH = ALO(LL,II)/THOMALO  
XXJ=XJ(LL,II)
IF(OPTMET1) THEN
  BETAR=OPAT_M_OLD(LL,II)*CLF/OPA !corrected (OPAT_M_OLD is mean opacity)
  XXJ=XXJ+ALOVTH*BETAR*(STRUE_M_NEW(LL,II)-STRUE_M_OLD(LL,II))
ENDIF

EXK = EXP(-HCKT*FREQ)  
HCFRE3 = HC2*FREQ**3  
HNUE1 = 2.D0/ (HC2*FREQ)  
!
!---    f corresponds to i3-i4, g to i1-i2
!
F = ALPHA*HNUE1* (XXJ-HCFRE3*EXK*ALOVTH*ALPHA*XNISTAR/OPA)  
G = ALPHA*HNUE1*EXK* (XXJ+ &
&    HCFRE3* (1.D0-ALOVTH*ALPHA*DEPART*XNISTAR/OPA))

SUM1 = SUM1 + F*WF  
SUM2 = SUM2 + G*WF  
!	SUM3,SUM4,SUM5 and SUM6 = 0. due to (1-EDGENU/nu) factor
SUM3=0.D0
SUM4=0.D0  
SUM5=0.D0
SUM6=0.D0
!
!----   rest of integration, with all necessary interpolations
!
KKLOOP: DO KK = ISTART + 1,IFRE1  

     WF = WFRE1(KK)  
     FREQ = FRE1(KK)  
     II = INDFIN(KK)  

     IF (II.GT.0.) THEN  
          ALPHA = ALPHAFS(NTR,II)  
          !OPA = OPAC(LL,II)*CLF !corrected 
          OPA = OPAC(LL,II)*CLF/OPA_EFF_RAT_OLD(LL,II) !back-corrected 
          THOM = XNE*SIG/OPA  
          XALO = ALO(LL,II)  
          THOMALO = 1.D0 - XALO*THOM  
          ALOVTH = XALO/THOMALO  
          XXJ = XJ(LL,II)
          XXJJ =XXJ
          IF(OPTMET1) THEN
            BETAR=OPAT_M_OLD(LL,II)*CLF/OPA !corrected
            XXJ=XXJ+ALOVTH*BETAR*(STRUE_M_NEW(LL,II)-STRUE_M_OLD(LL,II))
          ENDIF
          EXK = EXP(-HCKT*FREQ)  
          HCFRE3 = HC2*FREQ**3  
          HNUE1 = 2.D0/ (HC2*FREQ)  
          F = ALPHA*HNUE1* (XXJ-HCFRE3*EXK*ALOVTH*ALPHA*XNISTAR/ OPA)
          G = ALPHA*HNUE1*EXK* (XXJ+HCFRE3* (1.D0-ALOVTH*ALPHA* &
           DEPART*XNISTAR/OPA))
          FF = ALPHA*HNUE1* XXJJ
          GG = ALPHA*HNUE1*EXK* (XXJJ+ HCFRE3)
!	  FF=ALPHA*HNUE1*HCFRE3*EXK*ALOVTH*ALPHA*XNISTAR/OPA


          IF (F.LT.0. .OR. G.LT.0.) THEN  
               PRINT*,NTR,FRE1(ISTART),KK,F,G,XXJ,ALOVTH,DEPART
               PRINT*,II,FREQ,OPA,XNISTAR
               PRINT*,ALPHAFS(NTR,II)
               PRINT*,'EFFECTIVE OPACITY ISSUE?:',OPA_EFF_RAT(LL,II),OPA_EFF_RAT_OLD(LL,II)  
               PRINT*,OPAC(LL,II),CLF 
               PRINT *,'NEGATIVE CONTRIBUTION TO PHOTO-RATES1'  
               STOP  
          END IF  

     ELSE IF (II.LT.0) THEN  
!
!       logarithmic interpolation for desired quantities
!
          XF = LOG10(FREQ)  
          II = -II  
          IF (II.EQ.IIOLD) GO TO 10  
          IIOLD = II  
          II1 = II + 1  
          !OPA = OPAC(LL,II)*CLF !corrected  
          OPA = OPAC(LL,II)*CLF/OPA_EFF_RAT_OLD(LL,II) !back-corrected  
          THOM = XNE*SIG/OPA  
          XALO = ALO(LL,II)  
          THOMALO = 1.D0 - XALO*THOM  
          ALOVTH = XALO/THOMALO  
          XXJ = XJ(LL,II)  
          XXJJ =XXJ
          IF(OPTMET1) THEN
            BETAR=OPAT_M_OLD(LL,II)*CLF/OPA !corrected
            XXJ=XXJ+ALOVTH*BETAR*(STRUE_M_NEW(LL,II)-STRUE_M_OLD(LL,II))
          ENDIF
          FREQII = FRE(II)  
          EXK = EXP(-HCKT*FREQII)  
          HCFRE3 = HC2*FREQII**3  

          ALPHA = ALPHAFS(NTR,II)  
          AUX1 = XXJ/ (HCFRE3*ALPHA)  
          AUX11 = XXJJ/ (HCFRE3*ALPHA)  
          AUX2 = ALOVTH*XNISTAR/OPA  
! new evalution scheme, to allow for very low EXK (x-ray regime)
!          F1 = AUX1/EXK - AUX2  
          F1 = AUX1 - AUX2 * EXK  
          G1 = 1.D0/ALPHA + AUX1 - DEPART*AUX2  
!          FF1 = AUX11/EXK
          FF1 = AUX11
          GG1 = 1.D0/ALPHA + AUX11

          
          !OPA = OPAC(LL,II1)*CLF  !corrected
          OPA = OPAC(LL,II1)*CLF/OPA_EFF_RAT_OLD(LL,II1)  !back-corrected
          THOM = XNE*SIG/OPA  
          XALO = ALO(LL,II1)  
          THOMALO = 1.D0 - XALO*THOM  
          ALOVTH = XALO/THOMALO  
          XXJ = XJ(LL,II1)
          XXJJ =XXJ
          IF(OPTMET1) THEN
            BETAR=OPAT_M_OLD(LL,II1)*CLF/OPA  !corrected
            XXJ=XXJ+ALOVTH*BETAR*(STRUE_M_NEW(LL,II1)-STRUE_M_OLD(LL,II1))
          ENDIF
          FREQII1 = FRE(II1)   
          EXK = EXP(-HCKT*FREQII1)  
          HCFRE3 = HC2*FREQII1**3  

          ALPHA = ALPHAFS(NTR,II1)  
          AUX1 = XXJ/ (HCFRE3*ALPHAFS(NTR,II1))  
          AUX11 = XXJJ/ (HCFRE3*ALPHA)  
          AUX2 = ALOVTH*XNISTAR/OPA  
!          F2 = AUX1/EXK - AUX2  
          F2 = AUX1 - AUX2 * EXK  
          G2 = 1.D0/ALPHA + AUX1 - DEPART*AUX2  
!          FF2 = AUX11/EXK
          FF2 = AUX11
          GG2 = 1.D0/ALPHA + AUX11

          IF (F1.LT.0. .OR. G1.LT.0. .OR. F2.LT.0. .OR. G2.LT.0.) THEN
               PRINT*,NTR,FRE1(ISTART),KK,F1,G1,F2,G2,XXJ,ALOVTH,DEPART
               PRINT*,II,II1,FREQII,FREQII1
               PRINT*,ALPHAFS(NTR,II),ALPHAFS(NTR,II1)
               PRINT*,'effective issue?:',OPA_EFF_RAT(LL,II),OPA_EFF_RAT_OLD(LL,II)  
               PRINT*,OPAC(LL,II),CLF 
               PRINT *,'NEGATIVE CONTRIBUTION TO PHOTO-RATES1'  
               PRINT *,'NEGATIVE CONTRIBUTION TO PHOTO-RATES2'  
               STOP  
          END IF  

          XFREQ = LOG10(FREQII)  
          XFREQ1 = LOG10(FREQII1)  
!          XF1 = LOG10(F1) 
!          XF2 = LOG10(F2)
! ADD log10(exp(hnu/kt))
          XF1 = LOG10(F1) + HCKT*FREQII*LOG10E 
          XF2 = LOG10(F2) + HCKT*FREQII1*LOG10E
          XG1 = LOG10(G1)  
          XG2 = LOG10(G2)  

!          XFF1 = LOG10(FF1) 
!          XFF2 = LOG10(FF2)
! ADD log10(exp(hnu/kt))
          XFF1 = LOG10(FF1)+ HCKT*FREQII*LOG10E 
          XFF2 = LOG10(FF2)+ HCKT*FREQII1*LOG10E 
          XGG1 = LOG10(GG1)  
          XGG2 = LOG10(GG2)  


          DFREQ = 1.D0/ (XFREQ1-XFREQ)  
          FACF = (XF2-XF1)*DFREQ  
          FACG = (XG2-XG1)*DFREQ  
          FACFF = (XFF2-XFF1)*DFREQ  
          FACGG = (XGG2-XGG1)*DFREQ  

!	  xff1 = LOG10(FF1)
!	  xff2 = LOG10(FF2)
!	  facff = (xff2-xff1)*DFREQ

   10           CONTINUE  

          DFX = XF - XFREQ  
          XF = XF1 + FACF*DFX  
          XG = XG1 + FACG*DFX  
!          XF = 10.D0**XF  
!          XG = 10.D0**XG  

          XFF = XFF1 + FACFF*DFX  
          XGG = XGG1 + FACGG*DFX  
!          XFF = 10.D0**XFF  
!          XGG = 10.D0**XGG  
          CALL CROSSBF(NTR,ALPHA,FREQ)  
          HNUE1 = 2.D0/ (HC2*FREQ)  
          EXK = EXP(-HCKT*FREQ)  
          HCFRE3 = HC2*FREQ**3  
          FGCONST = LOG10(ALPHA*HNUE1*HCFRE3*ALPHA)-HCKT*FREQ*LOG10E  
          F = 10.D0**(FGCONST+XF)  
          G = 10.D0**(FGCONST+XG)  
          FF = 10.D0**(FGCONST+XFF)  
          GG = 10.D0**(FGCONST+XGG)  

! 	  XFF = XFF1 + FACF*DFX  
!          XFF = 10.D0**XFF  
!          FF = XFF*FGCONST

     ELSE  
          STOP ' II MUST NOT BE ZERO IN INDFIN'  
     END IF  

     X1 = F*WF  
     X2 = G*WF  
    
     X3 = FF*WF*(1.D0-EDGENU/FREQ)*HH*CLIGHT*FREQ 
     X4 = GG*WF*(1.D0-EDGENU/FREQ)*HH*CLIGHT*FREQ 
     X5 = X3*(HCKT*FREQ)/TEMP
     X6 = X4*(HCKT*FREQ)/TEMP 
            
     SUM1 = SUM1 + X1  
     SUM2 = SUM2 + X2  
     SUM3 = SUM3 + X3
     SUM4 = SUM4 + X4
     SUM5 = SUM5 + X5
     SUM6 = SUM6 + X6
     
     DR = ABS(X1/SUM1)  
     DR = MAX(DR,ABS(X2/SUM2))  
     DR = MAX(DR,ABS(X3/SUM3))  
     DR = MAX(DR,ABS(X4/SUM4))  
     DR = MAX(DR,ABS(X6/SUM6)) 
     
     DEX = 1.D0 - FRE1(KK-1)/FREQ
            
!     IF (DEX.LT.1.D-7) STOP ' ERROR IN DEX'  

! The limit has been changed to allow for close energy terms
! in detailed models for neutral ions (energy terms with almost
! the same ionization energy). Note that this will increase the 
! computation time, but I would not expect a huge effect
     IF (DEX.LT.1.D-9) STOP ' ERROR IN DEX'  
!     IF (DR.LT.RERR .AND. DEX.GT..01D0) EXIT  
END DO KKLOOP  

PI4 = 4.D0*PI  
X1 = PI4*SUM1  
X2 = PI4*SUM2  

X3 = PI4*SUM3
X4 = PI4*SUM4
X5 = PI4*SUM5
X6 = PI4*SUM6
!
!print*,ntr,x1,' ',x2,' ',dr,' ',istart,' ',kk
!
RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE INTERMEPHO1(NTR,DEPART,XNISTAR,LL,X1,X2,XNE,TEMP,CLF,OPTMET)
!
!----  as subroutine INTERMEPHO, but only calculating X1 and X2, in
!      the range INDFRQ1(NTR) ... FRECIN(NTR).
!
!      X1 and X2 are the additional rbf rates for OP-data, when ionizing
!      to excited levels, but resonances below corresponding edge is present
!      (analogous to INTERMEDR)  

!----  OPAC variable EFFECTIVE opacity. See comments.  

!----  comment by JO (Jan. 2016):
!      the scattering component of ALO (pure Thomson so far)
!      might need to be updated regarding the line pseudo continuum
!      (note that this refers to the 'old' approach, i.e.,
!       it is an old problem)
!
!----  remember: ALO=0 for cmf_all frequencies
!
USE nlte_type
USE nlte_dim
USE fund_const
USE princesa_var, ONLY: LABL, LABL4, FRECIN
USE nlte_var, ONLY: ALPHAFS, &
& FRE1=>FREF,WFRE1=>WFREF,INDFIN,INDFRQ1,IFRE1=>IFREF, &
& FRE,WFRE,HC,IFRE, &
& OPAC,ST=>STRUE,XJ,ALO, &
& OPAT_M_OLD,STRUE_M_OLD, STRUE_M_NEW, METALS_CONVERGED,OP_FLAG,FRECFN1

USE nlte_porvor, ONLY: OPA_EFF_RAT, OPA_EFF_RAT_OLD

IMPLICIT NONE
!
!---  new algorithm
!---  subroutine intemepho1 (see netmat). it calculates photo-integrals
!---  for some the formulae implemented in detail.
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
INTEGER(I4B), PARAMETER :: IFREFIN=ID_FREC2  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  DEPART,TEMP,X1,X2,XNE,X3,X4,X5,X6,EDGENU,XNISTAR,CLF 
INTEGER(I4B) ::  LL,NTR  
LOGICAL OPTMET,OPTMET1
!     ..
!     .. local scalars ..
REAL(DP) ::  ALOVTH,ALPHA,AUX1,AUX2,DEX,DFREQ,DFX,DR,EXK,F,F1, &
&                 F2,FACF,FACG,FGCONST,FREQ,FREQII,FREQII1,G,G1,G2, &
&                 HCFRE3,HCKT,HNUE1,OPA,PI4,RERR,SIG,SUM1,SUM2, &
&                 THOM,THOMALO,WF,XALO,XF,XF1,XF2,XFREQ,XFREQ1,XG, &
&                 XG1,XG2,XXJ,BETAR, &
&                 XXJJ,AUX11,LASTFREQ        
INTEGER(I4B) ::  II,II1,IIOLD,ISTART,KK
!     ..
!     .. external subroutines ..
EXTERNAL CROSSBF  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,EXP,LOG10,MAX  
!     ..

OPTMET1=OPTMET
IF(OPTMET1.AND.METALS_CONVERGED) OPTMET1=.FALSE.

HCKT = HKL/TEMP  
!
!       begin of integration of the two photo-integrals
!       (for definition, see the santolaya et al. paper)
!       note: integration over frequency, fre in kayser
!
IIOLD = 0  
RERR = 1.D-5  
SIG = AMH*SIGMAE  
ISTART = INDFRQ1(NTR)  
IF (ISTART.LE.0) THEN
  PRINT*,'TRANSITION',NTR,' AT',LL 
  PRINT*,'INDFRQ1 NOT DEFINED: CHECK ILOW OR IMAX (LTE AND NLTE!!!)'
ENDIF

SUM1 = 0.D0
SUM2 = 0.D0  
!
!---    note: first integration weight has to be modified
!
KK = ISTART  
WF = WFRE1(KK) - .5D0*CLIGHT* (FRE1(KK)-FRE1(KK-1))  
FREQ = FRE1(KK)  
!
!---    ii index corresponding to coarse freq. grid
!
EDGENU = FREQ ! maup
LASTFREQ=FRECIN(NTR)

! some tests
IF(LL.EQ.1) THEN
  IF(.NOT.OP_FLAG(NTR)) STOP ' INTERMEPHO1 CALLED AND OP_FLAG = F'
  IF(EDGENU.GE.LASTFREQ .OR. EDGENU.LT.FRECFN1(NTR)) &
    STOP ' SOMETHING ROTTEN WITH EDGENU -- INTERMEPHO1!'
ENDIF
  
II = INDFIN(KK)  

IF (II.GT.0.) THEN  
     ALPHA = ALPHAFS(NTR,II)  
ELSE  
    STOP ' II MUST NOT BE LT ZERO IN INDFIN AT FIRST FREQ. POINT'  
END IF  


!OPA = OPAC(LL,II)*CLF !corrected
OPA = OPAC(LL,II)*CLF/OPA_EFF_RAT_OLD(LL,II) !back-corrected
!---- since OPAC is already an effective quantity, back-corrected here
!     (and thus also below).
!NOTE: Use OLD values, since opat_m_old is used below
THOM = XNE*SIG/OPA
THOMALO = 1.D0 - ALO(LL,II)*THOM  
ALOVTH = ALO(LL,II)/THOMALO  
XXJ=XJ(LL,II)
IF(OPTMET1) THEN
  BETAR=OPAT_M_OLD(LL,II)*CLF/OPA !corrected (OPAT_M_OLD is mean opacity)
  XXJ=XXJ+ALOVTH*BETAR*(STRUE_M_NEW(LL,II)-STRUE_M_OLD(LL,II))
ENDIF

EXK = EXP(-HCKT*FREQ)  
HCFRE3 = HC2*FREQ**3  
HNUE1 = 2.D0/ (HC2*FREQ)  
!
!---    f corresponds to i3-i4, g to i1-i2
!
F = ALPHA*HNUE1* (XXJ-HCFRE3*EXK*ALOVTH*ALPHA*XNISTAR/OPA)  
G = ALPHA*HNUE1*EXK* (XXJ+ &
&    HCFRE3* (1.D0-ALOVTH*ALPHA*DEPART*XNISTAR/OPA))

SUM1 = SUM1 + F*WF  
SUM2 = SUM2 + G*WF  
!
!----   rest of integration, with all necessary interpolations
!
KKLOOP: DO KK = ISTART + 1,IFRE1  

     WF = WFRE1(KK)  
     FREQ = FRE1(KK)
     IF(FREQ.GE.LASTFREQ) EXIT KKLOOP
     II = INDFIN(KK)  

     IF (II.GT.0.) THEN  
          ALPHA = ALPHAFS(NTR,II)  
          !OPA = OPAC(LL,II)*CLF !corrected 
          OPA = OPAC(LL,II)*CLF/OPA_EFF_RAT_OLD(LL,II) !back-corrected 
          THOM = XNE*SIG/OPA  
          XALO = ALO(LL,II)  
          THOMALO = 1.D0 - XALO*THOM  
          ALOVTH = XALO/THOMALO  
          XXJ = XJ(LL,II)
          XXJJ =XXJ
          IF(OPTMET1) THEN
            BETAR=OPAT_M_OLD(LL,II)*CLF/OPA !corrected
            XXJ=XXJ+ALOVTH*BETAR*(STRUE_M_NEW(LL,II)-STRUE_M_OLD(LL,II))
          ENDIF
          EXK = EXP(-HCKT*FREQ)  
          HCFRE3 = HC2*FREQ**3  
          HNUE1 = 2.D0/ (HC2*FREQ)  
          F = ALPHA*HNUE1* (XXJ-HCFRE3*EXK*ALOVTH*ALPHA*XNISTAR/ OPA)
          G = ALPHA*HNUE1*EXK* (XXJ+HCFRE3* (1.D0-ALOVTH*ALPHA* &
           DEPART*XNISTAR/OPA))

          IF (F.LT.0. .OR. G.LT.0.) THEN  
               PRINT*,NTR,FRE1(ISTART),KK,F,G,XXJ,ALOVTH,DEPART
               PRINT*,II,FREQ,OPA,XNISTAR
               PRINT*,ALPHAFS(NTR,II)
               PRINT*,'EFFECTIVE OPACITY ISSUE?:',OPA_EFF_RAT(LL,II),OPA_EFF_RAT_OLD(LL,II)  
               PRINT*,OPAC(LL,II),CLF 
               PRINT *,'NEGATIVE CONTRIBUTION TO PHOTO-RATES1'  
               STOP  
          END IF  

     ELSE IF (II.LT.0) THEN  
!
!       logarithmic interpolation for desired quantities
!
          XF = LOG10(FREQ)  
          II = -II  
          IF (II.EQ.IIOLD) GO TO 10  
          IIOLD = II  
          II1 = II + 1  
          !OPA = OPAC(LL,II)*CLF !corrected  
          OPA = OPAC(LL,II)*CLF/OPA_EFF_RAT_OLD(LL,II) !back-corrected  
          THOM = XNE*SIG/OPA  
          XALO = ALO(LL,II)  
          THOMALO = 1.D0 - XALO*THOM  
          ALOVTH = XALO/THOMALO  
          XXJ = XJ(LL,II)  
          XXJJ =XXJ
          IF(OPTMET1) THEN
            BETAR=OPAT_M_OLD(LL,II)*CLF/OPA !corrected
            XXJ=XXJ+ALOVTH*BETAR*(STRUE_M_NEW(LL,II)-STRUE_M_OLD(LL,II))
          ENDIF
          FREQII = FRE(II)  
          EXK = EXP(-HCKT*FREQII)  
          HCFRE3 = HC2*FREQII**3  

          ALPHA = ALPHAFS(NTR,II)  
          AUX1 = XXJ/ (HCFRE3*ALPHA)  
          AUX11 = XXJJ/ (HCFRE3*ALPHA)  
          AUX2 = ALOVTH*XNISTAR/OPA  
! new evalution scheme, to allow for very low EXK (x-ray regime)
!          F1 = AUX1/EXK - AUX2  
          F1 = AUX1 - AUX2 * EXK  
          G1 = 1.D0/ALPHA + AUX1 - DEPART*AUX2  
          
          !OPA = OPAC(LL,II1)*CLF  !corrected
          OPA = OPAC(LL,II1)*CLF/OPA_EFF_RAT_OLD(LL,II1)  !back-corrected
          THOM = XNE*SIG/OPA  
          XALO = ALO(LL,II1)  
          THOMALO = 1.D0 - XALO*THOM  
          ALOVTH = XALO/THOMALO  
          XXJ = XJ(LL,II1)
          XXJJ =XXJ
          IF(OPTMET1) THEN
            BETAR=OPAT_M_OLD(LL,II1)*CLF/OPA  !corrected
            XXJ=XXJ+ALOVTH*BETAR*(STRUE_M_NEW(LL,II1)-STRUE_M_OLD(LL,II1))
          ENDIF
          FREQII1 = FRE(II1)   
          EXK = EXP(-HCKT*FREQII1)  
          HCFRE3 = HC2*FREQII1**3  

          ALPHA = ALPHAFS(NTR,II1)  
          AUX1 = XXJ/ (HCFRE3*ALPHAFS(NTR,II1))  
          AUX2 = ALOVTH*XNISTAR/OPA  
!          F2 = AUX1/EXK - AUX2  
          F2 = AUX1 - AUX2 * EXK  
          G2 = 1.D0/ALPHA + AUX1 - DEPART*AUX2  

          IF (F1.LT.0. .OR. G1.LT.0. .OR. F2.LT.0. .OR. G2.LT.0.) THEN
               PRINT*,NTR,FRE1(ISTART),KK,F1,G1,F2,G2,XXJ,ALOVTH,DEPART
               PRINT*,II,II1,FREQII,FREQII1
               PRINT*,ALPHAFS(NTR,II),ALPHAFS(NTR,II1)
               PRINT*,'effective issue?:',OPA_EFF_RAT(LL,II),OPA_EFF_RAT_OLD(LL,II)  
               PRINT*,OPAC(LL,II),CLF 
               PRINT *,'NEGATIVE CONTRIBUTION TO PHOTO-RATES1'  
               PRINT *,'NEGATIVE CONTRIBUTION TO PHOTO-RATES2'  
               STOP  
          END IF  

          XFREQ = LOG10(FREQII)  
          XFREQ1 = LOG10(FREQII1)  
!          XF1 = LOG10(F1) 
!          XF2 = LOG10(F2)
! ADD log10(exp(hnu/kt))
          XF1 = LOG10(F1) + HCKT*FREQII*LOG10E 
          XF2 = LOG10(F2) + HCKT*FREQII1*LOG10E
          XG1 = LOG10(G1)  
          XG2 = LOG10(G2)  

          DFREQ = 1.D0/ (XFREQ1-XFREQ)  
          FACF = (XF2-XF1)*DFREQ  
          FACG = (XG2-XG1)*DFREQ  

   10           CONTINUE  

          DFX = XF - XFREQ  
          XF = XF1 + FACF*DFX  
          XG = XG1 + FACG*DFX  
!          XF = 10.D0**XF  
!          XG = 10.D0**XG  

          CALL CROSSBF(NTR,ALPHA,FREQ)  
          HNUE1 = 2.D0/ (HC2*FREQ)  
          EXK = EXP(-HCKT*FREQ)  
          HCFRE3 = HC2*FREQ**3  
          FGCONST = LOG10(ALPHA*HNUE1*HCFRE3*ALPHA)-HCKT*FREQ*LOG10E  
          F = 10.D0**(FGCONST+XF)  
          G = 10.D0**(FGCONST+XG)  

     ELSE  
          STOP ' II MUST NOT BE ZERO IN INDFIN'  
     END IF  

     X1 = F*WF  
     X2 = G*WF  
    
     SUM1 = SUM1 + X1  
     SUM2 = SUM2 + X2  
     
     DR = ABS(X1/SUM1)  
     DR = MAX(DR,ABS(X2/SUM2))  
     
     DEX = 1.D0 - FRE1(KK-1)/FREQ
            
!     IF (DEX.LT.1.D-7) STOP ' ERROR IN DEX'  

! The limit has been changed to allow for close energy terms
! in detailed models for neutral ions (energy terms with almost
! the same ionization energy). Note that this will increase the 
! computation time, but I would not expect a huge effect
     IF (DEX.LT.1.D-9) STOP ' ERROR IN DEX'  
!     IF (DR.LT.RERR .AND. DEX.GT..01D0) EXIT  
END DO KKLOOP  

PI4 = 4.D0*PI  
X1 = PI4*SUM1  
X2 = PI4*SUM2  
!
!print*,ntr,x1,' ',x2,' ',dr,' ',istart,' ',kk
!
RETURN  

END
!
!
!-----------------------------------------------------------------------
!
SUBROUTINE INTERMEDR(I,LL,X1,X2,TEMP)

! CALCULATES DR RATES, XNUE=CM^-1

USE nlte_type
USE nlte_dim
USE fund_const
USE princesa_var, ONLY: DATA,IFORM4,INDAT4,NUMD4,KL,FRECIN,FRECFN,LABL,LABL4
USE nlte_var, ONLY: IFRE,FRE,XJ,MLOW,MUP

IMPLICIT NONE

REAL(DP), PARAMETER :: CONST1=E2MC2/HH*4.D0*PI**2  
!
!     .. scalar arguments ..
REAL(DP) ::  TEMP,X1,X2 
INTEGER(I4B) ::  I,LL  

!     .. local scalars ..
INTEGER(I4B) ::  N,II1,NDATA,NI,NTRAN,INDEX,K,ML,MU,MU1
REAL(DP) :: XNUE,FLU,XLAMC,BLU,AFAC,EHNUEKT,XXX,WILOG,J1,J2,JNUE
CHARACTER*5 :: LAB


N = IFORM4(I)  
IF(N.NE.21) STOP ' WRONG FORMULA IN INTERMEDR'

II1 = INDAT4(I) + 3  ! OFFSET
NDATA=NUMD4(I)

NTRAN=(NDATA-3)/2

X1=0.
X2=0.

!LOOP OVER ALL RESONANCES
K=2

DO NI=1,NTRAN

INDEX=II1+2*(NI-1)  
XNUE=DATA(INDEX)
!--- for tests of Adi's DR data:
!--- if ionization to excited states, first resonance should lie in between
!--- ground-state (FRECFN) and excited state (FRECIN) edge.
ML=MLOW(I)
MU=MUP(I)
MU1=KL(ML)
IF(NI.EQ.1 .AND. MU.NE.MU1) THEN
  IF(XNUE.LT.FRECFN(I) .OR. XNUE.GT.FRECIN(I)) THEN
    LAB=LABL(LABL4(I))
!JP May 2021: neglect small inconsistencies for OII in WM-Basic data base
    IF((LAB.EQ.'O2_5 ' .OR. LAB.EQ.'O2_17' .OR. LAB.EQ.'O2_15') .AND. &
    XNUE/FRECFN(I).GT.0.95 .AND. XNUE.LE.FRECIN(I)) THEN
      CONTINUE
    ELSE
    PRINT*,LABL(LABL4(I)),XNUE,FRECFN(I),FRECIN(I)
! after tests, this stop statement might be removed
    STOP ' PROBLEMS IN RESONANCES!!!'
    ENDIF
  ENDIF
ENDIF
  
FLU=DATA(INDEX+1)

IF (XNUE.GT.FRE(IFRE) .OR. XNUE.LT.FRE(1)) CYCLE

XLAMC=1.D0/XNUE

BLU = CONST1*XLAMC*FLU  
AFAC = HC2/XLAMC**3  
EHNUEKT = EXP(-HKL*XNUE/TEMP)  

!
!---------    log-log interpolation for JNUE
!
K = K-1  
! note: not all resonances in monotonic order
IF(K.NE.1.) THEN
  IF(XNUE.LT.FRE(K-1)) K = 1  
ENDIF

XXX = FRE(K)  
DO WHILE (XNUE.GT.XXX)  
  K=K+1  
  XXX=FRE(K)  
END DO  

IF(K.GT.IFRE) STOP ' INDEX NOT FOUND IN INTERMEDR'
! can be commented after tested
IF(XNUE.GT.FRE(K).OR.XNUE.LT.FRE(K-1)) STOP ' ERROR IN DR FRE-INDEX'

WILOG = LOG10(FRE(K)/XNUE)/LOG10(FRE(K-1)/FRE(K))

J1=XJ(LL,K)
J2=XJ(LL,K-1)
JNUE = J1*10.D0** (LOG10(J1/J2)*WILOG)  
!-------------------------------------------

! SUM UP RATES, ASSUMING BROAD RESONANCES (-> JNUE CAN BE USED)
X1=X1+BLU*JNUE
X2=X2+EHNUEKT*BLU*(AFAC+JNUE)

ENDDO

RETURN
END

!
!-----------------------------------------------------------------------
!
SUBROUTINE LINESOB(LL,VELR,VGRAD,NFIR,NLAS,BLEV0,RAT3,NDIM,XNEL, &
&                   SR,VMAX,NATO,XMUST,TEMPE,ERROLD,QLU,QLD)

! so far, this routine should be indendent on clumping, as long as
! all rates for energy equation remain unaffected  

!
!     new qualifiers for levels:
!     iqual = 0 ok
!     iqual = 1 problems, no high precision
!     iqual = 2 negative occup. numbers achieved, reset to old value!
!
!
USE nlte_type
USE nlte_dim
USE fund_const ! maup
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&       ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& DATA,NDAT, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
&       LABAT,LABL4,LABL1,LABU1,LABL3,IPARE5

USE nlte_var, ONLY: ABUNDND, &
& IONISST,IONIS,XNELTE, &
& ALEVEL,BLEVEL,ENIONND, &
& MLOW,MUP,MMLOW,MMUP,MCLOW,MCUP, &
& OPTCMF,OPTMIXED, &
& TLUMAT,TULMAT,INDEXRBB,INDEXCMF, &
& IQUAL, &
& PRECIS,UNASOL,MEGAS, &
& VTHER,NIST,XNKK

USE tcorr_var, ONLY : OPTTCOR
  

IMPLICIT NONE

!     .. parameters ..
INTEGER(I4B), PARAMETER :: NDIM1=ID_LLEVS  
INTEGER(I4B), PARAMETER :: NNREC=ID_LLEVS  
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  ERROLD,SR,TEMPE,VELR,VGRAD,VMAX,XMUST,XNEL  
INTEGER(I4B) ::  LL,NATO,NDIM,NFIR,NLAS  
!     ..
!     .. array arguments ..
REAL(DP) ::  BLEV0(NNREC),RAT3(NDIM1,NDIM1)  
REAL(DP), DIMENSION(NNREC,2) :: QLU,QLD  

!     ..
!     .. local scalars ..
REAL(DP) ::  BMAX,CASTA1,CASTA2,CLU,DET,ER,ERRMAX, &
&                 FIOLD,FLU,PRECIS1,SDP,TLU,TUL,XLAMBD,XLAMC,XNL, &
&                 XNLSTR,XNU,XNUSTR, &
&                 DTCLU, HHNU, clu1
INTEGER(I4B) ::  I,IFORM,II,III,INDI,INDMAT,INTEG,ITM, &
& J,JMAX,M,ML,MN,MU,N1,N2,NC,NDATOS,NFOR,NLO,NU
LOGICAL QCL,QDB,QRT,UNADOS  
!     ..
!     .. local arrays ..
REAL(DP) :: BOLD(NDIM1),DELTAN(NDIM1),DL(ID_NLONG), &
&                 ERRI(ID_LLEVS),F(NDIM1),FLUD(NDIM1), &
&                 FRECFN(ID_RBFTR),FRECIN(ID_RBFTR), &
&                 RAT2(NDIM1,NDIM1),RATMAT(NDIM1,NDIM1)

REAL(QP) :: AUX(NDIM1)

INTEGER(I4B) ::  IFPTR(ID_NTTRD),INDLUD(NDIM1),IQ(ID_LLEVS), &
& NFPTR(ID_NTTRD)  
!     ..
!     .. external subroutines ..
EXTERNAL LUBKSB,LUDCMP,TRATCOL,TRATRAD  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,INT,MAX,MIN  
!     ..

IF (NDIM.NE.NDIM1) STOP ' ERROR IN NDIM NE NDIM1'  

PRECIS1 = MIN(ERROLD/10.D0,1.D-3)  
PRECIS1 = MAX(1.D-5,PRECIS1)  
UNADOS = UNASOL  

ITM = 100  

MN = NLAS - NFIR + 1  

   10 CONTINUE  
   
F(1:MN)=0.D0
IQ(1:MN) = 0  
RATMAT(1:MN,1:MN) = RAT3(1:MN,1:MN)  

IILOOP: DO II = 1,INDEX1  
     ML = MMLOW(II)  
     MU = MMUP(II)  
     IF (ML.LT.NFIR .OR. ML.GE.NLAS) CYCLE  
     IF (MU.GT.NLAS) CYCLE

     XNL = BLEV0(ML)  
     XNU = BLEV0(MU)  

     XNLSTR = ALEVEL(ML)  
     XNUSTR = ALEVEL(MU)  
     IF(NIST(ML).EQ.0.) STOP ' NIST(ML=0)'
     IF(NIST(MU).EQ.0.) STOP ' NIST(MU=0)'
     INDI = INDAT1(II)  
     ER = DATA(INDI)  
     M = INT(ER)  

     IF (M.EQ.1) THEN  
!
!---  rbb transitions with occ.no. from previous iteration
!
          INDMAT = INDEXRBB(II)  
          TLU = TLUMAT(INDMAT,LL)  
          TUL = TULMAT(INDMAT,LL)  

          IF (UNADOS .AND. MEGAS) THEN  
               CASTA1 = XNL*TLU  
               CASTA2 = XNU*TUL  
               WRITE (22,FMT=9000) LL,'RBB',ML,MU,CASTA1,CASTA2, &
                CASTA2 - CASTA1
!               WRITE (22,FMT=9000) LL,'RBB',ML,MU,CASTA1*NIST(ML)/XNL, &
!&                CASTA2*NIST(MU)/XNU, (CASTA2 - CASTA1)/XNKK(ML)
          END IF  
     ELSE IF (M.EQ.2) THEN  

          NLO = LABL1(II)  
          NU = LABU1(II)  
          XLAMC = DATA(INDI+1)*1.D-8  
          NFOR = IFORM1(II)  

          CALL TRATCOL(II,NLO,NU,XLAMC,TEMPE,LABL1,LE,CLU,NFOR, &
           INDAT1,DATA,LL,'A')
          TLU = CLU*XNEL  
!
!---note: no correction for ne necessary, since transition in same ion
!
          TUL = TLU*XNLSTR/XNUSTR  

          IF(OPTTCOR) THEN
            CALL TRATCOL(II,NLO,NU,XLAMC,TEMPE*1.01,LABL1,LE,CLU1,NFOR, &
             INDAT1,DATA,LL,'B')
            dtclu=(clu1-clu)/(0.01*tempe)*xnel
! There is no difference in the energies by using XLAMC or the
! FL values
      	    HHNU=HH*CLIGHT/XLAMC 
!	    DTCBBD(II) = XNU*(XNLSTR/XNUSTR)*DTCLU*HHNU !does not work
            QLU(ML,1) = QLU(ML,1) + TLU*HHNU !(*x_l)
            QLD(MU,1) = QLD(MU,1) + TUL*HHNU !(*x_u)
            QLU(ML,2) = QLU(ML,2) + DTCLU*HHNU  !(*x_l)

            IF(LABAT(NATO).EQ.'H' .OR. LABAT(NATO).EQ.'HE') THEN 
! standard treatment for H/He
            QLD(MU,2) = QLD(MU,2) + (XNLSTR/XNUSTR)*DTCLU*HHNU - &
&             (TUL*HHNU) * HHNU/AKB/TEMPE/TEMPE !(*x_u)
! other elements
            ELSE ! to be consistent with the background treatment, see nlte_approx package
              IF(ABS(1.D0-TLU*BLEV0(ML)/(TUL*BLEV0(MU))).LT.0.001) THEN
                QLD(MU,2) = QLD(MU,2) + (XNLSTR/XNUSTR)*DTCLU*HHNU  ! to allow for cancellation
              ELSE                
                QLD(MU,2) = QLD(MU,2) + (XNLSTR/XNUSTR)*DTCLU*HHNU - &
&               (TUL*HHNU) * HHNU/AKB/TEMPE/TEMPE !(*x_u)
              ENDIF
            ENDIF  
          ENDIF   

          IF (UNADOS .AND. MEGAS) THEN  
               CASTA1 = XNL*TLU  
               CASTA2 = XNU*TUL  
               WRITE (22,FMT=9000) LL,'CBB',ML,MU,CASTA1,CASTA2, &
                CASTA2 - CASTA1
!               WRITE (22,FMT=9000) LL,'CBB',ML,MU,CASTA1*NIST(ML)/XNL, &
!&                CASTA2*NIST(MU)/XNU, (CASTA2 - CASTA1)/XNKK(ML)
          END IF  
	  
     ELSE  

          STOP 'ERROR IN TRANSITION TYPE RBB/CBB'  
     END IF  

     N1 = ML - NFIR + 1  
     N2 = MU - NFIR + 1  

     IF (N2.GT.MN .OR. N1.LE.0) STOP 'ERROR IN LINESOB'  
!
!------- coeficients matrix
!
     RATMAT(N1,N1) = RATMAT(N1,N1) - TLU  
     RATMAT(N1,N2) = RATMAT(N1,N2) + TUL  
     RATMAT(N2,N1) = RATMAT(N2,N1) + TLU  
     RATMAT(N2,N2) = RATMAT(N2,N2) - TUL  


END DO IILOOP

UNADOS = .FALSE.  
!
!---- newton-raphson for rate equations cont+lines
!
!---- changed: leave out line with largest occ.num
!
JMAX=MN
BMAX=0.

DO I=1,MN
     J=I+NFIR-1
     BOLD(I)=BLEV0(J)
     IF(BOLD(I).GT.BMAX) THEN
          BMAX=BOLD(I)
          JMAX=I
     ENDIF
END DO

DO I=1,MN
     RATMAT(JMAX,I)=1.D0
END DO
        
F(JMAX)=ABUNDND(NATO,LL)

FLUD(1:MN) = F(1:MN)  
RAT2(1:MN,1:MN) = RATMAT(1:MN,1:MN)  

!
!-----solution of rate equations
!
CALL LUDCMP(RATMAT,MN,NDIM,INDLUD,DET)  
!
CALL LUBKSB(RATMAT,MN,NDIM,INDLUD,F)  
!
!-----------------------------------------------------------------
!--------newton raphson improvement-cycle-------------------------
!-----------------------------------------------------------------
!
ITLOOP: DO III = 1,ITM  

! old version (if no qp precision available)
!     DO I = 1,MN  
!          SDP = -FLUD(I)  
!
!     high precision summation
!
!          DO J=1,MN
!               AUX(J)=RAT2(I,J)*F(J)
!          END DO
!
!          CALL SORT1(MN,AUX)     
!
!          DO J=1,MN
!               SDP=SDP+AUX(J)
!          END DO
!
!          DELTAN(I) = SDP  
!     END DO

!    new version (residuum calculated in qp precision
     AUX(1:MN)=MATMUL(REAL(RAT2(1:MN,1:MN),QP),REAL(F(1:MN),QP))-REAL(FLUD(1:MN),QP)
     DELTAN(1:MN)=AUX(1:MN)

     CALL LUBKSB(RATMAT,MN,NDIM,INDLUD,DELTAN)  

     ERRMAX = 0.D0  

     DO I = 1,MN  
!
!     changed order (at first udate, then error) to allow for f=0
!     after first iteration
!
          FIOLD = F(I)  
          F(I) = F(I) - DELTAN(I)  
          ERRI(I) = ABS(DELTAN(I)/F(I))  

          IF (ERRI(I).LT.PRECIS1) THEN  
               IQ(I) = 0  
          ELSE  
               IQ(I) = 1  
          END IF  

          IF (F(I).LT.0.D0) THEN  
               IF (FIOLD.LT.0.D0) THEN  
                    F(I) = BOLD(I)  
               ELSE  
                    F(I) = FIOLD  
               END IF  
               IQ(I) = 2  
          END IF  

          ERRMAX = MAX(ERRMAX,ERRI(I))  
     END DO
!
!     print*,'errmax: ',errmax
!
     IF (ERRMAX.LT.PRECIS1) THEN  

       DO I = 1,MN  
               J = I + NFIR - 1  
               BLEVEL(J) = F(I)  
!
!     for this end condition, all levels have to be 0
!
               IQUAL(J,LL) = IQ(I)  
          END DO

          GO TO 140  
     END IF  

END DO ITLOOP  
!
!     at this point, we have problems with our occup. numbers
!     only those errors are counted, which have "perfect" occup. numbers
!
PRINT *,'ATOM: ',NATO,'   DEPTH: ',LL  
ERRMAX = 0.D0  

DO I = 1,MN  
     J = I + NFIR - 1  
     BLEVEL(J) = F(I)  
     IQUAL(J,LL) = IQ(I)  
     IF (IQ(I).EQ.0) THEN  
          ERRMAX = MAX(ERRMAX,ERRI(I))  
     ELSE  
          WRITE (*,FMT=9010) J,IQ(I)  
     END IF  

END DO  

  140 CONTINUE  

RETURN  

 9000 FORMAT (1X,I2,3X,A3,2 (3X,I3),3 (2X,G16.6))  
 9010 FORMAT (' LEVEL(ABS. NUMBER)',I3,' WITH IQUAL = ',I1)  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE TRATCOL(II,NLO,NU,XLAMC,TEMPE,LABL1,LE,CLU,NFOR,INDAT1, &
&                   DATA,LL,CHAR)
!
!---- subroutine for treating cbb transitions.
!---- used (and partly checked so far)
!
!------ formula 1:  si
!------ formula 11: mg
!------ formula 13/113: h i (either Giovanardi et al. or Burke et al.)
!------ formula 14: he ii
!------ formula 15: he i (singlett)
!------ formula 16: he i (singlett, triplett)
!------ formula 16: he i (singlett)
!------ formula 17: he i (singlett, forbidden, intercomb.)
!------ formula 19: he i (triplett, forbidden)
!------ formula 20: he i (intercomb.)
!------ formula 21: he i (intercomb.)
!------ formula 22: mg (see formula 1) !------ formula 23: n iii 
!------ formula 24: si
!------ formula 25: si, general
!------ formula 26: c (bug identified and corrected, 05 Feb 2005)
!------ formula 32: mg 
!------ formula 35: he i (singlett, forbidden)
!       formula 37, 40: he i
!       formula 45: h i
!       formula 46: h i
!       formula 51: for Adi's input, omega with slope
!       formula 52: for Adi's input, omega = 1
!       formula 60-65 : n iii (fine structure fits for gamma) from Stafford (1993)
!       formula 66: niv (fine structure fits for gamma) from Ramsbottom (1994)
!
!       formula 100: h i/ he ii (fits to exactly calculated cross-sections, &
!                    either from Butler (2004) or from Aggarwal (1991)
!
!JO July 2016
!       formula 101: tabulated collision-strengths (e.g. CII/CIV from Aggarwal, CIII from Mitnik (2003)),
!                    in combination with formula 100. The first transition is read by formula 100
!                    (temperatures and coll. strengths), for the others formula 101 is used (only
!                     collision-strengths) when the T-grid is the same.
!                    NOTE!!!!! DIMENSION OF TGRID SET TO 50 (i.e., maximum number of grid-points = 50)
!                    NOTE!!!!! for reasons of comp. time, it's NOT checked whether NPT>50  
!
! note: A10HHe(hydrogen) uses formula 13 for all transitions
!            (from Giovanardi et al. for nl le 14, nu le 15,
!             otherwise according to Burke
!       A11HHe(hydrogen)
!           uses formula 100 (Butler) for all trans. with nl, nu  le 7 
!           uses formula 113 (Burke)  for all trans. with nl le 4, nl ge 8   
!           uses formula  46 (P & R)  for all trans. with nl ge 5, nl ge 8   

USE nlte_type
USE nlte_dim
USE fund_const
USE princesa_var, ONLY: ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS  

IMPLICIT NONE
!     .. parameters ..
  INTEGER(I4B), PARAMETER :: NGAU=6, NSP=10
  INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CLU,TEMPE,XLAMC  
INTEGER(I4B) ::  II,NFOR,NLO,NU,LL  
CHARACTER*1 :: CHAR
!     ..
!     .. array arguments ..
REAL(DP) ::  DATA(ID_NDATA)
INTEGER(I4B) ::  INDAT1(ID_RCBBT),LABL1(ID_RCBBT),LE(ID_LLEVS)  

REAL(DP) :: WLAG(NGAU),XLAG(NGAU),XSP(NSP)


REAL(DP), DIMENSION(ND1), SAVE :: TEMPA, TEMPB !TEMPB CORRESPONDS TO 1.01*TEMPA
REAL(DP), DIMENSION(0:8,ND1), SAVE :: TSA, TSB
REAL(DP), DIMENSION(0:8) :: TS
REAL(DP), DIMENSION(0:23) :: AI
REAL(DP), DIMENSION(0:9) :: ASP,ASP2
REAL(DP), DIMENSION(50), SAVE :: TGRID !for formula 100/101
!     ..
!     .. local scalars ..
REAL(DP) ::  ACON,ACON1,ADN,AIFIT,AN,AN2,AU,AU1,AU2,AU3, &
&                 C0,C1,C_1,C_2, &
&                 E02,ER,FLU,GAMMA,GAMMAE,GBAR,OMEGA,SCALE,T1, &
&                 U0,USU1,X,XLAMIO,EPS1,EPS2,FAC,F,E1,E2,P1,P2,Z,ZZR,RYDO, &
&                 XNULO,XNUUP,ULO,UUP, &
&                 RN,N2,NP2,RX,RX2,TWON2,TWON2RX,LOGTWNX,ANN,BN,RFAC,BNN, &
&                 CON2,YEX,YEX2,RY,RY2,Z0,ZEX,ZEX2,RZ,RZ2, &
&                 UPSFAC,DELTA,UPSLN,CON1,TE10D4,EE,TLOG,SLOPE,OM,  &
&                 XGAMMA,LGAMMA1,LGAMMA2,LGAMMA3,LGAMMA4,LGAMMA5,LGAMMA6, &
&                 GAMMA1,GAMMA2,GAMMA3,GAMMA4,GAMMA5,GAMMA6, &
&                 LTEMPE ,YP1,YPN

INTEGER(I4B) :: IJ,INDI,J,LEV,NA,NDELTA,NFIT,NIVL,NIVU,NNLO,NNUP, &
&               NPT,INDX,ILO,IUP,IPT,FILEIN,K,I

INTEGER(I4B), SAVE :: NPTSTORE

CHARACTER CAR*6,LAB*6  
LOGICAL FOUND
!     ..
!     .. external functions ..
REAL(DP) ::  ERFCM,EXPIN,EXPIN1,HCOL,PVR,PFTN,PRHCOL  
EXTERNAL ERFCM,EXPIN,EXPIN1,HCOL,PVR,PFTN,PRHCOL  
!     ..
!     .. intrinsic functions ..
!INTRINSIC DBLE,EXP,INT,LOG,MAX,MIN,SQRT  
!     ..
DATA XLAG,WLAG /0.222846604D0,1.1889321D0,2.99273633D0,5.77514357D0, &
&               9.83746742D0,15.982874D0,0.458964674D0,0.417000831D0, &
& 0.113373382D0,0.0103991975D0,0.000261017203D0,8.98547906D-07/

DATA XSP  / 3.50515  ,  3.90309  ,  4.20412 ,  4.50515  , 4.90309 ,&
&     5.20412 ,  5.50515 ,  5.90309  , 6.20412   ,   6.50515/

DATA TEMPA/ND1*0./
DATA TEMPB/ND1*0./

U0 = HH*CLIGHT/XLAMC/AKB/TEMPE  
ACON1 = 5.465727564D-11  
XLAMIO = 9.117532285D-6  
!
!---- atom determination
!
LEV = LABL1(II)  
NA = LE(LEV)  
INDI = INDAT1(II)  
!
! --- prepare chebychev polynomials for formula 60 to 65
!

IF (NFOR.GE.60 .AND. NFOR.LE.65) THEN 

  IF (CHAR.EQ.'A') THEN
    IF(TEMPE.EQ.TEMPA(LL)) THEN
       TS=TSA(:,LL)
    ELSE
       TEMPA(LL)=TEMPE
         XGAMMA=(0.542519D0*LOG(TEMPE))-5.620738D0
         TSA(0,LL)=1.D0
         TSA(1,LL)=XGAMMA
         DO I=2,8
           TSA(I,LL)=2.D0*XGAMMA*TSA(I-1,LL)-TSA(I-2,LL)
         ENDDO  
       TS=TSA(:,LL)
    ENDIF

  ELSE IF (CHAR.EQ.'B') THEN
    IF(TEMPE.EQ.TEMPB(LL)) THEN
       TS=TSB(:,LL)
    ELSE
       TEMPB(LL)=TEMPE
         XGAMMA=(0.542519D0*LOG(TEMPE))-5.620738D0
         TSB(0,LL)=1.D0
         TSB(1,LL)=XGAMMA
         DO I=2,8
           TSB(I,LL)=2.D0*XGAMMA*TSB(I-1,LL)-TSB(I-2,LL)
         ENDDO  
       TS=TSB(:,LL)
    ENDIF

  ELSE
    STOP ' WRONG CHAR IN TRATCOL'
  ENDIF

ENDIF

!   
!---- principal quantum number, only for hi and heii
!
IF (NFOR.EQ.13 .OR. NFOR.EQ.113 .OR. NFOR.EQ.14 .OR. NFOR.EQ.45 .OR. NFOR.EQ.46) THEN  
     LAB = LABL(NLO)  
     J = 1  

   10      CONTINUE  

     CAR = LAB(J:J)  

     IF (CAR.EQ.'0' .OR. CAR.EQ.'1' .OR. CAR.EQ.'2' .OR. &
&        CAR.EQ.'3' .OR. CAR.EQ.'4' .OR. CAR.EQ.'5' .OR. &
&        CAR.EQ.'6' .OR. CAR.EQ.'7' .OR. CAR.EQ.'8' .OR. &
&        CAR.EQ.'9') THEN
          J = J + 1  
          IJ = J  

   20           CONTINUE  

          CAR = LAB(J:J)  

          IF (CAR.EQ.'0' .OR. CAR.EQ.'1' .OR. CAR.EQ.'2' .OR. &
&            CAR.EQ.'3' .OR. CAR.EQ.'4' .OR. CAR.EQ.'5' .OR. &
&            CAR.EQ.'6' .OR. CAR.EQ.'7' .OR. CAR.EQ.'8' .OR. &
&            CAR.EQ.'9') THEN
               J = J + 1  
               GO TO 20  
          ELSE  
               J = J - 1  
               CAR = LAB(IJ:J)  
               READ (CAR,FMT='(I2)') NIVL  
!#             if(nivl.eq.0) nivl=10
               IF (NIVL.EQ.0) STOP ' NIVL = 0'  
          END IF  
     ELSE  
          J = J + 1  
          GO TO 10  
     END IF  

     LAB = LABL(NU)  
     J = 1  

   30      CONTINUE  

     CAR = LAB(J:J)  

     IF (CAR.EQ.'0' .OR. CAR.EQ.'1' .OR. CAR.EQ.'2' .OR. &
&        CAR.EQ.'3' .OR. CAR.EQ.'4' .OR. CAR.EQ.'5' .OR. &
&        CAR.EQ.'6' .OR. CAR.EQ.'7' .OR. CAR.EQ.'8' .OR. &
&        CAR.EQ.'9') THEN
          J = J + 1  
          IJ = J  
   40           CONTINUE  

          CAR = LAB(J:J)  

          IF (CAR.EQ.'0' .OR. CAR.EQ.'1' .OR. CAR.EQ.'2' .OR. &
&            CAR.EQ.'3' .OR. CAR.EQ.'4' .OR. CAR.EQ.'5' .OR. &
&            CAR.EQ.'6' .OR. CAR.EQ.'7' .OR. CAR.EQ.'8' .OR. &
&            CAR.EQ.'9') THEN
               J = J + 1  
               GO TO 40  
          ELSE  
               J = J - 1  
               CAR = LAB(IJ:J)  
               READ (CAR,FMT='(I2)') NIVU  
!#             if(nivu.eq.0) nivu=10
               IF (NIVU.EQ.0) STOP ' NIVU = 0'  
          END IF  
     ELSE  
          J = J + 1  
          GO TO 30  
     END IF  
END IF  
!
!---
!
IF (NFOR.EQ.1 .OR. NFOR.EQ.22) THEN  
!
!------- van regemorter
!
     FLU = DATA(INDI+2)  
     GBAR = DATA(INDI+3)  
     IF (GBAR.NE..2D0 .AND. GBAR.NE..7D0) STOP ' ERROR IN GBAR'  
!
!------- note that expin1(u0) is not e_1(u0) but e_1(u0)*exp(u0)
!
     GAMMAE = MAX(GBAR,.276D0*EXPIN1(U0))  
     ACON = ACON1* (XLAMC/XLAMIO)**2*FLU*U0  
     CLU = 14.5D0*ACON*SQRT(TEMPE)*EXP(-U0)*GAMMAE  

     RETURN  

ELSE IF(NFOR.EQ.11) THEN
!     
! warning: not checked by JO so far
!
      C_2 = DATA(INDI+2)
      C_1 = DATA(INDI+3)
      C0 =  DATA(INDI+4)
      C1 =  DATA(INDI+5)
      GAMMA = C_2/TEMPE**2 + C_1/TEMPE + C0 + C1*TEMPE
      CLU = ACON1 * SQRT(TEMPE) * GAMMA * EXP(-1.D0*U0)
!        
!        print *
!        print *, ' >>> c_2  : ', c_2
!        print *, ' >>> c_1  : ', c_1
!        print *, ' >>> c0   : ', c0
!        print *, ' >>> c1   : ', c1
!        print *, ' >>> tempe: ', tempe
!        print *, ' >>> u0   : ', u0
!        print *, ' >>> xlamc: ', xlamc
!        print *
!        print *, ' >>> gamma: ', gamma
!        print *, ' >>> clu  : ', clu
!        print *
!        
     RETURN

ELSE IF (NFOR.EQ.13) THEN  
!
!------- hydrogen atom
!
     IF (NIVL.GT.14 .OR. NIVU.GT.15) THEN  
!
          NDELTA = NIVU - NIVL  
!
!------- note that expin1(u0) is not e_1(u0) but e_1(u0)*exp(u0)
!------- idem with expin(5,u0)
!
          FLU = DATA(INDI+2)  
          ACON = ACON1* (XLAMC/XLAMIO)**2*FLU*U0  
          CLU = ACON*4.D0* (EXPIN1(U0)+0.148D0*U0*EXPIN(5,U0))* EXP(-U0)
          IF (NDELTA.NE.1) THEN  
               AU1 = 3.D0 - 1.2D0/DBLE(NIVL)  
               AU2 = 1.8D0 - 0.4D0/DBLE(NIVL)**2  
               AU3 = AU1 + 2.D0* (AU2-AU1)/DBLE(NDELTA)  
               CLU = CLU*AU3  
          END IF  
          CLU = CLU*SQRT(TEMPE)  

          RETURN  
     ELSE  
          CLU = HCOL(NIVL,NIVU,TEMPE)*EXP(-U0)  
!
!        here, we have included the correct temperature dependence,
!        hence:
!
         RETURN  
!
     END IF  


ELSE IF (NFOR.EQ.113) THEN  
!
!------- hydrogen atom, very old treatment (Burke et al. 1967)
!
          NDELTA = NIVU - NIVL  
!
!------- note that expin1(u0) is not e_1(u0) but e_1(u0)*exp(u0)
!------- idem with expin(5,u0)
!
          FLU = DATA(INDI+2)  
          ACON = ACON1* (XLAMC/XLAMIO)**2*FLU*U0  
          CLU = ACON*4.D0* (EXPIN1(U0)+0.148D0*U0*EXPIN(5,U0))* EXP(-U0)
          IF (NDELTA.NE.1) THEN  
               AU1 = 3.D0 - 1.2D0/DBLE(NIVL)  
               AU2 = 1.8D0 - 0.4D0/DBLE(NIVL)**2  
               AU3 = AU1 + 2.D0* (AU2-AU1)/DBLE(NDELTA)  
               CLU = CLU*AU3  
          END IF  
          CLU = CLU*SQRT(TEMPE)  

          RETURN 

ELSE IF (NFOR.EQ.14) THEN  

     FLU = DATA(INDI+2)  
     ACON = ACON1* (XLAMC/XLAMIO)**2*FLU*U0  
     CLU = ACON*EXP(1.D0)* (LOG(2.D0)+EXPIN1(U0))*EXP(-U0)  
     NDELTA = NIVU - NIVL  
     AN = DBLE(NIVL)  
     ADN = DBLE(NDELTA)  
     AN2 = AN - (AN-1.D0)/ADN  
     AU = MIN(ADN,AN2)  
     IF (NIVL.GT.1) AU = AU*1.1D0  
     CLU = CLU*AU*SQRT(TEMPE)  

     RETURN  

ELSE IF (NFOR.EQ.15) THEN  

     FLU = DATA(INDI+2)  
     ACON = ACON1* (XLAMC/XLAMIO)**2*FLU*U0  
     CLU = ACON*4.D0*EXPIN1(U0)*EXP(-U0)*SQRT(TEMPE)  

     RETURN  

ELSE IF (NFOR.EQ.16) THEN  

     E02 = .818730753D0  
     FLU = DATA(INDI+2)  
     ACON = ACON1* (XLAMC/XLAMIO)**2*FLU*U0  
     USU1 = U0 + 0.2D0  
     CLU = ACON*4.D0* (EXPIN1(U0)*EXP(-U0)- E02*U0/USU1*EXPIN1(USU1)* &
&      EXP(-USU1))
     CLU = CLU*SQRT(TEMPE)  

     RETURN  

ELSE IF (NFOR.EQ.17) THEN  

     X = U0 + 1.D0/DATA(INDI+5)  
     CLU = DATA(INDI+3)*EXP(-DATA(INDI+4)*U0)* (X*DATA(INDI+4)+2.D0)/X**3
     X = U0 + 1.D0/DATA(INDI+8)  
     CLU = CLU + DATA(INDI+6)*EXP(-DATA(INDI+7)*U0)* (X*DATA(INDI+7)+2.D0)/X**3
     CLU = DATA(INDI+2)*U0* (DATA(INDI+9)*EXPIN1(U0)*EXP(-U0)+U0*CLU)
     CLU = CLU*ACON1*SQRT(TEMPE)  

     RETURN  

ELSE IF (NFOR.EQ.19) THEN
  
     CLU = ACON1*DATA(INDI+2)*U0* (1.D0-U0*EXPIN1(U0))*EXP(-U0)* SQRT(TEMPE)

     RETURN  

ELSE IF (NFOR.EQ.20) THEN  

     ER = ERFCM(U0)*EXP(-U0**2)  
     CLU = 2.D0/3.D0* (1.D0+U0)*EXP(-U0) - SQPI*SQRT(U0)*ER* &
&      (1.D0+2.D0/3.D0*U0)
     CLU = CLU*ACON1*2.D0*DATA(INDI+2)*U0*U0*SQRT(TEMPE)  

     RETURN  

ELSE IF (NFOR.EQ.21) THEN  

     CLU = ACON1*DATA(INDI+2)*U0*U0* (EXPIN(2,U0)-EXPIN(3,U0))* &
      EXP(-U0)*SQRT(TEMPE)

     RETURN  

ELSE IF (NFOR.EQ.23) THEN	! taken from DETAIL, not checked by  JO
!	Hummer distorted wave

     FLU = DATA(INDI+12)/DATA(INDI+13)       
     IF (DATA(INDI+3).gt.0.D0) FLU=FLU*DATA(INDI+2)/DATA(INDI+3)
     TLOG = log10(TEMPE)-4.D0
     GAMMA = DATA(INDI+11)
     DO K =1,7
     	J=INDI+11-K
	GAMMA = GAMMA*TLOG+DATA(J)
     ENDDO
     
     CLU = GAMMA*EXP(-1.D0*U0)*FLU/SQRT(TEMPE)
     RETURN


ELSE IF (NFOR.EQ.24) THEN  
!
!------- allen's cross section
!
     OMEGA = DATA(INDI+2)  
     GAMMA = OMEGA/GL(LEV)*U0* (XLAMC/XLAMIO)  
     CLU = ACON1*SQRT(TEMPE)*EXP(-U0)*GAMMA  

     RETURN  

ELSE IF (NFOR.EQ.25) THEN  
!
!------- fit of effective collision strength
!
     T1 = DATA(INDI+2)  
     SCALE = DATA(INDI+3)  
     X = (TEMPE-T1)/SCALE  
     NFIT = INT(DATA(INDI+4))  
     GAMMA = 0.D0  

     DO J = 1,NFIT  
          AIFIT = DATA(INDI+4+J)  
          GAMMA = GAMMA + AIFIT*X** (J-1)  
     ENDDO

     CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*GAMMA  

     RETURN  

ELSE IF (NFOR.EQ.26) THEN  	! taken from DETAIL, not checked by  JO
!	powers of Log(T)

     T1 = DATA(INDI+2)  
     NFIT = INT(DATA(INDI+4)+0.5D0)  
     TLOG = LOG10(TEMPE)	
     GAMMA = 0.D0  
     
     DO J = 1,NFIT              ! bug detected/corrected Feb 2005, miguel
          AIFIT = DATA(NFIT-J+3+INDI)  
	  GAMMA = AIFIT + GAMMA*TLOG	
     ENDDO

     CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*GAMMA 

     RETURN  

ELSE IF (NFOR.EQ.32) THEN  	! taken from DETAIL, not checked by  JO
     
     ACON = 4.D0*ACON1*(109734.84D0*XLAMC/CLIGHT)**2*DATA(INDI+2)*U0
     CLU = ACON*SQRT(TEMPE)*(DATA(INDI+3)*EXPIN1(U0) + DATA(INDI+4))
     CLU = CLU*EXP(-1.D0*U0)
     RETURN


ELSE IF (NFOR.EQ.35) THEN  

     NNLO = INT(DATA(INDI+2)+0.5D0)  
     NNUP = INT(DATA(INDI+3)+0.5D0)  
!     CLU = HECOL(NNLO,NNUP,TEMPE)*EXP(-U0)  !old version for A7HHe
!    is replaced now by 
     CALL COLL21(TEMPE,NNLO,NNUP,GAMMA)
     CLU=GAMMA*EXP(-U0)
!
!        here, we have included the correct temperature dependence,
!        hence:
!
     RETURN  

ELSE IF (NFOR.EQ.37) THEN  


!-------- helium i

         GAMMA=PVR(U0)
         ACON=ACON1*U0*(XLAMC/XLAMIO)**2
         CLU=ACON*GAMMA*1.45D1*DATA(INDI+2)*EXP(-U0)*SQRT(TEMPE)

         RETURN

ELSE IF (NFOR.EQ.40) THEN  

!--------- helium i
      RYDO=3.29D15
      Z=ZL(NLO)+1.D0
      ZZR = Z * Z * RYDO
      ULO = FL(NLO) / ZZR
      UUP = FL(NU) / ZZR
      XNULO=1.0D0/SQRT(ULO)
      XNUUP=1.0D0/SQRT(UUP)
      
!        classical energy band

      EPS1=ULO-1.0D0/((XNUUP-0.5D0)**2)
      EPS2=ULO-1.0D0/((XNUUP+0.5D0)**2)
      P1=PFTN(TEMPE,ULO,EPS1)
      P2=PFTN(TEMPE,ULO,EPS2)
      E1=EXP(-1.579D5*EPS1/TEMPE)
      E2=EXP(-1.579D5*EPS2/TEMPE)
      IF ( EPS1.GT.0.0D0.AND.EPS2.GT.0.0D0 ) THEN
           F=E2*P2-E1*P1
      ELSEIF ( EPS1.LT.0.0D0.AND.EPS2.GT.0.0D0 ) THEN
           F=E2*P2-P1
      ELSEIF ( EPS1.LT.0.0D0.AND.EPS2.LT.0.0D0 ) THEN
           F=P2-P1
      ELSE
           WRITE(6,220) TEMPE,EPS1,EPS2
220        FORMAT(' TRATCOL : EPS CONDITION FAILS FOR TEM=',   &
                 E10.3,'  EPS1=',E10.3,'  EPS2=',E10.3/)
           STOP
      ENDIF
      CLU=8.69D-8*F*(1.579D5)**1.5D0
      FAC= DATA(INDI+2)*DATA(INDI+2) * 4.D0
      CLU=CLU/FAC * GL(NU)/TEMPE/TEMPE * SQRT(TEMPE)

      RETURN

ELSE IF(NFOR.EQ.45) THEN
        STOP ' formula 45 not in use'
!
!-------- hydrogen atom
!-------- Johnson 1972
!
        RN=1.D0/NIVL
        N2=NIVL*NIVL
        NP2=NIVU*NIVU
        X=1.D0-N2/NP2
        RX=1.D0/X
        RX2=RX*RX
        TWON2=2.D0*N2
        TWON2RX=TWON2*RX
        LOGTWNX=LOG(TWON2RX)
        ANN=DATA(INDI+2)*TWON2RX
        IF(NIVL.EQ.1) THEN
          BN=-6.03D-1
          RFAC=4.5D-1
        ELSE
          BN=(((-2.809D1*RN+3.624D1)*RN-1.863D1)*RN+4.D0)*RN
          RFAC=1.94D0*RN**1.57D0
        ENDIF
        RFAC=RFAC*X
        BNN=TWON2*TWON2*RX2/(NP2*NIVU)*(1.D0+4.D0/3.D0*RX+BN*RX2)
        CON1=5.436D-11*TWON2RX
        CON2=BNN-ANN*LOGTWNX 
        YEX=EXPIN1(U0)*EXP(-U0)
        YEX2=EXPIN(2,U0)*EXP(-U0)
        RY=1.D0/U0
        RY2=RY+0.5D0
        Z0=U0+RFAC
        ZEX=EXPIN1(Z0)*EXP(-Z0)
        ZEX2=EXPIN(2,Z0)*EXP(-Z0)
        RZ=1.D0/Z0
        RZ2=RZ+0.5D0
        CLU=CON1*U0*U0*SQRT(TEMPE)* &
&       (ANN*(RY2*YEX-RZ2*ZEX)+CON2*(RY*YEX2-RZ*ZEX2))
 
        
        RETURN
 
  ELSE IF(NFOR.EQ.46) THEN
!
!-------- hydrogen atom
!-------- Percival & Richards 1978
!
  
        Z = ZL(NLO) + 1.D0
        ACON = NIVL*NIVL*NIVL*NIVL * GL(NU)/(GL(NLO)*Z*Z)
        GAMMA = 0.D0
        UPSFAC = 8.631D-6 / SQRT(TEMPE)
        DO J=1,NGAU
          EE=6.33D-6*TEMPE*XLAG(J)
          GAMMA=GAMMA+WLAG(J)*PRHCOL(NIVL,NIVU,Z,EE)
        ENDDO
        CLU = GAMMA * ACON * UPSFAC * EXP(-U0)
        
        RETURN

ELSE IF(NFOR.EQ.51) THEN
! omega and slope as input
        OMEGA=DATA(INDI+2)
        SLOPE=DATA(INDI+3)  
        OM=OMEGA + (TEMPE - 2.D4)/1.D4 * SLOPE ! FROM ADI
        OM=MAX(OMEGA/10.,OM)
        CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*OM  

        RETURN

ELSE IF(NFOR.EQ.52) THEN
! omega = 1
        CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)  

        RETURN

!New formulas included by Jorge
ELSE IF(NFOR.EQ.60) THEN

        DO I=0,3
          AI(I)=DATA(INDI+2+I)
        ENDDO

        LGAMMA1=0.5D0*AI(0)+DOT_PRODUCT(AI(1:3),TS(1:3))

        GAMMA=EXP(LGAMMA1)

        CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*GAMMA

        RETURN

ELSE IF(NFOR.EQ.61) THEN

        DO I=0,7
          AI(I)=DATA(INDI+2+I)
        ENDDO

        LGAMMA1=0.5D0*AI(0)+DOT_PRODUCT(AI(1:3),TS(1:3))
        LGAMMA2=0.5D0*AI(4)+DOT_PRODUCT(AI(5:7),TS(1:3))

        GAMMA1=EXP(LGAMMA1)
        GAMMA2=EXP(LGAMMA2)

        GAMMA=GAMMA1+GAMMA2

        CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*GAMMA

        RETURN

ELSE IF(NFOR.EQ.62) THEN

        DO I=0,11
          AI(I)=DATA(INDI+2+I)
        ENDDO
      
        LGAMMA1=0.5D0*AI(0)+DOT_PRODUCT(AI(1:3),TS(1:3))
        LGAMMA2=0.5D0*AI(4)+DOT_PRODUCT(AI(5:7),TS(1:3))
        LGAMMA3=0.5D0*AI(8)+DOT_PRODUCT(AI(9:11),TS(1:3))

        GAMMA1=EXP(LGAMMA1)
        GAMMA2=EXP(LGAMMA2)
        GAMMA3=EXP(LGAMMA3)

        GAMMA=GAMMA1+GAMMA2+GAMMA3

        CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*GAMMA

        RETURN
        
ELSE IF(NFOR.EQ.63) THEN

        DO I=0,15
          AI(I)=DATA(INDI+2+I)
        ENDDO

        LGAMMA1=0.5D0*AI(0)+DOT_PRODUCT(AI(1:3),TS(1:3))
        LGAMMA2=0.5D0*AI(4)+DOT_PRODUCT(AI(5:7),TS(1:3))
        LGAMMA3=0.5D0*AI(8)+DOT_PRODUCT(AI(9:11),TS(1:3))
        LGAMMA4=0.5D0*AI(12)+DOT_PRODUCT(AI(13:15),TS(1:3))
       
        GAMMA1=EXP(LGAMMA1)
        GAMMA2=EXP(LGAMMA2)
        GAMMA3=EXP(LGAMMA3)
        GAMMA4=EXP(LGAMMA4)

        GAMMA=GAMMA1+GAMMA2+GAMMA3+GAMMA4

        CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*GAMMA

        RETURN
        
ELSE IF(NFOR.EQ.64) THEN

        DO I=0,23
          AI(I)=DATA(INDI+2+I)
        ENDDO

        LGAMMA1=0.5D0*AI(0)+DOT_PRODUCT(AI(1:3),TS(1:3))
        LGAMMA2=0.5D0*AI(4)+DOT_PRODUCT(AI(5:7),TS(1:3))
        LGAMMA3=0.5D0*AI(8)+DOT_PRODUCT(AI(9:11),TS(1:3))
        LGAMMA4=0.5D0*AI(12)+DOT_PRODUCT(AI(13:15),TS(1:3))
        LGAMMA5=0.5D0*AI(16)+DOT_PRODUCT(AI(17:19),TS(1:3))
        LGAMMA6=0.5D0*AI(20)+DOT_PRODUCT(AI(21:23),TS(1:3))

        GAMMA1=EXP(LGAMMA1) 
        GAMMA2=EXP(LGAMMA2) 
        GAMMA3=EXP(LGAMMA3) 
        GAMMA4=EXP(LGAMMA4)
        GAMMA5=EXP(LGAMMA5)
        GAMMA6=EXP(LGAMMA6)

        GAMMA=GAMMA1+GAMMA2+GAMMA3+GAMMA4+GAMMA5+GAMMA6

        CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*GAMMA

        RETURN
        
ELSE IF(NFOR.EQ.65) THEN

        DO I=0,17
          AI(I)=DATA(INDI+2+I)
        ENDDO


        LGAMMA1=0.5D0*AI(0)+DOT_PRODUCT(AI(1:8),TS(1:8))
        LGAMMA2=0.5D0*AI(9)+DOT_PRODUCT(AI(10:17),TS(1:8))

        GAMMA1=EXP(LGAMMA1) 
        GAMMA2=EXP(LGAMMA2) 

        GAMMA=GAMMA1+GAMMA2

        CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*GAMMA

        RETURN

ELSE IF(NFOR.EQ.66) THEN

       DO I=0,9
          ASP(I)=DATA(INDI+2+I)
       ENDDO

       LTEMPE=LOG10(TEMPE)

       CALL DEVP(XSP,ASP,NSP,YP1,YPN)

       CALL SPLINE_COL(XSP,ASP,NSP,YP1,YPN,ASP2)

       CALL SPLINTER(XSP,ASP,ASP2,NSP,LTEMPE,GAMMA)

       CLU = 8.631D-6/ (SQRT(TEMPE)*GL(LEV))*EXP(-U0)*GAMMA

       RETURN
!***********

ELSE IF (NFOR.EQ.100) THEN
! fits to "exact" data, used for hydrogen (Butler, 2004) or helium II

TE10D4=LOG10(TEMPE)

UPSFAC=8.631D-6/SQRT(TEMPE)

NPT=INT(DATA(INDI+2)+.5)

FOUND = .FALSE.

TGRID=0.
NPTSTORE=NPT
TGRID(1:NPT)=DATA(INDI+3:INDI+2+NPT)

IF (TGRID(1).GT.TE10D4) THEN
   INDX=0
   DELTA=0.D0
   FOUND=.TRUE.
ELSE IF (TGRID(NPT).LT.TE10D4) THEN 
   INDX=NPT
   DELTA=0.D0
   FOUND=.TRUE.
ELSE
  DO IPT = 1,NPT-1
    IF ((.NOT.FOUND).AND.(TGRID(IPT).LT.TE10D4) &
&    .AND.(TGRID(IPT+1).GE.TE10D4)) THEN
       FOUND=.TRUE.
       INDX=IPT
       DELTA = TE10D4-TGRID(IPT)
       DELTA = DELTA/(TGRID(IPT+1)-TGRID(IPT))
    ENDIF
  ENDDO
ENDIF

IF (INDX.EQ.0) THEN
   UPSLN=DATA(INDI+2+NPT+1)
ELSEIF (INDX.EQ.NPT) THEN
   UPSLN=DATA(INDI+2+NPT+NPT)
ELSE
   UPSLN=DATA(INDI+2+NPT+INDX+1)*DELTA
   UPSLN=UPSLN+(1.-DELTA)*DATA(INDI+2+NPT+INDX)
ENDIF

CON1=UPSLN/GL(NLO)

CLU=CON1*UPSFAC*EXP(-U0)                    

RETURN
!***********

ELSE IF (NFOR.EQ.101) THEN
! tabulated collision-strengths, in combination with formula 100
! the temperature grid is stored from the last call to formula 100 

TE10D4=LOG10(TEMPE)

UPSFAC=8.631D-6/SQRT(TEMPE)

NPT=INT(DATA(INDI+2)+.5)
IF(NPT.NE.NPTSTORE) THEN
  PRINT*,NPT,NPTSTORE
  STOP ' SOMETHING WRONG WITH FORMULA 101'
ENDIF

FOUND = .FALSE.

IF (TGRID(1).GT.TE10D4) THEN
   INDX=0
   DELTA=0.D0
   FOUND=.TRUE.
ELSE IF (TGRID(NPT).LT.TE10D4) THEN 
   INDX=NPT
   DELTA=0.D0
   FOUND=.TRUE.
ELSE
  DO IPT = 1,NPT-1
    IF ((.NOT.FOUND).AND.(TGRID(IPT).LT.TE10D4) &
&    .AND.(TGRID(IPT+1).GE.TE10D4)) THEN
       FOUND=.TRUE.
       INDX=IPT
       DELTA = TE10D4-TGRID(IPT)
       DELTA = DELTA/(TGRID(IPT+1)-TGRID(IPT))
    ENDIF
  ENDDO
ENDIF

IF (INDX.EQ.0) THEN
   UPSLN=DATA(INDI+3)
ELSEIF (INDX.EQ.NPT) THEN
   UPSLN=DATA(INDI+2+NPT)
ELSE
   UPSLN=DATA(INDI+2+INDX+1)*DELTA
   UPSLN=UPSLN+(1.-DELTA)*DATA(INDI+2+INDX)
ENDIF

CON1=UPSLN/GL(NLO)

CLU=CON1*UPSFAC*EXP(-U0)                    

RETURN
!***********

ELSE  

     PRINT *,'FORMULA NOT IMPLEMENTED YET'  
     PRINT *,NFOR  

     STOP  

END IF  

END
!
!----------------------------------------------------------------------- &
!
   FUNCTION PRHCOL(N,NP,Z,E)
    USE NLTE_TYPE
    USE NLTE_DIM
    IMPLICIT NONE
    REAL(DP) :: PRHCOL
    REAL(DP) :: L,Z,E,E2,Z2,TWOZNPE,RNP2,RS,FNNP,RFNNP, &
  &   FACN,A,D,F,G,FAC,XP,XM,Y,H,EGHTTHD, &
  &   TWOTHD,PIA02
    INTEGER(I4B) :: S,TWSP1,N,NP,N2
    PARAMETER(EGHTTHD=8.D0/3.D0,TWOTHD=2.D0/3.D0,PIA02=8.797D-17)
    E2 = E*E
    Z2 = Z*Z
    N2=N*N
    TWOZNPE=2.D0*Z/(N2*E)
    RNP2=1.D0/(NP*NP)
    S = NP-N
    TWSP1 = 2*S+1
    RS = 1.D0/S
    FNNP = NP*N
    RFNNP = 1.D0/FNNP
    FACN = NP*RS/N
    A = EGHTTHD*RS*FACN**3*(0.184D0-0.04D0*RS**TWOTHD)* &
  &      (1.D0-0.20D0*S*RFNNP)**TWSP1
    D = EXP(-Z2*RFNNP/E2)
    L = LOG((1.D0+0.53D0*E2*FNNP/Z2)/(1.D0+0.4D0*E))
    F = (1.D0-0.3D0*S*D*RFNNP)**TWSP1
    G = 0.5D0*(E*N2/(Z*NP))**3
    FAC = SQRT(2.D0-N2*RNP2)
    XP = TWOZNPE/(FAC+1.D0)
    XM = TWOZNPE/(FAC-1.D0)
    Y=1.D0-0.25D0*D*LOG(18.D0*S)*RS
    Y=1.D0/Y
    H=C2(XM,Y)-C2(XP,Y)
    PRHCOL=A*D*L+F*G*H
    RETURN
    CONTAINS
    REAL(DP) FUNCTION C2(X,Y)
    REAL(DP) :: X,Y
    C2=X*X*LOG(1.D0+TWOTHD*X)/(2.D0*Y+1.5D0*X)
  END FUNCTION C2
  END FUNCTION PRHCOL
!
!-----------------------------------------------------------------------
!
SUBROUTINE TRATRAD(L,NLO,NU,FLU,XLAMC,XNL,XNU,XNE,GL,VELR,GVEL, &
&                   TLU,TUL,BETO,SR,VMAX,XMUST,II,TEMPE,CLF, &
&                   VTHERM,FIC,TCL_FAC_LINE,TCL_FAC_CONT)

! UPDATED FOR CLUMPING

USE nlte_type
USE nlte_dim
USE fund_const
USE nlte_var, ONLY: AS,A2S,A3S,BS,B2S,B3S, &
& FRE,WFRE,HC,IFRE, &
& OPAC,STRUE,XJ,XXK,ALO, &
& OPACLIN, &
& TLUMAT,TULMAT,BETOLD,INDEXRBB,INDEXCMF, &
& WEIG9=>W9,WEIG5=>W5, &
& OPTLINES,THOMSON_LINES
!,RAYLEIGH

USE nlte_porvor, ONLY: OPA_EFF_RAT_OLD

IMPLICIT NONE
!
!---- radiative treatment. all elements needed for rate equation
!---- coefficientes are calculated (in the framework of sobolev plus!
!---- continuum absortion approach)
!
!---- calculation of bcIc acclerated
!
!     UPDATED FOR CLUMPING: present assumption: clumps small compared
!                           to Sobolev length; thus, "usual" averaging 
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
!
! maximum allowed tau_sob for inversion, to avoid too large corrections
REAL(DP), PARAMETER :: TAUMAX=2.3D0,SQTHIRD=0.5773502692D0
REAL(DP), PARAMETER :: CONST1=E2MC2/HH*4.D0*PI**2  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  BETO,FLU,GVEL,SR,TEMPE,TLU,TUL,VELR,VMAX, &
&                 VTHERM,XLAMC,XMUST,XNE,XNL,XNU,CLF
REAL(DP) :: FIC,TCL_FAC_LINE,TCL_FAC_CONT 
INTEGER(I4B) ::  II,L,NLO,NU  
!     ..
!     .. array arguments ..
REAL(DP) ::  GL(ID_LLEVS)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AUL,BB1,BB2,BCIC,BETAP,BETAS,BETBIG,BETBIM, &
&            BETLEM,BETLES,BLU,BUL,DELM1,DELM2,DN, &
&            OPACON,S1,S2,SCONT,TAUS,TAUS1,TAUS2,UBAR,WI1,WI2, &
&            WILOG,XKLI,XKLINE,XMUE1,XMUE2,XNUE,XXX,TAUSF, &
&            F1,F2,EDDF,SL,JBAR

REAL(DP) :: AAS,AA2S,AA3S,BBS,BB2S,BB3S

REAL(DP) :: TCL,FAC1 

INTEGER(I4B) ::  I,INDL,K  
LOGICAL FASTBETA,FASTBCIC,INVERSION,WARNING  
!     ..
!     .. external functions ..
REAL(DP) ::  BETMUE,BETINV,QFUNC,UPS,U2_INV
EXTERNAL BETMUE,BETINV,QFUNC,UPS,U2_INV
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,DBLE,LOG10
!     ..

INDL = INDEXRBB(II)  
!
!---- einstein coefficients and line opacity
!---- (warning! xlamc= central lambda in cm)
!
XNUE = 1.D0/XLAMC  

IF (XNUE.GT.FRE(IFRE) .OR. XNUE.LT.FRE(1)) THEN  
     TLU = 0.D0  
     TUL = 0.D0  
     RETURN  
END IF  

BLU = CONST1*XLAMC*FLU  
BUL = GL(NLO)*BLU/GL(NU)  
AUL = HC2/XLAMC**3*BUL  
DN=(XNL/GL(NLO)-XNU/GL(NU))/CLF  ! corrected

INVERSION=.FALSE.
IF(DN.LT.0.) INVERSION=.TRUE.

XKLI=PI*E2MC2*CLIGHT*GL(NLO)*FLU*DN         
XKLINE = XKLI*XLAMC*SR/VMAX  

     K = IFRE  
     XXX = FRE(IFRE)  
     DO WHILE (XNUE.LT.XXX)  
        K=K-1  
        XXX=FRE(K)  
     END DO  

!---- as discussed by JS & JP, we use MAX(tcl_line,tcl_bg) = MIN(fac1,opa_eff_rat_old)
!     (depth variable L, freq. variable K)
!---- inversion consistent with the non-inversion case
     TCL = ABS(XKLINE) * TCL_FAC_LINE *VMAX/SR    
!---- back-correct, since tcl_fac_line already includes SR/VMAX 
     FAC1 = (1.+FIC*TCL)/(1.+TCL)
!     IF (TCL.LT.0.0) FAC1=1.0D0   !old treatment of inversion (poor man's way..) 
     FAC1 = MIN( FAC1, OPA_EFF_RAT_OLD(L,K) )
     IF (FAC1.LE.0.0D0.OR.FAC1.GT.1.0D0) THEN 
        Print*,FAC1,OPA_EFF_RAT_OLD(L,K)
        Print*,TCL,L,K
        stop ' opa_eff > <opa>, TRATRAD_1'
     ENDIF
     !IF (XLAMC*1.d8.gt.1000.and.XLAMC*1.d8.lt.3000.) THEN 
     !   Print*,'test-JS-TRATRAD:'
     !   Print*,L,K,XLAMC*1.d8
     !   Print*,XKLINE,TCL_FAC_LINE 
     !   Print*,FAC1,OPA_EFF_RAT_OLD(L,K),L
     !   Print*,'-----------'
     !ENDIF
     !
     XKLINE = XKLINE * FAC1
!    now effective quantity
!
!    changed: linear interpolation for generalized blocking factors,
!    log-log interpolation for scont and opac
!
     WILOG = LOG10(FRE(K)/XNUE)/LOG10(FRE(K+1)/FRE(K))  
     WI1 = (FRE(K+1)-XNUE)/ (FRE(K+1)-FRE(K))  
     WI2 = 1.D0 - WI1  

! at first, check whether tausf is in critical range
! to do so, calculate at first eddington factor

     F1=XXK(L,K)/XJ(L,K)
     F2=XXK(L,K+1)/XJ(L,K+1)
     EDDF=F1*WI1+F2*WI2

     TAUSF=XKLINE/QFUNC(SQRT(EDDF),VELR,GVEL)

     FASTBCIC=.FALSE.
     FASTBETA=.FALSE.
     IF(.NOT.INVERSION) THEN 

!    slowbeta range has to include slowbcic range, otherwise betles to betbim
!    required for slowbeta are undefined
       
       IF(TAUSF.LE.0.01D0.OR.TAUSF.GT.8.D0) FASTBCIC=.TRUE.
       IF(TAUSF.LE.0.0001D0.OR.TAUSF.GT.10.D0) FASTBETA=.TRUE.
     ELSE
       IF(ABS(TAUSF).GT.TAUMAX) FASTBCIC=.TRUE.
     ENDIF
     
     IF(FASTBCIC.OR.INVERSION) THEN
       AAS = XJ(L,K)*10.D0** (LOG10(XJ(L,K)/XJ(L,K+1))* WILOG)
     ELSE 
       AAS = AS(K)*WI1 + AS(K+1)*WI2  
       BBS = BS(K)*WI1 + BS(K+1)*WI2  
       AA2S = A2S(K)*WI1 + A2S(K+1)*WI2  
       AA3S = A3S(K)*WI1 + A3S(K+1)*WI2  
       BB2S = B2S(K)*WI1 + B2S(K+1)*WI2  
       BB3S = B3S(K)*WI1 + B3S(K+1)*WI2  
     ENDIF
!
!---- s_cont and beta_puls
!

     IF(OPTLINES) THEN
!       S1 = STRUE(L,K) + XJ(L,K)* &
!&        (XNE*SIGMAE*AMH/CLF+THOMSON_LINES(L,K)+RAYLEIGH(L,K)/CLF)/OPAC(L,K)  
!       S2 = STRUE(L,K+1) + XJ(L,K+1)* &
!&        (XNE*SIGMAE*AMH/CLF+THOMSON_LINES(L,K+1)+RAYLEIGH(L,K+1)/CLF)/OPAC(L,K+1)  

!---- here only OPAC is effective, so we have to back-correct again 
!---- BUT again OPAC HAS *NOT* been updated in this loop yet
!---- (i.e. NLTE_APPROX and OPACITM called, and so OPA_EFF_RAT updated,
!      but OPACITC NOT called yet) Thus OPA_EFF_RAT_OLD. 
       S1 = STRUE(L,K) +   XJ(L,K)*(XNE*SIGMAE*AMH/CLF+ &
&        THOMSON_LINES(L,K))/OPAC(L,K) * OPA_EFF_RAT_OLD(L,K)  !back-corrected  
       S2 = STRUE(L,K+1) + XJ(L,K+1)*(XNE*SIGMAE*AMH/CLF+ &
&        THOMSON_LINES(L,K+1))/OPAC(L,K+1) * OPA_EFF_RAT_OLD(L,K+1) !back-corrected

     ELSE
     S1 = STRUE(L,K) +   XNE*SIGMAE*AMH*XJ(L,K  )/CLF/ OPAC(L,K) * OPA_EFF_RAT_OLD(L,K) !back-corrected  
     S2 = STRUE(L,K+1) + XNE*SIGMAE*AMH*XJ(L,K+1)/CLF/ OPAC(L,K+1) * OPA_EFF_RAT_OLD(L,K+1) !back-corrected 
     ENDIF

     SCONT = S1*10.D0** (LOG10(S1/S2)*WILOG)  
     
     OPACON = OPAC(L,K)*10.D0** (LOG10(OPAC(L,K)/OPAC(L,K+1))* WILOG) !already OK
!---- OPACON already effective 
     OPACLIN(INDL,L) = OPACON  
!
IF(.NOT.INVERSION) THEN
!
!--------------------------------------------------------------------
!
! this is the path for no inversion
!
BETAP = OPACON/XKLINE*SR*VTHERM/VMAX  ! ratio remains unaffected
!---- same goes here -- both OPACON and XKLINE are already effective 

IF(FASTBETA) THEN
!
! beta evaluated at MUE^2 = 1/3
!  
  TAUS = XKLINE/QFUNC(SQTHIRD,VELR,GVEL)  
  BETAS = BETMUE(TAUS)
  UBAR = UPS(TAUS,BETAP)
ELSE
!
!---- angular integration
!
  BETLES = 0.D0  
  BETBIG = 0.D0  
  BETLEM = 0.D0  
  BETBIM = 0.D0  
  UBAR = 0.D0  

!
!------- nine points for both intervals
!
  DELM1 = XMUST/8.D0  
  DELM2 = (1.D0-XMUST)/8.D0  
  DO I = 1,9  
          XMUE1 = DBLE(I-1)*DELM1  
          XMUE2 = XMUST + DBLE(I-1)*DELM2  
          TAUS1 = XKLINE/QFUNC(XMUE1,VELR,GVEL)  
          TAUS2 = XKLINE/QFUNC(XMUE2,VELR,GVEL)  
          BB1 = BETMUE(TAUS1)*DELM1  
          BB2 = BETMUE(TAUS2)*DELM2  
          BETLES = BETLES + BB1*WEIG9(I)  
          BETBIG = BETBIG + BB2*WEIG9(I)  
          BETLEM = BETLEM + BB1*XMUE1*WEIG9(I)  
          BETBIM = BETBIM + BB2*XMUE2*WEIG9(I)  
          UBAR = UBAR + (UPS(TAUS1,BETAP)*DELM1+ &
&           UPS(TAUS2,BETAP)*DELM2)*WEIG9(I)
  END DO

  BETAS = BETLES + BETBIG  
ENDIF  

IF(FASTBCIC) THEN
!
! betac_ic at MUE^2 = EDDF  
!
  BCIC=(1.D0-EXP(-TAUSF))/TAUSF*AAS
ELSE 
!
!---- more accurate integration: beta*i_inc integrated in mue and beta_sob
!
  BCIC = 0.5D0* (AAS+AA3S)*BETBIG + &
&         .5D0* (BBS-BB3S)*BETBIM + &
&         .5D0* (AA2S+AA3S)*BETLES + &
&         .5D0* (BB2S-BB3S)*BETLEM
ENDIF

IF (BCIC.LT.0.D0) then
  IF(BCIC.LT.-1.D-12) PRINT*,' WARNING: BETAC IC NEGATIVE AT TRANSITION:', &
&   II,XLAMC*1.D8
  BCIC=0.   
ENDIF

BETO = BETAS  
!
!---- terms for rate equations
!
TLU = BLU* (BCIC+SCONT*UBAR)  
TUL = BUL* (BCIC+SCONT*UBAR) + AUL* (BETAS+UBAR)  

!if (xlamc.lt.305.d-8.and.xlamc.gt.303.d-8) then
!  print*,l,bcic,betas,betap,ubar,scont,tlu,tul
!  print* 
!endif   
!
RETURN
!
!--------------------------------------------------------------------
!
ELSE ! path for inversion

! note: provide always the possibility for a cancellation of Jnue and Scont

WARNING=.FALSE.
XKLINE=ABS(XKLINE) ! already corrected
TAUSF=ABS(TAUSF)

BETAP = OPACON/XKLINE*SR*VTHERM/VMAX  

IF(FASTBCIC) THEN
!
! strong inversion, approx. treatment ok
!
  IF (TAUSF.LT.TAUMAX) STOP ' ERROR IN FASTBCIC'
  TAUSF=TAUMAX
  WARNING=.TRUE.
  BETAS=BETINV(TAUSF)

! ubar is now u2
  UBAR=U2_INV(TAUSF,BETAP)

! bcic now corresponds to Bc(Ic-Sc) = beta(sqrt(f))*(J-Sc)

ELSE
!
!---- angular integration, J drawn out of integral
!
  BETLES = 0.D0  
  BETBIG = 0.D0  
  UBAR = 0.D0  
!
!------- nine points for both intervals
!
  DELM1 = XMUST/8.D0  
  DELM2 = (1.D0-XMUST)/8.D0  
  DO I = 1,9  
          XMUE1 = DBLE(I-1)*DELM1  
          XMUE2 = XMUST + DBLE(I-1)*DELM2  
          TAUS1 = XKLINE/QFUNC(XMUE1,VELR,GVEL)  
          IF(TAUS1.GT.TAUMAX) THEN
            TAUS1=TAUMAX
            WARNING=.TRUE.
          ENDIF
          TAUS2 = XKLINE/QFUNC(XMUE2,VELR,GVEL)  
          IF(TAUS2.GT.TAUMAX) THEN
            TAUS2=TAUMAX
            WARNING=.TRUE.
          ENDIF
          BB1 = BETINV(TAUS1)*DELM1  
          BB2 = BETINV(TAUS2)*DELM2  
          BETLES = BETLES + BB1*WEIG9(I)  
          BETBIG = BETBIG + BB2*WEIG9(I)  
          UBAR = UBAR + (U2_INV(TAUS1,BETAP)*DELM1+ &
&           U2_INV(TAUS2,BETAP)*DELM2)*WEIG9(I)
  END DO

  BETAS = BETLES + BETBIG  

ENDIF

BCIC=BETAS*(AAS-SCONT)

SL=-AUL*XNU/(XNL*BLU-XNU*BUL)   ! here, CLF cancels out (thin and thick clumping)

JBAR=BCIC+SCONT*(1.+UBAR)+SL*UBAR

BETO = BETAS  
!
!---- terms for rate equations
!
IF (JBAR.GT.0.D0) THEN

TLU = BLU* (BCIC+SCONT*(1.D0+UBAR))  
TUL = BUL* (BCIC+SCONT*(1.D0+UBAR)) + AUL* (1.D0+UBAR)  

ELSE !can only happen if Jnue < Scont
PRINT*,'JBAR NEGATIVE AT TRANSITION ',II,XLAMC*1.D8

! jbar set to zero, use explicit rates
  
TLU = 0. 
TUL = AUL  
ENDIF

IF(WARNING) &
& PRINT*,' STRONG INVERSION AT',II,XLAMC*1.D8,L

RETURN  

ENDIF

END
!
!-----------------------------------------------------------------------
!
FUNCTION BETMUE(TAUS)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
REAL(DP) ::  TAUS,BETMUE
!     ..
!     .. local scalars ..
REAL(DP) ::  PREC  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP  
!     ..

PREC = 1.D-8  

IF (TAUS.LT.PREC) THEN  
     BETMUE = 1.D0 - 0.5D0*TAUS  
ELSE  
     BETMUE = (1.D0-EXP(-TAUS))/TAUS  
END IF  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION BETINV(TAUS)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
REAL(DP) ::  TAUS,BETINV
!     ..
!     .. local scalars ..
REAL(DP) ::  PREC  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP  
!     ..

PREC = 1.D-8  

IF (TAUS.LT.PREC) THEN  
     BETINV = 1.D0 + 0.5D0*TAUS  
ELSE  
     BETINV = (EXP(TAUS)-1.D0)/TAUS  
END IF  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION QFUNC(XMUE,VELR,GVEL)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
REAL(DP) ::  GVEL,VELR,XMUE,QFUNC
!     ..

QFUNC = VELR* (1.D0-XMUE**2) + GVEL*XMUE**2  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION UPS(TAU,BETA)  

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: UFUNG,UFUNK
IMPLICIT NONE
!
!---- approximation of function u (used in sobolev+continuum theory)
!---- all beta's and tau's are possible.
!---- e. santolaya, 1992 (inspired in gudrun taresch's)
!
!     .. scalar arguments ..
REAL(DP) ::  BETA,TAU,UPS
!     ..
!     .. local scalars ..
REAL(DP) ::  AUX,B1,B2,BETAI,BETAL,BETAS,BETAU2,T1,T2,TAUI, &
&                 TAUL,TAUS,U,UI,UII,UIS,US,USI,USS
INTEGER(I4B) ::  ICB,ICT,NB,NE,NN,NT  
!     ..
!     .. local arrays ..
REAL(DP) ::  BETAG(17),BETAK(76)  
!     ..
!     .. external functions ..
REAL(DP) ::  ERFCM  
EXTERNAL ERFCM  
!     ..
!     .. intrinsic functions ..
!INTRINSIC DBLE,INT,LOG10,SQRT  
!     ..
!     .. data statements ..
DATA BETAK/1.0D-10,2.8D-10,4.6D-10,6.4D-10,8.2D-10,1.0D-9,2.8D-9, &
&     4.6D-9,6.4D-9,8.2D-9,1.0D-8,2.8D-8,4.6D-8,6.4D-8,8.2D-8, &
&     1.0D-7,2.8D-7,4.6D-7,6.4D-7,8.2D-7,1.0D-6,2.8D-6,4.6D-6, &
&     6.4D-6,8.2D-6,1.0D-5,2.8D-5,4.6D-5,6.4D-5,8.2D-5,1.0D-4, &
&     2.8D-4,4.6D-4,6.4D-4,8.2D-4,1.0D-3,2.8D-3,4.6D-3,6.4D-3, &
&     8.2D-3,1.0D-2,2.8D-2,4.6D-2,6.4D-2,8.2D-2,1.0D-1,2.8D-1, &
&     4.6D-1,6.4D-1,8.2D-1,1.0D+0,2.8D+0,4.6D+0,6.4D+0,8.2D+0, &
&     1.0D+1,2.8D+1,4.6D+1,6.4D+1,8.2D+1,1.0D+2,2.8D+2,4.6D+2, &
&     6.4D+2,8.2D+2,1.0D+3,2.8D+3,4.6D+3,6.4D+3,8.2D+3,1.0D+4, &
&     2.8D+4,4.6D+4,6.4D+4,8.2D+4,1.0D+5/


DATA BETAG/1.D-10,1.D-9,1.D-8,1.D-7,1.D-6,1.D-5,1.D-4,1.D-3,1.D-2, &
&     1.D-1,5.D-1,1.D+0,1.D+1,1.D+2,1.D+3,1.D+4,1.D+5/
!     ..

TAUL = LOG10(TAU)  

BETAL = LOG10(BETA)  
!
!---- casuistry begins...
!
IF (TAUL.GT.2.D0) THEN  

     IF (BETAL.GT.5.D0) THEN  
          UPS = 1.D0  
          GO TO 10  
     END IF  
!
!------- interpolation indexes for tau and beta
!
     NT = INT(TAUL) - 1  
     NB = INT(BETAL+10.D0) + 1  

     IF (BETA.GE.0.5D0) NB = NB + 1  
!
!------- we continue
!
     IF (BETAL.LT.-10.D0) THEN  

          IF (TAUL.LT.10.D0) THEN  
               TAUI = DBLE(NT) + 1.D0  
               TAUS = TAUI + 1.D0  
               UI = UFUNG(NT,1)  
               US = UFUNG(NT+1,1)  
               T1 = TAUL - TAUI  
               T2 = TAUS - TAUL  
               U = UI*T2 + US*T1  
!#               ups=u*1.d10*beta**2
               UPS = U*BETA  
          ELSE  
               UPS = UFUNG(9,1)*BETA  
          END IF  

     ELSE  
!
!---------- weights for tau and beta (ufung region)
!
          TAUI = DBLE(NT) + 1.D0  
          TAUS = TAUI + 1.D0  

          IF (BETAL.EQ.5.D0) THEN  
               ICB = -1  
          ELSE  
               ICB = 1  
          END IF  

          BETAI = LOG10(BETAG(NB))  
          BETAS = LOG10(BETAG(NB+ICB))  
          B1 = (BETAL-BETAI)/ (BETAS-BETAI)  
          B2 = (BETAS-BETAL)/ (BETAS-BETAI)  
!
!---------- cases again...
!
          IF (TAUL.GE.10.D0) THEN  
               UI = UFUNG(9,NB)  
               US = UFUNG(9,NB+ICB)  
               UPS = UI*B2 + US*B1  

               IF (BETAL.LT.0.D0) UPS = UPS*BETA  

          ELSE  
               T1 = TAUL - TAUI  
               T2 = TAUS - TAUL  
               UII = UFUNG(NT,NB)  
               UIS = UFUNG(NT,NB+ICB)  
               USI = UFUNG(NT+1,NB)  
               USS = UFUNG(NT+1,NB+ICB)  
               UPS = UII*T2*B2 + UIS*T2*B1 + USI*T1*B2 + USS*T1* B1

               IF (BETAL.LT.0.D0) UPS = UPS*BETA  

          END IF  
     END IF  

ELSE  

     IF (TAUL.GE.-2.D0) THEN  
!
!---------- weights for tau and beta (ufunk region)
!
          NN = INT(BETAL+10.D0) - 10  
          AUX = BETA/10.D0**DBLE(NN)  
          NE = INT((AUX-1.D0)/1.8D0) + 1  
          NB = (NN+10)*5 + NE  
          NT = INT((TAUL+2.D0)/.2D0) + 1  

          IF (TAUL.EQ.2.D0) THEN  
               ICT = 0  
          ELSE  
               ICT = 1  
          END IF  

          TAUI = (DBLE(NT)-1.D0)*.2D0 - 2.D0  
          TAUS = TAUI + .2D0  
          T1 = (TAUL-TAUI)/.2D0  
          T2 = (TAUS-TAUL)/.2D0  
!
!---------- we continue...
!
          IF (BETAL.LT.-10.D0) THEN  
               UI = UFUNK(NT,1)  
               US = UFUNK(NT+ICT,1)  
               U = UI*T2 + US*T1  
               UPS = 10.D0**U  
!#               ups=ups*1.d10*beta**2
               UPS = UPS*BETA  

          ELSE IF (BETAL.LT.5.D0) THEN  
               BETAI = LOG10(BETAK(NB))  
               BETAS = LOG10(BETAK(NB+1))  
               B1 = (BETAL-BETAI)/ (BETAS-BETAI)  
               B2 = (BETAS-BETAL)/ (BETAS-BETAI)  
               UII = UFUNK(NT,NB)  
               UIS = UFUNK(NT,NB+1)  
               USI = UFUNK(NT+ICT,NB)  
               USS = UFUNK(NT+ICT,NB+1)  
               U = UII*T2*B2 + UIS*T2*B1 + USI*T1*B2 + USS*T1*B1  
               UPS = 10.D0**U  

               IF (BETA.LT.1.D0) UPS = UPS*BETA  

          ELSE  
               UI = UFUNK(NT,76)  
               US = UFUNK(NT+ICT,76)  
               U = UI*T2 + US*T1  
               UPS = 10.D0**U  
          END IF  

     ELSE IF (BETAL.LE.0.D0) THEN  

          UPS = BETA*.398942D0*TAU**2  

     ELSE IF (BETA*TAU.LT.10.D0) THEN  

          BETAU2 = BETA*TAU/SQRT(2.D0)  
          UPS = TAU/2.D0* (1.D0-ERFCM(BETAU2))  

     ELSE  

          UPS = TAU/2.D0 - .398942D0/BETA  
     END IF  

END IF  
!
!---- we finish!
!
   10 CONTINUE  

END
!
!-----------------------------------------------------------------------
!
FUNCTION U2_INV (TAU, BETA)  
!
! calculates u2 function for inverted levels;
! tau and beta are absolute values  
!
use nlte_type
use nlte_var, ONLY: N1_B,N2_B,N1_S,N2_S,N1_L1,N2_L1,N1_L2,N2_L2, &
&                   U2B,U2S,U2L1,U2L2
implicit none

REAL(DP) :: U2_INV
!
!
!     note, that the following number is not exactly sqrt(2 pi), &
!     however the numerical analogon used here to be consistent with the
!     numerical integrations performed to obtain u2.
!     don't change it!
!
!     .. parameters ..
REAL(DP), PARAMETER :: RDPI = 0.39908365D0  
!     ..
!     .. scalar arguments ..
REAL(DP) :: BETA, TAU
!     ..
!     ..
!     .. local scalars ..
REAL(DP) :: B1, B2, BETAX, BXL, T1, T2, TAUX, TXL, U2, XB, XX, &
 XXX, Y1, Y2, Y3, Y4, YB, YY, YYY
INTEGER(I4B) :: IB, IBL, IBL2, IBL2P1, IBLP1, IBP1, IBS, IBSP1, IT, &
 ITL, ITL2, ITL2P1, ITLP1, ITP1, ITS, ITSP1
!     ..
!     .. local arrays ..
REAL(DP), DIMENSION(N2_B) :: BETAB 
REAL(DP), DIMENSION(N2_L1) :: BETAL1
REAL(DP), DIMENSION(N2_L2) :: BETAL2
REAL(DP), DIMENSION(N2_S) :: BETAS

REAL(DP), DIMENSION(N1_B) :: TAUB 
REAL(DP), DIMENSION(N1_L1) :: TAUL1
REAL(DP), DIMENSION(N1_L2) :: TAUL2
REAL(DP), DIMENSION(N1_S) :: TAUS

!     ..
!     .. external functions ..
REAL(DP) :: DINTE, ZINTER  
EXTERNAL DINTE, ZINTER  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP, INT, LOG10  
!     ..
!     .. data statements ..
DATA TAUB / .10000000000000D-05, .17782794100389D-05, &
 .31622776601684D-05, .56234132519035D-05, .10000000000000D-04, &
 .17782794100389D-04, .31622776601684D-04, .56234132519035D-04, &
 .10000000000000D-03, .17782794100389D-03, .31622776601684D-03, &
 .56234132519035D-03, .10000000000000D-02, .17782794100389D-02, &
 .31622776601684D-02, .56234132519035D-02, .10000000000000D-01, &
 .17782794100389D-01, .31622776601684D-01, .56234132519035D-01, &
 .10000000000000D+00 /
DATA BETAB / .1000D-05, .1000D-04, .1000D-03, .1000D-02, &
 .1000D-01, .1000D+00 /

DATA TAUS /  - .100D+01, - .980D+00, - .960D+00, - .940D+00, &
 - .920D+00, - .900D+00, - .880D+00, - .860D+00, - .840D+00, &
 - .820D+00, - .800D+00, - .780D+00, - .760D+00, - .740D+00, &
 - .720D+00, - .700D+00, - .680D+00, - .660D+00, - .640D+00, &
 - .620D+00, - .600D+00, - .580D+00, - .560D+00, - .540D+00, &
 - .520D+00, - .500D+00, - .480D+00, - .460D+00, - .440D+00, &
 - .420D+00, - .400D+00, - .380D+00, - .360D+00, - .340D+00, &
 - .320D+00, - .300D+00, - .280D+00, - .260D+00, - .240D+00, &
 - .220D+00, - .200D+00, - .180D+00, - .160D+00, - .140D+00, &
 - .120D+00, - .100D+00, - .800D-01, - .600D-01, - .400D-01, &
 - .200D-01,0.000D-00, 0.200D-01, 0.400D-01, 0.600D-01, 0.800D-01, &
 0.100D+00, 0.120D+00, 0.140D+00, 0.160D+00, 0.180D+00, 0.200D+00, &
 0.220D+00, 0.240D+00, 0.260D+00, 0.280D+00, 0.300D+00, 0.320D+00, &
 0.340D+00, 0.360D+00, 0.380D+00, 0.400D+00, 0.420D+00, 0.440D+00, &
 0.460D+00, 0.480D+00, 0.500D+00, 0.520D+00, 0.540D+00, 0.560D+00, &
 0.580D+00, 0.600D+00, 0.620D+00, 0.640D+00, 0.660D+00, 0.680D+00, &
 0.700D+00, 0.720D+00, 0.740D+00, 0.760D+00, 0.780D+00, 0.800D+00, &
 0.820D+00, 0.840D+00, 0.860D+00, 0.880D+00, 0.900D+00, 0.920D+00, &
 0.940D+00, 0.960D+00, 0.980D+00, 0.100D+01, 0.102D+01, 0.104D+01, &
 0.106D+01, 0.108D+01, 0.110D+01, 0.112D+01, 0.114D+01, 0.116D+01, &
 0.118D+01, 0.120D+01, 0.122D+01, 0.124D+01, 0.126D+01, 0.128D+01, &
 0.130D+01, 0.132D+01, 0.134D+01, 0.136D+01, 0.138D+01, 0.140D+01, &
 0.142D+01, 0.144D+01, 0.146D+01, 0.148D+01, 0.150D+01, 0.152D+01, &
 0.154D+01, 0.156D+01, 0.158D+01, 0.160D+01, 0.162D+01, 0.164D+01, &
 0.166D+01, 0.168D+01, 0.170D+01, 0.172D+01, 0.174D+01, 0.176D+01, &
 0.178D+01, 0.180D+01, 0.182D+01, 0.184D+01, 0.186D+01, 0.188D+01, &
 0.190D+01, 0.192D+01, 0.194D+01, 0.196D+01, 0.198D+01, 0.200D+01, &
 0.202D+01, 0.204D+01, 0.206D+01, 0.208D+01, 0.210D+01, 0.212D+01, &
 0.214D+01, 0.216D+01, 0.218D+01, 0.220D+01, 0.222D+01, 0.224D+01, &
 0.226D+01, 0.228D+01, 0.230D+01, 0.232D+01, 0.234D+01, 0.236D+01, &
 0.238D+01, 0.240D+01, 0.242D+01, 0.244D+01, 0.246D+01, 0.248D+01, &
 0.250D+01, 0.252D+01, 0.254D+01, 0.256D+01, 0.258D+01, 0.260D+01, &
 0.262D+01, 0.264D+01, 0.266D+01, 0.268D+01, 0.270D+01, 0.272D+01, &
 0.274D+01, 0.276D+01, 0.278D+01, 0.280D+01, 0.282D+01 /
DATA BETAS / - .300D+01, - .298D+01, - .296D+01, - .294D+01, &
&- .292D+01, - .290D+01, - .288D+01, - .286D+01, - .284D+01, &
&- .282D+01, - .280D+01, - .278D+01, - .276D+01, - .274D+01, &
&- .272D+01, - .270D+01, - .268D+01, - .266D+01, - .264D+01, &
&- .262D+01, - .260D+01, - .258D+01, - .256D+01, - .254D+01, &
&- .252D+01, - .250D+01, - .248D+01, - .246D+01, - .244D+01, &
&- .242D+01, - .240D+01, - .238D+01, - .236D+01, - .234D+01, &
&- .232D+01, - .230D+01, - .228D+01, - .226D+01, - .224D+01, &
&- .222D+01, - .220D+01, - .218D+01, - .216D+01, - .214D+01, &
&- .212D+01, - .210D+01, - .208D+01, - .206D+01, - .204D+01, &
&- .202D+01, - .200D+01, - .198D+01, - .196D+01, - .194D+01, &
&- .192D+01, - .190D+01, - .188D+01, - .186D+01, - .184D+01, &
&- .182D+01, - .180D+01, - .178D+01, - .176D+01, - .174D+01, &
&- .172D+01, - .170D+01, - .168D+01, - .166D+01, - .164D+01, &
&- .162D+01, - .160D+01, - .158D+01, - .156D+01, - .154D+01, &
&- .152D+01, - .150D+01, - .148D+01, - .146D+01, - .144D+01, &
&- .142D+01, - .140D+01, - .138D+01, - .136D+01, - .134D+01, &
&- .132D+01, - .130D+01, - .128D+01, - .126D+01, - .124D+01, &
&- .122D+01, - .120D+01, - .118D+01, - .116D+01, - .114D+01, &
&- .112D+01, - .110D+01, - .108D+01, - .106D+01, - .104D+01, &
&- .102D+01, - .100D+01, - .980D+00, - .960D+00, - .940D+00, &
&- .920D+00, - .900D+00, - .880D+00, - .860D+00, - .840D+00, &
&- .820D+00, - .800D+00, - .780D+00, - .760D+00, - .740D+00, &
&- .720D+00, - .700D+00, - .680D+00, - .660D+00, - .640D+00, &
&- .620D+00, - .600D+00, - .580D+00, - .560D+00, - .540D+00, &
&- .520D+00, - .500D+00, - .480D+00, - .460D+00, - .440D+00, &
&- .420D+00, - .400D+00, - .380D+00, - .360D+00, - .340D+00, &
&- .320D+00, - .300D+00, - .280D+00, - .260D+00, - .240D+00, &
&- .220D+00, - .200D+00, - .180D+00, - .160D+00, - .140D+00, &
&- .120D+00, - .100D+00, - .800D-01, - .600D-01, - .400D-01, &
&- .200D-01,0.000D-00, 0.200D-01, 0.400D-01, 0.600D-01, 0.800D-01, &
&0.100D+00, 0.120D+00, 0.140D+00, 0.160D+00, 0.180D+00, 0.200D+00, &
&0.220D+00, 0.240D+00, 0.260D+00, 0.280D+00, 0.300D+00, 0.320D+00, &
&0.340D+00, 0.360D+00, 0.380D+00, 0.400D+00, 0.420D+00, 0.440D+00, &
&0.460D+00, 0.480D+00, 0.500D+00, 0.520D+00, 0.540D+00, 0.560D+00, &
&0.580D+00, 0.600D+00, 0.620D+00, 0.640D+00, 0.660D+00, 0.680D+00, &
&0.700D+00, 0.720D+00, 0.740D+00, 0.760D+00, 0.780D+00, 0.800D+00, &
&0.820D+00, 0.840D+00, 0.860D+00, 0.880D+00, 0.900D+00, 0.920D+00, &
&0.940D+00, 0.960D+00, 0.980D+00 /

DATA TAUL1 / - .600D+01, - .590D+01, - .580D+01, - .570D+01, &
 - .560D+01, - .550D+01, - .540D+01, - .530D+01, - .520D+01, &
 - .510D+01, - .500D+01, - .490D+01, - .480D+01, - .470D+01, &
 - .460D+01, - .450D+01, - .440D+01, - .430D+01, - .420D+01, &
 - .410D+01, - .400D+01, - .390D+01, - .380D+01, - .370D+01, &
 - .360D+01, - .350D+01, - .340D+01, - .330D+01, - .320D+01, &
 - .310D+01, - .300D+01, - .290D+01, - .280D+01, - .270D+01, &
 - .260D+01, - .250D+01, - .240D+01, - .230D+01, - .220D+01, &
 - .210D+01, - .200D+01, - .190D+01, - .180D+01, - .170D+01, &
 - .160D+01, - .150D+01, - .140D+01, - .130D+01, - .120D+01, &
 - .110D+01, - .100D+01, - .900D+00, - .800D+00, - .700D+00, &
 - .600D+00, - .500D+00, - .400D+00, - .300D+00, - .200D+00, &
 - .100D+00, 0.000D+00 /
DATA BETAL1 / -.100D+01, - .900D+00, - .800D+00, - .700D+00, &
 - .600D+00, - .500D+00, - .400D+00, - .300D+00, - .200D+00, &
 - .100D+00,0.000D+00, 0.100D+00, 0.200D+00, 0.300D+00, 0.400D+00, &
 0.500D+00, 0.600D+00, 0.700D+00, 0.800D+00, 0.900D+00, 0.100D+01, &
 0.110D+01, 0.120D+01, 0.130D+01, 0.140D+01, 0.150D+01, 0.160D+01, &
 0.170D+01, 0.180D+01, 0.190D+01, 0.200D+01, 0.210D+01, 0.220D+01, &
 0.230D+01, 0.240D+01, 0.250D+01, 0.260D+01, 0.270D+01, 0.280D+01, &
 0.290D+01, 0.300D+01, 0.310D+01, 0.320D+01, 0.330D+01, 0.340D+01, &
 0.350D+01, 0.360D+01, 0.370D+01, 0.380D+01, 0.390D+01, 0.400D+01, &
 0.410D+01, 0.420D+01, 0.430D+01, 0.440D+01, 0.450D+01, 0.460D+01, &
 0.470D+01, 0.480D+01, 0.490D+01, 0.500D+01, 0.510D+01, 0.520D+01, &
 0.530D+01, 0.540D+01, 0.550D+01, 0.560D+01, 0.570D+01, 0.580D+01, &
 0.590D+01, 0.600D+01 /

DATA TAUL2 / 0.0000D+00, 0.2500D+00, 0.5000D+00, 0.7500D+00, &
 1.0000D+00, 1.2500D+00, 1.5000D+00, 1.7500D+00, 2.0000D+00, &
 2.2500D+00, 2.5000D+00, 2.7500D+00 /
DATA BETAL2 /0.7500D+00, 1.0000D+00, 1.2500D+00, 1.5000D+00, &
 1.7500D+00, 2.0000D+00, 2.2500D+00, 2.5000D+00, 2.7500D+00, &
 3.0000D+00, 3.2500D+00, 3.5000D+00, 3.7500D+00, 4.0000D+00, &
 4.2500D+00, 4.5000D+00, 4.7500D+00, 5.0000D+00 /
!     ..
!r
!r    reset values of betax/taux if they can stop the program
!r
TAUX = TAU  

BETAX = BETA  
IF (BETAX.LT..1d-5) THEN  
   BETAX = .1d-05  
ENDIF  
!
! COMMENTED OUT, SINCE CAUGHT IN CALLING ROUTINE
!IF (TAUX.GT.0.65D3) THEN  
!   TAUX = 0.65D3
!ENDIF  
!r
!
TXL = LOG10 (TAUX)  
BXL = LOG10 (BETAX)  
!r
!r    set the values of u2 out of the table
!r    -------------------------------------
!
IF (TXL.LT. - .6D1) THEN  
   U2_INV = 0.5D0 * TAUX  

   RETURN  
ENDIF  
!
IF (BXL.GT..5D1.AND.TXL.GT.0.) THEN  
   U2_INV = RDPI / BETAX  

   RETURN  
ENDIF  
!r
IF (BXL.GT..6D1) THEN  
   U2_INV = RDPI / BETAX  

   RETURN  
ENDIF  
!
!r    "extrapolation" out of little2 (tau=last value in table)
!
IF (TXL.GE..275D1.AND.BXL.GE..1D1) THEN  
   ITL2 = N1_L2  
   YYY = (BXL - 1.D0) * 4.D0 + 1.D0  
   IBL2 = INT (YYY)  
   IBL2P1 = IBL2 + 1  
   Y1 = U2L2 (ITL2, IBL2)  
   Y2 = U2L2 (ITL2, IBL2P1)  
   U2 = DINTE (Y1, Y2, BXL, BETAL2 (IBL2), BETAL2 (IBL2P1) )  
   U2_INV = 10.D0**U2  

   RETURN  
ENDIF  
!r
IF (BXL.LE. - .6D1) THEN  
   U2_INV = (EXP (TAUX) - 1.D0) / TAUX - 1.D0  

   RETURN  
ENDIF  
!r
!r
IF (TXL.GT..278D1.AND.BXL.LT. - .1D1) THEN  
   U2_INV = (EXP (TAUX) - 1.D0) / TAUX - 1.D0  

   RETURN  
ENDIF  
!r
!r    "extrapolation" out of small (same as above)
!r
IF (TXL.GT..278D1.AND.BXL.GT. - .1D1.AND.BXL.LT..1D1) THEN  
   YY = (BXL + 3.D0) * 50.D0 + 1.D0  
   IBS = INT (YY)  
   ITS = N1_S  
   IBSP1 = IBS + 1  
   Y1 = U2S (ITS, IBS)  
   Y2 = U2S (ITS, IBSP1)  
   U2 = DINTE (Y1, Y2, BXL, BETAS (IBS), BETAS (IBSP1) )  
   U2_INV = 10.D0**U2  

   RETURN  
ENDIF  
!r
!r    interpolation regions
!r    --------------------------
!
!     small table - interpolation of log(u2) in log(t) & log(b)
!     range: -1>=log(tau)>=2.82 & -3>=log(beta)>=0.98
!r
IF (TXL.GE. - .1D1.AND.BXL.GE. - .3D1.AND.BXL.LE..98D0) THEN  
!r
   XX = (TXL + 1.D0) * 50.D0 + 1.D0  
   ITS = INT (XX)  
   ITSP1 = ITS + 1  
!
   YY = (BXL + 3.D0) * 50.D0 + 1.D0  
   IBS = INT (YY)  
   IBSP1 = IBS + 1  
!r
   Y1 = U2S (ITS, IBSP1)  
   Y2 = U2S (ITSP1, IBSP1)  
   Y3 = U2S (ITSP1, IBS)  
   Y4 = U2S (ITS, IBS)  
   T1 = TAUS (ITS)  
   T2 = TAUS (ITSP1)  
   B1 = BETAS (IBS)  
   B2 = BETAS (IBSP1)  
   U2 = ZINTER (Y1, Y2, Y3, Y4, T1, T2, B1, B2, TXL, BXL)  
   U2_INV = 10.D0**U2  
   RETURN  
!
ENDIF  
!r
!     little1 table - interpolation of log(u2) in log(t) & log(b)
!     range: -6>=log(tau)>=0 & -1>=log(beta)>=6
!
IF (BXL.GE. - 1.D0.AND.TXL.LE.0.D0) THEN  
!r
   XX = (TXL + 6.D0) * 10.D0 + 1.D0  
   ITL = INT (XX)  
   ITLP1 = ITL + 1  
!
   YY = (BXL + 1.D0) * 10.D0 + 1.D0  
   IBL = INT (YY)  
   IBLP1 = IBL + 1  
!r
   Y1 = U2L1 (ITL, IBLP1)  
   Y2 = U2L1 (ITLP1, IBLP1)  
   Y3 = U2L1 (ITLP1, IBL)  
   Y4 = U2L1 (ITL, IBL)  
   T1 = TAUL1 (ITL)  
   T2 = TAUL1 (ITLP1)  
   B1 = BETAL1 (IBL)  
   B2 = BETAL1 (IBLP1)  
   U2 = ZINTER (Y1, Y2, Y3, Y4, T1, T2, B1, B2, TXL, BXL)  
   U2_INV = 10.D0**U2  
   RETURN  
!
ENDIF  
!r
!     little2 table - interpolation of log(u2) in log(t) & log(b)
!     range: 0>=log(tau)>=2.75 & 0.75>=log(beta)>=5
!
IF (BXL.GT.0.75D0.AND.TXL.GT.0.D0) THEN  
!r
   XXX = TXL * 4.D0 + 1.D0  
   ITL2 = INT (XXX)  
   ITL2P1 = ITL2 + 1  
!
   YYY = (BXL - 0.75D0) * 4.D0 + 1.D0  
   IBL2 = INT (YYY)  
   IBL2P1 = IBL2 + 1  
!r
   Y1 = U2L2 (ITL2, IBL2P1)  
   Y2 = U2L2 (ITL2P1, IBL2P1)  
   Y3 = U2L2 (ITL2P1, IBL2)  
   Y4 = U2L2 (ITL2, IBL2)  
   T1 = TAUL2 (ITL2)  
   T2 = TAUL2 (ITL2P1)  
   B1 = BETAL2 (IBL2)  
   B2 = BETAL2 (IBL2P1)  
   U2 = ZINTER (Y1, Y2, Y3, Y4, T1, T2, B1, B2, TXL, BXL)  
   U2_INV = 10.D0**U2  
   RETURN  
!
ENDIF  
!r
!     big table - interpolation of u2 in tau & beta
!     range: 1d-6>=tau>=0.1 & 1d-6>=beta>=0.1
!
IF (TXL.LE. - 1.D0.AND.BXL.LT. - 1.D0) THEN  
   XB = (TXL + 6.D0) * 4.D0 + 1.D0  
   IT = INT (XB)  
!
   ITP1 = IT + 1  
!r
   YB = BXL + 7.D0  
   IB = INT (YB)  
!
   IBP1 = IB + 1  
!
   Y1 = U2B (IT, IBP1) / (.5D0 * TAUB (IT) )  
   Y2 = U2B (ITP1, IBP1) / (.5D0 * TAUB (ITP1) )  
   Y3 = U2B (ITP1, IB) / (.5D0 * TAUB (ITP1) )  
   Y4 = U2B (IT, IB) / (.5D0 * TAUB (IT) )  
   T1 = TAUB (IT)  
   T2 = TAUB (ITP1)  
   B1 = BETAB (IB)  
   B2 = BETAB (IBP1)  
   U2 = ZINTER (Y1, Y2, Y3, Y4, T1, T2, B1, B2, TAUX, BETAX)  
   U2_INV = .5D0 * TAUX * U2  

   RETURN  
ENDIF  
!
!     region of possible big numbers
!     good aprox. u2=(exp(tau)-1)/tau-1
!
IF (TXL.GT. - 1.D0.AND.BXL.LT. - 3.D0) THEN  
   PRINT * , 'Possible big number, no interpolation'  
   U2_INV = (EXP (TAUX) - 1.D0) / TAUX - 1.D0  

   RETURN  
ENDIF  
!
END FUNCTION U2_INV
!r
!r    ***************************************************************
!r    function zinter performs the interpolation
!r    ***************************************************************
FUNCTION ZINTER (Y1, Y2, Y3, Y4, T1, T2, B2, B1, TAUX, BETAX)
use nlte_type
implicit none

REAL(DP) :: ZINTER
!     .. scalar arguments ..
REAL(DP) :: B1, B2, BETAX, T1, T2, TAUX, Y1, Y2, Y3, Y4  
!     ..
!     .. local scalars ..
REAL(DP) :: T, U  
!     ..
T = (TAUX - T1) / (T2 - T1)  
U = (BETAX - B1) / (B2 - B1)  
ZINTER = (1.D0 - T) * (1.D0 - U) * Y1 + T * (1.D0 - U) * Y2 + T * &
 U * Y3 + (1.D0 - T) * U * Y4

RETURN  
END FUNCTION ZINTER
!r
!r    ***************************************************************
!r    function dinte interpolates only beta values
!r    *************************************************************** &
FUNCTION DINTE (Y1, Y2, BETAX, B, BP1)  
use nlte_type
implicit none

REAL(DP) :: DINTE
!     .. scalar arguments ..
REAL(DP) :: B, BETAX, BP1, Y1, Y2  
!     ..
!     .. local scalars ..
REAL(DP) :: U  
!     ..
U = BETAX - B  
U = U / (BP1 - B)  
DINTE = U * (Y2 - Y1) + Y1  

RETURN  
END FUNCTION DINTE
!
!-----------------------------------------------------------------------
!
SUBROUTINE UPDATE(XNH,XNE,TE,ILOW,IMAX,LL,NAT)  

USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS  

USE nlte_var, ONLY: OCCNUM,BLEVEL,ENIONND,OPTNEUPDATE  

IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  LL,NAT  
!     ..
!     .. array arguments ..
REAL(DP) ::  TE(ND1),XNE(ND1),XNH(ND1)  
INTEGER(I4B) ::  ILOW(ND1,KEL),IMAX(ND1,KEL)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AUX,ERR,REL,XNEH,XNENEW,XNEOLD,ZI  
INTEGER(I4B) ::  I,IZ,K,NNN,NNUU
INTEGER(I4B), DIMENSION(1) ::  MAIN
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,INT  
!     ..

XNEOLD = XNE(LL)  
XNENEW = 0.D0  

DO K = 1,NAT  

     DO I = ILOW(LL,K),IMAX(LL,K) + 1  

          ZI = ZEFF(K) + I - 1  
          XNENEW = XNENEW + ENIONND(K,I,LL)*ZI  
     END DO
! check for consistency
     DO I = 1, ILOW(LL,K) - 1
       IF (ENIONND(K,I,LL) .NE. 0.) THEN
         STOP ' ENIONND NE 0 FOR I < ILOW '
       ENDIF
     ENDDO  

     DO I = IMAX(LL,K) +2, ID_KISAT+1
       IF (ENIONND(K,I,LL) .NE. 0.) THEN
         STOP ' ENIONND NE 0 FOR I > IMAX +1 '
       ENDIF
     ENDDO  
END DO  

IF(.NOT.OPTNEUPDATE) XNENEW=XNEOLD !LEAVE NE AT ITS LTE VALUE
XNE(LL) = XNENEW
!print*,'xnecheck: in update',optneupdate
!if(ll.eq.nd1) print*,xne


REL = XNENEW/XNEOLD  
ERR = ABS(1.D0-REL)  

XNEH = XNENEW/XNH(LL)  
NNN = LL/6  
NNUU = NNN*6  
IF (LL.EQ.1 .OR. LL.EQ.47) NNUU = LL  

IF (NNUU.EQ.LL) THEN  
     PRINT *  
     PRINT *,'----------------------------------------------------------------'
     PRINT *,'   L=',LL,'   TE = ',TE(LL),' NE/NH = ',XNEH  
     PRINT *  
     PRINT *,' ATOM // Z(MAIN ION) // N(J,K)/N(H)'  

     DO K = 1,NAT  

          IZ = INT(ZEFF(K))  
          IZ = IZ - 1  
          AUX = 0.D0
          MAIN=MAXLOC(ENIONND(K,1:IMAX(LL,K),LL))
          WRITE (*,FMT=9000) LABAT(K), IZ+MAIN(1), (AUX,I= &
           1,IZ+ILOW(LL,K)), (ENIONND(K,I,LL)/XNH(LL), I=ILOW(LL,K) &
           ,IMAX(LL,K)+1)
     END DO

     PRINT *  
     PRINT *,' CHANGE IN ELECTRON DENSITY = ',ERR  
     PRINT *  

END IF  

RETURN  

 9000 FORMAT (/,A4,'/',I3,2X,'/',9 (E10.4,2X))  

END
!
!***********************************************************************
!
! subroutines: complex ones
! rateli and related (esp. cmf)
!
!***********************************************************************
!
SUBROUTINE RATELI(XNE,TEMP,CLF,ILOW,IMAX,ND,CONCON,R,V,GRADV,SR,VMAX, &
&                  TEFF,CORRFC)

! UPDATED FOR CLUMPING
! ILOW, IMAX, IMIANL AND IMAANL: NLTE VALUES ONLY!!!!

USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&    ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS

USE nlte_var, ONLY: BLEVEL,ENIONND, &
& INDXLAMC,XLAMCMF,&
& OPTCMF,OPTMIXED, &
& IMIANL,IMAANL,INDEXCMF,INDEXSA,&
& MODNAM,LTE_UPDATE  

USE nlte_porvor, ONLY: FIC,TCL_FAC_LINE,TCL_FAC_CONT

IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  

!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORRFC,SR,TEFF,VMAX  
INTEGER(I4B) ::  ND  
LOGICAL CONCON  
!     ..
!     .. array arguments ..
REAL(DP) ::  GRADV(ND1),R(ND1),TEMP(ND1),V(ND1),XNE(ND1),CLF(ND1)  

INTEGER(I4B) ::  ILOW(ND1,KEL),IMAX(ND1,KEL)  
!     ..
!     .. local scalars ..
REAL(DP) ::  VELR,XLAM,XMUST,TCL

INTEGER(I4B) ::  I,ICMF,ICOUNT,II,ILO,IMA, &
&        J,L,LL,LX,NATO,NFI,NFIR,NFIR1,NFIRM,NLA,NLAS, &
&        NLAS1,NLASM,NREC,NUMION
INTEGER(I4B), SAVE :: IPRES

LOGICAL ACCEL,STARTCMF
!     ..
!     .. local arrays ..
REAL(DP) ::  ABUND(ID_ATOMS), &
&                 BLEV0(ID_LLEVS),FL(ID_LLEVS),FLX(ID_XLEVS), &
&                 GL(ID_LLEVS),GLX(ID_XLEVS),GS0(ID_SLEVS), &
&                 GS1(ID_SLEVS),QD(ID_SLEVS),RYD(ID_SLEVS), &
&                 WEIGHT(ID_ATOMS),ZEFF(ID_ATOMS),ZL(ID_LLEVS)
!     ..
!     .. external functions ..
INTEGER(I4B) ::  IGENIO  
!
!     .. data statements ..
!
DATA STARTCMF/.TRUE./  
!     ..

! AFTER LTE-UPDATE, CREATE NEW CMF-LIST (CHANGE IN IMIN,IMAX)
IF(.NOT.STARTCMF.AND.LTE_UPDATE) STARTCMF=.TRUE.

NREC = ID_LLEVS  

IF (.NOT.CONCON) STOP 'ERROR IN RATELI'  
!
!-----sobolev case
!
IF (.NOT.OPTCMF) THEN  

     DO LL = 1,ND  

          READ (17,REC=LL) (BLEVEL(I),I=1,NREC)  
          CALL FACTINC(R(LL),LL)  
          VELR = V(LL)/R(LL)  
          XMUST = SQRT(1.D0-1.D0/R(LL)/R(LL))  

          DO NATO = 1,NAT  
               ILO = ILOW(LL,NATO)  
               IMA = IMAX(LL,NATO)  
!              print*,nato,'  ',ll,'  ',ilo,'  ',ima
               IF (IMA.LT.ILO) CYCLE  
               NUMION = IGENIO(NATO,ILO)  
               NFIR = IFIRSL(NUMION)  
               NUMION = IGENIO(NATO,IMA) + 1  
               IF (NUMION.GT.IONG) STOP 'ERROR IN RATELI - NUMION'  
               NLAS = IFIRSL(NUMION)  
               CALL SOBPREP(LL,VELR,GRADV(LL),NFIR,NLAS,BLEVEL, &
                XNE(LL),SR,VMAX,NATO,XMUST,TEMP(LL),CLF(LL), &
                FIC(LL),TCL_FAC_LINE(LL),TCL_FAC_CONT(LL))
          END DO

     END DO

     RETURN  
!
!-----cmf-treatment for some lines
!
ELSE  
!
!     reset of indexsa
!
     INDEXSA(1:ID_NTTRD) = 0  

DEPTHLOOP: DO LL = 1,ND  

          READ (17,REC=LL) (BLEVEL(I),I=1,NREC)  
!
!---  if all lines in cmf, the next statement can be skipped
!
          CALL FACTINC(R(LL),LL)  
          VELR = V(LL)/R(LL)  
          XMUST = SQRT(1.D0-1.D0/R(LL)/R(LL))  

NATOLOOP: DO NATO = 1,NAT  
               ILO = ILOW(LL,NATO)  
               IMA = IMAX(LL,NATO)  
!               print*,nato,'  ',ll,'  ',ilo,'  ',ima

               IF (IMA.LT.ILO) THEN
                 IF(NATO.EQ.1.AND.LL.EQ.1) THEN
                    PRINT*,NATO,'  ',LL,'  ',ILO,'  ',IMA
                    PRINT*,' NO LINES FOR NATO =1 AND LL=1'
                    PRINT*,' IN THIS CASE, CMFPREP IS NOT CALLED AND INDEXCMF'
                    PRINT*,' MIGHT BE NOT CORRECTLY UPDATED'
                    STOP ' NO LINES FOR NATO =1 AND LL=1'           
                 ENDIF
                 CYCLE NATOLOOP  
               ENDIF

               NUMION = IGENIO(NATO,ILO)  
               NFIR = IFIRSL(NUMION)  
               NUMION = IGENIO(NATO,IMA) + 1  
               IF (NUMION.GT.IONG) STOP 'ERROR IN RATELI - NUMION'  
               NLAS = IFIRSL(NUMION)  
!
!---     calculation of min/max levels for cmf-transfer: Here we
!        prepare the decision whether SA even in general CMF situation, &
!        in cases where we have not everywhere occupation numbers 
!
               IF (STARTCMF) THEN  
                    NFIRM = 0  
                    NLASM = 10000  
                    DO LX = 1,ND  
                         ILO = ILOW(LX,NATO)  
                         IMA = IMAX(LX,NATO)  
                         NUMION = IGENIO(NATO,ILO)  
                         NFIR1 = IFIRSL(NUMION)  
                         NUMION = IGENIO(NATO,IMA) + 1  
                         NLAS1 = IFIRSL(NUMION)  
                         NFIRM = MAX(NFIRM,NFIR1)  
                         NLASM = MIN(NLASM,NLAS1)
                    END DO    

                    NUMION = IGENIO(NATO,IMIANL(NATO))  
                    NFI = IFIRSL(NUMION)  
                    NUMION = IGENIO(NATO,IMAANL(NATO)) + 1
                    
                    IF (NUMION.GT.IONG) STOP 'ERROR IN RATELI - NUMION'

                    NLA = IFIRSL(NUMION)  

               END IF  
!
!        note: nfirm, nlasm only needed for startcmf = .true.,
!              i.e., at first depth point of first cmf iteration
!              the same is true for nfi,nla
!
!               print*,nato,'  ',ll,'  ',ilo,'  ',ima
!               print*,nato,'  ',ll,'  ',nfi,'  ',nla
!               print*,nato,'  ',ll,'  ',nfir,'  ',nlas
!               print*,nato,'  ',ll,'  ',nfirm,'  ',nlasm
               CALL CMFPREP(LL,VELR,GRADV(LL),NFIR,NLAS,BLEVEL, &
                XNE(LL),SR,VMAX,NATO,XMUST,TEMP(LL),CLF(LL),NFIRM,NLASM, &
                NFI,NLA,STARTCMF,OPTMIXED, IPRES,R,V, &
                FIC(LL),TCL_FAC_LINE(LL),TCL_FAC_CONT(LL))

          END DO NATOLOOP
!
!***  index array so that cmf-line wavelengths are in order
!
          ICOUNT = ID_NTTRD  

          CALL INDEXX(ICOUNT,XLAMCMF,INDXLAMC)
          
          IF (STARTCMF) THEN  
               PRINT *  
JLOOP:         DO J = 1,ID_NTTRD  
                    II = INDXLAMC(J)  
                    XLAM = XLAMCMF(II)  
                    ICMF = INDEXCMF(II)  
                    IF (XLAM.EQ.0.D0 .AND. ICMF.NE. 1) &
&                    STOP 'ERROR IN XLAM=0(1)!'
                    IF (ICMF.EQ.1) THEN  

                         IF (XLAM.NE.0.D0) PRINT *,' LINE AT ', &
                          XLAM, ' TREATED IN SOBOLEV APPROX.'

                    ELSE IF (ICMF.EQ.2) THEN  
                    PRINT *,' LINE AT ',XLAM,' TREATED IN CMF '  

                    ELSE  
                         STOP ' ERROR IN INDEXCMF(RATELI)'  
                    END IF  
               END DO JLOOP

               PRINT *  
               IF (IPRES.EQ.1) THEN  
!
!***     file indexcmf exists, no further action required
!
                    CONTINUE  

               ELSE IF (IPRES.EQ.2) THEN  

                    IF (OPTMIXED.EQ.1.) THEN  
                      stop ' optmixed not implemented yet'
                      OPEN (1,ERR=120,FILE=TRIM(MODNAM)//'/INDEXCMF', &
&                       STATUS='NEW')
                         WRITE (1,FMT=*) (INDEXCMF(II),II=1,ID_NTTRD)
                         CLOSE (1)  
                    END IF  

               ELSE  
                    STOP ' ERROR IN IPRES (RATELI)'  
               END IF  
          END IF  
!
!***  startcmf only for first depth point
!
          STARTCMF = .FALSE.  

     END DO DEPTHLOOP

! perform cmf-calc.

! so far, no necessity to update subroutine CMF, since all quantities already corrected 
     CALL CMF(R,V,GRADV,TEMP,VMAX,TEFF,CORRFC,ND)

     PRINT *  

     RETURN  

END IF  

  120 CONTINUE  

STOP 'ERROR IN OPENING FILE INDEXCMF'  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CMF(R,V,GRADV,TEMP,VMAX,TEFF,CORRFC,ND)

! performs cmf calculations, both for the single line as well as the
! overlap case

! note that all overlap calc. are done with respect to the "def"
! delta nu = nu_o * v/c and
! x = (nu/nu_o - 1) c/v  
  
USE nlte_type
USE fund_const, ONLY: CLIGHT
USE nlte_dim

USE nlte_var, ONLY: INDXLAMC,INDEXCMF,INDEXSA,XLAMCMF,VDOPCMF, &
& XMAXDOP,TLUMAT,TULMAT,BLUCMF,BULCMF,AULCMF,LABLINELO,LABLINEUP

USE nlte_opt, ONLY: OPTSING
USE cmf_multi, ONLY: INDOV, INDOVERLAP_ORDER, INFOTAB, &
&                    USAVE, VSAVE, IPLUS, IMINUS


IMPLICIT NONE

INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT, NP=ID_NPOIN  
INTEGER(I4B), PARAMETER :: NF=ID_NFCMF  


INTEGER(I4B) :: ND
REAL(DP) :: VMAX, TEFF, CORRFC
REAL(DP), DIMENSION(ND1) :: R,V,GRADV,TEMP

REAL(DP), DIMENSION(ND1) :: AJLMEAN, ALO


INTEGER(I4B), DIMENSION(:), ALLOCATABLE :: INDEX,INDEXOV
REAL(DP),     DIMENSION(:), ALLOCATABLE :: LAMBLUE
REAL(DP),     DIMENSION(:,:), ALLOCATABLE :: TLUAUX,TULAUX

INTEGER(I4B) :: II,II1,J,JJ,JJ1,JK,L,JSTART,JEND,NOV,NOV1,IOV 

REAL(DP) :: XLAM, VDOP, SNEW, XOV, XNF, XMAX, XLAM1, DL, DL1

LOGICAL :: FLAG,INVERSION
! prepare overlap calc.

IF(OPTSING) THEN
! forces single line treatment
  INDOV=0
  CALL CALCXMAX(VMAX,TEFF,TEMP,ND)
ELSE  
CALL PREPOV(R,V,GRADV,TEMP,VMAX,TEFF,ND)
ENDIF

! do the cmf transport for all lines, either in single or overlap approach

J=1
OPEN(333,FILE='OX')
CMFLOOP: DO  WHILE (J.LE.ID_NTTRD)  

    II = INDXLAMC(J)  
    XLAM = XLAMCMF(II)
    
    IF (INDEXCMF(II).EQ.2) THEN  

        VDOP = VDOPCMF(II)  
        IF (VDOP.EQ.0.D0) STOP ' ERROR IN VTHCMF'  

     
        IF(INDOV(J).EQ.0) THEN 
! no overlap

          CALL CMFSING &
&          (II,R,V,GRADV,TEMP,XLAM,VDOP,VMAX,TEFF,CORRFC,AJLMEAN,ALO,INVERSION)
           WRITE (*,FMT=9000) XLAM,LABLINELO(II),LABLINEUP(II)  

           DO L = 1,ND  
             SNEW = AJLMEAN(L) - ALO(L)*TULMAT(II,L)  
             IF (SNEW.LT.0.D0 .OR. ABS(ALO(L)).GT.1.D0) THEN
               IF(.NOT.INVERSION) PRINT*,' WARNING!!!! SOMETHING WRONG WITH LINE-ALO'
               IF(INVERSION) THEN
                 PRINT*,' WARNING!!!! Jbar corrected at',L
                 AJLMEAN(L)=0.                  
               ENDIF  
               ALO(L) = 0.D0  
               SNEW = AJLMEAN(L)  
             END IF  

             TLUMAT(II,L) = BLUCMF(II)*SNEW  
             TULMAT(II,L) = BULCMF(II)*SNEW + AULCMF(II)* (1.D0-ALO(L))
           END DO 

            
        ELSE 

! line overlap (intrinsic + wind induced)


        IF(INDOV(J).NE.1) STOP ' ERROR IN INDOV'
           JSTART=J
           J=J+1
           NOV=1
           DO WHILE (INDOV(J).NE.3)
             J=J+1
             IF (INDEXCMF(INDXLAMC(J)).EQ.2) NOV=NOV+1
           ENDDO
           JEND=J
           
           NOV=NOV+1
           NOV1=JEND-JSTART+1

           IF(NOV.NE.NOV1) THEN
           print*,jstart,jend,nov,nov1
           STOP ' SA LINE IN OVERLAP COMPLEX, SHOULD NEVER HAPPEN'
           ENDIF 
       
! prepare info concering blue-wing boundary condition
! ------------------------------------------------------------------------
           
! locals
           IF(ALLOCATED(LAMBLUE)) DEALLOCATE(LAMBLUE,INDEX,INDEXOV, &
&            TLUAUX,TULAUX)
! common
           IF(ALLOCATED(INDOVERLAP_ORDER)) &
&            DEALLOCATE(INDOVERLAP_ORDER,INFOTAB, &
&            USAVE,VSAVE,IPLUS,IMINUS)
           
           ALLOCATE(LAMBLUE(NOV),INDEX(NOV),INDOVERLAP_ORDER(NOV), &
&            INFOTAB(NOV),USAVE(ND1,NP-1,NOV),VSAVE(ND1-1,NP-1,NOV), &
&            IPLUS(NP-ND1+1,NOV),IMINUS(NP-1,NOV), &
&            INDEXOV(NOV),TLUAUX(ND,NOV),TULAUX(ND,NOV))
           
           INFOTAB%INDEX=0
           INFOTAB%INDBW=0
           INFOTAB%XOV=0.
           INFOTAB%IF=0.
           
           DO JJ=JSTART,JEND
              II = INDXLAMC(JJ)  
              VDOP = VDOPCMF(II)  
              IF (VDOP.EQ.0.D0) STOP ' ERROR IN VTHCMF'              
! blue-wing freq., from def. of dnu   
              LAMBLUE(JJ-JSTART+1) = &
&               XLAMCMF(II)/(1.D0 + XMAXDOP(II)*VDOP/CLIGHT)
           ENDDO

! sort lines for decreasing blue-wing frequencies
           CALL INDEXX(NOV,LAMBLUE,INDEX)

           DO JJ=1,NOV
              JJ1=INDEX(JJ)+JSTART-1
! JJ1 is now the index corresponding to blue-wing order in INDXLAMC
              II = INDXLAMC(JJ1)
! array INDOVERLAP_ORDER does not require any rearrangement,
! but can be used directly to find indices corresponding to
! blue wing order in overlap complex
              INDOVERLAP_ORDER(JJ) = II
           ENDDO

! Find out which line (II1) gets the b.w. condition from what line (II)
! and calculate frequency (XOV) where intensities from line (II) have
! to be saved.
! INFOTAB%IF then gives the corresponding freq. nr. 

! If NO intrinsic line overlap, find out which line (II1) gets the bw.
! cond. from the red wing of line (II). In this case, XOV is larger
! than XMAX(II), and
! INFOTAB%IF is set to NF


           DO JJ=1,NOV-1
             II=INDOVERLAP_ORDER(JJ)
             IF(JJ.EQ.1) INFOTAB(JJ)%INDEX=II
SEARCH:        DO JJ1=JJ+1,NOV
               II1=INDOVERLAP_ORDER(JJ1)
! defined with opposite sign, from def. of x
               XOV=(1.D0-XLAMCMF(II)/LAMBLUE(INDEX(JJ1)))* &
&                CLIGHT/VDOPCMF(II)
! since also xmax is positive 
               XMAX=XMAXDOP(II)
               IF(XOV.GT.XMAX) THEN              
! for narrow line, xov can be larger than xmax, altough iov was
! possible for prior line 
                 IF(INFOTAB(JJ1)%IF .EQ.0. .OR. INFOTAB(JJ1)%IF &
&                  .EQ. DBLE(NF)) THEN
! no intrinsic line-overlap, use closest line (this is a MUST), 
! since otherwise the range in between two lines would not be line-free!
                   
                    INFOTAB(JJ1)%INDEX=II1
                    INFOTAB(JJ1)%INDBW=II
                    INFOTAB(JJ1)%XOV=-XOV
                    INFOTAB(JJ1)%IF=DBLE(NF)
                    ENDIF 
                  EXIT SEARCH
               ENDIF
               IF(INFOTAB(JJ1)%IF.EQ.0. .OR. INFOTAB(JJ1)%IF.EQ.DBLE(NF)) THEN
! intrinsic line-overlap, use line with most blueward freq.
! (numerically most exact)
                  INFOTAB(JJ1)%INDEX=II1
                  INFOTAB(JJ1)%INDBW=II
                  INFOTAB(JJ1)%XOV=-XOV
                  XNF=1.D0+(XMAX+XOV)*(NF-1)*.5D0/XMAX !(-(-XOV)
                  IF(XNF.LE.1..OR.XNF.GE.DBLE(NF)) STOP ' ERROR IN XNF'
                  INFOTAB(JJ1)%IF=XNF
               ENDIF 
               ENDDO SEARCH
           ENDDO

! perform some tests
           IF(INFOTAB(1)%INDBW .NE. 0) STOP ' ERROR 1 IN INFOTABLE'
           DO JJ=2,NOV
             IF(INFOTAB(JJ)%IF .EQ. DBLE(NF)) THEN
               IF (INFOTAB(JJ)%INDBW .NE. INFOTAB(JJ-1)%INDEX) &
&                STOP ' ERROR 2 IN INFOTABLE'
             ENDIF
! for tests only
!           PRINT*,JJ,INFOTAB(JJ)%INDEX,INFOTAB(JJ)%INDBW, &
!             INFOTAB(JJ)%XOV,INFOTAB(JJ)%IF
           ENDDO           

! ----------------------------------------------------------------------



! perform cmf calc. for overlap interval, in order of descending
! bw. frequencies
           
           DO JJ=1,NOV
              JJ1=INDXLAMC(JJ+JSTART-1)
              II=INDOVERLAP_ORDER(JJ)
              XLAM = XLAMCMF(II)  
              VDOP = VDOPCMF(II)  
              DL=XLAM*XMAXDOP(II)*VDOP/CLIGHT

! ----------------------------------------------------------------------

! Find out intrinsically overlapping lines to allow for calc. of
! total background opacities/em.
              INDEXOV=0
              IOV=0
              DO JK=1,JJ-1
              II1=INDOVERLAP_ORDER(JK)
              XLAM1 = XLAMCMF(II1)    
              DL1=XLAM1*XMAXDOP(II1)*VDOPCMF(II1)/CLIGHT
              IF(ABS(XLAM1-XLAM).LT.DL+DL1) THEN
                IOV=IOV+1
                INDEXOV(IOV)=II1         
              ENDIF
              ENDDO
              DO JK=JJ+1,NOV
              II1=INDOVERLAP_ORDER(JK)
              XLAM1 = XLAMCMF(II1)    
              DL1=XLAM1*XMAXDOP(II1)*VDOPCMF(II1)/CLIGHT
              IF(ABS(XLAM1-XLAM).LT.DL+DL1) THEN
                IOV=IOV+1
                INDEXOV(IOV)=II1         
              ENDIF
              ENDDO
!test
              IF(IOV.EQ.0) THEN
                IF(JJ.NE.1..AND.INFOTAB(JJ)%IF.NE.DBLE(NF)) &
&                 STOP ' ERROR IN INDEXOV' 
!                PRINT*,'LINE AT ',XLAMCMF(II),':NO !INTRINSIC OVERLAP' 
              ELSE               
                IF(JJ.NE.1.AND.INFOTAB(JJ)%IF.NE.DBLE(NF)) THEN
                  FLAG=.FALSE.
                  DO JK=1,IOV
                    IF(INDEXOV(JK).EQ.INFOTAB(JJ)%INDBW) FLAG=.TRUE.
                  ENDDO
                  IF(.NOT. FLAG) THEN
                    PRINT*,INFOTAB(JJ)%INDEX,INFOTAB(JJ)%INDBW, &
&                     INFOTAB(JJ)%XOV,INFOTAB(JJ)%IF
                    PRINT*,INDEXOV(1:IOV)
                    STOP ' BW INDEX NOT FOUND IN INDEXOV'
                  ENDIF
                ENDIF  
!                PRINT*,'LINE AT ',XLAMCMF(II),':!INTRINSIC OVERLAP FOR ', &
!                INDEXOV(1:IOV) 
              ENDIF

! ----------------------------------------------------------------------
              
              CALL CMFMULTI(JJ,II,R,V,GRADV,TEMP,XLAM,VDOP,VMAX,TEFF, &
&               CORRFC,AJLMEAN,ALO,NOV,INDEXOV,IOV)
              IF(II.EQ.JJ1) THEN
                WRITE (*,FMT=9005) XLAM,LABLINELO(II),LABLINEUP(II),IOV  
              ELSE
                WRITE (*,FMT=9006) XLAM,LABLINELO(II),LABLINEUP(II),IOV  
              ENDIF
                
              DO L = 1,ND  
                SNEW = AJLMEAN(L) - ALO(L)*TULMAT(II,L)  
                IF (SNEW.LT.0.D0 .OR. ABS(ALO(L)).GT.1.D0) THEN
                  PRINT *,' WARNING!!!! SOMETHING WRONG WITH LINE-ALO'
                  ALO(L) = 0.D0  
                  SNEW = AJLMEAN(L)  
                END IF  
!
! avoid to overwrite TLUMAT (->OPALM) and TULMAT(->SLINEM) 
!
                TLUAUX(L,JJ) = BLUCMF(II)*SNEW  
                TULAUX(L,JJ) = BULCMF(II)*SNEW + AULCMF(II)* (1.D0-ALO(L))
              END DO 
           END DO
! overlap complex finished, can overwrite TLUMAT and TULMAT

           DO JJ=1,NOV
              II=INDOVERLAP_ORDER(JJ)
              DO L=1,ND
                TLUMAT(II,L) = TLUAUX(L,JJ)  
                TULMAT(II,L) = TULAUX(L,JJ)
              ENDDO
           ENDDO  
           
           J=JEND

        ENDIF !SINGLE OR OVERLAP   
           
    ELSE IF (INDEXCMF(II).EQ.1) THEN  

        IF (XLAM.EQ.0.D0) THEN  
           IF (INDEXSA(II).NE.0) STOP 'ERROR IN XLAM=0(2)!'  
        ELSE  
           WRITE (*,FMT=9010) XLAM,LABLINELO(II),LABLINEUP(II),INDEXSA(II)  
        END IF  
    END IF  

    J=J+1

END DO CMFLOOP 
CLOSE(333)

RETURN

 9000 FORMAT ('CMFT AT',F12.3,' A ',A6,' TO ',A6,' SINGLE LINE')  
 9005 FORMAT ('CMFT AT',F12.3,' A ',A6,' TO ',A6,' OVERLAP, IOV =',I2)  
 9006 FORMAT ('CMFT AT',F12.3,' A ',A6,' TO ',A6,' OVERLAP, ORDER CHANGED, IOV = ',I2)  
 9010 FORMAT ('SA-T AT',F12.3,' A ',A6,' TO ',A6,' FOR ',I3,' DEPTH', &
&       ' POINTS')

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE PREPOV(R,V,GRADV,TEMP,VMAX,TEFF,ND)
!
! prepares consideration of line overlap (intrinsic and wind induced)   
!  
USE nlte_type
USE fund_const, ONLY: CLIGHT
USE nlte_dim

USE nlte_var, ONLY: INDXLAMC,XLAMCMF,VDOPCMF,XMAXDOP,INDEXCMF, &
&                   TLUMAT,Z

USE nlte_opt, ONLY: IOV_ONLY
USE cmf_multi, ONLY: TAUSCRIT, INDOV, INDOVERLAP_ORDER, &
&                    LMAXJP, FMAX, FMIN, UV, ZRAY


IMPLICIT NONE

INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT, NP=ID_NPOIN  

INTEGER(I4B) :: ND
REAL(DP) :: VMAX, TEFF
REAL(DP), DIMENSION(ND1) :: R,V,GRADV,TEMP

INTEGER(I4B) :: II,J,JJ,JLAST,JSTART,JOLD,LL,JA,IA
INTEGER(I4B) :: JP,LMAX,L

REAL(DP) :: DELTA,DDOP,XLSTART,XLAM,XLAMOLD,TAUS,NEXTLAM,NEXTLAMOLD
REAL(DP) :: FM,ZL,ZL1

LOGICAL FIRST, CORE
LOGICAL, SAVE :: DO_ONCE =.TRUE.

NEXTLAMOLD=0.
INDOV=0
FIRST=.TRUE.
JLAST=0
DO J=ID_NTTRD,1,-1
     II = INDXLAMC(J)  
     IF (INDEXCMF(II).EQ.2) THEN
     JLAST=J  
     EXIT
     ENDIF
ENDDO     
IF (JLAST.EQ.0) STOP ' JLAST NOT FOUND!'

! calc. xmax
CALL CALCXMAX(VMAX,TEFF,TEMP,ND)

!NOTE: if SA line inside overlap complex, end complex

! INDOV = 0 SINGLE LINE OR SOBO
! INDOV = 1 BEGIN OF OVERLAP COMPLEX
! INDOV = 2 INSIDE OF OVERLAP COMPLEX
! INDOV = 3 END OF OVERLAP COMPLEX

!note: INDOV ordered in actual sequence of restwavelengths

DO J = 1,ID_NTTRD
  
     II = INDXLAMC(J)  

     XLAM = XLAMCMF(II)  
     IF (INDEXCMF(II).EQ.2) THEN  
          DO LL=1,ND
            TAUS=TLUMAT(II,LL)/GRADV(LL)
!           use velocities only where significant contribution
            IF(TAUS.GT.TAUSCRIT) EXIT
          ENDDO  
          IF(LL.GT.ND) LL=ND

!         maximum influence due to expansion and thermal line width
          DDOP =XLAM*XMAXDOP(II)*VDOPCMF(II)/CLIGHT 
          DELTA=XLAM*(2.*V(LL)*VMAX)/CLIGHT+DDOP  
!         note that previous line can have wider range of influence
          NEXTLAM=MAX(XLAM+DELTA,NEXTLAMOLD)  
          IF(FIRST) THEN
            XLSTART=XLAM
            JSTART=J
            NEXTLAMOLD=NEXTLAM
            XLAMOLD=XLAM
            JOLD=J
            FIRST=.FALSE.
            CYCLE
          ENDIF
          
          IF(J.EQ.JLAST) THEN

! for lines inside overlap complex, check whether SA-lines present;
! if so, break up overlap complex into subcomplexes in order
! to be able have SA-lines as single (correction of previous and
! actual lines required)
            
             IF(XLSTART.NE.XLAMOLD) THEN !OVERLAP
! first line
                INDOV(JSTART)=1
                DO JJ=JSTART+1,J-1 !all lines inside complex
                  IA = INDXLAMC(JJ)  
                  IF (INDEXCMF(IA).EQ.1) THEN !SA-LINE
                    INDOV(JJ)=0  
                    JA=INDOV(JJ-1)
                    IF(JA.EQ.1) INDOV(JJ-1)=0 !correction of previous index
                    IF(JA.EQ.2) INDOV(JJ-1)=3 !similar
                  ELSE
                    INDOV(JJ)=2 !standard, if no SA lines in between
                    JA=INDOV(JJ-1)
                    IF(JA.EQ.0) INDOV(JJ)=1 !correction of actual index
                  ENDIF
                ENDDO  
! last line J (end of complex)
                IA = INDXLAMC(J)  
                IF (INDEXCMF(IA).EQ.1) THEN !SA-LINE
                    INDOV(J)=0  
                    JA=INDOV(J-1)
                    IF(JA.EQ.1) INDOV(J-1)=0 !correction of previous index
                    IF(JA.EQ.2) INDOV(J-1)=3 !similar
                ELSE
                    INDOV(J)=3 !standard, if not SA line  
                    JA=INDOV(J-1)
                    IF(JA.EQ.0) INDOV(J)=0 !correction of actual index
                ENDIF

              ENDIF

!        corrected for thermal width
          ELSE IF(XLAM-DDOP.GT.NEXTLAMOLD) THEN
! same philosophy as above
            IF(XLSTART.NE.XLAMOLD) THEN !OVERLAP
                INDOV(JSTART)=1
                DO JJ=JSTART+1,JOLD-1
                  IA = INDXLAMC(JJ)  
                  IF (INDEXCMF(IA).EQ.1) THEN !SA-LINE
                    INDOV(JJ)=0  
                    JA=INDOV(JJ-1)
                    IF(JA.EQ.1) INDOV(JJ-1)=0 !correction of previous index
                    IF(JA.EQ.2) INDOV(JJ-1)=3 !similar
                  ELSE
                    INDOV(JJ)=2  !standard
                    JA=INDOV(JJ-1)
                    IF(JA.EQ.0) INDOV(JJ)=1 !correction of actual index
                  ENDIF
                ENDDO  
                IA = INDXLAMC(JOLD)  
                IF (INDEXCMF(IA).EQ.1) THEN !SA-LINE
                    INDOV(JOLD)=0  
                    JA=INDOV(JOLD-1)
                    IF(JA.EQ.1) INDOV(JOLD-1)=0 !correction of previous index
                    IF(JA.EQ.2) INDOV(JOLD-1)=3 !similar
                ELSE
                    INDOV(JOLD)=3  !standard
                    JA=INDOV(JOLD-1)
                    IF(JA.EQ.0) INDOV(JOLD)=0 !correction of actual index
                ENDIF

            ENDIF
            XLSTART=XLAM
            JSTART=J
          ENDIF  
   
          NEXTLAMOLD=NEXTLAM
          XLAMOLD=XLAM
          JOLD=J
     ENDIF
ENDDO

! for tests
!DO J = 1,ID_NTTRD
!     II = INDXLAMC(J)  
!     XLAM = XLAMCMF(II)  
!     PRINT*,J,XLAM,INDEXCMF(II),INDOV(J)
!ENDDO     

IF(IOV_ONLY.OR..NOT.DO_ONCE) RETURN

DO_ONCE=.FALSE.

RETURN

! prepare some variables required for wind induced overlap, old version
! NOT REQUIRED for new version

Stop ' Should never stop here (prepov)'
DO JP=1,NP-1
   LMAX=MIN0(ND,NP+1-JP)
   LMAXJP(JP)=LMAX
   CORE=LMAX.EQ.ND
   FM=Z(1,JP)/R(1)*V(1)
   FMAX(JP)=FM

      IF(.NOT.CORE) THEN
       FMIN(JP)=-FM
      ELSE
       FMIN(JP)=Z(ND,JP)*V(ND)
      ENDIF

      UV(1,JP)=FM
      ZRAY(1,JP)=Z(1,JP)
      DO L=2,LMAX
        ZL=Z(L,JP)
        UV(L,JP)=ZL/R(L)*V(L)
        ZRAY(L,JP)=ZL
      ENDDO

      IF(.NOT.CORE) THEN
        DO L=1,LMAX-1
          LL=2*LMAX-L
          UV(LL,JP)=-UV(L,JP)
          ZRAY(LL,JP)=-ZRAY(L,JP)
        ENDDO
      ENDIF

ENDDO
 
RETURN

END

!
!-----------------------------------------------------------------------
!
SUBROUTINE CALCXMAX(VMAX,TEFF,TEMP,ND)
!
! calculates xmax as function of beta
!
USE nlte_type
USE fund_const, ONLY: WPI=>SQPI
USE nlte_dim

USE nlte_var, ONLY: INDXLAMC,INDEXCMF,VDOPCMF,XMAXDOP,OPALM=>TLUMAT,OPACM

IMPLICIT NONE
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  

INTEGER(I4B) ND
REAL(DP) :: TEFF,VMAX
REAL(DP), DIMENSION(ND1) :: TEMP

INTEGER(I4B) :: II,J,LTEMP
REAL(DP) :: BETAP,BETA1,XMAX0

!   index for calculating beta_puls

DO J = 1,ND  
  IF (TEMP(J).GT.TEFF) GO TO 90
END DO 
STOP ' ERROR IN TEMP.STRAT.'  

90      CONTINUE  

LTEMP = J - 1  
PRINT *  

DO J = 1,ID_NTTRD
  
     II = INDXLAMC(J)  
     IF (INDEXCMF(II).EQ.2) THEN  

        BETAP = OPACM(II,LTEMP)*VDOPCMF(II)/ (OPALM(II,LTEMP)*VMAX)  
 
        IF (BETAP.LT.-1.D0) PRINT *,' WARNING: BETA_P NEGATIVE IN CALCXMAX'  

        XMAX0 = 3.D0  
        BETA1 = ABS(BETAP)  

        IF (BETA1.LT..01D0) &
&         XMAX0 = MAX(3.D0,SQRT(-LOG(.01D0*BETA1*WPI)))  
        XMAXDOP(II) = XMAX0
     ENDIF
   
ENDDO

RETURN

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CMFSING(II,R,V,DVDR,T,XLAMBDA,VDOP,VMAX,TEFF,CORRFC, &
&   AJLMEAN,ALO,INVERSION)

! prepares and carries out cmf transfer for single lines

USE nlte_type
USE nlte_dim
USE run_once, ONLY: START_CMFSING
USE nlte_var, ONLY: TAUZ1,TAU1,PP1, &
& AK1,AZ1,SLINEK, &
& Z,P, &
& OPALM=>TLUMAT,SLINEM=>TULMAT,OPACM,SCONTM, &
& INDEXRBB,INDEXCMF,W0=>VP,VP2,W0LAST,XMAXDOP  

IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND=ID_NDEPT,NP=ID_NPOIN  
INTEGER(I4B), PARAMETER :: NF=ID_NFCMF  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORRFC,TEFF,VDOP,VMAX,XLAMBDA  
INTEGER(I4B) ::  II  
!     ..
!     .. array arguments ..
REAL(DP) ::  AJLMEAN(ND),ALO(ND),DVDR(ND),R(ND),T(ND),V(ND)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AIC,AUX,DBDR,DELTAX,ETAK,OPAKK,OPALK,RL, &
&                 XMUE,XMUE2
INTEGER(I4B) ::  JP,K,L,LMAX,LZ  
LOGICAL INVERSION  
!     ..
!     .. local arrays ..
REAL(DP) ::  ETAC(ND),OPAC(ND),OPAL(ND),PHI(NF,ND),PP(ND), &
&                 PWEIGHT(NF,ND),SCONT(ND),SLINE(ND), &
&                 UBLUWI(ND,NP-1),UCMF(ND),VCMF(ND-1)
!     ..
!     .. external subroutines ..
EXTERNAL CONT2,DIFFUS,FGRID,RAY  
!     ..
!     .. intrinsic functions ..
!INTRINSIC MIN0  
!     ..
!     .. data statements ..
!
!     ..
!
IF (START_CMFSING) THEN  

!***  might have already been calculated in cmf_simple (nlte_approx)
!***  anyway, recalulate, since previous update
!
!***  line-independent quantities
!
JPLOOP1: DO JP = 1,NP - 1  
          LMAX = MIN0(NP+1-JP,ND)  
          LZ = LMAX - 1  

          DO L = 1,LZ  
               TAUZ1(L,JP) = 1.D0/ (Z(L,JP)-Z(L+1,JP))  
          END DO

          DO L = 2,LZ  
               TAU1(L,JP) = 2.D0/ (Z(L-1,JP)-Z(L+1,JP))  
          END DO
!
!     pp(l)=velocity gradient,projected on the present ray
!
          DO L = 1,LMAX  
               RL = R(L)  
               XMUE = Z(L,JP)/RL  
               XMUE2 = XMUE*XMUE  
               PP1(L,JP) = (XMUE2*DVDR(L)+ (1.D0-XMUE2)*V(L)/RL)  
          END DO

END DO JPLOOP1

START_CMFSING = .FALSE.  
END IF  

DO L = 1,ND  
     OPAL(L) = OPALM(II,L)  
     SLINE(L) = SLINEM(II,L)  
     OPAC(L) = OPACM(II,L)  
     SCONT(L) = SCONTM(II,L)  
     ETAC(L) = SCONT(L)*OPAC(L)  
END DO  
!
!     blue wing boundary
!
CALL DIFFUS(XLAMBDA,T,R,ND,AIC,DBDR)  

CALL CONT2(ND,NP,R,P,Z,SCONT,OPAC,AIC,DBDR,CORRFC,UBLUWI)  
!
CALL FGRID(NF,ND,PHI,PWEIGHT,DELTAX,VDOP,VMAX,TEFF,T,XMAXDOP(II))  

INVERSION = .FALSE.  

!  
! check whether inversion
!

INVCHECK: DO L = 1,ND  
  OPAKK = PHI((NF+1)/2,L)*OPAL(L)+OPAC(L)  
!  if(xlambda.gt.4686..and.xlambda.lt.4689.) then
!  print*,l,opakk,PHI((NF+1)/2,L),opal(l),opac(l)
!  endif
  IF(OPAKK.LT.0.D0.AND..NOT.INVERSION) THEN
    PRINT *,' INVERTED LEVELS FOR FOLLOWING TRANSITION '  
    INVERSION=.TRUE.
    EXIT INVCHECK
  ENDIF
END DO INVCHECK


IF(.NOT.INVERSION) THEN
!STANDARD TREATMENT
DO L = 1,ND
     DO K = 1,NF  
          OPALK = PHI(K,L)*OPAL(L)  
          ETAK = SLINE(L)*OPALK + ETAC(L)  
          OPAKK = OPALK + OPAC(L)  
          SLINEK(K,L) = ETAK/OPAKK         
!  if(xlambda.gt.4686..and.xlambda.lt.4689..and.k.eq.(nf+1)/2) then
!  print*,l,opakk,opalk,opac(l)
!  print*,l,etak,sline(l)*opalk,etac(l)
!  print*,l,slinek(k,l)
!  print*
!  endif
          AK1(K,L) = 1.D0/OPAKK  
          IF (L.NE.1) AZ1(K,L-1) = 2.D0/ (OPAKK+1.D0/AK1(K,L-1))  
     END DO
END DO  

ELSE
!INVERSION
DO L = 1,ND
     DO K = 1,NF  
          OPALK = PHI(K,L)*OPAL(L)  
          ETAK = SLINE(L)*OPALK + ETAC(L)  
          OPAKK = OPALK + OPAC(L)  
!
!   ATTENTION: for inversion, ak1 is now the actual opacity,  
!   and slinek is just the emisivity           
!
          AK1(K,L) = OPAKK
          SLINEK(K,L) = ETAK 
          IF (L.NE.1) AZ1(K,L-1) = (OPAKK+AK1(K,L-1))/ 2.D0  
     END DO
END DO  

ENDIF


AJLMEAN = 0.D0  
ALO = 0.D0  

JPLOOP2: DO JP = 1,NP - 1  
     LMAX = MIN0(NP+1-JP,ND)  
     LZ = LMAX - 1  
!
!     bluewing boundary condition
!
     UCMF(1) = UBLUWI(1,JP)  

     DO L = 1,LZ  
          UCMF(L+1) = UBLUWI(L+1,JP)  
          AUX = .5D0* (OPAC(L)+OPAC(L+1))  
          VCMF(L) = (UCMF(L+1)-UCMF(L))/AUX/ (Z(L,JP)-Z(L+1,JP))  
     END DO

     DO L = 1,LMAX  
          PP(L) = PP1(L,JP)/DELTAX  
     END DO

     IF(.NOT.INVERSION) THEN
! STANDARD SOLUTION OVER TAU
       CALL RAY(.TRUE.,1,1, &
         JP,Z(:,JP),R,ND,NP,UCMF,VCMF,SLINE,PHI,PWEIGHT, &
         DELTAX,OPAC,ETAC,OPAL,AIC,DBDR,CORRFC,W0(:,JP),W0LAST, &
         AJLMEAN,ALO,TAUZ1(:,JP),TAU1(:,JP),PP,XLAMBDA)

     ELSE
! NEW FORMULATION OVER Z
       CALL RAY_INV(.TRUE.,1,1, &
         JP,Z(:,JP),R,ND,NP,UCMF,VCMF,SLINE,PHI,PWEIGHT, &
         DELTAX,OPAC,ETAC,OPAL,AIC,DBDR,CORRFC,W0(:,JP),W0LAST, &
         AJLMEAN,TAUZ1(:,JP),TAU1(:,JP),PP)      
     ENDIF
       
END DO  JPLOOP2

!NOTE: SINCE ALO IS NOT MODIFIED IN THE INVERTED CASE, WE SIMPLY DON'T TOUCH IT
!
!      DO L=1,ND
!      print*,l,' ',sline(l),' ',ajlmean(l),'  ',alo(l),'  ',opal(l)
!      print*,l,' ',ajlmean(l),'  ',alo(l)
!      END DO
!
RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CMFMULTI(JJ,II,R,V,DVDR,T,XLAMBDA,VDOP,VMAX,TEFF,CORRFC, &
&   AJLMEAN,ALO,NOV,INDEXOV,IOV)

! prepares and carries out cmf transfer for given line in overlap complex

USE nlte_type
USE fund_const, ONLY: CLIGHT,WPI=>SQPI
USE nlte_dim
USE run_once, ONLY: START_CMFMULTI
USE nlte_var, ONLY: TAUZ1,TAU1,PP1, &
& AK1,AZ1,SLINEK, &
& Z,P, &
& OPALM=>TLUMAT,SLINEM=>TULMAT,OPACM,SCONTM, &
& INDEXRBB,INDEXCMF,W0=>VP,VP2,W0LAST,VTURB, &
& VDOPCMF,XLAMCMF,XMAXDOP 

USE nlte_opt, ONLY: IOV_ONLY
USE cmf_multi, ONLY: INFOTAB, USAVE, VSAVE

IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND=ID_NDEPT,NP=ID_NPOIN  
INTEGER(I4B), PARAMETER :: NF=ID_NFCMF  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORRFC,TEFF,VDOP,VMAX,XLAMBDA  
INTEGER(I4B) ::  JJ,II,NOV,IOV  
!     ..
!     .. array arguments ..
INTEGER(I4B) :: INDEXOV(IOV)
REAL(DP) ::  AJLMEAN(ND),ALO(ND),DVDR(ND),R(ND),T(ND),V(ND)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AIC,AUX,DBDR,DELTAX,ETAK,OPAKK,OPALK,RL, &
&            XMUE,XMUE2,OPABL,SBL,ETABL,XMAX,XMAX1, &
&            VDOPL,DEL,DEL1,CV,PHIKL,XLAMI,W,W1,XK,XKI,Q, &
&            LRED,LBLU,XMAXR,XMAXB,XOV,DEW,DELTA,DELTAX1

INTEGER(I4B) ::  I,JP,K,L,LMAX,LZ,II1,IBW1,IBW2,IIR,IIB  
LOGICAL INVERSION,FLAG  
!     ..
!     .. local arrays ..
REAL(DP) ::  ETAC(ND),OPAC(ND),OPAL(ND),PHI(NF,ND),PP(ND), &
&            PWEIGHT(NF,ND),SCONT(ND),SLINE(ND), &
&            UBLUWI(ND,NP-1),VBLUWI(ND-1,NP-1), &
&            UCMF(ND),VCMF(ND-1), &
&            OPABG(NF,ND),ETABG(NF,ND)

!     ..
!     .. external subroutines ..
EXTERNAL CONT2,DIFFUS,FGRID,RAY  
!     ..
!     .. intrinsic functions ..
!INTRINSIC MIN0  
!     ..
!     .. data statements ..
!
IF (INFOTAB(JJ)%INDEX .NE. II) STOP ' ERROR IN JJ <=> II'

IF (START_CMFMULTI) THEN  
!
!***  line-independent quantities
!
JPLOOP1: DO JP = 1,NP - 1  
          LMAX = MIN0(NP+1-JP,ND)  
          LZ = LMAX - 1  

          DO L = 1,LZ  
               TAUZ1(L,JP) = 1.D0/ (Z(L,JP)-Z(L+1,JP))  
          END DO

          DO L = 2,LZ  
               TAU1(L,JP) = 2.D0/ (Z(L-1,JP)-Z(L+1,JP))  
          END DO
!
!     pp(l)=velocity gradient,projected on the present ray
!
          DO L = 1,LMAX  
               RL = R(L)  
               XMUE = Z(L,JP)/RL  
               XMUE2 = XMUE*XMUE  
               PP1(L,JP) = (XMUE2*DVDR(L)+ (1.D0-XMUE2)*V(L)/RL)  
          END DO

END DO JPLOOP1

START_CMFMULTI = .FALSE.  
END IF  

DO L = 1,ND  
     OPAL(L) = OPALM(II,L)  
     SLINE(L) = SLINEM(II,L)  
END DO  

! continuum background
DO L=1,ND
  OPABL = OPACM(II,L)
  SBL   = SCONTM(II,L)
  ETABL = SBL * OPABL 
  OPAC(L) = OPABL
  SCONT(L) = SBL
  ETAC(L) = ETABL
  DO K=1,NF
     OPABG(K,L) = OPABL  
     ETABG(K,L) = ETABL  
  ENDDO
ENDDO

IF (IOV.GT.0) THEN 

! include overlapping lines

DEL1 = VDOP/VMAX  
XMAX = XMAXDOP(II)*DEL1  
DELTAX = 2.D0*XMAX/DBLE(NF-1)  
CV=CLIGHT/VMAX

DO I=1,IOV
  II1=INDEXOV(I)
  XLAMI=XLAMCMF(II1)
  XMAX1=XMAXDOP(II1)*VDOPCMF(II1)/VMAX
  W=XLAMI/XLAMBDA
  W1=(W-1.D0)*CV

  FLAG=.FALSE.
  DO L = 1,ND  
  OPABL = OPALM(II1,L)
  ETABL = SLINEM(II1,L) * OPABL

  VDOPL = (VDOPCMF(II1)**2-VTURB**2)*T(L)/TEFF  
  VDOPL=SQRT(VDOPL+VTURB**2)
  DEL = VMAX/VDOPL  
    
     DO K=1,NF
       XK = XMAX - (K-1)*DELTAX  
       XKI=W*XK+W1
       IF(ABS(XKI).LT.XMAX1) FLAG=.TRUE.
       PHIKL = EXP(- ((XKI*DEL)**2)) * DEL/WPI
       OPABG(K,L) = OPABG(K,L) + OPABL*PHIKL
       ETABG(K,L) = ETABG(K,L) + ETABL*PHIKL
     END DO

  ENDDO
  IF(.NOT.FLAG) THEN
    PRINT*,XLAMBDA,XLAMI,' SOMETHING WRONG IN !INTRINSIC OVERLAP'
  ENDIF  
ENDDO
ENDIF


! lower boundary
CALL DIFFUS(XLAMBDA,T,R,ND,AIC,DBDR)  

! freq. grid and integr. weights
CALL FGRID(NF,ND,PHI,PWEIGHT,DELTAX,VDOP,VMAX,TEFF,T,XMAXDOP(II))  

INVERSION = .FALSE.  
!  
! check whether inversion, assuming worst case scenario:
! background pure cont
!
INVCHECK: DO L = 1,ND  
  OPAKK = PHI((NF+1)/2,L)*OPAL(L)+OPAC(L)  
  IF(OPAKK.LT.0.D0.AND..NOT.INVERSION) THEN
    PRINT *,' INVERTED LEVELS FOR FOLLOWING TRANSITION '  
    INVERSION=.TRUE.
    EXIT INVCHECK
  ENDIF
END DO INVCHECK


!
! blue wing boundary
! a)  first line, pure cont 
!
IF(JJ.EQ.1) THEN
!
! check consistency
IF(INFOTAB(JJ)%IF .NE. 0.) STOP ' ERROR IN INFOTAB(1)'
CALL CONT2(ND,NP,R,P,Z,SCONT,OPAC,AIC,DBDR,CORRFC,UBLUWI)  

! b)  no intrinsic line overlap, only wind induced
ELSE IF (INFOTAB(JJ)%IF .EQ. DBLE(NF)) THEN
! check consistency
IF(USAVE(ND,NP-1,JJ).NE.DBLE(NF)) STOP ' ERROR IN U, V BW(1)'

  IF(IOV_ONLY) THEN
    CALL CONT2(ND,NP,R,P,Z,SCONT,OPAC,AIC,DBDR,CORRFC,UBLUWI)  
  ELSE
!
! old version
!    CALL CONT2(ND,NP,R,P,Z,SCONT,OPAC,AIC,DBDR,CORRFC,UBLUWI)  
!    CALL OVERLAP(JJ,VMAX,ND,UBLUWI,VBLUWI,OPAC,SCONT)
!    goto 100
!____________________________________________________________________________
! new version, cmf transport from red wing (blue) to blue wing (red)

    CV=CLIGHT/VMAX

    IIR=INFOTAB(JJ)%INDEX
    IIB=INFOTAB(JJ)%INDBW

    LRED = XLAMCMF(IIR)
    LBLU = XLAMCMF(IIB)

    XMAXR = XMAXDOP(IIR)*VDOPCMF(IIR)/VMAX
    XMAXB = XMAXDOP(IIB)*VDOPCMF(IIB)/VMAX
! for tests, remember that XOV with respct to blue comp.
    XOV=-INFOTAB(JJ)%XOV*VDOPCMF(IIB)/VMAX-XMAXB

    DEW=LBLU/LRED
! DELTA also with respect to blue comp.
    DELTA=(1.-DEW)*CV-XMAXB-XMAXR*DEW

    IF(ABS(1.-DELTA/XOV).GT.1.D-6) STOP '  DELTA AND XOV DIFFERENT'

    DELTAX1=DELTA/DBLE(NF-1)

    DO L = 1,ND
     DO K = 1,NF  
          OPAKK = OPAC(L)  
          SLINEK(K,L) = SCONT(L)         
          AK1(K,L) = 1.D0/OPAKK  
          IF (L.NE.1) AZ1(K,L-1) = 2.D0/ (OPAKK+1.D0/AK1(K,L-1))  
     END DO
    END DO  

    DO JP = 1,NP - 1  
     LMAX = MIN0(NP+1-JP,ND)  
     LZ = LMAX - 1  
!
!     bluewing boundary condition from previous line
!
        UCMF(1:LMAX)=USAVE(1:LMAX,JP,JJ)
        VCMF(1:LZ)  =VSAVE(1:LZ,JP,JJ)

     DO L = 1,LMAX  
          PP(L) = PP1(L,JP)/DELTAX1  
     END DO

     CALL RAYCONT(JP,Z(:,JP),R,ND,NP,UCMF,VCMF, &
        DELTAX1,OPAC,ETAC,AIC,DBDR,CORRFC,W0(:,JP),&
        TAUZ1(:,JP),TAU1(:,JP),PP)

     UBLUWI(1:LMAX,JP)=UCMF
     VBLUWI(1:LZ,JP)=VCMF

    END DO 
!100 continue
!_____________________________________________________________________________

  ENDIF

! c)  intrinsic line overlap
ELSE
! check consistency
W=INT(INFOTAB(JJ)%IF)+1.D0-INFOTAB(JJ)%IF
IF(USAVE(ND,NP-1,JJ).NE.W)   STOP ' ERROR IN U, V BW(2)'
!W=1.D0-W
!numerically consistent to last digits (calculated as in subroutine RAY)
W=INFOTAB(JJ)%IF+1.D0-(INT(INFOTAB(JJ)%IF)+1)
IF(VSAVE(ND-1,NP-1,JJ).NE.W) STOP ' ERROR IN U, V BW(3)'
ENDIF


IF(.NOT.INVERSION) THEN
!STANDARD TREATMENT
DO L = 1,ND
     DO K = 1,NF  
          OPALK = PHI(K,L)*OPAL(L)  
          ETAK = SLINE(L)*OPALK + ETABG(K,L)  
          OPAKK = OPALK + OPABG(K,L)  
          SLINEK(K,L) = ETAK/OPAKK         
          AK1(K,L) = 1.D0/OPAKK  
          IF (L.NE.1) AZ1(K,L-1) = 2.D0/ (OPAKK+1.D0/AK1(K,L-1))  
     END DO
END DO  

ELSE
! INVERSION

DO L = 1,ND
     DO K = 1,NF  
          OPALK = PHI(K,L)*OPAL(L)  
          ETAK = SLINE(L)*OPALK + ETABG(K,L)  
          OPAKK = OPALK + OPABG(K,L)  
!
!   ATTENTION: for inversion, ak1 is now the actual opacity,  
!   and slinek is just the emisivity           
!
          AK1(K,L) = OPAKK
          SLINEK(K,L) = ETAK 
          IF (L.NE.1) AZ1(K,L-1) = (OPAKK+AK1(K,L-1))/ 2.D0  
     END DO
END DO  

ENDIF

AJLMEAN = 0.D0  
ALO = 0.D0  

JPLOOP2: DO JP = 1,NP - 1  
     LMAX = MIN0(NP+1-JP,ND)  
     LZ = LMAX - 1  

     IF(JJ.EQ.1 .OR. (IOV_ONLY .AND. INFOTAB(JJ)%IF .EQ. DBLE(NF))) THEN
!
!     bluewing boundary condition from cont
!
        UCMF(1) = UBLUWI(1,JP)  

        DO L = 1,LZ  
          UCMF(L+1) = UBLUWI(L+1,JP)  
          AUX = .5D0* (OPAC(L)+OPAC(L+1))  
          VCMF(L) = (UCMF(L+1)-UCMF(L))/AUX/ (Z(L,JP)-Z(L+1,JP))  
        END DO
       
     ELSE IF (INFOTAB(JJ)%IF .EQ. DBLE(NF)) THEN
!
!     bluewing boundary condition from routine OVERLAP
!
        UCMF(1) = UBLUWI(1,JP)  
        DO L = 1,LZ  
          UCMF(L+1) = UBLUWI(L+1,JP)  
          VCMF(L)   = VBLUWI(L,JP)
        END DO
     ELSE
!
!     bluewing boundary condition from previous line
!
        UCMF(1:LMAX)=USAVE(1:LMAX,JP,JJ)
        VCMF(1:LZ)  =VSAVE(1:LZ,JP,JJ)
     ENDIF

     DO L = 1,LMAX  
          PP(L) = PP1(L,JP)/DELTAX  
     END DO

     IF(.NOT.INVERSION) THEN
! STANDARD SOLUTION OVER TAU
      CALL RAY(.FALSE.,II,NOV, &
        JP,Z(:,JP),R,ND,NP,UCMF,VCMF,SLINE,PHI,PWEIGHT, &
        DELTAX,OPAC,ETAC,OPAL,AIC,DBDR,CORRFC,W0(:,JP),W0LAST, &
        AJLMEAN,ALO,TAUZ1(:,JP),TAU1(:,JP),PP,XLAMBDA)

     ELSE
! NEW FORMULATION OVER Z
      CALL RAY_INV(.FALSE.,II,NOV, &
        JP,Z(:,JP),R,ND,NP,UCMF,VCMF,SLINE,PHI,PWEIGHT, &
        DELTAX,OPAC,ETAC,OPAL,AIC,DBDR,CORRFC,W0(:,JP),W0LAST, &
        AJLMEAN,TAUZ1(:,JP),TAU1(:,JP),PP)      
     ENDIF
       
END DO  JPLOOP2

!NOTE: SINCE ALO IS NOT MODIFIED IN THE INVERTED CASE, WE SIMPLY DON'T TOUCH IT

!
!      DO L=1,ND
!        PRINT*,l,' ',sline(l),' ',ajlmean(l),'  ',alo(l),'  ',opal(l)
!      END DO
!
RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE OVERLAP(JJ,VMAX,ND,UBLUWI,VBLUWI,OPAC,SCONT)

! transforms red wing radiation from line IIB to blue wing radiation for
! line IIR  
! UBLUWI and OPAC are CONTINUUM quantities at line IIR

USE nlte_type
USE fund_const, ONLY: CLIGHT
USE nlte_dim

USE nlte_var, ONLY: XLAMCMF,VDOPCMF,XMAXDOP,Z

USE cmf_multi, ONLY: LTO, INFOTAB, LMAXJP, FMAX, FMIN, UV, ZRAY,&
&                    USAVE,VSAVE,IPLUS,IMINUS,IPLUS_CONT,IMINUS_CONT

IMPLICIT NONE

INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT, NP=ID_NPOIN  

INTEGER(I4B) :: JJ, ND, IT
REAL(DP) :: VMAX
REAL(DP), DIMENSION (ND,NP-1) :: UBLUWI
REAL(DP), DIMENSION (ND-1,NP-1) :: VBLUWI
REAL(DP), DIMENSION (ND) :: OPAC, SCONT

REAL(DP) :: U(ND1), V(ND1-1), ZP(LTO), XIP(LTO), XIM(ND1), UVJ(LTO), &
& UCONT(ND1), VCONT(ND1-1), &
& XIPC(ND1), XIMC(ND1), AUX(ND1), VAUX(ND1), XIMOLD(ND1), &
& OPAP(LTO), SCONP(LTO), DTAU(LTO-1), FSP(LTO-1), FSM(2:ND)

INTEGER(I4B) :: IIR, IIB, JP, L, LL, LMAX, LTOT, LZ, LMINJ, LJ
REAL(DP) :: CV, LRED, LBLU, XMAXR, XMAXB, DEW, DELTA, XOV, &
&           XIPLUS, XIMINUS, VNEW, FMINJ, FMAXJ, &
&           UVR, F, DEL, AUXC, CPLUS, CMINUS, UMIN, SIGM, SIG1, &
&           ZMINUS,ZPLUS, DT, E0, E1, W1, W2


LOGICAL CORE

CV=CLIGHT/VMAX

IIR=INFOTAB(JJ)%INDEX
IIB=INFOTAB(JJ)%INDBW

LRED = XLAMCMF(IIR)
LBLU = XLAMCMF(IIB)

XMAXR = XMAXDOP(IIR)*VDOPCMF(IIR)/VMAX
XMAXB = XMAXDOP(IIB)*VDOPCMF(IIB)/VMAX
! for tests, remember that XOV with respct to blue comp.
XOV=-INFOTAB(JJ)%XOV*VDOPCMF(IIB)/VMAX-XMAXB


DEW=LBLU/LRED
! DELTA also with respect to blue comp.
DELTA=(1.-DEW)*CV-XMAXB-XMAXR*DEW

IF(ABS(1.-DELTA/XOV).GT.1.D-6) STOP '  DELTA AND XOV DIFFERENT'

DO JP=1,NP-1
   LMAX=LMAXJP(JP)
   LZ=LMAX-1
   CORE=LMAX.EQ.ND
   LTOT=2*LMAX-1
   IF(CORE) LTOT=ND
   UVJ(1:LTOT) = UV(1:LTOT,JP)
   ZP(1:LTOT) = ZRAY(1:LTOT,JP) 

!------------------------------------------
! prepare cont-transport for given p-ray
   IF(CORE) THEN
    OPAP(1:LTOT)=OPAC(1:LTOT)
    SCONP(1:LTOT)=SCONT(1:LTOT)
   ELSE
    OPAP(1:LMAX)=OPAC(1:LMAX)
    SCONP(1:LMAX)=SCONT(1:LMAX)
    DO L=LMAX+1,LTOT
      LL=2*LMAX-L
      OPAP(L)=OPAC(LL)
      SCONP(L)=SCONT(LL)
    ENDDO  
   ENDIF

   DO L=1,LTOT-1  
     DTAU(L) = 0.5D0* (OPAP(L)+OPAP(L+1))*(ZP(L)-ZP(L+1)) 
     DT=DTAU(L)
     IF (DT.LT.1.D-8) THEN  
          W1 = .5D0*DT  
          W2 = .5D0*DT  
     ELSE  
          E0 = EXP(-DT)  
          E1 = (1.D0-E0)/DT  
          W1 = 1.D0 - E1  
          W2 = E1 - E0  
     END IF  
     FSP(L)=SCONP(L)*W1+SCONP(L+1)*W2
     IF(L.LE.LMAX-1) FSM(L+1)=SCONP(L+1)*W1+SCONP(L)*W2
   ENDDO
!------------------------------------------

! calculate I_plus and I_minus (CONT) at blue wing of red line from
! UBLUWI, accounting for consistent boundary conditions

   UCONT(1) = UBLUWI(1,JP)  

   DO L = 1,LZ  
     UCONT(L+1) = UBLUWI(L+1,JP)  
     AUXC = .5D0* (OPAC(L)+OPAC(L+1))  
     VCONT(L) = (UCONT(L+1)-UCONT(L))/AUXC/ (ZP(L)-ZP(L+1))  
   END DO

   XIMINUS=IMINUS_CONT(JP)
   XIPC(1)=2.D0*UCONT(1)-XIMINUS
   XIMC(1)=XIMINUS

   IF(CORE) THEN
     XIPLUS=IPLUS_CONT(JP)
     XIPC(LMAX)=XIPLUS
     VNEW=2.D0*UCONT(LMAX)-XIPLUS
     IF(VNEW.GE.0.) THEN
      XIMC(LMAX)=VNEW
     ELSE
      XIMC(LMAX)=0.
     ENDIF
   ELSE
     XIMC(LMAX)=UCONT(LMAX)
     XIPC(LMAX)=XIMC(LMAX)
   ENDIF

! iterative solution for I_plus and I minus, &
! assuming I(l+1/2)=sqrt(I(l)*I(l+1)), see notes.

IF(LZ.GE.2) THEN   
! with this choice, we obtain the best results in cases of numerical problems
   UMIN=1.D-6*MINVAL(UCONT(1:LMAX))   

! start values
   DO L=2,LZ
     XIPC(L)=MAX(UCONT(L)+VCONT(L),UMIN)
     XIMC(L)=MAX(UCONT(L)-VCONT(L),UMIN)
   ENDDO  
   XIMOLD(2:LZ)=XIMC(2:LZ)

! iteration   
   DO IT = 1,20
     DO L=2,LZ
         CPLUS=SQRT(XIPC(L+1)/XIPC(L))
         CMINUS=SQRT(XIMC(L+1)/XIMC(L))
         AUXC=2.D0/(CPLUS+CMINUS)
         XIPC(L)=AUXC*(CMINUS*UCONT(L)+VCONT(L))
         XIPC(L)=MAX(XIPC(L),UMIN)
         XIMC(L)=AUXC*(CPLUS*UCONT(L)-VCONT(L))
         XIMC(L)=MAX(XIMC(L),UMIN)
     ENDDO     
     SIGM=SQRT(DOT_PRODUCT((XIMC(2:LZ)-XIMOLD(2:LZ)), &
&      (XIMC(2:LZ)-XIMOLD(2:LZ))))
     IF(IT.EQ.2) THEN
       SIG1=SIGM
     ELSE IF(IT.GT.2) THEN
       IF(SIGM.LT..05*SIG1) EXIT
     ENDIF
     XIMOLD(2:LZ)=XIMC(2:LZ)
   ENDDO
ENDIF
   
! test for continuum: solution should be "perfect" in u
   DO L=1,LMAX
       XIMOLD(L)=.5D0*(XIPC(L)+XIMC(L)) 
   ENDDO
   SIG1=MAXVAL(ABS(1.-XIMOLD(1:LMAX)/UCONT(1:LMAX)))
   IF (SIG1.GT..003D0) THEN
     PRINT*,' PROBLEM IN CONT-INTERPOLATION AT JP =',JP, SIG1
     DO L=1,LZ
     VAUX(L)=.5D0*(SQRT(XIPC(L)*XIPC(L+1))-SQRT(XIMC(L)*XIMC(L+1)))
     PRINT*,L,XIMOLD(L),UCONT(L),VAUX(L),VCONT(L)
     ENDDO
     L=LMAX
     PRINT*,L,XIMOLD(L),UCONT(L)
   ENDIF

!---
! calculate I_plus and I_minus at red wing of blue line from u and v, &
! accounting for consistent boundary conditions
 
   U(1:LMAX)=USAVE(1:LMAX,JP,JJ)
   V(1:LZ)  =VSAVE(1:LZ,JP,JJ)
! with this choice, we obtain the best results in cases of numerical problems.
! Note, that U may have been set to zero in CMF.
   UMIN=1.D-6*MINVAL(U(1:LMAX),MASK=U(1:LMAX).GT.0.D0)   
   IF(UMIN.EQ.0.D0) STOP ' UMIN = 0 IN OVERLAP'
   WHERE(U(1:LMAX).EQ.0D0) U(1:LMAX)=UMIN

! boundary conditions
   XIMINUS=IMINUS(JP,JJ)
   XIP(1)=2.D0*U(1)-XIMINUS
   XIM(1)=XIMINUS

   IF(CORE) THEN
     XIPLUS=IPLUS(JP,JJ)
     XIP(LMAX)=XIPLUS
     VNEW=2.D0*U(LMAX)-XIPLUS
     IF(VNEW.GE.0.) THEN
      XIM(LMAX)=VNEW
     ELSE
      XIM(LMAX)=0.
     ENDIF
   ELSE
     XIM(LMAX)=U(LMAX)
     XIP(LMAX)=XIM(LMAX)
   ENDIF

! iterative solution for I_plus and I minus for non-boundary points, &
! assuming I(l+1/2)=sqrt(I(l)*I(l+1)), see notes.
IF(LZ.GE.2) THEN   

! start values
   DO L=2,LZ
     XIP(L)=MAX(U(L)+V(L),UMIN)
     XIM(L)=MAX(U(L)-V(L),UMIN)
   ENDDO  
   XIMOLD(2:LZ)=XIM(2:LZ)

! iteration
   DO IT = 1,20
     DO L=2,LZ
         CPLUS=SQRT(XIP(L+1)/XIP(L))
         CMINUS=SQRT(XIM(L+1)/XIM(L))
!         print*,it,jp,l,xip(l+1),xip(l),xim(l+1),xim(l),umin !for tests
         AUXC=2.D0/(CPLUS+CMINUS)
         XIP(L)=AUXC*(CMINUS*U(L)+V(L))
         XIP(L)=MAX(XIP(L),UMIN)
         XIM(L)=AUXC*(CPLUS*U(L)-V(L))
         XIM(L)=MAX(XIM(L),UMIN)
       ENDDO
       SIGM=SQRT(DOT_PRODUCT((XIM(2:LZ)-XIMOLD(2:LZ)), &
&      (XIM(2:LZ)-XIMOLD(2:LZ))))
     IF(IT.EQ.2) THEN
       SIG1=SIGM
     ELSE IF(IT.GT.2) THEN
       IF(SIGM.LT..05*SIG1) EXIT
     ENDIF
     XIMOLD(2:LZ)=XIM(2:LZ)
   ENDDO
ENDIF   

! test for lines: solution should be "OK" in u
   DO L=1,LMAX
       XIMOLD(L)=.5D0*(XIP(L)+XIM(L)) 
   ENDDO
   SIG1=MAXVAL(ABS(1.-XIMOLD(1:LMAX)/U(1:LMAX)),MASK=U(1:LMAX).NE.UMIN)
   IF (SIG1.GT..1D0) THEN
     PRINT*,' PROBLEM IN LINE-INTERPOLATION AT JP =',JP, SIG1
! for tests only
!     DO L=1,LZ
!       VAUX(L)=.5D0*(SQRT(XIP(L)*XIP(L+1))-SQRT(XIM(L)*XIM(L+1)))
!       PRINT*,L,XIMOLD(L),U(L),VAUX(L),VSAVE(L,JP,JJ)
!     ENDDO
!     L=LMAX
!     PRINT*,L,XIMOLD(L),USAVE(L)
   ENDIF

   IF(.NOT.CORE) THEN
     DO L=1,LMAX-1
      LL=2*LMAX-L
      XIP(LL)=XIM(L)
     ENDDO
   ENDIF

!---
! find out which intensities at red wing (blue comp.) correspond to
! blue wing intensities for red comp.)
! In case of no interfering continuum, both intensities are equal 
 
   LMINJ=1
   FMINJ=FMIN(JP)
   FMAXJ=FMAX(JP)

PLUS: DO L=1,LMAX

     UVR=UVJ(L)
     F=UVR-DELTA
     LJ=L
     IF(F.LT.FMINJ) THEN

        DO LL=LJ,LMAX
            AUX(LL)=XIPC(LL)
!       in this case, Iplus is totally determined by the local blue-wing
!       continuum
!            PRINT*,'PLUS',JP,LL,XIPC(LL)   
        ENDDO
        EXIT PLUS

     ELSE

     DO LL=LMINJ,LTOT-1

       IF(F.LT.UVJ(LL).AND.F.GE.UVJ(LL+1)) THEN
         DEL=(F-UVJ(LL))/(UVJ(LL+1)-UVJ(LL))
! del corresponds to both uv and z, since assumed as linear in both cases
         IF(XIP(LL+1).EQ.0.D0) THEN
           IF(LL+1.EQ.LTOT) THEN ! outer boundary OK
             XIPLUS=XIP(LL)*(1.D0-DEL)+XIP(LL+1)*DEL
           ELSE
             STOP ' XIP EQ 0 AND LL+1 NE LTOT!'
           ENDIF  
         ELSE         
           XIPLUS=XIP(LL)**(1.D0-DEL)*XIP(LL+1)**DEL
         ENDIF
! xiplus input and output!
         CALL FORMACON_PLUS(JP,LTOT,L,LL,DEL,XIPLUS,XIPC(L),ZP,OPAP,SCONP, &
&          DTAU, FSP)
         AUX(L)=XIPLUS
!         PRINT*,'PLUS',JP,L,UVR,LL,LL+1,AUX(L)
!        This is the incident intensity(plus) which is additionally
!        modified by continuum procecesses between zmin(f) and z(l)
!        to yield the final blue wing intensity(plus) at z(l)
         LMINJ=LL
         CYCLE PLUS
       ENDIF

     ENDDO 
     
     PRINT*,' AT XBLUE = ',LBLU,' AND DELTA = ',DELTA
     STOP ' UVBLUE(XIP) NOT FOUND!'
 
     ENDIF

   ENDDO PLUS


      U(1:LMAX)=.5D0*AUX(1:LMAX)    !0.5 Iplus
   VAUX(1:LMAX) =  AUX(1:LMAX)    !    Iplus

MINUS: DO L=1,LMAX

     UVR=UVJ(L)
     F=UVR+DELTA
     LJ=L
     IF(F.GE.FMAXJ) THEN
       AUX(L)=XIMC(L)
!      in this case, Iminus is totally determined by the local blue-wing
!      continuum
!       PRINT*,'MINUS',JP,L,XIMC(L)   
       CYCLE MINUS
     ENDIF

     DO LL=1,LJ-1

       IF(F.LT.UVJ(LL).AND.F.GE.UVJ(LL+1)) THEN
         DEL=(F-UVJ(LL))/(UVJ(LL+1)-UVJ(LL))
         IF(XIM(LL).EQ.0.D0) THEN
           IF(LL.EQ.1) THEN ! outer boundary OK
             XIMINUS=XIM(LL)*(1.D0-DEL)+XIM(LL+1)*DEL
           ELSE
             STOP ' XIM EQ 0 AND LL NE 1!'
           ENDIF  
         ELSE         
           XIMINUS=XIM(LL)**(1.D0-DEL)*XIM(LL+1)**DEL
         ENDIF
! ximinus input and output!
         CALL FORMACON_MINUS(JP,LTOT,L,LL,DEL,XIMINUS,XIMC(L),ZP,OPAP,SCONP, &
&          DTAU, FSM)
         AUX(L)=XIMINUS
!         PRINT*,'MINUS',JP,L,UVR,LL,LL+1,AUX(L)
!        This is the incident intensity(minus) which is additionally
!        modified by continuum procecesses between zmax(f) and z(l)
!        to yield the final blue wing intensity(minus) at z(l)
         CYCLE MINUS
       ENDIF

     ENDDO

     PRINT*,' AT XBLUE = ',LBLU,' AND DELTA = ',DELTA
     STOP ' UVBLUE(XIM) NOT FOUND!'

   ENDDO MINUS

   UBLUWI(1:LMAX,JP)=U(1:LMAX)+.5D0*AUX(1:LMAX)  !.5 (Iplus + Iminus)

!  from construction (see above); here, we simply interpolate between
!  continuum quantities and line quantities at the border of influence
   DO L=1,LZ
     VBLUWI(L,JP)=.5D0*(SQRT(VAUX(L)*VAUX(L+1))-SQRT(AUX(L)*AUX(L+1)))
   ENDDO

ENDDO

RETURN
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FORMACON_PLUS(JP,LTOT,L,LL,DEL,XIPLUS,XIPCONT,ZP,OPAP,SCONP,DTAU,FSP)

USE nlte_type
USE nlte_dim
USE cmf_multi, ONLY: LTO

IMPLICIT NONE

INTEGER(I4B), PARAMETER :: ND=ID_NDEPT  

INTEGER(I4B) :: JP,LTOT,L,LL
REAL(DP) :: DEL, XIPLUS, XIPCONT
REAL(DP), DIMENSION (LTO) :: ZP, OPAP, SCONP
REAL(DP), DIMENSION (LTO-1) :: DTAU, FSP

INTEGER(I4B) :: I,II
REAL(DP), DIMENSION(L:LL+1) :: TAUCON
REAL(DP) :: ZLAST,OPALAST,SLAST,IPC,DT,E0,E1,W1,W2

IF(L.GE.LL+1) STOP ' ERROR IN FORMACON_PLUS'

! last, shorter step
OPALAST=OPAP(LL)*(1.D0-DEL)+OPAP(LL+1)*DEL
SLAST = SCONP(LL)*(1.D0-DEL)+SCONP(LL+1)*DEL
ZLAST = ZP(LL)*(1.D0-DEL)+ZP(LL+1)*DEL

IF(ZLAST.GT.ZP(LL).OR.ZLAST.LT.ZP(LL+1)) THEN
  PRINT*,JP,LL,ZLAST,ZP(LL),ZP(LL+1)
  STOP ' ERROR IN ZPLUS'
ENDIF  
    
TAUCON(L)=0.D0
DO I = L+1,LL  
     TAUCON(I) = TAUCON(I-1) + DTAU(I-1) ! note DTAU defined at "head"
END DO  
TAUCON(LL+1) = TAUCON(LL)+.5D0 * (OPAP(LL) + OPALAST)*(ZP(LL)-ZLAST)

IF(TAUCON(LL+1).GE.10.D0) THEN
! interfering continuum optically thick, use continuum value for consistency
  XIPLUS=XIPCONT
  RETURN
ENDIF

IPC=EXP(-TAUCON(LL+1))*XIPLUS !incident intensity


! first, shorter step
DT = TAUCON(LL+1) - TAUCON(LL)  

IF (DT.LT.1.D-8) THEN  
          W1 = .5D0*DT  
          W2 = .5D0*DT  
ELSE  
          E0 = EXP(-DT)  
          E1 = (1.D0-E0)/DT  
          W1 = 1.D0 - E1  
          W2 = E1 - E0  
END IF  
IPC = IPC + EXP(-TAUCON(LL))* (SCONP(LL)*W1+SLAST*W2)


DO I = LL-1,L,-1  
  IPC=IPC+EXP(-TAUCON(I))*FSP(I)     
ENDDO

! consistency check
!IF(TAUCON(LL+1).LT.1.D-3) THEN
!  IF(ABS(1.D0-IPC/XIPLUS).GT..02) STOP ' PROBLEMS IN FORMACON_PLUS'
!ENDIF

XIPLUS=IPC
RETURN
END
!
!
!-----------------------------------------------------------------------
!
SUBROUTINE FORMACON_MINUS(JP,LTOT,L,LL,DEL,XIMINUS,XIMCONT,ZP,OPAP,SCONP, &
&   DTAU,FSM)

USE nlte_type
USE nlte_dim
USE cmf_multi, ONLY: LTO

IMPLICIT NONE

INTEGER(I4B), PARAMETER :: ND=ID_NDEPT  

INTEGER(I4B) :: JP,LTOT,L,LL
REAL(DP) :: DEL, XIMINUS, XIMCONT
REAL(DP), DIMENSION (LTO) :: ZP, OPAP, SCONP
REAL(DP), DIMENSION (LTO-1) :: DTAU
REAL(DP), DIMENSION (2:ND) :: FSM

INTEGER(I4B) :: I,II
REAL(DP), DIMENSION(LL:L) :: TAUCON
REAL(DP) :: ZLAST,OPALAST,SLAST,IMC,DT,E0,E1,W1,W2

IF(L.LE.LL) STOP ' ERROR IN FORMACON_MINUS'

! last, shorter step
OPALAST=OPAP(LL)*(1.D0-DEL)+OPAP(LL+1)*DEL
SLAST = SCONP(LL)*(1.D0-DEL)+SCONP(LL+1)*DEL
ZLAST = ZP(LL)*(1.D0-DEL)+ZP(LL+1)*DEL

IF(ZLAST.GT.ZP(LL).OR.ZLAST.LT.ZP(LL+1)) THEN
  PRINT*,JP,LL,ZLAST,ZP(LL),ZP(LL+1)
  STOP ' ERROR IN ZMINUS'
ENDIF  
    
TAUCON(L)=0.D0
DO I = L-1,LL+1,-1  
     TAUCON(I) = TAUCON(I+1) + DTAU(I) ! note DTAU defined at "head"
END DO  
TAUCON(LL) = TAUCON(LL+1)+.5D0 * (OPAP(LL+1) + OPALAST)*(ZLAST-ZP(LL+1))

IF(TAUCON(LL).GE.10.D0) THEN
! interfering continuum optically thick, use continuum value for consistency
  XIMINUS=XIMCONT
  RETURN
ENDIF

IMC=EXP(-TAUCON(LL))*XIMINUS !incident intensity

! first, shorter step
DT = TAUCON(LL) - TAUCON(LL+1)  

IF (DT.LT.1.D-8) THEN  
          W1 = .5D0*DT  
          W2 = .5D0*DT  
ELSE  
          E0 = EXP(-DT)  
          E1 = (1.D0-E0)/DT  
          W1 = 1.D0 - E1  
          W2 = E1 - E0  
END IF  
IMC = IMC + EXP(-TAUCON(LL+1))* (SCONP(LL+1)*W1+SLAST*W2)

DO I = LL+2,L  
  IMC=IMC+EXP(-TAUCON(I))*FSM(I)     
ENDDO

! consistency check not performed, since I_minus oscillatory (and
! sometimes much smaller than scont, so that IMC different from
! XIMINUS even for low TAUCON(LL)

XIMINUS=IMC
RETURN
END
!
!-----------------------------------------------------------------------
!

SUBROUTINE CONT2(ND,NP,R,P,Z,SCONT,OPA,XIC1,XIC2,CORR,U)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!......................................................................
!
!     blue-wing intensities from pure formal solution,
!     only u and j are calculated
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,NP1=ID_NPOIN  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORR,XIC1,XIC2  
INTEGER(I4B) ::  ND,NP  
!     ..
!     .. array arguments ..
REAL(DP) ::  OPA(ND1),P(NP1),R(ND1),SCONT(ND1),U(ND1,NP1-1),Z(ND1,NP1)
!     ..
!     .. local scalars ..
REAL(DP) ::  PI2,THOM1  
INTEGER(I4B) ::  JP,L,LMAX  
LOGICAL CORE  
!     ..
!     .. local arrays ..
REAL(DP) ::  AKP(ND1),TA(ND1),TAUM(NP1),TB(ND1),TC(ND1)  
!     ..
!     .. external subroutines ..
EXTERNAL INVTRI,SETUP2  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ACOS,ATAN,MIN0  
!     ..

PI2 = ACOS(0.D0)  
THOM1 = 1.D0  

JPLOOP: DO JP = 1,NP - 1  
!
!---  implicit assumption: thomson(1):=1
!
     IF (JP.EQ.1) THEN  
          TAUM(1) = OPA(1)*R(1)* (1.D0/3.D0+2.D0*THOM1/3.D0)  
     ELSE  
       TAUM(JP) = OPA(1)*R(1)*R(1)* (THOM1/P(JP)* (PI2- ATAN(Z(1,JP)/P(JP)))+ &
&          (1.D0-THOM1)*R(1)*R(1)/2.D0/P(JP)**3* &
           (PI2-ATAN(Z(1,JP)/P(JP))-Z(1, JP)*P(JP)/R(1)/R(1)))
     END IF  

     IF (TAUM(JP).LT.0.D0) STOP 'TAUM NEGATIVE!'  

     LMAX = MIN0(NP+1-JP,ND)  
     CORE = (LMAX.EQ.ND)  

     CALL SETUP2(JP,LMAX,CORE,Z(:,JP),SCONT,OPA,XIC1,XIC2,CORR,AKP, &
      TA, TB,TC,TAUM(JP))

     CALL INVTRI(TA,TB,TC,AKP,LMAX)  

     DO L = 1,LMAX  
          U(L,JP) = AKP(L)  
     END DO

END DO JPLOOP  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE SETUP2(JP,LMAX,CORE,Z,SCONT,OPA,XIC1,XIC2,CORR,AKP,TA,TB, &
&                  TC,TAUM)

USE nlte_type
USE nlte_dim

USE nlte_opt, ONLY: IOV_ONLY
USE cmf_multi, ONLY: IPLUS_CONT, IMINUS_CONT



IMPLICIT NONE
!
!     sets up matrix elements for subroutine cont2
!     note: ta,tc negative compared to setup1
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  CORR,TAUM,XIC1,XIC2  
INTEGER(I4B) ::  JP,LMAX  
LOGICAL CORE  
!     ..
!     .. array arguments ..
REAL(DP) ::  AKP(ND1),OPA(ND1),SCONT(ND1),TA(ND1),TB(ND1), &
&                 TC(ND1),Z(ND1)
!     ..
!     .. local scalars ..
REAL(DP) ::  AK,AKDZ,BB,CC,DT0,DTM,DTP,DZ,E0,E1  
INTEGER(I4B) ::  L,LZ  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP  
!     ..

LZ = LMAX - 1  
!
!     outer boundary, 3rd order, corrected for i-minus
!
AK = .5D0* (OPA(1)+OPA(2))  
DZ = Z(1) - Z(2)  
AKDZ = .5D0*AK*DZ  
DTP = 1.D0/AK/DZ  
BB = 1.D0/DTP/3.D0  
CC = .5D0*BB  

IF (TAUM.LT.100.D0) THEN  
     E0 = EXP(-TAUM)  
     E1 = 1.D0 - E0  
ELSE  
     E1 = 1.D0  
END IF  

IF (AKDZ.LT..5D0) THEN  
     AKP(1) = -SCONT(1)* (BB+E1) - SCONT(2)*CC  
     TB(1) = -1.D0 - DTP - BB  
     TC(1) = -DTP + CC  
ELSE  
     AKP(1) = -SCONT(1)*E1  
     TB(1) = -1.D0 - DTP  
     TC(1) = -DTP  
END IF  

IF(.NOT.IOV_ONLY) IMINUS_CONT(JP)=SCONT(1)*E1
!
!     non boundary points
!
DO L = 2,LZ  
     DTM = DTP  
     AK = .5D0* (OPA(L)+OPA(L+1))  
     DZ = Z(L) - Z(L+1)  
     DTP = 1.D0/AK/DZ  
     DT0 = 2.D0/ (1.D0/DTM+1.D0/DTP)  
     AKP(L) = -SCONT(L)  
     TA(L) = -DT0*DTM  
     TC(L) = -DT0*DTP  
     TB(L) = -DT0* (DTM+DTP) - 1.D0  
END DO  

L = LMAX  
!
!     inner boundary,2nd order
!
IF (CORE) THEN  
     TA(L) = -DTP  
     AKP(L) = -XIC1 - Z(L)*XIC2/OPA(L)*CORR  
     TB(L) = -DTP - 1.D0  
     IF(.NOT.IOV_ONLY) IPLUS_CONT(JP)=-AKP(L)
ELSE  
     AKP(L) = -SCONT(L)  
     TA(L) = -2.D0*DTP*DTP  
     TB(L) = -2.D0*DTP*DTP - 1.D0  
END IF  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE FGRID(NF,ND,PHI,PWEIGHT,DELTAX,VDOP,VMAX,TEFF,T,XMAX0)  

USE nlte_type
USE nlte_dim
USE fund_const, ONLY: WPI=>SQPI
USE nlte_var, ONLY: VTURB
IMPLICIT NONE
!
!
!     calculates frequency scale,lineprofile phi and
!     renormalized integration weights pweight
!
!
!     ..
!     .. scalar arguments ..
REAL(DP) ::  DELTAX,TEFF,VDOP,VMAX,XMAX0  
INTEGER(I4B) ::  ND,NF  
!     ..
!     .. array arguments ..
REAL(DP) ::  PHI(NF,ND),PWEIGHT(NF,ND),T(ND)  

!     .. local scalars ..
REAL(DP) ::  DEL,DEL1,VDOPL,WS,XK,XMAX  
INTEGER(I4B) ::  K,L  
!     ..
!
! print*,' xmax0 = ',xmax0
!
DEL1 = VDOP/VMAX  
XMAX = XMAX0*DEL1  
DELTAX = 2.D0*XMAX/DBLE(NF-1)  

DO L = 1,ND  
     VDOPL = (VDOP**2-VTURB**2)*T(L)/TEFF  
     VDOPL=SQRT(VDOPL+VTURB**2)
     DEL = VMAX/VDOPL  
     WS = .0D0  
     DO K = 1,NF  
          XK = XMAX - (K-1)*DELTAX  
          PWEIGHT(K,L) = EXP(- ((XK*DEL)**2))  
          WS = WS + PWEIGHT(K,L)  
          PHI(K,L) = PWEIGHT(K,L)*DEL/WPI  
     END DO
!
!     renormalization
!
     DO K = 1,NF  
          PWEIGHT(K,L) = PWEIGHT(K,L)/WS  
     END DO

END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE RAY(SING,II,NOV, &
&              JP,Z,R,ND,NP,U,V,SLINE,PHI,PWEIGHT,DELTAX,OPA,ETA,&
&              OPAL,AIC,DBDR,CORRFC,W0,W0LAST,AJLMEAN,ALO,TAUZ1, &
&              TAU1,PP,XLAMBDA)

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: AAK1=>AK1,AAZ1=>AZ1,SLINEK  
USE cmf_multi, ONLY: INFOTAB, USAVE, VSAVE, IPLUS, IMINUS

IMPLICIT NONE
!
!
!     line radiation transfer in the comoving frame from a
!     given source function for a given impact parameter jp.
!     the integration is carried out in space and freq. to
!     yield the mean line intensity
!
!     new version: inlinen aller algebraischen routinen
!
!     j-bar
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,NF=ID_NFCMF  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  AIC,CORRFC,DBDR,DELTAX,W0LAST,XLAMBDA  
INTEGER(I4B) ::  II,NOV,JP,ND,NP  
!     ..
!     .. array arguments ..
REAL(DP) ::  AJLMEAN(ND),ALO(ND),OPA(ND),ETA(ND),OPAL(ND), &
&                 PHI(NF,ND),PP(ND1),PWEIGHT(NF,ND),R(ND), &
&                 SLINE(ND),TAU1(ND),TAUZ1(ND),U(ND),V(ND-1), &
&                 W0(ND),Z(ND)

LOGICAL SING
!     ..
!     .. local scalars ..
REAL(DP) ::  A,AK1,ALO1,AZ1,B,DAZ,DAZM,DBZ,DBZM,DT,DTZ,DTZM, &
&                 DX,DXZ,DXZ1,DXZM,DXZM1,ERFCK,ERFCKO,FF1,H1,HELP, &
&                 PW,RMAX,S1,SC,SL,TAUC,TAUS,TAUZ,TB1,W
INTEGER(I4B) ::  I,K,L,L1,LMAX,LZ,NI  
!     ..
!     .. local arrays ..
REAL(DP) ::  DDIA(0:ND1),EDIA(ND1+1),FF(ND1),GA(ND1),H(ND1), &
&                 QQ(ND1),S(ND1),TA(ND1),TB(ND1),TC(ND1),UB(ND1), &
&                 UOLD(ND1),V1OLD(ND1),VA(ND1),VB(ND1),VOLD(ND1)
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP,MAX,MIN0  
!     ..

LMAX = MIN0(NP+1-JP,ND)  
LZ = LMAX - 1  
RMAX = R(1)  

ERFCKO = PWEIGHT(1,1)  

UOLD(1:LMAX) = 0.D0  
VOLD(1:LZ) = 0.D0  
V1OLD(1:LZ) = 0.D0  

! until further evidence, use optically thick cont. without line background
! in case of multi-line transport

TAUC = OPA(1)*RMAX/3.D0  
SC = 0.D0  
ALO1 = 0.D0  

IF (TAUC.GE.1.D0) SC = ETA(1)/OPA(1)  

TAUS = OPAL(1)/ (PP(1)*DELTAX)  
SL = 0.D0  

!
!     loop for all original frequency points
!

KLOOP: DO K = 1,NF  

     IF (K.EQ.1) GO TO 170  

     ERFCK = ERFCKO + PWEIGHT(K,1)  
!
!         if(k.eq.nf.and.abs(erfck-1.d0).gt.1.d-10)
!     *   stop 'error in cmf: erfck'
!
!***  inlined subroutine cmfset
!----------------------------------------------------------------------
!
!***  outer boundary condition  -  2nd order
!
     AK1 = AAK1(K,1)  
     AZ1 = AAZ1(K,1)  
     DX = PP(1)*AK1  
!
!     approximate treatment for optically thick continuum
!     (rho^2 processes)
!     respectively optically thick line at outer boundary
!
!
     FF1=0.D0
     IF (TAUS.GE..5D0) THEN  
          FF1 = 1.D0 - EXP(-TAUS*ERFCK)  
          SL = SLINE(1)*FF1  
!
!      slp=sline(1)/(taus*deltax)*(exp(-taus*erfck)-exp(-taus*erfcko))
!      sl=sl-slp
!
     END IF  

     S(1) = MAX(SL,SC)  
     TC(1) = TAUZ1(1)*AK1  
     TB(1) = TC(1) + DX + 1.D0  
     UB(1) = DX  
     VB(1) = 0.D0  
     DTZM = TAUZ1(1)*AZ1  
!
!***  alo for outer boundary
!
     FF(1) = FF1  
     IF (S(1).EQ.SC) FF(1) = 0.D0  
!
!***  for g and h, the matrix element s are not different from
!***  inner points
!
     DXZM = .5D0* (PP(1)+PP(2))*AZ1  
     DXZM1 = 1.D0/ (1.D0+DXZM)  
     DAZM = DTZM*DXZM1  
     DBZM = DXZM*DXZM1  
     GA(1) = -DAZM  
     H(1) = DBZM  
!
!***  non-boundary points
!
     DO L = 2,LZ  
          AK1 = AAK1(K,L)  
          AZ1 = AAZ1(K,L)  
          S(L) = SLINEK(K,L)  
          DT = TAU1(L)*AK1  
          DTZ = TAUZ1(L)*AZ1  
          DX = PP(L)*AK1  
          DXZ = .5D0* (PP(L)+PP(L+1))*AZ1  
          DXZ1 = 1.D0/ (1.D0+DXZ)  
          DAZ = DTZ*DXZ1  
          DBZ = DXZ*DXZ1  
          TA(L) = DT*DAZM  
          TC(L) = DT*DAZ  
          TB(L) = TA(L) + TC(L) + DX + 1.D0  
          FF(L) = PHI(K,L)*OPAL(L)*AK1  
          UB(L) = DX  
          VA(L) = -DT*DBZM  
          VB(L) = DT*DBZ  
          GA(L) = -DAZ  
          H(L) = DBZ  
          DAZM = DAZ  
          DBZM = DBZ  
     END DO

     L = LMAX  

     IF (LMAX.EQ.ND) THEN  
!
!***  inner boundary condition (core rays)  -  only to first order
!***  diffusion-approximation
!
        AK1 = AAK1(K,L)  
        S(L) = AIC + Z(L)*AK1*DBDR*CORRFC  
        DX = PP(L)*AK1  
        DT = TAUZ1(L-1)*AK1  
        TA(L) = DT  
        TB(L) = DT + DX + 1.D0  
        FF(L) = 0.D0  
        UB(L) = DX  
        VA(L) = 0.0D0  
        VB(L) = 0.0D0  
     ELSE  
!
!***  inner boundary condition (non-core rays)  -  second order
!
        AK1 = AAK1(K,L)  
        S(L) = SLINEK(K,L)  
        TAUZ = Z(LZ)   
        DT = AK1/TAUZ  
        DX = PP(L)*AK1  
        TA(L) = 2.D0*DT*DAZM !instead of daz  
        TB(L) = TA(L) + DX + 1.D0  
        UB(L) = DX  
        VA(L) = -2.D0*DT*DBZM !instead of dbz
        FF(L) = PHI(K,L)*OPAL(L)*AK1  
      ENDIF
!      
!-----------------------------------------------------------------------
!***  continuation of subroutine ray
!
     ERFCKO = ERFCK  

     QQ(1) = 0.D0  

     DO L = 2,LZ  
          QQ(L) = VA(L)*V1OLD(L-1) + VB(L)*VOLD(L)  
     END DO

     QQ(LMAX) = VA(LMAX)*V1OLD(LZ)  
!
!     uold=ub * uold + qq + ff
!
     DO L = 1,LMAX  
          UOLD(L) = UB(L)*UOLD(L) + QQ(L) + FF(L)  
     END DO
!
!***     now, uold is the solution vector for the inhomgeneous
!***     equation corresponding to u(sl=1)-u(sl=0)
!
!     inlinen von diag mit option 1
!
!
     DDIA(0) = 0.D0  
     TA(1) = 0.D0
     TC(LMAX) = 0.D0

     DO L = 1,LMAX  
          DDIA(L) = TC(L)/ (TB(L)-TA(L)*DDIA(L-1))  
     END DO

     EDIA(LMAX+1) = 0.D0  

     DO L = LMAX,1,-1  
          EDIA(L) = TA(L)/ (TB(L)-TC(L)*EDIA(L+1))  
     END DO

     DO L = 1,LMAX  
          UOLD(L) = UOLD(L)/ ((1.D0-DDIA(L)*EDIA(L+1))* &
&          (TB(L)-TA(L)*DDIA(L-1)))
          UOLD(L) = MAX(UOLD(L),0.D0)  
     END DO
!
!***  now,uold is the solution corresponding to u(sl=1)-u(0)
!
     DO L = 2,LZ  
          L1 = L - 1  
          V1OLD(L1) = -GA(L1)*UOLD(L) + H(L1)*V1OLD(L1)  
          VOLD(L) = GA(L)*UOLD(L) + H(L)*VOLD(L)  
     END DO

     L = LZ  
     V1OLD(L) = -GA(L)*UOLD(LMAX) + H(L)*V1OLD(L)  
!
!***  now, v1old, vold are the v's corresponding to delta u
!
!     inlinen von vmalv
!
     B = V(1)  
     QQ(1) = VB(1)*B  


     DO L = 2,LZ  
          A = B  
          B = V(L)  
          QQ(L) = VA(L)*A + VB(L)*B  
     END DO

     QQ(LMAX) = VA(LMAX)*B  
!
!     u = ub * u + qq + s
!
     DO L = 1,LMAX  
          U(L) = UB(L)*U(L) + QQ(L) + S(L)  
     END DO
!
!***     inlining of invtri
!
     TB1 = 1.D0/TB(1)  
     TC(1) = TC(1)*TB1  
     U(1) = U(1)*TB1  

     DO L = 2,LZ  
          HELP = TB(L) - TA(L)*TC(L-1)  
          H1 = 1.D0/HELP  
          TC(L) = TC(L)*H1  
          U(L) = (U(L)+U(L-1)*TA(L))*H1  
     END DO

     U(LMAX) = (U(LMAX)+U(LZ)*TA(LMAX))/ (TB(LMAX)-TC(LZ)*TA(LMAX))

     DO L = 1,LZ  
          NI = LMAX - L  
          U(NI) = U(NI) + TC(NI)*U(NI+1)  
     END DO

     DO L = 1,LMAX  
          UOLD(L) = MAX(UOLD(L),0.D0)  
!
!     i don't like the following statement at all, but as one says
!     in germany:  man hat schon pferde kotzen sehen.
!     actually, i found one case where due to a very strange behaviour
!     of occupation numbers v became strongly negative and in the
!     course also qq, such that u became negative. once u is negative,
!     this can no longer be compensated. hence:
!
          U(L) = MAX(U(L),0.D0)  
     END DO
!
!     now u is the new field at index k
!
!     inlinen von gmalu und
!     v = h * v + gmalu
!
     B = U(1)  
     DO L = 1,LZ  
          A = B  
          B = U(L+1)  
          V(L) = H(L)*V(L) + GA(L)* (A-B)  
     END DO
!
!     now v is the new field at index k
!
!     adding the new u to ajlmean and to the 0.and 2nd moments aj,ak
!

     U(LMAX+1:ND)=0.D0
     V(LZ+1:ND-1)=0.D0

  170      CONTINUE  

     DO L = 1,LMAX  
          PW = PWEIGHT(K,L)*W0(L)  
          AJLMEAN(L) = AJLMEAN(L) + U(L)*PW  
          ALO(L) = ALO(L) + UOLD(L)*PW  
     END DO

IF(SING) CYCLE KLOOP

! store u,v if necessary

DO I=1,NOV
  
  IF(INFOTAB(I)%INDBW .EQ. II) THEN
     IF(INT(INFOTAB(I)%IF) .EQ. NF.AND. K.EQ.NF) THEN 
       USAVE(1:LMAX,JP,I)=U(1:LMAX)
       VSAVE(1:LZ,JP,I)=V(1:LZ)
       USAVE(ND,NP-1,I)=NF
       IMINUS(JP,I) = S(1)
       IF(LMAX.EQ.ND) IPLUS(JP,I)=S(ND)
!       IF(JP.EQ.1) PRINT*,II,'NF STORED',INFOTAB(I)%IF,' FOR L<INE ',I,' ',NF
     ELSE IF (INT(INFOTAB(I)%IF) .EQ. K) THEN
       W=K+1.D0-INFOTAB(I)%IF
       USAVE(1:LMAX,JP,I)=W*U(1:LMAX)
       VSAVE(1:LZ,JP,I)=W*V(1:LZ)
       USAVE(ND,NP-1,I)=W
!       IF(JP.EQ.1) PRINT*,II,K,' STORED (FIRST)',INFOTAB(I)%IF,' FOR LINE ',I,' ',W
     ELSE IF (INT(INFOTAB(I)%IF) .EQ. K-1) THEN
       W=INFOTAB(I)%IF+1.D0-K
       USAVE(1:LMAX,JP,I)=USAVE(1:LMAX,JP,I)+W*U(1:LMAX)
       VSAVE(1:LZ,JP,I)=VSAVE(1:LZ,JP,I)+W*V(1:LZ)
       VSAVE(ND-1,NP-1,I)=W
!       IF(JP.EQ.1) PRINT*,II,K,' STORED (2ND)',INFOTAB(I)%IF,' FOR LINE ',I,' ',W
     ENDIF 
  ENDIF

ENDDO


END DO KLOOP 

IF (JP.NE.NP-1) RETURN  
!
!***  final value for p = rmax and z=0 in optically thick case!!
!
TAUS = OPAL(1)/ (PP(1)*DELTAX)  

IF (SC.NE.0.D0 .OR. TAUS.GE..5D0) THEN  
     PRINT *,' OPTICALLY THICK OUTER BOUNDARY AT',XLAMBDA  
     ERFCKO = PWEIGHT(1,1)  

     DO K = 2,NF  
          ERFCK = ERFCKO + PWEIGHT(K,1)  
!
!      if(k.eq.nf.and.abs(erfck-1.d0).gt.1.d-10)
!     *stop 'error in cmf: erfck'
!
          ALO1 = 1.D0 - EXP(-TAUS*ERFCK)  
          SL = SLINE(1)*ALO1  
!
!      slp=sline(1)/(taus*deltax)*(exp(-taus*erfck)-exp(-taus*erfcko))
!      sl=sl-slp
!
          S1 = MAX(SL,SC)  
          PW = PWEIGHT(K,1)*W0LAST  
          AJLMEAN(1) = AJLMEAN(1) + S1*PW  
          IF (S1.EQ.SL) ALO(1) = ALO(1) + ALO1*PW  
          ERFCKO = ERFCK  
     END DO
END IF  

!  if(xlambda.gt.303..and.xlambda.lt.304.) then
!    do l=1,nd
!      write(*,100) l,alo(l),ajlmean(l),ajlmean(l)/sline(l),eta(l)/opa(l),opa(l)
!    enddo
!  endif
!  if(xlambda.gt.256..and.xlambda.lt.257.) then
!    do l=1,nd
!      write(*,100) l,alo(l),ajlmean(l),ajlmean(l)/sline(l),eta(l)/opa(l),opa(l)
!    enddo
!  endif
!  if(xlambda.gt.243..and.xlambda.lt.244.) then
!    do l=1,nd
!      write(*,100) l,alo(l),ajlmean(l),ajlmean(l)/sline(l),eta(l)/opa(l),opa(l)
!    enddo
!  endif
!  if(xlambda.gt.237..and.xlambda.lt.238.) then
!    do l=1,nd
!      write(*,100) l,alo(l),ajlmean(l),ajlmean(l)/sline(l),eta(l)/opa(l),opa(l)
!    enddo
!  endif

!100 format(i3,2x,f12.6,2x,e12.6,2x,f10.4,2x,e12.6,2x,e10.4)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE RAYCONT(JP,Z,R,ND,NP,U,V, &
        DELTAX,OPA,ETA,AIC,DBDR,CORRFC,W0, &
        TAUZ1,TAU1,PP)

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: AAK1=>AK1,AAZ1=>AZ1,SLINEK  

IMPLICIT NONE
!
!
!     line radiation transfer in the comoving frame from a
!     given source function for a given impact parameter jp.
!     the integration is carried out in space and freq. to
!     yield the mean line intensity
!
!     new version: inlinen aller algebraischen routinen
!
!     j-bar
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,NF=ID_NFCMF  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  AIC,CORRFC,DBDR,DELTAX  
INTEGER(I4B) ::  JP,ND,NP  
!     ..
!     .. array arguments ..
REAL(DP) ::  OPA(ND),ETA(ND), &
&            PP(ND1),PWEIGHT(NF,ND),R(ND), &
&            TAU1(ND),TAUZ1(ND),U(ND),V(ND-1), &
&            W0(ND),Z(ND)

!     ..
!     .. local scalars ..
REAL(DP) ::  A,AK1,AZ1,B,DAZ,DAZM,DBZ,DBZM,DT,DTZ,DTZM, &
&                 DX,DXZ,DXZ1,DXZM,DXZM1,ERFCK,ERFCKO,FF1,H1,HELP, &
&                 PW,RMAX,S1,SC,SL,TAUC,TAUS,TAUZ,TB1,W
INTEGER(I4B) ::  I,K,L,L1,LMAX,LZ,NI  
!     ..
!     .. local arrays ..
REAL(DP) ::       GA(ND1),H(ND1), &
&                 QQ(ND1),S(ND1),TA(ND1),TB(ND1),TC(ND1),UB(ND1), &
&                 VA(ND1),VB(ND1)
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP,MAX,MIN0  
!     ..

LMAX = MIN0(NP+1-JP,ND)  
LZ = LMAX - 1  
RMAX = R(1)  

! until further evidence, use optically thick cont. without line background
! in case of multi-line transport

TAUC = OPA(1)*RMAX/3.D0  
SC = 0.D0  

IF (TAUC.GE.1.D0) SC = ETA(1)/OPA(1)  

KLOOP: DO K = 2,NF  
!
!***  outer boundary condition  -  2nd order
!
     AK1 = AAK1(K,1)  
     AZ1 = AAZ1(K,1)  
     DX = PP(1)*AK1  
!
!     approximate treatment for optically thick continuum
!     (rho^2 processes)
!     respectively optically thick line at outer boundary
!
     S(1) = SC  
     TC(1) = TAUZ1(1)*AK1  
     TB(1) = TC(1) + DX + 1.D0  
     UB(1) = DX  
     VB(1) = 0.D0  
     DTZM = TAUZ1(1)*AZ1  
!
!***  for g and h, the matrix element s are not different from
!***  inner points
!
     DXZM = .5D0* (PP(1)+PP(2))*AZ1  
     DXZM1 = 1.D0/ (1.D0+DXZM)  
     DAZM = DTZM*DXZM1  
     DBZM = DXZM*DXZM1  
     GA(1) = -DAZM  
     H(1) = DBZM  
!
!***  non-boundary points
!
     DO L = 2,LZ  
          AK1 = AAK1(K,L)  
          AZ1 = AAZ1(K,L)  
          S(L) = SLINEK(K,L)  
          DT = TAU1(L)*AK1  
          DTZ = TAUZ1(L)*AZ1  
          DX = PP(L)*AK1  
          DXZ = .5D0* (PP(L)+PP(L+1))*AZ1  
          DXZ1 = 1.D0/ (1.D0+DXZ)  
          DAZ = DTZ*DXZ1  
          DBZ = DXZ*DXZ1  
          TA(L) = DT*DAZM  
          TC(L) = DT*DAZ  
          TB(L) = TA(L) + TC(L) + DX + 1.D0  
          UB(L) = DX  
          VA(L) = -DT*DBZM  
          VB(L) = DT*DBZ  
          GA(L) = -DAZ  
          H(L) = DBZ  
          DAZM = DAZ  
          DBZM = DBZ  
     END DO

     L = LMAX  

     IF (LMAX.EQ.ND) THEN  
!
!***  inner boundary condition (core rays)  -  only to first order
!***  diffusion-approximation
!
        AK1 = AAK1(K,L)  
        S(L) = AIC + Z(L)*AK1*DBDR*CORRFC  
        DX = PP(L)*AK1  
        DT = TAUZ1(L-1)*AK1  
        TA(L) = DT  
        TB(L) = DT + DX + 1.D0  
        UB(L) = DX  
        VA(L) = 0.0D0  
        VB(L) = 0.0D0  
     ELSE  
!
!***  inner boundary condition (non-core rays)  -  second order
!
        AK1 = AAK1(K,L)  
        S(L) = SLINEK(K,L)  
        TAUZ = Z(LZ)   
        DT = AK1/TAUZ  
        DX = PP(L)*AK1  
        TA(L) = 2.D0*DT*DAZM !instead of daz  
        TB(L) = TA(L) + DX + 1.D0  
        UB(L) = DX  
        VA(L) = -2.D0*DT*DBZM !instead of dbz
      ENDIF
!      
!-----------------------------------------------------------------------
!***  continuation of subroutine ray
!
!
!     inlinen von vmalv
!
     B = V(1)  
     QQ(1) = VB(1)*B  


     DO L = 2,LZ  
          A = B  
          B = V(L)  
          QQ(L) = VA(L)*A + VB(L)*B  
     END DO

     QQ(LMAX) = VA(LMAX)*B  
!
!     u = ub * u + qq + s
!
     DO L = 1,LMAX  
          U(L) = UB(L)*U(L) + QQ(L) + S(L)  
     END DO
!
!***     inlining of invtri
!
     TB1 = 1.D0/TB(1)  
     TC(1) = TC(1)*TB1  
     U(1) = U(1)*TB1  

     DO L = 2,LZ  
          HELP = TB(L) - TA(L)*TC(L-1)  
          H1 = 1.D0/HELP  
          TC(L) = TC(L)*H1  
          U(L) = (U(L)+U(L-1)*TA(L))*H1  
     END DO

     U(LMAX) = (U(LMAX)+U(LZ)*TA(LMAX))/ (TB(LMAX)-TC(LZ)*TA(LMAX))

     DO L = 1,LZ  
          NI = LMAX - L  
          U(NI) = U(NI) + TC(NI)*U(NI+1)  
     END DO

     DO L = 1,LMAX  
!
!     i don't like the following statement at all, but as one says
!     in germany:  man hat schon pferde kotzen sehen.
!     actually, i found one case where due to a very strange behaviour
!     of occupation numbers v became strongly negative and in the
!     course also qq, such that u became negative. once u is negative,
!     this can no longer be compensated. hence:
!
          U(L) = MAX(U(L),0.D0)  
     END DO
!
!     now u is the new field at index k
!
!     inlinen von gmalu und
!     v = h * v + gmalu
!
     B = U(1)  
     DO L = 1,LZ  
          A = B  
          B = U(L+1)  
          V(L) = H(L)*V(L) + GA(L)* (A-B)  
     END DO
!
!     now v is the new field at index k
!
!     adding the new u to ajlmean and to the 0.and 2nd moments aj,ak
!

     U(LMAX+1:ND)=0.D0
     V(LZ+1:ND-1)=0.D0
!     print*,jp,k,U
!     print*,jp,k,v

ENDDO KLOOP 

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE RAY_INV(SING,II,NOV, &
&               JP,Z,R,ND,NP,U,V,SLINE,PHI,PWEIGHT,DELTAX,OPA,ETA, &
&               OPAL,AIC,DBDR,CORRFC,W0,W0LAST,AJLMEAN,TAUZ1, &
&               TAU1,PP)

USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: AAK1=>AK1,AAZ1=>AZ1,SLINEK  
USE cmf_multi, ONLY: INFOTAB, USAVE, VSAVE, IPLUS, IMINUS

IMPLICIT NONE
!
!
!     line radiation transfer in the comoving frame from a
!     given source function for a given impact parameter jp.
!     the integration is carried out in space and freq. to
!     yield the mean line intensity, all for inverted transitions!
!
!     NOTE: here, we solve over z, which has been shown to
!           be stable (if implicit scheme used) by
!           Mihalas, Kunasz and Hummer, 1975, and has
!           been checked carefully by R. Venero (March 2001)
!
!
!     NOTE ALSO: no ALO calculated 
!     j-bar
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,NF=ID_NFCMF  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  AIC,CORRFC,DBDR,DELTAX,W0LAST  
INTEGER(I4B) ::  II,NOV,JP,ND,NP  
!     ..
!     .. array arguments ..
REAL(DP) ::  AJLMEAN(ND),ETA(ND),OPA(ND),OPAL(ND), &
&                 PHI(NF,ND),PP(ND1),PWEIGHT(NF,ND),R(ND), &
&                 SLINE(ND),TAU1(ND),TAUZ1(ND),U(ND),V(ND-1), &
&                 W0(ND),Z(ND)

LOGICAL SING
!     ..
!     .. local scalars ..
REAL(DP) ::  A,AK1,AZ1,B,DAZ,DAZM,DBZ,DBZM,DT,DTZ,DTZM, &
&                 DX,DXZ,DXZ1,DXZM,DXZM1,H1,HELP, &
&                 PW,RMAX,SC,TAUC,TAUS,TAUZ,TB1,W
INTEGER(I4B) ::  I,K,L,LMAX,LZ,NI  
!     ..
!     .. local arrays ..
REAL(DP) ::       GA(ND1),H(ND1), QQ(ND1),S(ND1),TA(ND1),TB(ND1),TC(ND1), &
&                 UB(ND1),VA(ND1),VB(ND1)
!     ..
!     .. intrinsic functions ..
!INTRINSIC MIN0  
!     ..


LMAX = MIN0(NP+1-JP,ND)  
LZ = LMAX - 1  
RMAX = R(1)  

SC = 0.D0  
TAUC = OPA(1)*RMAX/3.D0  
IF (TAUC.GE.1.D0) SC = ETA(1)  

TAUS = OPAL(1)/ (PP(1)*DELTAX)  
IF(TAUS.LT.-0.5) PRINT*,' WARNING!!!!: optically thick INVERTED line!!!'

!
!     loop for all original frequency points
!
KLOOP: DO K = 1,NF  

     IF (K.EQ.1) GO TO 170  

!
!***  inlined subroutine cmfset for inverted levels
!
!
!***  outer boundary condition  -  2nd order
!
     AK1 = AAK1(K,1)  
     AZ1 = AAZ1(K,1)  
     DX = PP(1) 
     S(1)=SC
     TC(1) = TAUZ1(1)  
     TB(1) = TC(1) + DX + AK1  
     UB(1) = DX  
     VB(1) = 0.D0  
     DTZM = TAUZ1(1)       
!
!***  for g and h, the matrix element s are not different from
!***  inner points
!
     DXZM = .5D0* (PP(1)+PP(2))  
     DXZM1 = 1.D0/ (AZ1+DXZM)  
     DAZM = DTZM*DXZM1  
     DBZM = DXZM*DXZM1  
     GA(1) = -DAZM  
     H(1) = DBZM  
!
!***  non-boundary points
!
     DO L = 2,LZ  
          AK1 = AAK1(K,L)  
          AZ1 = AAZ1(K,L)  
!r for inverted levels S(L) is just the emisivity
          S(L) = SLINEK(K,L)
          DT = TAU1(L)  
          DTZ = TAUZ1(L)  
          DX = PP(L)  
          DXZ = .5D0* (PP(L)+PP(L+1))  
          DXZ1 = 1.D0/ (AZ1+DXZ)  
          DAZ = DTZ*DXZ1  
          DBZ = DXZ*DXZ1  
          TA(L) = DT*DAZM  
          TC(L) = DT*DAZ  
          TB(L) = TA(L) + TC(L) + DX + AK1  
          UB(L) = DX  
          VA(L) = -DT*DBZM  
          VB(L) = DT*DBZ  
          GA(L) = -DAZ  
          H(L) = DBZ  
          DAZM = DAZ  
          DBZM = DBZ  
     END DO

     L = LMAX  

     IF (LMAX.EQ.ND) THEN  
!
!***  inner boundary condition (core rays)  -  only to first order
!***  diffusion-approximation
!
        AK1 = AAK1(K,L)  
        S(L) = AIC*AK1 + Z(L)*DBDR*CORRFC  
        DX = PP(L)  
        DT = TAUZ1(L-1)  
        TA(L) = DT  
        TB(L) = DT + DX + AK1          
        UB(L) = DX  
        VA(L) = 0.0D0  
        VB(L) = 0.0D0  
     ELSE  
!
!***  inner boundary condition (non-core rays)  -  second order
!
        AK1 = AAK1(K,L)  
!r   same as above
        S(L) = SLINEK(K,L)
        TAUZ = Z(LZ)   
        DT = 1.D0/TAUZ  
        DX = PP(L)
        TA(L) = 2.D0*DT*DAZM
        TB(L) = TA(L) + DX + AK1  
        UB(L) = DX  
        VA(L) = -2.D0*DT*DBZM
      ENDIF
!
!-----------------------------------------------------------------------
!***  continuation of subroutine ray
!
!
!     inlinen von vmalv
!
     B = V(1)  
     QQ(1) = VB(1)*B  


     DO L = 2,LZ  
          A = B  
          B = V(L)  
          QQ(L) = VA(L)*A + VB(L)*B  
     END DO

     QQ(LMAX) = VA(LMAX)*B  
!
!     u = ub * u + qq + s
!
     DO L = 1,LMAX  
          U(L) = UB(L)*U(L) + QQ(L) + S(L)  
     END DO
!
!***     inlining of invtri
!
     TB1 = 1.D0/TB(1)  
     TC(1) = TC(1)*TB1  
     U(1) = U(1)*TB1  

     DO L = 2,LZ  
          HELP = TB(L) - TA(L)*TC(L-1)  
          H1 = 1.D0/HELP  
          TC(L) = TC(L)*H1  
          U(L) = (U(L)+U(L-1)*TA(L))*H1  
     END DO

     U(LMAX) = (U(LMAX)+U(LZ)*TA(LMAX))/ (TB(LMAX)-TC(LZ)*TA(LMAX))

     DO L = 1,LZ  
         NI = LMAX  - L  
          U(NI) = U(NI) + TC(NI)*U(NI+1)  
     END DO

     DO L = 1,LMAX  
!      see comment in subroutine Ray
       U(L) = MAX(U(L),0.D0)  
     END DO


     B = U(1)  
     DO L = 1,LZ  
          A = B  
          B = U(L+1)  
          V(L) = H(L)*V(L) + GA(L)* (A-B)  
     END DO
!
!     now v is the new field at index k
!
!     adding the new u to ajlmean and to the 0.and 2nd moments aj,ak
!
     U(LMAX+1:ND)=0.D0
     V(LZ+1:ND-1)=0.D0

  170      CONTINUE  

     DO L = 1,LMAX  
          PW = PWEIGHT(K,L)*W0(L)  
          AJLMEAN(L) = AJLMEAN(L) + U(L)*PW  
     END DO

IF(SING) CYCLE KLOOP

! store u,v if necessary

DO I=1,NOV

  IF(INFOTAB(I)%INDBW .EQ. II) THEN
     IF(INT(INFOTAB(I)%IF) .EQ. NF.AND. K.EQ.NF) THEN 
       USAVE(1:LMAX,JP,I)=U(1:LMAX)
       VSAVE(1:LZ,JP,I)=V(1:LZ)
       USAVE(ND,NP-1,I)=NF
       IMINUS(JP,I) = S(1)
       IF(LMAX.EQ.ND) IPLUS(JP,I)=S(ND)/AAK1(NF,ND)
!       IF(JP.EQ.1) PRINT*,II,'NF STORED',INFOTAB(I)%IF,' FOR LINE ',I,' ',NF
     ELSE IF (INT(INFOTAB(I)%IF) .EQ. K) THEN
       W=K+1.D0-INFOTAB(I)%IF
       USAVE(1:LMAX,JP,I)=W*U(1:LMAX)
       VSAVE(1:LZ,JP,I)=W*V(1:LZ)
       USAVE(ND,NP-1,I)=W
!       IF(JP.EQ.1) PRINT*,II,K,' STORED (FIRST)',INFOTAB(I)%IF,' FOR LINE ',I,' ',W
     ELSE IF (INT(INFOTAB(I)%IF) .EQ. K-1) THEN
       W=INFOTAB(I)%IF+1.D0-K
       USAVE(1:LMAX,JP,I)=USAVE(1:LMAX,JP,I)+W*U(1:LMAX)
       VSAVE(1:LZ,JP,I)=VSAVE(1:LZ,JP,I)+W*V(1:LZ)
       VSAVE(ND-1,NP-1,I)=W
!       IF(JP.EQ.1) PRINT*,II,K,' STORED (2ND)',INFOTAB(I)%IF,' FOR LINE ',I,' ',W
     ENDIF 
  ENDIF

ENDDO

END DO KLOOP 

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CMFPREP(LL,VELR,VGRAD,NFIR,NLAS,BLEVEL,XNEL,SR,VMAX, &
&                   NATO,XMUST,TEMPE,CLF,NFIRM,NLASM,NFI,NLA,STARTCMF, &
&                   OPTMIXED,IPRES,R,VELO, FIC,TCL_FAC_LINE,TCL_FAC_CONT)

! UPDATED FOR CLUMPING

USE nlte_type
USE nlte_dim
USE fund_const
USE princesa_var, ONLY: ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& DATA,NDAT, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
&       LABL,LABL4,LABL1,LABU1,LABL3,IPARE5

USE nlte_var, ONLY:MLOW,MUP,MMLOW,MMUP,MCLOW,MCUP, &
& INDXLAMC,XLAMCMF,VDOPCMF,BLUCMF,BULCMF,AULCMF, &
& LABLINELO, LABLINEUP, &
& OPACLIN, &
& TLUMAT,TULMAT,BETOLD,INDEXRBB,INDEXCMF,INDEXSA, &
& MODNAM, &
& VTHER, FRE, LTE_UPDATE  

IMPLICIT NONE
!
!--- calculates rbb-rates from old occupation numbers for sob. treatment
!--- prepares cmf calculations and decides which lines are treated in
!--- cmf. (optmixed = 0.: all lines; optmixed = 1., lines according to
!--- the criteria specified below.)
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: NNREC=ID_LLEVS,ND=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  OPTMIXED,SR,TEMPE,VELR,VGRAD,VMAX,XMUST,XNEL,CLF
REAL(DP) :: FIC, TCL_FAC_LINE, TCL_FAC_CONT 
INTEGER(I4B) ::  IPRES,LL,NATO,NFI,NFIR,NFIRM,NLA,NLAS,NLASM  
LOGICAL STARTCMF  
!     ..
!     .. array arguments ..
REAL(DP) ::  BLEVEL(NNREC),R(ID_NDEPT),VELO(ID_NDEPT)  
!     ..
!     .. local scalars ..
REAL(DP) ::  BLU,BUL,ER,FLU,TAUC,TLU,TUL,VCONT,VSOBO, &
&                 VTHMAX,XLAM,XLAMC,XNL,XNU,MAXLAM
REAL(DP) ::  TCL, FAC 
INTEGER(I4B) ::  I,ICOUNT,IFORM,II, &
&        INDCMF,INDI,INDMAT,INTEG,L,M,ML,MU,NC,NDATOS
LOGICAL QCL,QDB,QRT,START  
!     ..
!     .. local arrays ..
REAL(DP) ::  DL(ID_NLONG),FRECFN(ID_RBFTR),FRECIN(ID_RBFTR)  
INTEGER(I4B) ::  IFIRS(ID_IONES),IFIRSL(ID_IONES),IFIRX(ID_IONES), &
&        IFPTR(ID_NTTRD),INDEXOLD(ID_NTTRD),ISA0(ID_ATOMS), &
&        ISA1(ID_ATOMS),IXA0(ID_ATOMS),IXA1(ID_ATOMS),KL(ID_LLEVS), &
&        KLX(ID_XLEVS),LA0(ID_ATOMS),LA1(ID_ATOMS),LE(ID_LLEVS), &
&        LI(ID_LLEVS),LIS(ID_SLEVS),LIX(ID_XLEVS),NFPTR(ID_NTTRD), &
&        NIONS(ID_ATOMS),NS0(ID_SLEVS),NS1(ID_SLEVS)
!     ..
!     .. external subroutines ..
EXTERNAL TRATCMF,TRATRAD  
!     ..
!     .. intrinsic functions ..
!INTRINSIC INT,MAX  
!     ..
!
!---  for cmf-lines, we have the following
!---- correspondance (before having performed the cmf treatment)
!     tlumat -- opal
!     tulmat -- sline
!
!     .. data statements ..
DATA START/.TRUE./  
!     ..

! AFTER LTE-UPDATE, CREATE NEW CMF-LIST (CHANGE IN IMIN,IMAX)
IF(.NOT.START.AND.LTE_UPDATE.AND.NATO.EQ.1.AND.LL.EQ.1) START=.TRUE.

IF (START) THEN  
! THIS BLOCK MUST BE CREATED ONLY ONCE AT NATO=1 AND LL=1, &
! SINCE ALL LINES (FROM ALL ELEMENTS) ARE INITIALIZED IN THE I-LOOP' 
     OPEN (1,ERR=10,FILE=TRIM(MODNAM)//'/INDEXCMF',STATUS='OLD')  
     REWIND 1  
     READ (1,FMT=*) (INDEXOLD(II),II=1,ID_NTTRD)  
     CLOSE (1)  
     IPRES = 1  
     GO TO 20  

   10      CONTINUE  

     IPRES = 2  

   20      CONTINUE  
     ICOUNT = 0  
     PRINT*,IPRES
     DO I = 1,INDEX1  
          INDI = INDAT1(I)  
          ER = DATA(INDI)  
          M = INT(ER)  

          IF (M.EQ.1) THEN  
               ICOUNT = ICOUNT + 1  
               INDEXRBB(I) = ICOUNT  
!
!---  at first, all lines are assumed as sobolev ones
!
               INDEXCMF(ICOUNT) = 1  
          ELSE IF (M.EQ.2) THEN  
               INDEXRBB(I) = 0  
          ELSE  
               STOP ' ERROR IN RBB OR CBB'  
          END IF  

     END DO

     IF (ICOUNT.NE.ID_NTTRD) STOP ' NOT ALL RBB TRANSITIONS FOUND'  
     START = .FALSE.  
END IF  
!
!---  now, according to the specified criteria, lines are changed from
!---  sobolev to cmf treatment
!---  only at first depth point, sucessively for all elements
!
IF (STARTCMF) THEN  
!
!***  this is the border velocity we require for the mixed treatment.
!
     VTHMAX = 0.2D0*VTHER(NATO)/VMAX  
     MAXLAM=1.D8/FRE(1)

ILOOP: DO I = 1,INDEX1  
          ML = MMLOW(I)  
          MU = MMUP(I)  
!
!---  note: since all ions are to be inspected,
!     we have             nfi   and nla   (depth indep., maximum set)
!     different from      nfir  and nlas  (actual, depth dependent)
!     and                 nfirm and nlasm (depth indep., minimum set)
!
          IF (ML.LT.NFI .OR. ML.GE.NLA) CYCLE  
          IF (MU.GT.NLA) CYCLE
          INDMAT = INDEXRBB(I)  
          IF (INDMAT.EQ.0) CYCLE  

          IF (INDEXCMF(INDMAT).NE.1) then
            PRINT*,STARTCMF,START,LTE_UPDATE,IPRES
            PRINT*,I,ML,MU,INDMAT,INDEXCMF(INDMAT)
            PRINT*,NFI,NLA
            STOP ' ERROR IN INDCMF-PHILOSOPHY'
          ENDIF
          INDI = INDAT1(I)  
          XLAM = DATA(INDI+1)
          XLAMCMF(INDMAT) = XLAM  
          LABLINELO(INDMAT) = LABL(ML)
          LABLINEUP(INDMAT) = LABL(MU)
!
!***  this is the present criterium to prespecify the maximum number
!     of cmf lines: lambda < 100000 (150000 to included H5->6 transition)
!     and to account only for ions which are everywhere present
!         
          IF (XLAM.LE.MAXLAM) THEN
! in this way, one can simulate the optmixed approach
!          IF (XLAM.LE.MAXLAM .and. nato.le.2) THEN  
               IF (ML.GE.NFIRM .AND. MU.LE.NLASM) THEN  
                    INDEXCMF(INDMAT) = 2  
               END IF  
          END IF  
!
!***  now, we specify the cmf lines according to the previous sa-results
!     in case of optmixed
!
          IF (OPTMIXED.EQ.1. .AND. INDEXCMF(INDMAT).EQ.2) THEN  
!-----
!-----note: if this path needs to be continued, take care to update
!           corresponding quantities if model (phot.struct.) has been updated 
               stop ' optmixed not implemented yet' 
               IF (IPRES.EQ.2) THEN  
                    VSOBO = 0.D0  
                    VCONT = VSOBO  
                    TAUC = 0.D0  

                    DO L = 1,ND  
!
!***  if the lines become optically thick (corresponding to a mean tau_s
!***  of unity) below the border velocity, they can be treated in sa.
!***  they can be treated also in sa, if the continuum becomes optically
!***  thick prior to the considered line.
!
                         IF (BETOLD(INDMAT,L).LT. 0.632D0) &
&                         VSOBO = MAX(VSOBO,VELO(L))
                         IF (L.NE.1) TAUC = TAUC + &
&                          .5D0* (OPACLIN(INDMAT,L-1)+ OPACLIN(INDMAT,L))* &
&                          (R(L-1)- R(L))*SR
                         IF (TAUC.GE.1.D0) VCONT = MAX(VCONT,VELO(L))
                    END DO

                    IF (VSOBO.LT.VTHMAX .OR. VSOBO.LT.VCONT) &
&                    INDEXCMF(INDMAT) = 1

               ELSE IF (IPRES.EQ.1) THEN  
                    INDEXCMF(INDMAT) = INDEXOLD(INDMAT)  
               ELSE  
                    STOP ' ERROR IN IPRES (CMFPREP)'  
               END IF  

          END IF  

     END DO ILOOP

END IF  
!
!**** general path
!
!---  setup of rbb rates with old occ.no.
!
IILOOP: DO II = 1,INDEX1  
     INDI = INDAT1(II)  
     ER = DATA(INDI)  
     M = INT(ER)  
     IF (M.NE.1) CYCLE  
     ML = MMLOW(II)  
     MU = MMUP(II)  

     IF (ML.LT.NFIR .OR. ML.GE.NLAS) CYCLE  
     IF (MU.GT.NLAS) CYCLE

     XNL = BLEVEL(ML)  
     XNU = BLEVEL(MU)  
! ALL OCCUPATION NUMBERS CONSIDERED HERE SHOULD BE DEFINED
     IF (XNL*XNU.EQ.0.) STOP ' OCC. NUM NOT DEFINED IN CMFPREP'

     XLAMC = DATA(INDI+1)*1.D-8  
     FLU = DATA(INDI+2)  

     INDMAT = INDEXRBB(II)  
     INDCMF = INDEXCMF(INDMAT)  
!
!---     final check if everything was ok
!
     IF (XLAMCMF(INDMAT).EQ.0.D0) STOP ' ERROR IN INDEX GYMNASTICS'  
!
!---  sobolev treatment
!
     IF (INDCMF.EQ.1) THEN  
          BETOLD(INDMAT,LL) = 0.D0  
          IF (LL.EQ.1) VDOPCMF(INDMAT) = 0.D0  
          CALL TRATRAD(LL,ML,MU,FLU,XLAMC,XNL,XNU,XNEL,GL,VELR, &
           VGRAD,TLU,TUL,BETOLD(INDMAT,LL),SR, VMAX,XMUST,II, &
           TEMPE,CLF,VTHER(NATO),FIC,TCL_FAC_LINE,TCL_FAC_CONT)

          TLUMAT(INDMAT,LL) = TLU  
          TULMAT(INDMAT,LL) = TUL  

          INDEXSA(INDMAT) = INDEXSA(INDMAT) + 1  
!
!---  cmf preparation
!
     ELSE IF (INDCMF.EQ.2) THEN  
       CALL TRATCMF(LL,ML,MU,FLU,XLAMC,XNL,XNU,XNEL,CLF,GL,SR,VMAX,II, &
            FIC,TCL_FAC_LINE,TCL_FAC_CONT)

          IF (LL.EQ.1) THEN  
               VDOPCMF(INDMAT) = VTHER(NATO)  
               BLU = E2MC2/HH*XLAMC*4.D0*PI**2*FLU  
               BUL = GL(ML)*BLU/GL(MU)  
               BLUCMF(INDMAT) = BLU  
               BULCMF(INDMAT) = BUL  
               AULCMF(INDMAT) = HC2/XLAMC**3*BUL  
          END IF  

     ELSE  
          STOP 'ERROR IN INDEXCMF(CMFPREP)'  
     END IF  

END DO IILOOP 

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE TRATCMF(L,NLO,NU,FLU,XLAMC,XNL,XNU,XNE,CLF,GL,SR,VMAX,II, &
&                  FIC,TCL_FAC_LINE,TCL_FAC_CONT)

! UPDATED FOR CLUMPING: same assumption as in TRATRAD, i.e., thin clumps small
! optically thick clumps as in TRATRAD 

  
USE nlte_type
USE nlte_dim
USE fund_const
USE nlte_var, ONLY: FRE,WFRE,HC,IFRE, &
& OPAC,STRUE,XJ,ALO, &
& TLUMAT,TULMAT,OPACM,SCONTM,INDEXRBB,INDEXCMF, &
& OPTLINES,THOMSON_LINES
!,RAYLEIGH

USE nlte_porvor, ONLY: OPA_EFF_RAT_OLD

IMPLICIT NONE
!
!---  calculation of line/continuum quantities for cmf-treatment
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  FLU,SR,VMAX,XLAMC,XNE,XNL,XNU,CLF 
REAL(DP) :: FIC,TCL_FAC_LINE,TCL_FAC_CONT
INTEGER(I4B) ::  II,L,NLO,NU  
!     ..
!     .. array arguments ..
REAL(DP) ::  GL(ID_LLEVS)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DELN,OPACON,S1,S2,SCONT,SLINE,WILOG,XKLI,XKLINE, &
&                 XNUE,XNUGL,XXX
REAL(DP) ::  TCL,FAC1
INTEGER(I4B) ::  INDL,K  
!     ..
!     .. intrinsic functions ..
!INTRINSIC LOG10  
!     ..
!     ..

INDL = INDEXRBB(II)  
!
!---- line opacity
!---- (warning! xlamc= central lambda in cm)
!
XNUE = 1.D0/XLAMC  

IF (XNUE.GT.FRE(IFRE) .OR. XNUE.LT.FRE(1)) STOP ' LINE OUT OF FREQ. RANGE'

XNUGL = XNU/GL(NU)  
DELN = (XNL/GL(NLO)-XNUGL) 

XKLI = PI*E2MC2*CLIGHT*GL(NLO)*FLU*DELN / CLF  ! corrected    
XKLINE = XKLI*XLAMC*SR/VMAX 

SLINE = 3.973D-16*XNUE**3*XNUGL/DELN  ! remains
!TLUMAT(INDL,L) = XKLINE  
!TULMAT(INDL,L) = SLINE  

K = IFRE  
XXX = FRE(IFRE)  
DO WHILE (XNUE.LT.XXX)  
   K=K-1  
   XXX=FRE(K)  
END DO  

!---- effective quantity calculated as in TRATRAD (incl. inversion)
TCL = ABS(XKLINE) * TCL_FAC_LINE * VMAX/SR  !SR/VMAX term included in tcl_fac_line 
FAC1 = (1.+FIC*TCL)/(1.+TCL)
!IF (FAC1.LT.0.0) FAC1=1.0D0   ! old treatment of inversion 
FAC1 = MIN ( FAC1, OPA_EFF_RAT_OLD(L,K) )
IF (FAC1.GT.1.0D0.OR.FAC1.LE.0.0D0) THEN 
   Print*,FAC1,OPA_EFF_RAT_OLD(L,K)
   Print*,L,K,TCL 
   Stop ' opa_eff > <opa>, TRATCMF_1'
ENDIF
!if(xlamc*1.d8.gt.237. .and. xlamc*1.d8.lt.238.) fac1=1.
!if(xlamc*1.d8.gt.243. .and. xlamc*1.d8.lt.244.) fac1=1.
!if(xlamc*1.d8.gt.256. .and. xlamc*1.d8.lt.257.) fac1=1.

!if(xlamc*1.d8.gt.237. .and. xlamc*1.d8.lt.238.) &
!  print*,'237',l,tcl,fac1,opa_eff_rat_old(l,k)
!if(xlamc*1.d8.gt.243. .and. xlamc*1.d8.lt.244.) &
!  print*,'243',l,tcl,fac1,opa_eff_rat_old(l,k)
!if(xlamc*1.d8.gt.256. .and. xlamc*1.d8.lt.257.) &
!  print*,'256',l,tcl,fac1,opa_eff_rat_old(l,k)
!if(xlamc*1.d8.gt.303. .and. xlamc*1.d8.lt.304.) &
!  print*,'303',l,tcl,fac1,opa_eff_rat_old(l,k)

!IF (XLAMC*1.d8.gt.1000.and.XLAMC*1.d8.lt.3000.) THEN 
!   Print*,'test-JS-CMF:'
!   Print*,L,K,XLAMC*1.d8
!   Print*,XKLINE,TCL_FAC_LINE 
!   Print*,FAC1,OPA_EFF_RAT_OLD(L,K),L
!   Print*,'-----------'
!ENDIF
XKLINE = XKLINE * FAC1 
!---- now effective 

TLUMAT(INDL,L) = XKLINE  
TULMAT(INDL,L) = SLINE  

!
!---- interpolation weights for frecuency magnitudes. interpolation
!---- carries out magnitude versus log(nue) (linear interp.)
!
WILOG = LOG10(FRE(K)/XNUE)/LOG10(FRE(K+1)/FRE(K))  
!
!---- s_cont and opac*sr
!

IF(OPTLINES) THEN
!       S1 = STRUE(L,K) + XJ(L,K)* &
!&        (XNE*SIGMAE*AMH/CLF+THOMSON_LINES(L,K)+RAYLEIGH(L,K)/CLF)/OPAC(L,K)  
!       S2 = STRUE(L,K+1) + XJ(L,K+1)* &
!&        (XNE*SIGMAE*AMH/CLF+THOMSON_LINES(L,K+1)+RAYLEIGH(L,K+1)/CLF)/OPAC(L,K+1)  

!---- same as in TRATRAD
  S1 = STRUE(L,K) +   XJ(L,K)*(XNE*SIGMAE*AMH/CLF+ &
&   THOMSON_LINES(L,K))/OPAC(L,K) * OPA_EFF_RAT_OLD(L,K)  !back-corrected
  S2 = STRUE(L,K+1) + XJ(L,K+1)*(XNE*SIGMAE*AMH/CLF+ &
&   THOMSON_LINES(L,K+1))/OPAC(L,K+1) * OPA_EFF_RAT_OLD(L,K+1) !back-corrected   
ELSE
!---- as above 
     S1 = STRUE(L,K)   + XNE*SIGMAE*AMH*XJ(L,K)  / CLF/ OPAC(L,K) * OPA_EFF_RAT_OLD(L,K)  !back-corrected   
     S2 = STRUE(L,K+1) + XNE*SIGMAE*AMH*XJ(L,K+1)/ CLF/ OPAC(L,K+1) * OPA_EFF_RAT_OLD(L,K+1) !back-corrected 
ENDIF

SCONT = S1*10.D0** (LOG10(S1/S2)*WILOG)  

OPACON = OPAC(L,K)*10.D0** (LOG10(OPAC(L,K)/OPAC(L,K+1))*WILOG)  

OPACM(INDL,L) = OPACON*SR  ! already OK
!---- OPACM now effective quantity 
SCONTM(INDL,L) = SCONT  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE INDEXX(N,ARR,INDX)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: M=7,NSTACK=50  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  ARR(N)  
INTEGER(I4B) ::  INDX(N)  
!     ..
!     .. local scalars ..
REAL(DP) ::  A  
INTEGER(I4B) ::  I,INDXT,IR,ITEMP,J,JSTACK,K,L  
!     ..
!     .. local arrays ..
INTEGER(I4B) ::  ISTACK(NSTACK)  
!     ..

DO J = 1,N  
     INDX(J) = J  
END DO  

JSTACK = 0  
L = 1  
IR = N  

   20 CONTINUE  

IF (IR-L.LT.M) THEN  

JLOOP: DO J = L + 1,IR  
          INDXT = INDX(J)  
          A = ARR(INDXT)  
          DO I = J - 1,1,-1  
               IF (ARR(INDX(I)).LE.A) GO TO 40  
               INDX(I+1) = INDX(I)  
          END DO
          I = 0  

   40           CONTINUE  

          INDX(I+1) = INDXT  
     END DO JLOOP

     IF (JSTACK.EQ.0) RETURN  
     IR = ISTACK(JSTACK)  
     L = ISTACK(JSTACK-1)  
     JSTACK = JSTACK - 2  

ELSE  

     K = (L+IR)/2  
     ITEMP = INDX(K)  
     INDX(K) = INDX(L+1)  
     INDX(L+1) = ITEMP  

     IF (ARR(INDX(L+1)).GT.ARR(INDX(IR))) THEN  
          ITEMP = INDX(L+1)  
          INDX(L+1) = INDX(IR)  
          INDX(IR) = ITEMP  
     END IF  

     IF (ARR(INDX(L)).GT.ARR(INDX(IR))) THEN  
          ITEMP = INDX(L)  
          INDX(L) = INDX(IR)  
          INDX(IR) = ITEMP  
     END IF  

     IF (ARR(INDX(L+1)).GT.ARR(INDX(L))) THEN  
          ITEMP = INDX(L+1)  
          INDX(L+1) = INDX(L)  
          INDX(L) = ITEMP  
     END IF  

     I = L + 1  
     J = IR  
     INDXT = INDX(L)  
     A = ARR(INDXT)  

   60      CONTINUE  

     I = I + 1  
     IF (ARR(INDX(I)).LT.A) GO TO 60  

   70      CONTINUE  

     J = J - 1  
     IF (ARR(INDX(J)).GT.A) GO TO 70  
     IF (J.LT.I) GO TO 80  
     ITEMP = INDX(I)  
     INDX(I) = INDX(J)  
     INDX(J) = ITEMP  
     GO TO 60  

   80      CONTINUE  

     INDX(L) = INDX(J)  
     INDX(J) = INDXT  
     JSTACK = JSTACK + 2  
     IF (JSTACK.GT.NSTACK) STOP 'NSTACK TOO SMALL IN INDEXX'  
     IF (IR-I+1.GE.J-L) THEN  
          ISTACK(JSTACK) = IR  
          ISTACK(JSTACK-1) = I  
          IR = J - 1  
     ELSE  
          ISTACK(JSTACK) = J - 1  
          ISTACK(JSTACK-1) = L  
          L = I  
     END IF  

END IF  

GO TO 20  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE SOBPREP(LL,VELR,VGRAD,NFIR,NLAS,BLEVEL,XNEL,SR,VMAX, &
&                   NATO,XMUST,TEMPE,CLF,FIC,TCL_FAC_LINE,TCL_FAC_CONT)

! UPDATED FOR CLUMPING

USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& DATA,NDAT, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
&       LABL4,LABL1,LABU1,LABL3,IPARE5

USE nlte_var, ONLY: MLOW,MUP,MMLOW,MMUP,MCLOW,MCUP, &
& TLUMAT,TULMAT,BETOLD,INDEXRBB,INDEXCMF, &
& VTHER  

IMPLICIT NONE
!
!---  calculates rbb-rates from old occupation numbers
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: NNREC=ID_LLEVS  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  SR,TEMPE,VELR,VGRAD,VMAX,XMUST,XNEL,CLF  
!THICK
REAL(DP) :: FIC,TCL_FAC_LINE,TCL_FAC_CONT
INTEGER(I4B) ::  LL,NATO,NFIR,NLAS  
!     ..
!     .. array arguments ..
REAL(DP) ::  BLEVEL(NNREC)  
!     ..
!     .. local scalars ..
REAL(DP) ::  ER,FLU,TLU,TUL,XLAMC,XNL,XNU  
REAL(DP) ::  TCL
INTEGER(I4B) ::  I,ICOUNT,IFORM,II, &
&        INDI,INDMAT,INTEG,M,ML,MU,NC,NDATOS
LOGICAL QCL,QDB,QRT,START  
!     ..
!     .. local arrays ..
REAL(DP) ::  DL(ID_NLONG),FRECFN(ID_RBFTR),FRECIN(ID_RBFTR)  
INTEGER(I4B) ::  IFIRS(ID_IONES),IFIRSL(ID_IONES),IFIRX(ID_IONES), &
&        IFPTR(ID_NTTRD),ISA0(ID_ATOMS),ISA1(ID_ATOMS), &
&        IXA0(ID_ATOMS),IXA1(ID_ATOMS),KL(ID_LLEVS),KLX(ID_XLEVS), &
&        LA0(ID_ATOMS),LA1(ID_ATOMS),LE(ID_LLEVS),LI(ID_LLEVS), &
&        LIS(ID_SLEVS),LIX(ID_XLEVS),NFPTR(ID_NTTRD), &
&        NIONS(ID_ATOMS),NS0(ID_SLEVS),NS1(ID_SLEVS)
!     ..
!     .. external subroutines ..
EXTERNAL TRATRAD  
!     ..
!     .. intrinsic functions ..
!INTRINSIC INT  
!     ..
!     .. data statements ..
!
DATA START/.TRUE./  
!     ..

IF (START) THEN  
     ICOUNT = 0  

     DO I = 1,INDEX1  
          INDI = INDAT1(I)  
          ER = DATA(INDI)  
          M = INT(ER)  

          IF (M.EQ.1) THEN  
               ICOUNT = ICOUNT + 1  
               INDEXRBB(I) = ICOUNT  
          ELSE IF (M.EQ.2) THEN  
               INDEXRBB(I) = 0  
          ELSE  
               STOP ' ERROR IN RBB OR CBB'  
          END IF  
     END DO

     IF (ICOUNT.NE.ID_NTTRD) STOP ' NOT ALL RBB TRANSITIONS FOUND'  
     START = .FALSE.  
END IF  
!
!---  setup of rbb rates with old occ.no.
!
DO II = 1,INDEX1  
     INDI = INDAT1(II)  
     ER = DATA(INDI)  
     M = INT(ER)  
     IF (M.NE.1) CYCLE  
     ML = MMLOW(II)  
     MU = MMUP(II)  
     IF (ML.LT.NFIR .OR. ML.GE.NLAS) CYCLE  
     IF (MU.GT.NLAS) CYCLE

     XNL = BLEVEL(ML)  
     XNU = BLEVEL(MU)  
     IF (XNL*XNU.EQ.0.) STOP ' OCC. NUM NOT DEFINED IN SOBPREP'

     XLAMC = DATA(INDI+1)*1.D-8  
     FLU = DATA(INDI+2)  
     INDMAT = INDEXRBB(II)  
     BETOLD(INDMAT,LL) = 0.D0  

     CALL TRATRAD(LL,ML,MU,FLU,XLAMC,XNL,XNU,XNEL,GL,VELR,VGRAD, &
      TLU,TUL,BETOLD(INDMAT,LL),SR,VMAX,XMUST,II,TEMPE,CLF, &
      VTHER(NATO),FIC,TCL_FAC_LINE,TCL_FAC_CONT)

     TLUMAT(INDMAT,LL) = TLU  
     TULMAT(INDMAT,LL) = TUL  

END DO  

RETURN  

END
!
!***********************************************************************
!
! subroutines: atomic quantities
!
!***********************************************************************
!
SUBROUTINE RAYLEIGH_SCAT(LL)

!      Set up the hydrogen/helium I cross-sections according to fits
!      lifted from the ATLAS program (Kurucz, 1970, SAO Special Report No.309).
!      NOTE however that here the cutoff for hydrogen is set to
!      2.40e15 Hz = 1250 A, whereas Kurucz goes down to 1026 A (where
!      the whole approximation is invalid because that's inmidst the Ly lines)

! This subroutine is presently not used, test calculations (not rigorous)
! for Vega have indicated that the influence is only weak.
! Note that for a final use the high frequency parts have to be modified!
   
USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: NAT, LABL
USE nlte_var, ONLY : FRE,IFRE,BLEVEL
!,RAYLEIGH  uncomment if subroutine used

IMPLICIT NONE

INTEGER(I4B), INTENT(IN) :: LL

INTEGER(I4B), SAVE :: LEVH=0, LEVHE=0
INTEGER(I4B) :: I

REAL(DP) :: LAM, LAM2

LOGICAL START

DATA START/.TRUE./

IF(START) THEN
  START=.FALSE.
  DO I=1,ID_LLEVS
    IF(LABL(I).EQ.'H11') THEN
      LEVH=I
      EXIT      
    ENDIF
  ENDDO

  DO I=1,ID_LLEVS
    IF(LABL(I).EQ.'HE11S1') THEN
      LEVHE=I
      EXIT
    ENDIF
  ENDDO
ENDIF

IF(LEVH.EQ.0) STOP ' NO HYDROGEN IN ATMOSPHERE!'

DO I=1,IFRE

! LAMBDA IN A
  LAM=1.D8/FRE(I)
  LAM=MAX(LAM,1250.D0)
  LAM2=LAM*LAM
!  RAYLEIGH(LL,I)= &    UNCOMMENT!!!!
!  BLEVEL(LEVH)*(5.799D-13/LAM2**2+1.422D-6/LAM2**3+2.784/LAM2**4)
ENDDO

IF(LEVHE.EQ.0) RETURN

DO I=1,IFRE

  LAM=1.D8/FRE(I)
  LAM=MAX(LAM,582.D0)
  LAM2=LAM*LAM
!  RAYLEIGH(LL,I)=RAYLEIGH(LL,I)+ BLEVEL(LEVHE)*5.484D-14/LAM2**2 * &
!&   ((5.94D10/(LAM2-2.90D5)+2.44D5)/LAM2+1.D0)**2   UNCOMMENT!!!
ENDDO

RETURN
END

!***********************************************************************
!
SUBROUTINE HMINUS(XNE,TEMP,LAM,EHNUEKT,LL,OPAHMINUS)
!
!  calculates bf and ff opacities for H-minus
!
!  lambda in A  
!
!  formulas and fits according to the references in:
!  D.G. Gray, "The observation and analysis of stellar photospheres",
!  2nd Edition (1992), Cambridge Astrophysics Series 20, Cambridge
!  University Press, p. 135-137  

USE nlte_type
USE nlte_dim
USE fund_const, ONLY: AKB
USE princesa_var, ONLY: NAT, LABAT
USE nlte_var, ONLY: ENIONND

IMPLICIT NONE

REAL(DP), INTENT(IN) :: XNE, TEMP, LAM, EHNUEKT
REAL(DP), INTENT(OUT) :: OPAHMINUS
INTEGER(I4B), INTENT(IN) :: LL


REAL(DP) :: THETA, THETALOG, NH, PE, OPABF, OPAFF, STIM, XLAM, F0, F1, F2

INTEGER(I4B) :: K

! fit coefficients for bf opacity
REAL(DP) :: A0=1.99654D0,A1=-1.18267D-5,A2=2.64243D-6,A3=-4.40524D-10, &
&           A4=3.23992D-14,A5=-1.39568D-18,A6=2.78701D-23

! fit coefficients for ff opacity
REAL(DP) :: B0=-2.2763D0, B1=-1.685D0, B2=.76661D0, B3=-.0533464D0
REAL(DP) :: C0=15.2827D0, C1=-9.2846D0, C2=1.99381D0, C3=-.142631D0
REAL(DP) :: D0=-197.789D0, D1=190.266D0, D2=-67.9775D0, D3=10.6913D0, &
&           D4=-0.625151D0

IF(LAM.LT.2000.) THEN
  OPAHMINUS=0.D0
  RETURN
ENDIF  

DO K=1,NAT
 IF(LABAT(K) .EQ. 'H') GOTO 1
ENDDO

STOP ' HYDROGEN NOT FOUND IN HMINUS'

1 NH=ENIONND(K,1,LL) !neutral hydrogen density
PE=XNE*AKB*TEMP   !electron pressure

THETA=5040.D0/TEMP

! bf opacity

OPABF=0.
IF (LAM .LE. 16000.) THEN
  OPABF=A0+LAM*(A1+LAM*(A2+LAM*(A3+LAM*(A4+LAM*(A5+LAM*A6)))))
  OPABF=OPABF*1.D-18
  STIM=1.-EHNUEKT
  OPABF=4.158D-10*OPABF*PE*THETA**2.5D0 *10.D0**(0.754D0*THETA)*STIM*NH
  IF(OPABF.LT.0.) OPABF=0.
ENDIF

THETALOG=LOG10(THETA)
XLAM=LOG10(LAM)

F0=B0+XLAM*(B1+XLAM*(B2+XLAM*B3))
F1=C0+XLAM*(C1+XLAM*(C2+XLAM*C3))
F2=D0+XLAM*(D1+XLAM*(D2+XLAM*(D3+XLAM*D4)))

OPAFF=1.D-26*PE*10.D0**(F0+F1*THETALOG+F2*THETALOG**2)*NH

OPAHMINUS=OPABF+OPAFF
RETURN

END
!
!-----------------------------------------------------------------------
!

FUNCTION AGAU(NIVEL,XNU,KEY,NUMI)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!       this function returns the bound-free gaunt factor for h and he
!       with the approximation used by herrero (tesis doctoral).
!
!     .. scalar arguments ..
REAL(DP) ::  XNU,AGAU  
INTEGER(I4B) ::  NIVEL,NUMI  
CHARACTER KEY*6  
!     ..
!     .. local arrays ..

REAL(DP) ::  A(3),B(3),C(3)  
!     ..
!     .. data statements ..
DATA A/0.9916D0,1.105D0,1.101D0/  
DATA B/2.71852D13,-2.37496D14,-9.863186D13/  
DATA C/-2.26846D30,4.07677D28,1.03537D28/  
!     ..

IF (KEY.EQ.'H') THEN  

     IF (NUMI.NE.1) STOP 'ERROR IN NUMI -H- GBF'  

     IF (NIVEL.LE.3) THEN  
          AGAU = A(NIVEL) + B(NIVEL)/XNU + C(NIVEL)/XNU/XNU  
          RETURN
     ELSE  
          AGAU = 1.D0  
          RETURN
     END IF  

ELSE IF (KEY.EQ.'HE') THEN  

     IF (NUMI.EQ.1) THEN  
          IF (NIVEL.LE.3) THEN  
               AGAU = A(NIVEL) + B(NIVEL)/XNU + C(NIVEL)/XNU/XNU  
               RETURN
          ELSE  
               AGAU = 1.D0  
               RETURN
          END IF  

     ELSE IF (NUMI.EQ.2) THEN  
          IF (NIVEL.LE.3) THEN  
               AGAU = A(NIVEL) + 4.D0*B(NIVEL)/XNU + 16.D0*C(NIVEL)/XNU/XNU
               RETURN
          ELSE  
               AGAU = 1.D0  
               RETURN
          END IF  
     ELSE  
          STOP 'ERROR IN NUMI -HE- GBF'  
     END IF  

ELSE  
     STOP 'AGAU INCORRECTLY CALLED, ONLY FOR H, HE'  
END IF  

END
!
!-----------------------------------------------------------------------
!
FUNCTION GAUNT(N,F)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!-----bound free gaunt factors for hydrogen like atoms, f = nue/z**2
!
!     .. scalar arguments ..
REAL(DP) ::  F,GAUNT  
INTEGER(I4B) ::  N  
!     ..
!     .. local scalars ..
REAL(DP) ::  OF,ONE  
!     ..
!     .. local arrays ..
REAL(DP) ::  A1(6),A2(6),B1(6),B2(6),C1(6),C2(6)  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA A1/1.078D0,1.0926D0,1.0983D0,1.0954D0,1.0912D0,1.0925D0/  
DATA B1/-8.754D14,-2.019D14,-9.45D13,-5.188D13,-3.2D13,-2.331D13/  
DATA C1/-1.791D29,1.836D28,9.177D27,3.552D27,1.576D27,9.325D26/  
DATA A2/0.798D0,0.768D0,0.793D0,0.831D0,0.758D0,0.79D0/  
DATA B2/5.358D15,6.242D15,5.48D15,4.094D15,6.633D15,5.808D15/  
DATA C2/-3.484D31,-3.208D31,-2.318D31,-1.43D31,-3.32D31,-2.844D31/  
DATA ONE/1.0D0/  
!     ..

IF (N.GT.6) GO TO 20  
OF = ONE/F  

IF (F.GT.1.0D16) GO TO 10  
!
!-----less than 1.0e16 hz
!
GAUNT = A1(N) + (B1(N)+C1(N)*OF)*OF  
RETURN  
!
!-----greater than 1.0e16 hz
!
   10 CONTINUE  

GAUNT = A2(N) + (B2(N)+C2(N)*OF)*OF  
RETURN  
				      !
!-----default
!
   20 CONTINUE  

GAUNT = ONE  
RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION GFF(FRQ,T,Z,NT,L,I)  

USE nlte_type  
USE nlte_dim  
USE nlte_var, ONLY: KIS => IONMAX
IMPLICIT NONE  
!
!-----generates thermally averaged free-free non-relativistic gaunt
!-----factor for a hydrogenic ion of charge z, with a maximum relative
!-----error of .oo7, (rms fitting error = .001) for temperature and
!-----frequency in intervals:
!                     10**-4 le u le 10**1.5,
!                     10**-3 le gams le 10**3,
!-----where
!             gams = z**2*ryd/k*t = z**2*1.579e5/t
!                                   max          t=1.6e8k
!                                   min          t=5000.k
!
!             u = h*nu/k*t = 1.44*fr(cm-1)/t
!                            max  .17      t=2400.k
!                                 10.      t=140ooo.k
!                            min  3.2e7    t=1.4e6k
!                                 1.e6     t=45000.k
!                                 1.1e5    t=5000.k
!
!-----to obtain the
!-----stated accuracy, the full number of significant figures in the
!-----coefficients must be retained.
!
!-----this subroutine uses a two-dimensional chebyshev expansion
!-----computed from expressions given by karzas and latter (ap.j.
!-----suppl., v.6, p.167, 1961) augmented by various limiting forms
!-----of energy-specific gaunt factors.
!     d.g.hummer, jila, may 1987.
!
!       subroutine arguments are:
!         z    =  nuclear charge
!         t    =  temperature in degrees kelvin
!         frq  =  frequency
!
!     called by opacitc and opacitm  
!     
!     adapted by A. Pauldrach and JP  
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT,KISND=KIS*ND1,KISNUM=8*KISND  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  FRQ,T,Z,GFF  
INTEGER(I4B) ::  I,L,NT  
!     ..
!     .. local scalars ..
REAL(DP) ::  TXU,XLF  
INTEGER(I4B) ::  IR,J  
!     ..
!     .. local arrays ..
REAL(DP) ::  B(8),C(8,KIS,ND1),CON(KIS,ND1)  
!     ..
!     .. external functions ..
REAL(DP) ::  GFREE  
EXTERNAL GFREE  
!     ..
!     .. external subroutines ..
EXTERNAL GFFINIT  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,LOG10  
!     ..
!     .. data statements ..
!
DATA CON,C/KISND*0.D0,KISNUM*0.D0/  
!     ..
!
!for first freq., calculate freq. independent part of gaunt-factor
!
IF (L.EQ.1) CALL GFFINIT(Z,T,C(:,I,NT),CON(I,NT))  
!
IF (CON(I,NT).EQ.100.D0) THEN  
     GFF = GFREE(FRQ,T,Z)  
     RETURN  

ELSE  
!
!-------------------------------------------------------------
!        h [erg/s] / 1 ryd = 3.0397e-16 ; 1 ryd = 2.18e-11 erg
!
     XLF = LOG10(FRQ*3.0397D-16)  
!-------------------------------------------------------------
     TXU = 0.72727273D0*XLF + CON(I,NT)  

     IF (ABS(TXU).GT.2.0D0) THEN  
          GFF = GFREE(FRQ,T,Z)  
          RETURN  
     END IF  

     IR = 6  
     B(8) = C(8,I,NT)  
     B(7) = TXU*B(8) + C(7,I,NT)  
     DO J = 1,6  
          B(IR) = TXU*B(IR+1) - B(IR+2) + C(IR,I,NT)  
          IR = IR - 1  
     END DO

     GFF = B(1) - B(3)  
END IF  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE GFFINIT(Z,T,C,CON)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!-----initiation of the nue-independent part of the f-f gaunt-factor
!-----called by 'gff' - m 4.91 adi
!
!     .. scalar arguments ..
REAL(DP) ::  CON,T,Z  
!     ..
!     .. array arguments ..
REAL(DP) ::  C(8)  
!     ..
!     .. local scalars ..
REAL(DP) ::  TXG,XLRKT  
INTEGER(I4B) ::  I,IR,J  
!     ..
!     .. local arrays ..
REAL(DP) ::  B(11),D(8,11)  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,LOG10  
!     ..
!     .. data statements ..
!
DATA ((D(I,J),I=1,8),J=1,6)/ &
&                      8.986940175D+00,-4.009515855D+00, &
&     8.808871266D-01, 2.640245111D-02,-4.580645915D-02, &
&    -3.568055702D-03, 2.827798067D-03, 3.365860195D-04, &
&    -8.006936989D-01, 9.466021705D-01, 9.043402532D-02, &
&    -9.608451450D-02,-1.885629865D-02, 1.050313890D-02, &
&     2.800889961D-03,-1.078209202D-03,-3.781305103D-01, &
&     1.102726332D-01,-1.543619180D-02, 8.310561114D-03, &
&     2.179620525D-02, 4.259726289D-03,-4.181588794D-03, &
&    -1.770208330D-03, 1.877213132D-02,-1.004885705D-01, &
&    -5.483366378D-02,-4.520154409D-03, 8.366530426D-03, &
&     3.700273930D-03, 6.889320423D-04, 9.460313195D-05, &
&     7.300158392D-02, 3.576785497D-03,-4.545307025D-03, &
&    -1.017965604D-02,-9.530211924D-03,-3.450186162D-03, &
&     1.040482914D-03, 1.407073544D-03,-1.744671550D-03, &
&     2.864013856D-02, 1.903394837D-02, 7.091074494D-03, &
&    -9.668371391D-04,-2.999107465D-03,-1.820642230D-03, &
&    -3.874082085D-04/
DATA ((D(I,J),I=1,8),J=7,11)/ &
&                     -1.707268366D-02,-4.694254776D-03, &
&     1.311691517D-03, 5.316703136D-03, 5.178193095D-03, &
&     2.451228935D-03,-2.277321615D-05,-8.182359057D-04, &
&     2.567331664D-04,-9.155339970D-03,-6.997479192D-03, &
&    -3.571518641D-03,-2.096101038D-04, 1.553822487D-03, &
&     1.509584686D-03, 6.212627837D-04, 4.098322531D-03, &
&     1.635218463D-03,-5.918883504D-04,-2.333091048D-03, &
&    -2.484138313D-03,-1.359996060D-03,-5.371426147D-05, &
&     5.553549563D-04, 3.837562402D-05, 2.938325230D-03, &
&     2.393747064D-03, 1.328839809D-03, 9.135013312D-05, &
&    -7.137252303D-04,-7.656848158D-04,-3.504683798D-04, &
&    -8.491991820D-04,-3.615327726D-04, 3.148015257D-04, &
&     8.909207650D-04, 9.869737522D-04, 6.134671184D-04, &
&     1.068883394D-04,-2.046080100D-04/
!     ..
!
!-----xlrxt is log(ryd/kt)
!
XLRKT = 5.1983649D0 - LOG10(T)  
TXG = 0.66666667D0* (2.0D0*LOG10(Z)+XLRKT)  

IF (ABS(TXG).GT.2.0D0) THEN  
     CON = 100.D0  
     RETURN  
END IF  

CON = 0.72727273D0*XLRKT + 0.90909091D0  

DO J = 1,8  
     IR = 9  
     B(11) = D(J,11)  
     B(10) = TXG*B(11) + D(J,10)  

     DO I = 1,9  
          B(IR) = TXG*B(IR+1) - B(IR+2) + D(J,IR)  
          IR = IR - 1  
     END DO

     C(J) = 0.25D0* (B(1)-B(3))  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION GFREE(FRQ,T,Z)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!-----free-free-gaunt-factors
!-----called by 'gff'
!
!-----------------------------------------------------------------------
!     .. scalar arguments ..
REAL(DP) ::  FRQ,T,Z,GFREE  
!     ..
!     .. local scalars ..
REAL(DP) ::  C1,C2,C3,C4,THET,X  
!     ..

THET = 5.0404D3/T  
THET = THET*Z*Z  
IF (THET.LT.4.0D-2) THET = 4.0D-2  
X = FRQ/2.997929D14  
X = X/Z/Z  
IF (X.GT.1.0D0) GO TO 10  
IF (X.LT.0.2D0) X = 0.2D0  

GFREE = (1.0823D0+2.98D-2/THET) + (6.7D-3+1.12D-2/THET)/X  

RETURN  

   10 CONTINUE  

C1 = (3.9999187D-3-7.8622889D-5/THET)/THET + 1.070192D0  
C2 = (6.4628601D-2-6.1953813D-4/THET)/THET + 2.6061249D-1  
C3 = (1.3983474D-5/THET+3.7542343D-2)/THET + 5.7917786D-1  
C4 = 3.4169006D-1 + 1.1852264D-2/THET  

GFREE = ((C4/X-C3)/X+C2)/X + C1  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION HCOL(NLO,NUP,T)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!          generates hydrogen collision rate coefficients
!          using fits given by giovanardi, natta, and
!          palla,  aa. suppl., 70, 269, 1987
!          coefficients have been summed over all states
!
!             nlo  =  lower principal quantum number (le.14)
!             nup  =  upper principal quantum number (le.15)
!             temp =  temperature  (2000 < t < 500000)
!
!             d.g.hummer, jila, nov 1989
!             modified k. butler, munich, jan 1990
!             extended to higher temperatures and modified for use
!             in detail (not checked so far)
!             modified by j. puls for different data
!                      used by david for lowest transitions
!             ly-alpha,ly-beta, balmer-alpha according to callaway
!             (at.data nucl.data tables 57, 9-20
!
!
!     .. scalar arguments ..
REAL(DP) ::  T,HCOL  
INTEGER(I4B) ::  NLO,NUP  
!     ..
!     .. local scalars ..
REAL(DP) ::  CUTOFF,GAMFAC,GLO,GUP,RYD,TR,TX,XNLOS,XNUPS  
INTEGER(I4B) ::  I,K  
!     ..
!     .. local arrays ..
REAL(DP) ::  BBA(0:6),BLA(0:5,2),BLB(0:7,3),C(420),D(420), &
&                 DAMP(2)
INTEGER(I4B) ::  N0(14)  
!     ..
!     .. intrinsic functions ..
!INTRINSIC DBLE,EXP,LOG,SQRT  
!     ..
!     .. data statements ..
!
!          n=1 lower state
!          n=2 lower state
!          n=3 lower state
!          n=4 lower state
!          n=5 lower state
!          n=6 lower state
!          n=7 lower state
!          n=8 lower state
!          n=9 lower state
!          n=10 lower state
!          n=11 lower state
!          n=12 lower state
!          n=13 lower state
!          n=14 lower state
!          for each successive lower state, data start at n0:
DATA RYD/6.336D-6/  
DATA DAMP/28.056D0,7.2945D0/  
DATA BLA/0.36105477D0, 1.73103797D0, 0.37297254D0,-0.43240114D0, &
&        0.14296095D0,-0.01670572D0, 0.25501041D0, 0.09973588D0, &
&       -0.05404749D0, 0.02520311D0,-0.00405306D0, 0.03508995D0/
DATA BLB/0.09895325D0, 0.23240113D0, 0.20112555D0, 0.87813331D0, &
&       -3.34239518D0, 3.70755260D0,-1.37320827D0, 0.D0,         &
&        0.06461453D0,-0.11160826D0, 0.76957269D0,-2.18918875D0, &
&        3.14216481D0,-2.23498961D0, 0.62651998D0, 0.01125150D0, &
&        0.05742134D0,-0.00609531D0, 0.93848281D0,-3.40393911D0, &
&        5.50742903D0,-4.25196238D0, 1.26871040D0, 0.00724136D0/
DATA BBA/21.6781D0,207.5215D0,-47.581D0,5.0449D0,-1.0262D0, &
&     0.4085D0,-5.93D-2/
DATA (C(I),I=1,56)/    5.732D-01, 1.829D-05,-1.158D-10, 9.429D-16,&
      2.122D-01,-9.190D-07, 8.772D-11,-5.679D-16, 6.323D-03, 2.237D-06,&
     -1.620D-11, 8.955D-17, 2.035D-02, 6.076D-07,-2.175D-13,-2.495D-18,&
      1.136D-02, 3.428D-07,-1.467D-13,-1.300D-18, 6.999D-03, 2.126D-07,&
     -9.963D-14,-7.672D-19, 4.624D-03, 1.410D-07,-6.969D-14,-4.927D-19,&
      3.217D-03, 9.836D-08,-5.031D-14,-3.361D-19, 2.329D-03, 7.135D-08,&
     -3.737D-14,-2.400D-19, 1.741D-03, 5.342D-08,-2.845D-14,-1.775D-19,&
      1.336D-03, 4.103D-08,-2.213D-14,-1.351D-19, 1.048D-03, 3.220D-08,&
     -1.754D-14,-1.053D-19, 8.369D-04, 2.574D-08,-1.413D-14,-8.368D-20,&
      6.791D-04, 2.090D-08,-1.154D-14,-6.763D-20/
DATA (D(I),I=1,56)/    5.856D-01, 1.551D-05,-9.669D-12, 5.716D-19,&
      1.537D-01, 3.548D-06,-3.224D-12,-7.626D-18, 2.400D-02, 1.419D-06,&
     -2.008D-12, 1.356D-18, 2.002D-02, 6.325D-07,-7.070D-13, 4.096D-19,&
      1.123D-02, 3.549D-07,-3.998D-13, 2.331D-19, 6.940D-03, 2.194D-07,&
     -2.483D-13, 1.453D-19, 4.593D-03, 1.453D-07,-1.648D-13, 9.667D-20,&
      3.199D-03, 1.012D-07,-1.150D-13, 6.758D-20, 2.318D-03, 7.334D-08,&
     -8.349D-14, 4.910D-20, 1.727D-03, 5.493D-08,-6.270D-14, 3.695D-20,&
      1.326D-03, 4.218D-08,-4.821D-14, 2.844D-20, 1.040D-03, 3.310D-08,&
     -3.786D-14, 2.236D-20, 8.305D-04, 2.645D-08,-3.028D-14, 1.790D-20,&
      6.740D-04, 2.147D-08,-2.460D-14, 1.455D-20/
! N=2 LOWER STATE
DATA (C(I),I=57,108)/  1.515D+01, 1.268D-03, 5.808D-09,-5.831D-14,&
      7.816D-01, 5.413D-04,-1.827D-09, 5.100D-17, 1.459D+00, 2.858D-04,&
     -2.207D-09, 9.028D-15, 7.172D-01, 1.440D-04,-1.139D-09, 4.755D-15,&
      4.107D-01, 8.360D-05,-6.699D-10, 2.823D-15, 2.591D-01, 5.319D-05,&
     -4.293D-10, 1.819D-15, 1.747D-01, 3.608D-05,-2.925D-10, 1.243D-15,&
      1.237D-01, 2.567D-05,-2.087D-10, 8.891D-16, 9.097D-02, 1.893D-05,&
     -1.543D-10, 6.585D-16, 6.896D-02, 1.438D-05,-1.174D-10, 5.017D-16,&
      5.356D-02, 1.119D-05,-9.150D-11, 3.913D-16, 4.247D-02, 8.887D-06,&
     -7.272D-11, 3.112D-16, 3.425D-02, 7.176D-06,-5.877D-11, 2.516D-16/
DATA (D(I),I=57,108)/  1.710D+01, 1.530D-03,-2.553D-09, 1.924D-15,&
      8.237D+00, 3.554D-04,-7.566D-10, 6.420D-16, 5.932D+00, 1.301D-04,&
     -2.912D-10, 2.535D-16, 2.991D+00, 6.419D-05,-1.444D-10, 1.260D-16,&
      1.733D+00, 3.689D-05,-8.324D-11, 7.267D-17, 1.102D+00, 2.333D-05,&
     -5.273D-11, 4.605D-17, 7.472D-01, 1.576D-05,-3.567D-11, 3.116D-17,&
      5.312D-01, 1.118D-05,-2.532D-11, 2.212D-17, 3.919D-01, 8.232D-06,&
     -1.865D-11, 1.630D-17, 2.977D-01, 6.245D-06,-1.416D-11, 1.237D-17,&
      2.312D-01, 4.855D-06,-1.101D-11, 9.622D-18, 1.838D-01, 3.851D-06,&
     -8.734D-12, 7.635D-18, 1.484D-01, 3.108D-06,-7.050D-12, 6.164D-18/
! N=3 LOWER STATE
DATA (C(I),I=109,156)/-1.290D+01, 2.059D-02, 5.469D-08,-9.082D-13,&
      3.562D+02, 7.337D-03,-9.622D-08, 5.596D-13, 5.744D+00, 3.570D-03,&
     -3.259D-08, 1.452D-13, 2.968D+00, 1.813D-03,-1.703D-08, 7.744D-14,&
      1.756D+00, 1.065D-03,-1.016D-08, 4.667D-14, 1.135D+00, 6.865D-04,&
     -6.601D-09, 3.053D-14, 7.802D-01, 4.713D-04,-4.558D-09, 2.116D-14,&
      5.615D-01, 3.390D-04,-3.292D-09, 1.532D-14, 4.189D-01, 2.528D-04,&
     -2.461D-09, 1.148D-14, 3.213D-01, 1.939D-04,-1.891D-09, 8.833D-15,&
      2.523D-01, 1.522D-04,-1.487D-09, 6.953D-15, 2.018D-01, 1.218D-04,&
     -1.192D-09, 5.576D-15/
DATA (D(I),I=109,156)/ 1.940D+02, 1.949D-02,-3.832D-08, 3.137D-14,&
      4.729D+02, 1.927D-03,-4.171D-09, 3.628D-15, 6.741D+01, 1.315D-03,&
     -3.145D-09, 2.814D-15, 3.444D+01, 6.477D-04,-1.560D-09, 1.399D-15,&
      2.031D+01, 3.744D-04,-9.054D-10, 8.130D-16, 1.311D+01, 2.388D-04,&
     -5.789D-10, 5.203D-16, 9.007D+00, 1.629D-04,-3.955D-10, 3.556D-16,&
      6.484D+00, 1.166D-04,-2.835D-10, 2.550D-16, 4.837D+00, 8.666D-05,&
     -2.108D-10, 1.896D-16, 3.711D+00, 6.631D-05,-1.614D-10, 1.452D-16,&
      2.914D+00, 5.194D-05,-1.265D-10, 1.138D-16, 2.332D+00, 4.150D-05,&
     -1.011D-10, 9.100D-17/
! N=4 LOWER STATE
DATA (C(I),I=157,200)/ 4.139D+03, 4.645D-01,-7.097D-06, 4.388D-11,&
      1.794D+03, 4.443D-02,-6.484D-07, 3.936D-12, 1.536D+01, 2.042D-02,&
     -2.065D-07, 9.734D-13, 8.730D+00, 1.033D-02,-1.074D-07, 5.161D-13,&
      5.434D+00, 6.084D-03,-6.423D-08, 3.116D-13, 3.628D+00, 3.938D-03,&
     -4.196D-08, 2.048D-13, 2.554D+00, 2.718D-03,-2.914D-08, 1.428D-13,&
      1.873D+00, 1.967D-03,-2.119D-08, 1.041D-13, 1.418D+00, 1.476D-03,&
     -1.594D-08, 7.843D-14, 1.102D+00, 1.138D-03,-1.232D-08, 6.075D-14,&
      8.744D-01, 8.987D-04,-9.746D-09, 4.809D-14/
DATA (D(I),I=157,200)/ 7.204D+03, 1.627D-01,-5.181D-07, 5.605D-13,&
      2.507D+03, 9.370D-03,-2.091D-08, 1.842D-14, 3.823D+02, 6.480D-03,&
     -1.600D-08, 1.452D-14, 1.950D+02, 3.161D-03,-7.869D-09, 7.157D-15,&
      1.154D+02, 1.823D-03,-4.561D-09, 4.154D-15, 7.486D+01, 1.165D-03,&
     -2.924D-09, 2.665D-15, 5.178D+01, 7.977D-04,-2.006D-09, 1.830D-15,&
      3.752D+01, 5.737D-04,-1.444D-09, 1.318D-15, 2.816D+01, 4.283D-04,&
     -1.080D-09, 9.858D-16, 2.174D+01, 3.293D-04,-8.307D-10, 7.587D-16,&
      1.717D+01, 2.592D-04,-6.544D-10, 5.978D-16/
! N=5 LOWER STATE
DATA (C(I),I=201,240)/-9.122D+02, 1.260D+00,-1.070D-05, 4.290D-11,&
      3.959D+01, 2.108D-01,-2.162D-06, 1.020D-11, 3.691D+01, 7.806D-02,&
     -8.485D-07, 4.166D-12, 2.352D+01, 3.911D-02,-4.365D-07, 2.179D-12,&
      1.542D+01, 2.296D-02,-2.601D-07, 1.310D-12, 1.062D+01, 1.487D-02,&
     -1.699D-07, 8.608D-13, 7.642D+00, 1.029D-02,-1.183D-07, 6.014D-13,&
      5.695D+00, 7.464D-03,-8.621D-08, 4.394D-13, 4.368D+00, 5.617D-03,&
     -6.508D-08, 3.323D-13, 3.430D+00, 4.348D-03,-5.051D-08, 2.583D-13/
DATA (D(I),I=201,240)/ 2.166D+04, 4.690D-01,-1.122D-06, 1.008D-12,&
      3.874D+03, 6.443D-02,-1.596D-07, 1.452D-13, 1.465D+03, 2.207D-02,&
     -5.556D-08, 5.082D-14, 7.410D+02, 1.062D-02,-2.698D-08, 2.476D-14,&
      4.374D+02, 6.096D-03,-1.556D-08, 1.430D-14, 2.841D+02, 3.889D-03,&
     -9.962D-09, 9.167D-15, 1.969D+02, 2.663D-03,-6.838D-09, 6.297D-15,&
      1.431D+02, 1.918D-03,-4.935D-09, 4.547D-15, 1.078D+02, 1.436D-03,&
     -3.698D-09, 3.409D-15, 8.353D+01, 1.107D-03,-2.854D-09, 2.632D-15/
! N=6 LOWER STATE
DATA (C(I),I=241,276)/-3.431D+03, 4.116D+00,-3.853D-05, 1.679D-10,&
      4.397D+01, 6.434D-01,-7.008D-06, 3.431D-11, 8.927D+01, 2.325D-01,&
     -2.667D-06, 1.350D-11, 6.153D+01, 1.152D-01,-1.354D-06, 6.957D-12,&
      4.165D+01, 6.729D-02,-8.024D-07, 4.156D-12, 2.923D+01, 4.349D-02,&
     -5.232D-07, 2.724D-12, 2.130D+01, 3.008D-02,-3.641D-07, 1.902D-12,&
      1.603D+01, 2.185D-02,-2.656D-07, 1.391D-12, 1.239D+01, 1.647D-02,&
     -2.008D-07, 1.054D-12/
DATA (D(I),I=241,276)/ 7.146D+04, 1.379D+00,-3.346D-06, 3.023D-12,&
      1.187D+04, 1.794D-01,-4.501D-07, 4.118D-13, 4.380D+03, 5.990D-02,&
     -1.527D-07, 1.405D-13, 2.192D+03, 2.846D-02,-7.324D-08, 6.759D-14,&
      1.288D+03, 1.621D-02,-4.197D-08, 3.881D-14, 8.351D+02, 1.031D-02,&
     -2.678D-08, 2.480D-14, 5.790D+02, 7.050D-03,-1.837D-08, 1.702D-14,&
      4.213D+02, 5.079D-03,-1.326D-08, 1.229D-14, 3.179D+02, 3.804D-03,&
     -9.944D-09, 9.226D-15/
! N=7 LOWER STATE
DATA(C(I),I=277,308)/ -9.280D+03, 1.116D+01,-1.122D-04, 5.167D-10,&
      6.658D+01, 1.651D+00,-1.884D-05, 9.487D-11, 2.172D+02, 5.833D-01,&
     -6.977D-06, 3.615D-11, 1.535D+02, 2.858D-01,-3.499D-06, 1.838D-11,&
      1.049D+02, 1.660D-01,-2.060D-06, 1.090D-11, 7.412D+01, 1.070D-01,&
     -1.339D-06, 7.118D-12, 5.428D+01, 7.389D-02,-9.304D-07, 4.963D-12,&
      4.103D+01, 5.366D-02,-6.786D-07, 3.629D-12/
DATA(D(I),I=277,308)/  1.954D+05, 3.426D+00,-8.397D-06, 7.624D-12,&
      3.057D+04, 4.266D-01,-1.080D-06, 9.917D-13, 1.103D+04, 1.392D-01,&
     -3.582D-07, 3.309D-13, 5.458D+03, 6.530D-02,-1.696D-07, 1.572D-13,&
      3.189D+03, 3.693D-02,-9.653D-08, 8.966D-14, 2.062D+03, 2.338D-02,&
     -6.136D-08, 5.707D-14, 1.429D+03, 1.595D-02,-4.200D-08, 3.910D-14,&
      1.039D+03, 1.148D-02,-3.029D-08, 2.282D-14/
! N=8 LOWER STATE
DATA (C(I),I=309,336)/-2.069D+04, 2.637D+01,-2.802D-04, 1.342D-09,&
      2.055D+02, 3.731D+00,-4.420D-05, 2.276D-10, 5.123D+02, 1.292D+00,&
     -1.599D-05, 8.442D-11, 3.578D+02, 6.265D-01,-7.922D-06, 4.235D-11,&
      2.438D+02, 3.616D-01,-4.633D-06, 2.494D-11, 1.721D+02, 2.322D-01,&
     -3.000D-06, 1.622D-11, 1.260D+02, 1.601D-01,-2.081D-06, 1.129D-11/
DATA (D(I),I=309,336)/ 4.651D+05, 7.527D+00,-1.859D-05, 1.694D-11,&
      6.930D+04, 9.038D-01,-2.302D-06, 2.121D-12, 2.450D+04, 2.891D-01,&
     -7.487D-07, 6.939D-13, 1.200D+04, 1.340D-01,-3.505D-07, 3.260D-13,&
      6.970D+03, 7.523D-02,-1.981D-07, 1.846D-13, 4.493D+03, 4.741D-02,&
     -1.254D-07, 1.170D-13, 3.106D+03, 3.226D-02,-8.559D-08, 7.997D-14/
! N=9 LOWER STATE
DATA (C(I),I=337,360)/-4.032D+04, 5.614D+01,-6.231D-04, 3.073D-09,&
      6.989D+02, 7.655D+00,-9.352D-05, 4.903D-10, 1.141D+03, 2.605D+00,&
     -3.313D-05, 1.777D-10, 7.755D+02, 1.250D+00,-1.624D-05, 8.808D-11,&
      5.234D+02, 7.175D-01,-9.437D-06, 5.153D-11, 3.677D+02, 4.590D-01,&
     -6.087D-06, 3.338D-11/
DATA (D(I),I=337,360)/ 9.956D+05, 1.506D+01,-3.741D-05, 3.418D-11,&
      1.425D+05, 1.754D+00,-4.489D-06, 4.146D-12, 4.949D+04, 5.510D-01,&
     -1.435D-06, 1.333D-12, 2.401D+04, 2.526D-01,-6.645D-07, 6.196D-13,&
      1.386D+04, 1.408D-01,-3.729D-07, 3.485D-13, 8.904D+03, 8.835D-02,&
      -2.350D-07, 2.200D-13/ 
! N=10 LOWER STATE
DATA (C(I),I=361,380)/-7.097D+04, 1.101D+02,-1.266D-03, 6.390D-09,&
      2.018D+03, 1.455D+01,-1.824D-04, 9.708D-10, 2.383D+03, 4.875D+00,&
     -6.348D-05, 3.449D-10, 1.569D+03, 2.319D+00,-3.081D-05, 1.691D-10,&
      1.046D+03, 1.323D+00,-1.779D-05, 9.830D-11/
DATA (D(I),I=361,380)/ 1.961D+06, 2.798D+01,-6.982D-05, 6.394D-11,&
      2.715D+05, 3.175D+00,-8.158D-06, 7.551D-12, 9.279D+04, 9.821D-01,&
     -2.567D-06, 2.390D-12, 4.460D+04, 4.458D-01,-1.178D-06, 1.100D-12,&
      2.561D+04, 2.468D-01,-6.566D-07, 6.150D-13/
! N=11 LOWER STATE
DATA (C(I),I=381,396)/-1.150D+05, 2.020D+02,-2.392D-03, 1.231D-08,&
      4.988D+03, 2.601D+01,-3.334D-04, 1.797D-09, 4.675D+03, 8.595D+00,&
     -1.142D-04, 6.273D-10, 2.986D+03, 4.054D+00,-5.491D-05, 3.046D-10/
DATA (D(I),I=381,396)/ 3.613D+06, 4.898D+01,-1.227D-04, 1.125D-10,&
      4.861D+05, 5.434D+00,-1.401D-05, 1.299D-11, 1.638D+05, 1.658D+00,&
      -4.348D-06, 4.054D-12, 7.810D+04, 7.456D-01,-1.976D-06, 1.850D-12/
! N=12 LOWER STATE
DATA (C(I),I=397,408)/-1.737D+05, 3.511D+02,-4.263D-03, 2.227D-08,&
      1.094D+04, 4.419D+01,-5.774D-04, 3.146D-09, 8.673D+03, 1.442D+01,&
     -1.950D-04, 1.082D-09/
DATA (D(I),I=397,408)/ 6.300D+06, 8.163D+01,-2.051D-04, 1.884D-10,&
      8.271D+05, 8.881D+00,-2.296D-05, 2.131D-11, 2.753D+05, 2.676D+00,&
     -7.037D-06, 6.571D-12/
! N=13 LOWER STATE
DATA (C(I),I=409,416)/-2.459D+05, 5.829D+02,-7.233D-03, 3.830D-08,&
      2.191D+04, 7.194D+01,-9.561D-04, 5.259D-09/
DATA (D(I),I=409,416)/ 1.049D+07, 1.305D+02,-3.288D-04, 3.025D-10,&
      1.348D+06, 1.396D+01,-3.617D-05, 3.361D-11/
! N=14 LOWER STATE
DATA (C(I),I=417,420)/-3.273D+05, 9.312D+02,-1.178D-02, 6.306D-08/
DATA (D(I),I=417,420)/ 1.680D+07, 2.016D+02,-5.089D-04, 4.687D-10/
!          FOR EACH SUCCESSIVE LOWER STATE, DATA START AT N0:
!
DATA (N0(I),I=1,14)/1,57,109,157,201,241,277,309,337,361,381,397, &
&     409,417/
!     ..
!
!          check quantum number and return zero if no data available
!
IF (NLO.EQ.NUP .OR. NLO.GT.14 .OR. NUP.GT.15) THEN  
     STOP 'ERROR IN H-COLL RATES'  
END IF  
!
!          set pointer for start of coefficients
!
XNLOS = DBLE(NLO**2)  
XNUPS = DBLE(NUP**2)  
GLO = 2.0D0*XNLOS  
GUP = 2.0D0*XNUPS  

IF (NLO.EQ.1 .AND. NUP.EQ.2) THEN  
!
!     ly-alpha
!
     TR = T*RYD  
     TX = 1.D0  
     GAMFAC = BLA(0,1) + BLA(0,2)  

     DO K = 1,4  
          TX = TX*TR  
          GAMFAC = GAMFAC + (BLA(K,1)+BLA(K,2))*TX  
     END DO

     CUTOFF = BLA(5,2)*LOG(DAMP(1)*TR)*EXP(-DAMP(2)*TR)  
     TX = TX*TR  
     GAMFAC = GAMFAC + BLA(5,1)*TX + CUTOFF  
     HCOL = GAMFAC*8.631D-6/ (SQRT(T)*GLO)  
     RETURN  

ELSE IF (NLO.EQ.1 .AND. NUP.EQ.3) THEN  
!
!     ly-beta
!
     TR = T*RYD  
     TX = 1.D0  
     GAMFAC = BLB(0,1) + BLB(0,2) + BLB(0,3)  

     DO K = 1,6  
          TX = TX*TR  
          GAMFAC = GAMFAC + (BLB(K,1)+BLB(K,2)+BLB(K,3))*TX  
     END DO

     CUTOFF = (BLB(7,2)+BLB(7,3))*LOG(DAMP(1)*TR)*EXP(-DAMP(2)*TR)  
     GAMFAC = GAMFAC + CUTOFF  
     HCOL = GAMFAC*8.631D-6/ (SQRT(T)*GLO)  
     RETURN  

ELSE IF (NLO.EQ.2 .AND. NUP.EQ.3) THEN  
!
!     balmer-alpha (packed in advance)
!
     TR = T*RYD  
     TX = 1.D0  
     GAMFAC = BBA(0)  

     DO K = 1,6  
          TX = TX*TR  
          GAMFAC = GAMFAC + BBA(K)*TX  
     END DO

     HCOL = GAMFAC*8.631D-6/ (SQRT(T)*GLO)  
     RETURN  

ELSE  
!
!     other transitions according to italian mafia
!
!       index of 1st coef.
     K = N0(NLO) + (NUP-NLO-1)*4  
!
!          evaluate rates at specified temperatures
!
     IF (T.GT.60000) THEN  
          GAMFAC = 8.631D-6* (D(K)+T* (D(K+1)+T* (D(K+2)+T*D(K+ &
           3))))/SQRT(T)
     ELSE  
          GAMFAC = 8.631D-6* (C(K)+T* (C(K+1)+T* (C(K+2)+T*C(K+ &
           3))))/SQRT(T)
     END IF  

     HCOL = GAMFAC/GLO  
     RETURN  

END IF  

END
!
!-----------------------------------------------------------------------
!
FUNCTION HECOL(NL,NU,T)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!        for specified t, 1000.le.t.le.30000, gives de-excitation
!        rate constants in array c(iu,il), where 11.gt.iu.ge.1.
!        certain transitions are missing: (8,6), (9,6), (10,7),
!        (11,7), and (9,8).  the states are coded as follows:
!                1: 1 sing s
!                2: 2 trip s
!                3: 2 sing s
!                4: 2 trip p
!                5: 2 sing p
!                6: 3 trip s
!                7: 3 sing s
!                8: 3 trip p
!                9: 3 trip d
!               10: 3 sing d
!               11: 3 sing p
!
!         based on calculations by berrington and kingston (1987)
!         d.g.hummer, july, 1987
!         modified k. butler. for use in detail july 1987.
!         range extended beyond 30000k by setting collision strength to
!         value at 30000k.
!
!
!     .. scalar arguments ..
REAL(DP) ::  T,HECOL  
INTEGER(I4B) ::  NL,NU  
!     ..
!     .. local scalars ..
REAL(DP) ::  C0,CM,CP,D1,D2,D3,F1,F2,F3,ST,X,X0,XM,XP  
INTEGER(I4B) ::  I,I0,IM,IP,K  
!     ..
!     .. local arrays ..
REAL(DP) ::  G(8,51),G1(80),G2(72),G3(64),G4(56),G5(48), &
&                 G6(40),G7(16),G8(16),G9(16),TN(8),W(11)
INTEGER(I4B) ::  IUMAX(9),IUMIN(9),NSTART(9)  
!     ..
!     .. intrinsic functions ..
!INTRINSIC SQRT  
!     ..
!     .. equivalences ..
EQUIVALENCE (G(1,1),G1), (G(1,11),G2), (G(1,20),G3), (G(1,28),G4), &
&             (G(1,35),G5), (G(1,41),G6), (G(1,46),G7), &
&            (G(1,48),G8), (G(1,50),G9)
!     ..
!     .. save statement ..

SAVE  
!     ..
!     .. data statements ..
DATA IUMIN/2,3,4,5,6,7,8,10,10/  
DATA IUMAX/11,11,11,11,11,11,9,11,11/  
DATA NSTART/0,10,19,27,34,40,45,47,49/  
DATA TN/1.0D3,2.0D3,5.0D3,1.0D4,1.5D4,2.0D4,2.5D4,3.0D4/  
DATA W/1.0D0,3.0D0,1.0D0,9.0D0,3.0D0,3.0D0,1.0D0,9.0D0,1.5D1, &
&     5.0D0,3.0D0/
DATA G1/3.09D-2,4.95D-2,6.82D-2,7.24D-2,7.22D-2,7.16D-2,7.10D-2, &
&       7.03D-2,1.60D-2,2.36D-2,3.31D-2,3.83D-2,4.08D-2,4.26D-2, &
&       4.39D-2,4.49D-2,6.37D-3,1.01D-2,1.71D-2,2.44D-2,2.92D-2, &
&       3.30D-2,3.61D-2,3.84D-2,3.02D-3,5.29D-3,1.05D-2,1.63D-2, &
&       2.09D-2,2.52D-2,2.91D-2,3.24D-2,1.82D-2,1.87D-2,1.82D-2, &
&       1.81D-2,1.88D-2,1.95D-2,2.00D-2,2.03D-2,9.36D-3,9.42D-3, &
&       9.18D-3,9.62D-3,1.06D-2,1.16D-2,1.23D-2,1.28D-2,5.86D-3, &
&       6.19D-3,7.01D-3,8.40D-3,1.01D-2,1.17D-2,1.30D-2,1.38D-2, &
&       8.94D-4,1.67D-3,2.34D-3,2.60D-3,2.80D-3,2.93D-3,2.99D-3, &
&       3.01D-3,4.49D-3,5.05D-3,5.19D-3,5.42D-3,5.93D-3,6.41D-3, &
&       6.75D-3,6.95D-3,6.56D-4,1.10D-3,2.25D-3,4.24D-3,6.74D-3, &
&       9.21D-3,1.13D-2,1.29D-2/
DATA G2/1.01D+0,1.60D+0,2.24D+0,2.40D+0,2.33D+0,2.21D+0,2.09D+0, &
&       1.98D+0,3.30D+0,5.59D+0,1.36D+1,2.62D+1,3.69D+1,4.58D+1, &
&       5.30D+1,5.86D+1,3.17D-1,4.79D-1,7.60D-1,9.71D-1,1.06D+0, &
&       1.09D+0,1.10D+0,1.09D+0,2.73D+0,2.89D+0,2.67D+0,2.47D+0, &
&       2.47D+0,2.54D+0,2.61D+0,2.65D+0,4.81D-1,4.99D-1,4.62D-1, &
&       4.03D-1,3.67D-1,3.41D-1,3.19D-1,3.01D-1,1.79D+0,1.84D+0, &
&       1.84D+0,1.83D+0,1.90D+0,1.98D+0,2.04D+0,2.07D+0,4.09D-1, &
&       7.45D-1,1.37D+0,2.14D+0,2.87D+0,3.51D+0,4.00D+0,4.35D+0, &
&       2.42D-1,2.73D-1,2.98D-1,3.16D-1,3.34D-1,3.47D-1,3.53D-1, &
&       3.53D-1,4.34D-2,8.14D-2,1.42D-1,1.69D-1,1.80D-1,1.84D-1, &
&       1.85D-1,1.84D-1/
DATA G3/9.96D-1,1.22D+0,1.51D+0,1.71D+0,1.76D+0,1.76D+0,1.73D+0, &
&       1.68D+0,1.10D+0,2.97D+0,9.16D+0,1.81D+1,2.53D+1,3.11D+1, &
&       3.56D+1,3.89D+1,8.73D-1,8.94D-1,7.48D-1,5.88D-1,5.01D-1, &
&       4.44D-1,4.03D-1,3.70D-1,6.20D-1,6.17D-1,6.23D-1,6.47D-1, &
&       7.11D-1,7.86D-1,8.53D-1,9.03D-1,5.90D-1,6.03D-1,6.12D-1, &
&       5.60D-1,5.14D-1,4.78D-1,4.48D-1,4.22D-1,9.61D-2,1.67D-1, &
&       2.79D-1,3.56D-1,4.08D-1,4.42D-1,4.60D-1,4.68D-1,5.29D-1, &
&       6.44D-1,8.73D-1,1.25D+0,1.67D+0,2.06D+0,2.36D+0,2.59D+0, &
&       1.14D-1,1.88D-1,2.93D-1,3.60D-1,4.11D-1,4.56D-1,4.90D-1, &
&       5.14D-1/
DATA G4/1.92D+0,2.60D+0,3.57D+0,4.38D+0,4.64D+0,4.75D+0,4.72D+0, &
&       4.72D+0,7.43D+0,7.11D+0,6.25D+0,5.93D+0,6.13D+0,6.44D+0, &
&       6.69D+0,6.86D+0,6.51D-1,8.04D-1,1.07D+0,1.06D+0,9.68D-1, &
&       8.91D-1,8.26D-1,7.70D-1,8.59D+0,1.05D+1,1.23D+1,1.37D+1, &
&       1.53D+1,1.68D+1,1.81D+1,1.89D+1,3.53D+0,5.71D+0,9.81D+0, &
&       1.54D+1,2.13D+1,2.67D+1,3.10D+1,3.42D+1,1.33D+0,1.50D+0, &
&       1.60D+0,1.62D+0,1.67D+0,1.70D+0,1.70D+0,1.68D+0,3.41D-1, &
&       5.22D-1,7.96D-1,9.14D-1,9.45D-1,9.47D-1,9.33D-1,9.10D-1/
DATA G5/1.21D+0,1.46D+0,1.38D+0,1.16D+0,1.03D+0,9.37D-1,8.67D-1, &
&       8.09D-1,6.55D-1,7.39D-1,8.04D-1,8.84D-1,1.01D+0,1.14D+0, &
&       1.24D+0,1.32D+0,1.55D+0,1.77D+0,1.87D+0,1.76D+0,1.66D+0, &
&       1.58D+0,1.50D+0,1.42D+0,5.71D-1,8.47D-1,1.24D+0,1.52D+0, &
&       1.70D+0,1.80D+0,1.85D+0,1.85D+0,3.23D+0,3.96D+0,5.19D+0, &
&       7.23D+0,9.68D+0,1.20D+1,1.39D+1,1.54D+1,6.12D-1,9.70D-1, &
&       1.57D+0,2.22D+0,2.90D+0,3.53D+0,4.03D+0,4.41D+0/
DATA G6/3.60D+0,4.03D+0,3.68D+0,2.92D+0,2.44D+0,2.12D+0,1.88D+0, &
&       1.70D+0,0.00D+0,0.00D+0,0.00D+0,0.00D+0,0.00D+0,0.00D+0, &
&       0.00D+0,0.00D+0,0.00D+0,0.00D+0,0.00D+0,0.00D+0,0.00D+0, &
&       0.00D+0,0.00D+0,0.00D+0,3.11D+0,3.26D+0,3.10D+0,2.78D+0, &
&       2.55D+0,2.36D+0,2.20D+0,2.06D+0,5.54D-1,9.44D-1,1.40D+0, &
&       1.50D+0,1.45D+0,1.38D+0,1.31D+0,1.23D+0/
DATA G7/3.36D+0,3.80D+0,3.70D+0,3.20D+0,2.81D+0,2.52D+0,2.28D+0, &
&       2.09D+0,4.00D-1,8.69D-1,1.54D+0,1.76D+0,1.77D+0,1.72D+0, &
&       1.64D+0,1.56D+0/
DATA G8/1.22D+1,1.27D+1,1.20D+1,1.06D+1,9.58D+0,8.75D+0,8.06D+0, &
&       7.46D+0,2.72D+0,3.71D+0,4.85D+0,5.27D+0,5.26D+0,5.08D+0, &
&       4.84D+0,4.58D+0/
DATA G9/6.08D+0,8.78D+0,1.19D+1,1.32D+1,1.37D+1,1.38D+1,1.36D+1, &
&       1.32D+1,1.29D+0,2.64D+0,4.66D+0,5.55D+0,5.75D+0,5.70D+0, &
&       5.54D+0,5.33D+0/
!     ..
!
!          check that temperature is in bounds
!
IF (NL.LT.1 .OR. NL.GT.9) THEN  
     WRITE (*,FMT=*) 'NL OUT OF RANGE IN HECOL.  NU = ',NU,' NL = ',NL
     STOP  
END IF  

IF (NU.LT.IUMIN(NL) .OR. NU.GT.IUMAX(NL)) THEN  
     WRITE (*,FMT=*) 'NU OUT OF RANGE IN HECOL.  NU = ',NU,' NL = ',NL
     STOP  
END IF  

I = NSTART(NL) + NU - NL  
ST = SQRT(T)  

IF (T.LT.TN(1)) THEN  
     HECOL = 8.63D-6*G(1,I)/ (W(NL)*ST)  
     RETURN  
END IF  

IF (T.GT.TN(8)) THEN  
     HECOL = 8.63D-6*G(8,I)/ (W(NL)*ST)  
     RETURN  
END IF  
!
!          find interpolation region
!
K = 2  

   10 CONTINUE  

IF (TN(K).LT.T) THEN  
     K = K + 1  
     GO TO 10  
END IF  

IF (K.EQ.2) THEN  
     IM = 1  
     I0 = 2  
     IP = 3  
ELSE  
     IM = K - 2  
     I0 = K - 1  
     IP = K  
END IF  
!
!          calculate interpolation weights
!
X = T  
XM = TN(IM)  
X0 = TN(I0)  
XP = TN(IP)  
D1 = X0 - XM  
D2 = XP - X0  
D3 = XP - XM  
F1 = X - XM  
F2 = X - X0  
F3 = X - XP  
CM = F2*F3/ (D1*D3)  
C0 = -F1*F3/ (D1*D2)  
CP = F1*F2/ (D2*D3)  
!
!          interpolate averaged collision strengths and
!          evaluate downwards collision rate coefficients
!
HECOL = G(IM,I)*CM + G(I0,I)*C0 + G(IP,I)*CP  
HECOL = 8.63D-6*HECOL/ (W(NL)*ST)  

RETURN  

END
!
!***********************************************************************
!
! subroutines: miscellaneous
!
!***********************************************************************
!
FUNCTION ERFCM(X)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!---- complementary error function of a given value (x)
!---- modified (e. santolaya, 23/7/93) so that it returns not
!---- erfc(x) but erfc(x)*exp(x**2)
!
!     .. scalar arguments ..
REAL(DP) ::  X,ERFCM
!     ..
!     .. local scalars ..
REAL(DP) ::  A1,A2,A3,A4,A5,P,T  
!     ..
!     .. data statements ..
!
DATA P/.3275911D0/  
DATA A1,A2,A3,A4,A5/.254829592D0,-.284496736D0,1.421413741D0, &
&     -1.453152027D0,1.061405429D0/
!     ..

T = 1.D0/ (1.D0+P*X)  
ERFCM = T* (A1+T* (A2+T* (A3+T* (A4+T*A5))))  

IF (ERFCM.LT.0.D0) STOP 'ERROR IN ERFCM'  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION EXPIN(N,X)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     returns exponential integral(degree n>1) up to factor exp(-x)
!
!     warning!!! this function does not return e_n(x) but
!     e_n(x)*exp(x). the result must be multiplied by exp(-x) to
!     get e_n(x)
!
!     .. scalar arguments ..
REAL(DP) ::  X,EXPIN
INTEGER(I4B) ::  N  
!     ..
!     .. local scalars ..
REAL(DP) ::  SUMM,XN,XNQ  
INTEGER(I4B) ::  K  
!     ..
!     .. external functions ..
REAL(DP) ::  EXPIN1,FAKUL  
EXTERNAL EXPIN1,FAKUL  
!     ..

IF(X.EQ.0.) THEN
  EXPIN=1.D0/(FLOAT(N)-1.)
  RETURN
ENDIF  

IF (N.GE.20 .OR. (N.GE.8.AND.X.GT.5.D0)) THEN  
     XN = X + N  
     XNQ = XN*XN  
     SUMM = 1.D0 + N/XNQ + N* (N-2.D0*X)/XNQ/XNQ + N* (6.D0*X*X- &
      8.D0*N*X+N*N)/XNQ/XNQ/XNQ
     EXPIN = SUMM/XN  
     RETURN  
END IF  

SUMM = 0.D0  

DO K = 0,N - 2  
     SUMM = SUMM + (-1)**K*X**K*FAKUL(N-K-2)  
END DO  

SUMM = SUMM + (-1)** (N-1)*X** (N-1)*EXPIN1(X)  
EXPIN = SUMM/FAKUL(N-1)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION EXPIN1(X)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     returns first exponential integral up to factor exp(-x)
!
!     warning!!! this function does not return e_1(x) but
!     e_1(x)*exp(x). the result must be multiplied by exp(-x) to
!     get e_1(x)
!
!
!     .. scalar arguments ..
REAL(DP) ::  X,EXPIN1
!     ..
!     .. local scalars ..
REAL(DP) ::  A0,A1,A2,A3,A4,A5,B1,B2,B3,B4,C1,C2,C3,C4,SUM1, &
&                 SUM2
!     ..
!     .. intrinsic functions ..

!INTRINSIC EXP,LOG  
!     ..
!     .. data statements ..
DATA A0,A1,A2,A3,A4,A5/-.57721566D0,.99999193D0,-.24991055D0, &
&     .05519968D0,-.00976004D0,.00107857D0/
DATA B1,B2,B3,B4/8.5733287401D0,18.0590169730D0,8.6347608925D0, &
&     .2677737343D0/
DATA C1,C2,C3,C4/9.5733223454D0,25.6329561486D0,21.0996530827D0, &
&     3.9584969228D0/
!     ..

IF (X.LE.1.D0) THEN  
     EXPIN1 = A0 + X* (A1+X* (A2+X* (A3+X* (A4+X*A5)))) - LOG(X)  
     EXPIN1 = EXPIN1* (EXP(X))  
     RETURN  
END IF  

SUM1 = B4 + X* (B3+X* (B2+X* (B1+X)))  
SUM2 = C4 + X* (C3+X* (C2+X* (C1+X)))  
EXPIN1 = SUM1/ (SUM2*X)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION FAKUL(N)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
REAL(DP) :: FAKUL
INTEGER(I4B) ::  N  
!     ..
!     .. local scalars ..
REAL(DP) ::  XJ  
INTEGER(I4B) ::  I  
!     ..
IF (N.LT.0) STOP 'ARGUMENT OF FAKULTAET NEGATIVE!'  

IF (N.LE.1) THEN  
     FAKUL = 1.D0  
     RETURN  
END IF  

XJ = 1.D0  

DO I = 1,N  
  XJ = XJ*I
END DO  

FAKUL = XJ  

RETURN  
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE INTEGRA(A,SUMM,ND,NN,DEL,II)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  DEL  
INTEGER(I4B) ::  II,ND,NN  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(ND1),SUMM(ND1)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  L  
!     ..

DO L = 1,ND  

     IF (II.EQ.1) THEN  
          SUMM(L) = A(L)*DEL  
     ELSE  
          SUMM(L) = SUMM(L) + A(L)*DEL  
     END IF  

END DO  

IF (II.NE.NN) RETURN  

A(1:ND)=SUMM(1:ND)

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE LINREG(X,Y,N,XM,B,R)  

USE nlte_type  
USE nlte_dim  
IMPLICIT NONE  
!
!
!-----one-dimensional linear regression
!
!
!     .. scalar arguments ..
REAL(DP) ::  B,R,XM  
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  X(N),Y(N)  
!     ..
!     .. local scalars ..
REAL(DP) ::  SIGX,SIGY,SUMX,SUMXX,SUMXY,SUMY,SUMYY,XL,YL  
INTEGER(I4B) ::  L  
!     ..
!     .. intrinsic functions ..
!INTRINSIC DBLE,SQRT  
!     ..

SUMX = 0.D0  
SUMY = 0.D0  
SUMXX = 0.D0  
SUMXY = 0.D0  
SUMYY = 0.D0  

DO L = 1,N  
     XL = X(L)  
     YL = Y(L)  
     SUMX = SUMX + XL  
     SUMY = SUMY + YL  
     SUMXX = SUMXX + XL*XL  
     SUMYY = SUMYY + YL*YL  
     SUMXY = SUMXY + XL*YL  
END DO  

XM = SUMXY - SUMX*SUMY/DBLE(N)  
XM = XM/ (SUMXX-SUMX*SUMX/DBLE(N))  
B = (SUMY-XM*SUMX)/DBLE(N)  
SIGX = SQRT((SUMXX-SUMX*SUMX/DBLE(N))/DBLE(N-1))  
SIGY = SQRT((SUMYY-SUMY*SUMY/DBLE(N))/DBLE(N-1))  
R = XM*SIGX/SIGY  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE ODEINT(YSTART,NVAR,X1,X2,EPS,H1,HMIN,NOK,NBAD,DERIVS, &
&                  RKQC)

! FROM V10.0 ON: HERE AND IN CORRESPONDING ROUTINES, NMAX CHANGED TO 4
USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: KMAX,KOUNT,DXSAV,XP,YP  

IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B) ::  MAXSTP,NMAX  
REAL(DP) ::  TWO,ZERO,TINY  
PARAMETER (MAXSTP=10000,NMAX=4,TWO=2.D0,ZERO=0.D0,TINY=1.D-30)  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  EPS,H1,HMIN,X1,X2  
INTEGER(I4B) ::  NBAD,NOK,NVAR  
!     ..
!     .. array arguments ..
REAL(DP) ::  YSTART(NVAR)  
!     ..
!     .. subroutine arguments ..
EXTERNAL DERIVS,RKQC  
!     ..
!     .. local scalars ..
REAL(DP) ::  H,HDID,HNEXT,X,XSAV  
INTEGER(I4B) ::  I,NSTP  
!     ..
!     .. local arrays ..
REAL(DP) ::  DYDX(NMAX),Y(NMAX),YSCAL(NMAX)  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,SIGN  
!     ..

X = X1  
H = SIGN(H1,X2-X1)  
NOK = 0  
NBAD = 0  
KOUNT = 0  

DO I = 1,NVAR  
     Y(I) = YSTART(I)  
END DO  

XSAV = X - DXSAV*TWO  

ITLOOP: DO NSTP = 1,MAXSTP  

     CALL DERIVS(X,Y,DYDX)  

     DO I = 1,NVAR  
          YSCAL(I) = ABS(Y(I)) + ABS(H*DYDX(I)) + TINY  
     END DO

     IF (KMAX.GT.0) THEN  

          IF (ABS(X-XSAV).GT.ABS(DXSAV)) THEN  
               IF (KOUNT.LT.KMAX-1) THEN  
                    KOUNT = KOUNT + 1  
                    XP(KOUNT) = X  
                    DO I = 1,NVAR  
                         YP(I,KOUNT) = Y(I)  
                    END DO

                    XSAV = X  
               END IF  
          END IF  
     END IF  

     IF ((X+H-X2)* (X+H-X1).GT.ZERO) H = X2 - X  

     CALL RKQC(Y,DYDX,NVAR,X,H,EPS,YSCAL,HDID,HNEXT,DERIVS)  

     IF (HDID.EQ.H) THEN  
          NOK = NOK + 1  
     ELSE  
          NBAD = NBAD + 1  
     END IF  

     IF ((X-X2)* (X2-X1).GE.ZERO) THEN  
          DO I = 1,NVAR  
               YSTART(I) = Y(I)  
          END DO

          IF (KMAX.NE.0) THEN  
               KOUNT = KOUNT + 1  
               XP(KOUNT) = X  
               DO I = 1,NVAR  
                    YP(I,KOUNT) = Y(I)  
               END DO
          END IF  

          RETURN  
     END IF  

     IF (ABS(HNEXT).LT.HMIN) STOP 'STEPSIZE SMALLER THAN MINIMUM.'  

     H = HNEXT  

END DO ITLOOP  

STOP 'TOO MANY STEPS.'  

END
!
!-----------------------------------------------------------------------

SUBROUTINE RKQC(Y,DYDX,N,X,HTRY,EPS,YSCAL,HDID,HNEXT,DERIVS)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B) ::  NMAX  
REAL(DP) ::  FCOR,ONE,SAFETY,ERRCON  
PARAMETER (NMAX=4,FCOR=.0666666667D0,ONE=1.D0,SAFETY=0.9D0, &
&          ERRCON=6.D-4)
!     ..
!     .. scalar arguments ..
REAL(DP) ::  EPS,HDID,HNEXT,HTRY,X  
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  DYDX(N),Y(N),YSCAL(N)  
!     ..
!     .. subroutine arguments ..
EXTERNAL DERIVS  
!     ..
!     .. local scalars ..
REAL(DP) ::  ERRMAX,H,HH,PGROW,PSHRNK,XSAV  
INTEGER(I4B) ::  I  
!     ..
!     .. local arrays ..
REAL(DP) ::  DYSAV(NMAX),YSAV(NMAX),YTEMP(NMAX)  
!     ..
!     .. external subroutines ..
EXTERNAL RK4  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,MAX  
!     ..

PGROW = -0.20D0  
PSHRNK = -0.25D0  
XSAV = X  

DO I = 1,N  
     YSAV(I) = Y(I)  
     DYSAV(I) = DYDX(I)  
END DO  

H = HTRY  

   20 CONTINUE  

HH = 0.5D0*H  

CALL RK4(YSAV,DYSAV,N,XSAV,HH,YTEMP,DERIVS)  

X = XSAV + HH  

CALL DERIVS(X,YTEMP,DYDX)  
CALL RK4(YTEMP,DYDX,N,X,HH,Y,DERIVS)  

X = XSAV + H  

IF (X.EQ.XSAV) PRINT *,'STEPSIZE NOT SIGNIFICANT IN RKQC.'  

CALL RK4(YSAV,DYSAV,N,XSAV,H,YTEMP,DERIVS)  
ERRMAX = 0.D0  

DO I = 1,N  
     YTEMP(I) = Y(I) - YTEMP(I)  
     ERRMAX = MAX(ERRMAX,ABS(YTEMP(I)/YSCAL(I)))  
END DO  

ERRMAX = ERRMAX/EPS  

IF (ERRMAX.GT.ONE) THEN  
     H = SAFETY*H* (ERRMAX**PSHRNK)  
     GO TO 20  
ELSE  
     HDID = H  
     IF (ERRMAX.GT.ERRCON) THEN  
          HNEXT = SAFETY*H* (ERRMAX**PGROW)  
     ELSE  
          HNEXT = 4.D0*H  
     END IF  
END IF  

DO I = 1,N  
     Y(I) = Y(I) + YTEMP(I)*FCOR  
END DO  

RETURN  
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE RK4(Y,DYDX,N,X,H,YOUT,DERIVS)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B) ::  NMAX  
PARAMETER (NMAX=4)  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  H,X  
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  DYDX(N),Y(N),YOUT(N)  
!     ..
!     .. subroutine arguments ..
EXTERNAL DERIVS  
!     ..
!     .. local scalars ..
REAL(DP) ::  H6,HH,XH  
INTEGER(I4B) ::  I  
!     ..
!     .. local arrays ..
REAL(DP) ::  DYM(NMAX),DYT(NMAX),YT(NMAX)  
!     ..

HH = H*0.5D0  
H6 = H/6.D0  
XH = X + HH  

DO I = 1,N  
     YT(I) = Y(I) + HH*DYDX(I)  
END DO  

CALL DERIVS(XH,YT,DYT)  

DO I = 1,N  
     YT(I) = Y(I) + HH*DYT(I)  
END DO  

CALL DERIVS(XH,YT,DYM)  


DO I = 1,N  
     YT(I) = Y(I) + H*DYM(I)  
     DYM(I) = DYT(I) + DYM(I)  
END DO  


CALL DERIVS(X+H,YT,DYT)  


DO I = 1,N  
     YOUT(I) = Y(I) + H6* (DYDX(I)+DYT(I)+2.D0*DYM(I))  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE SORT(N,RA)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  RA(N)  
!     ..
!     .. local scalars ..
REAL(DP) ::  RRA  
INTEGER(I4B) ::  I,IR,J,L  
!     ..

L = N/2 + 1  
IR = N  

   10 CONTINUE  

IF (L.GT.1) THEN  
     L = L - 1  
     RRA = RA(L)  
ELSE  
     RRA = RA(IR)  
     RA(IR) = RA(1)  
     IR = IR - 1  

     IF (IR.EQ.1) THEN  
          RA(1) = RRA  
          RETURN  
     END IF  
END IF  

I = L  
J = L + L  

   20 CONTINUE  

IF (J.LE.IR) THEN  

     IF (J.LT.IR) THEN  
          IF (RA(J).LT.RA(J+1)) J = J + 1  
     END IF  

     IF (RRA.LT.RA(J)) THEN  
          RA(I) = RA(J)  
          I = J  
          J = J + J  
     ELSE  
          J = IR + 1  
     END IF  

     GO TO 20  

END IF  

RA(I) = RRA  

GO TO 10  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE SORT1(N,RA)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     sorts for absolute values
!                                                                       
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  RA(N)  
!     ..
!     .. local scalars ..
REAL(DP) ::  RRA  
INTEGER(I4B) ::  I,IR,J,L  
!     ..

L = N/2 + 1  
IR = N  

   10 CONTINUE  

IF (L.GT.1) THEN  
     L = L - 1  
     RRA = RA(L)  
ELSE  
     RRA = RA(IR)  
     RA(IR) = RA(1)  
     IR = IR - 1  

     IF (IR.EQ.1) THEN  
          RA(1) = RRA  
          RETURN  
     END IF  
END IF  

I = L  
J = L + L  

   20 CONTINUE  

IF (J.LE.IR) THEN  

     IF (J.LT.IR) THEN  
          IF (ABS(RA(J)).LT.ABS(RA(J+1))) J = J + 1  
     END IF  

     IF (ABS(RRA).LT.ABS(RA(J))) THEN  
          RA(I) = RA(J)  
          I = J  
          J = J + J  
     ELSE  
          J = IR + 1  
     END IF  

     GO TO 20  

END IF  

RA(I) = RRA  

GO TO 10  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE SPLINE(X,Y,A,B,C,N)  

Use nlte_type
USE nlte_dim
IMPLICIT NONE
!
!-----computation of cubic spline coefficients a,b,c
!-----for approximation of a data array (x,y) such that
!-----f(x)=y(i)+del*(a(i)+del*(b(i)+c(i)*del)) in the
!-----interval x(i)-x-x(i+1), where del=x-x(i).
!-----vanishing curvature at the boundaries is assumed:
!-----f"(x(i))=f"(x(n))=0
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(N),B(N),C(N),X(N),Y(N)  
!     ..
!     .. local scalars ..
REAL(DP) ::  X1,Z1  
INTEGER(I4B) ::  I  
!     ..
!     .. local arrays ..
REAL(DP) ::  XX(1000)  
!     ..

IF (N.GT.1000) STOP ' N IN SPLINE TOO LARGE'  

DO I = 1,N  
     XX(I) = X(I)  
END DO  

X1 = X(1)  

DO I = 2,N  
     Z1 = X(I)  
     X(I) = Z1 - X1  
     X1 = Z1  
END DO  

DO I = 2,N - 1  
     A(I) = X(I+1)/ (X(I)+X(I+1))  
     B(I) = 1.D0 - A(I)  
     C(I) = 6.D0* ((Y(I+1)-Y(I))/X(I+1)- (Y(I)-Y(I-1))/X(I))/ (X(I)+X(I+1))
END DO  

B(2) = 2.D0  

DO I = 3,N - 1  
     X1 = B(I)/B(I-1)  
     B(I) = 2.D0 - X1*A(I-1)  
     C(I) = C(I) - X1*C(I-1)  
END DO  

C(N-1) = C(N-1)/B(N-1)  

DO I = N - 2,2,-1  
     C(I) = (C(I)-A(I)*C(I+1))/B(I)  
END DO  

C(1) = 0.D0  
C(N) = 0.D0  

DO I = 1,N - 1  
     A(I) = (Y(I+1)-Y(I))/X(I+1) - X(I+1)* (2.D0*C(I)+C(I+1))/ 6.D0
     B(I) = .5D0*C(I)  
     C(I) = (C(I+1)-C(I))/X(I+1)/6.D0  
END DO  

DO I = 1,N  
     X(I) = XX(I)  
END DO  

RETURN  

END
!
!***********************************************************************
!
! subroutines: algebraic ones
!
!***********************************************************************
!
SUBROUTINE INV(N,NDIM,A)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     scaled matrix inversion
!
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N,NDIM  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(NDIM,NDIM)  
!     ..
!     .. local scalars ..
REAL(DP) ::  QI  
INTEGER(I4B) ::  I,II  
!     ..
!     .. local arrays ..
REAL(DP) ::  Q(200)  
!     ..
!     .. external subroutines ..
EXTERNAL MATINV  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,MAX  
!     ..

DO I = 1,N  
     QI = 0.D0  

     DO II = 1,N  
          QI = MAX(QI,ABS(A(II,I)))  
     END DO

     Q(I) = QI  
     DO II = 1,N  
          A(II,I) = A(II,I)/QI  
     END DO
END DO  

CALL MATINV(A,N,NDIM)  

DO II = 1,N  
     QI = Q(II)  
     DO I = 1,N  
          A(II,I) = A(II,I)/QI  
     END DO
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE MATINV(A,N,NO)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!
!-----matinv executes matrix inversion by lu decomposition
!-----inversion is accomplished in place,
!-----and the original matrix is replaced by its inverse
!-----note that n must be smaller than no
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N,NO  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(NO,NO)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DIV,SUMM  
INTEGER(I4B) ::  I,II,IM1,IP1,J,JJ,JM1,JP1,K,KO,L  
!     ..
!     .. intrinsic functions ..
!INTRINSIC MAX0  
!     ..
ILOOP: DO I = 2,N  
     IM1 = I - 1  

JLOOP1: DO J = 1,IM1  
          JM1 = J - 1  

          IF (A(J,J).EQ.0.0D0) A(J,J) = 1.0D-20  

          DIV = A(J,J)  
          SUMM = 0.0D+00  
          IF (JM1.LT.1) GO TO 20  

          DO L = 1,JM1  
               SUMM = SUMM + A(I,L)*A(L,J)  
          END DO

   20           CONTINUE  

          A(I,J) = (A(I,J)-SUMM)/DIV  

     END DO JLOOP1

     DO J = I,N  
          SUMM = 0.0D+00  

          DO L = 1,IM1  
               SUMM = SUMM + A(I,L)*A(L,J)  
          END DO
          A(I,J) = A(I,J) - SUMM  
     END DO

END DO ILOOP  

IILOOP1: DO II = 2,N  
     I = N + 2 - II  
     IM1 = I - 1  
     IF (IM1.LT.1) CYCLE  

     DO JJ = 1,IM1  
          J = I - JJ  
          JP1 = J + 1  
          SUMM = 0.0D+00  

          IF (JP1.GT.IM1) GO TO 80  

          DO K = JP1,IM1  
               SUMM = SUMM + A(I,K)*A(K,J)  
          END DO

   80           CONTINUE  

          A(I,J) = -A(I,J) - SUMM  

     END DO
END DO IILOOP1 

IILOOP2: DO II = 1,N  
     I = N + 1 - II  

     IF (A(I,I).EQ.0.0D0) A(I,I) = 1.0D-20  

     DIV = A(I,I)  
     IP1 = I + 1  
     IF (IP1.GT.N) GO TO 130  

     DO JJ = IP1,N  
          J = N + IP1 - JJ  
          SUMM = 0.0D+00  
          DO K = IP1,J  
               SUMM = SUMM + A(I,K)*A(K,J)  
          END DO

          A(I,J) = -SUMM/DIV  
     END DO

  130      CONTINUE  

     A(I,I) = 1.D0/A(I,I)  
END DO IILOOP2 

BACK: DO I = 1,N  
JLOOP2: DO J = 1,N  
          KO = MAX0(I,J)  
          IF (KO.EQ.J) GO TO 170  
          SUMM = 0.0D+00  

  150           CONTINUE  

          DO K = KO,N  
               SUMM = SUMM + A(I,K)*A(K,J)  
          END DO

          GO TO 180  

  170           CONTINUE  

          SUMM = A(I,KO)  
          IF (KO.EQ.N) GO TO 180  
          KO = KO + 1  

          GO TO 150  

  180           CONTINUE  

          A(I,J) = SUMM  
     END DO JLOOP2

END DO BACK 

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE INVTRI(A,B,C,Q,N)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     solution of tridiagonal equation system with
!     tridiag. matrix  -a(l),b(l),-c(l)  ,
!     right side: vector q(l),solution overwrites q
!
!     for this program package we assume without proving that n > 1
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND=ID_NDEPT  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(ND),B(ND),C(ND),Q(ND)  
!     ..
!     .. local scalars ..
REAL(DP) ::  H  
INTEGER(I4B) ::  I,NI  
!     ..

C(1) = C(1)/B(1)  
Q(1) = Q(1)/B(1)  

!      if (n.eq.1) return

DO I = 2,N - 1  
     H = B(I) - A(I)*C(I-1)  
     C(I) = C(I)/H  
     Q(I) = (Q(I)+Q(I-1)*A(I))/H  
END DO  

Q(N) = (Q(N)+Q(N-1)*A(N))/ (B(N)-C(N-1)*A(N))  

DO I = 1,N - 1  
     NI = N - I  
     Q(NI) = Q(NI) + C(NI)*Q(NI+1)  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE INVTRI3(TA,TB,TC,A,K,N)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
INTEGER(I4B) ::  K,N  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(ID_NDEPT,ID_NDEPT),TA(ID_NDEPT),TB(ID_NDEPT), &
&                 TC(ID_NDEPT)
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,J,L  
!     ..
!     .. local arrays ..
REAL(DP) ::  D(0:100),DIA(100),E(101)  
!     ..

IF (N.GT.100) STOP 'TOO MANY GRID POINTS IN DIAG'  

D(0) = 0.D0  

DO L = 1,K  
     D(L) = -TC(L)/ (TB(L)+TA(L)*D(L-1))  
END DO  

E(K+1) = 0.D0  

DO L = K,1,-1  
     E(L) = -TA(L)/ (TB(L)+TC(L)*E(L+1))  
END DO  

DO L = 1,K  
     DIA(L) = 1.D0/ ((1.D0-D(L)*E(L+1))* (TB(L)+TA(L)*D(L-1)))  
     A(L,L) = DIA(L)  
END DO  

DO J = 2,K  
     DO I = J - 1,1,-1  
          A(I,J) = A(I+1,J)*D(I)  
     END DO
END DO  

DO J = 1,K - 1  
     DO I = J + 1,K  
          A(I,J) = A(I-1,J)*E(I)  
     END DO
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE LUBKSB(A,N,NP,INDX,B)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N,NP  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(NP,NP),B(N)  
INTEGER(I4B) ::  INDX(N)  
!     ..
!     .. local scalars ..
REAL(DP) ::  SUMM  
INTEGER(I4B) ::  I,II,J,LL  
!     ..

II = 0  

DO I = 1,N  
     LL = INDX(I)  
     SUMM = B(LL)  
     B(LL) = B(I)  

     IF (II.NE.0) THEN  
          DO J = II,I - 1  
               SUMM = SUMM - A(I,J)*B(J)  
          END DO
     ELSE IF (SUMM.NE.0.D0) THEN  
          II = I  
     END IF  

     B(I) = SUMM
END DO  

DO I = N,1,-1  
     SUMM = B(I)  

     DO J = I + 1,N  
          SUMM = SUMM - A(I,J)*B(J)  
     END DO

     B(I) = SUMM/A(I,I)  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE LUDCMP(A,N,NP,INDX,D)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. parameters ..
INTEGER(I4B) ::  NMAX  
PARAMETER (NMAX=ID_LLEVS+1)  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  D  
INTEGER(I4B) ::  N,NP  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(NP,NP)  
INTEGER(I4B) ::  INDX(N)  
!     ..
!     .. local scalars ..
REAL(DP) ::  AAMAX,DUM,SUMM  
INTEGER(I4B) ::  I,IMAX,J,K  
!     ..
!     .. local arrays ..
REAL(DP) ::  VV(NMAX)  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS  
!     ..

D = 1.D0  

DO I = 1,N  
     AAMAX = 0.D0  
     DO J = 1,N  
          IF (ABS(A(I,J)).GT.AAMAX) AAMAX = ABS(A(I,J))  
     END DO
     VV(I) = 1.D0/AAMAX  
END DO  

JLOOP: DO J = 1,N  

     DO I = 1,J - 1  
          SUMM = A(I,J)  

          DO K = 1,I - 1  
               SUMM = SUMM - A(I,K)*A(K,J)  
          END DO

          A(I,J) = SUMM  
     END DO

     AAMAX = 0.D0  

     DO I = J,N  
          SUMM = A(I,J)  

          DO K = 1,J - 1  
               SUMM = SUMM - A(I,K)*A(K,J)  
          END DO

          A(I,J) = SUMM  
          DUM = VV(I)*ABS(SUMM)  

          IF (DUM.GE.AAMAX) THEN  
               IMAX = I  
               AAMAX = DUM  
          END IF  
     END DO

     IF (J.NE.IMAX) THEN  

          DO K = 1,N  
               DUM = A(IMAX,K)  
               A(IMAX,K) = A(J,K)  
               A(J,K) = DUM  
          END DO

          D = -D  
          VV(IMAX) = VV(J)  
     END IF  

     INDX(J) = IMAX  

     IF (A(J,J).EQ.0.) STOP ' MATRIX SINGULAR'  

     IF (J.NE.N) THEN  
          DUM = 1.D0/A(J,J)  
          DO I = J + 1,N  
               A(I,J) = A(I,J)*DUM  
          END DO
     END IF  

END DO JLOOP  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE MADD(A,B,N,NDIM)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     a(i,j)=a(i,j)+b(i,j)
!
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N,NDIM  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(NDIM,NDIM),B(NDIM,NDIM)  
!     ..

A(1:N,1:N) = A(1:N,1:N) + B(1:N,1:N)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE MDMV(A,B,N,NDIM)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     matrix a(voll)=: matrix b(diag) * matrix a(voll)
!
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N,NDIM  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(NDIM,NDIM),B(NDIM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,J  
!     ..

DO I = 1,N  
     DO J = 1,N  
          A(I,J) = B(I)*A(I,J)  
     END DO
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE MDV(A,V,N)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     matrix(diagonal) * vector v
!     result overwrites v
!
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(N),V(N)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I  
!     ..

DO I = 1,N  
     V(I) = A(I)*V(I)  
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE MVMD(A,B,N,NDIM)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     matrix a(voll)=: matrix a(voll) * matrix b(diag)
!
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N,NDIM  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(NDIM,NDIM),B(NDIM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,J  
!     ..

DO J = 1,N  
     DO I = 1,N  
          A(I,J) = A(I,J)*B(J)  
     END DO
END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE MVV(WX,B,W,JMAX,JMM,JP)  

USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     matrix(voll) b *vektor w= vektor wx
!     format: wx(jmax)=b(jmax,jmm)*w(jmm)
!
!     .. scalar arguments ..
INTEGER(I4B) ::  JMAX,JMM,JP  
!     ..
!     .. array arguments ..
REAL(DP) ::  B(JP,JP),W(JP),WX(JP)  
!     ..
!     .. local scalars ..
REAL(DP) ::  WXI  
INTEGER(I4B) ::  I,K  
!     ..

DO I = 1,JMAX  
     WXI = 0.D0  

     DO K = 1,JMM  
          WXI = WXI + B(I,K)*W(K)  
     END DO

     WX(I) = WXI  

END DO  

RETURN  

END
!
!-----------------------------------------------------------------------
!
!
SUBROUTINE VADD(A,B,N)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     vector addition a=a+b
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(N),B(N)  
!     ..

A = A + B

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE VSUB(A,B,N)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!
!     a(i)=a(i)-b(i)
!
!
!     .. scalar arguments ..
INTEGER(I4B) ::  N  
!     ..
!     .. array arguments ..
REAL(DP) ::  A(N),B(N)  
!     ..

A = A - B

RETURN  

END
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cc                                                                  ccc
!cc      subroutines and functions for new atomic data (partly not used)
!cc      (NOT ADAPTED FOR FAST F90)                                  ccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
FUNCTION PVR(Y)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!         generates van regemorter function for neutrals
!
!     .. scalar arguments ..
REAL(DP) ::  Y,PVR
!     ..
!     .. local scalars ..
REAL(DP) ::  PPL,YYL  
INTEGER(I4B) ::  I,IL  
!     ..
!     .. local arrays ..
REAL(DP) ::  PL(11),YL(11)  
!     ..
!     .. intrinsic functions ..
!INTRINSIC LOG,LOG10,SQRT  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA YL/1.0D0,0.60206D0,0.30103D0,0.0D0,-0.39794D0,-0.69897D0, &
&     -1.0D0,-1.39794D0,-1.69897D0,-2.0D0,-2.30103D0/
DATA PL/-1.6383D0,-1.398D0,-1.201D0,-1.0D0,-0.68D0,-0.4802D0, &
&     -0.3072D0,-0.1203D0,-0.01954D0,0.06446D0,0.11442D0/
!     ..
IF (Y.GT.1.0D1) THEN  

     PVR = 0.066D0/SQRT(Y)  
ELSE IF (Y.LE.5.0D-3) THEN  

     PVR = 0.2757D0* (-0.57722D0-LOG(Y))  
ELSE  
     YYL = LOG10(Y)  
     DO I = 2,11  
          IF (YYL.GE.YL(I)) THEN  
               IL = I  
               EXIT   
          END IF  
     ENDDO
        
     PPL = PL(IL) + (YYL-YL(IL))* (PL(IL-1)-PL(IL))/ (YL(IL-1)-YL( &
      IL))
     PVR = 10.0D0**PPL  

END IF  

RETURN  
END
!
!-----------------------------------------------------------------------
!
FUNCTION PFTN(T,U,EPS)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!          basic function for classical collisional excitation
!
!     .. scalar arguments ..
REAL(DP) ::  EPS,T,U,PFTN
!     ..
!     .. local scalars ..
REAL(DP) ::  RAT1,RAT2,RES1,RES2  
!     ..
!     .. external functions ..
REAL(DP) ::  EXPINO  
EXTERNAL EXPINO  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,EXP  
!     ..
!     .. save statement ..
SAVE  
!     ..
IF (ABS(EPS).LE.1.0E-35) THEN  
     RAT1 = U*1.579D5/T  
     RES1 = EXPINO(RAT1)  

     PFTN = (2.0D0+RAT1)*EXP(RAT1)*RES1 - (1.0D0+ (1.0D0/RAT1))  
ELSE  
     RAT1 = (U+ABS(EPS))*1.579D5/T  
     RAT2 = U*1.579D5/T  
     RES1 = EXPINO(RAT1)  
     RES2 = EXPINO(RAT2)  
     PFTN = ((U/ABS(EPS))+1.0D0)*EXP(RAT1)*RES1 - ((U/ABS(EPS))- &
      1.0D0)*EXP(RAT2)*RES2 - 1.0D0/RAT2

END IF  

RETURN  
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE COLL21(T,KL,KU,C)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     generates collisional rate coefficients among the 19 states of
!     helium with n = 1, 2, 3, and 4, using rates evaluated from the
!     cross sections calculated by berrington and kingston (j. phys.b.
!     20, 6631(1987)). collisional rate coefficients have been evaluated
!     numerically by p.j.storey from the unpublished computer output
!     files of berrington and kingston.
!
!     the states included in the calculation are labelled sequentially
!     in order of increasing energy:
!                           1          1 sing s
!                           2          2 trip s
!                           3          2 sing s
!                           4          2 trip p
!                           5          2 sing p
!                           6          3 trip s
!                           7          3 sing s
!                           8          3 trip p
!                           9          3 trip d
!                          10          3 sing d
!                          11          3 sing p
!                          12          4 trip s
!                          13          4 sing s
!                          14          4 trip p
!                          15          4 trip d
!                          16          4 sing d
!                          17          4 trip f
!                          18          4 sing f
!                          19          4 sing p
!
!     this ordering differs slightly from that of berrington and
!     kingston, in which 15 and 16, and 17 and 18, were interchanged.
!
!     .. scalar arguments ..
REAL(DP) ::  C,T  
INTEGER(I4B) ::  KL,KU  
!     ..
!     .. local scalars ..
REAL(DP) ::  C1,C2,HALF,ONE,TEMP,TFAC,TWO,XXX,ZERO  
INTEGER(I4B) ::  I,IR,J,JJ,N1,NF,NT,NTM2  
!     ..
!     .. local arrays ..
REAL(DP) ::  A(929),B(10),ENER(19),S(19),STWT(19)  
INTEGER(I4B) ::  KEY(4,4,2),L(19),N(19),NSTART(172)  
CHARACTER IDEN(19)*14  
!     ..
!     .. intrinsic functions ..
!INTRINSIC LOG10,SQRT  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA N/1,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4/  
DATA L/0,0,0,1,1,0,0,1,2,2,1,0,0,1,2,2,3,3,1/  
DATA S/0,1,0,1,0,1,0,1,1,0,0,1,0,1,1,0,1,0,0/  
DATA KEY/1,3,7,13,0,5,11,19,0,0,10,16,0,0,0,18,0,2,6,12,0,4,8,14, &
&     0,0,9,5,0,0,0,17/
DATA ENER/0.0D0,19.8198D0,20.6160D0,20.96432D0,21.2182D0, &
&     22.7187D0,22.9206D0,23.00731D0,23.0739D0,23.0743D0,23.0873D0, &
&     23.5942D0,23.6738D0,23.7081D0,23.7363D0,23.7366D0,23.7373D0, &
&     23.7373D0,23.7423D0/
DATA STWT/1.0D0,3.0D0,1.0D0,9.0D0,3.0D0,3.0D0,1.0D0,9.0D0,1.5D1, &
&     5.0D0,3.0D0,3.0D0,1.0D0,9.0D0,1.5D1,5.0D0,2.1D1,7.0D0,3.0D0/
DATA IDEN/'HE I 1 SING S','HE I 2 TRIP S','HE I 2 SING S', &
&     'HE I 2 TRIP P','HE I 2 SING P','HE I 3 TRIP S', &
&     'HE I 3 SING S','HE I 3 TRIP P','HE I 3 TRIP D', &
&     'HE I 3 SING D','HE I 3 SING P','HE I 4 TRIP S', &
&     'HE I 4 SING S','HE I 4 TRIP P','HE I 4 TRIP D', &
&     'HE I 4 SING D','HE I 4 TRIP F','HE I 4 SING F', &
&     'HE I 4 SING P'/
DATA NSTART/1,6,11,16,20,28,32,40,44,52,57,62,67,72,77,82,88,92, &
&     98,104,110,114,120,125,129,135,139,147,151,157,164,170,177, &
&     183,190,195,202,208,213,220,225,232,236,243,247,251,260,266, &
&     273,278,285,290,300,304,309,316,320,324,329,333,338,343,347, &
&     352,357,362,367,372,376,382,386,391,395,401,405,410,414,421, &
&     425,431,435,440,445,449,454,459,465,470,475,480,487,491,497, &
&     503,508,515,520,525,530,536,542,547,552,559,564,571,576,581, &
&     587,592,598,603,608,613,617,623,630,635,642,646,650,655,660, &
&     666,671,677,683,689,695,702,707,713,718,723,728,732,737,741, &
&     745,750,754,759,765,771,777,782,789,796,801,805,810,815,819, &
&     824,831,837,844,850,856,861,868,873,877,882,890,895,905,909, &
&     913,920,925,930/

DATA (A(I),I=1,95)/ &
& 1.7339D-07, 2.7997D-08,-1.3812D-08, 2.6639D-09, 1.7776D-09,&
& 2.9820D-07, 7.5210D-08,-3.5975D-09, 3.2270D-09, 1.5245D-09,&
& 1.5601D-05, 1.5340D-06,-2.2122D-06,-1.1073D-07, 1.9249D-07,&
& 2.3682D-08, 1.0638D-08, 2.0959D-09, 2.8381D-10, 3.0497D-05,&
& 1.9252D-05, 6.3109D-06, 6.9098D-07,-2.8039D-07,-2.1128D-07,&
&-1.2192D-07,-4.4417D-08, 1.3896D-06, 1.5715D-07,-8.4358D-08,&
&-2.8800D-08, 5.9599D-08, 3.4756D-08, 1.2183D-08, 3.7999D-09,&
& 8.5500D-10,-3.9428D-10,-5.3999D-10,-2.2962D-10, 2.2510D-06,&
& 6.1436D-07,-1.2437D-07,-7.1718D-08, 5.9026D-05, 3.8150D-05,&
& 1.1426D-05, 9.2886D-07,-6.4827D-07,-4.4270D-07,-1.8611D-07,&
&-5.6403D-08, 5.8752D-06, 2.5167D-06, 2.0787D-07,-2.3353D-07,&
&-8.9900D-08, 5.6334D-08, 2.9313D-09, 1.7775D-09, 5.5494D-10,&
&-5.8914D-10, 8.1939D-06, 2.4014D-07, 3.8681D-07, 2.8446D-07,&
&-6.1936D-08, 1.8173D-06,-4.7530D-07,-6.6432D-08, 5.4898D-08,&
&-1.2377D-08, 2.0732D-05, 6.5991D-07, 1.5840D-06, 4.9920D-07,&
&-1.8332D-07, 3.2273D-06,-4.9880D-07,-2.4929D-07, 1.1964D-07,&
&-2.1996D-08, 9.7096D-08, 1.3557D-08, 7.4404D-09, 1.3858D-09,&
&-1.3778D-09,-8.4885D-10, 3.5068D-06,-5.0675D-07,-1.2252D-07,&
& 6.0514D-08, 7.0524D-06, 1.4454D-06, 8.3966D-07, 2.7203D-07/
DATA (A(I),I=96,190)/ &
&-2.3854D-08,-8.6693D-08, 7.1193D-06, 8.3111D-09,-9.3916D-07,&
&-2.7944D-08, 1.7803D-07,-3.9216D-08, 9.2760D-06, 2.4761D-06,&
& 1.0095D-06, 3.4039D-07,-8.6900D-08,-1.1156D-07, 2.5036D-05,&
&-5.7791D-06,-1.8197D-06, 1.0630D-06, 9.8746D-09, 3.1048D-09,&
& 1.2172D-09, 1.8411D-10,-1.5835D-10,-1.4443D-10, 1.9240D-06,&
& 1.5012D-07, 7.9741D-08, 2.9323D-08,-1.2796D-08, 5.0457D-07,&
&-5.5146D-08,-2.7506D-08, 1.4341D-09, 1.3321D-05, 2.6960D-06,&
& 2.8059D-07, 1.7548D-08,-2.3623D-07,-1.4619D-07, 1.5294D-06,&
&-9.4925D-08,-1.1347D-07, 8.1980D-09, 1.3324D-04, 7.8068D-05,&
& 2.9238D-05, 4.2718D-06,-1.6556D-06,-1.4529D-06,-8.6046D-07,&
&-2.1062D-07, 2.8510D-06,-4.7936D-07,-2.4042D-07, 6.2333D-08,&
& 1.3493D-09, 3.5113D-10,-4.7269D-11, 2.4872D-11,-9.7136D-12,&
&-1.4217D-11, 1.5680D-06, 8.9272D-07, 2.8313D-07, 5.2456D-08,&
&-2.3404D-08,-2.7304D-08,-8.4539D-09, 1.7967D-07, 6.2133D-08,&
&-3.2814D-09,-3.7299D-09,-1.9587D-09,-1.6685D-09, 1.2707D-05,&
& 7.5029D-06, 2.8330D-06, 6.7478D-07,-1.5012D-07,-2.3916D-07,&
&-8.9594D-08, 7.6160D-07, 2.0422D-07,-2.5881D-08,-1.7624D-08,&
&-8.3518D-09,-4.1743D-09, 4.6044D-05, 2.2425D-05, 3.3079D-06,&
& 3.6752D-07,-8.4476D-08,-4.8455D-07,-2.5391D-07, 7.0824D-07/
DATA (A(I),I=191,285)/ &
& 1.4917D-07,-1.2005D-07,-1.6983D-08, 1.1545D-08, 7.2866D-04,&
& 3.8907D-04, 9.6365D-05, 2.3153D-05, 1.8462D-07,-9.0627D-06,&
&-5.4332D-06, 1.0458D-08, 1.6430D-09, 5.6453D-10, 2.0276D-10,&
&-1.6501D-10,-1.0656D-10, 5.2198D-07, 4.2898D-08,-1.0332D-08,&
&-5.6754D-09,-7.1960D-09, 3.0390D-06, 1.5385D-06, 6.2110D-07,&
& 1.4111D-07,-3.7415D-08,-4.8395D-08,-1.7219D-08, 2.7227D-06,&
& 1.4381D-07,-5.5030D-08,-2.2192D-10,-2.8583D-08, 1.8417D-05,&
& 9.3860D-06, 3.9999D-06, 1.0424D-06,-2.1337D-07,-3.3271D-07,&
&-1.2684D-07, 4.6063D-06,-6.7185D-07,-2.5830D-07, 2.9976D-08,&
& 7.8012D-05, 3.4921D-05, 8.3012D-06, 1.5643D-06,-4.3892D-07,&
&-9.5318D-07,-4.6534D-07, 1.7584D-05,-2.8817D-06,-1.0081D-06,&
& 1.3259D-07, 1.8561D-05, 2.6602D-06,-1.7243D-06,-3.5533D-07,&
& 2.3315D-08, 1.6239D-08, 7.3739D-09, 1.9570D-09,-2.5667D-10,&
&-5.8101D-10,-2.3947D-10, 4.5821D-12, 4.5024D-11, 3.8796D-07,&
& 1.1239D-07,-2.9033D-08,-5.7237D-09, 2.4186D-09,-2.9670D-09,&
& 1.0887D-06, 4.3737D-07, 8.3753D-08, 3.6771D-08, 1.6545D-09,&
&-1.3849D-08,-5.8855D-09, 2.1028D-06, 4.3725D-07,-1.7621D-07,&
&-3.0743D-08, 1.7119D-08, 8.4698D-06, 4.5395D-06, 1.5774D-06,&
& 4.1647D-07,-7.9865D-08,-1.6003D-07,-6.0171D-08, 3.1692D-06/
DATA (A(I),I=286,380)/ &
& 3.6965D-07,-4.9333D-07,-2.8856D-08, 4.2819D-08, 2.0492D-04,&
& 1.3705D-04, 4.0972D-05, 2.9565D-06,-1.4061D-06,-1.8775D-06,&
&-1.6306D-06,-3.6380D-07, 3.1213D-07, 2.0912D-07, 1.1965D-05,&
& 9.9088D-07,-1.3565D-06,-2.3128D-07, 1.1379D-05, 2.6632D-06,&
&-1.6877D-06,-4.0541D-07, 9.8921D-08, 5.9262D-03, 3.2332D-03,&
& 9.2954D-04, 1.6597D-04,-3.7972D-05,-7.1646D-05,-4.0073D-05,&
& 2.5789D-08, 5.2124D-09,-2.2517D-10,-7.9400D-10, 3.3181D-06,&
& 1.9117D-07, 1.0299D-07,-7.7443D-08, 2.2953D-06,-1.0879D-06,&
& 3.4368D-07,-1.1230D-07, 1.6826D-08, 6.9774D-06, 9.2721D-07,&
&-1.8983D-08,-1.6068D-07, 1.5843D-06,-3.5616D-08,-1.1262D-07,&
&-4.3051D-08, 8.8140D-09, 3.4244D-05, 8.0163D-06, 2.9703D-06,&
&-1.6480D-07,-6.3282D-07, 2.3791D-06,-4.9305D-07,-1.7122D-07,&
& 1.2147D-07, 7.2131D-05, 1.7699D-05, 8.4281D-06,-9.3966D-07,&
&-6.8048D-07, 5.3785D-05, 6.9745D-06,-2.3554D-06,-7.2141D-07,&
& 3.8501D-07, 4.4794D-06,-6.2435D-07,-2.7340D-07, 2.3127D-08,&
& 4.2314D-08, 3.3784D-06,-5.1356D-07,-2.4321D-07,-6.2698D-09,&
& 3.0812D-08, 5.6419D-08, 1.6889D-08, 2.7037D-09,-1.7433D-09,&
&-9.4507D-10, 1.9714D-06, 4.2743D-08,-7.4114D-08,-2.9304D-08,&
& 2.8973D-06, 1.0079D-06, 2.9561D-07,-6.2977D-09,-4.5782D-08/
DATA (A(I),I=381,475)/ &
&-2.4331D-08, 4.3284D-06, 2.8398D-07,-3.2705D-07,-1.5079D-07,&
& 5.2686D-06, 1.6539D-06, 3.6079D-07,-1.1589D-07,-5.4904D-08,&
& 7.0939D-06,-1.9534D-06, 3.2391D-08, 5.5702D-08, 3.9072D-05,&
& 1.7299D-05, 5.1530D-06,-6.0911D-07,-1.2652D-06,-4.6507D-07,&
& 1.1506D-05,-2.0422D-06,-6.1195D-07, 5.8641D-08, 1.0150D-05,&
&-4.6977D-07,-6.9446D-07, 2.2516D-08, 1.1109D-07, 3.6715D-05,&
& 5.6186D-06,-3.2309D-06,-1.6403D-06, 4.1051D-05, 2.3335D-05,&
& 7.8106D-06, 2.0279D-07,-8.1139D-07,-3.7619D-07,-1.4982D-07,&
& 2.5621D-05,-1.0798D-05, 1.4607D-06, 3.2421D-07, 6.5478D-09,&
& 2.7233D-09, 3.8646D-10,-1.9143D-10,-1.0483D-10,-4.8664D-11,&
& 1.0698D-06, 2.7752D-07,-2.6636D-08,-3.7583D-08, 2.6989D-07,&
& 3.0325D-08,-2.4613D-08,-1.0828D-08, 2.2522D-09, 8.0777D-06,&
& 2.2558D-06, 7.4760D-08,-2.4140D-07,-5.4292D-08, 8.8362D-07,&
& 9.5045D-08,-5.0543D-08,-1.9134D-08, 1.1284D-05, 4.0659D-06,&
& 7.6246D-07,-9.0394D-08,-8.7408D-08, 1.2192D-06,-2.0362D-07,&
&-1.0700D-07, 2.1990D-08, 1.2519D-08, 5.2758D-05, 2.0175D-05,&
& 3.6690D-06,-4.9014D-07,-7.0474D-07,-3.9931D-07, 4.7638D-05,&
& 1.5546D-05, 6.2294D-07,-1.3428D-06,-1.8177D-07, 4.1203D-06,&
&-4.9340D-07,-3.0198D-07,-1.5902D-08, 2.8567D-08, 2.2397D-06/
DATA (A(I),I=476,570)/ &
&-2.1306D-08,-1.7897D-07, 2.8178D-08, 4.1722D-08, 2.0594D-04,&
& 1.2838D-04, 5.8219D-05, 1.1735D-05,-6.9778D-06,-6.2486D-06,&
&-1.3800D-06, 2.9170D-06,-7.0833D-07,-1.2673D-07, 4.5069D-08,&
& 1.0974D-09, 4.7908D-10, 3.0294D-11,-7.0815D-11,-1.2039D-11,&
& 5.6357D-12, 8.1274D-07, 4.5158D-07, 1.1655D-07,-2.1481D-08,&
&-2.2493D-08,-6.5884D-09, 9.8696D-08, 3.6499D-08, 1.4768D-09,&
&-8.0141D-09,-2.8147D-09, 5.8287D-06, 3.2469D-06, 9.4927D-07,&
&-7.2550D-08,-1.4231D-07,-5.8408D-08,-1.4468D-08, 4.6070D-07,&
& 1.5956D-07,-3.2311D-09,-3.3487D-08,-7.9013D-09, 4.0092D-06,&
& 1.2195D-06, 1.1192D-07,-1.3870D-07,-6.2194D-08, 5.4918D-07,&
&-5.1133D-08,-5.4844D-08,-3.1054D-09, 6.1049D-09, 2.4833D-05,&
& 1.1280D-05, 2.2680D-06,-4.4637D-07,-2.8942D-07,-1.2382D-07,&
& 6.8223D-05, 3.6505D-05, 7.8792D-06,-1.7946D-06,-1.4925D-06,&
&-4.6018D-07, 2.5677D-06,-7.2440D-08,-2.2441D-07,-4.1769D-08,&
& 2.3833D-08, 1.1885D-06, 4.0457D-09,-1.2273D-07,-2.3271D-08,&
& 1.5149D-08, 9.1912D-05, 5.6621D-05, 1.4338D-05,-2.6072D-06,&
&-2.3688D-06,-6.2490D-07,-1.4386D-07, 1.0821D-06,-1.6304D-07,&
&-7.9756D-08,-4.9881D-09, 9.9527D-09, 1.5288D-03, 9.8517D-04,&
& 3.4966D-04, 3.4238D-05,-4.3610D-05,-3.2734D-05,-8.3676D-06/
DATA (A(I),I=571,665)/ &
& 9.6630D-09, 3.2976D-09, 2.1545D-10,-3.3938D-10,-1.4397D-10,&
& 3.4140D-07, 7.6536D-08,-2.2485D-08,-1.8244D-08,-2.0611D-09,&
& 1.3128D-06, 6.4165D-07, 1.4728D-07,-2.7600D-08,-3.0125D-08,&
&-1.0320D-08, 1.6295D-06, 3.3697D-07,-5.7547D-08,-7.3134D-08,&
&-1.5301D-08, 8.0375D-06, 4.0695D-06, 1.1441D-06,-5.8423D-08,&
&-1.6488D-07,-7.5920D-08, 2.1173D-06,-1.9184D-07,-2.4986D-07,&
& 7.2656D-09, 2.7560D-08, 7.9040D-06, 3.3963D-06, 4.8082D-07,&
&-3.1619D-07,-1.6501D-07, 5.8017D-06,-5.5173D-07,-6.3613D-07,&
& 1.9633D-08, 7.8681D-08, 9.4568D-06, 9.8227D-08,-7.8983D-07,&
&-2.0343D-07, 7.0351D-05, 3.6017D-05, 8.0504D-06,-1.7276D-06,&
&-1.5329D-06,-4.7799D-07, 3.5263D-05, 1.8639D-05, 4.2070D-06,&
&-4.2643D-07,-5.2726D-07,-2.5481D-07,-1.0144D-07, 4.4613D-06,&
&-8.7802D-07,-3.9633D-07, 6.7953D-08, 4.1539D-08, 1.8780D-04,&
& 1.0547D-04, 2.0983D-05,-3.7076D-06,-3.7090D-06,-1.5934D-06,&
&-2.6146D-07, 1.6384D-05,-3.3765D-06,-7.8390D-07, 1.2382D-07,&
& 1.5748D-05,-9.4352D-07,-4.8936D-07,-4.9967D-07, 3.8177D-10,&
& 9.6781D-11,-1.9622D-11,-1.7859D-11,-4.4781D-12, 3.0310D-07,&
& 1.1193D-07,-5.4059D-09,-1.4249D-08,-1.9549D-09, 4.7664D-08,&
& 8.5684D-09,-5.0948D-09,-3.1313D-09, 4.9557D-10, 4.2702D-10/
DATA (A(I),I=666,760)/ &
& 2.3433D-06, 8.3824D-07, 2.6585D-08,-7.5944D-08,-1.5093D-08,&
& 1.8775D-07, 2.7614D-08,-2.1193D-08,-1.0264D-08, 1.9660D-09,&
& 1.1637D-09, 7.7890D-06, 3.2901D-06,-2.9753D-07,-5.3153D-07,&
&-5.2098D-08, 3.3947D-08, 5.5937D-07, 5.1363D-08,-8.5390D-08,&
&-2.2580D-08, 1.0857D-08, 3.5157D-09, 4.1303D-05, 2.1947D-05,&
& 3.9480D-06,-1.4234D-06,-9.0220D-07,-2.4485D-07, 1.1031D-04,&
& 6.6726D-05, 1.9581D-05,-4.3637D-07,-2.8911D-06,-1.4332D-06,&
&-3.2332D-07, 3.0922D-06, 2.0624D-07,-4.0771D-07,-1.0402D-07,&
& 4.8807D-08, 1.2491D-06, 2.1187D-07,-1.5951D-07,-7.2537D-08,&
& 1.4035D-08, 9.6514D-09, 2.5146D-05,-9.4490D-08,-3.4103D-06,&
& 2.4020D-07, 2.9564D-07, 6.6566D-07,-3.0772D-08,-1.0055D-07,&
&-3.1082D-09, 1.4897D-08, 7.7189D-04, 3.3710D-04, 3.9258D-05,&
&-1.5271D-05,-6.0652D-06, 2.1986D-04,-3.3135D-05,-8.5682D-06,&
&-9.2988D-07, 3.8085D-06,-1.3259D-07,-3.8517D-07,-5.7276D-08,&
& 3.7226D-08, 2.0825D-09, 1.6707D-10,-1.2443D-10,-3.9117D-11,&
& 1.0068D-07, 6.2278D-09,-5.9169D-09,-3.9535D-09, 5.3334D-07,&
& 2.2022D-07, 1.4522D-08,-2.8440D-08,-8.1406D-09, 6.0187D-07,&
& 5.8318D-08,-2.7772D-08,-2.6639D-08, 3.0929D-06, 1.0648D-06,&
& 1.2317D-07,-9.9233D-08,-4.0796D-08, 1.5217D-06, 8.3095D-08/
DATA (A(I),I=761,855)/ &
&-1.9819D-07,-6.0417D-08, 2.9553D-08, 9.0905D-09, 8.6651D-06,&
& 3.9125D-06,-2.2871D-07,-6.4651D-07,-7.4440D-08, 4.3543D-08,&
& 4.4505D-06, 2.7350D-07,-4.6428D-07,-2.2293D-07, 5.5275D-08,&
& 3.6505D-08, 8.3471D-06, 5.0215D-07,-1.1189D-06,-2.5404D-07,&
& 1.3175D-07, 1.1687D-04, 7.0674D-05, 2.0219D-05,-1.2136D-06,&
&-3.2299D-06,-1.3981D-06,-2.8271D-07, 3.7298D-05, 2.2054D-05,&
& 5.5510D-06,-9.1515D-07,-1.0832D-06,-3.6657D-07,-4.8581D-08,&
& 2.2506D-06,-2.9469D-07,-2.1641D-07,-3.9630D-08, 5.2879D-08,&
& 2.7894D-05,-3.9312D-06,-2.6413D-06, 5.5848D-07, 8.7829D-06,&
&-1.4800D-06,-7.0982D-07,-2.1645D-08, 1.3258D-07, 1.2248D-05,&
&-1.4244D-06,-7.6696D-07,-1.5029D-07, 6.8467D-08, 1.9392D-04,&
&-2.6676D-05,-8.0536D-06,-6.3609D-07, 2.3074D-05, 7.2856D-07,&
&-2.3751D-06,-5.8464D-07, 1.3685D-07, 1.9646D-08, 1.2918D-08,&
& 4.6924D-09, 5.1593D-10,-4.4715D-10,-3.4091D-10,-9.7254D-11,&
& 3.4117D-07, 1.2731D-07,-2.2300D-08,-2.5297D-08,-1.1877D-10,&
& 2.3436D-09, 9.4463D-07, 4.9464D-07, 8.9138D-08,-2.2158D-08,&
&-1.2008D-08,-4.3715D-09,-2.8719D-09, 1.4091D-06, 4.8184D-07,&
&-9.6135D-08,-1.0945D-07,-9.3174D-09, 9.0166D-09, 4.3747D-06,&
& 2.1053D-06, 1.7702D-07,-3.0956D-07,-1.2902D-07,-8.9418D-09/
DATA (A(I),I=856,929)/ &
& 1.4297D-06, 7.7612D-08,-1.7188D-07, 6.4348D-10, 1.1572D-08,&
& 7.4828D-06, 4.0177D-06, 1.0890D-06, 8.6393D-08,-5.4701D-08,&
&-5.7656D-08,-3.3030D-08, 5.0379D-06, 1.2861D-07,-7.1313D-07,&
&-3.3703D-09, 7.9367D-08, 6.2235D-06, 9.8196D-07,-4.6416D-07,&
&-2.3278D-07, 2.9519D-05, 1.5275D-05, 3.1448D-06,-6.4571D-07,&
&-3.4445D-07, 3.6508D-05, 2.0524D-05, 3.1203D-06,-2.3606D-06,&
&-1.5012D-06,-2.5807D-07, 1.3807D-07, 9.9656D-08, 3.1628D-06,&
&-5.6361D-08,-3.8861D-07,-1.0385D-09, 2.9930D-08, 3.1351D-04,&
& 2.3774D-04, 1.0342D-04, 1.2429D-05,-1.4107D-05,-9.2597D-06,&
&-1.7338D-06, 8.4903D-07, 7.3795D-07, 2.2065D-07, 1.2134D-05,&
& 4.6651D-07,-9.3659D-07,-2.7485D-07, 1.2462D-05, 6.2559D-07,&
&-6.1634D-07,-5.1080D-07, 1.2493D-02, 8.2057D-03, 2.9562D-03,&
& 3.3090D-04,-3.7349D-04,-2.8886D-04,-7.4606D-05, 1.2726D-05,&
&-1.6480D-07,-1.5006D-06,-1.1181D-07, 1.4846D-07, 8.7160D-03,&
& 4.2652D-03, 5.2455D-04,-2.6363D-04,-7.9836D-05/

DATA ZERO,ONE,TWO,HALF/0.D0,1.D0,2.D0,0.5D0/  
!     ..
C1 = HALF*LOG10(5.0D7)  
C2 = HALF*LOG10(5.0D1)  
TEMP = T  
!     if(t.gt.30000.) temp = 30000.
!     if(t.lt.2000.) temp = 2000.
!
!        evaluate gammas for requested levels
!
XXX = TWO* (LOG10(TEMP)-C1)/C2  
TFAC = ONE/SQRT(TEMP)  
!     mapping in cheby array
J = ((KU*KU-3*KU+4)/2) + KL - 1  
!     location of first cheby coef
N1 = NSTART(J)  
!     location of last cheby coef
NF = NSTART(J+1) - 1  
NT = NF - N1 + 1  
NTM2 = NT - 2  
!
!          clenshaw summation
!
B(NT) = A(NF)  
B(NT-1) = XXX*B(NT) + A(NF-1)  
IR = NTM2  
JJ = NF - 2  
DO 10 J = 1,NTM2  
     B(IR) = XXX*B(IR+1) - B(IR+2) + A(JJ)  
     IR = IR - 1  
     JJ = JJ - 1  
   10 END DO  
C = (B(1)-B(3))*TFAC*STWT(KU)/STWT(KL)  

RETURN  
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE PIX11 (ZED, AW, N, LP, NFR, FQ, ANL)  
!
!pix11 (formula RBF 16) checked and inserted on 30.01.01
!---- from detail_v201
!---- modified by e. santolaya (18-11-94) so that only one
!---- frequency is calculated each time
!---- fq in hz

use nlte_type
implicit none

!     .. scalar arguments ..
REAL(DP) ::  AW, ZED  
INTEGER(I4B) :: LP, N, NFR  
!     ..
!     .. array arguments ..
REAL(DP), DIMENSION(NFR) ::  ANL, FQ   
!     ..
!     .. local scalars ..
REAL(DP) :: A, AI, AL, ALFAC, CLIGHT, CON1, CON2, CON3, E, EE, &
 FAC, FLL, FLM, FN, FOUR, FRFAC, FTH, FTHHZ, G11, G12, G21, G22, &
 G31, G32, GN0, GN1E, GNE, ONE, P, P1, P2, RYD, SE, SL, SL4, SM, &
 SM4, SN, SN4, SUM, T1, T2, TEN, TERM, TWO, X, ZERO
INTEGER(I4B) :: I, IF, IL, ILMAX, J, JF, K, L, LL, LLK, LLL, LLM, LM, &
 LMAX1, M, MULP, MULR, MULS
LOGICAL :: DONE  
!     ..
!     .. local arrays ..
REAL(DP), DIMENSION(1) :: AB, FREQ
REAL(DP), DIMENSION(1, 2) :: G2, G3
REAL(DP), DIMENSION(1500) :: ALO, FAL

INTEGER(I4B), DIMENSION(1) :: MM  
!     ..
!     .. external functions ..
REAL(DP) :: EXP1  
EXTERNAL EXP1  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS, ATAN, DBLE, LOG10, MIN, SQRT  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA DONE / .FALSE. /  
!     ..
IF (.NOT.DONE) THEN  
!
!         double precision constants
!
   ZERO = 0.0D0  
   ONE = 1.0D0  
   TWO = 2.0D0  
   FOUR = 4.0D0  
   TEN = 1.0D1  
   CON1 = 8.5594D-19  
   CLIGHT = 2.997925D10  
!
!
!          evaluate log factorials
!
   FAL (1) = ZERO  
   DO I = 2, 1500  
      AI = DBLE (I - 1)  
      ALO (I - 1) = LOG10 (AI)  
      FAL (I) = ALO (I - 1) + FAL (I - 1)  
   END DO  
!
   DONE = .TRUE.  

ENDIF  
!          rydberg constant
!
RYD = 109737.312D0 / (ONE+ONE / (1822.84D0 * AW) )  
!
!         frequency scaling
!
FRFAC = ONE / (ZED**2 * RYD * CLIGHT)  
!
!          calculation of hydrogenic photoionization cross section
!
!
!         initialization for n
!
FN = DBLE (N)  
SN = FN * FN  
SN4 = FOUR * SN  
CON2 = CON1 * (FN / ZED) **2  
FTH = ONE / SN  
FTHHZ = FTH / FRFAC * 10.D0  
GN0 = 2.3052328943D0 - 2.302585093D0 * FAL (N + N) - FN * &
 0.61370563888D0 + ALO (N) * (FN + ONE) * 2.30258093D0
LMAX1 = MIN (LP, N - 1)  
ILMAX = N - LP  
IF (LP.GT.100) THEN  
   LMAX1 = N - 1  
   ILMAX = N  
ENDIF  
!
!     reverse frequencies and keep the range down to avoid numerical
!     problems
!
JF = NFR  
DO IF = 1, NFR  
   ANL (IF) = 0.D0  
   IF (FQ (IF) .LT.FTHHZ) THEN  
      FREQ (JF) = FQ (IF)  
   ELSE  
      FREQ (JF) = FTHHZ  
   ENDIF  
   JF = JF - 1  
END DO  
!
!         initialize g's
!
DO I = 1, NFR  
   MM (I) = 0  
   DO J = 1, 2  
      G2 (I, J) = ZERO  
      G3 (I, J) = ZERO  
   END DO  
END DO  
!
!          l loop
!
LLOOP: DO IL = 1, ILMAX  
   L = N - IL  
   M = 0  
   AL = DBLE (L)  
   K = N - L - 2  
   CON3 = CON2 / (TWO * AL + ONE)  
!
!          frequency loop (freq units after multiplication by frfac
!          are ryd*zed**2)
!
   JF = NFR  
IFLOOP:   DO IF = 1, NFR  
      IF ( (FRFAC * FREQ (IF) ) .LT.FTH) THEN  
         IF (L.LE.LMAX1) ANL (JF) = ANL (JF) + ZERO  
         JF = JF - 1  
         CYCLE IFLOOP
      ENDIF  
      G11 = ZERO  
      M = M + 1  
      SE = (FRFAC * FREQ (IF) ) - FTH  
      E = SQRT (SE)  
      X = ONE+SN * SE  
      IF (K.LT.0) THEN  

         IF (E.GE.0.314E0) THEN  
            EE = 6.2831853D0 / E  
            P = 0.5D0 * LOG10 (EXP1 ( - EE) )  
         ELSE  
            P = ZERO  
         ENDIF  

         IF (E.GT.1.0D-6) THEN  
            A = TWO * (FN - ATAN (FN * E) / E)  
         ELSE  
            A = ZERO  
         ENDIF  

         AB (M) = (GN0 + A) / 2.302585D0 - P - (FN + TWO) * LOG10 &
          (X)
         GNE = 0.1D0  
         GN1E = X * GNE / (FN + FN)  
         G3 (M, 2) = GNE  
         G3 (M, 1) = GN1E  
         G2 (M, 2) = GNE * FN * X * (FN + FN - ONE)  
         G2 (M, 1) = GN1E * (FN + FN - ONE) * (FOUR + (FN - ONE) &
          * X)
      ENDIF  
      G22 = G2 (M, 2)  
      G32 = G3 (M, 2)  
      G21 = G2 (M, 1)  
      G31 = G3 (M, 1)  
      MULS = MM (M)  
      IF (K) 90, 100, 50  
!
!                     l.lt.n-2
!
   50       CONTINUE  
      IF (K.GT.1) GOTO 60  
      LL = N - 1  
      LM = N - 2  
   60       CONTINUE  
      SL = DBLE (LL * LL)  
      SL4 = FOUR * SL  
      FLL = DBLE (LL)  
      G12 = (SN4 - SL4 + (TWO * SL - FLL) * X) * G22 - SN4 * &
       (SN - SL) * (ONE+ (FLL + ONE) **2 * SE) * G32
      IF (L.EQ.0) GOTO 70  
      SM = DBLE (LM * LM)  
      SM4 = FOUR * SM  
      FLM = DBLE (LM)  
      G11 = (SN4 - SM4 + (TWO * SM + FLM) * X) * G21 - SN4 * &
       (SN - (FLM + ONE) **2) * (ONE+SM * SE) * G31
      G31 = G21  
      G21 = G11  
   70       CONTINUE  
      G32 = G22  
      G22 = G12  
      IF (IF.NE.NFR) GOTO 80  
      LL = LL - 1  
      IF (L.EQ.0) GOTO 80  
      LM = LM - 1  
   80       CONTINUE  
      IF (G12.LT.1.D20) GOTO 110  
      MULS = MULS + 35  
      G22 = G22 * 1.D-35  
      G32 = G32 * 1.D-35  
      G12 = G12 * 1.D-35  
      IF (L.EQ.0) GOTO 110  
      G11 = G11 * 1.D-35  
      G21 = G21 * 1.D-35  
      G31 = G31 * 1.D-35  
      GOTO 110  
!
!                    l.eq.n-1
!
   90       CONTINUE  
      G11 = G31  
      IF (L.EQ.0) G11 = ZERO  
      G12 = G32  
      GOTO 110  
!
!                    l.eq.n-2
!
  100       CONTINUE  
      G11 = G21  
      IF (L.EQ.0) G11 = ZERO  
      G12 = G22  
  110       CONTINUE  
      MM (M) = MULS  
      G2 (M, 2) = G22  
      G3 (M, 2) = G32  
      G2 (M, 1) = G21  
      G3 (M, 1) = G31  
      ALFAC = FAL (N + L + 1) - FAL (N - L) + TWO * (AL - FN) &
       * ALO (2 * N)
      P1 = ONE  
      LLL = L + 1  
      LLM = L - 1  
      MULR = 0  
      IF (LLM.LT.1) GOTO 130  
      DO 120 I = 1, LLM  
         AI = DBLE (I)  
         P1 = P1 * (ONE+AI * AI * SE)  
         IF (P1.GE.1.D20) THEN  
            P1 = P1 * 1.D-10  
            MULR = MULR + 10  
         ENDIF  
  120       END DO  
  130       CONTINUE  
      P2 = P1  
      LLK = LLM + 1  
      IF (LLK.LT.1) LLK = 1  
      DO 140 I = LLK, LLL  
         AI = DBLE (I)  
         P2 = P2 * (ONE+AI * AI * SE)  
  140       END DO  
      MULP = 0  
  150       CONTINUE  
      IF (G12.LT.ONE) GOTO 160  
      MULP = MULP - 10  
      G12 = G12 * 1.D-10  
      IF (L.EQ.0) GOTO 150  
      G11 = G11 * 1.D-10  

      GOTO 150  
  160       CONTINUE  
      SUM = ALFAC + DBLE (MULR) + TWO * (AB (M) + DBLE (MULS - &
       MULP + 1) )
      FAC = ZERO  
      IF (ABS (SUM) .LT.50.D0) FAC = TEN**SUM  
  170       CONTINUE  
      IF (L.EQ.0) GOTO 180  
      G11 = G11 * P1 * G11  
      T1 = FAC * G11 * X * CON3  
      IF (T1.EQ.ZERO) GOTO 220  
  180       CONTINUE  
      G12 = G12 * P2 * G12  
      T2 = FAC * G12 * X * CON3  
      IF (T2.EQ.ZERO) GOTO 220  

      IF (L.LE.LMAX1) THEN  
         TERM = T1 * AL + T2 * (AL + ONE)  

         IF (LP.LT.100) THEN  
            ANL (JF) = TERM  
         ELSE  
            FAC = (2.D0 * AL + ONE) / SN  
            ANL (JF) = ANL (JF) + FAC * TERM  
         ENDIF  
      ENDIF  
      JF = JF - 1  
  END DO IFLOOP  
END DO LLOOP  

DO IF = 1, NFR  

   IF (FQ (IF) .GE.FTHHZ) THEN  
      X = FTHHZ / FQ (IF)  
      X = X**3  
      ANL (IF) = ANL (IF) * X  
   ENDIF  

END DO  

RETURN  

  220 CONTINUE  
WRITE (6, FMT = 9000) N, L, IF  

STOP  
 9000 FORMAT (1X,'EXPONENT UNDERFLOW FOR N =',I4,'  L =',I3,'  FREQUEN', &
&       'CY NUMBER =',I4)
END SUBROUTINE PIX11
!
!-----------------------------------------------------------------------
!
SUBROUTINE PIX21 (IS, IL, N, XNU, ALPHA)  
!  
!     subroutine used only for formula 17 (rbf).
!     cross section is placed in alpha.
!     xnu is the frecuency, in hertzs

use nlte_type
implicit none


!     .. scalar arguments ..
REAL(DP) ::  ALPHA, XNU  
INTEGER(I4B) :: IL, IS, N  
!     ..
!     .. local scalars ..
REAL(DP) :: FL, P, X  
INTEGER(I4B) :: I, ILL, ISS, J, K, NSL0  
!     ..
!     .. local arrays ..
REAL(DP), DIMENSION(53) :: A, B, FL0, XFITM
REAL(DP), DIMENSION(4, 53) :: COEF

INTEGER(I4B), DIMENSION(3,2) :: IST, N0   
!     ..
!     .. intrinsic functions ..

!INTRINSIC LOG10  
!     ..
!     .. data statements ..
DATA IST / 1, 36, 20, 11, 45, 28 /  
DATA N0 / 1, 2, 3, 2, 2, 3 /  
DATA FL0 /  15.7742D0, 14.9831D0, 14.6054D0, 14.3443D0, 14.1440D0, &
 13.9814D0, 13.8445D0, 13.7262D0, 13.6223D0, 13.5294D0, 15.0618D0, &
 14.6550D0, 14.3806D0, 14.1726D0, 14.0050D0, 13.8646D0, 13.7438D0, &
 13.6378D0, 13.5433D0, 14.5634D0, 14.3134D0, 14.1195D0, 13.9611D0, &
 13.8272D0, 13.7111D0, 13.6088D0, 13.5173D0, 14.5636D0, 14.3135D0, &
 14.1196D0, 13.9612D0, 13.8273D0, 13.7112D0, 13.6089D0, 13.5174D0, &
 14.9110D0, 14.5597D0, 14.3105D0, 14.1171D0, 13.9591D0, 13.8254D0, &
 13.7096D0, 13.6075D0, 13.5172D0, 14.9426D0, 14.5822D0, 14.3277D0, &
 14.1310D0, 13.9707D0, 13.8354D0, 13.7184D0, 13.6152D0, 13.5231D0 /
DATA XFITM /3.262D-01, 6.135D-01, 9.233D-01, 8.438D-01, &
 1.020D+00, 1.169D+00, 1.298D+00, 1.411D+00, 1.512D+00, 1.602D+00, &
 7.228D-01, 1.076D+00, 1.206D+00, 1.404D+00, 1.481D+00, 1.464D+00, &
 1.581D+00, 1.685D+00, 1.777D+00, 9.586D-01, 1.187D+00, 1.371D+00, &
 1.524D+00, 1.740D+00, 1.854D+00, 1.955D+00, 2.046D+00, 9.585D-01, &
 1.041D+00, 1.371D+00, 1.608D+00, 1.739D+00, 1.768D+00, 1.869D+00, &
 1.803D+00, 7.360D-01, 1.041D+00, 1.272D+00, 1.457D+00, 1.611D+00, &
 1.741D+00, 1.855D+00, 1.870D+00, 1.804D+00, 9.302D-01, 1.144D+00, &
 1.028D+00, 1.210D+00, 1.362D+00, 1.646D+00, 1.761D+00, 1.863D+00, &
 1.954D+00 /
DATA A /      6.95319D-01, 1.13101D+00, 1.36313D+00, 1.51684D+00, &
 1.64767D+00, 1.75643D+00, 1.84458D+00, 1.87243D+00, 1.85628D+00, &
 1.90889D+00, 9.01802D-01, 1.25389D+00, 1.39033D+00, 1.55226D+00, &
 1.60658D+00, 1.65930D+00, 1.68855D+00, 1.62477D+00, 1.66726D+00, &
 1.83599D+00, 2.50403D+00, 3.08564D+00, 3.56545D+00, 4.25922D+00, &
 4.61346D+00, 4.91417D+00, 5.19211D+00, 1.74181D+00, 2.25756D+00, &
 2.95625D+00, 3.65899D+00, 4.04397D+00, 4.13410D+00, 4.43538D+00, &
 4.19583D+00, 1.79027D+00, 2.23543D+00, 2.63942D+00, 3.02461D+00, &
 3.35018D+00, 3.62067D+00, 3.85218D+00, 3.76689D+00, 3.49318D+00, &
 1.16294D+00, 1.86467D+00, 2.02110D+00, 2.24231D+00, 2.44240D+00, &
 2.76594D+00, 2.93230D+00, 3.08109D+00, 3.21069D+00 /
DATA B /      - 1.29300D+00, - 2.15771D+00, - 2.13263D+00, - &
 2.10272D+00, - 2.10861D+00, - 2.11507D+00, - 2.11710D+00, - &
 2.08531D+00, - 2.03296D+00, - 2.03441D+00, - 1.85905D+00, - &
 2.04057D+00, - 2.02189D+00, - 2.05930D+00, - 2.03403D+00, - &
 2.02071D+00, - 1.99956D+00, - 1.92851D+00, - 1.92905D+00, - &
 4.58608D+00, - 4.40022D+00, - 4.39154D+00, - 4.39676D+00, - &
 4.57631D+00, - 4.57120D+00, - 4.56188D+00, - 4.55915D+00, - &
 4.41218D+00, - 4.12940D+00, - 4.24401D+00, - 4.40783D+00, - &
 4.39930D+00, - 4.25981D+00, - 4.26804D+00, - 4.00419D+00, - &
 4.47251D+00, - 3.87960D+00, - 3.71668D+00, - 3.68461D+00, - &
 3.67173D+00, - 3.65991D+00, - 3.64968D+00, - 3.48666D+00, - &
 3.23985D+00, - 2.95758D+00, - 3.07110D+00, - 2.87157D+00, - &
 2.83137D+00, - 2.82132D+00, - 2.91084D+00, - 2.91159D+00, - &
 2.91336D+00, - 2.91296D+00 /
DATA ( (COEF (I, J), I = 1, 4), J = 1, 10) /         8.734D-01, &
 - 1.545D+00, - 1.093D+00, 5.918D-01,   9.771D-01, - 1.567D+00, &
 - 4.739D-01, - 1.302D-01, 1.174D+00, - 1.638D+00, - 2.831D-01, &
 - 3.281D-02,   1.324D+00,-1.692D+00, - 2.916D-01,   9.027D-02, &
   1.445D+00, - 1.761D+00,-1.902D-01,   4.401D-02,   1.546D+00, &
 - 1.817D+00, - 1.278D-01, 2.293D-02,   1.635D+00, - 1.864D+00, &
 - 8.252D-02,   9.854D-03, 1.712D+00, - 1.903D+00, - 5.206D-02, &
   2.892D-03,   1.782D+00,-1.936D+00, - 2.952D-02, - 1.405D-03, &
   1.845D+00, - 1.964D+00,-1.152D-02, - 4.487D-03 /
DATA ( (COEF (I, J), I = 1, 4), J = 11, 19) /        7.377D-01, & 
 - 9.327D-01, - 1.466D+00, 6.891D-01,   9.031D-01, - 1.157D+00, &
 - 7.151D-01,   1.832D-01, 1.031D+00, - 1.313D+00, - 4.517D-01, &
   9.207D-02,   1.135D+00,-1.441D+00, - 2.724D-01,   3.105D-02, &
   1.225D+00, - 1.536D+00,-1.725D-01,   7.191D-03,   1.302D+00, &
 - 1.602D+00, - 1.300D-01, 7.345D-03,   1.372D+00, - 1.664D+00, &
 - 8.204D-02, - 1.643D-03, 1.434D+00, - 1.715D+00, - 4.646D-02, &
 - 7.456D-03, 1.491D+00, - 1.760D+00, - 1.838D-02, - 1.152D-02 /
DATA ( (COEF (I, J), I = 1, 4), J = 20, 27) /          1.258D+00, &
 - 3.442D+00, - 4.731D-01, - 9.522D-02,   1.553D+00, - 2.781D+00, &
 - 6.841D-01, - 4.083D-03,   1.727D+00, - 2.494D+00, - 5.785D-01, &
 - 6.015D-02,   1.853D+00, - 2.347D+00, - 4.611D-01, - 9.615D-02, &
   1.955D+00, - 2.273D+00, - 3.457D-01, - 1.245D-01,   2.041D+00, &
 - 2.226D+00, - 2.669D-01, - 1.344D-01,   2.115D+00, - 2.200D+00, &
 - 1.999D-01, - 1.410D-01,   2.182D+00, - 2.188D+00, - 1.405D-01, &
 - 1.460D-01 /
DATA ( (COEF (I, J), I = 1, 4), J = 28, 35) /          1.267D+00, &
 - 3.417D+00, - 5.038D-01, - 1.797D-02,   1.565D+00, - 2.781D+00, &
 - 6.497D-01, - 5.979D-03,   1.741D+00, - 2.479D+00, - 6.099D-01, &
 - 2.227D-02,   1.870D+00, - 2.336D+00, - 4.899D-01, - 6.616D-02, &
   1.973D+00, - 2.253D+00, - 3.972D-01, - 8.729D-02,   2.061D+00, &
 - 2.212D+00, - 3.072D-01, - 1.060D-01,   2.137D+00, - 2.189D+00, &
 - 2.352D-01, - 1.171D-01,   2.205D+00, - 2.186D+00, - 1.621D-01, &
 - 1.296D-01 /
DATA ( (COEF (I, J), I = 1, 4), J = 36, 44) /          1.129D+00, &
 - 3.149D+00, - 1.910D-01, - 5.244D-01,   1.431D+00, - 2.511D+00, &
 - 3.710D-01, - 1.933D-01,   1.620D+00, - 2.303D+00, - 3.045D-01, &
 - 1.391D-01,   1.763D+00, - 2.235D+00, - 1.829D-01, - 1.491D-01, &
   1.879D+00, - 2.215D+00, - 9.003D-02, - 1.537D-01,   1.978D+00, &
 - 2.213D+00, - 2.066D-02, - 1.541D-01,   2.064D+00, - 2.220D+00, &
   3.258D-02, - 1.527D-01,   2.140D+00, - 2.225D+00,   6.311D-02, &
 - 1.455D-01,   2.208D+00, - 2.229D+00,   7.977D-02, - 1.357D-01 /

DATA ( (COEF (I, J), I = 1, 4), J = 45, 53) /          1.204D+00, &
 - 2.809D+00, - 3.094D-01,   1.100D-01,   1.455D+00, - 2.254D+00, &
 - 4.795D-01,   6.872D-02,   1.619D+00, - 2.109D+00, - 3.357D-01, &
 - 2.532D-02,   1.747D+00, - 2.065D+00, - 2.317D-01, - 5.224D-02, &
   1.853D+00, - 2.058D+00, - 1.517D-01, - 6.647D-02,   1.943D+00, &
 - 2.055D+00, - 1.158D-01, - 6.081D-02,   2.023D+00, - 2.070D+00, &
 - 6.470D-02, - 6.800D-02,   2.095D+00, - 2.088D+00, - 2.357D-02, &
 - 7.250D-02,   2.160D+00, - 2.107D+00,   1.065D-02, - 7.542D-02 /
!     ..
!
!       check input parameters
!
IF (IS.NE.1.AND.IS.NE.3) THEN  
   WRITE ( *, FMT = 9000) IS  
   STOP  
ENDIF  

IF (IL.LT.0.OR.IL.GT.2) THEN  
   WRITE ( *, FMT = 9010) IL  
   STOP  
ENDIF  

IF (N.GT.10) THEN  
WRITE ( * , FMT = '(A)') 'N TOO LARGE IN PIX21. GREATER THAN 10'  
   STOP  
ENDIF  
!
!     selected beginning and end of coefficientes
!
ISS = (IS + 1) / 2  
ILL = IL + 1  
NSL0 = N0 (ILL, ISS)  

IF (N.LT.NSL0) THEN  
   WRITE ( * , FMT = '(A)') 'N TOO SMALL IN PIX21 '  
   STOP  
ENDIF  

I = IST (ILL, ISS) + N - NSL0  
!
!     calculation of cross section for frec. xnu
!
FL = LOG10 (XNU)  

X = FL - FL0 (I)  
IF (X.GE. - 0.001D0) THEN  

! allows freq slightly below threshold

   IF (X.LT.XFITM (I) ) THEN  
      P = COEF (4, I)  
      DO K = 1, 3  
         P = X * P + COEF (4 - K, I)  
      END DO  
      ALPHA = 10.D0**P * 1.D-18  

   ELSE  
      ALPHA = 10.D0** (A (I) + B (I) * X) * 1.D-18  
   ENDIF  

ELSE  
   ALPHA = 0.D0  
ENDIF  

RETURN  
 9000 FORMAT ('  MULTIPLICITY S=',I3,'  NOT ALLOWED FOR HE I')  
 9010 FORMAT ('   ANGULAR MOMENTUM L=',I3,'   NOT ALLOWED FOR HE I')  
END SUBROUTINE PIX21
!
!-----------------------------------------------------------------------
!
SUBROUTINE HECION(T,YNE,N,C)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
REAL(DP) ::  C,T,YNE  
INTEGER(I4B) ::  N  
!     ..
!     .. local scalars ..
REAL(DP) ::  GAM,TFAC,U,UI,UL,Y  
INTEGER(I4B) :: I  
!     ..
!     .. local arrays ..
REAL(DP) ::  A(6),ALPHA(73),BETA(3),E(73)  
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP,LOG,LOG10,SQRT  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
!
!        data for the helium i atom
!
!        ionization energies in cm-1
!
!
!        threshold photoionization cross sections *10**18
!
!
!        data for ground state ionization
!
DATA (E(I),I=1,73)/198310.76D0,38454.69D0,32083.21D0,29223.69D0, &
&     27175.76D0,15073.87D0,13445.82D0,12746.07D0,12209.11D0, &
&     12205.70D0,12101.29D0,8012.55D0,7370.43D0,7093.62D0, &
&     6866.17D0,6864.20D0,6858.78D0,6858.77D0,6817.94D0,4963.67D0, &
&     4647.13D0,4509.95D0,4393.51D0,4392.37D0,4389.58D0,4389.57D0, &
&     4389.03D0,4368.19D0,3374.53D0,3195.76D0,3117.85D0,3050.59D0, &
&     3049.90D0,3048.27D0,3048.26D0,3047.89D0,3035.72D0,2442.41D0, &
&     2331.72D0,2283.36D0,2241.03D0,2240.54D0,2239.49D0,2239.48D0, &
&     2239.24D0,2231.52D0,1849.34D0,1775.88D0,1743.94D0,1715.58D0, &
&     1715.22D0,1714.59D0,1714.59D0,1714.41D0,1709.25D0,1448.72D0, &
&     1397.78D0,1375.34D0,1355.48D0,1355.24D0,1354.72D0,1354.72D0, &
&     1354.60D0,1350.97D0,1165.48D0,1128.59D0,1112.42D0,1097.88D0, &
&     1097.69D0,1097.33D0,1097.33D0,1097.22D0,1094.52D0/
DATA (ALPHA(I),I=1,73)/7.5D0,5.5D0,9.5D0,16.0D0,13.5D0,8.0D0, &
&     14.9D0,28.5D0,18.5D0,18.1D0,27.0D0,10.7D0,21.1D0,41.6D0, &
&     36.7D0,35.7D0,2*19.1D0,41.7D0,13.6D0,27.9D0,55.8D0,55.1D0, &
&     55.3D0,2*41.0D0,17.5D0,53.9D0,16.8D0,35.2D0,71.3D0,74.1D0, &
&     71.3D0,2*71.7D0,26.7D0,75.7D0,20.0D0,43.2D0,87.7D0,94.0D0, &
&     90.2D0,2*86.0D0,35.6D0,95.1D0,23.6D0,51.5D0,105.4D0,115.1D0, &
&     110.0D0,2*109.0D0,44.4D0,115.9D0,27.2D0,60.5D0,124.5D0, &
&     137.1D0,130.3D0,2*132.6D0,52.9D0,138.0D0,31.0D0,70.0D0, &
&     144.5D0,160.3D0,152.1D0,2*156.9D0,59.9D0,161.4D0/
DATA A/1.4999D-8,5.6656D-10,-6.0821D-9,-3.589D-9,1.5529D-9, &
&     1.3207D-9/
DATA GAM/3.1373D-8/  
DATA BETA/4.7893D-8,-7.7359D-7,3.7366D-6/  
!     ..
!
!        ground state - fits to experimental data
!
IF (N.EQ.1) THEN  
! =kt/eion
     U = .69501D0*T/E(1)  
     UI = 1.0D0/U  
     IF (U.LE.1.0D0) THEN  
          UL = LOG10(U)  

          C = (A(1)+UL* (A(2)+UL* (A(3)+UL* (A(4)+UL* (A(5)+ &
           UL*A(6))))))*SQRT(U)*EXP(-UI)
     ELSE  
          C = SQRT(U)* (GAM*LOG(U)+BETA(1)+UI* (BETA(2)+UI*BETA(3) &
           ))

     END IF  

     C = YNE*C  
ELSE  
!
!        excited states
!
     TFAC = 1.0D0/SQRT(T)  
     Y = 1.4388D0*E(N)/T  
     C = YNE*1.1D-6*ALPHA(N)*TFAC*EXP(-Y)/Y  
END IF  
!

RETURN  
END
!
!-----------------------------------------------------------------------
!
FUNCTION EXP1(X)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
REAL(DP) ::  X,EXP1
!     ..
!     .. local scalars ..
REAL(DP) ::  DX  
!     ..
!     .. intrinsic functions ..
!INTRINSIC ABS,EXP  
!     ..
!     .. save statement ..
SAVE  
!     ..
DX = ABS(X)  
IF (DX.LT.1.0D-9) GO TO 10  
IF (DX.LT.1.0D-5) GO TO 20  
IF (DX.LT.1.0D-3) GO TO 30  
EXP1 = 1.0D0 - EXP(X)  

RETURN  
   10 CONTINUE  
EXP1 = -X  

RETURN  
   20 CONTINUE  
EXP1 = ((-X*0.D0)-1.0D0)*X  

RETURN  
   30 CONTINUE  
EXP1 = (((-X*0.1666666667D0)-0.5D0)*X-1.0D0)*X  

RETURN  
END
!
!
!-----------------------------------------------------------------------
!
FUNCTION EXPINO(X)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     exponential integral for positive arguments after cody and
!     thacher, math. of comp.,22,641(1968)
!     .. scalar arguments ..
REAL(DP) ::  X,EXPINO
!     ..
!     .. local scalars ..
REAL(DP) ::  A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,C0,C1,C2,C3,C4, &
&                 C5,C6,D1,D2,D3,D4,D5,D6,E0,E1,E2,E3,E4,E5,E6,EX, &
&                 EX1,F1,F2,F3,F4,F5,F6,X1
!     ..
!     .. intrinsic functions ..
!INTRINSIC EXP,LOG  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA X1/-1.E20/  
DATA A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4/-44178.5471728217D0, &
&     57721.7247139444D0,9938.31388962037D0,1842.11088668000D0, &
&     101.093806161906D0,5.03416184097568D0,76537.3323337614D0, &
&     32597.1881290275D0,6106.10794245759D0,635.419418378382D0, &
&     37.2298352833327D0/
DATA C0,C1,C2,C3,C4,C5,C6,D1,D2,D3,D4,D5,D6/4.65627107975096D-7, &
&     .999979577051595D0,9.04161556946329D0,24.3784088791317D0, &
&     23.0192559391333D0,6.90522522784444D0,.430967839469389D0, &
&     10.0411643829054D0,32.4264210695138D0,41.2807841891424D0, &
&     20.4494785013794D0,3.31909213593302D0,.103400130404874D0/
DATA E0,E1,E2,E3,E4,E5,E6,F1,F2,F3,F4,F5,F6/-.999999999998447D0, &
&     -26.6271060431811D0,-241.055827097015D0,-895.927957772937D0, &
&     -1298.85688746484D0,-545.374158883133D0,-5.66575206533869D0, &
&     28.6271060422192D0,292.310039388533D0,1332.78537748257D0, &
&     2777.61949509163D0,2404.01713225909D0,631.657483280800D0/
!     ..
IF (X.EQ.X1) GO TO 40  
EX = EXP(-X)  
X1 = X  
IF (X.GT.4.E0) GO TO 10  
IF (X.GT.1.E0) GO TO 20  
IF (X.GT.0.E0) GO TO 30  
EX1 = 0.D0  

GO TO 40  
   10 CONTINUE  
EX1 = (EX+EX* (E0+ (E1+ (E2+ (E3+ (E4+ (E5+E6/X)/X)/X)/X)/X)/X)/ &
&      (X+F1+ (F2+ (F3+ (F4+ (F5+F6/X)/X)/X)/X)/X))/X

GO TO 40  
   20 CONTINUE  
EX1 = EX* (C6+ (C5+ (C4+ (C3+ (C2+ (C1+C0*X)*X)*X)*X)*X)*X)/ &
&      (D6+ (D5+ (D4+ (D3+ (D2+ (D1+X)*X)*X)*X)*X)*X)

GO TO 40  
   30 CONTINUE  
EX1 = (A0+ (A1+ (A2+ (A3+ (A4+A5*X)*X)*X)*X)*X)/ &
&      (B0+ (B1+ (B2+ (B3+ (B4+X)*X)*X)*X)*X) - LOG(X)
   40 CONTINUE  
EXPINO = EX1  

RETURN  
END
!
!***********************************************************************
!
! subroutines: temperature correction
!
!***********************************************************************
!
SUBROUTINE TEMPCORR_START

USE tcorr_var, ONLY : QFFH,QFFC,QRBFI,QRBFR, &
&        QCBBD, QCBBU, DTQFFH,DTQFFC,DTQCBFH, &
&        DTQCBFC,DTQCBBD,DTQCBBU,DTQRBFR,DTQRBFI, &
&        QCBFR, QCBFI

! Energies
QFFH=0.D0
QFFC=0.D0
QCBBD=0.D0
QCBBU=0.D0
QRBFR=0.D0
QRBFI=0.D0
QCBFR=0.D0
QCBFI=0.D0
! Derivatives
DTQFFH=0.D0    
DTQFFC=0.D0    
DTQCBFH=0.D0
DTQCBFC=0.D0
DTQCBBU=0.D0
DTQCBBD=0.D0
DTQRBFR=0.D0
DTQRBFI=0.D0

RETURN
END
!
!-------------------------------------------------------------------------
!
SUBROUTINE TEMPCORR_MAIN(T,XNE,TAUR,DTFCORR,FLUXERR,TEFF,NCOR,EMAXTC,EXPANSION,OUTTHB)

! no update for clumping required, since all rates refer to clumps (incl. OPAFF)
! warning: update the algorithm if adiabatic EXPANSION is accounted for

USE princesa_var
USE nlte_type
USE nlte_dim
USE fund_const
USE nlte_var, ONLY : MODNAM, XNELTE
USE tcorr_var, ONLY : TAURLIM, TMIN1, TMINABS

IMPLICIT NONE

INTEGER(I4B), PARAMETER :: ND=ID_NDEPT
! .
! .. scalar arguments
LOGICAL :: EXPANSION,OUTTHB
INTEGER(I4B) :: NCOR
REAL(DP) :: EMAXTC,TEFF
! .
! .. array arguments
REAL(DP) :: T(ND),DTFCORR(ND),FLUXERR(ND),XNE(ND),TAUR(ND)
! .
! .. local scalars
!
CHARACTER*3 :: NCNAME
REAL(DP) :: RTAU23,EMAXNE,TLIM,TAULAST,W1,W2,WTOT
INTEGER(I4B) :: I,IREL,ITAURLIM,ILAST
! .
! .. local arrays
REAL(DP) :: DELTAT(ND),DUMMY(ND),R(ND),TOLD(ND),DTOLD(ND),XNESAVED(ND), &
& XNELTE_TAUR(ND),TAUR_TAUR(ND),DT(ND),REL_ERR(ND)

DATA DTOLD/ND*0.D0/

DELTAT = 0.D0
TOLD=T
CALL THERMAL_BALANCE(TAUR,XNE,T,DELTAT,EXPANSION,NCOR+1,OUTTHB,DTOLD,REL_ERR,IREL)


DO I=ND,1,-1
  IF(TAUR(I).LT.TAURLIM) EXIT
ENDDO
ITAURLIM=I+1  

!check for outermost point where dtfcorr is provided, including one safety point
DO I=ND,1,-1
  IF (DTFCORR(I).EQ.0.D0) GOTO 10
ENDDO
STOP ' DTFCORR EQ 0 NOT FOUND'

10 ILAST=I+1
TAULAST=TAUR(ILAST)
DO I=ND,1,-1
  IF(TAUR(I).LT.3.*TAULAST) EXIT ! ANOTHER SAFETY FACTOR, TO ALLOW
                                 ! AT LEAST A FACTOR OF 3 FOR BORDER EFFECTS
ENDDO
ILAST=I+1
IF(TAUR(ILAST).GT.TAUR(ITAURLIM)) ILAST=ITAURLIM

!check for irel and modify it if necessary

print*,' TAURLIM = ',TAURLIM
print*,' TRANSITION REGION AT MAXIMUM UNTIL TAUR = ',TAUR(ILAST)
print*,' TAU(REL_ENER <  1.D-3) = ',TAUR(IREL)
 
! itaurlim: minimum tau until we do flux-conservation in any case (inside)
! irel    : maximum tau from where on we do thermal balance in any case (outside)

IF(IREL.LT.ILAST) IREL=ILAST !THIS IS THE MAXIMUM WE ALLOW  

IF(ITAURLIM-IREL.LT.2) THEN
  IREL=ITAURLIM-2 ! at least two points for transition region
  TLIM=TAUR(IREL)
ELSE 
  TLIM=TAUR(IREL)
ENDIF  

print*,' FLUX-CORRECTION FROM FLUX-ERROR PERFORMED UNTIL TAUR > ',TAURLIM
print*,' FLUX-CORRECTION FROM TH.BALANCE PERFORMED FROM  TAUR < ',TLIM
print*,' TRANSITION REGION FROM ',TAUR(ITAURLIM-1),' UNTIL ',TAUR(IREL)


EMAXTC=0.D0
PRINT *,'TEMPERATURE CORRECTION!!!'
write(*,*) ' ND   Tau Ross.      Temp.         dT (ThB)    dT (FlCorr.)    dT (used)'
DO I=ND,1,-1
        IF (TAUR(I) .LT. TLIM) THEN  
! outside
                DT(I)=DELTAT(I)
        	T(I) = T(I) + DT(I)	
                W1=1.
                W2=0.
               
        ELSE IF(TAUR(I).GT.TAURLIM) THEN
!inside
                DT(I)=DTFCORR(I)
		T(I) = T(I) + DT(I)
                W1=0.
                W2=1.
        ELSE
!intermediate range: interpolation
          WTOT=TAURLIM-TLIM
          IF(WTOT.LE.0.) STOP ' ERROR IN TAURLIM,TLIM'
          W1=1.-EXP(-(TAURLIM-TAUR(I))/WTOT)
!          W1=1.-EXP(-(TAURLIM-TAUR(I))/WTOT/10.)
          W2=1.-W1
          DT(I)=W1*DELTAT(I)+W2*DTFCORR(I)
! for tests
!          IF(TAUR(I).LE.1. .AND. TAUR(I).GT. 0.01) DT(I)=DTFCORR(I)
          T(I)=T(I)+DT(I)
        ENDIF

! to avoid problems, we fix the minimum temperature that can be reached
! at a given value	
        IF (T(I) .LT. TMIN1*TEFF) THEN
          T(I) = TMIN1*TEFF
          DT(I)=0.
        ENDIF  
        IF (T(I) .LT. TMINABS) THEN
          T(I) = TMINABS
          DT(I)=0.
        ENDIF  
	EMAXTC=MAX(EMAXTC,ABS(DT(I)/T(I)))
        write (*,FMT='(1X,I2,5(1X,G14.7),2(1X,F4.2))') I,TAUR(I),&
&          T(I),DELTAT(I),DTFCORR(I),DT(I),W1,W2
ENDDO
PRINT *,'Maximum relative T correction ',EMAXTC
DTOLD=T-TOLD

IF(OUTTHB) THEN
  CALL NCOR_NAME(NCOR+1,NCNAME)
ELSE
  NCNAME='_LA'
ENDIF
	open (7,FILE=TRIM(MODNAM)//'/DIFF_TC'//TRIM(NCNAME)//'.dat', &
&		STATUS='UNKNOWN')
        rewind 7
	write(7,*) ' ND   Tau Ross.    Temp.       dT (ThB)    dT (FlCorr.)   dT (used)    XNE '
	print *,' '
        DO I=ND,1,-1		
        	write(7,FMT='(1X,I3,1X,6(G12.5,1X))') I,TAUR(I), &
&   		T(I),DELTAT(I),DTFCORR(I),DT(I),XNE(I)
	ENDDO
	CLOSE(7)	

! updating the TEMP file (including XNELTE from last T-iteration)

REWIND 21
READ (21,*) (R(I),I=ND,1,-1), (DUMMY(I),I=ND,1,-1)  
READ (21,*) (XNESAVED(I),I=ND,1,-1)  
READ (21,*) RTAU23

!just to be sure
OPEN (1,FILE=TRIM(MODNAM)//'/TAU_ROS',STATUS='UNKNOWN',FORM='FORMATTED')  
REWIND 1
  DO I=1,ND
       READ(1,*) TAUR_TAUR(I),XNELTE_TAUR(I)
       IF(TAUR(I).NE.0.D0.AND.ABS(1.-TAUR(I)/TAUR_TAUR(I)).GT.1.D-13) &
&        STOP ' TAUR .NE. TAUR IN TAU_ROS'
       IF(ABS(1.-XNELTE(I)/XNELTE_TAUR(I)).GT.1.D-13) &
&        STOP ' XNELTE .NE. XNELTE IN TAU_ROS' !not exact, since formattedly written/read 
  ENDDO
CLOSE (1)
!
EMAXNE=0.
DO I=1,ND
   EMAXNE=MAX(EMAXNE,ABS(1.-XNELTE(I)/XNESAVED(I)))
ENDDO

PRINT* 
PRINT*,' TEMPERATURE AND LTE ELECTRON DENSITY (LAST ITERATION) UPDATED IN FILE TEMP!'  
PRINT*,' MAX. CHANGE IN XNELTE = ',EMAXNE
PRINT*

REWIND 21
WRITE (21,*) (R(I),I=ND,1,-1), (T(I),I=ND,1,-1)  
WRITE (21,*) (XNELTE(I),I=ND,1,-1)  
WRITE (21,*) RTAU23 

RETURN
END
!
!________________________________________________________________________
!
!
SUBROUTINE MeanMolWeight(meanmw,elecmw)
!
! check if EXPANSION is incorporated
!
USE nlte_type
USE nlte_dim
USE princesa_var, ONLY : ZEFF,WEIGHT,ABUND
USE nlte_var, ONLY : ENIONND
USE fund_const
IMPLICIT NONE
!
INTEGER(I4B), PARAMETER :: ND=ID_NDEPT,KEL=ID_ATOMS,KIS=ID_KISAT
!
INTEGER(I4B) :: i,i1,i2,aux,aux2, tecla, j
REAL(DP) :: ion_par(kel,kis+1,nd),meanmw(ND),elecmw(ND),sumaele,sumapar
!
ion_par(:,:,:)=0.D0
meanmw(:)=0.D0
elecmw(:)=0.D0

do i=ND,1,-1
	do i1=1,kel
		sumapar=0.D0
		aux=int(ZEFF(i1))
		do i2=1,kis+1
			sumapar=sumapar+ENIONND(i1,i2,i)
		enddo
		do i2=aux+1,kis+1
			aux2=i2-aux			
			ion_par(i1,i2,i)=ENIONND(i1,aux2,i)/sumapar			
		enddo
	enddo
	do i1=1,kel
		sumapar=0.D0
		sumaele=0.D0
		aux=INT(ZEFF(i1))
		do i2=1,kis+1
			sumapar=sumapar+ion_par(i1,i2,i)*(1.d0 &
&			+ aux)
			sumaele=sumaele+ion_par(i1,i2,i)*aux
			aux=aux+1
		enddo
		sumapar=ABUND(i1)*sumapar/WEIGHT(i1)
		sumaele=ABUND(i1)*sumaele/WEIGHT(i1)
		meanmw(i)=meanmw(i)+sumapar
		elecmw(i)=elecmw(i)+sumaele
	enddo
	meanmw(i)=1.D0/meanmw(i)
	elecmw(i)=1.D0/elecmw(i)
enddo

RETURN

END
!_________________________________________________________________
!
!
SUBROUTINE THERMAL_BALANCE(TAUR,XNE,T,DELTAT,EXPANSION,NCOR,OUTTHB,DTOLD,REL_ERR,IREL)

! It computes all the terms needed to check the thermal balance of the 
! electrons, giving the T-Corr. 
! If OUTTHB is true, then the numbers are save into a number of ascii 
! files on every TCorr cycle. WARNING: if a model is restarted to perform
! some new T corrections, then the output files will be overwritten


! rates always formulated via heating-cooling:
! a final positive term means that the net process heats the medium 

USE princesa_var, ONLY : NAT,INDEX1
USE nlte_type
USE nlte_dim
USE fund_const

USE nlte_var, ONLY : IFRE,FRE,WFRE,XJ,MODNAM	

USE tcorr_var, ONLY : FFOPA,DTFFOPA,QFFH, &
&   QFFC,QCBFR,QCBFI,QRBFR,QRBFI,DTQRBFR,DTQRBFI, &
&   DTQFFC,DTQFFH,DTQCBFH,DTQCBFC,QCBBU,QCBBD,  &
&   DTQCBBU,DTQCBBD,ENATCOR, &
&   FFOPA_M,DTFFOPA_M,QCBFR_M,QCBFI_M,DTQCBFR_M,DTQCBFI_M, &
&   QRBFR_M,QRBFI_M,DTQRBFR_M,DTQRBFI_M,&
&   QCBBU_M,QCBBD_M,DTQCBBU_M,DTQCBBD_M


IMPLICIT NONE

INTEGER(I4B), PARAMETER :: ND=ID_NDEPT, NF=ID_FREC1
!REAL(DP), PARAMETER :: RSUN=6.9599D10
! . 
! .. scalar arguments
INTEGER(I4B) :: NCOR,IREL
LOGICAL :: EXPANSION,OUTTHB
! .
! .. array arguments
REAL(DP) :: XNE(ND),DELTAT(ND),T(ND),TAUR(ND),DTOLD(ND),REL_ERR(ND)	    
! .
! .. local scalars
CHARACTER*3 :: NCNAME
INTEGER(I4B) :: I,J,IIFREQ,IINAT
REAL(DP) :: AUX,X,EXPO
! .
! .. local arrays
REAL(DP), DIMENSION(ND,NF) :: X1,X2,X3,X4,X5,X6,x7,x8
REAL(DP), DIMENSION(ND) :: JINT, BINT
REAL(DP) :: ENERGIES(ND,9),DTQ(ND,9), &
&	    TAUX1(ND,6),TAUX2(ND,6), TAUX3(ND,6), &
&	    MEANMW(ND), DUMMY(ND), &
&           REL_ENE(ND,6), ENERDIFFOLD(ND), DTQOLD(ND), AUXOLD(ND)
!           ,REL_ENE2(ND,5),RADIATIVEEQ(ND,9),AUX1(ND,NF)

DATA ENERDIFFOLD /ND*0.D0/
!     ..
!     .. external subroutines ..
EXTERNAL MeanMolWeIght,NCOR_NAME
!     .. 	
!     .. external functions
REAL(DP) :: XINTFRE,FFTOT,DTFFTOT
EXTERNAL XINTFRE
!     ..
!     .. IntrInsIc functIons ..
!INTRINSIC EXP, MAX, MIN, ABS  
!	

AUX=0.D0
ENERGIES=0.D0
DTQ=0.D0
X1=0.D0
X2=0.D0
X3=0.D0
X4=0.D0
X5=0.D0
! RADIATIVEEQ=0.D0

DO I=1,ND ! depth loop
! FREE-FREE, other terms are calculated In NETMAT, RATEEQ and LINESOB
	DO IIFREQ=1,IFRE ! frequency loop
	   AUX = (XJ(I,IIFREQ)+HC2*FRE(IIFREQ)**3) * &
&            EXP(-HKL*FRE(IIFREQ)/T(I))
             FFTOT=FFOPA(I,IIFREQ)+FFOPA_M(I,IIFREQ)
           DTFFTOT=DTFFOPA(I,IIFREQ)+DTFFOPA_M(I,IIFREQ)

	   X1(I,IIFREQ)=FFTOT*XJ(I,IIFREQ)
           X3(I,IIFREQ)=FFTOT*AUX
! derivatives vs. temperature
	   X2(I,IIFREQ)=DTFFTOT*XJ(I,IIFREQ)
	   X4(I,IIFREQ)=AUX*(DTFFTOT + FFTOT*HKL*FRE(IIFREQ)/(T(I)**2))
! Planck function, to check consistency
           X=HKL*FRE(IIFREQ)/T(I)
           IF(X.LT.200.D0) THEN
              X5(I,IIFREQ) = HC2*FRE(IIFREQ)**3/(EXP(X)-1.D0)
           ELSE
              EXPO=LOG(HC2*FRE(IIFREQ)**3)-X        
              X5(I,IIFREQ)=EXP(EXPO)
           ENDIF   
        ENDDO

	QFFH(I)=4.D0*PI*XINTFRE(X1(I,:),WFRE,1,IFRE,IFRE,FRE,4,'OLD')
	QFFC(I)=4.D0*PI*XINTFRE(X3(I,:),WFRE,1,IFRE,IFRE,FRE,4,'OLD')
	DTQFFH(I)=4.D0*PI*XINTFRE(X2(I,:),WFRE,1,IFRE,IFRE,FRE,4,'OLD')
	DTQFFC(I)=4.D0*PI*XINTFRE(X4(I,:),WFRE,1,IFRE,IFRE,FRE,4,'OLD')	
	JINT(I)=XINTFRE(XJ(I,:),WFRE,1,IFRE,IFRE,FRE,4,'OLD')	
	BINT(I)=XINTFRE(X5(I,:),WFRE,1,IFRE,IFRE,FRE,4,'OLD')	
ENDDO

! TEST OF FREQ. GRID AND INTEGRATION, JUST IN CASE
AUX = SIGSB/PI*T(1)**4 
IF (ABS(1.-BINT(1)/AUX) .GT. 1.D-2) THEN
  PRINT*,AUX, BINT(1)
  STOP ' THERMAL BALANCE: SOMETHING WRONG IN FREQ. INTEGRATION (OUTSIDE)'
ENDIF
AUX = SIGSB/PI*T(ND)**4 
IF (ABS(1.-BINT(ND)/AUX) .GT. 1.D-2) THEN
  PRINT*,AUX, BINT(ND)
!  STOP ' THERMAL BALANCE: SOMETHING WRONG IN FREQ. INTEGRATION (INSIDE)'
  PRINT*,' WARNING!! THERMAL BALANCE: FREQ. INTEGRATION (INSIDE) INACCURATE'
  PRINT*,' WARNING!! THERMAL BALANCE: FREQ. INTEGRATION (INSIDE) INACCURATE'
  PRINT*,' WARNING!! THERMAL BALANCE: FREQ. INTEGRATION (INSIDE) INACCURATE'
ENDIF

DO I=1,ND ! depth loop
! G heating
! L cooling
! negative rates=cooling
        ENERGIES(I,1)=QFFH(I)	! Gamma_FF
 	ENERGIES(I,2)=QFFC(I)	! Lambda_FF
        DTQ(I,1)=DTQFFH(I)	! dGamma_FF/dT	
	DTQ(I,2)=DTQFFC(I)	! dLambda_FF/dT

! radiative equilibrium terms
!       RADIATIVEEQ(I,1) = RADIATIVEEQ(I,1) + QFFC(I)	!REQ heatIng
!	RADIATIVEEQ(I,2) = RADIATIVEEQ(I,2) + QFFH(I)	!REQ coolIng
!		
		
	ENERGIES(I,3) = QCBFR(I)	! G CBF
 	ENERGIES(I,4) = QCBFI(I)	! L CBF
        ENERGIES(I,5) = QRBFI(I)	! G RBF
 	ENERGIES(I,6) = QRBFR(I)	! L RBF
	ENERGIES(I,7) = QCBBD(I)	! G CBB
	ENERGIES(I,8) = QCBBU(I)	! L CBB	

	DTQ(I,3) = DTQCBFH(I)    	! dG/dT CBF
	DTQ(I,4) = DTQCBFC(I)	        ! dL/dT CBF
	DTQ(I,5) = DTQRBFI(I)   	! dG/dT RBF
	DTQ(I,6) = DTQRBFR(I)    	! dL/dT RBF
	DTQ(I,7) = DTQCBBD(I)	        ! dG/dT CBB
	DTQ(I,8) = DTQCBBU(I)	        ! dL/dT CBB 
! radiative equilibrium terms
!     	do IINAT=1,NAT ! atom loop
!               RADIATIVEEQ(I,3) = RADIATIVEEQ(I,3) + RERBFH(I,IINAT)
!		RADIATIVEEQ(I,4) = RADIATIVEEQ(I,4) + RERBFC(I,IINAT)
!	enddo	
	        DTQ(I,3) = DTQ(I,3) + DTQCBFR_M(I)	! dG/dT CBF_METALS
		DTQ(I,4) = DTQ(I,4) + DTQCBFI_M(I)	! dL/dT CBF_METALS
		DTQ(I,5) = DTQ(I,5) + DTQRBFI_M(I)	! dG/dT RBF_METALS
		DTQ(I,6) = DTQ(I,6) + DTQRBFR_M(I)	! dL/dT RBF_METALS
		DTQ(I,7) = DTQ(I,7) + DTQCBBD_M(I)	! dG/dT CBB_METALS
		DTQ(I,8) = DTQ(I,8) + DTQCBBU_M(I)	! dL/dT CBB_METALS 
! in case, print out the different contributions
!                print*,i,energies(i,3)-energies(i,4),qcbfr_m(i)-qcbfi_m(i)
!                print*,i,energies(i,5)-energies(i,6),qrbfi_m(i)-qrbfr_m(i)
!                print*,i,energies(i,7)-energies(i,8),qcbbd_m(i)-qcbbu_m(i)
!                print*
        	ENERGIES(I,3) = ENERGIES(I,3) + QCBFR_M(I)	! G CBF_METALS
 		ENERGIES(I,4) = ENERGIES(I,4) + QCBFI_M(I)	! L CBF_METALS
        	ENERGIES(I,5) = ENERGIES(I,5) + QRBFI_M(I)	! G RBF_METALS
 		ENERGIES(I,6) = ENERGIES(I,6) + QRBFR_M(I)	! L RBF_METALS
   		ENERGIES(I,7) = ENERGIES(I,7) + QCBBD_M(I)	! G CBB_METALS
		ENERGIES(I,8) = ENERGIES(I,8) + QCBBU_M(I)	! L CBB_METALS


	ENERGIES(I,9) = (ENERGIES(I,1) - ENERGIES(I,2)) + &
&                       (ENERGIES(I,3) - ENERGIES(I,4)) + &
&                       (ENERGIES(I,5) - ENERGIES(I,6)) + &
&                       (ENERGIES(I,7) - ENERGIES(I,8))	! Sum (G - L)
  	DTQ(I,9) = (DTQ(I,1) - DTQ(I,2)) + &
&                  (DTQ(I,3) - DTQ(I,4)) + &
&                  (DTQ(I,5) - DTQ(I,6)) + &
&                  (DTQ(I,7) - DTQ(I,8))	! Sum (dG/dT - dL/dT)
!
!        RADIATIVEEQ(I,7) = RADIATIVEEQ(I,1) + RADIATIVEEQ(I,3) + &
!	        RADIATIVEEQ(I,5) 
!        RADIATIVEEQ(I,8) = RADIATIVEEQ(I,2) + RADIATIVEEQ(I,4) + &
!	        RADIATIVEEQ(I,6) 
!	RADIATIVEEQ(I,9) = RADIATIVEEQ(I,7) - RADIATIVEEQ(I,8)	
!	

! relative differences
	rel_ene(I,1)=1.D0-ENERGIES(I,1)/ENERGIES(I,2)	! 1-g/l FF
	rel_ene(I,2)=1.D0-ENERGIES(I,3)/ENERGIES(I,4)	! 1-g/l CBF

        IF (ENERGIES(I,6) .NE. 0.D0) THEN 
	  rel_ene(I,3)=1.D0-ENERGIES(I,5)/ENERGIES(I,6)	! 1-g/l rbf
        ELSE 
          IF (ENERGIES(I,5) .NE. 0.D0) STOP ' something wrong with rbf energies'
          rel_ene(I,3)=0.D0
        ENDIF 

        IF (ENERGIES(i,8) .NE. 0.D0) THEN 
	  rel_ene(I,4)=1.D0-ENERGIES(I,7)/ENERGIES(I,8)	! 1-g/l CBB
        ELSE 
          IF (ENERGIES(I,7) .NE. 0.D0) STOP ' something wrong with cbb energies'
          rel_ene(I,4)=0.D0
        ENDIF 
          
	REL_ENE(I,5)=1.D0-(ENERGIES(I,1)+ENERGIES(I,3)  &
&	  + ENERGIES(I,5)+ENERGIES(I,7))/(ENERGIES(I,2) &
&	  + ENERGIES(I,4)+ENERGIES(I,6)+ENERGIES(I,8))	! 1-G/L ALL	

	REL_ENE(I,6)=1.D0-JINT(I)/BINT(I)	        ! 1-J/B
	
! relative differences for partials (not required presently)
!	rel_ene2(I,1)=1.D0-ABS(DTQ(I,1))/ABS(DTQ(I,2))	! 1-abs(Dg)/abs(Dl) FF
!	rel_ene2(I,2)=1.D0-ABS(DTQ(I,3))/ABS(DTQ(I,4))	! 1-abs(Dg)/abs(Dl) CBF
!	rel_ene2(I,3)=1.D0-ABS(DTQ(I,5))/ABS(DTQ(I,6))	! 1-abs(Dg)/abs(Dl) RBF

!        IF (DTQ(I,8) .NE. 0.D0) THEN 
!	  rel_ene2(I,4)=1.D0-ABS(DTQ(I,7))/ABS(DTQ(I,8)) ! 1-g/l CBB
!        ELSE 
!          IF (DTQ(I,7) .ne. 0.D0) STOP ' something wrong with cbb partials'
!          rel_ene2(I,4)=0.D0
!        ENDIF  
!	REL_ENE2(I,5) = 0.D0

! absolute differences
	TAUX1(I,1)=ENERGIES(I,1)-ENERGIES(I,2)	! G - L FF
	TAUX1(I,2)=ENERGIES(I,3)-ENERGIES(I,4)	! G - L CBF
	TAUX1(I,3)=ENERGIES(I,5)-ENERGIES(I,6)	! G - L RBF
	TAUX1(I,4)=ENERGIES(I,7)-ENERGIES(I,8)	! G - L CBB
	TAUX1(I,5)=TAUX1(I,1) + TAUX1(I,2) + &
&                      TAUX1(I,3) + TAUX1(I,4)	
        TAUX1(I,6)=REL_ENE(I,5)                  ! 1-G/L ALL	 
	 
	TAUX2(I,1)=DTQ(I,1)-DTQ(I,2)	! dG/dT - dL/dT FF
	TAUX2(I,2)=DTQ(I,3)-DTQ(I,4)	! dG/dT - dL/dT CBF
	TAUX2(I,3)=DTQ(I,5)-DTQ(I,6)	! dG/dT - dL/dT RBF
	TAUX2(I,4)=DTQ(I,7)-DTQ(I,8)	! dG/dT - dL/dT CBB
	TAUX2(I,5)=TAUX2(I,1) + TAUX2(I,2) + &	
&		   TAUX2(I,3) + TAUX2(I,4)	

ENDDO ! depth loop


DELTAT=0.D0 ! T-Corr
!call EXPAN_COOLING(R,XNE,V,T,DELTATE,ENERGIES(:,9),DTQ(:,9),VDIV,OPT, &
!&                  IND2,RHO,XMLOSS,ESCALARADIO,ESCALAVELO) 
CALL MeanMolWeIght(meanmw,dummy)
DO I=1,ND
	TAUX3(I,1)=meanmw(I)
	TAUX3(I,2)=T(I)
	TAUX3(I,3)=TAUR(I)
        AUX= -ENERGIES(I,9)/DTQ(I,9)  ! delta T = (G-L)/(dL/dT-dG/dT)
! in case the temperature iteration (controlled by the partials) does
! not work, consider the following possibility (set taur(i) to 1.d-2). 
        IF(ENERDIFFOLD(I).NE.0..AND.DTOLD(I).NE.0..and.taur(i).lt.-100.) THEN
          dtqold(i)=(energies(i,9)-enerdiffold(i))/dtold(i) 
          if(enerdiffold(i)*energies(i,9).lt.0.d0) then
            auxold(i)=-energies(i,9)/dtqold(i) !bisect
          else if(abs(energies(i,9)).lt.abs(enerdiffold(i))) then
            auxold(i)=(.5d0*energies(i,9)-energies(i,9))/dtqold(i) 
          else
            auxold(i)=(.5d0*enerdiffold(i)-energies(i,9))/dtqold(i) 
          endif 
        ELSE
          AUXOLD(I)=aux
      ENDIF
        aux=auxold(i)
! restrict correction to 7.5%
        IF (ABS(AUX) .GT. 0.075*T(I)) THEN
          TAUX3(I,4)= AUX / ABS(AUX)*0.075*T(I) !correct sign
        ELSE
          TAUX3(I,4) = AUX
        ENDIF  
	TAUX3(I,5)=TAUX3(I,4)/T(I)	     ! delta T / T
	TAUX3(I,6)=XNE(I)		     ! electron density
!	TAUX3(I,5)=DELTAT(I)		     ! delta T from EXPAN_COOLING
	DELTAT(I) = TAUX3(I,4)	! ThB  deltaT-Corr 
ENDDO
ENERDIFFOLD=ENERGIES(:,9)

IF (OUTTHB) THEN
	call NCOR_NAME(NCOR,NCNAME)
ELSE
  NCNAME='_LA'
ENDIF  
	open (7,FILE=TRIM(MODNAM)//'/ENERGY_TC'//TRIM(NCNAME)//'.dat', &
&	  STATUS='UNKNOWN') !	
	rewind 7
	do I=1,ND
		write(7,FMT='(1X,I3,1X,9(G12.5,1X))') I,(ENERGIES(I,J),&
&		 J=1,9)
	enddo
	close (7)
	open (7,FILE=TRIM(MODNAM)//'/DERIVATIVES_TC'//TRIM(NCNAME)//'.dat', &
&	  STATUS='UNKNOWN')
	rewind 7
	do I=1,ND
       		write(7,FMT='(1X,I3,1X,9(G12.5,1X))') I,(DTQ(I,J), &
& 		 J=1,9)
	enddo
	close (7)
	open (7,FILE=TRIM(MODNAM)//'/ENERDIFF_TC'//TRIM(NCNAME)//'.dat', &
&  	  STATUS='UNKNOWN')
	rewind 7
	write(7,*) ' ND   FF (G-L)     CBF (G-L)     RBF (G-L)     CBB (G-L)    TOTAL     1-G/L ALL'
	do I=1,ND
       		write(7,FMT='(1X,I3,1X,6(G12.5,1X))') I,(TAUX1(I,J),&
&		J=1,6)
	enddo
	close (7)	
	open (7,FILE=TRIM(MODNAM)//'/DERIVADIFF_TC'//TRIM(NCNAME)//'.dat', &
&		STATUS='UNKNOWN')
	rewind 7
	write(7,*) 'dG/dT-dL/dT   FF      CBF           RBF        CBB          TOTAL       DUMMY  '
	do I=1,ND
       		write(7,FMT='(1X,I3,1X,5(G12.5,1X))') I,(TAUX2(I,J), &
&		  J=1,5)
	enddo
	close (7)	
	open (7,FILE=TRIM(MODNAM)//'/DELTAT_TC'//TRIM(NCNAME)//'.dat',&
&		STATUS='UNKNOWN')
	rewind 7
!	write(7,*) ' ND  MeanMolW      T (K)        Tau Ross      dT (K)         dT/T         XNE '
	do I=1,ND
       		write(7,FMT='(1X,I3,1X,6(G12.5,1X))') I,(TAUX3(I,J),&
&		J=1,6)
	enddo
	close (7)
	open (7,FILE=TRIM(MODNAM)//'/REL_ENERGY_TC'//TRIM(NCNAME)//'.dat', &
&         STATUS='UNKNOWN')
	rewind 7
	write(7,*) ' ND   1-g/l FF    1-g/l CBF      1-g/l RBF    1-g/l CBB    1-G/L ALL   1-J/B '
	do I=1,ND
		write(7,FMT='(1X,I3,1X,6(G12.5,1X))') I,(rel_ene(I,J), &
&			J=1,6)
	enddo
	close (7)

!	open (7,FILE=TRIM(MODNAM)//'/RADEQ_ENERGIES.dat',STATUS='UNKNOWN')
!	rewind 7
!	do I=1,ND
!		write(7,FMT='(1X,I3,1X,9(G12.5,1X))') I,(radIatIveeq(I,tecla),&
!&		tecla=1,9)
!	enddo
!	close (7)

! find point where relative differences are too low to allow for successfull correction

REL_ERR=TAUX1(:,6)

DO I=ND-1,1,-1
  IF(ABS(TAUX1(I,6)).GT.1.D-3) GOTO 10
ENDDO

DO I=ND-1,1,-1
  IF(ABS(TAUX1(I,6)).GT.1.D-4) GOTO 10
ENDDO

STOP ' REL DIFF IN HEATING/COOLING NEVER LARGER THAN 1.D-4!'

10 IREL=I+1


RETURN
END
!
!-----------------------------------------------------------------
!
SUBROUTINE NCOR_NAME(N,NNAME)

USE nlte_type
USE nlte_dim
USE fund_const

IMPLICIT NONE
INTEGER(I4B) :: N
CHARACTER*3 :: NNAME
INTEGER(I4B) :: I,J


I = N/10
IF (I .LT. 1) THEN
   NNAME=TRIM(CHAR(N+48))
ELSE 
  J = N - I*10
  NNAME=TRIM(CHAR(I+48))//TRIM(CHAR(J+48))
END IF
NNAME=TRIM(NNAME)

RETURN
END  	   
!
!***********************************************************************
!
! subroutines: recent ones (from vers. 9.0 on)
!
!***********************************************************************
!
SUBROUTINE OP_RBFSETUP(II,NUNNAME,NDATOSMAX,NRECORD,NDATOS,FILEOP)
!
! for formula 20
! changed Sept. 2016. most important, FRECFN1 is calculated and overwrites
! the start value (=FRECFN)
!
use nlte_type
use nlte_dim
use fund_const
use princesa_var, only: LABL, LABL4, FRECIN, FRECFN, FL
use nlte_var, only: IFRE, FRE, EOPDIFF, OP_FLAG, OP_FLAG2, &
              OPPATH, OP_DATA, MODNAM, FRECFN1
use nlte_opt, only: OPDEBUG


IMPLICIT NONE
!	..
!	.. parameters ..
INTEGER(I4B), PARAMETER :: IMAX=4000
!JP (July 2013)
INTEGER(I4B), PARAMETER :: NMAX_LOW = 2000 ! for structured part
!INTEGER(I4B), PARAMETER :: NMAX_HIGH = 200 ! for smooth high freq. tail
INTEGER(I4B), PARAMETER :: NMAX_HIGH = 2000 ! for smooth high freq. tail

REAL(DP), PARAMETER :: RESOL=100. ! AVERAGE resolution of Gaussian, &
                                  ! corresponding to 3000 km/s (FWHM)
!
!	..
!	.. scalar arguments ..

INTEGER(I4B) :: II, NDATOSMAX, NRECORD, NDATOS
REAL (DP) :: NUNNAME
CHARACTER*30 :: FILEOP
!
!	..
!	.. local scalars

INTEGER(I4B) :: J,ESTADO,K,L1,L2,IVOID,NOUT, L2START
REAL(DP) :: XNAUX, ATMRYD,DVOID,ERYD,RELDIFF, PEQ=0.D0, Q, Q1, ALPHA, XNAUXMIN

CHARACTER*30 :: CHECKFILE
CHARACTER*80 :: DUMMYCHAR
!
!	..
!	.. local arrays ..

REAL(DP) :: FGS(IMAX),PICS(IMAX),AUX2(IMAX),FGSOLD(2)
REAL(DP), DIMENSION(NMAX_LOW+NMAX_HIGH) :: EOUT,PICSOUT

IF(NDATOSMAX.GT.IMAX) STOP ' NDATOSMAX > IMAX'

FGS(:)=0.D0
PICS(:)=0.D0
ATMRYD=0.D0
ESTADO=1
ERYD =0.D0
ALPHA=0.D0
RELDIFF=0.D0

OP_FLAG2(II)=.FALSE.

CALL OP_CODEX(INT(NUNNAME),FILEOP,CHECKFILE,ATMRYD) ! filename decoder (CODEX)

OPEN(771,FILE=OPPATH//TRIM(FILEOP),STATUS='OLD',ACCESS='DIRECT',  &  
&		    RECL=2*NDATOSMAX*8,IOSTAT=ESTADO)  

IF (ESTADO.NE.0) THEN 
    PRINT *, 'FILE OP_RBF ',OPPATH//TRIM(FILEOP),' NOT FOUND'
    STOP
ENDIF

READ(771,REC=NRECORD)(FGS(K),K=1,NDATOS),(PICS(J),J=1,NDATOS)
CLOSE(771)  

IF (OPDEBUG) THEN
   DUMMYCHAR=trim(modnam)//'/opori-crossbf-'//trim(labl(labl4(ii)))//'.ascii'
   OPEN (771,FILE=TRIM(DUMMYCHAR),STATUS='UNKNOWN')
   WRITE(771,FMT='(1X,A10,3(1X,G18.8))') TRIM(LABL(LABL4(II))),ERYD, &
&    FL(LABL4(II))/CLIGHT,FRECIN(II)
     DO K=1,NDATOS
 	 XNAUX = FGS(K)/CLIGHT
 	 WRITE(771,FMT='(1X,2(G18.8,1X))') XNAUX,PICS(K)
     ENDDO   
   CLOSE(771)	       
ENDIF    

!interpolate OPACITY data when zero values are present

DO K=1,NDATOS
  IF (PICS(K) .EQ. 0.) THEN
      GOTO 56
  ENDIF 
ENDDO

GOTO 57

56 CONTINUE
IF (OPDEBUG) THEN
PRINT*
PRINT *,'WARNING!!! ZEROES FOUND IN OPACITY DATA FOR LEVEL ',TRIM(LABL(LABL4(II)))
ENDIF

CALL OP_FIT(FGS,PICS,IMAX,NDATOS) 

IF (OPDEBUG) THEN
   DUMMYCHAR=trim(modnam)//'/opori-mod-crossbf-'//trim(labl(labl4(ii)))//'.ascii'
   OPEN (771,FILE=TRIM(DUMMYCHAR),STATUS='UNKNOWN')
   WRITE(771,FMT='(1X,A10,3(1X,G18.8))') TRIM(LABL(LABL4(II))),ERYD, &
&    FL(LABL4(II))/CLIGHT,FRECIN(II)
     DO K=1,NDATOS
 	 XNAUX = FGS(K)/CLIGHT
 	 WRITE(771,FMT='(1X,2(G18.8,1X))') XNAUX,PICS(K)
     ENDDO   
   CLOSE(771)	       
ENDIF      
!*******************
!
!
!   Is there any offset in the (energy) level definition between the OP data
!   and DETAIL?
!   This is only checked once.
!	
!   We need to compare the FL variable (ionization energy w.r.t. ground-state)
!   with the OP energy (w.r.t. ground-state), not FRECIN (actual ionization energy),
!   because an ionization transition to an excited level is possible. 
!   Remember that [FL] = s^{-1};
!   NOTE: variable FREFN consistent with FL (though in Kayser)
!  
57 IF (.NOT. OP_FLAG2(II)) THEN	
    ESTADO=1
    OPEN(771,FILE=OPPATH//TRIM(CHECKFILE),STATUS='OLD',IOSTAT=ESTADO)
    IF (ESTADO.NE.0) THEN 
	PRINT*,' CHECK FILE OP_RBF ',TRIM(CHECKFILE),' NOT FOUND'
	STOP
    ENDIF
    READ(771,*) DUMMYCHAR
    READ(771,*) DUMMYCHAR

    DO L1=1,NRECORD
            READ(771,*) IVOID,IVOID,DVOID,DVOID,DVOID,ERYD,DVOID
    ENDDO
    CLOSE(771)
    ERYD = ERYD*(-1.D0)*ATMRYD	
!    eopdiff(ii) = eryd - frecin(ii)    
!    reldiff=eopdiff(ii)/frecin(ii)
!
    XNAUX = FL(LABL4(II))/CLIGHT
    IF(ABS(1.-XNAUX/FRECFN(II)).GT.1.D-15) STOP ' INCONSISTENCY BETWEEN FL AND FRECFN!'
    EOPDIFF(II) = ERYD - XNAUX
    RELDIFF=EOPDIFF(II)/XNAUX
!
! WRITE(*,*) ' Relative energy difference at the edge ', &
!    TRIM(LABL(LABL4(II))),' ', RELDIFF
!
! Just a warning, proceed at user's risk
    IF (ABS(RELDIFF) .GT. 0.2) THEN      
       PRINT*,' RBF OP-DATA WARNING -- level ',TRIM(LABL(LABL4(II))), &
&	      ' shows a large (rel) energy difference ',RELDIFF
       STOP ' TOO LARGE DIFFERENCE IN IONIZATION EDGES OP-DATA VS. DETAIL' 
    ENDIF

    IF (ABS(RELDIFF) .GT. 1.D-3) THEN       
	PRINT*
	print *,'RBF OP-DATA WARNING -- shifting cross section for ',TRIM(LABL(LABL4(II))), &
&              ' from ',ERYD,' cm^(-1) to ',ERYD-EOPDIFF(II),' cm^(-1)'
        FGSOLD(1) = FGS(NDATOS)/CLIGHT
	FGSOLD(2) = FGS(1)/CLIGHT
        FGS(:) = FGS(:) - ( EOPDIFF(II)*CLIGHT ) 
    ENDIF	  	 
!   
    OP_FLAG2(II)=.TRUE.
    IF (OPDEBUG) THEN
      DUMMYCHAR=trim(modnam)//'/op-crossbf-'//trim(labl(labl4(ii)))//'.ascii'
      OPEN (771,FILE=TRIM(DUMMYCHAR),STATUS='UNKNOWN')
      WRITE(771,FMT='(1X,A10,3(1X,G18.8))') TRIM(LABL(LABL4(II))), &
&       ERYD,FL(LABL4(II))/CLIGHT,FRECIN(II)
      DO K=1,NDATOS
	 XNAUX = FGS(K)/CLIGHT
	 WRITE(771,FMT='(1X,2(G18.8,1X))') XNAUX,PICS(K)
      ENDDO   
      CLOSE(771)	       

      DUMMYCHAR=trim(modnam)//'/detsrf-crossbf-'//trim(labl(labl4(ii)))//'.ascii'
      OPEN (771,FILE=TRIM(DUMMYCHAR),STATUS='UNKNOWN')
      WRITE(771,FMT='(1X,A10,3(1X,G18.8))') TRIM(LABL(LABL4(II))),ERYD, &
&        FL(LABL4(II)),FRECIN(II)*CLIGHT
      WRITE(771,FMT='(1X,A2,1X,G18.8,1X,A6)') 'F1',FGS(1),TRIM(LABL(LABL4(II)))
      WRITE(771,FMT='(1X,A2,1X,G18.8,1X,A6)') 'F2',FGS(NDATOS), &
&        TRIM(LABL(LABL4(II)))
      
      CLOSE(771)	       
    ENDIF      

ENDIF	  
!
AUX2(:) = FGS(:) / CLIGHT
!
IF (AUX2(NDATOS) .GT. FRECIN(II)) THEN
	PRINT*,' OP STOP  --- rbf cross-section does not include the ionization edge' 
	PRINT*,'   Level ',trim(labl(labl4(ii))),'  -----   record ',nrecord
	PRINT*,'   Level defined at ',fl(labl4(ii))/clight,' cm^{-1}'
	PRINT*,'   Ionization edge located at ',frecin(ii),' cm^{-1}'
	PRINT*,'   while the available range is ',aux2(ndatos),' -- ',aux2(1),' cm^{-1} '
	PRINT*,'   <--> before correction was: ',fgsold(1),' -- ',fgsold(2)
	PRINT*,' CHECK THE UPPER LEVEL for this transition !!!! '
	STOP
ENDIF		
!
! smoothing with Gaussian: energy units used!
! the logical variable controls output (no output if set to .false.)
!
!CALL SMOOTH(AUX2,PICS,NDATOS,RESOL,NMAX_LOW,NMAX_HIGH,FRECIN(II),EOUT,PICSOUT,NOUT,.false.)
!JO: changed Sept. 2016, to allow for frequencies lower than the
!standard ionization energy. Note that for ionizations to excited levels,
!resonances are possible in the range down or even below the ground-state edge
CALL SMOOTH(AUX2,PICS,NDATOS,RESOL,NMAX_LOW,NMAX_HIGH,AUX2(NDATOS),EOUT,PICSOUT,NOUT,.FALSE.)
If(EOUT(1).NE.FGS(1)/CLIGHT) &
&     STOP ' error in fgs(1)' !highest energy, MUST be identical 
If(ABS(1.D0-EOUT(NOUT)/(FGS(NDATOS)/CLIGHT)).GT.1.D-14) &
&     STOP ' error in fgs(ndatos)' !lowest energy, identical within precision 
EOUT(NOUT)=FGS(NDATOS)/CLIGHT ! to remain consistent
!
! Once the cross-section has been read and convolved, &
! we map the cross-section on FASTWIND's frequency grid

! units: fre, frecin, eout in cm^-1; xnaux, fgs Hz
! energy grid fre  ordered from red to blue
! energy grid eout ordered from blue to red

L2START=NOUT-1

!JO Sept2016; for some transitions, data not down to groundstate edge
IF(EOUT(NOUT).GT.FRECFN(II)) THEN 
   IF(EOUT(NOUT).GT.FRECIN(II)) STOP ' SOMETHING WRONG WITH OP-DATA (SUBR. OP_RBFSETUP)'
ENDIF

XNAUXMIN=0.
DO L1=1,IFRE
    XNAUX=FRE(L1) 
!    IF (FRE(L1) .LE. EOUT(1) .AND. FRE(L1) .GE. FRECIN(II) ) THEN
!JO: changed Sept. 2016, to allow for frequencies lower than the
!standard ionization energy. Note that for ionizations to excited levels,
!resonances are possible in the range down or even below the ground-state edge
    
    IF (XNAUX .LE. EOUT(1) .AND. XNAUX .GE. EOUT(NOUT)) THEN
    IF(XNAUXMIN.EQ.0.) XNAUXMIN=XNAUX
!   emax > fre(l1) > edge, starting close to edge
!   we do not allow lower frequencies than the effective edge

        DO L2=L2START,1,-1
	   IF (XNAUX.LE.EOUT(L2) .AND. XNAUX.GT.EOUT(L2+1)) THEN
	      J=L2
	      K=L2+1
              IF(PICSOUT(J) .LE. 0.D0) STOP ' sout(j) le 0.'  
              IF(PICSOUT(K) .LE. 0.D0) STOP ' sout(k) le 0.'  
              GOTO 10
	   ENDIF	 
	ENDDO
        STOP ' FRE NOT FOUND IN EOUT'
! logarithmic interpolation  
10      L2START=L2
        Q=LOG10(XNAUX/EOUT(K))/LOG10(EOUT(J)/EOUT(K))
        Q1=1.D0-Q
        ALPHA = Q*LOG10(PICSOUT(J))+Q1*LOG10(PICSOUT(K))
        ALPHA = 10.D0**ALPHA
		
        IF (ALPHA .LE. 0.D0) THEN
	   PRINT*,'ERROR IN ',trim(oppath//trim(fileop)),' alpha zero or negative' 
	   PRINT *,'   record ',nrecord,'  level ',trim(labl(labl4(ii)))

           DUMMYCHAR=trim(modnam)//'/crossbf-error-'//trim(labl(labl4(ii)))//'.dat'
	   OPEN (77,FILE=TRIM(DUMMYCHAR),STATUS='UNKNOWN')
	   DO K=1,IFRE
		WRITE(77,*) FRE(K),OP_DATA(II,K)
	   ENDDO   
	   CLOSE(77)	       
	   STOP
	ENDIF   

	OP_DATA(II,L1) = ALPHA

    ELSE IF (XNAUX .GT. EOUT(1)) THEN  ! nu**3 extrapolation for high energies, just in case
	   OP_DATA(II,L1)=PICSOUT(1)*(EOUT(1)/XNAUX)**3

    ELSE ! just in case, we will never reach this frequency
	   OP_DATA(II,L1)=0.D0	
    ENDIF

ENDDO

!redefine FRECFN1
FRECFN1(II)=MAX(FRECFN(II),XNAUXMIN)   

!PRINT*,LABL(LABL4(II)),FRECIN(II),FRECFN(II),FRECFN1(II)

IF (OPDEBUG) THEN
   DUMMYCHAR=trim(modnam)//'/fstwnd-crossbf-'//trim(labl(labl4(ii)))//'.ascii'
   OPEN (771,FILE=TRIM(DUMMYCHAR),STATUS='UNKNOWN')
   WRITE(771,FMT='(1X,A10,2(1X,G18.8))') TRIM(LABL(LABL4(II))),FL(LABL4(II))/CLIGHT,FRECIN(II)
   DO K=1,IFRE
	XNAUX=FRE(K)*CLIGHT
	WRITE(771,FMT='(1X,2(G18.8,1X))') FRE(K),OP_DATA(II,K)
   ENDDO   
   CLOSE(771)	       
ENDIF   

RETURN
END
!
!-----------------------------------------------------------------
!
SUBROUTINE OP_RBFSETUP1(II,KEY,IZ,NRECORD,FILEOP)
!
! for formula 101
! similar to OP_RBFSETUP, adapted for new DETAIL format
!
use nlte_type
use nlte_dim
use fund_const
use princesa_var, only: LABL, LABL4, FRECIN, FL

use nlte_var, only: OPPATH, MODNAM, IFRE, FRE, OP_DATA

use nlte_opt, only: OPDEBUG

IMPLICIT NONE
!	..
!	.. parameters ..
INTEGER(I4B), PARAMETER :: NMAX_LOW = 1000 ! for structured part
!INTEGER(I4B), PARAMETER :: NMAX_HIGH =200 ! for smooth high freq. tail
INTEGER(I4B), PARAMETER :: NMAX_HIGH =1000 ! for smooth high freq. tail

REAL(DP), PARAMETER :: RESOL=100. ! AVERAGE resolution of Gaussian, &
                                  ! corresponding to 3000 km/s (FWHM)
!
!	..
!	.. scalar arguments ..

INTEGER(I4B) :: II, NRECORD,IZ
CHARACTER*30 :: FILEOP
CHARACTER*6  :: KEY 
!
!	..
!	.. local scalars

INTEGER(I4B) :: ESTADO,ICODE,NLAB,J,K,NDATOS,ILAB,NOUT,L1,L2,L2START 

REAL(DP) :: XNAUX, ERYD, Q, Q1, ALPHA

CHARACTER*30 :: LABEL,LABEL1,CHECKFILE

CHARACTER*80 :: DUMMYCHAR

LOGICAL :: LAB0 ! indicates that alpha = 0 for a certain range close
                ! to the edge
!
!	..
!	.. local arrays ..

REAL(DP), DIMENSION(:), ALLOCATABLE:: FGS, PICS, AUX2

REAL(DP), DIMENSION(NMAX_LOW+NMAX_HIGH) :: EOUT,PICSOUT

CHARACTER*6, DIMENSION(30) :: NAMES

DATA NAMES/'H ','HE','LI','BE','B ','C ', &
&          'N ','O ','F ','NE','NA','MG', &
&          'AL','SI','P ','S ','CL','AR',&
&          'K ','CA','SC','TI','V ','CR',&
&          'MN','FE','CO','NI','CU','ZN'/


WRITE(LABEL,*) NRECORD
LABEL='LABEL'//ADJUSTL(LABEL)

ERYD =0.D0
LAB0=.FALSE.

OPEN(771,FILE=OPPATH//TRIM(FILEOP),STATUS='OLD',IOSTAT=ESTADO)  

IF (ESTADO.NE.0) THEN 
    PRINT *, 'FILE OP_RBF ',OPPATH//TRIM(FILEOP),' NOT FOUND'
    STOP
ENDIF

READ(771,*) ICODE, NLAB
J=ICODE/100
IF (TRIM(NAMES(J)).NE.TRIM(KEY)) STOP ' OP_RBF FILE INCONSISTENT (ELEMENT)'
K=ICODE-J*100
IF (K+1.NE.IZ)  STOP ' OP_RBF FILE INCONSISTENT (ION)'

DO 
   READ(771,*,END=10) LABEL1,NDATOS
   IF(TRIM(LABEL1).EQ.TRIM(LABEL)) GOTO 20
   READ(771,*) (XNAUX,J=1,2*NDATOS)
ENDDO

10 STOP ' LABEL NOT FOUND IN OP_RBF FILE'

20 ALLOCATE(FGS(NDATOS+1),PICS(NDATOS+1),AUX2(NDATOS+1)) ! one additional point may be required

READ(771,*) (FGS(J),J=1,NDATOS),(PICS(J),J=1,NDATOS)

CLOSE(771)  

IF (OPDEBUG) THEN
   DUMMYCHAR=trim(modnam)//'/opori-crossbf-'//trim(labl(labl4(ii)))//'.ascii'
   OPEN (771,FILE=TRIM(DUMMYCHAR),STATUS='UNKNOWN')
   WRITE(771,FMT='(1X,A10,3(1X,G18.8))') TRIM(LABL(LABL4(II))),ERYD, &
&    FL(LABL4(II))/CLIGHT,FRECIN(II)
     DO K=1,NDATOS
 	 XNAUX = FGS(K)/CLIGHT
 	 WRITE(771,FMT='(1X,2(G18.8,1X))') XNAUX,PICS(K)
     ENDDO   
   CLOSE(771)	       
ENDIF      
!
!   Is there any offset in the (energy) level definition between the OP data
!   and DETAIL?
!   In contrast to OP_RBFSETUP, this is NOT checked here, since we
!   do not have the OP energy info. Thus, we do not shift the energy, 
!   and "believe" that everything is OK
!   (Norbert P. has promised that he has done the control by hand, &
!    and shifted, if necessary).

!   If such an info will become available, we suggest to proceed as in the
!   alternative routine

!
AUX2 = FGS / CLIGHT
!
IF (AUX2(NDATOS) .GT. FRECIN(II)) THEN
IF (OPDEBUG) THEN
	PRINT*,' OP --- rbf cross-section does not include the ionization edge' 
	PRINT*,'   Level ',trim(labl(labl4(ii))),'  -----  ',label
	PRINT*,'   Level defined at ',fl(labl4(ii))/clight,' cm^{-1}'
	PRINT*,'   Ionization edge located at ',frecin(ii),' cm^{-1}'
	PRINT*,'   while the available range is ',aux2(ndatos),' -- ',aux2(1),' cm^{-1} '
        PRINT*,'   Assumed that cross-section is low in missing area !!!! '
ENDIF        
LAB0=.TRUE.        
ENDIF		

IF (LAB0) THEN ! one additional point at low cross-section, to prevent problems
               ! in the photo integration routine (integral does not change)  
NDATOS=NDATOS+1
AUX2(NDATOS)=FRECIN(II)
FGS(NDATOS)=FRECIN(II)*CLIGHT
PICS(NDATOS)=PICS(NDATOS-1)/1000.
ENDIF
!
! smoothing with Gaussian: energy units used!
! the logical variable controls output (no output if set to .false.)
!
CALL SMOOTH(AUX2,PICS,NDATOS,RESOL,NMAX_LOW,NMAX_HIGH,FRECIN(II),EOUT,PICSOUT,NOUT,.FALSE.)
!
If(EOUT(1).NE.FGS(1)/CLIGHT) &
&     STOP ' error in fgs(1)' !highest energy, MUST be identical 
!print*,abs(1.-eout(nout)/(fgs(ndatos)/clight))
If(ABS(1.D0-EOUT(NOUT)/(FGS(NDATOS)/CLIGHT)).GT.5.D-14) &
&     STOP ' error in fgs(ndatos)' !lowest energy, identical within precision 
EOUT(NOUT)=FGS(NDATOS)/CLIGHT ! to remain consistent
!
DEALLOCATE(FGS,PICS,AUX2)

! Once the cross-section has been read and convolved, &
! we map the cross-section on FASTWIND's frequency grid

! units: fre, frecin, eout in cm^-1; xnaux, fgs Hz
! energy grid fre  ordered from red to blue
! energy grid eout ordered from blue to red

L2START=NOUT-1

DO L1=1,IFRE
    XNAUX=FRE(L1) 
    IF (FRE(L1) .LE. EOUT(1) .AND. FRE(L1) .GE. FRECIN(II) ) THEN
!   emax > fre(l1) > edge, starting close to edge
                          ! we do not allow lower frequencies than 

        DO L2=L2START,1,-1    ! the edge one
	   IF (XNAUX.LE.EOUT(L2) .AND. XNAUX.GE.EOUT(L2+1)) THEN
	      J=L2
	      K=L2+1
! can happen under bad conditions (high resol) due to numerical problems of FFT
              IF(PICSOUT(J) .LE. 0.D0) STOP ' sout(j) le 0.'  
              IF(PICSOUT(K) .LE. 0.D0) STOP ' sout(k) le 0.'  
              GOTO 50
	   ENDIF	 
	ENDDO	      
!        IF(LAB0) THEN ! in this case, everything should be OK
!        ALPHA=0.
!        GOTO 60
!        ENDIF
        STOP ' fre not found in eout'
! logarithmic interpolation  
50      L2START=L2
        Q=LOG10(XNAUX/EOUT(K))/LOG10(EOUT(J)/EOUT(K))
        Q1=1.D0-Q
        ALPHA = Q*LOG10(PICSOUT(J))+Q1*LOG10(PICSOUT(K))
        ALPHA = 10.D0**ALPHA
		
        IF (ALPHA .LE. 0.D0) THEN
	   PRINT*,'ERROR IN ',trim(oppath//trim(fileop)),' alpha zero or negative' 
	   PRINT *,'   label ',label,'  level ',trim(labl(labl4(ii)))

           DUMMYCHAR=trim(modnam)//'/crossbf-error-'//trim(labl(labl4(ii)))//'.dat'
	   OPEN (77,FILE=TRIM(DUMMYCHAR),STATUS='UNKNOWN')
	   DO K=1,IFRE
		WRITE(77,*) FRE(K),OP_DATA(II,K)
	   ENDDO   
	   CLOSE(77)	       
	   STOP
	ENDIF   

60	OP_DATA(II,L1) = ALPHA

    ELSE IF (XNAUX .GT. EOUT(1)) THEN  ! nu**3 extrapolation for high energies, just in case
	   OP_DATA(II,L1)=PICSOUT(1)*(EOUT(1)/XNAUX)**3

    ELSE ! just in case, we will never reach this frequency
	   OP_DATA(II,L1)=0.D0	
    ENDIF

ENDDO

IF (OPDEBUG) THEN
   DUMMYCHAR=trim(modnam)//'/fstwnd-crossbf-'//trim(labl(labl4(ii)))//'.ascii'
   OPEN (771,FILE=TRIM(DUMMYCHAR),STATUS='UNKNOWN')
   WRITE(771,FMT='(1X,A10,2(1X,G18.8))') TRIM(LABL(LABL4(II))),FL(LABL4(II))/CLIGHT,FRECIN(II)
   DO K=1,IFRE
	XNAUX=FRE(K)*CLIGHT
	WRITE(771,FMT='(1X,2(G18.8,1X))') FRE(K),OP_DATA(II,K)
   ENDDO   
   CLOSE(771)	       
ENDIF   

PRINT*,' PHOTO-CROSS SECTION FOR LEVEL',trim(labl(labl4(ii))),' CONVOLVED WITH GAUSSIAN'
RETURN
END
!
!-----------------------------------------------------------------
!
subroutine smooth(e,s,n,resol,nmax_low,nmax_high,edge,eout,sout,nout,message)

use nlte_type
use fund_const

IMPLICIT NONE

integer(i4b), intent(in) :: n,nmax_low,nmax_high
real(dp), dimension(n), intent(in) :: e,s
real(dp), intent(in) :: resol,edge
logical :: lab0,message,low

real(dp), dimension(nmax_low+nmax_high), intent(out) :: eout, sout

integer(i4b), intent(out) :: nout

integer(i4b) :: nmax,i,istart

real(dp) :: dsx,rdst_max,ratio

nmax=nmax_low+nmax_high
eout=0.
sout=0.

if (4*resol .gt. nmax_low) stop ' nmax too low (change resol?)' 
nout=4*nint(resol)

! define smooth and non-smooth regime(s)
rdst_max = 0.

! ordered from high to low energies
if(e(1).lt.e(n)) stop ' wrong order of energy in crossbf file'

do i = 1,n-1
dsx = (1.-e(i+1)/e(i))
  rdst_max = max(rdst_max,dsx)
enddo

!JP (July 2013)
low=(edge.lt.10000.) ! for low ionisation stages with edge > 10000 A
if (low) then
  ratio=40.
else
  ratio=20.
endif

istart=1

5 do i=istart,n-1
dsx = (1.-e(i+1)/e(i))
!print*,istart,i,dsx,rdst_max,e(i),s(i)
if(dsx.lt.0.05*rdst_max) exit
enddo

i=max(i,2) ! in case i=1
! changed from 10 to 20 in vers. 9.2.5, to allow for Si2 cross-sect.
! additional change by JP (July 2013) to allow for low-enery edges (e.g., MgI)
if (e(i-1)/max(e(n),edge).gt.ratio) then ! inclusion of e(i) to obtain better overlap
  ! but might be "normal" edge
  ! "real" edge (which differs from e(n), particularly if ionisation to excited state) included
istart=i+1
if (istart .ge. n-1) &
& stop ' resonance part too broad, further reduction required'
goto 5
endif

!in case that everything smooth
if(i.eq.n) then
if(message) then
print*
print*,' smooth distribution from',e(1),'to',e(i)
print*,' no convolution required'
print*
endif
eout(1:i)=e
sout(1:i)=s
nout=n
return
endif

if(message) then
print*
print*,' smooth energy tail from',e(1),'to',e(i)
print*,'     resonance part from',e(i+1),'to',e(n)
print*
endif

call convol(e(i-1:n),s(i-1:n),n+2-i,eout,sout,nout,resol,message,low)

! add high freq. tail
if(sout(1).ne.s(i-1)) stop ' error in signal at transition freq.'

nout=nout+i-2
if(nout.gt.nmax) then
  print*,nout,nmax
  stop ' nout > nmax'
endif

eout=cshift(eout,-(i-2))
sout=cshift(sout,-(i-2))
eout(1:i-2)=e(1:i-2)
sout(1:i-2)=s(1:i-2)

!for tests
!open(1,file='cross_out')
!do i=1,nout
!  write(1,*),eout(i),sout(i)
!enddo
!close(1)

end
!
!-----------------------------------------------------------------
!
subroutine convol(e,s,n,eout,sout,nout,resol,message,low)
!
! convolution with gaussian of width resol (e/de, FWHM, see below)
! (assumed at the center of the given range)
!
! always resample input, since padding on both sides necessary  
!  
! n is original dimension (e,s)
! n1 refers to orginal plus padded data 
!  
! convolved signal s on modified, equidistant grid e is returned as sout(eout)
! (typically, 4 points per average resol. element)

use nlte_type
use fund_const

IMPLICIT NONE

integer(i4b), parameter :: nwmax=6 !maximum width of gaussian, in doppler units

integer(i4b) :: n,nout
real(dp), dimension(n), intent(in) :: e,s
real(dp), dimension(nout), intent(out) :: eout,sout
real(dp), allocatable, dimension(:) :: x,y,sxx,syy,px,py,logx,logy
real(dp) :: resol
logical :: message, swap, low

real(dp), parameter :: rsmpl_fac=0.7d0, max_resol=30000.  

integer(i4b) :: n1, nn, pow, k, i, j, kk, min_cnr, max_cnr, cnr
real(dp) :: s1,sn,x1,xn,rdst, rdst_min, rdst_max, dsx, meanw, eps, diff, &
&           xpow,q,q1,fac,emean,vdop,dedop,percent,pxunit,errmax


if(max_resol.lt.10.*resol) stop ' increase max_resol'
n1=n+2*nwmax ! account for additional points for padding

allocate(x(n1),y(n1))

if (e(1) .gt. e(n)) then ! change order (assumed from low to high)
  swap=.true.
  do k=1,n
   i=n+1-k
   x(i)=e(k)
   y(i)=s(k)
  enddo
else
  swap=.false.
  x(1:n)=e
  y(1:n)=s
endif  

s1=y(1)
sn=y(n)
x1=x(1)
xn=x(n)

fac = 2.*sqrt(alog(2.))

! JP (July 2013)
! in most cases, resol assumed to be valid at center of range
if(.not.low) then
  emean      = 0.5*(x1+xn)
else
  emean      = 0.75*x1+0.25*xn !x1+0.25(xn-x1)
endif

vdop=clight/(resol*fac)
dedop = vdop*(emean/clight)

! end padding, assuming symmetric response function (refering to no. of points)
do k=1,nwmax
  x(n+k)=xn+k*dedop
  y(n+k)=sn ! constant end
enddo

x=cshift(x,-nwmax) ! shift by nwmax to the right
y=cshift(y,-nwmax)

! padding at begin, again assuming symmteric response
do k=1,nwmax
  x(k)=x1-(nwmax+1-k)*dedop
  y(k)=s1 ! constant beginning 
enddo

if(x(nwmax+1) .ne. x1) stop ' error in x1'
if(x(n+nwmax) .ne. xn) stop ' error in xn'

if(message) then
  print*,' min(e) = ',x1,' incl. padded reg.',x(1)
  print*,' max(e) = ',xn,' incl. padded reg.',x(n1)
  print*
endif  

allocate(logx(n1),logy(n1))
logx=log10(x)
logy=log10(y)

rdst = (x(n1)-x(1))/(n1-1) ! resampling distance
!
! resample x-vector always
!
! dx from actual minimum * rsmpl_fac or max. resol

rdst_min = 1.d10
rdst_max = 0.
do k = 1,n1-1
dsx = x(k+1)-x(k)
  rdst_min = min(rdst_min,dsx)
  rdst_max = max(rdst_max,dsx)
enddo

! JP (July 2013)
if(.not.low) then
  meanw      = 0.5*(x(1)+x(n1))
else
  meanw      = 0.75*x(1)+0.25*x(n1) !x1+0.25(xn1-x1)
endif

eps        = meanw/max_resol
diff       = rdst_max-rdst_min 

if (message) then
  print*,' Max(delta dE): ', diff
  print*
  print*,' RDMIN = ',rdst_min,', RDMAX = ',rdst_max, ' EPS = ',eps
  print*
endif

rdst = max(rdst_min * rsmpl_fac, eps) 
nn = (x(n1)-x(1))/rdst + 1

xpow = log(dble(nn))/log(2.d0)
pow = int(xpow+1.e-15)

!print*, xpow,pow,nn,low

if(pow .le. 2)  stop ' too few points in convol!'

if(.not.low .and. pow .gt. 15) then
if (message) then
  print*,' pow = ',pow,': too many points in convol!'
  print*,' reset to pow =15'
endif  
pow=15
endif

if(low .and. pow .gt. 16) then
if (message) then
  print*,' pow = ',pow,': too many points in convol!'
  print*,' reset to pow =16'
endif  
pow=16
endif

pow = pow + 1
nn = 2**pow
  
rdst = (x(n1)-x(1))/(nn-1)

if(rdst .gt. rdst_min .and. message) then
     print*,' WARNING WARNING WARNING WARNING WARNING WARNING WARNING!' 
     print*,' WARNING WARNING WARNING WARNING WARNING WARNING WARNING!' 
     print*,' WARNING WARNING WARNING WARNING WARNING WARNING WARNING!'
     print*
     print*,' rdst > rdst_min: extreme narrow spikes may be not well sampled!'
     print*
endif
   
allocate(sxx(nn),syy(nn))

do k=0,nn-1
  sxx(k+1)=x(1)+k*rdst
enddo

if (abs(1.d0-sxx(1)/x(1)) .gt.1.d-14) stop ' error in sxx(1)'
if (abs(1.d0-sxx(nn)/x(n1)).gt.1.d-14) stop ' error in sxx(nn)'
   
!first part, padded
do j=1,nn
  if(sxx(j).ge.x1) exit
  syy(j)=s1
enddo

kk=1

! interpolation of orginal signal onto fine grid
do k=j,nn-1

     if(sxx(k).gt.xn) exit ! padded region 

     do i=kk,n1-1
       if(sxx(k).gt.x(i) .and. sxx(k).le. x(i+1)) goto 10
     enddo
     stop ' sxx(k) not found in x'
10   kk=i

!    logarithmic interpolation
     q=(log10(sxx(k))-logx(i+1))/(logx(i)-logx(i+1))
     q1=1.d0-q
     syy(k)=q*logy(i)+q1*logy(i+1)
     syy(k)=10.**syy(k)
enddo

! padded end region
do i=k,nn
  syy(i)=sn
enddo

if (message) then
   print*,' resampled grid with resampling distance: ',rdst
   print*,' resampled spectrum has ',nn,' points'
   print*
endif
 
! 
! Gauss profile
!

min_cnr = 5                  
max_cnr = nn*0.8

if (message) then 
   write(*,FMT="(' Vdop=',F7.0,' km/s  <=>',F7.0,' Kayser')") vdop/1.e5, dedop 
   write(*,FMT="(' FWHM=',F7.0,' km/s  <=>',F7.0,' Kayser')") fac*vdop/1.e5,fac*dedop
   print*
endif

cnr = (4.*dedop/rdst) + 1       

if (max_cnr .le. cnr) then
   percent = cnr/FLOAT(nn)*100.
   write(*,FMT="(' Gauss profile is very broad:', i2, &
&         ' % of spectrum points are forming the profile')") percent
endif   

if (min_cnr .ge. cnr) then
   write(*,FMT="(' Gauss profile is very narrow: only ', i2, &
&         ' data points are forming the profile')") cnr
endif

allocate(px(nn),py(nn))

pxunit=rdst/dedop
do i=1,nn
  k=i-1-nn/2
  px(i) = k*pxunit
enddo

py=0.

where(abs(px) .le. float(nwmax))
 py = exp(-(px**2))/(sqrt(pi)*dedop)
endwhere

py=cshift(py,-nn/2) ! wrap around zero


!check whether zero padding was successful:
!on the left  side, signal has to be e1 where py has finite values
!on the right side, signal has to be en where py has finite values

do k=1,nn
  if(py(k).eq.0.d0) exit
  if(syy(k).ne.s1) stop ' error in padding on right side'
enddo

do k=nn-1,1,-1
  if(py(k).eq.0.d0) exit
  if(syy(k).ne.sn) stop ' error in padding on right side'
enddo

!for tests
!do k=1,nn
!  print*,k,sxx(k),syy(k),py(k)
!enddo

call conv_spec(sxx,syy,py,nn)       

!do k=1,nn
!  print*,k,sxx(k),syy(k)
!enddo

deallocate(px,py)

! interpolation onto appropriate freq. grid
! (4 points per average resol. element)
rdst=(xn-x1)/(nout-1)

if(swap) then ! change order

do k=1,nout
  eout(k)=xn-(k-1)*rdst  
enddo

kk=nn

do k=1,nout
     do i=kk,2,-1
       if(eout(k).le.sxx(i) .and. eout(k).ge. sxx(i-1)) goto 20
     enddo
     stop ' eout(k) not found in sxx'
20   kk=i

!    linear interpolation, since sxx well resolved
     q=(eout(k)-sxx(i-1))/(sxx(i)-sxx(i-1))
     q1=1.d0-q
     sout(k)=q*syy(i)+q1*syy(i-1)
!     print*,k,eout(k),sout(k)
enddo

!check consistency
!JP (July 2013)
if (.not.low) then
  errmax=2.
else
  errmax=10.
endif

if (abs(1.-sout(1)/sn).gt.0.1) then
  if (message) print*,'WARNING!!!! Problem at begin of high energy tail',sout(1),sn
!JO Sept. 2016 commented out
!  if (abs(1.-sout(1)/sn).gt.errmax)  stop ' error at begin of high energy tail (swap)'
! might need to be commented out 
endif

sout(1)=sn
sout(nout)=s1  ! to obtain similar alpha0
               ! (sout(nout) might not be equal to s1 due to convolution)

else

do k=1,nout
  eout(k)=x1+(k-1)*rdst  
enddo

kk=1

do k=1,nout
     do i=kk,nn-1
       if(eout(k).gt.sxx(i) .and. eout(k).le. sxx(i+1)) goto 30
     enddo
     stop ' eout(k) not found in sxx'
30   kk=i

!    linear interpolation, since sxx well resolved
     q=(eout(k)-sxx(i+1))/(sxx(i)-sxx(i+1))
     q1=1.d0-q
     sout(k)=q*syy(i)+q1*syy(i+1)
enddo

!check consistency
if (abs(1.-sout(nout)/sn).gt.0.1) then
  if (message) print*,'WARNING!!!! Problem at begin of high energy tail',sout(nout),sn
!JO Sept. 2016 commented out
!  if (abs(1.-sout(nout)/sn).gt.2.)  stop ' error at begin of high energy tail (noswap)'
! might need to be commented out 
endif

sout(nout)=sn
sout(1)=s1  ! to obtain identical alpha0
            ! (sout(1) might not be equal to s1 due to convolution)

endif

!for tests
!open(1,file='cross_out1')
!do i=1,nn
!  if(sxx(i).lt.x1) cycle
!  if(sxx(i).gt.xn) exit
!  write(1,*),sxx(i),syy(i)
!enddo
!close(1)
deallocate(x,y,logx,logy,sxx,syy)

return
end

!
!------------------------------------------------------------------------
!
subroutine conv_spec (sx,sy,py,n)

! note1:
! in contrast to the idl routine (convol.pro), we wrap the response function
! around zero, to enable padding in a simple way.
! This is necessary here since we deal with functions which are completely
! non-periodical (s(1) and s(N) have completely different orders),
! in contrast to convolving rectified spectra which begin and end at unity.
!
! note2:
! this routine uses the "slow" way. By exploiting the fact that
! signal and response are real functions, one could save a factor of
! roughly two. Since the convolution(s) have to be done only once per
! frequ. grid, we use here the "standard" way for clarity. 
  
use nlte_type
use nlte_dim
use fund_const

IMPLICIT NONE
integer(i4b) :: n,i
real(dp), dimension(n) :: sx,sy,py
complex(dp), allocatable, dimension(:) :: sy_t,py_t

allocate(sy_t(n),py_t(n))

sy_t = sy
py_t = py

call fft(sy_t,1,n) ! forward
call fft(py_t,1,n) ! forward
sy_t=sy_t*py_t     ! convolution
call fft(sy_t,-1,n)! backwards
sy = dble(sy_t)    ! real part
sy = ((sx(n)-sx(1))/(n-1)) * sy  ! Physical domain (hn = Hn*Delta)

deallocate(sy_t,py_t)

return

end
!
!------------------------------------------------------------------------
!
subroutine fft(x,dir,length)
! fast-fourier transformation. length has to be a power of 2.
use nlte_type
implicit none

integer(i4b), intent(in) :: dir,length
complex(dp),dimension(0:length-1) :: x
real(dp), parameter :: pi=3.141592653589793_dp

complex(dp) :: wp,w,ws,temp
real(dp) :: theta, pi1
integer(i4b) :: nob, i, j, mmax, istep, m, bit_reverse

nob=nint(log(real(length))/log(2.0))
if (2**nob .ne. length) stop ' input data length must be a power of 2'

do i=1,length-2
  j=bit_reverse(i,nob)
  if(j.gt.i) call swap(x(i),x(j))
enddo

if (dir .eq. 1) then
  pi1=-pi
else if (dir .eq. -1) then
  pi1=pi
else
  stop ' wrong dir in fft'
endif  

mmax=1
do
  if(length .le. mmax) exit
  istep=2*mmax

  theta=pi1/mmax

  wp=cmplx(-2.0d0*sin(.5d0*theta)**2,sin(theta))
  w=cmplx(1.d0,0.d0,kind=dp)

  do m=1,mmax
    ws=w
    do i=m-1,length-1,istep
      j=i+mmax
      temp=ws*x(j)
!      if(dir.eq.-1) print*,mmax,i,j,ws,wp,temp,x(i),x(j)
      x(j)=x(i)-temp
      x(i)=x(i)+temp
!      if(dir.eq.-1) print*,mmax,i,j,x(i),x(j)
    enddo
    w=w+w*wp
  enddo
  mmax=istep
enddo

if (dir.eq.-1) x=x/length

end subroutine
!
!------------------------------------------------------------------------
!
function bit_reverse(i,size)
use nlte_type
implicit none
integer(i4b) :: i, size, bit_reverse
integer(i4b) :: length, temp

temp=i
do length =size,2,-1
  temp=ishftc(temp,-1,length)
enddo

bit_reverse=temp
end function
!
!------------------------------------------------------------------------
!
subroutine swap(a,b)
use nlte_type
implicit none
complex(dp) :: a,b,temp
temp=a
a=b
b=temp
end subroutine
!
!------------------------------------------------------------------------
!
SUBROUTINE ILOWIMAX_LTE(ILOW,IMAX,TE,TEFF,ND,NTEMP,HELONE, &
&   OPTCMF1,OPTOUT,LTE_UPDATE)
! 
! definition of ilow and imax for LTE calculations (first iteration cycle)
! and H/He always (in the old spirit, which worked) 
! 
! most of the input parameters refer to the older version, and might
! come into handy in future again
!  
USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: NAT, LABAT,NIONS, LE, LI, IFIRSL

USE nlte_var, ONLY: MAINION,IONIS,ENIONND,IMIA,IMAA,ICONVER,BLEVEL,ALEVEL

IMPLICIT NONE

!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS
INTEGER(I4B), PARAMETER :: NREC=ID_LLEVS  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND,NTEMP  
REAL(DP) :: TEFF
LOGICAL :: HELONE,OPTCMF1,OPTOUT,LTE_UPDATE
!     ..
!     .. array arguments ..
REAL(DP) ::  TE(ND1)  
INTEGER(I4B) ::  ILOW(ND1,KEL),IMAX(ND1,KEL)  

INTEGER(I4B) :: K,L,I,MAIN,IONI,IMI,IMA,I1,I2,DELTAI,NATO,INATO,IJ,NLAS,IGENIO

CHARACTER AT*6  

! all ionisation stages relative (1 is lowest ion in DETAIL)

DO K = 1,NAT  
     DO L = NTEMP,ND  
          MAIN = MAINION(L,K)  
          IONI = IONIS(K,L)  
          AT = LABAT(K)  

          IF (AT.EQ.'H') THEN
               I1=1
               I2=1

          ELSE IF (AT.EQ.'HE') THEN  
! exception for He, since option HELONE might be set to .false.
! (for very hot stars)

               IF (HELONE) THEN   !this is the standard
                    I1 = 1  
               ELSE  
                    I1 = 2  
               END IF  

               IF (MAIN.GE.2) THEN  
                    IF (MAIN.EQ.3) STOP 'CHECK IONIZATION INDEXING!'
                    I2 = 2  
                    IF (ENIONND(K,3,L)/ENIONND(K,2,L).LT. 1.D-12) I2 = 1
               ELSE IF (ENIONND(K,3,L)/ENIONND(K,1,L).GE.1.D-12) THEN
                    I2 = 2  
               ELSE  
                    IF (.NOT.HELONE) STOP ' NO HE I, BUT DOMINANT ION!!!'
                    I2 = 1  ! for very cool stars
               END IF
! JO Oct. 2021: to avoid different ilow, imax for cool stars, following hack
!               (was already present in v11)
               IF(TEFF.GT.15000.) THEN
                 IF(I2.NE.2) THEN
                   PRINT*,L,' IMAX(HE) SET TO 2 (ILOWIMAX_LTE)'
                   I2=2
                 ENDIF  
               ELSE
                 IF(.NOT.HELONE) STOP ' HELONE = F FOR Teff < 15 kK (ILOWIMAX_LTE)'
                 IF(I2.NE.1) THEN
! JO Oct. 2021, to allow for HeII under specific conditions
                   IF(I2.NE.2) STOP ' IMAX(HE) ne (1 or 2)'
                   PRINT*,L,' IMAX(HE) NE 1 (ILOWIMAX_LTE) for Teff le 15 kK'
                 ENDIF
               ENDIF  
             ELSE ! ALL OTHER ELEMENTS
               IF(MAIN.EQ.IONI) THEN
                 I1=MAIN-2
                 I2=MAIN
               ELSE  
                 I1=MAIN-1
                 I2=MAIN+1
               ENDIF  
          ENDIF
               
          ILOW(L,K) = MAX(I1,1)  
          IMAX(L,K) = MIN(I2,IONI)
          DELTAI=IMAX(L,K)-ILOW(L,K)
          IF (DELTAI.LT.0) STOP ' ERROR IN ILOW,IMAX'
          IF (LABAT(K).NE.'H' .AND. LABAT(K) .NE.'HE' .AND. DELTAI.LT.1) &
&  PRINT*,' WARNING!!! K=',K,' L=',L,' ILOW = ',ILOW(L,K),' IMAX = ',IMAX(L,K)
     ENDDO
END DO  
!
!------------ determination of imia, imaa
!
DO K = 1,NAT
     IMI = NIONS(K)  
     IMA = 0  

     DO L = 1,ND  
          IMI = MIN0(ILOW(L,K),IMI)  
          IMA = MAX0(IMAX(L,K),IMA)  
     END DO

     IF (IMA.EQ.0) STOP ' SOMETHING WRONG WITH IMA'  
     IMIA(K) = IMI  
     IMAA(K) = IMA  
!     IF (OPTOUT .AND. ICONVER.EQ.1)
     WRITE (*,FMT=9010) K,IMI,IMA  
     PRINT*
END DO  

IF(.NOT.LTE_UPDATE) RETURN

! from v10 on: check for new levels
! ONLY HERE, ILOW and IMAX can change regarding He.
! If they change to higher stages, 
! subroutine OPACITC has no info how to deal with this.
! 
! Example: in a certain iteration, ILOW=1, IMAX=1 at a certain L.
! after ILOWIMAX_LTE has been called (either due to restart, or because
! of temperature update, ILOW/IMAX might change to 2/2. Although the occupation
! numbers of the ground-level of i=2 is known, all other levels as well
! as the bf-transition remain unkown. THUS: 


DO L=1,ND

   READ (17,REC=L) (BLEVEL(I),I=1,NREC)  
   READ (15,REC=L) (ALEVEL(I),I=1,NREC)

   DO I=1,NREC
      NATO  = LE(I)  
      IF (LABAT(NATO).NE.'HE') CYCLE
      INATO = LI(I)  
      IJ = IGENIO(NATO,IMAX(L,NATO))+1
      NLAS = IFIRSL(IJ)  
! slightly modified from V6.1 on

      IF(BLEVEL(I).EQ.0. .AND. &
&      ((INATO.GE.ILOW(L,NATO).AND.INATO.LE.IMAX(L,NATO)).OR. I.EQ.NLAS)) THEN
! VERY SMALL OCCUP. NUMBER
         BLEVEL(I)=ALEVEL(I)*1.D-10
        PRINT*,'NEW LEVEL ',I,' AT L = ',L,': OCC MODIFIED TO OCC-LTE*1.D-10'
!           print*,i,nato,inato,ilow(l,nato),imax(l,nato),blevel(i)
      ENDIF
   ENDDO  
! so far, we assume that this happens only after T-update.
! Thereafter, restart is invoked, which reads from file 27 (instead of 17)
! If something goes wrong, try to uncomment the following line  
!   WRITE (17,REC=L) (BLEVEL(I),I=1,NREC)  
   WRITE (27,REC=L) (BLEVEL(I),I=1,NREC)  
END DO  

 9010 FORMAT (' ELEM. NO. ',I2,' WITH ILOW= ',I1,' AND IMAX= ',I1, ' (LTE!!!, ZEFF CORRECTED)')  

RETURN
END
!
!------------------------------------------------------------------------
!
SUBROUTINE ILOWIMAX_OLD(ILOW,IMAX,TE,TEFF,ND,NTEMP,HELONE,OPTCMF1,OPTOUT)
! 
! definition of ilow and imax as done in versions until 8.7.1, 
! kept for consistency to allow for a treatment of z=0 or comparison
! calculations 
!  
USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: NAT, LABAT, NIONS

USE nlte_var, ONLY: MAINION,IONIS,ENIONND,IMIA,IMAA,ICONVER

IMPLICIT NONE

!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND,NTEMP  
REAL(DP) :: TEFF
LOGICAL :: HELONE,OPTCMF1,OPTOUT
!     ..
!     .. array arguments ..
REAL(DP) ::  TE(ND1)  
INTEGER(I4B) ::  ILOW(ND1,KEL),IMAX(ND1,KEL)  

INTEGER(I4B) :: K,L,MAIN,IONI,I1,I2,IMI,IMA,IMI1,IMA1

REAL(DP) :: EN,REL

CHARACTER AT*6  

! all ionisation stages relative (1 is lowest ion in DETAIL)
! occmain maximum fraction of all ions EXCLUDING the highest one without
! transitions

DO K = 1,NAT  

     DO L = NTEMP,ND  
          MAIN = MAINION(L,K)  
          IONI = IONIS(K,L)  
          AT = LABAT(K)  
          IF (AT.EQ.'H') THEN  
               I1 = 1  
               I2 = 1  

          ELSE IF (AT.EQ.'HE') THEN  
               IF (HELONE) THEN  
                    I1 = 1  
               ELSE  
                    I1 = 2  
               END IF  

               IF (MAIN.GE.2) THEN  
               IF (MAIN.EQ.3) STOP 'CHECK IONIZATION INDEXING!'
                    I2 = 2  
                    IF (ENIONND(K,3,L)/ENIONND(K,2,L).LT. 1.D-12) I2 = 1
               ELSE IF (ENIONND(K,3,L)/ENIONND(K,1,L).GE.1.D-12) THEN
                    I2 = 2  
               ELSE  
                    IF (.NOT.HELONE) STOP ' NO HE I, BUT DOMINANT ION!!!'
                    I2 = 1  
               END IF
!
!          ELSE IF (AT.EQ.'SI' .or. at.eq.'N') THEN
! if old N-atom (only NII to NIV), use this block
          ELSE IF (AT.EQ.'SI') THEN
! NOTE THAT PRESENTLY ONLY SiII to SiIV are available from detail
! thus, ilow,imax = 1 correspond to SiII, ilow,imax=3 to SiIV and so on
               IF(AT.EQ.'SI'.AND.IONI.NE.3) THEN
                PRINT*,' EITHER: si atomic model has changed, update the following blocks'
                PRINT*,'     OR: IONIS (Si) ne 3'
                STOP ' PROBLEMS WITH HIGHEST ION FROM SILICON'
               ENDIF 
               IF (MAIN.EQ.1.OR.TE(L).LE.15000.) THEN  
                    I1 = 1  
                    I2 = 2  
               ELSE IF (MAIN.EQ.IONI) THEN  
                    EN = ENIONND(K,MAIN,L)  
                    REL = ENIONND(K,MAIN-2,L)/EN  
                    IF (REL.GT.1.D-10) THEN  
                         I1 = MAIN - 2  
                         I2 = MAIN  
                    ELSE  
                         I1 = MAIN - 1  
                         I2 = MAIN  
                    END IF  
               ELSE  
                    EN = ENIONND(K,MAIN,L)  
                    REL = ENIONND(K,MAIN+2,L)/EN  
                    IF (REL.GT.1.D-7) THEN  
                      I1 = MAIN - 1  
                      I2 = MAIN + 1
                    ELSE
                      I1 = MAIN - 1  
                      I2 = MAIN 
                    ENDIF
               END IF  
!
!--- C, O, Mg for present Detail models (only TWO ions!)
!
          ELSE IF (AT.EQ.'C' .OR.  &
&                     AT.EQ.'O' .OR. AT .EQ.'MG') THEN
             IF (MAIN.EQ.3) STOP 'CHECK IONIZATION INDEXING!'
               I1 = 1
               I2 = 2
               IF (ENIONND(K,1,L)/ENIONND(K,2,L).LT.1.D-12) THEN
                 I1 = 2
                 I2 = 2
               ENDIF
               IF (ENIONND(K,2,L)/ENIONND(K,1,L).LT.1.D-12) THEN
                 I1 = 1
                 I2 = 1
               ENDIF                 
!

          ELSE IF (AT.EQ.'N') THEN 
! for new model atom from NII to NV
               IF(MAIN.EQ.IONI) THEN
                 I1=MAIN-2
                 I2=MAIN
               ELSE  
                 I1=MAIN-1
                 I2=MAIN+1
               ENDIF  

          ELSE  
               STOP ' ATOM NOT IMPLEMENTED YET'  
          END IF  

          ILOW(L,K) = MAX(I1,1)  
          IMAX(L,K) = MIN(I2,IONI)  
          END DO

END DO  
!
!------------ determination of imia, imaa
!
DO K = 1,NAT  

     AT = LABAT(K)  

!------------ want to have at least SiII/III or SiIII/IV at
!             ALL depth points (minimum set) if CMF treatment aimed at
     
     IF ((AT.EQ.'SI'.or.AT.EQ.'N').AND.OPTCMF1) THEN
      IMI1=MAXVAL(ILOW(:,K))
      IMA1=MINVAL(IMAX(:,K))
      IF ((IMA1-IMI1).LT.0) STOP ' Si: SOMETHING WRONG WITH IMI1,IMA1 IN OCCLTE'
      IF((IMA1-IMI1).EQ.0) THEN  ! ONLY ONE ION PRESENT AT ALL DEPTH PONTS
        IF (IMA1.EQ.3) THEN
          DO L = 1,ND
            IF(IMAX(L,K).NE.3) STOP ' Si: INCONSISTENCY IN IMA1'
            ILOW(L,K)=MIN0(ILOW(L,K),2)
          ENDDO  
        ELSE IF (IMA1.EQ.1) THEN
          DO L = 1,ND
            IF(ILOW(L,K).NE.1) STOP ' Si: INCONSISTENCY IN IMI1'
            IMAX(L,K)=MAX0(IMAX(L,K),2)
          ENDDO  
        ELSE
          IF(IMA1.NE.2.OR.IMI1.NE.2) STOP ' Si: INCONSISITENCY IN IMA1,IMI1'
! decide wether II/III or III/IV
          IF(TEFF.LE.20000.) THEN !might be changed
            DO L = 1,ND
              ILOW(L,K)=1
            ENDDO             
          ELSE
            DO L = 1,ND
              IMAX(L,K)=3
            ENDDO             
          ENDIF  
        ENDIF

      ENDIF
! check once more
      IMI1=MAXVAL(ILOW(:,K))
      IMA1=MINVAL(IMAX(:,K))
      IF ((IMA1-IMI1).LE.0) STOP ' Si: unsuccessful def. of ilow, imax'
      PRINT*
      PRINT*,' MINIMUM IONIC SET FOR SILICON:',IMI1,' TO ',IMA1 
      PRINT*
     ENDIF 
!------------ Si treatment

     IMI = NIONS(K)  
     IMA = 0  

     DO L = 1,ND  
          IMI = MIN0(ILOW(L,K),IMI)  
          IMA = MAX0(IMAX(L,K),IMA)  
     END DO

     IF (IMA.EQ.0) STOP ' SOMETHING WRONG WITH IMA'  
     IMIA(K) = IMI  
     IMAA(K) = IMA  
     IF (OPTOUT .AND. ICONVER.EQ.1) WRITE (*,FMT=9010) K,IMI,IMA  
     PRINT*
END DO  

 9010 FORMAT (' ELEM. NO. ',I2,' WITH ILOW= ',I1,' AND IMAX= ',I1, ' (LTE!!!, ZEFF CORRECTED)')  

RETURN
END
!
!------------------------------------------------------------------------
!
SUBROUTINE RBF_CHECK
!
! checks for unique rbf transition frequencies and modifies them in case
USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: NF=>ID_RBFTR, FRECIN

implicit none

integer(i4b) :: i,j,ilog
real(dp) :: f1,f2

! first run: change
do i=1,nf-1
  f1=frecin(i)
  do j=i+1,nf
    f2=frecin(j)
    if(f2.eq.f1) then
! change 7th digit (such a frequency is in the MHz domain and not present here) 
    ilog=log10(f2)-7
    print*,' RBF transition frequencies changed'
    print*,' old frequency = ',f2
    f2=f2+10.**ilog
    frecin(j)=f2
    print*,' new frequency = ',f2
    print*
    endif
  enddo
enddo

! 2nd run: check! due to bad luck, new conincidence might be created
do i=1,nf-1
  f1=frecin(i)
  do j=i+1,nf
    f2=frecin(j)
    if(f2.eq.f1) then
      print*,' still identical frequencies',f1
      stop ' INDENTICAL RBF FREQUENCIES' 
    endif
  enddo
enddo

return
end
!
!------------------------------------------------------------------------
!
subroutine OP_FIT(FGS,PICS,IMAX,NDATOS)

!interpolation for original OPACITY data where sigma = 0.
!additionally, frequency grid is adapted

use nlte_type

IMPLICIT NONE
!	.. scalar arguments ..
INTEGER(I4B) :: NDATOS,IMAX

!	.. array arguments ..
REAL(DP), DIMENSION(IMAX) :: FGS,PICS,PICS2
!	..
!	.. local scalars

INTEGER(I4B) :: L,IA,IB,IA1,IB1,K
REAL(DP) :: X1,X2,Y1,Y2,DSDE,DF

!	....
!	.. automatic array ..
REAL(DP),DIMENSION(NDATOS) :: YOP

YOP(1:NDATOS)=PICS(1:NDATOS)

IF (YOP(1).EQ.0. .OR. YOP(NDATOS).EQ.0.) STOP ' sig = 0 at boundaries' 

BIGLOOP: DO

DO K=1,NDATOS-1 
  IF (YOP(K).EQ.0.) THEN
    IA=K
    GOTO 100
  ENDIF  
ENDDO

IF (K.NE.NDATOS) STOP ' ERROR1 IN OP-FIT'
EXIT BIGLOOP

100 DO K=IA,NDATOS 
  IF (YOP(K).NE.0.) THEN
    IB=K-1
    GOTO 110 
  ENDIF  
ENDDO

STOP ' OP_FIT: SHOULD NOT STOP HERE' 

!LOG-LOG INTERPOLATION
110 IA1=IA-1
    IB1=IB+1

X1=FGS(IA1)
X2=FGS(IB1)

Y1=YOP(IA1)
Y2=YOP(IB1)

DSDE=LOG10(Y2/Y1)/LOG10(X2/X1)
DF=(X1-X2)/DBLE(IA1-IB1)

!PRINT*,X1,X2,Y1,Y2
DO K=IA,IB 
  FGS(K)=FGS(IA1)+(K-IA+1)*DF ! works for both increasing and decreasing grid
  X2=LOG10(Y1)+DSDE*LOG10(FGS(K)/X1)
  YOP(K)=10.D0**X2
!  PRINT*,K,FGS(K),YOP(K)
ENDDO
!PRINT*

ENDDO BIGLOOP

PICS(1:NDATOS)=YOP(1:NDATOS)

RETURN
END
!
!------------------------------------------------------------------------
!
SUBROUTINE MODVL_UPDATE(ND,R,VELO,DVDR,RHO,XNE,XNELTE,XNH,TEMP,TAU,TAUR,CLF, &
&   TEFF,GGRAV,RSTARNOM,XMLOSS,BETA,NDIV,YHEIN,YHE,HEI,RTAU23,UPDATE_DONE)

! BASIC ASSUMPTION: NO CLUMPING IN PHOTOSPHERE (I.E., BELOW TRANSITION POINT)
!
! UPDATE OF PHOTOSPHERIC STRATIFICATION WITH GIVEN TEMP, CHIBAR_H AND NE 
! CLUMPING FACTOR ASSUMED TO BE UNITY IN PHOTOSPHERIC REGION
!
  
!NOTE: CHIBAR_H calculated from NLTE (with XNE), TAUR calculated from
!      LTE (with XNELTE). Can lead to inconsistencies for cooler stars,
!      when H recombines

USE nlte_type
USE nlte_dim
USE fund_const, ONLY: PI, XMH=>AMH, AKB
USE nlte_opt, ONLY: OPT_PHOTCLUMP

USE princesa_var, ONLY: NAT, LABAT, ABUND, WEIGHT

USE nlte_var, ONLY: MODNAM, VMAX, SR, SRVMIN, SRNOM, IDUM, &
&             GG=>GGRAV, TE=>TEFF, A2TE, DELMU

USE photstruc, ONLY: NDMOD, XT, CONST, EXPO, SIGEMIN, &
&                    XMEXT, TEXT, XM, CHIBAR_H, DELCHI, &
&                    POLD=>PKUR, TAUROLD=>XNEKUR, SIG=>GRADKUR

USE nlte_porvor, ONLY : OPTTHICK

IMPLICIT NONE
!
!     .. parameters ..
REAL(DP), PARAMETER :: RSU=6.96D10, SIGMAE=6.65D-25  

INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT
!     ..
!     .. scalar arguments ..
REAL(DP)                 ::  TEFF,GGRAV,XMLOSS,BETA,YHE  
REAL(DP), intent(in)     ::  RSTARNOM,YHEIN,HEI  
REAL(DP), intent(inout)  ::  RTAU23
INTEGER(I4B), intent(in) ::  ND,NDIV  
!     ..
!     .. array arguments ..
REAL(DP), intent(in)    :: TAUR(ND1)

REAL(DP), intent(inout) :: R(ND1),VELO(ND1),DVDR(ND1),RHO(ND1), &
&                          TEMP(ND1),CLF(ND1),XNE(ND1),XNELTE(ND1), &
&                          XNH(ND1),TAU(ND1)
!     ..
!     .. local scalars ..
REAL(DP) :: VMIN,B,H2,XM0,ALPHA,DM, &
&           R0,YHE1,SUMATOM,SUMMASS,XMU,XMU1,VSONIC,A2, &
&           XMMAX,DELTAM,DELTAM1,X1,X2,P0,T0,RHO0,TAUR0,PL,TL,RL,TAURL,HS, &
&           DMDR,SRNEW,RMAX,DPX,SIGEM,TAUMIN,SUMM,DEL,DCHIDM, DELCHIBAR, CHIBAR, &
&           AVERAGE,XTIN,SAFETY, &
&           RR1, RR2, DET, V1TEST, V2TEST, RR0, VSMOOTH, RHOSMOOTH


INTEGER(I4B) ::  L,NS,NENE,ISI,K,NOK,NBAD,LL,I,NSIG  
!     ..
!     .. local arrays ..
REAL(DP) ::  RADACC(ND1),P(ND1),TNEW(ND1),Y(4), &
&            RNEW(ND1),RHONEW(ND1),V(ND1),GRADV(ND1), &
&            RHOCL(ND1),RHONEWCL(ND1),XNECL(ND1), &
&            ARHO(ND1),BRHO(ND1),CRHO(ND1),TAURNEW(ND1), &
&            RIN(ND1),VELOIN(ND1),DVDRIN(ND1),RHOIN(ND1), &
&            XNELTEIN(ND1),XNHIN(ND1),CLFIN(ND1), &
&            ZZ1(4),ZZ2(4),VMAT(4,4),YY(4),GRADVNEW(ND1)

INTEGER(I4B) :: INDEX(ND1), INDLUD(4)

LOGICAL :: UPDATED
LOGICAL, INTENT(OUT) :: UPDATE_DONE

CHARACTER*7 :: CHAR

EXTERNAL DERIVS_UPD, RKQC
!     ..

! CHECK WHETHER UPDATE NEEDS TO BE PERFORMED
! ARHO: START VALUE OF RADACC
OPEN(61,FILE=TRIM(MODNAM)//'/GRAD.OUT',STATUS='UNKNOWN')
DO L=1,ND
   READ(61,*) CHAR,I,ARHO(L)
ENDDO
CLOSE(61)

! CRHO: ACTUAL VALUE OF RADACC
REWIND(60)
DO L=1,ND
READ(60,*) I,BRHO(L),CRHO(L) 
ENDDO

! TEST CONSISTENCY
DO L=1,ND
  IF (ABS(1.-BRHO(L)/CHIBAR_H(L)).GT.1.D-13) THEN
    PRINT*,BRHO(L),CHIBAR_H(L)
    STOP ' INCONSISTENCY IN CHIBAR_H'
  ENDIF
ENDDO  

DO L=1,ND
  IF(ARHO(L).NE.0.D0) EXIT
ENDDO
I=L

IF(I.GE.ND-5) STOP ' SOMETHING WRONG WITH RADACC_START NE 0'

AVERAGE=0.
DO L=I,I+4
  AVERAGE=AVERAGE+CRHO(L)/ARHO(L)
ENDDO  
AVERAGE=AVERAGE/5.
PRINT*, ' AVERAGE DEVIATION IN RADACC AT OUTER PHOTOSPHERE =',AVERAGE

IF (AVERAGE.LT.1.) THEN
! NO UPDATE PERFORMED

PRINT*
PRINT*,' NO UPDATE PERFORMED'
PRINT*

REWIND 20  
READ (20) TEFF,GGRAV,SR,YHE,XMU,VMAX,XMLOSS,BETA, &
&      (RIN(I),I=1,ND), (VELOIN(I),I=1,ND), (DVDRIN(I),I=1,ND), &
&      (RHOIN(I),I=1,ND), &
&      (XNELTEIN(I),I=1,ND), (XNHIN(I),I=1,ND), (CLFIN(I),I=1,ND), &
&      (INDEX(I),I=1,ND), (P(I),I=1,ND), NS, UPDATED, XTIN 

! CHECK CONSISTENCY
DO I=1,ND
  IF(R(I).NE.RIN(I)) STOP ' MODVL_UPDATE: R NE RIN'
  IF(VELO(I).NE.VELOIN(I)) STOP ' MODVL_UPDATE: VELO NE VELOIN'
  IF(DVDR(I).NE.DVDRIN(I)) STOP ' MODVL_UPDATE: DVDR NE DVDRIN'
  IF(RHO(I).NE.RHOIN(I)) STOP ' MODVL_UPDATE: RHO NE RHOIN'
  IF(XNH(I).NE.XNHIN(I)) STOP ' MODVL_UPDATE: XNH NE XNHIN'
  IF(CLF(I).NE.CLFIN(I)) STOP ' MODVL_UPDATE: CLF NE CLFIN'
ENDDO

IF(XT.NE.XTIN) STOP ' MODVL_UPDATE: XT NE XTIN'


!NOTE: XNELTEIN NE XNELTE!!! 
!      XNELTEIN -> REWRITTEN TO MODEL
!      XNE, XNELTE (ARGUMENTS) REMAIN UNCHANGED

UPDATED=.TRUE.
UPDATE_DONE=.FALSE.

REWIND 20
WRITE (20) TEFF,GGRAV,SR,YHE,XMU,VMAX,XMLOSS,BETA, &
&      (R(I),I=1,ND), (VELO(I),I=1,ND), (DVDR(I),I=1,ND), (RHO(I),I=1,ND), &
&      (XNELTEIN(I),I=1,ND), (XNH(I),I=1,ND), (CLF(I),I=1,ND), &
&      (INDEX(I),I=1,ND), (P(I),I=1,ND), NS, UPDATED, XT 
RETURN
ENDIF


PRINT* 
PRINT*,' =========== UPDATE OF PHOTOSPHERIC STRUCTURE ============'
PRINT*

! check that no clumping except for tests

IF (.NOT. OPT_PHOTCLUMP) THEN
  DO L=NDIV,ND
    IF (CLF(L).NE.1.D0) STOP ' PHOTOSPHERIC CLUMPING FOUND (ROUTINE MODVL_UPDATE)' 
  ENDDO
ENDIF

! transfer to module nlte_var (needed in DERIVS)
SRNOM=RSTARNOM*RSU

RHOCL=RHO*CLF ! OLD QUANTITY

NDMOD=ND ! for module photstruc

RNEW=R*SR ! in cgs, will contain updated radius
V=VELO*VMAX ! in cgs, will contain updated velocity
RHONEW=RHO  ! will contain updated density
TNEW=TEMP   ! will contain updated temperature
TAURNEW=TAUR ! will contain updated taur (guess-value)
GRADV=DVDR*VMAX/SR

ALLOCATE(XMEXT(NDMOD),TEXT(NDMOD),POLD(NDMOD),TAUROLD(NDMOD),SIG(NDMOD))
XMEXT=0.  ! standard grid (needed for dT/dm)
TEXT=TEMP 

!  integration assuming rho(r)=r^alpha*b 
!  => int = 1/(alpha+1)*(rho(r2)*r2-rho(r1)*r1)
!  with
!  alpha=delta log rho/delta log r and
!  (previous version)
!  b=rho(r)/r^alpha  => rho(x) = rho(r)*(x/r)^alpha with x on staggered grid
DO L=2,ND
   ALPHA=LOG10(RHO(L-1)/RHO(L))/LOG10(R(L-1)/R(L))
   XMEXT(L)=XMEXT(L-1)+(R(L-1)*RHO(L-1)-R(L)*RHO(L))/(ALPHA+1.) ! on normal grid
ENDDO
XMEXT = XMEXT*SR

! recalculate XM0 = M(NDIV) and check consistency
VMIN=V(NDIV)

B = 1.D0 - (VMIN/VMAX)** (1.D0/BETA)  

IF (BETA.EQ.1.D0) THEN  
  H2 = -LOG(1.D0-B)  
ELSE  
  H2 = (1.D0- (VMIN/VMAX)** ((1.D0-BETA)/BETA))/ (1.D0-BETA)
END IF  

R0=RNEW(NDIV)

XM0 = XMLOSS/ (4.D0*PI*R0*VMAX)/B*H2  

DM=XM0/XMEXT(NDIV)

!IF(ABS(LOG10(DM)).GT.0.05) STOP ' PROBLEMS IN COLUMN DENSITY (ROUTINE MODVL_UPDATE)'
!JO Oct. 2017
IF(ABS(LOG10(DM)).GT.0.20) then
  DO L=1,NDIV
    PRINT*,RHO(L),R(L),XMEXT(L)
  ENDDO
  PRINT*
  PRINT*,XM0,XMEXT(NDIV),DM,LOG10(DM)  
  STOP ' PROBLEMS IN COLUMN DENSITY (ROUTINE MODVL_UPDATE)'
endif
  
XMEXT=XMEXT*DM

!JO Oct. 2017
IF(ABS(1.-XMEXT(NDIV)/XM0).GT.1.D-15) STOP ' XMEXT(NDIV) NE XM0 (ROUTINE MODVL_UPDATE)'
XMEXT(NDIV)=XM0 !This is important for index-finding (IT) in routine DERIVE_UPD

PRINT*,'CHIBAR_H'
DO L=2,ND1 !XMEXT(1)=0.
  PRINT*,L,' ',LOG10(XMEXT(L)),' ',CHIBAR_H(L)
ENDDO

NS=NDIV

PRINT *  
PRINT *,' DIVISION AT ',VMIN*1.D-5,' KM/S (L = ',NS,')'  
PRINT *,' CORRESPONDING TO ',R0/SRNOM,' NOM. RADII'  
PRINT *,' AND LG(M0) = ',LOG10(XM0)  
PRINT *  

! to be consistent to routine MODVL
NENE = 0  

DO ISI = 1,NAT  
     IF (LABAT(ISI).EQ.'HE') NENE = ISI  
END DO  

IF (NENE.EQ.0) THEN  
     PRINT *,'HE NOT FOUND - MODVL_UPDATE'  
     IF (HEI.NE.0.D0) THEN  
          PRINT *,' NO HELIUM BUT ELSCAT-CONTRIBUTION!!!'  
          PRINT *, ' EITHER SET HEI TO ZERO OR CALCULATE WITH HELIUM!!!'
! comment the following line out for specific tests
          STOP ' INCONSISTENT TREATMENT OF HELIUM'  
     END IF  
     YHE1 = YHEIN  
ELSE  
     YHE1 = ABUND(NENE)  
     IF (ABS(1.D0-YHE1/YHEIN).GT.1.D-5) STOP ' ERROR IN YHE - MODVL_UPDATE'
END IF  

IF(YHE1.NE.YHE) STOP ' ERROR IN YHE - MODVL_UPDATE'

! transfer to module nlte_var (needed in DERIVS)
TE = TEFF  
GG = GGRAV

SUMATOM = 0.D0  
SUMMASS = 0.D0  

DO K = 1,NAT  
     SUMATOM = SUMATOM + ABUND(K)  
     SUMMASS = SUMMASS + WEIGHT(K)*ABUND(K)  
END DO  

IF (NENE.EQ.0 .AND. YHE.NE.0.) THEN  
     SUMATOM = SUMATOM + YHE  
     SUMMASS = SUMMASS + 4.D0*YHE  
END IF  

SUMMASS = SUMMASS*XMH

XMU = (1.D0+4.D0*YHE)/ (2.D0+YHE* (1.D0+HEI))  

DO L = 1,ND  
! clumping cancels out
   XMU1 = SUMMASS/ (SUMATOM+XNE(L)/XNH(L))/XMH  
!
!---  delmu defined in order to correct a2te (vs(iso)**2/t)
!---  for ionisation effects
!
    DELMU(L) = XMU/XMU1  
!   PRINT*,L,DELMU(L)
END DO

XM=XMEXT ! XM new grid, as calculated here

VSONIC = SQRT(AKB*TE/XMH/XMU)
A2 = VSONIC**2
A2TE = A2/TE

T0 = TNEW(NS)
RHO0 = RHONEW(NS)                                                            
P0 = A2TE*DELMU(NS)*RHO0*T0                                               
TAUR0 = TAURNEW(NS)

! P0 AND POLD (FROM 'MODEL') DIFFERENT, DUE TO DIFFERENT TEMPERATURE
! THUS, RECALCULATE P0LD 


DO L=NS,ND
   POLD(L) = A2TE*DELMU(L)*RHO(L)*TEMP(L)
!   SIG(L) = XNE(L)*SIGMAE/RHOCL(L)
! CHANGED FROM v10.1.3, SINCE USED FOR TAUR(=LTE)
   SIG(L) = XNELTE(L)*SIGMAE/RHOCL(L)
ENDDO
TAUROLD=TAUR
                                                                          
IF (.NOT.OPTTHICK) THEN
! chibar_h calculated with xne(NLTE)
  SIGEMIN=MINVAL(CHIBAR_H)  
!IF (SIGEMIN.LT.MINVAL(SIG)) &
! CHANGED FROM v10.1.3, SINCE NOW SIG REFERS TO LTE, WHILST SIGEMIN TO NLTE
  IF (SIGEMIN.LT.MINVAL(SIG*XNE/XNELTE)) &
& STOP ' MIN(CHIBAR_H) < SIG(THOMSON). MAYBE ERROR IN WIND-OPACITIES'

ELSE
! check only in unclumped part. But note that later on SIG(L) will be used.
! let's hope that there are no effects from wind conditions that influence
! the region below NS
!JO  at some point, this needs to be improved
  SIGEMIN=MINVAL(CHIBAR_H(NS:ND))  
! see above
  IF (SIGEMIN.LT.MINVAL(SIG(NS:ND)*XNE(NS:ND)/XNELTE(NS:ND))) &
& STOP ' MIN(CHIBAR_H) < SIG(THOMSON) BELOW NS'
ENDIF

!JO Jan 2016: SIGEMIN redefined. Otherwise log10(CHI-SIGEMIN) (in CHIH_PARAMS)
!becomes infinite at that point where CHI = SIGEMIN
IF(OPTTHICK) SIGEMIN=MINVAL(SIG(NS:ND)*XNE(NS:ND)/XNELTE(NS:ND))

DO L=NS,ND
  IF (ABS(CHIBAR_H(L)-SIGEMIN).GT.0.05) EXIT
ENDDO
NSIG=MAX(NS,L-1)
! IN CASES WHERE ONLY THOMSON SCATTERING OR CONSTANT GRAD
IF(ND-NSIG .LE. 4) THEN
  DO L=NS,ND
    IF (ABS(CHIBAR_H(L)-SIGEMIN).GT.0.01) EXIT
  ENDDO
  NSIG=MAX(NS,L-1)
  IF(ND-NSIG .LE. 4) STOP ' TOO FEW POINTS FOR REGRESSION OF CHIBAR_H'
ENDIF

! PARAMETERIZE CHIBAR_H IN THE RANGE NSIG...ND, &
! AND EVALUATE CHIBAR_H(FIT)/CHIBAR_H(ACTUAL) IN THE RANGE NS...NSIG
CALL CHIH_PARAMS(CHIBAR_H,POLD,TEMP,DELCHI,SIGEMIN,CONST,EXPO,ND,NS,NSIG)

XMMAX=.995*XM(ND) ! SAFETY FACTOR, TO REMAIN INSIDE GRID

DELTAM = LOG10(XMMAX/XM0)/ (ND-NS-5)                                      

IF (ABS(1.D0-XM(NS)/XM0).GT.1.D-14) STOP ' ERROR IN XM0 - MODVL_UPDATE'
XM(NS)=LOG10(XM0)
                                                                          
RADACC=0.                                                                 
                                                                          
DO L = NS + 1,ND                                                          
     DELTAM1 = DELTAM                                                     
     IF (L.LE.NS+8) DELTAM1 = DELTAM/2.D0                                 
     IF (L.LE.NS+4) DELTAM1 = DELTAM/3.D0                                 
     IF (L.LE.NS+2) DELTAM1 = DELTAM/6.D0                                 
     XM(L) = XM(L-1) + DELTAM1                                            
END DO                                                                    
                                                                          
DO L = NS,ND                                                              
     XM(L) = 10.D0**XM(L)                                                 
END DO                                                                    
!JO Jan/April 2016: last point close to previous one, to improve diffusion approx.
IF(ND.GE.61) XM(ND)=XM(ND-1)*1.1
                                                                          
X1 = XM0                                                                  
Y(1) = P0                                                                 
Y(2) = T0                                                                 
Y(3) = R0                                                                 
Y(4) = TAUR0
P(NS) = P0
! TNEW(NS),RHONEW(NS) DO NOT CHANGE 
                                                                          
! LOOP OVER ALL XM
! FROM V10.1 ON: WHILE LOOP, TO ALLOW FOR RESET OF L
!-------------------------------------------------------------------------
L = NS +1

DO WHILE (L < ND+1)

PL=Y(1)                                                                   
TL=Y(2)                                                                   
RL=Y(3)                                                                   
TAURL=Y(4)
X2 = XM(L)                                                                
HS = (X2-X1)/20.D0                                                        
IDUM = L                                                                  

! new photosheric structure, &
! using grad(m) via CHIBAR_H and approximate update of temperature
CALL ODEINT(Y,4,X1,X2,1.D-5,HS,0.D0,NOK,NBAD,DERIVS_UPD,RKQC)              

P(L) = Y(1)  
TNEW(L) = Y(2)  
RNEW(L) = Y(3)  
TAURNEW(L) = Y(4)

! one point just below SRNOM 
IF(RNEW(L).LT.SRNOM.AND.RNEW(L-1).GT.SRNOM) THEN
   IF(L.LE.NS+4) THEN
      PRINT*
      PRINT*,' WARNING! PHOTOSPHERE AT ',L,' VERY CLOSE TO DIVISION POINT AT',NS
      PRINT*
   ENDIF
! new try
   SAFETY=1.1D0
95 DMDR=(XM(L)-XM(L-1))/(RNEW(L-1)-RNEW(L))
   DELTAM=SAFETY*DMDR*(RNEW(L-1)-SRNOM) !including safety factor
   IF (DELTAM.LT.0.) STOP ' ERROR IN DELTAM < 0'

! FROM V10.1 ON: AVOID TOO SMALL SEPARATIONS (except L - 1 = NS, i.e., 
!           last point = starting point)
   IF (DELTAM/XM(L-1) .LT. 0.1 .AND. L-1 .NE. NS) THEN ! FACTOR 1.1
! SKIP LAST POINT
     XM(L-1)=XM(L-1)+DELTAM
     L = L - 1
     X1 = XM(L-1)
     Y(1) = P(L-1) 
     Y(2) = TNEW(L-1)  
     Y(3) = RNEW(L-1)  
     Y(4) = TAURNEW(L-1)
     IDUM = L
   ELSE
! ACCEPT LAST POINT, RECALCULATE NEW POINT
     XM(L)=XM(L-1)+DELTAM
     Y(1)=PL
     Y(2)=TL
     Y(3)=RL  
     Y(4)=TAURL
   ENDIF  
     
   X2=XM(L)
   HS = (X2-X1)/20.D0  
   CALL ODEINT(Y,4,X1,X2,1.D-5,HS,0.D0,NOK,NBAD,DERIVS_UPD,RKQC)

   P(L) = Y(1)  
   TNEW(L) = Y(2)  
   RNEW(L) = Y(3)  
   TAURNEW(L) = Y(4)
   
! stop statement removed. shall occur only very seldom, &
! and should be of no harm
   IF(R(L)/SRNOM.GE.1.0D0) THEN
! NEW PHILOSOPHY FROM V10.1.3 ON
     IF(SAFETY.EQ.1.1D0) THEN
       PRINT*,'WARNING!!!  PROBLEMS WITH POINT CLOSE TO PHOTOSPHERE (MODVL_UPDATE)!'
       SAFETY=1.2
       GOTO 95
     ENDIF 
  
     PRINT*,XM(L-2),XM(L-1),XM(L)
     PRINT*,R(L-2)/SRNOM,R(L-1)/SRNOM,R(L)/SRNOM
     PRINT*,R(L)/SRNOM
     STOP ' PROBLEMS WITH POINT CLOSE TO PHOTOSPHERE CANNOT BE CURED (MODVL_UPDATE)!'
   ENDIF

! define new xm grid (equidistant)
   DELTAM = LOG10(XMMAX/X2)/ (ND-L)  
   DELTAM=10.D0**DELTAM
   DO LL = L + 1,ND  
      XM(LL) = XM(LL-1)*DELTAM   
   END DO
!           
ENDIF  

X1 = X2  
! re-calculate R^2*GRAD

DO I=1,NDMOD-1  
  IF (X1 .GE. XMEXT(I).AND. X1 .LT. XMEXT(I+1)) EXIT
ENDDO
IF (I .EQ. NDMOD) STOP ' x1 not found in XM - MODVL_UPDATE'

DCHIDM=(DELCHI(I+1)-DELCHI(I))/(XMEXT(I+1)-XMEXT(I))
DELCHIBAR=DELCHI(I)+DCHIDM*(X1-XMEXT(I))
CHIBAR=DELCHIBAR*(CONST*P(L)*TNEW(L)**EXPO + SIGEMIN)
RADACC(L) = 5.67D-5/2.9979D10*TEFF**4*CHIBAR  
IF (RADACC(L).LE.0.) STOP ' RADACC = 0 IN MODVL_UPDATE'

L = L + 1

END DO
!-------------------------------------------------------------------------

DO L = NS + 1,ND  
   IF (P(L).LT.0.D0) STOP 'PRESSURE NEGATIVE!'  
   RHONEW(L) = P(L)/ (A2TE*DELMU(L)*TNEW(L))  
!   PRINT*,L,XM(L),XMEXT(L)
!   PRINT*,RNEW(L)/SRNOM,R(L)*SR/SRNOM
!   PRINT*,RHONEW(L),RHO(L)
!   PRINT*,TNEW(L), TEMP(L)
!   PRINT*,TAURNEW(L), TAUR(L)
!   PRINT*,RADACC(L),5.67D-5/2.9979D10*TEFF**4*CHIBAR_H(L)
!   WRITE(*,FMT='(I3,4(2X,E10.4))') L,XM(L),RADACC(L),XMEXT(L),5.67D-5/2.9979D10*TEFF**4*CHIBAR_H(L)
!   PRINT*
END DO

SRNEW = RNEW(ND)  

IF(ABS(1.-SR/SRNEW).GT.0.3) STOP ' TOO LARGE CHANGE IN SR -- MODVL_UPDATE'

SR=SRNEW

RMAX = RNEW(1)/SR  
RTAU23 = SRNOM/SR  
SRVMIN = R0  
IF (1.D0-1.D0/RTAU23.GT..1D0) PRINT *,' WARNING!!! EXTENDED PHOTOSPHERE'

PRINT *,' LOWER (PHOT.) RADIUS AT ',1.D0/RTAU23,' NOM. RADII'  
PRINT *,' MAX. RADIUS (RMAX) = ',RMAX,' (IN SR)'  

DO L = NS + 1,ND  
   V(L) = XMLOSS/ (4.D0*PI*RNEW(L)**2*RHONEW(L))  
END DO
!
!    modified to ensure smooth transition
!
! new version
DO L = NS,ND - 1  
   DM = LOG(V(L)/V(L-1))/ (RNEW(L)-RNEW(L-1))  
   DPX =LOG(V(L)/V(L+1))/ (RNEW(L)-RNEW(L+1))  
   GRADV(L) = .5D0* (DM+DPX) * V(L)   
END DO

L = ND  
GRADV(L) = LOG(V(L)/V(L-1))/ (RNEW(L)-RNEW(L-1))*V(L)  

! NORMALIZATION
DO K = 1,ND  
     RNEW(K) = RNEW(K)/SR  
     V(K) = V(K)/VMAX  
     GRADV(K) = GRADV(K)*SR/VMAX  
     P(K) = RHONEW(K)* (A2TE*DELMU(K)*TNEW(K))  
END DO  

RNEW(ND) = 1.D0  

!-------------
!JO Apri16: now we 'smooth' the velocity at the transition point,
! requiring a 3rd degree polynom and prescribed v, gradv above and below
! transition point (4 unknowns for the parabola, four conditions to fix them)

RR1=RNEW(NS-1)
RR2=RNEW(NS+1)

ZZ1=(/RR1**3,RR1**2,RR1,1.D0/)
ZZ2=(/RR2**3,RR2**2,RR2,1.D0/)
VMAT(1,:)=ZZ1
VMAT(2,:)=ZZ2
VMAT(3,:)=(/3.D0*RR1**2,2.D0*RR1,1.D0,0.D0/)
VMAT(4,:)=(/3.D0*RR2**2,2.D0*RR2,1.D0,0.D0/)

YY=(/V(NS-1),V(NS+1),GRADV(NS-1),GRADV(NS+1)/)

CALL LUDCMP(VMAT,4,4,INDLUD,DET)  
CALL LUBKSB(VMAT,4,4,INDLUD,YY)  

!test precision
V1TEST=DOT_PRODUCT(YY,ZZ1)
IF(ABS(1.D0-V1TEST/V(NS-1)).GT.1.D-4) STOP ' PRECISION OF V1 NOT SUFFICIENT (ROUTINTE MODVL)'
V2TEST=DOT_PRODUCT(YY,ZZ2)
IF(ABS(1.D0-V2TEST/V(NS+1)).GT.1.D-4) STOP ' PRECISION OF V2 NOT SUFFICIENT (ROUTINTE MODVL)'

!calculate smoothed velocity at NS
RR0=RNEW(NS)
ZZ1=(/RR0**3,RR0**2,RR0,1.D0/)
VSMOOTH=DOT_PRODUCT(YY,ZZ1)
PRINT* 
PRINT*,'OLD/NEW V-VALUES AROUND NS:'
WRITE(*,FMT='(3(F12.6,2X))') V(NS-1),V(NS),V(NS+1)
WRITE(*,FMT='(3(F12.6,2X))') V(NS-1),VSMOOTH,V(NS+1)

!recalculate rho and p, assuming r^2*rho*v = const
RHOSMOOTH=RHONEW(NS)*V(NS)/VSMOOTH
V(NS)=VSMOOTH
P(NS) = P(NS)/RHONEW(NS)*RHOSMOOTH
RHONEW(NS)=RHOSMOOTH

GRADVNEW=GRADV
DO L = NS-1,NS+1 
    DM = LOG(V(L)/V(L-1))/ (R(L)-R(L-1))  
    DPX =LOG(V(L)/V(L+1))/ (R(L)-R(L+1))  
    GRADVNEW(L) = .5D0* (DM+DPX) * V(L)   
END DO

!for tests
PRINT*,'OLD/NEW DVDR-VALUES AROUND NS:'
WRITE(*,FMT='(5F12.6,2X)') GRADV(NS-2:NS+2)
WRITE(*,FMT='(5F12.6,2X)') GRADVNEW(NS-2:NS+2)

GRADV=GRADVNEW

!----------------------------------------------------------------------------

DO L = 1,ND  
     IF (GRADV(L).LE.0.D0) THEN  
       GRADV(L) = -GRADV(L)  
            PRINT *, &
&             ' MODVL_UPDATE: NEGATIVE VEL. GRADIENT -- DENSITY INVERSION! AT N =',L
            WRITE(999,*) &
&             ' MODVL_UPDATE: NEGATIVE VEL. GRADIENT -- DENSITY INVERSION! AT N =',L
     END IF  
END DO  

!----from here on, we have to consider clumping
CALL CLUMPING(RNEW,V,RHONEW,CLF,RHONEWCL,RTAU23,ND,NS,VMAX,GRADV)

! scaling of ne and nh
DO L = 1,ND  
     DM=RHONEWCL(L)/RHOCL(L)
     XNE(L) = XNE(L)*DM
     XNELTE(L) = XNELTE(L)*DM
     XNH(L) = XNH(L)*DM
END DO  

! assumed that xne for ithyd ne 1 has been corrected

XNECL=XNE/CLF ! corrected for integration of tau_e (see subr. MODVL)

SIGEM = SIGMAE*(1.D0+HEI*YHE)/SUMMASS    

TAUMIN = SIGEM * XMLOSS/(4.*PI*VMAX*RNEW(1)*SR)  

CALL SPLINE(RNEW,XNECL,ARHO,BRHO,CRHO,ND)  
TAU(1) = TAUMIN  
SUMM = TAUMIN  

DO L = 1,ND - 1  
     DEL = RNEW(L+1) - RNEW(L)  
     SUMM = SUMM - SIGMAE*SR*DEL* (XNECL(L)+ DEL* (ARHO(L)/2.D0+DEL* &
&      (BRHO(L)/3.D0+ DEL*CRHO(L)/4.D0)))
     TAU(L+1) = SUMM  
END DO  

PRINT *  
PRINT *,' TAU-THOMSON (MICRO-CLUMPED) AT RSTAR = ',TAU(ND)  
PRINT *  

DO L = 1,ND  
     INDEX(L) = L  
END DO  

!
!---- file 'model' is written: XNE, XNH include CLFAC
!     (RHO average density without CLFAC)
!     from version 9.0 on: NS as last entry
!     from version 10.0 on UPDATED as last entry
!

! note: (hopefully) xnelte (and xnh) from model will not be used during
!        restart;
!        otherwise, they might need to be replaced by better estimates

! UPDATE done, thus
UPDATED=.TRUE.
UPDATE_DONE=.TRUE.

REWIND 20  
WRITE (20) TEFF,GGRAV,SR,YHE,XMU,VMAX,XMLOSS,BETA, (RNEW(I),I=1,ND), &
&   (V(I),I=1,ND), (GRADV(I),I=1,ND), (RHONEW(I),I=1,ND), &
!JO changed July 2016 
!&   (XNE(I),I=1,ND), (XNH(I),I=1,ND), (CLF(I),I=1,ND), &
&   (XNELTE(I),I=1,ND), (XNH(I),I=1,ND), (CLF(I),I=1,ND), &
&   (INDEX(I),I=1,ND), (P(I),I=1,ND), NS, UPDATED, XT

!  read photospheric rad. acceleration (start values)
OPEN(61,FILE=TRIM(MODNAM)//'/GRAD.OUT',STATUS='UNKNOWN')
DO L=1,ND
   READ(61,*) CHAR,I,ARHO(L)
ENDDO

!  store current photospheric rad. acceleration (start values)
REWIND 61
DO L=1,ND
   WRITE(61,*) 'RADACC ',L,' ',RADACC(L)
ENDDO
CLOSE (61)


! store photospheric rad. acceleration (start)
OPEN(61,FILE=TRIM(MODNAM)//'/GRAD_START.OUT',STATUS='UNKNOWN')
DO L=1,ND
   WRITE(61,*) 'RADACC ',L,' ',ARHO(L)
ENDDO
CLOSE(61)

DO LL = 1,ND  
     L = INDEX(ND+1-LL)  
     ARHO(LL) = RNEW(L)  
     BRHO(LL) = TNEW(L)  
END DO  
!
!---- file 'temp' is written
!
REWIND 21  
WRITE (21,*) ARHO, BRHO  
WRITE (21,*) (XNELTE(I),I=ND,1,-1)
WRITE (21,*) RTAU23

PRINT*,' MODEL UPDATED'
PRINT*
!REMEMBER: TAU=TAU-TH ONLY IN MICRO-CLUMPING APPROX.
CALL PRMODEL(RNEW,V,GRADV,RHONEW,XNE,XNH,TNEW,TAU,CLF,TEFF,ND)

! update of variables
R=RNEW
VELO=V
DVDR=GRADV
RHO=RHONEW
TEMP=TNEW

RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE DERIVS_UPD(X,Y,DY)  
! basic idea: T(taur) remains preserved, we have only to calculate
!             the new taur from the old one and using
! dtau/dm approx se + C rho T^(-x) approx se + C' p T^(-x-1)
! thus, dtaunew/dm approx [(dtauold/dm -se)*pnew/pold (tnew/told)^(-x-1)]+se 
!
! with se = ne * sigmae/rho := sig 

! Y(1) --- pressure
! Y(2) --- temperature
! Y(3) --- radius
! Y(4) --- taur_new

USE nlte_type
USE nlte_dim
USE nlte_opt, ONLY: OPT_PHOTCLUMP
USE nlte_var, ONLY: SRNOM,IDUM,GGRAV,TEFF,A2TE,DELMU
USE photstruc, ONLY: NDMOD, XT, CONST, EXPO, SIGEMIN, &
&                    XMEXT, TEXT, DELCHI, &
&                    POLD=>PKUR, TAUROLD=>XNEKUR, SIG=>GRADKUR, CLF_CONST

IMPLICIT NONE
!
!     delmu  corrects for constant mue
!
!     .. scalar arguments ..
REAL(DP) ::  X  
!     ..
!     .. array arguments ..
REAL(DP) ::  DY(4),Y(4)  
!     ..
!     .. local scalars ..
REAL(DP) ::  DELCHIBAR, DCHIDM, CHIBAR, DELM, DY1, DPDM, PO, DTDM, TO, DTAUDM
!     ..
INTEGER(I4B) :: IT

IF(NDMOD.NE.ID_NDEPT) STOP ' ERROR IN NDMOD - DERIVS_UPD'

! CHIBAR_H, POLD, TOLD  on regular grid
DO IT=1,NDMOD-1  
  IF (X .GE. XMEXT(IT).AND. X .LT. XMEXT(IT+1)) EXIT
ENDDO
IF (IT .EQ. NDMOD) STOP ' x not found in XM - DERIVS_UPD'

DCHIDM=(DELCHI(IT+1)-DELCHI(IT))/(XMEXT(IT+1)-XMEXT(IT))
DELCHIBAR=DELCHI(IT)+DCHIDM*(X-XMEXT(IT))

DPDM=(POLD(IT+1)-POLD(IT))/(XMEXT(IT+1)-XMEXT(IT))
PO=POLD(IT)+DPDM*(X-XMEXT(IT))

DTDM=(TEXT(IT+1)-TEXT(IT))/(XMEXT(IT+1)-XMEXT(IT))
TO=TEXT(IT)+DTDM*(X-XMEXT(IT))

DTAUDM=(TAUROLD(IT+1)-TAUROLD(IT))/(XMEXT(IT+1)-XMEXT(IT))

! NEW (former version: < 1.*SIG(IT)
IF(DTAUDM.LE..95*SIG(IT)) THEN
  PRINT*,X,IT,DTAUDM,SIG(IT)
  print*
  DO IT=1,NDMOD  
    PRINT*,IT,XMEXT(IT),DELCHI(IT),POLD(IT),TEXT(IT),TAUROLD(IT),SIG(IT)
  ENDDO
  STOP ' DTAUDM < SIG_TH'
ENDIF

! FROM REGRESSION
CHIBAR=DELCHIBAR*(CONST*Y(1)*Y(2)**EXPO+SIGEMIN)

DY1 = GGRAV - 5.67D-5/2.9979D10*TEFF**4*CHIBAR
! this is the ONLY hack to ensure that rho_phot => rho_phot/fcl
IF (OPT_PHOTCLUMP) DY1 = DY1/CLF_CONST
DY(1) = DY1* (SRNOM/Y(3))**2  

!DTAUDM FROM OLD TAUR, POLD AND TOLD AND NEW P AND T, + EXPONENT
DY(4)=((DTAUDM-SIG(IT))*(Y(1)/PO)*(TO/Y(2))**(XT+1.))+SIG(IT)

! FIND INTERVAL WITH RESPECT TO NEW TAUR
DO IT=1,NDMOD-1
  IF (Y(4) .GE. TAUROLD(IT).AND. Y(4).LT. TAUROLD(IT+1)) EXIT
ENDDO
IF (IT .EQ. NDMOD) IT=NDMOD-1

!DTDM = DTDTAU|new * DTAUDM
DY(2)=(TEXT(IT+1)-TEXT(IT))/(TAUROLD(IT+1)-TAUROLD(IT))*DY(4)

DELM = .5D0* (DELMU(IDUM-1)+DELMU(IDUM))  

DY(3) = -Y(2)/Y(1)*A2TE*DELM  
RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CHIH_PARAMS(CHI,P,T,DELCHI,SIGEMIN,CONST,EXPO,ND,NS,NSIG)
!
! parameterizes CHIBAR_H
!
USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT
IMPLICIT NONE

INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT

INTEGER(I4B), INTENT(IN) :: ND, NS, NSIG
REAL(DP), DIMENSION(ND), INTENT(IN) :: CHI,P,T
REAL(DP), DIMENSION(ND), INTENT(OUT) :: DELCHI

REAL(DP), INTENT(IN)  :: SIGEMIN
REAL(DP), INTENT(OUT) :: CONST,EXPO

REAL(DP), DIMENSION(ND1) :: FIT

INTEGER(I4B) :: K, IMIN, I
REAL(DP) :: CHI2MIN, EXPOI, CONSTI, CORR, CHI2

CHI2MIN=1.D5
EXPO=0.

! test for best interval to perform the fit

DO IMIN=NSIG,ND-4
! regression from IMIN on
  CALL LINREG(LOG10(T(IMIN:ND)),LOG10((CHI(IMIN:ND)-SIGEMIN)/P(IMIN:ND)), &
&   ND+1-IMIN,EXPOI,CONSTI,CORR)

! fit quality for "all" points (from NSIG on)  
  FIT(NSIG:ND)=(10.**CONSTI *T(NSIG:ND)**EXPOI *P(NSIG:ND))+SIGEMIN
  CHI2=SUM((CHI(NSIG:ND)-FIT(NSIG:ND))**2/CHI(NSIG:ND))
!JO Jan 2016 -- Note:
!actually, the denominator should be CHI^2 because we want to minimize
!(fit/chi-1)^2 = (fit-chi)^2/chi^2 = (chi-fit)^2/chi^2,
!but for consistency with older versions we leave this here.
!(This routine is only called for OPTCMF_FULL = F, and in this case
!the parameterization makes no problem anyway, independent of the choice of chi2 

  IF (CHI2 .LT. CHI2MIN) THEN
    CHI2MIN=CHI2
    EXPO=EXPOI
    CONST=CONSTI
  ENDIF  
ENDDO

IF (EXPO.EQ.0.) STOP ' REGRESSION NOT SUCCESSFUL (CHIH_PARAMS)'

CONST=10.**CONST
! for all photospheric points (from NS on)
FIT(NS:ND)=(CONST *T(NS:ND)**EXPO *P(NS:ND))+SIGEMIN
DELCHI(NS:ND)=CHI(NS:ND)/FIT(NS:ND)
!DO I=NS,ND
!  PRINT*,I,' ',CHI(I),' ',FIT(I),' ',DELCHI(I) 
!ENDDO

IF(MAXVAL(DELCHI(NS:ND)).GT.2. .OR. MINVAL(DELCHI(NS:ND)).LT.0.2) THEN
  DO I=NS,ND
    PRINT*,I,' ',CHI(I),' ',FIT(I),' ',DELCHI(I) 
  ENDDO
  STOP ' NO REGRESSION POSSIBLE (CHIH_PARAMS)! RE-TRY WITH UPDATE_STRUCT = .FALSE.'
ENDIF

RETURN
END
!
!-----------------------------------------------------------------------
!
! next three subroutines for TRATCOL, NFOR=66
SUBROUTINE  SPLINE_COL(X,Y,N,YP1,YPN,Y2)

!Cubic spline interpolation (Numerical Recipes p.88)
!Given arrays X and Y of legth N containing a tabulated function
!y(i)=f(x(i))
!and given values YP1 and YPN for the first derivative of the
!interpolating function at points 1 and N, returns an array Y2
!of length N which contains second derivatives of the interpolating
!function at tabulated points x(i)

!uses natural splines only if derivatives at boundary are extremely large

USE nlte_type 
IMPLICIT NONE

INTEGER(I4B), PARAMETER  :: NMAX=100
INTEGER(I4B) :: I,N,K

REAL(DP), DIMENSION(N)   :: X,Y,Y2
REAL(DP), DIMENSION (NMAX)  :: U

REAL(DP) ::  YP1,YPN,SIG,P,QN,UN


IF (YP1.GT..99D30) THEN
   Y2(1)=0.
   U(1)=0.
ELSE
  Y2(1)=-0.5
  U(1)=(3./(X(2)-X(1)))*((Y(2)-Y(1))/(X(2)-X(1))-YP1)
ENDIF

DO I=2,N-1

  SIG=(X(I)-X(I-1))/(X(I+1)-X(I-1))
  P=SIG*Y2(I-1)+2
  Y2(I)=(SIG-1)/P
  U(I)=(6.*((Y(I+1)-Y(I))/(X(I+1)-X(I))-(Y(I)-Y(I-1))/ &
& (X(I)-X(I-1)))/(X(I+1)-X(I-1))-SIG*U(I-1))/P

ENDDO

IF (YPN.EQ..99D30) THEN
   QN=0
   UN=0
ELSE
   QN=0.5
   UN=(3./(X(N)-X(N-1)))*(YPN-(Y(N)-Y(N-1))/(X(N)-X(N-1)))
ENDIF

Y2(N)=(UN-QN*U(N-1))/(QN*Y2(N-1)+1.)

DO K=N-1,1,-1
   Y2(K)=Y2(K)*Y2(K+1)+U(K)
ENDDO

RETURN

END SUBROUTINE SPLINE_COL
!
!-----------------------------------------------------------------------
!
SUBROUTINE SPLINTER(XA,YA,Y2A,N,X,Y)

!Cubic spline interpolation (Numerical Recipes p.89)
!Given arrays xa aand ya of length n, which tabulated function and given 
!array y2a (output from spline_col) and given a value of x, returns a
!cubic spline interpolated value y

USE nlte_type 

Implicit NONE
           
INTEGER(I4B)             :: K ,KLO,KHI,N,KHO
REAL(DP) ,DIMENSION(N)   :: XA,YA,Y2A

REAL(DP)                ::   H,A,B,Y,X
      
KLO = 1
KHI = N
      
1 IF (KHI-KLO.GT.1) THEN
      K = (KHI+KLO)/2

      IF (XA(K).GT.X) THEN
            KHI = K
      ELSE
            KLO = K
      END IF
      GOTO 1
END IF

H = XA(KHI)-XA(KLO)

IF (H.EQ.0.D0) STOP 'Bad xa input in routine SPLINTER'

A = (XA(KHI)-X)/H
B = (X-XA(KLO))/H
Y = A*YA(KLO)+B*YA(KHI) + &
&   ((A**3-A)*Y2A(KLO)+(B**3-B)*Y2A(KHI))*(H**2)/6.
    
RETURN
END SUBROUTINE SPLINTER
!
!-----------------------------------------------------------------------
!
SUBROUTINE DEVP(X,Y,N,YP1,YPN)

!Given an array of length N, returns first derivative at points 1 and N 

USE nlte_type

IMPLICIT NONE

INTEGER(I4B) :: N
REAL(DP), DIMENSION (N) :: X, Y
REAL(DP) :: H1,H2,YP1,YPN

H1=X(2)-X(1)
H2=X(N)-X(N-1)

YP1=((Y(2))-Y(1))/H1
YPN=((Y(N)-Y(N-1)))/H2

RETURN
END SUBROUTINE DEVP
!
!-----------------------------------------------------------------------
!
SUBROUTINE DERIV_3P(D,X,Y,N)

! Perform numerical differentiation using 3-point, Lagrangian interpolation.
! df/dx = y0*(2x-x1-x2)/(x01*x02)+y1*(2x-x0-x2)/(x10*x12)+y2*(2x-x0-x1)/(x20*x21)
! Where: x01 = x0-x1, x02 = x0-x2, x12 = x1-x2, etc.

! taken from IDL; note that cshift (f90) and shift(idl) use different sign conventions

USE nlte_type

IMPLICIT NONE

INTEGER(I4B) :: N, N2
REAL(DP), DIMENSION (N) :: X, Y, D

REAL(DP), DIMENSION(N) :: X12, X01, X02

IF (N.LT.3) STOP 'N LT 3 IN DERIV_3P'

X12 = X - CSHIFT(X,1)     !x1 - x2
X01 = CSHIFT(X,-1) - X    !x0 - x1
X02 = CSHIFT(X,-1) - CSHIFT(X,1) !x0 - x2

!Middle points
D = CSHIFT(Y,-1) * (X12 / (X01*X02)) + Y * (1./X12 - 1./X01) - &
&   CSHIFT(Y,1) * (X01 / (X02 * X12))

! Formulae for the first and last points:
D(1) =  Y(1) * (X01(2)+X02(2))/(X01(2)*X02(2)) - & 
&       Y(2) * X02(2)/(X01(2)*X12(2)) + &
&       Y(3) * X01(2)/(X02(2)*X12(2))

N2 = N-1
D(N) = -Y(N-2) * X12(N2)/(X01(N2)*X02(N2)) + &
&       Y(N-1) * X02(N2)/(X01(N2)*X12(N2)) - &
&         Y(N) * (X02(N2)+X12(N2)) / (X02(N2)*X12(N2))

RETURN
END
