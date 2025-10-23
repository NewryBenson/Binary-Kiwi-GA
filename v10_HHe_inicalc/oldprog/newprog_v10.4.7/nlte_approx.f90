module version_nlte_approx
! WRITTEN: 
!        02/00 by JP
!
! HISTORY:
!         this is a modified version
!         of the corresponding subroutines of "dice3.f" (written by
!         uwe) in the spirit of Abbott 1982
!         completely updated from version 4.0 on to account for
!         separation of different spin systems
!
! version 1.0 03/02/2000
!
! version 1.1 06/05/2000 - check for tau-line to decide which
!                          zcorr has to be taken
!                        - thermalisation of lines
!                        - ALI   
!
! version 1.1 Oct. 19th 2000; accuracy for deltatradj changed
!
! version 2.0 Feb. 12th 2001; changes for A-stars included
! NOTE: if NLTE-treatment only for hydrogen, helium not considered also here:
!       indirect influence over different electron density accounted for. 

! version 2.1 Feb. 26th 2001; changes for A-stars etc. finalized
! version 2.2 March 1st 2001: ALI treatment modified (ZETA term), 
!                             line collisions now consistent
!
! version 3.0 March 14th 2001: averaged metal line background included
!
! version 3.1 April 2nd 2001: finalized metallic line background
!
! version 3.1.1 April 10th 2001: smaller corrections, &
!                                mostly  to allow for -nan compilation
!
! version 3.1.2 April 12th 2001: bug in line treatment (lines from ground
!                                levels) cured, freq. range for line
!                                list at lower temp. modified;  
!                                metal line background only calculated
!                                at cont. freqs.
!
! version 3.1.3 April 24th 2001: xmet can be either
!                                positive (scaling of metal MASS fractions)
!                                or
!                                negative(scaling of metal NUMBER fractions)
!
! version 3.1.4 May 3rd 2001: correct abundances by Anders/Grevesse and updates
!
! version 3.1.5 May 18th 2001: vturb also in line background
!
! version 3.1.6 June 21st 2001: blocking window enlarged for hot temperatures
!
! version 4.0   Jan 2003: approx. nlte completely changed (most important:
!                         separation of different spin systems)
!
! version 3.1.6.alpha May 7th 2002: begin of work with explicit NLTE atoms
!
! version 3.1.6.beta  April 2003: Including Tcorr with metal background
!
! version 4.1 June 2003: unification of ver 4.0 and 3.1.6.beta
!
! version 5.0 June 2003: rigorous NLTE treatment for selected atoms
!                        (with jatom_full = 1, chosen in subroutine select)
!                        rates simplified where possible, &
!                        T-correction from corresponding transitions
!                        (-> new atomic data file-path: ATOMDAT_NEW!)
!
! version 5.1 July 2003: inclusion of clumping, version compatible with
!                        nlte_8.2 and higher  
! version 5.1.1 Sept 2003: inclusion of net derivatives of CBB heating- &
!                        cooling term to ensure consistent treatment
!                        also in "LTE" region!
!                        (note that the direct term cancel by def.) 
! version 5.1.2 Oct 2003: some smaller changes, bug in nindex (sumopal) removed
!                         rigorous check of packing algorithm in sumopal performed
!
! version 5.2 Nov 2004:  new atomic data checked, averaging of collion/ 
!                        scattering terms in sumopal reformulated (strict
!                        consistency with opacity mean  
! 
! version 5.3 March 2004: improved treatment of sampling to ensure
!                         that at each value of vturb a similar sampling
!                         width is present.
!                         bug in subroutine opacity identified, which
!                         has a small influence on the photospheric
!                         density stratification. (Too few edges considered)  
! version 5.3.1 June 2004: check for occ(meta)=0 (in partit) only if
!                         Teff > 9500, &
!                         since for lower Teff it can become zero because
!                         of underflow. (without this hack, program stops
!                         with 'error in m-state(done)'  
!  
! version 5.3.2 June 2004: use all elements (except H/He) for blocking calculations
!                          different treatment for explicit elements
!                          only with respect to continuum opacities and
!                          cooling/heating. Might lead to somewhat too strong
!                          self-shadowing, but is more consistent than  
!                          old approach
!                          note: opametinf gives edges for calculations of
!                                bf opacities, indexopa gives all edges!  
! version 5.3.3 May 2005:  full line list in ram if parameter ram='big'
!                          this is the default, requiring roughly 100 MB RAM
!                          old approach (reading each time) requires ram='small'
!
! version 5.3.4 Sept.2005: output of abundances to file ABUNDAN
!                          file METAL_IDL always created, &
!                          solar Helium abundance 0.1 in
!                          background abundances (used for scaling)
!                          solar-abundances from data file
!                          (ATOMDAT_NEW/abund_solar)
!                          abundance changes for explicit and background
!                          elements now from INDAT.DAT  
!
! version 5.4 March 2006:  for nlte.f90 from version 9.0 and higher
!                          REQUIRES NEW nlte_type.f90
!                          among other things,
!                          routines RESTART and SAVE_METAL_INFO changed,
!                          required for nlte versions 9.0 and higher  
!                          iterative improvement of solution of rate-equations
!                          calculated with residuum in qp precision 
!                          (possible with recent versions of intel compiler)
!
! version 6.0 April 2006:   new treatment of bf transitions with excited upper
!                           levels for "selected" elements
!                           (non-selected elements still treated in the old way, 
!                            since method depends on assumption of ionization
!                            to ground-states).
!                            for consistency, transition edges  
!                           new file atomnl2_meta_v2.0 required!!!!
!                           (inclusion of upper level of bf transition)
!                           remaining approximation: use ONE representative
!                           transition in case more bf transitions from one
!                           lower lever are present (the strongest one, with
!                           summed up alphas) 
!                           selected elements assumed to be in LTE only from
!                           LWION1 on (taur > 2)  
!
! version 6.0.1 Nov. 2006:  convergence of bg elemets even if emax > 0.03, 
!                           to allow convergence even if explicit ions oscillate  
!
! version 6.1 June 2009:    metal ionization fractions of fjk stored and restored
!                           at restart (file METAL_RESTART), to obtain consistent   
!                           ILOW, IMAX (explicit elements) in subroutine
!                           ILOWIMAX_NLTE. Old models can still be used. In 
!                           this case, the lte version of fjk is exploited. 
!                           smaller modifications w.r.t. restart 
!
! version 6.2 July 2009:    inclusion of dielectronic recombination to
!                           background elements (similar as in nlte.f90, v9.6)
!                           new data-file atomnldr required
!
! version 6.3 Sept 2009:    Stark broadening of (quasi-) resonance lines from
!                           metals and H-lines close to Lyman-jump for bg-elements.
!                           (OPTSTARK=.TRUE.)
!                           Global control options in new module nlte_opt
!                           needs nlte-version v9.9 or higher and nlte_type v1.2 
!
! version 7.0 Sept 2009:    Inclusion of work done in March 2009
!                           correct treatment of photospheric/cmf line transfer for
!                           lines from selected background elements
!                           (OPTPHOTLINES=.TRUE.)
!                           DELTATRADJ explicitely set to zero in IONIS when
!                           LTE conditions (to ensure correct calculation
!                           after update of phot.struct).  
!
! version 7.1 Dec 2009:     New routine jbar_phot_diff
!             Jan 2010:     occ(imax+1) discarded from the global
!                           error budget if ifrac(imax+1) < 1.d-12 

!                           as well, changes from minor ions discarded in the
!                           outer wind as long as they are below 3.d-2.  

!                           kl_tables moved to DATA/kl_tables, path updated

!                           if restart and changes in abundances of specific
!                           elements > 20%, metallic background re-calculated

! version 7.2 June 2010:    BUG in ILOWIMAX_NLTE (zeff correction) removed
!                           standard check for numerical precision removed
!
! version 7.2.1 April 2012: names1 moved to module nlte_app
!
! version 7.2.2 Nov/Dez 2012: bug in subroutine select fixed (when very low z) 
!                           subroutine indexfre slightly changed to improve restart
!
! version 7.2.3 April 2013: Treatment of nlambox_ly slightly changed
!                           (in correspondance of changes in nlte.f90 V10.1.5) 
!                           affected subroutines: SUMOPAL_NLTE AND SUMOPAL_LTE
!
! version 7.3.0 October 2014: several changes to allow for inclusion of X-ray 
!                           transfer and to improve convergence of bg elements
!
! version 7.4.0 October 2014: inclusion of optically thick clumping
!                           (porosity and vorosity), programmed by Jon Sundqvist)   
!
! version 7.4.1 July 2014:  inclusion of specific modifications from CMF-path (v8.0)
!                           in particular:
!                           (i) old hack in routine partit changed, since old hack
!                           could lead to oscillations. New approach ensures
!                           a smooth transition to lower excitation factors
!                           if original approach leads to too large values.  
!                           (ii) taulast changed to 0.1 (instead of 1.0) in
!                           subr. jbar_phot_diff. Increases number of lines
!                           from selected bg elements with RT!
!version 7.4.2 Dec. 2016:   photo-integrals for ground-state slightly changed
!                           in NETMET_MET, to allow for AlIV
  
character*10 :: ver_nlte_app='7.4.2' 
end module version_nlte_approx
!
!--------------------------------------------------------------------------- &
!
module nlte_app
!
! NOTE: after one run of nlte_app, the following info is present within
! the various occupation numbers:
!
! occng: 'exact' numbers for selected ions within met_imin, met_imax, 
! in dependence of depth, until LWION1-1. For not selected elements
! and ions outside this range, only part of the occup. are present
! (the others are zero, see below), as calculated by the approx. approach.
!
! For l= LWION1, ND, all occup. numbers are present (until a certain
! ionization stage, but in LTE.
!
! occnglte: all LTE occupation numbers, identical with occng in the range
! LWION1, nd 
!
! if LTEOPT = .TRUE., occng as calculated from IONIS AND IONIS_FULL_LTE
! (the latter only for selected elements) and OCCUP1/2 should be equal for
! l=LWION,ND for all levels and fairly equal for gs, 's' and 'm' levels
! above. In particular, the ionization fractions should be (almost) equal
! (checked in IONIS_FULL_LTE)  
!
! occngold: 'exact' numbers for selected elements from previous iteration.
! As occng, but only in the range met_imin, met_imax and l=1,LWION1-1.
! All other occup. numbers are zero.
!
! COMMENT: the above has been re-checked by JO in Sept. 2013 
!
! more detailed NOTES (Jan 2016):
! in approximate approach, the following philosophy is present:
! a) subr. IONIS/OCCUP
! LTE: occng = LTE (for given xne) for l=LWION (<LWION1), ND for all levels
!      occng = LTE for l=1,LWION-1 for gs, 's' and 'm' levels
!      occng = 0 for l=1,LWION-1 for other levels
!NLTE: occng = LTE (for given xne) for l=LWION (<LWION1), ND
!      occng = approx. NLTE for l=1,LWION-1 for gs, 's' and 'm' levels
!      occng = 0 for l=1,LWION-1 for other levels
! b) subr. IONIS_FULL_LTE/OCCUP overwrites, for all levels at all depth points,
!      occng with true LTE values if LTEOPT=.TRUE., for selected elements 
! c) subr. RATEEQ_MET overwrites, for all levels and for l=1,LWION1-1 
!      occng with improved NLTE values for selected elements 
!      for l=LWION1,ND, LTE-values (for actual XNE) from IONIS/IONIS_FULL_LTE
!      remain. 
!  
! NOTE: also for L = LWION1, ND occng and occnglte only identical
! for LTEOPT = .TRUE. Otherwise, differences since occng calculated with
! xne(NLTE).
!  
!
! IMPORTANT COMMENT: self shadowing not correctly accounted for
! 

USE nlte_type      
USE nlte_dim, ONLY: ID_ATOMS, ID_NDEPT
USE princesa_var, ONLY: nat,abundin=>abund,weight,labat
implicit none

!nout=1 output, = 0 no output
!ndebug= 1 minimum output (major ionization stages)
!ndebug= 2 including ionization fractions in subr. 'ionout', file 'METAL_IDL' generated
!ndebug= 3 including partition functions

integer(i4b), parameter :: ndebug=2

! cutoff : cutoff energy (in units of ionization energy) for
!          excited levels ('s' and 'm') to be included for opacities

real(sp), parameter :: cutoff=0.75    !(1.0 = all)

! minimum number fraction njk/ntot to be present if to be included in opacities

real(sp), parameter :: fracmin=1.d-12

integer(i4b) :: nwne, nout, lwion=ID_NDEPT

integer(i4b), parameter :: natom=30, nion=9, nlev=50, nrec=149 !, &
!&                          levmax=5000
integer(i4b), parameter :: nion1=nion-1

integer(i4b) :: lomin

character*(*), parameter :: fpath='../inicalc/ATOMDAT_NEW/' 

!NOTE: indexel(k) =  1 for explicit elements 
!      indexel(k) = -1 for background elements, except for
!H/He: indexel(1,2) = 0 if H/He not explicit.
!e.g., for only H as explicit element, indexel(1)=1, but indexel(2)=0, and
!      indexel(k) = -1 for the rest 
integer(i4b), dimension(natom) :: imax, jatom=0, jmax, indexel=-1, indexbg=-1, jatom_full=0
integer(i4b), dimension(natom) :: louter, lstat
integer(i4b), dimension(nrec)  :: kel, iel, ielev, levdr, ldadr, ilin=0, icol=0, irbb=0, icbb=0, irdr=0 
integer(i4b), dimension(nrec, nlev)  :: indexex, indexopa, opametinf
integer(i4b), dimension(natom, nion) :: indrec, indexfrec, calcion, indexelion
integer(i4b), dimension(natom, nion, nlev) :: ibfup, ilow, ipop
integer(i4b), dimension(natom,ID_NDEPT) :: met_imin=0,met_imax=0

real(dp), dimension(ID_NDEPT) :: te,wne,wion,xmuee
real(dp), dimension(natom) :: abund, vth, abund_h12
real(dp) :: vth8
real(dp), dimension(natom, nion) :: xion, zeta
real(dp), dimension(nrec, nlev) :: crodat,alpha,beta,s
real(dp), dimension(:,:), allocatable :: alphamet
real(dp), dimension(nrec, nlev,ID_NDEPT) :: occng=0.d0,occnglte=0.d0
real(dp), dimension(nrec, nlev,ID_NDEPT) :: coll_therm = 1.d0,corr_beta=1.d0 ! for first NLTE
real(dp), dimension(natom,nion,nlev) :: gstat, ergion, trans, gfval
real(dp), dimension(natom,nion,ID_NDEPT) :: fjk, upart, occ1=0.d0, occ1lte=0.d0, c_n 
real(dp), dimension(natom,nion,ID_NDEPT) :: taus1, taus1new, zcorr, deltatradj=1.d0

real(dp), dimension(natom,nion,id_ndept) :: occ1old=0.  ! for errors etc.
real(dp), dimension(nrec,nlev,id_ndept)  :: occngold=0. ! for betas etc.

real(dp), dimension(:,:), allocatable :: jbar_bg, alo_bg

real(dp), dimension (nlev,ID_NDEPT) :: c_mj
real(dp), dimension (nlev) :: zeta_mj

character*2, dimension(natom) :: name, names1 
character*1,  dimension(natom,nion,nlev) :: metall=' '

! for info and checks
data names1 /'H ','HE','LI','BE','B ','C ', &
&            'N ','O ','F ','NE','NA','MG', &
&            'AL','SI','P ','S ','CL','AR',&
&            'K ','CA','SC','TI','V ','CR',&
&            'MN','FE','CO','NI','CU','ZN'/

real(dp) :: teff, xmet, yhe
real(dp) :: summas, xhy, xhe

type rbb
  integer(i4b) :: low
  integer(i4b) :: lup
  real(dp) :: wave
  real(dp) :: gf
  integer(i4b) :: index
  integer(i4b) :: index_jbar
end type

type cbb
  integer(i4b) :: low
  integer(i4b) :: lup
  real(dp) :: omega
  real(dp) :: slope
end type

type rdr
  integer(i4b) :: low
  integer(i4b) :: ncomp
  integer(i4b) :: index
  real(dp) :: ener
  real(dp) :: flu
end type

type(rbb), dimension(35000) :: met_rbb
type(cbb), dimension(1500) :: met_cbb
type(rdr), dimension(3500) :: met_rdr
  
end module nlte_app
!
!-----------------------------------------------------------------------
!
module nlte_lines
!
! module for approximate line treatment
!  
USE nlte_type      
USE nlte_dim, ONLY: ID_NDEPT

implicit none

! ram = 'small': line list not put into ram (old approach)
! ram = 'big'  : line list only read once, put into ram
character*(*), parameter :: ram='big'

integer(i4b), parameter :: nbunch=50000
integer(i4b) :: nsubinter

integer(i4b), dimension(ID_NDEPT) :: nblock

real(dp), dimension(ID_NDEPT) :: vanreg, &
&                                u0,wexp,clufac,clufac_forb,vanreg1

real(dp), dimension(:,:), allocatable :: opal_aux, sum_th, sum_nth

real(dp), dimension(:,:), allocatable :: opa_eff_aux, opa_eff_grid 

real(dp), dimension(:,:), allocatable :: opacgrid, opalgrid, &
&                                        fracth,fracnth

integer(i4b), dimension(:), allocatable :: calcopal
real(dp), dimension(:), allocatable :: fgrid
real(dp), dimension(:,:), allocatable :: tradj_fg

integer(i4b) :: ntotnl3, nfullrec, nrest, nlam, iblue, ired
integer(i4b) :: nlamb_old, nlambox_old, nlambox_ly

real(dp) :: wavblue, wavred, wavcon, wavcon1
real(sp) :: xlamold

integer(i4b), dimension(:), allocatable:: id
real(sp), dimension(:), allocatable :: gf, xlam

logical :: nochange_in_metals=.false., metconv1=.true.

! parameters and data for approx. Stark-broadening
! average broadening parameter per electron according to Sahal-Brechot &
! Segre 1971 for important UV resonance lines is a few times 10^-6.
!10^-5 chosen here from comparing with Tlusty fluxes.
real(dp), parameter :: gamma_stark = 1.d-5
real(dp), parameter :: xnemin = 3.d10 ! from here on, Stark broadening
real(dp), parameter :: cutoff_stark=0.2 ! Stark profile until opal < 0.2 opac


real(dp),dimension(2:20) :: gamma_hyd
real(dp),dimension(0:5,2:20) :: coeff

! from fits to Stark broadened (Griem-approx) hydrogen profiles
! average from values for Teff = 10 ... 60 kK and log ne=12...16
! directory nlte_pub/newtemp_inner, routine stark.pro
data gamma_hyd/6.07201e-05, 0.000265054, 0.000713401, 0.00145657, &
&   0.00271228, 0.00426579, 0.00621346,  0.00838173,  0.0113937,  &
&   0.0149051,  0.0190546,  0.0239883,   0.0292864,   0.0349409,  &
&   0.0433179,  0.0482318,  0.0532926,   0.0630957,   0.0713400/
! in case, individual values might be included, &
! see Table stark_table and routine rstark.pro
! corresponding 1-sig errors in the log: 
!               0.155549    0.100630,    0.0776079,   0.0927858, &
!   0.0844182,  0.0702213,  0.0784915,   0.0817200,   0.0858360, &
!   0.0827682,  0.0805156,  0.0886683,   0.112444,    0.138174, &
!   0.140156,   0.162063,   0.174066,    0.183829,    0.208001

! polynomial coefficients for gamma_hyd, levels n=2,20 and
! fit according to
! log gamma = a(0)+a(1)*log(te/1000.)+a(2)*log(te/1000.)^2+ &
! a(3)*log(ne)+a(4)*log(ne)^2+a(5)*log(ne)^3
data coeff/ &
&-127.27362,    -0.57055,     0.29998,    26.02858,    -1.81936,     0.04210, &
& -67.17953,    -0.17496,     0.10088,    13.44877,    -0.93925,     0.02168,  &
& -28.26527,    -0.13439,     0.07713,     5.66609,    -0.41958,     0.01022,  &
&   5.92708,    -0.00452,     0.01904,    -1.46047,     0.07689,    -0.00127,  &
&   2.57007,     0.33799,    -0.09992,    -0.73362,     0.02434,     0.00001,  &
&  15.34906,     0.07536,     0.00987,    -3.54246,     0.23310,    -0.00510,  &
&  44.65504,    -0.12654,     0.07736,    -9.97078,     0.70548,    -0.01660,  &
&  20.36860,    -0.09383,     0.05338,    -4.68978,     0.32778,    -0.00766,  &
&  56.14668,     0.03935,     0.01457,   -12.60169,     0.90737,    -0.02172,  &
&  47.12978,     0.33801,    -0.09993,   -10.80985,     0.78941,    -0.01916,  &
&  37.27353,     0.12374,    -0.01363,    -8.60983,     0.63023,    -0.01534,  &
&  47.03411,     0.11592,    -0.01388,   -10.74534,     0.78712,    -0.01917,  &
&  49.41928,     0.29291,    -0.05713,   -11.34250,     0.83526,    -0.02045,  &
&  54.17860,     0.11922,     0.00540,   -12.48347,     0.92934,    -0.02301,  &
&  53.43937,     0.07391,     0.01566,   -12.35654,     0.92496,    -0.02302,  &
&  79.17628,     0.45576,    -0.08873,   -18.10460,     1.34606,    -0.03325,  &
&  73.80367,     0.23820,    -0.02172,   -16.78883,     1.24455,    -0.03068,  &
&  55.35773,    -0.01992,     0.06904,   -12.83411,     0.96821,    -0.02430,  &
&  29.93176,     0.75624,    -0.19610,    -7.46984,     0.58742,    -0.01538/  

end module nlte_lines
!
!-----------------------------------------------------------------------
!
module phot_lines
!
! module for the calculation of jbar/alo for photospheric lines
! from bg. elements
!  
USE nlte_type      

implicit none

integer(i4b), parameter :: nbdim=131, ntdim=2001, nf=21

real(dp), parameter:: fac=1.1d0     ! critical ratio of betaq(i,j)/betaq(i,j+1)
real(dp), parameter:: neglect=1.d-5 ! when to neglect element of Lambda-matrix


! precalculated tables
real(sp), dimension(nbdim, ntdim) :: xk2atab, xl2atab, xl2tab, xl3atab
real(sp), dimension(nbdim) :: betatab
real(sp), dimension(ntdim) :: tautab
real(sp) :: betamin, betamax, taumin, taumax

real(dp), dimension(nf) :: phi, pweight


end module phot_lines
!
!-----------------------------------------------------------------------
!
module phot_lines_diff
!
! module for the calculation of jbar/alo for photospheric lines
! from bg. elements
!  
USE nlte_type      

implicit none

integer(i4b), parameter :: nf=15, nmu=3

real(dp), dimension(nf) :: phi, pweight

real(dp), dimension(nmu) ::  mu,musq,wmu,wmu1,wmu2
! angles and weights according to Sykes, 1951, MNRAS 111
! double Gauss-Legendre, valid for weigths 1, mu and mu^2

data mu /0.8872983346d0,0.5d0,0.1127016654d0/
data musq /0.787298335d0,0.25d0,0.01270166538d0/
data wmu /0.277777777778d0,0.444444444444d0,0.277777777778d0/
data wmu1 /0.2464717596d0,0.222222222222d0,0.03130601817d0/
data wmu2 /0.2186939818d0,0.111111111111d0,0.003528240383d0/

end module phot_lines_diff
!      
!-----------------------------------------------------------------------
!
subroutine nlte_approx(teffin,yhein,xmetin,rho,xne,dilfac,temp, &
&   dvdr,vr,velo,clf,nd,frenew,lteopt,ntemp,optout)
!
!      calculates ionization equilibrium
!      (the old versions in the spirit of Uwe's approach)
!
!      INPUT: 
!      atomnl2_meta_v2.0, with all required data,
!      metastable levels designated by 'm',
!      first-order subordinate levels designated by 's'
!
!      dilfac with respect to R(taur=2/3). dilfac reset to unity for
!      lower radii in order to allow for transition to LTE
!
!-----------------------------------------------------------------------
!
USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT

USE fund_const, ONLY: amh
USE version_nlte_approx

USE nlte_opt, ONLY: optphotlines

USE nlte_var, ONLY: optlines,lines,lines_in_model,almost_converged
USE nlte_app, ONLY: teff,yhe,xmet,xmuee,wne,wion,te,nwne, &
& lwion,nout,ndebug,alphamet,occ1,occ1lte,occng,occnglte,taus1, &
& jatom,jatom_full,natom,nrec,kel
USE nlte_lines, ONLY: nochange_in_metals

USE nlte_porvor, ONLY: fic,tcl_fac_line,tcl_fac_cont

implicit none
integer(i4b), intent(in) :: nd,ntemp
real(dp), dimension(nd), intent(in) :: rho, xne, dilfac, temp, dvdr, vr, velo, clf !in cgs
real(dp), intent(in) :: teffin, yhein, xmetin
logical, intent(in) :: frenew, lteopt, optout

integer(i4b) :: i,k
logical start,firstnlte
data start/.true./
data firstnlte/.true./

if(optout) nout=1 

if (ID_NDEPT.ne.nd) stop ' nlte_approx: error in nwne'  
if (start.and..not.frenew) stop ' start and .not. frenew in nlte_approx'

if(.not.start.and.frenew) then
    if(.not.allocated(alphamet)) stop ' error in allocation status alphamet'  
    deallocate(alphamet)
endif    

! copy variables to module nlte_app

if (start) then
! FASTWIND LOG
  write(999,*) '          VERSION ',VER_NLTE_APP,' (nlte_approx)'
!
  taus1=300.
  nwne=nd
  teff=teffin
  yhe=yhein
  xmet=xmetin
endif

te=temp
if(lteopt) then 
    wion=1.
    do i=ntemp,nd
      if(dilfac(i).eq.0.75) goto 10 ! only for finished LTE
    enddo
    do i=ntemp,nd
      if(dilfac(i).eq.0.5) goto 10  ! usual LTE
    enddo
    stop' lwion not found in LTE' 
10    lwion=i+1
else
    wion=dilfac
    do i=ntemp,nd
      if(wion(i).eq.1.) exit
    enddo  
   lwion=i
endif  

wne=wion/xne
xmuee=rho*clf/(xne*amh) !corrected (xne contains clf)

! now let's calculate the ionization structure

if (start) then
  call abundan(yhe,xmet,teff)
  call atomdat
  call fgrid_lines(teff)
  call open_linedat
  start=.false.
endif 

if (lteopt) call select(lteopt)  ! select atoms
!if(frenew .and. .not.lteopt .and. .not.firstnlte) &
!& stop' update of freq. grid in course of nlte-iteration'

if (frenew) then
  call indexfre(lteopt)
  if (.not.lteopt) call indexfrelines
else if (.not.lteopt.and.firstnlte) then
  if(.not.allocated(alphamet)) stop ' error in allocation status alphamet'  
  deallocate(alphamet)
  call indexfre(lteopt)  
  firstnlte=.false. 
endif

call ionis(lteopt,ntemp,xne,dvdr,vr,clf) 
! update all occup with approximate (N)LTE-values

if (lteopt) then
! LTE treatment for selected elements at ALL depth points  
  call ionis_full_lte(ntemp,xne)
endif

! update all occupation numbers
call occup1(ntemp)         ! ground state n_1/n_e
call occup2(ntemp)         ! n_i/(g_i * n_e)

! times n_e: now absolute values, include clf
do i=ntemp,nd
occ1(:,:,i)=occ1(:,:,i)*xne(i)
occng(:,:,i)=occng(:,:,i)*xne(i)
enddo

if (lteopt) then
  occ1lte=occ1
  occnglte=occng
else
! full NLTE treatment for selected elements  

! select atoms and ions (from approx. method)
  if(.not.almost_converged) call select(lteopt)  
! calculate exact value of jbar for photospheric lines from selected elements
! (obsframe or cmf)
! line opacities from occngold(1...LWION1-1)/occnglte(LWION1...ND) for
! selected elements and occng for approx. elements
  if(optphotlines) call jbar_metals(xne,dvdr,vr,clf,velo)

! rate coefficients from occngold(1...LWION1-1)  
  call rateeq_met(xne,dvdr,vr,dilfac,clf,velo)
! occ1, occng and fjk (jatom_full) are explicitly overwritten

endif


! output of ionisation structure
! new since V6.1:
! since after restart METAL_IONS/METAL_IDL would be overwritten by first
! LTE run, IONOUT now creates output in dependence of LTEOPT
! ------------------------------
if(nout.ne.0) call ionout(lteopt)

if(.not.lines.or.(lteopt.and..not.lines_in_model)) return
if(nochange_in_metals) return

call neglect_ions
call opacitl(xne,lteopt,ntemp,dvdr,vr,velo,clf,frenew)
optlines=.true.

return
end
!
!------------------------------------------------------------------------
!
SUBROUTINE ILOWIMAX_NLTE(ILOWNL,IMAXNL,ILOWLTE,IMAXLTE,TE,TEFF,ND,OPTCMF1,NS)
! called from nlte.f90
! kept here because module nlte_app required
! 
! definition of ilownl and imaxnl for metals, using the info obtained
! in nlte_approx 
! (for z=0, ILOWIMAX_OLD is used, whereas H/HE are always treated
! in the old spirit, via ILOWIMAX_LTE  
!
! HERE YOU CAN PLAY AROUND IF NECESSARY!!!
  
USE nlte_type
USE nlte_dim
USE nlte_opt, ONLY: GLOBAL
USE princesa_var, ONLY: NAT, LABAT, NIONS, ZEFF, LE, LI, IFIRSL

USE nlte_var, ONLY: IONIS,ENIONND,IMIA,IMAA,IMIANL,IMAANL,BLEVEL,ALEVEL
USE nlte_app, ONLY: NATOM, NION, JMAX, INDEXEL, FJK

USE nlte_xrays, ONLY: OPTXRAY

IMPLICIT NONE

! new parameter GLOBAL (from V6.1 on) set in nlte_opt
! if set to .true., ILOWNL, IMAXNL will be constant throughout complete atmosphere, &
! controlled by the average in between NS-5 to NS+5
! if set to .false., ILOWNL, IMAXNL will be locally consistent (but with
! disadvantages regarding the CMF lines, see below

!     .. parameters ..
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS
INTEGER(I4B), PARAMETER :: NREC=ID_LLEVS  
INTEGER(I4B), PARAMETER :: ND1=ID_NDEPT  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  ND, NS
REAL(DP) :: TEFF
LOGICAL :: OPTCMF1
!     ..
!     .. array arguments ..
REAL(DP) ::  TE(ND1)  
INTEGER(I4B) ::  ILOWNL(ND1,KEL),IMAXNL(ND1,KEL), &
&                ILOWLTE(ND1,KEL),IMAXLTE(ND1,KEL)  

INTEGER(I4B) :: I,J,K,L,IM,IM1,IM2,IXLOW,IXMAX,MEAN,NI,IMI,IMA,IONI, &
&               NATO, INATO, IJ, NLAS, IGENIO
INTEGER(I4B), DIMENSION(1) :: IMAX0,IMAX1,IMAX2
!INTEGER(I4B) :: MAIN,I1,I2,IMI1,IMA1,ICONVER

REAL(DP), DIMENSION(NION) :: F

REAL(DP) :: IMEAN

!REAL(DP) :: EN,REL

CHARACTER AT*6  
CHARACTER*2, DIMENSION(NATOM) :: NAMES

!LOGICAL :: HELONE, OPTOUT
LOGICAL :: LOW, HIGH

EXTERNAL IGENIO

! copy from nlte_approx
DATA NAMES /'H ','HE','LI','BE','B ','C ', &
&          'N ','O ','F ','NE','NA','MG', &
&          'AL','SI','P ','S ','CL','AR',&
&          'K ','CA','SC','TI','V ','CR',&
&          'MN','FE','CO','NI','CU','ZN'/

! remember: ilow = 1 refers to lowest ion in DETAIL
!           in fjk, on the other hand, we count from neutrals on

DO K = 3, NATOM  !(not H/He) 
  I=INDEXEL(K)
  IF (I.GT.0) THEN
! at first, perform consistency check
    IF(LABAT(I) .NE. NAMES(K)) STOP ' explicit metals not found in NAMES'

      PRINT*,' ATOM: ',LABAT(I)
      DO L = 1,ND

        F=FJK(K,:,L)
        F(JMAX(K)+1:NION)=0.
        IMAX0=MAXLOC(F)
        IM=IMAX0(1)
        IMAX1=MAXLOC(F,MASK=F.LT.F(IM))
        IM1=IMAX1(1)
        IMAX2=MAXLOC(F,MASK=F.LT.F(IM1))
        IM2=IMAX2(1)
        IXLOW=MIN0(IM,IM1,IM2)
        IXMAX=MAX0(IM,IM1,IM2)
        IF(IXLOW.EQ.0) STOP' IXLOW = 0' 
        IF(IXMAX.GT.JMAX(K)) STOP' IXMAX > JMAX' 
        IF(IXMAX-IXLOW.NE.2) THEN
          IF(IXMAX-IXLOW.GT.3) &
&           STOP' SOMETHING REALLY WRONG WITH IXLOW, IXMAX'
            IM1=IXMAX
            IM2=IXLOW
            IF(F(IM1).LT.F(IM2)) THEN 
               IXMAX=IXMAX-1
            ELSE   
               IXLOW=IXLOW+1
            ENDIF
            IF(IXMAX-IXLOW.NE.2)  THEN
              PRINT*,L
              PRINT*,FJK(K,:,L)
              PRINT*,IXLOW,IXMAX
              STOP' PROBLEMS IXLOW, IXMAX COULD NOT BE CURED'   
            ENDIF 
        ENDIF
        ILOWNL(L,I)=IXLOW
        IMAXNL(L,I)=IXMAX
! for tests
!        PRINT*,L,I,IXLOW,IXMAX
      ENDDO

! calculate average ionization in most decisive region, assumed here
! between NS-5 and NS+5  (with NS the transition point) 
! MIGHT NEED TO BE MODIFIED. (FROM NS-10 TO NS+10)
    IF(NS+5 .LE. 0) STOP ' SOMETHING WRONG WITH TRANSITION POINT'

    IMEAN=0.
    DO L=NS-5,NS+5
      IMEAN=IMEAN+(ILOWNL(L,I)+1.)
    ENDDO
    IMEAN=IMEAN/11.

    PRINT*,' AVERAGE IONIS. IN LINE-FORMING REGION (1 = neutral): ',IMEAN

! consider effective charge (DETAIL!)
! HERE WAS A BRUTAL BUT. REMOVED FROM V7.2 ON
    ILOWNL(:,I)=ILOWNL(:,I)-ZEFF(I)
    IMAXNL(:,I)=IMAXNL(:,I)-ZEFF(I)

    MEAN=NINT(IMEAN-ZEFF(I))

    NI=NIONS(I)-1 !(WITHOUT K-LEVEL)
    LOW=.FALSE.
    HIGH=.FALSE.

    IF(MEAN.LT.1) THEN
      PRINT*,' WARNING!!!!!'
      PRINT*,' WARNING!!!!!'
      PRINT*,' WARNING!!!!!'
      PRINT*,' WARNING!!!!! MEAN (CHARGE CORRECTED < 1, INCLUDE LOWER IONISATION STAGES IN DETAIL-FILE'
      PRINT*,' WARNING!!!!!'
      PRINT*,' WARNING!!!!!'
      PRINT*,' PROCEED AT OWN RISK!!!'
      MEAN=MEAN+1
      LOW=.TRUE.
      IF(MEAN.LT.1) STOP ' STILL MEAN < 1, CANNOT PROCEED'
    ENDIF  

    IF(MEAN.GT.NI) THEN
      PRINT*,' WARNING!!!!!'
      PRINT*,' WARNING!!!!!'
      PRINT*,' WARNING!!!!!'
      PRINT*,' WARNING!!!!! MEAN (CHARGE CORRECTED > NION, INCLUDE HIGHER IONISATION STAGES IN DETAIL-FILE'
      PRINT*,' WARNING!!!!!'
      PRINT*,' WARNING!!!!!'
      PRINT*,' PROCEED AT OWN RISK!!!'
      MEAN=MEAN-1
      HIGH=.TRUE.
      IF(MEAN.GT.NI) STOP ' STILL MEAN < NION, CANNOT PROCEED'
    ENDIF  

    IF (LOW.AND.HIGH) STOP' SOMETHING WRONG WITH LOW OR HIGH(1)'

! IF YOU WANT TO HAVE LOCALLY CONSISTENT IONIZATION FRACTIONS (GLOBAL=.FALSE.)
! WARNING: BY DOING SO, BE AWARE THAT SOME CMF LINES WILL BE CACULATED IN SOBO 
    IF (.NOT.GLOBAL) GOTO 100
       
! IF THE NEXT LINES ARE EXECUTED (GLOBAL=.TRUE.), YOU WILL HAVE CONSTANT
! ILOW/IMAX FOR ALL DEPTH POINTS, WHICH CAN BE ADVANTAGEOUS W.R.T. CMF-TRANSFER 
    IF (MEAN.EQ.1) THEN
      ILOWNL(:,I)=MEAN
      IF(NI.EQ.1) THEN
        IMAXNL(:,I) = MEAN
      ELSE IF(NI.EQ.2 .OR. LOW) THEN  !ONLY TWO STAGES, OTHERWISE NUMERICAL PROBLEMS
        IMAXNL(:,I) = MEAN+1
      ELSE
        IMAXNL(:,I) = MEAN+2
      ENDIF

    ELSE IF (MEAN.EQ.NI) THEN

      IMAXNL(:,I)=MEAN
      IF(NI.EQ.1) THEN
        ILOWNL(:,I) = MEAN
      ELSE IF(NI.EQ.2. .OR. HIGH) THEN !ONLY TWO STAGES, OTHERWISE NUMERICAL PROBLEMS
        ILOWNL(:,I) = MEAN-1
      ELSE
        ILOWNL(:,I) = MEAN-2
      ENDIF

    ELSE
      IF(LOW.OR.HIGH) STOP' SOMETHING WRONG WITH LOW OR HIGH(2)'
      ILOWNL(:,I)=MEAN-1 
      IMAXNL(:,I)=MEAN+1 
    ENDIF

    IF(OPTXRAY) THEN
!think about O and P
!      IF (LABAT(I).EQ.'C' .AND. (IMAXNL(1,I)+ZEFF(I)).LT.5) THEN
!        ILOWNL(:,I)=3-ZEFF(I) 
!        IMAXNL(:,I)=5-ZEFF(I) 
!      ENDIF      
      IF (LABAT(I).EQ.'N' .AND. (IMAXNL(1,I)+ZEFF(I)).LT.5) THEN
        ILOWNL(:,I)=3-ZEFF(I) 
        IMAXNL(:,I)=5-ZEFF(I) 
      ENDIF      
      IF (LABAT(I).EQ.'O' .AND. (IMAXNL(1,I)+ZEFF(I)).LT.6) THEN
        ILOWNL(:,I)=4-ZEFF(I) 
        IMAXNL(:,I)=6-ZEFF(I) 
      ENDIF      
      IF (LABAT(I).EQ.'P' .AND. (IMAXNL(1,I)+ZEFF(I)).LT.5) THEN
        ILOWNL(:,I)=3-ZEFF(I) 
        IMAXNL(:,I)=5-ZEFF(I) 
      ENDIF      
    ENDIF

100 CONTINUE

    DO L=1,ND  ! IMAX (LTE) LOWER?
      IONI = IONIS(I,L)  
      IMAXNL(L,I)=MIN(IONI,IMAXNL(L,I))
      IF(IMAXNL(L,I)-ILOWNL(L,I) .LT.0) STOP' IMAX TOO LOW (IONIS)!!!'
    ENDDO

ENDIF
ENDDO  
!
!------------ determination of imia, imaa (nlte)
!
DO K = 1,NAT
  IF (LABAT(K).EQ.'H' .OR. LABAT(K).EQ.'HE') THEN
    ILOWNL(:,K)=ILOWLTE(:,K)
    IMAXNL(:,K)=IMAXLTE(:,K)
    IMIANL(K)=IMIA(K)
    IMAANL(K)=IMAA(K)
    CYCLE
  ENDIF

     IMI = NIONS(K)  
     IMA = 0  

     DO L = 1,ND  
          IMI = MIN0(ILOWNL(L,K),IMI)  
!COMMENTED (V6.1)
!          IF(IMI.NE.ILOWNL(L,K)) STOP' ERROR IN IMI PHILOSOPHY'
          IMA = MAX0(IMAXNL(L,K),IMA)  
     END DO

     IF (IMA.EQ.0) STOP ' SOMETHING WRONG WITH IMA'  
     IMIANL(K) = IMI  
     IMAANL(K) = IMA  
     IF(IMI.LT.IMIA(K)) STOP' IMI < IMI(LTE)'
     IF(IMA.GT.IMAA(K)) STOP' IMA < IMA(LTE)'

!     IF (OPTOUT .AND. ICONVER.EQ.1)
     WRITE (*,FMT=9010) K,IMI,IMA  
     PRINT*
END DO  

 9010 FORMAT (' ELEM. NO. ',I2,' WITH ILOW= ',I1,' AND IMAX= ',I1, ' (NLTE, ZEFF CORRECTED)')  


! check for new levels
! FROM VER 6.1 ON, ONLY HERE (and in ILOWIMAX_LTE, but not in COPYOCC):
! ILOW and IMAX can change. If they change to higher stages, 
! subroutine OPACITC has no info how to deal with this.
! Example: in a certain iteration, ILOW=1, IMAX=3 at a certain L.
! after ILOWIMAX_NLTE has been called (either due to restart, or because
! of temperature update, ILOW/IMAX change to 2/4. Altough the occupation
! numbers of the ground-level of i=4 is known, all other levels as well
! as the bf-transition 4_1 to 5_1 remain unkown. THUS: 


DO L=1,ND

   READ (17,REC=L) (BLEVEL(I),I=1,NREC)  
   READ (15,REC=L) (ALEVEL(I),I=1,NREC)

   DO I=1,NREC
      NATO  = LE(I)  
      INATO = LI(I)  
      IJ = IGENIO(NATO,IMAXNL(L,NATO))+1  
      NLAS = IFIRSL(IJ)  
! slightly modified from V6.1 on
      IF(BLEVEL(I).EQ.0. .AND.   &
&      ((INATO.GE.ILOWNL(L,NATO).AND.INATO.LE.IMAXNL(L,NATO)).OR. I.EQ.NLAS)) THEN
! VERY SMALL OCCUP. NUMBER
         BLEVEL(I)=ALEVEL(I)*1.D-10
        PRINT*,'NEW LEVEL ',I,' AT L = ',L,': OCC MODIFIED TO OCC-LTE*1.D-10'
!           print*,i,nato,inato,ilow(l,nato),imax(l,nato)
!        stop
      ENDIF
   ENDDO  
   WRITE (17,REC=L) (BLEVEL(I),I=1,NREC)  
END DO  

RETURN
END
!      
!-----------------------------------------------------------------------
!
subroutine abundan(yhe,xmet,teff)
!
!----------------------------------------------------------------------
!     determines abundances of the elements
!  
!     all metals (without the prespecified ones, see below) are rescaled
!     to a fraction of xmet of
!     their corresponding solar MASS-fraction (xmet > 0)
!
!     or
!  
!     their corresponding solar NUMBER-fractions (xmet < 0)
!
!     helium content is changed to yhe
!     prespecified metall abundances (with respect to H) are conserved!  
!
!     solar abundances are from Anders & Grevesse
!     (1989, Geochim. Cosmochim. Acta 53)
!
!     with corrections from
!           
!     Holweger, Heise and Kock, 1990, A&A 232, 510,
!     St"urenberg & Holweger,   1990, A&A 237, 125,
!     Holweger, Kock and Bard,  1995, A&A 296, 233)
!     N. Grevesse & A. Sauval,  1998, Space Science Reviews 85, 161
!
!
!     INPUT:
!        yhe   : specified Helium abundance (with respect to H)
!        xmet  : total metal abundance with respect to solar mass-fractions
!        prespecified abundanced for specific elements (with respect to H)
!
!     if OPTKUR is set to true during compilation, Kurucz abundances are
!     used (only for test calculations!)
!
!     OUTPUT: 
!        abund(natom)  : abundances (with respect to all particles)
!        summas        : mean atomic weight (in m_H)
!        xhy  : hydrogen mass fraction
!        xhe  : helium         "      
!
!
!     VERSION:
!        1.0   03/02/2000 JP 
!        2.0   24/04/2001 JP
!        3.0   02/05/2001 JP!
!        4.0   end 2005
!  
! Note, that Dumpfbacke U.S. has confused the abundances from Anders &
! Grevesse with the old ones by Holweger.
! This has been corrected now in version 3.0. 
!
!   from version 4.0 on, solar abundances are read from file 
!
!----------------------------------------------------------------------
USE nlte_type
USE fund_const, only: amh,akb
USE nlte_var, only: vturb, modnam, nat_bg, names_bg, abundbg, abund_have_changed
USE nlte_app, only: ndebug, nat, natom, abund, abundin, labat, weight, &
&                   summas, xhy, xhe, indexel, indexbg, vth, fpath, &
&                   names1, abund_h12
USE nlte_opt, only: optkur

implicit none
real(dp), intent(in) :: yhe,xmet,teff

real(dp), dimension(natom) :: solar,xsol,xmass,aweight,abkur,abundold

integer(i4b) :: i,j,k,index,ios
real(dp) :: rest, zsolar, zme, corr, sumabu

character*6, dimension(natom) :: names

logical :: exist
!
!     
!                             atomic number Z and name
!                   ---------------------------------------------      
!                    1 H     2 He    3 Li    4 Be    5 B     6 C
!                    7 N     8 O     9 F    10 Ne   11 Na   12 Mg
!                   13 Al   14 Si   15 P    16 S    17 Cl   18 Ar
!                   19 K    20 Ca   21 Sc   22 Ti   23 V    24 Cr
!                   25 Mn   26 Fe   27 Co   28 Ni   29 Cu   30 Zn
!
! for info and checks -> moved to nlte_app from v7.2.1
! data names1 /'H ','HE','LI','BE','B ','C ', &
!   &          'N ','O ','F ','NE','NA','MG', &
!   &          'AL','SI','P ','S ','CL','AR',&
!   &          'K ','CA','SC','TI','V ','CR',&
!   &          'MN','FE','CO','NI','CU','ZN'/

!
!
!standard set of SOLAR abundances used for background elements
!(sources see above)
!
!now read from file ATOMDAT_NEW/abundan_solar
!
!if you don't like these numbers, change this file.
!Take care since these numbers must be the SOLAR ones, used for
!renormalization if xmet > 0!


!DATA solar !old!     /12.000,10.930, 1.100, 1.400, 2.550, 8.520, &
!Finally, we use 0.1 as standard Helium abundance
!DATA solar      /12.000,11.000, 1.100, 1.400, 2.550, 8.520, &
!                  7.920, 8.830, 4.560, 8.080, 6.330, 7.580, &
!                  6.470, 7.550, 5.450, 7.330, 5.500, 6.400, &
!                  5.120, 6.360, 3.170, 5.020, 4.000, 5.670, &
!                  5.390, 7.500, 4.920, 6.250, 4.210, 4.600/

! Kurucz abundances, normalized to ntot approx n_H + n_He
data abkur /   0.91100,  0.08900, -10.88, -10.89,  -9.44,  -3.48, &
&             -3.99,     -3.11,    -7.48,  -3.95,  -5.71,  -4.46, &
&             -5.57,     -4.49,    -6.59,  -4.83,  -6.54,  -5.48, &
&             -6.82,     -5.68,    -8.94,  -7.05,  -8.04,  -6.37, &
&             -6.65,     -4.37,    -7.12,  -5.79,  -7.83,  -7.44/ 


data aweight / 1.008,  4.003,  6.941,  9.012, 10.811, 12.011, & 
&             14.007, 16.000, 18.998, 20.180, 22.990, 24.305,&
&             26.982, 28.085, 30.974, 32.066, 35.453, 39.948,&
&             39.098, 40.078, 44.956, 47.88,  50.941, 51.996,&
&             54.938, 55.847, 58.933, 58.69,  63.546, 65.39/  

open(1,file=fpath//'abundan_solar',status='old')
do i=1,natom
  read(1,*) j,names(i),solar(i)
  if(names(i).ne.names1(i)) stop' error in abundan_solar'
enddo
close(1)

if(optkur) print*,' KURUCZ ABUNDANCES USED!!!'
print*
if(ndebug.ge.2) then
IF(xmet.gt.0.) then
print* 
print*,'YHE = ',yhe,' XMET = ',xmet,' SCALING TO SOLAR MASS-FRACTIONS'
print*  
else
print* 
print*,'YHE = ',yhe,' XMET = ',abs(xmet),' SCALING TO SOLAR NUMBER-FRACTIONS'
print*  
endif
print*,'PRESPECIFIED EXPLICIT ELEMENTS'
do k=1,nat
  print*,'ELEMENT ',labat(k),' WITH ABUNDANCE(TO H):',abundin(k)
enddo  
print* 
endif

!hydrogen and helium a priori
indexel(1)=0
indexel(2)=0

!find out which elements are prespecified
do i = 1,natom
  do k = 1,nat
    if(labat(k).eq.names(i)) then
      indexel(i)=k
      if(i.eq.1.and.abundin(k).ne.1.d0) stop 'error in hydrogen-abundance' 
    endif  
  enddo
enddo  

!prespecified background elements
if(nat_bg.gt.0) then
print*,'PRESPECIFIED BACKGROUND ELEMENTS'

do k = 1,nat_bg
  do i = 1,natom
    if(names_bg(k).eq.names(i)) then
      print*,'ELEMENT ',names_bg(k),' WITH ABUNDANCE(TO H):',abundbg(k)
      indexbg(i)=k
      exit
    endif  
  enddo
  if(i.eq.natom+1) then
    print*,' wrong element name:',names_bg(k)
    stop' wrong element name'
  endif
enddo  

!perform test (indexel and indexbg must be mutually exclusive)
do i=1,natom
  if(indexel(i).gt.-1.and.indexbg(i).gt.0) then
    print*,'element',i,': indexel and indexbg inconsistent'
    stop' indexel and indexbg inconsistent'
  endif
enddo

endif

if(xmet.gt.0..and..not.optkur) then

!path for scaling to solar mass fraction

! solar values
! ------------
!
summas=0.
rest=0.

do i=1,natom
   abund(i) = 10.**(solar(i)-solar(1)) !now normalized to hydrogen
   xsol(i) = abund(i)*aweight(i)
   if(indexel(i).gt.-1.or.indexbg(i).gt.0) then
     summas = summas + xsol(i)
   else 
     rest = rest + xsol(i)
   endif 
enddo    

summas=summas+rest
zsolar=rest/summas ! only for not prespecified metals

!redefine now solar values (if prespecified) and perform some checks
!Note that all prespecified input values (from DETAIL-ed file)
!have been already normalized to hydrogen.

do i = 1,natom
  k=indexel(i)
  if(k.gt.0) then
  if(abs(1.-weight(k)/aweight(i)).gt.0.01) stop ' error in atomic masses'
  abund(i)=abundin(k)
  endif

  k=indexbg(i)
  if(k.gt.0) then
  abund(i)=abundbg(k)
  endif
enddo  
abund(2)=yhe  ! use always prespecified Helium abundance (from INDAT.DAT)

! Now rescaling to xmet


!At first, same game as above for actual values (which have been just created)
summas=0.
rest=0.

do i=1,natom
   xmass(i) = abund(i)*aweight(i)
   if(indexel(i).gt.-1.or.indexbg(i).gt.0) then
     summas = summas + xmass(i)
   else 
     rest = rest + xmass(i)
   endif 
enddo    

zme=xmet*zsolar
!
! from approach:
!1.*w(1)+yhe*w(2)+sum(a(k)*w(k)(prespec)+corr*sum(a(k)*w(k))(solar) = 
! summas+corr*rest
!
! and thus
!
!(corr*rest)/(summas+corr*rest) =: zme
!
corr=(summas*zme)/(rest*(1.-zme))

!renormalisation of rest metals
renorm: do  i = 3,natom
     if(indexel(i).gt.0.or.indexbg(i).gt.0) cycle renorm !don't touch prespecified values
     abund(i)=abund(i)*corr 
enddo renorm   

!----------------------------------------------------------------------

else if(xmet.lt.0..and..not.optkur) then
!
!path for scaling to number fractions

! solar values
! ------------
!

do i=1,natom
   abund(i) = 10.**(solar(i)-solar(1)) !now normalized to hydrogen
enddo


! scale with xmet (scaling of abundance normalized to H is identical
! to scaling of abundances normalized to ntot, as long as metals are
! concerned)

renorm1: do  i = 3,natom
     if(indexel(i).gt.0.or.indexbg(i).gt.0) cycle renorm1 !don't touch prespecified values
     abund(i)=abund(i)*abs(xmet) 
enddo renorm1   


!redefine now solar values (if prespecified) and perform some checks
!Note that all prespecified input values (from DETAIL-ed file)
!have been already normalized to hydrogen.

do i = 1,natom
  k=indexel(i)
  if(k.gt.0) then
  if(abs(1.-weight(k)/aweight(i)).gt.0.01) stop ' error in atomic masses'
  abund(i)=abundin(k)
  endif

  k=indexbg(i)
  if(k.gt.0) then
  abund(i)=abundbg(k)
  endif
enddo  
abund(2)=yhe  ! use always prespecified Helium abundance (from INDAT.DAT)

else if(xmet.lt.0..and.optkur) then

!
!path for scaling to number fractions, using Kurucz abundances

! solar values
! ------------
!

corr = 1.D0 + abkur(2)/abkur(1)
abund(1) = abund(1)*corr
abund(2) = abund(2)*corr
do i=3,natom
   abund(i) = 10.**abkur(i) *corr  !now normalized to hydrogen
enddo


! scale with xmet (scaling of abundance normalized to H is identical
! to scaling of abundances normalized to ntot, as long as metals are
! concerned)

renorm2: do  i = 3,natom
     if(indexel(i).gt.0.or.indexbg(i).gt.0) cycle renorm2 !don't touch prespecified values
     abund(i)=abund(i)*abs(xmet) 
enddo renorm2   


!redefine now solar values (if prespecified) and perform some checks
!Note that all prespecified input values (from DETAIL-ed file)
!have been already normalized to hydrogen.

do i = 1,natom
  k=indexel(i)
  if(k.gt.0) then
  if(abs(1.-weight(k)/aweight(i)).gt.0.01) stop ' error in atomic masses'
  abund(i)=abundin(k)
  endif

  k=indexbg(i)
  if(k.gt.0) then
  abund(i)=abundbg(k)
  endif
enddo  
abund(2)=yhe  ! use always prespecified Helium abundance (from INDAT.DAT)

else
  stop ' wrong specification of abundances'

endif

index = 6*(natom/6.)
!----------------------------------------------------------------------
! from v7.1 on
!
! check whether ABUNDAN exists, and, in case, test whether old abundances
! are the same as new abundances;
! if no and restart, metal-background will be recalculated
! (-> in restart, METALS-CONVERGED set to false
abund_have_changed=.false.

inquire(FILE=trim(modnam)//'/ABUNDAN',exist=exist)
if (exist) then
  open(1,file=trim(modnam)//'/ABUNDAN',status='old',iostat=ios)
  if (ios.ne.0) STOP' ABUNDAN DOES NOT EXIST BUT SHOULD'
  read(1,*)
  read(1,*)
  read(1,*)
  do i = 0,index-6,6
     read(1,15) (k,abundold(j+i),j=1,6)
  end do
  close(1)

  do i=1,natom
    if (abundold(i).eq.0.) then
      if(abund(i).eq.0.) cycle
      abund_have_changed=.true.
      exit
    endif  
    if(abs(1.-abund(i)/abundold(i)) .gt. 0.2) then
      abund_have_changed=.true.
      exit
    endif  
  enddo

  if(abund_have_changed) then
    print*
    print*,'AT LEAST ONE ELEMENT WITH SIGNIF. DIFFERENT ABUNDANCE THAN IN PREVIOUS MODEL'
    print*
    do i=1,natom
       if(abundold(i).eq.0. .and. abund(i).ne.0.) then
         write(*,fmt="(' ELEMENT ',a3,' old abund: ',e8.2,' new abund: ',e8.2)") &
&        names(i),abundold(i),abund(i)
       else if(abs(1.-abund(i)/abundold(i)) .gt. 0.2) then
         write(*,fmt="(' ELEMENT ',a3,' old abund :',e8.2,' new abund :',e8.2)") &
&         names(i),abundold(i),abund(i)
       endif
    enddo  
  endif 
endif
!
! continue with first output (also written to file ABUNDAN)
! ------
open(1,file=trim(modnam)//'/ABUNDAN',status='unknown')
write(*,10) ' NUMBER FRACTIONS OF THE ELEMENTS WITH RESPECT TO HYDROGEN'
write(1,10) ' NUMBER FRACTIONS OF THE ELEMENTS WITH RESPECT TO HYDROGEN'

do i = 0,index-6,6
   write(*,15) (j+i,abund(j+i),j=1,6)
   write(1,15) (j+i,abund(j+i),j=1,6)
end do

if(natom.gt.6*index)then
   write(*,15) (j,abund(j),j=index+1,natom)
   write(1,15) (j,abund(j),j=index+1,natom)
endif

!save for Raymond-Smith code
abund_h12=abund

! now final mass fractions
summas=0.

do i=1,natom
   xmass(i) = abund(i)*aweight(i)
   summas = summas + xmass(i)
enddo    
xmass=xmass/summas

xhy = xmass(1)
xhe = xmass(2)

! now number fractions (with respect to all particles)
! --------------------
sumabu = 0.

do i = 1,natom
  sumabu = sumabu + abund(i)
end do

! normalization
summas = 0.               ! mean atomic weight of mixture, sum(nk*mk)/sum(nk)
do i = 1,natom
   abund(i) = abund(i)/sumabu ! ni/sum(nk)
   summas = summas + abund(i)*aweight(i)
enddo

  write(*,10) ' NUMBER FRACTIONS OF THE ELEMENTS'
  write(1,10) ' NUMBER FRACTIONS OF THE ELEMENTS'
  index = 6*(natom/6.)

  do i = 0,index-6,6
     write(*,15) (j+i,abund(j+i),j=1,6)
     write(1,15) (j+i,abund(j+i),j=1,6)
  end do

  if(natom.gt.6*index)then
     write(*,15) (j,abund(j),j=index+1,natom)
     write(1,15) (j,abund(j),j=index+1,natom)
  endif

write(*,25) ' MASS FRACTION BY CLASS: H, He, Me',&
     &     xmass(1),xmass(2),1.-xmass(1)-xmass(2)
write(1,25) ' MASS FRACTION BY CLASS: H, He, Me',&
     &     xmass(1),xmass(2),1.-xmass(1)-xmass(2)

  write(*,10) ' MASS FRACTIONS OF THE ELEMENTS'
  write(1,10) ' MASS FRACTIONS OF THE ELEMENTS'
  index = 6*(natom/6.)

  do i = 0,index-6,6
    write(*,15) (j+i,xmass(j+i),j=1,6)
    write(1,15) (j+i,xmass(j+i),j=1,6)
  end do

  if(natom.gt.6*index)then
    write(*,15) (j,xmass(j),j=index+1,natom)
    write(1,15) (j,xmass(j),j=index+1,natom)
  endif

write(*,20) ' MEAN ATOMIC MASS OF MIXTURE: ', summas
write(1,20) ' MEAN ATOMIC MASS OF MIXTURE: ', summas

! vthermal at teff
do i=1,natom
  vth(i)=sqrt(2.*akb*teff/(amh*aweight(i))+vturb*vturb)
enddo

close(1)

return

 10   format(/,a,/)
 15   format(1x,10(i4,1x,e8.2))
 20   format(/a,2x,f5.3)
 25   format(/a,3(2x,g10.3))

end
!
!----------------------------------------------------------------------
!
subroutine atomdat
!----------------------------------------------------------------------
!     reads all required atomic data for calculation of
!     ionisation structure, occupation numbers and opacities
!
!     INPUT : 
!       'generalinfo' : general counting information
!       'partit'      : statist. weight of ground state of
!                       (J_max+1) ionisation state
!       'atomnl2_meta_v2.0': level information, generated by 'meta.f'
!                       includes info concerning upper levels of bf transitions
!       'atomcol'     : collision data
!       'atomgf3'     : line data
!       'atomnldr'    : dielectronic data
!
!     CALLED BY: 
!        nlte_approx
!
!     CHANGES:
!        10/91 zeta-values calculated (contributions from considered
!              levels only)
!        02/00 to f90 and consistency with nlte
!
!        02/01 eta recalculated to be consistent
!
!        10/02 new definition of zeta
!
!        04/06 ionization to excited levels enabled
!        07/09 inclusion of dielectronic recomb.
!----------------------------------------------------------------------
!
! crodat=alpha_i *g_i *nu_i^2/(alpha_1*g_1*nu_1^2)
!
USE nlte_type
USE nlte_app, only: fpath, natom, nion, nrec, &
& jatom, jmax, kel, iel, ielev, ibfup, ilow, ipop, indrec, &
& abund, xion, zeta, crodat, gstat, ergion, &
& name, metall, alpha, beta, s, trans, gfval, &
& ilin, icol, irbb, icbb, met_rbb, met_cbb, &
& levdr, ldadr, met_rdr, irdr

USE nlte_opt, ONLY: optdr

USE princesa_var, ONLY: nat, labat

implicit none
integer(i4b) :: i, idum, ios, irec, j, jdum, k, lev, level, n, jj, kk
integer(i4b) :: eastat1, eastat2, nr, lines, ind, npart, nrest, low, lup, nc, &
& nrbb, ncbb, nrdr, ndr, levdr1, ldadr1, ilowdr, icompdr
real(dp) :: recsum, wave, gf, slope, omega, ener, flu
character*1 :: met

name=(/' H', 'He', 'Li', 'Be', ' B', ' C', ' N', ' O', ' F', &
& 'Ne', 'Na', 'Mg', 'Al', 'Si', ' P', ' S', 'Cl', 'Ar', &
& ' K', 'Ca', 'Sc', 'Ti', ' V', 'Cr', 'Mn', 'Fe', 'Co', &
& 'Ni', 'Cu', 'Zn'/)


open(11,file=fpath//'generalinfo',status='old',err=100,iostat=ios)
open(12,file=fpath//'partit',status='old',err=101,iostat=ios)

eastat1=0
eastat2=0
nrbb=0
ncbb=0
nrdr=0

! read in general information
! ---------------------------

do i=1,nrec
   read(11,*)irec,kel(i),iel(i),idum,idum,idum,ilin(i),levdr(i),ldadr(i),ielev(i)
   indrec(kel(i),iel(i))=irec
enddo

! read maximum ionisation stage J_k and U(J_k+1)
! ----------------------------------------------

read(12,*) (jmax(i),i=1,natom)
read(12,*) (gstat(i,jmax(i)+1,1),i=1,natom)

close(11)
close(12)

if(maxval(jmax)+1.gt.nion) stop ' error in nion'

open(11,file=fpath//'atomgf3',status='old',err=103,iostat=ios)
open(12,file=fpath//'atomcol',status='old',err=104,iostat=ios)
open(13,file=fpath//'atomnl2_meta_v2.0',status='old',err=102,iostat=ios)
if(optdr) &
& open(14,file=fpath//'atomnldr',status='old',err=105,iostat=ios)


! read level information
! ----------------------


do i = 1,nrec
  k = kel(i)
  j = iel(i)
  read(13,*) n,idum,idum,level,xion(k,j)

  if((n.ne.i).or.(level.ne.ielev(i))) then
    write(*,*) 'error reading atomnl2_meta in subr. atomdat',& 
&     n,level,i,ielev(i)
    stop
  endif

  recsum = 0.

  do lev = 1,level
      read(13,*) jdum,ibfup(k,j,lev),ilow(k,j,lev),ergion(k,j,lev),gstat(k,j,lev),& 
&       metall(k,j,lev),ipop(k,j,lev),trans(k,j,lev),gfval(k,j,lev),alpha(i,lev), &
&       crodat(i,lev),beta(i,lev),s(i,lev)
! for highest ion, ibfup MUST be 1

! to create ionization-energy file
!      met=metall(k,j,lev)
!      if (met .eq. ' ') then
!        if (lev .eq. 1) then
!          met='g'
!        else
!          met='o'
!        endif
!      endif  
!      if(lev.le.5)  print*, &
!&       name(k),' ',j,' ',lev,' ',met,' ',1.d8/(xion(k,j)-ergion(k,j,lev))
!
      if(j.eq.jmax(k)) then
        if(ibfup(k,j,lev).ne.1) then
!          print*,' upper level (bf) reset to ground-state: ',k,' ',j,' ',lev
           ibfup(k,j,lev)=1
        endif  
      endif  

! used in SR partit (only for ground-state transitions and ipop ne 0 cases)
      if((metall(k,j,lev).eq.'s'.and.ilow(k,j,lev) .eq. 1) .or.  &
         (ipop(k,j,lev) .ne. 0)) &
&      recsum =   recsum + crodat(i,lev)
   enddo
   if(crodat(i,1) .ne. 1. ) stop' error in crodat(ground)' 
   zeta(k,j) = 1.D0/(recsum + 1.D0)


! read level transition information from file 'atomgf3'
   if(eastat1 .eq. 0) then
   read(11,*,end=150) nr,lines
   goto 160

150 eastat1=1
   goto 190

160 if(nr.ne.i) then
! for missing transition information, do nothing
! (currently element 23I ... 23III)
      if(ilin(i).ne.0) stop 'error in ilin'
        backspace(11)
   else
      if(lines .ne. ilin(i))then
        write(*,*) 'error in atomgf3:', nr,lines,ilin(nr)
        stop
      endif

      irbb(i)=nrbb+1
      do jj=1,ilin(nr)
       read(11,*) ind,wave,gf
       nrbb=nrbb+1
       if(nrbb.gt.35000) stop' nrbb > 35000!'
       npart = ind/10000
       nrest = ind - npart*10000
       low = nrest/100
       lup = nrest - 100*low
       met_rbb(nrbb)%low=low
       met_rbb(nrbb)%lup=lup
       met_rbb(nrbb)%wave=wave
       met_rbb(nrbb)%gf=gf
       met_rbb(nrbb)%index=0
      enddo
   endif
190 continue
   endif

! read level transition information from file 'atomcol'
   if(eastat2 .eq. 0) then
   read(12,*,end=200) nc,icol(i)
   goto 210
200 eastat2=1
   goto 250

210 if(nc.ne.i) then
      icol(i)=0
      backspace(12)
  else
      icbb(i)=ncbb+1
      do jj=1,icol(i)
       ncbb=ncbb+1
       if(ncbb.gt.1500) stop' ncbb > 1500!'
        read(12,*) low,lup,slope,omega
!       if(i.eq.18 .and. low.ne.1 .and. lup.ne.2) stop'error in cbb'
!       if(i.eq.18) then
!         omega=1.d12
!         slope=0.0001
!       endif  
       met_cbb(ncbb)%low=low
       met_cbb(ncbb)%lup=lup
       met_cbb(ncbb)%omega=omega
       met_cbb(ncbb)%slope=slope
      enddo
  endif
250 continue
  endif

! DR data
  if(.not.optdr) cycle
! read level transition information from file 'atomnldr'
  read(14,*,end=300) ndr, levdr1, ldadr1
! for gfortran
!  read(14,*,end=300,err=300) ndr, levdr1, ldadr1

  if(ndr.ne.i) then
      irdr(i)=0
      backspace(14)
  else
    if(levdr1.ne.levdr(i) .or. ldadr1.ne. ldadr(i)) &
&     stop' error in DR-data - levdr or ldadr'
     irdr(i)=nrdr+1
     do kk=1,levdr1
     read(14,*) ilowdr,icompdr
      do jj=1,icompdr
       nrdr=nrdr+1
       if(nrdr.gt.3500) stop' nrdr > 3500!'
       read(14,*) ener,flu
       met_rdr(nrdr)%low=ilowdr
       met_rdr(nrdr)%ncomp=icompdr
       met_rdr(nrdr)%index=0
       met_rdr(nrdr)%ener=ener
       met_rdr(nrdr)%flu=flu
      enddo
     enddo
  endif

300 continue

enddo

!testing DR-data (this is the way to read it)
!do i=1,nrec
!   if(levdr(i).eq.0) cycle
!   if(irdr(i).eq.0) stop
!   ndr=irdr(i)  
!   do kk=1,levdr(i)
!     icompdr=met_rdr(ndr)%ncomp
!     do jj=1,icompdr
!       print*,i,met_rdr(ndr)%low,met_rdr(ndr)%ener,met_rdr(ndr)%flu
!     ndr=ndr+1  
!     enddo
!     print*
!   enddo
!enddo   

! list of considered atom species
do i = 1,nrec
!
! for tests
!  print*,i,ilin(i),icol(i),irbb(i),icbb(i)
!  jj=irbb(i)
!  if(ilin(i).ne.0) &
!&   print*,met_rbb(jj)%low,met_rbb(jj)%lup,met_rbb(jj)%wave,met_rbb(jj)%gf
!  jj=icbb(i)
!  if(icol(i).ne.0) &
!&   print*,met_cbb(jj)%low,met_cbb(jj)%lup,met_cbb(jj)%omega,met_cbb(jj)%slope
!
   k = kel(i)
   if(abund(k) .ne. 0.) jatom(k) = k
enddo

if(nat.eq.1) then
! check whether NLTE atom is hydrogen
  if (labat(1).ne.'H ') stop ' nat = 1 and element ne hydrogen'
! to forbid approx. treatment of helium, uncomment following line
!  jatom(2) = 0
endif

close(11)
close(12)
close(13)
if (optdr) close(14)

return

100  write(*,*)' error with generalinfo, iostat=',ios
     stop
101  write(*,*)' error with partit, iostat=',ios
     stop
102  write(*,*)' error with atomnl2_meta, iostat=',ios
     stop
103  write(*,*)' error with atomgf3, iostat=',ios
     stop
104  write(*,*)' error with atomcol, iostat=',ios
     stop
105  write(*,*)' error with atomnldr, iostat=',ios
     stop
end
!
!----------------------------------------------------------------------
!
subroutine ionis(lteopt,ntemp,xne,dvdr,vr,clf)
!  
!----------------------------------------------------------------------
!     calculation of ionisation stratification in wind 
! NOTE:
! in case of xrays, we have tested (in indexfre) that all required
! entries in indexfrec are ne 0, avoiding the approx. Trad = Teff
! (in other words, the freq. grid extends to high enough frequencies)
!
!     WRITTEN: 
!        03/96, U.S.
!
!     CHANGES: 
!        06/96  J split into components J_i = W_i B(T_i)
!               for photoionization integrals
!        10/96  different J for ground and excited levels
!        01/97  SIMPLIFIED version for force multiplier calculation!
!        02/00 to f90 and consistency with nlte, JP
!
!        10/02  completetly changed; note that ZETA different from previous
!               definition
!
!        july/03 update for clumping: only line-transitions are affected, &
!                                     since ne already corrected
!
!        version 6.0: old approach preserved, see comment in routine partit
!----------------------------------------------------------------------
USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT
USE fund_const, ONLY: c2 => hkl, saha
USE nlte_var, ONLY: tradj, tradj1,xj,xxk
USE nlte_app, ONLY: nwne,nion,natom,indexfrec, &
&                   jatom, jmax, lwion, xion, &
&                   zeta, upart, fjk, &
&                   teff, te, wion, &
&                   summas, abund, xmuee, taus1, taus1new, deltatradj, &
&                   coll_therm, indrec, ielev, opametinf, indexopa, indexex,&
&                   ergion, alpha, metall, crodat, zcorr, &
&                   ilow, ipop, trans, occng, gstat, gfval, corr_beta, c_n

USE nlte_porvor, ONLY: fic,tcl_fac_line,tcl_fac_cont

implicit none

!typical resonance line with f=1 at 1000 A.
real(dp), parameter:: const = 0.02654 * 1. * 1000.d-8
real(dp), parameter:: c_vanreg= 0.5 * 2.055d-23

integer(i4b),intent(in) :: ntemp
logical,intent(in) :: lteopt
real(dp),dimension(nwne),intent(in) :: xne,dvdr,vr,clf

real(dp) :: tcl  

real(dp), dimension(0:nion) :: psi,prod

integer(i4b) :: j, js, k, l, it, n, ilev, lev, ind, inde, il
real(dp) :: ci, trad1, taus, ratlog, summ, uratio, xnk, &
& eps, epskj, beta, vanreg, tau, tau_mult, lam, eddf, corrb

real(dp) :: el,add1,add2
character*1 :: met

real(dp), dimension(natom,nion) :: lambdares

ci=0.5*saha

lambdares=0.

eps=1.
it=0

ittau: do while (eps.gt.1.d-3.and.it.le.20) 

it=it+1

! 1. calculation of partition functions
! -------------------------------------
!
call partit(lteopt,ntemp)


! 2. calculation of f_jk = N_jk/N_k
! ---------------------------------
radius: do js = ntemp,nwne

! psi(j) := log (N_j+1/N_j)

  atom:  do k = 1,natom
     if(jatom(k) .eq. 0) cycle atom

     do j = 1,jmax(k)
       if(lteopt.or.js.ge.lwion) then
         trad1=te(js)
       else
         if(indexfrec(k,j).eq.0) then
!          wien tail, Bnue(trad) approx W Bnue(Teff) => &
!          1/trad=1/teff -k ln W/(h nue)           
           trad1=1./teff-log(wion(js))/(c2*xion(k,j))
           trad1=1./trad1
         else  
         trad1=tradj1(js,indexfrec(k,j))
         endif
       endif  
       uratio = upart(k,j+1,js)/upart(k,j,js)

! OLD VERSION: J is locally approximated by W_ion*B(T_ion) 

!       ratlog = log(wne(js)) + .5*log(te(js)/trad1) + 1.5*log(trad1) &
!&        - log(ci) - c2*xion(k,j)/trad1
!

! NEW VERSION: J is locally approximated by B(T_RAD): improves convergence 
   
       ratlog = -log(xne(js)) + .5*log(te(js)/trad1) + 1.5*log(trad1) &
&        - log(ci) - c2*xion(k,j)/trad1
       
! note that zcorr = 1 for wion = 1 (allows for thermalization!) if
! all rad. temperatures are equal       

       if(lteopt.or.js.ge.lwion) then
         zcorr(k,j,js) =1.
! From V7 on: deltatradj might be different from 1 if lwion changes after
!             update of phot. struct. and optlte = .true.
!             Changed that always unity under LTE conditions
         deltatradj(k,j,js)=1.
       else  

! correction to allow for different rad-temp in ionizing continua
! and  res. lines
! performed only for NOT prespecified elements; all quantities have been calculated in
! subroutine partit         
         n=indrec(k,j)
         ilev=ielev(n)
         do lev = 2,ilev
           met=metall(k,j,lev)
           if(js.eq.ntemp) then          
               if(met.eq.'s'.and.lambdares(k,j).eq.0.) then
               el=ergion(k,j,lev)-ergion(k,j,ilow(k,j,lev))
! consistency  
               if(abs(el-trans(k,j,lev)).gt.el*1.d-5) stop' el ne trans'
! wavelength of 1st res. line
               lambdares(k,j)=1.d8/el
               cycle
             endif           
           endif
         enddo         
 
         zcorr(k,j,js)=zeta(k,j)*(1.d0+c_n(k,j,js))
         taus=taus1(k,j,js) 
!         if (k.eq.7) &
!&        print*,j,js,zcorr(k,j,js),zeta(k,j),c_n(k,j,js),taus

       endif  

! for deltatradj, see note from above 
       psi(j) = log(uratio)+log(zcorr(k,j,js)*deltatradj(k,j,js))+ratlog
!       if(k.eq.7) print*,j,psi(j),log(uratio), &
!&        log(zcorr(k,j,js)*deltatradj(k,j,js)),ratlog
       prod(j) = 0.
     enddo !j
            
     summ = 0.
     psi(0) = 0.
     prod(0) = 0.        
            
     do j = 0,jmax(k)
        do l = 0,j
          prod(j) = prod(j) + psi(l)
        enddo
        summ = summ + exp(prod(j))
     enddo

     do j = 1,jmax(k)+1
          fjk(k,j,js) = exp(prod(j-1)) / summ
!     if(k.eq.7) print*,j,js,upart(k,j,js),fjk(k,j,js)   
     enddo
     
  enddo atom
  
enddo radius

if(lteopt) exit ittau
!
! check for taus1
!
eps=0.

atom1: do k = 1,natom
    if(jatom(k).eq.0) cycle atom1

         xnk = const*abund(k)/summas

ions:    do j = 1,jmax(k)
         
! this quantity is affected by clumping, since all quantities
! calculated inside clumps (xmuee, xne), whereas taus involves spatial average
         taus1new(k,j,1:lwion-1) = xnk * xmuee(1:lwion-1) * xne(1:lwion-1) * &
&            fjk(k,j,1:lwion-1)/(dvdr(1:lwion-1) * clf(1:lwion-1)) !corrected 

         epskj=maxval(abs(1.-taus1new(k,j,1:lwion-1)/taus1(k,j,1:lwion-1))) 
         eps=max(eps,epskj) 
         enddo ions           ! ion loop
enddo atom1                    ! atom loop

taus1=taus1new

enddo ittau

if (it.eq.21) then

print*
print*,'WARNING: TAUSOB ITERATION FOR METALS NOT CONVERGED!!!'
PRINT* 

endif
!
!perform consistency checks (can be skipped after couple of tests)
! 
goto 50
do js = 1,lwion-1
   do k = 1,natom
     if (jatom(k).eq.0) cycle
     do j = 1,jmax(k)
         n=indrec(k,j)
         ilev=ielev(n)
         do lev = 2,ilev
           met=metall(k,j,lev)       
           if(met.eq.'s' .or. met.eq.'m') then 
              if(occng(n,lev,js).eq.0.) stop 'IONIS: gs/s/m level and occng=0'
            else 
              if(occng(n,lev,js).ne.0.) stop' IONIS: other level and occng ne 0'
           endif
         enddo
     enddo 
   enddo 
enddo              
! 
do js = lwion,nwne
   do k = 1,natom
     if (jatom(k).eq.0) cycle
     do j = 1,jmax(k)
         n=indrec(k,j)
         ilev=ielev(n)
         do lev = 2,ilev
              if(occng(n,lev,js).eq.0.) stop' IONIS occng=0'
         enddo
     enddo 
   enddo 
enddo              

50 continue
!
! calculate van Regemorter eps and coll_therm for next iteration
!
if(.not.lteopt) then
coll_therm=0.
corr_beta=1.

do js = 1,lwion-1
atom3: do k = 1,natom
     if (jatom(k).eq.0) cycle atom3
     do j = 1,jmax(k)

!  lambdares=0: only metastable levels considered, do not require coll_therm
       
         if(taus1(k,j,js).gt.1.D0.and.lambdares(k,j).ne.0.) then
         n=indrec(k,j)
         ilev=ielev(n)
       
         do lev = 2,ilev
           if(metall(k,j,lev).ne.'s') cycle   
           lam=1.d8/trans(k,j,lev)
! occup and gstat from lower level!
           il=ilow(k,j,lev)
! gstat(k,j,1)*fval =gfval(k,j,lev)
           if(il.eq.1) then  
             tau_mult=gfval(k,j,lev)*lam/1000./upart(k,j,js)
           else
! gstat(k,j,il)*fval =gfval(k,j,lev)
             if(metall(k,j,il).ne.'m') stop' inconsistency in lower level' 
             if(occng(n,il,js) .eq. 0.) stop ' occup = 0 for lower level'
             tau_mult=gfval(k,j,lev)*lam/1000.*occng(n,il,js)/upart(k,j,js)
           endif

!---- so far, the possibility of choosing the continuum opacity reduction
!     NOT included here (would require additional loop etc.)  
!     since taus1 is propto Sobo depth with v-grad in actual units:
!     --> tcl = taus1 * tau_mult * dvdr * tcl_fac_line
!     (since tcl_fac_line includes already sr/vmax/dvdr = 1/dvdr in actual units) 
           tcl=taus1(k,j,js)*tau_mult * dvdr(js) * tcl_fac_line(js)
! no additional correction for micro-clumping required, since taus1
! already corrected

           tau=taus1(k,j,js)*tau_mult * (1.+fic(js)*tcl)/(1.+tcl)
           
           add1=(3.*tau)/(1.+2.*vr(js)/dvdr(js))  ! at mue^2=1/3
           if(add1.lt.1.d-5) then
             beta=1.d0
           else
             beta=(1.-exp(-add1))/add1
           endif  

! assumption: collisions inside clumps, thus no correction
!             (beta depends on non-local quantities anyway)
           
           vanreg=(1.-exp(-1.4388d8/(lam*te(js)))) * c_vanreg * &
&            lam**3 * xne(js) / sqrt(te(js))
           coll_therm(n,lev,js)=vanreg/(vanreg+beta)
!           
           ind=indexex(n,lev)
           if(ind.ne.0) then
             eddf=xxk(js,ind)/xj(js,ind)
             if(eddf.gt.1.1.or.eddf.lt.0.1) then
               print*,lam,ind,js,eddf
               stop 'error in eddf'
             endif  
             add2=tau/(eddf+(1.-eddf)*vr(js)/dvdr(js)) ! at mue^2 = eddf
             if(add2.lt.1.d-5) then
                add2=1.d0
             else
                add2=(1.-exp(-add2))/add2 
             endif  
!             print*,k,j,lev,js,ind,lam,xxk(js,ind),xj(js,ind),eddf,add2,beta
!             print*,k,j,lev,js,ind,lam,taus1(k,j,js),tau,tau_mult,gfval(k,j,lev)
             corr_beta(n,lev,js)=min(add2/beta,1.d0)
           endif  
!          print*,js,k,j,lev,coll_therm(n,lev,js),beta,eddf,corr_beta(n,lev,js)
         enddo
         endif
     enddo
  enddo atom3
enddo
endif

end
!
!----------------------------------------------------------------------
!
subroutine partit(lteopt,ntemp)
!----------------------------------------------------------------------
!     calculation of modified partition function
!
!     since the partition function is a sum over all bound states,
!     the occupation numbers are calculated here as well
!
! NOTE:
! in case of xrays, we have tested (in indexfre) that all required
! entries in indexex and indexopa are ne 0, avoiding the approx. Trad = Teff
! (in other words, the freq. grid extends to high enough frequencies)
!
!     WRITTEN: 
!        09/96, U.S.
!
!     CHANGES: 
!        02/00 to f90 and consistency with nlte, JP.
!        note, that for w=1 ALL levels are now included in the partition
!        function, in order to allow for correct thermalization  
!       10/02  completely new
!
!     independent of clumping, since all values (beta!) already calculated
!     in ionis
!
!     note that this routine preserves the old approach of considering
!     ionizations to groundstates
!     (even if the upper bf state is an excited one). This will lead to
!     stronger deviations between the approximate and the almost exact method, &
!     compared to the results displayed in the FASTWIND paper.  
!
!---------------------------------------------------------------------- &
!  
USE nlte_type
USE fund_const, ONLY: c2 => hkl
USE nlte_var, ONLY: tradj,fre,ifre
USE nlte_app, ONLY: nwne, natom, jatom, nrec, jmax, lwion, &
& kel, iel, ielev, indexex, &
& occng, gstat, ergion, upart, metall, te, wion, teff, &
& coll_therm, corr_beta, &
& xion, indexopa, ilow, ipop, trans, opametinf, zeta, zeta_mj, crodat, &
& c_n, c_mj

implicit none
integer(i4b),intent(in) :: ntemp
logical,intent(in) :: lteopt


character*1 :: met
integer(i4b) :: i, j, js, k, lev, ind, il, ip, jstart, jj, kk
real(dp) :: et, trad1, t1, ti, tm, ground, e1, ei, em, el, deim, alphasum, add1, dummy
real(dp) :: integral,aux,xil,xip,tradmean,fac,weight

do js = ntemp,nwne
    do k = 1,natom
      upart(k,jmax(k)+1,js) = gstat(k,jmax(k)+1,1)
    enddo
enddo

occng=0. ! required

c_n=0.

rec: do i = 1,nrec
         k = kel(i)
         if(jatom(k) .eq. 0) cycle
         j = iel(i)
         ground=xion(k,j)
         e1=ergion(k,j,1)
         
! ground state contribution

         do js = ntemp,nwne
            upart(k,j,js) = gstat(k,j,1) 
         enddo

         if(ntemp.gt.lwion) stop 'ntemp > lwion'
         if(lwion.gt.nwne) stop ' something wrong with lwion'
         
         if (lteopt) then
           jstart=ntemp
         else
           jstart=lwion
         endif

! first loop
! lte case (for all depth points or only from lwion on)
         levlte: do lev = 2,ielev(i)

             
            met=metall(k,j,lev)
! excitation energy of considered levels
            el=ergion(k,j,lev)-e1

! "normal" levels 
            if(met .eq. 'm' .or. met .eq. 's') then
               do js = jstart,nwne
                  et = c2*el/te(js)
                  occng(i,lev,js) = exp(-et)
               enddo 

! for lowest part of atmosphere, include all remaining levels            
            else 
! for wind, set these levels explicitely to zero, in case lwion or
! ntemp has changed!
               do js=1,lwion-1
                  occng(i,lev,js)=0.
               enddo
               do js=lwion,nwne
                  et = c2*el/te(js)
                  occng(i,lev,js) = exp(-et)  ! wion=1 assumed
               enddo 
            endif 

         enddo levlte

         if(lteopt) goto 100   ! all occupation numbers done

! 2nd loop: 's' levels connected to ground states or ipop ne 0 levels (m and s)

         alphasum=0.

levground:  do lev = 2,ielev(i)

            met=metall(k,j,lev)
            el=ergion(k,j,lev)-e1

            il=ilow(k,j,lev)
            ip=ipop(k,j,lev)

            if(met .eq. 'm' .and. ip .ne. 0) then
! population over higher lying excited levels 
              alphasum=alphasum+crodat(i,lev)
              do js=ntemp,lwion-1   
                 if(indexex(i,lev).eq.0) then
                   trad1=teff
                 else  
                   trad1=tradj(js,indexex(i,lev))
                 endif  
!HERE: trans definitely different from "el"
                 et = c2*el/trad1
                 occng(i,lev,js) = exp(-et)  
              enddo
! normal 's' levels
            else if(met .eq. 's' .and. il .eq. 1) then
! consistency check 
              if(abs(el-trans(k,j,lev)).gt.el*1.d-5) stop' el ne trans'
              alphasum=alphasum+crodat(i,lev)
              do js=ntemp,lwion-1   
                 if (indexex(i,lev).eq.0) then
                   trad1=teff
                 else  
                   trad1=tradj(js,indexex(i,lev))
                 endif
                 et = c2*el/trad1
                 occng(i,lev,js) = &
&                  corr_beta(i,lev,js)*wion(js)*(1.-coll_therm(i,lev,js))*exp(-et)+ &
&                  coll_therm(i,lev,js)*exp(-et*trad1/te(js))
              enddo

! excitation from meta-stable state (ipop ne 0)
            else if(met .eq. 's' .and. ip .ne. 0) then

              if(il.ge.lev) stop 'il ge lev for s-levels'  
              deim=ergion(k,j,lev)-ergion(k,j,il)                  
! one more consistency check 
              if(abs(deim-trans(k,j,lev)).gt.deim*1.d-5) stop' deim ne trans'
              alphasum=alphasum+crodat(i,lev)
              do js=ntemp,lwion-1   
                 if (indexex(i,lev).eq.0) then
                   trad1=teff
                 else  
                   trad1=tradj(js,indexex(i,lev))
                 endif
                 et = c2*deim/trad1
! statistical weight of gm cancels out 
                 if(occng(i,il,js).eq.0.) & ! for tests only
&                  stop ' error in lower (meta-stable) state (ipop ne 0)!'
                 occng(i,lev,js) = occng(i,il,js)* &
&               (corr_beta(i,lev,js)*wion(js)*(1.-coll_therm(i,lev,js))*exp(-et)+ &
&                  coll_therm(i,lev,js)*exp(-et*trad1/te(js)))
              enddo

            endif 
         enddo levground
! consistency check
         alphasum=1./(1.+alphasum)
         if(abs(1.-alphasum/zeta(k,j)).gt.1.d-5) stop ' error in alphasum'


! 3rd loop: 's' levels connected to metastable states with ipop = 0, &
!           occng with respect to nm 
         
levmeta:  do lev = 2,ielev(i)

            met=metall(k,j,lev)

            ip=ipop(k,j,lev)
            il=ilow(k,j,lev)
! excitation from meta-stable state (ip eq 0)
            if(met .eq. 's' .and. il.ne.1 .and. ip .eq. 0) then

              if(il.ge.lev) stop 'il ge lev for s-levels'  
               deim=ergion(k,j,lev)-ergion(k,j,il)                  
! one more consistency check 
              if(abs(deim-trans(k,j,lev)).gt.deim*1.d-5) stop' deim ne trans'

              do js = ntemp,lwion-1
                 if (indexex(i,lev).eq.0) then
                   trad1=teff
                 else  
                   trad1=tradj(js,indexex(i,lev))
                 endif
                 et = c2*deim/trad1
! statistical weight of gm cancels out 
                 if(occng(i,il,js).ne.0.) stop ' error in lower (meta-stable) state(ipop =0)!'
! so far, only excitation
                 occng(i,lev,js) = &
&               (corr_beta(i,lev,js)*wion(js)*(1.-coll_therm(i,lev,js))*exp(-et)+ &
&                  coll_therm(i,lev,js)*exp(-et*trad1/te(js)))
              enddo
            endif
          enddo levmeta  

! 4th loop: sum up ni/n1*Rik/R1k and ni/nm*Rik/Rmk to obtain C_N, C_Mj and Zeta_Mj

          c_mj=0.
          zeta_mj=0.
levsum:   do lev = 2,ielev(i)

            met=metall(k,j,lev)
! excitation energy of considered levels
            el=ergion(k,j,lev)-e1

! ioniz. energy, modified by e1 for consistency
            ei=ground-el

            il=ilow(k,j,lev)
            ip=ipop(k,j,lev)

            if((met .eq. 's' .and. il .eq. 1) .or. ip .ne. 0) then
! connected to ground state (including ipop ne 0 states (m and s) 
            do js=ntemp,lwion-1         

             ind=indexopa(i,1)
             if(ind.ne.0) then
               t1=tradj(js,ind)
             else
               t1=teff
             endif 

             ind=indexopa(i,lev)
             if(ind.ne.0.and.t1.ne.teff) then
               ti=tradj(js,ind)
             else
               ti=teff
             endif 
! ni/n1*Rik/R1k
             if(occng(i,lev,js).eq.0.) stop ' error in s-state (c_n)!'
             add1=crodat(i,lev)*ti/t1*exp(c2*(ground/t1-ei/ti))*occng(i,lev,js)
             c_n(k,j,js)=c_n(k,j,js)+add1
            enddo
                          

          else if(met .eq. 's' .and. il .ne. 1 .and. ip .eq. 0) then
! connected to meta-stable-state
          
            do js=ntemp,lwion-1         

             ind=indexopa(i,1)
             if(ind.ne.0) then
               t1=tradj(js,ind)
             else
               t1=teff
             endif 

             ind=indexopa(i,lev)
             if(ind.ne.0.and.t1.ne.teff) then
               ti=tradj(js,ind)
             else
               ti=teff
             endif 

             ind=indexopa(i,il)
             if(ind.ne.0.and.t1.ne.teff) then
              tm=tradj(js,ind)
             else
              tm=teff
             endif 
! ni/nm*Rik/Rmk
             if(occng(i,lev,js).eq.0.) stop ' error in s-state (c_m)!'
             if(occng(i,il,js).ne.0.) stop ' error in m-state (c_m)!'

             em=ground-(ergion(k,j,il)-e1)
             add1=crodat(i,lev)/crodat(i,il)*ti/tm*exp(c2*(em/tm-ei/ti))*occng(i,lev,js)
             c_mj(il,js)=c_mj(il,js)+add1
            enddo
            zeta_mj(il)=zeta_mj(il)+crodat(i,lev)           
            endif

          enddo levsum
          
! now we can calculate nm/n1 in the 5th loop

levdone:   do lev = 2,ielev(i)

            met=metall(k,j,lev)
            ip=ipop(k,j,lev)
            il=ilow(k,j,lev)
            if(met .eq. 'm' .and. ip .eq. 0) then
             em=ground-(ergion(k,j,lev)-e1)

!            check for singular meta-stable levels (for tests only)
             if(zeta_mj(lev).eq.0.) then
                do jj=2,ielev(i)
                  if (ilow(k,j,jj).eq.lev) then
                    print*,k,j,jj,ilow(k,j,jj)
                    stop' error in singular metastable levels'
                  endif  
                enddo
             endif
! final result for zeta_mj 
             zeta_mj(lev)=1./(crodat(i,lev)+zeta_mj(lev))

             do js=ntemp,lwion-1         

              ind=indexopa(i,1)
              if(ind.ne.0) then
                t1=tradj(js,ind)
              else
                t1=teff
              endif 

              ind=indexopa(i,lev)
              if(ind.ne.0.and.t1.ne.teff) then
               tm=tradj(js,ind)
              else
               tm=teff
              endif 

              if(occng(i,lev,js).ne.0.) stop ' error in m-state (before done)!'
! avoid too strongly differing Trads for low-lying metastable levels if edge
! between ground-state and metastable levels; thus, "exact" integration
! cf subr. netmat_met

              if(ind.ne.0.and.ind.lt.ifre.and.t1.ne.teff) then
                kk=ind
                if(em.le.fre(ind+1)) then
                  xil=c2*em/tm
                  xip=xil*fre(ind+1)/em
                else
! may happen because of definition of em
                  xil=c2*fre(ind)/tm
                  xip=xil*fre(ind+1)/fre(ind)
                endif  
                integral=tm*(exp(-xil)-exp(-xip))
                if(integral.le.0.) stop' integral < 0 in partit'
                
                do kk=ind+1,indexopa(i,1)-1
                 tradmean=.5*(tradj(js,kk)+tradj(js,kk+1))
                 xil=c2*fre(kk)/tradmean
                 xip=xil*fre(kk+1)/fre(kk)
                 integral=integral+tradmean*(exp(-xil)-exp(-xip))
                enddo
                tm=tradj(js,kk)
                xil=c2*fre(kk)/tm
                integral=integral+tm*exp(-xil)
                aux=1.d0/integral
              else
                aux=exp(c2*em/tm)/tm
              endif
              if(aux.le.0.) then
                print*,js,i,lev
                stop' ioniz. integral negative in partit'
              endif  
!------------------------------
! this is the old version with hack. Can lead to oscillations (s50inv), &
! since correction is not analytic (abrupt change).

! now, aux corresponds to exp(-energie_m/kTrad)/Trad with average Trad
! JO Oct. 2014 -> should read exp(+energie ...
!              occng(i,lev,js)=aux*t1*exp(-c2*ground/t1)/crodat(i,lev)* &
!&               zeta(k,j)/zeta_mj(lev) *(1.+ c_n(k,j,js))/(1.+ c_mj(lev,js))
!
! here comes the new hack
!              if(occng(i,lev,js).gt.1.) then
! assume that c_mj is of the same order as c_n, and only small because of missing levels
!                dummy=aux*t1*exp(-c2*ground/t1)/crodat(i,lev)* &
!&                 zeta(k,j)/zeta_mj(lev)
!                occng(i,lev,js)=min(dummy,occng(i,lev,js))
! note: under certain conditions, n_m/n_1 can become quite large, leading
! to very large partition functions. An example is NiV where the meta-stable
! levels ionize to excited levels of NiVI.
! For X-ray conditions, the ground state has a higher Trad than the metastable
! levels, leading to quite high ratios (100) and partition functions (10^4).
!              endif  

! New version (from CMF-path v8.0, Jan 2015). Make a smooth transition between dummy*fac and dummy, &
! by using a smoothed heaviside function 
!------------------------------

! now, aux corresponds to exp(-energie_m/kTrad)/Trad with average Trad
              dummy=aux*t1*exp(-c2*ground/t1)/crodat(i,lev)* &
&               zeta(k,j)/zeta_mj(lev)
              fac=(1.+ c_n(k,j,js))/(1.+ c_mj(lev,js))
              if(fac.le.1.d0) then
! use standard version (above: min will be occng(i,lev,js)
                 occng(i,lev,js)=dummy*fac
              else
! new hack 
! smoothed heaviside function around x=dummy*fac=1., with smoothing
! parameter k=5.d0 (k=10.d0 already too steep). Argument is log!
! leads to a smooth transition from dummy*fac (weight=0, for low values
! of occng) to dummy (weight=1, for large values of occng) 
                 weight=1.d0/(1.d0+exp(-2.d0*5.d0*log10(dummy*fac)))
                 occng(i,lev,js)=dummy*weight+dummy*fac*(1.d0-weight)
              endif

! some check for isolated ground and metastable states
              if(c_n(k,j,js).eq.0. .and. c_mj(lev,js).eq.0.) then
                if(zeta(k,j).ne.1.) stop ' isolated states, zeta .ne. 1'
                if(abs(zeta_mj(lev)-1./crodat(i,lev)).gt.1.d-10) &
&                 stop ' isolated states, zeta_mj .ne. 1/crodat'
              endif
             enddo

! finally, we obtain the occup. numbers for those s-levels connected to m states
            else if(met .eq. 's' .and. il .ne. 1 .and. ip .eq. 0) then
             if(il.ge.lev) stop 'il ge lev for s-levels'  
! check also for underflow (highest ions in cool stars)
             do js=ntemp,lwion-1
! might happen in cool stars for high ions
               if(occng(i,il,js).eq.0.d0.and.teff.gt.9500.d0) &
&               stop ' error in m-state (done)!'
               occng(i,lev,js)=occng(i,il,js)*occng(i,lev,js)
             enddo
            endif 
          enddo levdone

!... and calculate the partition function for record i
100 part:     do lev = 2,ielev(i)
            do js = ntemp,nwne
             upart(k,j,js) = upart(k,j,js)+gstat(k,j,lev)*occng(i,lev,js)
            enddo
          enddo part

enddo rec

return
end
!
!----------------------------------------------------------------------
!
subroutine occup1(ntemp)
!----------------------------------------------------------------------
!     ground state occupation numbers
!     occ1 is n_1/n_e
!     thus indepedent of clumping  
!----------------------------------------------------------------------
USE nlte_type
USE nlte_app, ONLY: nwne, natom, jmax, &
&                   gstat, fjk, upart, occ1, &
&                   abund, xmuee, summas, jatom
implicit none
integer(i4b),intent(in) :: ntemp

integer(i4b) :: j, js,  k
real(dp) :: xjk, xnucl

! ground state occupation numbers
! -------------------------------
do k=1,natom
   if(jatom(k) .eq. 0) cycle 

   do j=1,jmax(k)+1
         do js = ntemp,nwne
            xnucl = abund(k)*xmuee(js)/summas
            xjk = fjk(k,j,js)
            occ1(k,j,js) = xjk*xnucl*gstat(k,j,1)/upart(k,j,js)
         enddo
   enddo                  ! ionization loop
enddo                     ! atom loop

return
end
!
!----------------------------------------------------------------------
!
subroutine occup2(ntemp)
!----------------------------------------------------------------------
!     occupation numbers of all levels from file 
!
!     OUTPUT:
!        occng(i,l,js): n_kjl/g_kjl/N_e
!        thus indepedent of clumping  
!
!
!     CHANGES: 
!        07/96 test for abundance conservation
!        09/96 simplified (already calculated in SR ionis)
!        02/00 to f90 and consistency with nlte, JP.
!----------------------------------------------------------------------
USE nlte_type
USE nlte_app, ONLY: nwne, natom, nrec, &
&                   jmax, kel, iel, ielev, indrec, &
& occng, occ1, gstat, abund, xmuee, summas, lwion, jatom

implicit none
integer(i4b),intent(in) :: ntemp

integer(i4b) :: i, irec, j, js, k, l, lev, jjs, jm
real(dp) :: occsum, xnk

! compute occupation numbers for all existing levels
do i = 1,nrec
   k = kel(i)
   if(jatom(k).eq.0) cycle
   j = iel(i)
         
   do lev = 1,ielev(i)

! ground states

      if(lev .eq. 1)then
         do js = ntemp,nwne
           occng(i,lev,js) = occ1(k,j,js)/gstat(k,j,1)
         enddo

! others, as determined in SR partit
      else
         do js = ntemp,nwne
           occng(i,lev,js) = occng(i,1,js)*occng(i,lev,js)
         enddo
      endif
   enddo                  ! level loop
enddo                     ! record loop

! test for abundance conservation
atom: do k = 1,natom
    if(jatom(k).eq.0) cycle atom

       do js = ntemp,nwne
! local number of particles in unites of ne(js) -- checked!
          xnk = abund(k)*xmuee(js)/summas
          occsum = 0.

          jm=jmax(k)
          do j = 1,jm
               irec = indrec(k,j)
               do l = 1,ielev(irec)
                  occsum = occsum + gstat(k,j,l)*occng(irec,l,js)
               enddo
          enddo
          j=jm+1
          occsum=occsum+occ1(k,j,js) ! last ion, only ground-state, occ1 without gstat    

          if(abs(xnk/occsum - 1.).ge.1.d-6) then
             write(*,*) ' abundance not conserved in occup2'
             write(*,*) ' k,js, xnk/occsum -1: ',k,js,xnk/occsum-1.
             print*,lwion,ntemp
             stop
          endif

       enddo                  ! radius loop
enddo atom                    ! atom loop

return
end
!
!----------------------------------------------------------------------
!
      subroutine ionout(lteopt)
!----------------------------------------------------------------------
!     output of ionisation structure as calculated in sr ionis
!
!     WRITTEN:
!        04/96, U.S
!     CHANGES: 
!        02/00 to f90 and consistency with nlte, JP.
!----------------------------------------------------------------------
USE nlte_type
USE nlte_var, ONLY: modnam
USE nlte_app, ONLY: nwne, natom, nion, &
&                   imax,indexel,jatom, jmax, abund, fjk, upart, name, &
&                   xhy, xhe, wne, wion, te, ndebug


implicit none

logical,intent(in) :: lteopt

integer(i4b) :: i, j, js, k

real(dp) :: xmax, xold, ak

! output of modified partition functions

if(ndebug.ge.3) then
    write(*,'(/,a)')' MOD. PARTITION FUNCTION'
    do js = 1,nwne
      if(js.ne.1.and.js.ne.nwne.and.mod(js,6).ne.0.and.wion(js).ne.0.5&
     .and.wion(js).ne.0.75) cycle
        write(*,'(/,a,i3,/)')' RADIUS SHELL = ',js
          do k=1,natom
             if(jatom(k).ne.0) then
                  write(*,10) k,(upart(k,j,js),j=1,jmax(k)+1)
             endif
          enddo
    enddo
endif

10   format(i4,2x,9(f8.1,1x))

! output of maximum ionisation stage

if(ndebug.ge.1) then

  write(*,'(//,a)') ' OUTPUT OF IONISATION STRUCTURE'
  write(*,*)         '=============================='
  write(*,'(/,a)') ' main ionisation stage of elements'
  write(*,*)        '---------------------------------'
  write(*,20)'#',(name(i),i=1,natom)
  write(*,21)

  do js = 1,nwne
!    if(js.ne.1.and.js.ne.nwne.and.mod(js,6).ne.0.and.wion(js).ne.0.5&
!     .and.wion(js).ne.0.75) cycle
     atom: do k = 1,natom
       imax(k) = 0
       xold = 0.
       if(jatom(k).eq.0)then
          imax(k) = 0
          cycle atom
       endif

       do j = 1,jmax(k)+1
          xmax=max(xold,fjk(k,j,js))
          if(xmax.gt.xold) imax(k)=j
          xold=xmax
       enddo
     enddo atom
     write(*,22) js,(imax(k),k=1,natom)
  enddo
endif

20   format(/,a4,30(1x,a2))
21   format(94('-'))
22   format(i4,30(1x,i2))

! output of main species (HI or HeII)-Ionisation structure

if(ndebug.ge.2) then
  if(xhy.gt.xhe) then
       write(*,'(/,a)') ' RUN OF H IONISATION: N_j/N_H'
  else
       write(*,'(/,a)') ' RUN OF He IONISATION: N_j/N_He'
  endif
  write(*,25)
25      format(49('-'))

  do js = 1,nwne
    if(js.ne.1.and.js.ne.nwne.and.mod(js,6).ne.0.and.wion(js).ne.0.5&
     .and.wion(js).ne.0.75) cycle
    if(xhy.gt.xhe) then
        write(*,30)js,(fjk(1,j,js),j=1,2)
    else
        write(*,30)js,(fjk(2,j,js),j=1,3)
    endif
  enddo
endif

30   format(i4,2x,3(g10.3,2x),f7.4)

! output of ionisation structure of all elements

if(ndebug.ge.2) then
! new from V6.1 on
   if (lteopt) then
   open(87,file = trim(modnam)//'/METAL_IDL_LTE',status='unknown')
   open(88,file = trim(modnam)//'/METAL_IONS_LTE',status='unknown')
   else
   open(87,file = trim(modnam)//'/METAL_IDL',status='unknown')
   open(88,file = trim(modnam)//'/METAL_IONS',status='unknown')
   endif

   write(88,'(a)') '# (f_jk=N_jk/N_k)'
!   write(*,'(a)') '# (f_jk=N_jk/N_k)'
     
   do js=1,nwne

     do k=1,natom
          write(87,34) k,(fjk(k,j,js),j=1,nion)
     enddo

     if(js.ne.1.and.js.ne.nwne.and.mod(js,6).ne.0.and.wion(js).ne.0.5&
     .and.wion(js).ne.0.75) cycle

       write(88,33) js,wne(js),te(js),wion(js)
!       write(*,33) js,wne(js),te(js),wion(js)
       do k=1,natom
            if(abund(k).ne.0.) then
                write(88,34) k,(fjk(k,j,js),j=1,nion)
!                write(*,34) k,(fjk(k,j,js),j=1,nion)
            endif
       enddo
   enddo
   print*     
   print* 
!
! output of ionisation structure of NLTE-elements with respect to N_H
!
   write(88,'(a)') '# (N_jk/N_H)'
!   write(*,'(a)') '# (N_jk/N_H)'
     
   do js = 1,nwne
     if(js.ne.1.and.js.ne.nwne.and.mod(js,6).ne.0.and.wion(js).ne.0.5&
     .and.wion(js).ne.0.75) cycle
       write(88,33) js,wne(js),te(js),wion(js)
!       write(*,33) js,wne(js),te(js),wion(js)
       do k=1,natom
            if(indexel(k).gt.0) then
                ak=abund(k)/abund(1)
                write(88,35) k,(fjk(k,j,js)*ak,j=1,nion)
!                write(*,35) k,(fjk(k,j,js)*ak,j=1,nion)
            endif
       enddo
   enddo
   close(87)     
   close(88)
endif
return

33   format(/,'# i = ',i4,' wne = ',e10.4,' T_e = ',f7.0,' dilfac = ',e10.4)
34   format(i4,2x,1p,9(g10.3,1x))
35   format(i4,2x,9(e8.2,1x))

end
!
!----------------------------------------------------------------------
!
subroutine indexfre(lteopt)
!  
!calculates indexfile for metal opacities (start-freq.), cross-sections
!for most important bf transitions and index-files for radiation temperatures
!used in NLTE  
!
!specific caution has to be taken for elements simultaneously present
!in NLTE and approximate treatment, since ionization edges may be
!defined at (slightly) different frequencies
!
! 8/07/02 small changes to discard only those ionization stages of
! preespecified atoms that are not included  (m. urbaneja)
!
! 04/2006: allows for bf transitions to excited levels
!
! 10/2014
! in case of xrays, we test that all required entries in
! indexopa, indexfrec and indexex are ne 0, i.e., contain an index
! (in other words, the freq. grid extends to high enough frequencies)
!
USE nlte_type
USE nlte_var, ONLY: fre,ifre
USE nlte_app, ONLY: natom,nout,ndebug, &
&                   indexel,indexex,indexfrec,indexopa,indrec, &
&                   jatom,jatom_full,jmax,xion, &
&                   nrec,kel,iel,ielev,cutoff,ergion,metall, &
&                   opametinf,alphamet,alpha,beta,s, &
&                   indexelion, & ! maup, indexelion
&                   ibfup,ilow,ipop,trans
USE princesa_var, ONLY: index4, labl4, le, nions, frecin, zl, &
&                       nat, zeff 

USE nlte_xrays, ONLY: optxray

implicit none
logical, intent(in) :: lteopt

integer(i4b) :: k,knlte,kl,i,ilo,iz,j,lev,n,nionnlte,niontest,number,iup
real(dp) :: a,diffe,diffe1,edge,energy,ground,new_energy,frec,x, &
&           alphai,betai,si


if(.not.lteopt) then
! changed Dec. 11/2012: put as first block because in NLTE case
! xion is changed, and this has an effect on indexopa
! otherwise, restart does not work correctly

! find ground state transition edges of NLTE ions
! thus far (ver 6.0), we do not modifiy this loop 
niontest=0
atomloop: do k=1,natom
   if (jatom(k).eq.0) cycle atomloop
   knlte=indexel(k)
   if (knlte.le.0) cycle atomloop
   ionloop: do j=1,jmax(k)
      ground=ergion(k,j,1)
      energy=xion(k,j)-ground 
      diffe=energy 
      new_energy=0.
      tranloop: do i=1,index4
          a=frecin(i)
          ilo=labl4(i)
          iz=zl(ilo)
          kl=le(ilo)
!
!check only for ions k,j
!
          if(iz+1.ne.j.or.kl.ne.knlte) cycle tranloop
          diffe1=abs(a-energy)
          if(diffe1.lt.diffe) then
            diffe=diffe1
            new_energy=a
          endif
      enddo tranloop 
      if(new_energy.eq.0.) then
        if(nout.eq.1.and.ndebug.ge.3) write(*,10) k,j
      else  
        niontest=niontest+1
        if(nout.eq.1.and.ndebug.ge.3) &
&         write(*,20) k,j,1.d8/energy,1.d8/new_energy
        if (abs(1.-energy/new_energy).gt.0.01) &
&        stop ' something wrong with ionization energies' 
        xion(k,j) = new_energy+ground
      endif
   enddo ionloop
enddo atomloop    

nionnlte=sum(nions)-nat
if(niontest.ne.nionnlte) stop ' not all NLTE-ions found in indexfre'
if(nout.eq.1.and.ndebug.ge.3) print*

endif

! set up the indexopa-array (starting point for opacities) for ALL edges
! and provide info which opacities shall be used in opacitm.
! the corresponding index file "opametinf" includes
!  i) all levels for the "selected" atoms, &
! ii) the ground, 's' and 'm' levels down to cutoff fraction of
!     ionization energy for non-selected atoms and  
!iii) discards all explicit IONS (present in NLTE)

indexopa=0 
opametinf=0

indexelion=0 ! maup, seleccion on preespecified NLTE ions 
rec0: do n = 1,nrec
         k = kel(n)
         if (indexel(k) .le. 0) cycle rec0
         j = indexel(k)
         do lev = 1, nions(j) - 1
            number = lev + INT(zeff(j))	    
            indexelion(k,number) = 1
         enddo
enddo rec0

! at first, set up information file which levels are used
number=0
rec1: do n = 1,nrec
       k = kel(n)
       if (jatom(k).eq.0) cycle rec1
!       if (indexel(k).gt.0) cycle rec1
       j = iel(n)
       if ( (indexelion(k,j).eq.1) .and. (indexel(k).gt.0)) cycle rec1 
       ground=xion(k,j)
level1: do lev = 1,ielev(n)
          energy=ground-ergion(k,j,lev) 
          if (jatom_full(k).ne.1.and.lev.ne.1.and.metall(k,j,lev).eq.' ') cycle level1
          if ((1.-energy/ground).gt.cutoff) cycle level1

! to avoid difficulties in convergence (excited levels in same cont)

! with new versions, this seems to be no longer necessary

! allows also for HeI and HeII for approx calculation if database
! ATOMDAT is used          

! check for cool stars (carbon has groundstate and 2nd level in Balmer-cont
! should be included to allow for depopulation
!          print*,k,j,lev,ground,energy
!          if     (ground.gt.436881.) then    !shortward HeII
!            if (lev.ne.1.and.energy.gt.436881.) cycle level1 
!          else if(ground.gt.198412.) then    !shortward HeI
!            if (lev.ne.1.and.energy.gt.198412.) cycle level1
!          else if(ground.gt.109649.) then    !shortward H-Ly
!            if (lev.ne.1.and.energy.gt.109649.) cycle level1
!          else if(ground.gt. 27419.) then    !shortward H-Ba
!            if (lev.ne.1.and.energy.gt. 27419.) cycle level1
!          else
!            if (lev.ne.1) cycle level1       ! H-Pa continuum
!          endif
          
          number=number+1          
          opametinf(n,lev)=number
       enddo level1
enddo rec1

if(nout.eq.1.and.ndebug.ge.3)then
  print*
  print*,number,' EDGES LOCATED FOR METAL OPACITIES'
  print*
endif

!now setup "edge" information and
!calculate cross sections for ALL considered levels and frequencies 

allocate(alphamet(number,ifre))

rec2: do n = 1,nrec
    k = kel(n)
    if(jatom(k).eq.0) cycle rec2
    j = iel(n)
    ground=xion(k,j)

!  ionization to ground states (for non-selected elements)
!  AND excited states (for selected elements)

level2: do lev = 1,ielev(n)
!       number=opametinf(n,lev)     
!       if(number.eq.0) cycle level2
       edge=ground-ergion(k,j,lev) 
       iup=ibfup(k,j,lev)       
       if (jatom_full(k).eq.1 .and. iup.ne.1) edge=edge+ergion(k,j+1,iup)
!
          frecloop2: do i=2,ifre
          frec=fre(i)
          if(edge.le.frec) then
            if(indexopa(n,lev).eq.0) then
              indexopa(n,lev)=i 
!  cross-sections required only if included in opacities
              number=opametinf(n,lev)
!              if(number.ne.0) print*,k,j,lev,edge,frec,1.d8/frec,i,number
              if(number.eq.0) cycle level2
            endif  
            x=edge/frec             
            alphai=alpha(n,lev)*1.d-18
            betai=beta(n,lev)
            si=s(n,lev)
            alphamet(number,i)=alphai*(betai*x**si+(1.-betai)*x**(si+1))
            if(alphamet(number,i).le.0.d0) then
                if (betai.ge.0.) stop' ALPHA NEGATIVE AND BETA POSITIV (nlte_approx)'
                alphamet(number,i)=1.d-40
            endif
          endif  
      enddo frecloop2    
    enddo level2
enddo rec2    

!tests in case of xrays
if(optxray) then
do n = 1,nrec
    k = kel(n)
    if(jatom(k).eq.0) cycle

    do lev = 1,ielev(n)
    if(indexopa(n,lev).eq.0) stop' indexopa = 0'
    enddo
enddo
endif

if(lteopt) return

indexfrec=0
indexex=0

! now we can set up the indexfrec-array (ground-state ionization only)
!
! in former versions, this seemed to be superfluous, since indexfrec(k,j)
! should be indexopa(n,1)
! from version 6.0 on, this is no longer necessarily true, due to the
! possibility of ionizations to excited levels

niontest=0
atomloop1: do k=1,natom
   if (jatom(k).eq.0) cycle atomloop1
   ionloop1: do j=1,jmax(k)
      energy=xion(k,j)-ergion(k,j,1)
!
! assumes that energies are orderer from low to high      
!   
      frecloop: do i=2,ifre
         frec=fre(i)
         if(energy.eq.frec) then 
            niontest=niontest+1
            indexfrec(k,j)=i
            if(nout.eq.1.and.ndebug.ge.3) print*,k,j,'EXACTLY FOUND'
            cycle ionloop1
         else if(energy.lt.frec) then
              indexfrec(k,j)=i
              n=indrec(k,j)
              if(i.ne.indexopa(n,1) .and. ndebug.eq.3) then
	           print*,' no ground-state ionization'
	           print *,'index ',n,' frequency ',frec,i
		   print *,indexopa(n,1),' atom ',k,' ion ',j
	      endif	   
            cycle ionloop1
         endif
      enddo frecloop    

!tests in case of xrays
      if(optxray) stop' indexfrec = 0'
      if (indexfrec(k,j).ne.0) stop ' not found 1'
      
   enddo ionloop1
enddo atomloop1    

if(nout.eq.1.and.ndebug.ge.3) then
   print*
   print*,niontest,' NLTE-EDGES EXACTLY FOUND IN FREQ. GRID'
   print*
endif

! now we can set up the indexex-array (radiation temperatures for excitation)

rec: do n = 1,nrec
       k = kel(n)
       if(jatom(k).eq.0) cycle rec
       j = iel(n)
       ground=ergion(k,j,1)
level: do lev = 2,ielev(n)
          if(metall(k,j,lev).eq.'s') then
             if(ilow(k,j,lev).eq.1) then
               energy=ergion(k,j,lev)-ground
             else  
               energy=ergion(k,j,lev)-ergion(k,j,ilow(k,j,lev))
             endif
          else if(metall(k,j,lev).eq.'m'.and.ipop(k,j,lev).ne.0) then
              energy=ergion(k,j,ipop(k,j,lev))-ground   
          else
              cycle level
          endif
!consistency check
          if(abs(energy-trans(k,j,lev)).gt.energy*1.d-5) then
            print*,k,j,lev,energy,trans(k,j,lev)
            stop' energy ne trans'
          endif  
          frecloop1: do i=2,ifre
          frec=fre(i)
          if(energy.le.frec) then
            diffe =abs(energy-frec)
            diffe1=abs(energy-fre(i-1))
            if (diffe.le.diffe1) then
              indexex(n,lev)=i
            else
              indexex(n,lev)=i-1
            endif  
            cycle level
          endif
          enddo frecloop1    

!tests in case of xrays
          if(optxray) stop' indexex = 0'
          if(indexex(n,lev).ne.0) stop ' not found 2'
    enddo level
enddo rec    

return

10 format(' ION(',i2,',',i2,') not in DETAIL')
20 format(' ION(',i2,',',i2,') WITH OLD EDGE ',f12.6,' CHANGED TO ',f12.6)

end
!
!----------------------------------------------------------------------
!
subroutine opacitm(nd,xne,temp,clf,lteopt,ntemp,opteta,frenew,it0)

! calculates opacities (and emissivities for opteta) for
! all levels and frequencies specified in routine indexfre, i.e., &
! for the most important metallic levels excluding those elements &
! treated in NLTE  
!
! clumping included
! philosophy (as in opacitc): scale all opacities by CLF^(-1)
! leave OPAFF (only needed for energy balance)

USE nlte_type  
USE nlte_dim, ONLY: ID_NDEPT,ID_FREC1
USE fund_const

USE nlte_opt, ONLY: optstark

USE nlte_var, ONLY: fre,ifre,precis,opat_m_new,opat_m_nolines, &
&                   strue_m_new,thomson_lines,optlines,xnelte, &
&                   gffnlte, ionmax, optneupdate,lwion1,lam_ly

USE nlte_app, ONLY: natom, nion, nrec, lwion, kel, iel, ielev, ibfup, &
&                   indexel, indexopa, jatom, jmax, opametinf, &
&                   alphamet,gstat,occ1,occ1lte,occng,occnglte, &
&                   abund,fjk,fracmin, xmuee, summas, &
&                   jatom_full,met_imin,met_imax,indrec

USE tcorr_var, ONLY: ffopa_m,dtffopa_m

USE nlte_lines, ONLY: wavblue,wavred,wavcon,nlam, &
& opalgrid,opacgrid,fracth,fracnth,nlambox_ly, &
& opa_eff_grid,calcopal 

USE nlte_porvor, ONLY: epsi,epsi1,optthick, &
  fic,tcl_fac_line,tcl_fac_cont,opa_eff_rat


implicit none
!
integer(i4b), intent(in) :: nd,ntemp
real(dp), dimension(nd), intent(in) :: xne, temp, clf
logical, intent(in) :: lteopt,opteta,frenew,it0

integer(i4b) :: iz,k,kk,j,lev,ll,n,number,index,nlambox,iup,nup,lw
real(dp) :: bn,bn1,depart,etab,freq,gf,gff,gs, &
&           opab,xni,xnistar,xmuene, xnk,z, &
&           opaff,opaff1,opal,slinel,chiff,chiffc,etaffc,bnue,xlam,opacg, &
&           opabfaux,x,expo

real(dp), dimension(ID_NDEPT) :: xnerat
real(dp), dimension(ID_FREC1) :: hcfre3
real(dp), dimension(ID_FREC1,ID_NDEPT) :: ehnuekt, opabf, etabf

real(dp) :: tcl,fac2

external gff,bnue

ffopa_m=0.
dtffopa_m=0.


if (lteopt.or.it0) then  
do ll = ntemp,nd  
   if (xne(ll).ne.xnelte(ll)) stop ' error in ne -- lteopt'  
   xnerat(ll) = 1.d0  
enddo
else  
do ll = ntemp,nd  
   if (optneupdate.and.xne(ll).eq.xnelte(ll)) stop ' error in ne -- nlteopt'  
   xnerat(ll) = xne(ll)/xnelte(ll)  
enddo
end if  

if(frenew) then ! prepare ff gaunt-factors for metal ions

 if(nion.ne.ionmax) stop ' nion incompatible in nlte and nlte_approx!'

  do ll=ntemp,nd
    do kk=1,ifre
      freq=fre(kk)
      do iz=1,nion
          z=float(iz)
          gffnlte(iz,ll,kk) = gff(freq*clight,temp(ll),z,ll,kk,iz)*z*z  
      enddo        
    enddo
  enddo    
endif

do kk=1,ifre
  freq = fre(kk)  
  do ll=ntemp,nd
     ehnuekt(kk,ll) = exp(-hkl*freq/temp(ll))  
   enddo
enddo

opabf=0.d0
if (opteta) then
  hcfre3(1:ifre) = hc2*fre(1:ifre)**3  
  etabf=0.d0
endif  

! no correction in this block
rec: do n = 1,nrec

level: do lev = 1,ielev(n)
   number=opametinf(n,lev)
   if(number.eq.0) cycle level
   index=indexopa(n,lev) 
   if(index.eq.0) cycle level !eq.0 for edges larger than max(fre)
   
   k = kel(n)
   j = iel(n)
!  for tests
!   print*,'opacitm',k,j,lev
   gs=gstat(k,j,lev) 
   iup=ibfup(k,j,lev)
   if(jatom_full(k).ne.1) iup=1    
   
   do kk=index,ifre
     if(alphamet(number,kk).eq.0.d0) stop ' error in alphamet'
   enddo

depth: do ll=ntemp,nd
   if(abund(k)*fjk(k,j,ll).lt.fracmin) cycle depth
   if(.not.lteopt.and.jatom_full(k).eq.1..and.ll.lt.lwion1) then !here was the bug
     if(met_imin(k,ll).eq.0 .or. met_imax(k,ll).eq.0) stop' error in met_imin/imax'
     if(j.lt.met_imin(k,ll).or.j.gt.met_imax(k,ll)) cycle depth
   endif  
   xni=occng(n,lev,ll)*gs
   if (iup.eq.1 .or. &
&      (.not.lteopt .and. jatom_full(k).eq.1. .and. ll.lt.lwion1 &
       .and. j.eq.met_imax(k,ll))) then
! also for ionization to excited levels if j = imax
! see subr. opacitc (nlte.f90) and notes
     xnk=occ1(k,j+1,ll)   ! includes stat. weight
     if (occnglte(n,lev,ll) .eq. 0.) then
       stop 'inconsistency in number'
     endif  
     xnistar = xnk*occnglte(n,lev,ll)*gs/occ1lte(k,j+1,ll)*xnerat(ll)  
   else
     nup=indrec(k,j+1)
     xnk=occng(nup,iup,ll)*gstat(k,j+1,iup)
     if(xnk .eq. 0.) stop' xnk = 0 in opacitm'     
     xnistar = xnk*occnglte(n,lev,ll)*gs/ &
&      (occnglte(nup,iup,ll)*gstat(k,j+1,iup))*xnerat(ll)  
   endif  

   depart = xni/xnistar  

!   if(ll.eq.nd) print*,k,j,lev,ll,depart 
   lw=lwion
   if(jatom_full(k).eq.1) lw=lwion1
   if ((lteopt.or.ll.ge.lw) .and. abs(depart-1.d0).gt.precis) then
       print*,lteopt,k,j,lev,ll,depart,precis 
       stop ' error in depart - lte'
   endif

   opabf(index:ifre,ll)=opabf(index:ifre,ll)+ &
&    xnistar*alphamet(number,index:ifre)*(depart-ehnuekt(index:ifre,ll))

!   if(ll.eq.15.and.alphamet(number,658).ne.0.) &
!&    print*,'658',k,j,lev,depart,xnistar*alphamet(number,658)
!   if(ll.eq.15.and.alphamet(number,659).ne.0.) &
!&    print*,'659',k,j,lev,depart,xnistar*alphamet(number,659)

! for tests of groundstate
!   if(lev.eq.1) then
!     opabfaux=xnistar*alphamet(number,index)*(depart-ehnuekt(index,ll))
!     print*,ll,k,j,index,depart,opabfaux/clf(ll)
!   endif
   if (opteta) etabf(index:ifre,ll)=etabf(index:ifre,ll)+ &
&    xnistar*alphamet(number,index:ifre)  

   enddo depth

enddo level
enddo rec    
!
!-------- now we have calculated bf opacity for all transitions and depths
!-------- we continue with ff opacity and summ up everything, 
!-------- including clumping correction
!
do ll=ntemp,nd

  xmuene=xmuee(ll)*xne(ll)/summas !(absolute number, including clf)

  do kk=1,ifre


     freq=fre(kk)
     opaff = 0.d0  
     chiff = 1.3695d-23*xne(ll)/ (freq**3*sqrt(temp(ll)))  
     chiffc = chiff* (1.d0-ehnuekt(kk,ll))  
     etaffc = chiff  

     atom: do k = 1,natom
        if(jatom(k).eq.0) cycle atom
        if (indexel(k).gt.0) cycle atom ! included in nlte

        ion: do j=2,jmax(k)
          xnk=abund(k)*fjk(k,j,ll)
          if(xnk.lt.fracmin) cycle ion  
          xnk=xnk*xmuene  !total occupation of ion
          iz=j-1
          gf = gffnlte(iz,ll,kk)
          if(gf.le.0.) stop ' error in ff gaunt-factors'   
          opaff = opaff + gf*xnk  
        enddo ion

        j=jmax(k)+1  ! highest stage
        xnk=abund(k)*fjk(k,j,ll)
        if(xnk.ge.fracmin) then  

! calculate total occupation of ion to check consistency
          xnk=xnk*xmuene
          if(abs(1.-xnk/occ1(k,j,ll)).gt.precis) then
            print*,lteopt,frenew
            print*,abund(k),fjk(k,j,ll),xmuene
            print*,k,j,ll,xnk,occ1(k,j,ll)
            stop ' error in occ.num of highest ion'
            endif
          iz=j-1
          gf = gffnlte(iz,ll,kk)
          if(gf.le.0.)stop ' error in ff gaunt-factors'   
          opaff = opaff + gf*xnk  
        endif

     enddo atom                    ! atom loop

     opaff1 = opaff*chiffc  
     ffopa_m(ll,kk) = opaff * chiff !tcorr, without clf correction
     dtffopa_m(ll,kk) = (-1.d0)*ffopa_m(ll,kk)/(2.d0*temp(ll))

     opab   = (opabf(kk,ll) + opaff1)/clf(ll) ! now corrected

! include line-opacities

     
     nlambox=0
     opal=0.
     if(optlines) then
     xlam=1.d8/freq
        if(xlam.gt.wavblue.and.xlam.lt.wavred) then
          nlambox = 1 + log(xlam/wavblue)/wavcon

          if(nlambox.ge.nlam) stop ' error in nwavbox'
          if(optstark) then
! approximate opacities redwards from jump by Stark-broadened data
! not affected by changes in V7.2.3
            if(nlambox.eq.nlambox_ly-1 .and. xlam.ge.lam_ly) nlambox=nlambox_ly
          endif
          
          opal=opalgrid(nlambox,ll) ! already corrected (from OPACITL)
          opacg=opacgrid(nlambox,ll) !should be corrected (check)!!!  
! REMEMBER -- opalgrid uses MEAN quantities 
          opa_eff_rat(ll,kk) = opa_eff_grid(nlambox,ll)
       endif
     else 
     endif

!     if(.not.optthick.and.abs(opa_eff_rat(ll,kk)-1.d0).gt.epsi) then
     if(.not.optthick.and.abs(opa_eff_rat(ll,kk)-1.d0).gt.1.d-3) then
       print*,ll,kk,opa_eff_rat(ll,kk)-1.d0
!       print*,abs(opa_eff_rat(ll,kk)-1.d0),epsi
!       stop' optically thin clumping and opa_eff_rat ne 1 in opacitm'
     endif
! this is a good test, since also in thin clumping opa_eff_grid is
! calculated in sumopal. Test proves appropriate averaging
     
! always
     opat_m_nolines(ll,kk) = opab
     opab=opab+opal   
     opat_m_new(ll,kk) = opab   

! here is where opa_eff_rat = opa_eff/<opa_tot>  
! is checked and finally transferred to nlte.f90
! NOTE -- In case we're OUTSIDE the range wavblue,wavred,
! opa_eff_rat is calculated in opacitc. 
     if (opa_eff_rat(ll,kk).gt.epsi1.or.opa_eff_rat(ll,kk).le.0.d0) then 
           print*,ll,kk,opa_eff_rat(ll,kk),xlam
           print*,opat_m_new(ll,kk),opal,opacg 
           print*,nlambox,opab,opa_eff_grid(nlambox,ll)
           print*,calcopal(nlambox)
           stop'opa_eff_rat(ll,kk) out of bounds, opacitm_1'
     endif
! now effective opacity transfered
! as discussed, we DON'T do anything with emissivities and opacities here,
! since always mean values required (except for opac and related quantities) 
     if (opteta) then  

       if(ehnuekt(kk,ll) .ne. 0.d0) then 
       
         etab = ehnuekt(kk,ll)*hcfre3(kk) * (etabf(kk,ll)+opaff*etaffc)  
         etab = etab/clf(ll) ! now corrected
 
         bn = hcfre3(kk)/ (1.d0/ehnuekt(kk,ll)-1.d0)  

       else
! high energies
         x=hkl*fre(kk)/temp(ll)
         if(x.lt.200.d0) stop' opacitm: problem at high energies'
         expo=log(hcfre3(kk) * (etabf(kk,ll)+opaff*etaffc))-x
         etab=exp(expo)
         etab = etab/clf(ll) ! now corrected

         expo=log(hcfre3(kk))-x
         bn=exp(expo)
       endif
         
!  check for LTE in metals 
!       lw=lwion ! for selected and non-selected elements; original version
                  ! for tests: marginal influence, deteroriates convergence
       lw=lwion1 ! for selected and non-selected elements:
       if (lteopt.or.ll.ge.lw) then  !opab should never be zero 
                                         !(minimum is ff-contrib)  
          if(nlambox.ne.0) then
            !include lines
          slinel=bn
          etab=etab+opal*slinel !since (<opal>+opac)*(<fth>+<fnth>)=<opal>
          thomson_lines(ll,kk)=0.
          else
            if(thomson_lines(ll,kk).ne.0.) then
!            stop' LTE, no lines and Thomson ne 0'
              thomson_lines(ll,kk)=0.
            endif
          endif
          bn1 = etab/opab  
          if (abs(1.d0-bn1/bn).gt.precis) then 
!           print*,'!warning!',abs(1.d0-bn1/bn)
           !stop ' error in lte -- opacitm!!!'
          endif
       else
          if(nlambox.ne.0) then
            !include lines
          slinel=fracth(nlambox,ll)*bn
          etab=etab+(opal+opacg)*slinel
          thomson_lines(ll,kk)=(opal+opacg)*fracnth(nlambox,ll) ! corrected, since OPAL corrected
          else
            if(thomson_lines(ll,kk).ne.0.) then
!            stop' NLTE, no lines and Thomson ne 0'
              thomson_lines(ll,kk)=0.
            endif
          endif
       endif 
!       if(opal.ne.0) then
!       print*,kk,nlambox,xlam,ll,lwion,thomson_lines(ll,kk)/opal
!       else
!       print*,kk,nlambox,xlam,ll,lwion,thomson_lines(ll,kk),'opal=0'
!       endif
       
       strue_m_new(ll,kk) = etab/opab  
!       if (strue_m_new(ll,kk).eq.0.) stop ' strue_metall = 0!'  
     end if  
  enddo !freq-loop
enddo !depth-loop

return  
end
!
!-----------------------------------------------------------------------
!
subroutine trad(nd,temp,dilfac1,xne,clf,iit,optcmf,emax,lastlte)
 
USE nlte_type
USE fund_const, ONLY: hkl, hc2, pi, hk, akb
USE nlte_dim
USE nlte_var, ONLY: fre,ifre,tradj,tradj1,xj,&
&                   xnelte,alo,opac,metals_converged,lwion1
USE nlte_app, ONLY: lwion, nrec,ielev,opametinf,indexopa,indexfrec, &
&                  kel,iel,gstat,abund,alphamet,fjk,fracmin, &
&                  occng,occ1,occnglte,occ1lte,zeta, &
&                  deltatradj,xion,zcorr,alpha,ergion,jatom_full
USE nlte_lines, ONLY: nochange_in_metals,metconv1

USE tcorr_var, ONLY: enatcor,opttcor,temp_converged, &
&                    qrbfr_m,qrbfi_m,dtqrbfr_m,dtqrbfi_m

USE nlte_opt, ONLY: opttcor_simple

USE nlte_porvor, ONLY: fic,tcl_fac_line,tcl_fac_cont,opa_eff_rat_old,&
&                      conv_eff_opa_rat

implicit none
!
!
! calculates trad from jnue and "ALO"(-> DELTATRADJ) for approx. NLTE
! clumping included.
!
! for selected elements (jatom_full=1), deltatradj = 1 always
!
! for non-selected elements, old approach (ground-state ioniz. only)
!
!     .. parameters ..
integer(i4b), parameter :: nd1=ID_NDEPT  
integer(i4b), parameter :: ifretot=ID_FREC1  

!     ..
!   
!     .. scalar arguments ..
real(dp), intent(in) :: emax
integer(i4b), intent(in) :: nd,iit
logical :: start, optcmf, lastlte
!
!     .. array arguments ..
real(dp), intent(in) ::  temp(nd),dilfac1(nd),xne(nd),clf(nd)  
!     ..
integer(i4b) ::  j,k,kk,index,lev,ll,n,number
!     ..
!     .. local arrays ..
real(dp), dimension(ifretot,nd1) ::  ehnuekt
real(dp), dimension(nd1) :: aux, xnerat
!
real(dp) ::  xlambda, freq, gs, alphai, xni, xnk, &
&            xnistar,depart,opabf,opa,beta, &
&            diff1,tradold,tempnd,err,fac,betaalo1, &
&            kh,edge,const,xi,te,xil,xel,x3,x4,x6,err1

real(dp) :: tcl

data start/.true./

tempnd=temp(nd)

do k=1,ifre
   xlambda=1./fre(k)
   aux=xj(:,k)/dilfac1

!  for excited levels
   
   tradj(:,k) = hkl*fre(k)/ (log(hc2/xlambda**3/aux+1.d0))       

!  for ground states

   tradj1(:,k) = hkl*fre(k)/ (log(hc2/xlambda**3/xj(:,k)+1.d0))
   

!  for test (trad=teff)

!   tradj(:,k)=teff 
!   aux=dilfac1*bnue(xlambda*1.d8,teff)
!   tradj1(:,k) = hkl*fre(k)/ (log(hc2/xlambda**3/aux+1.d0))
enddo 

! check for thermalization
err=0.
do k=1,ifre
  err=max(err,abs(1.-tradj(nd,k)/tempnd))
!  print*,k,1.d8/fre(k),tradj(nd,k),tempnd
  if(1.d8/fre(k).gt.20.d0) err1=err
enddo
print* 
print*,'MAX. ERROR BETWEEN TRAD(ND,K) AND TEMP(ND) = ',err
print*,'MAX. ERROR FOR LAMBDA > 20 A               = ',err1
print*
if(err1.gt.0.01d0) stop' something wrong with thermalization'


! for iteration zero or 1 iteration after restart
if (start) then
  print*,'iteration zero or 1st iteration after restart: deltatradj = 1!'
  print*
  start=.false.
  if(deltatradj(1,1,1).ne.1.) stop' something wrong in deltatradj'
  return
endif

!if (tcorr and previous LTE solution)
if (lastlte) then
  print*,'1st iteration after LTE update: deltatradj = 1!'
  print*
  deltatradj=1.
  return
endif  

! calculation of "new" radiation temperature for the ALI-method of metals

deltatradj=1.
xnerat=xne/xnelte

do kk=1,ifre
  freq = fre(kk)  
  do ll=1,nd
     ehnuekt(kk,ll) = exp(-hkl*freq/temp(ll))  
   enddo
enddo

! convergence of metals allowed only if lines treated

metals_converged = .false.
nochange_in_metals=.true.

!if(.not.optcmf.and.iit.ge.12.and.emax.lt.0.03) metals_converged = .true.
!if(     optcmf.and.iit.ge.22.and.emax.lt.0.03) metals_converged = .true.
! allow convergence even if explicit ions oscillate
if(.not.optcmf.and.iit.ge.12) metals_converged = .true.
if(     optcmf.and.iit.ge.22) metals_converged = .true.

! to allow convergence, tcorr has to be converged
if(enatcor.and..not.temp_converged) metals_converged = .false.
!NOTE - additional constraint from conv_opa_eff_rat 
if(conv_eff_opa_rat.gt.0.01) metals_converged = .false.

rec: do n = 1,nrec

!only ground-state considered

   lev = 1
   number=opametinf(n,lev)
   if(number.eq.0) cycle rec
   index=indexopa(n,lev) 
   if(index.eq.0.) cycle rec !eq.0 for edges larger than max(fre)

   k = kel(n)
   j = iel(n)
   if(jatom_full(k).eq.1) then
! ensure that 
     if (deltatradj(k,j,lwion1-1).ne.1.d0) &
&      stop' deltatradj ne 1 for selected elements'
     cycle rec
   endif  
! once more, check consistency
   if(index.ne.indexfrec(k,j)) stop ' inconsistency in ground-state indices'

   gs=gstat(k,j,lev) 

   kk=index
   alphai=alphamet(number,kk)
   if(alphai.eq.0.d0) stop ' error in alphamet'

   depth: do ll=1,lwion-1
      if(alo(ll,kk).lt.0.1) cycle depth
      if(abund(k)*fjk(k,j,ll).lt.fracmin) cycle depth 
      xni=occng(n,lev,ll)*gs
      xnk=occ1(k,j+1,ll)   ! includes stat. weight
      xnistar = xnk*occnglte(n,lev,ll)*gs/occ1lte(k,j+1,ll)*xnerat(ll)  
      depart = xni/xnistar  
!      print*,'yyy',k,j,ll,depart,zcorr(k,j,ll)
      opabf=xnistar*alphai*(depart-ehnuekt(kk,ll))*opa_eff_rat_old(ll,kk)/clf(ll) !corrected
      opa=opac(ll,kk) ! already corrected
!opac should be EFFECTIVE from previous iteration, 
!thus corrected below with opa_eff_ratio to get correct ratio bf to opac
      beta=opabf/opa  
      if(beta.gt.1.d0) stop' beta > 1 in Trad'

      !corrected 

!old version
!      tradold=tradj(ll,kk)
!      diff1=1.-beta*alo(ll,kk)/(depart*dilfac1(ll)*zcorr(k,j,ll))* &
!&       temp(ll)/tradold* &
!&       exp(-xion(k,j)*hkl*(1./temp(ll)-1./tradold))

!new version(without W)
      tradold=tradj1(ll,kk)
      fac=zeta(k,j)/(zcorr(k,j,ll)*depart)*temp(ll)/tradold* &
&       exp(-xion(k,j)*hkl*(1./temp(ll)-1./tradold))
      betaalo1=1.-beta*alo(ll,kk)*zeta(k,j)
      diff1=1.-beta*alo(ll,kk)*fac

      if(diff1.lt.0.) then 
        print*,'warning'
! check precision        
! if Lambda * beta close to unity, large correction induced if
! fac (=rest/depart) deviates only slightly from unity. To avoid this problem, &
! we check accuracy of fac and compare it to precision (0.01/(1-Lambda*beta)        
!        
      else if (beta.gt.0.9.and.abs(1.-fac).lt.0.01/betaalo1) then  
         deltatradj(k,j,ll)=1.
         write(*,202) k,j,1.d8/fre(kk),ll,beta,tradold
!         print*,temp(ll),depart,zcorr(k,j,ll),zeta(k,j),alo(ll,kk)
      else
         deltatradj(k,j,ll) = diff1/betaalo1
      endif    

      if (deltatradj(k,j,ll).lt.0.8.or.deltatradj(k,j,ll).gt.1.2) then
        deltatradj(k,j,ll)=1.
        metals_converged = .false.
        nochange_in_metals=.false.
        write(*,201) k,j,1.d8/fre(kk),ll,beta,tradold
!        print*,temp(ll),depart,zcorr(k,j,ll),zeta(k,j),alo(ll,kk)
      endif       
      if (deltatradj(k,j,ll).lt.0.99.or.deltatradj(k,j,ll).gt.1.01) then
        metals_converged = .false.
        nochange_in_metals=.false.
        write(*,200) k,j,1.d8/fre(kk),ll,beta,tradold,deltatradj(k,j,ll)
!        print*,temp(ll),depart,zcorr(k,j,ll),zeta(k,j),alo(ll,kk)
      endif

! for tests
!      write(*,200) k,j,1.d8/fre(kk),ll,beta,tradold,deltatradj(k,j,ll)
!      print*,temp(ll),depart,zcorr(k,j,ll),zeta(k,j),alo(ll,kk)
!      print*,xni,xnk,xnistar,opabf,opa
!      print*
   enddo depth

enddo rec    

! specific metals have to be converged in any case
if(.not.metconv1) then
metals_converged=.false.
nochange_in_metals=.false.
endif

if(metals_converged) then
print* 
print*,' METALS CONVERGED !'
print*
endif

if(.not.opttcor) return
if(.not.opttcor_simple) return
nochange_in_metals=.false. !(update cbb cooling/heating)


! Heating/cooling rates for elements in approx NLTE; (beta=1, s=2 assummed)
! => Qik = kTrad Rik, Qki= kTe Rki
! no special treatment for clumping required
qrbfr_m=0.d0
qrbfi_m=0.d0
dtqrbfr_m=0.d0
dtqrbfi_m=0.d0

kh=1./hk
rec1: do n = 1,nrec

level1: do lev = 1,ielev(n)
   number=opametinf(n,lev)
   if(number.eq.0) cycle level1
   index=indexopa(n,lev) 
   if(index.eq.0) cycle level1 !eq.0 for edges larger than max(fre)

   k = kel(n)
   j = iel(n)

   gs=gstat(k,j,lev) 

   alphai=alpha(n,lev)*1.d-18
   edge=xion(k,j)-ergion(k,j,lev)

   const=8.d0*pi*edge**2*alphai*kh*akb  !8pi alpha (nu_i/c)^2 * k/h * k
   xi=hkl*edge  !(hnu_i/k)
   
!   if(k.eq.6.and.j.eq.4) print*,'civ',edge,fre(index),alphai
   depth1: do ll=1,nd1

      if(abund(k)*fjk(k,j,ll).lt.fracmin) cycle depth1 

      tradold=tradj(ll,index)
      te=temp(ll)
      xil=xi/tradold !hnu_i/(kTrad)
      xel=xi/te      !hnu_i/(kTe)
      x3=const*tradold*exp(-xil)*tradold*dilfac1(ll)
      x4=const*     te*exp(-xel)*te
      x6=x4/te*(2.d0+xel)  !dx4/dT

! cf subroutine NETMAT, after call of INTERMEPHO
      xni=occng(n,lev,ll)*gs
      xnk=occ1(k,j+1,ll)   ! includes stat. weight
      xnistar = xnk*occnglte(n,lev,ll)*gs/occ1lte(k,j+1,ll)*xnerat(ll)  
!     if(k.eq.6.and.j.eq.4) print*,'civ',ll,x3,x4,xni,xnistar
      qrbfr_m(ll)=qrbfr_m(ll)+xnistar*x4
      qrbfi_m(ll)=qrbfi_m(ll)+xni*x3
      dtqrbfr_m(ll)=dtqrbfr_m(ll)+xnistar*(x6-x4*(1.5d0+xel)/te)
      dtqrbfi_m(ll)=dtqrbfi_m(ll)-xni*x3*(1.5d0+xel)/te
    enddo depth1

enddo level1
enddo rec1    
print*,' TCORR IN TRAD DONE'
return

200 format(2(i2,1x),f10.5,1x,i2,f10.5,1x,f10.0,1x,f10.5)
201 format(2(i2,1x),f10.5,1x,i2,f10.5,1x,f10.0,1x,'no corr')
202 format(2(i2,1x),f10.5,1x,i2,f10.5,1x,f10.0,1x,'precision')
end
!
!-----------------------------------------------------------------------
!
subroutine restart(nd,xne,dilfac1,optmet,ilow,imax) 

USE nlte_type
USE nlte_dim

USE nlte_var, ONLY: modnam,opat_m_new,opat_m_nolines,strue_m_new, &
& thomson_lines,metals_converged,lines,optlines,fre,ifre,abund_have_changed
USE nlte_app, ONLY: lwion,coll_therm,corr_beta, occ1old, occngold, fjk
USE tcorr_var, ONLY : enatcor
USE nlte_porvor, ONLY: epsi1,opa_eff_rat

implicit none

!     .. parameters ..
integer(i4b), parameter :: ifretot=ID_FREC1, nd1=ID_NDEPT  
!
!     .. scalar arguments ..
integer(i4b) :: nd
!
!     .. array arguments ..
real(dp) ::  xne(nd), dilfac1(nd)
integer(i4b) :: ilow(nd,id_atoms), imax(nd,id_atoms)  

real(dp), dimension(:), allocatable :: freold
real(dp), dimension(:,:), allocatable :: aux1,aux2,aux3,aux4,aux5

logical optmet

integer(i4b) ::  i,ifreold,ks,k 
real(dp) :: frelast,q,q1

call copyocc(xne,nd,ilow,imax)

if(.not.optmet) return

if(lines) optlines=.true.

open (1,err=100,file=trim(modnam)//'/METAL_RESTART', &
&   status='unknown',form='unformatted')

  rewind 1  
! changed from version 7.4 on (old models can no longer be read)
! namely opa_eff_rat included
  read (1,end=100) opat_m_new, opat_m_nolines, strue_m_new, thomson_lines, opa_eff_rat
! changed from version 6.1 on. Old models without fjk can still be read
  read (1,err=10) coll_therm, corr_beta, occ1old, occngold, fjk
  goto 20
10 print*,' PREVIOUS MODEL DID NOT CONTAIN FJK(METALS,NLTE)'
   print*,' FJK(LTE) USED INSTEAD. ILOW, IMAX MIGHT CHANGE!'

20  read (1,end=100) metals_converged,ifreold,frelast

  print* 
  print*,' METAL_INFO FROM PREVIOUS RUN READ SUCCESSFULLY'
  print*
  print*,' METALS_CONVERGED (old model) = ',metals_converged
  print*

  if(enatcor) metals_converged = .FALSE.
! uncomment for tests
!  metals_converged = .FALSE.
! form v7.1 on
  if (abund_have_changed) then
    print*,' SIGNIFICANT CHANGE OF ABUNDANCE, METALLIC BACKGROUND RECALCULATED '
    metals_converged = .FALSE.
  endif

do i=1,nd
  if(dilfac1(i).eq.1.) then
    lwion=i
    exit  
  endif
enddo

if(ifre.ne.ifreold .or. frelast .ne. fre(ifre)) then
 ! frequency grid has changed, interpolation necessary
 print*,' NEW FREQUENCY GRID, INTERPOLATION REQUIRED'
 print*
 allocate(freold(ifreold))
 read(1,end=100) freold
 if(freold(1).ne.fre(1)) stop' RESTART: error in first frequency' 

 allocate(aux1(nd1,ifre),aux2(nd1,ifre),aux3(nd1,ifre), &
          aux4(nd1,ifre),aux5(nd1,ifre))

 ks=1
 do k=2,ifre
   do i=ks,ifreold-1
     if(fre(k).ge.freold(i).and.fre(k).le.freold(i+1)) goto 50
   enddo  
   if(fre(k) .gt. frelast) goto 50 ! extrapolation necessary
   stop' fre(k) not found in freold'

50 ks=i
   if(ks.eq.ifreold) ks=ks-1 !for extrapolation
   q=log10(fre(k)/freold(ks+1))/log10(freold(ks)/freold(ks+1))
   q1=1.d0-q

   aux1(:,k)=q*log10(opat_m_new(:,ks))    +q1*log10(opat_m_new(:,ks+1))
   aux2(:,k)=q*log10(opat_m_nolines(:,ks))+q1*log10(opat_m_nolines(:,ks+1))

   where (strue_m_new(:,ks).eq.0..or.strue_m_new(:,ks+1).eq.0.)
     aux3(:,k)=0.
   elsewhere     
     aux3(:,k)=q*log10(strue_m_new(:,ks))   +q1*log10(strue_m_new(:,ks+1))
   endwhere
   
   where (thomson_lines(:,ks).eq.0..or.thomson_lines(:,ks+1).eq.0.)
     aux4(:,k)=0.
   elsewhere     
     aux4(:,k)=q*log10(thomson_lines(:,ks)) +q1*log10(thomson_lines(:,ks+1))
   endwhere
!porosity
   aux5(:,k)=q*log10(opa_eff_rat(:,ks))    +q1*log10(opa_eff_rat(:,ks+1))

enddo

 opat_m_new(:,2:ifre)    =10.d0**aux1(:,2:ifre) ! k=1 identical
 opat_m_nolines(:,2:ifre)=10.d0**aux2(:,2:ifre)
!porosity
 opa_eff_rat(:,2:ifre)=10.d0**aux5(:,2:ifre)

 If (maxval(opa_eff_rat(:,2:ifre)).gt.epsi1.or. &
& minval(opa_eff_rat(:,2:ifre)).le.0.d0) then    
    print*,maxval(opa_eff_rat(:,2:ifre)),minval(opa_eff_rat(:,2:ifre))
    stop'opa_eff_rat interpolation out of bounds at restart'
 endif
!--------------------------------

 where(aux3(:,2:ifre).ne.0.)
  strue_m_new(:,2:ifre)   =10.d0**aux3(:,2:ifre)
 endwhere

 where(aux4(:,2:ifre).ne.0.)
  thomson_lines(:,2:ifre) =10.d0**aux4(:,2:ifre)
 endwhere

 deallocate(freold,aux1,aux2,aux3,aux4,aux5)

endif

close (1)  

return


100 stop ' ERROR IN READING FILE METAL_RESTART -- check DIMENSIONS!'
end
!
!-----------------------------------------------------------------------
!
subroutine save_metal_info
!
! changed from version 5.4 on.
! freq. grid required to allow for interpolations if freq. grid has changed
! (approx.) nlte numbers required to allow for smooth restart (needed for betas)
! deltatradj no longer required, since always calculated
!  
USE nlte_type
USE nlte_dim
USE nlte_var, ONLY: modnam,opat_m_new,opat_m_nolines,strue_m_new, &
& thomson_lines,metals_converged,fre,ifre
USE nlte_app, ONLY: coll_therm, corr_beta, occ1old, occngold, fjk
USE nlte_porvor, ONLY: opa_eff_rat, opa_eff_rat_old

implicit none

open (1,file=trim(modnam)//'/METAL_RESTART',status='UNKNOWN', &
& form='UNFORMATTED')

rewind 1  
!
! opa_eff_rat(new) added from version 7.4 on
!NOTE: opa_eff_rat is saved (rather than opa_eff_rat_old), since 
!first call after restart (where save_metal_info is used) is to
!OPACITC, where the outside range becomes updated 
!-------------------------------------------------
write (1) opat_m_new,opat_m_nolines,strue_m_new,thomson_lines,opa_eff_rat
! fjk added from version 6.1 on
write (1) coll_therm, corr_beta, occ1old, occngold, fjk
write (1) metals_converged,ifre,fre(ifre)
write (1) fre

close (1)  
return
end
!
!-----------------------------------------------------------------------
!
! PACKAGE FOR APPROXIMATE LINE TRANSFER
!
!-----------------------------------------------------------------------
!
subroutine fgrid_lines(teff)

USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT
USE fund_const, ONLY: akb,amh,clight


USE nlte_var, ONLY: fre, vturb

USE nlte_app, ONLY: xion, vth, vth8

USE nlte_lines, ONLY: wavblue, wavred, wavcon, wavcon1, &
& nsubinter, nlam, calcopal, fgrid, &
& opacgrid, opalgrid, fracth, fracnth, tradj_fg, opal_aux, sum_th, sum_nth, &
& opa_eff_aux, opa_eff_grid 

USE tcorr_var, ONLY: enatcor

USE nlte_porvor, ONLY: w_bg_red,w_bg_blue 

implicit none
integer(i4b), parameter :: sample_width=120 ! number of channels for vturb=10km/s
! Don't fiddle around with this number, if Stark-broadening should be used:
! Hydrogen Voigt parameters fitted for this value;


logical :: enatcor_simple=.false.

real(dp), intent(in) :: teff

real(dp) :: const, const1, vth10
integer(i4b) :: i

!include all IR lines (costs almost nothing)
wavred=0.95d8/fre(1)

if (teff <= 10000.) then
  wavblue=248.d0

else if (teff <= 20000.) then
!He2 edge (229 A)
  wavblue=1.d8/xion(2,2)

!-------------------------------------------------------------------------
!else if (teff <= 30000.) then
!He2 edge (229 A)
!  wavblue=1.d8/xion(2,2)
! that was the old version (until 5.x)  
!else if (teff <= 37000.) then
!He2 edge (229 A)
!  wavblue=1.d8/xion(2,2)

!else if (teff <= 43000.) then
!C4 edge (192 A)
!  wavblue=1.d8/xion(6,4)

!else if (teff <= 55000.) then
!N4 edge (160 A)
!  wavblue=1.d8/xion(7,4)

!else if (teff <= 70000.) then
!O5 edge (109 A)
!  wavblue=1.d8/xion(8,5)

!else
!O6 edge (90 A)
!  wavblue=1.d8/xion(8,6)
!end if
!-------------------------------------------------------------------------
! from version 6.0 on, we use the full line list for Teff ge 25000, to
! ensure correct fluxes also below 229 A  
else if (teff < 25000.) then
!He2 edge (229 A)
  wavblue=1.d8/xion(2,2)

else
!O6 edge (90 A)
  wavblue=1.d8/xion(8,6)
end if

! other important edges (but at the moment not used)
!                         /172.43  /!Fe55-edge 
!                         /164.12  /!Fe51-edge
!                         /154.53  /!Mg31-edge

! to ensure that for each value of v_turb the same sample-width is present
! use oxygen as average representative, with same mass as in aweight, and
! at nominal value of vturb = 10 km/s


! calculate actual number of channels (the larger vturb, the lower number of channels)
vth10=sqrt((2.*akb*teff/(amh*16.000d0))+10.d5**2)

nsubinter=sample_width/2*(vth10/vth(8)) !using actual vturb

! check for truncation-effects at different compilers
! check removed from v7.2 on
!if(abs(1.-vturb/10.d5).lt.1.d-5.and.nsubinter.ne.sample_width/2) &
!& stop' error in nsubinter'

!renormalize vth(8) to get rid of truncation errors:
! sample_width * vth10 =: 2 * nsubinter * vth8
vth8=sample_width/2*vth10/dble(nsubinter) 
if(abs(1.-vth8/vth(8)).gt.0.05) then
 print*,nsubinter,vth(8),vth8
 print*,'vturb most probably too large'
 stop' truncation error in renormalized vth(8) too large'
endif

const = 1.d0 + sample_width*vth10/clight !sample width (independent of vturb) 
wavcon = log(const)

!Delta nue for summation of opals, nsubinter (order sample_width/2) subintervals
const1 = const**(1.d0/float(nsubinter))
wavcon1 = log(const1)

nlam = int(log(wavred/wavblue)/wavcon) + 2 ! since xlam(1)=wavblue

allocate(fgrid(nlam))
allocate(calcopal(nlam))
allocate(opacgrid(nlam,ID_NDEPT))
allocate(opalgrid(nlam,ID_NDEPT))
allocate(fracth(nlam,ID_NDEPT))
allocate(fracnth(nlam,ID_NDEPT))
allocate(opal_aux(nsubinter,ID_NDEPT))
allocate(sum_th(nsubinter,ID_NDEPT))
allocate(sum_nth(nsubinter,ID_NDEPT))

allocate(opa_eff_aux(nsubinter,ID_NDEPT))
allocate(opa_eff_grid(nlam,ID_NDEPT))

!if (enatcor) allocate(tradj_fg(nlam,ID_NDEPT)) 
if (enatcor_simple) allocate(tradj_fg(nlam,ID_NDEPT)) 

do i = 1,nlam
  fgrid(i) = wavblue*const**(i-1)
enddo
wavred=fgrid(nlam) !redefinition to avoid end-effects

!variables to compute OPA_EFF_RAT outsied wavred,wavblue edges
!within opacitc
w_bg_red = wavred
w_bg_blue = wavblue 

return
end
!
!-----------------------------------------------------------------------
!
subroutine open_linedat
!
! open line files (binaries) and according info file
!
USE nlte_type
USE nlte_app, ONLY: fpath
USE nlte_lines, ONLY: ntotnl3, nfullrec, nrest, nbunch, ram, id, gf, xlam, &
& wavblue, wavred, iblue, ired

implicit none
integer(i4b) :: ios, npack, jrec, j, nlast, &
&               jrecblue

! atomic transition information

open(111,file=fpath//'nl3i_all',form='unformatted',status='old', &
&        err=102,iostat=ios)
open(112,file=fpath//'nl3a_all',form='unformatted',status='old', &
&        err=103,iostat=ios)
open(113,file=fpath//'nl3g_all',form='unformatted',status='old', &
&        err=104,iostat=ios)
open(114,file=fpath//'nl3info_all',status='old',err=105,iostat=ios)

read(114,*) ntotnl3, nfullrec, nrest
close (114)

if(ram .eq. 'small') then
  allocate(id(nbunch),gf(nbunch),xlam(nbunch))
  iblue=1
  ired=nbunch
  return
endif

if(ram .ne. 'big') stop' wrong ram-parameter'
 
allocate(id(ntotnl3),gf(ntotnl3),xlam(ntotnl3))

! read complete line list
npack=nbunch
jrecblue=0
do jrec = 1,nfullrec+1
   if(jrec .eq. nfullrec+1) npack=nrest
   nlast=(jrec-1)*nbunch
   read(111)(id(j),j=nlast+1,nlast+npack)
   read(112)(xlam(j),j=nlast+1,nlast+npack)
   read(113)(gf(j),j=nlast+1,nlast+npack)
   if (xlam(nlast+npack).lt.wavblue) jrecblue=jrec
!   print*,' lines up to',xlam(npack),'A read'
enddo

! define iblue
do j=jrecblue*nbunch+1,(jrecblue+1)*nbunch
      if (xlam(j).ge.wavblue) exit
enddo
iblue=j
!print*,wavblue,xlam(iblue),iblue

! define ired
do j=ntotnl3,1,-1
      if (xlam(j).le.wavred) exit
enddo
ired=j
!print*,wavred,xlam(ired), ired
return

102  write(*,*) ' error in reading file nl3i, iostat=',ios
      stop
103  write(*,*) ' error in reading file nl3a, iostat=',ios
      stop
104  write(*,*) ' error in reading file nl3g, iostat=',ios
      stop
105  write(*,*) ' error in reading file nl3info, iostat=',ios
      stop

end
!
!-----------------------------------------------------------------------
!
subroutine neglect_ions
!
! find ions which can be neglected
!
USE nlte_type
USE nlte_app, ONLY: fracmin,abund,fjk,natom,jatom,jmax,calcion

implicit none

integer(i4b) :: k,j
real(dp) :: fjkmax

calcion=0

do k = 1,natom
  if(jatom(k) .eq. 0) cycle 
  do j = 1,jmax(k)
    fjkmax=maxval(fjk(k,j,:))
    if(abund(k)*fjkmax.gt.fracmin) calcion(k,j)=1
  enddo
enddo
return
end
!
!-----------------------------------------------------------------------
!
subroutine opacitl(xne,lteopt,ntemp,dvdr,vr,velo,clf,frenew)
!
! reads line data in packets of nbunch lines and calls line opacity summation
!
! line transfer including clumping
! assumption: clumps small against Sobolev length!
! method: scale all opacities with CLF^(-1), consider that cont. opacities
!         from nlte.f90 have been scaled already  
  
USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT, ID_FREC1
USE fund_const, ONLY : sigmae, amh, hc2, pi

USE nlte_var, ONLY: ifre, fre, opac_nolines, tradj

USE nlte_app, ONLY: te, wion, vth8

USE nlte_lines, ONLY: ram, nbunch, nsubinter, nfullrec, nrest, iblue, ired, &
& id, gf, xlam, &
& wavblue,wavred,wavcon, &
& opacgrid, opalgrid, opal_aux, &
& sum_th, sum_nth, fracth, fracnth, vanreg, vanreg1, &
& nlam, calcopal, fgrid, nlamb_old, tradj_fg, u0, wexp, &
& clufac,clufac_forb,nblock, &
& gamma_stark, xnemin, opa_eff_grid

USE tcorr_var, ONLY: enatcor,opttcor,qcbbu_m,qcbbd_m,dtqcbbu_m,dtqcbbd_m

USE nlte_opt, ONLY: opttcor_simple

USE nlte_porvor, ONLY: fic,tcl_fac_line,tcl_fac_cont

implicit none

real(dp), parameter:: c_vanreg= 0.5 * 2.055d-23
real(dp), parameter:: c_forbid= 8.63d-6/(0.5*1.3707d-7)

logical :: enatcor_simple=.false.

integer(i4b), intent(in) :: ntemp
real(dp),dimension(ID_NDEPT), intent(in) :: xne, dvdr, vr, velo, clf

logical, intent(in) :: lteopt, frenew

logical  start

real(dp), dimension(ID_NDEPT) :: opath, betafac, collfac, avoigtp, tcl_arr

integer(i4b) :: npack, jrec, j, nlinesr,nlinesm, nlamdiff, &
&               ifreold, k, ll, nlambox, jm, jm1, nb, nb1, inemin

integer(i4b), dimension(1) :: jb

integer(i4b), save :: ifold
real(dp), dimension(ID_FREC1), save :: freold

real(dp) :: fregrid, q, q1, xlamc, velratio, diff1, diff2

real(dp) :: tcl,fac2

data start/.true./ 

if (frenew) then
! calculate indexarray where line-opacities have to be calculated
  calcopal=0
  do j=1,ifre

    xlamc=1.d8/fre(j)
    if(xlamc.gt.wavblue.and.xlamc.lt.wavred) then
      nlambox = 1 + log(xlamc/wavblue)/wavcon
      if(nlambox.ge.nlam) stop ' error in nwavbox'
      calcopal(nlambox)=1
    endif
  enddo
endif

if (lteopt.and.start) then
! only very first LTE iteration
   if(ntemp.ne.1) stop ' something wrong with ntemp in opacitl'
   start=.false.
   opath = sigmae*amh*xne/clf ! corrected
! so far, nothing to be done (only for start) 
   do j=nlam-1,1,-1
     opacgrid(j,:) = opath
   enddo  

else

if (start) then
! if lines only in NLTE, however not in model
! may only happen in case of no T-corr! can also lead to small inconsistencies
! if restart
!if (enatcor) stop' tcorr: something wrong with start in opacitl'   
if (enatcor_simple) stop' tcorr: something wrong with start in opacitl'   
start=.false.
ifold=ifre
freold=fre
endif
  
! prepare opacities at grid centers; note, that fre(1) at low end

ifreold=1
do j=nlam-1,1,-1

  if(calcopal(j).eq.0) cycle ! only those are needed for opacities
  
  fregrid=.5d8*(1./fgrid(j)+1./fgrid(j+1))

! use old freq. grid since opac_nolines has been calculated in this grid;
! if model finished, both grids will coincide, of course  

  do k=ifreold,ifold-1
    if(fregrid.ge.freold(k).and.fregrid.lt.freold(k+1)) goto 10
  enddo  
!  
! extrapolation at end possible
!  
  if(fgrid(j+1).gt.1.d8/freold(ifold)) then
    k=ifold-1
    goto 10
  endif
! something wrong  
  print*,ifreold,ifold,fregrid
  print*,freold(ifold-1),freold(ifold)
  print*,j,fgrid(j),fgrid(j+1)

10  ifreold=k
  q=(freold(k)-fregrid)/(freold(k)-freold(k+1))
  q1=1.d0-q
  do ll=ntemp,ID_NDEPT

    opacgrid(j,ll)=q1*opac_nolines(ll,k)+q*opac_nolines(ll,k+1) ! already corrected
    if (opacgrid(j,ll).eq.0.) then
      print*,j,ll,k,k+1
      stop
    endif  
!    if(ll.eq.40) &
!    print*,j,k,opac_nolines(ll,k),opacgrid(j,ll),opac_nolines(ll,k+1) 
  enddo

enddo

endif

! prepare Stark broadening
avoigtp=gamma_stark*xne/(2.*pi*sqrt(pi)) ! modified voigt parameter (see notes) 
        ! excludes 2*dnue, includes additional sqrt(pi)

do ll=1,id_ndept
  if(xne(ll).ge.xnemin) goto 15
enddo
stop' xnemin not found in xne'

15 inemin=ll

!different pathes for LTE and NLTE
!--------------------------------------------------------------------

if(.not.lteopt) then

do ll=1,id_ndept
  velratio=velo(ll)/(2.*vth8)
  if(velratio.gt.nsubinter) then
    nblock(ll)=1
  else if (velratio.lt.1.) then
    nblock(ll)=nsubinter
  else
    jm=velratio
    nb=nsubinter/jm
    nb1=nsubinter/(jm+1)
! now other way round, since integer operations
    jm=nsubinter/nb
    jm1=nsubinter/nb1
    if(nsubinter-jm*nb.eq.nsubinter-jm1*nb1) then
      diff1=abs(velratio-jm)
      diff2=abs(velratio-jm1)
      if(diff1.le.diff2) then
        nblock(ll)=nb
      else
        nblock(ll)=nb1
      endif  
    else if(nsubinter-jm*nb.lt.nsubinter-jm1*nb1) then
      nblock(ll)=nb
    else
      nblock(ll)=nb1
    endif  
  endif
! in any case, there must be a chance that photons can escape unless the
! line-density is really high  
    nblock(ll)=max(nblock(ll),4)
enddo
!NLTE path

!check, just in case
if(ifold.ne.ifre) stop ' error in freq. grid: opacitl, nlte!'

!if(opttcor) then
if(opttcor_simple) then

!check, just in case
if(ntemp.ne.1) stop' subr. opacitl: something wrong with ntemp'

!interpolate radiation temperatures to fgrid
ifreold=1
tradj_fg=0.

do j=nlam,1,-1

  fregrid=1.d8/fgrid(j) !if opttcorr, ALL tradj_fg's are needed

  do k=ifreold,ifre-1
    if(fregrid.ge.fre(k).and.fregrid.lt.fre(k+1)) goto 20
  enddo  
  print*,k,fre(k),fre(k+1),j,fregrid
  stop' subr. opacitl: fregrid not found in fre'
20  ifreold=k
  q=(fre(k)-fregrid)/(fre(k)-fre(k+1))
  q1=1.d0-q

  tradj_fg(j,:)=q1*tradj(:,k)+q*tradj(:,k+1)
enddo
endif


! prepare common factors to calculate coll_therm per line
betafac=(dvdr+2.*vr)/3.D-8  ! D-8, since xlam in Angst!
collfac=c_vanreg*xne/sqrt(te)

! for first freq. interval (nlambox=1) calculate vanreg in advance
u0=1.4388d8/(fgrid(1)*te)
vanreg=collfac*fgrid(1)**3*(1.d0-exp(-u0))
vanreg1=vanreg*c_forbid/fgrid(1)
vanreg=vanreg/(1.+vanreg)

! additional quantities to calculate heating/cooling rates
! if(opttcor) then
if(opttcor_simple) then
! heating/cooling rates inside clumps, no scaling required
   qcbbu_m=0.
   qcbbd_m=0.
   dtqcbbu_m=0.
   dtqcbbd_m=0.
   wexp=wion*exp(u0*(1.d0-te/tradj_fg(1,:)))
!
!  lambda in Angstrom!
!   
!  cooling rate = nl*Clu * hnu_lu (cf. subroutine linesob)
!  Clu (van Regemorter) = Cul/Aul * flu * 6.67d15 / lam^2 * exp (-u0)
!                        = collfac * flu * 6.67d15 * lam * exp(-u0)
!  clufac= collfac * 6.67d15 * exp(-u0) * lam * (0.5*hc2*1.d8) / lam
!  cooling rate=nl*clufac*flu
!  heating rate=nu * (nl/nu)* clufac*flu  with nu = nl*(nu/nl)
!   
   clufac=collfac*6.6702082d15*exp(-u0)*0.5d8*hc2
   clufac_forb=clufac*c_forbid/fgrid(1)
endif

npack=nbunch
nlinesr = 0
nlinesm = 0

opalgrid=0.
fracth=0.
fracnth=0.
opal_aux=0.
sum_th=0.
sum_nth=0.
nlamb_old=0 !initialize

! set to zero, to finally check whether nlambox has been considered or not
! (see end of routine)
opa_eff_grid=0.

if(ram.eq.'small') then
rewind 111
rewind 112
rewind 113

do jrec = 1,nfullrec+1
   if(jrec .eq. nfullrec+1) npack=nrest
   read(111)(id(j),j=1,npack)
   read(112)(xlam(j),j=1,npack)
   read(113)(gf(j),j=1,npack)
!   print*,' lines up to',xlam(npack),'A read'
   if (xlam(npack).lt.wavblue) cycle
   if (xlam(1).gt.wavred) exit
   call sumopal_nlte(iblue,npack,nlinesr,nlinesm,betafac, &
&    collfac,clf,avoigtp,nblock,xne,inemin)
enddo

else
   call sumopal_nlte(iblue,ired,nlinesr,nlinesm,betafac, &
&    collfac,clf,avoigtp,nblock,xne,inemin)
endif
!--------------------------------------------------------------------
else
!LTE path

npack=nbunch
nlinesr = 0
nlinesm = 0

opalgrid=0.
opal_aux=0.
nlamb_old=0 !initialize

! set to zero, to finally check whether nlambox has been considered or not
! (see end of routine)
opa_eff_grid=0.

if(ram.eq.'small') then
rewind 111
rewind 112
rewind 113

do jrec = 1,nfullrec+1
   if(jrec .eq. nfullrec+1) npack=nrest
   read(111)(id(j),j=1,npack)
   read(112)(xlam(j),j=1,npack)
   read(113)(gf(j),j=1,npack)
!   print*,' lines up to',xlam(npack),'A read'
   if (xlam(npack).lt.wavblue) cycle
   if (xlam(1).gt.wavred) exit
   call sumopal_lte(ntemp,iblue,npack,nlinesr,nlinesm,clf,avoigtp,xne,inemin)
enddo

else
   call sumopal_lte(ntemp,iblue,ired,nlinesr,nlinesm,clf,avoigtp,xne,inemin)
endif

endif

!NOTE: In certain cases, particularly in the IR/radio, certain NLAMBOXes might
!be not explicitly considered even if CALCOPAL=1. This will happen if there
!are no lines in the specific boxes.
!In these cases, we obtain opalgrid(etc)=0. as initialized and as correct.
!If we include optically thick clumping, however, these boxes need to be
!considered, though only for the continuum part. This is done in the following.
do k=1,nlam 
   if (calcopal(k).eq.1 .and. opa_eff_grid(k,1).eq.0.) then !opa_eff_grid initialized above 
      if (opa_eff_grid(k,2).ne.0.) stop' error in opa_eff_grid, opacitl'
      tcl_arr = opacgrid(k,:)*tcl_fac_cont
      opa_eff_grid(k,:) = (1.+fic*tcl_arr)/(1.+tcl_arr)
   endif
enddo


!--------------------------------------------------------------------
! FOR TESTS
!do ll =ntemp,ID_NDEPT
!  do j=nlam-1,1,-1
!
!  if(calcopal(j).eq.0) cycle
!
!  if(mod(ll,6).eq.0) print*,j,ll,fgrid(j),opalgrid(j,ll)/opacgrid(j,ll)
!  enddo
!enddo

print* 
print*,' line summation finished'
print*,' considered lines at cont. freqs: ',nlinesr,'(r,s) and ',nlinesm,'(m)'

ifold=ifre
freold=fre

return
end
!
!-----------------------------------------------------------------------
!
subroutine sumopal_nlte(nstart,npack,nlinesr,nlinesm, &
&   betafac,collfac,clf,avoigtp,nblock,xne,inemin)
!
! calculates summed line-opacities (assuming box car profiles of width
! 2 deltanuedop, which cancels out, see notes) within fgrid

! alternative version accounting for line wings can be found in
! oldprog/nlte_approx_7.0_photlines_only.f90:
!  84% (const1) into central 2 channels (from int[exp(-x^2)] between -1 to +1) 
!  7% (const2)  into neighbouring 2x2 channels (from int[exp(-x^2)]
!     between -3 to -1 and between 1 to 3.

! Stark-broadening for (quasi-) resonance lines of metals and Hydrogen

! clumping (thick and thin) included

USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT
USE fund_const, ONLY : hc2, hkl, pi, wpi=>sqpi, pihalf

USE nlte_var, ONLY: lwion1,lam_ly

USE nlte_app, ONLY: jatom,calcion,te,vth8,indexel,indrec, &
& metall,occng,wion,gstat,ergion,xion,jatom_full,met_imin,met_imax, &
& lwion,vth

USE nlte_lines, ONLY: ram, id, gf, xlam, &
& wavblue, wavred, wavcon, wavcon1, &
& nlamb_old, nlambox_old, xlamold, &
& opacgrid, opalgrid, &
& opal_aux, sum_th, sum_nth, &
& fracth, fracnth, &
& calcopal, fgrid, vanreg, tradj_fg, u0, wexp, &
& clufac, clufac_forb, vanreg1, nsubinter, &
& cutoff_stark, gamma_stark, gamma_hyd, coeff, nlambox_ly, & 
& opa_eff_aux, opa_eff_grid

USE tcorr_var, ONLY: opttcor, qcbbu_m, qcbbd_m, dtqcbbu_m, dtqcbbd_m

USE nlte_opt, ONLY: opttcor_simple, optstark 

USE nlte_porvor, ONLY: epsi1,fic,tcl_fac_line,tcl_fac_cont

implicit none

real(sp), parameter :: cross = 0.02654

real(dp), parameter:: c_forbid= 8.63d-6/(0.5*1.3707d-7)
real(dp), parameter :: gfcut=1.d-5

integer(i4b), intent(in) :: nstart,npack,inemin
integer(i4b), dimension(ID_NDEPT), intent(in) :: nblock
integer(i4b), intent(inout) :: nlinesr,nlinesm

real(dp), dimension(ID_NDEPT), intent(in) :: betafac,collfac,clf,avoigtp,xne

integer(i4b) :: idum

real(sp) :: gfl,test,opalt,error

integer(i4b) :: irest,irec,j,l,low,k,nlamb,nlambox,ll,up,nindex,ni,jm,nj, &
&               nsubint,kmax,kk,nred,nblue

real(dp) :: dnue, opl, delta, xnell, tell, gammah, rat
real(dp) :: tcl,fac2

real(dp), dimension(ID_NDEPT) :: coll, opal, beta, n2n1, &
& qcul, qclu, dtqcul, dtqclu, van, op, ap, norm

integer(i4b), dimension(ID_NDEPT) :: kindex

character*1 :: met,metup

lines: do  l = nstart,npack

   if(ram.eq.'small') then
!  additional tests required  
     if(xlam(l).lt.wavblue) cycle lines
     if(xlam(l).gt.wavred) goto 100
   endif  

   nlambox = 1 + log(xlam(l)/wavblue)/wavcon

! consider only relevant intervals
! if opttcor, info for all transitions is needed
!   if(.not.opttcor .and. calcopal(nlambox).eq.0) cycle lines 
   if(.not.opttcor_simple .and. calcopal(nlambox).eq.0) cycle lines 
   k = id(l)/1000000
   if(jatom(k).eq.0) cycle lines
!   stop ' wrong atom in line list'

! Vers. 8.5 include all elements
!   if (indexel(k).gt.0) cycle lines  ! included in nlte
   if (.not.optstark .and. (k.eq.1 .or. k.eq. 2)) cycle lines  ! H/He

   irest = id(l) - k*1000000
   j = irest/100000

   if(calcion(k,j).eq.0) cycle lines !too low abundance

   irest = irest - j*100000
   low = irest/100
   up  = irest-low*100
   met=metall(k,j,low)
   gfl=cross*gf(l)

!  has been checked in sumopal_lte
!   if(low.ne.1.and.met.eq.' ') stop' inconsistent linelist'
   if(low.ne.1.and.met.eq.' '.and.jatom_full(k).eq.0) cycle lines
!
! test input data
!   if((k.lt.1 .or. k.gt.natom) .or. (j.lt.1 .or. j.gt.nion1) &
!&   .or. (low.lt.1)) then
!      print*,'error in input data / sumopal:'
!      print*,l,xlam(l),id(l),gf(l)
!      stop
!   endif

   irec = indrec(k,j)

   nlamb = 1 + log(xlam(l)/wavblue)/wavcon1
   nindex=mod(nlamb-1,nsubinter)+1 ! here was the bug

   if(nlamb_old.eq.0) nlamb_old=nlamb !for very first line
! the frequential integral over one interval has to yield chi-bar;
! thus, instead of
! dnue=vth(k)*2.d8/xlam(l)   
! we use
  dnue=vth8*2.d8/xlam(l) 
  if(nlamb.ne.nlamb_old) then
     nlamb_old=nlamb

     nlambox_old = 1 + log(xlamold/wavblue)/wavcon

     if(nlambox.ne.nlambox_old) then     

! build quantities per channel including continuum
       do ni=1,nsubinter
         sum_nth(ni,:)=opal_aux(ni,:)
! also for coupled channels, see notes
! effective opacity parameter added
! to recorrect, we need to multiply by 2.*vth8 (lambda remains in tau_sob)
         opa_eff_aux(ni,:) = opal_aux(ni,:)*2.*vth8*tcl_fac_line + &
              opacgrid(nlambox_old,:)*tcl_fac_cont           
         ! opa_eff_aux is here *total* tcl!
         opal_aux(ni,:)=opal_aux(ni,:)+opacgrid(nlambox_old,:) 
         !Add continuum to summed mean opacity 
         opa_eff_aux(ni,:) = opal_aux(ni,:) * & 
              (1.+fic*opa_eff_aux(ni,:))/(1.+opa_eff_aux(ni,:))
         !Now summed effective opacity in each SUB-bin 
       enddo

! sum up channels
       do ll=1,ID_NDEPT 
         if(nblock(ll).eq.1) then  !only one big channel present, should not happen so far!
           stop' nblock = 1!'
           do ni=2,nsubinter
              opal_aux(1,ll)=opal_aux(1,ll)+opal_aux(ni,ll)
              opa_eff_aux(1,ll)=opa_eff_aux(1,ll)+opa_eff_aux(ni,ll)   
               sum_th(1,ll)=  sum_th(1,ll)+  sum_th(ni,ll)  
              sum_nth(1,ll)= sum_nth(1,ll)+ sum_nth(ni,ll) 
           enddo
           opalgrid(nlambox_old,ll) = 1.d0/opal_aux(1,ll)
           opa_eff_grid(nlambox_old,ll) = 1.d0/opa_eff_aux(1,ll)
           fracth(nlambox_old,ll)=sum_th(1,ll)/opal_aux(1,ll)
           fracnth(nlambox_old,ll)=sum_nth(1,ll)/opal_aux(1,ll)
          
         else if(nblock(ll).ne.nsubinter) then
           ! jm coupled channels, neglecting truncated channels (at the red)
           ! carefully checked regarding opalgrid; if nblock low, large increase in opacity
           ! if strong lines present; maximum for nblock = 1: opalgrid => opalmax
           ! nblock = 2 and only 1 line: opalgrid => opac
           jm=nsubinter/nblock(ll)
           nsubint=nblock(ll)*jm
           do ni=1,nsubint,jm ! sum up coupled channels 
             do nj=1,jm-1  ! sum up subchannels
               opal_aux(ni,ll)=opal_aux(ni,ll)+opal_aux(ni+nj,ll)
               opa_eff_aux(ni,ll)=opa_eff_aux(ni,ll)+opa_eff_aux(ni+nj,ll)
               sum_th(ni,ll)= sum_th(ni,ll)+ sum_th(ni+nj,ll) 
               sum_nth(ni,ll)=sum_nth(ni,ll)+sum_nth(ni+nj,ll) 
!             print*,ll,nblock(ll),jm,ni,nj,nsubinter,nsubint
             enddo  
             opalgrid(nlambox_old,ll) = &
&              opalgrid(nlambox_old,ll)+1./opal_aux(ni,ll)
             opa_eff_grid(nlambox_old,ll) = &
&              opa_eff_grid(nlambox_old,ll)+1./opa_eff_aux(ni,ll)             
             fracth(nlambox_old,ll) = &
&              fracth(nlambox_old,ll)+sum_th(ni,ll)/opal_aux(ni,ll)
             fracnth(nlambox_old,ll) = &
&              fracnth(nlambox_old,ll)+sum_nth(ni,ll)/opal_aux(ni,ll)
           enddo
         else
           do ni=1,nsubinter ! all subchannels present, e.g. in photosphere
           opalgrid(nlambox_old,ll) = &
&            opalgrid(nlambox_old,ll)+1./opal_aux(ni,ll)
           opa_eff_grid(nlambox_old,ll) = &
&            opa_eff_grid(nlambox_old,ll)+1./opa_eff_aux(ni,ll)       
           fracth(nlambox_old,ll) = &
&            fracth(nlambox_old,ll)+sum_th(ni,ll)/opal_aux(ni,ll)
           fracnth(nlambox_old,ll) = &
&            fracnth(nlambox_old,ll)+sum_nth(ni,ll)/opal_aux(ni,ll)
           enddo
         endif
       enddo

       do ll=1,ID_NDEPT  ! normalize with respect to complete interval
         jm=nsubinter/nblock(ll)
         nsubint=jm*nblock(ll)
! opalgrid checked
         opalgrid(nlambox_old,ll)= &
&        float(nsubint)/float(jm**2)/opalgrid(nlambox_old,ll)-opacgrid(nlambox_old,ll)
! here we're NOT subtracting continuum in effective factor, see notes  
         opa_eff_grid(nlambox_old,ll)= &
&        float(nsubint)/float(jm**2)/opa_eff_grid(nlambox_old,ll)
! here's where final comp. of effective opacity parameter 
! opa_eff_grid = chi_eff/<chi> is done 
         opa_eff_grid(nlambox_old,ll) = opa_eff_grid(nlambox_old,ll)/ &
&        (opalgrid(nlambox_old,ll)+opacgrid(nlambox_old,ll))
! critical test for opa_eff_grid -- since working on total opacity, 
! it should never be negative 
         if (opa_eff_grid(nlambox_old,ll).gt.epsi1.or. &
&            opa_eff_grid(nlambox_old,ll).le.0.D0) then
! somewhat strange numerical precision thing: it indicates 1.0 precisely 
! but can hit stop anyway (at least in LTE sumopal routine coming below) 
               print*,opa_eff_grid(nlambox_old,ll),&
&              opalgrid(nlambox_old,ll)+opacgrid(nlambox_old,ll)
               print*,abs(opa_eff_grid(nlambox_old,ll)-1.d0)
               print*,opalgrid(nlambox_old,ll),opacgrid(nlambox_old,ll)
               stop' opa_eff > <opa>, sumopal_nlte_1'
         endif
         !now effective 
!         if (ll.eq.2) &
!&         print*,fgrid(nlambox_old),nlambox_old,ll,nsubinter,nsubint,nblock(ll), &
!&         opalgrid(nlambox_old,ll),opacgrid(nlambox_old,ll)
         if(opalgrid(nlambox_old,ll).lt.0.) opalgrid(nlambox_old,ll)=0. 
! fracth and fracnth hopefully correct (missing notes)
         fracth(nlambox_old,ll) = &
&              fracth(nlambox_old,ll)*float(jm)/float(nsubint)
         fracnth(nlambox_old,ll) = &
&              fracnth(nlambox_old,ll)*float(jm)/float(nsubint)
         if(fracnth(nlambox_old,ll).ge.1.d0 .or. & 
&          fracnth(nlambox_old,ll).lt.fracth(nlambox_old,ll)) then 
           print*,nlambox_old,ll,fracnth(nlambox_old,ll),fracth(nlambox_old,ll)
           stop' error in fracnth'
         endif  
! this is the crucial test to check whether averages are OK and 
! whether thermalization occurs for J = B (see notes)
! test the following condition: (<opal>+opac)*<fracth+fracnth> =: <opal> 

! no such test for opa_eff_aux, since for effective we're working with
! the total opacity, not separating line and continuum contributions 
         opalt=opalgrid(nlambox_old,ll)
         test=fracnth(nlambox_old,ll)*(opalt+opacgrid(nlambox_old,ll))         
         if(opalt.ne.0.and.test.ne.0.) then
           error=abs(1.-test/opalt)
           if(error.gt.1.d-3.and.opalt.gt.opacgrid(nlambox_old,ll)*1.d-3) &
&           print*,nlambox_old,ll,test,opalt,' sumopal_nlte: error1 in average'
!           stop' error in average'
         else           
!           test=abs(test-opalt)=test
! without abs, since all quantities > 0
           error=max(test,opalt)
           if(error.gt.opacgrid(nlambox_old,ll)*1.d-3) &
&           stop' sumopal_nlte: error2 in average'
         endif        
! finally
         fracnth(nlambox_old,ll) = fracnth(nlambox_old,ll)-fracth(nlambox_old,ll)
         enddo
         
       opal_aux=0.
       opa_eff_aux=0.  !JO not necessary, but does not hurt
       sum_th=0.
! don't need to reset sum_nth 

!       print*,nlambox_old,fgrid(nlambox_old),fracth(nlambox_old,42)
!       print*,nlambox_old,fgrid(nlambox_old), &
!&        fracth(nlambox_old,15),fracnth(nlambox_old,15) 
!       print*,nlambox_old,fgrid(nlambox_old), &
!&        fracth(nlambox_old,25),fracnth(nlambox_old,25) 
!       print*,nlambox_old,fgrid(nlambox_old), &
!&        fracth(nlambox_old,40),fracnth(nlambox_old,40) 

!test output for opaline.pro
!        print*,fgrid(nlambox_old),opalgrid(nlambox_old,20)/xne(20),opalgrid(nlambox_old,20)/opacgrid(nlambox_old,20)
!        print*,fgrid(nlambox_old),opalgrid(nlambox_old,24)/xne(24),opalgrid(nlambox_old,24)/opacgrid(nlambox_old,24)
!        print*,fgrid(nlambox_old),opalgrid(nlambox_old,30)/xne(30),opalgrid(nlambox_old,30)/opacgrid(nlambox_old,30)
!        print*,fgrid(nlambox_old),opalgrid(nlambox_old,40)/xne(40),opalgrid(nlambox_old,40)/opacgrid(nlambox_old,40)
              
       nlambox_old=nlambox
! calculate new vanreg
       u0=1.4388d8/(fgrid(nlambox)*te)
       vanreg=collfac*fgrid(nlambox)**3*(1.d0-exp(-u0))
       vanreg1=vanreg*c_forbid/fgrid(nlambox)
       vanreg=vanreg/(1.+vanreg)
   

!       if(opttcor) then
       if(opttcor_simple) then
! calculate new wexp = dilfac*exp(hnu/k*(1./te-1./trad))
         wexp=wion*exp(u0*(1.d0-te/tradj_fg(nlambox,:)))
! calculate new clufac (cf. subr. opacitl)
         clufac=collfac*6.6702082d15*exp(-u0)*0.5d8*hc2
         clufac_forb=clufac*c_forbid/fgrid(nlambox)
       endif
     endif
   endif                

   if(jatom_full(k).eq.1) then
      do ll=1,id_ndept
        if((j.lt.met_imin(k,ll).or.j.gt.met_imax(k,ll)).and.ll.lt.lwion1) then
! for lower depth points, we have lte!
          opal(ll)=0.
          beta(ll)=1.d20 
! can only happen here
        else
          opal(ll) = gfl*occng(irec,low,ll)/clf(ll) !corrected
          beta(ll)=betafac(ll)/(opal(ll)*xlam(l))
        endif
      enddo
   else
! for tests
!     if(low.ne.1.and.met.eq.' ') stop' met = empty'
     opal = gfl*occng(irec,low,:)/clf  !corrected
! must have occupation numbers
     beta=betafac/(opal*xlam(l))
   endif

   if(gf(l).le.gfcut) then
     van=vanreg1/gf(l)
     van=van/(1.+van)
   else
     van=vanreg
   endif
   
   where(beta.gt..99)
!  coll=0.  here was a bug (corrected April 2003), &
!           however in this case coll=vanreg small anyway
     coll=van  
   elsewhere  
     coll=van/(van*(1.d0-beta)+beta)  ! see comment in partit
   endwhere  

!---------------------------------------------------------
! although these quantities are not needed in case of calcopal = 0 (opttcor), &
! they are calculated to obtain consistency with nlambox_old if-block above
   opal = opal/dnue

! for tests; simulate missing FeIV lines
!   if (k.eq.26.and.j.eq.4.and.xlam(l).gt.911. .and. xlam(l).lt.1400.) opal=opal*5.

!---------------------------------------------------------
! inlined, to save time
! account for wings of Stark-broadened (quasi-)resonance lines (see notes)
! for ne > xnemin (ll >ienimin)

if(optstark .and. (low.eq.1.or.met.eq.'m')) then

! at first, prepare quantities and calculate max width

   if(k.ne.1) then
! (quasi-) resonance lines from metals, simple version, 
!  assuming avoigt/sqrt(pi) << 1

     do ll=inemin,ID_NDEPT
      op(ll)=opal(ll)*avoigtp(ll)/dnue !dnue includes factor 2, conventional normalization
!  until opal < cutoff_stark (0.2) * opac
      kindex(ll)=0.5*sqrt(op(ll)/cutoff_stark/opacgrid(nlambox,ll)+1.)
     enddo

   else
! hydrogen Lyman lines, "exact" formulation for all voigt-params (to avoid too many if-blocks
     if (met.eq.'m') stop' meta-stable levels in hydrogen-model found!'

     delta=vth8/vth(1) ! account for grid-spacing of sub-intervals

     do ll=inemin,ID_NDEPT
! calculation of gamma_h according to multi-linear fit

      xnell=xne(ll)
      if(xnell.lt.1.d12) xnell=1.d12
      if(xnell.gt.1.d16) xnell=1.d16

      tell=te(ll)/1000.
      if(tell.lt.10.) tell=10.
      if(tell.gt.60.) tell=60.

      xnell=log10(xnell)
      tell=log10(tell)

      gammah=coeff(0,up)+coeff(1,up)*tell+coeff(2,up)*tell**2+ &
&      coeff(3,up)*xnell+coeff(4,up)*xnell**2+coeff(5,up)*xnell**3

!      if(gammah.gt.log10(gamma_hyd(up))+0.5 .or. gammah.lt.log10(gamma_hyd(up))-0.5) then
!       print*,gammah,log10(gamma_hyd(up))
!       stop' wrong fit values for gamma_hyd'
!     endif
!     print*,xlam(l),ll,tell,xnell,gammah,log10(gamma_hyd(up))
          
      gammah=10.**gammah

      ap(ll)=avoigtp(ll)*wpi/dnue*gammah/gamma_stark
      norm(ll)=1.d0+(pihalf-atan(1.d0/ap(ll)))/(wpi*delta)
      opal(ll)=opal(ll)/norm(ll) ! additional normalization
      op(ll)=opal(ll)/(2.d0*delta*wpi)

!  until opal < cutoff_stark (0.2) * opac
      rat=ap(ll)*(2.d0/tan(cutoff_stark*opacgrid(nlambox,ll)/op(ll))-1.d0)
      if (rat.lt.15.d0) then
      kindex(ll)=1  
      else
      kindex(ll)=0.5d0*sqrt(rat+1.d0)
      endif
!     print*,ll,ap(ll),opal(ll),opacgrid(nlambox,ll),op(ll),kindex(ll)

      enddo
   endif  !end preparation

   kmax=maxval(kindex(inemin:id_ndept))

! add wing contribution
   if(kmax.ge.2) then
! for tests
!     do ll=inemin,ID_NDEPT 
!        if(kindex(ll).ge.2.and.k.eq.1) then
!          write(*,111) ll,k,j,low,up,xlam(l),nindex,kindex(ll)
!        endif
!     enddo  
!111 format(5(I2,2X),F12.5,2(2X,I2))
  
     do ll=inemin,ID_NDEPT
! this formulation gives better results than the older one (shifting
! the line-centers), even though some wing contribution entering the
! neighbouring nlamboxes (both on the red and on the blue) is missing 
       if(kindex(ll).lt.2) cycle

       do kk=1,kindex(ll)
        nred =nindex+kk
        nblue=nindex-kk
  
        if(nred.gt.nsubinter.and.nblue.lt.1) exit

        if (k.ne.1) then
! metals 
          opl=op(ll)/dble(4*kk*kk-1)
        else
          opl=op(ll)*atan(2.d0/(1.d0+dble(4*kk*kk-1)/ap(ll)))
! hydrogen
        endif

        if(nred.le.nsubinter) then
          opal_aux(nred,ll)=opal_aux(nred,ll) + opl
          sum_th(nred,ll)=sum_th(nred,ll)+opl*coll(ll)
        endif
        if(nblue.ge.1) then
          opal_aux(nblue,ll)=opal_aux(nblue,ll) + opl
          sum_th(nblue,ll)=sum_th(nblue,ll)+opl*coll(ll)
        endif
       enddo

     enddo
   endif ! end addition of wings

endif ! Stark broadened lines

!---------------------------------------------------------

!core always
   opal_aux(nindex,:)=opal_aux(nindex,:) + opal(:) ! correctly normalized in both cases
   sum_th(nindex,:)=sum_th(nindex,:)+opal(:)*coll(:)
!   opl=opal(44)/opacgrid(nlambox,44)
!   if(xlam(l).gt.1000. .and. xlam(l).lt.2000. .and.opl.gt.0.01) &
!&  write(*,110) k,j,low,xlam(l),nindex1,opl

110 format(3(I2,2X),F10.5,2X,I2,2x,F10.5)


   if(met.eq.'m') then
     nlinesm=nlinesm+1
   else
     nlinesr=nlinesr+1
   endif

   xlamold=xlam(l)

!   if (.not. opttcor) cycle lines
   if (.not. opttcor_simple) cycle lines
!----------------------------------------------------------
! heating and cooling rates for all lines (including calcopal = 0)
! no special treatment for clumping required 
! cooling rate
    if(up.eq.0) then
      metup=' '
    else
      metup=metall(k,j,up)
    endif
    
    if (up.ne.0 .and. metup.eq.'m'.or.gf(l).lt.1.d-5) then
     qclu=occng(irec,low,:)*clufac_forb  ! occng = nl/gl
    else
     qclu=occng(irec,low,:)*clufac*gf(l) ! occng = nl/gl
    endif

! derivatives (see notes)
! Important: leave out any partials with respect to n!
     dtqclu=qclu*(u0-.5d0)/te 

     if (up.ne.0.and.metup.ne.' ') then

! NLTE value (nu/nl) times (nu/nl)* for lines  where both levels are present
       if(low.eq.1.and.metall(k,j,up).ne.' ') then
!for consistency in strong resonance lines (cf. subr. partit)
          n2n1=occng(irec,up,:)/occng(irec,low,:)*exp(hkl*ergion(k,j,up)/te)
       else   
          n2n1=occng(irec,up,:)/occng(irec,low,:)*exp(u0)
       endif
       qcul=qclu*n2n1
       dtqcul= -qcul*0.5d0/te 

     else
! approx. NLTE value (nu/nl) times (nu/nl)* for lines with lower level only
       n2n1=wexp*(1.d0-coll)+coll
! heating rate
       qcul=qclu*n2n1
! important: use following formulation for dtqcul 
       dtqcul=qclu*coll*u0/te - qcul*0.5d0/te 
     endif
     
     qcbbu_m(ll)=qcbbu_m(ll)+qclu(ll)
     qcbbd_m(ll)=qcbbd_m(ll)+qcul(ll)
     dtqcbbu_m(ll)=dtqcbbu_m(ll)+dtqclu(ll)
     dtqcbbd_m(ll)=dtqcbbd_m(ll)+dtqcul(ll)

enddo lines

! approximate first boxes after Lyman jump by values from reliable boxes
! slightly changed in V7.2.3; what we actually want to have is that
! the lyman jump is in box nlambox_ly-1
if (optstark) then
  if (lam_ly.eq.0.) stop' Optstark and no hydrogen: lam_ly = 0. Change approach!'
! note: this assumes that hydrogen ground levels are denoted by 'H11' and H21'
! otherwise, change in frescal
  nlambox_ly = 1 + log(lam_ly/wavblue)/wavcon + 1 !(the +1 ensures correct position)
  if (calcopal(nlambox_ly).ne.1) stop' box next to Lyman jump (redwards) not considered'
  if (fgrid(nlambox_ly-1) .gt. lam_ly) &
&   stop' problems with nlamboxes around Lyman jump'
endif

return

! in case, summation for last interval
100 nlambox_old = 1 + log(xlamold/wavblue)/wavcon

       do ni=1,nsubinter
         sum_nth(ni,:)=opal_aux(ni,:)
         opal_aux(ni,:)=opal_aux(ni,:)+opacgrid(nlambox_old,:)
       enddo

! sum up channels
       do ll=1,ID_NDEPT 
         if(nblock(ll).eq.1) then  !only one big channel present, should not happen so far
           stop' nblock = 1!'
           do ni=2,nsubinter
             opal_aux(1,ll)=opal_aux(1,ll)+opal_aux(ni,ll)
             opa_eff_aux(1,ll)=opa_eff_aux(1,ll)+opa_eff_aux(ni,ll)
             sum_th(1,ll)=  sum_th(1,ll)+  sum_th(ni,ll)  
             sum_nth(1,ll)= sum_nth(1,ll)+ sum_nth(ni,ll) 
           enddo
           opalgrid(nlambox_old,ll) = 1.d0/opal_aux(1,ll)
           opa_eff_grid(nlambox_old,ll) = 1.d0/opa_eff_aux(1,ll)
           fracth(nlambox_old,ll)=sum_th(1,ll)/opal_aux(1,ll)
           fracnth(nlambox_old,ll)=sum_nth(1,ll)/opal_aux(1,ll)
          
         else if(nblock(ll).ne.nsubinter) then ! jm coupled channels 
           jm=nsubinter/nblock(ll)
           nsubint=jm*nblock(ll)
           do ni=1,nsubint,jm ! sum up coupled channels 
             do nj=1,jm-1  ! sum up subchannels
               opal_aux(ni,ll)=opal_aux(ni,ll)+opal_aux(ni+nj,ll)
               opa_eff_aux(ni,ll)=opa_eff_aux(ni,ll)+opa_eff_aux(ni+nj,ll)
                sum_th(ni,ll)= sum_th(ni,ll)+ sum_th(ni+nj,ll) 
               sum_nth(ni,ll)=sum_nth(ni,ll)+sum_nth(ni+nj,ll) 
             enddo  
             opalgrid(nlambox_old,ll) = &
&              opalgrid(nlambox_old,ll)+1./opal_aux(ni,ll)
             opa_eff_grid(nlambox_old,ll) = &
&              opa_eff_grid(nlambox_old,ll)+1./opa_eff_aux(ni,ll)
             fracth(nlambox_old,ll) = &
&              fracth(nlambox_old,ll)+sum_th(ni,ll)/opal_aux(ni,ll)
             fracnth(nlambox_old,ll) = &
&              fracnth(nlambox_old,ll)+sum_nth(ni,ll)/opal_aux(ni,ll)
           enddo
         else
           do ni=1,nsubinter ! all subchannels present, e.g. in photosphere
           opalgrid(nlambox_old,ll) = &
&            opalgrid(nlambox_old,ll)+1./opal_aux(ni,ll)
           opa_eff_grid(nlambox_old,ll) = &
&            opa_eff_grid(nlambox_old,ll)+1./opa_eff_aux(ni,ll)
           fracth(nlambox_old,ll) = &
&            fracth(nlambox_old,ll)+sum_th(ni,ll)/opal_aux(ni,ll)
           fracnth(nlambox_old,ll) = &
&            fracnth(nlambox_old,ll)+sum_nth(ni,ll)/opal_aux(ni,ll)
           enddo
         endif
       enddo

       do ll=1,ID_NDEPT  ! normalize with respect to complete interval
! for optically thick treatment, see above
         jm=nsubinter/nblock(ll)
         nsubint=jm*nblock(ll)
         opalgrid(nlambox_old,ll)= &
&        float(nsubint)/float(jm**2)/opalgrid(nlambox_old,ll)-opacgrid(nlambox_old,ll)
         opa_eff_grid(nlambox_old,ll)= &
&        float(nsubint)/float(jm**2)/opa_eff_grid(nlambox_old,ll)
         opa_eff_grid(nlambox_old,ll) = opa_eff_grid(nlambox_old,ll)/ & 
&        ( opalgrid(nlambox_old,ll) + opacgrid(nlambox_old,ll) ) 
         if (opa_eff_grid(nlambox_old,ll).gt.epsi1.or.opa_eff_grid(nlambox_old,ll).le.0.d0) then 
               print*,opa_eff_grid(nlambox_old,ll),&
&              opalgrid(nlambox_old,ll)+opacgrid(nlambox_old,ll)
               print*,opalgrid(nlambox_old,ll),opacgrid(nlambox_old,ll)
               stop' opa_eff > <opa>!, sumopal_nlte_2'
         endif
         if(opalgrid(nlambox_old,ll).lt.0.) opalgrid(nlambox_old,ll)=0. 
         fracth(nlambox_old,ll) = &
&              fracth(nlambox_old,ll)*float(jm)/float(nsubint)
         fracnth(nlambox_old,ll) = &
&              fracnth(nlambox_old,ll)*float(jm)/float(nsubint)
         if(fracnth(nlambox_old,ll).ge.1.d0 .or. & 
&          fracnth(nlambox_old,ll).lt.fracth(nlambox_old,ll)) then 
           print*,nlambox_old,ll,fracnth(nlambox_old,ll)
           stop' error in fracnth'
         endif  
! finally
         fracnth(nlambox_old,ll) = fracnth(nlambox_old,ll)-fracth(nlambox_old,ll)
       enddo

!       print*,nlambox_old, &
!&        fracth(nlambox_old,30),fracnth(nlambox_old,30) &
       

return
end

!
!-----------------------------------------------------------------------
!
subroutine sumopal_lte(nt,nstart,npack,nlinesr,nlinesm,clf,avoigtp,xne,inemin_in)
!
! calculates summed line-opacities (assuming box car profiles of width
! 2 deltanuedop, which cancels out, see notes) within fgrid
!
! no velocity-field effects, since only needed for grad in lower atmosphere
!  
! alternative version accounting for line wings can be found in
! oldprog/nlte_approx_7.0_photlines_only.f90:
!  84% (const1) into central 2 channels (from int[exp(-x^2)] between -1 to +1) 
!  7% (const2)  into neighbouring 2x2 channels (from int[exp(-x^2)]
!     between -3 to -1 and between 1 to 3.

! Stark-broadening for (quasi-) resonance lines of metals and Hydrogen
!
! clumping included, optically thick treament in analogy to sumopal_nlte
!  
USE nlte_type
USE nlte_dim, ONLY: nd=>ID_NDEPT
USE fund_const, ONLY : pi, wpi=>sqpi, pihalf

USE nlte_var, ONLY: lam_ly

USE nlte_app, ONLY: jatom,calcion,te,vth8,indexel,indrec, &
& metall,occng,jatom_full,vth

USE nlte_lines, ONLY: ram, id, gf, xlam, &
& wavblue, wavred, wavcon, wavcon1, &
& nlamb_old, nlambox_old, xlamold, &
& opacgrid, opalgrid, opal_aux, calcopal, fgrid, nsubinter, &
& cutoff_stark, gamma_stark, gamma_hyd, coeff, nlambox_ly, & 
& opa_eff_aux, opa_eff_grid

USE nlte_opt, ONLY: optstark 

USE nlte_porvor, ONLY: epsi1,fic,tcl_fac_line,tcl_fac_cont

implicit none
integer(i4b), intent(in) :: nt,nstart,npack,inemin_in
integer(i4b), intent(inout) :: nlinesr,nlinesm

real(dp), dimension(nd), intent(in) :: clf, avoigtp, xne

real(sp), parameter :: cross = 0.02654

real(sp) :: gfl

integer(i4b) :: irest,irec,j,l,low,k,nlamb,nlambox,nindex,ni, &
&               ll,up,kmax,kk,nred,nblue,inemin

real(dp) :: dnue, opl, delta, xnell, tell, gammah, rat
real(dp) :: tcl,fac2

real(dp), dimension(nd) :: opal, op, ap, norm

integer(i4b), dimension(nd) :: kindex

character*1 :: met

inemin=inemin_in
if(optstark .and. nt.gt.inemin) inemin=nt

lines: do  l = nstart,npack

   if(ram.eq.'small') then
!  additional tests required  
     if(xlam(l).lt.wavblue) cycle lines
     if(xlam(l).gt.wavred) goto 100
   endif  

   nlambox = 1 + log(xlam(l)/wavblue)/wavcon   
   if(calcopal(nlambox).eq.0) cycle lines ! consider only relevant intervals

   k = id(l)/1000000

   if(jatom(k).eq.0) cycle lines
!   stop ' wrong atom in line list'

! as above 
!   if (indexel(k).gt.0) cycle lines  ! included in nlte
   if (.not.optstark .and. (k.eq.1 .or. k.eq. 2)) cycle lines  ! H/He

   irest = id(l) - k*1000000
   j = irest/100000

   if(calcion(k,j).eq.0) cycle lines !too low abundance

   irest = irest - j*100000
   low = irest/100
   up  = irest-low*100
   met=metall(k,j,low)
!   if(low.ne.1.and.met.eq.' ') stop' inconsistent line list'
!   if(low.ne.1.and.met.eq.' ') cycle lines
   if(low.ne.1.and.met.eq.' '.and.jatom_full(k).eq.0) cycle lines

   gfl=cross*gf(l)

! test input data
!   if((k.lt.1 .or. k.gt.natom) .or. (j.lt.1 .or. j.gt.nion1) &
!&   .or. (low.lt.1)) then
!      print*,'error in input data / sumopal:'
!      print*,l,xlam(l),id(l),gf(l)
!      stop
!   endif

   irec = indrec(k,j)

   nlamb = 1 + log(xlam(l)/wavblue)/wavcon1
   nindex=mod(nlamb-1,nsubinter)+1 ! here was the bug
   
   if(nlamb_old.eq.0) nlamb_old=nlamb !for very first line
! the frequential integral over one niterval has to yield chi-bar;
! thus, instead of
! dnue=vth(k)*2.d8/xlam(l)   
! we use
  dnue=vth8*2.d8/xlam(l) 

   if(nlamb.ne.nlamb_old) then
     nlamb_old=nlamb
     
     nlambox_old = 1 + log(xlamold/wavblue)/wavcon

     if(nlambox.ne.nlambox_old) then

       do ni=1,nsubinter
         opa_eff_aux(ni,nt:nd) = opal_aux(ni,nt:nd)*2.*vth8*tcl_fac_line(nt:nd)+ &
&            opacgrid(nlambox_old,nt:nd)*tcl_fac_cont(nt:nd)
         opal_aux(ni,nt:nd)=opal_aux(ni,nt:nd)+opacgrid(nlambox_old,nt:nd)
         !Add continuum to summed mean opacity 
         opa_eff_aux(ni,nt:nd) = opal_aux(ni,nt:nd) * & 
              (1.+fic(nt:nd)*opa_eff_aux(ni,nt:nd))/(1.+opa_eff_aux(ni,nt:nd))
         !Now summed effective opacity in each SUB-bin 
         opalgrid(nlambox_old,nt:nd) = &
&            opalgrid(nlambox_old,nt:nd)+1./opal_aux(ni,nt:nd)
         opa_eff_grid(nlambox_old,nt:nd) = &
&            opa_eff_grid(nlambox_old,nt:nd)+1./opa_eff_aux(ni,nt:nd)
         enddo
       opalgrid(nlambox_old,nt:nd)= &
&      float(nsubinter)/opalgrid(nlambox_old,nt:nd)-opacgrid(nlambox_old,nt:nd)
       opa_eff_grid(nlambox_old,nt:nd)= &
&      float(nsubinter)/opa_eff_grid(nlambox_old,nt:nd)
       
       do ll=nt,nd   
          !final effective opacity fraction opa_eff_grid = chi_eff/<chi> 
          opa_eff_grid(nlambox_old,ll) = opa_eff_grid(nlambox_old,ll)/&
&         ( opalgrid(nlambox_old,ll) + opacgrid(nlambox_old,ll) )
          if (opa_eff_grid(nlambox_old,ll).gt.epsi1.or.opa_eff_grid(nlambox_old,ll).le.0.d0) then
                print*,opa_eff_grid(nlambox_old,ll),&
&               opalgrid(nlambox_old,ll)+opacgrid(nlambox_old,ll)
                print*,abs(opa_eff_grid(nlambox_old,ll)-1.d0)
                print*,opalgrid(nlambox_old,ll),opacgrid(nlambox_old,ll)
                stop' opa_eff > <opa>, sumopal_lte_1'
          endif
          if(opalgrid(nlambox_old,ll).lt.0.) opalgrid(nlambox_old,ll)=0.     
       enddo
       opal_aux=0.
       opa_eff_aux=0. !JO not necessary, but does not hurt
       
     endif
   endif                

   opal(nt:nd)=gfl*occng(irec,low,nt:nd)/(clf(nt:nd)*dnue) ! corrected
! for tests; simulate missing FeIV lines
!   if (k.eq.26.and.j.eq.4.and.xlam(l).gt.911. .and. xlam(l).lt.1400.) opal(nt:nd)=opal(nt:nd)*5.

!---------------------------------------------------------
! inlined, to save time
! account for wings of Stark-broadened (quasi-)resonance lines (see notes)
! for ne > xnemin (ll >ienimin)

if(optstark .and. (low.eq.1.or.met.eq.'m')) then

! at first, prepare quantities and calculate max width

   if(k.ne.1) then
! (quasi-) resonance lines from metals, simple version, 
!  assuming avoigt/sqrt(pi) << 1

     do ll=inemin,nd
      op(ll)=opal(ll)*avoigtp(ll)/dnue !dnue includes factor 2, conventional normalization
!  until opal < cutoff_stark (0.2) * opac
      kindex(ll)=0.5*sqrt(op(ll)/cutoff_stark/opacgrid(nlambox,ll)+1.)
     enddo

   else
! hydrogen Lyman lines, "exact" formulation for all voigt-params (to avoid too many if-blocks
     if (met.eq.'m') stop' meta-stable levels in hydrogen-model found!'

     delta=vth8/vth(1) ! account for grid-spacing of sub-intervals

     do ll=inemin,nd
! calculation of gamma_h according to multi-linear fit

      xnell=xne(ll)
      if(xnell.lt.1.d12) xnell=1.d12
      if(xnell.gt.1.d16) xnell=1.d16

      tell=te(ll)/1000.
      if(tell.lt.10.) tell=10.
      if(tell.gt.60.) tell=60.

      xnell=log10(xnell)
      tell=log10(tell)

      gammah=coeff(0,up)+coeff(1,up)*tell+coeff(2,up)*tell**2+ &
&      coeff(3,up)*xnell+coeff(4,up)*xnell**2+coeff(5,up)*xnell**3

!      if(gammah.gt.log10(gamma_hyd(up))+0.5 .or. gammah.lt.log10(gamma_hyd(up))-0.5) then
!       print*,gammah,log10(gamma_hyd(up))
!       stop' wrong fit values for gamma_hyd'
!     endif
!     print*,xlam(l),ll,tell,xnell,gammah,log10(gamma_hyd(up))
          
      gammah=10.**gammah

      ap(ll)=avoigtp(ll)*wpi/dnue*gammah/gamma_stark
      norm(ll)=1.d0+(pihalf-atan(1.d0/ap(ll)))/(wpi*delta)
      opal(ll)=opal(ll)/norm(ll) ! additional normalization
      op(ll)=opal(ll)/(2.d0*delta*wpi)

!  until opal < cutoff_stark (0.2) * opac
      rat=ap(ll)*(2.d0/tan(cutoff_stark*opacgrid(nlambox,ll)/op(ll))-1.d0)
      if (rat.lt.15.d0) then
      kindex(ll)=1  
      else
      kindex(ll)=0.5d0*sqrt(rat+1.d0)
      endif
!      print*,ll,ap(ll),opal(ll),opacgrid(nlambox,ll),op(ll),kindex(ll),xnell

      enddo
   endif  !end preparation

   kmax=maxval(kindex(inemin:nd))

! add wing contribution
   if(kmax.ge.2) then
! for tests
!     do ll=inemin,nd 
!        if(kindex(ll).ge.2.and.k.eq.1) then
!          write(*,111) ll,k,j,low,up,xlam(l),nindex,kindex(ll)
!        endif
!     enddo  
!111 format(5(I2,2X),F12.5,2(2X,I2))
  
     do ll=inemin,nd
! this formulation gives better results than the older one (shifting
! the line-centers), even though some wing contribution entering the
! neighbouring nlamboxes (both on the red and on the blue) is missing 
       if(kindex(ll).lt.2) cycle

       do kk=1,kindex(ll)
        nred =nindex+kk
        nblue=nindex-kk
  
        if(nred.gt.nsubinter.and.nblue.lt.1) exit

        if (k.ne.1) then
! metals 
          opl=op(ll)/dble(4*kk*kk-1)
        else
          opl=op(ll)*atan(2.d0/(1.d0+dble(4*kk*kk-1)/ap(ll)))
! hydrogen
        endif

        if(nred.le.nsubinter) then
          opal_aux(nred,ll)=opal_aux(nred,ll) + opl
        endif
        if(nblue.ge.1) then
          opal_aux(nblue,ll)=opal_aux(nblue,ll) + opl
        endif
       enddo

     enddo
   endif ! end addition of wings

endif ! Stark broadened lines

!---------------------------------------------------------

!core in any case
   opal_aux(nindex,nt:nd)=opal_aux(nindex,nt:nd) + opal(nt:nd)

   if(met.eq.'m') then
     nlinesm=nlinesm+1
   else
     nlinesr=nlinesr+1
   endif

   xlamold=xlam(l)

enddo lines

! approximate first boxes after Lyman jump by values from reliable boxes
! slightly changed in V7.2.3; what we actually want to have is that
! the lyman jump is in box nlambox_ly-1
if (optstark) then
  if (lam_ly.eq.0.) stop' Optstark and no hydrogen: lam_ly = 0. Change approach!'
! note: this assumes that hydrogen ground levels are denoted by 'H11' and H21'
! otherwise, change in frescal
  nlambox_ly = 1 + log(lam_ly/wavblue)/wavcon + 1 !(the +1 ensures correct position)
  if (calcopal(nlambox_ly).ne.1) stop' box next to Lyman jump (redwards) not considered'
  if (fgrid(nlambox_ly-1) .gt. lam_ly) &
&   stop' problems with nlamboxes around Lyman jump'

!  print*,nlambox_ly,calcopal(nlambox_ly),wavblue,wavcon
!  print*,lam_ly,fgrid(nlambox_ly-1),fgrid(nlambox_ly),fgrid(nlambox_ly+1)
endif


return

! in case, summation for last interval
100 nlambox_old = 1 + log(xlamold/wavblue)/wavcon

do ni=1,nsubinter
    opa_eff_aux(ni,nt:nd) = opal_aux(ni,nt:nd)*2.*vth8*tcl_fac_line(nt:nd)+ &
&            opacgrid(nlambox_old,nt:nd)*tcl_fac_cont(nt:nd)
    opal_aux(ni,nt:nd)=opal_aux(ni,nt:nd)+opacgrid(nlambox_old,nt:nd)
    !Add continuum to summed mean opacity 
    opa_eff_aux(ni,nt:nd) = opal_aux(ni,nt:nd) * & 
&            (1.+fic(nt:nd)*opa_eff_aux(ni,nt:nd))/(1.+opa_eff_aux(ni,nt:nd))
    !Now summed effective opacity in each SUB-bin 
    opalgrid(nlambox_old,nt:nd) = &
&            opalgrid(nlambox_old,nt:nd)+1./opal_aux(ni,nt:nd)
    opa_eff_grid(nlambox_old,nt:nd) = &
&            opa_eff_grid(nlambox_old,nt:nd)+1./opa_eff_aux(ni,nt:nd)
enddo

opalgrid(nlambox_old,nt:nd)= &
&  float(nsubinter)/opalgrid(nlambox_old,nt:nd)-opacgrid(nlambox_old,nt:nd)
opa_eff_grid(nlambox_old,nt:nd)= &
&  float(nsubinter)/opa_eff_grid(nlambox_old,nt:nd)

do ll=nt,nd   
   opa_eff_grid(nlambox_old,ll) = opa_eff_grid(nlambox_old,ll)/& 
&  ( opalgrid(nlambox_old,ll) + opacgrid(nlambox_old,ll) )   
   if (opa_eff_grid(nlambox_old,ll).gt.epsi1.or.opa_eff_grid(nlambox_old,ll).le.0.d0) then  
         print*,opa_eff_grid(nlambox_old,ll),&
&        opalgrid(nlambox_old,ll)+opacgrid(nlambox_old,ll)
         print*,opalgrid(nlambox_old,ll),opacgrid(nlambox_old,ll)
         stop' opa_eff > <opa>, sumopal_lte_2'
   endif
   if(opalgrid(nlambox_old,ll).lt.0.) opalgrid(nlambox_old,ll)=0.
enddo

!
return
end
!
!-----------------------------------------------------------------------
!
! PACKAGE FOR SOLUTION OF COMPLETE RATEEQUATIONS
!
!-----------------------------------------------------------------------
!
subroutine select(lteopt)

! define which atoms should be treated via complete rate equations
!
! and find out which ions have to be considered 
!
USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT 
USE nlte_var, ONLY: almost_converged
USE nlte_app, ONLY: nion,fracmin,abund,fjk,natom,jatom,jatom_full,indexel, &
& jmax,met_imin,met_imax

implicit none

logical, intent(in) :: lteopt
integer(i4b) :: k,j,js,im,im1,im2
integer(i4b), dimension(1) :: imax,imax1,imax2

real(dp) :: fjkmax
real(dp), dimension(nion) :: f

!------------------------------------------------------------------
if(lteopt) then
! select atoms 
jatom_full(6)=1
jatom_full(7)=1
jatom_full(8)=1
jatom_full(10)=1
jatom_full(12)=1
jatom_full(14)=1
jatom_full(15)=1
jatom_full(16)=1
jatom_full(18)=1
!jatom_full(20)=1 !Ca, in case
jatom_full(26)=1
jatom_full(28)=1

do k=1,natom
  if (jatom_full(k).eq.1.and.jatom(k).eq.0) &
&   stop' inconsistent definition of jatom_full(1)'
! Vers. 8.5 calculate all elements
!  if (jatom_full(k).eq.1.and.indexel(k).gt.0) jatom_full(k)=0
enddo

return
!------------------------------------------------------------------
else
!
! don't change imin and imax for metals when close to convergence
if(almost_converged.and.maxval(met_imax).ne.0) return
! select ions
! hydrogen
met_imin(1,:)=1
met_imax(1,:)=1

!helium
met_imin(2,:)=1
met_imax(2,:)=2

!rest
met_imin(3:natom,:)=0
met_imax(3:natom,:)=0

print*,' subroutine select(nlte) called'
print*
do k = 3,natom
  if(jatom_full(k) .eq. 0) cycle 

!changed by JO (14.11.2012)
  fjkmax=maxval(fjk(k,1:jmax(k),:))
  if(abund(k)*fjkmax.lt.fracmin) cycle ! irrelevant species

  do js = 1,id_ndept     
    f=fjk(k,:,js)
    f(jmax(k)+1:nion)=0.
    imax=maxloc(f)
    im=imax(1)
    imax1=maxloc(f,mask=f.lt.f(im))
    im1=imax1(1)
    imax2=maxloc(f,mask=f.lt.f(im1))
    im2=imax2(1)
    met_imin(k,js)=min0(im,im1,im2)
    met_imax(k,js)=max0(im,im1,im2)
!    print*,k,js,met_imin(k,js),met_imax(k,js)
!    print*,f
    if(met_imin(k,js).eq.0) stop' met_imin = 0' 
    if(met_imax(k,js).gt.jmax(k)) stop' met_imax > jmax' 
    if(met_imax(k,js)-met_imin(k,js).ne.2) then
      if(met_imax(k,js)-met_imin(k,js).gt.3) &
&     stop' something really wrong with met_imin, met_imax'
      im1=met_imax(k,js)
      im2=met_imin(k,js)
      if(f(im1).lt.f(im2)) then 
         met_imax(k,js)=met_imax(k,js)-1
      else   
         met_imin(k,js)=met_imin(k,js)+1
      endif
      if(met_imax(k,js)-met_imin(k,js).ne.2) &
&     stop' problems with met_imin, met_imax could not be cured'   
    endif 
  enddo
enddo

print*
print*,' subroutine select: met_imin and met_imax defined'
print*
return

endif
end

!
!----------------------------------------------------------------------
!
subroutine indexfrelines
!
! calculated freq. index for line transitions of selected elements
! (including DR stabilizing lines) 
!  
USE nlte_type
USE nlte_var, ONLY: ifre,fre
USE nlte_app, ONLY: nrec,jatom_full,kel,irbb,ilin,met_rbb, &
&                   levdr,irdr,met_rdr

USE nlte_opt, ONLY: optdr

implicit none
integer(i4b) :: n,jj,ii,i,ind,istart,istartold,ndr,kk,icompdr
real(dp) :: lam,energy,frec,diffe,diffe1,energyold

! indices for line transitions
energyold=1.d100
istartold=2

do n=1,nrec
  if(jatom_full(kel(n)).eq.0) cycle

    jj=irbb(n)-1
    do ii=1,ilin(n)
      lam=met_rbb(jj+ii)%wave 
      energy=1.d8/lam
      if(energy.gt.fre(ifre).or.energy.lt.fre(1)) then
        ind=0
        goto 10
      endif
      istart=2
      if(energy.gt.energyold) istart=istartold
      freloop: do i=istart,ifre
          frec=fre(i)
          if(energy.le.frec) then
            diffe =abs(energy-frec)
            diffe1=abs(energy-fre(i-1))
            if (diffe.le.diffe1) then
              ind=i
            else
              ind=i-1
            endif  
            energyold=energy
            istartold=i
            exit freloop
          endif
      enddo freloop
10      met_rbb(jj+ii)%index=ind
!      print*,lam,ind
    enddo 
enddo

if(.not.optdr) return

! indices for DR stabilizing lines
energyold=1.d100
istartold=2

drloop: do n=1,nrec
  if(jatom_full(kel(n)).eq.0) cycle drloop
  if(levdr(n).eq.0) cycle drloop
  if(irdr(n).eq.0) stop' error in DR-index -- netmat'
  ndr=irdr(n)  
    do kk=1,levdr(n)
     icompdr=met_rdr(ndr)%ncomp
     do jj=1,icompdr
       energy=met_rdr(ndr)%ener
       if(energy.gt.fre(ifre).or.energy.lt.fre(1)) then
         ind=0
         goto 20
       endif
       istart=2
       if(energy.gt.energyold) istart=istartold
       freloop1: do i=istart,ifre
          frec=fre(i)
          if(energy.le.frec) then
            diffe =abs(energy-frec)
            diffe1=abs(energy-fre(i-1))
            if (diffe.le.diffe1) then
              ind=i
            else
              ind=i-1
            endif  
            energyold=energy
            istartold=i
            exit freloop1
          endif
      enddo freloop1
20    met_rdr(ndr)%index=ind
      ndr=ndr+1  
     enddo
   enddo
enddo drloop  

return
end
!
!----------------------------------------------------------------------
!
subroutine ionis_full_lte(ntemp,xne)
!----------------------------------------------------------------------
!  calculation of LTE ionisation stratification for selected elements
!  in complete atmosphere 
!
!----------------------------------------------------------------------
USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT
USE fund_const, ONLY: c2 => hkl, saha
USE nlte_app, ONLY: nwne, nion, natom, jatom_full, jmax, xion, upart, fjk, te, lwion 

implicit none

integer(i4b),intent(in) :: ntemp
real(dp),dimension(nwne),intent(in) :: xne

real(dp), dimension(0:nion) :: psi,prod

integer(i4b) :: j, js, k, l
real(dp) :: ci, trad1, ratlog, summ, uratio, fjkold

ci=0.5*saha

!
! 1. calculation of partition functions
! -------------------------------------
!
call partit_full_lte(ntemp)

! 2. calculation of f_jk = N_jk/N_k
! ---------------------------------
radius: do js = ntemp, nwne

! psi(j) := log (N_j+1/N_j)

  atom:  do k = 1,natom
     if(jatom_full(k) .eq. 0) cycle atom

     do j = 1,jmax(k)
       trad1=te(js)
       uratio = upart(k,j+1,js)/upart(k,j,js)
       ratlog = -log(xne(js)) + 1.5*log(trad1) &
&        - log(ci) - c2*xion(k,j)/trad1
       psi(j) = log(uratio) + ratlog
       prod(j) =0.
     enddo
       
     summ = 0.
     psi(0) = 0.
     prod(0) = 0.        
            
     do j = 0,jmax(k)
        do l = 0,j
          prod(j) = prod(j) + psi(l)
        enddo
        summ = summ + exp(prod(j))
     enddo

     do j = 1,jmax(k)+1
       fjkold=fjk(k,j,js)
       fjk(k,j,js) = exp(prod(j-1)) / summ
! consisteny check: fjk should be identical with previous value calculated
! in IONIS for l=LWION,ND
! in the outer region, both values should be fairly equal, but sometimes
! larger differences are possible (up to 50%) 
       if(js.ge.lwion) then
         if(abs(1.-fjk(k,j,js)/fjkold).gt.1.d-13) then
           print*,k,j,js,fjkold,fjk(k,j,js)
           stop' fjk from IONIS and IONIS_FULL_LTE not equal in inner region'
         endif
       endif
     enddo

  enddo atom
enddo radius

end
!
!----------------------------------------------------------------------
!
subroutine partit_full_lte(ntemp)
!----------------------------------------------------------------------
!  calculation of LTE partition function for subroutine ionis_full_lte 
!
!---------------------------------------------------------------------- &
!  
USE nlte_type
USE fund_const, ONLY: c2 => hkl
USE nlte_app, ONLY: nwne, natom, jatom_full, nrec, jmax, &
& kel, iel, ielev, occng, gstat, ergion, upart, te, met_imin, met_imax

implicit none

integer(i4b), intent(in) :: ntemp
integer(i4b) :: i, j, js, k, lev, jm
real(dp) :: e1, el, et

do js = ntemp,nwne
    do k = 1,natom
!JO Jan. 2016 
      if(jatom_full(k) .eq. 0) cycle
      upart(k,jmax(k)+1,js) = gstat(k,jmax(k)+1,1)
    enddo
enddo

rec: do i = 1,nrec
         k = kel(i)
         if(jatom_full(k) .eq. 0) cycle
         j = iel(i)
         e1=ergion(k,j,1)
         
! ground state contribution

         do js = ntemp,nwne
           upart(k,j,js) = gstat(k,j,1) 
         enddo

         levlte: do lev = 2,ielev(i)

! excitation energy of considered levels
               el=ergion(k,j,lev)-e1
               do js=ntemp,nwne
                  et = c2*el/te(js)
                  occng(i,lev,js) = exp(-et)
               enddo
         enddo levlte      

!... and calculate the partition function for record i
part:    do lev = 2,ielev(i)
            do js = ntemp,nwne
             upart(k,j,js) = upart(k,j,js)+gstat(k,j,lev)*occng(i,lev,js)
            enddo
         enddo part
enddo rec

return
end
!
!----------------------------------------------------------------------
!
subroutine rateeq_met(xne,dvdr,vr,dilfac,clf,velo)
!
! nlte solution for metals, including clumping
! as in nlte.f90, heating/cooling rates are considered inside clumps
! thus remain uncorrected
!
USE nlte_type
USE nlte_dim, ONLY: id_ndept
USE nlte_var, ONLY: lwion1,xnelte,tradj,ifre,fre,almost_converged
USE nlte_xrays, ONLY: optxray

USE nlte_app, ONLY: te, nwne, lwion, natom, jatom_full, &
& met_imin, met_imax, nion, indrec, ielev, occng, gstat, occ1, nlev, nrec, &
& occnglte, occ1lte, name, fjk, occ1old, occngold, &
& abund, xmuee, summas

USE tcorr_var, ONLY: opttcor,error_met,qcbbu_m,qcbbd_m,dtqcbbu_m,dtqcbbd_m, &
& qcbfr_m,qcbfi_m,dtqcbfr_m,dtqcbfi_m,qrbfr_m,qrbfi_m,dtqrbfr_m,dtqrbfi_m


USE nlte_lines, ONLY: metconv1

USE nlte_porvor, ONLY: fic,tcl_fac_line,tcl_fac_cont

implicit none

real(dp),dimension(nwne),intent(in) :: xne,dvdr,vr,dilfac,clf,velo
integer(i4b) :: ll,k,ilow,imax,mn,n,ilev,j,i,lev,kk
integer(i4b), dimension(nion) :: noff

real(dp), dimension(3*nlev+1) :: blev0, blevel  ! occ/g, occ
real(dp), dimension(natom) ::  errork
real(dp), dimension(natom,id_ndept) :: devfjkmax
real(dp), dimension(id_ndept) :: meandev,meandevsq

real(dp) :: xnerat,gs,depart,xni,xnk,xnistar,summ,sumj,sumfjk,errk1,errk2, &
&           fjkim1,occmax,bold,error,error1,summ1,aux,fjknew

real(dp) :: tcl,fac2

logical :: damped
integer(i4b) :: indi,irat,irat_aux

if(opttcor) then
   qcbfr_m=0.
   qcbfi_m=0.
   dtqcbfr_m=0.
   dtqcbfi_m=0.

   qrbfr_m=0.
   qrbfi_m=0.
   dtqrbfr_m=0.
   dtqrbfi_m=0.

   qcbbu_m=0.
   qcbbd_m=0.
   dtqcbbu_m=0.
   dtqcbbd_m=0.
endif

errork=0.

devfjkmax=0.
meandev=0.
meandevsq=0.

if(lwion1.lt.lwion) stop' something wrong with lwion1'

print* 
print*,' Rigorous NLTE treatment for elements (ions at lwion1-1 =',lwion1-1,')'
print*
! changed; now we assume LTE only for taur > 2 for selected elements
depthloop: do ll = 1,lwion1-1 ! LTE at greater depths 
! if this is changed here, corresponding changes in jbar_phot etc. are required as well

kloop: do k = 1,natom
  if(jatom_full(k).eq.0) cycle
          ilow = met_imin(k,ll)  
          imax = met_imax(k,ll)  
          if (imax.eq.0) then
            if(ilow.ne.0) stop' imax = 0 and ilow ne 0'
            cycle  
          endif
!define vector of old occupation numbers and offset-vector
          mn=0
          noff(ilow)=mn
          do j = ilow, imax
            n=indrec(k,j)
            ilev=ielev(n)
            if(occngold(n,1,ll).eq.0.) then   ! start, change of ionization stages
               blev0(noff(j)+1:noff(j)+ilev)=occng(n,1:ilev,ll)
! check consistency
            else
               blev0(noff(j)+1:noff(j)+ilev)=occngold(n,1:ilev,ll)
            endif
            mn=mn+ilev
            noff(j+1)=mn
          enddo
          mn = mn + 1
          if(occ1old(k,imax+1,ll).eq.0.) then
            blev0(mn)=occ1(k,imax+1,ll)/gstat(k,imax+1,1)          
          else
            blev0(mn)=occ1old(k,imax+1,ll)/gstat(k,imax+1,1)          
          endif

!-----setup of nlte equations and newton-raphson iteration
! attention: blev0  (INPUT)  is occ/gstat 
!            blevel(OUTPUT)  is occ
          call netmat_met(k,ilow,imax,ll,mn,noff,blev0(1:mn),blevel(1:mn), &
&           te,xne,velo,vr,xnelte(ll),dvdr(ll),vr(ll),dilfac(ll),clf(ll),lwion1-1, &
&           fic(ll),tcl_fac_line(ll),tcl_fac_cont(ll))
          
          do i=1,mn
            if(blevel(i).lt.0.d0) then
            print*,ll,k,j,i
            stop' blevel < 0 found'
            endif
          enddo

          summ=sum(blevel(1:mn))

          summ1=abund(k)*xmuee(ll)/summas*xne(ll) ! includes enhanced density anyway
          if(abs(1.-summ/summ1).gt.1.d-10) then
            print*,k,1.-summ/summ1
            stop' particle conservation violated (rateeq_met)'
          endif

          if(optxray.and.almost_converged) then
! damp changes. Typically, these are levels which oscillate 
          damped=.false.
          do j = ilow, imax + 1
            if(j.eq.imax+1) then
              ilev=1
            else  
              n=indrec(k,j)
              ilev=ielev(n)
            endif
            do i=1,ilev
              indi=noff(j)+i
              bold=blev0(indi)*gstat(k,j,i)
              if(bold.eq.0.d0) then
                print*,ll,k,j,i
                print*,' bold = 0 in rateeq_met'
                bold=blevel(indi)
!                stop' bold = 0 in rateeq_met'
              endif
              error1=1.-bold/blevel(indi)
              error=abs(error1)
              if(error.gt.0.05) then  ! log = -1.3
!large changes damped using geometrical mean
                blevel(indi)=sqrt(blevel(indi)*bold)
!                if(error1.gt.0) then
!                    blevel(indi)=bold/0.975
!                else
!                    blevel(indi)=bold/1.025
!                endif
                write(*,15) ll,k,j,i,error1
15              format(' L = ',i3,': update of level ',3(2x,i3),2x,f10.5,' damped')
                damped=.true.

              else if(error.gt.0.025) then 
!smaller changes damped to 1.5% (note: convergeqnce at 1%)
                  if(error1.gt.0) then
                    blevel(indi)=bold/0.985
                else
                    blevel(indi)=bold/1.015
                endif
!                write(*,15) ll,k,j,i,error1
!15              format(' L = ',i3,': update of level ',3(2x,i3),2x,f10.5,' damped')
                damped=.true.
              endif  

            enddo
          enddo
! renormalize level occupation
          if(damped) then
            summ1=sum(blevel(1:mn))
            error=summ/summ1
            if(error.gt.1.05.or.error.lt.0.95) then
              print*,error
!              stop' too large renormalization of damped corrections'
              print*,' large renormalization of damped corrections!!!!!!'
            endif
            blevel(1:mn)=blevel(1:mn)*error           
!            print*,' level occup. renormalized by factor',error
          endif

          endif! damping
          
          
          do j = ilow, imax
            occ1(k,j,ll)=blevel(noff(j)+1)
          enddo
          occmax=maxval(occ1(k,ilow:imax,ll))

          do j = ilow, imax
            n=indrec(k,j)
            ilev=ielev(n)
            errk1=abs(1.-occ1old(k,j,ll)/occ1(k,j,ll))
! if in outer wind and error not from major ion, discard error
            if (occ1(k,j,ll).ne.occmax .and. velo(ll).gt.50d0) then
              if(errk1 .lt. 0.03) errk1=0.
            endif  
            errork(k)=max(errork(k),errk1)
!            print*,'errork ',ll,k,ilow,imax,j,errk1,occ1(k,j,ll)

! for tests
            occ1old(k,j,ll)=occ1(k,j,ll) ! for next iteration

            sumj=sum(blevel(noff(j)+1:noff(j)+ilev))
!            if(k.eq.26) print*,ll,j,fjk(k,j,ll),sumj/summ

! NOTE: fractions here only w.r.t. ilow...imax+1!
!            print*,'old',ll,k,j,fjk(k,j,ll),sumj/summ
! check consistency of approx. and refined method for main ions 
            fjknew=sumj/summ
            aux=fjk(k,j,ll)/fjknew
            aux=(1.d0-aux)/(1.d0+aux)
            if(j.eq.ilow) then            
               devfjkmax(k,ll)=aux
            else
               if(fjknew.gt.fjk(k,j-1,ll)) devfjkmax(k,ll)=aux
            endif 
!            if(ll.eq.1) print*,k,j,fjk(k,j,ll),fjknew,devfjkmax(k,ll)
            fjk(k,j,ll)=fjknew
            occng(n,1:ilev,ll)=blevel(noff(j)+1:noff(j)+ilev)/gstat(k,j,1:ilev)

            occngold(n,1:ilev,ll)=occng(n,1:ilev,ll) !for next iteration

!            if(k.eq.26.and.ll.eq.1) then
!              do lev=1,ilev
!              print*,j,lev,gstat(k,j,lev),occngold(n,lev,ll)/occngold(n,1,ll)
!            enddo
!            endif
          enddo
   
          occ1(k,imax+1,ll)=blevel(mn)

!          if(k.eq.26) print*,ll,imax+1,fjk(k,imax+1,ll),blevel(mn)/summ

          fjkim1=blevel(mn)/summ 
          fjk(k,imax+1,ll)=fjkim1  !update

! to discared purely numerical problems
          errk2=0.
!          if(fjkim1.gt.1.d-12) then
!JO new precision (Oct. 2014)
          if(fjkim1.gt.1.d-6) then
            errk2=abs(1.-occ1old(k,imax+1,ll)/occ1(k,imax+1,ll))
            errork(k)=max(errork(k),errk2)
          endif

!          print*,'errork ',ll,k,ilow,imax,imax+1,errk2,occ1(k,imax+1,ll)

          occ1old(k,imax+1,ll)=occ1(k,imax+1,ll)  ! for next iteration

!for tests  
!          xnerat=xne(ll)/xnelte(ll)
!          do j=ilow, imax
!            n=indrec(k,j)
!            ilev=ielev(n)
!            do i=1,ilev
!            gs=gstat(k,j,i)
!            xni=occngold(n,i,ll)*gs
!            xnk=occ1old(k,j+1,ll)   ! includes stat. weight
!            xnistar = xnk*occnglte(n,i,ll)*gs/occ1lte(k,j+1,ll)*xnerat  
!            depart = xni/xnistar  
!            print*,ll,k,j,i,depart
!            enddo
!          enddo
!          print*

! check error in ionization fractions (due to additional, approximated stages)
! particularly if there are X-rays, a couple of ions might be outside
! considered range.
! Note that this range is estimated from the very approx. treatment (subr. ionis)
!
     sumfjk=sum(fjk(k,:,ll))-1.d0
     if(abs(sumfjk).gt.5.d-3) then
       print*
       write(*,fmt = &
&      "(' !warning! additional approx. ion(s) outside ilow, imax+1:',2x,i3,2x,i3,2x,e10.4)"), &
&      k,ll,sumfjk
!       print*,fjk(k,:,ll)
     endif

     if(ll.eq.lwion1-1) &
&    print*,' ',name(k),' from ',ilow,' to ',imax,'(+1) max. error (all depths) = ',errork(k)
     end do kloop
end do depthloop  
error_met=maxval(errork)
print* 
print*,' max. error for above elements = ',error_met
print*

metconv1=.false.
if(error_met.lt.0.01) metconv1=.true.

!compare approx. and refined method
print*,' mean deviation of ionization fractions approx. method/almost exact method'
print*,' for selected elements, main ionization stage (range from -1 to +1)'
do ll=1,lwion-1
     irat=0
     do k=1,natom
        if(devfjkmax(k,ll).eq.0.d0) cycle
        meandev(ll)=meandev(ll)+devfjkmax(k,ll)
        meandevsq(ll)=meandevsq(ll)+devfjkmax(k,ll)**2
        irat=irat+1          
     enddo

     if(irat.ne.sum(jatom_full)) then
! element might have not been considered, because of too low abundance
! (see subr. select)
       do k=1,natom
        if(jatom_full(k).eq.0) cycle
        ilow = met_imin(k,ll)  
        imax = met_imax(k,ll)  
        if (imax.eq.0.and.ilow.eq.0) irat_aux=irat+1
       enddo
       if(irat_aux.ne.sum(jatom_full)) stop' error in irat'
     endif
     
     meandev(ll)=meandev(ll)/irat
     aux=sqrt((meandevsq(ll)-irat*meandev(ll)**2)/float(irat-1))
     meandevsq(ll)=aux
     print*,ll,irat,meandev(ll),' +/-',meandevsq(ll)
enddo
print*

return
end
!
!----------------------------------------------------------------------
!
subroutine netmat_met(k,ilow,imax,ll,mn,noff,blev0,blevel, &
&   tend,xnend,velond,vrnd,xnelte,dvdr,vr,dilfac,clf,lastl, &
&   fic,tcl_fac_line,tcl_fac_cont)
!
! solution of simplified rateeq. for atom k, ionization stages ilow,imax+1 and
! depth point ll, including clumping 
!
! note: in contrast to comments in earlier versions, collisional ionization/ &
!       recombination IS treated (Cik) ...
  
! attention: blev0  (INPUT)  is occ/gstat 
!            blevel(OUTPUT)  is occ

USE nlte_type
USE fund_const, ONLY: hk,pi,hkl,hc2,akb,sigmae,amh,e2mc2,hh

USE nlte_var, ONLY: tradj1,tradj,xj,xxk,alo,opac,ifre,fre,lthin,lwion1

USE nlte_app, ONLY: nwne,nion,nlev,indrec,ielev,ibfup,occ1lte,gstat, &
& icol,icbb,met_cbb, &
& ilin,irbb,met_rbb,occnglte,abund,xmuee,summas, &
& indexopa,alpha,xion,ergion,metall,vth,indexel, &
& levdr,irdr,met_rdr

USE nlte_app, ONLY: louter, lstat, jbar_bg, alo_bg

USE tcorr_var, ONLY: opttcor,qcbbu_m,qcbbd_m,dtqcbbu_m,dtqcbbd_m, &
& qrbfr_m,qrbfi_m,dtqrbfr_m,dtqrbfi_m,qcbfr_m,qcbfi_m,dtqcbfr_m,dtqcbfi_m

USE nlte_opt, ONLY: optphotlines, optdr

USE nlte_xrays, ONLY: optxray, optauger, &
&               n_kedges, k_nmin, k_nmax, eth, z_k=>z, n_k=>n, aug_1_6

USE nlte_porvor, ONLY: epsi1,opa_eff_rat, opa_eff_rat_old

implicit none
integer(i4b), parameter :: ndim=3*nlev+1
real(dp), parameter :: gfcut=1.d-5
real(dp), parameter :: c_vanreg = 0.5*1.3707d-7 !average value since also
                                                !subordinate lines
real(dp), parameter :: c_tau = 1.d-8*0.02654
! conversion to Angstrom
real(dp), parameter :: c_aul = 6.6702082d15, c_bul=1.d-24/hc2 

real(dp), parameter :: c_blu=e2mc2/hh*4.d0*pi**2 

integer, intent(in) :: k,ilow,imax,ll,mn,lastl
integer, dimension(nion), intent(in) :: noff

real(dp),dimension(nwne),intent(in) :: tend,xnend,velond,vrnd

real(dp),intent(in) :: xnelte, dvdr, vr, dilfac, clf

real(dp), intent(in) :: fic,tcl_fac_line,tcl_fac_cont
 
real(dp), dimension(mn), intent(in) :: blev0
real(dp), dimension(mn), intent(out) :: blevel

integer(i4b), dimension(1) :: jm
integer(i4b) :: j,n,ilev,i,n1,n2,m1,m2,jj,ii,jmax,index,ind,kk,lli,iup,nup, &
&               ndr,icompdr,nlcounter,indicator,itrans,l,ji
integer(i4b), dimension(ndim) :: indlud


real(dp), dimension(ndim,ndim) :: ratmat,rat2
real(dp), dimension(ndim-1,ndim-1) :: culmat
real(dp), dimension(ndim) :: f,flud,deltan

real(qp), dimension(ndim) :: aux

real(dp) :: te, xne, mustar, eddf1, aux1, aux2, err1, prob

real(dp) :: xnerat,nenk,xxlev,x1,x2,clu,cul,omega,slope,lam,gf,tlu,tul, &
& det,sdp,kh,alphai,edge,const,xi,tradold,xil,xel,culconst1,culconst2,om, &
& tau0,gup,tau_mult,xxj,eddf,taub,betacic,beta,tau,aul,bul,blu, &
& energy,bnue,nllow, &
& u0,qcul,qclu,dtqcul,dtqclu,x3,x4,x6,xni,xnk,xnistar, &
& cikconst,eel,cik,cikhnu, &
& integral,tradmean,xip,betap,sl,depart,opabf,betaopa,aloeff,alofac, &
& ener,flu,xlamc,afac,add,x1a
!x31,x41,ed,hckt,freq,xxjj,exk,hcfre3,aux0,aux1,aux2

real(dp) :: tcl,fac2


te=tend(ll)
xne=xnend(ll)

xnerat = xne/xnelte  
!
!------ initialitation of rate matrix and solution vector
!
f(1:mn)=0.d0
ratmat(1:mn,1:mn)=0.d0
!
!------ radiative/collisional b-f transitions
!
cikconst=xne*1.55d13/sqrt(te) ! Mihalas, p. 134

kh=1./hk
bfloop: do j = ilow,imax  
    n=indrec(k,j)
    ilev=ielev(n)
    nenk=xnerat/occ1lte(k,j+1,ll)   ! includes stat. weight

    do i=1,ilev

       index=indexopa(n,i) 
! specified for all edges in indexfre; note, that fre(index) slightly
! larger than edge itself
           
    
       alphai=alpha(n,i)*1.d-18
       if(alphai.eq.0.) stop' alphai = 0 in netmat_met!'
       edge=xion(k,j)-ergion(k,j,i)
       iup=ibfup(k,j,i)
       if(iup.ne.1) edge=edge+ergion(k,j+1,iup)
       xi=hkl*edge    !(hnu_i/k)   
       xel=xi/te      !hnu_i/(kTe)
       eel=exp(-xel)
! charge of ion (higher one) is just j (e.g., j=1 is neutral (z=0), +1 = j)
       cik=cikconst*alphai*eel/xel*min(0.1*j,0.3)

! preparation for all states
       n1 = i+noff(j)  
       xxlev = occnglte(n,i,ll)*gstat(k,j,i)*nenk ! (ni/nk)*
       if(iup.eq.1.or.j.eq.imax) then
! see subroutine opacitm and opacitc (nlte.f90)
         n2 = 1+noff(j+1)
       else
         nup=indrec(k,j+1)
         if(occnglte(nup,iup,ll).eq.0.) stop' error in excited level'
         xxlev = xxlev*occ1lte(k,j+1,ll)/(occnglte(nup,iup,ll)*gstat(k,j+1,iup)) 
         n2 = iup+noff(j+1)
       endif

       if(index.eq.0) then !eq.0 for edges larger than max(fre)
         x1 = 0.
         x2 = 0.
       else  
         const=8.d0*pi*edge**2*alphai*kh  !8pi alpha (nu_i/c)^2 * k/h
!------------------------------------------------------------------------
! in case and 
! for ground-states, ALI with tradj1 (i.e., without dilution factor)
! Note that so far there is no indication that ALI is needed
! Thus: commented out until further need
!         if(i.eq.1) then
!            kk=index
!            tradold=tradj1(ll,kk) 
! xni, xnk and xxlev already calculated above
!            xni=blev0(n1)*gstat(k,j,1)  
!            if(j.eq.imax) then
!              xnk=blev0(n2)*gstat(k,j+1,1)
!            else
!              xnk=blev0(n2)*gstat(k,j+1,iup)
!            endif
!            xnistar=xnk*xxlev
!            depart=xni/xnistar
! in accordance to assumption 
!            opabf=xnistar*alphai*(edge/fre(kk))**2*(depart-eel*fre(kk)/edge)/clf !corrected
! opac already corrected
!            betaopa=opabf/opac(ll,kk)
! assume maximum alo (constant over integration regime)
! delta Jnu = Jnu-ALO* * Bnu/b = Jnu *(1-ALO* * Bnu/(b*Jnu) 
!           = Jnu * alofac, which is integrated over frequency
!            aloeff=betaopa*alo(ll,kk)
!            alofac=1.-aloeff*bnue(1.d8/fre(kk),te)/(depart*xj(ll,kk))

!            if(alofac.le.0. .or. alofac.gt.1.1) then
! 1.1 for safety, might happen in first iterations
!               print*,' warning! problems with alofac in netmat_met:',ll,k,j
! no ALI
!               x1=const*tradold*exp(-xi/tradold)
!               x2=const*te*eel
!            else
! ALI 
!               x1=const*tradold*exp(-xi/tradold)*alofac
!               x2=const*te*eel*(1.-aloeff) ! stimulated emission neglected
!            endif
!------------------------------------------------------------------------
! in case of X-rays, calculate ground-state ionization from complete integrals 
!         if(optxray .and. edge.gt.440528. .and. i.eq.1) then
! the uppper statement give sometimes (very seldom) a better convergence, but
! is also sometimes somewhat worses than the expression below. For a final
! judgement which statement to use, more tests on large grids are required! 
! comment by JO (Dec. 2015): seems to be OK for standard treatment

!         if(optxray .and. i.eq.1) then
!JO changed Dec. 2016 (example: AlIV has edge just above 100 A,
!and index+1 is no longer defined  
         if(optxray .and. i.eq.1 .and. index+1.le.ifre) then
! tested: sufficient to consider ground-state
           tradold=tradj(ll,index)
             xil=xi/tradold !hnu_i/(kTrad)
             if(edge.le.fre(index+1)) then
               xip=xil*fre(index+1)/edge
             else
               xil=xil*fre(index)/edge
               xip=xil*fre(index+1)/fre(index)
             endif  
             integral=tradold*(exp(-xil)-exp(-xip))             
             if(integral.lt.0.) then
               print*,ll,k,j,i,edge,fre(index),fre(index+1),xil,xip
               stop' ground-state integral < 0 in netmat_met (1)'
             endif
             do kk=index+1,ifre-1
               tradmean=.5*(tradj(ll,kk)+tradj(ll,kk+1))
               xil=xi*fre(kk)/(tradmean*edge)
               xip=xil*fre(kk+1)/fre(kk)                
               add=tradmean*(exp(-xil)-exp(-xip))
               if(add.lt.0.) stop' ground-state integral < 0 in netmat_met (2)'
               integral=integral+add
! derived from exp(-xil) < 1.d-7 exp(-xi/trad) with 7*log(10.) = 16.
! contribution from rest neglible (including saftey for high trad) 
               if((xil-xi/tradold).gt.16.) then
                 if(add/integral.gt.1.d-5) stop' netmat_met: integral not converged'
                 exit
               endif
             enddo
             x1=const*integral*dilfac 
! tested; following hack not necessary
!             tradold=tradj(ll,index)
!             xil=xi/tradold !hnu_i/(kTrad)
!             x1a=const*tradold*exp(-xil)*dilfac
!             if(ll.eq.10) print*,k,j,x1,x1a
! to be consistent with older versions and to avoid convergence problems,
! use complete integral only if significant influence of X-rays 
!             x1=max(x1,x1a) 
             x2=const*te*eel            
             
! to account for varying radiation temperatures, if ground and metastable
! levels are situated on different sides of strong edges (e.g, OIII)
         else if(metall(k,j,i).eq.'m'.and.index.lt.indexopa(n,1)) then
! comprises case indexopa(n,1) = 0; in this case, index > indexopa(n,1), and
! nothing is done, as it should be 
             tradold=tradj(ll,index)
             xil=xi/tradold !hnu_i/(kTrad)
             kk=index
             if(edge.le.fre(index+1)) then
               xip=xil*fre(index+1)/edge
             else
               xil=xil*fre(index)/edge
               xip=xil*fre(index+1)/fre(index)
             endif  
             integral=tradold*(exp(-xil)-exp(-xip))             
             if(integral.lt.0.) then
               print*,ll,k,j,i,edge,fre(index),fre(index+1),xil,xip
               stop' integral < 0 in netmat_met'
             endif
             do kk=index+1,indexopa(n,1)-1
               tradmean=.5*(tradj(ll,kk)+tradj(ll,kk+1))
               xil=xi*fre(kk)/(tradmean*edge)
               xip=xil*fre(kk+1)/fre(kk)
               integral=integral+tradmean*(exp(-xil)-exp(-xip))
             enddo
             tradold=tradj(ll,kk)
             xil=xi*fre(kk)/(edge*tradold)
             integral=integral+tradold*exp(-xil)
             x1=const*integral*dilfac  
             x2=const*te*eel
! normal path (non ground and non meta-stable states)
! plus ground state if no ALI
         else
             tradold=tradj(ll,index)
             xil=xi/tradold !hnu_i/(kTrad)
             x1=const*tradold*exp(-xil)*dilfac
             x2=const*te*eel
         endif
       endif
! difference from here on; ionization to excited states allowed as well

       x1=x1+cik
       x2=x2+cik
       
       ratmat(n1,n1) = ratmat(n1,n1) - x1  
       ratmat(n1,n2) = ratmat(n1,n2) + x2*xxlev  
       ratmat(n2,n1) = ratmat(n2,n1) + x1  
       ratmat(n2,n2) = ratmat(n2,n2) - x2*xxlev  
    enddo

end do bfloop  

! DR stabilizing transitions
if (optdr) then
drloop: do j = ilow,imax  
    n=indrec(k,j)
    if(levdr(n).eq.0) cycle drloop
    if(irdr(n).eq.0) stop' error in DR-index -- netmat'
    ndr=irdr(n)  
levloop: do kk=1,levdr(n)

     x1=0.
     x2=0.

     i=met_rdr(ndr)%low 
!    everything w.r.t groundstate
     nenk=xnerat/occ1lte(k,j+1,ll)   ! includes stat. weight
     xxlev = occnglte(n,i,ll)*gstat(k,j,i)*nenk ! (ni/nk)*

     n1=i+noff(j)
     n2=1+noff(j+1)

     icompdr=met_rdr(ndr)%ncomp
     do jj=1,icompdr
! consistency check
       if (i.ne.met_rdr(ndr)%low) stop' something wrong in DR levels'
       index=met_rdr(ndr)%index
       if (index.eq.0) cycle
       ener=met_rdr(ndr)%ener
       flu= met_rdr(ndr)%flu

       xlamc=1./ener
       blu=c_blu*xlamc*flu
       afac=hc2/xlamc**3
       xil=xj(ll,index)
       eel=exp(-hkl*ener/te)       
       x1=x1+blu*xil
       x2=x2+eel*blu*(afac+xil)
       ndr=ndr+1  
     enddo

     ratmat(n1,n1) = ratmat(n1,n1) - x1  
     ratmat(n1,n2) = ratmat(n1,n2) + x2*xxlev  
     ratmat(n2,n1) = ratmat(n2,n1) + x1  
     ratmat(n2,n2) = ratmat(n2,n2) - x2*xxlev  

   enddo levloop

enddo drloop  
endif

! Auger-ionization
! in its present form (see table), only for K-shell ionization
! NOT only for two ejected electrons, but generalized
if (optxray.and.optauger) then
augerloop: do j = ilow,imax  
  if(j.lt.k_nmin .or. j.gt.k_nmax) cycle augerloop
  
  do l=1,n_kedges
    if(z_k(l).eq.k .and. n_k(l).eq.j) goto 10
  enddo
! element (or ion) not present in k_shell_Auger_data
  cycle augerloop

10 itrans=l
  if(eth(itrans).ge.fre(ifre)) cycle augerloop
  
  call intermepho_auger(x1,itrans,ll)   
     
  n=indrec(k,j)
  ilev=ielev(n)  
  do i=1,ilev
    n1=i+noff(j)   

! loop over all probabilities (1,2,3 etc. ejected electrons)    
   proba: do jj=j+1,imax+1    
       n2=1+noff(jj)
! e.g., if jj=j+1, we have jj-j = 1 ejected electrons, for
!          jj=j+2  we have jj-j = 2 ejected electrons (typical case), and so on
       prob=aug_1_6(itrans,jj-j)
       if(prob.eq.0.d0) cycle proba
! for tests
!       if(ll.eq.1) print*,k,j,itrans,jj,i,n1,n2,prob,x1
       ratmat(n1,n1) = ratmat(n1,n1) - x1*prob  
       ratmat(n2,n1) = ratmat(n2,n1) + x1*prob  
    enddo proba
  enddo  
enddo augerloop
endif !Auger

culconst1=xne*8.63d-6/sqrt(te) ! omega = 1.
culconst2=xne*c_vanreg/sqrt(te)

tau0=c_tau/(dvdr/3.+2./3.*vr)  ! at mue^2 = 1/3

iiloop: do j = ilow,imax  

    n=indrec(k,j)

! at first CUL for all possible transitions (omega = 1)
    ilev=ielev(n)
    do m2=2,ilev
      gup=gstat(k,j,m2)
      n2=noff(j)+m2
      do m1=1,m2-1
        n1=noff(j)+m1
        culmat(n1,n2)=culconst1/gup
      enddo
    enddo


! now RBB data and CLU via van Regemorter, if allowed
! for RBB (dependent on tau, we have to correct for clumping)
    jj=irbb(n)-1
    do ii=1,ilin(n)
      ji=jj+ii
      m1=met_rbb(ji)%low  
      m2=met_rbb(ji)%lup  
      lam=met_rbb(ji)%wave
      gf=met_rbb(ji)%gf
      ind=met_rbb(ji)%index
      nlcounter=met_rbb(ji)%index_jbar

      
! note: nlcounter can be zero, because of
!  i) ind=0
! ii) ionization above/below imax,ilow (ll=louter(k),lwion1-1) as defined in
! routine jbar_metals 

      gup=gstat(k,j,m2)

      n1=noff(j)+m1
      n2=noff(j)+m2
      
      if(ind.eq.0) then
         energy=1.d8/lam
         if(energy.gt.fre(1).and.energy.lt.fre(ifre)) &
&          stop' index array met_rbb%index inconsistent'
         xxj=bnue(lam,te)
         eddf=1./3.
      else
         xxj=xj(ll,ind)   
         eddf=xxk(ll,ind)/xxj
         if(ll.lt.lthin(ind)) then
!           mustar=1.d0-2.d0*dilfac (checked, correct, but irrelevant)
           aux1=velond(lthin(ind))/vrnd(lthin(ind))
           aux2=velond(ll)/vrnd(ll)
!JO: I don't understand the next comment any longer, might be a typo
!           mustar=1.d0-(aux2/aux1)**2*(1.d0-mustar**2) ! correction for lthin
           mustar=1.-(aux1/aux2)**2 !correction for lthin (correct)
           mustar=sqrt(mustar)
           eddf1=(1.d0-mustar**3)/3.d0/(1.d0-mustar)! checked, correct
!           if(eddf/eddf1.gt.2.) & ! eddf1 lower
!             print*,'problem with eddf',lam,ll,lthin(ind),eddf,eddf1
!           aux1=(eddf*dvdr+(1.-eddf)*vr)
!           aux2=(eddf1*dvdr+(1.-eddf1)*vr)
!           if(aux1/aux2.gt.3.) then ! problems only if aux2 lower
!             print*,'problem with taub',lam,ll,lthin(ind),eddf,eddf1,aux1,aux2
           eddf=eddf1
!           endif
         endif  
      endif  

! clumping included
      tau_mult=gf*lam*blev0(n1)/clf ! includes gstat, ind. emission negl.
! optically thick clumping as before 
      tcl = tau_mult*c_tau * tcl_fac_line
      fac2 = (1.+fic*tcl)/(1.+tcl)
      if (ind.ne.0) then
        if(opa_eff_rat(ll,ind).ne.opa_eff_rat_old(ll,ind)) &
&         stop' opa_eff_rat .ne. opa_eff_rat_old in netmet_met!!!' 
          fac2 = MIN ( fac2, opa_eff_rat(ll,ind) )
        endif
      if (fac2.gt.epsi1.or.fac2.le.0.d0) then 
            print*,fac2,opa_eff_rat(ll,ind)
            print*,tcl,ll,ind,lam
            stop' opa_eff > <opa> in netmat_met_1'
      endif
      tau_mult = tau_mult * fac2 
!effective quantity 
      taub=c_tau*tau_mult/(eddf*dvdr+(1.-eddf)*vr) ! at mue^2 = eddf
      if(taub.lt.1.d-5) then
          betacic=xxj  
      else
          betacic=(1.-exp(-taub))/taub*xxj
      endif

      tau=tau0*tau_mult  ! at mue^2 = 1/3
      if(tau.lt.1.d-5) then
          beta=1.d0
      else
          beta=(1.-exp(-tau))/tau
      endif
! influence of continuum negligible!!!!
!      if(tau.gt.1.d0.and.ind.ne.0) then
!          betap=opac(ll,ind)*vth(k)/(c_tau*tau_mult) ! corrected
!          if(betap.gt.100.d0 .or. (tau.gt.10d0 .and. betap.gt.tau)) then
!          print*,ll,k,j,m1,m2,lam,tau,betap
!           sl = strue(ll,ind)+xxj*(xne*sigmae*amh/clf+thomson_lines(ll,ind))/opac(ll,ind)   !corrected
!           betacic=betacic+sl ! U = 1 assumed
!           beta=beta+1.d0 ! U = 1 assumed
!          endif
!      endif

! ----------------------------------------------------------------------
! exact treatment of photospheric lines
      if(optphotlines .and. blev0(n1) .ne. 0.d0) then

! check treatment of line indicator = jbar_bg(nlcounter, lwion1)
! indicator = 1 indicates very weak line -> Sobo approach correct (check)
! indicator = 2 indicates that line has been treated with phot. RT
! indicator = 3 indicates that line has been treated with cmf RT
!from v7.2 on
      if(nlcounter.ne.0) then
      indicator=jbar_bg(nlcounter,lwion1)       

! check consistency with (Sobo-)results from jbar_bg, alo_bg
      if(ll.eq.louter(k) .and. ind.ne.0 .and. indicator.ne.3) then
          if (betacic.ne.jbar_bg(nlcounter,ll)) then 
           print*,k,j,ii,ind,lam,betacic,jbar_bg(nlcounter,ll)       
            stop' error in betacic'
          endif
          if (1.d0-beta.ne.alo_bg(nlcounter,ll)) then
            print*,k,j,ii,ind,lam,1.d0-beta,alo_bg(nlcounter,ll)       
            stop' error in beta'
          endif
      endif
      endif

! final consistency check
      if(ll.gt.louter(k) .and. ind.ne.0) then
        if (nlcounter.eq.0) then
!          print*,k,j,ii,ll,lam,ind
          stop' nlcounter unexpectedly = 0'
! still, there is the chance that ll > lwion-1
        endif  
      endif

! in case, use photospheric/cmf values: jbar-alo*sline and 1.-alo
      if(nlcounter.ne.0) then

        if(indicator.eq.0) then
! check, do nothing (use SA-quantities)
          if (ll.gt. louter(k)) stop' indicator = 0 (netmat)'

        else if (indicator.eq.2) then
! use "exact" quantities from louter(k)+1 ... lwion-1
          if (ll .gt. louter(k)) then
!            if(ii.eq.1) then 
!            write(*,101) k,j,ii,ll,beta,betacic,1.-alo_bg(nlcounter,ll),jbar_bg(nlcounter,ll)
!            endif
            beta=alo_bg(nlcounter,ll)
            if(beta.eq.0.) stop' something rotten with beta1 (indicator=2)'
            beta=1.-beta ! alo is 1-beta
            betacic=jbar_bg(nlcounter,ll) ! Jbar-alo*sl 
          endif  

        else if (indicator.eq.3) then
! in contrast to previous approaches, we use ALL cmf-values.
             if (ll .eq. 4) then
! testing the the agreement between cmf and sobo at depth-point 4
! (to get rid of boundary effects)
                err1=abs(1.-beta/(1.-alo_bg(nlcounter,ll)))            
!                if (err1.gt.0.2) then
!if at all, the cmf quantity is larger; warning if larger by factor of 2.
                    if (err1.gt.0.5) then 
                    write(*,101) k,j,ii,ll,beta,betacic,1.-alo_bg(nlcounter,ll),jbar_bg(nlcounter,ll)
                    print*,' warning: SA and CMF not consistent at ll=4!!!'
                    print*   
                endif
              endif
              beta=alo_bg(nlcounter,ll)
              if(beta.eq.0. .and. ll.ne.1) then
                    write(*,101) k,j,ii,ll,beta,betacic,1.-alo_bg(nlcounter,ll),jbar_bg(nlcounter,ll)
                    stop' something rotten with beta1 (indicator=3)'
              endif
              beta=1.-beta ! alo is 1-beta
              betacic=jbar_bg(nlcounter,ll) ! Jbar-alo*sl  
          
        else
! check, do nothing (use SA-quantities)
          if (indicator.ne.1) stop' something wrotten with indicator'
        endif
      endif
101   format(4(i3,2x),4(e10.4,2x))        

      endif
! ----------------------------------------------------------------------

      aul=c_aul/lam**2*gf/gup
      bul=aul*c_bul*lam**3
      blu=gup*bul/gstat(k,j,m1)

      tlu = blu*betacic
      tul = bul*betacic + aul*beta
! for tests
!      if(ll.eq.35) print*,m1,m2,lam,gf,beta,betacic,aul,bul,tlu,tul
! update "allowed" collisions by van regemorter

      if(gf .gt. gfcut) then
        culmat(n1,n2)=culconst2/gup*gf*lam
      endif

!      if(j.eq.4.and.m1.eq.1.and.ll.eq.1) then
!         print*,m1,m2,lam,gf,beta,betacic,aul,bul,tlu,tul
!      endif
!
!------- coefficient matrix, updated for rbb rates
!
      ratmat(n1,n1) = ratmat(n1,n1) - tlu  
      ratmat(n1,n2) = ratmat(n1,n2) + tul  
      ratmat(n2,n1) = ratmat(n2,n1) + tlu  
      ratmat(n2,n2) = ratmat(n2,n2) - tul  
   enddo

! finally CLU where omega available 

    jj=icbb(n)-1
    do ii=1,icol(n)
      ji=jj+ii
      m1=met_cbb(ji)%low  
      m2=met_cbb(ji)%lup  
      omega=met_cbb(ji)%omega
      slope=met_cbb(ji)%slope
      om=omega + (te - 2.d4)/1.d4 * slope ! from adi
      om=max(omega/10.,om)
      n1=noff(j)+m1
      n2=noff(j)+m2
      culmat(n1,n2)=culconst1/gstat(k,j,m2)*om
    enddo


! now we can update the rate-matrix concerning all cbb rates
   do m1=1,ilev-1
     n1=noff(j)+m1
     nllow=1.d0/(occnglte(n,m1,ll)*gstat(k,j,m1))
     do m2=m1+1,ilev
        n2=noff(j)+m2
        cul=culmat(n1,n2)
!
!---note: no correction for ne necessary, since transition in same ion
!
        clu = cul*occnglte(n,m2,ll)*gstat(k,j,m2)*nllow ! (n/g)*g  
        if(clu.le.0.) stop
!
!------- coefficient matrix, updated for cbb rates
!
      ratmat(n1,n1) = ratmat(n1,n1) - clu  
      ratmat(n1,n2) = ratmat(n1,n2) + cul  
      ratmat(n2,n1) = ratmat(n2,n1) + clu  
      ratmat(n2,n2) = ratmat(n2,n2) - cul  
     enddo
   enddo

end do iiloop

!... and we are finished.

! now let's solve the rate-equations in the same spirit as for the detailed atoms
!
!---- newton-raphson for rate equations cont+lines
!
!---- leave out line with largest occ.num
!
jm=maxloc(blev0)  !includes gstat, should not matter
jmax=jm(1)

do i=1,mn
     ratmat(jmax,i)=1.d0
end do
        
f(jmax)=abund(k)*xmuee(ll)/summas*xne ! includes enhanced density anyway

flud(1:mn) = f(1:mn)  
rat2(1:mn,1:mn) = ratmat(1:mn,1:mn)  

!for tests
!if (k.eq.12.and.ll.eq.6) then
!  print*,mn
!  do i=1,mn
!    do j=1,mn
!      print*,i,j,ratmat(i,j)
!    enddo
!enddo      
!
!-----solution of rate equations
!
call ludcmp1(ratmat,mn,ndim,indlud,det)  
!
call lubksb(ratmat,mn,ndim,indlud,f)  
!
!-----------------------------------------------------------------
!-----one newton raphson improvement
!-----------------------------------------------------------------
!
! old version
!     do i = 1,mn  
!          sdp = -flud(i)  
!
!     high precision summation
!
!          do j=1,mn
!               aux(j)=rat2(i,j)*f(j)
!          end do
!
!          call sort1(mn,aux)     
!
!          do j=1,mn
!               sdp=sdp+aux(j)
!          end do
!          deltan(i) = sdp
!     end do

!new version
     aux(1:mn)=matmul(real(rat2(1:mn,1:mn),qp),real(f(1:mn),qp))-real(flud(1:mn),qp)
     deltan(1:mn)=aux(1:mn)

     call lubksb(ratmat,mn,ndim,indlud,deltan)  
     do i = 1,mn  
          blevel(i) = f(i) - deltan(i)  
     enddo

if(.not.opttcor) return
! update of heating and cooling rates for most important elements

! Vers. 8.5
! heating rates updated in nlte
if(indexel(k).gt.0) return



! cbf/rbf heating and cooling
! JO: in case update ground-state heating rates in accordance with ionis. rates

do j = ilow,imax  
    n=indrec(k,j)
    ilev=ielev(n)
    nenk=xnerat/occ1lte(k,j+1,ll)   ! includes stat. weight

    do i=1,ilev

       index=indexopa(n,i) 
! specified for all edges in indexfre; note, that fre(index) slightly
! larger than edge itself
           
       alphai=alpha(n,i)*1.d-18
       edge=xion(k,j)-ergion(k,j,i)
       iup=ibfup(k,j,i)
       if(iup.ne.1) edge=edge+ergion(k,j+1,iup)
       xi=hkl*edge  !(hnu_i/k)   
       xel=xi/te      !hnu_i/(kTe)
       eel=exp(-xel)
       cikhnu=cikconst*alphai*eel/xel*min(0.1*j,0.3)*edge*0.5*hc2 !*hnue

       if(index.ne.0) then !eq.0 for edges larger than max(fre)
       const=8.d0*pi*edge**2*alphai*kh*akb  !8pi alpha (nu_i/c)^2 * k/h * k
       xi=hkl*edge  !(hnu_i/k)   
       tradold=tradj(ll,index)
       xil=xi/tradold !hnu_i/(kTrad)
       x3=const*tradold*exp(-xil)*tradold*dilfac
       x4=const*     te*eel      *te
       x6=x4/te*(2.d0+xel)  !dx4/dT
! test for accuray of above approximation.
! Result: Although not completely exact, the total rates are fairly
!  well represented and of order of 1% of the H/He rates. Reason:
!  Although occupation numbers similar, integrals much smaller, since
!  proportional to exp(-hnu_ion/kT), where the lowest ionization edges
!  of metals lie around 500 A! 
       
!       x31=0.
!       x41=0.
!       ed=fre(index)
!       if(ed.lt.edge) stop' netmat_met: error in edge!'
!       hckt = hkl/te
!       do kk=index,ifre 
!          freq=fre(kk)
!          xxjj=xj(ll,kk)
!          exk = exp(-hckt*freq)  
!          hcfre3 = hc2*freq**3  
!          aux0 = alphai*(ed/freq)**2*(1.-ed/freq)
!          aux1 = aux0*xxjj
!          aux2 = aux0*(xxjj+hcfre3)*exk 
!          x31=x31+wfre(kk)*aux1
!          x41=x41+wfre(kk)*aux2
!        enddo
!        x31=4.*pi*x31         
!        x41=4.*pi*x41
       endif

       n1 = i+noff(j)  
       xni=blevel(n1)
       xxlev = occnglte(n,i,ll)*gstat(k,j,i)*nenk ! (ni/nk)*

       if(iup.eq.1.or.j.eq.imax) then
! see subroutine opacitm and opacitc (nlte.f90)
         n2 = 1+noff(j+1)
       else
         nup=indrec(k,j+1)
         if(occnglte(nup,iup,ll).eq.0.) stop' error in excited level'
         xxlev = xxlev*occ1lte(k,j+1,ll)/(occnglte(nup,iup,ll)*gstat(k,j+1,iup)) 
         n2 = iup+noff(j+1)
       endif

       xnk=blevel(n2)
       xnistar=xnk*xxlev

       qcbfr_m(ll)=qcbfr_m(ll)+xnistar*cikhnu
       qcbfi_m(ll)=qcbfi_m(ll)+xni*cikhnu
!      (dnistar/dT = -nistar*(1.5+u0)/te
!       dCik/dT    = -Cik   *(1.5-u0)/te
!      d(nistar*Cik)/dT = -(nistar*Cik)*3/te
! not rigorously checked so far
       dtqcbfr_m(ll)=dtqcbfr_m(ll)-xnistar*cikhnu*3.d0/te 
       dtqcbfi_m(ll)=dtqcbfi_m(ll)-xni    *cikhnu*3.d0/te
       
       if(index.ne.0) then
       qrbfr_m(ll)=qrbfr_m(ll)+xnistar*x4
       qrbfi_m(ll)=qrbfi_m(ll)+xni*x3
       dtqrbfr_m(ll)=dtqrbfr_m(ll)+xnistar*(x6-x4*(1.5d0+xel)/te)
       dtqrbfi_m(ll)=dtqrbfi_m(ll)-xni*x3*(1.5d0+xel)/te
!       if(k.eq.7) print*,ll,n1,n2,xni*x3-xnistar*x4,xni/xnistar
       endif
    enddo
enddo

!missing dr heating/cooling

! cbb heating/cooling
do j = ilow,imax  
    n=indrec(k,j)
    ilev=ielev(n)
    do m1=1,ilev-1
     n1=noff(j)+m1
     nllow=1.d0/(occnglte(n,m1,ll)*gstat(k,j,m1))
     do m2=m1+1,ilev
        n2=noff(j)+m2
        energy=(ergion(k,j,m2)-ergion(k,j,m1))*0.5*hc2 !hnue
        if(energy.eq.0.d0) then
          cycle
!          print*,n2,blevel(n2),n1,blevel(n1),cul,clu
!          print*,culmat(n1,n2),energy          
!          print*,k,j,m1,m2,ergion(k,j,m1),ergion(k,j,m2)
!          stop
        endif  
        cul=culmat(n1,n2)*energy
        clu = cul*occnglte(n,m2,ll)*gstat(k,j,m2)*nllow ! (n/g)*g  
        qclu= blevel(n1)*clu  ! occng = nl/gl
        qcul= blevel(n2)*cul
        u0=energy/(akb*te) 
! derivatives (see notes)
! Important: leave out any partials with respect to n!
        if(abs(1.d0-qclu/qcul).lt.0.001) then
! qclu/qcul = b_l/b_u; this condition is valid if we are close
! to LTE, since then the total partial should approach zero
          dtqclu= -qclu*.5d0/te 
!          dtqclu=qclu*(u0-.5d0)/te 
        else
! outside (non-LTE), we have to use to alternative formulation
! which increases the derivative and prevents too large and unstable corr.
          dtqclu=qclu*(u0-.5d0)/te 
        endif
        dtqcul= -qcul*0.5d0/te 
!        if(ll.ge.30.and.ll.le.40.and.k.eq.7.and.j.eq.4) then
!           print*,ll,j,m1,m2,1.d8/energy,qcul-qclu
!        endif
        qcbbu_m(ll)=qcbbu_m(ll)+qclu
        qcbbd_m(ll)=qcbbd_m(ll)+qcul
        dtqcbbu_m(ll)=dtqcbbu_m(ll)+dtqclu
        dtqcbbd_m(ll)=dtqcbbd_m(ll)+dtqcul

        if(ll.eq.lastl) then
! derivatives for cbb heating-cooling in LTE region, to obtain
! consistent rates;
! assumption: Qlu=Qul; neglect weak dependence of omega(T)
          do lli=lastl+1,nwne
            dtqcul=-occnglte(n,m2,lli)*gstat(k,j,m2)*cul*xnend(lli)/xne* &
&             sqrt(te/tend(lli))*u0*te/(tend(lli)**2)
            dtqcbbd_m(lli)=dtqcbbd_m(lli)+dtqcul
          enddo
        endif   
     enddo 
    enddo
enddo

return  
end
!
!-----------------------------------------------------------------------
!
subroutine ludcmp1(a,n,np,indx,d)  

USE nlte_type
USE nlte_app, ONLY: nlev
implicit none
!
!     .. parameters ..
integer(i4b) ::  nmax  
parameter (nmax=3*nlev+1)  
!     ..
!     .. scalar arguments ..
real(dp) ::  d  
integer(i4b) ::  n,np  
!     ..
!     .. array arguments ..
real(dp) ::  a(np,np)  
integer(i4b) ::  indx(n)  
!     ..
!     .. local scalars ..
real(dp) ::  aamax,dum,summ  
integer(i4b) ::  i,imax,j,k  
!     ..
!     .. local arrays ..
real(dp) ::  vv(nmax)  
!     ..
!     .. intrinsic functions ..
intrinsic abs  
!     ..

d = 1.d0  

do i = 1,n  
     aamax = 0.d0  
     do j = 1,n  
          if (abs(a(i,j)).gt.aamax) aamax = abs(a(i,j))  
     end do
     vv(i) = 1.d0/aamax  
end do  

jloop: do j = 1,n  

     do i = 1,j - 1  
          summ = a(i,j)  

          do k = 1,i - 1  
               summ = summ - a(i,k)*a(k,j)  
          end do

          a(i,j) = summ  
     end do

     aamax = 0.d0  

     do i = j,n  
          summ = a(i,j)  

          do k = 1,j - 1  
               summ = summ - a(i,k)*a(k,j)  
          end do

          a(i,j) = summ  
          dum = vv(i)*abs(summ)  

          if (dum.ge.aamax) then  
               imax = i  
               aamax = dum  
          end if  
     end do

     if (j.ne.imax) then  

          do k = 1,n  
               dum = a(imax,k)  
               a(imax,k) = a(j,k)  
               a(j,k) = dum  
          end do

          d = -d  
          vv(imax) = vv(j)  
     end if  

     indx(j) = imax  

     if (a(j,j).eq.0.) stop ' matrix singular in ludcmp1'  

     if (j.ne.n) then  
          dum = 1.d0/a(j,j)  
          do i = j + 1,n  
               a(i,j) = a(i,j)*dum  
          end do
     end if  

end do jloop  

return  

end
!
!-----------------------------------------------------------------------
!
! PACKAGE FOR PHOTOSPHERIC LINE TRANSFER
!
!-----------------------------------------------------------------------
!
subroutine jbar_metals(xne,dvdr,vr,clf,velo)
! 
! line transfer for background elements: calculation of jbar and alo
! photosphere: by means of lambda-operator using freq. integrated exp. integrals
! wind:        using Sobolev approx. 
! interpolation for transition region
!
! remember: velo, vr, dvdr in absolute numbers
! dimensionless quantities are r1, velo1, dvdr1  

USE nlte_type
USE fund_const, ONLY: sigmae,amh,hc2
USE nlte_dim, ONLY: id_ndept

USE nlte_opt, ONLY: optphotlines_diff

USE run_once, ONLY: start_jbar_metals

USE nlte_var, ONLY: modnam, fpath, lwion1, opac, xj, xxk, lthin, &
& strue, thomson_lines, fre

USE nlte_var, ONLY: corrfc,sr,vmax,vsound,vdiv,ndiv=>ndiv_calc

USE nlte_app, ONLY: nwne, lwion, natom, jatom_full, &
& met_imin, met_imax, indrec, ielev, irbb, ilin, met_rbb, &
& gstat, occngold, occng, occnglte, vth, louter, lstat, &
& jbar_bg, alo_bg, temp=>te, lomin

USE phot_lines

USE nlte_porvor, ONLY: epsi1,fic,tcl_fac_line,tcl_fac_cont, &
& opa_eff_rat, opa_eff_rat_old

implicit none
integer(i4b), parameter :: nd=id_ndept

real(dp), parameter :: c_tau = 1.d-8*0.02654, hc2_8=hc2*1.d24

real(dp),dimension(nwne),intent(in) :: xne,dvdr,vr,clf,velo

integer(i4b), dimension(natom) :: ilow, imax

integer(i4b) :: i,k,j,n,ilev,jj,ii,ll,m1,m2,ind,lo,ls,nlines,nlcounter, &
&               nb, nt, nb1, nt1, nlcounter1, nlcounter_cmf, ji

real(dp), dimension(nd) :: opal, sline, betap, sc, ajlmean, alo, &
& opac1, etac

real(dp), dimension(nd), save :: r,r1,v1,vr1,dvdr1

real(dp) :: lam,gf,gup,occlow,occup,tau_mult,tau_mult1, &
&           vth1,vo,xxj,eddf,eddf1, &
&           aux1,aux2,mustar,tau,tau0,taub,betacic,beta,beta1,bnue, &
&           db, dt, betapmin, taulast, xlambda, srvmax

real(dp) :: tcl,fac2,fac1

logical first, flag, flagcmf, optdebug
data first/.true./
data optdebug/.false./

if(lwion1.lt.lwion) stop' something wrong with lwion1'
if(nd.ne.nwne) stop' nd ne nwne in jbar_metals'
print* 
print*,' Radiative transfer for bg. elements: calculation of Jbar and ALO'
print*

if (first) then
! read in tables
! only necessary if jbar_phot(integral method) called
! presently, we use the differential method, jbar_phot_diff
if (optphotlines_diff) goto 5

open (901, file = fpath//'kl_tables/k2atable.dat')  
open (902, file = fpath//'kl_tables/l2atable.dat')  
open (903, file = fpath//'kl_tables/l2table.dat')  
open (904, file = fpath//'kl_tables/l3atable.dat')  

read (901, * ) nb, nt  
read (902, * ) nb1, nt1  
read (903, * ) nb1, nt1  
read (904, * ) nb1, nt1  

if (nb.ne.nb1) stop ' error in nb(1)'  
if (nt.ne.nt1) stop ' error in nt(1)'  

if (nb.ne.nbdim) stop ' error in nb(2)'  
if (nt.ne.ntdim) stop ' error in nt(2)'  

read (901, * ) betamin, betamax, db  
read (901, * ) taumin, taumax, dt  
read (901, * ) betatab  
read (901, * ) tautab  

do i = 1, nb  
read (901, * ) (xk2atab (i, j), j = 1, nt)  
enddo  
do i = 1, nb  
read (902, * ) (xl2atab (i, j), j = 1, nt)  
enddo  
do i = 1, nb  
read (903, * ) (xl2tab (i, j), j = 1, nt)  
enddo  
do i = 1, nb  
read (904, * ) (xl3atab (i, j), j = 1, nt)  
enddo  

close (901)  
close (902)  
close (903)  
close (904)  

! calculate profile functions etc. for integration of exponential integrals

call xgrid

! calculate profile functions etc. for jbar_phot_diff
5 if (optphotlines_diff) call xgrid_diff

first=.false.
endif

! after each update
if(start_jbar_metals) then
! radius grid (in absolute units)

r=velo/vr

! dimensionless quantities
srvmax=sr/vmax
r1=r/sr
if(abs(r1(nd)-1.d0).gt.1.d-14) stop' error in radius grid'
v1=velo/vmax
vr1=vr*srvmax
dvdr1=dvdr*srvmax

! louter and lstat, in dependence of k

lomin=nd
do k = 1, natom
  if(jatom_full(k).eq.0) cycle

! this is the outermost point where the hydrostatic, pp rt-solution is valid
  if (vth(k).gt.vsound) then
! large vturb
    vth1=vsound
  else  
    vth1 = amin1 (vth(k),vdiv*vsound)  
  endif
! this is the innermost point where the Sobo-approx is valid
! n thermal widths above sonic point
  vo = vsound + 2.*vth(k) !factor 2 empirically  

  lo=0
  ls=0

  do ll = nd, 1, - 1  
   if (velo (ll) .gt. vo) then  
      lo = ll  
      exit  
   endif  
  end do  

  do ll = nd, 1, - 1  
   if (velo (ll) .gt. vth1) then  
      ls = ll + 1  
      exit  
   endif  
  enddo  

  if(lo.eq.0) stop' louter not found'
  if(ls.eq.0) stop' lstat not found'

  if(ndiv.eq.0) stop' ndiv = 0 in jbar_metals'
  ls=min0(ls,ndiv)

! v10.1: next line commented; might happen for very thick winds, &
!        shouldn't matter then  
!  if (ls.ge.lwion1-1) stop' lstat(k) ge lwion1-1'
  if (lo.ge.ls)       stop' louter(k) ge lstat(k)'
  lomin=min0(lomin,lo)
! lomin no longer needed (but see below)

  louter(k)=lo
  lstat(k)=ls

enddo

start_jbar_metals=.false.
endif

!calculate dimension of jbar, alo and allocate
nlines=0

do k = 1,natom
   if(jatom_full(k).eq.0) cycle

   ilow(k) = minval(met_imin(k,louter(k):lwion1-1))  
   imax(k) = maxval(met_imax(k,louter(k):lwion1-1))  
!   print*,k,ilow(k),imax(k) 
!   do ll=louter(k),lwion1-1
!      print*,k,ll,met_imin(k,ll),met_imax(k,ll)       
!   enddo
   if (imax(k).eq.0) then
     if(ilow(k).ne.0) stop' imax = 0 and ilow ne 0'
     cycle  
   endif
   if(imax(k)-ilow(k) .gt. 3) then
! assuming a maximum of 5 stages in parallel
     if(imax(k)-ilow(k) .gt. 4) then
       print*,k,ilow(k),imax(k)
       stop' more than 5 stages in parallel'
     endif
   do ll=louter(k),lwion1-1
      print*,k,ll,met_imin(k,ll),met_imax(k,ll)       
   enddo
   print*,' more than 4 ionization stages present in lower atmosphere!'
   print*,' lower and uppermost only in Sobo'   
   endif

   do j=ilow(k),imax(k)
      n=indrec(k,j)
      nlines=nlines+ilin(n)
   enddo
enddo

print*,' ',nlines,' lines considered'
print*

if (allocated (jbar_bg)) deallocate(jbar_bg, alo_bg)

! lot of ram spent, but most simple way.
! actually, for phot. RT we would need only nlcounter1 * (lomin:lwion1) variables, &
!      and for cmf RT we would need only nlcounter_cmf * (1:lwion1) variables.
! Since we do not know nlcounter1 and nlcounter_cmf in advance, this would
! require some good guesses.
! Delayed until required.

! lwion1 for indicator
allocate(jbar_bg(nlines,1:lwion1),alo_bg(nlines,1:lwion1))


! prepare radiation transfer
! for cmf-tests
if (optdebug) open(200,file='OUT.IDL')
open(201,file=trim(modnam)//'/BG_LINES.CMF')

nlcounter=0
nlcounter1=0
nlcounter_cmf=0

!reset index at each iteration, to avoid leftovers from previous calcs.
met_rbb(:)%index_jbar=0


kloop: do k = 1,natom

    if(jatom_full(k).eq.0) cycle kloop
    if (imax(k).eq.0) cycle kloop
        
jloop: do j=ilow(k),imax(k)

    n=indrec(k,j)
    ilev=ielev(n)
    jj=irbb(n)-1

iiloop: do ii=1,ilin(n)
! opacities and soure functions
         ji=jj+ii
         m1=met_rbb(ji)%low  
         m2=met_rbb(ji)%lup  
         lam=met_rbb(ji)%wave
         gf=met_rbb(ji)%gf
         ind=met_rbb(ji)%index
         gup=gstat(k,j,m2)

! outside freq. range; will be approximated in Sobo -> subroutine netmat
         if (ind.eq.0) cycle iiloop
         xlambda=1.d8/fre(ind)

         nlcounter=nlcounter+1
         met_rbb(ji)%index_jbar=nlcounter

! all depth points, since cmf transfer required for strong lines
!--------------------------------------------------------

!         do ll=louter(k),nwne
         do ll=1,nwne

         if(occngold(n,1,ll).eq.0.) then   ! start, change of ionization stages
               occlow=occng(n,m1,ll)
! check consistency
         else
               occlow=occngold(n,m1,ll)
         endif

         if (occlow.eq.0.) then
! outside considered ionization
!             if(j.ge.met_imin(k,ll) .and. j.le.met_imax(k,ll)) &
!&              stop' routine jbar_metals: error in occng' 
             occlow=occnglte(n,m1,ll)
             if(occlow.eq.0.) &
&              stop' routine jbar_metals: error in occnglte'
             if (ll.lt.lstat(k)) occlow = occlow/100. ! to avoid wrong impact in transition region
         endif  

         if(occngold(n,1,ll).eq.0.) then   ! start, change of ionization stages
               occup=occng(n,m2,ll)
! check consistency
         else
               occup=occngold(n,m2,ll)
         endif

         if (occup.eq.0.) then
! outside considered ionization
!             if(j.ge.met_imin(k,ll) .and. j.le.met_imax(k,ll)) &
!&              stop' routine jbar_metals: error in occng' 
             occup=occnglte(n,m2,ll)
             if(occup.eq.0.) &
&              stop' routine jbar_metals: error in occnglte'
             if (ll.lt.lstat(k)) occup = occup/100. ! to avoid wrong impact in transition region
         endif  

         tau_mult1=gf*lam/clf(ll)
         if (occlow.gt.occup) then
           tau_mult=tau_mult1*(occlow-occup)  
         else 
           tau_mult=tau_mult1*occlow
         endif
           
         xxj=xj(ll,ind)   
         
         opal(ll)=tau_mult*c_tau/vth(k)
! optically thick clumping as before
         tcl = tau_mult*c_tau*tcl_fac_line(ll)          
         fac2 = (1.+fic(ll)*tcl)/(1.+tcl)
         if(opa_eff_rat(ll,ind).ne.opa_eff_rat_old(ll,ind)) &
           stop' opa_eff_rat .ne. opa_eff_rat_old in jbar_metals'
         fac2 = Min( fac2,opa_eff_rat(ll,ind) )
         if (fac2.gt.epsi1.or.fac2.le.0.d0) then 
               print*,fac2,opa_eff_rat(ll,ind),ll,ind
               stop'eff_opa > <opa> in jbar_metals_1'
            endif
         opal(ll) = opal(ll) * fac2 
! effective opacity 
         if(ll.lt.lwion1) then 
           if (occlow.gt.occup) then
             sline(ll)=hc2_8/lam**3/(occlow/occup-1.d0)
           else  
             sline(ll)=bnue(lam,temp(ll))
           endif
         else  
! to be consistent with radiation field.
! Note that lam might be not consistent with Delta E (due to packing), &
! since lam refers to the transition with the strongest gf value, &
! whereas the LTE occupation numbers are calculated according to the given
! energy levels. In other words, Bnue(lam) ne Sline (occlte).  
! Example: SiV, transition 19 to 14 (ii=323): lam = 2844, whereas Delta E
! corresponds to 3324.
!
! Note: maybe, we will need here even a change to lambda(ind)
             sline(ll)=bnue(lam,temp(ll))
         endif
         
         opac1(ll)=opac(ll,ind)
! this is the EFFECTIVE opac from previous iteration 
                  
         betap(ll)=opac1(ll)/opal(ll) ! corrected
! both opac, opal are corrected 
! caution: since not necessarily the same reduction, 
! (max tcl option for opal) ratio may not be preserved. 

!         sc(ll) = strue(ll,ind)+xxj*(xne(ll)*sigmae*amh/clf(ll)+ &
!&          thomson_lines(ll,ind))/opac(ll,ind)   !corrected
! since ONLY opac is corrected, need to back-correct 
         sc(ll) = strue(ll,ind)+xxj*(xne(ll)*sigmae*amh/clf(ll)+ &
&          thomson_lines(ll,ind))/opac(ll,ind) * opa_eff_rat(ll,ind)  !back-corrected
 
!         print*,k,j,ii,ll,opal(ll),sline(ll),betap(ll),sc(ll)

         etac(ll)=opac1(ll)*sc(ll)
         
!--------------------------------------------------------
!
! calculation of Sobolev quantity at louter (needed for interpolation)
! identical approach as in netmat (checked there) 
!

         if (ll.eq.louter(k)) then

           eddf=xxk(ll,ind)/xxj
           if(ll.lt.lthin(ind)) then
!             mustar=1.d0-2.d0*dilfac (checked, correct, but irrelevant)
             aux1=velo(lthin(ind))/vr(lthin(ind))
             aux2=velo(ll)/vr(ll)
!JO: I don't understand the next comment any longer, might be a typo
!             mustar=1.d0-(aux2/aux1)**2*(1.d0-mustar**2) ! correction for lthin
             mustar=1.-(aux1/aux2)**2 !correction for lthin (correct)
             mustar=sqrt(mustar)
             eddf1=(1.d0-mustar**3)/3.d0/(1.d0-mustar)!checked, correct
!            if(eddf/eddf1.gt.2.) & ! eddf1 lower
!               print*,'problem with eddf',lam,ll,lthin(ind),eddf,eddf1
!               aux1=(eddf*dvdr+(1.-eddf)*vr)
!               aux2=(eddf1*dvdr+(1.-eddf1)*vr)
!               if(aux1/aux2.gt.3.) then ! problems only if aux2 lower
!               print*,'problem with taub',lam,ll,lthin(ind),eddf,eddf1,aux1,aux2
             eddf=eddf1
!            endif
           endif    

! stimulated emission neglected, to be consistent with netmat
           tau_mult=gf*lam*occlow/clf(ll)
! again using max tcl 
           tcl = tau_mult*c_tau*tcl_fac_line(ll)          
           fac2 = (1.+fic(ll)*tcl)/(1.+tcl)
! indentity of opa_eff_rat and opa_eff_rat_old already checked above
           fac2 = Min( fac2,opa_eff_rat(ll,ind) )
           if (fac2.gt.epsi1.or.fac2.le.0.d0) then 
                 print*,fac2,opa_eff_rat(ll,ind),ll,ind
                 stop'eff_opa > <opa>, jbar_metals_2'
           endif
           tau_mult = tau_mult * fac2
! effective opacity 
           taub=c_tau*tau_mult/(eddf*dvdr(ll)+(1.-eddf)*vr(ll)) ! at mue^2 = eddf
           if(taub.lt.1.d-5) then
             betacic=xxj  
           else
             betacic=(1.-exp(-taub))/taub*xxj
           endif

           tau0=c_tau/(dvdr(ll)/3.+2./3.*vr(ll))  ! at mue^2 = 1/3
           tau=tau0*tau_mult  ! at mue^2 = 1/3

           if(tau.lt.1.d-5) then
             beta=1.
             beta1=0.5*tau ! expanded
           else
             beta=(1.-exp(-tau))/tau
             beta1=1.-beta
           endif

           jbar_bg(nlcounter,ll)=betacic
           alo_bg(nlcounter,ll)=1.-beta ! to be consistent with netmat
         endif ! sobo quantities
!--------------------------------------------------------

         enddo ! depth loop

!--------------------------------------------------------

! calculation of jbar and alo (including interpolation within louter, lstat)
! actually, jbar-alo*sline is calculated
!
! flag = .true.    -- jbar(phot) calculated         (implies flagcmf = .false. 
! flag = .false.   -- weak line, Sobo sufficient    (implies flagcmf = .false.)

! flagcmf = .true. -- very strong line, cmf required (implies flag= .false.)
         
         if(imax(k)-ilow(k) .gt. 3) then
           if (j.eq.ilow(k) .or. j.eq.imax(k)) then
! only Sobo
             flag=.false.
             goto 100
           endif  
         endif

         betapmin=minval(betap(lstat(k):lwion1-1))

         if (betapmin .gt. 1.d3) then
! tested; for beta > betapmin Sobo-solution OK
           flag = .false.
           flagcmf = .false.
         else  

!           write(*,10),'new line',k,j,ii,louter(k),lstat(k),lam,betapmin

! this is the differential method
           if (optphotlines_diff) then
             call jbar_phot_diff(ajlmean, alo, r, velo, temp, opal, sline, betap, &
&            opac1, sc, beta1, betacic, louter(k), lstat(k), xlambda, corrfc, &
&            flag, flagcmf)
           else
! this is the integral method
             call jbar_phot(ajlmean, alo, r, velo, opal, sline, betap, &
&            opac1, sc, beta1, betacic, louter(k), lstat(k), corrfc, taulast, &
&            flag, flagcmf)
           endif
           if(flag) then
! check for negative jbar
           do ll=louter(k),nd
              if(ajlmean(ll).lt.0.) then
              print*,'jbar-alo*sline < 0 (jbar_phot)'
              print*,ll,ajlmean(ll),alo(ll),sline(ll)
              write(*,10),'line-err',k,j,ii,louter(k),lstat(k),lam,betapmin
              print*,' needs to be calculated in cmf'
              flagcmf=.true.
              flag=.false.
!              stop' jbar-alo*sline < 0'
              endif
           enddo
           endif
         endif

         if(flagcmf) then
           nlcounter_cmf=nlcounter_cmf+1
!           write(*,11) flag,flagcmf,k,j,ii,louter(k),1.-beta,betacic
           write(201,10) 'cmf line',k,j,ii,louter(k),lstat(k),lam,betapmin
           if (flag) stop' something wrong in flag policy'
           call cmf_simple(ajlmean,alo,r1,v1,vr1,dvdr1,temp, &
&            opal,sline,opac1,etac,sc,vth(k),xlambda,corrfc,xxj,betapmin)
! check for negative jbar
           do ll=1,nd
              if(ajlmean(ll).lt.0.) then
                print*,'jbar-alo*sline < 0 (jbar_cmf)'
                print*,ll,ajlmean(ll),alo(ll),sline(ll)
                write(*,10),'line-err',k,j,ii,louter(k),lstat(k),lam,betapmin
                stop' jbar-alo*sline < 0'
              endif
           enddo

!           print*,k,j,ii,lam
!           do ll=1,nd
!             print*,ll,ajlmean(ll),alo(ll)
!           enddo  

           endif   

!---------------------------------------------
! for tests (comparison with external cmf-solution)
if (optdebug) then
          if(flagcmf) then
!         if(betapmin.lt.1000.) then
!         if(k.eq.18.and.j.eq.7.and.ii.eq.220) then
!          aux2 = rraic
           print*
           print*,'new line',k,j,ii,lam,betapmin
!           call cmf_ext(opal*vth(k),sline,opac1,etac,xlambda,k,aux2,xxj,flagcmf)
           if (.not.flagcmf) then
           write(200,12) 'new line',k,j,ii,louter(k),lam,betapmin,betacic/beta*aux2
!           else 
!           nlcounter_cmf=nlcounter_cmf+1
!           write(201,*) 'cmf line-transfer required',k,j,ii,lam,betapmin
           endif
         endif
endif
!---------------------------------------------

100      if (.not.flag .and. .not.flagcmf) then
! this is the indicator that line needs to be treated in Sobo
             jbar_bg(nlcounter,lwion1)=1.d0 
         else 

           nlcounter1=nlcounter1+1

           if(flagcmf) then           
! this is the indicator that line has been treated in cmf
             jbar_bg(nlcounter,lwion1)=3.d0 
! results from cmf_simple
             do ll=1,lwion1-1
! this quantity is jbar-alo*sline
              jbar_bg(nlcounter,ll)=ajlmean(ll)
              alo_bg(nlcounter,ll)=alo(ll)
            enddo         

           else
! final consistency check 
             if(.not.flag) stop' something really rotten with flag'
! this is the indicator that line has been treated in phot. approx.
             jbar_bg(nlcounter,lwion1)=2.d0 
! results from jbar_phot, interpolated until louter(k)
! per definition, sobo and RT quantities are identical at louter(k). Checked.
             do ll=louter(k)+1,lwion1-1
! this quantity is jbar-alo*sline
              jbar_bg(nlcounter,ll)=ajlmean(ll)
              alo_bg(nlcounter,ll)=alo(ll)
             enddo         
           endif
         endif
enddo iiloop

enddo jloop

enddo kloop

if (nlcounter.gt.nlines) stop' nlcounter > nlines'
print*
print*,'     nlcounter = ',nlcounter, ' (all lines from selected bg elements)'
print*,'    nlcounter1 = ',nlcounter1,' (lines from selected bg elements with RT)'
print*,' nlcounter_cmf = ',nlcounter_cmf,' (lines from selected bg elements with CMF-RT)'

if(optdebug) close(200)
close(201)

return


10 format(a9,2x,5(i3,2x),f10.2,2x,1(e9.3,2x))
11 format(2(L1,2x),4(i3,2x),2(e10.4,2x))        
12 format(a9,2x,4(i3,2x),f10.2,2x,2(e9.3,2x))

end
!
!-----------------------------------------------------------------------
!
subroutine jbar_phot(ajlmean, alo, r, velo, opal, sline, beta, opac, sc, &
&          ps1, betacic, louter, lstatin, corrfc, taulast, flag, flagcmf)
! 
! NOTE!!!
! this routine solves the R.T. down to ND. Actually, this is not necessary
! (lowermost point which needs to be considered is LWION1-1) 
! might be changed in future; due to speed of routine, presently not
! necessary.
! When this is changed, attention must be paid regarding the construction
! of the upper triangle of the Lambda matrix (iend etc)
  
! calculates jbar and alo from constructing the lambda operator via frequency
! integrated exponential integrals
!
! output is jbar-alo*sline and alo  
!  
! consistency to cmf-transport has been carefully checked
! between louter and lstat, an interpolation between static and Sobolev
! transfer is performed.
! test programs located in oldstuff_lrz/test_new 

! r and v in absolute units

! here, beta is opac/opal, and ps1 is 1.-beta(sob) at louter(kel)

! somewhere, there is a bug for alo(nd). it should be a factor of two larger
! than calculated here.  

USE nlte_type
USE nlte_dim,only: id_ndept
USE phot_lines, only: fac, neglect

implicit none
integer(i4b), parameter :: nd=id_ndept

real(dp), dimension(nd), intent(in) :: r, velo, opal, sline, beta, opac, sc
real(dp), dimension(nd), intent(out):: ajlmean, alo

integer(i4b), intent(in) :: louter, lstatin
real(dp), intent(in) :: ps1, betacic, corrfc

logical, intent(out) :: flag, flagcmf

real(dp), dimension(nd) :: taunue
real(dp), dimension(nd+1) :: z, tauz, taucz

real(dp), dimension(:,:), allocatable :: lamline, lamcont, betaq
real(dp), dimension(:), allocatable :: ibeta
integer(i4b), dimension(:), allocatable :: istart,iend

real(dp) :: del, dt, tau, tauzj, bi, taulast, bet, betq1, betq2, betqm, &
&           tau1, tau2, dt1, dt2, change, rat, e1, diff, &
&           dv, xouter1, xinner1, bb1, xouter2, xinner2, bb2,&
&           term1, term2, rstar, &
&           xk21,xl21, xk22, xl22, xk20, xl20, xl3a, aic, ajm, rraic

real(dp) :: xk2a, xk2a_simple, xl2

integer(i4b) :: ndstat, l, i, li, j, lj, icount1, icount2, is1, ie1, ii, jj, lstat

logical done

! for the complete range, to avoid edge effects
lstat=louter

ndstat = nd-lstat + 1  
!
ajlmean = 0.  
!
!tau-grid for all depth points
taunue (1) = 1.d-8  

do l = lstat, nd-1  
   i=l+1-lstat
   del = r (l) - r (l + 1)  
   dt = del * 0.5 * (opal(l) + opal(l+1))
   taunue(i+1)=taunue(i) + dt
end do  

!do i=1,ndstat
!  l=i+lstat-1
!  print*,i,l,taunue(i),beta(l)
!enddo
!print*,'new line',taunue(ndstat)
taulast=taunue(ndstat)

if (taulast.lt.1.) then
! line very weak, Sobo sufficient
  flag=.false.
  flagcmf=.false.
  return
endif  

if(taunue(3).gt.1.) then
! optically thick around louter, needs cmf transfer
  flag=.false.
  flagcmf=.true.
  return
endif  


! staggered grid and corresponding tau
z(1)=r(lstat)
do l=lstat+1,nd
  i=l+1-lstat
  z(i)=0.5*(r(l-1)+r(l))
enddo
z(ndstat+1)=r(nd)

tauz(1)=taunue(1)
taucz(1) = 0.
do i = 1, ndstat  
   l=i+lstat-1
   del = z (i) - z (i + 1)  
   dt = del * opal(l)
   tauz(i+1)=tauz(i) + dt
   taucz(i+1)=taucz(i) + del * opac(l)
end do  

!check consistency
if(abs(taunue(ndstat)-tauz(ndstat+1)).gt.1.d-5) stop' error in staggered grid'
tauz(ndstat+1)=taunue(ndstat) ! numerics

allocate(ibeta(ndstat),istart(ndstat+1),iend(0:ndstat))
allocate(lamline(ndstat,ndstat),lamcont(ndstat,ndstat),betaq(ndstat,ndstat+1))
lamline=0.
lamcont=0.

! pre-calculate ibeta = int_tau_out^tau (beta)
ibeta(1)=0.
do i=2,ndstat
  li=i+lstat-1
! this is the version if beta is integrated over tau-line
!  ibeta(i)=ibeta(i-1)+0.5*(beta(li-1)+beta(li))*(taunue(i)-taunue(i-1))
  ibeta(i)=ibeta(i-1)+0.5*(opac(li-1)+opac(li))*(r(li-1)-r(li))
enddo

!do i=1,ndstat
!  write(*,11) i+lstat-1,taunue(i),tauz(i),ibeta(i),taucz(i),beta(i+lstat-1)
!enddo  
!11 format(i2,5(2x,e9.3))

do i=1,ndstat
  tau=taunue(i)
  li=i+lstat-1
  do j=1,ndstat+1
    lj=j+lstat-1
    tauzj=tauz(j)
!    print*,i,j,tau,tauzj
    dt=tau-tauzj
    if (abs(dt).lt.1.d-13) dt=sign(1.d-13,dt) !numerical precision
    
    if(i.eq.1 .and. j.eq.1 .or. i.eq.ndstat .and. j.eq. ndstat+1) then
      betaq(i,j)=beta(li)

!    else if (abs(dt).lt.1.d-12) then !numerical precision
!      betaq(i,j)=beta(li)

    else

      if (dt.gt.0.) then
        if(j.ne.1) then
! this is the version if beta is integrated over tau-line
!      bi=(0.25*beta(lj-1)+0.75*beta(lj))*(taunue(j)-tauzj)+ibeta(i)-ibeta(j)
          bi=ibeta(i)-taucz(j)
        else
          if(taunue(j)-tauzj .ne. 0) stop' betaq: error 1' 
          bi=ibeta(i)-ibeta(j)
        endif
        betaq(i,j)=bi/dt
      
      else if (dt.lt.0.) then
        if(j.ne.ndstat+1) then
! this is the version if beta is integrated over tau-line
!      bi=ibeta(j-1)-ibeta(i)+(0.75*beta(lj-1)+0.25*beta(lj))*(tauzj-taunue(j-1))
          bi=taucz(j)-ibeta(i)
        else
          if(tauzj-taunue(j-1) .ne. 0) stop' betaq: error 2' 
          bi=ibeta(j-1)-ibeta(i)
        endif
        betaq(i,j)=-bi/dt

      else

!   might happen for very weak lines
!      if(tau.lt.1.d-7 .and. tauzj.lt.1.d-7) then
!        flag=.false.
!        flagcmf=.false.
!        return
!      endif
!   otherwise
        print*,i,j,ndstat,tau,tauzj
        print* 
        do ii=1,ndstat
          l=ii+lstat-1
          print*,ii,l,taunue(ii),beta(l)
        enddo
!        flag=.false.
!        flagcmf=.false.
!        return
        stop ' no path to here'
      endif

  endif

    if(betaq(i,j) .lt. 0.) then
! numerical problems (might also lead to dt<0 below)
! thus
      flag=.false.
      flagcmf=.true.
      return
    endif  
  enddo
enddo

! build up lambda matrix (assuming constant S over staggered grid)
!icount1=0
!icount2=0

istart(1)=1
istart(2)=1
do i=2,ndstat
  tau=taunue(i)
  done=.false.
  do j=istart(i),i-1
!  do j=1,i-1
!lower triangle (radiation from above)
    l=j+lstat-1
    bet=beta(l)
    betq1=betaq(i,j)
    betq2=betaq(i,j+1)
    betqm=0.5*(betq1+betq2)
    tau1=tauz(j)
    tau2=tauz(j+1)
    dt1=(tau-tau1)
    dt2=(tau-tau2)
    change=betq1/betq2
    if(change.gt.fac .or. change.lt. 1./fac) then
!      print*,'1',i,j,bet,betq1,betq2,xk21,xl21
      if(bet.gt.5.) then
         if(betq1.gt.5.) then
           rat=betq1/bet
!           icount2=icount2+1  
           xk21=xk2a(betq1,dt1,xl21) 
           xk21=xk21*rat
           xl21=xl21*rat
         else
!           icount1=icount1+1  
           xk21=xk2a_simple(betq1,bet,dt1,xl21) 
         endif      

         if(betq2.gt.5.) then
           rat=betq2/bet
!           icount2=icount2+1  
           xk22=xk2a(betq2,dt2,xl22) 
           xk22=xk22*rat
           xl22=xl22*rat
         else
!           icount1=icount1+1  
           xk22=xk2a_simple(betq2,bet,dt2,xl22) 
         endif      
      else  
!         icount1=icount1+2  
         xk21=xk2a_simple(betq1,bet,dt1,xl21) 
         xk22=xk2a_simple(betq2,bet,dt2,xl22) 
      endif
    else
!      icount2=icount2+2  
      xk21=xk2a(betqm,dt1,xl21) 
      xk22=xk2a(betqm,dt2,xl22)
    endif
      lamline(i,j)=(xk22-xk21)
      lamcont(i,j)=bet*(xl22-xl21)
      if(.not.done) then
        if(lamline(i,j).gt.neglect .or. lamcont(i,j).gt.neglect) then
          istart(i+1)=j
          done=.true.
        endif
      endif
  enddo
  if(.not.done) istart(i+1)=i-1
enddo

do i=1,ndstat
  tau=taunue(i)
!diagonal (local radiation)
  j=i
    l=j+lstat-1
    bet=beta(l)
    betq1=betaq(i,j)
    betq2=betaq(i,j+1)
    betqm=0.5*(betq1+betq2)
    tau1=tauz(j)
    tau2=tauz(j+1)
    dt1=(tau-tau1)
    dt2=(tau2-tau)
! tau inside interval, change of sign 
!    icount2=icount2+1
    xk20=xk2a(bet,0.d0,xl20) !always, since only dependent on bet 
    change=betq1/betq2
    if(change.gt.fac .or. change.lt. 1./fac) then
!      print*,'2',i,j,bet,betq1,betq2
      if(bet.gt.5.) then
         if(betq1.gt.5.) then
           rat=betq1/bet
!           icount2=icount2+1  
           xk21=xk2a(betq1,dt1,xl21) 
           xk21=xk21*rat
           xl21=xl21*rat
         else
!           icount1=icount1+1  
           xk21=xk2a_simple(betq1,bet,dt1,xl21) 
         endif      

         if(betq2.gt.5.) then
           rat=betq2/bet
!           icount2=icount2+1  
           xk22=xk2a(betq2,dt2,xl22) 
           xk22=xk22*rat
           xl22=xl22*rat
         else
!           icount1=icount1+1  
           xk22=xk2a_simple(betq2,bet,dt2,xl22) 
         endif      
      else  
!         icount1=icount1+2  
         xk21=xk2a_simple(betq1,bet,dt1,xl21) 
         xk22=xk2a_simple(betq2,bet,dt2,xl22) 
      endif
    else
!      icount2=icount2+2  
      xk21=xk2a(bet,dt1,xl21) 
      xk22=xk2a(bet,dt2,xl22)
    endif
      lamline(i,i)=2.d0*xk20-xk21-xk22
      lamcont(i,i)=bet*(2.d0*xl20-xl21-xl22)
      if (lamline(i,i).le.0.) lamline(i,i)=1.e-10
      if (lamcont(i,i).le.0.) lamcont(i,i)=1.e-10
enddo

iend(ndstat)=ndstat
iend(ndstat-1)=ndstat
do i=ndstat-1,1,-1
  tau=taunue(i)
  done=.false.
  do j=iend(i),i+1,-1
!  do j=ndstat,i+1,-1
!upper triangle (radiation from below)
    l=j+lstat-1
    bet=beta(l)
    betq1=betaq(i,j)
    betq2=betaq(i,j+1)
    betqm=0.5*(betq1+betq2)
    tau1=tauz(j)
    tau2=tauz(j+1)
    dt1=(tau1-tau)
    dt2=(tau2-tau)
    change=betq1/betq2
    if(change.gt.fac .or. change.lt. 1./fac) then
!      print*,'3',i,j,bet,betq1,betq2
      if(bet.gt.5.) then
         if(betq1.gt.5.) then
           rat=betq1/bet
!           icount2=icount2+1  
           xk21=xk2a(betq1,dt1,xl21) 
           xk21=xk21*rat
           xl21=xl21*rat
         else
!           icount1=icount1+1  
           xk21=xk2a_simple(betq1,bet,dt1,xl21) 
         endif      

         if(betq2.gt.5.) then
           rat=betq2/bet
!           icount2=icount2+1  
           xk22=xk2a(betq2,dt2,xl22) 
           xk22=xk22*rat
           xl22=xl22*rat
         else
!           icount1=icount1+1  
           xk22=xk2a_simple(betq2,bet,dt2,xl22) 
         endif      
      else  
!         icount1=icount1+2  
         xk21=xk2a_simple(betq1,bet,dt1,xl21) 
         xk22=xk2a_simple(betq2,bet,dt2,xl22) 
      endif
    else
!      icount2=icount2+2  
      xk21=xk2a(betqm,dt1,xl21) 
      xk22=xk2a(betqm,dt2,xl22)
    endif
      lamline(i,j)=(xk21-xk22)
      lamcont(i,j)=bet*(xl21-xl22)
      if(.not.done) then
        if(lamline(i,j).gt.neglect .or. lamcont(i,j).gt.neglect) then
          iend(i-1)=j
          done=.true.
        endif
      endif
  enddo
  if(.not.done) iend(i-1)=i+1
enddo

!print*,' icount1 = ',icount1,' icount2 = ',icount2

do i=1,ndstat
  l=i+lstat-1
  alo(l)=lamline(i,i)
!  write(*,999) 'alo',l,alo(l),beta(l),opal(l)*r(nd),taunue(i),lamcont(i,i),sc(l)
enddo
999 format(a3,2x,i2,2x,6(e9.3,2x))


do i=1,ndstat

! calculate Jbar from full lambda matrices
l=i+lstat-1
is1=istart(i)+lstat-1
ie1=iend(i)  +lstat-1
ajlmean(l)=dot_product(lamline(i,istart(i):iend(i)),sline(is1:ie1))+ &
& dot_product(lamcont(i,istart(i):iend(i)),sc(is1:ie1))
!print*,i,l,ajlmean(l),sline(l),sc(l)
enddo

! forget outer boundary condition; wrong in any case.

! lower boundary condition
aic=sc(nd) !assuming that the model is thermalized

do i=ndstat,1,-1
l=i+lstat-1
betq1=betaq(i,ndstat)
dt=taunue(ndstat)-taunue(i)
e1=xl2 (betq1, dt, xl3a)

! until further evidence neglect diffuse term
! (assuming that tau_tot is large and that only quantities until lwion1-1 are needed)
! 
!diff = xl3a * corrfc * dbdr / opal (nd)
!if (beta(nd).gt.1.) diff=diff*betq1/beta(nd) ! rough correction for beta(nd) ne betaqq

ajlmean(l) = ajlmean(l)+  e1 * aic ! + diff
enddo

! now interpolation between louter (sobo) and lstat (static approach)

rstar=r(lstatin)

! interpolation log y vs log v
! note:  correction for dilution (r^2 terms); only "source terms, i.e., &
! betacic and jbar, need to be considered. alo and beta (-> ps1) unaffected.

! from numerous tests using realist models, it turned out that in most
! cases only a correction for the interpolation is required; for the lower
! photosphere, it leads to erroneous results. Only for very weak winds the
! correction improves the consistency with cmf-calculations


! note; previously, we interpolated between louter and lstat+1

dv=log(velo(louter)/velo(lstatin))
xouter1=ps1
xinner1=alo(lstatin)
bb1=log(xouter1/xinner1)/dv

xouter2=betacic*(r(louter)/rstar)**2
xinner2=(ajlmean(lstatin)-alo(lstatin)*sline(lstatin))
bb2=log(xouter2/xinner2)/dv

do l=louter,nd
   if(l.lt.lstatin) then
     term1=xouter1*(velo(l)/velo(louter))**bb1
! diluted quantity, needs to be corrected before interpolation
     term2=xouter2*(velo(l)/velo(louter))**bb2*(rstar/r(l))**2 ! correction for dilution
     alo(l)=term1
     ajlmean(l)=term2
!  note that only values l ge louter+1 are finally transfered
!  but checked that term1 and term2 at l=louter exactly = 1.-beta and betacic, respectively
   else  
!     term2=(ajlmean(l)-alo(l)*sline(l))*(rstar/r(l))**2 ! correction for dilution
     term2=(ajlmean(l)-alo(l)*sline(l)) ! no longer corrected
     ajlmean(l)=term2
! alo already defined
   endif

enddo

!rstar=r(nd)
! for tests
!do l=louter,nd
!  ajm=ajlmean(l)+alo(l)*sline(l) ! in Sobo, alo=1.-beta = ps1
!  rraic=(r(l)/rstar)**2/aic
!  write(*,10) l,ajm*rraic,alo(l),ajlmean(l)*rraic
!enddo

flag=.true.
flagcmf=.false.

return

10 format(i2,3(2x,e10.4))
end
!
!-----------------------------------------------------------------------
!
subroutine xgrid

USE nlte_type
USE phot_lines, ONLY: nf, phi, pweight

implicit none

real(dp), parameter :: xmax=4.d0, wpi= 1.7724539d0

integer(i4b) :: i

real(dp) :: dx, summ, x

dx=xmax/float(nf-1)

summ=0.

do i=1,nf
x=xmax-(i-1)*dx
phi(i)=exp(-x*x)/wpi
summ=summ+phi(i)
enddo

pweight=0.5*phi/summ  ! normalized to 0.5 since integration from 0 to xmax
  
return
end
!
!-----------------------------------------------------------------------
!
function xk2a (beta, tau1, xl2aval)  

USE nlte_type
USE phot_lines

implicit none

real(dp) :: xk2a

real(dp), intent(in) :: beta, tau1
real(dp), intent(out) :: xl2aval

real(dp), parameter :: xk2_0=0.19947114d0  !0.5/sqrt(2pi)

integer(i4b) :: nb, nt

real(dp) :: tau, xtau, xbeta, xm, xk2a1, xk2a2, xl2a1, xl2a2, xl2a, expint2

if (tau1.lt.1.d-10) then
  tau=1.d-10
else
  tau=tau1
endif
  
xtau = log10 (tau)  
  
xbeta = log10 (beta)  

if (xbeta.lt.betamin) xbeta = betamin  

if (xbeta.gt.betamax) then  
   Xl2aval = 0.5 * expint2 (tau1*beta)/beta  
   xk2a = xl2aval * 2. * xk2_0
return  
endif  

if (xtau.lt.taumin.or.xtau.gt.taumax) then  
   print * , xtau, ' out of range in kl-tables'  
   stop ' xtau out of range in kl-tables'  

endif  
nb = nint (xbeta * 10.) + 101  
nt = nint (xtau * 100.) + 1001  
if (abs (xbeta - betatab (nb) ) .gt.0.051d0) then
  print*,xbeta, betatab(nb)  
  stop ' error in beta-gridpoint: xk2a'
endif

if (abs (xtau - tautab (nt) ) .gt.0.0051d0) then
  print*,xtau,tautab(nt)
  stop ' error in tau-gridpoint'
endif

!tauout=10.**tautab(nt)
!interpolation in beta
if(nb .eq. nbdim) nb=nbdim-1
xm=(xbeta-betatab(nb))/(betatab(nb+1)-betatab(nb))  

xk2a1 = xk2atab (nb, nt)
xk2a2 = xk2atab (nb+1, nt)  
if (xk2a1.ne.0. .and. xk2a2 .ne.0.) then
  xk2a1=log10(xk2a1)
  xk2a2=log10(xk2a2)
  xk2a=xm*(xk2a2-xk2a1)+xk2a1
  xk2a=10.**xk2a
else 
  xk2a=0.d0
endif  

xl2a1 = xl2atab (nb, nt)  
xl2a2 = xl2atab (nb+1, nt)
if (xl2a1.ne.0. .and. xl2a2 .ne.0.) then
  xl2a1=log10(xl2a1)
  xl2a2=log10(xl2a2)
  xl2a=xm*(xl2a2-xl2a1)+xl2a1
  xl2aval=10.**xl2a
else 
  xl2aval=0.
endif  

!print*,betatab(nb),betatab(nb+1),xbeta
!print*,xk2a1,xk2a2,xk2a
!print*,xl2a1,xl2a2,xl2aval
!print*
return  

end function xk2a
!
!-------------------------------------------------------------------
!
function xl2 (beta, tau1, xl3a)  

USE nlte_type
USE phot_lines

implicit none

real(dp) :: xl2

real(dp), intent(in) :: beta, tau1
real(dp), intent(out) :: xl3a

integer(i4b) :: nb, nt
real(dp) :: tau, xbeta, xtau, arg, expint2, expin

if (tau1.lt.1.d-10) then
  tau=1.d-10
else
  tau=tau1
endif

xbeta = log10 (beta)  
xtau = log10 (tau)  
! needs to be checked!
if (xbeta.lt.betamin) xbeta = betamin  
if (xbeta.gt.betamax) then  
   arg= beta *tau1
   xl2 = 0.5 * expint2 (arg)  
   xl3a = 0.5 * expin (3, arg) * exp (- arg) / beta  
   return  

endif  
if (xtau.lt.taumin.or.xtau.gt.taumax) then  
   print * , xtau, ' out of range in kl-tables'  
   stop ' xtau out of range in kl-tables'  

endif  
nb = nint (xbeta * 10.) + 101  
nt = nint (xtau * 100.) + 1001  
if (abs (xbeta - betatab (nb) ) .gt.0.051) stop ' error in beta-gridpoint: xl2_2'
if (abs (xtau - tautab (nt) ) .gt.0.0051) stop ' error in tau-gridpoint'

xl2 = xl2tab (nb, nt)  
xl3a = xl3atab (nb, nt)  

return  

end function xl2
!
!-------------------------------------------------------------------
!
function xk2a_simple(beta1,beta2,tau,xl2a)
! simple integration, xmax=4 and nf=21 sufficient
! factor 0.5 missing, since integration only from 0 to xmax

USE nlte_type
USE phot_lines, ONLY: nf, phig=>phi, pweight

implicit none

real(dp) :: xk2a_simple

real(dp), intent(in) :: beta1, beta2, tau
real(dp), intent(out) :: xl2a

integer(i4b) :: i
real(dp) :: sumk, suml, phi, arg, xint, expint2

sumk=0.
suml=0.
do i=1,nf
phi=phig(i)
arg=(beta1+phi)*tau
xint=pweight(i)/(beta2+phi)*expint2(arg)
sumk=sumk+phi*xint  
suml=suml+xint
enddo
xk2a_simple=sumk
xl2a=suml

return
end
!
!----------------------------------------------------------------------------
!
function expint2 (x)  
! fast 2nd exponential integral

USE nlte_type

implicit none

real(dp), intent(in) :: x
real(dp) expint2

real(dp) :: a0, a1, a2, a3, a4, a5, b1, b2, b3, b4, c1, c2, c3, c4

real(dp) :: e1, sum1, sum2

data a0, a1, a2, a3, a4, a5 / - .57721566, .99999193, - .24991055, &
 .05519968, - .00976004, .00107857 /
data b1, b2, b3, b4 / 8.5733287401, 18.0590169730, 8.6347608925, &
 .2677737343 /
data c1, c2, c3, c4 / 9.5733223454, 25.6329561486, 21.0996530827, &
 3.9584969228 /
!

if (x.gt.100.) then  
   expint2 = 0.  
   return  
endif  
!
if (x.eq.0.) then  
   expint2 = 1.  
   return  
endif  
!
if (x.le.1.) then  
   e1 = a0 + x * (a1 + x * (a2 + x * (a3 + x * (a4 + x * a5) &
    ) ) ) - log (x)  
else
sum1 = b4 + x * (b3 + x * (b2 + x * (b1 + x) ) )  
sum2 = c4 + x * (c3 + x * (c2 + x * (c1 + x) ) )  
e1 = exp ( - x) / x * sum1 / sum2  
endif

expint2 = exp ( - x) - x * e1
return  
end function expint2
!
!-------------------------------------------------------------------
!
subroutine xgrid_diff

USE nlte_type
USE phot_lines_diff, ONLY: nf, phi, pweight

implicit none

real(dp), parameter :: xmax=4.d0, wpi= 1.7724539d0

integer(i4b) :: i

real(dp) :: dx, summ, x

dx=xmax/float(nf-1)

do i=1,nf
x=xmax-(i-1)*dx
phi(i)=exp(-x*x)/wpi
enddo

summ=0.
do i=1,nf-1
pweight(i)=2.*phi(i)  ! positive and negative frequencies
summ=summ+pweight(i)
enddo

pweight(nf)=phi(nf) ! zero frequency only once
summ=summ+pweight(nf)


pweight=pweight/summ  ! normalized to 0.5 since integration from 0 to xmax

return
end
!
!-----------------------------------------------------------------------
!
subroutine jbar_phot_diff(ajlmean, alo, r, velo, temp, &
&          opal, sline, beta, opac, sc, &
&          ps1, betacic, louter, lstatin, xlambda, corrfc, flag, flagcmf)
! 
! NOTE!!!
  
! calculates jbar and alo from solving the 2nd order differential equation
! of RT (formal solution only) 
!
! so far, a constant Doppler width at vth(k) is assumed

!
! output is jbar-alo*sline and alo  
!  
! between louter and lstatin, an interpolation between static and Sobolev
! transfer is performed.

! r and v in absolute units

! here, beta is opac/opal, and ps1 is 1.-beta(sob) at louter(kel)


USE nlte_type
USE nlte_dim,only: id_ndept
USE phot_lines_diff

implicit none
integer(i4b), parameter :: nd=id_ndept

real(dp), dimension(nd), intent(in) :: r, velo, temp, opal, sline, &
&                                      beta, opac, sc
real(dp), dimension(nd), intent(out):: ajlmean, alo

integer(i4b), intent(in) :: louter, lstatin
real(dp), intent(in) :: ps1, betacic, xlambda, corrfc

logical, intent(out) :: flag, flagcmf

real(dp), dimension(nd) :: taunue, dtau, s, source, ta, tb, tc, tc1

integer(i4b) :: lstat, ndstat, l, i, k, j

real(dp) :: del, dt, taulast, aic, dbdr, phik, betal, fac, op, opold, &
&           dtp, dtm, dt0, aux, aux2, mu2, dv, xouter1, xinner1, bb1, &
&           xouter2, xinner2, bb2, term1, term2, rstar, rraic, ajm

! for the complete range, to avoid edge effects
lstat=louter

ndstat = nd-lstat + 1  
!
ajlmean = 0.  
alo = 0.

! check at first whether sobo-transfer sufficient or
! whether cmf-transport is necessary
!
!tau-grid
taunue (1) = 1.d-8  

do l = lstat, nd-1  
   i=l+1-lstat
   del = r (l) - r (l + 1)  
   dt = del * 0.5 * (opal(l) + opal(l+1))
   taunue(i+1)=taunue(i) + dt
end do  

taulast=taunue(ndstat)

!if (taulast.lt.1.0) then
!changed by JP (Jan. 2015), to avoid oscillations (with and without RT)
if (taulast.lt.0.1) then
! line very weak, Sobo sufficient
  flag=.false.
  flagcmf=.false.
  return
endif  



if(taunue(3).gt.1.) then
! optically thick around louter, needs cmf transfer
  flag=.false.
  flagcmf=.true.
  return
endif  


! diffusion approx. for lower boundary; works also when r in absolute units
call diffus (xlambda, temp, r, nd, aic, dbdr)  

! now formal solution for all frequencies (nf) and angles (nmu)
! profile functions and weights from xgrid_diff;
! angles and weights from module phot_lines_diff

do k=1,nf
! angle independent quantities
phik=phi(k)  
! outermost point
l=lstat
betal=beta(l)
fac=phik+betal
op=opal(l)*fac
s(l)=phik/fac*sline(l)+betal/fac*sc(l)

! other points
do l=lstat+1,nd
opold=op
betal=beta(l)
fac=phik+betal
op=opal(l)*fac
dtau(l-1)=0.5*(opold+op)*(r(l-1)-r(l))
s(l)=phik/fac*sline(l)+betal/fac*sc(l)
enddo

do j=1,nmu
! build up tri-diagonal matrix (-ta, tb, -tc) and source term 

! outer boundary condition, 2nd order, a la Mihalas
! (multiplied with 2 mu /dtp to be consistent with ALO)
l=lstat
dtp=dtau(l)
aux=mu(j)/dtp
aux2 = 2.d0*aux
ta(l)=0.
tb(l)=(aux+1.d0+0.5d0/aux)*aux2
tc(l)=aux*aux2
source(l)=s(l)

! intermediate points
do l=lstat+1,nd-1
dtm=dtp
dtp=dtau(l)
dt0=0.5d0*(dtm+dtp)
mu2=musq(j)
ta(l)=mu2/(dtm*dt0)
tb(l)=mu2/dt0*(1.d0/dtm+1.d0/dtp)+1.d0
tc(l)=mu2/(dtp*dt0)
! source needs to be updated, since overwritten in INVTRI
source(l)=s(l)
enddo

! inner boundary condition, 2nd order, a la Mihalas
! (multiplied with 2 mu /dtp to be consistent with ALO)
l=nd
aux=mu(j)/dtp !dtp from above is dtm local
aux2=2.d0*aux
ta(l)=aux*aux2
tb(l)=(aux+1.d0+0.5d0/aux)*aux2
tc(l)=0.
source(l)=(aic+corrfc*mu(j)*dbdr/op)*aux2+s(l)  ! op angle-independent, survived from upper loop

tc1=tc ! save for alo calculation

! solution of tridiag system; source and tc overwritten
call invtri(ta(lstat:nd),tb(lstat:nd),tc(lstat:nd),source(lstat:nd),ndstat)

! calculation of the diagonal of the inverse of T = alo_tot (k,j)
call diag(ta(lstat:nd),tb(lstat:nd),tc1(lstat:nd),tc(lstat:nd),ndstat,0)
! integration of jbar and alo
do l=lstat,nd
  ajlmean(l)=ajlmean(l)+source(l)*wmu(j)*pweight(k)
  alo(l)=alo(l)+tc(l)*wmu(j)*pweight(k)*phik/(phik+beta(l))
enddo

enddo ! angle loop
enddo ! frequency loop

rstar=r(lstatin)

! interpolation log y vs log v
! note:  correction for dilution (r^2 terms); only "source terms, i.e., &
! betacic and jbar, need to be considered. alo and beta (-> ps1) unaffected.

! from numerous tests using realist models, it turned out that in most
! cases only a correction for the interpolation is required; for the lower
! photosphere, it leads to erroneous results. Only for very weak winds the
! correction improves the consistency with cmf-calculations


! note; previously, we interpolated between louter and lstat+1

dv=log(velo(louter)/velo(lstatin))
xouter1=ps1
xinner1=alo(lstatin)
bb1=log(xouter1/xinner1)/dv

xouter2=betacic*(r(louter)/rstar)**2
xinner2=(ajlmean(lstatin)-alo(lstatin)*sline(lstatin))
bb2=log(xouter2/xinner2)/dv

do l=louter,nd
   if(l.lt.lstatin) then
     term1=xouter1*(velo(l)/velo(louter))**bb1
! diluted quantity, needs to be corrected before interpolation
     term2=xouter2*(velo(l)/velo(louter))**bb2*(rstar/r(l))**2 ! correction for dilution
     alo(l)=term1
     ajlmean(l)=term2
!  note that only values l ge louter+1 are finally transfered
!  but checked that term1 and term2 at l=louter exactly = 1.-beta and betacic, respectively
   else  
!     term2=(ajlmean(l)-alo(l)*sline(l))*(rstar/r(l))**2 ! correction for dilution
     term2=(ajlmean(l)-alo(l)*sline(l)) ! no longer corrected
     ajlmean(l)=term2
! alo already defined
   endif

enddo

!rstar=r(nd)
! for tests
!print*
!print*,louter,lstatin,xlambda
!do l=louter,nd
!  ajm=ajlmean(l)+alo(l)*sline(l) ! in Sobo, alo=1.-beta = ps1
!  rraic=(r(l)/rstar)**2/aic
!  write(*,10) l,ajm*rraic,alo(l),ajlmean(l)*rraic
!enddo
!print*

!if(xlambda .gt. 1238.148 .and. xlambda .lt. 1238.15) stop

flag=.true.
flagcmf=.false.

return

10 format(i2,3(2x,e10.4))
end
!
!----------------------------------------------------------------------------
!
subroutine cmf_simple(ajlmean,alo,r,v,vr,dvdr,temp, &
&            opalin,sline,opacin,etacin,scont,vdop,xlambda,corr,xjin,betamin)
!
! simplified cmf transfer for background elements (where necessary)
! output is alo and jbar-alo*sline

! here, r, v, vr and dvdr are dimensionless (normalized), 
! and the opacities/emissivities need to be multiplied by
! Rstar and Rstar/vinf to be consistent with the standard approach  
!
! opal needs additional correction by vdop = vth(k) = vdop(teff,vmicro)
! correction for depth dependent vdop in fgrid  

USE nlte_type
USE nlte_dim,only: id_ndept, id_npoin, id_nfcmf

USE run_once, only: start_cmf_simple

USE nlte_var, only : sr,vmax

USE nlte_var, only : z,p,tauz1,tau1,pp1,slinek,ak1,az1,w0=>vp,w0last

USE nlte_app, only : teff

implicit none
integer(i4b), parameter :: nd=id_ndept, np=id_npoin
integer(i4b), parameter :: nf=id_nfcmf  

real(dp), dimension(nd), intent(out):: ajlmean, alo
real(dp), dimension(nd), intent(in) :: r, v, vr, dvdr, temp
real(dp), dimension(nd), intent(in) :: opalin, sline, opacin, etacin, scont

real(dp), intent(in) :: xlambda, corr, xjin, betamin

real(dp), dimension(nd) :: opal, opac, etac, ucmf, pp
real(dp), dimension(nd-1) :: vcmf
real(dp), dimension(nd,np-1) :: ubluwi

real(dp), dimension(nf,nd) :: phi, pweight

real(dp) :: aic, dbdr, srvmax, rl, xmue, xmue2, deltax, vdop, xmaxdop, &
&           xmax0, xmax1, aux, opalk, etak, opakk

integer(i4b) :: jp, lmax, l, lz, k


! line-independent quantities
if (start_cmf_simple) then
! might have already been calculated in cmf (nlte), &
! but recalculate in any case (previous update possible)

jploop1: do jp = 1,np - 1  
          lmax = min0(np+1-jp,nd)  
          lz = lmax - 1  

          do l = 1,lz  
               tauz1(l,jp) = 1.d0/ (z(l,jp)-z(l+1,jp))  
          end do

          do l = 2,lz  
               tau1(l,jp) = 2.d0/ (z(l-1,jp)-z(l+1,jp))  
          end do
!
!     pp(l)=velocity gradient,projected on the present ray
!
          do l = 1,lmax  
               rl = r(l)  
               xmue = z(l,jp)/rl  
               xmue2 = xmue*xmue  
               pp1(l,jp) = xmue2*dvdr(l)+ (1.d0-xmue2)*vr(l)
          end do

end do jploop1

start_cmf_simple=.false.
endif

call diffus (xlambda, temp, r, nd, aic, dbdr)  
!print * ,' dbdr=', dbdr, corr  

!
!     dimensioning of opacities
!
srvmax=vdop*sr/vmax
opac=opacin*sr
etac=etacin*sr
opal=opalin*srvmax

!
!     calculation of continuum radiation field at the blue wind frequenc
!
call cont2 (nd, np, r, p, z, scont, opac, aic, dbdr, corr, ubluwi)


!if (abs(1.-scont(nd)/aic).gt.0.05) stop' inconsistency between ic-scont'
!if (abs(1.-xjin/aic).gt.0.05) stop' inconsistency between ic-xjin'

! calculation of xmax (opal/ddop (xmax) = 0.01 opac)
xmax0=3.
! const = 0.01*wpi
aux = - log (.0177245 * betamin)
if (aux.gt.0.) then
  xmax1 = sqrt (aux)  
  if (xmax1.gt.xmax0) xmax0 = xmax1
endif

!print*,betamin,xmax0

call fgrid(nf,nd,phi,pweight,deltax,vdop,vmax,teff,temp,xmax0)  

do l = 1,nd
     do k = 1,nf  
          opalk = phi(k,l)*opal(l)  
          etak = sline(l)*opalk + etac(l)  
          opakk = opalk + opac(l)  
          slinek(k,l) = etak/opakk         
          ak1(k,l) = 1.d0/opakk  
          if (l.ne.1) az1(k,l-1) = 2.d0/ (opakk+1.d0/ak1(k,l-1))  
     end do
end do  

ajlmean = 0.d0  
alo = 0.d0  

jploop2: do jp = 1,np - 1  
     lmax = min0(np+1-jp,nd)  
     lz = lmax - 1  
!
!     bluewing boundary condition
!
     ucmf(1) = ubluwi(1,jp)  

     do l = 1,lz  
          ucmf(l+1) = ubluwi(l+1,jp)  
          aux = .5d0* (opac(l)+opac(l+1))  
          vcmf(l) = (ucmf(l+1)-ucmf(l))/aux/ (z(l,jp)-z(l+1,jp))  
     end do

     do l = 1,lmax  
          pp(l) = pp1(l,jp)/deltax
     end do
     
     call ray(.true.,1,1,&
&         jp,z(:,jp),r,nd,np,ucmf,vcmf,sline,phi,pweight, &
&         deltax,opac,etac,opal,aic,dbdr,corr,w0(:,jp),w0last, &
&         ajlmean,alo,tauz1(:,jp),tau1(:,jp),pp,xlambda)

end do  jploop2

! this is the output quantity
ajlmean=ajlmean-alo*sline

return
end
