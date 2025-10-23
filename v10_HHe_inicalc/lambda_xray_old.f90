! routines for x-ray treatment
! version 1.0   October 2014, programmed by Luiz Carneiro and Jo
! version 1.1   July 2016, programmed by Jo and Koushik Sen, to
!               allow for a better description of radiative shocks
!               (see version control in nlte.f90)
! version 1.1.1 Sept 2020: one bug found by Sarah Brands removed  
! version 1.1.2 Oct  2020: compatible with gfortran  
!
!-----------------------------------------------------------------------
!
SUBROUTINE TSHOCK(TS,V,R,RTAU23,XMU,ND)
! calculates stratification of X-ray shock-temperatures, basically following
! Pauldrach et al. 1994.
! Change this routine if you want to test other stratificationsq 

!NOTE: The minimum emission radius is set from the minimum of
!      the velocity (V_em > MX*VSOUND) or radius constraint (R_em > RMINX) 
  
USE nlte_type
USE nlte_dim, ONLY: ID_NDEPT
USE fund_const, ONLY: AMH, AKB
USE nlte_var, ONLY: MODNAM, DELMU, VMAX, VSOUND
USE nlte_xrays, ONLY: FX, PX, LXMIN, RMINX_EFF

IMPLICIT NONE

INTEGER(I4B), INTENT(IN) :: ND
REAL(DP), DIMENSION(ND), INTENT(IN) :: V,R
REAL(DP), DIMENSION(ND), INTENT(OUT) :: TS

REAL(DP), DIMENSION(ID_NDEPT) :: UJUMP

REAL(DP) :: RTAU23,XMU
REAL(DP) :: GAMX,MX,UINFX,TS_CONST,VMIN,RMINX

INTEGER(I4B) :: L,L1,L2

LOGICAL :: OPTXRAY

OPEN (1,FILE=TRIM(MODNAM)//'/INXRAY.DAT',STATUS='OLD')  
REWIND 1
READ (1,FMT=*) OPTXRAY,FX
IF(.NOT.OPTXRAY) STOP ' OPTXRAY = FALSE IN CATALOGUE/INXRAY.DAT'
READ (1,FMT=*) GAMX,MX,RMINX,UINFX,PX 
CLOSE(1)

! for simplicity, we calculate the shock temperature from the approximate
! expression, neglecting terms with O(vsound), i.e., assuming strong shocks
! (also w.r.t. consistency with Feldmeier et al., Eq. 19)
! Note: the exact expression would be
! Tshock = 3/16 mu mh/k * (u^2 + (14/5 a^2 * (1 - 3/14 a^2/u^2)))


TS_CONST=3.D0/16.D0*XMU*AMH/AKB

UJUMP=UINFX*1.D5*V**GAMX

!now assuming delmu = 1 (to ensure consistent restart)
TS=TS_CONST*UJUMP**2
!TS=TS_CONST/DELMU*UJUMP**2
! corrected for variable xmu; xmu_actual=xmu/delmu

!print*, 'XMU=',XMU
!print*, 'AMH=',AMH
!print*, 'AKB=',AKB
!print*, 'V= ',V
!print*, 'GAMX =',GAMX
!print*, 'UJUMP=',UJUMP
!print*, 'TS=',TS


VMIN=MX*VSOUND

DO L=1,ND
  IF(V(L)*VMAX.LT.VMIN) EXIT
ENDDO
L1=L

!JO changed July 2016
!RMINX in units of RSTAR (not SR)
DO L=1,ND
  IF(R(L)/RTAU23.LT.RMINX) EXIT
ENDDO
L2=L

LXMIN=MAX(L1,L2)-1
RMINX_EFF=R(LXMIN)

DO L=LXMIN+1,ND
  TS(L)=0.D0
ENDDO

PRINT*
PRINT*,'SHOCK TEMPERATURES CALCULATED, FROM L=1 TO L=',LXMIN
PRINT*,'MINIMUM EMISSION RADIUS (in units of RSTAR) = ',RMINX_EFF/RTAU23
PRINT*,'MINIMUM EMISSION RADIUS (in units of SR) = ',RMINX_EFF
PRINT*
PRINT*,'SHOCK TEMPERATURE AT RMAX = ',TS(1)/1.D6,' MK'
PRINT*,'MINIMUM SHOCK TEMPERATURE = ',TS(LXMIN)/1.D6,' MK'

RETURN
END
!
!---------------------------------------------------------------------------
!
subroutine lambda_xray(xmu,ts,r,v,rho,xne,xnh,fcl,nd)

! calculate X-ray cooling functions accounting for cooling zones, both for
! radiative and adiabatic shocks, following Feldmeier et al. 1997
! The basic cooling functions are calculated via Raymond & Smith 1989

! Note: you might play with Theta and xne;
! so far, xne = 1.d10 fixed in cooling function
!JO July 2016
! for the electron density used to calculate the cooling time/length, we
! now use the hot-plasma conditions (IHe=2), but check for exceptions  
  
!version 0.1, programmed by Luiz Carneiro (2014)
!version 0.2, modified by Jo (using input from Koushik Sen) (July 2016)
  
USE nlte_type
USE fund_const, only: hh,clight,ev
USE nlte_var, only: delmu, yhe=>yhein
USE nlte_xrays, only: enerx, spec=>lambdax, lxmin, fxl, px

IMPLICIT NONE

integer(i4b) :: nd
real(dp), dimension(nd), intent(in) :: ts,r,v,rho,xne,xnh,fcl
real(dp) :: xmu

!input for rs-code: variables/data
integer(i4b), parameter :: nf=1000 ! <= 1000, because required by Raymond-Smith

!parameters which may be changed
integer(i4b) :: ntdec=25!number of t-points per decade
!if you change the binsize, change also nf
real(dp) :: x_binsyz =2.5d0 ! (eV)


integer(i4b) :: i_nbin, &
                i_nbln=0, &
                i_num=12, &
                i_iprint=0, &
                i_jprint=0, &
                i_iphot=0, &
                i_idens=0, &
                i_icx=0

real(dp) :: x_binmin=1., &
            x_blnmin=0., &            
            x_blnsyz=0.

real(dp) :: logt, tinput, distance

!***************************************************************************
!NOIE: so far and for computational efficiency,
!we calculate Lambda with constant electron density ne=10^10.
!Tests have shown that this is basically OK, but certain frequencies
!around 10 to 100 ev react weakly on variations
real(dp) :: logne=10.d0

!NOIE: so far we assume theta = T_rev/T_for (post) as unity.
!This is just a guess and needs to be tested
real(dp) :: theta=1.d0
real(dp) :: wr=0.5d0, wf=0.5d0

logical :: isothermal = .false.
! if true, isothermal shocks will be calculated (check that wr+wf=1.)

!regarding the adiabatic double-shock structure:
!if theta = 1, the structure of the cooling zones behind the forward and 
!the reverse shocks are identical (but somewhat different from the (single) 
!radiative one). Since we now have two shocks per volume, this would require 
!a different filling factor (factor two lower), and thus we divide the 
!total emission by a factor of two (using weighting factors of 0.5).
!If we would use wr=1 and wf=0, we would only count the contribution from 
!the reverse shock (for theta=1 this is identical to 0.5/0.5), and vice versa
!for the forward shock with wr=0 and wf=1.
!Note that reverse shocks are destroyed around 10 Rstar. 
!Realistic weighting factors need to be derived from hydro-sims. 
!We suggest to keep the above parameterization until further evidence.

!NOTE: As in Feldmeier, we assume that the parameterized
! jump velocity always refers to the reverse-shock temperature
! in the radiative case, but to the forward-shock temperature
! in the adiabatic case. The reason for doing so is that the forward-shock
! structure in the latter is independent of theta, whilst the density of
! the reverse shock changes significantly. Thus, normalizing to the
! forward shock retains a somewhat stable solution if theta is changed
!
!***************************************************************************


integer(i4b), parameter :: NFELD = 1000 ! number of integration points over cooling zone

integer(i4b) :: i, j, k, nt, n, l

real(sp), dimension(nf) :: energy, lambdanu

real(dp) :: tmin, tmax, dt, w1, w2, t1, t2, tt
real(dp) :: Fract_tc_tf, Fract_Lc_rs, xne_av, xnh_av, kappa_f, dxi, xne_app_last

real(dp), allocatable, dimension(:) :: tgrid,lambdainter
real(dp), allocatable, dimension(:,:) :: lambda

!Declaration of variables in the subroutines

real(dp), dimension(NFELD) :: f_ksi, g_ksi

real(dp), dimension (NFELD) :: f_eta_rev, g_eta_rev,   &
                               f_eta_for, g_eta_for

real(dp), dimension(NFELD) :: f_rev, g_rev


!JO July 2016
!check consistency of provided ('cool') electron density at lowermost point
!with approximate hot plasma xne-value. If differences, then most likely
!other elements than H/He are dominating, and another approach must be provided.
xne_app_last=xnh(nd)*(1.+2.*YHe)
if(abs(1.-xne_app_last/xne(nd)).gt.0.05) then
  print*,' approximate electron-density used for X-ray emission erroneous!'
  print*,' most likely, H/He not dominating in this model!'
  print*,' change approach!'
  print*,xne(nd),xne_app_last
  stop ' approximate electron-density used for X-ray emission erroneous!'
endif


!create grid of models
!
tmin=4.d0
if(tmin.gt.ts(lxmin)) stop 'tmin > ts(lxmin). Modify minimum emission radius!' 
tmax=max(ts(1),ts(1)*theta) ! to account for the double shock structure
tmax=int(log10(tmax))+1.
print*
print*,'minimum shock temp. (log10) for grid = ',tmin
print*,'maximum shock temp. (log10) for grid = ',tmax
print*,'number of T-points per decade = ',ntdec

dt=tmax-tmin
nt=dt*ntdec+2
dt=1.d0/float(ntdec)

allocate(tgrid(nt))

do i=1,nt
  tgrid(i)=tmin+(i-1)*dt
!  print*,i,tgrid(i)
enddo

if(tgrid(nt).le.tmax) stop ' tgrid(nt) < tmax'

!actually i_nbin can be also smaller than nf, if other freq. grid required, e.g.
!i_nbin = int(1000/x_binsyz),
!which would give 500 points for x_binsyz = 2.)

i_nbin=nf    

!print*,nt
allocate(lambda(i_nbin,nt))
allocate(lambdainter(i_nbin))

do i=1,nt

! note: RS-code in single precision
  call raymond_smith(sngl(tgrid(i)),sngl(logne), energy, lambdanu, &
                   i_nbin,sngl(x_binmin),sngl(x_binsyz), &
                   i_nbln,sngl(x_blnmin),sngl(x_blnsyz), &
&                  i_num,i_iprint,i_jprint,i_iphot,i_idens,i_icx)   
!********
!********Added this block to make lambdainter = 0
!********in case of lamda = 0    05/08/14
!********
  where(lambdanu.ne.0.d0)
     lambda(:,i)=log10(lambdanu(:))
  elsewhere
     lambda(:,i)=0.d0
  endwhere
enddo

! independent of local conditions
call radiative_part(g_ksi,f_ksi,NFELD)

if(isothermal) then
g_ksi=1.d0
f_ksi=1.d0
endif

allocate(spec(nf,lxmin))
spec=0.d0

print*

if(.not.allocated(fxl)) allocate(fxl(nd))
fxl=0.d0

print* 
if(px.lt.-999.99) then
  print*,' Old approach with constant filling factor,'
  print*,' giving rise to rho^2-dependent emission'
  print*,' for radiative and adiabatic shocks!'
else
  print*,' New approach with radius-dependent filling factor,'
  print*,' giving rise to rho-dependent emission for radiative shocks,'
  print*,'          and rho^2-dependent emission for adiabatic ones!'
endif
print*

do l=1,lxmin

logt=log10(ts(l))

tinput=logt

!xne_av=xne(l)/fcl(l)
xnh_av=xnh(l)/fcl(l)
!modified
xne_av=xnh_av*(1.+2.*YHe)

call cooling_flowing_time &
!& (Fract_tc_tf,Fract_Lc_rs,ts(l),r(l),v(l),rho(l),xne_av,xnh_av,xmu/delmu(l), &
!now assuming delmu=1 to ensure consistent restart
& (Fract_tc_tf,Fract_Lc_rs,ts(l),r(l),v(l),rho(l),xne_av,xnh_av,xmu, &
&  theta,kappa_f,l)

if (Fract_tc_tf .lt. 1.d0) then
      if(kappa_f.ne.0.d0) stop ' radiative and kappa_f ne 0!'
      g_rev = g_ksi
      f_rev = f_ksi
 
else
      call adiabatic_part(theta,kappa_f,f_eta_rev,g_eta_rev,f_eta_for,g_eta_for,NFELD)

if(isothermal) then
f_eta_rev=1.d0
g_eta_rev=1.d0
f_eta_for=1.d0
g_eta_for=1.d0
endif

! g normalized to forward shock, g(eta_f=1)=1
! rev. shock temperature = : T_for* g_rev =: T_jump * g_rev with T_jump = ts
      g_rev = g_eta_rev
      f_rev = f_eta_rev

!do i = 1,NFELD
!print*, i,f_eta_rev(i),g_eta_rev(i),f_eta_for(i),g_eta_for(i)
!end do
end if

do k=1,NFELD            ! radiative cooling
   logt = tinput

   logt = logt + log10(g_rev(k))
   
!The emission of temperature log(ts) < 4.0 is too low and has no big contribution.
   if (logt .lt. 4.0d0) then
     lambdainter=0.d0
     goto 15
   endif
     
! interpolate the cooling function at logt from the grid   
   do i=nt,1,-1
     if(logt.le.tgrid(i) .and. logt.gt. tgrid(i-1)) goto 10 
   enddo

   print*,logt
   print*,tgrid
   print*,'logt not found in tgrid - 1'
   stop


10 t1= 1.d0  / 10.d0 ** tgrid(i-1)
   t2 = 1.d0 / 10.d0 ** tgrid(i)
   tt = 1.d0 / 10.d0 ** logt
 
   w1=(t2-tt)/(t2-t1)
   w2=1.d0-w1

!print*,w1,w2

!********
!********Added this block to make lambdainter = 0
!********in case of lamda = 0    05/08/14
!********
   where(lambda(:,i-1).ne.0.d0 .and. lambda(:,i).ne.0.d0) 
      lambdainter(:)=10.d0**(w1*lambda(:,i-1)+w2*lambda(:,i))
   elsewhere
      lambdainter(:)=0.d0
   endwhere  

15 continue   
! integration weights assumed to be equal (neglecting boudary effects)
   if (Fract_tc_tf .lt. 1.d0) then
! no weight, one shock
     spec(:,l) = spec(:,l) + f_rev(k)**2 * lambdainter(:) 
   else

! contrib. from reverse shock     
! two shocks, weighted
     spec(:,l) = spec(:,l) + wr * f_rev(k)**2 * lambdainter(:)


!***************************************************************************
! contrib. from forward shock 
      logt = tinput

! g normalized to forward shock, g(eta_f=1)=1
! forw. shock temperature = : T_for* g_eta_for =: T_jump * g_eta_for with T_jump = ts
      logt = logt + log10(g_eta_for(k))

      if (logt .lt. 4.0d0) then
        lambdainter=0.d0
        goto 25
      endif

      do i=nt,1,-1
        if(logt.le.tgrid(i) .and. logt.gt. tgrid(i-1)) goto 20 
      enddo

      print*,'logt not found in tgrid - 2'
      stop


20    t1= 1.d0 / 10.d0 ** tgrid(i-1)
      t2 = 1.d0 / 10.d0 ** tgrid(i)
      tt = 1.d0 / 10.d0 ** logt
 
      w1=(t2-tt)/(t2-t1)
      w2=1.d0-w1

!print*,w1,w2

!********
!********Added this block to make lambdainter = 0
!********in case of lamda = 0    05/08/14
!********

      where(lambda(:,i-1).ne.0.d0 .and. lambda(:,i).ne.0.d0) 
         lambdainter(:)=10.d0**(w1*lambda(:,i-1)+w2*lambda(:,i))
      elsewhere
         lambdainter(:)=0.d0
      endwhere  

25    continue

! two shocks, weighted
      spec(:,l) = spec(:,l) + wf * f_eta_for(k)**2 * lambdainter(:)

endif
!***************************************************************************

end do ! cooling zones

enddo ! depth loop

dxi=1./float(nfeld)

spec = spec * dxi * 1.d-23 ! since RS codes in these units

allocate(enerx(0:nf))

enerx(0:nf-1)=energy(1:nf)
enerx(nf)=energy(nf)+x_binsyz

! for tests
!do l=1,lxmin
!do l=9,9
!  write(*,'(i4,3es12.4)') (i,enerx(i),spec(i,l),10.d0**lambda(i,66)*1.d-23, i=1,i_nbin)
!enddo
print*
print*,'X-ray cooling function calculated for all energies between'
print*,energy(1),' to ',energy(nf)+x_binsyz,' eV'
print*,'with bin-width = ',x_binsyz,' eV'

!transform eV grid to frequency grid (in Kayser)
dxi=hh*clight/ev
dxi=1.d0/dxi
enerx=enerx*dxi

return

end
!
!---------------------------------------------------------------------------
!
subroutine raymond_smith(xlogt,xlogne, hnuvec, binout, &
                         i_nbin,x_binmin,x_binsyz,i_nbln,x_blnmin,x_blnsyz, &
&                        i_num,i_iprint,i_jprint,i_iphot,i_idens,i_icx)
  
!  CALLS XSPCT FOR VARIOUS PARAMETERS:
!  Dec 17, 1991 : modify GAUNT5 to improve transition 1 of Li-like ions
!  Sep 21, 1993 : add option to choose the solar photospheric and corona
!  abundances from Anders and Grevesse (1989, Geochimica et Cosmochimica
!  53, 197) as well as the cosmic abundances from Allen 1973 (SAD)
!  Note that the default abundances are Allen 1973 cosmic values.

USE nlte_app, only: abund_h12

character*(*), parameter :: fpath='../inicalc/RaymondSmith/'


real, dimension(1000) :: hnuvec, binout
  
character (len=1) :: IREP, IAB, IABSEL  
COMMON / HLN / EXT (11), HALF, HALFC, ALFLY, ALFLYC, HBET, HBETC, &
 TWOP
COMMON / BLN / BLN (1000), BLNMIN, BLNSYZ, NBLN  
common / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
COMMON / PT / RF (500), TAU (150), TAUHE (150), TPLUS (150)  
COMMON / CONTIN / BRMEV (1000), RECEV (1000), TUFEV (1000)  
common / com / cnc (12, 30), PTOT (12, 220), abin (1000), bin ( &
 1000)
DIMENSION abcor (12), abphot (12)  
DATA ABUNJ / 10.93, 8.52, 7.96, 8.82, 7.92, 7.42, 7.52, 7.2, 6.9, &
 6.3, 7.6, 6.3 /, ABCOR / 10.14, 7.90, 7.40, 8.30, 7.46, 7.59, &
 7.55, 6.93, 5.89, 6.46, 7.65, 6.22 /, ABPHOT / 10.99, 8.56, 8.05, &
 8.93, 8.09, 7.58, 7.55, 7.21, 6.56, 6.36, 7.67, 6.25 /

DIMENSION iabindex(12)
DATA IABINDEX /2,6,7,8,10,12,14,16,18,20,26,28/

DATA RF / 500 * 1.0 /  

logical start
data start / .true. /

open (unit = 1, file = fpath//'file1.dat')  
open (unit = 2, file = fpath//'atomic.dat')  
open (unit = 3, file = fpath//'file2.dat')  

!open (unit = 7, file = 'answers.dat')  

NBIN=I_NBIN
BINMIN=X_BINMIN
BINSYZ=X_BINSYZ  

NBLN=I_NBLN
BLNMIN=X_BLNMIN
BLNSYZ=X_BLNSYZ  


NUM=I_NUM
IPRINT=I_IPRINT
JPRINT=I_JPRINT
IPHOT=I_IPHOT
IDENS=I_IDENS
ICX=I_ICX  

CALL ATREAD (NUM)  
IF (IPHOT.NE.0) THEN  
   print * , 'RF1?'  
   READ *, RF1  
   CALL RADRD (1)  
   DO 10 N = 1, NBIN  
   10    ABIN (N) = RF1 * ABIN (N)  

ENDIF  

   20 continue  


    1 FORMAT (A1)  

IAB='N'
! this needs to be changed

IF (IAB.NE.'Y'.AND.IAB.NE.'y') GOTO 30  
   25 CONTINUE  
print * , 'WANT CORONAL(C), PHOTOSPHERIC(P), OR 1 by 1(1)?'  
READ (5, 1) IABSEL  
IF (IABSEL.EQ.'C'.OR.IABSEL.EQ.'c') goto 251  
IF (IABSEL.EQ.'P'.OR.IABSEL.EQ.'p') goto 253  
print * , 'NO, ABUND?'  
READ (5, * ) NO, ABUND  
!
ABUNJ (NO) = ABUND  
!
print * , ' ANOTHER ABUNDANCE MODIFICATION?'  
READ (5, 1) IAB  
IF (IAB.EQ.'Y'.OR.IAB.EQ.'y') GOTO 25  
GOTO 30  
  251 CONTINUE  
DO 252 I = 1, 12  
  252 ABUNJ (I) = ABCOR (I)  
GOTO 30  
  253 CONTINUE  
DO 254 I = 1, 12  
  254 ABUNJ (I) = ABPHOT (I)  
   30 CONTINUE  

!here we overwrite our own abundances
!we do this only once, since ABUNJ in DATA statement
if(start) then
print*
print*,' used abundances (consistent)'
do i=1,12
ki=iabindex(i)
xnewab=log10(abund_h12(ki))+12.d0
print*,i,ki,xnewab
abunj(i)=xnewab
enddo
start=.false.
endif

T = 10.**XLOGT  
DENE = 10.**XLOGNE  

! COMPUTE RATIO OF IONIZED TO NEUTRAL HYDROGEN
CIONH = CION (1, 1, 13.6, T)  
if (iphot.ne.0) cionh = cionh + phot (1, 1, 13.6, t, dene) &
 / dene
BE = 157890. / T  
CREC = 2.06E-11 * (.4288 + .5 * ALOG (BE) + .469 * BE** ( - &
 .33333) ) / SQRT (T)

RHY = CIONH / CREC  

!write (7, * ) ' rhy = ', rhy  
!
!
!for tests
!print*,nbin,binmin,binsyz
!print*,nbln,blnmin,blnsyz
!print*,num,iprint,jprint,iphot,idens,icx
!print*,t,dene



CALL FELINE (T, DENE, NUM, IPRINT, JPRINT, 0, IPHOT, IDENS, ICX)  
!  put hydrogen emission into bins
!     st = sqrt(t)
!     call hline(rhy,0.,t,st,be,0)
!     critn = 100.*sqrt(t)
!     p2ph = ext(11)*critn/(critn+dene)
!     ext(9) = ext(9) + ext(11)*(1.-critn/(critn+dene))
!     imax = (12399./1215.6 - binmin)/binsyz
!     imax= min0(imax,nbin)
!     if (imax .le. 1) go to 76
!     DO 75 ik= 1,imax
!     r = (binmin+binsyz*(ik-.5)) * 1215.6/12399.
!     tufev(ik) = 12.*p2ph*r*r*(1.-r)*1215.6*binsyz/12399.
!     print *,ik,tufev(ik)
!  75 bin(ik) = bin(ik) + tufev(ik)
!  76 continue
!     ilmin = (1215.6 - blnmin) / blnsyz + 1
!       DO 85 ibln = ilmin,nbln
!       wbln = (ibln - 0.5)*blnsyz + blnmin
!       ephot = 12399. / wbln
!       ibn = (ephot-binmin) / binsyz + 1
!       de = ephot - 12399. / (wbln + blnsyz)
!       if (ibn .ge. 1 .and. ibn .le. nbin) bln(ibln) = bln(ibln) +
!    1  tufev(ibn) * de / binsyz
!  85   continue
ibn = (10.2 - binmin) / binsyz + 1  
ibl = 0
if (ibn.ge.1.and.ibn.le.nbin) bin (ibn) = bin (ibn) + ext (6) &
 + ext (9)
if (blnsyz.gt.0) ibl = (1215.6 - blnmin) / blnsyz + 1  

if (ibl.ge.1.and.ibl.le.nbln) bln (ibl) = bln (ibl) + ext (6) &
 + ext (9)
IF (NBIN.EQ.0) GOTO 70  
WRITE (1, 50) T, BINMIN, BINSYZ, ABUNJ  
   50 FORMAT (' TEMPERATURE=',E10.2,' BINMIN, &
&  BINSYZ=',2F7.1,' ABUNJ',12F6.2)
WRITE (1, 60) (BIN (K), K = 1, NBIN)  

   60 FORMAT (10E12.3)  
IF (NBLN.EQ.0) GOTO 70  
WRITE (3, 50) T, BLNMIN, BLNSYZ, ABUNJ  
WRITE (3, 60) (BLN (K), K = 1, NBLN)  
   70 CONTINUE  

IREP='n'  
IF (IREP.EQ.'Y'.OR.IREP.EQ.'y') GOTO 20    

DO I = 1, NBIN  
   HNUVEC(I) = BINMIN + I * BINSYZ - BINSYZ  
   BINOUT(I)= BIN (I)
ENDDO

CLOSE(1)
CLOSE(2)
CLOSE(3)



RETURN
end 


SUBROUTINE ATREAD (NUM)  
!  CALLS ATRD NUM TIMES TO READ IN ATOMIC DATA
COMMON / FEL / WJ (12, 220), FJ (12, 220), E3J (12, 220), PWR (12, &
 220), C1J (12, 30), C2J (12, 30), EJ (12, 30), EAJ (12, 30), &
 S2J (12, 30), LLJ (12, 30), S3J (12, 30), S4J (12, 30), S5J (12, &
 30)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / DAT / E (30), EA (30), S2 (30), WAVE (220), E3 (220), &
 F (220), LL (30), S3 (30), S4 (30), S5 (30)
!
DO 100 NO = 1, NUM  
   CALL ATRD (NO)  
   N = NJ (NO)  
   DO 50 J = 1, N  
      EJ (NO, J) = E (J)  
      EAJ (NO, J) = EA (J)  
      S2J (NO, J) = S2 (J)  
      LLJ (NO, J) = LL (J)  
      S3J (NO, J) = S3 (J)  
      S4J (NO, J) = S4 (J)  
      S5J (NO, J) = S5 (J)  
   50    END DO  
   DO 70 L = 1, 220  
      E3J (NO, L) = E3 (L)  
      FJ (NO, L) = F (L)  
      WJ (NO, L) = WAVE (L)  
   70    END DO  
  100 END DO  
RETURN  

END subroutine ATREAD
SUBROUTINE FELINE (T, DENE, NUM, IPR, JPR, JCONT, IPHOT, IDENS, &
 ICX)
COMMON / BLN / BLN (1000), BLNMIN, BLNSYZ, NBLN  
COMMON / FEL / WJ (12, 220), FJ (12, 220), E3J (12, 220), PWR (12, &
 220), C1J (12, 30), C2J (12, 30), EJ (12, 30), EAJ (12, 30), &
 S2J (12, 30), LLJ (12, 30), S3J (12, 30), S4J (12, 30), S5J (12, &
 30)
COMMON / DAT / E (30), EA (30), S2 (30), WAVE (220), E3 (220), &
 F (220), LL (30), S3 (30), S4 (30), S5 (30)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / RESULT / CONCE (30), CA (30), POWER (220), RHY, HENEUT, &
 HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
COMMON / AUG / CAJ (12, 30)  
COMMON / RATES / C1 (30), C2 (30), CTH  
COMMON / COM / CNC (12, 30), PTOT (12, 220), ABIN (1000), BIN ( &
 1000)
COMMON / CONTIN / BRMEV (1000), RECEV (1000), TUFEV (1000)  
DIMENSION PC (12)  
!
DO 10 I = 1, NBIN  
   RECEV (I) = 0.  
   TUFEV (I) = 0.  
   BRMEV (I) = 0.  
   10 BIN (I) = 0.  
NLINES = 220  
ST = SQRT (T)  
PCOOL = 0.  
ABUND = 12.  
CONCE (2) = RHY / (1. + RHY)  
IF (NBIN.NE.0) CALL BREMS (T, 1)  
IF (NBIN.NE.0) CALL RECEMS (T, 1, 0, 0)  
!
!  ELEMENT LOOP
DO 140 NO = 1, NUM  
   RE = 0.  
   TU = 0.  
   ABUND = ABUNJ (NO)  
   DO 20 L = 1, NLINES  
      POWER (L) = 0.  
      WAVE (L) = WJ (NO, L)  
      F (L) = FJ (NO, L)  
   20    E3 (L) = E3J (NO, L)  
   N = NJ (NO)  
   NN = N + 1  
   DO 21 J = 1, N  
      E (J) = EJ (NO, J)  
      EA (J) = EAJ (NO, J)  
      S2 (J) = S2J (NO, J)  
      CONCE (J) = CNC (NO, J)  
      LL (J) = LLJ (NO, J)  
      S3 (J) = S3J (NO, J)  
      S4 (J) = S4J (NO, J)  
   21    S5 (J) = S5J (NO, J)  
   E (N + 1) = 0.  
   LL (N + 1) = 0.  
   CONCE (N + 1) = CNC (NO, N + 1)  
!
   CALL SINGLE (T, DENE, NO, IO, IPR, JPR, JCONT, IPHOT, IDENS, &
    ICX)
!
   IF (NO.EQ.1) HENEUT = CONCE (1)  
   IF (NO.EQ.1) HEPLUS = CONCE (2)  
   DO 25 J = 1, NN  
      IF (JCONT.EQ.0) CNC (NO, J) = CONCE (J)  
      CAJ (NO, J) = CA (J)  
      C1J (NO, J) = C1 (J)  
   25    C2J (NO, J) = C2 (J)  
   IF (N.EQ.18.OR.N.EQ.20.OR.N.EQ.28) GOTO 26  
   IF (NBIN.NE.0) CALL BREMS (T, N)  
   IF (NBIN.NE.0) CALL RECEMS (T, N, 0, 0)  
   26    CONTINUE  
!
   DO 27 L = 1, 220  
   27    PWR (NO, L) = POWER (L)  
   IF (NBIN.NE.0) CALL BUND (NO)  
   PC (NO) = PCOOL + TU  
  140 END DO  
PCOOL = 0.  
DO 145 NO = 1, NUM  
  145 PCOOL = PCOOL + PC (NO)  
IF (NBIN.EQ.0) GOTO 170  
DO 150 I = 1, NBIN  
  150 BIN (I) = BIN (I) + RECEV (I) + BRMEV (I) + TUFEV (I)  
IF (JPR.EQ.0) GOTO 170  
WRITE (7, 151) T, BINMIN, BINSYZ  
  151 FORMAT (1H1,5H TEMP,F10.0,10H    BINMIN,F7.2,10H    BINSYZ,F7.2,/, &
&  ('  BIN    HNU',9X,'RECEMS     BREMS    2 PHOT     TOTAL')//)
DO 160 I = 1, NBIN  
   HNU = BINMIN + I * BINSYZ - BINSYZ  
   WRITE (7, 155) I, HNU, RECEV (I), BRMEV (I), TUFEV (I), &
    BIN (I)
  155 FORMAT ((1X,I5,F10.1,3X,4E10.3,1X))  
  160 END DO  
  170 CONTINUE  
IF (NBLN.NE.0) CALL BLND (NUM)  
RETURN  

END SUBROUTINE FELINE
SUBROUTINE ATRD (NO)  
COMMON / DAT / E (30), EA (30), S2 (30), WAVE (220), E3 (220), &
 F (220), LL (30), S3 (30), S4 (30), S5 (30)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
INTEGER :: AA  
!     READ ATOMIC DATA
IX = 0  
   25 READ (2, 26) N, J, E (J), EA (J), S2 (J), S3 (J), S4 (J), S5 (J), &
 LL (J)
   26 FORMAT (2I5,F5.0,5F7.0,I5)  
IY = LL (J)  
IF (IY.LE.0) GOTO 30  
DO 28 L = 1, IY  
   AA = IX + 3 * L - 3  
   28 READ (2, 29) (WAVE (AA + K), E3 (AA + K), F (AA + K), K = 1, 3)  
   29 FORMAT (9F6.0)  
IX = IX + 3 * IY  
   30 IF (N.GT.J) GOTO 25  
NJ (NO) = N  
RETURN  

END SUBROUTINE ATRD
SUBROUTINE SINGLE (T, DENE, NO, IO, IPRINT, JPRINT, JCONT, IPHOT, &
 IDENS, ICX)
INTEGER :: BB  
COMMON / DAT / E (30), S (30), C (30), WAVE (220), E3 (220), &
 F (220), LL (30), SIG (30), ALF (30), ES (30)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / RATES / C1 (30), C2 (30), CTH  
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
COMMON / STRMET / POPM (4, 12)  

DIMENSION JDUM (5), LDUM (5), WDUM (5), PDUM (5)  
PCOOL = 0.  
ABUND = 12.  
ST = SQRT (T)  
!  IONIZATION STATE OF H MUST BE PASSED FROM OUTSIDE
RE = 0.  
TU = 0.  
ABUND = ABUNJ (NO)  
N = NJ (NO)  
IX = 0  
NN = N + 1  
E (N + 1) = 0.  
LL (N + 1) = 0  
CALL NQUIL (T, N, DENE, JCONT, IPHOT, IDENS, ICX)  
POPM (1, NO) = PM (1)  
POPM (2, NO) = PM (2)  
POPM (3, NO) = PM (3)  
POPM (4, NO) = PM (4)  
LOC = 0  
IF (IPRINT.EQ.0) GOTO 50  
  300 WRITE (7, 31) N, ABUND, T, DENE  
   31 FORMAT ('0LINES FOR N=',I2,'     ABUND=',F5.2,'     T=',E9.3, &
& '     DENE=',E9.3)
IF (N.GE.6) WRITE (7, 32) N, PM  
WRITE (7, 38) (K, CONCE (K), K = 1, NN)  
   32 FORMAT (' CONCE FOR N=',I3,' METASTABLES ',4E10.3)  
   38 FORMAT (10(I3,E10.2),/,10(I3,E10.2),/,10(I3,E10.2),/)  
IF (IPRINT.GE.3) WRITE (7, 38) (K, C1 (K), K = 1, NN)  
IF (IPRINT.GE.3) WRITE (7, 38) (K, C2 (K), K = 1, NN)  
   50 IIK = IX  
IX = 0  
DO 270 IL = 1, N  
   IY = 3 * LL (IL)  
   IF (IY.LE.0.) GOTO 270  
   CN = CONCE (IL)  
   PW = 0.  
   DO 260 L = 1, IY  
      BB = IX + L  
      IF (CN.LT..00001) GOTO 230  
      IF (WAVE (BB) .LE.0.) GOTO 260  
      G = GAUNT_RS (T, E3 (BB), N, IL, L, DENE)  
      PW = 10.** (ABUND+11.) * CN * (ALPHADI (N, IL, L, BB, T) &
       * E3 (BB) * 1.602E-12 + 2.72E-15 * F (BB) * EXP ( - 11590 * &
       E3 (BB) / T) * G / ST) * 12398. / (WAVE (BB) * E3 (BB) )
  230       POWER (BB) = PW  
  255       PCOOL = PCOOL + PW  
  260    END DO  
   IF (IL.EQ.N - 1) CALL HESEQ (N, IL, T, DENE, IX)  
   IF (IL.EQ.N) CALL HYSEQ (N, IL, T, DENE, IX)  
  270 IX = IX + IY  
IF (IPRINT.LE.1) GOTO 380  
IX = 0  
DO 370 IL = 1, N  
   IY = 3 * LL (IL)  
   IF (CONCE (IL) .LE..001) GOTO 370  
   DO 360 L = 1, IY  
      BB = IX + L  
      IF (WAVE (BB) .LE.0.) GOTO 360  
      LOC = LOC + 1  
      JDUM (LOC) = IL  
      LDUM (LOC) = L  
      WDUM (LOC) = WAVE (BB)  
      PDUM (LOC) = POWER (BB)  
      IF (LOC.EQ.5) WRITE (7, 345) N, (JDUM (K), LDUM (K), &
       WDUM (K), PDUM (K), K = 1, 5)
  345 FORMAT   (I4,5(I4,I3,F8.2,E10.3))  
      LOC = MOD (LOC, 5)  
  360    END DO  
  370 IX = IX + IY  
IF (LOC.NE.0) WRITE (7, 345) N, (JDUM (K), LDUM (K), WDUM (K), &
 PDUM (K), K = 1, LOC)
  380 CONTINUE  
IF (IPRINT.NE.0) WRITE (7, 40) PCOOL, RE, TU  
   40 FORMAT (' PCOOL=',E10.3,' RE=',E10.3, &
&  ' TU=',E10.3)
  140 CONTINUE  
  170 CONTINUE  
RETURN  
END SUBROUTINE SINGLE


FUNCTION CION (N, J, E, T)  
!  SM YOUNGER JQSRT 26, 329; 27, 541; 29, 61   WITH MOORES FOR UNDONE
!  A0 FOR B-LIKE ION HAS TWICE 2S PLUS ONE 2P  AS IN SUMMERS ET AL
!  CHI = kT / I

DIMENSION A0 (30), A1 (30), A2 (30), A3 (30), B0 (30), B1 (30), &
 B2 (30), B3 (30), C0 (30), C1 (30), C2 (30), C3 (30), D0 (30), &
 D1 (30), D2 (30), D3 (30)
DATA A0 / 13.5, 27.0, 9.07, 11.80, 20.2, 28.6, 37.0, 45.4, 53.8, &
 62.2, 11.7, 38.8, 37.27, 46.7, 57.4, 67.0, 77.8, 90.1, 106., &
 120.8, 135.6, 150.4, 165.2, 180.0, 194.8, 209.6, 224.4, 239.2, &
 154.0, 268.8 /
DATA A1 / - 14.2, - 60.1, 4.30, 27 * 0. /  
DATA A2 / 40.6, 140., 7.69, 27 * 0. /  

DATA A3 / - 17.1, - 89.8, - 7.53, 27 * 0. /  
DATA B0 / - 4.81, - 9.62, - 2.47, - 3.28, - 5.96, - 8.64, - 11.32, &
 - 14.00, - 16.68, - 19.36, - 4.29, - 16.7, - 14.58, - 16.95, &
 - 19.93, - 23.05, - 26.00, - 29.45, - 34.25, - 38.92, - 43.59, &
 - 48.26, - 52.93, - 57.60, - 62.27, - 66.94, - 71.62, - 76.29, &
 - 80.96, - 85.63 /
DATA B1 / 9.77, 33.1, - 3.78, 27 * 0. /  
DATA B2 / - 28.3, - 82.5, - 3.59, 27 * 0. /  

DATA B3 / 11.4, 54.6, 3.34, 27 * 0. /  
DATA C0 / 1.85, 3.69, 1.34, 1.64, 2.31, 2.984, 3.656, 4.328, 5.00, &
 5.672, 1.061, 1.87, 3.26, 5.07, 6.67, 8.10, 9.92, 11.79, 7.953, &
 8.408, 8.863, 9.318, 9.773, 10.228, 10.683, 11.138, 11.593, &
 12.048, 12.505, 12.96 /
DATA C1 / 0., 4.32, .343, 27 * 0. /  
DATA C2 / 0., - 2.527, - 2.46, 27 * 0. /  

DATA C3 / 0., .262, 1.38, 27 * 0. /  
DATA D0 / - 10.9, - 21.7, - 5.37, - 7.58, - 12.66, - 17.74, &
 - 22.82, - 27.9, - 32.98, - 38.06, - 7.34, - 28.8, - 24.87, &
 - 30.5, - 37.9, - 45.3, - 53.8, - 64.6, - 54.54, - 61.70, - 68.86, &
 - 76.02, - 83.18, - 90.34, - 97.50, - 104.66, - 111.82, - 118.98, &
 - 126.14, - 133.32 /
DATA D1 / 8.90, 42.5, - 12.4, 27 * 0. /  
DATA D2 / - 35.7, - 131., - 8.09, 27 * 0. /  

DATA D3 / 16.5, 87.4, 1.23, 27 * 0. /  
CION = 0.  
CHIR = T / (11590. * E)  
IF (CHIR.LE..0115) RETURN  
CHI = AMAX1 (CHIR, 0.1)  
CH2 = CHI * CHI  
CH3 = CH2 * CHI  
ALPHA = (.001193 + .9764 * CHI + .6604 * CH2 + .02590 * CH3) &
 / (1.0 + 1.488 * CHI + .2972 * CH2 + .004925 * CH3)
BETA = ( - .0005725 + .01345 * CHI + .8691 * CH2 + .03404 * CH3) &
 / (1.0 + 2.197 * CHI + .2457 * CH2 + .002503 * CH3)
J2 = J * J  
J3 = J2 * J  

ISO = N - J + 1  
A = A0 (ISO) + A1 (ISO) / J + A2 (ISO) / J2 + A3 (ISO) / J3  
B = B0 (ISO) + B1 (ISO) / J + B2 (ISO) / J2 + B3 (ISO) / J3  
C = C0 (ISO) + C1 (ISO) / J + C2 (ISO) / J2 + C3 (ISO) / J3  

D = D0 (ISO) + D1 (ISO) / J + D2 (ISO) / J2 + D3 (ISO) / J3  
!  FE II EXPERIMENTAL IONIZATION MONTAGUE ET AL: D. NEUFELD FIT
IF (N.NE.26.OR.J.NE.2) GOTO 10  
A = - 13.825  
B = - 11.0395  
C = 21.07262  
D = 0.  

   10 CONTINUE  
CH = 1. / CHI  
FCHI = 0.3 * CH * (A + B * (1. + CH) + (C - (A + B * (2. + CH) ) &
 * CH) * ALPHA + D * BETA * CH)
if (iso.ge.4.and.iso.le.10) fchi = fchi * 1.59  
!  correct Younger JQSRT 27, 541 from table to graphs
CION = 2.2E-6 * SQRT (CHIR) * FCHI * EXP ( - 1. / CHIR) / (E * &
 SQRT (E) )
RETURN  
END FUNCTION CION
FUNCTION ALPHADI (N, J, L, LN, T)  
!  DIELECTRONIC RECOMBINATION : BURGESS AND TWORKOWSKI FOR H-LIKE

COMMON / DAT / E (30), S (30), C (30), WAVE (220), E3 (220), &
 F (220), LL (30), SIG (30), ALF (30), ES (30)
!  BURGESS AND TWORKOWSKI
Z12 = J - 12.  
TWORK = .84 + .5 / J**2 + .03 * Z12 / (1. + 4.5E-5 * Z12**3)  
IF (N.NE.J) TWORK = 1.  
ALPHADI = 0.  
DL = DELT (N, J, L)  
IF (DL.LE.0.) RETURN  
ZF = J - 1.  
B = SQRT (ZF) * (ZF + 1.) **2.5 / SQRT (ZF * ZF + 13.4)  
X = E3 (LN) / ( (ZF + 1.) * 13.6)  
A = SQRT (X) / (1. + .105 * X + .015 * X * X)  
EBAR = E3 (LN) / (1. + .015 * ZF**3 / (ZF + 1.) **2)  
if (n - j.ne.1) goto 25  
b = (0.5 * zf / sqrt (zf) ) * b  
x = .75 * j  
a = sqrt (x) * 0.5 / (1. + 0.21 * x + 0.03 * x * x)  
!  younger jqsrt 29, 67 for he-like ions

   25 continue  
ALPHADI = .0030 * T** ( - 1.5) * A * B * F (LN) * TWORK * DL * &
 EXP ( - EBAR * 11590. / T)
RETURN  

END FUNCTION ALPHADI
FUNCTION AUTOIN (N, J, T)  
!  INNERSHELL EXCITATION FOLLOWED BY AUTOIONIZATION FOR NA,MG,AL,SI
!  SEQUENCES.  CRANDALL ET AL PRA 25,143 FOR MG,SI.  MANN FOR FE* .75.
!  COMPROMISE BETWEEN COWAN AND MANN AND DOUBLE THAT.
!  USE 2S AT 2P ENERGY, ETC FOR B,C,... SEQUENCES
DIMENSION ET (20), OM (20)  
DATA ET / 0., 55., 0., 115., 0., 190., 0., 280., 0., 375., 5 * 0., &
 802., 0., 1002., 0., 0. /
DATA OM / 0., .34, 0., .16, 0., .18, 0., .20, 0., .22, 5 * 0., &
 .29, 0., .3, 0., 0. /
AUTOIN = 0.  
IF (N.LE.10) RETURN  
I = N - J + 1  
IF (I.LE.10.OR.I.GE.15) RETURN  
AUTOIN = 8.63E-6 * OM (N - 10) * EXP ( - 11590. * ET (N - 10) &
 / T) / SQRT (T)
!  ASSUMES THAT NUMBER OF 32,3P ELECTRONS DOESN'T MATTER
RETURN  

END FUNCTION AUTOIN

FUNCTION ALFLO (N, J, T)  
!  Low Temperature Dielectronic Recombination from Nussbaumer and
!  Storey: C - Si for 1000 - 60,000 K

dimension iion (8, 16), a (25), b (25), c (25), d (25), f (25)  

data iion / 40 * 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 4, 5, 6, 7, 0, 0, &
 0, 0, 8, 9, 10, 11, 12, 0, 0, 8 * 0, 0, 13, 14, 15, 16, 17, 18, &
 19, 8 * 0, 0, 20, 6 * 0, 0, 21, 22, 5 * 0, 0, 23, 24, 25, 4 * 0, &
 8 * 0, 8 * 0 /

data a / .0108, 1.8267, 2.3196, 0.0, 0.0320, - .8806, .4134, 0.0, &
 - .0036, 0.0, .0061, - 2.8425, 0.0, .0129, 3.6781, - .0254, &
 - .0141, 19.928, 5.4751, 1.2044, .0219, .7086, - .0219, 3.2163, &
 .1203 /

data b / - .1075, 4.1012, 10.7328, .6310, - .6624, 11.2406, &
 - 4.6319, .0238, .7519, 21.879, .2269, .2283, 0.0, - .1779, &
 14.1481, 5.5365, 33.8479, 235.0536, 203.9751, - 4.6836, - .4528, &
 - 3.1083, .4364, - 12.0571, - 2.690 /

data c / .2810, 4.8443, 6.8830, .1990, 4.3191, 30.7066, 25.9172, &
 .0659, 1.5252, 16.2730, 32.1419, 40.4072, 0., .9353, 17.1175, &
 17.0727, 43.1608, 152.5096, 86.9016, 7.6620, 2.5427, 7.0422, &
 .0684, 16.2118, 19.1943 /

data d / - .0193, .2261, - .1824, - .0197, .0003, - 1.1721, &
 - 2.2290, .0349, - .0838, - .7020, 1.9939, - 3.4956, 0., - .0682, &
 - .5017, - .7225, - 1.6072, 9.1413, - 7.4568, - .5930, - .1678, &
 .5998, - .0032, - .5886, - .1479 /

data f / - .1127, .5960, .4101, .4398, .5946, .6127, .2360, .5334, &
 .2769, 1.1899, - .0646, 1.7558, 0., .4516, .2313, .1702, .1942, &
 .1282, 2.5145, 1.6260, .2276, .4194, .1342, .5613, .1118 /
alflo = 0.  
if (j.eq.1.or.j.gt.8) return  

if (n.gt.14) return  

t4 = t * .0001  

if (t4.lt.0.1.or.t4.gt.6.) return  
if (n.eq.6.and.j.eq.2.and.t4.lt.0.2) return  
if (n.eq.8.and.j.eq.4.and.t4.lt.0.2) return  
if (n.eq.8.and.j.eq.6.and.t4.lt.0.4) return  
if (n.eq.10.and.j.eq.8.and.t4.lt.0.2) return  
if (n.eq.12.and.j.eq.2.and.t4.lt.0.15) return  

if (n.eq.14.and.j.eq.3.and.t4.lt.0.15) return  
ij = iion (j, n)  

if (ij.eq.0) return  

alflo = 1.0e-12 * (a (ij) / t4 + b (ij) + c (ij) * t4 + d (ij) &
 * t4 * t4) * exp ( - f (ij) / t4) / t4**1.5
return  



end FUNCTION ALFLO
!      FUNCTION ALFLO(N,J,T)
!  LOW TEMPERATURE DIELECTRONIC RECOMBINATION FROM NUSSBAUMER AND STOREY
!  C THROUGH O AND T < 60000 K
!      DIMENSION IION(8,3),A(12),B(12),C(12),D(12),F(12)
!        dimension asi(4),bsi(4),csi(4),dsi(4),fsi(4)
!      DATA IION/0,1,2,3,0,0,0,0, 0,4,5,6,7,0,0,0, 0,8,9,10,11,12,0,0/
!      DATA A/.0108,1.8267,2.3196,0.0,.032,-.8806,.4134,0.,-.0036,0.,
!     1  .0061,-2.8425/
!      DATA B/-.1075,4.1012,10.7328,.6310,-.6624,11.2406,-4.6319,.0238,
!     1  .7519,21.8790,.2269,.2283/
!      DATA C/.2810,4.8443,6.883,.199,4.3191,30.7066,25.9172,.0659,
!     1  1.5252,16.273,32.1419,40.407/
!      DATA D/-.0193,.2261,-.1824,-.0197,.0003,-1.1721,-2.229,.0349,
!     1  -.0838,-.7020,1.9939,-3.4956/
!      DATA F/-.1127,.5960,.4101,.4398,.5946,.6127,.2360,.5334,.2769,
!     1  1.1899,-.0646,1.7558/
!        DATA ASI/0.,-.0219,3.2163,0.1203/
!        DATA BSI/0.,.4364,-12.0571,-2.690/
!        DATA CSI/0.,.0684,16.2118,19.1943/
!        DATA DSI/0.,-.0032,-.5886,-.0099/
!        DATA FSI/0.,.1342,.5613,.1118/
!       ALFLO = 0.
!       IF (J .EQ. 1 .OR. N-J .LE. 1) RETURN
!       T4 = T *.0001
!       IF (T4 .GT. 6.0) RETURN
!       IF(T4 .LT. 0.1) RETURN
!        IF (N .EQ. 6 .AND. J .EQ. 2 .AND. T4 .LT. 0.2) RETURN
!        IF (N .EQ. 8 .AND. J .EQ. 4 .AND. T4 .LT. 0.2) RETURN
!        IF (N .EQ. 8 .AND. J .EQ. 6 .AND. T4 .LT. 0.4) RETURN
!       IF (N .EQ. 12 .OR. N .EQ. 14) GO TO 900
!IF (N .LT. 6 .OR. N .GT. 8) RETURN
!IJ = IION(J,N-5)
!ALFLO = 1.0E-12 * (A(IJ)/T4 + B(IJ) + C(IJ)*T4 + D(IJ)*T4**2) *
!    1  EXP(-F(IJ) / T4) / T4 ** 1.5
! 900   IF (N .EQ. 14) GO TO 950
!       IF (J .NE. 2) RETURN
!       ALFLO = 1.0E-12 * (1.2044/T4 -4.6836 +7.6620*T4 -.5930*T4**2) *
!    1  EXP(-1.6260 / T4) / T4 ** 1.5
!       RETURN
! 950   IF (J .GT. 4) RETURN
!       ALFLO = 1.0E-12*(ASI(J)/T4 +BSI(J) +CSI(J)*T4 +DSI(J)*T4**2) *
!    1  EXP(-FSI(J) / T4) / T4 ** 1.5
!       RETURN
!END
FUNCTION DIMET (N, J, T, B, DD)  
DIMENSION NOJ (30), E (12, 3), F (12, 3)  
DATA NOJ / 0, 1, 3 * 0, 2, 3, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, &
 10, 5 * 0, 11, 0, 12, 2 * 0 /
DATA E / 0, 10.55, 13.4, 16.3, 22.1, 27.9, 40.0, 40.5, 46.7, 52.8, &
 76.3, 86.2, 0., 12.28, 16.06, 19.8, 27.4, 35.0, 42.7, 51.0, 60.9, &
 73.5, 122., 145., 5 * 0., 4.46, 9.55, 14.5, 19.5, 24.5, 40.7, 51. &
 /
DATA F / 0., .26, .21, .18, .16, .12, .10, .091, .081, .075, .003, &
 .002, 0., .16, .17, .14, .084, .076, .071, .067, .063, .059, .049, &
 .046, 5 * 0., .61, .564, 0.458, .408, .366, .30, .25 /
IJ = N - J - 2  
IF (IJ.EQ.9) IJ = 3  
Z = J - 1  
NO = NOJ (N)  
X = E (NO, IJ) / (J * 13.6)  
A = SQRT (X) / (1. + .105 * X + .015 * X * X)  
EBAR = E (NO, IJ) / (1. + .015 * Z * Z * Z / (J * J) )  
DIMET = .003 * T** ( - 1.5) * A * B * F (NO, IJ) * EXP ( - 11590. &
 * EBAR / T) * DD
RETURN  

END function DIMET
SUBROUTINE BUND (NO)  
!  PUTS EMISSION LINES INTO ARRAY BIN.
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / COM / CNC (12, 30), PTOT (12, 220), ABIN (1000), BIN ( &
 1000)
COMMON / DAT / V (30), S (30), C (30), WAVE (220), E3 (220), &
 F (220), LL (30), SIG (30), ALF (30), ES (30)
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, PPU, POT, RE, TU, PM (4)
IBN (E) = (E-BINMIN) / BINSYZ + 1  
DO 100 L = 1, 220  
   IF (WAVE (L) .LE.0) GOTO 100  
   E = 12399. / WAVE (L)  
   IK = IBN (E)  
   IF (IK.LT.1) GOTO 100  
   IF (IK.GT.NBIN) GOTO 100  
!
!  MOVE PHOTONS IN SAME BIN AS POPULAR K EDGE BUT HIGHER ENERGY UP
!  TO NEXT HIGHER BIN:  B,C,N,O,F,NE,AL EDGES
!
   KEDGE = IBN (187.03)  
   IF (IK.EQ.KEDGE.AND.E.GT.187.03) IK = IK + 1  
   KEDGE = IBN (284.05)  
   IF (IK.EQ.KEDGE.AND.E.GT.284.05) IK = IK + 1  
   KEDGE = IBN (400.07)  
   IF (IK.EQ.KEDGE.AND.E.GT.400.06) IK = IK + 1  
   KEDGE = IBN (532.09)  
   IF (IK.EQ.KEDGE.AND.E.GT.532.09) IK = IK + 1  
   KEDGE = IBN (692.14)  
   IF (IK.EQ.KEDGE.AND.E.GT.692.14) IK = IK + 1  
   KEDGE = IBN (874.16)  
   IF (IK.EQ.KEDGE.AND.E.GT.874.16) IK = IK + 1  
   KEDGE = IBN (1559.3)  
   IF (IK.EQ.KEDGE.AND.E.GT.1559.3) IK = IK + 1  
   BIN (IK) = BIN (IK) + POWER (L)  
  100 END DO  
RETURN  

END SUBROUTINE BUND


FUNCTION FECOOL (DENE, T, IPR)  
!  COOLING DUE TO FE II AND FE III ASSUMING THAT CONCE=CONCE(MG)
!  IF IRON IS NOT EXPLICITLY CALCULATED.
!  NUSSBAUMER AND STOREY FE II OMEGA'S AND ROBB FE III
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / COM / CNC (12, 30), PTOT (12, 220), ABIN (1000), BIN ( &
 1000)
COMMON / FE / FEINF, FEPERM, FEFORB, FEIII, FEIIIO  

common / feir / fe16, fe86, fe25  
ST = SQRT (T)  
C2 = CNC (6, 2)  
C3 = CNC (6, 3)  
IF (NJ (11) .EQ.26) C2 = CNC (11, 2)  
IF (NJ (11) .EQ.26) C3 = CNC (11, 3)  
FCT = 1.6E11 * 10.** (ABUNJ (11) - 12.) * 8.63E-6 * C2 / ST  
FEINF = FCT * (.31 * .048 * EXP ( - 553. / T) + .42 * .99 * EXP ( &
 - 11400. / T) )
FE16 = FCT * .39 * .99 * EXP ( - 11400. / T) * 1130. * ST / &
 (DENE+1130. * ST)
FE86 = FCT * .075 * 1.67 * EXP ( - 19400. / T) * 7050. * ST / &
 (DENE+7050. * ST)
FE25 = FCT * .3 * .048 * exp ( - 553. / T) * 580. * ST / ( &
 DENE+580. * ST)
FEINF = FEINF * 580. * ST / (DENE+580. * ST)  
FEPERM = FCT * 3.4 * 4.47 * EXP ( - 55000. / T)  
FEFORB = FCT * 0.1 * 2.9 * EXP ( - 33000. / T) * 1.2E6 * ST / &
 (DENE+1.2E6 * ST)
FCT = 1.6E11 * 10.** (ABUNJ (11) - 12.) * 8.63E-6 * C3 / ST  
FEIII = FCT * .30 * .054 * EXP ( - 625. / T) * 1590. * ST / &
 (DENE+1590. * ST)

FEIIIO = FCT * .43 * 2.66 * EXP ( - 30840. / T) * 521000. * ST / &
 (DENE+521000. * ST)
FC = FEINF + FEPERM + FEFORB + FEIII + FEIIIO  
FECOOL = FC  
IF (IPR.NE.0) PRINT 1, FC, FEINF, FEPERM, FEFORB, FEIII, FEIIIO  
    1 FORMAT   (' IRON COOLING; FC,II IR, II UV, II OPT, III IR, III &
&  OPT ',6F9.6)
RETURN  

END FUNCTION FECOOL
FUNCTION CTHR (N, J, T)  
!  SCOTT'S RATES FOR H I + J TO H II + (J-1)
!  THROUGH ARGON : NEUFELT'S FE II - FE III
!
DIMENSION A (30), B (30)  
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
DATA A / 0., 0., .00001, 0., 5.9, 6.2, 7.8, 9.4, 11., 13., 14., &
 16.1, 17.6, 19., 20.4, 21.8, 23.3, 13 * 0. /
DATA B / 4 * 0., .25, .18, .13, .1, .08, .05, .04, 19 * 0. /  
!
CTHR = 0.  
IF (J.EQ.1) RETURN  
T4 = AMAX1 (T * .0001, 0.1)  
T4 = AMIN1 (T4, 10.)  
!  NEUFELD'S FE II - FE III, FE III-FE IV, AND NI II - NI III RATES
IF (N.EQ.26.AND.J.EQ.3) CTHR = 1.0E-9 * (1.25 + .25 * ALOG10 (T4) &
 )
IF (N.EQ.26.AND.J.EQ.4) CTHR = 3.4E-9 * SQRT (T4)  
IF (N.EQ.28.AND.J.EQ.3) CTHR = 1.0E-9 * (.34 + 1.86 * T4)  
IF (N.GT.18) RETURN  
CTHR = A (J) * (1. + B (J) * T4) * 1.0E-9  
IF (J.GE.6) RETURN  
IF (N.NE.2) GOTO 6  
CTHR = 1.9E-15  
IF (J.EQ.3) CTHR = 1.7E-13  
RETURN  
    6 IF (N.NE.6) GOTO 7  
!  ALBERT'S C III TRIPLET P
IF (J.EQ.3) CTHR = 2.5E-9 * PM (1) * EXP ( - 15000. / T)  
IF (J.EQ.4) CTHR = 2.9E-9  
IF (J.EQ.5) CTHR = 7.6E-10 * T4**1.48  
RETURN  
    7 IF (N.NE.7) GOTO 8  
CTHR = 1.1E-12 / (1. + .1 * T4)  
IF (J.EQ.3) CTHR = 5.2E-10  
IF (J.EQ.4) CTHR = 2.7E-9 * T4**.926  
IF (J.EQ.5) CTHR = 1.7E-10 * T4**1.40  
!  ALBERT'S N VI
IF (J.EQ.6) CTHR = 6.6E-10  
RETURN  
    8 IF (N.NE.8) GOTO 10  
CTHR = 4.0E-10  
IF (J.EQ.3) CTHR = 7.7E-10 * SQRT (T4)  
IF (J.EQ.4) CTHR = 2.1E-9  
IF (J.EQ.5) CTHR = 1.4E-10 * (1. + T4)  
!  ALBERT'S O VII : NEED O VI
IF (J.EQ.7) CTHR = 5.4E-8  
RETURN  
   10 IF (N.NE.10) GOTO 12  
IF (J.EQ.4) CTHR = 3.8E-9 * SQRT (T4)  
RETURN  
   12 IF (N.NE.12) GOTO 14  
IF (J.EQ.4) CTHR = 4.4E-9 * (1. + .37 * T4)  
RETURN  
   14 IF (N.NE.14) GOTO 16  
IF (J.EQ.3) CTHR = 4.0E-9 * T4**.23  
IF (J.EQ.4) CTHR = 4.1E-10  
IF (J.EQ.5) CTHR = 2.2E-9 * (1. + .1 * T4)  
RETURN  
   16 IF (N.NE.16) GOTO 18  
IF (J.EQ.4) CTHR = 2.5E-9  
IF (J.EQ.5) CTHR = 7.0E-9  
RETURN  
   18 IF (N.NE.18) GOTO 20  
IF (J.EQ.4) CTHR = 4.4E-8 * T4**.27  
RETURN  
   20 RETURN  

END FUNCTION CTHR
FUNCTION CTHI (N, J, T)  
!  SCOTT'S RATES FOR   H II  +  J    TO    HI + J+1
!  THROUGH ARGON
T4 = AMAX1 (.0001 * T, 0.1)  
T4 = AMIN1 (T4, 10.)  
CTHI = 0.  
IF (J.GT.2) RETURN  
IF (J.EQ.2) GOTO 2  
IF (N.EQ.6) CTHI = 2.5E-15  
IF (N.EQ.7) CTHI = 3.3E-13 * T4**.12  
IF (N.EQ.8) CTHI = 3.3E-10  
IF (N.EQ.16) CTHI = 6.7E-10  
RETURN  
    2 IF (N.EQ.12) CTHI = 2.6E-11 * EXP ( - 6.0 / T4)  
IF (N.EQ.14) CTHI = 9.6E-10 * EXP ( - 2.74 / T4)  
!  NEUFELD'S FE II - FE III RATES
IF (N.EQ.26) CTHI = 1.0E-9 * 1.666 * (1.25 + .25 * ALOG10 (T4) ) &
 * EXP ( - 3.005 / T4)
RETURN  

END FUNCTION CTHI
FUNCTION CTHER (N, J, T)  
!  FITS TO SCOTT'S RATES FOR  HE I + J   TO   HE II + J-1
!  1000  TO 100,000 K THROUGH ARGON
DIMENSION A (2, 9)  
DATA A / 4 * 0., .059, .00001, 1.1, 0.7, .00001, 1.7, .075, 2.2, &
 .096, 1.2, 1.1, .0008, .00001, 1.0 /
!
CTHER = 0.  
IF (J.EQ.1) RETURN  
T4 = AMAX1 (.0001 * T, 0.1)  
T4 = AMIN1 (T4, 10.)  
CR = 5.4E-10 * T4  
IF (J.EQ.2.OR.N.GT.18) RETURN  
CTHER = AMAX1 (CR, ( - .1 + .4 * J) * 1.0E-9)  
IF (J.GE.6) RETURN  
IF (J.NE.3) GOTO 10  
CTHER = 0.  
!  INCLUDE ALBERT'S RATE TO 1D OF N+
IF (N.EQ.7) CTHER = 3.0E-10 * (1. + .26 * T4) + 6.2E-11  
IF (N.EQ.8) CTHER = 3.3E-10 * T4**.7  
IF (N.EQ.18) CTHER = 1.3E-10  
RETURN  
   10 IF (N.NE.7) GOTO 11  
CTHER = 1.1E-10  
IF (J.EQ.5) CTHER = 2.0E-9  
RETURN  
   11 CONTINUE  
IN = N / 2  
CTHER = A (J - 3, IN) * 1.0E-9  
IF (J.EQ.5) GOTO 12  
IF (N.EQ.6) CTHER = 5.1E-11 * T4**1.46  
IF (N.EQ.14) CTHER = 9.6E-10 * T4**.55  
IF (N.EQ.16) CTHER = 1.1E-9 * T4**.63  
RETURN  
   12 IF (N.EQ.18) CTHER = 1.0E-9 * T4**.91  
RETURN  

END FUNCTION CTHER
FUNCTION CTHEI (N, J, T, DENE)  
!  SCOTT'S RATES FOR HE+  + J  TO HE0  + J+1
!  THROUGH ARGON  0.1 < T4 < 10.
CTHEI = 0.  
T4 = 0.0001 * T  
IF (J.EQ.1.OR.J.GE.4) RETURN  
IF (J.EQ.3) GOTO 10  
IF (N.EQ.6) CTHEI = 5.3E-10 * EXP ( - 13.2 / T4)  
!  ALBERT'S RATES FOR N+ FROM 1D LEVEL
IF (N.NE.7) GOTO 70  
QDW = 5.14E-06 * DENE / SQRT (T)  
QUP = 2.86E-6 * EXP ( - 21826. / T)  
R21 = QUP / (QDW + .0041)  
D1 = R21 / (1. + R21)  
P3 = 1. - D1  
CTHEI = P3 * 4.1E-10 * EXP ( - 10.4 / T4) + D1 * 1.1E-10 * EXP ( - &
 3.59 / T4)
   70 CONTINUE  
IF (N.EQ.14) CTHEI = 2.9E-10 * EXP ( - 8.7 / T4)  
IF (N.EQ.18) CTHEI = 1.1E-10 * EXP ( - 3.57 / T4)  
RETURN  
   10 CONTINUE  
IF (N.EQ.14) CTHEI = 3.4E-9 * EXP ( - 10.5 / T4)  
IF (N.EQ.16) CTHEI = 4.8E-10 * EXP ( - 15.7 / T4)  
IF (N.EQ.26) CTHEI = 1.2E-9 * SQRT (T4) * EXP ( - 12. / T4)  
RETURN  

END FUNCTION CTHEI

FUNCTION POPMET (N, IJ, EMET, DENE, T)  
!  RETURNS POPULATION OF METASTABLE LEVEL OF BE,B,C, OR MG-LIKE ION;  CA
!  RESONANCE CONTRIBUTIONS CALC FOR HIGH Z WITH SMITH ET AL METHOD
!  PASSES REDUC BACK TO GAUNT_RS FOR INTERCOMBINATION LINES

REAL :: MGMM, MGM1, MGM5, MG21, MG5, MG1  
DIMENSION RM5 (12), EBE (12), pops (4)  
DIMENSION MGMM (3, 7), MGM1 (7), MG5 (7), MG21 (2, 7), EMG (7)  
DIMENSION NJ (30), A21 (2, 12), B21 (3, 12), C21 (3, 12), RM1 (12)  
DIMENSION RMM (3, 12), BM1 (12), BMM (3, 12), CM1 (3, 12), &
 CMM (2, 12)
DIMENSION ECM (2, 12), R (3, 3), X (3), C (3), CPR (3)  
DIMENSION RB5 (10), RC5 (10), RRES (4, 10), ERES (4, 10)  

DIMENSION RBE (10), BEE (10)  
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
COMMON / INTERC / REDUC (4)  

COMMON / METLN / PLN (4, 3, 8), TLN (4, 3, 8)  
DATA RM5 / 0., 18.6, 11.7, 6.09, 2.79, 1.54, .93, .61, .41, .31, &
 .13, .10 /
DATA EBE / 0., 6.20, 7.87, 9.51, 12.81, 16.1, 19.5, 23.11, 26.9, &
 28.9, 55.5, 63.1 /
DATA MG21 / 430., 5.8E-4, 22000., .008, 1.63E+5, .000, 9.E+5, .12, &
 3.3E+6, .26, 4.5E+7, 1.00, 8.E+7, 2. /


DATA MG5 / 0.2, 0.5, .26, .035, .026, .0058, .0039 /  
!  GILES QUOTED BY MENDOZA FOR S V, BALUJA FOR SI III
DATA MGM1 / .045, .35, .072, .026, .016, .0069, .0047 /  
DATA MGMM / .5, .38, 1.5, 1.4, 1.06, 4.1, .25, .39, 1.32, .29, &
 .21, .82, .17, .13, .51, .033, .25, .43, .02, .13, .30 /
DATA EMG / 1.63, 3.72, 5.50, 7.13, 8.72, 13.9, 17. /  
DATA NJ / 0, 1, 0, 0, 0, 2, 3, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, &
 10, 5 * 0, 11, 0, 12, 0, 0 /
DATA A21 / 2 * 0., 77.2, .00282, 471., .00629, 1880., .0118, &
 16200., .0306, 84500., .0629, 320000., .112, 972600., .1823, &
 2.546E+6, .2781, 5.914E+6, .4029, 7.3E+7, 1.21, 1.4E+8, 1.8 /
DATA B21 / 3 * 0., 332., 130., 82., 1183., 130., 347., 4210., &
 499., 1470., 28400., 3170., 8910., 152000., 19800., 56200., &
 705000., 102000., 288000., 43790., 11613., 27300., 4.28E+06, &
 7.08E+05, 2.42E+06, 1.02E+07, 1.55E+06, 5.62E+06, 8.E+7, 1.E+7, &
 5.E+7, 2.E+8, 2.E+7, 1.E+8 /


DATA C21 / 3 * 0., .00031, .0026, 32., .0041, .0342, 190., .022, &
 .16, 385., .599, 4.64, 6910., 4.67, 37.1, 45200., 26.1, 200., &
 197000., 115., 856., 680000., 424., 3000., 2.0E+06, 1370., 9090., &
 5.32E+06, 29400., 128000., 6.7E+7, 95000., 380000., 1.7E+8 /
!  REDUCED COLLISION RATES * 10. ** 7


DATA RM1 / 0., 9.384, 4.944, 2.814, 1.340, .7265, .4718, .3164, &
 .2243, .1732, .08, .062 /
!  INCLUDES PROTON RATES FROM MUNRO THROUGH SI, EXT F OR S, NONE ABOVE A
DATA RMM / 3 * 0., 29.9, 13.4, 50.8, 17.7, 8.41, 32.2, 11.2, 6.00, &
 21.8, 6.48, 4.19, 13.9, 3.96, 3.04, 9.68, 2.91, 2.67, 7.96, 2.19, &
 2.42, 6.83, 1.25, 0.40, 1.79, .533, .317, 1.40, .046, .16, .67, &
 .020, .12, .52 /
DATA BM1 / 0., 14., 8.1, 5.3, 4.1, 3.01, 2.0, 1.5, 1.1, .95, .67, &
 .58 /
DATA BMM / 3 * 0., 23., 7.9, 31., 17.1, 5.73, 20.8, 12.3, 4.17, &
 14.1, 6.52, 2.39, 7.91, 3.84, 1.47, 3.01, 2.65, 1.01, 3.16, 18.6, &
 13.9, 38., 1.68, .57, 1.9, 1.34, .43, 1.45, .69, .18, .69, .55, &
 .14, .54 /
DATA CM1 / 3 * 0., 12.1, 12.9, 10., 51., 29.5, 23., 12.0, 8.6, &
 4.18, 9.1, 6.4, 2.92, 6.75, 4.72, 2.1, 5.39, 3.86, 1.55, 4.17, &
 3.04, 1.19, 2.84, 2.02, 1.00, 2.43, 1.57, 0.90, 1.1, .75, .60, &
 1.0, .56, .54 /
DATA CMM / 2 * 0., 12.9, .0005, 32.5, .001, 29., .0015, 17.5, &
 .00205, 11.2, .00173, 7.42, .00173, 5.44, .00518, 4.1, .0069, &
 3.52, .0138, 1.7, .03, 1.5, .05 /
DATA ECM / 2 * 0., 1.26, 2.68, 1.90, 4.05, 2.42, 5.34, 3.76, 7.93, &
 5.09, 10.6, 6.58, 13.4, 8.34, 16.5, 10.6, 20.1, 13.5, 24.5, 30.1, &
 45., 51., 62. /
DATA RB5 / 0., 25., 21.6, 16.9, 8.80, 5.23, 3.64, 15.5, 2 * 0. /  
DATA RC5 / 10 * 0. /  
DATA BEE / 0., 10.5, 13.4, 16.3, 22.1, 28., 34., 40.5, 2 * 0. /  
DATA RBE / 0., 115., 95.4, 78.4, 47.7, 32.1, 24.6, 18.2, 2 * 0. /  
DATA ERES / 4 * 0., 12.7, 9.25, 7.94, 0., 16.3, 12.5, 11.5, 0., &
 19.7, 15.8, 14.9, 0., 27.0, 22.3, 21.9, 0., 33.8, 28.8, 28.9, &
 4.35, 40.9, 35.8, 36.3, 10.3, 48.4, 11.6, 44.1, 15.8, 8 * 0. /

DATA RRES / 4 * 0., 327., 95., 5.9, 0., 306., 111., 74.9, 0., &
 229., 97.8, 139., 0., 141., 60.7, 111., 0., 96., 41.3, 70.3, 68.0, &
 71.7, 30.3, 37.1, 956., 52.5, 22.0, 26.5, 64.3, 8 * 0. /
DO 9 I = 1, 3  
   DO 8 J = 1, 3  
    8    R (I, J) = 0.  

    9 END DO  
!  avoid overflow for low t
REDUC (IJ) = 0.  
POPMET = 0.  
IF (EMET * 11590. / T.GE.30.) RETURN  
ST = SQRT (T)  
NO = NJ (N)  
V = DENE * 1.E-7 / ST  
GOTO (1, 2, 3, 4), IJ  
    1 CONTINUE  
CN = CONCE (N - 3)  
R (1, 1) = - (RM1 (NO) + RMM (1, NO) * 3. + RMM (2, NO) * 5.) &
 * V
R (2, 2) = - A21 (1, NO) - V * (RM1 (NO) + RMM (1, NO) + RMM (3, &
 NO) * 1.6666667)
R (3, 3) = - A21 (2, NO) - V * (RM1 (NO) + RMM (2, NO) + RMM (3, &
 NO) )
R5 = RM5 (NO) * V * EXP ( - 11590. * EBE (NO) / T)  
R (1, 1) = R (1, 1) - R5  
R (2, 2) = R (2, 2) - R5  
R (3, 3) = R (3, 3) - R5  
R (2, 1) = V * RMM (1, NO) * 3.  
R (3, 1) = V * RMM (2, NO) * 5.  
R (1, 2) = V * RMM (1, NO)  
R (1, 3) = V * RMM (2, NO)  
R (3, 2) = V * 1.666667 * RMM (3, NO)  
R (2, 3) = V * RMM (3, NO)  
C (1) = - RM1 (NO) * EXP ( - EMET * 11590. / T) * V  
C (2) = C (1) * 3.  
C (3) = C (1) * 5.  
GOTO 10  
    2 CONTINUE  
!  B-LIKE IONS
CN = CONCE (N - 4)  
R5 = RB5 (NO) * V * EXP ( - 11590. * (ERES (2, NO) - EMET) &
 / T)
R (1, 1) = - (BM1 (NO) + BMM (1, NO) * 2. + BMM (2, NO) * 3.) &
 * V - B21 (1, NO)
R (2, 2) = - (BM1 (NO) + BMM (1, NO) + BMM (3, NO) * 1.5) * V - &
 B21 (2, NO)
R (3, 3) = - (BM1 (NO) + BMM (2, NO) + BMM (3, NO) ) * V - B21 (3, &
 NO)
R (2, 1) = V * BMM (1, NO) * 2.  
R (3, 1) = V * BMM (2, NO) * 3.  
R (1, 2) = V * BMM (1, NO)  
R (1, 3) = V * BMM (2, NO)  
R (3, 2) = V * 1.5 * BMM (3, NO)  
R (2, 3) = V * BMM (3, NO)  
C (1) = - BM1 (NO) * EXP ( - EMET * 11590. / T) * V / 3.  
C (2) = C (1) * 2.  
C (3) = C (1) * 3.  
GOTO 10  
    3 CONTINUE  
!  C-LIKE IONS
CN = CONCE (N - 5)  
R5 = RC5 (NO) * V * EXP ( - 11590. * (ERES (3, NO) - EMET) &
 / T)
E1 = EXP ( - ECM (1, NO) * 11590. / T)  
E2 = EXP ( - ECM (2, NO) * 11590. / T)  
E3 = EXP ( - EMET * 11590. / T)  
R (1, 1) = - (CM1 (1, NO) + CMM (1, NO) * .2 * E2 / E1 + CMM (2, &
 NO) * E3 / E1) * V - C21 (1, NO)
R (2, 2) = - (CM1 (2, NO) + CMM (1, NO) ) * V - C21 (2, NO)  
R (3, 3) = - (CM1 (3, NO) + CMM (2, NO) ) * V - C21 (3, NO)  
R (2, 1) = V * CMM (1, NO) * .2 * E2 / E1  
R (1, 2) = V * CMM (1, NO)  
R (3, 1) = V * CMM (2, NO) * E3 / E1  
R (1, 3) = V * CMM (2, NO)  
R (3, 2) = 0.  
R (2, 3) = 0.  
C (1) = - CM1 (1, NO) * V * 5. * E1 / 9.  
C (2) = - CM1 (2, NO) * V * E2 / 9.  
C (3) = - CM1 (3, NO) * V * E3 * 5. / 9.  
GOTO 10  
    4 CONTINUE  
CN = CONCE (N - 11)  
V = 8.63E-6 * DENE / ST  
NP = NO - 5  
MG1 = V * MGM1 (NP)  
MGM5 = V * MG5 (NP) * EXP ( - 11590. * EMG (NP) / T)  
IF (N.EQ.14) MGM5 = MGM5 / (1. + 8.6E-6 * T)  
IF (N.EQ.14) MG1 = MG1 / (1. + T * 3.5E-6)  
R (1, 1) = - MG1 - MGM5 - MGMM (1, NP) * V - MGMM (2, NP) * V  
R (2, 2) = - MG1 - MGM5 - MGMM (1, NP) * V / 3 - MGMM (3, NP) &
 * V / 3 - MG21 (1, NP)
R (3, 3) = - MG1 - MGM5 - MG21 (2, NP) - MGMM (2, NP) * V / 5. - &
 MGMM (3, NP) * V / 5.
R (2, 1) = V * MGMM (1, NP)  
R (3, 1) = V * MGMM (2, NP)  
R (1, 2) = V * MGMM (1, NP) / 3.  
R (1, 3) = V * MGMM (2, NP) / 5.  
R (3, 2) = V * MGMM (3, NP) / 3.  
R (2, 3) = V * MGMM (3, NP) / 5.  
C (1) = - MG1 * EXP ( - 11590. * EMET / T)  
C (2) = C (1) * 3.  
C (3) = C (1) * 5.  
   10 CONTINUE  
NN = NOSON (R, X, 3, 3)  
IF (NN.EQ.0) PRINT 99  
   99 FORMAT (' BAD INVERSION')  
DO 20 I = 1, 3  
   CPR (I) = 0.  
   DO 15 J = 1, 3  
   15    CPR (I) = CPR (I) + R (I, J) * C (J)  
   20 END DO  
SUM = 1. + CPR (1) + CPR (2) + CPR (3)  
POPS (1) = 1. / SUM  
POPS (2) = CPR (1) / SUM  
POPS (3) = CPR (2) / SUM  
POPS (4) = CPR (3) / SUM  
   30 CONTINUE  
POPMET = (CPR (1) + CPR (2) + CPR (3) ) / SUM  
IF (IJ.EQ.3) POPMET = CPR (3) / SUM  
FC = 1.E23 * 1.6E-12 * CN / DENE  
GOTO (51, 52, 53, 54), IJ  
!  ERGS * 10-23 / S / ATOM
   51 PLN (1, 1, NO) = (POPS (1) * RRES (1, NO) * V * EXP ( - 11590. * &
 ERES (1, NO) / T) + POPMET * R5) * FC * ERES (1, NO)
PLN (1, 2, NO) = POPMET * V * RBE (NO) * EXP ( - 11590. * BEE (NO) &
 / T) * FC * BEE (NO)
PLN (1, 3, NO) = POPS (3) * A21 (1, NO) * EMET * FC  
SUMC = - C (1) - C (2) - C (3)  
REDUC (1) = (POPS (3) * A21 (1, NO) + POPS (4) * A21 (2, NO) ) &
 / SUMC
RETURN  
   52 PLN (2, 1, NO) = (POPS (1) * RRES (2, NO) * V * EXP ( - 11590. * &
 ERES (2, NO) / T) + POPMET * R5) * FC * ERES (2, NO)
PLN (2, 3, NO) = (POPS (2) * B21 (1, NO) + POPS (3) * B21 (2, NO) &
 + POPS (4) * B21 (3, NO) ) * FC * EMET
SUMC = - C (1) - C (2) - C (3)  
REDUC (2) = (POPS (2) * B21 (1, NO) + POPS (3) * B21 (2, NO) &
 + POPS (4) * B21 (3, NO) ) / SUMC
RETURN  
   53 PLN (3, 1, NO) = (POPS (1) * RRES (3, NO) * V * EXP ( - 11590. * &
 ERES (3, NO) / T) + POPMET * R5) * FC * ERES (3, NO)
PLN (3, 3, NO) = POPS (4) * C21 (3, NO) * FC * EMET  
REDUC (3) = - POPS (4) * C21 (3, NO) / C (3)  
RETURN  
   54 PLN (4, 1, NO) = (POPS (1) * RRES (4, NO) * V * EXP ( - 11590. * &
 ERES (4, NO) / T) + POPMET * MGM5) * FC * ERES (4, NO)
PLN (4, 3, NO) = POPS (3) * MG21 (1, NP) * FC * EMET  
SUMC = - C (1) - C (2) - C (3)  
REDUC (4) = (POPS (3) * MG21 (1, NP) + POPS (4) * MG21 (2, NP) ) &
 / SUMC
RETURN  

END FUNCTION POPMET
FUNCTION NOSON (A, X, L, LMAX)  
!     RUDOLF LOESER, 28 JULY 1965
!     NOSON IS A MATRIX INVERSION ROUTINE, USING THE METHOD OUTLINED ON
!     P 434 OF HILDEBRAND, INTRODUCTION TO NUMERICAL ANALYSIS,
!     (NEW YORK,1956).
!     A IS A MATRIX OF ORDER L, WITH COLUMN DIMENSION LMAX, ITS ELEMENTS
!     ARE ASSUMED TO BE STORED COLUMNWISE, IN THE USUAL FORTRAN MANNER,
!     X IS WORKING STORAGE OF LENGTH L.
!     THE INVERSE OF A WILL REPLACE A.
!     UPON RETURN, NOSON=1 IF INVERSION WENT OK, =0 IF A DIVISOR IS
!     ZERO (IN WHICH CASE A MAY CONTAIN GARBAGE UPON RETURN).
DIMENSION A (1), X (1)  
N = L - 1  
MAX = N * LMAX + L  
DO 100 I = 1, L  
   X (I) = 1.  
  100 END DO  
K1 = - LMAX  
DO 110 K = 1, L  
   K1 = K1 + LMAX  
   K2 = K1 + K  
   IF (A (K2) ) 101, 115, 101  
  101    DO 104 I = 1, L  
      J1 = K1 + I  
      IF (A (J1) ) 102, 104, 102  
  102       F = 1. / A (J1)  
      X (I) = X (I) * F  
      DO 103 J1 = I, MAX, LMAX  
         A (J1) = A (J1) * F  
  103       END DO  
  104    END DO  
   A (K2) = X (K)  
   X (K) = 1.  
   DO 108 I = 1, L  
      KI = K - I  
      IF (KI) 105, 108, 105  
  105       J1 = K1 + I  
      IF (A (J1) ) 106, 108, 106  
  106       A (J1) = 0.  
      DO 107 J2 = I, MAX, LMAX  
         J1 = J2 + KI  
         A (J2) = A (J2) - A (J1)  
  107       END DO  
  108    END DO  
  110 END DO  
DO 113 I = 1, N  
   IF (X (I) ) 111, 115, 111  
  111    F = 1. / X (I)  
   DO 112 J1 = I, MAX, LMAX  
      A (J1) = A (J1) * F  
  112    END DO  
  113 END DO  
NOSON = 1  
  114 RETURN  
  115 NOSON = 0  
RETURN  
END FUNCTION NOSON

SUBROUTINE NQUIL (T, N, DENE, JCONT, IPHOT, IDENS, ICX)  
!   ************************  METASTABLE POPULATION DATA
!  July 93: CD1 and CD2 for Li-like ions work well for high Z, but
!  underestimates density effects for low Z
DIMENSION EMETJ (30, 4), CD1 (28), CD2 (28)  
DIMENSION RAT (30), PROD (30), C5 (30), C6 (30), OVER (30), &
 IMETA (30)

DIMENSION SEC (30)  
COMMON / DAT / E (30), EA (30), S2 (30), WAVE (220), E3 (220), &
 F (220), LL (30), S3 (30), S4 (30), S5 (30)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / RATES / C1 (30), C2 (30), CTH  

COMMON / RESULT / CONCE (30), CA (30), POWER (220), RHY, HENEUT, &
 HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
DATA EMETJ / 5 * 0., 6.5, 8.35, 10.2, 0., 14., 0., 17.6, 0., 21.6, &
 0., 25.3, 0., 29.0, 0., 32.7, 5 * 0., 47.0, 0., 51.8, 0., 0., 5 * &
 0., 5.34, 7.10, 8.87, 0., 12.5, 0., 16.0, 0., 19.8, 0., 24.0, 0., &
 28.6, 0., 34.2, 5 * 0., 57.1, 0., 64.8, 0., 0., 5 * 0., 4.18, &
 5.78, 7.48, 0., 11.0, 0., 14.7, 0., 18.6, 0., 23.0, 0., 28.0, 0., &
 33.9, 5 * 0., 60.4, 0., 69.2, 0., 0., 11 * 0., 2.72, 0., 6.55, 0., &
 10.4, 0., 14.2, 0., 18.1, 5 * 0., 29.7, 0., 38.9, 2 * 0. /
!
DATA IMETA / 0, 0, 0, 1, 2, 3, 5 * 0, 4, 18 * 0 /  
DATA CD1 / .0024, - .0005, .00, .061, .027, .011, .005, .005, 0., &
 .0107, .09, .13, .11, .081, .075, .066, .051, 11 * 0. /
DATA CD2 / 0., .01485, .30, .108, .107, .024, .075, .051, .054, &
 .0167, .36, 17 * 0. /
DATA SEC / 4 * 1., .9, .9, 6 * 1., .6, .5, 1., .7, .9, 13 * 1. /  
IX = 0  
ABHE = 10.** (ABUNJ (1) - 12.)  
DO 678 I = 1, 4  
  678 PM (I) = 0.  
ORANGE = 1.  
C5 (1) = 0.  
C5 (N + 1) = 0.  
JLOW = 1  
JMAX = N + 1  
JTOP = N + 1  
C6 (1) = 0  
T6 = T * 1.0E-6  
ST = SQRT (T)  
DO 301 J = 1, N  
!  IONIZATION RATE FROM YOUNGER
   EION = E (J)  
   C1 (J) = CION (N, J, EION, T)  
   IF (N - J.LE.1) GOTO 104  
!  IONIZATION FROM METASTABLE LEVEL : BE,B,C,MG SEQUENCES
   IF (IDENS.EQ.0) GOTO 105  
   IMET = IMETA (N - J + 1)  
   IF (IMET.EQ.0) GOTO 105  
   EMET = EMETJ (N, IMET)  
   FMET = POPMET (N, IMET, EMET, DENE, T)  
   PM (IMET) = FMET  
   EM = E (J) - EMET  
   C1 (J) = C1 (J) * (1. - FMET) + FMET * CION (N, J, EM, T)  
  105    CONTINUE  
!  INNERSHELL IONIZATION
   IF (N - J.LE.1) GOTO 104  
   EION = EA (J)  
   JP = N - 1  
   IF (N - J.GE.10) JP = N - 9  
   C1 (J) = C1 (J) + CION (N, JP, EION, T)  
  104    CONTINUE  
!  IONIZATION FOLLOWING INNERSHELL EXCITATION  :  COWAN AND MANN
!  INCREASED FOR FE
   C1 (J) = C1 (J) + AUTOIN (N, J, T)  
!  ****************      PHOTOIONIZATION
   IF (IPHOT.NE.0) C1 (J) = C1 (J) + PHOT (N, J, E (J), T, DENE) &
    / DENE
!  ****************      RADIATIVE RECOMBINATION
   ZEFF = J  
   BETA = ZEFF * ZEFF / (6.34 * T6)  
    2    C2 (J + 1) = 2.07E-11 * ZEFF * ZEFF * (0.4288 + 0.5 * ALOG ( &
    BETA) + 0.469 * (BETA** ( - 1. / 3.) ) ) / SQRT (T)
   APPLE = C2 (J + 1)  
   PLUM = 0  
   RECN = 0.  
   RGND = 0.  
   IF (N.EQ.J) GOTO 50  
   NVAL = GND (N - J) + 0.1  
   STARN = SQRT (13.6 / E (J) ) * ZEFF  
   DO 5000 NQM = 1, NVAL  
      QM = NQM  
      BE = BETA / (QM * QM)  
 5000    C2 (J + 1) = C2 (J + 1) - 5.197E-14 * J * SQRT (BE) * SEATON ( &
    BE, QM)
   IF (C2 (J + 1) .LT.0) C2 (J + 1) = 0  
   I = N - J  
!  RECOMBINATION TO GROUND STATE
   IF (I.LE.17) RGND = GREC (N, J, E (J), T)  
   IF (I.LE.1) GOTO 50  
   IF (I.GE.4.AND.I.LE.9) GOTO 50  
   EI = E (J)  
   IF (I.GE.17) GOTO 40  
!  RECOMBINATION TO OTHER STATES IN SAME SHELL
   LION = 1  
   IF (I.EQ.12) LION = 3  
   EI = EI - E3 (IX + LION)  
   40    STARN = SQRT (13.595 / EI) * ZEFF  
   BE = BETA / (STARN * STARN)  
   RECN = EFFNEW (N - J, T, ZEFF, N) * SEATON (BE, STARN) * &
    2.599E-14 * J * SQRT (EI * 11590. / T) / (STARN * STARN)
   50    C2 (J + 1) = C2 (J + 1) + RECN + RGND  
   IF (BETA.GE.0.5) GOTO 62  
   CHI = 1.2 * BETA + (0.55 + 1.04 * ALOG (BETA) ) * BETA * BETA + &
    ( - 0.43 + 1.01 * ALOG (BETA) ) * BETA**3
   GOTO 63  
   62    CHI = 0.5 * (0.735 + ALOG (BETA) + 1. / (3. * BETA) )  
   63    CONTINUE  
   DO 64 NQM = 1, NVAL  
      QM = NQM  
   64    CHI = CHI - P (BETA, QM)  
   CHI = CHI + EFFN (N - J, ZEFF, T) * P (BETA, STARN) / (2. * &
    STARN * STARN)
    6    C6 (J + 1) = E (J) * 1.6027E-12 * (C2 (J + 1) + 4.22E-22 * &
    SQRT (T / 1.037E-11) * CHI)
!  DIELECTRONIC RECOMBINATION ***** DENSITY DEPENDENCE AND SECONDARY
!                             ***** AUTOIONIZATION
   IY = LL (J) * 3  
   IF (J.EQ.1) GOTO 370  
   ZEFF = J - 1  
   RHO = (DENE / ZEFF**7.) **.2  
   DD = 1.  
   IF (RHO.GE.3) DD = 1. / (1. + (CD1 (N - J + 1) + CD2 (N - J + &
    1) / ZEFF) * RHO)
   DO 320 L = 1, IY  
      LN = IX + L  
      PLUM = PLUM + ALPHADI (N, J, L, LN, T) * AMIN1 (DD, SEC (N - &
       J + 1) )
  320    END DO  
!  DIELECTRONIC RECOMBINATION FOR METASTABLE LEVELS OF BE,B,C,MG ISO
   IMET = IMETA (N - J + 1)  
   IF (IMET.EQ.0) GOTO 321  
   ZF = J - 1  
   B = SQRT (ZF) * (ZF + 1.) **2.5 / SQRT (ZF * ZF + 13.4)  
   if (idens.gt.0) PLUM = PLUM * (1. - FMET)  
   if (idens.gt.0) PLUM = PLUM + FMET * DIMET (N, J, T, B, DD)  
  321    CONTINUE  
!   *************** ADD DIELECTRONIC RECOMB AND STOREY'S LOW T DIELEC RE
   C2 (J) = C2 (J) + PLUM + ALFLO (N, J, T)  
  370    OVER (J) = PLUM / ORANGE  
   IF (J.NE.1) C5 (J) = PLUM * E (J - 1) * 10.** (ABUND-1.) &
    * 1.6027
   IX = IX + IY  
  301 ORANGE = APPLE  
if (ICX.EQ.0) goto 306  
!  CHARGE TRANSFER CONTRIBUTION
FPLUS = RHY / (1. + RHY)  
DENH = 1. / (.003 + FPLUS + ABHE * HEPLUS + ABHE * 2. * (1. - &
 HENEUT - HEPLUS) )
C1 (N + 1) = 0.  
C2 (1) = 0.  
NN = N + 1  
!  **********************************  S. Butler Charge Transfer Rates
DO 305 J = 1, N  
   C1 (J) = C1 (J) + CTHI (N, J, T) * DENH * FPLUS + CTHEI (N, J, &
    T, DENE) * HEPLUS * ABHE
   C2 (J + 1) = C2 (J + 1) + CTHR (N, J + 1, T) * DENH * (1. - &
    FPLUS) + CTHER (N, J + 1, T) * HENEUT * ABHE
  305 END DO  
  306 continue  
!  AUGER IONIZATION
IF (IPHOT.EQ.2) CALL APHOT (N, DENE, JCONT)  
IF (JCONT.EQ.1) GOTO 725  
DO 400 J = 1, N  
   C2 (J + 1) = ABS (C2 (J + 1) )  
   IF (C1 (J) / C2 (J + 1) .LE.0.) GOTO 31  
   RAT (J) = C2 (J + 1) / C1 (J)  
   IF (J.EQ.1.AND.T.GE.1.0E8) RAT (J) = AMIN1 (RAT (J), 1.0E4)  
   GOTO 32  
   31    RAT (J) = 1.0E+6  
   32    CONTINUE  
   IF (RAT (J) .GE.1.0E-5) GOTO 34  
   JLOW = J + 1  
   34    IF (RAT (J) .LT.1.0E+5) GOTO 400  
   JMAX = J  
   GOTO 41  
  400 END DO  
IF (JLOW.EQ.JMAX) GOTO 525  
   41 JUMP = max(JMAX - 1,1)  
PROD (JUMP) = RAT (JUMP)  
KTOP = JUMP - JLOW  
SUM = 1.00000 + PROD (JUMP)  
IF (KTOP.EQ.0) GOTO 550  
DO 500 K = 1, KTOP  
   PROD (JUMP - K) = RAT (JUMP - K) * PROD (JMAX - K)  
   PROD (JUMP - K) = AMIN1 (PROD (JUMP - K), 1.0e30)  
  500 SUM = SUM + PROD (JUMP - K)  
GOTO 550  
  525 SUM = 1.0000  
  550 CONTINUE  
DO 600 J = 1, JTOP  
   CONCE (J) = 0  
  600 C5 (J) = 0.  
CONCE (JMAX) = 1.000 / SUM  
KMAX = JMAX - JLOW  
IF (KMAX.EQ.0) GOTO 725  
DO 700 K = 1, KMAX  
  700 CONCE (JMAX - K) = PROD (JMAX - K) * CONCE (JMAX)  
  725 CONTINUE  
IF (N.EQ.2) DNE = 0.  
WORL = 0.  
DEI = 0.  
DO 901 J = 1, JTOP  
   DNE = (J - 1) * CONCE (J) * 10.** (ABUND-12.) + DNE  
    8    C6 (J) = C6 (J) * CONCE (J) * 10.0** (11. + ABUND)  
   RE = RE+C6 (J)  
   C5 (J) = C5 (J) * CONCE (J)  
   PCOOL = PCOOL + C5 (J)  
   PCOOL = PCOOL + C6 (J)  
  901 END DO  
RETURN  
END SUBROUTINE NQUIL

SUBROUTINE HESEQ (N, J, T, DENE, IX)  
!  HELIUM-LIKE IONS  :  RECOMBINATION TO EXCITED LEVELS, INNER-SHELL
!  IONIZATION OF LI-LIKE IONS  AND DENSITY DEPENDENCE USING PRADHAN,
!  MEWE & SCHRIJVER,  BERRINGTON, FON & KINGSTON, DRAKE
!
COMMON / DAT / E (30), S (30), C (30), WAVE (220), E3 (220), &
 F (220), LL (30), SIG (30), ALF (30), ES (30)
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
common / twosv / hetwop  
!
DIMENSION BR (30), CR (3, 30)  
DATA BR / 0., 1.0, 3 * 0., .89, .78, .71, 0., .67, 0., .63, 0., &
 .58, 0., .50, 0., .40, .0, .33, 5 * 0., .20, 0., .18, 0., 0. /
DATA CR / 3 * 0., 7.6E6, 4540., 2210., 9 * 0., 1.9E10, 1.3E11, &
 9.6E11, 4.9E10, 6.5E11, 2.3E12, 1.25E11, 3.3E12, 5.8E12, 3 * 0., &
 2.4E12, 8.4E13, 1.4E14, 3 * 0., 2.3E13, 1.1E15, 1.9E15, 3 * 0., &
 2.3E14, 1.5E15, 2.7E16, 3 * 0., 9.5E14, 7.6E16, 1.3E17, 3 * 0., &
 3.9E15, 3.9E17, 6.2E17, 3 * 0., 1.6E16, 2.0E18, 3.0E18, 15 * 0., &
 4.5E17, 5.3E19, 8.2E19, 3 * 0., 1.0E18, 1.0E20, 2.0E20, 6 * 0. /
IF (CONCE (J) .LT..00001) GOTO 1000  
ABU = 10.** (ABUND-12.)  
TU = 0.  
EN = N  
!  EXCITATION OF 1S2S 1S LEVEL PRADHAN ; BALUJA,CALLAWAY,HENRY: 1.2 FOR
Y = 11590. * E3 (IX + 1) / T  
CC = ALOG ( (Y + 1.) / Y) - 0.4 / (Y + 1) **2  
gbar = (.06168 + .0002522 * n) + ( - .02404 - .001948 * n) &
 * y * cc + ( - .007604 + .002416 * n) * (y - y * y * cc)
!         GBAR = (.03014+.0009855*N)+(.2392-.007019*N)*Y*CC
!    1  +(-.1292+.003543*N)*(Y-Y*Y*CC)
IF (N.EQ.2) GBAR = .048 * (t / 10000.) ** (.201)  
OM = 1.2 * 14.5 * F (IX + 1) * GBAR * 13.6 / E3 (IX + 1)  
P2PH = (8.63E-6 * OM / SQRT (T) ) * EXP ( - Y) * ABU * CONCE (N - &
 1) * E3 (IX + 1) * 1.6E11

TU = P2PH  
!  INNERSHELL IONIZATION OF LI-LIKE ION FROM MEWE AND SCHRIJVER
IF (N.EQ.2) GOTO 10  
EIS = 13.6 * EN * EN * 3.** (.24 - .43 / ALOG10 (EN) )  
CIONIZE = 2.5E-8 * SQRT (T) * EXP ( - EIS * 11590. / T) / EIS**2  
ADDIT = CONCE (J - 1) * CIONIZE * E3 (IX + 2) * 1.6E11 * ABU  
POWER (IX + 2) = POWER (IX + 2) + ADDIT * .75  

P2PH = P2PH + ADDIT * .25  
   10 CONTINUE  
! RADIATIVE RECOMBINATION
CN = CONCE (J + 1) * ABU  
Z = N - 1.  
POWER (IX + 2) = POWER (IX + 2) + (5.9E-13 * Z**1.8 * T** ( - 0.4) &
 ) * (1. + 17 * Z**0.4 * T** ( - 0.2) ) * E3 (IX + 2) * 1.6E11 * &
 CN
POWER (IX + 3) = POWER (IX + 3) + (3.6E-11 * Z**2.4 * T** ( - 0.7) &
 + 3.6E-10 * Z**2.8 * T** ( - .9) ) * 1.6E11 * E3 (IX + 3) * CN
POWER (IX + 1) = POWER (IX + 1) + 1.6E11 * CN * E3 (IX + 1) &
 * 1.4E-11 * Z**2.52 * T** ( - .76) * (1. + 10. * Z**0.4 * T** ( - &
 0.2) )

P2PH = P2PH + CN * 1.6E11 * E3 (IX + 1) * (5.4E-13 * Z * Z * T** ( &
 - 0.5) ) * (1. + 17. * Z** (0.4) * T** ( - 0.2) )

hetwop = p2ph  
ZF = N - 1  
EFFECN = SQRT (ZF * ZF * 13.6 / (E (J) - E3 (IX + 7) ) )  
XN = 157890. * ZF * ZF / (T * EFFECN * EFFECN)  
RECRAT = 5.2E-14 * ZF * SQRT (XN) * SEATON (XN, EFFECN)  
ADDIT = (5. / 12.) * RECRAT * CN * (12399. / WAVE (IX + 7) ) &
 * 1.6E11
!       HYDROGENIC RECOMBINATION FOR 3D LINES EXCEPT HELIUM FROM ROBBINS
IF (N.EQ.2) ADDIT = CN * 5.00E-14 * (.0001 * T) ** ( - 1.333) &
 * 1.6E11 * 12399. / WAVE (IX + 7)
POWER (IX + 7) = POWER (IX + 7) + ADDIT  
POWER (IX + 8) = POWER (IX + 8) + .3333 * ADDIT * WAVE (IX + 7) &
 / WAVE (IX + 8)
! DIELECTRONIC RECOMB
POWER (IX + 1) = POWER (IX + 1) + 1.6E11 * E3 (IX + 1) * CN * ADH &
 (1, N, T)
POWER (IX + 2) = POWER (IX + 2) + 1.6E11 * E3 (IX + 2) * CN * ADH &
 (2, N, T)
POWER (IX + 3) = POWER (IX + 3) + 1.6E11 * E3 (IX + 3) * CN * ADH &
 (3, N, T)

P2PH = P2PH + 1.6E11 * E3 (IX + 1) * CN * ADH (4, N, T)  
! DENSITY DEPENDENCE :  PRADHAN; BERRINGTON, FON & KINGSTON FOR HE I
CRIT1 = CR (1, N)  
CRIT2 = CR (2, N)  
CRIT3 = CR (3, N)  
CRITM = 1. / (1. / CRIT1 + 1. / CRIT2 + 1. / CRIT3)  
CRIT = 5000. * Z**8 * SQRT (T)  
IF (N.EQ.2) CRIT = 3.1E5 * SQRT (T)  
P2DUM = P2PH  

P2PH = (P2PH + POWER (IX + 2) * (CRITM / CRIT3) * DENE / (DENE+ &
 CRITM) ) * CRIT / (DENE+CRIT)
POWER (IX + 1) = POWER (IX + 1) + P2DUM * DENE / (DENE+CRIT) &
 + POWER (IX + 2) * (CRITM / CRIT2) * DENE / (DENE+CRITM)
BRANCH = BR (N)  
PDUM = POWER (IX + 3)  
POWER (IX + 3) = POWER (IX + 3) * (1. - BRANCH) + POWER (IX + 2) &
 * (CRITM / CRIT1) * (DENE / (DENE+CRITM) )
FC = BRANCH * (1. - BRANCH)  
IF (N.EQ.2) FC = 2.03E-5  
POWER (IX + 6) = PDUM * BRANCH * WAVE (IX + 3) / WAVE (IX + 6) &
 + POWER (IX + 2) * (WAVE (IX + 2) / WAVE (IX + 6) ) * FC * &
 (DENE / CRIT1) / (1. + DENE / CRITM)
POWER (IX + 2) = (POWER (IX + 2) + BRANCH * PDUM) * CRITM / &
 (DENE+CRITM)
!
WAV2 = WAVE (IX + 1)  
IF (NBIN.NE.0) CALL TWOPH (WAV2, P2PH, 0)  
 1000 CONTINUE  
RETURN  

END SUBROUTINE HESEQ
FUNCTION ADH (L, N, T)  
!  DIELECTRONIC RECOMBINATION TO HE-LIKE EXCITED STATES
!  USING MEWE&SCHRIJVER
P = 33 * (N - 1.) **0.6 * T** ( - 0.3)  
Z = N  
Z2 = Z * Z  
Z3 = Z * Z2  
Z4 = Z * Z3  
ZPT = (Z + .5) * (Z + .5)  
A = 6.46E-8 * Z4 * T** ( - 1.5)  
EX1 = EXP ( - 78900. * ZPT / T)  
EX2 = EXP ( - 101800. * Z2 / T)  
EX3 = EXP ( - 118400. * Z2 / T)  
GOTO (1, 2, 3, 4), L  
    1 ADH = A * (12. * EX1 / (1. + 6.E-6 * Z4) + 18. * EX2 / (1. + &
 3.0E-5 * Z4) + 69. * EX3 / (1. + 5.0E-3 * Z3) )
RETURN  
    2 ADH = A * (9. * EX1 / (1. + 7.E-5 * Z4) + 27. * EX2 / (1. + &
 8.0E-5 * Z4) + 380. * EX3 / ( (1. + P) * (1. + 5.0E-3 * Z3) ) )
RETURN  
    3 ADH = A * (18. * EX1 / 9.5 + 54 * EX2 / (1. + 1.9E-4 * Z4) &
 + 380. * EX3 * P / ( (1. + P) * (1. + 5.0E-3 * Z3) ) )
RETURN  
    4 ADH = A * (3. * EX1 / (1. + 3.0E-6 * Z4) + 0.5 * EX2 / (1. + &
 2.2E-5 * Z4) + 6.3 * EX3 / (1. + 5.0E-3 * Z3) )
RETURN  

END FUNCTION ADH

SUBROUTINE HYSEQ (N, J, T, DENE, IX)  
!  HYDROGENIC IONS : ADDS RECOMBINATION, DENSITY DEPENDENCE
!  HAYES AND SEATON 2S EXCITATION,  HYDROGENIC RECOMBINATION
COMMON / DAT / E (30), S (30), C (30), WAVE (220), E3 (220), &
 F (220), LL (30), SIG (30), ALF (30), ES (30)
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
IF (CONCE (J) .LT..00001) GOTO 1000  
ABU = 10.** (ABUND-12.)  
EN = N  
CN = CONCE (J) * ABU  
Y = E3 (IX + 1) * 11590. / T  
CC = ALOG ( (Y + 1) / Y) - 0.4 / (Y + 1) **2  
gbar = 0.047  
!     GBAR = 0.20 * Y * CC + 0.276 * CC
IF (N.EQ.1) GBAR = .017 + .1 * CC  
!  1.2 FOR CASCADES : TU IS DIRECT EXCITATION TWO PHOTON
OM = 1.2 * 14.5 * .416 * GBAR * 13.6 / E3 (IX + 1)  
P2PH = (8.63E-6 * OM / SQRT (T) ) * EXP ( - Y) * E3 (IX + 1) &
 * 1.6E11 * CN

TU = TU + P2PH  
!  RADIATIVE RECOMBINATION : CASE A : exponent revised Aug. '86
TP = .0001 * T / N**2  
CN = CONCE (J + 1) * ABU  
RC = N * 16.7E-14 * TP** ( - 0.91)  
POWER (IX + 1) = POWER (IX + 1) + RC * CN * E3 (IX + 1) * 1.6E11  
RC = N * 3.61E-14 * TP** ( - 0.72)  
POWER (IX + 2) = POWER (IX + 2) + RC * CN * E3 (IX + 2) * 1.6E11  
RC = N * 1.40E-14 * TP** ( - 0.757)  
POWER (IX + 3) = POWER (IX + 3) + RC * CN * E3 (IX + 3) * 1.6E11  
RC = N * .693E-14 * TP** ( - .750)  
POWER (IX + 4) = POWER (IX + 4) + RC * CN * E3 (IX + 4) * 1.6E11  
RC = N * 7.84E-14 * TP** ( - 1.04)  
IF (N.EQ.2) RC = 2 * 11.8E-14 * TP** ( - 1.02)  
!  CASE B FOR HELIUM PLUS
POWER (IX + 5) = POWER (IX + 5) + RC * CN * 1.6E11 * 12399. / &
 WAVE (IX + 5)
RC = N * 3.18E-14 * TP** ( - .610)  

P2PH = P2PH + RC * CN * 1.6E11 * E3 (IX + 1)  
!  DENSITY DEPENDENCE OF METASTABLE LEVEL MEWE AND GRONENSCHILD
CRIT = 100. * EN**8 * SQRT (T)  
POWER (IX + 1) = POWER (IX + 1) + P2PH * DENE / (DENE+CRIT)  
P2PH = P2PH * CRIT / (DENE+CRIT)  
WAV2 = WAVE (IX + 1)  
IF (NBIN.NE.0) CALL TWOPH (WAV2, P2PH, 0)  
POWER (IX + 5) = POWER (IX + 5) + POWER (IX + 2) * .11 * WAVE (IX &
 + 2) / WAVE (IX + 5)
POWER (IX + 2) = .89 * POWER (IX + 2)  
POWER (IX + 3) = .86 * POWER (IX + 3)  
POWER (IX + 4) = .78 * POWER (IX + 4)  
 1000 CONTINUE  
RETURN  
END SUBROUTINE HYSEQ
SUBROUTINE RECEMS (T, N, IPRINT, JPRINT)  
REAL :: KT  

LOGICAL :: NOTGRC (25), DONE (25), SWITCH  
DIMENSION DRAT (20), LION (2, 20), ZETA (2, 20), IEDGE (26), &
 EDGE (25), CHI (25), ESMIN3 (25), ALFX (25), JEDGE (25), IMAP (25)

DIMENSION S2X (25), S3X (25), S4X (25), S5X (25)  
COMMON / DAT / E (30), EA (30), S2 (30), WAVE (220), E3 (220), &
 F (220), LL (30), S3 (30), S4 (30), S5 (30)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / RATES / C1 (30), C2 (30), CTH  
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)

COMMON / CONTIN / BRMEV (1000), RECEV (1000), TUF (1000)  
IBN (Z) = INT ( (Z - BINMIN) / BINSYZ + 0.5) + 1  
ECEN (I) = BINMIN + (I - 0.5) * BINSYZ  

GFUNC (X, Y) = 1. + 0.1728 * (Y * X) ** ( - 2. / 3.) * (X - 2.) &
 - 0.0496 * (Y * X) ** ( - 4. / 3.) * (X * X - X * 2. / 3. + 2. / &
 3.)
DATA DRAT / 2., .5, 2., .5, 6., 2.5, 2.22, 2.25, .667, .167, 2., &
 .5, 6., 2.5, 2.22, 2.25, .667, .167, 2., .5 /
DATA LION / 0, 0, 3, 7, 1, 5, 2, 4, 4, 0, 4, 0, 5, 0, 2, 0, 2, 0, &
 3, 0, 1, 5, 1, 4, 3, 5, 1, 3, 1, 4, 1, 3, 1, 3, 1, 2, 4, 0, 6, 0 /

DATA ZETA / 0., 0., 2., 3., 2., 3., 1.75, 3., 3., 0., 3., 0., 3., &
 0., 3., 0., 3., 0., 3., 0., 3., 4., 2.83, 4., 2.67, 4., 2.5, 4., &
 2.33, 4., 2.17, 4., 2., 4., 1.83, 4., 4., 0., 4., 0. /
T6 = T / 1.E6  
KT = 86.17 * T6  
! UPPER LIMIT TO BINNING IS CHOSEN WHERE EXP = 1.E-20
IMAX = NBIN  
IF (IMAX.LT.1) RETURN  
AB = 10.** (ABUND-12.)  
EBK = EXP (BINSYZ / KT)  
Q = AB * KT * (EBK - 1.) * EXP ( - BINMIN / KT) / T6**1.5  
qprime = ab * kt / t6**1.5  
QGREC = Q * 1.31E-8  
QTK = Q * 6.52 / 12399.  
Z = N  
QHYD = Q * 5.28E-4 * (Z**4)  
qgrecpr = qprime * 1.31e-8  
qtkpr = qprime * 6.52 / 12399.  
qhydpr = qprime * 5.28e-4 * (z**4)  
EBK = 1. / EBK  
! LIMIT TO SIGNIFICANT CONTRIB FROM AN EDGE; ASSUMES #IONS PRESENT
!  GOES LIKE SQRT(Z); COMPARE THRESH RECEMS TO EDGLIM*(CURRENT CONTIN)
EDGLIM = 0.01 / SQRT (Z)  
! FIRST CONSTRUCT TABLE OF SIGNIFICANT EDGES
IX = 0  
J = 1  
ISO = N  
    4 IF (ISO.LE.20) GOTO 5  
IX = IX + 3 * LL (J)  
ISO = ISO - 1  
J = J + 1  
GOTO 4  
    5 IEMAX = 0  
   10 IF (ISO.EQ.1) GOTO 200  
IF (CONCE (J + 1) .LT.1.E-4) GOTO 190  
CHITRY = E (J)  
IGND = IBN (CHITRY)  
IF (IGND.GE.nbin) GOTO 100  
IF (IGND.LT.1) IGND = 1  
THRESH = S2 (J) + S3 (J) + S4 (J) + S5 (J)  
!
if (chitry / kt.gt.46.) recev (ignd) = recev (ignd) + qgrecpr * &
 conce (j + 1) * drat (iso) * (chitry**3) * thresh
if (chitry / kt.gt.46.) goto 100  
!
EDGTRY = QGREC * CONCE (J + 1) * DRAT (ISO) * (CHITRY**3) * &
 THRESH * EXP (CHITRY / KT)
EN = ECEN (IGND)  
X = CHITRY / EN  
X2 = X * X  
EDG = EDGTRY * (EBK**IGND) * (S2 (J) / X + S3 (J) + S4 (J) &
 * X + S5 (J) * X2) / THRESH
BINNOW = BRMEV (IGND) + RECEV (IGND)  
IF (EDG.LT.EDGLIM * BINNOW) GOTO 190  
IEMAX = IEMAX + 1  
IEDGE (IEMAX) = IGND  
EDGE (IEMAX) = EDGTRY  
CHI (IEMAX) = CHITRY  
JEDGE (IEMAX) = J  
NOTGRC (IEMAX) = .FALSE.  
S2X (IEMAX) = S2 (J) / THRESH  
S3X (IEMAX) = S3 (J) / THRESH  
S4X (IEMAX) = S4 (J) / THRESH  
S5X (IEMAX) = S5 (J) / THRESH  
   99 IF (IEMAX.EQ.25) GOTO 395  
  100 LN = LL (J) * 3  
IF (LN.EQ.0) GOTO 190  
DO 110 K = 1, 2  
   L = LION (K, ISO)  
   IF (L.EQ.0.OR.L.GT.LN) GOTO 110  
   E3X = E3 (IX + L)  
   IF (E3X.LT.1.) GOTO 110  
   IXL = IX + L  
   CHITRY = E (J) - E3 (IXL)  
   IGND = IBN (CHITRY)  
   IF (IGND.GE.nbin) GOTO 110  
   IF (IGND.LT.1) IGND = 1  
!
   if (chitry / kt.gt.46.) recev (ignd) = recev (ignd) + qtkpr * &
    conce (j + 1) * zeta (k, iso) * ( (chitry / 13.6) **2)
   if (chitry / kt.gt.46.) goto 110  
!
   EDGTRY = QTK * CONCE (J + 1) * ZETA (K, ISO) * ( (CHITRY / &
    13.6) **2) * EXP (CHITRY / KT)
   EN = ECEN (IGND)  
   EDG = EDGTRY * EBK**IGND  
   BINNOW = BRMEV (IGND) + RECEV (IGND)  
   IF (EDG.LT.EDGLIM * BINNOW) GOTO 110  
   IEMAX = IEMAX + 1  
   IEDGE (IEMAX) = IGND  
   EDGE (IEMAX) = EDGTRY  
   CHI (IEMAX) = CHITRY  
   JEDGE (IEMAX) = J  
   NOTGRC (IEMAX) = .TRUE.  
   DONE (IEMAX) = .TRUE.  
   IF (IEMAX.EQ.25) GOTO 395  
  110 END DO  
  190 IX = IX + 3 * LL (J)  
J = J + 1  
ISO = ISO - 1  
GOTO 10  
  200 IF (CONCE (N + 1) .LT.1.E-6) GOTO 400  
IF (CONCE (N + 1) .LT.1.E-4.AND.N.NE.2) GOTO 400  
DO 250 K = 1, 2  
   Y = K  
   CHITRY = 13.6 * (Z / Y) **2  
   IGND = IBN (CHITRY)  
   IF (IGND.GE.nbin) GOTO 250  
   IF (IGND.LT.1) IGND = 1  
   EN = ECEN (IGND)  
   X = EN / CHITRY  
   GFTR = GFUNC (X, Y)  
   IF (X.GT.100.) GFTR = 7. / SQRT (X)  
   GMXTRY = 1.1 - 1. / (10. * Y**1.5)  
   IF (GFTR.GT.GMXTRY) GFTR = GMXTRY  
!
   if (chitry / kt.gt.46.) recev (ignd) = recev (ignd) + qhydpr * &
    conce (n + 1) * gftr / y**3
   if (chitry / kt.gt.46.) goto 250  
!
   EDGTRY = QHYD * CONCE (N + 1) * EXP (CHITRY / KT) / Y**3  
   EDG = EDGTRY * GFTR * EBK**IGND  
   BINNOW = BRMEV (IGND) + RECEV (IGND)  
   IF (EDG.LT.EDGLIM * BINNOW) GOTO 250  
   IEMAX = IEMAX + 1  
   IEDGE (IEMAX) = IGND  
   EDGE (IEMAX) = EDGTRY  
   CHI (IEMAX) = CHITRY  
   JEDGE (IEMAX) = J  
   ESMIN3 (IEMAX) = GMXTRY  
   ALFX (IEMAX) = Y  
   NOTGRC (IEMAX) = .TRUE.  
   DONE (IEMAX) = .FALSE.  
   IF (IEMAX.EQ.25) GOTO 395  
  250 END DO  
GOTO 400  
  395 PRINT 396, N  
  396 FORMAT(1X//' IEMAX OVERRUN ON ELEMENT N =',I3//)  
  400 IF (IPRINT.GT.0) PRINT 401, N, T, IEMAX  
  401 FORMAT(1X//' RECEMS FOR ELEMENT N =',I3,'  AT T =',1PE10.3, &
&  ',   WITH',I3,'  SIGNIF. EDGES'/)
IF (IEMAX.EQ.0) RETURN  
!  BUBBLE SORT TO PUT IEDGE IN INCREASING ORDER; PARALLEL SORT OF IMAP
DO 410 IE = 1, 25  
  410 IMAP (IE) = IE  
  415 SWITCH = .FALSE.  
IEDGE (IEMAX + 1) = 0  
IF (IEMAX.LT.25) IEDGE (IEMAX + 2) = 0  
IF (IEMAX.EQ.1) GOTO 430  
IE = 1  
  420 IESAVE = IEDGE (IE)  
IF (IEDGE (IE+1) .GE.IESAVE) GOTO 425  
IMSAVE = IMAP (IE)  
IEDGE (IE) = IEDGE (IE+1)  
IMAP (IE) = IMAP (IE+1)  
IEDGE (IE+1) = IESAVE  
IMAP (IE+1) = IMSAVE  
IF (IE.GT.1) SWITCH = .TRUE.  
  425 IE = IE+1  
IF (IE.LT.IEMAX) GOTO 420  
IF (SWITCH) GOTO 415  
!  END SORT, PRINT TABLE OF EDGES SIGNIFICANT ENOUGH TO BE INCLUDED
  430 IF (JPRINT.EQ.0) GOTO 500  
PRINT 441  
  441 FORMAT(2(11X,'J     CHI',8X,'EDGE',7X,'N GRC   SHRT')/)  
IEP = IEMAX / 2 + 1  
DO 450 IE = 1, IEP  
   JB = IE * 2  
   IB = JB - 1  
   JP = IMAP (JB)  
   IP = IMAP (IB)  
   IF (IEDGE (IB) .EQ.0) GOTO 450  
   IF (IEDGE (JB) .EQ.0) GOTO 444  
   PRINT 445, JEDGE (IP), CHI (IP), EDGE (IP), NOTGRC (IP), &
    DONE (IP), JEDGE (JP), CHI (JP), EDGE (JP), NOTGRC (JP), &
    DONE (JP)
   GOTO 450  
  444    PRINT 445, JEDGE (IP), CHI (IP), EDGE (IP), NOTGRC (IP), &
    DONE (IP)
  445 FORMAT(2(10X,I2,0PF10.1,1PE12.2,2L8))  
  450 END DO  
!  BEGIN BINNING, USING LIST IEDGE TO GOVERN NUMBER OF EDGES INCLUDED
  500 IMIN = IEDGE (1)  
IEMAX = 1  
FIMIN = IMIN - 1  
EXPROD = EBK**FIMIN  
DO 550 I = IMIN, nbin  
  510    IF (I.NE.IEDGE (IEMAX + 1) ) GOTO 520  
   IEMAX = IEMAX + 1  
   GOTO 510  
  520    EN = ECEN (I)  
   BINSUM = 0.  
   DO 540 IE = 1, IEMAX  
      IPNT = IMAP (IE)  
      BINN = EDGE (IPNT)  
      IF (NOTGRC (IPNT) ) GOTO 530  
      X = CHI (IPNT) / EN  
      BINN = BINN * (S2X (IPNT) / X + S3X (IPNT) + S4X (IPNT) &
       * X + S5X (IPNT) * X * X)
      GOTO 540  
  530       IF (DONE (IPNT) ) GOTO 540  
      Y = ALFX (IPNT)  
      X = EN / CHI (IPNT)  
      GFTR = 7. / SQRT (X)  
      IF (X.LT.100.) GFTR = GFUNC (X, Y)  
      GMAX = ESMIN3 (IPNT)  
      IF (GFTR.GT.GMAX) GFTR = GMAX  
      BINN = BINN * GFTR  
  540    BINSUM = BINSUM + BINN  
   EXPROD = EXPROD * EBK  
   RECEV (I) = RECEV (I) + BINSUM * EXPROD  
  550 END DO  
RETURN  

END SUBROUTINE RECEMS
FUNCTION FBG (U, GAM)  
!  39th values of A1, A2, and A3 revised to get rid of 5% bump
!  in Kellog et al fit to Karzas and Latter at gamma2 = 0.1,
!  u = .3-1. : uses numbers from Larry Molnar, Jan 1988
!  and corrections from Jack Hughes
!
REAL (8) :: T, AI, AK, U4  
DIMENSION A (6, 7, 3), A1 (126), GAM2 (6), GAM3 (6)  
EQUIVALENCE (A, A1)  
DATA GAM2 / .7783, 1.2217, 2.6234, 4.3766, 20., 70. /  
DATA GAM3 / 1., 1.7783, 3., 5.6234, 10., 30. /  
DATA (A1 (I), I = 1, 42) / 1.001, 1.004, 1.017, 1.036, 1.056, &
 1.121, 1.001, 1.005, 1.017, 1.046, 1.073, 1.115, .9991, 1.005, &
 1.030, 1.055, 1.102, 1.176, .9970, 1.005, 1.035, 1.069, 1.134, &
 1.186, .9962, 1.004, 1.042, 1.100, 1.193, 1.306, .9874, .9962, &
 1.047, 1.156, 1.327, 1.485, .9681, .9755, 1.02009, 1.208, 1.525, &
 1.965 /
DATA (A1 (I), I = 43, 84) / .30290, .16160, .04757, .01300, &
 .00490, - .00320, .49050, .21550, .08357, .02041, .00739, .00029, &
 .65400, .28330, .08057, .03257, .00759, - .00151, 1.0290, .39100, &
 .12660, .05149, .01274, .00324, .95690, .48910, .17640, .05914, &
 .01407, - .00024, 1.2360, .75790, .32600, .10770, .02800, .00548, &
 1.3270, 1.0170, 0.60166, .20500, .06050, .00187 /

DATA (A1 (I), I = 85, 126) / - 1.3230, - .25400, - .01571, &
 - .001000, - .000184, .00008, - 4.7620, - .33860, - .03571, &
 - .001786, - .000300, .00001, - 6.3490, - .42060, - .02571, &
 - .003429, - .000234, .00005, - 13.231, - .59000, - .04571, &
 - .005714, - .000445, - .00004, - 7.6720, - .68520, - .06430, &
 - .005857, - .000420, .00004, - 7.1430, - .99470, - .12000, &
 - .010070, - .000851, - .00004, - 3.1750, - 1.1160, - .22695, &
 - .018210, - .001729, .00023 /
GAM1 = GAM * 1000.  
IF (GAM1.GT.100.) GOTO 602  
U2 = U**2  
!
!*****COMPUTE BORN APPROXIMATION GAUNT_RS FACTOR
!
U1 = U / 2.  
T = U1 / 3.75  
U4 = U1 / 2.  
IF (U1.GT.2.) GOTO 299  
AI = 1.0 + 3.5156229 * T**2 + 3.0899424 * T**4 + 1.2067492 * T**6 &
 + 0.2659732 * T**8 + 0.0360768 * T**10 + 0.0045813 * T**12
AK = - 1. * LOG (U4) * AI - .57721566 + .42278420 * U4**2 + &
 .23069758 * U4**4 + .0348859 * U4**6 + .00262698 * U4**8 + &
 .00010750 * U4**10 + .0000074 * U4**12
GOTO 297  
!
  299 AK = 1.2533141 - .07832358 / U4 + .02189568 / U4**2 - .01062446 / &
 U4**3 + .00587872 / U4**4 - .00251540 / U4**5 + .00053208 / U4**6
AK = AK / (EXP (U1) * SQRT (U1) )  
  297 BORN = .5513 * EXP (U1) * AK  
!
!*****COMPUTE POLYMONIAL FACTOR TO MULTIPLY BORN APPROXIMATION
!
IF (GAM1.LT.1.) GOTO 401  
IF (U.LT..003) goto 401  
IF (U.LE..03) N = 1  
IF ( (U.LE..3) .AND. (U.GT..03) ) N = 2  
IF ( (U.LE.1.) .AND. (U.GT..3) ) N = 3  
IF ( (U.LE.5.) .AND. (U.GT.1.) ) N = 4  
IF ( (U.LE.15.) .AND. (U.GT.5.) ) N = 5  
IF (U.GT.15.) N = 6  
IF (GAM1.LE.1.7783) M = 1  
IF ( (GAM1.LE.3.) .AND. (GAM1.GT.1.7783) ) M = 2  
IF ( (GAM1.LE.5.6234) .AND. (GAM1.GT.3.) ) M = 3  
IF ( (GAM1.LE.10.) .AND. (GAM1.GT.5.6234) ) M = 4  
IF ( (GAM1.LE.30.) .AND. (GAM1.GT.10.) ) M = 5  
IF ( (GAM1.LE.100.) .AND. (GAM1.GT.30.) ) M = 6  
M1 = M + 1  
G1 = (A (N, M, 1) + A (N, M, 2) * U + A (N, M, 3) * U2) * BORN  
G2 = (A (N, M1, 1) + A (N, M1, 2) * U + A (N, M1, 3) * U2) &
 * BORN
P = (GAM1 - GAM3 (M) ) / GAM2 (M)  
FBG = (1.0 - P) * G1 + P * G2  
RETURN  
  602 POWER = - .134 / (GAM**.2097)  
FBG = 1.5 * (3. * U) **POWER  
RETURN  
  401 FBG = BORN  
RETURN  

END FUNCTION FBG
SUBROUTINE BREMS (T, N)  
REAL :: KT  
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / CONTIN / BRMEV (1000), RECEV (1000), TUF (1000)  
DIMENSION X (30)  
HC = 12399.  
J2 = N + 1  
J1 = 2  
T6 = T / 1.E6  
KT = 86.17 * T6  
EBK = EXP (BINSYZ / KT)  
Q = 10.** (ABUND-12.) * 86.17 * SQRT (T6) * 20.4 * (EBK - 1.) &
 * EXP ( - BINMIN / KT) / HC
EBK = 1. / EBK  
DO 10 J = J1, J2  
   X (J) = 0.  
   IF (CONCE (J) .GT.1.E-5) X (J) = CONCE (J) * Q * (J - 1) **2  
   10 END DO  
I2 = MIN0 (NBIN, INT (46. * KT / BINSYZ) + 1)  
EXPROD = 1.  
DO 40 I = 1, I2  
   EN2 = BINSYZ * I + BINMIN  
   EN1 = EN2 - BINSYZ  
   Y = (EN2 + EN1) * .5 / KT  
   CHRGSM = 0.  
   DO 30 J = J1, J2  
      IF (X (J) .EQ.0.) GOTO 30  
      Z = (J - 1) **2 * .158 / T6  
      CHRGSM = CHRGSM + X (J) * FBG (Y, Z)  
   30    END DO  
   EXPROD = EXPROD * EBK  
   40 BRMEV (I) = BRMEV (I) + CHRGSM * EXPROD  
RETURN  

END SUBROUTINE BREMS
SUBROUTINE TWOPH (WAV2, P2PH, III)  
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / CONTIN / BRMEV (1000), RECEV (1000), TUFEV (1000)  
IF (P2PH.LE.1.E-10) RETURN  
ITMAX = (12399. / WAV2 - BINMIN) / BINSYZ  
IMAX = MIN0 (ITMAX, NBIN)  
IF (IMAX.LE.1) RETURN  
DO 15 IK = 1, IMAX  
   R = (BINMIN + BINSYZ * (IK - .5) ) * WAV2 / 12399.  
   15 TUFEV (IK) = TUFEV (IK) + 12. * P2PH * R * R * (1. - R) * WAV2 * &
 BINSYZ / 12399.
RETURN  
END SUBROUTINE TWOPH


FUNCTION GAUNT_RS (T, E, N, J, L, DENE)  
!     MAY '82 VERSION RELYING HEAVILY ON MANN AND ROBB CALCULATIONS,
!     BHATIA FOR DELTA N = 0 AND BHATIA AND MASON DELTA N = 1
!  DEc 17, 1991 : Modify to improve trans 1 of Li-like ions
DIMENSION AH (5), BH (5), GHE (24), GBE (8, 5)  
DIMENSION GBE1 (4), GBE2 (4), GBE3 (11), GBE4 (11), GBE5 (11), &
 BESPEC (4, 3)
DIMENSION GB (8, 6), GC (8, 9), GN (8, 4), GOX (8, 8), GF (8, 9), &
 GNE (5, 12)
DIMENSION GNA (8, 2), GNA4 (3, 3), GMGII (12)  
DIMENSION GMG (10), GMGP (8, 2)  
DIMENSION GAL (10), GAL2 (10), A (10), B (10), C (10)  
DIMENSION GMSHELL (6, 4), GK (10), GCA (10)  

dimension a1 (12), b1 (12), c1 (12)  
COMMON / INTERC / REDUC (4)  

COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
DATA AH / 0., .08, .08, .08, 0. /  
DATA BH / .22, .16, .19, .20, 0. /  
DATA GHE / .03785, .0002038, .04214, - .001901, - .03263, .004462, &
 .28, 0., .02327, - .0007424, - .06369, .001865, .07711, .000058, &
 0., 0., - .03168, .0008439, .182, - .007289, - .05873, .009605, &
 0., 0. /
DATA GBE / .7114, .00558, - .9277, - .00153, .2711, .00523, 0.0, &
 0.0, 0.0, 0.0, - .044, .00735, .482, - .01855, 0.0, 0.0, .44, 0.0, &
 - .010, 5 * 0.0, .077, .00666, 6 * 0., - .2943, - .00084, .2574, &
 .01167, .04852, - .00777, .2671, .00555 /
DATA GBE1 / - .0264, 0.03, 0.0, 0.20 /  
DATA GBE2 / .0881, 0.10, .117, 0.10 /  
DATA GBE3 / 0.0, .064, .069, .048, .035, .0303, .025, .02, .02, &
 .032, .032 /
DATA GBE4 / 0.0, .342, .39, .192, .103, .0836, .063, .043, .043, &
 0.0, 0.0 /
DATA GBE5 / 0.0, 3 * 0.0, 55000., 64000., 86000., 97000., 108000., &
 156000., 168000. /
DATA BESPEC / .00069, .122, .0059, .0063, .0016, .13, .0073, &
 .0058, .0017, .127, .0087, .0040 /
DATA GB / .2392, .00543, .3723, .0335, .0321, - .0253, .4591, &
 - .00232, .2317, .00654, - .1479, .03186, .3038, - .02130, .3904, &
 - .00302, .0811, .01318, - .2523, .04591, .2122, - .02192, .2304, &
 .00639, - .112, .0063, .050, .011, .0808, - .005, .239, .0023, &
 .126, 0.0, .287, 0.0, 0.0, 0.0, .28, 0.0, .128, - .025, - .46, &
 .133, .138, - .023, - .70, .128 /
DATA GC / .3539, .00039, .2314, .02314, .0358, - .01534, .5449, &
 - .00858, .2005, .00395, - .3012, .04332, - .09599, .01606, .3727, &
 - .001517, .3229, - .002883, .0212, .02481, .0783, - .01326, &
 .4671, - .008106, - .112, .0063, .050, .011, .0808, - .005, .239, &
 .0023, .128, - .025, - .46, .133, .138, - .023, - .70, .128, &
 - .269, - .00117, .0318, .0233, - .0102, - .0075, .235, .0172, &
 .22, .0056, 6 * 0., .198, .0069, 6 * 0., 8 * 0. /
DATA GN / - .01876, .0011, .0083, .00182, .0135, - .00083, .040, &
 .00040, .51, - .1, - 1.84, .53, .55, - .091, - 2.80, .51, .128, &
 - .025, - .46, .133, .138, - .023, - .70, .128, - .07504, .0042, &
 .033, .00726, .054, - .0033, .159, .0015 /
!
DATA GOX / .69, 0.0, 4 * 0.0, .46, 0.0, - .112, .0063, .050, .011, &
 .0808, - .005, .2391, .0023, - .112, .0063, .050, .011, .0808, &
 - .005, .2391, .0023, .128, - .025, - .46, .133, .138, - .023, &
 - .70, .128, .51, - .1, - 1.84, .53, .55, - .091, - 2.80, .51, &
 .22, .0056, 6 * 0.0, 8 * 0., .1261, 0.0, .2869, 0.0, 0.0, 0.0, &
 .28, 0.0 /
DATA GF / .2246, .0075, .3055, .0062, - .0575, - .0017, .4248, &
 - .0049, - .063, 0., .407, 0., - .0238, 0., .478, 0., - .0157, 0., &
 .188, 0., .0195, 0., .283, 0., - .001, 0., .134, 0., .123, 0., &
 .158, 0., .51, - .1, - 1.84, .53, .55, - .091, - 2.8, .51, .039, &
 .0154, 6 * 0., 8 * 0., .22, .056, 6 * 0., 8 * 0. /
DATA GNE / - .72117, 1.2406, 11.746, 8.2169, - 7.7772, 1.0227, &
 .70828, 4.5400, 4.1450, - 3.8657, - 1.04290, .84411, 6.7031, &
 3.1927, - 3.1693, - .039582, .26837, .25803, .086929, - .086641, &
 - .017323, .29514, .29301, .13223, - .13071, - .071593, .15504, &
 .12598, - .000449, .000523, .040202, .25113, .14858, .030780, &
 - .030745, .45761, .38477, .52142, .92153, - .91649, - .17862, &
 .32249, .28172, .040677, - .040639, - .062670, .14921, 1.5354, &
 1.0586, - 1.0219, - .057871, .030701, .36471, .14784, - .14293, &
 .093106, - .001108, - .067652, - .021663, .021657 /
!
DATA GNA / .1586, .007375, .1866, .04156, .02403, - .02416, .3100, &
 - .00098, - .3245, 0.0, .5548, 0.0, - .1562, 0.0, .266, 0. /
DATA GNA4 / 1.153, - .0333, 0.0, .001, .0053, .058, .67, - .0133, &
 0.0 /
DATA GMGII / .24, 37.3, 68.3, 96., 96., 2.39, 5.99, 5.99, 0., 0., &
 0., .142 /
DATA GMG / .9, .2, .25, .23, .23, .2, .2, .2, .14, .0094 /  
DATA GMGP / - .1333, .0155, - .6758, .0577, .5057, - .0323, .314, &
 0.0, - .294, 0.0, .5043, 0.0, - .1619, 0.0, .2606, 0.0 /
DATA GAL / 0.0139, .01, .06, 0., 0.2, 3 * 0., 1.64, .11 /  
DATA GAL2 / 0.0, .28, .28, 0., 0., 3 * 0., .00, .28 /  
DATA A / .022, .625, .660, .240, .061, .256, .0368, .271, .833, &
 .696 /
DATA B / .0035, .360, .432, .019, .364, .354, .343, .794, .404, &
 .387 /
DATA C / .1261, .1261, .1261, .477, .162, .0108, .138, .0719, &
 .1261, .1261 /
DATA GMSHELL / .453, .91, .13, .93, .2, .2, .7, .98, .13, .93, 0., &
 0., .59, .95, .13, .93, 0., 0., .51, .95, .20, .29, 0., 0. /
DATA GK / .35, .35, 1.1, .092, .91, .97, 1.1, 3 * 0. /  

DATA GCA / .35, .21, 43., .14, .12, .43, 37., .20, .11, 4. /  
data a1 / 0., .4508, .3957, .5584, .5055, .5055, .5537, .5537, &
 .5108, .5108, 1.235, 1.235 /
data b1 / 0., - .02787, .4532, .1363, .3461, .3461, .3468, .3468, &
 .3304, .3304, - .6625, - .6625 /


data c1 / 0., .1570, - .1219, .1158, - .01872, - .01872, .00516, &
 .00516, .1007, .1007, .8310, .8310 /
!  SHOULD PUT IN SENSIBLE F'S FOR POTASSIUM ISOSEQUENCE
T4 = T / 10000.  
ST = SQRT (T)  
GAUNT_RS = 0.2  
Y = E * 11590. / T  
CC = EXINT1 (Y, 2)  
IF (J.EQ.1.AND.N.GT.2) GOTO 150  
!
II = N - J + 1  
IF (II.GE.18) GOTO 50  
GOTO (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 14), &
 II
!
    1 CONTINUE  
!  HYDROGENIC IONS: MEWE; HAYES AND SEATON: 3S,3D FROM THOMAS; OTHERS ME
GAUNT_RS = AH (L) + BH (L) * Y * CC + .276 * CC  
IF (L.EQ.5) GAUNT_RS = .212 - .203 * (Y - Y * Y * CC) + .0256 * CC  
IF (N.GT.2) RETURN  
IF (L.EQ.5) GAUNT_RS = .1198 * Y * CC - .1194 * (Y - Y * Y * CC) &
 + .0328 * CC
GOTO 200  
    2 CONTINUE  
!  HELIUM-LIKE IONS:  PRADHAN+CASCADES FOR N=2, MEWE N=3,4
!  NEUTRAL HE FROM Berrington et al J Phys B 18,4135; Aggarwal Ap J 278,
!  L = 9 HAS GAUNT_RS = 0.0,  SATELLITE LINES : L = 6 DONE BY BRANCH FROM L
IF (N.EQ.2) GOTO 25  
IF (L.GT.3) GOTO 21  
GAUNT_RS = GNCRCH (GHE, L, Y, CC, N)  
RETURN  
   21 IF (L.EQ.4) GAUNT_RS = (1. + 7. / N) * (.053 + .022 * (Y * Y * Y * &
 CC - Y * Y + Y) + .276 * CC)
IF (L.EQ.5) GAUNT_RS = (1 + 1.5 / N) * (.053 + .022 * (Y * Y * Y * &
 CC - Y * Y + Y) + .276 * CC)
IF (L.EQ.6) GAUNT_RS = 0.  
IF (L.EQ.7) GAUNT_RS = .04 * (Y - Y * Y * CC)  
IF (L.EQ.8) GAUNT_RS = .02  
IF (L.EQ.9) GAUNT_RS = 0.  
RETURN  
   25 IF (L.EQ.1) GAUNT_RS = AMIN1 (.00656 * T4** (.89), .30 * CC)  
IF (L.EQ.4.OR.L.EQ.5) GAUNT_RS = .30 * CC  
IF (L.EQ.2) GAUNT_RS = AMIN1 (.0283 * T4** ( - .10), .359 * T4** ( - &
 .569) )
IF (L.EQ.3) GAUNT_RS = AMIN1 (.0105 * T4** (.30), .095 * T4** ( - &
 .479) )
IF (L.EQ.6) GAUNT_RS = 0.0  
IF (L.EQ.7) GAUNT_RS = 4.08E-7 * T**.941  
IF (L.EQ.8) GAUNT_RS = 2.46E-8 * T**1.22  
IF (L.EQ.9) GAUNT_RS = 0.0  
GOTO 200  
    3 CONTINUE  
!  LITHIUM-LIKE IONS: MEWE MODIFIED TO FIT CLOSE-COUPLING CALCULATIONS O
!  L=1 modified Dec 17, 1991
i = n - 4  
if (n.ge.10) i = n / 2  
if (n.ge.26) i = n / 2 - 2  
if (l.eq.1) gaunt = a1 (i) + (b1 (i) * y - c1 (i) * y * y + 0.279) &
 * cc + c1 (i) * y
X = 1. / (N - 3.)  
!     IF (L .EQ. 1) GAUNT_RS = (0.68 + .02*N) * ((.7+.35*X) +
!    1  ((1.-.8*X)*Y+(.5-.5*X)*Y*Y +.28)*CC - (.5-.5*X)*Y)
IF (L.EQ.2) GAUNT_RS = .053 + .16 * CC  
IF (L.EQ.3.OR.L.EQ.4) GAUNT_RS = - (.16 + .32 * X) + ( (.8 - .56 * X) &
 * Y + .2 * Y * Y + .28) * CC - .2 * Y
IF (L.EQ.5) GAUNT_RS = (.19 + .25 * X) + .079 * CC  
IF (L.EQ.6) GAUNT_RS = .31 - .1 * Y * CC  
IF (L.EQ.7) GAUNT_RS = .096 + .32 * X  
IF (L.EQ.8) GAUNT_RS = .13  
GOTO 200  
    4 CONTINUE  
!  BERYLLIUM-LIKE IONS:  QUB FOR N=2 UP THROUGH NE; MANN, ROBB & SAMPSON
!  MANN AND MALINOVSKY FOR N = 3 AND QUB O V N=3 FROM WIDING
!  N = 4 FROM JOHNSTON & KUNZE, MASON & STOREY (FE XXII) AND LI-LIKE
I = N - 5  
IF (N.GE.10) I = N / 2 - 1  
IF (N.GE.26) I = N / 2 - 3  
IF (L.NE.1) GOTO 41  
TL = ALOG10 (T)  
GAUNT_RS = .54 + .0125 * N + .135 * CC  
IF (N.LE.10) GAUNT_RS = GBE1 (I) + GBE2 (I) * TL  
GOTO 200  
   41 IF (L.NE.2) GOTO 42  
GNT = GBE3 (I) / (1. + GBE4 (I) / Y) + GBE5 (I) / T  
IF (N.EQ.6) GNT = .046 / (1. + T * T / 6.25E10)  
GAUNT_RS = GNT * REDUC (1)  
GOTO 200  
   42 CONTINUE  
IF (L.LE.7) GAUNT_RS = GNCRCH (GBE, L - 2, Y, CC, N)  
IF (L.EQ.8) GAUNT_RS = .1261 + .2869 * Y * CC + .276 * CC  
IF (N.GT.8) GOTO 200  
EM = - 4.67 + 1.86 * N  
EX = EXP ( - EM * 11590. / T)  
M = N - 5  
IF (L.EQ.10) GAUNT_RS = BESPEC (1, M) * (1. - PM (1) ) * EX + BESPEC &
 (2, M) * PM (1)
IF (L.EQ.11) GAUNT_RS = BESPEC (3, M) * (1. - PM (1) ) * EX + BESPEC &
 (4, M) * PM (1)
GOTO 200  
    5 CONTINUE  
!  BORON ISOSEQUENCE: N=2 INTERPOLATED FROM O IV AND FE XXII OF ROBB; GO
!  NA-S, DERE ET AL AR, MANN C II;  OVEREST BY 36% NEAR THRESHOLD WRT RO
!  N = 3 INTERPOLATED C II AND FE XXII FROM MANN; 2S-3L SCALED FROM BE-,
!  TO 8% FOR 2S-3P.  N = 4 FROM MASON & STOREY
!  L = 9 INTERCOMBINATION LINE
IF (L.LE.6) GAUNT_RS = GNCRCH (GB, L, Y, CC, N)  
IF (L.EQ.9) GAUNT_RS = REDUC (2)  
IF (L.EQ.12) GAUNT_RS = 0.  
GOTO 200  
    6 CONTINUE  
!  CARBON ISOSEQUENCE: N=2 INTERP FROM MANN AND ROBB;  AGREES TO 5-10% W
!  N=3,4 GENERAL B-F; AGREES 3-10% WITH MANN 2P-3D; L=6 INTERCOMB
IF (L.LE.5) GAUNT_RS = GNCRCH (GC, L, Y, CC, N)  
IF (L.EQ.7) GAUNT_RS = .198 + .0069 * N  
IF (L.EQ.8) GAUNT_RS = .22 + .0056 * N  
IF (L.EQ.9) GAUNT_RS = .1261 + (.2869 * Y + .28) * CC  
IF (L.EQ.6) GAUNT_RS = REDUC (3)  
IF (L.EQ.6.AND.N.EQ.6) GAUNT_RS = SQRT (.0001 * T) * REDUC (3)  
IF (L.EQ.12) GAUNT_RS = 0.  
GOTO 200  
    7 CONTINUE  
!  NITROGEN SEQUENCE:   N=2 FROM MANN FE XX AND MASON & BHATIA MG,SI,S,A
!  N=3 AND 4 GENERAL B-F
IF (L.EQ.1) GAUNT_RS = (1.086 - 1.71 / J) * (.3642 + .9358 * Y * CC - &
 .3758 * (Y - Y * Y * CC) + .3586 * CC)
IF (L.GE.2.AND.L.LE.5) GAUNT_RS = GNCRCH (GN, L - 1, Y, CC, N)  
IF (L.EQ.12) GAUNT_RS = 0.  
IF (L.EQ.8) GAUNT_RS = .22 + .0056 * N  
IF (L.EQ.9) GAUNT_RS = .1261 + .2869 * Y * CC + .276 * CC  
GOTO 200  
    8 CONTINUE  
!  OXYGEN ISOSEQUENCE:  N=2 FROM MANN, ROBB FE XIX AND FROM BHATIA,F&D S
!  OTHERS GENERIC B-F
IF (L.LE.8) GAUNT_RS = GNCRCH (GOX, L, Y, CC, N)  
IF (L.EQ.10) GAUNT_RS = .16 + .0015 * N  
IF (L.EQ.15) GAUNT_RS = 0.  
GOTO 200  
    9 CONTINUE  
!  FLUORINE SEQUENCE:   N=2 FROM MANN AL V AND ROBB FE XVIII;   OTHERS G
IF (L.LE.8) GAUNT_RS = GNCRCH (GF, L, Y, CC, N)  
IF (L.EQ.9) GAUNT_RS = .1261 + (.2869 * Y + .28) * CC  
GOTO 200  
   10 CONTINUE  
!     NEON SEQUENCE: SMITH ET AL INCLUDING CASCADES AND RESONANCES
!     ASSUME THEY ARE ALL LIKE FE XVII
IF (L.LE.12) GAUNT_RS = GNE (1, L) + (GNE (2, L) + GNE (3, L) &
 * Y + GNE (4, L) * Y * Y) * EXINT1 (Y, 2) + GNE (5, L) * Y
GOTO 200  
   11 CONTINUE  
!     SODIUM SEQUENCE:  MANN, FLOWER & NUSSBAUMER, AND BLAHA
IF (N.NE.12) GOTO 111  
GAUNT_RS = GMGII (L)  
IF (L.EQ.1) GAUNT_RS = .112 + (.0269 * Y - .0998 * Y * Y + .318) &
 * CC + .0998 * Y
IF (L.EQ.2) GAUNT_RS = 141 + (59.3 * Y - 671 * Y * Y + .858) * CC + &
 671 * Y
RETURN  
  111 CONTINUE  
IF (L.GT.2) GOTO 112  
GAUNT_RS = GNCRCH (GNA, L, Y, CC, N - 10)  
IF (N.EQ.14.AND.L.EQ.2) GAUNT_RS = - .0172 + (.832 * Y + .029 * Y * &
 Y + .3513) * CC - .029 * Y
RETURN  
  112 CONTINUE  
IF (L.GT.5) GOTO 113  
LL = L - 2  
GAUNT_RS = GNA4 (1, LL) + GNA4 (2, LL) * N + GNA4 (3, LL) * CC  
RETURN  
  113 IF (L.EQ.6) GAUNT_RS = - .16 + .8 * Y * CC - .2 * (Y - Y * Y * CC) &
 + .276 * CC
IF (L.EQ.7) GAUNT_RS = .44 - 0.1 * Y * CC  
IF (L.EQ.8) GAUNT_RS = .15 - .05 * Y * CC  
IF (L.EQ.9.OR.L.EQ.10) GAUNT_RS = 0.03  
IF (L.EQ.11) GAUNT_RS = 0.07  
IF (L.EQ.12) GAUNT_RS = 0.15  
RETURN  
   12 CONTINUE  
!     MAGNESIUM SEQUENCE:   INTERPOLATE SI TO FE FOR 3P, 4P EXCITATIONS;
!     USE MANN FE GAUNT_RS FACTORS FOR 3D, 4S, 4F, AND INTERCOMB.  ASSUME
!     G FOR 4D = G FOR 4F;  L=10 IS INTERCOMBINATION LINE
GAUNT_RS = GMG (L)  
IF (L.LE.2) GAUNT_RS = GNCRCH (GMGP, L, Y, CC, N)  
IF (L.EQ.10) GAUNT_RS = REDUC (4)  
IF (L.EQ.10.AND.N.EQ.14) GAUNT_RS = REDUC (4) * (20000. / T) ** ( &
 .406)
RETURN  
   13 CONTINUE  
!     ALUMINUM ISO-SEQUENCE:   SI II FROM ROBERTS, S IV 3P FROM MANN, 3D
!     FE XIV FROM BLAHA USED FOR NI AND FOR ALL N=4
!     RESONANCES INCLUDED FOR METASTABLE USING BHATIA COLLISION STRENGTH
!     SI II 1814 LINES FROM BROWN, FERRAZ AND JORDAN
!     FE XIV RESONANCES COMPUTED AS IN SMITH ET AL FROM BLAHA OMEGAS
IF (L.GE.4.AND.L.LE.8) GOTO 132  
IF (N.GT.14) GOTO 131  
GAUNT_RS = GAL (L) + GAL2 (L) * CC  
IF (L.EQ.1) GAUNT_RS = GAL (1) * 1.9E7 * ST / (DENE+ST * 1.9E7)  
RETURN  
  131 IF (N.GT.20) GOTO 132  
CN = ST * 2.7 * 10.** (7. + J / 2.)  
IF (L.EQ.1) GAUNT_RS = ( (.128 + (.3449 * Y + .3544 * Y * Y) * CC - &
 .3544 * Y) * 1.8 / (1. + .25 / Y) ) * .069 * CN / (DENE+CN)
IF (L.EQ.2) GAUNT_RS = .0405 + (.2052 * Y + .0328 * Y * Y + .2311) &
 * CC - .0328 * Y
IF (L.EQ.3) GAUNT_RS = .142 + .147 * CC  
IF (L.EQ.9) GAUNT_RS = .1330 + (.3833 * Y + .0934 * Y * Y + .1611) &
 * CC - .0934 * Y
IF (L.EQ.10) GAUNT_RS = .1684 + (.2432 * Y + .0484 * Y * Y + .2638) &
 * CC - .0484 * Y
RETURN  
  132 CONTINUE  
GAUNT_RS = A (L) + B (L) * (1. - 1. / (1. + C (L) / Y) )  
IF (L.EQ.1) GAUNT_RS = (.0121 + .0036 / (1. + .1261 / Y) ) * 6.0E17 / &
 (DENE+6.0E17)
RETURN  
!
!  SILICON THROUGH CHLORINE SEQUENCES
   14 GAUNT_RS = GMSHELL (L, II - 13)  
IF (L.EQ.3.AND.II.NE.17) GAUNT_RS = GMSHELL (L, II - 13) - .114 / &
 (1. + .138 / Y)
RETURN  
   50 CONTINUE  
IF (II.GT.18) GOTO 19  
RETURN  
   19 IF (II.GT.19) GOTO 20  
GAUNT_RS = GK (L)  
IF (N.EQ.20) GAUNT_RS = GCA (L)  
RETURN  
   20 IF (L.LE.3) GAUNT_RS = AMAX1 (.2, - .606 * ALOG10 (Y) - .052)  
RETURN  
  150 CONTINUE  
GAUNT_RS = .01  
IF (Y.LE.10.) GAUNT_RS = AMIN1 (EXP ( - .7 * ALOG (Y) - 2.3), &
 .28 * CC)
  200 CONTINUE  
RETURN  

END FUNCTION GAUNT_RS
FUNCTION GNCRCH (R, L, Y, CC, N)  
DIMENSION R (1)  
IL = 8 * (L - 1)  
A = R (IL + 1) + R (IL + 2) * N  
B = R (IL + 3) + R (IL + 4) * N  
C = R (IL + 5) + R (IL + 6) * N  
D = R (IL + 7) + R (IL + 8) * N  
GNCRCH = A + B * Y * CC + C * (Y - Y * Y * CC) + D * CC  
RETURN  

END FUNCTION GNCRCH
FUNCTION DELT (N, J, L)  
!  OUR VERY OWN MODIFICATIONS TO JACOBS
!  DENSITY DEPENDENCE EXTERNAL
!  TAKE 0.5 FOR MERTS ET AL   DELTA N .NE. 0 EXCEPT S
!  ASSUME 3D OF N,O,F INTERP BETWEEN C AND NE ISOSEQUENCES
!  THIS USES YOUNGER'S CLAIM THAT DELT=1 FOR HE-LIKE RESONANCE LINE;
!  TRY BELY-DUBAU??????
DELT = 0.  
IF (J.EQ.1) RETURN  
I = N - J + 1  
IF (L.EQ.12.AND.I.NE.10) GOTO 20  
IF (L.NE.1.OR.I.LE.2) GOTO 50  
IF (I.EQ.13) GOTO 50  
   20 DELT = 1.  
GOTO 200  
   50 CONTINUE  
IF (I.GE.19) GOTO 19  
IF (I.EQ.18) GOTO 18  
IF (I.GE.15) GOTO 14  
GOTO (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), I  
    1 CONTINUE  
!   BURGESS & TWORKOWSKI ARE IN ALPHADI
IF (L.EQ.1) DELT = 1.  
IF (L.GT.1) DELT = 0.1  
IF (L.EQ.5) DELT = 0.  
GOTO 200  
    2 IF (L.EQ.4.OR.L.EQ.5) DELT = 0.1  
IF (L.EQ.9) DELT = 1.0  
GOTO 200  
    3 IF (L.GE.2.AND.L.LE.4) DELT = .5 * ( - .35 + 1.26 * N / (N + 11.) &
 )
GOTO 200  
    4 IF (L.EQ.7) DELT = AMAX1 (0.5 * ( - .55 + 1.50 * N / (N + 11.) ), &
 0.05)
GOTO 200  
    5 IF (L.LE.6) DELT = 1.  
IF (L.EQ.4.OR.L.EQ.5) DELT = 0.5 * ( - .54 + 2.2 * N / (N + 11.) )  
GOTO 200  
    6 IF (L.LE.5) DELT = 1.  
IF (L.EQ.4) DELT = AMAX1 (0.5 * ( - 1.28 + 3.2 * N / (N + 11) ), &
 0.05)
GOTO 200  
    7 IF (L.EQ.4) DELT = 1.  
IF (L.EQ.5) DELT = AMAX1 (0.5 * ( - 1.44 + 3.4 * N / (N + 11.) ), &
 0.05)
IF (L.EQ.7) DELT = AMAX1 (0.25 * ( - 1.44 + 3.4 * N / (N + 11.) ), &
 0.025)
GOTO 200  
    8 IF (L.LE.3) DELT = 0.5 * ( - 1.60 + 3.6 * N / (N + 11.) )  
IF (L.EQ.4.OR.L.EQ.5) DELT = 1.  
IF (L.EQ.7) DELT = .25 * ( - 1.6 + 3.6 * N / (N + 11.) )  
IF (L.EQ.15) DELT = 1.  
GOTO 200  
    9 IF (L.LE.3) DELT = 0.5 * ( - 1.75 + 3.8 * N / (N + 11.) )  
IF (L.EQ.5) DELT = 1.  
IF (L.EQ.6) DELT = 0.25 * ( - 1.75 + 3.8 * N / (N + 11.) )  
GOTO 200  
   10 IF (L.LE.2) DELT = 1.  
IF (L.GE.3.AND.L.LE.5) DELT = 0.5 * ( - 1.9 + 4.0 * N / (N + 11.) &
 )
IF (L.GE.6.AND.L.LT.8) DELT = .25 * ( - 1.9 + 4. * N / (N + 11.) )  
IF (L.EQ.14) DELT = 1.  
GOTO 200  
   11 CONTINUE  
IF (L.EQ.2.OR.L.EQ.3.OR.L.EQ.6) DELT = .25  
GOTO 200  
   12 IF (L.EQ.2) DELT = .25  
GOTO 200  
   13 IF (L.EQ.2.OR.L.GE.9) DELT = 1.  
IF (L.EQ.3.OR.L.EQ.5) DELT = .5  
IF (L.EQ.7) DELT = 0.25  
GOTO 200  
   14 IF (L.LE.2) DELT = 1.  
IF (L.EQ.3.OR.L.EQ.4) DELT = .25  
GOTO 200  
   18 DELT = 0.25  
IF (L.LE.3) DELT = 1.  
IF (N.NE.20) RETURN  
DELT = 0.  
IF (L.EQ.6) DELT = 0.05  
GOTO 200  
   19 DELT = 0.25  
  200 CONTINUE  
RETURN  
END FUNCTION DELT


FUNCTION PHOT (N, J, E, T, DENE)  
!  USES REILMAN & MANSON FOR INNER SUBSHELL PHOTOIONIZATION:  2S FOR B,C
!  SEQUENCES,; 3S FOR AL,SI,P SEQUENCES AND 3P FOR K,CA,SC SEQUENCES
!  ASSUMES THAT OUTER SHELL IONIZATION IS IN FOUR TERM POLYNOMIAL,
!  S2*X**2 + S3*X**3 + S4*X**4 + S5*X**5
COMMON / COM / CNC (12, 30), PTOT (12, 220), ABIN (1000), BIN ( &
 1000)
COMMON / HOT / HE, HEA (2)  
COMMON / HETOT / HEATING  
COMMON / DAT / EE (30), EA (30), S2 (30), WAVE (220), E3 (220), &
 F (220), LL (30), S3 (30), S4 (30), S5 (30)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / PT / RF (500), TAU (150), TAUHE (150), TPLUS (150)  
COMMON / RESULT / CONCE (30), GNDREC (30), POWER (220), RHY, &
 HENEUT, HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
DIMENSION INNER (30), AINNER (12, 9), BINNER (12, 9), EINNER (12, &
 9)
DATA INNER / 4 * 0, 1, 2, 3, 5 * 0, 4, 5, 6, 3 * 0, 7, 8, 9, 9 * &
 0 /
DATA EINNER / 0., 30.9, 55.8, 87.6, 172., 283., 422., 589., 784., &
 1006., 1842., 2178., 0., 15.6, 36.7, 63.8, 139., 241., 371., 528., &
 713., 925., 1731., 2056., 2 * 0., 20.3, 42.6, 108., 201., 321., &
 469., 644., 847., 1622., 1938., 6 * 0., 22.9, 57.6, 105.2, 165., &
 421., 531., 6 * 0., 13.5, 43.8, 87.6, 144., 388., 493., 7 * 0., &
 30.7, 70.4, 123., 356., 458., 9 * 0., 28., 213., 296., 9 * 0., &
 38., 190., 271., 10 * 0., 169., 246. /
DATA AINNER / 0., 2.357, 2.745, 1.443, .6024, .3391, .2058, .1385, &
 .09804, .0764, .04443, .0376, 0., 18.26, 6.950, 2.817, .9583, &
 .5001, .2672, .1685, .1184, .0913, .04837, .0407, 2 * 0., 27.80, &
 8.139, 1.630, .7415, .3621, .2394, .1478, .112, .05442, .0455, 6 * &
 0., 4.961, 2.695, 1.401, .8416, .2929, .2254, 6 * 0., 10.489, &
 5.507, 1.575, 1.409, .4378, .3263, 7 * 0., 12.06, 4.726, 2.351, &
 .6321, .4799, 9 * 0., 29.6, 2.452, 1.788, 9 * 0., 53.8, 3.265, &
 2.070, 10 * 0., 4.078, 2.351 /
DATA BINNER / 0., - .7352, .0201, .4321, .4094, .2605, .1935, &
 .1423, .111, .0865, .04050, .0343, 0., - 6.256, - .2887, 1.463, &
 1.029, .5726, .4080, .2885, .2132, .164, .07817, .0654, 2 * 0., &
 - 14.93, 1.194, 2.298, 1.106, .7338, .4742, .3510, .267, .1244, &
 .104, 6 * 0., - 4.607, - 1.863, - .6435, - .2594, - .0225, &
 - .0111, 6 * 0., - 10.412, - 4.613, - 1.118, - .4648, - .0221, &
 .0105, 7 * 0., - 11.09, - 3.357, - 1.014, - .0255, - .0078, 9 * &
 0., - 28.3, - .386, - .360, 9 * 0., - 45.8, - .610, - .178, 10 * &
 0., - .834, .0037 /
PHOT = 0.  
HE = 0.  
DO 995 NOJ = 1, 12  
   NO = NOJ  
   IF (N.LE.NJ (NOJ) ) GOTO 996  
  995 END DO  
  996 CONTINUE  
IF (J.NE.1) GOTO 5  
IF (N.EQ.12) PHOT = 1.0E-10  
IF (N.EQ.16) PHOT = 2.0E-10  
IF (N.EQ.26) PHOT = 3.0E-10  
IF (N.NE.6) GOTO 5  
!  PHOTOIONIZATION OF C I FROM 1D LEVEL
T4 = AMIN1 (.0001 * T, 3.)  
OM = 1.16 * T4 * (1.085 - .07507 * T4 - .02150 * T4 * T4)  
CN = SQRT (T) * 5. * 3.26E-4 / (8.63E-6 * OM)  
FD1 = 0.4 * EXP ( - 1.45 / T4) * DENE / (DENE+CN)  
ILY = (10.2 - BINMIN) / BINSYZ + 1  
PHOT = 2.5E-10  
IF (ILY.GE.1) PHOT = PHOT + FD1 * 1.03E-17 * 6.12E10 * ABIN (ILY)  
    5 CONTINUE  
IMIN = (E-BINMIN) / BINSYZ + 1  
IF (E.LT.BINMIN + (IMIN + 0.5) * BINSYZ) IMIN = IMIN + 1  
IF (IMIN.GE.NBIN) RETURN  
IMIN = MAX0 (IMIN, 1)  
IF (N.NE.1) GOTO 10  
SS2 = 0.00  
SS3 = 8.44  
SS4 = - 2.14  
SS5 = 0.00  
FACT = 1.6E-12 * 1.E23 / (1. + RHY)  
GOTO 15  
   10 CONTINUE  
SS2 = S2 (J)  
SS3 = S3 (J)  
SS4 = S4 (J)  
SS5 = S5 (J)  
FACT = 1.6E-12 * 1.E23 * 10.** (ABUND-12.) * CONCE (J)  
   15 CONTINUE  
ISO = N - J + 1  
IA = (EA (J) - BINMIN) / BINSYZ  
IA = MAX0 (IA, 1)  
IMAX = NBIN  
IF (ISO.GE.3) IMAX = MIN0 (NBIN, IA)  
IISO = INNER (ISO)  
IF (IISO.EQ.0) GOTO 991  
EIN = EINNER (NO, IISO)  
IIN = (EIN - BINMIN) / BINSYZ + 1  
IF (EIN.LT.BINMIN + (IIN + 0.5) * BINSYZ) IIN = IIN + 1  
IIN = MAX0 (IIN, 1)  
IMAX = MIN0 (IIN - 1, NBIN)  
  991 CONTINUE  
HNU = BINMIN + (IMIN - 0.5) * BINSYZ  
DO 20 I = IMIN, IMAX  
   HNU = HNU + BINSYZ  
   X = E / HNU  
   X2 = X * X  
   SG = SS2 * X2 + SS3 * X2 * X + SS4 * X2 * X2 + SS5 * X2 * X2 * &
    X
   PHOT1 = SG * (1.E-18 / 1.6E-12) * RF (I) * ABIN (I) / HNU  
   PHOT = PHOT + PHOT1  
   HE = HE+PHOT1 * HNU * FACT  
   20 END DO  
   30 CONTINUE  
IF (N.EQ.2) HEA (J) = HE  
!   INNER SUBSHELL
IF (IISO.EQ.0) goto 980  
IF (IIN.GE.NBIN) goto 980  
IMAX = MIN0 (IA, NBIN)  
SS2 = AINNER (NO, IISO)  
SS3 = BINNER (NO, IISO)  
DO 979 I = IIN, IMAX  
   HNU = BINMIN + BINSYZ * (I - 0.5)  
   X = EIN / HNU  
   SG = SS2 * X * X + SS3 * X * X * X  
   PHOT1 = SG * (1.E-18 / 1.6E-12) * RF (I) * ABIN (I) / HNU  
   PHOT = PHOT + PHOT1  
   HE = HE+PHOT1 * HNU * FACT  
  979 END DO  
  980 IF (N.GE.6) HEATING = HEATING + HE  
RETURN  

END FUNCTION PHOT

SUBROUTINE APHOT (N, DENE, ICONT)  
!  AUGER PHOTOIONIZATION : N=1 AND N=2 SHELLS
!  USES COX & DALTABUIT AND REILMAN & MANSON CROSS SECTIONS
COMMON / COM / CNC (12, 30), PTOT (12, 220), ABIN (1000), BIN ( &
 1000)
COMMON / HOT / HE, HEA (2)  
common / hetot / heating  
COMMON / DAT / E (30), EA (30), S2 (30), WAVE (220), E3 (220), &
 F (220), LL (30), S3 (30), S4 (30), S5 (30)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / PT / RF (500), TAU (150), TAUHE (150), TPLUS (150)  
COMMON / RESULT / CONCE (30), CA (30), POWER (220), RHY, HENEUT, &
 HEPLUS, DNE, PCOOL, POU, POT, RE, TU, PM (4)
COMMON / RATES / C1 (30), C2 (30), CTH  
DIMENSION A2P (20, 20), B2P (20, 20)  
DATA A2P / 20 * 0, 12.60, 20.58, 38 * 0., 4.520, 3.945, 5.546, &
 7.213, 36 * 0., 1.286, 1.316, 1.843, 2.556, 3.363, 3.654, 34 * 0., &
 .6109, .6256, .8755, 1.111, 1.254, 1.578, 1.919, 2.307, 32 * 0., &
 .4075, .4438, .4801, .5164, .6798, .8431, .9660, 1.089, 1.288, &
 1.488, 110 * 0., .1339, .1531, .1722, .1909, .2095, .2320, .2545, &
 .3022, .3499, .3628, .3756, .4630, .5504, .5820, .6136, .6679, 24 &
 * 0., .1145, .1294, .1409, .1525, .1644, .1763, .1797, .1831, &
 .2005, .2179, .2425, .2670, .3393, .4116, .4266, .4415, .5339, &
 .6263, 42 * 0. /
DATA B2P / 20 * 0., .1887, - 7.098, 38 * 0., 1.349, 5.109, 4.291, &
 6.750, 36 * 0., 2.408, 2.546, 3.116, 3.060, 2.981, 5.273, 34 * 0., &
 1.377, 1.456, 1.782, 1.930, 2.384, 2.572, 3.023, 3.451, 32 * 0., &
 .7788, .9714, 1.164, 1.356, 1.488, 1.620, 1.869, 2.118, 2.261, &
 2.405, 110 * 0., .3454, .3781, .4107, .4565, .5023, .5590, .6157, &
 .6283, .6409, .7392, .8375, .8258, .8142, .9381, 1.062, 1.207, 24 &
 * 0., .2541, .2787, .3087, .3387, .3770, .4152, .4907, .5661, &
 .6055, .6499, .6729, .7010, .6410, .5809, .6970, .8130, .7768, &
 .7406, 42 * 0. /
!
he = 0.  
DO 10 J = 1, 30  
   10 CA (J) = 0.  
RN = N  
NN = N - 2  
IF (N.LE.2) RETURN  
DO 1000 J = 1, NN  
   FACT = 1.6E-12 * 1.E23 * 10.** (ABUND-12.) * CONCE (J)  
   ISO = N - J + 1  
   RISO = ISO  
   PA1 = 0.  
   PA2 = 0.  
   EKS = 13.6 * RN * RN * RISO** (.24 - .43 / ALOG10 (RN) )  
   IK = (EKS - BINMIN) / BINSYZ  
   IMAX = NBIN  
   IF (ISO.GE.11) IMAX = MIN0 (IK, NBIN)  
   IMIN = (EA (J) - BINMIN) / BINSYZ + 1  
   IF (EA (J) .LT.BINMIN + (IMIN + 0.5) * BINSYZ) IMIN = IMIN + 1  
   IMIN = MAX0 (1, IMIN)  
   IF (IMIN.GE.NBIN) GOTO 1000  
!
   IF (ISO.LE.10) GOTO 500  
   A2 = A2P (ISO - 10, N - 10)  
   B2 = B2P (ISO - 10, N - 10)  
   DO 200 I = IMIN, IMAX  
      HNU = BINMIN + BINSYZ * (I - 0.5)  
      X = EA (J) / HNU  
      SG = A2 * X * X + B2 * X * X * X  
      PHOT1 = SG * (1.0E-18 / 1.6E-12) * RF (I) * ABIN (I) &
       / HNU
      PA1 = PA1 + PHOT1  
      HE = HE+PHOT1 * HNU * FACT  
  200    END DO  
  500    CONTINUE  
   IMIN = MAX0 (IK + 1, 1)  
   IF (IMIN.GE.NBIN) GOTO 610  
   SSS = 1.66 * EKS** (0.071)  
   DO 600 I = IMIN, NBIN  
      HNU = BINMIN + BINSYZ * (I - 0.5)  
      X = EKS / HNU  
      SG = (295. / EKS) * X**SSS  
      PHOT1 = SG * (1.0E-18 / 1.6E-12) * RF (I) * ABIN (I) &
       / HNU
      PA2 = PA2 + PHOT1  
      HE = HE+PHOT1 * HNU * FACT  
  600    END DO  
  610    CONTINUE  
   FLUOR = 0.5 * (RN / 26.) **4  
   CA (J) = (PA1 + PA2) * (1. - FLUOR) / DENE  
   C1 (J) = C1 (J) + (PA1 + PA2) * FLUOR / DENE  
   IF (ISO.NE.3) GOTO 700  
   C1 (J) = C1 (J) + PA2 / DENE  
   CA (J) = 0.  
  700    IF (ISO.NE.11) GOTO 710  
   CA (J) = (1. - FLUOR) * PA2 / DENE  
   C1 (J) = C1 (J) + (PA2 * FLUOR + PA1) / DENE  
  710    CONTINUE  
 1000 END DO  
 1001 CONTINUE  
!
heating = heating + he  
!
IF (ICONT.NE.0) RETURN  
!
!  AVRETT'S METHOD FOR INCLUDING AUTOIONIZATION IN EQUILIBRIUM
!
Q = 0.  
X = CA (1)  
NN = N - 1  
DO 1500 J = 1, NN  
   C1 (J) = C1 (J) + X  
   IF (C1 (J) .LE.0.) GOTO 1501  
   Q = C2 (J + 1) * CA (J) / C1 (J)  
   X = CA (J + 1) + Q  
 1500 END DO  
 1501 CONTINUE  
RETURN  

END SUBROUTINE APHOT
FUNCTION SECONT (B, Q)  
external gamma  
A = B / (Q * Q)  
R = Q**.6666667  
IF (A - 50.0) 20, 20, 10  
   10 SECONT = (1.0 - .1728 / R) - 0.0496 / (R * R)  
GOTO 30  
   20 CONTINUE  
SECONT = 1.0 + (.1728 / R) * ( (.33333 - 2. * A) * GAMMA (.33333, &
 A) + 1.0) - (.0496 / (R * R) ) * (.66667 * (1. - A - 3. * A * A) &
 * GAMMA (.66667, A) + (1. + 2. * A) )
   30 CONTINUE  
RETURN  

END FUNCTION SECONT
SUBROUTINE LNPRT (NUM)  
COMMON / HLN / EXT (11), HALF, HALFC, ALFLY, ALFLYC, HBET, HBETC, &
 TWOP
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / FLN / FWV (320), FTOT (320), FOUT (320), FCOOL  
COMMON / FEL / WJ (12, 220), FJ (12, 220), E3J (12, 220), PWR (12, &
 220), C1J (12, 30), C2J (12, 30), EJ (12, 30), SJ (12, 30), &
 CJ (12, 30), LLJ (12, 30), ALFJ (12, 30), ESJ (12, 30), SIGJ (12, &
 30)
COMMON / COM / CNC (12, 30), PTOT (12, 220), ABIN (1000), BIN ( &
 1000)
INTEGER :: BB  
COMMON / LNPR / JMX (11), NF (11), LDUM (5), JDUM (5), WDUM (5), &
 PDUM (5)
WRITE (7, 200) HBET, HBETC, TWOP  
  200 FORMAT ('    HBET, HBETC, TWOP    ',3E10.3)  
C = 100. / (HBET + HBETC)  
DO 1500 NO = 1, NUM  
   N = NJ (NO)  
   WRITE (7, 201) N  
  201 FORMAT   (' LINES FOR ELEMENT',I5)  
   LOC = 1  
   IX = 0  
   JT = AMIN0 (JMX (NO), N)  
   DO 1450 J = 1, JT  
      IY = LLJ (NO, J) * 3  
      IF (IY) 1450, 1450, 1400  
 1400       CONTINUE  
      DO 1426 L = 1, IY  
         BB = IX + L  
         IF (WJ (NO, BB) ) 1426, 1426, 1425  
 1425          CONTINUE  
         LDUM (LOC) = L  
         JDUM (LOC) = J  
         WDUM (LOC) = WJ (NO, BB)  
         PDUM (LOC) = PTOT (NO, BB) * C  
         IF (LOC.EQ.5) WRITE (7, 1415) N, (JDUM (K), LDUM (K), &
          WDUM (K), PDUM (K), K = 1, 5)
 1415 FORMAT (I4,5(I4,I3,F8.2,E10.3))  
         LOC = MOD (LOC, 5) + 1  
 1426       END DO  
 1450    IX = IX + IY  
   IL = LOC - 1  
   IF (LOC.NE.1) WRITE (7, 1415) N, (JDUM (K), LDUM (K), WDUM (K), &
    PDUM (K), K = 1, IL)
   IF (N.EQ.2) GOTO 1500  
   WRITE (7, 1460) N  
 1460 FORMAT ('  ***** FORBIDDEN LINES ***** N=',I3)  
   NLOW = NF (NO - 1) + 1  
   NHIGH = NF (NO)  
   LOC = 1  
   DO 1470 I = NLOW, NHIGH  
      IF (FWV (I) * FTOT (I) .LE.0.) GOTO 1470  
      LDUM (LOC) = I  
      WDUM (LOC) = FWV (I)  
      PDUM (LOC) = FTOT (I) * C  
      IF (LOC.EQ.5) WRITE (7, 212) N, (LDUM (K), WDUM (K), &
       PDUM (K), K = 1, 5)
  212 FORMAT (1X,I2,5(I6,F9.1,E10.3))  
      LOC = MOD (LOC, 5) + 1  
 1470    END DO  
   IL = LOC - 1  
   IF (LOC.NE.1) WRITE (7, 212) N, (LDUM (K), WDUM (K), PDUM (K), &
    K = 1, IL)
 1500 END DO  
RETURN  

END SUBROUTINE LNPRT

SUBROUTINE RADRD (ICONT)  
!     Read in radiation profile.

character (len=100) :: label  
COMMON / COM / CNC (12, 30), POWER (12, 220), ABIN (1000), &
 BIN (1000)
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
IF (ICONT.EQ.0) RETURN  
IRD = NBIN / 10  
read (3, 1394) label  
 1394 format (a100)  
DO 6003 I = 1, IRD  
   IL = 10 * I - 10  
 6003 READ (3, 1395) (ABIN (IL + K), K = 1, 10)  
 1395 FORMAT (13X,10E10.2)  
RETURN  

END SUBROUTINE RADRD
FUNCTION SOLVX (F, E, H)  
COMMON / SLV / DENZ, PRESS, PB, DYNAM  
!     CALCULATE COMPRESSION
AX2 = 2. * DENZ * (H - E) / PB  
AX1 = - 5. * PRESS / PB  
AX0 = 4. * DYNAM / PB  
QX = AX2 * AX2 / 9. - AX1 / 3.  
RX = (3. * AX0 - AX1 * AX2) / 6. + (AX2**3) / 27.  
PLINK = RX / QX**1.5  
IF (PLINK.GT.0.997) GOTO 431  
THETAX = ACOS (PLINK) / 3.  
SOLVX = SQRT (QX) * (COS (THETAX) + 1.732 * SIN (THETAX) ) &
 - AX2 / 3.
RETURN  
  431 X = ( - AX1 + SQRT (AX1 * AX1 - 4. * AX2 * AX0) ) / (2. * AX2)  
DO 432 K = 1, 4  
  432 X = ( - AX1 + SQRT (AX1 * AX1 - 4. * AX2 * (AX0 + X**3) ) ) &
 / (2. * AX2)
SOLVX = X  
RETURN  

END FUNCTION SOLVX
SUBROUTINE HLINE (RHY, A, T, ST, BETA, ALTER)  
COMMON / HLN / EXT (11), HALF, HALFC, ALFLY, ALFLYC, HBET, HBETC, &
 TWOP
RH = RHY / (1. + RHY)  
Y = .75 * 157890. / T  
CC = ALOG ( (Y + 1.) / Y) - 0.4 / (Y + 1.) **2  
!     CONTINUUM TO N .GE. 3
EXT (1) = 3.46 * RH / ST  
!     CONTINUUM TO N .EQ. 2
EXT (2) = (45.0 / ST) * SECONT (BETA, 2.) * RH / 8.  
!     CONTINUUM TO GROUND
EXT (3) = 45.0 * SECONT (BETA, 1.) * RH / ST  
IF (ALTER.GE.1.) EXT (3) = 0.  
!     TOTAL RECOMBINATION LINES
CHI = (.735 + ALOG (BETA) + .3333 / BETA) / 2.  
IF (BETA.LE..5) CHI = 1.2 * BETA + (.55 + 1.04 * ALOG (BETA) ) &
 * BETA * BETA + (1.01 * ALOG (BETA) - 0.43) * BETA**3
CHI = CHI - P (BETA, 1.)  
EX = .4288 + ALOG (BETA) / 2. + 0.469 * BETA** ( - .333) - SEATON &
 (BETA, 1.)
EXT (4) = (2.06 * 1.607 * 13.6 / ST) * (EX + CHI / BETA - SECONT ( &
 BETA, 2.) / 8. - .077) * RH
!     HBETA FROM RECOMBINATION
EXT (5) = .01275 * EXP ( - 410. / T) * RH / (.0001 * T) **.937  
!     LYMAN ALPHA FROM RECOMBINATION
EXT (6) = 2.06 * 1.6027 * 13.6 * .75 * EX * RH / ST  
!     TOTAL FROM COLLISIONAL EXCITATION
!     STILL USING OLD LYMAN ALPHA EXCITATION
SUM = .416 * EXP ( - .75 * BETA) * 1.195 + .07912 * EXP ( - .889 * &
 BETA) * 1.076 + .02899 * EXP ( - .9375 * BETA) * 1.041 + .01394 * &
 EXP ( - .96 * BETA) * 1.026 + .007799 * EXP ( - .972 * BETA) &
 * 1.018
EXT (7) = 4.5E-6 * 1.6027E-12 * (1.E23 / SQRT (13.6) ) * SUM / &
 ( (1. + RHY) * BETA**.12)
!     HBETA FROM COLLISIONAL EXCITATION : OLD H BETA
SUM = (.02899 * .739 * 3 / 16.) * EXP ( - .9375 * BETA) * 1.110 + &
 .0745 * .01394 * EXP ( - .96 * BETA) * 1.068 + .007799 * .079 * &
 EXP ( - .972 * BETA) * 1.047
EXT (8) = (4.5E-6 * 1.6027E11 / SQRT (13.6) ) * SUM / ( (1 + RHY) &
 * BETA**.12)
!     LYMAN ALPHA FROM COLLISIONAL EXCITATION
SUM = .0791 * EXP ( - .889 * BETA) * 1.210 + .02899 * EXP ( - &
 .9375 * BETA) * 1.110 + .01394 * EXP ( - .96 * BETA) * 1.068 + &
 .007799 * EXP ( - .972 * BETA) * 1.047
EXT (9) = (4.5E-6 * 1.6027E-12 * 1.E23 / SQRT (13.6) ) * SUM / &
 (BETA**.12 * (1. + RHY) )
!  AGGARWAL FITS TO MCDOWELL AND CALLAWAY WITH OLD CASCADES
TP = AMIN1 (T, 5.0E5)  
OM = .5 * (.335 + 1.45E-5 * TP + 1.39E-10 * TP * TP - 5.66E-15 * &
 TP * TP * TP)
IF (TP.GE.25000.) OM = .5 * (.328 + 1.43E-5 * TP - 6.55E-12 * TP * &
 TP - 2.69E-18 * TP * TP * TP)
IF (T.GE.5.0E5) OM = 8.05 * .276 * CC  
EXT (9) = EXT (9) + 8.63E-6 * OM * 10.2 * 1.6E11 * EXP ( - &
 118000. / T) / (SQRT (T) * (1. + RHY) )
!  H ALPHA EXCITATION; AGGARWAL ; CASE B, NO CASCADES
OM = (.175 - 1.31E-7 * TP - 4.08E-11 * TP * TP + 3.10E-15 * TP * &
 TP * TP)
IF (TP.GE.25000.) OM = (.138 + 2.50E-6 * TP - 4.43E-12 * TP * TP + &
 3.59E-18 * TP * TP * TP)
EXT (10) = 8.63E-6 * OM * 1.89 * 1.6E11 * EXP ( - 140300. / T) &
 / (SQRT (T) * (1. + RHY) )
!  TWO PHOTON FROM AGGARWAL;  NO CASCADES INCLUDED
OM = .5 * (.195 + 1.28E-5 * TP - 4.69E-10 * TP * TP + 6.19E-15 * &
 TP * TP * TP)
IF (TP.GE.25000.) OM = .5 * (.308 + 3.58E-7 * TP + 6.15E-13 * TP * &
 TP - 1.08E-18 * TP * TP * TP)
EXT (11) = 8.63E-6 * OM * 10.2 * 1.6E11 * EXP ( - 118000. / T) &
 / (SQRT (T) * (1. + RHY) )
HALF = HALF + EXT (5) * A * (2.654 + 1730. / T)  
HALFC = HALFC + EXT (10) * A  
HBET = HBET + EXT (5) * A  
HBETC = HBETC + EXT (8) * A  
ALFLY = ALFLY + EXT (6) * A  
ALFLYC = ALFLYC + EXT (9) * A  
TWOP = TWOP + EXT (11) * A  
RETURN  


END SUBROUTINE HLINE
!
FUNCTION HCOOL (T, RHY, ALTER)  
COMMON / HLN / EXT (11), HALF, HALFC, ALFLY, ALFLYC, HBET, HBETC, &
 TWOP
BETA = 157890. / T  
ST = SQRT (T)  
CALL HLINE (RHY, 0., T, ST, BETA, ALTER)  
HCOOL = EXT (1) + EXT (2) + EXT (3) + EXT (4) + EXT (8) + EXT (9) &
 + EXT (10) + EXT (11)
RETURN  

END FUNCTION HCOOL
FUNCTION TAUP (W, IT)  
COMMON / PT / RF (500), TAU (150), TAUHE (150), TPLUS (150)  
TAUP = 1.E-8  
IF (W.GT.912.) RETURN  
TAUP = TAUP + TAU (IT) * (W / 912.) **3  
IF (W.LT.504.6) TAUP = TAUP + TAUHE (IT) * (.763 * (W / 504.6) ** &
 1.99 + .237 * (W / 504.6) **2.99)
IF (W.LT.228.) TAUP = TAUP + TPLUS (IT) * (W / 228.) **3  
RETURN  

END FUNCTION TAUP
FUNCTION EXPF (X)  
EXPF = 1.0E-37  
IF (X.GE. - 75..AND.X.LE.75.) EXPF = EXP (X)  
IF (X.GT.75.) EXPF = 1.0E37  
RETURN  

END FUNCTION EXPF
FUNCTION GAMMA (A, X)  
REAL (8) :: CAN1, CAN2, CBN1, CBN2, AN, BN, CAN, CBN  
!     EXP(-X) * X ** A   TAKEN OUTSIDE
CAN2 = 1.  
CAN1 = 0.  
CBN2 = 0.  
CBN1 = 1.  
AN = 1.  
BN = X  
N = 1  
CAN = BN * CAN1 + AN * CAN2  
CBN = BN * CBN1 + AN * CBN2  
FN1 = CAN / CBN  
DO 100 N = 2, 1000  
   IF (CAN.LE.1.E30) GOTO 50  
   CAN1 = CAN1 * 1.E-30  
   CAN = CAN * 1.E-30  
   CBN1 = CBN1 * 1.E-30  
   CBN = CBN * 1.E-30  
   50    CONTINUE  
   IN = N  
   CAN2 = CAN1  
   CBN2 = CBN1  
   CAN1 = CAN  
   CBN1 = CBN  
   MN = MOD (N, 2)  
   BN = MN * X + (1. - MN)  
   NN = N / 2  
   AN = NN - (1. - MN) * A  
   CAN = BN * CAN1 + AN * CAN2  
   CBN = BN * CBN1 + AN * CBN2  
   FN = CAN / CBN  
   IF (ABS ( (FN - FN1) / FN1) .LE..0001.AND.N.GE.20) GOTO 200  
   FN2 = FN1  
  100 FN1 = FN  
GAMMA = (FN + FN2) / 2.  
RETURN  
  200 GAMMA = FN  
RETURN  

END FUNCTION GAMMA
FUNCTION FABLE (X, Y, Z)  
IF (X - Y) 10, 20, 10  
   10 DUCK = (EXPF ( - X * Z) - EXPF ( - Y * Z) ) / (Y - X)  
GOTO 30  
   20 DUCK = Z * EXP ( - X * Z)  
   30 FABLE = DUCK  
CONTINUE  
RETURN  

END FUNCTION FABLE



SUBROUTINE BLND (NUM)  
!  COMPUTES EMISSION SPECTRUM IN WAVELENGTH BINS : FIRST BLN EXTENDS FRO
!  BLNMIN TO BLNMIN+BLNSYZ
!  RESOLUTION AND RANGE OF CONTINUUM ARE ONLY AS GOOD AS THE
!  RESOLUTION AND RANGE OF THE ENERGY BIN COMPUTATION.  BE SURE THAT
!  NBIN, BINMIN AND BINSYZ PROVIDE THE NECESSARY INTERVAL.  BLN'S
!  OUTSIDE THE RANGE OF THE ENERGY BINS WILL HAVE NO CONTINUUM
!  CONTRIBUTION, BUT PLACEMENT OF LINES WILL NOT BE AFFECTED.
!  SEPARATES RESONANCE DOUBLETS OF LI-LIKE AND NA-LIKE LINES
!  FEB. 1983
COMMON / BLN / BLN (1000), BLNMIN, BLNSYZ, NBLN  
COMMON / PARAMS / NJ (12), ABUNJ (12), ABUND, BINMIN, BINSYZ, &
 NBIN
COMMON / CONTIN / BRMEV (1000), RECEV (1000), TUFEV (1000)  
COMMON / FEL / WJ (12, 220), FJ (12, 220), E3J (12, 220), PWR (12, &
 220), C1J (12, 30), C2J (12, 30), EJ (12, 30), SJ (12, 30), &
 CJ (12, 30), LLJ (12, 30), SIGJ (12, 30), ALFJ (12, 30), ESJ (12, &
 30)
DIMENSION L1 (12), L2 (12), WLI (2, 12), WNA (2, 12)  
DATA L1 / 0, 28, 37, 43, 52, 85, 103, 127, 88, 148, 184, 166 /  
DATA L2 / 5 * 0., 13, 28, 43, 34, 64, 85, 88 /  
DATA WLI / 0., 0., 1548.2, 1550.8, 1238.8, 1242.8, 1031.9, 1037.6, &
 770.4, 780.3, 609.8, 624.9, 499.4, 520.7, 417.6, 445.8, 353.9, &
 389.1, 302.2, 344.8, 192.03, 255.11, 165.42, 234.20 /
DATA WNA / 10 * 0., 2796.4, 2803.5, 1393.8, 1402.8, 933.4, 944.5, &
 700.4, 714.0, 557.7, 574.0, 335.4, 360.8, 292.0, 320.6 /
!
!  CONTINUUM
!
DO 1000 IBLN = 1, NBLN  
   WBLN = (IBLN - 0.5) * BLNSYZ + BLNMIN  
   EPHOT = 12399. / WBLN  
   IBN = (EPHOT - BINMIN) / BINSYZ + 1  
   DE = EPHOT - 12399. / (WBLN + BLNSYZ)  
   IF (IBN.GE.1.AND.IBN.LE.NBIN) BLN (IBLN) = (BRMEV (IBN) &
    + RECEV (IBN) + TUFEV (IBN) ) * DE / BINSYZ
 1000 END DO  
!
!  LINES
!
DO 2000 NO = 1, NUM  
   DO 1500 L = 1, 220  
      IF (L.EQ.L1 (NO) ) GOTO 1100  
      IF (L.EQ.L2 (NO) ) GOTO 1200  
      GOTO 1400  
 1100       IBLN = (WLI (1, NO) - BLNMIN) / BLNSYZ + 1  
      IF (IBLN.GE.1.AND.IBLN.LE.NBLN) BLN (IBLN) = BLN (IBLN) &
       + 0.6666666 * PWR (NO, L)
      IBLN = (WLI (2, NO) - BLNMIN) / BLNSYZ + 1  
      IF (IBLN.GE.1.AND.IBLN.LE.NBLN) BLN (IBLN) = BLN (IBLN) &
       + 0.333333 * PWR (NO, L)
      GOTO 1499  
 1200       IBLN = (WNA (1, NO) - BLNMIN) / BLNSYZ + 1  
      IF (IBLN.GE.1.AND.IBLN.LE.NBLN) BLN (IBLN) = BLN (IBLN) &
       + 0.666666 * PWR (NO, L)
      IBLN = (WNA (2, NO) - BLNMIN) / BLNSYZ + 1  
      IF (IBLN.GE.1.AND.IBLN.LE.NBLN) BLN (IBLN) = BLN (IBLN) &
       + 0.33333 * PWR (NO, L)
      GOTO 1499  
 1400       CONTINUE  
      IBLN = (WJ (NO, L) - BLNMIN) / BLNSYZ + 1  
      IF (IBLN.GE.1.AND.IBLN.LE.NBLN) BLN (IBLN) = BLN (IBLN) &
       + PWR (NO, L)
 1499       CONTINUE  
 1500    END DO  
 2000 END DO  
RETURN  
END SUBROUTINE BLND

FUNCTION EFFNEW (I, T, Z, N)  
T3 = .001 * T / Z**2  
XX = .4342 * LOG (T3)  
IF (T3 - 1.) 101, 101, 102  
  101 F1 = .266  
F2 = .13  
F3 = .13  
GOTO 105  
  102 IF ( (T3 - 1.E5) .GE.0.) GOTO 104  
F1 = .266 + .1068 * XX - .074 * SIN (1.2566 * XX)  
F2 = .130 + .1160 * XX - .074 * SIN (1.2566 * XX)  
F3 = .130 - .012 * XX + .05 * EXP ( - (XX - 2.) * (XX - 2.) )  
GOTO 105  
  104 F1 = .80  
F2 = .71  
F3 = .07  
  105 CONTINUE  
IF (I.EQ.2.OR.I.EQ.3) EFFNEW = 8. * (1. - F1)  
IF (I.EQ.10.OR.I.EQ.11) EFFNEW = 18. * (1. - F2)  
IF (I.GE.12.AND.I.LE.17) EFFNEW = 18. * (1. - 3. * F3 - F2)  
IF (I.GE.18) EFFNEW = (28. - N + Z) * 1.8 * (1. - 3. * F3 - F2)  
RETURN  

END FUNCTION EFFNEW
FUNCTION EFFN (I, ZEFF, T)  
  100 T3 = T / (ZEFF * ZEFF * 1000.0)  
XX = 0.4342 * LOG (T3)  
IF (T3 - 1.0) 101, 101, 102  
  101 F1 = 0.266  
F2 = 0.13  
F3 = 0.13  
GOTO 105  
  102 IF (T3 - 10.**5) 103, 104, 104  
  103 F1 = 0.266 + 0.1068 * XX - 0.074 * SIN (1.2566 * XX)  
F2 = 0.130 + 0.1160 * XX - 0.074 * SIN (1.2566 * XX)  
F3 = 0.130 - 0.0120 * XX + 0.050 * EXP ( - (XX - 2.) * (XX - 2.) )  
GOTO 105  
  104 F1 = 0.80  
F2 = 0.71  
F3 = 0.07  
  105 CONTINUE  
IF (I.EQ.0) GOTO 1  
IF (I - 18) 110, 110, 18  
  110 GOTO (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, &
 18), I
    1 EYE = I  
EFFN = 2. - EYE  
GOTO 1000  
    2 EFFN = 8.0  
GOTO 1000  
    3 EFFN = 8. - (4. * F1)  
GOTO 1000  
    4 EFFN = 8. * (1. - F1)  
GOTO 1000  
    5 EFFN = 6.6667 * (1. - F1)  
GOTO 1000  
    6 EFFN = 5.33333 * (1. - F1)  
GOTO 1000  
    7 EFFN = 4. * (1. - F1)  
GOTO 1000  
    8 EFFN = 2.6667 * (1. - F1)  
GOTO 1000  
    9 EFFN = 1.33333 * (1. - F1)  
GOTO 1000  
   10 EFFN = 18.  
GOTO 1000  
   11 EFFN = 18. - (9. * F2)  
GOTO 1000  
   12 EFFN = 18.0 * (1. - F2)  
GOTO 1000  
   13 EFFN = 18. * (1. - F2) - 1. * (9. * F3)  
GOTO 1000  
   14 EFFN = 18. * (1. - F2) - 2. * (9. * F3)  
GOTO 1000  
   15 EFFN = 18. * (1. - F2) - 3. * (9. * F3)  
GOTO 1000  
   16 EFFN = 18. * (1. - F2) - 4. * (9. * F3)  
GOTO 1000  
   17 EFFN = 18. * (1. - F2) - 45.0 * F3  
GOTO 1000  
   18 NO = 28 - I  
EFFN = NO * 1.8 * (1. - 3. * F3 - F2)  
IF (EFFN) 106, 1000, 1000  
  106 EYE = I  
EFFN = 60. - EYE  
!     GUARD PACKAGE FOR I = 17
!     PROBABLY UNNECESSARY
IF (EFFN.LE.0) EFFN = 1.0  
 1000 CONTINUE  
RETURN  

END FUNCTION EFFN
FUNCTION GREC (N, J, E, T)  
COMMON / DAT / V (30), EA (30), S2 (30), WAVE (220), E3 (220), &
 F (220), LL (30), S3 (30), S4 (30), S5 (30)
DIMENSION DRAT (30)  
DATA DRAT / 2., .5, 2., .5, 6., 2.5, 2.22, 2.25, .67, .167, 2., &
 .5, 6., 2.5, 2.22, 2.25, .67, .167, 12 * 0. /
DEG = DRAT (N - J + 1)  
X1 = E * 11590. / T  
X12 = X1 * X1  
X14 = X12 * X12  
G = 5.238E-23 * T**1.5 * DEG * (S2 (J) * X12 + S4 (J) * X12 * X1 + &
 X12 * X1 * S5 (J) * (0.5 - X1 / 2.) + EXINT1 (X1, 2) * (S3 (J) &
 * X12 * X1 - X14 * S4 (J) + X1 * X14 * S5 (J) / 2.) )
GREC = G  
if (X1.ge.300.) grec = 5.238e-23 * t**1.5 * deg * (s2 (j) + s3 (j) &
 + s4 (j) + s5 (j) ) * x12
RETURN  

END FUNCTION GREC
FUNCTION SEATON (X, Q)  
!     R. MOORE OCTOBER 1976
IMPLICIT REAL (8)(A, B)  
DOUBLEPRECISION XD, XS1, XS2, X3  
DATA A10 / 8.7469604697013D-4 /, B10 / - 1.2007397521051D-4 /  
DATA A11 / .20406771231267D0 /, B11 / - .055223640825293D0 /  
DATA A12 / - 2.1524482972354D0 /, B12 / .029171138841798D0 /  
DATA A13 / 12.663578339302D0 /, B13 / .25091093604147D0 /  
DATA A14 / - 43.153566859883D0 /, B14 / - .94344532109356D0 /  
DATA A15 / 61.113098339262D0 /  
DATA A20 / .011834608639468D0 /, B20 / - 1.2467753947278D-3 /  
DATA A21 / - 2.9889195903436D-3 /, B21 / - .043871158058636D0 /  
DATA A22 / - .10946027271945D0 /, B22 / .013617064094285D0 /  
DATA A23 / .097410292577482D0 /, B23 / - 5.2824884665512D-3 /  
DATA A24 / - .039676727608179D0 /, B24 / 8.9490487211065D-4 /  
DATA A25 / 6.2318768197420D-3 /  
DATA A30 / .018985613015081D0 /, B30 / - .010963734653233D0 /  
DATA A31 / - .064707516794785D0 /, B31 / - .028119928428050D0 /  
DATA A32 / 6.2172804659938D-3 /, B32 / 1.1971209805431D-3 /  
DATA A33 / - 4.1777961107942D-4 /, B33 / - 4.8739343472085D-5 /  
DATA A34 / 1.5264422582645D-5 /, B34 / 8.4274360230135D-7 /  
DATA A35 / - 2.2708774951499D-7 /  
IF (X.GE.20.) GOTO 40  
IF (X.GT.2.) GOTO 30  
IF (X.GT.0.2) GOTO 20  
IF (X.GT.0.02) GOTO 10  
XS1 = .4629 * X * (1. + 4. * X) - 1.0368 * X**1.3333333333333 * &
 (1. + 1.875 * X)
XS2 = - .0672 * X * (1. + 3. * X) + .1488 * X**1.6666666666667 * &
 (1. + 1.8 * X)
GOTO 50  
   10 XS1 = A10 + (A11 + (A12 + (A13 + (A14 + A15 * X) * X) * X) &
 * X) * X
XS2 = B10 + (B11 + (B12 + (B13 + B14 * X) * X) * X) * X  
GOTO 50  
   20 XS1 = A20 + (A21 + (A22 + (A23 + (A24 + A25 * X) * X) * X) &
 * X) * X
XS2 = B20 + (B21 + (B22 + (B23 + B24 * X) * X) * X) * X  
GOTO 50  
   30 XS1 = A30 + (A31 + (A32 + (A33 + (A34 + A35 * X) * X) * X) &
 * X) * X
XS2 = B30 + (B31 + (B32 + (B33 + B34 * X) * X) * X) * X  
GOTO 50  
   40 X3 = 3. * X  
XS1 = - .1728 * X**.33333333333333 * (1. + ( - 8. + (70. + &
 ( - 800. + 11440. / X3) / X3) / X3) / X3)
XS2 = - .0496 * X**.66666666666667 * (1. + ( - 3. + (32. - 448. / &
 X3) / X3) / X3)
   50 X3 = X**.33333333333333 * Q**.66666666666667  
SEATON = EXINT1 (X, 3) + (XS1 + XS2 / X3) / X3  
RETURN  

END FUNCTION SEATON
FUNCTION P (Y, Q)  
A = Y / (Q * Q)  
IF (A.LE.75) GOTO 20  
P = 1. / Q  
RETURN  
   20 P = A * (1. - EXINT1 (A, 3) ) / Q  
RETURN  

END FUNCTION P
FUNCTION GND (NUM)  
GND = 4.  
IF (NUM.LT.28) GND = 3.  
IF (NUM.LT.10) GND = 2.  
IF (NUM.LT.2) GND = 1.  
RETURN  

END FUNCTION GND
FUNCTION EXINT1 (X, JUMP)  
!     R. MOORE OCTOBER 1976
!   JUMP=1    EXINT1=E1(X)
!   JUMP=2    EXINT1=EXP(X)*E1(X)
!   JUMP=3    EXINT1=X*EXP(X)*E1(X)
IF (X.GE.1.) GOTO 30  
EXINT1 = ( ( ( ( ( ( ( (7.122452D-7 * X - 1.766345D-6) * X + &
 2.928433D-5) * X - .0002335379D0) * X + .001664156D0) * X - &
 .01041576D0) * X + .05555682D0) * X - .2500001D0) * X + &
 .9999999D0) * X - LOG (X) - .57721566490153D0
GOTO (9999, 10, 20), JUMP  
   10 EXINT1 = EXP (X) * EXINT1  
RETURN  
   20 EXINT1 = X * EXP (X) * EXINT1  
RETURN  
   30 X2 = X * X  
X3 = X2 * X  
X4 = X3 * X  
EXINT1 = (X4 + 8.5733287401D0 * X3 + 18.059016973D0 * X2 + &
 8.6347608925D0 * X + .2677737343D0) / (X4 + 9.5733223454D0 * X3 + &
 25.6329561486D0 * X2 + 21.0996530827D0 * X + 3.9584969228D0)
GOTO (40, 50, 9999), JUMP  
   40 EXINT1 = EXINT1 * EXP ( - X) / X  
RETURN  
   50 EXINT1 = EXINT1 / X  
 9999 RETURN  

END FUNCTION EXINT1
FUNCTION ETWO (X)  
ETWO = EXP ( - X) - X * EXINT1 (X, 1)  
RETURN  
END FUNCTION ETWO
!
!---------------------------------------------------------------------------
!
Subroutine radiative_part(g_ksi,f_ksi,NFELD)
!**************************************************************************
! calculates the functions f(ksi) g(ksi), h(ksi) 
! being ksi divided in NFELD parts, along Lc (cooling length),
! between 0 and 1. ksi is a parametrization of the radiative cooling
! length as introduced in Feldmeier et al. 1997
!**************************************************************************

use nlte_type
implicit none  

INTEGER(i4b), intent(in) ::  NFELD      
REAL(dp), DIMENSION (NFELD) , intent(out) :: g_ksi, f_ksi

REAL(dp), parameter ::   A = 0.8722544d0 !see Feldmeier Eq. 11

REAL(dp), parameter :: twoseventh=2.d0/7.d0, third=1.d0/3.d0

Real(dp) :: Xi, h_ksi         

INTEGER(i4b) ::  i      

!************************************************************************** 

!integration from eps to 1


DO i= 1,NFELD
        Xi=(FLOAT(i)/FLOAT(NFELD))**twoseventh     !See that Xi never reachs the value zero.
        h_ksi  = A * Xi * (1.0d0+(1.0d0/A-1.0d0) *Xi)
        f_ksi(i)=1.0d0/h_ksi
        g_ksi(i)=third * h_ksi *(4.0d0-h_ksi)
!        print*,xi,f_ksi(i),g_ksi(i)
end DO

end Subroutine radiative_part
!
!---------------------------------------------------------------------------
!
Subroutine adiabatic_part(theta,kappa_f,f_eta_rev,g_eta_rev,f_eta_for,g_eta_for,NFELD)
!*************************************************************************** 
! calculates the functions eta_f (for the forward shock)
! and eta_r (reverse shock).
! For details, see Feldmeier et al. 1997
!*************************************************************************** 

use nlte_type
implicit none  

INTEGER(i4b), intent(in) ::  NFELD

! theta = T_r/T_f (just after jump), defined in the calling routine
REAL(dp), intent(in) :: theta, kappa_f !as defined by Feldmeier
REAL(dp), DIMENSION (NFELD) , intent(out) :: f_eta_rev, g_eta_rev,   &
&                                            f_eta_for, g_eta_for

!local
Integer(i4b) :: i

REAL(dp) ::  eta_c,  &                ! Contact discontinuity inside the shell
             eta_f,  &                ! Location of the foward shock front (= 1.)
             eta_r,  &                ! Position of the reverse shock front
             h_eta_rev, &
             h_eta_for, &
             XI

REAL(dp) :: eta_rev, eta_for

REAL(dp), parameter :: mfn=-4.d0/9.d0


! *************************************************************************** 


eta_c = 1.d0-0.1314d0*kappa_f-0.02857d0*kappa_f**2

eta_f = 1.d0  !position of forward shock

eta_r = eta_c - (1.d0-eta_c)*(1.d0-kappa_f+kappa_f**2)*theta**  &
&            (0.5d0-0.31d0*kappa_f+0.55d0*kappa_f**2)
!   print*, eta_c, eta_f, eta_r

!**************
!REVERSE SHOCKS
!**************
DO i= 1,NFELD
       XI = float(i) / float(NFELD)
       eta_rev   = eta_c - XI * (eta_c-eta_r)
       h_eta_rev    = ((eta_c/eta_rev)**3 - 1.d0) / ((eta_c/eta_r)**3 - 1.d0)
       f_eta_rev(i) =  1.d0/theta * (eta_r/eta_rev)**2 * h_eta_rev**mfn
       g_eta_rev(i) =  1.d0/f_eta_rev(i)
!       print*, eta_rev, f_eta_rev(i), g_eta_rev(i)
!       print*, xi, f_eta_rev(i), g_eta_rev(i)
end DO

!************
!FOWARD SHOCKS
!************
DO i= 1,NFELD
       XI = float(i) / float(NFELD)
       eta_for   = eta_c + XI * (1.d0-eta_c)
       h_eta_for    = ((eta_c/eta_for)**3 - 1.d0) /((eta_c/eta_f)**3 - 1.d0)
       f_eta_for(i) =  1.d0 * (eta_f/eta_for)**2 * h_eta_for**mfn
       g_eta_for(i) =  1./f_eta_for(i)
!       print*, eta_for, f_eta_for(i), g_eta_for(i)
!       print*, xi, f_eta_for(i), g_eta_for(i)
end DO

end subroutine adiabatic_part
!
!---------------------------------------------------------------------------
!
Subroutine cooling_flowing_time(Fract_tc_tf,Fract_Lc_rs,tsl,rl,vl,rhol,xnel,xnhl, &
&                               xmu,theta,kappa_f,l)

!**************************************************************************
! calculates t_cool/t_flow and L_cool/r_sh,
! as well as kappa_f (Eq. 19, with vinf replaced by v(r)=vl
! as introduced in Feldmeier et al. 1997
!
! Note that as in Feldmeier, the jump-temperature as calculated in
! subroutine Tshock refers to the reverse shock (radiative case)
! and to the forward shock (adiabtic case, kappa_f needed)  
! Thus, Tsh_rev = Tjump*theta , and Tsh_for = Tjump in the adiabtic case,
! whilst Tsh_rev = Tjump in the radiative case.
!**************************************************************************

USE nlte_type
USE nlte_var, only: vmax, sr
USE fund_const, only: akb, amh, pi
USE nlte_xrays, only: fx,px,fxl,rminx_eff


IMPLICIT NONE

real(dp), intent(out) :: Fract_tc_tf,Fract_Lc_rs,kappa_f
real(dp), intent(in)  :: tsl,rl,vl,rhol,xnel,xnhl,xmu,theta
integer(i4b), intent(in) :: l
! xnel and xnhl average values, re-correct for clumping

! constant for freq. integrated cooling function, Feldmeier Eq. 7
real(dp), parameter :: ar=1.64d-19
real(dp), parameter :: fac=(93.d0*sqrt(3.d0)-40.d0*pi)/10.d0

real(dp) :: tfpo

Fract_tc_tf=40./7./ar*(16./3.)**1.5/4.**4*akb/(xmu*amh)
Fract_tc_tf=Fract_tc_tf*(vl*vmax)/(rl*sr)*rhol/(xnel*xnhl)*tsl**1.5

!hack to ensure adiabatic treatment
!Fract_tc_tf=1.1

Fract_Lc_rs=fac/ar*(16./3.)**2/4.**5*(akb/(xmu*amh))**1.5
Fract_Lc_rs=Fract_Lc_rs/(rl*sr)*rhol/(xnel*xnhl)*tsl**2

!either use old approach with constant filling factor
!(giving rise to rho^2 dependent emissivities in all cases)
if (px.lt.-999.99) then
  fxl(l)=fx
!or calculate new X-ray filling-factor according to work by Koushik Sen + JP,
!unifying the Feldmeier+ and Owocki+ (2013) work 
else
  fxl(l)=fx*(rminx_eff/rl)**px*amin1(Fract_Lc_rs,1.)
endif
!NOTE: this routine is only called for l=1,lxmin,
!and all fxl(l) for l>lxmin are set to zero

if(Fract_tc_tf.lt.1.d0) then
  kappa_f= 0.d0
  write(*,10) l,rl,tsl,fract_tc_tf,fract_Lc_rs,fxl(l)
else
  tfpo=tsl  !see comment above
  kappa_f=1.d0+sqrt(3.d0*xmu*amh/(16.*akb*tfpo))*vl*vmax
  kappa_f=1.d0/kappa_f
  write(*,20) l,rl,tsl,fract_tc_tf,fract_Lc_rs,fxl(l),kappa_f
endif

 10 format('L =',I3,' R = ',f9.3,' Tshock = ',e9.3,' tcool/tflow = ',e9.3,' Lcool/rsh = ',e9.3,' fvol(l) = ',e9.3,' RADIATIVE!')
 20 format('L =',I3,' R = ',f9.3,' Tshock = ',e9.3,' tcool/tflow = ',e9.3,' Lcool/rsh = ',e9.3,' fvol(l) = ',e9.3,&
&           ' ADIABATIC! kappa = ',f9.3)  

end Subroutine cooling_flowing_time
!
!---------------------------------------------------------------------------
!
Subroutine interpol_xray
!  
!**************************************************************************
!interpolates lambda_xray (per bin) from equidistant energy grid
!onto nlte-freq. grid in such a way that integrals lambda_nu dnu
!remain preserved  
!on output, integrals(k,l) contain emitted frequential X-ray energy
!**************************************************************************

USE nlte_type
USE nlte_dim, only: nd => ID_NDEPT
USE fund_const, only: clight
USE nlte_xrays, only: enerx, lambdax,integrals=>lambdanu, lxmin
USE nlte_var, only: fre, wfre, ifre 

IMPLICIT NONE

real(dp), parameter :: lamred = 3500. ! start of x-ray treatment

real(dp), allocatable, dimension(:,:) :: lamint
real(dp), allocatable, dimension(:,:) :: dum

real(dp), dimension(nd) :: eps0, intlow, intup

real(dp) :: dx, ered, x1, q, q1, dx1, dxhalf, h, eb, er, error

integer(i4b) :: nf, k, i, ired, kp, inext, l, kred, kblue


if(lbound(enerx,1).ne.0) stop ' lower bound of enerx ne 0'
nf=ubound(enerx,1)

if(1.d8/lamred .lt. enerx(1)) stop ' interpol_xray: lamred outside xray grid'
if(fre(ifre) .gt. enerx(nf)) stop ' interpol_xray: fre(ifre) outside xray grid'

!test whether equidistant grid
dx=enerx(1)-enerx(0)

if(abs(1.-dx/(enerx(nf)-enerx(nf-1))).gt.1.d-12) stop ' enerx not equidistant'

allocate(lamint(0:nf,lxmin))
allocate(dum(ifre,lxmin))

if (allocated(integrals)) then
  deallocate(integrals)
  allocate(integrals(ifre,lxmin))
else
  allocate(integrals(ifre,lxmin))
endif

!indices i correspond to enerx (xray bins), and indices k to fre-grid

!integrate lambdax (sum up, since already energy per bin)

!summation might provide problems for very low numbers.
!tests showed that also quadruple precision and reverse summation does not help
!Anyway, problems only when emissivity very low. 
lamint(nf,:)=0.d0
integrals=0.d0
dum=0.d0

! integrating from high to low frequencies, to avoid numerical problems
! note indices
do i=nf-1,0,-1
    lamint(i,:)=lamint(i+1,:)+lambdax(i+1,:)
enddo

!check for precision
do l=1,lxmin
  do i=0,nf-1
    if(lamint(i,l).le.lamint(i+1,l)) then
      if(lambdax(i+1,l).ne.0.d0) then
        print*,i,l,lamint(i,l),lamint(i+1,l),lambdax(i+1,l)
        stop ' interpol_xray: problems with precision'
      endif
    endif
  enddo
enddo  

ered=1.d8/lamred

do i=1,nf
  if(ered.ge.enerx(i-1) .and. ered .lt. enerx(i)) goto 5
enddo     
stop ' 1 in interpol_xray'

5 continue

!ered inside bin i-1,i, corresponding to energy i (see RS manual)
ired=i

!calculate integrals between fre(k-1) and fre(k)

!lower freq. for first integral = ered
q=(ered-enerx(ired))/(enerx(ired-1)-enerx(ired))
q1=1.d0-q

intlow(1:lxmin)=q*lamint(ired-1,:)+q1*lamint(ired,:)

!define start index (upper freq. for first integral)
do k=1,nf
  if(fre(k).gt.ered) goto 10
enddo
stop ' 2 in interpol_xray'

10 continue

kp=k
inext=ired

do k=kp,ifre !over all frequencies

x1=fre(k)
do i=inext,nf
  if(x1 .ge. enerx(i-1) .and. x1 .lt. enerx(i)) goto 15
enddo     
stop ' 3 in interpol_xray'

15 continue
!interpolate energy-integrals at fre(k)

inext=i !for next index

q=(x1-enerx(inext))/(enerx(inext-1)-enerx(inext))
q1=1.d0-q

intup(1:lxmin)=q*lamint(inext-1,:)+q1*lamint(inext,:)

!do l=1,lxmin
!  print*,l,intlow(l),intup(l)
!enddo

!reverse order, since integration from high to low frequencies
!JO Sept 2020: bug found by Sarah Brands removed 
!integrals(k,:)=intlow-intup
integrals(k,:)=intlow(1:lxmin)-intup(1:lxmin)
if (minval(integrals(k,:)).lt.0.d0) stop ' interpol_xrays: integrals negative'

intlow=intup

enddo ! all frequencies

!total energy
do l=1,lxmin
  h=sum(integrals(kp:ifre,l))
  if (h.gt.lamint(1,l)) stop ' interpol_xrays: summ(energies) > lamint!'
  eps0(l)=h !slightly different from lamint, due to edge effects
enddo


!this is now the frequential energy on the coarse grid w.r.t. staggered frequency grid
k=kp
integrals(k,:)=integrals(k,:)/(clight*(fre(k)-ered))
do k=kp+1,ifre
  integrals(k,:)=integrals(k,:)/(clight*(fre(k)-fre(k-1)))
enddo

!now, interpolate frequential energy onto original grid, conserving the energy
!(see notes)
!first frequency
k=kp
   q=(fre(k)-ered)/(fre(k+1)-ered)
   q1=(fre(k+1)-fre(k))/(fre(k+1)-ered)
   dum(k,:)=q*integrals(k,:)+q1*integrals(k+1,:)

do k=kp+1,ifre-1
   q=(fre(k)-fre(k-1))/(fre(k+1)-fre(k-1))
   q1=(fre(k+1)-fre(k))/(fre(k+1)-fre(k-1))
   dum(k,:)=q*integrals(k,:)+q1*integrals(k+1,:)
enddo

!last frequency (no action required)
dum(ifre,:)=integrals(ifre,:)

!integrals now contain the frequential energies on the original grid
integrals=dum


!--------------------------------------------------------------------------
!Now lets test a specific, line contaminated interval, say between 10 to 20 A
!corresponding to 0.6 to 1.2 keV
!can be skipped if everything is running

do k=kp,ifre-1
  if(1.d8/fre(k).gt.20.d0 .and. 1.d8/fre(k+1).le.20.d0) exit
enddo     
kred=k
do k=kred,ifre-1
  if(1.d8/fre(k).gt.10.d0 .and. 1.d8/fre(k+1).le.10.d0) exit
enddo     
kblue=k+1
er=fre(kred)
eb=fre(kblue)

!at first, exact integral
do i=1,nf
  if(er.ge.enerx(i-1) .and. er.lt.enerx(i)) exit 
enddo     
!er inside bin i-1,i, corresponding to energy i (see RS manual)
ired=i
!lower freq. for first integral = ered
q=(er-enerx(ired))/(enerx(ired-1)-enerx(ired))
q1=1.d0-q
intlow(1:lxmin)=q*lamint(ired-1,:)+q1*lamint(ired,:)


do i=ired,nf
  if(eb.ge.enerx(i-1) .and. eb.lt.enerx(i)) exit 
enddo     
!eb inside bin i-1,i, corresponding to energy i (see RS manual)
ired=i
!lower freq. for first integral = ered
q=(eb-enerx(ired))/(enerx(ired-1)-enerx(ired))
q1=1.d0-q
intup(1:lxmin)=q*lamint(ired-1,:)+q1*lamint(ired,:)

intup=-(intup-intlow)

!now, integral over interpolated values
intlow(1:lxmin)=0.d0
do k=kred,kblue-1
intlow(1:lxmin)=intlow(1:lxmin) + &
&  0.5d0*(integrals(k,:)+integrals(k+1,:))*(fre(k+1)-fre(k))*clight
enddo

error=0.d0
do l=1,lxmin
  if (intup(l).ne.0.d0) error=max(error,abs(1.d0-intlow(l)/intup(l)))
enddo

print*
print*,'test for interval = ',1.d8/er,'to',1.d8/eb,' A'
print*,'energy not conserved by ',error
if(error.gt.0.05) stop ' interpol_xrays: energy of interpolated values not conserved(1)!'

!--------------------------------------------------------------------------
!energy conservation in total freq. range, and
!renormalization w.r.t. standard integration weights

error=0.d0
do l=1,lxmin
  h=0.d0
  do k=kp,ifre
   h=h+wfre(k)*integrals(k,l)
  enddo
  if(eps0(l).ne.0.d0) then
    error=max(error,abs(1.d0-h/eps0(l)))
    integrals(kp:ifre,l)=integrals(kp:ifre,l)*eps0(l)/h
  else
    if (h.ne.0.d0) stop ' interpol_xrays: eps0 = 0 and h ne 0!'
  endif
enddo

print*
print*,'test for total range = ',1.d8/ered,'to',1.d8/fre(ifre),' A'
print*,'energy not conserved by ',error
if(error.gt.0.05) stop ' interpol_xrays: energy of interpolated values not conserved(2)!'

print*
print*,'interpolation of x-ray emission to coarse grid done'

!can be deleted when everything is running
do k=1,kp-1
  do l=1,lxmin
    if(integrals(k,l).ne.0d0) stop ' problem in integrals'
  enddo
enddo  

deallocate(lamint,dum)

return

!for tests at l
l=10
do i=1,nf
  print*,i,1.d8/(enerx(i)-0.5*dx),lambdax(i,l)/(clight*dx)
enddo

do k=kp,ifre
  print*,k,1.d8/fre(k),integrals(k,l)
enddo
stop

end
!
!---------------------------------------------------------------------------
!
subroutine read_kshell_data

USE nlte_type
USE fund_const,only: hh,clight,ev
USE nlte_xrays, only: n_kedges, z, n, eth, sigma, s, zeff, aug_1_6

IMPLICIT NONE


integer :: i

real(dp) :: ZL10, E_thre, cross_s, s_calc, const, eth_kaastra, err, summ

character*(*), parameter :: fpath='../inicalc/ATOMDAT_NEW/'

open (unit = 2, file = fpath//'k_shell_Auger_data',status='old')
read(2,*)
read(2,*)
read(2,*)

! aug_1_6 contains fraction of number of electrons ejected
! during complete process (1 K-shell electron + n-1 Auger electrons)

err=0.d0
do i=1,n_kedges
!N is number of electrons
read(2,*) z(i),n(i),eth(i),eth_kaastra,sigma(i),s(i),zeff(i), &
&        aug_1_6(i,1),aug_1_6(i,2),aug_1_6(i,3), &
&        aug_1_6(i,4),aug_1_6(i,5),aug_1_6(i,6)
!sigma(i)=0.0
err=max(abs(1.-eth(i)/eth_kaastra),err)
summ=sum(aug_1_6(i,:))
if(summ.ne.10000.) stop ' error in Auger probabilities!'

!**************************************************
! Remember to multiply sigma(i) by 1.E-18
!**************************************************
!print*, z(i),n(i),eth(i),sigma(i),s(i),zeff(i) 
end do
close(2)

goto 10
!for tests
        i = 12
        ZL10=LOG10(float(Z(i)))
        E_thre =  10.d0**((0.24d0-0.43d0/ZL10)*LOG10(float(N(i)))+2.d0*ZL10+1.133539d0)
        cross_s = 10.d0**(-LOG10(ETH(i)) + 2.47d0)*1.d-18
        s_calc  = 10.d0**(0.07d0*LOG10(ETH(i)) + 0.221d0)

print*, E_thre, cross_s, s_calc
print*
print*, ETH(i), sigma(i)*1.d-18,S(i)


10 continue
!now calculate ionization stages (astronomical units)
n=z-n+1
sigma=sigma*1.d-18

!transform eV to Kayser
const=hh*clight/ev
const=1.d0/const
eth=eth*const

print*
print*,'K-shell data read'
print*,'max deviation Daltabuit & Cox vs. Kaastra & Mewe (thresholds) = ',err
if(err.gt.5.d-2) stop ' deviation too large!'
print*

aug_1_6=aug_1_6/10000.

return

end
!
!---------------------------------------------------------------------------
!
subroutine opacit_kshell(nd,xne,ilow,imax)
! calculates k-shell opacities (clumping included in occupation numbers),
! but not corrected
  
USE nlte_type

USE nlte_dim, ONLY: id_atoms, id_ndept

USE princesa_var, only: nat, labat, zeff

USE nlte_var, only: concon, imia, imaa, imianl, imaanl, enionnd, ifre, fre

USE nlte_app, only: natom, indexel, names1, jatom_full, &
              met_imin, met_imax, &
              abund, fjk, xmuee, summas

USE nlte_xrays, only: n_kedges, k_nmin, k_nmax, &
  z, nionk=>n, eth, sigma, s, name, opa_kshell

IMPLICIT NONE

integer(i4b), parameter :: kel=id_atoms, nd1=id_ndept  

integer(i4b), intent(in) :: nd
real(dp), intent(in) :: xne(nd)

integer(i4b), intent(in) ::  ilow(nd1,kel),imax(nd1,kel)  

real(dp), dimension(:), allocatable :: alpha_fre

real(dp), dimension(nd1) :: const, occ

integer(i4b) :: nk, zold, k, kex, iex, iz, j, ll, kk, kmin

integer(i4b), dimension(natom) :: iminbg, imaxbg
integer(i4b), dimension(kel) :: iminex, imaxex

real(dp) :: eth_min

!minimum and maximum ions for bg-elements
do k=1,natom
  if (jatom_full(k).ne.1) cycle
  iminbg(k)=minval(met_imin(k,:))
  imaxbg(k)=maxval(met_imax(k,:))+1  !accounting for highest ion with ground-state
enddo  

!minimum and maximum ions for explicit elements
do k=1,nat
  iminex(k)=minval(ilow(:,k))
  imaxex(k)=maxval(imax(:,k))+1  !accounting for highest ion with ground-state
  if(.not.concon) then
    if(iminex(k).ne.imia(k)) stop ' opacit_kshell: imia not consistent'
    if(imaxex(k)-1.ne.imaa(k)) stop ' opacit_kshell: imaa not consistent'
  else
    if(iminex(k).ne.imianl(k)) stop ' opacit_kshell: imianl not consistent'
    if(imaxex(k)-1.ne.imaanl(k)) stop ' opacit_kshell: imaanl not consistent'
  endif
enddo  

eth_min=1.d99
do k=1,n_kedges
  if (nionk(k).lt.k_nmin) cycle !below k_nmin (typically 3, set in nlte_xrays)
  if (nionk(k).gt.k_nmax) cycle !above k_nmax (typically 8, set in nlte_xrays)
  if (eth(k).gt.2.d7) cycle     !below 5 a (maximum energy in fre, set in frescal) 
  eth_min=min(eth_min,eth(k))
enddo

do kk=1,ifre
    if(eth_min.eq.fre(kk)) goto 5
enddo
stop ' min edge not found in opacit_kshell'


5 kmin=kk

if (allocated(opa_kshell)) then
   deallocate(opa_kshell)
   allocate(opa_kshell(nd1,kmin:ifre),alpha_fre(kmin:ifre))
else
   allocate(opa_kshell(nd1,kmin:ifre),alpha_fre(kmin:ifre))
endif
opa_kshell=0.d0

nk=0
zold=0
! only edges from ionization stage iii on
do k=1,n_kedges
  if (nionk(k).lt.k_nmin) cycle !below k_nmin (typically 3, set in nlte_xrays)
  if (nionk(k).gt.k_nmax) cycle !above k_nmax (typically 8, set in nlte_xrays)
  if (eth(k).gt.2.d7) cycle     !below 5 a (maximum energy in fre, set in frescal) 
  zold=z(k)

! different treatment of explicit and background element 

!bg element
  if(indexel(zold).eq.-1) then

     if(jatom_full(zold).ne.1) stop ' K-shell absorption from not selected bg-element!'
     if(name(zold).ne.names1(zold)) stop ' opacit_kshell: names not consistent (1)!'
!     print*,k,z(k),nionk(k),iminbg(zold),imaxbg(zold)
     if(nionk(k).lt.iminbg(zold) .or. nionk(k).gt.imaxbg(zold)) cycle     
     print*,' k-shell opacity from bg element', zold, name(zold), nionk(k)

     const=xmuee*xne/summas*abund(zold) !depth dependent
     occ=0.d0
     do ll=1,nd
       j=nionk(k)
       if(j.lt.met_imin(zold,ll) .or. j.gt.met_imax(zold,ll)+1) cycle
       occ(ll)=const(ll)*fjk(zold,j,ll)
     enddo
       
     !explicit element
  else
     kex=indexel(zold)
     if(name(zold).ne.labat(kex)) stop ' opacit_kshell: names not consistent (2)!'
! account for zeff
     iz=int(zeff(kex))
     if(float(iz)-zeff(kex).gt.1.d-12) stop ' opacit_kshell: something wrong with int(zeff)!'
     iex=nionk(k)-iz ! this is the index w.r.t. ionization state of expl. elements
     if(iex.lt.iminex(kex) .or. iex.gt.imaxex(kex)) cycle     
     print*,' k-shell opacity from exp. elem.', kex, labat(kex), nionk(k),'(',iex,')'
     occ=0.d0
     do ll=1,nd
       if(iex.lt.ilow(ll,kex) .or. iex.gt.imax(ll,kex)+1) cycle
       occ(ll)=enionnd(kex,iex,ll)
     enddo
  endif       

  do kk=1,ifre
    if(eth(k).eq.fre(kk)) goto 10
  enddo
  stop ' edge not found in opacit_kshell'

10 continue
  
  alpha_fre(kk:ifre)=sigma(k)*(eth(k)/fre(kk:ifre))**s(k) 
  do ll=1,nd
     if(occ(ll).ne.0.d0) &
       opa_kshell(ll,kk:ifre)=opa_kshell(ll,kk:ifre)+occ(ll)*alpha_fre(kk:ifre)
  enddo
     
  nk=nk+1
enddo

print*
print*,nk,' k-shell opacities calculated'

return
end
!
!-----------------------------------------------------------------------
!
subroutine intermepho_auger(x1,itrans,ll)
!
!calculates photo-integrals for k-shell absorption required for Auger-ionization
!
USE nlte_type
USE nlte_dim
USE fund_const
use nlte_var, ONLY: fre,wfre,ifre,xj
USE nlte_xrays, ONLY: n_kedges,eth,sigma,s

IMPLICIT NONE
integer(i4b), intent(in) :: itrans, ll
real(dp), intent(out) :: x1

integer(i4b) :: k,kk

real(dp) :: wf, quant, xint, const

if(itrans.le.0 .or. itrans.gt.n_kedges) stop ' intermepho_auger: error in itrans'

do kk=1,ifre
   if(eth(itrans).eq.fre(kk)) goto 10
enddo
stop ' intermepho_auger: edge not found'

10 continue
  
xint=0.

do k=kk,ifre
!
!---    note: first integration weight has to be modified
!
if(k.eq.kk) then
  wf = wfre(kk) - .5d0*clight* (fre(kk)-fre(kk-1))
else
  wf = wfre(k)  
endif
quant = sigma(itrans)*(eth(itrans)/fre(k))**s(itrans)/fre(k)*xj(ll,k)
xint = xint + wf * quant

enddo

const = 4.d0*pi/(clight*hh)  
x1 = const*xint  

if(x1.le.0.) stop ' INTERMEPHO_AUGER!'

return  
end
