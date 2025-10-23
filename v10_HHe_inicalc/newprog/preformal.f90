MODULE preformal_var
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!for STAFIL and LINETAB 
!
character*(*), parameter :: fpath='../inicalc/DATA/'
!----------------------------------------------------------------------
!comcha
!
! ID_LLEVS+2, to allow for one additional lower and upper level used
!             within approximate treatment (NCOMP = 1!)
CHARACTER*6, DIMENSION(ID_ATOMS) :: LABAT 
CHARACTER*6, DIMENSION(ID_LLEVS+2) :: LABL 
!----------------------------------------------------------------------
!comnui
!
INTEGER(I4B) ::  NALIN,NAT,NL,NLEVH21 !level no of H21 for AT
INTEGER(I4B), DIMENSION(ID_LLEVS+2) :: LE

LOGICAL AT
!----------------------------------------------------------------------
!comnur
!
REAL(DP), DIMENSION(ID_ATOMS) :: WEIGHT,ZEFF
REAL(DP), DIMENSION(ID_LLEVS+2) :: FL,GL,ZL
!----------------------------------------------------------------------
!charge,griem
!
INTEGER(I4B) ::  NS  
REAL(DP) ::  AS,ODOP,PS,ZZ  
REAL(DP), DIMENSION(69) :: SS,SX
!----------------------------------------------------------------------
!inout
!
INTEGER(I4B) ::  IU,IOU

END MODULE preformal_var
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccc                                                                cccc
!ccc      preparation of formal solucion                            cccc
!ccc                                                                cccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
SUBROUTINE PREFORMAL(FILE,NCOMP,LEVEL,LEVEU,LINENO,NSTARK,VTURB,YHE, &
                     BALMER_LEMKE, PB_LEMKE)
!
USE nlte_type
USE nlte_dim
USE fund_const, ONLY: PI
USE preformal_var, ONLY: FPATH,NAT,NL,FL,GL,ZL,WEIGHT,ZEFF,IU,IOU,AT,NLEVH21
USE ffr_error
IMPLICIT NONE
!
!------ program run before the formal solution is performed
!------ it uses two files as input:
!
!            i) ATOM_FILE: contains the files of atomic data and stark
!                          profiles
!
!------ and two files as output:
!
!            i) IX.DAT: it will be used by the formal solution as input
!           ii) STARK.BIN: binary file containing the stark profile
!                          of the transition(s) considered
!
!  stark broadening options:
!  0: pure doppler
!  1: stark (from tables or griem)
!  for NSTARK = 0 or 1
!     if lineno =0, standard approach
!     if lineno =1, approximate treatment of transitions between
!     high lying levels, and data from LINES.dat
!
!  2: voigt, with gammal and gammac from LINES.dat
!  3: voigt, with gammal and gammac approximated
!    *works only for ncomp=1*: 
!      lineno   =0: gammal and gammac calculated
!      lineno ne 0: gammal calculated, gammac=0.
!
!NOTE: if GammaL and GammaC not available, can be approximated quite well
!by using one line of the multiplet with lineno=0 and stark=3
!(checked, e.g., by comparing Si line values with given data)
!Works particularly good for lines with low lying lower level.

!
!------ enrique santolaya, november 1993, version 1.0
!
!       version 1.1 april 30th 1997 (j.puls)
!                  to be used as subroutine in 'formalsol'
!
!
!       version 1.2 july 4th 1997
!                  ix.dat contains now gf-value,
!                  broadening option "2" for voigt profiles incorporated
!
!
!       version 3.0.0 jan 21st 1999
!                  changed to f90, inclusion of modification by
!                  christian wiethaus: wavelenghts now for air!
!
!       version 3.1 march 11th 1999
!                   more f90 changes, obsolete features removed
!
!
!       version 4.0 april 16th 1999
!                   ISO-lines can now be treated, 
!                   vturb included by additional folding of tabulated data 
!
!       version 4.1 may 20th 1999
!                   GAMMAL and GAMMAC approximated for lines were no other
!                   information given (corresponds to NSTARK = 3)
!
!       version 4.1.1 june 16th 1999 
!                   error corrected (treatment of lines with two components
!		    with the same parent levels). This error was corrected
!		    by J. Puls, but new versions (4.2 and 5.0) have this 
!		    error. I have corrected it again for Carrie's version
!		    (Miguel A. Urbaneja, Bremen, january 3rd 2002)
!
!       version 4.2 june 28th 2000 inconsistency in griem (range = 6, &
!                   xmax in err = 5) removed 
!
!       version 5.0 feb 7th 2001 file paths to STAFIL and LINETAB
!                   changed via parameter fpath
!
!----lineno  = 0 : line from detail input, assuming that levels are
!                   not packed
!----lineno ne 0 : line for lines.dat, levels are packed
!
!       version 5.1 april 2004 new broadening routine for "iso1" levels
!                   according to the data by Dimitrijevic & Sahal-Brechot.
!                   So far, proton/HeII contribution only approximate
!                   from YHe and HeI input values (assumed to be constant
!                   throughout the atmosphere)
!
!       version 5.2 may 2006: stop statements in transic1 removed; allows
!                   to have lines in LINES.DAT from levels which are NOT
!                   present in the model
!
!       version 6.0 july 2009: generalization to arbitrary NCOMP 
!                   old bug removed: calculation with pure Doppler-broadening
!                   and more than one component possible
!
!       version 6.1 april 2012: approximate treatment (AT) of transitions
!                   between high lying levels (mm and submm range), for ALMA
!                   coding by NSTARK = 0 or 1 and LINENO=1 (NCOMP=1)
!
!       version 6.2 may 2013: Stark-broadening from Lemke (1997) will be used
!                   for H_Balmer lines if BALMER_LEMKE = .true.
!
!       version 6.2.1 february 2015: in function HALLCL, correction for
!                   atmospheric refraction only down to 2000 AA:
!                   Additional IF statement to avoid that UV lines are
!                   corrected for air.
!                   This had to be done because lines as HEII1640 and
!                   Lyman_beta (not present in LINES.DAT) were corrected
!                   while specific non-H/He lines such as NV use wavelengths
!                   from LINES.DAT as given (here: in vaccum).
!
!       version 7.0 july 2016: routines RDATOM, HALLCL, and IDENT renamed,
!                   to RDATOM1, HALLCL1, and IDENT1,
!                   because of overlap with PRINCESA, and function IGENIO
!                   included 
!                   subr. TRANSIC1: QCL = .FALSE. inserted
!                                         (old bug, found in March 2015)
!
!       version 7.0.1 july 2016:
!                   consistency with version 7.0 and 7.0.1 from CMF-path
!
!       version 7.0.2 oct 2020: compatible with gfortran
!
!       version 7.0.3 nov 2022: potential Lemke broadening for
!                   Paschen/Brackett lines (hydrogen) included
!
!       initializing data
!     .. scalar arguments ..
INTEGER(I4B) ::  NCOMP  
REAL(DP) :: VTURB,YHE
LOGICAL :: BALMER_LEMKE, PB_LEMKE
!     ..
!     .. array arguments ..
INTEGER(I4B), DIMENSION(NCOMP) ::  LINENO, NSTARK  
CHARACTER, DIMENSION(NCOMP) :: LEVEL*6,LEVEU*6  
!     ..
!     .. local scalars ..
REAL(DP) ::  AUX  
INTEGER(I4B) ::  I,IAUX,NS,NSUM
LOGICAL RTABLE
CHARACTER DC*6,KEY*6,FICHERO*32,LINETAB*32,STAFIL*32,RET*4,FILE*60
!     ..
!     .. local arrays ..
REAL(DP), ALLOCATABLE, DIMENSION(:) ::  FLU,GAMMAC,GAMMAL,XLAMB,SUMLO,SUMUP  
INTEGER(I4B), ALLOCATABLE, DIMENSION(:) ::  NATOM,NLEVL,NLEVU,INDEX  
LOGICAL, ALLOCATABLE, DIMENSION(:) :: LOW  
!     ..
!     .. external subroutines ..
EXTERNAL FFRACC,PRESTARK,RDATOM1,TRANSIC1,TRANSIC2,TRANSIC_H_AT  
!     ..
!     .. intrinsic functions ..
INTRINSIC MIN  
!     ..

ALLOCATE(FLU(NCOMP),GAMMAC(NCOMP),GAMMAL(NCOMP),XLAMB(NCOMP),SUMLO(NCOMP),SUMUP(NCOMP))
ALLOCATE(NATOM(NCOMP),NLEVL(NCOMP),NLEVU(NCOMP),INDEX(NCOMP))
ALLOCATE(LOW(NCOMP))

RTABLE = .FALSE.  
AT = .FALSE.  
LOW = .FALSE.  

NLEVH21=0

NL=0
NAT=0
WEIGHT=0.
ZEFF=0.
FL=0.
GL=0.
ZL=0.

NATOM=0
NLEVL=0
NLEVU=0
XLAMB=0.
FLU=0.
GAMMAL=0.
GAMMAC=0.
SUMLO=0.
SUMUP=0.

DO I =1,NCOMP
     IF (NSTARK(I).EQ.2 .OR. LINENO(I).NE.0) RTABLE = .TRUE.  ! works also for nstark=4
     IF (NSTARK(I).EQ.3 .AND. LINENO(I).NE.0) STOP ' NSTARK = 3 AND LINENO NE 0'
     IF (NSTARK(I).LE.1 .AND. LINENO(I).EQ.1) AT=.TRUE. 
END DO  

IU = 37  
IOU = 38  
DC = ':T'  

OPEN (1,FILE='ATOM_FILE',STATUS='OLD')  
REWIND 1  
READ (1,FMT='(A)') FICHERO  
READ (1,FMT='(A)') STAFIL  
IF (RTABLE) READ (1,FMT='(A)') LINETAB  
CLOSE (1)  

OPEN (UNIT=IU,FILE=FICHERO,FORM='FORMATTED',STATUS='OLD')  
OPEN (UNIT=IOU,FILE='control.dat',STATUS='UNKNOWN')  
!
!       rewind!!!
!
IU = -37  

CALL FFRACC(IU,RET)
IF (ON_ERROR(RET))  GOTO 60

RLOOP: DO  

CALL RDATOM1(RET)  
IF (RET.NE.'RET0'.AND.RET.NE.'RET3')  GOTO 70
IF (RET.EQ.'RET3') GOTO 50

IF(.NOT.AT) THEN
! standard path
   CALL TRANSIC1(LEVEL,LEVEU,NCOMP,LINENO,NLEVL,NLEVU,NATOM,XLAMB,FLU,LOW, &
&             SUMLO,SUMUP,NSTARK,GAMMAC,RET)
   IF (ON_ERROR(RET))  GOTO 70

ENDIF
END DO RLOOP 
!
!       normal exit: end of file
!
   50 CONTINUE  

WRITE (*,FMT='(A)') ' DETAIL FILE IS OVER. SUCCESSFUL!! '  
CLOSE (IU)  

!  now, all contributions to SUMLO, SUMUP are calculated, &
!  hence we can calculate GAMMAL.
!  the constant below corresponds to 8 pi^2 e^2/(m_e c), if lambda in A

IF(.NOT.AT) THEN
  DO I=1,NCOMP
  IF(NSTARK(I).EQ.3) THEN
    GAMMAL(I)=6.6702082D15*(SUMLO(I)+SUMUP(I))/(4.*PI)
    PRINT*,GAMMAL(I),GAMMAC(I)
  ENDIF
  END DO  
ELSE
! approximate treatment
  CALL TRANSIC_H_AT(LEVEL,LEVEU,NCOMP,LINENO,NLEVL,NLEVU,NATOM, &
&                   XLAMB,FLU,NSTARK)
ENDIF 

IF (RTABLE) THEN  
     OPEN (UNIT=IU,FILE=fpath//LINETAB,FORM='FORMATTED',STATUS='OLD')  
!
!       rewind!!!
!
     IU = -37  
     CALL FFRACC(IU,RET)  
     IF (ON_ERROR(RET))  GOTO 60
!
!       note here: flu corresponds to gf/sum(g), see above
!
     CALL TRANSIC2(LEVEL,LEVEU,NCOMP,NSTARK,LINENO,NLEVL,NLEVU, &
&                  NATOM,XLAMB,FLU,GAMMAL,GAMMAC,RET)
     IF (ON_ERROR(RET))  GOTO 70

     WRITE (*,FMT='(A)') ' LINE FILE IS OVER. SUCCESSFUL!! '  
     CLOSE (IU)  
END IF  

GO TO 90  
!
!       error in input unit exit
!
   60 CONTINUE  
WRITE (*,FMT='(A)') ' ERROR IN INPUT UNIT NUMBER '  
STOP

!       error conditions
70 SELECT CASE(RET)
CASE ('RET1') !       end of file "no esperado" exit  
  WRITE (*,FMT='(A)') ' EOF NOT EXPECTED. THERE IS SOMETHING WRONG! '  
  STOP
CASE ('RET2') !       error in ffr subroutines exit  
  WRITE (*,FMT='(A)') ' ERROR IN FFR SUBROUTINES '  
  STOP
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN PREFORMAL'
END SELECT

   90 CONTINUE  
CLOSE (IOU)  

DO I = 1,NCOMP  
    IF (NLEVL(I).EQ.0 .OR. NLEVU(I).EQ.0) STOP 'LEVELS NOT FOUND'  
    IF (XLAMB(I).LE.0.D0) STOP 'ERROR IN LAMBDA'  
    IF (FLU(I).LE.0.D0) STOP 'ERROR IN FLU'  
    IF (NATOM(I).EQ.0) STOP 'ERROR IN ATOM'  
END DO  
!
!---    sorting for wavelengths (blue component at first)
!
CALL INDEXX(NCOMP,XLAMB,INDEX)
!
!---    interchange of components
!
XLAMB=XLAMB(INDEX)
FLU=FLU(INDEX)
GAMMAL=GAMMAL(INDEX)
GAMMAC=GAMMAC(INDEX)
NLEVL=NLEVL(INDEX)
NLEVU=NLEVU(INDEX)
NATOM=NATOM(INDEX)
NSTARK=NSTARK(INDEX)
!
!---    from here on, comp 1 = blue component
!
!
!------ nstark=0 : doppler
!------ nstark=1 : stark
!------ nstark=2 : voigt with data from LINETAB (LINES.DAT)
!------ nstark=3 : voigt with approx. for GAMMAL/GAMMAC (see above comments)
!

NSUM=0
DO I = 1,NCOMP  
     IF (NSTARK(I).EQ.1) NSUM = NSUM + 1  
END DO  
!
!------ nsum=0 : no stark;
!------ nsum=1 : only one comp. stark (nsum, precs.)
!------ nsum=n : n comps. stark
PRINT *,'NUMBER OF STARK-COMPONENTS',NSUM  

IF (NSUM.NE.0) &
& CALL PRESTARK(FILE,NSUM,STAFIL,NCOMP,NLEVL,NLEVU,XLAMB,FLU,NSTARK,VTURB,YHE, &
                BALMER_LEMKE,PB_LEMKE)
!
!------ output file ix.dat
!
OPEN (1,FILE=TRIM(FILE)//'/IX.DAT',STATUS='UNKNOWN',FORM='FORMATTED')  
REWIND 1  

WRITE (1,FMT=*) NCOMP  
DO I = 1,NCOMP  
     WRITE (1,FMT=*) XLAMB(I)  
     WRITE (1,FMT=*) GL(NLEVL(I)),GL(NLEVU(I))  
     WRITE (1,FMT=*) GL(NLEVL(I))*FLU(I)  
     WRITE (1,FMT=*) NLEVL(I),NLEVU(I)  
     WRITE (1,FMT=*) WEIGHT(NATOM(I))  
     NS = NSTARK(I)  
     WRITE (1,FMT=*) NS  
     IF (NS.EQ.2.OR.NS.EQ.3) WRITE (1,FMT=*) GAMMAL(I),GAMMAC(I)  
END DO  
CLOSE (1)  

DEALLOCATE(FLU,GAMMAC,GAMMAL,XLAMB,SUMLO,SUMUP)
DEALLOCATE(NATOM,NLEVL,NLEVU,INDEX,LOW)

RETURN  

END
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
SUBROUTINE RDATOM1(RETCHAR)  
!
USE nlte_type
USE nlte_dim
USE preformal_var, ONLY: LABAT,LABL,NAT,NL,LE,FL,GL,ZL,WEIGHT,ZEFF
USE ffr_error
IMPLICIT NONE
!
!     .. local scalars ..
REAL(DP) ::  REALO,ZEF  
INTEGER(I4B) ::  INTEG,NC  
CHARACTER LABEL*4,KEY*6,RETCHAR*4,RET*4  
!     ..
!     .. external subroutines ..
EXTERNAL FFRKEY,FFRLOC,FFRNUM  
!     ..

LABEL = 'ATOM'  
CALL FFRLOC(LABEL,RET)  
SELECT CASE(RET)
CASE('RET1')
  GOTO 50
CASE('RET2')
  GOTO 60
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN IN RDATOM1'
END SELECT

NAT = NAT + 1  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 60
LABAT(NAT) = KEY  

CALL FFRNUM(REALO,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 60
ZEFF(NAT)=REALO

CALL FFRNUM(REALO,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 60
WEIGHT(NAT) = REALO  

CALL FFRNUM(REALO,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 60

ZEF = ZEFF(NAT) - 1.D0  

   10 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 60

!------------- l  levels
IF (KEY.EQ.'L') THEN  
     ZEF=ZEF+1.D0
   20      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 60
     IF (KEY.EQ.'0') GO TO 10  
     NL = NL + 1  
     LABL(NL) = KEY  
     ZL(NL)=ZEF
     
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60
     GL(NL) = REALO
     
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60
     FL(NL) = REALO  
     LE(NL) = NAT  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     GO TO 20  

!----------------- x  levels
ELSE IF (KEY.EQ.'X') THEN  
   30      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 60
     
     IF (KEY.EQ.'0') GO TO 10  

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     GO TO 30  

!-------------- s  levels
ELSE IF (KEY.EQ.'S') THEN  
   40      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     IF (KEY.EQ.'0') GO TO 10  

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 60

     GO TO 40  

!--------------------  k  levels
ELSE IF (KEY.EQ.'K') THEN  
     NL = NL + 1  
     ZEF=ZEF+1
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 60
     LABL(NL) = KEY  

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 60
     GL(NL) = REALO  

     CALL FFRNUM(REALO,INTEG,RET)  
     FL(NL) = REALO  
     ZL(NL)=ZEF

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 60

END IF  

!       the atom is over.
RETCHAR='RET0'
RETURN  
!       no hay mas atomos que leer. ya se ha leido todo el fichero. es
!       cuando acaba el programa principal (prince)
   50 CONTINUE  

RETCHAR='RET3'
RETURN 

! error handling
60 SELECT CASE(RET)
CASE('RET1') !       end of file exit 
  WRITE (*,FMT='(A)') ' END OF FILE '  
  RETCHAR='RET1'
  RETURN
CASE('RET2') !       error exit 
  WRITE (*,FMT='(A)') 'ERROR IN FFR SUBROUTINES '  
  RETCHAR='RET2'
  RETURN
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN RDATOM1'
END SELECT

END
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
SUBROUTINE TRANSIC1(LEVEL,LEVEU,NCOMP,LINENO,NLEVL,NLEVU,NATOM, &
&   XLAMB,FLU,LOW,SUMLO,SUMUP,NSTARK,GAMMAC,RETCHAR)
!
USE nlte_type
USE nlte_dim
USE fund_const, ONLY: CLIGHT
USE preformal_var, ONLY: GL,LABL,NL,LE,IU
USE ffr_error
IMPLICIT NONE
!
!       subrutina que lee los datos referentes a las transiciones, los
!       almacena y escribe (para control) en un fichero de salida.
!       espero poner suficientes explicaciones a lo largo de la
!       subrutina, de modo que este todo lo mas claro posible.
!
!     .. scalar arguments ..
INTEGER(I4B) ::  NCOMP  
!     ..
!     .. array arguments ..
REAL(DP) ::  FLU(NCOMP),XLAMB(NCOMP),SUMLO(NCOMP),SUMUP(NCOMP),GAMMAC(NCOMP) 
INTEGER(I4B) ::  LINENO(NCOMP),NATOM(NCOMP),NLEVL(NCOMP),NLEVU(NCOMP), &
&                NSTARK(NCOMP)
LOGICAL LOW(NCOMP)  
CHARACTER LEVEL(NCOMP)*6,LEVEU(NCOMP)*6  
!     ..
!     .. local scalars ..
REAL(DP) ::  REALO,TL,WLC  
INTEGER(I4B) ::  I,IFORM,INTEG,IQI,LABL1,LABU1,M,NC, &
&        NDATM,NDATOS,NLONG,NN,NNU
LOGICAL QCL,QRT  
CHARACTER KEY*6,RETCHAR*4,RET*4  
!     ..
!     .. local arrays ..
CHARACTER TABLA(8)*6  
!     ..
!     .. external functions ..
REAL(DP) ::  HALLCL1,CALCGAM  
EXTERNAL HALLCL1,CALCGAM  
!     ..
!     .. external subroutines ..
EXTERNAL FFRKEY,FFRNUM,IDENT1  
!     ..
!     .. data statements ..
DATA TABLA/'RBB','CBB','RBX','CBX','CBS','CBF','RBF','RFF'/  
!     ..

   10 CONTINUE  
!JO March 2015: this is a very important statement, and had been forgotten
!in previous versions; if left out, certain reading errors can occur.
   QCL = .FALSE.
   
CALL FFRKEY(KEY,NC,RET)  

IF (ON_ERROR(RET))  GOTO 250

IF (KEY.EQ.'TY') THEN  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET))  GOTO 250

     CALL IDENT1(KEY,TABLA,8,M)  
     CALL FFRNUM(REALO,IFORM,RET)  
     IF (ON_ERROR(RET))  GOTO 250

     CALL FFRNUM(REALO,NDATOS,RET)  
     IF (ON_ERROR(RET))  GOTO 250
!          son los numeros de formula y de datos, respectivamente
     NDATM = NDATOS  

     GO TO 10  

!       muestreo en anchuras doppler. el muestreo queda en dl.
ELSE IF (KEY.EQ.'DB') THEN  
     CALL FFRNUM(REALO,NLONG,RET)  
     IF (ON_ERROR(RET))  GOTO 250
     DO I = 1,NLONG  
          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 250
     END DO

     GO TO 10  

!       muestreo en angstroms
ELSE IF (KEY.EQ.'DL') THEN  
     CALL FFRNUM(REALO,NLONG,RET)  
     IF (ON_ERROR(RET))  GOTO 250
     DO I = 1,NLONG  
          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 250
     END DO

     GO TO 10  

!       temperatura de linea
ELSE IF (KEY.EQ.'TL') THEN  
     CALL FFRNUM(TL,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 250

     GO TO 10  

!       longitud central de la linea
ELSE IF (KEY.EQ.'CL') THEN  
     QCL = .TRUE.  
     GO TO 10  

!       tipo de transicion
ELSE  
     CALL IDENT1(KEY,TABLA,8,M)  
     IF (M.EQ.0) THEN  
          GO TO 270  
     ELSE  
          QRT = .FALSE.  
          IF ((M.EQ.1) .OR. (M.EQ.3)) QRT = .TRUE.  
          GO TO (40,40,130,130,160,160,190,220) M  
!  goto calculado, para que haga una cosa u otra segun el tipo de trans.


     END IF  

END IF  

!-------------transiciones rbb o cbb ------------
   40 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

   50 CONTINUE  
DO I = 1,NCOMP  
     LOW(I) = .FALSE.  
     IF (KEY.EQ.LEVEL(I) .AND. LINENO(I).EQ.0) LOW(I) = .TRUE.  
END DO  

DO IQI = 1,NL  
     IF (LABL(IQI).EQ.KEY) THEN  
          NN = IQI  
          GO TO 80  
     END IF  
END DO  

STOP 'ERROR IN TRANSIC1 - LABEL L NOT FOUND, BB'  
   80 CONTINUE  

LABL1 = NN

CALL FFRKEY(KEY,NC,RET)  
!print*,labl(labl1),key

IF (ON_ERROR(RET))  GOTO 250

DO IQI = 1,NL
     IF (LABL(IQI).EQ.KEY) THEN  
          NNU = IQI  
          GO TO 100  
     END IF  
END DO  
STOP 'ERROR IN TRANSIC1 - LABEL U NOT FOUND, BB'  
  100 CONTINUE  

LABU1 = NNU  
!print*,labl(labl1),' ',labl(labu1),qrt,qcl

!       transicion radiativa
IF (QRT) THEN  
     IF (QCL) THEN  
          CALL FFRNUM(WLC,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 250
     ELSE  
          WLC = HALLCL1(LABL1,LABU1)  
     END IF  

     DO I = 1,NCOMP  

!  sumlo and sumup contain sum of (gi fij/gj/lambda^2) from all transitions
!  downwards from lower and upper level. Needed to calculate Sum(Aji) for
!  GAMMAL!
!JO changed July 2016, following CMF-path (problem detected Sept. 2015)
!   new version might be still erroneous, partic. for NSTARK=2
!  -according to header, this works only for NSTARK=3 and NCOMP=1
!  -as far as I can see, this works only if in the component list
!   any upper and lower level does not appear more than once
!   (which is usually the case for individual lines complexes).
!  (-does no longer work for 'range' calculations)     
!  -in any case, GAMMAL only calculated for NSTARK=3
!  -thus, seperate reading of oscillator strength
!  (question: what about oscillator strength if NSTARK = 2 in old version???) 
!       IF(NSTARK(I).NE.2 .AND. LINENO(I).EQ.0) THEN !old version
       IF(NSTARK(I).EQ.3 .AND. LINENO(I).EQ.0) THEN !new version (July. 2016) 
          IF (LABL(NNU).EQ.LEVEL(I)) THEN
            CALL FFRNUM(REALO,INTEG,RET)  
            IF (ON_ERROR(RET))  GOTO 250
            NDATM = NDATOS - 1  
            SUMLO(I)=SUMLO(I)+GL(NN)*REALO/(GL(NNU)*WLC*WLC)
!            PRINT*,'LO ',I,' ',LABL(NN),LABL(NNU),GL(NN),GL(NNU),WLC,REALO
          ELSE IF(LABL(NNU).EQ.LEVEU(I)) THEN
            CALL FFRNUM(REALO,INTEG,RET)  
            IF (ON_ERROR(RET))  GOTO 250
            NDATM = NDATOS - 1  
            SUMUP(I)=SUMUP(I)+GL(NN)*REALO/(GL(NNU)*WLC*WLC)
!            PRINT*,'UP ',I,' ',LABL(NN),LABL(NNU),GL(NN),GL(NNU),WLC,REALO
          ENDIF   
! new else block, allowing to read osc. strength
       ELSE
          IF (KEY.EQ.LEVEU(I) .AND. LOW(I)) THEN  
            CALL FFRNUM(REALO,INTEG,RET)  
            IF (ON_ERROR(RET))  GOTO 250
            NDATM = NDATOS - 1  
          ENDIF          
       ENDIF


       IF (KEY.EQ.LEVEU(I) .AND. LOW(I)) THEN  
               XLAMB(I) = WLC  
!  osc. strength already read 
               FLU(I) = REALO
               NLEVL(I) = LABL1  
               NLEVU(I) = LABU1  
               NATOM(I) = LE(LABL1)  
               IF(NSTARK(I).EQ.3) GAMMAC(I)=CALCGAM(LABL1,LABU1,NATOM(I))
       END IF  
     END DO

END IF  

!       lectura de los datos
DO I = 1,NDATM  
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 250
END DO  

NDATM = NDATOS  

!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

IF (KEY.EQ.'0') THEN  
     QCL = .FALSE.  
     GO TO 10  
END IF  

GO TO 50  

!--------------- transiciones rbx o cbx ------------
  130 CONTINUE  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

CALL FFRKEY(KEY,NC,RET)
IF (ON_ERROR(RET))  GOTO 250
  140 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

!       transicion radiativa
IF (QRT) THEN  
     IF (QCL) THEN  
          CALL FFRNUM(WLC,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 250
     END IF  
END IF  

!       lectura de datos
DO I = 1,NDATOS  
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 250
END DO  

!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

IF (KEY.EQ.'0') THEN  
     QCL = .FALSE.  
     GO TO 10  
END IF  

GO TO 140  

!------------- transiciones  cbs o cbf ------------------
  160 CONTINUE  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

  170 CONTINUE  

!       lectura de datos
DO I = 1,NDATOS  
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 250
END DO  

!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

IF (KEY.EQ.'0') GO TO 10  

GO TO 170  

!------------------ transicion  rbf -------------------
  190 CONTINUE  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

  200 CONTINUE  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

!       leer datos
DO I = 1,NDATOS  
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 250
END DO  

!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

IF (KEY.EQ.'0') GO TO 10  

GO TO 200  

!----------------- transiciones rff ------------------------
  220 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 250

  230 CONTINUE  

!       lectura de datos
DO I = 1,NDATOS  
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 250
END DO  

!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)
IF (ON_ERROR(RET))  GOTO 250

IF (KEY.EQ.'0') GO TO 10  

GO TO 230  

! error handling
250 SELECT CASE(RET)
CASE('RET1') !       end of file exit 
  WRITE (*,FMT='(A)') ' END OF FILE '  
  RETCHAR='RET1'
  RETURN
CASE('RET2') !       error exit 
  WRITE (*,FMT='(A)') 'ERROR IN FFR SUBROUTINES '  
  RETCHAR='RET2'
  RETURN
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN TRANSIC1'
END SELECT

  270 CONTINUE  
BACKSPACE IU  

RETURN  

END
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
SUBROUTINE TRANSIC2(LEVEL,LEVEU,NCOMP,NSTARK,LINENO,NLEVL,NLEVU, &
&                    NATOM,XLAMB,FLU,GAMMAL,GAMMAC,RETCHAR)
!
USE nlte_type
USE nlte_dim
USE fund_const, ONLY: CLIGHT
USE preformal_var, ONLY: LABL,NL,LE,GL,AT
USE ffr_error
IMPLICIT NONE
!
!       finds line-transitions from file lines.dat
!
!     .. scalar arguments ..
INTEGER(I4B) ::  NCOMP  
!     ..
!     .. array arguments ..
REAL(DP) ::  FLU(NCOMP),GAMMAC(NCOMP),GAMMAL(NCOMP),XLAMB(NCOMP)
INTEGER(I4B) ::  LINENO(NCOMP),NATOM(NCOMP),NLEVL(NCOMP),NLEVU(NCOMP), &
&        NSTARK(NCOMP)
CHARACTER LEVEL(NCOMP)*6,LEVEU(NCOMP)*6,RETCHAR*4,RET*4  
!     ..
!     .. local scalars ..
REAL(DP) ::  REALO,SUMGLO  
INTEGER(I4B) ::  I,INTEG,IQI,LABL1,LABU1,NC,NDATOS,NLINE, &
&        NN,NNU,IQISTART
CHARACTER KEY*6  
!     ..
!     .. local arrays ..
LOGICAL LOW(NCOMP)  
!     ..
!     .. external subroutines ..
EXTERNAL FFRKEY,FFRNUM  
!     ..

   10 CONTINUE  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 110
IF (KEY.NE.'CL') GO TO 120  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 110
IF (KEY.NE.'TY') GO TO 120  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 110
IF (KEY.NE.'RBB') GO TO 120  

CALL FFRNUM(REALO,INTEG,RET)  
IF (ON_ERROR(RET))  GOTO 110
IF (INTEG.NE.2) GO TO 120

CALL FFRNUM(REALO,INTEG,RET)  
IF (ON_ERROR(RET))  GOTO 110
NDATOS = INTEG  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 110
IF (KEY.NE.'RBB') GO TO 120  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 110

   20 CONTINUE  
DO I = 1,NCOMP  
     LOW(I) = .FALSE.  
     IF (KEY.EQ.LEVEL(I) .AND. LINENO(I).NE.0) LOW(I) = .TRUE.  
END DO  

IQISTART=1
IF(AT) THEN
  IF (NL.NE.ID_LLEVS+2) STOP ' AT AND NL NE ID_LLEVS+2 IN TRANSIC2'
  IQISTART=ID_LLEVS+1
ENDIF

DO IQI = IQISTART,NL  
     IF (LABL(IQI).EQ.KEY) THEN  
          NN = IQI  
          GO TO 50  
     END IF  
END DO  
! stop statement removed: not ALL lines in LINES.dat must be present in level-list
!STOP 'ERROR IN TRANSIC2 - LABEL L NOT FOUND, BB'  
   50 CONTINUE  


LABL1 = NN  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 110

DO IQI = IQISTART,NL  
     IF (LABL(IQI).EQ.KEY) THEN  
          NNU = IQI  
          GO TO 70  
     END IF  
END DO  
! stop statement removed: not ALL lines in LINES.dat must be present in level-list
!STOP 'ERROR IN TRANSIC2 - LABEL U NOT FOUND, BB'  

   70 CONTINUE  
LABU1 = NNU  

CALL FFRNUM(REALO,INTEG,RET)  
IF (ON_ERROR(RET))  GOTO 110

NLINE = INTEG  

DO I = 1,NCOMP  
     IF (KEY.EQ.LEVEU(I) .AND. LOW(I) .AND. NLINE.EQ.LINENO(I)) THEN
          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 110
          XLAMB(I) = REALO  

          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 110
          FLU(I) = REALO  

          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 110
          GAMMAL(I) = REALO  

          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 110

          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 110

          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 110
          GAMMAC(I) = REALO  

          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 110
          SUMGLO = REALO  
          NLEVL(I) = LABL1  
          NLEVU(I) = LABU1  
          NATOM(I) = LE(LABL1)  

          GO TO 100  

     END IF  

END DO  

!       lectura de los datos
DO I = 1,NDATOS  
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 110
END DO  

!       leer nueva keyword
  100 CONTINUE  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET))  GOTO 110

IF (KEY.EQ.'0') GO TO 10  

GO TO 20  
!
! error handling
!
110 SELECT CASE(RET)
CASE('RET1') !       end of file exit 
  WRITE (*,FMT='(A)') ' END OF FILE '  
  RETCHAR='RET1'
  RETURN
CASE('RET2') !       error exit 
  WRITE (*,FMT='(A)') 'ERROR IN FFR SUBROUTINES '  
  RETCHAR='RET2'
  RETURN
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN TRANSIC2'
END SELECT

120  IF (KEY.EQ.'THEEND') THEN
    RETCHAR='RET0'
    RETURN
ELSE
    STOP ' WRONG END CONDITION IN TRANSIC2'
ENDIF

END
!-----------------------------------------------------------------------
!
FUNCTION CALCGAM(II1,II2,NATOM)  
!
USE nlte_type
USE nlte_dim
USE preformal_var, ONLY: FL,ZL,WEIGHT
USE fund_const, ONLY: CLIGHT,PI
IMPLICIT NONE
!
!  calculated GAMMAC in approximate weigh, using the relation between
!  neff and energy
!
!     ..
REAL(DP) :: CALCGAM
!
!     .. scalar arguments ..
INTEGER(I4B) ::  II1,II2,NATOM  
!     ..
REAL(DP) RYD,Z

Z=ZL(II1)
IF(Z.NE.ZL(II2)) STOP ' SOMETHING WRONG WITH NET CHARGE'
Z=Z+1.

RYD=109737.312*(1.+1./(1836.*WEIGHT(NATOM)))
CALCGAM=4.335D-7/(4.*PI)*Z*Z*(RYD*CLIGHT)**2 &
&       *(1./(FL(II2)*FL(II2))+1./(FL(II1)*FL(II1)))

RETURN  

END
!
!
!-----------------------------------------------------------------------
!
FUNCTION HALLCL1(II1,II2)  
!
USE nlte_type
USE nlte_dim
USE preformal_var, ONLY: FL
IMPLICIT NONE
!
!        calcula la longitud de onda central de una transicion rbb,
!        a partir de los datos atomicos de los niveles involucrados
!
!     ..
REAL(DP) :: HALLCL1
!
!     .. scalar arguments ..
INTEGER(I4B) ::  II1,II2  
!     ..
!     .. local scalars ..
REAL(DP) ::  CLIGHT,FC,FF1,FF2,XKW,XN  
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS  

!     ..
!     .. data statements ..
DATA CLIGHT/2.997925E18/  
!     ..
FF1 = FL(II1)  
FF2 = FL(II2)  
FC = ABS(FF1-FF2)  
IF (FC.EQ.0.D0) FC = 1.D0  
HALLCL1 = CLIGHT/FC  
!---    primitive way to change hei 4026 line (to avoid
!---    false wavelength and order of components)
IF (ABS(HALLCL1-4026.72472212120d0).LT.1.D-10) HALLCL1 = 4027.3d0  
!--- same for hei 4388
IF (ABS(HALLCL1-4388.82518552703d0).LT.1.D-10) HALLCL1 = 4389.16d0  
!
!-----AVOID THAT UV LINES HAVE THEIR WAVELENGTHS CONVERGED TO AIR
!-----CHANGE MADE BY Luiz Carneiro - 25/02/2015
IF (HALLCL1.LE.2000.d0) RETURN
!
!------CONVERSION TO WAVELENGTH IN AIR
!
XKW=1.D4/HALLCL1      
XN=1.D0+1.D-7*(643.28D0+294981.D0/(146.D0-XKW**2)+2554.D0/(41.D0-XKW**2))
HALLCL1=HALLCL1/XN
RETURN  

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE IDENT1(KEY,TABLE,NMAX,J)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!        compara el caracter key con los de la tabla 'table', y devuelve
!        en j el indice que corresponda a key en table, si esta, o 0 en
!        otro caso.
!     .. scalar arguments ..
INTEGER(I4B) ::  J,NMAX  
CHARACTER KEY*6  
!     ..
!     .. array arguments ..
CHARACTER TABLE(*)*6  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I  
!     ..
I = 1  
   10 CONTINUE  

IF (KEY.EQ.TABLE(I)) THEN  
     J = I  
     RETURN  
END IF  

I = I + 1  
IF (I.LE.NMAX) THEN  
     GO TO 10  
END IF  

J = 0  

RETURN  

END
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
SUBROUTINE PRESTARK(FILE,NSUM,STAFIL,NB,NLEVL,NLEVU,XLAMB,FLU,NSTARK,VTURB,YHE, &
                    BALMER_LEMKE, PB_LEMKE)  
!
USE nlte_type
USE nlte_dim
USE preformal_var, ONLY: FPATH,LABL,LE,WEIGHT,IU
USE ffr_error
IMPLICIT NONE
!
!---- preparation of stark solution. stark profiles are read and
!---- the desired transition(s) is(are) stored in file 'stark.bin'
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: MAXW=ID_MAXWW,MAXT=ID_MAXTT,MAXNE=ID_MAXNE, &
& MAXPS=ID_MAXPS,NDATAMAX=50
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  NB,NSUM  
REAL(DP) :: VTURB,YHE
LOGICAL :: BALMER_LEMKE, PB_LEMKE
CHARACTER STAFIL*32, FILE*60  
!     ..
!     .. array arguments ..
REAL(DP) ::  XLAMB(NB),FLU(NB)  
INTEGER(I4B) ::  NLEVL(NB),NLEVU(NB),NSTARK(NB),LL,LU  
!     ..
!     .. local scalars ..
REAL(DP) ::  AA,BB,EE,FAC,OSC,PESO,REALO,TT,XL0,WAVEAIR  
INTEGER(I4B) ::  I,IJK,INTEG,ISTARK,J,JE,JT,MFULL, &
&        NC,NDATA,NIN,NN,NN1,NN2,NNAT,NNN,NPS,NTSISO,NFSISO
LOGICAL QLOG
CHARACTER KEY*6,KEY2*6,RET*4,LLCHAR*6,LUCHAR*6
!     ..
!     .. local arrays ..
REAL(DP) ::  DATA(NDATAMAX),DDWS(35),SQ1(35),EEES(11), &
&            POLD(MAXPS),SQ(MAXW),TTTS(6),PROF(MAXW)
INTEGER(I4B) ::  NWSN 
LOGICAL Q  

REAL(DP), ALLOCATABLE, DIMENSION(:,:) :: DWS, ES, P, TS
INTEGER(I4B), ALLOCATABLE, DIMENSION(:) :: NES, NTS, NWS
LOGICAL, ALLOCATABLE, DIMENSION(:) :: DONE, QHALF

!     ..
!     .. external functions ..
INTEGER(I4B) ::  NPPAL  
EXTERNAL NPPAL  
!     ..
!     .. external subroutines ..
EXTERNAL FFRACC,FFRKEY,FFRLOC1,FFRNUM,GRIEM  
!     ..
!     .. intrinsic functions ..
INTRINSIC DBLE,LOG10  
!     ..
!     .. data statements ..

DATA TTTS,EEES/4.,4.301,4.602,4.903,5.204,5.505,10.5,11.5,12., &
&     12.5,13.,13.5,14.,14.5,15.,16.,17./
!     ..

IF(NSUM.NE.1.AND.NSUM.NE.2) STOP ' MORE THAN TWO STARK-COMPONENTS, CHECK!'

ALLOCATE(DWS(MAXW,NB),ES(MAXNE,NB),P(MAXPS,NB),TS(MAXT,NB))
ALLOCATE(NES(NB),NTS(NB),NWS(NB))
ALLOCATE(DONE(NB),QHALF(NB))

DONE=.TRUE.

OPEN (1,FILE=FPATH//STAFIL,STATUS='OLD',FORM='FORMATTED')  

! NEEDS TO BE CHANGED FOR MORE THAN TWO STARK LINES
DO I = 1,NB  
     IF(NSTARK(I).EQ.1) DONE(I) = .FALSE.  
END DO  

!BIG_LOOP

BIG_LOOP: DO NNN=1,NB

IF (NSTARK(NNN).NE.1) CYCLE

!
!     rewind!!!
!
IU = -1  
CALL FFRACC(IU,RET)  
SELECT CASE(RET)
CASE('RET1')
  GOTO 220
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN IN PRESTARK'
END SELECT

!---- read stafil

   20      CONTINUE  
!
!        note! 'line' has to be situated within first five characters of
!        input-line
!
     CALL FFRLOC1('LINE',RET)  
     SELECT CASE(RET)
     CASE('RET1')
       GOTO 150
     CASE('RET2')
       GOTO 230
     CASE('RET0')
       CONTINUE
     CASE DEFAULT
       STOP ' WRONG RETURN IN PRESTARK'
     END SELECT
       
     CALL FFRKEY(KEY,NC,RET)  
     SELECT CASE(RET)
     CASE('RET1')
       GOTO 230
     CASE('RET2')
       GOTO 230
     CASE('RET0')
       CONTINUE
     CASE DEFAULT
       STOP ' WRONG RETURN IN PRESTARK'
     END SELECT
!   
!----------------------------------------------------------------------
!
!  frequency grid for ISO lines
!     
     IF(KEY.EQ.'DL') THEN

     CALL FFRNUM(REALO,NC,RET)  
     IF (ON_ERROR(RET))  GOTO 230
     IF(NC.GT.MAXW) STOP ' NWS(ISO) > MAXW'
     NWS(NNN)=NC
     DO I = 1,NC
          CALL FFRNUM(REALO,INTEG,RET)  
          IF (ON_ERROR(RET))  GOTO 230
          DWS(I,NNN) = REALO
     END DO
     GOTO 20

     ENDIF
!   
!----------------------------------------------------------------------
!
!
!  treat ISO lines
!     
     IF (KEY.EQ.'ISO') THEN

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET))  GOTO 230
 
     CALL FFRKEY(KEY2,NC,RET)  
     IF (ON_ERROR(RET))  GOTO 230
     Q = (LABL(NLEVL(NNN)).EQ.KEY .AND. LABL(NLEVU(NNN)).EQ.KEY2)
     
     IF (.NOT. Q)  GOTO 20
     CALL FFRNUM(REALO,INTEG,RET)  
     WAVEAIR=REALO
     IF (ABS(WAVEAIR-XLAMB(NNN)).GT. 0.1) &
&     PRINT*,' CHECK WAVELENGTHS (ISO)! ',WAVEAIR,' ',XLAMB(NNN) 
!
!    take value provided
!     
     XLAMB(NNN)=WAVEAIR
     CALL FFRNUM(REALO,INTEG,RET)  
     OSC=REALO
     IF (ABS(LOG10(OSC/FLU(NNN))).GT. 0.1)&
&     PRINT*,' CHECK OSCILLATOR STRENGTH (ISO)! ',OSC,' ',FLU(NNN) 
!
!    take value provided
!     
     FLU(NNN)=OSC
!
!    read and write data in following format     
!----OSC,DLP,DENS,NTS,NFS,
!----TS(NTS),WS(NTS),DS(NTS),AS(NTS),
!----(DLF,FORB(NTS)) FOR EACH NFS
!     
     J=1
     DATA(J)=OSC
     J=J+1 
     CALL FFRNUM(REALO,INTEG,RET)  
     DATA(J)=REALO  !DLP
     J=J+1 
     CALL FFRNUM(REALO,INTEG,RET)  
     DATA(J)=REALO  !DENS
     J=J+1 
     CALL FFRNUM(REALO,INTEG,RET)  
     NTSISO=INTEG 
     DATA(J)=INTEG  !NTS
     J=J+1 
     CALL FFRNUM(REALO,INTEG,RET)  
     NFSISO=INTEG
     DATA(J)=INTEG  !NFS
!
!----NDATA=5+4*NTS+NFS*(1+NTS)
!   
     NDATA=5+4*NTSISO+NFSISO*(1+NTSISO)
     IF(NDATA.GT.NDATAMAX) STOP ' NDATA > NDATAMAX'

     DO I=1,4*NTSISO
        J=J+1 
        CALL FFRNUM(REALO,INTEG,RET)  
        DATA(J)=REALO  !TS,WS,DS,AS
     END DO

     DO I=1,NFSISO*(1+NTSISO)
        J=J+1 
        CALL FFRNUM(REALO,INTEG,RET)  
        DATA(J)=REALO  !(DLF,FORB) for each forbidden component
     END DO

     NWSN=NWS(NNN)
     IF(DWS(1,NNN)*DWS(NWSN,NNN).GE.0.D0) STOP ' ERROR IN ISO-FREQ (QHALF)'
     QHALF(NNN)=.FALSE.
     NNAT = LE(NLEVL(NNN))  
     PESO = WEIGHT(NNAT)  
     XL0 = XLAMB(NNN)  
     NTS(NNN) = 6  
     NES(NNN) = 11  
     NPS = NWS(NNN)*NTS(NNN)*NES(NNN)  
     IF (NPS.GT.MAXPS) STOP 'TOO MANY POINTS IN PROFILE'  
     DO IJK = 1,6  
       TS(IJK,NNN) = TTTS(IJK)  
     END DO

     DO IJK = 1,11  
       ES(IJK,NNN) = EEES(IJK)  
     END DO

     NIN=0
     JELOP: DO JE = 1,11  
       DO JT = 1,6  
         TT = 10.D0**TS(JT,NNN)  
         EE = 10.D0**ES(JE,NNN)  
         CALL ISO(DATA,NDATA,TT,EE,PESO,VTURB,XL0,DWS(1:NWSN,NNN),NWSN,SQ)
         DO IJK = 1,NWSN  
           NIN=NIN+1
           P(NIN,NNN) = LOG10(SQ(IJK))  
         END DO 
       END DO 
     END DO JELOP
     IF(NIN.NE.NPS) STOP ' ERROR IN NPS(ISO)'

     DONE(NNN) = .TRUE.  
     IF (NNN.NE.NB) THEN
        CYCLE
     ELSE  
        EXIT
     ENDIF

     ENDIF
!   
!----------------------------------------------------------------------
!
!
!  treat ISO1 lines
!     
     IF (KEY.EQ.'ISO1') THEN

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET))  GOTO 230
 
     CALL FFRKEY(KEY2,NC,RET)  
     IF (ON_ERROR(RET))  GOTO 230
     Q = (LABL(NLEVL(NNN)).EQ.KEY .AND. LABL(NLEVU(NNN)).EQ.KEY2)
     
     IF (.NOT. Q)  GOTO 20
     CALL FFRNUM(REALO,INTEG,RET)  
     WAVEAIR=REALO
     IF (ABS(WAVEAIR-XLAMB(NNN)).GT. 0.1) &
&     PRINT*,' CHECK WAVELENGTHS (ISO1)! ',WAVEAIR,' ',XLAMB(NNN) 
!
!    take value provided
!     
     XLAMB(NNN)=WAVEAIR
     CALL FFRNUM(REALO,INTEG,RET)  
     OSC=REALO
     IF (ABS(LOG10(OSC/FLU(NNN))).GT. 0.1)&
&     PRINT*,' CHECK OSCILLATOR STRENGTH (ISO1)! ',OSC,' ',FLU(NNN) 
!
!    take value provided
!     
     FLU(NNN)=OSC
!
!    read and write data in following format     
!----OSC,DLP,DENS,NTS, &
!----TS(NTS),WS(NTS),DS(NTS),   electrons
!----WS1(NTS),DS1(NTS),WS2(NTS),DS2(NTS) protons, HeII
!
!    NOTE: tables contain FULL half-widths, in contrast to ISO tables
!          provided by Griem
!     
     J=1
     DATA(J)=OSC
     J=J+1 
     CALL FFRNUM(REALO,INTEG,RET)  
     DATA(J)=REALO  !DLP
     J=J+1 
     CALL FFRNUM(REALO,INTEG,RET)  
     DATA(J)=REALO  !DENS
     J=J+1 
     CALL FFRNUM(REALO,INTEG,RET)  
     NTSISO=INTEG 
     DATA(J)=INTEG  !NTS
     J=J+1
     CALL FFRNUM(REALO,INTEG,RET)  
     DATA(J)=REALO  !CRIT

!
!----NDATA=5+NTS+6*NTS
!   
     NDATA=5+7*NTSISO
     IF(NDATA.GT.NDATAMAX) STOP ' NDATA > NDATAMAX'

     DO I=1,7*NTSISO
        J=J+1 
        CALL FFRNUM(REALO,INTEG,RET)  
        DATA(J)=REALO  !TS,WS,DS,WS1,DS1,WS2,DS2
     END DO

     NWSN=NWS(NNN)
     IF(DWS(1,NNN)*DWS(NWSN,NNN).GE.0.D0) STOP ' ERROR IN ISO-FREQ (QHALF)'
     QHALF(NNN)=.FALSE.
     NNAT = LE(NLEVL(NNN))  
     PESO = WEIGHT(NNAT)  
     XL0 = XLAMB(NNN)  
     NTS(NNN) = 6  
     NES(NNN) = 11  
     NPS = NWS(NNN)*NTS(NNN)*NES(NNN)  
     IF (NPS.GT.MAXPS) STOP 'TOO MANY POINTS IN PROFILE'  
     DO IJK = 1,6  
       TS(IJK,NNN) = TTTS(IJK)  
     END DO

     DO IJK = 1,11  
       ES(IJK,NNN) = EEES(IJK)  
     END DO

     NIN=0
     JELOP1: DO JE = 1,11  
       DO JT = 1,6  
         TT = 10.D0**TS(JT,NNN)  
         EE = 10.D0**ES(JE,NNN)  
         CALL ISO1(DATA,NDATA,TT,EE,PESO,VTURB,XL0,DWS(1:NWSN,NNN),NWSN,SQ,YHE)
         DO IJK = 1,NWSN  
           NIN=NIN+1
           P(NIN,NNN) = LOG10(SQ(IJK))  
         END DO 
       END DO 
     END DO JELOP1
     IF(NIN.NE.NPS) STOP ' ERROR IN NPS(ISO)'

     DONE(NNN) = .TRUE.  
     IF (NNN.NE.NB) THEN
        CYCLE
     ELSE  
        EXIT
     ENDIF

     ENDIF
!   
!----------------------------------------------------------------------
!
     LLCHAR=LABL(NLEVL(NNN))
     LUCHAR=LABL(NLEVU(NNN))
     IF(LLCHAR(1:2).EQ.'H1') THEN
       READ(LLCHAR(3:4),'(I2)') LL
       READ(LUCHAR(3:4),'(I2)') LU
     ENDIF
!
!   "normal" lines
!     
!   in case, use Balmer-lines from Lemke
     IF(LLCHAR.eq.'H12   '.AND. BALMER_LEMKE) THEN
       QHALF(NNN) = .TRUE.
       QLOG=.FALSE.
       CALL VCS_BALMER(XLAMB(NNN),LL,LU,NWS(NNN),DWS(:,NNN), &
&         NTS(NNN),TS(:,NNN),NES(NNN),ES(:,NNN),NPS,P(:,NNN))

       IF(VTURB.NE.0.) THEN 
         NWSN=NWS(NNN)
         NIN=0
         DO I = 1,NTS(NNN)*NES(NNN)
           DO J=1,NWSN
             NIN=NIN+1
             PROF(J) = P(NIN,NNN)  
           END DO

           CALL CONVVTURB(PROF(1:NWSN),DWS(1:NWSN,NNN),NWSN,QLOG, &
&            QHALF(NNN),I,VTURB,XLAMB(NNN))
           NIN=NIN-NWSN
           DO J=1,NWSN
             NIN=NIN+1
             P(NIN,NNN) = PROF(J)  
           END DO
         END DO
         IF(NIN.NE.NPS) STOP ' ERROR IN NPS(CONVVTURB)'
       ENDIF

!   in case, use Brackett-lines from Lemke
!   
     ELSE IF((LLCHAR.eq.'H13   '.OR. LLCHAR.eq.'H14   ') &
              .AND. PB_LEMKE .AND. LU.LE.10) THEN
       QHALF(NNN) = .TRUE.
       QLOG=.FALSE.
       CALL VCS_PB(XLAMB(NNN),LL,LU,NWS(NNN),DWS(:,NNN), &
&         NTS(NNN),TS(:,NNN),NES(NNN),ES(:,NNN),NPS,P(:,NNN))

       IF(VTURB.NE.0.) THEN 
         NWSN=NWS(NNN)
         NIN=0
         DO I = 1,NTS(NNN)*NES(NNN)
           DO J=1,NWSN
             NIN=NIN+1
             PROF(J) = P(NIN,NNN)  
           END DO

           CALL CONVVTURB(PROF(1:NWSN),DWS(1:NWSN,NNN),NWSN,QLOG, &
&            QHALF(NNN),I,VTURB,XLAMB(NNN))
           NIN=NIN-NWSN
           DO J=1,NWSN
             NIN=NIN+1
             P(NIN,NNN) = PROF(J)  
           END DO
         END DO
         IF(NIN.NE.NPS) STOP ' ERROR IN NPS(CONVVTURB)'
       ENDIF
       
     ELSE

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET))  GOTO 230
 
     CALL FFRKEY(KEY2,NC,RET)  
     IF (ON_ERROR(RET))  GOTO 230

     Q = (LABL(NLEVL(NNN)).EQ.KEY .AND. LABL(NLEVU(NNN)).EQ.KEY2)

     IF (.NOT. Q)  GOTO 20
     
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 230
     MFULL = INTEG  
     QHALF(NNN) = MFULL .EQ. 0  
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 230
!---------- number of wavelengths
     IF (INTEG.GT.MAXW) STOP 'TOO MANY LAMBDAS'  
     NWS(NNN) = INTEG  
     DO I = 1,INTEG  
        CALL FFRNUM(REALO,INTEG,RET)  
        IF (ON_ERROR(RET))  GOTO 230
        DWS(I,NNN) = REALO  
     END DO 

     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 230
!---------- number of temperatures
     IF (INTEG.GT.MAXT) STOP 'TOO MANY TEMPS'  
     NTS(NNN) = INTEG  
     DO I = 1,INTEG  
        CALL FFRNUM(REALO,INTEG,RET)
        IF (ON_ERROR(RET))  GOTO 230
        TS(I,NNN) = REALO  
     END DO
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 230
!---------- number of electron densities
     IF (INTEG.GT.MAXNE) STOP 'TOO MANY ELECTRON DENS.'  
     NES(NNN) = INTEG  
     DO I = 1,INTEG  
        CALL FFRNUM(REALO,INTEG,RET)  
        IF (ON_ERROR(RET))  GOTO 230
        ES(I,NNN) = REALO  
     END DO
     CALL FFRNUM(REALO,INTEG,RET)
     IF (ON_ERROR(RET))  GOTO 230
     FAC = REALO  
     CALL FFRNUM(REALO,INTEG,RET)  
     IF (ON_ERROR(RET))  GOTO 230
     QLOG = INTEG .EQ. 1  
!---------- number of profile points
     NPS = NWS(NNN)*NTS(NNN)*NES(NNN)  
     IF (NPS.GT.MAXPS) STOP 'TOO MANY POINTS IN PROFILE'  

     IF(VTURB.EQ.0.) THEN
        DO I = 1,NPS
           CALL FFRNUM(REALO,INTEG,RET)  
           IF (ON_ERROR(RET))  GOTO 230
           P(I,NNN) = REALO*FAC  
           IF (QLOG) P(I,NNN) = LOG10(P(I,NNN))  
        END DO
     ELSE 
        NWSN=NWS(NNN)
        NIN=0
        DO I = 1,NTS(NNN)*NES(NNN)
          DO J=1,NWSN
            CALL FFRNUM(REALO,INTEG,RET)  
            IF (ON_ERROR(RET))  GOTO 230
            PROF(J)=REALO*FAC
          END DO
          CALL CONVVTURB(PROF(1:NWSN),DWS(1:NWSN,NNN),NWSN,QLOG, &
&           QHALF(NNN),I,VTURB,XLAMB(NNN))
          IF(QLOG) THEN
            DO J=1,NWSN  
              PROF(J)=LOG10(PROF(J))
            END DO  
          ENDIF
          DO J=1,NWSN
            NIN=NIN+1
            P(NIN,NNN) = PROF(J)  
          END DO
        END DO
        IF(NIN.NE.NPS) STOP ' ERROR IN NPS(CONVVTURB)'
     ENDIF

     ENDIF
!
     DONE(NNN) = .TRUE.  
     IF (NNN.NE.NB) THEN
        CYCLE
     ELSE  
        EXIT
     ENDIF

STOP ' ERROR IN PATH'

!   
!----------------------------------------------------------------------
!
!---- stark profile(s) not in tables. we calculate them in Griem-approx,
!     if hydrogen or HeII

  150 CONTINUE  
!
IF (DONE(NNN)) STOP ' WRONG PATH'  
!
     PRINT *,'STARK PROFILES NOT PRESENT; COMPONENT:',NNN,' GRIEM USED'  
     KEY = LABL(NLEVL(NNN))  
     IF (KEY(:3).EQ.'HE2' .OR. KEY(:2).EQ.'H1') THEN  
          QHALF(NNN) = .TRUE.  
          NNAT = LE(NLEVL(NNN))  
          PESO = WEIGHT(NNAT)  
          XL0 = XLAMB(NNN)  
          NN1 = NPPAL(KEY)  
          KEY = LABL(NLEVU(NNN))  
          NN2 = NPPAL(KEY)  
          AA = DBLE(NN1)  
          BB = DBLE(NN2)  
          NTS(NNN) = 6  
          NES(NNN) = 11  
          NWS(NNN) = 35  
          NIN = 0  
          DO IJK = 1,6  
               TS(IJK,NNN) = TTTS(IJK)  
          END DO

          DO IJK = 1,11  
               ES(IJK,NNN) = EEES(IJK)  
          END DO

          JELOOP: DO JE = 1,11  
               DO JT = 1,6  
                    TT = 10.D0**TS(JT,NNN)  
                    EE = 10.D0**ES(JE,NNN)  
                    CALL GRIEM(TT,EE,SQ1,PESO,XL0,AA,BB,DDWS,VTURB)  
                    DO IJK = 1,35  
                         SQ(IJK)=SQ1(IJK)
                         IF(SQ(IJK).LT.0.) STOP ' NEG. PROFILE IN GRIEM'
                         NIN = NIN + 1  
                         DWS(IJK,NNN) = DDWS(IJK)  
                         P(NIN,NNN) = LOG10(SQ(IJK))  
                    END DO 
                END DO 
          END DO JELOOP
     ELSE  
          PRINT *,'STARK CALCULATION NOT IMPLEMENTED'  
          PRINT *,'LINE:',LABL(NLEVL(NNN)),'-',LABL(NLEVU(NNN))  
          STOP  
     END IF  

DONE(NNN) = .TRUE.  
IF (NNN.NE.NB) THEN
  CYCLE
ELSE  
  EXIT
ENDIF

END DO BIG_LOOP

CLOSE(1)

!---- output file is written

OPEN (1,FILE=TRIM(FILE)//'/STARK.BIN',STATUS='UNKNOWN',FORM='UNFORMATTED')  
REWIND 1  

DO I = 1,NB 
IF(NSTARK(I).NE.1) CYCLE
IF(.NOT.DONE(I))  STOP ' ERROR IN PRESTARK, BEFORE WRITING'
     NN = NWS(I)*NTS(I)*NES(I)  
     WRITE (1) NLEVL(I),NLEVU(I),QHALF(I),NWS(I), (DWS(J,I),J=1, &
      NWS(I)),NTS(I), (TS(J,I),J=1,NTS(I)),NES(I), (ES(J,I),J=1, &
      NES(I)),NN, (P(J,I),J=1,NN)
END DO  
CLOSE (1)  

DEALLOCATE(DWS,ES,P,TS)
DEALLOCATE(NES,NTS,NWS)
DEALLOCATE(DONE,QHALF)

RETURN  

!---- error in input unit
  220 CONTINUE  

STOP 'ERROR IN INPUT UNIT'  
!
! error handling
!
230 SELECT CASE(RET)
CASE('RET1') !       end of file exit 
  WRITE (*,FMT='(A)') ' END OF FILE '  
  STOP 
CASE('RET2') !       error exit 
  WRITE (*,FMT='(A)') 'ERROR IN FFR SUBROUTINES '  
  STOP
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN PRESTARK'
END SELECT

END
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
SUBROUTINE GRIEM(TT,EE,SQ,WEIGHT,WAVC,AA,BB,DWS,VTURB)  
!
USE nlte_type
USE nlte_dim
USE fund_const, ONLY: XMH=>AMH,AKB,PI
USE preformal_var, ONLY: NS,AS,ODOP,PS,ZZ,SS,SX

IMPLICIT NONE
!
!-----create a profile table using the griem theory as used by a and m
!-----in their ap j s24 paper
!     ..
!     .. scalar arguments ..
REAL(DP) ::  AA,BB,EE,TT,WAVC,WEIGHT,VTURB  
!     ..
!     .. array arguments ..
REAL(DP) ::  DWS(35),SQ(35)  
!     ..
!     .. local scalars ..
REAL(DP) ::  ASQ,BETA,BETAP,BETAW,BSQ,CLIGHT,CONKB,CUE,DIVKAB, &
&                 DOPKAB,DW,F0,FAC,FPW,GAM0,GAMA,GAML,GAMW,OBA, &
&                 RFT,SRE,SRT,SSS,SUMM,VMOT,WSQ,XKABF0
INTEGER(I4B) ::  I,JM,JP,NBET  
!     ..
!     .. local arrays ..
REAL(DP) ::  BET(35),DDWS(35)  
!     ..
!     .. external functions ..
REAL(DP) ::  DCONV,TBG,TH  
EXTERNAL DCONV,TBG,TH  
!     ..
!     .. intrinsic functions ..
INTRINSIC LOG,LOG10,MIN,SQRT  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA DDWS/0.D0,1.5849D-3,2.5119D-3,3.9811D-3,6.3096D-3,1.D-2, &
&     1.5849D-2,2.5119D-2,3.9811D-2,6.3096D-2,1.D-1,1.585D-1, &
&     1.995D-1,2.512D-1,3.162D-1,3.981D-1,5.012D-1,6.310D-1, &
&     7.943D-1,1.00D0,1.259D0,1.585D0,1.995D0,2.512D0,3.981D0, &
&     6.310D0,1.0D1,1.585D1,2.512D1,3.891D1,6.310D1,1.0D2,1.585D2, &
&     2.512D2,3.981D2/


DATA CLIGHT/2.997925D18/
DATA NBET/35/  
!     ..

SRT = SQRT(TT)  

!---- calculation of clight/v_thermal
VMOT = CLIGHT*1.D-8/SQRT(2.D0*AKB*TT/XMH/WEIGHT+VTURB**2)  

!-----doppler width (a)
RFT = WAVC/CLIGHT  
ODOP = VMOT/WAVC  

!-----upper and lower level parameters
ASQ = AA*AA  
BSQ = BB*BB  
WSQ = WAVC*WAVC  
OBA = 1.0d0/ (BSQ-ASQ)  

!-----normal field strength
SRE = SQRT(EE)  
CUE = EE**0.3333333333d0  
F0 = 1.25d-9*CUE*CUE  

!-----ratio stark/doppler width
DOPKAB = 5.5d-5* (ASQ*BSQ)**2*OBA*F0*ODOP/ZZ**5  

!-----1/stark width     (a)**-1
DIVKAB = ODOP/DOPKAB  

!---- calculation  of beta grid
XKABF0 = DOPKAB/ODOP  

DO I = 1,NBET  
  BET(I) = DDWS(I)/XKABF0
END DO  

BETAW = 6.951d-9*WSQ*ZZ*TT*DIVKAB/BSQ  
BETAP = 2.995d-15*WSQ*SRE*DIVKAB  
BETAP = MIN(BETAP,.99d0*BETAW)  
GAM0 = 3.78d-5* (BSQ-3.D0*AA)*BSQ*OBA*CUE*LOG10(4.d6*ZZ*TT/BSQ/SRE)/SRT/ZZ
GAML = LOG(BETAW/BETAP)  
GAMW = 4.712389d0/SQRT(BETAW)  
FPW = GAM0/GAML  

!-----initialize gama for zero,betap
GAMA = GAMW + GAM0  

!-----generate s(beta) on a fixed basis
JM = NBET  
JP = JM  

DO I = 1,NBET  
     BETA = BET(I)  
     IF (BETA.LT.BETAP) GO TO 20  
     IF (BETA.GT.BETAW) GO TO 40  

!-----   assign gama for betap,betaw
     GAMA = GAMW + FPW*LOG(BETAW/BETA)  

!-----   look for regions of the gama,beta plane where fast methods are
!-----   this is not just to save time as the t(beta,gama) analysis
!-----   is numerically unstable in these same regions
   20      CONTINUE  
     IF (GAMA.GT.25.) GO TO 50  
     IF (GAMA.LT..01D0*BETA) GO TO 30  
     IF (GAMA.LT..01) GO TO 30  

!-----   full case
     SSS = TBG(BETA,GAMA)  
     GO TO 60  

!-----   beta gt 100*gama for beta lt 1
   30      CONTINUE  
     SSS = TH(BETA)  
     IF (BETA.LT.1.0d0) GO TO 60  

!-----   gama lt  .01  or beta gt 100*gama for beta gt 1
     SSS = SSS + GAMA/ (PI*BETA*BETA)  
     GO TO 60  

!-----   beta gt betaw
   40      CONTINUE  
     SSS = 2.0D0*TH(BETA)  
     GO TO 60  

!-----   gama gt 25.
   50      CONTINUE  
     SSS = GAMA/ (PI* (GAMA*GAMA+BETA*BETA))  

!-----   fill the symmetric ss,sx vectors
   60      CONTINUE  
     SS(JM) = SSS  
     SX(JM) = -BETA*DOPKAB  
     SS(JP) = SSS  
     SX(JP) = BETA*DOPKAB  
     JM = JM - 1  
     JP = JP + 1  
END DO  


!-----make the asymtotic power law constants
NS = 2*NBET - 1  
PS = LOG(SS(NS)/SS(NS-1))/LOG(SX(NS)/SX(NS-1))  
CONKB = -2.0D0  
PS = MIN(PS,CONKB)  
AS = SS(NS)/SX(NS)**PS  

!-----normalize to unit frequency integral
SUMM = 0.0D0  
DO I = 2,NS  
     SUMM = SUMM + (SS(I)+SS(I-1))* (SX(I)-SX(I-1))  
END DO  

FAC = 2.0D0*VMOT*RFT/SUMM  

!-----convolve with doppler function
DO I = 1,35  
     DW = DDWS(I)  
     SQ(I) = FAC*DCONV(DW)  
     DWS(I) = DDWS(I)
END DO  

RETURN  


END
!
!-----------------------------------------------------------------------
!
FUNCTION TBG(BET,GAM)  
!
USE nlte_type
USE nlte_dim
USE fund_const, ONLY: PI
IMPLICIT NONE
!
!     .. scalar arguments ..
REAL(DP) ::  BET,GAM,TBG
!     ..
!     .. local scalars ..
REAL(DP) ::  T  
!     ..
!     .. external functions ..
REAL(DP) ::  TF,TG,TH  
EXTERNAL TF,TG,TH  
!     ..
!     .. save statement ..
SAVE  
!     ..

IF (GAM.GT.1.d-2) GO TO 10  

!     small gamma
TBG = TH(BET)  
IF (BET.GT.1.) TBG = TBG + GAM/PI/BET**2  
RETURN  

!     normal case
   10 CONTINUE  
IF (BET/GAM.GT.1.d2) GO TO 20  
T = GAM* (TF(BET,GAM)+TF(-BET,GAM)+TG(BET,GAM)+TG(-BET,GAM))/PI  
TBG = T  
RETURN  

   20 CONTINUE  
TBG = TH(BET) + GAM/PI/BET**2  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION TG(BET,GAM)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
REAL(DP) ::  BET,GAM,TG
!     ..
!     .. local scalars ..
REAL(DP) ::  B,BET2,C,C4,C5,C6,CC,COSA2,D1,D2,D3,D4,D5,D6,D7, &
&                 D8,D9,G2,P,P2,PI2,Q,Q2,R,SINA,SINA2,SUM1,SUM2, &
&                 SUM3,SUM4,X1,X2,X3,X4,X5,Y1,Y2
!     ..
!     .. intrinsic functions ..
INTRINSIC ATAN,LOG,SQRT  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA C4,C5,C6/1.5,3.4636008d1,-1.3253986d2/  
DATA PI2/1.57079632679489D0/  
!     ..

G2 = GAM*GAM  
B = 2.0D0*BET  
C = BET*BET + G2  
BET2 = BET*BET  
CC = C*C  
R = SQRT(C)  
Q = SQRT(R)  
Q2 = Q*Q  
P = 1.D0/Q  
P2 = P*P  
SINA = GAM/R  
COSA2 = SQRT(0.5D0* (1.D0-BET/R))  
SINA2 = SQRT(0.5D0* (1.D0+BET/R))  
X5 = BET + 4.D0  
Y2 = LOG((X5*X5+G2)/16.D0)  
IF (X5/GAM.GT.1.0) GO TO 10  
Y1 = PI2 - ATAN(X5/GAM)  

GO TO 20  

   10 CONTINUE  
Y1 = ATAN(GAM/X5)  

   20 CONTINUE  
D2 = C/192.D0  
D3 = -B/32.D0  
D4 = 0.25D0* (3.D0*BET2-G2)/C  
D5 = B* (G2-BET2)/CC  
D6 = (BET2* (BET2-6.0D0*G2)+G2*G2)/CC  
SUM1 = ((D6*Y1)/GAM+ (D4+D3)) + D2  
SUM1 = (SUM1+D5*Y2)*C5/CC  
D1 = C/1024.D0  
D2 = -B/192.D0  
D3 = (3.0D0*BET2-G2)/ (32.D0*C)  
D4 = BET* (G2-BET2)/CC  
D5 = 0.5D0* (G2*G2+5.D0*BET2* (BET2-2.0D0*G2))/ (C*CC)  
D6 = -BET* (BET2*BET2+5.0D0*G2* (G2-2.0D0*BET2))/ (CC*C)  
SUM2 = (D5*Y2+D3) + D1  
SUM2 = (SUM2+ (D2+D4+ (D6*Y1)/GAM))*C6/CC  
D7 = C4/C  
D8 = D7* (B*B/C-1.D0)/ (2.D0*Q*Q2*SINA)  
D9 = D7* (B/ (C*C))/ (2.D0*P*P2*SINA)  
X1 = (4.0D0-Q2)/ (4.0D0*Q*SINA2)  
X2 = (Q* (Q+4.D0*COSA2)+4.D0)/ (Q* (Q-4.D0*COSA2)+4.D0)  
X3 = (0.25D0-P2)/ (P*SINA2)  
X4 = (P* (P+COSA2)+0.25D0)/ (P* (P-COSA2)+0.25D0)  
IF (X1.GT.1.) GO TO 30  
Y1 = PI2 - ATAN(X1)  

GO TO 40  

   30 CONTINUE  
Y1 = ATAN(1.D0/X1)  

   40 CONTINUE  
IF (X3.GT.-1.) GO TO 50  
Y2 = -ATAN(1.D0/X3)  

GO TO 60  

   50 CONTINUE  
Y2 = PI2 + ATAN(X3)  

   60 CONTINUE  
SUM3 = D8* (2.D0*COSA2*Y1-SINA2*LOG(X2))  
SUM4 = D9* (2.D0*COSA2*Y2+SINA2*LOG(X4))  
TG = (SUM4+D7* (1.D0/12.D0-B/C)) + SUM3  
TG = TG + SUM1 + SUM2  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION TF(B,G)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!     .. scalar arguments ..
REAL(DP) ::  B,G,TF  
!     ..
!     .. local scalars ..
REAL(DP) ::  C0,C1,C2,C3,D,D1,D2,D3,G2,X1  
!     ..
!     .. intrinsic functions ..
INTRINSIC ATAN,LOG  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA C0,C1,C2,C3/1.0007744d-1,4.93208719d-3,-7.09873526d-3, &
&     7.11559325d-4/
!     ..
D3 = C3  
D2 = C2 - 3.0D0*B*C3  
D1 = (3.0D0*C3*B-2.0D0*C2)*B + C1  
D = ((-C3*B+C2)*B-C1)*B + C0  
G2 = G*G  
X1 = B + 4.D0  
TF = 4.0D0*D2 + 4.0D0*D3* (B+2.0D0) + &
&     0.5D0* (D1-G2*D3)*LOG((X1*X1+G2)/ (B*B+G2)) + &
&     (D-G2*D2)* (ATAN(X1/G)-ATAN(B/G))/G

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION DCONV(DLAM)  
!
USE nlte_type
USE nlte_dim
USE preformal_var, ONLY: NS,ODOP,SS,SX
IMPLICIT NONE
!
!-----
!-----convolution of gaussian profile with s function  6 aug 77
!-----
!     .. scalar arguments ..
REAL(DP) ::  DLAM,DCONV  
!     ..
!     .. local scalars ..
REAL(DP) ::  CON,CR,DB,DX,GR,HALF,RANGE,SRTPI,TERX,TEX,X1,XKB, &
&                 XN,ZERO
INTEGER(I4B) ::  I,J,JM,N0  
!     ..
!     .. local arrays ..
REAL(DP) ::  ERX(69),EX(69),X(69)  
!     ..
!     .. external functions ..
REAL(DP) ::  ASINT,ERR  
EXTERNAL ASINT,ERR  
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS,EXP  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA HALF/0.5d0/,ZERO/0.0d0/,SRTPI/5.6418958d-1/  
DATA RANGE/6.0d0/  
!     ..

DB = ABS(DLAM)*ODOP  
X1 = SX(1) + DB  
IF (X1.LT.RANGE) GO TO 10  

!-----asint -range,range
DCONV = ASINT(-RANGE,RANGE,DB)  
RETURN  

   10 CONTINUE  
DCONV = ZERO  

!-----asint -range,x1
IF (X1.GT.-RANGE) DCONV = DCONV + ASINT(-RANGE,X1,DB)  

!-----asint xn,range
XN = SX(NS) + DB  
IF (XN.LT.RANGE) DCONV = DCONV + ASINT(XN,RANGE,DB)  

!-----set up x in the gaussian frame
!-----store all exponential and error functions once
N0 = 2  
CON = HALF*SRTPI  

DO I = 1,NS  
     XKB = SX(I) + DB  
     X(I) = XKB  
     IF (XKB.LT.-RANGE) N0 = I + 1  
     EX(I) = EXP(-XKB*XKB)  
     ERX(I) = ERR(XKB,EX(I))  
END DO  

!-----by s segment
TERX = ZERO  
TEX = ZERO  

DO J = N0,NS  
     JM = J - 1  
     DX = X(J) - X(JM)  
     IF (DX.LT.0.1) GO TO 30  
     GR = (SS(J)-SS(JM))/DX  
     CR = SS(J) - GR*X(J)  
     TERX = TERX + CR* (ERX(J)-ERX(JM))  
     TEX = TEX + GR* (EX(J)-EX(JM))  

     GO TO 40  
   30      CONTINUE  
     DCONV = DCONV + CON* (SS(J)*EX(J)+SS(JM)*EX(JM))*DX  
   40      CONTINUE  
     IF (X(J).GT.RANGE) EXIT  
END DO  

DCONV = DCONV + HALF* (TERX-SRTPI*TEX)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION ASINT(X0,X1,DB)  
!
USE nlte_type
USE nlte_dim
USE preformal_var, ONLY: AS,PS
IMPLICIT NONE
!
!-----
!-----integral over asymtotic region by simpsons rule
!-----
!     .. scalar arguments ..
REAL(DP) ::  DB,X0,X1,ASINT
!     ..
!     .. local scalars ..
REAL(DP) ::  C23,DX,H,HALF,ONE,SRTPI,STEP,TWO,X  
INTEGER(I4B) ::  I,N  
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS,DBLE,EXP,INT  
!     ..
!     .. statement functions ..
REAL(DP) ::  F  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA SRTPI/5.6418958d-1/,C23/6.666666667d-1/  
DATA HALF,ONE,TWO/0.5d0,1.0d0,2.0d0/  
DATA STEP/0.2d0/  
!     ..
!     .. statement function definitions ..
F(X) = EXP(-X*X)*ABS(X-DB)**PS  
!     ..

!-----chose h to be sept
N = INT((X1-X0)/STEP+ONE)  
DX = (X1-X0)/DBLE(N)  
H = HALF*DX  
X = X0 - H  
ASINT = F(X0)  

DO I = 1,N  
     X = X + DX  
     ASINT = ASINT + TWO*F(X) + F(X+H)  
END DO  

ASINT = ASINT*AS*SRTPI*H*C23  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION ERR(X,EX)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!-----error function
!     .. scalar arguments ..
REAL(DP) ::  EX,X,ERR
!     ..
!     .. local scalars ..
REAL(DP) ::  ONE,T  
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS,SIGN  
!     ..
!     .. save statement ..
SAVE  
!     ..

ONE = 1.0D0  
T = ABS(X)  
IF (T.GT.6.) GO TO 10  
T = 1.D0/ (1.D0+0.3275911D0*T)  
ERR = ((((1.061405429D0*T-1.453152027D0)*T+1.421413741D0)*T- &
&      0.284496736D0)*T+0.254829592D0)*T
ERR = SIGN(ONE-EX*ERR,X)  

RETURN  

   10 CONTINUE  
ERR = SIGN(ONE,X)  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION TH(X)  
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!
!-----griem microfield fudge - normalized to unit b integral
!     .. scalar arguments ..
REAL(DP) ::  X,TH
!     ..
!     .. local scalars ..
REAL(DP) ::  B,B2,C0,C1,C2,C3,C4,C5,C6,SRT  
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS,SQRT  
!     ..
!     .. save statement ..
SAVE  
!     ..
!     .. data statements ..
DATA C0,C1,C2,C3/1.0007744d-1,4.93208719d-3,-7.09873526d-3, &
&     7.11559325d-4/
DATA C4,C5,C6/1.5,3.4636008d1,-1.3253986d2/  
!     ..

B = ABS(X)  
IF (B.GT.4.) GO TO 10  
TH = ((C3*B+C2)*B+C1)*B + C0  

RETURN  

   10 CONTINUE  
SRT = SQRT(B)  
B2 = B*B  
TH = ((C6/B+C5)/B2+C4/SRT)/B2  

RETURN  

END
!
!-----------------------------------------------------------------------
!
FUNCTION NPPAL(LAB)  
!
USE nlte_type
USE nlte_dim
USE preformal_var, ONLY: ZZ
IMPLICIT NONE
!
!---- this function determines the principal quantum number, n,
!---- of a certain level of an hydrogenic ion, if the level's label
!---- is written in the "normal detail way"
!
!      ..
INTEGER(I4B) :: NPPAL
!
!     .. scalar arguments ..
CHARACTER LAB*6  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  IJ,J  
CHARACTER CAR*1,AT*2,CARA*2  
!     ..
!     .. save statement ..
SAVE  
!     ..

J = 1  

   10 CONTINUE  
CAR = LAB(J:J)  
IF (CAR.EQ.'0' .OR. CAR.EQ.'1' .OR. CAR.EQ.'2' .OR. &
&    CAR.EQ.'3' .OR. CAR.EQ.'4' .OR. CAR.EQ.'5' .OR. &
&    CAR.EQ.'6' .OR. CAR.EQ.'7' .OR. CAR.EQ.'8' .OR. &
&    CAR.EQ.'9') THEN
!----------- atom determination for atomic charge
     AT = LAB(1:J-1)  

     IF (AT.EQ.'H') THEN  
          ZZ = 1.D0  
     ELSE IF (AT.EQ.'HE') THEN  
          ZZ = 2.D0  
     ELSE  
          PRINT *,'ATOM ',AT,' NOT IMPLEMENTED - NPPAL'  
          STOP  
     END IF  

     J = J + 1  
     IJ = J  

   20      CONTINUE  
     CAR = LAB(J:J)  
     IF (CAR.EQ.'0' .OR. CAR.EQ.'1' .OR. CAR.EQ.'2' .OR. &
&        CAR.EQ.'3' .OR. CAR.EQ.'4' .OR. CAR.EQ.'5' .OR. &
&        CAR.EQ.'6' .OR. CAR.EQ.'7' .OR. CAR.EQ.'8' .OR. &
&        CAR.EQ.'9') THEN
          J = J + 1  
          GO TO 20  
     ELSE  
          J = J - 1  
          CARA = LAB(IJ:J)  
          READ (CARA,FMT='(I2)') NPPAL  
          IF (NPPAL.EQ.0) NPPAL = 10  
     END IF  

ELSE  
     J = J + 1  
     GO TO 10  
END IF  

RETURN  
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE CONVVTURB(PROF,DDWS,NS,QLOG,QHALF,ISTART,VTURB,LAMBDA0)
!
!convolution with vturb, assuming profile is linear between grid points.
!for istart=1, convolution matrix will be calculated, for istart > 1
!"only" applied  
!
!
USE nlte_type
USE nlte_dim
USE fund_const, ONLY: CLIGHT
IMPLICIT NONE

!Input data
INTEGER(I4B) :: NS,ISTART
REAL(DP) :: PROF(NS),DDWS(NS),VTURB,LAMBDA0
LOGICAL :: QLOG, QHALF

INTEGER(I4B), PARAMETER :: NSMAX=ID_MAXWW,NSMAX2=2*NSMAX-1
REAL(DP), PARAMETER     :: SRTPI1=0.5641895835D0
!
INTEGER(I4B) :: I,J,JMIN,JMAX,J1,J2,NS2
REAL(DP) :: CONST,ERR,EX,X0,W
REAL(DP) :: Q,Q1,DX,DX1,DX2,DELTAEXP,DELTAERF,X1,X2,EXP1,EXP2,ERFC1,ERFC2
REAL(DP) :: DW(NSMAX2),X(0:NSMAX2+1),P(0:NSMAX2+1), &
&           A(NSMAX,0:NSMAX2+1),PNEW(NSMAX) 

SAVE A

!check whether input from blue to red as it is assumed
IF(DDWS(1).GT.DDWS(NS)) STOP ' INPUT FROM RED TO BLUE IN CONVVTURB'

!change order from red to blue and complete variables if QHALF

NS2=NS

IF (QHALF) THEN
   NS2=2*NS-1
   DO I=1,NS
     J=I+NS-1
     DW(I)=DDWS(NS+1-I)
     P(I)=PROF(NS+1-I)
     DW(J)=-DDWS(I)
     P(J)=PROF(I)
   END DO
ELSE
   DO I=1,NS2
     DW(I)=DDWS(NS2+1-I)
     P(I)=PROF(NS2+1-I)
   END DO
ENDIF

IF (.NOT.QLOG) THEN
  DO I=1,NS2
    P(I)=10.**P(I)
  END DO
ENDIF

!calculate matrix once

IF(ISTART.EQ.1) THEN

A=0.
!conversion to Doppler units (with respect to vturb)

CONST=CLIGHT/(VTURB*LAMBDA0)
DO I=1,NS2
  X(I)=-DW(I)*CONST
END DO
X(0)=X(1)-5.01D0
X(NS2+1)=X(NS2)+5.1D0

!append first and last element assuming constant profiles
P(0)=P(1)
P(NS2+1)=P(NS2)

JMIN=0
JMAX=0

!set up matrix, calculate full (NS2=NS) or only half (NS2 NE NS) range
BIG: DO I=1,NS
X0=X(I)

!find range
DO J=JMIN,NS2
  IF(X(J)-X0.GT.-5.D0) EXIT
ENDDO  
JMIN=J-1

DO J=JMAX,NS2+1
  IF(X(J)-X0.GE.5.D0) EXIT
ENDDO  
JMAX=J

!calculate weights
!first element
J1=JMIN
J2=J1+1
X1=X(J1)
X2=X(J2)
DX=X2-X1
DX1=X1-X0
DX2=X2-X0
Q=-DX1/DX
Q1=1.D0-Q
EXP1=EXP(-DX1*DX1)
EXP2=EXP(-DX2*DX2)
DELTAEXP=EXP1-EXP2
ERFC1=1.D0-ERR(DX1,EXP1)
ERFC2=1.D0-ERR(DX2,EXP2)
DELTAERF=ERFC1-ERFC2
W=Q1*DELTAERF-SRTPI1*DELTAEXP/DX
A(I,J1)=.5D0*W

!middle elements
DO J=JMIN+1,JMAX-1

!old values
W=Q*DELTAERF+SRTPI1*DELTAEXP/DX

!new values
J1=J
J2=J1+1
X1=X(J1)
X2=X(J2)
DX=X2-X1
DX1=X1-X0
DX2=X2-X0
Q=-DX1/DX
Q1=1.D0-Q
EXP1=EXP(-DX1*DX1)
EXP2=EXP(-DX2*DX2)
DELTAEXP=EXP1-EXP2
ERFC1=1.D0-ERR(DX1,EXP1)
ERFC2=1.D0-ERR(DX2,EXP2)
DELTAERF=ERFC1-ERFC2
W=W+Q1*DELTAERF-SRTPI1*DELTAEXP/DX
A(I,J1)=.5D0*W
END DO

!last element
W=Q*DELTAERF+SRTPI1*DELTAEXP/DX
A(I,JMAX)=.5D0*W

END DO BIG

ENDIF

!convolution
PNEW(1:NS)=MATMUL(A(1:NS,0:NS2+1),P(0:NS2+1))

!test
W=0.D0
Q=0.D0
DO I=1,NS-1
DX=.5*(DW(I+1)-DW(I))
W=W+(P(I)+P(I+1))*DX
Q=Q+(PNEW(I)+PNEW(I+1))*DX
END DO

!error in equivalent width
DX=1.-Q/W
IF(ABS(DX).GT.0.1D0) PRINT*,' WARNING!!! ERROR IN CONVOLUTION LARGE: ',DX

!renormalization
DO I=1,NS
  PNEW(I)=PNEW(I)*W/Q
END DO

!reversal of order: either pnew is ordered from -xmax to 0 (symmetric case)
! or from -xmax to +xmax (asymmetric case)

DO I=1,NS
  PROF(I)=PNEW(NS+1-I)
END DO

IF(.NOT.QLOG) THEN
  DO I=1,NS
    PROF(I)=LOG10(PROF(I))
  END DO
ENDIF

RETURN
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE ISO(DATA,NDATA,TT,EE,WEIGHT,VTURB,LAMBDA0,DWS,NWS,SQ)
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE


!-----ISOLATED AND MULTI COMPONENT HE I LINE PROFILES
!-----INPUT DATA ARE - OSC,DLP,DENS,NTS,NFS,
!-----                 TS(NTS),WS(NTS),DS(NTS),AS(NTS),
!-----                 (DLF,FORB(NTS)) FOR EACH NFS
!      
!-----Updated/corrected by j. puls 13/04/99
!     depends majorly on treatment by Barnard, Cooper, Smith, 1974,
!     J.Q.S.R.T. 14, 1025
!
!      
!-----NDATA=5+4*NTS+NFS*(1+NTS)
!
INTEGER(I4B), PARAMETER :: MAXWW=ID_MAXWW
REAL(DP), PARAMETER :: CLIGHT=2.997925D18
!
!input values
!
INTEGER(I4B) :: NDATA,NWS
REAL(DP) :: DATA(NDATA),TT,EE,WEIGHT,VTURB,LAMBDA0,DWS(NWS),SQ(NWS)

!local values
INTEGER(I4B) :: I,IB,IF,J,K,L,NFS,NTS
REAL(DP) :: A,ALF,CON,D,DB,DENS,DLF,DLP,FT,FQ(MAXWW),P,RBA,RBHZ,RFT,RWT, &
&           RHOM,SIGMA,SRT,VA,VB,VTH,VMOT,VP,W,WF,WQ,WT,X,Y

REAL(DP) :: TS(4),WS(4),DS(4),AS(4),DLS(3),FB(6,3),FOS(3) 

!     .. external functions ..
REAL(DP) ::  VOIGT  
EXTERNAL VOIGT  

!-----frequential quantities

DO I=1,NWS
  WQ=LAMBDA0+DWS(I)
  FQ(I)=CLIGHT/WQ
END DO

FT=CLIGHT/LAMBDA0
RFT=1.D0/FT

SRT=SQRT(TT)

VTH=SQRT(1.6504D8*TT/WEIGHT+VTURB**2)
VMOT=CLIGHT*1.D-8/VTH

WT=CLIGHT*RFT
RWT=FT/CLIGHT

!     move DATA to local

I = 1
DLP=DATA(I+1)
DENS=DATA(I+2)
NTS=DATA(I+3)+0.5D0
NFS=DATA(I+4)+1.5D0
K=I+4

DO J=1,NTS
  K=K+1
  TS(J)=DATA(K)
ENDDO

DO J=1,NTS
  K=K+1
  WS(J)=DATA(K)
ENDDO

DO J=1,NTS
  K=K+1
  DS(J)=DATA(K)
ENDDO

DO J=1,NTS
  K=K+1
  AS(J)=DATA(K)
ENDDO

IF(NFS.GT.1) THEN
  DO L=2,NFS
    K=K+1
    DLS(L)=DATA(K)
    DO J=1,NTS
      K=K+1
      FB(J,L)=DATA(K)
    ENDDO
  ENDDO
ENDIF

!
!     RBHZ = 1/deltanu-dop
!     RBA  = nu0/vth  (for transformation dlambda -> dnue   
!
RBHZ=VMOT*RFT
RBA=VMOT*RWT

DLP=DLP*RBA
IF(NFS.GT.1) THEN
  DO L=2,NFS
    DLS(L)=DLS(L)*RBA
  ENDDO
ENDIF

!-----SET UP INTERPOLATION IN T
DO J=2,NTS
  IF=J
  IF(TS(J).GE.TT) EXIT
ENDDO

IB=IF-1
WF=(TT-TS(IB))/(TS(IF)-TS(IB))

IF(TT.LT.TS(IB)) WF = 0.0D0
IF(TT.GT.TS(IF)) WF = 1.0D0
   
!-----PERTURBER QUANTITIES
Y=EE/DENS
!
!       VP=8.78D0/SRT !factor cannot be understood, should be
!
!       factor = [(1./A_perturber)+(1./A_radiator)]^(-1/2), A atomic weight
!       A_perturber = hydrogen = 1, A_radiatior = helium = 4    
!
!
VP=1.D0/(SQRT(1.+1./WEIGHT)*SRT)  !inverse of mean vel., without factors
!
RHOM=0.62035D0/EE**0.333333D0     ! mean distance, from 4pi/3 Ne rhom^3=1

!-----IMPACT WIDTH WIDTH (A)
W=(WF*(WS(IF)-WS(IB))+WS(IB))*Y

!-----RATIO IMPACT SHIFT/WIDTH
D=WF*(DS(IF)-DS(IB))+DS(IB)

!-----ION BROADENING PARAMETERS
ALF=(WF*(AS(IF)-AS(IB))+AS(IB)) *Y**0.25D0
!
!      factor accounts for transformation dlambda -> dnu and appropriate units
!      factor = sqrt(pi*m_hyd/(8 k_botz)) * clight * 1.e8 
!        
SIGMA=2.06D14*W*RHOM*VP*RWT*RWT      ! sig=w*rhom/v (*c/lam0^2)
X=ALF**0.888889D0/SIGMA**0.333333D0  ! alf^(8/9) sig^(-1/3) propto w=wi/we

!-----TOTAL WIDTH IN DOPPLER UNITS
A=W*(1.0D0+1.36D0*X)*RBA                ! we*(1+wi/we) at line-center
DLS(1)=W*D*(1.0D0+2.36D0*X/ABS(D))*RBA  ! de*(1+di/de) at line-center, no Debye shielding
                                        ! and di/de=(di/we)/(de/we), BCS 6.6
FOS(1)=1.0D0

!-----SATELLITE COMPONENTS
X=FOS(1)
IF(NFS.GT.1) THEN
   DO L=2,NFS
     FOS(L)=WF*(FB(IF,L)-FB(IB,L))+FB(IB,L)
     X=X+FOS(L)
   ENDDO
ENDIF
!
!      con includes transformation to Doppler units via RBHZ,
!      factor 1/sqrt(pi) from normalization of Voigt profile and
!      factor 1/9 from using (8:1) weighted Voigt profiles to account
!      for fine-structure of lower component, level 2P3.         
!      Works also for other cases if separation is set to zero
!        
CON=6.268773D-2*RBHZ/X

!-----COMPUTE PROFILE
DO J=1,NWS                                                             
  DB=(FT-FQ(J))*RBHZ                                                     
  P=0.D0                                                                  
  DO L=1,NFS                                                         
    VA=DB-DLS(L)                                                        
    VB=VA-DLP                                                           
    P=P+FOS(L)*(8.0D0*VOIGT(A,VA)+VOIGT(A,VB))                            
  END DO                                                               
  SQ(J)=CON*P                                                            
END DO                                                                  

RETURN
END
!
!-----------------------------------------------------------------------
!
SUBROUTINE ISO1(DATA,NDATA,TT,EE,WEIGHT,VTURB,LAMBDA0,DWS,NWS,SQ,YHE)
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE


!-----ISOLATE HE I LINE PROFILES according to data from
!     Dimitrijevic & Sahal-Brechot.
!     So far, proton/HeII contribution only approximate
!     from YHe input values and HeI (assumed to be constant
!     throughout the atmosphere)
!     If HeIII major component (corresp. to HeI = 2), same
!     values as for HeII used. 


!-----INPUT DATA ARE - OSC,DLP,DENS,NTS,
!-----                 TS(NTS),WS(NTS),DS(NTS),  for electrons
!-----                 WS1(NTS),DS1(NTS),WS2(NTS),DS2(NTS) for protons and HeII
!      
!-----programmed by j.p. 04/04 in analogy to subroutine iso from above
!
!      
!-----NDATA=4+7*NTS
!
INTEGER(I4B), PARAMETER :: MAXWW=ID_MAXWW
REAL(DP), PARAMETER :: CLIGHT=2.997925D18
!
!input values
!
INTEGER(I4B) :: NDATA, NWS
REAL(DP) :: DATA(NDATA),TT,EE,WEIGHT,VTURB,LAMBDA0,DWS(NWS),SQ(NWS),YHE

!local values
INTEGER(I4B) :: I,IB,IF,J,K,L,NTS
REAL(DP) :: A,CON,CRIT,D,D1,D2,DB,DENS,DLP,DLS,FT,FQ(MAXWW),P,RBA,RBHZ,RFT,RWT, &
& SRT,VA,VB,VTH,VMOT,W,W1,W2,WF,WQ,WT,Y, &
& NP,NHE,HEI

REAL(DP) :: TS(6),WS(6),DS(6),WS1(6),DS1(6),WS2(6),DS2(6) 

!     .. external functions ..
REAL(DP) ::  VOIGT  
EXTERNAL VOIGT  

!-----frequential quantities

DO I=1,NWS
  WQ=LAMBDA0+DWS(I)
  FQ(I)=CLIGHT/WQ
END DO

FT=CLIGHT/LAMBDA0
RFT=1.D0/FT

SRT=SQRT(TT)

VTH=SQRT(1.6504D8*TT/WEIGHT+VTURB**2)
VMOT=CLIGHT*1.D-8/VTH

WT=CLIGHT*RFT
RWT=FT/CLIGHT

!     move DATA to local

I = 1
DLP=DATA(I+1)
DENS=DATA(I+2)
NTS=DATA(I+3)+0.5D0
CRIT=DATA(I+4)
K=I+4

DO J=1,NTS
  K=K+1
  TS(J)=DATA(K)
ENDDO

DO J=1,NTS
  K=K+1
  WS(J)=DATA(K)
ENDDO

DO J=1,NTS
  K=K+1
  DS(J)=DATA(K)
ENDDO

DO J=1,NTS
  K=K+1
  WS1(J)=DATA(K)
ENDDO

DO J=1,NTS
  K=K+1
  DS1(J)=DATA(K)
ENDDO
DO J=1,NTS
  K=K+1
  WS2(J)=DATA(K)
ENDDO

DO J=1,NTS
  K=K+1
  DS2(J)=DATA(K)
ENDDO

!
!     RBHZ = 1/deltanu-dop
!     RBA  = nu0/vth  (for transformation dlambda -> dnue   
!
RBHZ=VMOT*RFT
RBA=VMOT*RWT

DLP=DLP*RBA

!-----SET UP INTERPOLATION IN T
DO J=2,NTS
  IF=J
  IF(TS(J).GE.TT) EXIT
ENDDO

IB=IF-1
WF=(TT-TS(IB))/(TS(IF)-TS(IB))

IF(TT.LT.TS(IB)) WF = 0.0D0
IF(TT.GT.TS(IF)) WF = 1.0D0
   
!-----PERTURBER QUANTITIES
Y=EE/DENS
!
!-----IMPACT WIDTH WIDTH (A)
W=(WF*(WS(IF)-WS(IB))+WS(IB))*Y
W=.5D0*W !tables give full halfwidths 

!-----SHIFT (A)
D=(WF*(DS(IF)-DS(IB))+DS(IB))*Y

! the same for protons and helium ions (note, that heiii approximated by heii!)
! test for crit

! rough estimate
IF(TT.GE.25000.) THEN
  HEI=2.
ELSE IF (TT.GE.10000.) THEN
  HEI=1.
ELSE
  HEI=0.
ENDIF  

NP=EE/(1.+HEI*YHE) !valid also for hei=0
Y=NP/DENS

W1=(WF*(WS1(IF)-WS1(IB))+WS1(IB))
IF(W1.EQ.0.) THEN
  D1=0.
ELSE
  IF(CRIT/W1.LE.NP) THEN
    W1=.5D0*W1*Y !tables give full halfwidths 
!-----SHIFT (A)
    D1=(WF*(DS1(IF)-DS1(IB))+DS1(IB))*Y
  ELSE ! NOT RELIABLE
    W1=0.
    D1=0.
  ENDIF
ENDIF


IF(HEI.EQ.0.) THEN
!no contribution from heii/heiii
  W2=0.
ELSE
  NHE=YHE*NP !(heii/heiii)  
  Y=NHE/DENS
  W2=(WF*(WS2(IF)-WS2(IB))+WS2(IB))
ENDIF

IF(W2.EQ.0.) THEN
  D2=0.
ELSE
  IF(W2.NE.0.AND.CRIT/W2.LE.NHE) THEN
    W2=.5D0*W2*Y !tables give full halfwidths 
!-----SHIFT (A)
    D2=(WF*(DS2(IF)-DS2(IB))+DS2(IB))*Y
  ELSE ! NOT RELIABLE
    W2=0.
    D2=0.
  ENDIF
ENDIF

!-----TOTAL WIDTH IN DOPPLER UNITS
A=(W+W1+W2)*RBA
DLS=(D+D1+D2)*RBA
!
!      con includes transformation to Doppler units via RBHZ,
!      factor 1/sqrt(pi) from normalization of Voigt profile and
!      factor 1/9 from using (8:1) weighted Voigt profiles to account
!      for fine-structure of lower component, level 2P3.         
!      Works also for other cases if separation is set to zero
!        
CON=6.268773D-2*RBHZ

!-----COMPUTE PROFILE

DO J=1,NWS                                                             
  DB=(FT-FQ(J))*RBHZ
  VA=DB-DLS                                                        
  VB=VA-DLP                                                           
  P=8.0D0*VOIGT(A,VA)+VOIGT(A,VB)
  SQ(J)=CON*P                                                            
END DO                                                                  

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
SUBROUTINE TRANSIC_H_AT(LEVEL,LEVEU,NCOMP,LINENO,NLEVL,NLEVU,NATOM, &
&                   XLAMB,FLU,NSTARK)
USE nlte_type
USE nlte_dim
USE fund_const, ONLY: CLIGHT
USE preformal_var, ONLY: FL,GL,ZL,LABL,NL,LE,IU,NLEVH21
IMPLICIT NONE
!
! provides info on high lying H-levels
!
!     .. scalar arguments ..
INTEGER(I4B) ::  NCOMP  
!     ..
!     .. array arguments ..
REAL(DP) ::  FLU(NCOMP),XLAMB(NCOMP) 
INTEGER(I4B) ::  LINENO(NCOMP),NATOM(NCOMP),NLEVL(NCOMP),NLEVU(NCOMP), &
&                NSTARK(NCOMP)
CHARACTER LEVEL(NCOMP)*6,LEVEU(NCOMP)*6  

! local variables
REAL(DP), PARAMETER :: RYD=109677.58D0

INTEGER(I4B) ::  NPPAL,NN1,NN2,I,LEH20  

! final consistency check
IF (NCOMP.NE.1) STOP ' NCOMP NE 1 IN TRANSIC_H_app'
IF (LINENO(NCOMP).NE.1) STOP ' LINENO NE 1 IN TRANSIC_H_app'
IF (NSTARK(NCOMP).GT.1) STOP ' NSTARK > 1 IN TRANSIC_H_app'

IF(NL.NE.ID_LLEVS) STOP ' NL NE ID_LLEVS IN TRANSIC_H_AT'

NL=NL+2 !two more levels

LABL(ID_LLEVS+1)=LEVEL(1)
LABL(ID_LLEVS+2)=LEVEU(1)

!levels of neutral hydrogen
ZL(ID_LLEVS+1)=0.
ZL(ID_LLEVS+2)=0.

! two extra levels
NLEVL(1)=ID_LLEVS+1
NLEVU(1)=ID_LLEVS+2

NN1=NPPAL(LEVEL(1))
NN2=NPPAL(LEVEU(1))

PRINT*,'approximate treatment for level ',level(1),'(',NN1,') and ',leveu(1),'(',NN2,')'

GL(ID_LLEVS+1)=2.*NN1**2
GL(ID_LLEVS+2)=2.*NN2**2

FL(ID_LLEVS+1)=RYD/FLOAT(NN1**2)*CLIGHT
FL(ID_LLEVS+2)=RYD/FLOAT(NN2**2)*CLIGHT

DO I=1,ID_LLEVS
  IF (LABL(I).EQ.'H21') GOTO 10
ENDDO
STOP ' H21 NOT FOUND IN LABL'

10 NLEVH21=I
LEH20=LE(NLEVH21-1) !since le(k-level) =0; thus with H120

LE(ID_LLEVS+1)=LEH20
LE(ID_LLEVS+2)=LEH20

NATOM(1)=LE(NN1)

!XLAMB and FLU from TRANSIC2

RETURN
END
!
!-----------------------------------------------------------------------
!
subroutine vcs_balmer(xlamb,ll,lu,nws,dws,nts,ts,nes,es,nps,p)
! ***
! **********************************************************************
! ***
! ***	Read VCS tables, interpolate onto common wavelength grid, &
! ***   and transform to p(nu)d(nu) 
! ***
! ***	From A&AS paper `Extended VCS Stark broadening tables for hydrogen
! ***	by Michael Lemke (michael@io.as.utexas.edu,
! ***                     ai26@a400.sternwarte.uni-erlangen.de)
! ***
! ***
! ***    adapted May 2013 by JP
! ***    read according to original data format Nov 2022 by JP
! ***
! ***	 2-AUG-1996 18:00:10.47		ML / Bamberg.
! ***
! **********************************************************************
! ***

USE nlte_type
USE nlte_dim
USE fund_const, ONLY: clight

implicit none  

logical, parameter :: optout=.false. !if true, original values will be tabulated

integer(i4b), parameter :: maxw=id_maxww,maxt=id_maxtt,maxne=id_maxne,maxps=id_maxps, &
&                          nnws=35

real(dp) :: xlamb

real(dp) :: dws(maxw), ts(maxt), es(maxne), p(maxps), ddws(0:nnws-1)

integer(i4b) :: nws1, nts, nes, nps, nws, ll, lu


integer(i4b), parameter :: pmne = 17, pmt = 7, pmp = 65, mline = 20  
                                             ! max # of n_e
                                             ! max # of T
                                             ! max # of profile points
                                             ! max # of H lines
                                             ! VCS profiles

real(dp), parameter :: f0const = 2.5985d0*4.8032d-10

real(dp) :: svcs (pmt, pmne, 0:pmp), alpha(pmp), alpha_loc(pmp), svcs_loc(pmt, pmne, 0:nnws-1)
                                             ! log Delta_alpha_min

real(dp) :: log_alpha0 (mline), log_ne0 (mline), log_t0 (mline)  
                                             ! log n_e_min
                                             ! log T_min
                                             ! Delta log Delta_alpha

real(dp) :: log_alpha_inc (mline), log_t_inc (mline), log_ne_inc (mline)
                                             ! Delta log n_e
                                             ! Delta log T
integer(i4b) :: mp (mline), mne (mline), mt (mline)
integer(i4b), dimension(mline) :: nlow, nup

integer(i4b) :: i, j, k, line, nline, iline, it, kk, nnl, nnu

real(dp) :: ne, f0, x, q, q1, renorm

character (len=1) :: null (pmt)  

character*11 :: charline

character*(*), parameter :: fpath='../inicalc/DATA/'

data ddws/0.d0,1.5849d-3,2.5119d-3,3.9811d-3,6.3096d-3,1.d-2, &
&     1.5849d-2,2.5119d-2,3.9811d-2,6.3096d-2,1.d-1,1.585d-1, &
&     1.995d-1,2.512d-1,3.162d-1,3.981d-1,5.012d-1,6.310d-1, &
&     7.943d-1,1.00d0,1.259d0,1.585d0,1.995d0,2.512d0,3.981d0, &
&     6.310d0,1.0d1,1.585d1,2.512d1,3.891d1,6.310d1,1.0d2,1.585d2, &
&     2.512d2,3.981d2/


!
!       READ IN VCS ARRAYS
!
open(1,file=fpath//'balmer_lemke.dat',status='OLD')  

read (1, * ) nline  
if (nline.gt.mline) then  
write (*,  * ) 'Table too big.  Not more than ', mline, ' lines.'  
   stop ' Table too big!' 
endif  

do i = 1, nline  
read (1, * ) nlow (i), nup (i), log_alpha0 (i), log_ne0 (i), log_t0 ( &
 i), log_alpha_inc (i), log_ne_inc (i), log_t_inc (i), mp (i), &
 mne (i), mt (i)
if (mp (i) .gt.pmp.or.mne (i) .gt.pmne.or.mt (i) .gt.pmt) then  
   write (*, * ) 'Table too big in one of these:'  
   write (*, * ) 'mp:', mp (i) , ' >', pmp  
   write (*, * ) 'mne:', mne (i) , ' >', pmne  
   write (*, * ) 'mt:', mt (i) , ' >', pmt  
   stop ' Table too big in one of these!'  
endif  
enddo  

do line = 1, nline  
read (1,fmt='(A11)') charline
read(charline(4:5),'(i2)') nnl
read(charline(10:11),'(i2)') nnu

if (nnl.ne.nlow (line) .or.nnu.ne.nup (line) ) then  
   write (*, * ) 'Inconsistency in table for', nlow (line) , ' ->', &
    nup (line)
   stop ' Inconsistency in stark_balmer_lemke!'  
endif  
read (1, * ) ( ( (svcs (i, j, k), k = 0, mp (line) ), &
 i = 1, mt (line) ), j = 1, mne (line) )

if(nnl.eq.ll .and. nnu.eq.lu) goto 100

enddo  
stop ' line not found stark_balmer_lemke'

100 close(1)
iline=line

nws1=mp(iline)
nts=mt(iline)
nes=mne(iline)
nws=nnws

nps=nts*nes*nws ! and not nws1

if(nts.gt.maxt)  stop ' nts > maxt'
if(nes.gt.maxne) stop ' nes > maxne'
if(nps.gt.maxps) stop ' nps > maxps'

do k = 1,nws1  
  alpha(k) = log_alpha0 (iline) + log_alpha_inc (iline) * (k - 1)  
enddo

do i = 1, nts  
  it = nint (10.d0**(log_t0 (iline) + log_t_inc (iline) * (i - 1) ) )
  ts(i)=float(it)
enddo

do j = 1, nes  
  es(j) = 10.d0**(log_ne0 (iline) + log_ne_inc (iline) * (j - 1) )  
enddo

line = iline  
write (*,  * ) 'nl = ', nlow (line) , ';  nu = ', nup (line),' Stark broadening according to Lemke (1997)'

if(optout) then
do j = 1, nes  
ne = 10.d0**(log_ne0 (line) + log_ne_inc (line) * (j - 1) )  
write (*, * ) ne, ' cm^-3'  
! svcs(i,j,0): quality flag (0 = OK)
write (*, '(a5,1x,a8,1x,7(:'' ('',i1,'')'',i6))') 'alpha', &
          'lambda', (abs (int (svcs (i + 1, j, 0) ) ) , &
           nint (10.d0**(log_t0 (line) + log_t_inc (line) * i) ) , i = 0, nts  - 1)

f0 = f0const * ne** (2.d0 / 3.d0)  

do k = 1, nws1  
  x = log_alpha0 (line) + log_alpha_inc (line) * (k - 1)  
  do i = 1, nts  
    null (i) = ' '  
  enddo  
write (*, '(f5.1,1x,1p,e8.3e1,1x,0p,7(f9.5,a))') x, 10.d0**x * f0, &
& (real (svcs (i, j, k) ) , null (i) , i = 1, nts)
enddo  
write (*, * )  
enddo  

endif

! now we interpolate the profile functions onto a common wavelength grid.
! interpolation is linear in log profile vs. log dalpha

! as a reference wavelength grid, we use the one from subr. GRIEM
! (consistent with the one from thom_new.dat)

! interpolation for all models

! rember: nws1 is number of freq. points in original grid
!       : nws is number of freq. points for output grid, with indices(0:nws-1)


do j=1,nes
  !recalculate alpha for given lambda 
  f0 = f0const * es(j)** (2.d0 / 3.d0)  

  do k=1,nws-1
    alpha_loc(k)=log10(ddws(k)/f0)
  enddo

  do k=1,nws-1
      if(alpha_loc(k).lt.alpha(1)) then
        do i=1,nts
          svcs_loc(i,j,k)=svcs(i,j,1) ! constant for (very) low dw
        enddo
      else 
        do kk=1,nws1-1
           if(alpha_loc(k).ge.alpha(kk) .and. alpha_loc(k).le.alpha(kk+1)) exit
        enddo       
!end condition
        if(kk.eq.nws1) kk=nws1-1 
        q=(alpha_loc(k)-alpha(kk))/(alpha(kk+1)-alpha(kk))
        q1=1.d0-q
        do i=1,nts
          svcs_loc(i,j,k)=q1*svcs(i,j,kk) + q*svcs(i,j,kk+1)         
        enddo
      endif
  enddo

! value at dlam = 0, taken from first entry at orignal table
  do i=1,nts
    svcs_loc(i,j,0)=svcs(i,j,1)
  enddo

!for tests
!  print*,j,log10(es(j))
!  k=0
!  print*,alpha_loc(1)-10.,svcs_loc(1,j,k),svcs_loc(7,j,k)
!  do k=1,nws-1
!    print*,alpha_loc(k),svcs_loc(1,j,k),svcs_loc(7,j,k)
!  enddo

enddo

!finally, write dws (starting with dlam=0)
do k=0,nws-1
  dws(k+1)=ddws(k)
enddo

!... and profile p, including transformation
! remember that all wavelengths (and alpha) are in Angstrom, &
! and that the original normalization is 0.5 (and not unity) for 0<alpha<inf

do j=1,nes
  f0 = f0const * es(j)** (2.d0 / 3.d0)  
  do k=0,nws-1
    renorm=(xlamb+ddws(k))**2/(f0*clight*1.d8)
    renorm=log10(renorm)
      do i=1,nts
        svcs_loc(i,j,k)=svcs_loc(i,j,k)+renorm
      enddo
  enddo  
enddo

! write transformed profile to output p (sequential order)
kk=0
do j=1,nes
  do i=1,nts
    do k=0,nws-1
      kk=kk+1
      p(kk)=svcs_loc(i,j,k)
    enddo
  enddo  
enddo


if (kk.ne.nps) stop ' kk ne nps in vcs_balmer!'

ts(1:nts)=log10(ts(1:nts))
es(1:nes)=log10(es(1:nes))

return

END subroutine vcs_balmer
!
!-----------------------------------------------------------------------
!
subroutine vcs_pb(xlamb,ll,lu,nws,dws,nts,ts,nes,es,nps,p)
! ***
! **********************************************************************
! ***
! ***	Read VCS tables, interpolate onto common wavelength grid, &
! ***   and transform to p(nu)d(nu) 
! ***
! ***	From A&AS paper `Extended VCS Stark broadening tables for hydrogen
! ***	by Michael Lemke (michael@io.as.utexas.edu,
! ***                     ai26@a400.sternwarte.uni-erlangen.de)
! ***
! ***
! ***    adapted for Paschen & Brackett series, Nov 2022 by JP
! ***
! ***	 2-AUG-1996 18:00:10.47		ML / Bamberg.
! ***
! **********************************************************************
! ***

USE nlte_type
USE nlte_dim
USE fund_const, ONLY: clight

implicit none  

logical, parameter :: optout=.false. !if true, original values will be tabulated

integer(i4b), parameter :: maxw=id_maxww,maxt=id_maxtt,maxne=id_maxne,maxps=id_maxps, &
&                          nnws=35

real(dp) :: xlamb

!character*6 :: lablow, labup

real(dp) :: dws(maxw), ts(maxt), es(maxne), p(maxps), ddws(0:nnws-1)

integer(i4b) :: nws1, nts, nes, nps, nws, ll, lu


integer(i4b), parameter :: pmne = 17, pmt = 7, pmp = 65, mline = 20  
                                             ! max # of n_e
                                             ! max # of T
                                             ! max # of profile points
                                             ! max # of H lines
                                             ! VCS profiles

real(dp), parameter :: f0const = 2.5985d0*4.8032d-10

real(dp) :: svcs (pmt, pmne, 0:pmp), alpha(pmp), alpha_loc(pmp), svcs_loc(pmt, pmne, 0:nnws-1)
                                             ! log Delta_alpha_min

real(dp) :: log_alpha0 (mline), log_ne0 (mline), log_t0 (mline)  
                                             ! log n_e_min
                                             ! log T_min
                                             ! Delta log Delta_alpha

real(dp) :: log_alpha_inc (mline), log_t_inc (mline), log_ne_inc (mline)
                                             ! Delta log n_e
                                             ! Delta log T
integer(i4b) :: mp (mline), mne (mline), mt (mline)
integer(i4b), dimension(mline) :: nlow, nup

integer(i4b) :: i, j, k, line, nline, iline, it, kk, nnl, nnu

real(dp) :: ne, f0, x, q, q1, renorm

character (len=1) :: null (pmt)  

character*11 :: charline

character*(*), parameter :: fpath='../inicalc/DATA/'

data ddws/0.d0,1.5849d-3,2.5119d-3,3.9811d-3,6.3096d-3,1.d-2, &
&     1.5849d-2,2.5119d-2,3.9811d-2,6.3096d-2,1.d-1,1.585d-1, &
&     1.995d-1,2.512d-1,3.162d-1,3.981d-1,5.012d-1,6.310d-1, &
&     7.943d-1,1.00d0,1.259d0,1.585d0,1.995d0,2.512d0,3.981d0, &
&     6.310d0,1.0d1,1.585d1,2.512d1,3.891d1,6.310d1,1.0d2,1.585d2, &
&     2.512d2,3.981d2/


!
!       READ IN VCS ARRAYS
!
select case(ll)

case(3)  
open(1,file=fpath//'paschen_lemke.dat',status='OLD')  
case(4)  
open(1,file=fpath//'brackett_lemke.dat',status='OLD')  
case default
  print*,ll
  stop 'wrong lower level in vcs_pb'

end select  

if(lu.gt.10) stop 'upper level > 10 in vcs_pb'

read (1, * ) nline  
if (nline.gt.mline) then  
write (*,  * ) 'Table too big.  Not more than ', mline, ' lines.'  
   stop ' Table too big!' 
endif  

do i = 1, nline  
read (1, * ) nlow (i), nup (i), log_alpha0 (i), log_ne0 (i), log_t0 ( &
 i), log_alpha_inc (i), log_ne_inc (i), log_t_inc (i), mp (i), &
 mne (i), mt (i)
if (mp (i) .gt.pmp.or.mne (i) .gt.pmne.or.mt (i) .gt.pmt) then  
   write (*, * ) 'Table too big in one of these:'  
   write (*, * ) 'mp:', mp (i) , ' >', pmp  
   write (*, * ) 'mne:', mne (i) , ' >', pmne  
   write (*, * ) 'mt:', mt (i) , ' >', pmt  
   stop ' Table too big in one of these!'  
endif  
enddo  

do line = 1, nline  
read (1,fmt='(A11)') charline
read(charline(4:5),'(i2)') nnl
read(charline(10:11),'(i2)') nnu

if (nnl.ne.nlow (line) .or.nnu.ne.nup (line) ) then  
   write (*, * ) 'Inconsistency in table for', nlow (line) , ' ->', &
    nup (line)
   stop ' Inconsistency in table (vcs_pb)!'  
endif  
read (1, * ) ( ( (svcs (i, j, k), k = 0, mp (line) ), &
 i = 1, mt (line) ), j = 1, mne (line) )

if(nnl.eq.ll .and. nnu.eq.lu) goto 100

enddo  
stop ' line not found lemke tables'

100 close(1)
iline=line

nws1=mp(iline)
nts=mt(iline)
nes=mne(iline)
nws=nnws

nps=nts*nes*nws ! and not nws1

if(nts.gt.maxt)  stop ' nts > maxt'
if(nes.gt.maxne) stop ' nes > maxne'
if(nps.gt.maxps) stop ' nps > maxps'

do k = 1,nws1  
  alpha(k) = log_alpha0 (iline) + log_alpha_inc (iline) * (k - 1)  
enddo

do i = 1, nts  
  it = nint (10.d0**(log_t0 (iline) + log_t_inc (iline) * (i - 1) ) )
  ts(i)=float(it)
enddo

do j = 1, nes  
  es(j) = 10.d0**(log_ne0 (iline) + log_ne_inc (iline) * (j - 1) )  
enddo

line = iline  
write (*,  * ) 'nl = ', nlow (line) , ';  nu = ', nup (line),' Stark broadening according to Lemke (1997)'

if(optout) then
do j = 1, nes  
ne = 10.d0**(log_ne0 (line) + log_ne_inc (line) * (j - 1) )  
write (*, * ) ne, ' cm^-3'  
! svcs(i,j,0): quality flag (0 = OK)
write (*, '(a5,1x,a8,1x,7(:'' ('',i1,'')'',i6))') 'alpha', &
          'lambda', (abs (int (svcs (i + 1, j, 0) ) ) , &
           nint (10.d0**(log_t0 (line) + log_t_inc (line) * i) ) , i = 0, nts  - 1)

f0 = f0const * ne** (2.d0 / 3.d0)  

do k = 1, nws1  
  x = log_alpha0 (line) + log_alpha_inc (line) * (k - 1)  
  do i = 1, nts  
    null (i) = ' '  
  enddo  
write (*, '(f5.1,1x,1p,e8.3e1,1x,0p,7(f9.5,a))') x, 10.d0**x * f0, &
& (real (svcs (i, j, k) ) , null (i) , i = 1, nts)
enddo  
write (*, * )  
enddo  

endif

! now we interpolate the profile functions onto a common wavelength grid.
! interpolation is linear in log profile vs. log dalpha

! as a reference wavelength grid, we use the one from subr. GRIEM
! (consistent with the one from thom_new.dat)

! interpolation for all models

! rember: nws1 is number of freq. points in original grid
!       : nws is number of freq. points for output grid, with indices(0:nws-1)


do j=1,nes
  !recalculate alpha for given lambda 
  f0 = f0const * es(j)** (2.d0 / 3.d0)  

  do k=1,nws-1
    alpha_loc(k)=log10(ddws(k)/f0)
  enddo

  do k=1,nws-1
      if(alpha_loc(k).lt.alpha(1)) then
        do i=1,nts
          svcs_loc(i,j,k)=svcs(i,j,1) ! constant for (very) low dw
        enddo
      else 
        do kk=1,nws1-1
           if(alpha_loc(k).ge.alpha(kk) .and. alpha_loc(k).le.alpha(kk+1)) exit
        enddo       
!end condition
        if(kk.eq.nws1) kk=nws1-1 
        q=(alpha_loc(k)-alpha(kk))/(alpha(kk+1)-alpha(kk))
        q1=1.d0-q
        do i=1,nts
          svcs_loc(i,j,k)=q1*svcs(i,j,kk) + q*svcs(i,j,kk+1)         
        enddo
      endif
  enddo

! value at dlam = 0, taken from first entry at orignal table
  do i=1,nts
    svcs_loc(i,j,0)=svcs(i,j,1)
  enddo

!for tests
!  print*,j,log10(es(j))
!  k=0
!  print*,alpha_loc(1)-10.,svcs_loc(1,j,k),svcs_loc(7,j,k)
!  do k=1,nws-1
!    print*,alpha_loc(k),svcs_loc(1,j,k),svcs_loc(7,j,k)
!  enddo

enddo

!finally, write dws (starting with dlam=0)
do k=0,nws-1
  dws(k+1)=ddws(k)
enddo

!... and profile p, including transformation
! remember that all wavelengths (and alpha) are in Angstrom, &
! and that the original normalization is 0.5 (and not unity) for 0<alpha<inf

do j=1,nes
  f0 = f0const * es(j)** (2.d0 / 3.d0)  
  do k=0,nws-1
    renorm=(xlamb+ddws(k))**2/(f0*clight*1.d8)
    renorm=log10(renorm)
      do i=1,nts
        svcs_loc(i,j,k)=svcs_loc(i,j,k)+renorm
      enddo
  enddo  
enddo

! write transformed profile to output p (sequential order)
kk=0
do j=1,nes
  do i=1,nts
    do k=0,nws-1
      kk=kk+1
      p(kk)=svcs_loc(i,j,k)
    enddo
  enddo  
enddo


if (kk.ne.nps) stop ' kk ne nps in vcs_pb!'

ts(1:nts)=log10(ts(1:nts))
es(1:nes)=log10(es(1:nes))

return

END subroutine vcs_pb
!
!-----------------------------------------------------------------------
!
function igenio(kk,ii)  

USE nlte_type  
USE nlte_dim  
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&     ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS

implicit none  
!
!------ this function returns the value of 'general ion' for atom kk and
!       ion ii
!
!     .. scalar arguments ..
integer(i4b) ::  ii,kk,igenio  
!     ..
!     .. local scalars ..
integer(i4b) ::  i  
!     ..
!     .. local arrays ..
real(dp) ::  abund(id_atoms),fl(id_llevs),flx(id_xlevs), &
&                 gl(id_llevs),glx(id_xlevs),gs0(id_slevs), &
&                 gs1(id_slevs),qd(id_slevs),ryd(id_slevs), &
&                 weight(id_atoms),zeff(id_atoms),zl(id_llevs)
!     ..

igenio = 0  

do i = 1,kk - 1  
     igenio = igenio + nions(i)  
end do  

igenio = igenio + ii  
if (igenio.gt.iong) stop ' error in general ion - igenio '  

return  

end
