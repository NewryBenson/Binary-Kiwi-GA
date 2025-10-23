! version 1.7 (Oct 2020)
! compatible with gfortran
!
! version 1.6 (March 2017)
! one more freq. point at 303.797 (HeII Ly-alpha)
!
! version 1.5 (Jan 2015)
! frequency grid changed between 1600 and 911, consistent with nlte.f90
! checked July 2016 that v. 1.3.4  from Feb 2015 was already correctly included:
! FRESCAL_0 CHANGED FOR BETTER RESOLUTION 
!
! version 1.4 
! allows for X-ray treatment (FRESCAL0 adapted for higher frequencies and K-shells
!
! version 1.3.3 
! (July 2013): NFESC set to 500 (for new formalsol)
!
! version 1.3.2
! (April 2013): Improved treatment of freq. spacing between 911 and 925 A, &
! to ensure a maximum separation of 3 A (minimum box size for nsubinter = 120
! is 3.6 A, can lead to problems with nlambox_ly when many ions included)
!
! version 1.3.1
! (March 2013) - number of freq. points changed (rounded + 200)
! higher resolution in UV range until Lyman edge
! new default values for nd1 and np1

! version 1.3 until March 2013
MODULE nlte_param
!
USE nlte_type
IMPLICIT NONE
!
! fix parameters for dimes etc.
!
! nlte
!INTEGER(I4B), PARAMETER :: NF2 = 550             !freq. fine grid (HHE)
INTEGER(I4B), PARAMETER :: NF2 = 4000             !freq. fine grid (large atoms)
INTEGER(I4B), PARAMETER :: LEVMAX = 4
INTEGER(I4B), PARAMETER :: LEVMIN1 = 3
INTEGER(I4B), PARAMETER :: LEVMIN2 = 2
INTEGER(I4B), PARAMETER :: NMAXION = 5           !maximum number of ions per
!                                                 atom (without highest one) 
!INTEGER(I4B), PARAMETER :: ND1 = 47              !depth points (for purely optical analysis) 
!INTEGER(I4B), PARAMETER :: NP1 = 52              !p-rays
INTEGER(I4B), PARAMETER :: ND1 = 51              !suggested if IR should be calculated 
INTEGER(I4B), PARAMETER :: NP1 = 56              !p-rays
!INTEGER(I4B), PARAMETER :: ND1 = 67 !depth points for high resol. of photosphere
!INTEGER(I4B), PARAMETER :: NP1 = 72 !p-rays for high resol. of photosphere
INTEGER(I4B), PARAMETER :: NFCMF = 21            !cmf-freq.
! formal
INTEGER(I4B), PARAMETER :: NFFORMAL = 3000       !depth points (fine grid)
INTEGER(I4B), PARAMETER :: NCFORMAL = 10         !core rays 
INTEGER(I4B), PARAMETER :: NFOBS = 161           !obs. frame freq.
INTEGER(I4B), PARAMETER :: NFESC = 500           !cmf frame freq. 
! stark
INTEGER(I4B), PARAMETER :: NFSTARK = 80
INTEGER(I4B), PARAMETER :: NTSTARK = 50
INTEGER(I4B), PARAMETER :: NESTARK = 50
INTEGER(I4B), PARAMETER :: NTOTSTARK = 10000
END MODULE nlte_param
!
!----------------------------------------------------------------------
!
MODULE dimes_var 
!
USE nlte_type
IMPLICIT NONE
!
!dimensiones
!
INTEGER(I4B) ::  NAT=0
INTEGER(I4B) ::  IONG=0
INTEGER(I4B) ::  NL=0
INTEGER(I4B) ::  NLX=0
INTEGER(I4B) ::  NS=0
INTEGER(I4B) ::  INDTT=0
INTEGER(I4B) ::  INDEX1=0,INDEX2=0,INDEX3=0,INDEX4=0,INDEX5=0
INTEGER(I4B) ::  INDTR=0
INTEGER(I4B) ::  NDAT=0
INTEGER(I4B) ::  NPROF=0
!
!inout
!
INTEGER(I4B) ::  IU
INTEGER(I4B) ::  IOU
!
! xray-treatment
INTEGER(I4B), PARAMETER :: N_KEDGES = 59 ! in ATOMDAT_NEW/k_shell_data
INTEGER(I4B), PARAMETER :: K_NMIN = 3, K_NMAX = 8  ! min/max ion. stage to be included in freq. grid

REAL(DP), DIMENSION(N_KEDGES) :: ETH,SIGMA,S,ZEFF
INTEGER(I4B), DIMENSION(N_KEDGES) :: Z,N

END MODULE dimes_var
!
!----------------------------------------------------------------------
!
MODULE prince_var
!
USE nlte_type
IMPLICIT NONE
!----------------------------------------------------------------------
!atdnin
!
INTEGER(I4B) ::  NL=0,NLX=0,NS=0,NAT=0
INTEGER(I4B) ::  ION,IONG=0
INTEGER(I4B), DIMENSION(:), ALLOCATABLE ::  LA0,LA1
INTEGER(I4B), DIMENSION(:), ALLOCATABLE ::  IXA0,IXA1
INTEGER(I4B), DIMENSION(:), ALLOCATABLE ::  ISA0,ISA1
INTEGER(I4B), DIMENSION(:), ALLOCATABLE ::  NIONS
INTEGER(I4B), DIMENSION(:), ALLOCATABLE ::  IFIRSL,IFIRX,IFIRS
INTEGER(I4B), DIMENSION(:), ALLOCATABLE ::  KL,LE,LI
INTEGER(I4B), DIMENSION(:), ALLOCATABLE ::  KLX,LIX
INTEGER(I4B), DIMENSION(:), ALLOCATABLE ::  LIS,NS0,NS1
!----------------------------------------------------------------------
!atdnre
!
REAL(DP), DIMENSION(:), ALLOCATABLE :: ZEFF,WEIGHT,ABUND
REAL(DP), DIMENSION(:), ALLOCATABLE :: GL,FL,ZL
REAL(DP), DIMENSION(:), ALLOCATABLE :: GLX,FLX
REAL(DP), DIMENSION(:), ALLOCATABLE :: GS0,GS1,RYD,QD
!----------------------------------------------------------------------
!atomdatc
!
CHARACTER*6, DIMENSION(:), ALLOCATABLE :: LABAT 
CHARACTER*6, DIMENSION(:), ALLOCATABLE :: LABL,LKL 
CHARACTER*6, DIMENSION(:), ALLOCATABLE :: LABX,LKX 
CHARACTER*6, DIMENSION(:), ALLOCATABLE :: LABLS,KLS 
!----------------------------------------------------------------------
!dat
!
INTEGER(I4B) :: NDAT=0
REAL(DP), DIMENSION(:), ALLOCATABLE :: DATA
!----------------------------------------------------------------------
!frec
!
INTEGER(I4B) :: NF=0 !only needed for perfil
!----------------------------------------------------------------------
!inout
!
INTEGER(I4B) ::  IU,IOU
!----------------------------------------------------------------------
!tranin
!
INTEGER(I4B) :: NLONG,INDTT=0,INDTR=0
INTEGER(I4B) :: INDEX1=0,INDEX2=0,INDEX3=0,INDEX4=0,INDEX5=0
INTEGER(I4B), DIMENSION(:), ALLOCATABLE :: IAUX2
INTEGER(I4B), DIMENSION(:), ALLOCATABLE :: IFORM1,NUMD1,INDAT1,LABL1,LABU1
INTEGER(I4B), DIMENSION(:), ALLOCATABLE :: IFORM2,NUMD2,INDAT2
INTEGER(I4B), DIMENSION(:), ALLOCATABLE :: IFORM3,NUMD3,INDAT3,LABL3
INTEGER(I4B), DIMENSION(:), ALLOCATABLE :: IFORM4,NUMD4,INDAT4,LABL4
INTEGER(I4B), DIMENSION(:), ALLOCATABLE :: IFORM5,NUMD5,INDAT5,IPARE5
INTEGER(I4B), DIMENSION(:,:), ALLOCATABLE :: IAUX1
!----------------------------------------------------------------------
!tranlo
!
LOGICAL :: QCL,QDB,QRT
!----------------------------------------------------------------------
!tranre
!
REAL(DP) :: ANCDO,TL,WLC
REAL(DP), DIMENSION(:), ALLOCATABLE :: DL
REAL(DP), DIMENSION(:), ALLOCATABLE :: FRECIN,FRECFN
!----------------------------------------------------------------------
!transc
!
CHARACTER*6, DIMENSION(:), ALLOCATABLE :: LABL2,LABU2,PAREN2 
CHARACTER*6, DIMENSION(:), ALLOCATABLE :: PAREN3 
CHARACTER*6, DIMENSION(:), ALLOCATABLE :: PAREN4 

END MODULE prince_var

PROGRAM DIMENATOM  
!
!     programa para calcular las dimensiones necesarias para los datos
!     atomicos
!
!     14/01/99 changed to f90 by (j.puls)
!     02/03/99 more f90 changes
!     11/03/99 obsolete fortran features removed (j.puls)
!  
!     version 1.1 (march 2006) : initial freq. grid from DETAIL input
!                 no longer read!!! (can be present or absent in file)
!                 stop enforced if NMAXION is too low
!
!                 calls modified version of princesa to read in all data
!                 these data are used to calculate a pessimistic guess
!                 of ID_FREC1 (in a modified version of FRESCAL)
!                 (in order to save disk space when saving the model-files)
!                 NOTE: ID_FREC2 is still input data, since only used inside
!                       FASTWIND 
USE nlte_type
USE dimes_var, ONLY: IU,IOU
USE ffr_error

IMPLICIT NONE

!     ..
!     .. local scalars ..
INTEGER(I4B) NF1

CHARACTER DC*6,FICHERO*32,RET*4  
!     ..
!     .. external subroutines ..
EXTERNAL FFRACC,FFRCDC,FFROUT,FREQCON,OUTSUB,RDATOM_0,TRANSIC_0  
!     ..
!     ..

IU = 37  
IOU = 38  
DC = ':T'  

OPEN (1,FILE='ATOM_FILE',STATUS='OLD')  
REWIND 1  
READ (1,FMT='(A)') FICHERO  
CLOSE (1)  

OPEN (UNIT=IU,FILE=FICHERO,FORM='FORMATTED',STATUS='OLD')  
OPEN (UNIT=IOU,FILE='control_'//FICHERO,STATUS='UNKNOWN')  

CALL FFROUT(IOU,RET)  
IF (ON_ERROR(RET))  GOTO 9000

CALL FFRACC(IU,RET)  
IF (ON_ERROR(RET))  GOTO 9100

CALL FFRCDC(DC)  

RLOOP: DO  

CALL RDATOM_0(RET)  
IF (RET.NE.'RET0'.AND.RET.NE.'RET3')  GOTO 9200
IF (RET.EQ.'RET3') GOTO 9990

CALL TRANSIC_0(RET)  
IF (ON_ERROR(RET))  GOTO 9200

END DO RLOOP

!       error in output unit exit
 9000 CONTINUE  
WRITE (*,FMT='(A)') ' ERROR IN OUTPUT UNIT NUMBER '  
STOP

!       error in input unit exit
 9100 CONTINUE  
WRITE (*,FMT='(A)') ' ERROR IN INPUT UNIT NUMBER '  
STOP

!       error conditions
9200 SELECT CASE(RET)
CASE ('RET1') !       end of file "no esperado" exit  
  WRITE (*,FMT='(A)') ' EOF NO ESPERADO. ALGO FALLA! '  
  STOP
CASE ('RET2') !       error in ffr subroutines exit  
  WRITE (*,FMT='(A)') ' ERROR IN FFR SUBROUTINES '  
  STOP
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN DIMENATOM'
END SELECT

!       normal exit: end of file
 9990 CONTINUE  
WRITE (*,FMT='(A)') ' FILE IS OVER. SUCCESSFUL!! '  

CLOSE (IU)  
CLOSE (IOU)  

! read in all data
CALL PRINCE_RUN_FIRST

! read k-shell data
! commented out by JO; only required if Xrays included
!CALL READ_KSHELL_DATA

! calculate pessimistic guess for NF1
CALL FRESCAL_0(NF1)

! write nlte_dim.f90
CALL OUTSUB(FICHERO,NF1)  

END


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


SUBROUTINE RDATOM_0(RETCHAR)  
!       lee y almacena los datos atomicos
!       lista de variables (por orden de appearance):
!               nat: numero de atomo
!               ion: numero de ion dentro del atomo
!               iong: numero de ion en el computo general
USE nlte_type
USE nlte_param, ONLY: NMAXION
USE dimes_var, ONLY: NAT,IONG,NL,NLX,NS
USE ffr_error
IMPLICIT NONE

!     .. local scalars ..
REAL(DP) :: REALVAR  
INTEGER(I4B) :: INTEG,ION,NC  
CHARACTER LABEL*4,KEY*6,RETCHAR*4,RET*4  
!     ..
!     .. external subroutines ..
EXTERNAL FFRKEY,FFRLOC,FFRNUM  
!     ..

LABEL = 'ATOM'  
CALL FFRLOC(LABEL,RET)  
SELECT CASE(RET)
CASE('RET1')
  GOTO 9980
CASE('RET2')
  GOTO 9990
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN IN RDATOM'
END SELECT

NAT = NAT + 1  

ION = 0  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRNUM(REALVAR,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRNUM(REALVAR,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRNUM(REALVAR,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  100 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

!------------- l  levels
IF (KEY.EQ.'L') THEN  
     ION = ION + 1  
     IONG = IONG + 1  

 1000      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     IF (KEY.EQ.'0') THEN  
          GO TO 100  
     ELSE  
          NL = NL + 1  
          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          CALL FFRKEY(KEY,NC,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          GO TO 1000  


     END IF  

!----------------- x  levels
ELSE IF (KEY.EQ.'X') THEN  
 2000      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     IF (KEY.EQ.'0') GO TO 100  
     NLX = NLX + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     GO TO 2000  

!-------------- s  levels
ELSE IF (KEY.EQ.'S') THEN  
 3000      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     IF (KEY.NE.'0') THEN  
          NS = NS + 1  
          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          CALL FFRKEY(KEY,NC,RET)  
          IF (ON_ERROR(RET)) GOTO 9990

          GO TO 3000  

     END IF  

     GO TO 100  

!--------------------  k  levels
ELSE IF (KEY.EQ.'K') THEN  
     NL = NL + 1  
     ION = ION + 1  
     IONG = IONG + 1  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

END IF  

!       the atom is over.
RETCHAR='RET0'

IF(ION-1.GT.NMAXION) THEN
  PRINT*,' INCREASE NMAXION TO ',ION-1
  STOP ' NMAXION TOO LOW!'
ENDIF

RETURN  
!       no hay mas atomos que leer. ya se ha leido todo el fichero. es
!       cuando acaba el programa principal (prince)
 9980 CONTINUE  

RETCHAR='RET3'
RETURN 

! error handling
9990 SELECT CASE(RET)
CASE('RET1') !       end of file exit 
  WRITE (*,FMT='(A)') ' END OF FILE '  
  RETCHAR='RET1'
  RETURN
CASE('RET2') !       error exit 
  WRITE (*,FMT='(A)') 'ERROR IN FFR SUBROUTINES '  
  RETCHAR='RET2'
  RETURN
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN RDATOM'
END SELECT

END


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


SUBROUTINE TRANSIC_0(RETCHAR)  
!       subrutina que lee los datos referentes a las transiciones, los
!       almacena y escribe (para control) en un fichero de salida.
!       espero poner suficientes explicaciones a lo largo de la subr.
!       de modo que este todo lo mas claro posible.
!       variables (por orden de aparicion):
!          tabla: posibles tipos de transiciones
!          qdb: bandera para el caso db (muestreo en anchuras doppler)
!          nlong: numero de puntos en el muestreo del perfil
!          qcl: bandera para el caso cl (longitud central)
!          qrt: bandera para el caso de transicion rbb o rbx
!          indtt: indice para llevar la cuenta del total de transiciones
USE nlte_type
USE dimes_var, ONLY: NPROF,INDTT,INDTR,NDAT, &
&                    INDEX1,INDEX2,INDEX3,INDEX4,INDEX5,IU
USE ffr_error
IMPLICIT NONE

!     .. local scalars ..
REAL(DP) :: REALVAR,TL,WLC  
INTEGER(I4B) :: I,IFORM,INTEG,M,NC,NDATOS,NLONG  
LOGICAL QCL,QDB,QRT  
CHARACTER KEY*6,PARE*6,RETCHAR*4,RET*4  
!     ..
!     .. local arrays ..
CHARACTER TABLA(8)*6  
!     ..
!     .. external subroutines ..
EXTERNAL FFRKEY,FFRNUM,IDENT_0  
!     ..
!     .. intrinsic functions ..
INTRINSIC MAX  
!     ..
!     .. data statements ..
DATA TABLA/'RBB','CBB','RBX','CBX','CBS','CBF','RBF','RFF'/  
!     ..

! JP changed May 3rd 2001
   QCL = .FALSE.

   10 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'TY') THEN  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL IDENT_0(KEY,TABLA,8,M)  
     CALL FFRNUM(REALVAR,IFORM,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL FFRNUM(REALVAR,NDATOS,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

!          son los numeros de formula y de datos, respectivamente

     GO TO 10  

!       muestreo en anchuras doppler. el muestreo queda en dl.
ELSE IF (KEY.EQ.'DB') THEN  
     QDB = .TRUE.  
     CALL FFRNUM(REALVAR,NLONG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     NPROF = MAX(NPROF,NLONG)  
     DO I = 1,NLONG  
          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
     END DO

     GO TO 10  

!       muestreo en angstroms
ELSE IF (KEY.EQ.'DL') THEN  
     QDB = .FALSE.  
     CALL FFRNUM(REALVAR,NLONG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     DO I = 1,NLONG  
          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
     END DO

     GO TO 10  

!       temperatura de linea
ELSE IF (KEY.EQ.'TL') THEN  
     CALL FFRNUM(TL,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     GO TO 10  

!       longitud central de la linea
ELSE IF (KEY.EQ.'CL') THEN  
     QCL = .TRUE.  

     GO TO 10  
!       tipo de transicion
ELSE  
     CALL IDENT_0(KEY,TABLA,8,M)  
     IF (M.EQ.0) THEN  
          GO TO 10000  

     ELSE  
          QRT = .FALSE.  
          IF ((M.EQ.1) .OR. (M.EQ.3)) QRT = .TRUE.  
          GO TO (100,100,200,200,300,300,400,500) M  
! goto calculado, para que haga una cosa u otra segun el tipo de trans.


     END IF  


END IF  

!-------------transiciones rbb o cbb ------------
  100 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  105 CONTINUE  
INDTT = INDTT + 1  
INDEX1 = INDEX1 + 1  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

!       transicion radiativa
IF (QRT) THEN  
     IF (QCL) THEN
          CALL FFRNUM(WLC,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
     END IF  

     INDTR = INDTR + 1  
!                nf=nf+nlong
END IF  

!       lectura de los datos
NDAT = NDAT + 1  
NDAT = NDAT + 1  
DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
END DO  

!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') THEN  
     QCL = .FALSE.  
     GO TO 10  
END IF  


GO TO 105  

!--------------- transiciones rbx o cbx ------------
  200 CONTINUE  
CALL FFRKEY(PARE,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  205 CONTINUE  
INDTT = INDTT + 1  
INDEX2 = INDEX2 + 1  


CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

!       transicion radiativa
IF (QRT) THEN  
     IF (QCL) THEN  
          CALL FFRNUM(WLC,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
     END IF  

END IF  

!       lectura de datos
DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
END DO  

!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') THEN  
     QCL = .FALSE.  
     GO TO 10  
END IF  

GO TO 205  

!------------- transiciones  cbs o cbf ------------------
  300 CONTINUE  
CALL FFRKEY(PARE,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  305 CONTINUE  
INDTT = INDTT + 1  
INDEX3 = INDEX3 + 1  

!       lectura de datos
DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
END DO  

!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') GO TO 10  

GO TO 305  

!------------------ transicion  rbf -------------------
  400 CONTINUE  
CALL FFRKEY(PARE,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  405 CONTINUE  

INDTT = INDTT + 1  
INDEX4 = INDEX4 + 1  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

!       leer datos
DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
END DO  
!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') GO TO 10  

GO TO 405  

!----------------- transiciones rff ------------------------
  500 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  505 CONTINUE  

INDTT = INDTT + 1  

INDEX5 = INDEX5 + 1  
!       lectura de datos
DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
END DO  

!       leer nueva keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') GO TO 10  

GO TO 505  

! error handling
9990 SELECT CASE(RET)
CASE('RET1') !       end of file exit 
  WRITE (*,FMT='(A)') ' END OF FILE '  
  RETCHAR='RET1'
  RETURN
CASE('RET2') !       error exit 
  WRITE (*,FMT='(A)') 'ERROR IN FFR SUBROUTINES '  
  RETCHAR='RET2'
  RETURN
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN TRANSIC'
END SELECT

10000 CONTINUE  

BACKSPACE IU  
RETCHAR='RET0'

RETURN  

END


!------------------------------------------------------------------


SUBROUTINE IDENT_0(KEY,TABLE,NMAX,J)  
!        compara el caracter key con los de la tabla 'table', y devuelve
!        en j el indice que corresponda a key en table, si esta, o 0 en
!        otro caso.
!     .. scalar arguments ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) :: J,NMAX  
CHARACTER KEY*6  
!     ..
!     .. array arguments ..
CHARACTER TABLE(*)*6  
!     ..
!     .. local scalars ..
INTEGER(I4B) :: I  
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


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


SUBROUTINE OUTSUB(FICHERO,NF1)  
USE nlte_type
USE nlte_param
Use dimes_var, ONLY: NAT,IONG,NL,NLX,NS,INDEX1,INDEX2,INDEX3,INDEX4, &
&              INDEX5,INDTT,INDTR,NDAT,NPROF

IMPLICIT NONE
!
!     For fix parameters (freq., depth points etc., see module nlte_param) 
!
!     .. scalar arguments ..
INTEGER(I4B) :: NF1
CHARACTER FICHERO*32  
!     ..
!     .. local scalars ..
INTEGER(I4B) :: IREST,N,NDEP  
CHARACTER ALPHA*1  
!     ..
!     .. intrinsic functions ..
INTRINSIC MOD  
!     ..

ALPHA = ' '  

OPEN (UNIT=3,FILE='nlte_dim.f90',STATUS='UNKNOWN')  

WRITE (3,FMT='(A)') 'Module nlte_dim'  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') 'Use nlte_type'  
WRITE (3,FMT='(A)') 'IMPLICIT NONE'  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A,2X,A)') '!  DIMENSIONS FROM FILE',FICHERO  
WRITE (3,FMT='(A)') ALPHA  
WRITE (3,FMT='(A)') ALPHA  

IF (NAT.EQ.0) STOP ' NAT = 0!'
WRITE (3,FMT='(A)') '!    NUMBER OF ATOMS'  
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_ATOMS = ',NAT
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF FREQUENCIES (coarse mesh, modify in case)'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_FREC1 = ',NF1  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF FREQUENCIES (fine mesh, input)'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_FREC2 = ',NF2  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    LEVMAX (see frescal)'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_LEVMA = ',LEVMAX  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    LEVMIN1 (see frescal and fresfin)'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_LEVM1 = ',LEVMIN1  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    LEVMIN2(see frescal)'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_LEVM2 = ',LEVMIN2  
WRITE (3,FMT='(A)') ALPHA  

IF (IONG.EQ.0) STOP ' IONG = 0!'  
WRITE (3,FMT='(A)') '!    NUMBER OF IONS'  
WRITE (3,FMT='(A,I3)') 'INTEGER(I4B), PARAMETER :: ID_IONES = ',IONG  
WRITE (3,FMT='(A)') ALPHA  

IF (NL.EQ.0) STOP ' NL = 0!'
WRITE (3,FMT='(A)') '!    NUMBER OF L + K LEVELS'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_LLEVS = ',NL  
WRITE (3,FMT='(A)') ALPHA  

IF (NLX.NE.0) THEN
PRINT*,' WARNING!!! X LEVELS FOUND!'
ELSE
NLX = 1
ENDIF
WRITE (3,FMT='(A)') '!    NUMBER OF X LEVELS '  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_XLEVS = ',NLX  
WRITE (3,FMT='(A)') ALPHA  

IF (NS.NE.0) THEN
PRINT*,' WARNING!!! S LEVELS FOUND!'
ELSE
NS = 1
ENDIF
WRITE (3,FMT='(A)') '!    NUMBER OF S LEVELS'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_SLEVS = ',NS  
WRITE (3,FMT='(A)') ALPHA  

IF (INDEX1.EQ.0) STOP ' INDEX1 = 0!'  
WRITE (3,FMT='(A)') '!    NUMBER OF RBB + CBB TRANSITIONS'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_RCBBT = ',INDEX1  
WRITE (3,FMT='(A)') ALPHA  

IF (INDEX2.NE.0) STOP ' RBX AND/OR CBX TRANSITIONS FOUND!'
INDEX2 = 1  
WRITE (3,FMT='(A)') '!    NUMBER OF RBX + CBX TRANSITIONS'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_RCBXT = ',INDEX2  
WRITE (3,FMT='(A)') ALPHA  

IF (INDEX3.EQ.0) STOP ' INDEX3 = 0!'
WRITE (3,FMT='(A)') '!    NUMBER OF CBS + CBF TRANSITIONS'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_CBSFT = ',INDEX3  
WRITE (3,FMT='(A)') ALPHA  

IF (INDEX4.EQ.0) STOP ' INDEX4 = 0!'  
WRITE (3,FMT='(A)') '!    NUMBER OF RBF TRANSITIONS'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_RBFTR = ',INDEX4  
WRITE (3,FMT='(A)') ALPHA  

IF (INDEX5.EQ.0) STOP ' INDEX5 = 0!'  
WRITE (3,FMT='(A)') '!    NUMBER OF RFF TRANSITIONS'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_RFFTR = ',INDEX5  
WRITE (3,FMT='(A)') ALPHA  

IF (INDTT.EQ.0) STOP ' INDTT = 0!'  
WRITE (3,FMT='(A)') '!    NUMBER OF TOTAL TRANSITIONS'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_NTRAN = ',INDTT  
WRITE (3,FMT='(A)') ALPHA  

IF (INDTR.EQ.0) STOP ' INDTR = 0!'
WRITE (3,FMT='(A)') '!   NUMBER OF RBB + RBX TRANSITIONS'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_NTTRD = ',INDTR  
WRITE (3,FMT='(A)') ALPHA  

IF (NDAT.EQ.0) STOP ' NDAT = 0!'
WRITE (3,FMT='(A)') '!    NUMBER OF DATA'  
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_NDATA = ',NDAT  
WRITE (3,FMT='(A)') ALPHA  

IF (NPROF.EQ.0) NPROF = 1  
WRITE (3,FMT='(A)') '!    NUMBER OF FREQ. PER LINE (from input)'
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_NLONG = ',NPROF  
WRITE (3,FMT='(A)') ALPHA  

!       maximum number of ions/atom excluding the highest one
WRITE (3,FMT='(A)') '!    MAX. NUMBER OF IONS PER ATOM EXCL. LAST ONE (set)'  
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_KISAT = ',NMAXION  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF DEPTH POINTS (ND, ND1)'  
NDEP = ND1  
IF (MOD(NDEP,2).EQ.0) STOP ' NUMBER OF DEPTH POINTS MUST BE ODD!!!'
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_NDEPT =  ',ND1  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF P-RAYS'    
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_NPOIN = ',NP1  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF CMF-FREQUENCIES'    
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_NFCMF = ',NFCMF  
WRITE (3,FMT='(A)') ALPHA  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!   DIMENSIONS FOR FORMAL SOLUTION'  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF DEPTH POINTS (FINE MESH)'    
WRITE (3,FMT='(A,I4)') 'INTEGER(I4B), PARAMETER :: ID_DEPFI = ',NFFORMAL  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF CORE RAYS'    
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_CORES = ',NCFORMAL  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF NON-CORE RAYS MODULO 4 (+1)'  

IREST = MOD(NDEP-1,4)  
IF (IREST.EQ.0) THEN  

     N = NDEP/4  
ELSE IF (IREST.EQ.2) THEN  

     N = NDEP/4 + 1  
ELSE  

     STOP ' SOMETHING WAS WRONG IN MY PHILOSOPHY'  

END IF  
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_NOCOR = ',N  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF OBS. FRAME FREQUENCIES'    
WRITE (3,FMT='(A,I3)') ' INTEGER(I4B), PARAMETER :: ID_NFOBS = ',NFOBS  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    NUMBER OF CMF FREQUENCIES (EL. SCAT.)'    
WRITE (3,FMT='(A,I3)') ' INTEGER(I4B), PARAMETER :: ID_NFESC = ',NFESC  
WRITE (3,FMT='(A)') ALPHA  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    DIMENSIONS FOR STARK EFFECT TREATMENT'  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    MAX. NUMBER OF WAVELENGTHS'  
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_MAXWW = ',NFSTARK
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    MAX. NUMBER OF TEMPERATURES'  
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_MAXTT = ',NTSTARK
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    MAX. NUMBER OF ELECTRON DENSITIES' 
WRITE (3,FMT='(A,I2)') 'INTEGER(I4B), PARAMETER :: ID_MAXNE = ',NESTARK  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') '!    MAX. NUMBER IN PERFIL'    
WRITE (3,FMT='(A,I5)') 'INTEGER(I4B), PARAMETER :: ID_MAXPS = ',NTOTSTARK  
WRITE (3,FMT='(A)') ALPHA  

WRITE (3,FMT='(A)') 'END MODULE nlte_dim'  

CLOSE (3)  

RETURN  
END
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!ccc                                                                cccc
!ccc         read atomic data  subroutines (e. santolaya, 1991)     cccc
!ccc                                                                cccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!  15/01/99 change to f90 by j. puls
!  03/03/99 more f90 changes
!  11/03/99 obsolete fortran features removed (j.puls)
!   
!
!     version 1.1 (march 2006) : initial freq. grid from DETAIL input
!                 no longer read!!! (can be present or absent in file)
!
SUBROUTINE PRINCE_RUN_FIRST
!
USE nlte_type
!USE nlte_dim
USE prince_var
USE ffr_error
IMPLICIT NONE
!
!       main program for reading and storaging atomic data
!       variables
!     ..
!     .. local scalars ..
CHARACTER DC*6,FICHERO*32,RET*4
!     ..
!     .. external subroutines ..
EXTERNAL ABU,FFRACC,FFRCDC,FFROUT,RDATOM,TRANSIC  
!     ..

CALL ALLOC

IU = 37  
IOU = 38  
DC = ':T'  
OPEN (1,FILE='ATOM_FILE',STATUS='OLD')  
REWIND 1  
READ (1,FMT='(A)') FICHERO  

CLOSE (1)  
OPEN (UNIT=IU,FILE=FICHERO,FORM='FORMATTED',STATUS='OLD')  
OPEN (UNIT=IOU,FILE='control.dat',STATUS='UNKNOWN')  

CALL FFROUT(IOU,RET)  
IF (ON_ERROR(RET))  GOTO 9000

CALL FFRACC(IU,RET)  
IF (ON_ERROR(RET))  GOTO 9100

CALL FFRCDC(DC)  

RLOOP: DO  

CALL RDATOM(RET)  
IF (RET.NE.'RET0'.AND.RET.NE.'RET3')  GOTO 9200
IF (RET.EQ.'RET3') GOTO 9990

CALL TRANSIC(RET)  
IF (ON_ERROR(RET))  GOTO 9200

END DO RLOOP

!       error in output unit exit
 9000 CONTINUE  
WRITE (*,FMT='(A)') ' ERROR IN OUTPUT UNIT NUMBER '  
STOP

!       error in input unit exit
 9100 CONTINUE  
WRITE (*,FMT='(A)') ' ERROR IN INPUT UNIT NUMBER '  
STOP

!       error conditions
9200 SELECT CASE(RET)
CASE ('RET1') !       end of file "no esperado" exit  
  WRITE (*,FMT='(A)') ' EOF NO ESPERADO. ALGO FALLA! '  
  STOP
CASE ('RET2') !       error in ffr subroutines exit  
  WRITE (*,FMT='(A)') ' ERROR IN FFR SUBROUTINES '  
  STOP
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN PRINCESA'
END SELECT

!       normal exit: end of file
 9990 CONTINUE  
CALL ABU  
WRITE (*,FMT='(A)') ' FILE IS OVER. SUCCESSFUL!! '  
CLOSE (IU)  

CLOSE (IOU)  
10000 CONTINUE  

RETURN  


END


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

SUBROUTINE RDATOM(RETCHAR)  
!
USE nlte_type
USE dimes_var, ONLY: ID_IONES=>IONG
USE prince_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&       ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS, &
& IU,IOU  
USE ffr_error

IMPLICIT NONE
!
!       it reads and storages atomic data
!       variables
!               nat: atom number (for internal counting)
!               ion: number of ion inside each atomo
!               iong: number of ion in the general counting
!               labat( ): atom label (name)
!               zeff( ): charge of the nucleus (=atomic number)
!               weight( ): atomic weight
!               abund( ): abundance
!               la0( ): index of first l level of the atom
!               ixa0( ): same with levels x
!               isa0( ): same with levels s
!               zef: effective charge of the atom
!               ifirsl( ): first l level of a given ion
!               nl: index for levels l
!               kl( ): index of the first level of the next ion
!               labl( ): level l label
!               gl( ): stadistic weight of level l
!               fl( ): ionization frequency of level l
!               lkl( ): label of parental level of a given l level
!               le( ): number of atom of a level l
!               li( ): number of ion of a level l
!               zl( ): effective charge of level l
!               ifirx( ): first x level of a given ion
!               nlx: index for levels x
!               labx( ): labels of x levels
!               glx( ): stadistic weight of levels x
!               flx( ): ionization frequency for levels x
!               lkx( ): label of parent level to a x level
!               klx( ): index of the first l level of the next ion
!                       (for x levels)
!               lix( ): number of ion of a level x
!               ifirs( ): first s level of a given ion
!               ns: index for levels s
!               labls( ): label of levels s
!               gs0( ): stadistic weight of first level of the serial
!               gs1( ): same with the last
!               ryd( ): rydberg constant of level s
!               qd( ): quantum deffect of level s
!               ns0( ): index of the first level of the serial
!               ns1( ): same with the last
!               kls( ): label of the parent level for a s level
!               lis( ): ion of level s
!               la1( ): number of l levels in the atom (can be avoid)
!               ixa1( ): same with levels x (can be avoid)
!               isa1( ): same with levels s (can be avoid)
!               nions( ): total number of ions in a given atom
!     ..
!     .. local scalars ..
REAL(DP) ::  REALVAR,ZEF  
INTEGER(I4B) ::  I,IJ,INTEG,NC
CHARACTER LABEL*4,KEY*6,RETCHAR*4,RET*4  
!     ..
!     .. external subroutines ..
EXTERNAL FFRKEY,FFRLOC,FFRNUM  
!     ..
!     .. common blocks ..
!     ..

LABEL = 'ATOM'  

CALL FFRLOC(LABEL,RET)  
SELECT CASE(RET)
CASE('RET1')
  GOTO 9980
CASE('RET2')
  GOTO 9990
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN IN RDATOM'
END SELECT

NAT = NAT + 1  

ION = 0  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990
LABAT(NAT) = KEY  

CALL FFRNUM(REALVAR,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 9990
ZEFF(NAT) = REALVAR  

CALL FFRNUM(REALVAR,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 9990
WEIGHT(NAT) = REALVAR  

CALL FFRNUM(REALVAR,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 9990
IF (REALVAR.LT.0.) REALVAR = 10**REALVAR  

ABUND(NAT) = REALVAR  
LA0(NAT) = NL + 1  
IXA0(NAT) = NLX + 1  
ISA0(NAT) = NS + 1  

ZEF = ZEFF(NAT) - 1.D0  
  100 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

!------------- l  levels
IF (KEY.EQ.'L') THEN  
     ZEF = ZEF + 1.D0  
     ION = ION + 1  
     IONG = IONG + 1  

     IFIRSL(IONG) = NL + 1  
 1000      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     IF (KEY.EQ.'0') THEN  
          DO I = IFIRSL(IONG),NL  
                KL(I) = NL + 1  
          END DO 
          DO IJ = IONG + 1,ID_IONES  
                IFIRSL(IJ) = NL + 1  
          END DO

          GO TO 100  
     ELSE  
          NL = NL + 1  
          LABL(NL) = KEY  
          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          GL(NL) = REALVAR  
          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          FL(NL) = REALVAR  
          CALL FFRKEY(KEY,NC,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          LKL(NL) = KEY  
          LE(NL) = NAT  
          LI(NL) = ION  
          ZL(NL) = ZEF  

          GO TO 1000  


     END IF  

!----------------- x  levels
ELSE IF (KEY.EQ.'X') THEN  

     IFIRX(IONG) = NLX + 1  
 2000      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     IF (KEY.EQ.'0') THEN  
          DO IJ = IONG + 1,ID_IONES  
                IFIRX(IJ) = NLX + 1  
          END DO
          GO TO 100  

     END IF  
     NLX = NLX + 1  
     LABX(NLX) = KEY  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     GLX(NLX) = REALVAR
     
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     FLX(NLX) = REALVAR  

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     LKX(NLX) = KEY  
     KLX(NLX) = NL + 1  
     LIX(NLX) = ION  


     GO TO 2000  

!-------------- s  levels
ELSE IF (KEY.EQ.'S') THEN  
     IFIRS(IONG) = NS + 1  
 3000      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     IF (KEY.NE.'0') THEN  
          NS = NS + 1  
          LABLS(NS) = KEY  
          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          GS0(NS) = REALVAR  

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          GS1(NS) = REALVAR  

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          RYD(NS) = REALVAR  

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          QD(NS) = REALVAR  

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          NS0(NS) = INTEG  

          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          NS1(NS) = INTEG  

          CALL FFRKEY(KEY,NC,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          KLS(NS) = KEY  
          LIS(NS) = ION  

          GO TO 3000  

     END IF  
     DO IJ = IONG + 1,ID_IONES  
           IFIRS(IJ) = NS + 1  
     END DO

     GO TO 100  

!--------------------  k  levels
ELSE IF (KEY.EQ.'K') THEN  
     NL = NL + 1  
     ZEF = ZEF + 1  
     ION = ION + 1  
     IONG = IONG + 1  
     IFIRSL(IONG) = NL  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     LABL(NL) = KEY  

     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     GL(NL) = REALVAR  

     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     FL(NL) = REALVAR  

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     LKL(NL) = KEY  
     ZL(NL) = ZEF  
     LE(NL) = NAT  
     LI(NL) = ION  
     KL(NL) = NL  

END IF  

!       the atom is over.
LA1(NAT) = NL - LA0(NAT) + 1  
IXA1(NAT) = NLX - IXA0(NAT) + 1  
ISA1(NAT) = NS - ISA0(NAT) + 1  


NIONS(NAT) = ION  

RETCHAR='RET0'
RETURN  
!       no more atoms to read. the main subroutine (prince) ends
 9980 CONTINUE  

RETCHAR='RET3'
RETURN

! error handling
9990 SELECT CASE(RET)
CASE('RET1') !       end of file exit 
  WRITE (*,FMT='(A)') ' END OF FILE '  
  RETCHAR='RET1'
  RETURN
CASE('RET2') !       error exit 
  WRITE (*,FMT='(A)') 'ERROR IN FFR SUBROUTINES '  
  RETCHAR='RET2'
  RETURN
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN RDATOM'
END SELECT

END

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

SUBROUTINE TRANSIC(RETCHAR)  
!
USE nlte_type
USE dimes_var,only: id_nttrd=>indtr
USE fund_const, ONLY: CLIGHT
USE prince_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&       ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
& ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
& LABAT,LABL,LKL,LABX,LKX,LABLS,KLS, &
& DATA,NDAT, &
& IU,IOU, &
& NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
&       LABL4,LABL1,LABU1,LABL3,IPARE5, &
& QDB,QCL,QRT, &
& TL,ANCDO,WLC,DL,FRECIN,FRECFN, &
& LABL2,PAREN2,PAREN3,PAREN4,LABU2
USE ffr_error

IMPLICIT NONE
!
!       it reads and storages all data concerning transitions
!       variables:
!                tabla: posible transition types
!                iform: number of formula
!                ndatos: number of data required for the transition
!                qdb: flag for the db case (doppler widths sampling)
!                nlong: number of points in the profile sampling
!                dl( ): points for the sampling in lambda
!                tl: line temperatura
!                ancdo: doppler width
!                qcl: flag for the cl case (central wavelenght)
!                qrt: flag for rbb or rbx transition
!                indtt: index for the total number of transitions
!                iaux1( , ): matrix for storaging the transition type
!                            and the pointer for other transitions
!                            counts
!                index1( ): index for rbb or cbb transistions
!                iform1( ): formula number for rbb or cbb
!                numd1( ): same with the number of data
!                labl1( ): lower level
!                labu1( ): upper level
!                indat1( ): index of the first data of this transition,
!                          storaged in data
!                ndat1( ): number of data in data for this transition
!                ndat: pointer for data
!                data( ): data
!                wlc: central wavelength
!                for the other transition type, joint in five groups,
!                (rbb-cbb; rbx-cbx; cbs-cbf; rbf; rff)
!                similar vector and matrices are defined, but using
!                the numbers 2, 3, 4 and 5 instead of 1. there are
!                only one different:
!                paren_( ): parent level label
!                frecin( ): actual edge frequency  (rbf)
!                frecfn( ): edge frequency with respect to next ground
!                           state
!
!     ..
!     .. local scalars ..
REAL(DP) ::  DELTAE,FRECIN2,REALVAR  
INTEGER(I4B) ::  I,IFORM,IJ,INTEG,IQI,ISI,L0,M,NC,NDATOS,NN,NNU
CHARACTER KEY*6,PARE*6,RETCHAR*4,RET*4  
!     ..
!     .. local arrays ..
INTEGER(I4B) ::  IFPTR(ID_NTTRD),NFPTR(ID_NTTRD)  
CHARACTER TABLA(8)*6  
!     ..
!     .. external functions ..
REAL(DP) ::  DOPPLER  
INTEGER(I4B) ::  IGENIO  
EXTERNAL DOPPLER,IGENIO  
!     ..
!     .. external subroutines ..
EXTERNAL FFRKEY,FFRNUM,HALLCL,HALLCX,IDENT,PERFIL  
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS,DBLE  
!     ..
!     .. data statements ..
DATA TABLA/'RBB','CBB','RBX','CBX','CBS','CBF','RBF','RFF'/  
!     ..

! JP changed May 3rd 2001
   QCL=.FALSE.

   10 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'TY') THEN  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     CALL IDENT(KEY,TABLA,8,M)  

     CALL FFRNUM(REALVAR,IFORM,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL FFRNUM(REALVAR,NDATOS,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

!          son los numeros de formula y de datos, respectivamente

     GO TO 10
     
!       sampling in doppler widths. the sampling is storage in dl.
ELSE IF (KEY.EQ.'DB') THEN  
     QDB = .TRUE.  
     CALL FFRNUM(REALVAR,NLONG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     
     DO I = 1,NLONG  
          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          DL(I) = REALVAR  
     END DO


     GO TO 10  

!       sampling in angstroms
ELSE IF (KEY.EQ.'DL') THEN  
     QDB = .FALSE.  
     CALL FFRNUM(REALVAR,NLONG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     DO I = 1,NLONG  
          CALL FFRNUM(REALVAR,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
          DL(I) = REALVAR  
     END DO


     GO TO 10  

!       line temperature
ELSE IF (KEY.EQ.'TL') THEN  
     CALL FFRNUM(TL,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     ANCDO = DOPPLER(TL,WEIGHT(NAT))  

     GO TO 10  

!       central wavelength
ELSE IF (KEY.EQ.'CL') THEN  
     QCL = .TRUE.  

     GO TO 10  

!       transition type
ELSE  
     CALL IDENT(KEY,TABLA,8,M)  
     IF (M.EQ.0) THEN  
          GO TO 10000  

     ELSE  
          QRT = .FALSE.  
          IF ((M.EQ.1) .OR. (M.EQ.3)) QRT = .TRUE.  
          GO TO (100,100,200,200,300,300,400,500) M  
!       goto calculado, para que haga una cosa u otra segun el tipo de trans.


     END IF  


END IF  

!-------------rbb or cbb transitions ------------
  100 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  101 CONTINUE  
DO IQI = 1,NL  
     IF (LABL(IQI).EQ.KEY) THEN  
          NN = IQI  
          GO TO 103  
     END IF  
END DO  

STOP ' ERROR IN TRANSIC - LABEL L NOT FOUND, BB'  

  103 CONTINUE  
INDTT = INDTT + 1  
IAUX1(INDTT,1) = M  
INDEX1 = INDEX1 + 1  
IFORM1(INDEX1) = IFORM  
NUMD1(INDEX1) = NDATOS + 2  
LABL1(INDEX1) = NN  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

DO IQI = 1,NL  
     IF (LABL(IQI).EQ.KEY) THEN  
          NNU = IQI  
          GO TO 115  
     END IF  
END DO  

STOP ' ERROR IN TRANSIC - LABEL U NOT FOUND, BB'  

  115 CONTINUE  
LABU1(INDEX1) = NNU  
INDAT1(INDEX1) = NDAT + 1  

IAUX1(INDTT,2) = INDEX1  

!       radiative transition
IF (QRT) THEN  
     IF (QCL) THEN  
          CALL FFRNUM(WLC,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
     ELSE  
          CALL HALLCL(INDEX1)  
     END IF  

     CALL PERFIL  
ELSE  
     CALL HALLCL(INDEX1)  

END IF  

!       read data
NDAT = NDAT + 1  
DATA(NDAT) = DBLE(M)  
NDAT = NDAT + 1  
DATA(NDAT) = WLC  

DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     DATA(NDAT) = REALVAR  
END DO  

!       read a new keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') THEN  
     QCL = .FALSE.  
     GO TO 10  
END IF  

GO TO 101  

!--------------- rbx or cbx transitions -----------
  200 CONTINUE  
CALL FFRKEY(PARE,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  205 CONTINUE  
INDTT = INDTT + 1  
IAUX1(INDTT,1) = M  
INDEX2 = INDEX2 + 1  
IFORM2(INDEX2) = IFORM  
NUMD2(INDEX2) = NDATOS  
INDAT2(INDEX2) = NDAT + 1  
PAREN2(INDEX2) = PARE  
IAUX1(INDTT,2) = INDEX2  
LABL2(INDEX2) = KEY  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

LABU2(INDEX2) = KEY  

!       radiative transition
IF (QRT) THEN  
     IF (QCL) THEN  
          CALL FFRNUM(WLC,INTEG,RET)  
          IF (ON_ERROR(RET)) GOTO 9990
     ELSE  
          CALL HALLCX(INDEX2)  
     END IF  

     CALL PERFIL  

END IF  

!       read data
DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     DATA(NDAT) = REALVAR  
END DO  

!       read new keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') THEN  
     QCL = .FALSE.  
     GO TO 10  
END IF  

GO TO 205  

!------------- cbs or cbf transitions  ------------------
  300 CONTINUE  
CALL FFRKEY(PARE,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  305 CONTINUE  
INDTT = INDTT + 1  
IAUX1(INDTT,1) = M  
INDEX3 = INDEX3 + 1  
IFORM3(INDEX3) = IFORM  
NUMD3(INDEX3) = NDATOS  
INDAT3(INDEX3) = NDAT + 1  
IAUX1(INDTT,2) = INDEX3  
PAREN3(INDEX3) = PARE  

DO ISI = 1,NL  
     IF (LABL(ISI).EQ.KEY) THEN  
          NN = ISI  
          GO TO 310  
     END IF  
END DO  

DO ISI = 1,NS  
     IF (LABLS(ISI).EQ.KEY) THEN  
          NN = -ISI  
          GO TO 310  
     END IF  
END DO  

STOP ' ERROR IN TRANSIC - LABEL L OR S NOT FOUND -CBF-CBS'  

  310 CONTINUE  

LABL3(INDEX3) = NN  

!       read data
DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     DATA(NDAT) = REALVAR  
END DO  

!       read new keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') GO TO 10  

GO TO 305  

!------------------ rbf transition  -------------------
  400 CONTINUE  
CALL FFRKEY(PARE,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  405 CONTINUE  
INDTT = INDTT + 1  
IAUX1(INDTT,1) = M  
INDEX4 = INDEX4 + 1  
IFORM4(INDEX4) = IFORM  
NUMD4(INDEX4) = NDATOS  
INDAT4(INDEX4) = NDAT + 1  
IAUX1(INDTT,2) = INDEX4  
PAREN4(INDEX4) = PARE

DO ISI = 1,NL  
     IF (LABL(ISI).EQ.KEY) THEN  
          NN = ISI  
          GO TO 410  
     END IF  
END DO  

STOP ' ERROR IN TRANSIC - LABEL L NOT FOUND'  

  410 CONTINUE  
LABL4(INDEX4) = NN  
FRECFN(INDEX4) = FL(NN)  
!
!       calculation of actual edge frequency (if not ground state
!       ionization)
!
DO ISI = 1,NL  
     IF (LABL(ISI).EQ.PARE) THEN  
          NN = ISI  
          GO TO 411  
     END IF  
END DO  

STOP ' ERROR IN TRANSIC - LABEL PARE NOT FOUND'  

  411 CONTINUE  
IJ = IGENIO(LE(NN),LI(NN))  
L0 = IFIRSL(IJ)  
IF (L0.EQ.NN) THEN  
!       groundstate ionization
     DELTAE = 0.D0  
ELSE  
!       excited state ionization
     DELTAE = FL(L0) - FL(NN)  
END IF  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

FRECIN2 = FRECFN(INDEX4) + DELTAE  
!
FRECIN(INDEX4) = FRECIN2/CLIGHT  
FRECFN(INDEX4) = FRECFN(INDEX4)/CLIGHT  
!
!       a la salida de aqui frecin y frecfn estan en cm-1
!       read data
DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     DATA(NDAT) = REALVAR  
END DO  

!       read new keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') GO TO 10  

GO TO 405  

!----------------- rff transitions ------------------------
  500 CONTINUE  

CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  505 CONTINUE  
INDTT = INDTT + 1  
IAUX1(INDTT,1) = M  
INDEX5 = INDEX5 + 1  
IFORM5(INDEX5) = IFORM  
NUMD5(INDEX5) = NDATOS  
INDAT5(INDEX5) = NDAT + 1  
IAUX1(INDTT,2) = INDEX5

DO ISI = 1,NL  
     IF (LABL(ISI).EQ.KEY) THEN  
          NN = ISI  
          GO TO 510  
     END IF  
END DO  

STOP ' ERROR IN TRANSIC - LABEL L NOT FOUND'  

  510 CONTINUE  
IPARE5(INDEX5) = NN  
!       read data
DO I = 1,NDATOS  
     NDAT = NDAT + 1  
     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     DATA(NDAT) = REALVAR  
END DO  

!       read new keyword
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

IF (KEY.EQ.'0') GO TO 10  

GO TO 505  

!--------------------------------------------------------
! error handling
9990 SELECT CASE(RET)
CASE('RET1') !       end of file exit 
  WRITE (*,FMT='(A)') ' END OF FILE '  
  RETCHAR='RET1'
  RETURN
CASE('RET2') !       error exit 
  WRITE (*,FMT='(A)') 'ERROR IN FFR SUBROUTINES '  
  RETCHAR='RET2'
  RETURN
CASE DEFAULT
  STOP ' WRONG ERROR CONDITION IN TRANSIC'
END SELECT

10000 CONTINUE  
BACKSPACE IU  

RETCHAR='RET0'
RETURN  

END

!-----------------------------------------------------------------------

SUBROUTINE PERFIL  
!
Use nlte_type
USE dimes_var,only: id_nttrd=>indtr

Use prince_var, ONLY: NF, &
& NLONG,INDTT,INDTR,IAUX2, &
& QDB, &
& ANCDO,WLC,DL

IMPLICIT NONE
!
!        calculates the sampling of the atomic profile for radiative
!        bound-bound transitions
!     ..
!     .. local scalars ..
REAL(DP) ::  CLIGHT  
INTEGER(I4B) ::  I
!     ..
!     .. local arrays ..
INTEGER(I4B) ::  IFPTR(ID_NTTRD),NFPTR(ID_NTTRD)  
!     ..
!     ..
!     .. data statements ..

DATA CLIGHT/2.997925E18/  
!     ..
! this routine should never be called, since profiles separately treated, THUS

INDTR = INDTR + 1  
IAUX2(INDTR) = INDTT  
NFPTR(INDTR) = NLONG  

IFPTR(INDTR) = NF + 1  
!       sampling in doppler width units
IF (QDB) THEN  
     DO I = 1,NLONG  
!          DL(I) = DL(I)* (ANCDO*WLC)  
      dl(i)=0.
     END DO
END IF  
!       now dl is in angstroms
!       construction of the sampling
!        do 20 i=1,nlong
!                nf=nf+1
!                fq(nf)=clight/(wlc+dl(i))
!20      continue

RETURN  

END

!-----------------------------------------------------------------------

SUBROUTINE HALLCL(INDEX)  
!
Use nlte_type
!Use nlte_dim
Use prince_var, ONLY: LABL1,LABU1,FL,WLC
IMPLICIT NONE
!
!        it calculates the central wavelength for a rbb transition
!        from the atomic data
!       character*6 key1,key2
!     .. scalar arguments ..
INTEGER(I4B) ::  INDEX  
!     ..
!     .. local scalars ..
REAL(DP) ::  CLIGHT,FC,FF1,FF2  
INTEGER(I4B) :: II1,II2
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS  
!     ..
!     .. data statements ..
DATA CLIGHT/2.997925E18/  
!
!     ..
!        key1=labl1(index)
!        key2=labu1(index)
!        nmax=nl
!        call ident(key1,labl,nmax,ii1)
!        call ident(key2,labl,nmax,ii2)
II1 = LABL1(INDEX)  
II2 = LABU1(INDEX)  
FF1 = FL(II1)  
FF2 = FL(II2)  
FC = ABS(FF1-FF2)  
IF (FC.EQ.0.) THEN
  WLC=1.D20 !if coincidentally levels are equal
ELSE
  WLC = CLIGHT/FC  
ENDIF
RETURN  

END

!-----------------------------------------------------------------------

SUBROUTINE HALLCX(INDEX)  
!
Use nlte_type
!Use nlte_dim
Use prince_var, ONLY: LABL2,LABU2,LABL,NL,LABX,NLX,FL,FLX,WLC
IMPLICIT NONE
!
!        it calculates central wavelength for a rbx transition from the
!        atomic data of the levels inolved
!     .. scalar arguments ..
INTEGER(I4B) ::  INDEX  
!     ..
!     ..
!     .. local scalars ..
REAL(DP) ::  CLIGHT,FC,FF1,FF2  
INTEGER(I4B) :: II1,II2
CHARACTER KEY1*6,KEY2*6  
!     ..
!     .. external subroutines ..
EXTERNAL IDENT  
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS  
!     ..
!     .. data statements ..
DATA CLIGHT/2.997925E18/  
!     ..

KEY1 = LABL2(INDEX)  
KEY2 = LABU2(INDEX)  
CALL IDENT(KEY1,LABL,NL,II1)  

CALL IDENT(KEY2,LABX,NLX,II2)  
FF1 = FL(II1)  
FF2 = FLX(II2)  
FC = ABS(FF1-FF2)  
WLC = CLIGHT/FC  

RETURN  

END

!-----------------------------------------------------------------------

SUBROUTINE IDENT(KEY,TABLE,NMAX,J)  
!
Use nlte_type
!Use nlte_dim
IMPLICIT NONE
!
!        it compares the character 'key' with those of table 'table',
!        returning in j the value of the corresponding index; if 'key'
!        is not in 'table', j=0
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

!-----------------------------------------------------------------------

FUNCTION DOPPLER(T,W)  
!
Use nlte_type
!Use nlte_dim
IMPLICIT NONE
!
!        it calculates the doppler width for a given temperature t
!        and atomic weight w.
!        the output is an adimensional number that multiplied by the
!        central wavelength gives the width in angstroms, or by the
!        central frequency, the width in herzts
!     .. scalar arguments ..
REAL(DP) ::  T,W,DOPPLER  
!     ..
!     .. local scalars ..
REAL(DP) ::  AMU,CBOLTZ,RCCMS  
!     ..
!     .. intrinsic functions ..
INTRINSIC SQRT  
!     ..

AMU = 1.67333D-24  

CBOLTZ = 1.3806D-16  

RCCMS = 3.3356405D-11  

DOPPLER = RCCMS*SQRT(2.D0*CBOLTZ*T/ (AMU*W))  

END

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

SUBROUTINE ABU  
!
Use nlte_type
!Use nlte_dim
Use prince_var, ONLY: NAT,ABUND,LABAT
IMPLICIT NONE
!
!       this subroutine allows the abundances in the atomic data to be
!       written either relative to h or to 1 (always in number of
!       particles, not in mass). after this subroutine, abundances are
!       always relative to hydrogen, so a certain abundance of h is
!       always required (it can be very small, but not 0)
!     ..
!     .. local scalars ..
REAL(DP) ::  ABH,SUMM  
INTEGER(I4B) ::  I, NENE
!     ..

SUMM = 0.D0  
DO I = 1,NAT  
     SUMM = SUMM + ABUND(I)  
END DO

IF (SUMM.LE.1.) THEN  
     NENE = 0  
     DO I = 1,NAT  
          IF (LABAT(I).EQ.'H') NENE = I  
     END DO

     IF (NENE.EQ.0) STOP ' ERROR. H NOT FOUND - ABU'  
     ABH = ABUND(NENE)  
     DO I = 1,NAT  
          ABUND(I) = ABUND(I)/ABH  
     END DO
END IF  

RETURN  
END

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

SUBROUTINE ALLOC
!
USE nlte_type
USE prince_var
USE dimes_var, ONLY: &
               id_atoms=>nat,   id_iones=>iong,   id_llevs=>nl, &
&              id_xlevs=>nlx,   id_slevs=>ns,     id_data=>ndat, &
&              id_nttrd=>indtr, id_rcbbt=>index1, id_rcbxt=>index2, &
&              id_cbsft=>index3,id_rbftr=>index4, id_rfftr=>index5, &
&              id_ntran=>indtt, id_nlong=>nprof

IMPLICIT NONE
!----------------------------------------------------------------------

allocate(la0(id_atoms),la1(id_atoms))
LA0=0
LA1=0

allocate(ixa0(id_atoms),ixa1(id_atoms))
IXA0=0
IXA1=0

allocate(isa0(id_atoms),isa1(id_atoms))
ISA0=0
ISA1=0

allocate(nions(id_atoms))
NIONS=0

allocate(ifirsl(id_iones),ifirx(id_iones),ifirs(id_iones))
IFIRSL=1
IFIRX=1
IFIRS=1

allocate(kl(id_llevs),le(id_llevs),li(id_llevs))
KL=0
LE=0
LI=0

allocate(klx(id_xlevs),lix(id_xlevs))
KLX=0
LIX=0

allocate(lis(id_slevs),ns0(id_slevs),ns1(id_slevs))
LIS=0
NS0=0
NS1=0

allocate(zeff(id_atoms),weight(id_atoms),abund(id_atoms))
ZEFF=0.
WEIGHT=0.
ABUND=0.

allocate(gl(id_llevs),fl(id_llevs),zl(id_llevs))
GL=0.
FL=0.
ZL=0.

allocate(glx(id_xlevs),flx(id_xlevs))
GLX=0.
FLX=0.

allocate(gs0(id_slevs),gs1(id_slevs),ryd(id_slevs),qd(id_slevs))
GS0=0.
GS1=0.
RYD=0.
QD=0.

allocate(labat(id_atoms))
allocate(labl(id_llevs),lkl(id_llevs))
allocate(labx(id_xlevs),lkx(id_xlevs))
allocate(labls(id_slevs),kls(id_slevs))

allocate(data(id_data))
DATA=0.

allocate(iaux2(id_nttrd))
allocate(iform1(id_rcbbt),numd1(id_rcbbt),indat1(id_rcbbt),labl1(id_rcbbt),labu1(id_rcbbt))
allocate(iform2(id_rcbxt),numd2(id_rcbxt),indat2(id_rcbxt))
allocate(iform3(id_cbsft),numd3(id_cbsft),indat3(id_cbsft),labl3(id_cbsft))
allocate(iform4(id_rbftr),numd4(id_rbftr),indat4(id_rbftr),labl4(id_rbftr))
allocate(iform5(id_rfftr),numd5(id_rfftr),indat5(id_rfftr),ipare5(id_rfftr))
allocate(iaux1(id_ntran,2))
allocate(dl(id_nlong))
allocate(frecin(id_rbftr),frecfn(id_rbftr))

allocate(labl2(id_rcbxt),labu2(id_rcbxt),paren2(id_rcbxt))
allocate(paren3(id_cbsft))
allocate(paren4(id_rbftr))

return
end

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

FUNCTION IGENIO(KK,II)  

USE nlte_type  

USE dimes_var, ONLY: id_atoms=>nat, id_llevs=>nl, &
&                    id_xlevs=>nlx, id_slevs=>ns

USE prince_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
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

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


SUBROUTINE FRESCAL_0(IFRE)  
!
!-----  this subroutine provides a pessimistic guess for ID_FREC1
!       in a similar spirit as the original version.
!       if major changes are done there, they should be applied also here!!!!  
!
!       present philosophy requires that EACH level of a DETAILed ion
!       is represented with two corresponding frequencies, namely at
!       the corresponding edge and at edge-epslon(edge)  
!   
!       in this version, we use only LEVMAX and ALL ions
!
USE nlte_type  

USE dimes_var, ONLY: kel=>nat, id_rbftr=>index4

USE dimes_var, ONLY: N_KEDGES, K_NMIN, K_NMAX, Z, NIONK=>N, ETH

USE nlte_param, ONLY: levmax, id_frec2=>nf2, kis=>nmaxion

USE fund_const

USE prince_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
&       ISA0,ISA1,NIONS,IFIRSL,IFIRX,IFIRS,KL,LE,LI,KLX,LIX,NS0,NS1,LIS, &
&       ZEFF,WEIGHT,ABUND,GL,FL,ZL,GLX,FLX,GS0,GS1,RYD,QD, &
&       LABAT,LABL,LKL,LABX,LKX,LABLS,KLS, &
&       NLONG,INDTT,INDEX1,INDEX2,INDEX3,INDEX4,INDEX5, &
&       INDTR,IAUX2,IFORM1,IFORM2,IFORM3,IFORM4,IFORM5,NUMD1,NUMD2, &
&       NUMD3,NUMD4,NUMD5,INDAT1,INDAT2,INDAT3,INDAT4,INDAT5,IAUX1, &
&       LABL4,LABL1,LABU1,LABL3,IPARE5, &
&       TL,ANCDO,WLC,DL,FRECIN,FRECFN, &
&       LABL2,PAREN2,PAREN3,PAREN4,LABU2  

IMPLICIT NONE  
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: NFMIN=300
LOGICAL, PARAMETER      :: OPTOUT = .FALSE.

INTEGER(I4B) :: IFRE
 
!     ..
!     .. local scalars ..
REAL(DP) :: EDGE,EPS,FREMIN,FRENEW,AAA,EMAX,DMIN,DMIN1,DMIN2,DM,DNEW,DNEW1, &
&           XMIN,XMAX,DFL         

INTEGER(I4B) :: NOPA,I,J,K,L,M,N,IHI,IZ,ISI,LEVUSED,LEVTOT,LEVPORG,II,ILAB, &
&               LEVE,LEVP,LLI,IF1,NPER,IMP,JI,INEW,IL,IO,LL,NK


CHARACTER LAB*6,PARE*6  
!     ..
!     .. local arrays ..
REAL(DP) ::  FLAUX(ID_RBFTR),FREEDGE(ID_RBFTR),FRE(ID_FREC2)  
INTEGER(I4B) ::  INDEDGE(ID_RBFTR),IOPA(KIS*KEL),INDEX(ID_FREC2)
!     ..
!     .. statement functions ..
REAL(DP) ::  EPSLON  
!     ..
!     .. statement function definitions ..
!
EPSLON(EDGE) = 5.D0*10.D0** (DBLE(INT(LOG10(EDGE)))-6.D0)  
!     ..
IF(KEL.NE.NAT) STOP ' SOMETHING WRONG WITH KEL,NAT'

IF (ID_RBFTR.GT.999) STOP ' TOO MANY RBF-TRANSITIONS, CODING OF INDEX AFFECTED'  
!
!       most abundant element k1
!
!
NOPA=0

DO K = 1,NAT  
     DO M = 1,NIONS(K)-1
          NOPA = NOPA + 1  
          IOPA(NOPA) = K*100 + M  
     END DO
END DO  

IF (NOPA.GT.KIS*KEL) STOP ' ERROR IN NOPA'  

PRINT *  
PRINT *,' USED IONS FOR FREQUENCY GRID'  
PRINT *  

DO N = 1,NOPA  
     IHI = IOPA(N)/100  
     IZ = INT(ZEFF(IHI))  
     ISI = IOPA(N) - 100*IHI  
     IF (NIONS(IHI).EQ.ISI) STOP ' ERROR IN NIONS OR ISI'  
     LEVUSED = LEVMAX  
     WRITE (*,FMT=9000) LABAT(IHI),ISI,IZ + ISI - 1,LEVUSED  
END DO  

PRINT *  

IFRE = 0  
LEVTOT = 0  
DO 150 N = 1,NOPA  

     K = IOPA(N)/100  
     I = IOPA(N) - 100*K
     
     IF (I.EQ.NIONS(K)) STOP ' ERROR IN NIONS OR ISI'  
     IF (I.LE.0) STOP ' ERROR IN ION - FRESCAL_0 '  

     LEVPORG = 0  
     DO II = 1,ID_RBFTR  
          ILAB = LABL4(II)  
          IF (LE(ILAB).EQ.K .AND. LI(ILAB).EQ.I) THEN  
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
!     always levmax levels (assuming levmax < levporg)
!
     LEVE =  LEVMAX  

     LEVP = LEVPORG  

     DO L = 1,LEVP  
          IFRE = IFRE + 1  
          LLI = LEVPORG + 1 - L  
          DO II = 1,ID_RBFTR  
               IF (FLAUX(LLI).EQ.FRECIN(II)) GO TO 130  
          END DO
          STOP ' FLAUX NOT FOUND IN FRECIN'  

  130           CONTINUE  
          INDEX(IFRE) = 100000*K + I*1000 + II  
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

IF (IFRE.GT.ID_RBFTR) STOP ' ERROR IN IFRE'  ! THIS IS THE NEW ONE

PRINT *,' NUMBER OF CONSIDERED EDGES = ',IFRE  
PRINT *  

DO I = 1,IFRE  
     EPS = EPSLON(FRE(I))  
     FRE(I+IFRE) = FRE(I) - EPS  
     FREEDGE(I) = FRE(I)  
END DO  

IF1 = IFRE  
IFRE = 2*IFRE  

!K-shell edges
NK=0

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

!AAA = 5.D6  !  ( 20 A) sort of minimum
!new version including x-ray treatment
AAA = 2.D7  !  (  5 A)

!HYDROGEN ONLY
!changed March 2012, Valparaiso
IF (NAT.EQ.1.AND.IOPA(1)/100.EQ.1) AAA=4.D5 !250 A
!might still lead to problems with ifremax for cool stars

PRINT *
PRINT *, ' FRE(IFRE): ', 1.0D8/FRE(IFRE)
PRINT *, ' MIN      : ', 1.0D8/AAA
PRINT *

EMAX = AAA  
EMAX=MAX(EMAX,FRE(IFRE)*1.2D0)

IF(EMAX.GT.7.5D5) THEN ! in case, resolve HeII edge
  IFRE = IFRE + 1  
  FRE(IFRE) = 7.5D5  !(133 A) 
ENDIF

IF(EMAX.GT.1.2D6) THEN
  IFRE = IFRE + 1  
  FRE(IFRE) = 1.2D6  !(86 A, half way between 133 A and first K-shell at 39A)
ENDIF

IF(EMAX.GT.5.D6) THEN ! xray treatment
  IFRE = IFRE + 1  
  FRE(IFRE) = 5.D6  !(20 A)
ENDIF

IFRE = IFRE + 1  
FRE(IFRE) = EMAX  
!
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
!changed Jan 27 2015, for better resol between 2000 and 1600 of non-HHe models
!          DMIN1 = DMIN  
          DMIN1 = DMIN/4.  
     ELSE IF (FRE(I+1).LT.1.D8/910.) THEN ! > 910 A   
          DMIN1 = DMIN/4.
!
!   minimum separation for lambda < 227 a  is approx. 2  A
!
     ELSE IF (FRE(I+1).GT.440528.D0) THEN  ! < 227 A
          DMIN2 = 2.D-8 * FRE(I+1)  !  DELTA LAM / LAM   
          IF(FRE(I+1).GT.5.D6) DMIN2=0.3
          DMIN1=DMIN2
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
                         IF (OPTOUT) WRITE (*,FMT=9030) 1.D8/FRE(IFRE),DNEW1, &
&                          DNEW1/FRE(IFRE)
                    END DO

               ELSE IF (IMP.EQ.1 .AND. J.EQ.2) THEN  
                    DNEW1 = DNEW/2.D0  
                    FRENEW = FRE(IFRE) + DNEW1  
                    IFRE = IFRE + 1  
                    FRE(IFRE) = FRENEW  
                    IF (OPTOUT) WRITE (*,FMT=9030) 1.D8/FRE(IFRE),DNEW1, &
&                     DNEW1/FRE(IFRE)
               END IF  

               IF (J.NE.INEW) THEN  
                    FRENEW = FRE(I) + J*DNEW  
                    IFRE = IFRE + 1  
                    FRE(IFRE) = FRENEW  
                    IF (OPTOUT) WRITE (*,FMT=9030) 1.D8/FRE(IFRE),DNEW1, &
&                     DNEW1/FRE(IFRE)

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

CALL SORT(IFRE,FRE)  
!
!    check resolution for
!    hydrogen resonance lines and hei singlet resonance line
!

!XMIN=1.D0/1030.D-8
XMIN=1.D0/1025.D-8
XMAX=1.D0/912.D-8

!following block changed Feb 2015, to be on the safe side,
!and to include Lyman beta (important for Halpha)
DO I = 1,IFRE-1  

  IF (FRE(I).LE.1.D0/1215.D-8 .AND. FRE(I+1).GT.1.D0/1215.D-8) THEN !LYMAN ALPHA
  DMIN1=0.005D0  !VERY HIGH RESOLUTION
  DM = FRE(I+1)/FRE(I) - 1.D0  
  IF(DM .LT. DMIN1) CYCLE

! new treatment
  ELSE IF (FRE(I).LE.XMIN .AND. FRE(I+1).GT.XMIN) THEN !LYMAN BETA
  DMIN1=0.001D0  !EVEN HIGHER RESOLUTION
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
       IF (OPTOUT) WRITE (*,FMT=9031) 1.D8/FRE(IFRE),DNEW, DNEW/FRE(IFRE)
     END DO

ENDDO

!JO CHANGED March 2017
! ONE ADDITIONAL POINT EXACTLY AT HEII LY-ALPHA
IFRE = IFRE + 1
FRE(IFRE)=1.D8/303.797
IF (OPTOUT) WRITE (*,FMT=9034) 1.D8/FRE(IFRE)


CALL SORT(IFRE,FRE)  

IF (OPTOUT) THEN  
    PRINT *,' WAVELENGTH(A)     FREQ(HZ)'  
     DO 250 I = 1,IFRE  

          DO J = 1,IF1  
               IF (FRE(I).EQ.FREEDGE(J)) GO TO 240  
          END DO

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
          WRITE (*,FMT=9020) 1.D0/FRE(I)*1.D8,CLIGHT*FRE(I), &
           LABAT(K),IO,IZ + IO - 1,LAB,PARE
  250      CONTINUE  
     PRINT *  
END IF  

PRINT *,' TOTAL NUMBER OF FREQUENCIES = ',IFRE  
PRINT *  
I=LOG10(FLOAT(IFRE))  ! ROUNDING TO NEXT FULL 100
I=10**I
K=IFRE/I
IZ=IFRE-K*I
IZ=IZ/100

!IFRE=K*I+(IZ+1)*100
IFRE=K*I+(IZ+2)*100
!changed at March 14th 2013, after request (problem) by roberto
PRINT *,' USED MAX. NUMBER OF FREQUENCIES = ',IFRE 
PRINT *  


RETURN  

 9000 FORMAT (1X,A2,I2,' Z = ',I2,' MAX.NO.OF RESOLVED EDGES = ',I2)  
 9010 FORMAT (1X,F12.3,6X,E12.6)  
 9020 FORMAT (1X,F12.3,6X,E12.6,3X,A2,I2,' Z=',I2,'  FROM ',A6,' TO ', &
&       A6)
 9030 FORMAT (' ADDIT. POINT AT ',F12.3,6X,F12.3,' DELTA E / E = ', &
&       F5.3)
 9031 FORMAT (' ADDIT. POINT AT ',F12.3,6X,F12.3,' DELTA E / E = ', &
&       F5.3,' (RESONANCE LINES!)')
 9034 FORMAT (' ADDIT. POINT AT ',F12.3,' HEII LY_ALPHA')

END
!
!-----------------------------------------------------------------------
!
SUBROUTINE SORT(N,RA)  

USE nlte_type
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
!--------------------------------------------------------------------------
!
subroutine read_kshell_data

USE nlte_type
USE fund_const,only: hh,clight,ev
USE dimes_var, only: n_kedges, z, n, eth, sigma, s, zeff

IMPLICIT NONE

integer :: i

real(dp) :: ZL10, E_thre, cross_s, s_calc, const, dummy

character*(*), parameter :: fpath='../inicalc/ATOMDAT_NEW/'

open (unit = 2, file = fpath//'k_shell_Auger_data',status='old')
read(2,*)
read(2,*)
read(2,*)


do i=1,n_kedges
!N is number of electrons
read(2,*) Z(i),N(i),ETH(i),dummy,sigma(i),S(i),Zeff(i) 
!**************************************************
! Remeber to multiply sigma(i) by 1.E-18
!**************************************************
!print*, Z(i),N(i),ETH(i),sigma(i),S(i),Zeff(i) 
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
!now calculate ionization stages (astronomical notation)
n=z-n+1
sigma=sigma*1.d-18

!transform eV to Kayser
const=hh*clight/ev
const=1.d0/const
eth=eth*const

print*
print*,'K-shell data read'

return

end
