! version 1.2 (see below)
MODULE princesa_var
!
USE nlte_type
USE nlte_dim
IMPLICIT NONE
!----------------------------------------------------------------------
!atdnin
!
INTEGER(I4B) ::  NL=0,NLX=0,NS=0,NAT=0
INTEGER(I4B) ::  ION,IONG=0
INTEGER(I4B), DIMENSION(ID_ATOMS) ::  LA0=0,LA1=0
INTEGER(I4B), DIMENSION(ID_ATOMS) ::  IXA0=0,IXA1=0
INTEGER(I4B), DIMENSION(ID_ATOMS) ::  ISA0=0,ISA1=0
INTEGER(I4B), DIMENSION(ID_ATOMS) ::  NIONS=0
INTEGER(I4B), DIMENSION(ID_IONES) ::  IFIRSL=1,IFIRX=1,IFIRS=1
INTEGER(I4B), DIMENSION(ID_LLEVS) ::  KL=0,LE=0,LI=0
INTEGER(I4B), DIMENSION(ID_XLEVS) ::  KLX=0,LIX=0
INTEGER(I4B), DIMENSION(ID_SLEVS) ::  LIS=0,NS0=0,NS1=0
!----------------------------------------------------------------------
!atdnre
!
REAL(DP), DIMENSION(ID_ATOMS) :: ZEFF=0.,WEIGHT=0.,ABUND=0.
REAL(DP), DIMENSION(ID_LLEVS) :: GL=0.,FL=0.,ZL=0.
REAL(DP), DIMENSION(ID_XLEVS) :: GLX=0.,FLX=0.
REAL(DP), DIMENSION(ID_SLEVS) :: GS0=0.,GS1=0.,RYD=0.,QD=0.
!----------------------------------------------------------------------
!atomdatc
!
CHARACTER*6, DIMENSION(ID_ATOMS) :: LABAT 
CHARACTER*6, DIMENSION(ID_LLEVS) :: LABL,LKL 
CHARACTER*6, DIMENSION(ID_XLEVS) :: LABX,LKX 
CHARACTER*6, DIMENSION(ID_SLEVS) :: LABLS,KLS 
!----------------------------------------------------------------------
!dat
!
INTEGER(I4B) :: NDAT=0
REAL(DP), DIMENSION(ID_NDATA) :: DATA=0.
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
INTEGER(I4B), DIMENSION(ID_NTTRD) :: IAUX2
INTEGER(I4B), DIMENSION(ID_RCBBT) :: IFORM1,NUMD1,INDAT1,LABL1,LABU1
INTEGER(I4B), DIMENSION(ID_RCBXT) :: IFORM2,NUMD2,INDAT2
INTEGER(I4B), DIMENSION(ID_CBSFT) :: IFORM3,NUMD3,INDAT3,LABL3
INTEGER(I4B), DIMENSION(ID_RBFTR) :: IFORM4,NUMD4,INDAT4,LABL4,DREXPLICIT
INTEGER(I4B), DIMENSION(ID_RFFTR) :: IFORM5,NUMD5,INDAT5,IPARE5
INTEGER(I4B), DIMENSION(ID_NTRAN,2) :: IAUX1
!----------------------------------------------------------------------
!tranlo
!
LOGICAL :: QCL,QDB,QRT
!----------------------------------------------------------------------
!tranre
!
REAL(DP) :: ANCDO,TL,WLC
REAL(DP), DIMENSION(ID_NLONG) :: DL
REAL(DP), DIMENSION(ID_RBFTR) :: FRECIN,FRECFN
REAL(DP), DIMENSION(ID_CBSFT) :: FLCBF
!----------------------------------------------------------------------
!transc
!
CHARACTER*6, DIMENSION(ID_RCBXT) :: LABL2,LABU2,PAREN2 
CHARACTER*6, DIMENSION(ID_CBSFT) :: PAREN3 
CHARACTER*6, DIMENSION(ID_RBFTR) :: PAREN4 

END MODULE princesa_var
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
!     version 1.2 (july 2009) : new array DREXPLICIT(INDEX4): set to 1 if
!                 explicit dr data are present (otherwise set to 0)
!     version 1.3 (dec. 2016) : new array FLCBF(INDEX3): frequencies
!                 for CBF transitions
!
SUBROUTINE PRINCE  
!
USE nlte_type
USE nlte_dim
USE princesa_var, ONLY: IOU,IU
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
USE nlte_dim
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
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
USE nlte_dim
USE fund_const, ONLY: CLIGHT
USE princesa_var, ONLY: NL,NLX,NS,NAT,ION,IONG,LA0,LA1,IXA0,IXA1, &
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
& LABL2,PAREN2,PAREN3,PAREN4,LABU2,DREXPLICIT,FLCBF
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
!                frecin( ): actual edge energy  (rbf) [in Kayser]
!                frecfn( ): edge energy with respect to next ground
!                           state [in Kayser]
!                flcbf( ): edge frequency with respect to next ground
!                           state for cbf ionization
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

STOP 'ERROR IN TRANSIC - LABEL L NOT FOUND, BB'  

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

STOP 'ERROR IN TRANSIC - LABEL U NOT FOUND, BB'  

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

STOP 'ERROR IN TRANSIC - LABEL L OR S NOT FOUND -CBF-CBS'  

  310 CONTINUE  

LABL3(INDEX3) = NN  
FLCBF(INDEX3) = FL(NN)  

!       calculation of actual edge frequency (if not ground state
!       ionization)
!
DO ISI = 1,NL  
     IF (LABL(ISI).EQ.PARE) THEN  
          NN = ISI  
          GO TO 311  
     END IF  
END DO  

STOP 'ERROR IN TRANSIC - LABEL PARE (CBF) NOT FOUND'  

  311 CONTINUE  
IJ = IGENIO(LE(NN),LI(NN))  
L0 = IFIRSL(IJ)  
IF (L0.EQ.NN) THEN  
!       groundstate ionization
     DELTAE = 0.D0  
ELSE  
!       excited state ionization
     DELTAE = FL(L0) - FL(NN)  
END IF  

FLCBF(INDEX3) = FLCBF(INDEX3) + DELTAE  

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
! NEW; CHECK WHETHER DR WITH EXPLICIT STABILIZING RATES
IF (IFORM.EQ.21) THEN
DREXPLICIT(INDEX4)=1
ELSE
DREXPLICIT(INDEX4)=0
ENDIF

DO ISI = 1,NL  
     IF (LABL(ISI).EQ.KEY) THEN  
          NN = ISI  
          GO TO 410  
     END IF  
END DO  

STOP 'ERROR IN TRANSIC - LABEL L NOT FOUND'  

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

STOP 'ERROR IN TRANSIC - LABEL PARE NOT FOUND'  

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

STOP 'ERROR IN TRANSIC - LABEL L NOT FOUND'  

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
Use nlte_dim
Use princesa_var, ONLY: NF, &
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
Use nlte_dim
Use princesa_var, ONLY: LABL1,LABU1,FL,WLC
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
Use nlte_dim
Use princesa_var, ONLY: LABL2,LABU2,LABL,NL,LABX,NLX,FL,FLX,WLC
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
Use nlte_dim
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
Use nlte_dim
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
Use nlte_dim
Use princesa_var, ONLY: NAT,ABUND,LABAT
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

     IF (NENE.EQ.0) STOP 'ERROR. H NOT FOUND - ABU'  
     ABH = ABUND(NENE)  
     DO I = 1,NAT  
          ABUND(I) = ABUND(I)/ABH  
     END DO
END IF  

RETURN  
END
