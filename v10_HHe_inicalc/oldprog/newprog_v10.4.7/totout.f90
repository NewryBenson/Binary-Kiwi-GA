PROGRAM TOTOUT  
!
!     03/03/99 changed to f90
!      july 2003 number of minor modifications
!     march 2006 dimension of enionzc
!      sept 2006 reading/output of clf
!       oct 2014 length of cat (*50) and related
!
Use nlte_type
Use nlte_dim
IMPLICIT NONE
!
!     generates complete output for considered model
!     in the following form
!
!     changed to treat both old model files (without pressure)
!             and new model files (with pressure)
!
!     nd,natom,nkis,nlev,nfreq
!     r (dim=nd)
!     v (dim=nd)
!     dvdr (dim=nd)
!     rho (dim=nd)
!     p (dim=nd)
!     t (dim=nd)
!     ne (dim=nd)
!     tauross (dim=nd)
!     coldens (dim=nd)
!     clf(dim=nd)
!     level names (dim=nlev)
!     departure coeff (dim=nlev,nd)
!     ionization fractions (dim=nkis+1,natom,nd)
!     lambda(dim=nfreq)
!     h-nue (dim=nfreq)
!     trad_eff(dim=nfreq)
!
!
!     .. parameters ..
INTEGER(I4B), PARAMETER :: ND=ID_NDEPT  
INTEGER(I4B), PARAMETER :: KEL=ID_ATOMS,KIS=ID_KISAT  
INTEGER(I4B), PARAMETER :: IFRETOT=ID_FREC1  
INTEGER(I4B), PARAMETER :: NLEVS=ID_LLEVS  
!     ..
!     .. local scalars ..
REAL(DP) ::  BETA,CSOUND,DUM,GGRAV,REL,RVMIN1,SR,SUMM,TEFF, &
&                 VMAX,XMLOSS,XMU,YHE
INTEGER(I4B) :: I,ICORR,IFRE,J,K,L,NL,NLEV,NU,ZEFFK,NDEF,KISMAX
CHARACTER CAT*50,LTEPOP*60,NLTEPO*60,OUT_TOT*60  
!     ..
!     .. local arrays ..
REAL(DP) ::  A(NLEVS),B(NLEVS),DEP(NLEVS), &
&                 ENIONND(KEL,KIS+1,ND), &
&                 FNUE(IFRETOT),FRE(IFRETOT),GV(ND), &
&                 OPAC(ND,IFRETOT),P(ND),R(ND),CLF(ND),RHO(ND), &
&                 SCONT(ND,IFRETOT),TAUR(ND),TEMP(ND), &
&                 TRAD(IFRETOT),V(ND),XLAM(IFRETOT),XM(ND),XNE(ND), &
&                 XNELTE(ND),XNH(ND),ZEFF(KEL)

REAL(DP), DIMENSION (:,:,:), ALLOCATABLE :: ENIONZC

INTEGER(I4B) :: INDEX(ND),NUPPER(NLEVS)  
INTEGER(I4B), DIMENSION(:,:), ALLOCATABLE :: DEFINED
CHARACTER LEVNAM(NLEVS)*6  
!     ..
!     .. external subroutines ..
EXTERNAL ATFILE  
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS,INT,LOG10,MAX  
!     ..

PRINT *,'GIVE IN MODEL NAME (CATALOG)'  
READ (*,FMT='(A)') CAT  

NLTEPO  = TRIM(CAT)//'/NLTE_POP'  
LTEPOP  = TRIM(CAT)//'/LTE_POP'  
OUT_TOT = TRIM(CAT)//'/OUT_TOT'  

PRINT *,' LEVEL STRUCTURE'  
PRINT *  

OPEN (10,FILE=OUT_TOT,STATUS='UNKNOWN',FORM='FORMATTED')  

NLEV = NLEVS  
NL=0
CALL ATFILE(ZEFF,NUPPER,NLEV,NL,LEVNAM)  

ZEFFK=MAXVAL(ZEFF)
KISMAX=KIS+ZEFFK
ALLOCATE(ENIONZC(KEL,KISMAX+1,ND))

IF (NL.GT.NLEVS) STOP ' NL > NLEVS'  

OPEN (2,FILE=TRIM(CAT)//'/CONT_FORMAL',STATUS='OLD',FORM='UNFORMATTED')  
READ (2) IFRE, (FRE(I),I=1,IFRETOT), &
&  ((SCONT(I,J),I=1,ND),J=1,IFRETOT), &
&  ((OPAC(I,J),I=1,ND),J=1,IFRETOT), (TEMP(I),I=1,ND)
CLOSE (2)  

WRITE (10,FMT=*) ND,KEL,KISMAX,NL,IFRE  

OPEN (2,FILE=TRIM(CAT)//'/MODEL',STATUS='OLD',FORM='UNFORMATTED')  
READ (2,ERR=10) TEFF,GGRAV,SR,YHE,XMU,VMAX,XMLOSS,BETA, &
&  (R(I),I=1,ND), (V(I),I=1,ND), (GV(I),I=1,ND), (RHO(I),I=1,ND), &
&  (XNE(I),I=1,ND), (XNH(I),I=1,ND), (CLF(I),I=1,ND), &
&  (INDEX(I),I=1,ND), (P(I),I=1,ND)
! ne from hydro model (no T-corrections)
GOTO 30
  

   10 CONTINUE

! to allow reading of very old version
PRINT *  
PRINT *,' PRESSURE ONLY FOR MU = CONST'  
PRINT *  
CSOUND = 1.3806D-16/ (XMU*1.673D-24)  
!
DO L = 1,ND  
     P(L) = RHO(L)*TEMP(L)*CSOUND  
END DO  
!   
   30 CONTINUE

CLOSE (2)  

PRINT *  
PRINT *,' MODEL-PARAMETERS'  
PRINT *  

PRINT *,'TEFF = ',TEFF,' LOG G =',LOG10(GGRAV),' RSTAR = ',SR/6.96D10
PRINT *,' YHE = ',YHE  
PRINT *,' MDOT = ',XMLOSS*3.1557D7/1.989D33,' VINF = ',VMAX*1.D-5, &
&  ' BETA = ',BETA
PRINT *  

OPEN (2,FILE=TRIM(CAT)//'/TAU_ROS',STATUS='OLD',FORM='FORMATTED')  
DO L = 1,ND  
     READ (2,FMT=*) TAUR(L),XNELTE(L)  
END DO  
CLOSE (2)  

XM(1) = 0.D0  
DO L = 2,ND  
     XM(L) = XM(L-1) + .5D0* (RHO(L-1)+RHO(L))* (R(L-1)-R(L))  
END DO  

DO L = 1,ND  
     XM(L) = XM(L)*SR  
END DO  

OPEN (1,FILE='OUT_LASTMODEL',STATUS='UNKNOWN',FORM='FORMATTED')  
WRITE (1,FMT='(A)') &
&' N      R            V           DV/DR          RHO            P '&
&'          XNH         TEMP        TAUROSS      COL.DENS.     CLF'
!23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
DO I = 1,ND  
     WRITE (1,FMT=9000) I,R(I),V(I),GV(I),RHO(I),P(I),XNH(I), &
      TEMP(I),TAUR(I),XM(I),CLF(I)
END DO  

OPEN (2,FILE=TRIM(CAT)//'/ENION',STATUS='OLD',FORM='UNFORMATTED')  
READ (2) ENIONND,XNE  
CLOSE (2)  

WRITE (10,FMT='(3(F16.10,2x))') (R(I),I=1,ND)  
WRITE (10,FMT='(3(E17.10,2x))') (V(I),I=1,ND)  
WRITE (10,FMT='(3(E17.10,2x))') (GV(I),I=1,ND)  
WRITE (10,FMT='(3(E17.10,2x))') (RHO(I),I=1,ND)  
WRITE (10,FMT='(3(E17.10,2x))') (P(I),I=1,ND)  
WRITE (10,FMT='(3(F16.5,2x))') (TEMP(I),I=1,ND)  
WRITE (10,FMT='(3(E17.10,2x))') (XNE(I),I=1,ND)  
WRITE (10,FMT='(3(E17.10,2x))') (TAUR(I),I=1,ND)  
WRITE (10,FMT='(3(E17.10,2x))') (XM(I),I=1,ND)  
WRITE (10,FMT='(A)') (LEVNAM(L),L=1,NL)  

OPEN (11,FILE=LTEPOP,STATUS='OLD',ACCESS='DIRECT',RECL=NL*8)  
OPEN (12,FILE=NLTEPO,STATUS='OLD',ACCESS='DIRECT',RECL=NL*8)  

REL = 0.D0  
DO L = 1,ND  
     XNELTE(L) = XNELTE(L)/XNE(L)  
     REL = MAX(REL,ABS(1.D0-XNELTE(L)))  
END DO  
PRINT *  
PRINT *,' MAX DEV. BETWEEN NE(LTE) AND NE(NLTE) = ',REL  
PRINT *  

ALLOCATE(DEFINED(NL,ND))

DO I = 1,ND
     READ (11,REC=I) (A(J),J=1,NL)  
     READ (12,REC=I) (B(J),J=1,NL)
     WHERE(B.EQ.0)
       DEFINED(:,I)=0
     ELSEWHERE  
       DEFINED(:,I)=1
     ENDWHERE  
     DO J=1,NL
!     print*,i,j,a(j),b(j)
       NU=NUPPER(J)
       IF(B(NU).NE.0.) THEN
         DEP(J)=B(J)/A(J)*A(NU)/B(NU)*XNELTE(I)
! for comparison with Adi
!          DEP(J)=B(J)/A(J)*XNELTE(I)
       ELSE
         DEP(J)=0.D0
       ENDIF  
     ENDDO

     WRITE (10,FMT='(3(E17.10,2x))') (DEP(J),J=1,NL)  
END DO  

DO J=1,NL
  NDEF=0
  DO I=1,ND
    IF (DEFINED(J,I).EQ.0) ndef=1
  ENDDO
  IF(NDEF.EQ.1) THEN
    PRINT*,'LEVEL ',LEVNAM(J),' ',J,' NOT DEFINED AT ALL DEPTH-POINTS'
!  ELSE
!    PRINT*,'LEVEL ',LEVNAM(J),' ',J,' DEFINED AT ALL DEPTH-POINTS'
  ENDIF
ENDDO  
!
!     note: enionzc corrected for effective charge
!
LLOOP: DO L = 1,ND  
     KLOOP: DO K = 1,KEL  
          SUMM = 0.D0  
          ZEFFK = INT(ZEFF(K))  
          DO I = 1,KIS + 1  
               SUMM = SUMM + ENIONND(K,I,L)  
          END DO
          DO I = 1,ZEFFK  
               ENIONZC(K,I,L) = 0.D0  
          END DO
          DO I = ZEFFK + 1,KISMAX + 1  
               ICORR = I - ZEFFK  
               ENIONZC(K,I,L) = ENIONND(K,ICORR,L)/SUMM  
          END DO
     END DO KLOOP
     WRITE (10,FMT='(3(E17.10,2x))') ((ENIONZC(K,I,L),I=1,KISMAX+1),K=1,KEL)  
END DO LLOOP  

OPEN (2,FILE=TRIM(CAT)//'/FLUXCONT',STATUS='OLD',FORM='FORMATTED')  
READ (2,FMT=*)  
DO K = 1,IFRE  
     READ (2,FMT=*) I,XLAM(I),FNUE(I),DUM,TRAD(I)  
END DO  

READ (2,FMT=*) RVMIN1  
PRINT *  
PRINT *,' EFFECTIVE RADIUS = ',SR*RVMIN1/6.96D10  
CLOSE (2)  

WRITE (10,FMT='(3(E17.10,2x))') (XLAM(I),I=1,IFRE)  
WRITE (10,FMT='(3(F16.10,2x))') (FNUE(I),I=1,IFRE)  
WRITE (10,FMT='(3(F16.5,2x))')  (TRAD(I),I=1,IFRE)  
! this is new (appended in order to allow reading of older files)
WRITE (10,FMT='(3(E17.10,2x))') (CLF(I),I=1,ND)  


CLOSE (10)  
CLOSE (11)  
CLOSE (12)  

PRINT *,'FILE OUT_TOT IN CATALOG ',TRIM(CAT),' WRITTEN'  

 9000 FORMAT (1X,I2,10 (1X,G12.4))  

END
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
SUBROUTINE ATFILE(ZEFF,NUPPER,NLEV,NL,LEVNAM)  

USE nlte_type
USE nlte_dim
USE ffr_error

IMPLICIT NONE

!     .. scalar arguments ..
INTEGER(I4B) :: NL,NLEV  
!     ..
!     .. array arguments ..
REAL(DP) ::  ZEFF(ID_ATOMS)  
INTEGER(I4B) :: NUPPER(NLEV)  
CHARACTER LEVNAM(NLEV)*6  
!     ..
!     .. scalars in common ..
INTEGER(I4B) :: IOU,IU,NAT  
!     ..
!     .. local scalars ..
INTEGER(I4B) :: NPREV  
CHARACTER DC*6,FICHERO*20,RET*4  
!     ..
!     .. external subroutines ..
EXTERNAL FFRACC,FFRCDC,FFROUT,RDATOM  
!     ..
!     .. common blocks ..
COMMON /INOUT/IU,IOU,NAT  
!     ..
IU = 37  
IOU = 38  
DC = ':T'  

NAT = 0  
OPEN (1,FILE='ATOM_FILE',STATUS='OLD',FORM='FORMATTED')  
READ (1,FMT='(A)') FICHERO  
CLOSE (1)  

OPEN (UNIT=IU,FILE=FICHERO,FORM='FORMATTED',STATUS='OLD')  
OPEN (UNIT=IOU,FILE='control.dat',STATUS='UNKNOWN')  

CALL FFROUT(IOU,RET)  
IF (ON_ERROR(RET))  GOTO 9000

CALL FFRACC(IU,RET)  
IF (ON_ERROR(RET))  GOTO 9100

CALL FFRCDC(DC)  

NPREV = 1  

RLOOP: DO

CALL RDATOM(ZEFF,NPREV,NUPPER,NLEV,NL,LEVNAM,RET)  
IF (RET.NE.'RET0'.AND.RET.NE.'RET3')  GOTO 9200
IF (RET.EQ.'RET3') GOTO 9990

END DO RLOOP  
!
!       error in output unit exit
!
 9000 CONTINUE  

WRITE (*,FMT='(A)') ' ERROR IN OUTPUT UNIT NUMBER '  
STOP
!
!       error in input unit exit
!
 9100 CONTINUE  
WRITE (*,FMT='(A)') ' ERROR IN INPUT UNIT NUMBER '  
STOP
!  
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

RETURN  

END


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


SUBROUTINE RDATOM(ZEFF,NPREV,NUPPER,NLEV,NL,LEVNAM,RETCHAR)  
!
USE nlte_type
USE nlte_dim
USE ffr_error

IMPLICIT NONE

!       lee y almacena los datos atomicos
!       lista de variables (por orden de appearance):
!               nat: numero de atomo
!               ion: numero de ion dentro del atomo
!               iong: numero de ion en el computo general
!     .. scalar arguments ..
INTEGER(I4B) :: NL,NLEV,NPREV  
!     ..
!     .. array arguments ..
REAL(DP) ::  ZEFF(ID_ATOMS)  
INTEGER(I4B) :: NUPPER(NLEV)  
CHARACTER LEVNAM(NLEV)*6  
!     ..
!     .. scalars in common ..
INTEGER(I4B) :: IOU,IU,NAT  
!     ..
!     .. local scalars ..
REAL(DP) ::  REALVAR  
INTEGER(I4B) :: II,INTEG,NC,NS  
CHARACTER LABEL*4,KEY*6,RETCHAR*4,RET*4  
!     ..
!     .. external subroutines ..
EXTERNAL FFRKEY,FFRLOC,FFRNUM  
!     ..
!     .. common blocks ..
COMMON /INOUT/IU,IOU,NAT  
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
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRNUM(REALVAR,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 9990
ZEFF(NAT) = REALVAR  

PRINT *,' ATOM = ',KEY,' ZEFF = ',ZEFF(NAT)  
CALL FFRNUM(REALVAR,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 9990

CALL FFRNUM(REALVAR,INTEG,RET)  
IF (ON_ERROR(RET)) GOTO 9990

  100 CONTINUE  
CALL FFRKEY(KEY,NC,RET)  
IF (ON_ERROR(RET)) GOTO 9990

!------------- l  levels

IF (KEY.EQ.'L') THEN  
     PRINT *,'L LEVELS'  
     DO II = NPREV,NL  
          NUPPER(II) = NL + 1  
     END DO
     NPREV = NL + 1  

 1000      CONTINUE  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     IF (KEY.EQ.'0') THEN  
          GO TO 100  
     ELSE  
          NL = NL + 1  
          PRINT *,' LEVEL: ',KEY,' CORRESPONDING NO.',NL  
          LEVNAM(NL) = KEY  
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

     STOP 'X-LEVEL FOUND'  
!
!2000            call ffrkey(key,nc,*9990,*9999)
!                        if (key.eq.'0') goto 100
!                nlx=nlx+1
!                call ffrnum(REALVAR, integ,*9990,*9999)
!                call ffrnum(REALVAR,integ,*9990,*9999)
!                call ffrkey(key,nc,*9990,*9999)
!                goto 2000
!

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
     PRINT *,'K LEVEL'  
     DO II = NPREV,NL + 1  
          NUPPER(II) = NL + 1  
     END DO 

     NPREV = NL + 2  
     NL = NL + 1  
     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990
     PRINT *,' LEVEL: ',KEY,' CORRESPONDING NO.',NL  
     LEVNAM(NL) = KEY  

     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL FFRNUM(REALVAR,INTEG,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

     CALL FFRKEY(KEY,NC,RET)  
     IF (ON_ERROR(RET)) GOTO 9990

END IF  
!       the atom is over.


IF (NLEV.LT.NL) STOP 'MORE THAN NLEVS LEVELS IN FILE!!'  

RETCHAR='RET0'
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
