MODULE ffr_error
CONTAINS
LOGICAL FUNCTION ON_ERROR(RET)
IMPLICIT NONE
CHARACTER*4 :: RET
ON_ERROR = RET.NE.'RET0'
RETURN
END FUNCTION ON_ERROR

END MODULE ffr_error
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccc                                                              ccccc
!cccc   free-field read routines : source library                  ccccc
!cccc                                                              ccccc
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!cccc
!cccc   status : 03/03/87
!cccc
!cccc   revison record:
!cccc      date: 02/03/87   programmer: d.husfeld
!cccc            function 'ffrlin' added
!cccc      date: 03/03/87   programmer: d.husfeld
!cccc            subroutines 'ffrkey', 'ffrnch', 'ffrnum' modified
!cccc            to accelerate execution
!cccc      date: 31/03/92   programmer: e. santolaya rey
!cccc            all integer variables changed into implicit type (i-n).
!cccc            implicit double precision (a-h,o-z)
!cccc      date: 14/01/99 polished and changed to f90 (J.Puls)
!cccc      date: 20/01/99 lower case character set included (J.Puls)
!cccc      date: 10/03/99 obsolete fortran features removed (J.Puls)
!cccc      date: march 2006 underscore allowed in key character set (J.Puls)

SUBROUTINE FFRACC(IUN,RETCHAR)  
!
!docs ffracc
!        free-field read : access new input unit
!docc    call ffracc ( iun,retchar )
!           iun    : an integer variable or integer array element which
!                    absolute value identifies the unit number
!                    to be used for further input if it is in the range
!                    from 1 to 999. the referenced unit must already
!                    be opened for sequential and formatted access.
!                    if the value of 'iun' is negative the referenced
!                    unit will be rewound; if it is positive the unit
!                    will be not repositioned.
!                    if the value of 'iun' is zero when ffracc is
!                    called, 'iun' receives the current input unit
!                    number and no further action is taken.
!           ret1   : a statement label in the calling program to which
!                    control is returned from ffracc in case of an error
!docr    latest revision : 05/11/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!     free-field read : unit table                               ffrutab
!     latest revision : 05/11/86                                 ffrutab
!     nun = number of units in table                             ffrutab
!     inpt= input unit number table                              ffrutab
!     icolt= column number table                                 ffrutab
!     iseqt= line sequence number table                          ffrutab
!     lint= line image table                                     ffrutab
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

CHARACTER*6 PRGNAM  
PARAMETER (PRGNAM='FFRACC')  
INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
INTEGER(I4B) ::  MUN  
PARAMETER (MUN=10)  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  IUN  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL,NUN  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!     ..
!     .. arrays in common ..
INTEGER(I4B) ::  ICOLT(MUN),INPT(MUN),ISEQT(MUN)  
CHARACTER LINT(MUN)*80  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IERR  
LOGICAL REW,UOP  
CHARACTER ACMODE*10,FMMODE*10,RETCHAR*4  
!     ..
!     .. intrinsic functions ..
INTRINSIC IABS  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
COMMON /FFRUT1/NUN,INPT,ICOLT,ISEQT  
COMMON /FFRUT2/LINT  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
!.....return current input unit if un is zero
IF (IUN.EQ.0) THEN  
     IUN = INP  
     RETCHAR='RET0'
     RETURN  
END IF  
!
!.....save current file status if unit still exists in table
DO 110 I = 1,NUN  
     IF (INPT(I).EQ.INP) THEN  
          ICOLT(I) = ICOL  
          LINT(I) = LIN  
          ISEQT(I) = ISEQ  

     END IF  
  110 END DO  
!
!.....check iun
REW = IUN .LT. 0  
IUN = IABS(IUN)  
IF (IUN.GT.999) THEN  
     ERRPRG = PRGNAM  
     WRITE (ERRMSG,FMT='(A,I10)') 'NON-ACCEPTABLE UNIT NUMBER :', &
      IUN

     GO TO 9999  

END IF  
INQUIRE (UNIT=IUN,IOSTAT=IERR,OPENED=UOP,ACCESS=ACMODE, &
&        FORM=FMMODE)
IF ((.NOT.UOP) .OR. (IERR.NE.0) .OR. (ACMODE.NE.'SEQUENTIAL') .OR. &
&    (FMMODE.NE.'FORMATTED')) THEN
     ERRPRG = PRGNAM  
    WRITE (ERRMSG,FMT='(A,I3,A)') 'UNIT #',IUN, &
&      ' NOT OPENED PROPERLY'

     GO TO 9999  
END IF  
!
!.....access new unit
DO 210 I = 1,NUN  
     IF (INPT(I).EQ.IUN) THEN  
!        ...unit already exist in table
          INP = IUN  
          ICOL = ICOLT(I)  
          ISEQ = ISEQT(I)  
          LIN = LINT(I)  
          EOF = .FALSE.  

          GO TO 2100  

     END IF  
  210 END DO  
!     ...unit is not contained in table
IF (NUN.GE.MUN) THEN  
     ERRPRG = PRGNAM  
     ERRMSG = 'UNIT BUFFER IS FULL'  

     GO TO 9999  

END IF  
NUN = NUN + 1  
INPT(NUN) = IUN  
IF (INP.NE.IUN) THEN  
     INP = IUN  
     LIN = ' '  
     ICOL = 81  
     ISEQ = 0  
     EOF = .FALSE.  

END IF  
 2100 CONTINUE  
!
!.....rewind unit if requested
IF (REW) THEN  
     REWIND INP  
     LIN = ' '  
     ICOL = 81  
     ISEQ = 0  
     EOF = .FALSE.  
END IF  
!
!.....verify
IF (IVER.GE.1) WRITE (IOUT,FMT='(A,I3,A)', &
&    IOSTAT=IERR) ' ** UNIT #',IUN,' ACCESSED'
     RETCHAR='RET0'
RETURN  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET1'
RETURN   
!
END
BLOCK DATA FFRBDT  
!
!docs ffrbdt
!        free-field read : define initial values
!docr    latest revision : 20/01/99
!docp    programmer      : d.husfeld/chr.wiethaus
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!     free-field read : unit table                               ffrutab
!     latest revision : 05/11/86                                 ffrutab
!     nun = number of units in table                             ffrutab
!     inpt= input unit number table                              ffrutab
!     icolt= column number table                                 ffrutab
!     iseqt= line sequence number table                          ffrutab
!     lint= line image table                                     ffrutab
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
INTEGER(I4B) ::  MUN  
PARAMETER (MUN=10)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL,NUN  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!     ..
!     .. arrays in common ..
INTEGER(I4B) ::  ICOLT(MUN),INPT(MUN),ISEQT(MUN)  
CHARACTER LINT(MUN)*80  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
COMMON /FFRUT1/NUN,INPT,ICOLT,ISEQT  
COMMON /FFRUT2/LINT  
!     ..
!     .. data statements ..
!
!     ffrctrl
!
!     ffrerrs
!
!     ffrstat
!
!     ffrutab
DATA IOUT/0/,IVER/0/,NOR/.TRUE./,CDI/.FALSE./,SDI/.FALSE./, &
& CDC/' '/,SDC/' '/,INT/':'/ 
!
!     fixing the problem when using small letters in formal_input.
!     because of limitation of columns in fortran and due to portability
!     i had to devide 'kcs' in two pieces.
!     'kcl' is set to '62'.
!
!     13/3/1998 chr. wiethaus
!
DATA KCS(1:26)  /'ABCDEFGHIJKLMNOPQRSTUVWXYZ'/
DATA KCS(27:63) /'abcdefghijklmnopqrstuvwxyz0123456789_'/ 
DATA KCL /63/
!
DATA ERRPRG/' '/,ERRMSG/' '/  
DATA INP/0/,ICOL/81/,ISEQ/0/,EOF/.FALSE./,LIN/' '/  
DATA NUN/0/,INPT/MUN*0/,ICOLT/MUN*81/,ISEQT/MUN*0/,LINT/MUN*' '/  
!     ..
!
END
SUBROUTINE FFRCDC(DC)  
!
!docs ffrcdc
!        free-field read : define comment delimiting character
!docc    call ffrcdc ( dc )
!           dc     : a character variable that defines in its first
!                    position the new string delimiting character.
!                    however, if dc contains in its first five positions
!                    the string 'clear', the delimiting character
!                    becomes undefined.
!docr    latest revision : 10/10/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
CHARACTER DC* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  IOUT,IVER,KCL  
LOGICAL CDI,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  IERR  
!     ..
!     .. intrinsic functions ..
INTRINSIC LEN  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
IF ((LEN(DC).GE.5) .AND. (DC(1:5).EQ.'CLEAR')) THEN  
     CDI = .FALSE.  
     CDC = ' '  

    IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&        IOSTAT=IERR) ' ** COMMENT DELIMITER CLEARED'
ELSE  
     CDI = .TRUE.  
     CDC = DC(1:1)  
    IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&        IOSTAT=IERR) ' ** COMMENT DELIMITER IS SET TO '''//CDC// &
&        ''''

END IF  
RETURN  

END
SUBROUTINE FFRDMP  
!
!docs ffrdmp
!        free-field read : dump current status
!docc    call ffrdmp
!docr    latest revision : 05/11/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IO,NUN  
LOGICAL LCL,LOP  
CHARACTER WORD1*7,WORD2*7  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
!     ..
!     .. data statements ..
DATA LOP/.FALSE./,LCL/.FALSE./  
!     ..
!
!     determine output unit for dump
IF (IOUT.NE.0) THEN  
!     ...if output unit is defined for ffr-routines then use that

     IO = IOUT  
ELSE  
     INQUIRE (FILE='OUTPUT',OPENED=LOP,NUMBER=NUN)  
     IF (LOP) THEN  
!     ...if file 'output' is opened use its unit number

          IO = NUN  
     ELSE  
!     search free unit number and open file 'output' via this unit
          DO 110 I = 1,999  
               INQUIRE (UNIT=I,OPENED=LOP)  
               IF (LOP) GO TO 110  
               IO = I  
               OPEN (UNIT=IO,FILE='OUTPUT',STATUS='UNKNOWN')  
               LCL = .TRUE.  

               GO TO 1110  
  110           CONTINUE  
!        ...output file can not be accessed, return
          RETURN  
!
 1110           CONTINUE  

     END IF  
END IF  
!
!.....print error status
IF (ERRMSG.EQ.' ') THEN  

     WRITE (IO,FMT='(A)') ' NO ERROR DETECTED'  
ELSE  
     WRITE (IO,FMT='(A)') ' *** ERROR DETECTED BY '//ERRPRG// &
      ' : '//ERRMSG
END IF  
!
!.....print input file status
IF (EOF) THEN  

     WRITE (IO,FMT='(A,I3,A,I5,A)') ' CURRENT INPUT UNIT #',INP, &
      ' AT EOF (FOUND AFTER LINE #',ISEQ,')'
ELSE  
     WRITE (IO,FMT='(A,I3,A,I5)') ' CURRENT INPUT UNIT #',INP, &
      ' AT LINE #',ISEQ
     IF (ICOL.LE.40) THEN  

          WRITE (IO,FMT='(41A1,A,I2)') (' ',I=0,ICOL-1),'!', &
           (' ',I=ICOL+1,40),' LINE AT COLUMN #',ICOL
     ELSE  
          WRITE (IO,FMT='(A17,I2,21X,50A1)') ' LINE AT COLUMN #', &
           ICOL, (' ',I=41,ICOL-1),'!'

     END IF  
     WRITE (IO,FMT='(1X,A)') LIN  

END IF  
!
!.....print control variables
WRITE (IO,FMT='(A,I1,A,I3)') ' VERIFICATION LEVEL ',IVER, &
&  ' , PROTOICOL DIRECTED TO UNIT #',IOUT
WORD1 = 'NOT SET'  
WORD2 = 'NOT SET'  
IF (CDI) WORD1 = ''''//CDC//''''  
IF (SDI) WORD2 = ''''//SDC//''''  
WRITE (IO,FMT='(A)') ' INTERRUPT CHARACTER '''//INT// &
&  ''', STRING DELIMITER '//WORD2//', COMMENT DELIMITER '//WORD1
WRITE (IO,FMT='(A)') ' KEYWORD CHARACTER SET '''//KCS(1:KCL)//''''  
!
!.....close output file if permitted
IF (LCL) CLOSE (UNIT=IO)  
!

RETURN  
END
SUBROUTINE FFRINT(IC)  
!
!docs ffrint
!        free-field read : define interrupt character
!docc    call ffrint ( ic )
!           ic     : a character variable that defines in its first
!                    position the new interrupt character
!docr    latest revision : 26/09/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
CHARACTER IC* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  IOUT,IVER,KCL  
LOGICAL CDI,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  IERR  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
INT = IC(1:1)  
IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&    IOSTAT=IERR) ' ** INTERRUPT CHARACTER IS SET TO '''//INT//''''
RETURN  

END
SUBROUTINE FFRIPR  
!
!docs ffripr
!        free-field read : interrupt processing
!docc    *** for internal use only
!        call ffripr
!docr    latest revision : 20/01/99
!docp    programmer      : d.husfeld/ chr. wiethaus
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IERR,J  
CHARACTER CHR*1,DEL*1,EQU*1  
!     ..
!     .. external subroutines ..
EXTERNAL FFRIFC  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
!     ..
!     .. data statements ..
DATA EQU/'='/  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
!
!.....confirm call
IF (LIN(1:1).NE.INT) RETURN  
!
CHR = LIN(2:2)  
DEL = LIN(3:3)  
!
!.....c  comment delimiter
IF (CHR.EQ.'C') THEN  
     IF (DEL.EQ.EQU) THEN  
          CDI = .TRUE.  
          CDC = LIN(4:4)  

        IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&            IOSTAT=IERR) ' ** COMMENT DELIMITER SET TO '''//CDC// &
&            ''''
     ELSE  
          CDI = .FALSE.  
          CDC = ' '  
        IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&            IOSTAT=IERR) ' ** COMMENT DELIMITER CLEARED'
     END IF  
!
!.....d  string delimiter
ELSE IF (CHR.EQ.'D') THEN  
     IF (DEL.EQ.EQU) THEN  
          SDI = .TRUE.  
          SDC = LIN(4:4)  

        IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&            IOSTAT=IERR) ' ** STRING DELIMITER IS SET TO '''// &
&            SDC//''''
     ELSE  
          SDI = .FALSE.  
          SDC = ' '  
        IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&            IOSTAT=IERR) ' ** STRING DELIMITER CLEARED'
     END IF  
!
!.....i  interrupt character
ELSE IF (CHR.EQ.'I') THEN  
     INT = LIN(3:3)  
    IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&        IOSTAT=IERR) ' ** INTERRUPT CHARACTER IS SET TO '''//INT// &
&        ''''
!
!.....k  keyword character set
ELSE IF (CHR.EQ.'K') THEN  
     IF (DEL.EQ.EQU) THEN  
          KCS(1:1) = LIN(4:4)  
          KCL = 1  
          DO 210 I = 2,KCLM  
               J = I + 3  
               IF (LIN(J:J).EQ.KCS(KCL:KCL)) GO TO 1210  
               KCS(I:I) = LIN(J:J)  
               KCL = I  
  210           CONTINUE  

 1210           CONTINUE  
     ELSE  
          KCS(1:26)  = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
          KCS(27:62) = 'abcdefghijklmnopqrstuvwxyz0123456789'
          KCL = 62                                                           
     END IF  
    IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&        IOSTAT=IERR) ' ** NEW KEYWORD CHARACTER SET IS :', &
&        '    '''//KCS(1:KCL)//''''
!
!.....s  send line to interface routine 'ffrifc'
ELSE IF (CHR.EQ.'S') THEN  
     CALL FFRIFC(LIN)  
    IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&        IOSTAT=IERR) ' ** FOLLOWING LINE SENT TO INTERFACE :', &
&        ''''//LIN//''''
!
!.....v  verification level
ELSE IF (CHR.EQ.'V') THEN  
     READ (LIN(3:3),FMT='(I1)',IOSTAT=IERR) I  
     IF (IERR.EQ.0) IVER = I  
     IF (IOUT.EQ.0) IVER = 0  
!
!.....any other character : comment line
ELSE  
     IF (IVER.GE.1) WRITE (IOUT,FMT='(A)',IOSTAT=IERR) ' *'// &
      LIN(3:80)
END IF  
!
RETURN  

END
SUBROUTINE FFRKCS(CS)  
!
!docs ffrkcs
!        free-field read : defined keyword character set
!docc    call ffrkcs ( cs )
!           cs     : a character variable that defines the character set
!                    acceptable in any keyword following.
!                    if the length of cs exceeds the maximum length
!                    allowed, cs will be truncated.
!           sl     : a statement label in the calling program to which
!                    control is returned from ffrkcs in case of an error
!docr    latest revision : 26/09/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
CHARACTER CS* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  IOUT,IVER,KCL  
LOGICAL CDI,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  IERR,L  
!     ..
!     .. intrinsic functions ..
INTRINSIC LEN  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
L = LEN(CS)  
IF (L.GT.KCLM) THEN  
    IF (IVER.GE.1) WRITE (IOUT,FMT='(A,I3,A)',IOSTAT=IERR) &
&        ' ** NEW KEYWORD CHARACTER SET TRUNCATED TO ',KCLM, &
&        ' CHARACTERS'
     KCL = KCLM  

     KCS = CS(1:KCLM)  
ELSE  
     KCL = L  
     KCS(1:KCL) = CS  
END IF  
!
IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&    IOSTAT=IERR) ' ** NEW KEYWORD CHARACTER SET IS :', &
&    '    '''//KCS(1:KCL)//''''
RETURN  

END
SUBROUTINE FFRKEY(KEY,NC,RETCHAR)  
!
!docs ffrkey
!        free-field read : get next keyword
!docc    call ffrkey ( key,nc, retchar )
!           key    : a character variable that receives the next keyword
!                    found in the input stream
!           nc     : an integer variable that returns the number
!                    of characters stored into key
!           ret1    : a statement label in the calling program to which
!                    control is returned from ffrkey when an eof is
!                    found during search for a keyword.
!                    note that the keyword will be successfully read
!                    even when it is terminated by an eof.
!           ret2    : a statement label in the calling program to which
!                    control is returned from ffrkey in case of an error
!docr    latest revision : 03/03/87
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  NC  
CHARACTER KEY* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  IOUT,IVER,KCL  
LOGICAL CDI,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IERR,L  
CHARACTER CH*1, RETCHAR*4, RET*4  
!     ..
!     .. external subroutines ..
EXTERNAL FFRNCH  
!     ..
!     .. intrinsic functions ..
INTRINSIC INDEX,LEN  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
L = LEN(KEY)  
KEY = ' '  
IF (IVER.GE.3) WRITE (IOUT,FMT='(A)', &
&    IOSTAT=IERR) ' **** READ NEXT KEYWORD'
!
!.....scan input for first acceptable character
 1000 CONTINUE  
CALL FFRNCH(CH,RET)  

SELECT CASE(RET)
CASE('RET1')
  GOTO 9000
CASE('RET2')
  GOTO 9999
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN CONDITION'
END SELECT

IF (CH.EQ.' ') GO TO 1000  
IF (INDEX(KCS(1:KCL),CH).EQ.0) GO TO 1000  
!
!.....first acceptable character found.
!     now icollect characters in key until either key is full
!     or next non-acceptable character occurs.
KEY(1:1) = CH  
NC = 1  
DO 110 I = 2,L  
     CALL FFRNCH(CH,RET)  

     SELECT CASE(RET)
     CASE('RET1')
       GOTO 1110
     CASE('RET2')
       GOTO 9999
     CASE('RET0')
       CONTINUE
     CASE DEFAULT
       STOP ' WRONG RETURN CONDITION'
     END SELECT

     IF (INDEX(KCS(1:KCL),CH).EQ.0) GO TO 1110  
     KEY(I:I) = CH  
     NC = I  
  110 END DO  
CALL FFRNCH(CH,RET)  

SELECT CASE(RET)
CASE('RET1')
  GOTO 1110
CASE('RET2')
  GOTO 9999
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN CONDITION'
END SELECT

 1110 CONTINUE  
!
IF (IVER.GE.2) WRITE (IOUT,FMT='(A,A)', &
&    IOSTAT=IERR) ' *** KEYWORD FOUND : ',KEY(1:NC)
!
RETCHAR='RET0'
RETURN  
!
!.....eof condition exit
 9000 CONTINUE  
RETCHAR='RET1'
RETURN  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET2'
RETURN   
!
END
CHARACTER*(*) FUNCTION FFRLIN()  
!
!docs ffrlin
!        free-field read : supply current input line
!docc    line = ffrlin()
!           line   : a character variable, character variable substring,
!                    character array element or character array element
!                    substring that receives a copy of the current input
!                    line.
!docr    latest revision : 02/03/87
!docp     programmmer    : d. husfeld
!doce
!
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!     .. scalars in common ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  ICOL,INP,ISEQ  
LOGICAL EOF  
CHARACTER LIN*80  
!     ..
!     .. common blocks ..
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
!     ..
FFRLIN = LIN  

RETURN  
END
SUBROUTINE FFRLOC(STR,RETCHAR)  
!
!docs ffrloc
!        free-field read : locate specified string and position read
!        pointer behind it.
!docc    call ffrloc ( str,retchar )
!           str    : a character expression that specifies the string
!                    to be located.
!           ret1   : a statement label in the calling program to which
!                   control is returned from ffrloc when an eof is found
!           ret2   : a statement label in the calling program to which
!                    control is returned from ffrloc in case of an error
!docr    latest revision : 05/11/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
CHARACTER STR* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IERR,L  
CHARACTER CH1*1,CH2*1,RETCHAR*4,RET*4  
!     ..
!     .. external subroutines ..
EXTERNAL FFRNCH  
!     ..
!     .. intrinsic functions ..
INTRINSIC LEN  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
IF (IVER.GE.3) WRITE (IOUT,FMT='(A,A)', &
&    IOSTAT=IERR) ' **** LOCATING STRING : ',STR
!
L = LEN(STR)  
CH2 = STR(1:1)  
   10 CONTINUE  
CALL FFRNCH(CH1,RET)  

SELECT CASE(RET)
CASE('RET1')
  GOTO 9998
CASE('RET2')
  GOTO 9999
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN CONDITION'
END SELECT

   20 CONTINUE  
IF (CH1.NE.CH2) GO TO 10  
DO 30 I = 2,L  
     CALL FFRNCH(CH1,RET)  

     SELECT CASE(RET)
     CASE('RET1')
       GOTO 9998
     CASE('RET2')
       GOTO 9999
     CASE('RET0')
       CONTINUE
     CASE DEFAULT
       STOP ' WRONG RETURN CONDITION'
     END SELECT

     IF (CH1.NE.STR(I:I)) GO TO 20  
   30 END DO  
!
!.....verify
IF (IVER.GE.2) WRITE (IOUT,FMT='(A,I5)', &
&    IOSTAT=IERR) ' *** SPECIFIED STRING LOCATED IN LINE #',ISEQ
!
RETCHAR='RET0'
RETURN  
!
!.....eof condition exit
 9998 CONTINUE  
ERRPRG = 'FFRLOC'  
ERRMSG = 'EOF WHILE LOCATING '''//STR//''''  
RETCHAR='RET1'
RETURN  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET2'
RETURN  
!
END
SUBROUTINE FFRNCD(RETCHAR)  
!
!docs ffrncd
!        free-field read : load next card into memory
!docc    call ffrncd ( retchar )
!           ret1   : a statement label in the calling program to which
!                   control is returned from ffrncd when an eof is found
!           ret2   : a statement label in the calling program to which
!                    control is returned from ffrncd in case of an error
!docr    latest revision : 05/11/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

CHARACTER*6 PRGNAM  
PARAMETER (PRGNAM='FFRNCD')  
INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!
CHARACTER*4 :: RETCHAR
!     ..
!     .. external subroutines ..
EXTERNAL FFRIPR  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
!.....return immediately if eof flag is already set
IF (EOF) THEN
  RETCHAR='RET1'
  RETURN   
ENDIF
!
 1000 CONTINUE  
READ (INP,FMT='(A)',END=9000,ERR=9998) LIN  
ICOL = 1  
ISEQ = ISEQ + 1  
EOF = .FALSE.  
!
!.....check for interrupt
IF (LIN(1:1).EQ.INT) THEN  
     CALL FFRIPR  
     GO TO 1000  
END IF  
!
!.....normal exit
RETCHAR='RET0'
RETURN  
!
!.....eof condition exit
 9000 CONTINUE  
ERRPRG = PRGNAM  
WRITE (ERRMSG,FMT='(A,I3)') 'EOF ENCOUNTERED ON UNIT #',INP  
ICOL = 1  
EOF = .TRUE.  
LIN = ' '  
RETCHAR='RET1'
RETURN
!
!.....read error exit
 9998 CONTINUE  
ERRPRG = PRGNAM  
WRITE (ERRMSG,FMT='(A,I3)') 'READ ERROR ON UNIT #',INP  
GO TO 9999  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET2'
RETURN   
!
END
SUBROUTINE FFRNCH(CH,RETCHAR)  
!
!docs ffrnch
!        free-field read : get next character
!docc    *** for internal use only
!        call ffrnch ( ch,retchar )
!           ch     : a character variable that receives in its first po-
!                    sition the next character found in the input stream
!           ret1   : a statement label in the calling program to which
!                   control is returned from ffrnch when an eof is found
!           ret2   : a statement label in the calling program to which
!                    control is returned from ffrnch in case of an error
!docr    latest revision : 03/03/87
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
CHARACTER CH* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IICOL  
CHARACTER*4 :: RETCHAR,RET
!     ..
!     .. external subroutines ..
EXTERNAL FFRNCD  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
IF (ICOL.GT.80) THEN
  CALL FFRNCD(RET)  
  SELECT CASE(RET)
    CASE('RET1')
    GOTO 9000
  CASE('RET2')
    GOTO 9999
  CASE('RET0')
    CONTINUE
  CASE DEFAULT
    STOP ' WRONG RETURN CONDITION'
  END SELECT
ENDIF

CH(1:1) = LIN(ICOL:ICOL)  
ICOL = ICOL + 1  
RETCHAR='RET0'
IF (.NOT.CDI) RETURN  
IF (CH(1:1).NE.CDC) RETURN  
!     ...comment delimiter found. skip to next comment delimiter or end
!        of line and return a blank in ch.
CH(1:1) = ' '  
IICOL = ICOL  
DO 100 I = IICOL,80  
     ICOL = I  
     IF (LIN(I:I).EQ.CDC) GO TO 110  
  100 END DO  
  110 CONTINUE  
ICOL = ICOL + 1  
RETURN  
!
!.....eof condition exit
 9000 CONTINUE  
RETCHAR='RET1'
RETURN  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET2'
RETURN  
!
END
SUBROUTINE FFRNUM(REALVAR,INTEG,RETCHAR)  
!
!docs ffrnum
!        free-field read : get next number
!docc    call ffrnum ( realvar,integ,retchar )
!           realvar: is a real variable that receives the number read
!           integ  : an integer variable that receives the number read
!           ret1   : a statement label in the calling program to which
!                    control is returned from ffrnum when an eof is
!                    found during search for a valid number.
!                    note that the number will be correctly read even
!                    when it is terminated by an eof.
!           ret2   : a statement label in the calling program to which
!                   control is returned from ffrnum in case of an error.
!docr    latest revision : 03/03/87
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

CHARACTER*6 PRGNAM  
PARAMETER (PRGNAM='FFRNUM')  
INTEGER(I4B) ::  ML  
PARAMETER (ML=50)  
INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
REAL(DP) ::  REALVAR  
INTEGER(I4B) ::  INTEG  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  IOUT,IVER,KCL  
LOGICAL CDI,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,KCS* (KCLM)  
!     ..
!     .. local scalars ..
REAL(DP) ::  INTMAX  
INTEGER(I4B) ::  I,IERR,JUMP,L  
LOGICAL DIGIT  
CHARACTER CH*1,FORM1*10,SET*15,NUM* (ML),RETCHAR*4,RET*4
!     ..
!     .. external subroutines ..
EXTERNAL FFRNCH  
!     ..
!     .. intrinsic functions ..
INTRINSIC ABS,INDEX,MIN,NINT,SIGN  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
!     ..
!     .. data statements ..
DATA SET/'.0123456789-+ED'/,INTMAX/32767/  
!     ..
!    +                  ,intmax/281474976710655/
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
ERRPRG = ' '  
ERRMSG = ' '  
!
IF (IVER.GE.3) WRITE (IOUT,FMT='(A)', &
&    IOSTAT=IERR) ' **** READ NEXT NUMBER'
!
!.....clear
 1000 CONTINUE  
L = 0  
NUM = ' '  
DIGIT = .FALSE.  
!
!.....skip over non-acceptable characters
 1010 CONTINUE  
CALL FFRNCH(CH,RET)

SELECT CASE(RET)
CASE('RET1')
  GOTO 9000
CASE('RET2')
  GOTO 9999
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN CONDITION'
END SELECT

IF (CH.EQ.' ') GO TO 1010  
I = INDEX(SET,CH)  
IF (I.EQ.0) GO TO 1010  
!
!.....sign field
 2000 CONTINUE  
IF (I.EQ.12) THEN  
     JUMP=2100  
     GO TO 7000  
END IF  
!
!.....integer field
 2100 CONTINUE  
JUMP=2100  
IF ((I.GE.2) .AND. (I.LE.11)) THEN  
     DIGIT = .TRUE.  

     GO TO 7000  
END IF  
!
!.....decimal point
 2200 CONTINUE  
IF (I.EQ.1) THEN  
     JUMP=2300  
     GO TO 7000  
END IF  
!
!.....fractional field
 2300 CONTINUE  
JUMP=2300  
IF ((I.GE.2) .AND. (I.LE.11)) THEN  
     DIGIT = .TRUE.  
     GO TO 7000  
END IF  
!
!.....restart read if no digit was found yet
IF (.NOT.DIGIT) GO TO 1000  
!
!.....exponent indicator field
 2400 CONTINUE  
IF ((I.EQ.14) .OR. (I.EQ.15)) THEN  
     DIGIT = .FALSE.  
     JUMP=2500  
     GO TO 7000  
END IF  
!
!.....exponent sign field
 2500 CONTINUE  
IF ((I.EQ.12) .OR. (I.EQ.13)) THEN  
     DIGIT = .FALSE.  
     JUMP=2600  
     GO TO 7000  
END IF  
!
!.....exponent integer field
 2600 CONTINUE  
JUMP=2600  
IF ((I.GE.2) .AND. (I.LE.11)) THEN  
     DIGIT = .TRUE.  
     GO TO 7000  
END IF  
!
!.....store '0' if no exponent integer was read yet
IF (.NOT.DIGIT) THEN  
     IF (L.GE.ML) GO TO 8000  
     L = L + 1  
     NUM(L:L) = '0'  
END IF  
!
!.....read number into real variable
!
 3000 CONTINUE  
WRITE (FORM1,FMT='(A,I2,A)') '(E',L,'.0)'  
!
!.....high art of fortran: if input has decimal point, location of decimal
!     point overrides format specification: thus, only length of field has
!     to be correct
!
READ (NUM,FMT=FORM1,ERR=9998) REALVAR  
!
!.....convert into integer variable
INTEG = NINT(SIGN(MIN(ABS(REALVAR),INTMAX),REALVAR))  
!
!.....verify
IF (IVER.GE.2) WRITE (IOUT,FMT='(A,1PG21.14,A,I6)') &
&    ' *** NUMBER FOUND :',REALVAR,' <-> ',INTEG
!
!.....normal exit
RETCHAR='RET0'
RETURN  
!
!.....procedure to store current character and to fetch new one
 7000 CONTINUE  
IF (L.GE.ML) GO TO 8000  
L = L + 1  
NUM(L:L) = CH  
CALL FFRNCH(CH,RET)

SELECT CASE(RET)
CASE('RET1')
  GOTO 3000
CASE('RET2')
  GOTO 9999
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN CONDITION'
END SELECT

I = INDEX(SET,CH)  
SELECT CASE (JUMP)
  CASE (2000)
    GOTO 2000
  CASE (2100)
    GOTO 2100
  CASE (2200)
    GOTO 2200
  CASE (2300)
    GOTO 2300
  CASE (2400)
    GOTO 2400
  CASE (2500)
    GOTO 2500
  CASE (2600)
    GOTO 2600
  CASE DEFAULT
    STOP 'ERROR IN JUMP'
END SELECT      
!
!.....error exit when read buffer 'num' is full
 8000 CONTINUE  
ERRPRG = PRGNAM  
ERRMSG = 'BUFFER FULL DURING READ OF NUMBER'  
GO TO 9999  
!
!.....eof condition exit
 9000 CONTINUE  
RETCHAR='RET1'
RETURN
!
!.....read error exit
 9998 CONTINUE  
ERRPRG = PRGNAM  
ERRMSG = 'ERROR DURING READ OF NUMBER'  
GO TO 9999  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET2'
RETURN  
!
END
SUBROUTINE FFROUT(IUN,RETCHAR)  
!
!docs ffrout
!        free-field read : define output unit number
!docc    call ffrout ( iun,retchar )
!           iun    : an integer variable that identifies the unit number
!                    to be used for any further output of the free-field
!                    read routines if its value is in the range from 1
!                    to 999. the referenced output unit must already be
!                    be opened for sequential and formatted access.
!                    if the value of iun is zero when ffrout is called
!                    un receives the current output unit number and no
!                    further action is taken.
!           ret1   : a statement label in the calling program to which
!                    control is returned from ffrout in case of an error
!docr    latest revision : 10/10/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  IUN  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  IOUT,IVER,KCL  
LOGICAL CDI,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  IERR  
LOGICAL UOP  
CHARACTER ACMODE*10,FMMODE*10,RETCHAR*4
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
!.....return current output unit number if iun is zero
RETCHAR='RET0'
IF (IUN.EQ.0) THEN  
     IUN = IOUT  
     RETURN  
END IF  
!
!.....check iun
IF ((IUN.LT.1) .OR. (IUN.GT.999)) THEN  
    IF (IVER.GE.1) WRITE (IOUT,FMT='(A,I10)', &
&        IOSTAT=IERR) ' *** NON-ACCEPTABLE UNIT NUMBER ',IUN

     GO TO 9999  

END IF  
INQUIRE (UNIT=IUN,IOSTAT=IERR,OPENED=UOP,ACCESS=ACMODE, &
&        FORM=FMMODE)
IF ((.NOT.UOP) .OR. (IERR.NE.0) .OR. (ACMODE.NE.'SEQUENTIAL') .OR. &
&    (FMMODE.NE.'FORMATTED')) THEN
    IF (IVER.GE.1) WRITE (IOUT,FMT='(A,I3,A)', &
&        IOSTAT=IERR) ' *** UNIT #',IUN,' NOT OPENED PROPERLY'

     GO TO 9999  
END IF  
!
!.....accept unit number
IF (IVER.GE.1) WRITE (IOUT,FMT='(A,I3)', &
&    IOSTAT=IERR) ' ** SWITCHING TO OUTPUT UNIT #',IUN
IOUT = IUN  
RETURN  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET1'
RETURN
!
END
SUBROUTINE FFRRTN(IUN)  
!
!docs ffrrtn
!        free-field read : return specified input unit
!docc    call ffrrtn ( iun )
!           iun     : an integer variable which value defines the unit
!                    to be closed for free-field read
!docr    latest revision : 26/09/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!     free-field read : unit table                               ffrutab
!     latest revision : 05/11/86                                 ffrutab
!     nun = number of units in table                             ffrutab
!     inpt= input unit number table                              ffrutab
!     icolt= column number table                                 ffrutab
!     iseqt= line sequence number table                          ffrutab
!     lint= line image table                                     ffrutab
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
INTEGER(I4B) ::  MUN  
PARAMETER (MUN=10)  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  IUN  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL,NUN  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!     ..
!     .. arrays in common ..
INTEGER(I4B) ::  ICOLT(MUN),INPT(MUN),ISEQT(MUN)  
CHARACTER LINT(MUN)*80  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IERR,J  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
COMMON /FFRUT1/NUN,INPT,ICOLT,ISEQT  
COMMON /FFRUT2/LINT  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
!.....search unit number in table
DO 120 I = 1,NUN  
     IF (INPT(I).EQ.IUN) THEN  
!           ...shift all following entries one step down
          DO 110 J = I + 1,NUN  
               INPT(J-1) = INPT(J)  
               ICOLT(J-1) = ICOLT(J)  
               LINT(J-1) = LINT(J)  
  110           CONTINUE  
          NUN = NUN - 1  

          GO TO 1200  

     END IF  
  120 END DO  
 1200 CONTINUE  
IF (IVER.GE.1) WRITE (IOUT,FMT='(A,I3,A)', &
&    IOSTAT=IERR) ' ** UNIT #',IUN,' DISCARDED'
RETURN  

END
SUBROUTINE FFRSDC(DC)  
!
!docs ffrsdc
!        free-field read : define string delimiting character
!docc    call ffrsdc ( dc )
!           dc     : a character variable that defines in its first
!                    position the new string delimiting character.
!                    however, if dc contains in its first five positions
!                    the string 'clear', the delimiting character
!                    becomes undefined.
!docr    latest revision : 26/09/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
CHARACTER DC* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  IOUT,IVER,KCL  
LOGICAL CDI,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  IERR  
!     ..
!     .. intrinsic functions ..
INTRINSIC LEN  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
IF ((LEN(DC).GE.5) .AND. (DC(1:5).EQ.'CLEAR')) THEN  
     SDI = .FALSE.  
     SDC = ' '  

    IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&        IOSTAT=IERR) ' ** STRING DELIMITER CLEARED'
ELSE  
     SDI = .TRUE.  
     SDC = DC(1:1)  
    IF (IVER.GE.1) WRITE (IOUT,FMT='(A)', &
&        IOSTAT=IERR) ' ** STRING DELIMITER IS SET TO '''//SDC// &
&        ''''

END IF  
RETURN  

END
SUBROUTINE FFRSTR(STR,NC,RETCHAR)  
!
!docs ffrstr
!        free-field read : get string
!docc    call ffrstr ( str,nc,retchar )
!           str    : a character variable that receives the string
!           nc     : an integer variable that returns the number of
!                    characters stored into str
!           ret1   : a statement label in the calling program to which
!                   control is returned from ffrstr when an eof is found
!           ret2   : a statement label in the calling program to which
!                    control is returned from ffrstr in case of an error
!docr    latest revision : 05/11/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  NC  
CHARACTER STR* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  IOUT,IVER,KCL  
LOGICAL CDI,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IERR,L  
LOGICAL CDISAV  
CHARACTER CH*1,RETCHAR*4,RET*4
!     ..
!     .. external subroutines ..
EXTERNAL FFRNCH  
!     ..
!     .. intrinsic functions ..
INTRINSIC LEN  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
IF (IVER.GE.3) WRITE (IOUT,FMT='(A)', &
&    IOSTAT=IERR) ' **** READ STRING'
!
L = LEN(STR)  
STR = ' '  
NC = 0  
!
!.....inhibit temporarily the comment delimiter detection
CDISAV = CDI  
CDI = .FALSE.  
!
!.....icollect in str all characters until either str is full
!     or the string delimiter (if any is defined) is encountered
 1000 CONTINUE  
DO 110 I = 1,L  
     CALL FFRNCH(CH,RET)  

     SELECT CASE(RET)
     CASE('RET1')
       GOTO 9000
     CASE('RET2')
       GOTO 9999
     CASE('RET0')
       CONTINUE
     CASE DEFAULT
       STOP ' WRONG RETURN CONDITION'
     END SELECT

     IF ((SDI) .AND. (CH.EQ.SDC)) GO TO 1110  
     STR(I:I) = CH  
     NC = I  
  110 END DO  
 1110 CONTINUE  
IF (NC.EQ.0) GO TO 1000  
!
!.....restore comment delimiter
CDI = CDISAV  
!
!.....verify
IF (IVER.GE.2) WRITE (IOUT,FMT='(A,A)', &
&    IOSTAT=IERR) ' *** STRING FOUND : ',STR(1:NC)
!
RETCHAR='RET0'
RETURN  
!
!.....eof condition exit
 9000 CONTINUE  
RETCHAR='RET1'
RETURN  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET2'
RETURN
!
END
SUBROUTINE FFRVER(IV)  
!
!docs ffrver
!        free-field read : define verification level
!docc    call ffrver ( iv )
!           iv     : an integer variable that defines (after being taken
!                    to the modulus of 10) the new verification level
!docr    latest revision : 10/10/86
!docp    programmer      : d.husfeld
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
INTEGER(I4B) ::  IV  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  IOUT,IVER,KCL  
LOGICAL CDI,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,KCS* (KCLM)  
!     ..
!     .. intrinsic functions ..
INTRINSIC MOD  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
IVER = MOD(IV,10)  
IF (IOUT.EQ.0) IVER = 0  
RETURN  

END
SUBROUTINE FFRIFC(LINE)  
!     dummy subroutine to satisfy external reference in the free-field
!     program package. it may be used to handle interface interrupts
!     found in the input field which are presently ignored.
!     .. scalar arguments ..
USE nlte_type
IMPLICIT NONE

CHARACTER LINE* (*)  
!     ..

RETURN  
END
!
!----------------------------------------------------------------------
SUBROUTINE FFRLOC1(STR,RETCHAR)  
!
!docs ffrloc1
!        free-field read : locate specified string and position read
!        pointer behind it-only first five characters are checked per
!        line for first character of str!!!
!docc    call ffrloc ( str,retchar )
!           str    : a character expression that specifies the string
!                    to be located.
!           ret    : a statement label in the calling program to which
!                   control is returned from ffrloc when an eof is found
!           ret2   : a statement label in the calling program to which
!                    control is returned from ffrloc in case of an error
!docr    latest revision : 30/04/97
!docp    programmer      : j.puls
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
CHARACTER STR* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IERR,L  
CHARACTER CH1*1,CH2*1,RETCHAR*4,RET*4  
!     ..
!     .. external subroutines ..
EXTERNAL FFRNCH,FFRNCH1  
!     ..
!     .. intrinsic functions ..
INTRINSIC LEN  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
IF (IVER.GE.3) WRITE (IOUT,FMT='(A,A)', &
&    IOSTAT=IERR) ' **** LOCATING STRING : ',STR
!
L = LEN(STR)  
CH2 = STR(1:1)  
   10 CONTINUE  
CALL FFRNCH1(CH1,RET)

SELECT CASE(RET)
CASE('RET1')
  GOTO 9998
CASE('RET2')
  GOTO 9999
CASE('RET0')
  CONTINUE
CASE DEFAULT
  STOP ' WRONG RETURN CONDITION'
END SELECT

   20 CONTINUE  
IF (CH1.NE.CH2) GO TO 10  
DO 30 I = 2,L  
     CALL FFRNCH(CH1,RET)

     SELECT CASE(RET)
     CASE('RET1')
       GOTO 9998
     CASE('RET2')
       GOTO 9999
     CASE('RET0')
       CONTINUE
     CASE DEFAULT
       STOP ' WRONG RETURN CONDITION'
     END SELECT
  
     IF (CH1.NE.STR(I:I)) GO TO 20  
   30 END DO  
!
!.....verify
IF (IVER.GE.2) WRITE (IOUT,FMT='(A,I5)', &
&    IOSTAT=IERR) ' *** SPECIFIED STRING LOCATED IN LINE #',ISEQ
!
RETCHAR='RET0'
RETURN  
!
!.....eof condition exit
 9998 CONTINUE  
ERRPRG = 'FFRLOC1'  
ERRMSG = 'EOF WHILE LOCATING '''//STR//''''  
RETCHAR='RET1'
RETURN  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET2'
RETURN
!
END
!----------------------------------------------------------------------
SUBROUTINE FFRNCH1(CH,RETCHAR)  
!
!docs ffrnch
!        free-field read : get next character in the first five columns
!docc    *** for internal use only
!        call ffrnch ( ch,retchar )
!           ch     : a character variable that receives in its first po-
!                    sition the next character found in the input stream
!           ret1   : a statement label in the calling program to which
!                   control is returned from ffrnch when an eof is found
!           ret2   : a statement label in the calling program to which
!                    control is returned from ffrnch in case of an error
!docr    latest revision : 30/04/93
!docp    programmer      : j.puls
!doce
!
!     free-field read : control variables                        ffrctrl
!     latest revision : 10/10/86                                 ffrctrl
!     iout = output unit number                                  ffrctrl
!     iver = verification level                                  ffrctrl
!     nor = no run indicator                                     ffrctrl
!     cdi = comment delimiting indicator                         ffrctrl
!     sdi = string delimiting indicator                          ffrctrl
!     cdc = comment delimiting character                         ffrctrl
!     sdc = string delimiting character                          ffrctrl
!     int = interrupt character                                  ffrctrl
!     kcs = keyword character set                                ffrctrl
!     kcl = keyword character set length                         ffrctrl
!     free-field read : error status table                       ffrerrs
!     latest revision : 26/09/86                                 ffrerrs
!     errprg = name of routine that detected the error           ffrerrs
!     errmsg = error message                                     ffrerrs
!     free-field read : file status variables                    ffrstat
!     latest revision : 05/11/86                                 ffrstat
!     inp = input unit number                                    ffrstat
!     icol = column number in line                               ffrstat
!     iseq = line sequence number                                ffrstat
!     eof = end-of-file indicator                                ffrstat
!     lin = line image                                           ffrstat
!
!     free-field read : error status clearance                   ffreclr
!     latest revision : 26/09/86                                 ffreclr
!     .. parameters ..
USE nlte_type
IMPLICIT NONE

INTEGER(I4B) ::  KCLM  
PARAMETER (KCLM=64)  
!     ..
!     .. scalar arguments ..
CHARACTER CH* (*)  
!     ..
!     .. scalars in common ..
INTEGER(I4B) ::  ICOL,INP,IOUT,ISEQ,IVER,KCL  
LOGICAL CDI,EOF,NOR,SDI  
CHARACTER CDC*1,INT*1,SDC*1,ERRPRG*6,ERRMSG*40,LIN*80,KCS* (KCLM)  
!     ..
!     .. local scalars ..
INTEGER(I4B) ::  I,IICOL  
CHARACTER*4 :: RETCHAR,RET
!     ..
!     .. external subroutines ..
EXTERNAL FFRNCD  
!     ..
!     .. common blocks ..
COMMON /FFRCT1/IOUT,IVER,NOR,CDI,SDI,KCL  
COMMON /FFRCT2/CDC,SDC,INT,KCS  
COMMON /FFRERR/ERRPRG,ERRMSG  
COMMON /FFRST1/INP,ICOL,EOF,ISEQ  
COMMON /FFRST2/LIN  
!     ..
ERRPRG = ' '  
ERRMSG = ' '  
!
IF (ICOL.GT.5) THEN
  CALL FFRNCD(RET)  
  SELECT CASE(RET)
    CASE('RET1')
    GOTO 9000
  CASE('RET2')
    GOTO 9999
  CASE('RET0')
    CONTINUE
  CASE DEFAULT
    STOP ' WRONG RETURN CONDITION'
  END SELECT
ENDIF

CH(1:1) = LIN(ICOL:ICOL)  
ICOL = ICOL + 1  
RETCHAR='RET0'
IF (.NOT.CDI) RETURN  
IF (CH(1:1).NE.CDC) RETURN  
!     ...comment delimiter found. skip to next comment delimiter or end
!        of line and return a blank in ch.
CH(1:1) = ' '  
IICOL = ICOL  
DO 100 I = IICOL,5  
     ICOL = I  
     IF (LIN(I:I).EQ.CDC) GO TO 110  
  100 END DO  
  110 CONTINUE  
ICOL = ICOL + 1  
RETURN  
!
!.....eof condition exit
 9000 CONTINUE  
RETCHAR='RET1'
RETURN  
!
!.....error exit
 9999 CONTINUE  
RETCHAR='RET2'
RETURN
!
END
