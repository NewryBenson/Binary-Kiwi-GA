! version 1.2.1 (adapted for X-ray treatment)
! version 1.2 (for nlte.f90 vers. 9.0 and higher)
! new constants sqrt(pi) and pi/2.
! new precison qp: possible for recent intel compilers
! version 1.3: new constant eV
MODULE nlte_type
  INTEGER, PARAMETER :: I4B = SELECTED_INT_KIND(9)
  INTEGER, PARAMETER :: SP = KIND(1.0)
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
  INTEGER, PARAMETER :: QP = SELECTED_REAL_KIND(33,4931)
END MODULE nlte_type     

MODULE fund_const
!
USE nlte_type
IMPLICIT NONE
!
!coconst
!
REAL(DP), PARAMETER :: HH=6.6262D-27,AKB=1.38062D-16  
REAL(DP), PARAMETER :: PM=1.67265D-24,AMH=1.67352D-24,CLIGHT=2.997925D10
REAL(DP), PARAMETER :: SAHA=4.141584D-16,SIGSB=5.66956D-5,SIGMAE=0.3977246D0  
REAL(DP), PARAMETER :: HC2 = 2.D0*HH*CLIGHT, HK = HH/AKB  
REAL(DP), PARAMETER :: HKL = HH*CLIGHT/AKB, PI = 3.1415926536D0  
REAL(DP), PARAMETER :: E2MC2=2.817508719D-13  
REAL(DP), PARAMETER :: SQPI = 1.772453851D0, PIHALF=0.5D0*PI  
REAL(DP), PARAMETER :: EV=1.602D-12 !CONVERSION eV to erg
REAL(DP), PARAMETER :: LOG10E=LOG10(EXP(1.D0)) ! used for new interpol. in intermepho

END MODULE fund_const
