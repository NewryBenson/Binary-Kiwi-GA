      program vaul
C modified version for A-SUPERGIANTS using file FLUXCONT from Jo_code
C used for cspns with new absolute flux calibration
C
C        SV     : Filterfunktion nach Matthews & Sandage (1963) ApJ 138,30
C                 Table A1, 3.column, Interpolation nach Detlev Koester
C        WV     : Wellenlaengengitter fuer in die Integration
C        FV     : Flusswerte auf dem Gitter WV
C        OBJEKT : Objektname
C        XMV    : Absolute Helligkeit in mag
C        TEF    : Effektivtemperatur in Kelvin
C        xl     : wavelenght of model input
C        trad   : radiation temperature of model fluxes at model wavel.
C
      CHARACTER HEADLINE*80
      REAL SV(300),WV(300),FV(300),xl(300),trad(300)
      LOGICAL APPROX
C
C
C
C                       Filterfunktion
      DATA  (SV(I), I=1, 130)  /
     $  0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.007,
     $  0.020,0.035,0.053,0.073,0.095,0.120,0.147,0.177,0.209,0.243,
     $  0.280,0.341,0.409,0.483,0.563,0.650,0.743,0.843,0.949,1.061,
     $  1.180,1.275,1.371,1.468,1.565,1.664,1.763,1.864,1.965,2.067,
     $  2.170,2.259,2.345,2.430,2.513,2.594,2.673,2.750,2.825,2.899,
     $  2.970,3.024,3.074,3.118,3.158,3.194,3.224,3.250,3.272,3.288,
     $  3.300,3.312,3.320,3.325,3.326,3.323,3.316,3.305,3.290,3.272,
     $  3.250,3.234,3.216,3.196,3.174,3.150,3.124,3.096,3.066,3.034,
     $  3.000,2.972,2.944,2.915,2.886,2.856,2.826,2.795,2.764,2.732,
     $  2.700,2.666,2.630,2.594,2.558,2.520,2.482,2.442,2.402,2.362,
     $  2.320,2.285,2.251,2.218,2.185,2.153,2.121,2.090,2.059,2.029,
     $  2.000,1.967,1.935,1.902,1.869,1.836,1.803,1.770,1.737,1.703,
     $  1.670,1.634,1.597,1.559,1.521,1.483,1.443,1.403,1.363,1.322
     $  /
      DATA  (SV(I), I= 131, 261)  /
     $  1.280,1.243,1.206,1.169,1.132,1.096,1.060,1.025,0.990,0.955,
     $  0.920,0.889,0.859,0.830,0.801,0.774,0.747,0.722,0.697,0.673,
     $  0.650,0.626,0.602,0.579,0.556,0.534,0.512,0.491,0.470,0.450,
     $  0.430,0.411,0.392,0.374,0.356,0.339,0.322,0.306,0.290,0.275,
     $  0.260,0.248,0.236,0.226,0.216,0.207,0.199,0.192,0.185,0.180,
     $  0.175,0.168,0.162,0.156,0.151,0.146,0.141,0.136,0.132,0.128,
     $  0.125,0.121,0.118,0.115,0.112,0.109,0.107,0.105,0.103,0.101,
     $  0.100,0.097,0.094,0.092,0.089,0.086,0.083,0.080,0.076,0.073,
     $  0.070,0.068,0.065,0.063,0.061,0.059,0.057,0.055,0.053,0.052,
     $  0.050,0.048,0.046,0.044,0.042,0.040,0.038,0.036,0.034,0.032,
     $  0.030,0.029,0.027,0.026,0.025,0.024,0.023,0.022,0.021,0.021,
     $  0.020,0.019,0.018,0.017,0.016,0.015,0.014,0.013,0.012,0.011,
     $  0.010,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,
     $  0.000  /
C ******************************************************************
C   INPUT
C
      write(*,*)' type   T_eff'
      read(*,*) tef
C
      write(*,*)' type de-reddened absolute magnitude'
      read(*,*) xmv

      approx=.false.
C1    write(*,*)' from theoretical fluxes or approx? '
C     write(*,*)' fluxes = 0, approx = 1'
C     read(*,*) inp
C     if (inp.ne.1.and.inp.ne.0) then
C       print*,' wrong input, try once more!!!'
C       goto 1
C     endif
C     if (inp.eq.1) approx=.true.

C     if(.not.approx) then
C     write(*,*)' ?? file FLUXCONT copied from sub-directory ??'

C
      open(1,file='FLUXCONT')
C     read(1,*)HEADLINE
      read(1,*)
      do i=1,300
        read(1,*) ida,xl(i),dum,dum,trad(i),dum,idum
        if(xl(i).lt.4700) goto 301
      enddo
      stop' lambda=4700 not found'
301   close (1)

C     else
C       write(*,*)' type   T_rad/T_eff at V band (example: 0.9)'
C       read(*,*) trtef
C       tradi=trtef*tef
C     endif
C
C *******************************************************************
C
C     Setzen: Zahl der Stuetzstellen
C
      NANU=261
      H=10.
C --- Erzeugung des Wellenlaengengitters
      WV(1)=4700.
      DO 3 I=2,NANU
         WV(I)=WV(I-1)+H
3     CONTINUE
C ***********************************************************************
C
C    generating interpolated flux values for all WV points
C
C    finding out wavelength points in V-filter of model fluxes
C
      wmin=WV(1)
      wmax=WV(NANU)
      do i=1,nanu
      wx=wv(i)

      if(.not.approx) then
      do j=1,299
        if(xl(j).gt.wx.and.xl(j+1).le.wx) goto 101
      enddo
      stop' j not found'
101   deltrad=(trad(j)-trad(j+1))/(xl(j)-xl(j+1))
      tradi=trad(j)+deltrad*(wx-xl(j))
      endif
      fv(i)=hlam(tradi,wx)
      enddo
c
C     open (3,file='vaul.out')
C     do i=1,nanu
C       write(3,*)wv(i),fv(i)
C     enddo
C     close (3)
C     write(*,*)' interpol. H_l on vaul.out'
C
C*******************************************************************
C
C --- Integration des interpolierten Flusses
      XIN=0.0
      DO 8 I=2,NANU
         IM=I-1
         XIN=(FV(I)*SV(I)+FV(IM)*SV(IM))/2.+XIN
8     CONTINUE
      XIN=XIN*4.*H
C --- Berechnung weiterer Groessen
      V=-2.5*ALOG10(XIN)
c old calib.
      XLR=5.914-0.2*(XMV-V)
c      XLR=5.923-0.2*(XMV-V)
c      instead 3.66e-9 we used new calibration
c      with 3.46e-9, hopefully the sign is correct
c rechanged July 2010, because of comparison with Martins05
c Might be a circular conclusion, needs to be further checked 
      R=10.**XLR
      xll=tef/5770.
      xll=xll*xll*xll*xll
      xll=xll*r*r
      xll=alog10(xll)
c  BC = M_bol  - M_v
c                      M_bol/sun=4.75, T_eff/sun=5770 adopted
c
      bc=4.75-2.5*xll-xmv
C --- Ausdruck Ergebnisse
C
      write(*,*)' T_eff',TEF,' M_v',XMV
      write(*,*)' R/R_sun',R,' log L/L_sun',xll
      write(*,*)' M_bol',xmv+bc,' B.C.',BC
      write(*,*)R
      write(*,*)V
      STOP
      END
c____________________________________________________________
C---------------------------------------------------------------------
      function hlam(trad,wl)
c
c  calculates H_lambda in erg/cm**2/s/A from t_rad
c  wl is wavelength in A
c
c  ca=speed of light in A/s
      ca=2.997929e18
      xnue=ca/wl
c const=h*ca/c/c/2 where c is speed of light in cm/s
      const=1.10513e-29
      y=const*xnue*xnue*xnue/wl/wl
      ee=exp(4.79944e-11*xnue/trad)
      hlam=y/(ee-1.)
      return
      end
c---------------------------------------------------------------------
c----------------------------------------------------------------------
      subroutine polint(xa,ya,n,x,y)
c from numerical recipies p. 103
      dimension xa(n),ya(n)
      parameter (nmax=10)
      dimension c(nmax),d(nmax)
      ns=1
      dif=abs(x-xa(1))
      do 11 i=1,n
           dift=abs(x-xa(i))
           if (dift.lt.dif) then
               ns=i
               dif=dift
           endif
           c(i)=ya(i)
           d(i)=ya(i)
11    continue
      y=ya(ns)
      ns=ns-1
      do 13 m=1,n-1
           do 12 i=1,n-m
              ho=xa(i)-x
              hp=xa(i+m)-x
              w=c(i+1)-d(i)
              den=ho-hp
              if(den.eq.0)stop 'failure in polint'
              den=w/den
              d(i)=hp*den
              c(i)=ho*den
12            continue
           if (2*ns.lt.n-m)then
              dy=c(ns+1)
           else
              dy=d(ns)
              ns=ns-1
           endif
           y=y+dy
13    continue
      return
      end
c-----------------------------------------------------------------------
