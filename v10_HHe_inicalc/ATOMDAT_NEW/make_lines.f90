program make_lines  
!
!creates file LINES_ADI.DAT with info on depacked components,
!for specified elements
!default: lambda = 900 ... 10000 A, plus resonance lines

!wavelength in air for lambda > 2000 A
  
implicit none

integer*4, parameter :: kmax=2 !number of elements
integer*4, dimension(kmax) :: ielem = (/6,7/) !carbon and oxygen

real*8, parameter :: lmin=900.,lmax=10000.
real*4, parameter :: zero=0.

character*(*), parameter :: fp = '/home/klein/uh101aw/nlte_cmf/inicalc/ATOMDAT_NEW/'
integer*4, parameter :: natom=30, nion=9, nlev=50, nrec=149, nbunch=50000

integer*4 :: ntotnl3, nfullrec, nlast, k, i, j, n, irec, idum, level, lev, &
&            jdum, ibfup, ilow, ios, npack, jrec, nrest, ilinmax, ncount          

integer*4 :: kkk, kk, jj, l, irest, low, up, ncomp

real*8 :: ergion,flu,xl,xkw,xn

integer*4, dimension(nrec) :: kel, iel, ilin
integer*4, dimension(natom) :: jmax

integer*4, dimension(:), allocatable :: id
real*4, dimension(:), allocatable :: xlam, gf  
real*8, dimension(natom,nion,nlev) :: gstat

character*2, dimension(natom) :: name

character*1, dimension(nion) :: arabic
character*4, dimension(nion) :: roman

character*2 :: levno
character*6 :: lablo, labup
character*6, dimension(:), allocatable :: labl, labu
integer*4, dimension(:), allocatable:: nco

name=(/' H', 'He', 'Li', 'Be', ' B', ' C', ' N', ' O', ' F', &
& 'Ne', 'Na', 'Mg', 'Al', 'Si', ' P', ' S', 'Cl', 'Ar', &
& ' K', 'Ca', 'Sc', 'Ti', ' V', 'Cr', 'Mn', 'Fe', 'Co', &
& 'Ni', 'Cu', 'Zn'/)

data arabic /'1','2','3','4','5','6','7','8','9'/
data roman /'I','II','III','IV','V','VI','VII','VIII','IX'/

open(11,file=fp//'generalinfo',status='old',err=100,iostat=ios)
open(12,file=fp//'partit',status='old',err=101,iostat=ios)

! read maximum ionisation stage J_k and U(J_k+1)
! ----------------------------------------------

do i=1,nrec
   read(11,*) irec,kel(i),iel(i),idum,idum,idum,ilin(i)
enddo

ilinmax=0

do kkk=1,kmax
  kk=ielem(kkk)
  do i=1,nrec
   if(kel(i).eq.kk) ilinmax=max0(ilinmax,ilin(i)) 
  enddo                
enddo

read(12,*) (jmax(i),i=1,natom)
read(12,*) (gstat(i,jmax(i)+1,1),i=1,natom)

close(11)
close(12)

if(maxval(jmax)+1.gt.nion) stop ' error in nion'

open(13,file=fp//'atomnl2_meta_v2.0',status='old',err=102,iostat=ios)


! read statistical weights
! ----------------------


do i = 1,nrec
  k = kel(i)
  j = iel(i)
  read(13,*) n,idum,idum,level

  if(n.ne.i) then
    write(*,*) 'error reading atomnl2_meta in subr. atomdat',n,i
    stop
  endif

  do lev = 1,level
      read(13,*) jdum,ibfup,ilow,ergion,gstat(k,j,lev)
!      print*,k,j,lev,gstat(k,j,lev)
  enddo
enddo  

close(13)
!print*
!
open (20, file = fp//'nl3i_all',form='unformatted')  
open (21, file = fp//'nl3a_all',form='unformatted')  
open (22, file = fp//'nl3g_all',form='unformatted')  

open (33, file = fp//'nl3info', status = 'unknown')  
read (33, * ) ntotnl3, nfullrec, nrest
!print*,ntotnl3,nfullrec,nrest


allocate(id(ntotnl3),gf(ntotnl3),xlam(ntotnl3))

! read complete line list
npack=nbunch
do jrec = 1,nfullrec+1
   if(jrec .eq. nfullrec+1) npack=nrest
   nlast=(jrec-1)*nbunch
   read(20)(id(j),j=nlast+1,nlast+npack)
   read(21)(xlam(j),j=nlast+1,nlast+npack)
   read(22)(gf(j),j=nlast+1,nlast+npack)
   print*,' lines up to',xlam(nlast+npack),'A read'
enddo

close (20)  
close (21)  
close (22)  
close (33)  

!for tests
!do l=1,ntotnl3
!  print*,l,id(l),xlam(l),gf(l)
!enddo
!stop
allocate(labl(ilinmax),labu(ilinmax),nco(ilinmax))
print*
open(1, file='LINES_ADI.dat')
!JO note: fmt is required, since otherwise not starting at first column
write(1,fmt="(':T')")
write(1,fmt="(':T Lines.dat: Version compatible to FASTWIND')")
write(1,fmt="(':T to be used in cmf_all.f90 for depacking only')")
write(1,fmt="(':T')")
write(1,fmt="(':T 3rd column: component #')")
write(1,fmt="(':T 4th column: wavelengths in air for lambda > 2000, else in vacuum')")
write(1,fmt="(':T 5th column: f_ij (depacked) = gi f_ij/sum(gi), where sum(gi) statistical weight')")
write(1,fmt="(':T             of packed lower level (last column)')")
write(1,fmt="(':T')")
write(1,fmt="(':T')")
write(1,fmt="(':T *******************************************************************')")
write(1,fmt="(':T elements: ',8(A2,1x))") (name(ielem(i)),i=1,kmax)
!JO: in case, change '8' to larger values
write(1,fmt="(':T *******************************************************************')")
write(1,fmt="(':T')")
write(1,fmt="(':V2')")

do kkk=1,kmax
  kk=ielem(kkk)
  do jj=1,jmax(kk)
    print*,' element ',name(kk), ' ion = ',jj
    write(1,fmt="(':T *******************************************************************')")
    write(1,fmt="(':T depacked ion: ',A)") trim(name(kk))//roman(jj)
    write(1,fmt="(':T *******************************************************************')")
    write(1,fmt="(':T')")
    write(1,fmt="('CL  TY  RBB  2  7 RBB')")
    ncount=0
    nco=0
    labl=''
    labu=''
    do l=1,ntotnl3
       ncomp=0
       k = id(l)/1000000
       if(k.ne.kk) cycle
       irest = id(l) - k*1000000
       j = irest/100000
       if(j.ne.jj) cycle
       irest = irest - j*100000
       low = irest/100
       up  = irest-low*100
       if(up.eq.0) cycle
       if(xlam(l).lt.lmin .and. low.ne.1) cycle !including resonance lines
       if(xlam(l).gt.lmax) exit
       if (low.le.9) then
         levno=achar(low+48)
       else
         i=low/10
         idum=low-i*10
         levno=achar(i+48)//achar(idum+48)
       endif  
       lablo=trim(name(kk))//trim(arabic(jj))//'_'//trim(levno)
       if (up.le.9) then
         levno=achar(up+48)
       else
         i=up/10
         idum=up-i*10
         levno=achar(i+48)//achar(idum+48)
       endif 
       labup=trim(name(kk))//trim(arabic(jj))//'_'//trim(levno)
!JO: note: in present line list, optical NII lines already at air wavelengths
       if(xlam(l).lt.2000. .or.(k.eq.7 .and. j.eq.2)) then
         xl=xlam(l)
       else
         xkw=1.d4/xlam(l)      
         xn=1.d0+1.d-7*(643.28d0+294981.d0/(146.d0-xkw**2)+2554.d0/(41.d0-xkw**2))
         xl=xlam(l)/xn
       endif
       flu=gf(l)/gstat(k,j,low)
       if(ncount.eq.0) then
         ncount=1
         labl(ncount)=lablo
         labu(ncount)=labup
         nco(ncount)=1
         ncomp=1
       else
         do i=1,ncount
           if(labl(i).eq.lablo .and. labu(i).eq.labup) then
             nco(i)=nco(i)+1
             ncomp=nco(i)
             goto 90
           endif
         enddo    
         ncount=ncount+1
         labl(ncount)=lablo
         labu(ncount)=labup
         nco(ncount)=1
         ncomp=1
90       continue
       endif  
       write(1,50) lablo,labup,ncomp,xl,flu,zero,zero,zero,zero,gstat(k,j,low)
    enddo
    write(1,fmt="('0')")
  enddo
enddo
write(1,fmt="('THEEND')")

close(1)
print* 
print*,'file LINES_ADI.dat successfully created'
print*
stop

100  write(*,*)' error with generalinfo, iostat=',ios
     stop
101  write(*,*)' error with partit, iostat=',ios
     stop
102  write(*,*)' error with atomnl2_meta, iostat=',ios
     stop

50 format(a6,3x,a6,3x,i3,3x,f8.2,3x,g10.4,3x,4(f2.0,3x),f5.0)
     
end program make_lines
