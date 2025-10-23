program conv_ascii_add_HHelines  
!
!     converts nl3*_ascii files to binary, adds additional HHe-lines from
!     file nl3_add_HHelines
!     nl3_add_HHelines doesn't need to be ordered
!
implicit none

integer, parameter :: i4b = selected_int_kind(9)
integer, parameter :: sp = kind(1.0)

integer(i4b), parameter :: nbunch = 50000
!integer(i4b), dimension(nbunch) :: lidnew   
!real(sp), dimension(nbunch) :: angnew, gfnew  

integer(i4b), dimension(:), allocatable:: id
real(sp), dimension(:), allocatable :: gf, xlam

integer(i4b), dimension(:), allocatable:: ihhe
real(sp), dimension(:), allocatable :: ghhe, ahhe

integer(i4b) :: j,nhhe,npack, jrec,nlast, jstart, i
integer(i4b) :: ntotnl3, nfullrec, nrest

real(sp) :: dummy

! read additional lines (ascii)
j=0
open (20, file = 'nl3_add_HHelines')  
do
   read(20,*,end=10) dummy
   j=j+1
enddo


10 nhhe=j
print*,nhhe,' additional H/He lines'

allocate(ihhe(nhhe),ghhe(nhhe),ahhe(nhhe))

rewind 20
do j=1,nhhe
  read(20,*) ahhe(j),ghhe(j),ihhe(j)
enddo

close(20)

!
!     ASCII input files
open (20, file = 'nl3i_all_ascii')  
open (21, file = 'nl3a_all_ascii')  
open (22, file = 'nl3g_all_ascii')  

open (23, file = 'nl3info', status = 'unknown')  

read (23, * ) ntotnl3, nfullrec, nrest  
print*, ntotnl3, nfullrec, nrest, 'read'  

allocate(id(ntotnl3+nhhe),gf(ntotnl3+nhhe),xlam(ntotnl3+nhhe))

! read complete line list (ascii)
npack=nbunch
do jrec = 1,nfullrec+1
   if(jrec .eq. nfullrec+1) npack=nrest
   nlast=(jrec-1)*nbunch
   read(20, *)(id(j),j=nlast+1,nlast+npack)
   read(21, *)(xlam(j),j=nlast+1,nlast+npack)
   read(22, *)(gf(j),j=nlast+1,nlast+npack)
enddo
close (20)  
close (21)  
close (22)  
close (23)  

!include additional nlines
do i=1,nhhe

do j=1,ntotnl3
  if (xlam(j).gt.ahhe(i)) exit
enddo  

jstart=j
ntotnl3=ntotnl3+1

do j=ntotnl3-1,jstart,-1
   id(j+1)=id(j)
   xlam(j+1)=xlam(j)
   gf(j+1)=gf(j)
enddo

id(jstart)=ihhe(i)
xlam(jstart)=ahhe(i)
gf(jstart)=ghhe(i)
enddo

nrest=nrest+nhhe
if (nrest.gt.nbunch) stop' nrest > nbunch, modify approach'

!     Binary output files
open (30, file = 'nl3i_all', form = 'unformatted', status = 'unknown')
open (31, file = 'nl3a_all', form = 'unformatted', status = 'unknown')
open (32, file = 'nl3g_all', form = 'unformatted', status = 'unknown')

open (33, file = 'nl3info_all', status = 'unknown')  

! write complete line list (binary, with recl nbunch)

npack=nbunch
do jrec = 1,nfullrec+1
   if(jrec .eq. nfullrec+1) npack=nrest
   nlast=(jrec-1)*nbunch
   write(30)(id(j),j=nlast+1,nlast+npack)
   write(31)(xlam(j),j=nlast+1,nlast+npack)
   write(32)(gf(j),j=nlast+1,nlast+npack)
   print * , ' record ', jrec, ' converted'  
enddo

write (33, * ) ntotnl3, nfullrec, nrest  
print*, ntotnl3, nfullrec, nrest,' written'  

close (30)  
close (31)  
close (32)  
close (33)  

print * , 'success'  
end program
