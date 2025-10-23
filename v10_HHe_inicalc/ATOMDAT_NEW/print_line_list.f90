program print_line_list  
!
implicit none

integer*4 :: ntotnl3, nfullrec, nlast, k, i
real*8 :: lmin,lmax

integer*4 :: lidnew (50000)  

real*4 :: angnew (50000), gfnew (50000)  

character*(*), parameter :: fp = '/home/klein/uh101aw/nlte_v10/inicalc/ATOMDAT_NEW/'

print*,'input range (lambda_min, lambda_max)'
read*,lmin,lmax


!
!     ASCII input files
open (20, file = fp//'nl3i_all_ascii')  
open (21, file = fp//'nl3a_all_ascii')  
open (22, file = fp//'nl3g_all_ascii')  
!     Binary output files

open (33, file = fp//'nl3info', status = 'unknown')  
read (33, * ) ntotnl3, nfullrec, nlast  
!print *, ntotnl3, nfullrec, nlast  
do k = 1, nfullrec  
read (20, * ) (lidnew (i), i = 1, 50000)  
read (21, * ) (angnew (i), i = 1, 50000)  
read (22, * ) (gfnew (i), i = 1, 50000)  

!print * , ' record ', k, ' read'  

do i=1,50000
  if (angnew(i).ge.lmin .and. angnew(i).le.lmax) then
  print*,angnew(i),lidnew(i),gfnew(i)
  endif
enddo  

if (angnew(50000).gt. lmax) goto 100 

enddo  
read (20, * ) (lidnew (i), i = 1, nlast)  
read (21, * ) (angnew (i), i = 1, nlast)  
read (22, * ) (gfnew (i), i = 1, nlast)  

!print * , ' last record read'  

do i=1,nlast
  if (angnew(i).ge.lmin .and. angnew(i).le.lmax) then
  print*,angnew(i),lidnew(i),gfnew(i)
  endif
enddo  


100 continue

close (20)  
close (21)  
close (22)  
close (33)  
!print * , 'success'  
end program print_line_list
