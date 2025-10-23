      program conv_ascii
c
      integer*4 lidnew(50000)
      real*4 angnew(50000),gfnew(50000)
C
 
C     ASCII input files
      open(20,file = 'nl3i_all_ascii')
      open(21,file = 'nl3a_all_ascii')
      open(22,file = 'nl3g_all_ascii')
C     Binary output files
      open(30,file = 'nl3i_all',form = 'unformatted', 
     &     status = 'unknown')
      open(31,file = 'nl3a_all',form = 'unformatted', 
     &     status = 'unknown')
      open(32,file = 'nl3g_all',form = 'unformatted', 
     &     status = 'unknown')

      open(33,file = 'nl3info', status = 'unknown')

      read(33,*) ntotnl3,nfullrec,nlast
      print*,ntotnl3,nfullrec,nlast
      do k=1,nfullrec
      read(20,*) (lidnew(i),i=1,50000)
      read(21,*) (angnew(i),i=1,50000)
      read(22,*)  (gfnew(i),i=1,50000)

      write(30) (lidnew(i),i=1,50000)
      write(31) (angnew(i),i=1,50000)
      write(32)  (gfnew(i),i=1,50000)
      print*,' record ',k,' converted'  
      end do
      
      read(20,*) (lidnew(i),i=1,nlast)
      read(21,*) (angnew(i),i=1,nlast)
      read(22,*)  (gfnew(i),i=1,nlast)

      write(30) (lidnew(i),i=1,nlast)
      write(31) (angnew(i),i=1,nlast)
      write(32)  (gfnew(i),i=1,nlast)
      print*,' last record converted'  
 
      close(20)
      close(21)
      close(22)
      close(30)
      close(31)
      close(32)
      close(33)
      print*,'success'
      end
