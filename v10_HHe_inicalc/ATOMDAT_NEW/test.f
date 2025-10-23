      program conv_ascii
c
      integer*4 lidnew(50000)
      real*4 angnew(50000),gfnew(50000)
C
 
      open(30,file = 'nl3i_all',form = 'unformatted', 
     &     status = 'unknown')
      open(31,file = 'nl3a_all',form = 'unformatted', 
     &     status = 'unknown')
      open(32,file = 'nl3g_all',form = 'unformatted', 
     &     status = 'unknown')

      open(33,file = 'nl3info_all', status = 'unknown')

      read(33,*) ntotnl3,nfullrec,nlast
      print*,ntotnl3,nfullrec,nlast
      do k=1,nfullrec

      read(30) (lidnew(i),i=1,50000)
      read(31) (angnew(i),i=1,50000)
      read(32)  (gfnew(i),i=1,50000)
      do i=1,50000
        print*,lidnew(i),angnew(i),gfnew(i)
      enddo  
      print*,' record ',k,' converted'  
      end do
      

      read(30) (lidnew(i),i=1,nlast)
      read(31) (angnew(i),i=1,nlast)
      read(32)  (gfnew(i),i=1,nlast)
      do i=1,nlast
        print*,lidnew(i),angnew(i),gfnew(i)
      enddo  
      print*,' last record converted'  
 
      close(30)
      close(31)
      close(32)
      close(33)
      print*,'success'
      end
