program test
  integer,dimension(40,300) :: arr
  do i=1,300
    arr(:,i)=5
  enddo

  do k=1,40
    print*,sum(arr(k,5:9))/5
  enddo
  end  
