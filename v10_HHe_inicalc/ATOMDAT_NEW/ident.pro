pro ident,lmin,lmax
  readcol,'out_NIII',format='F,L,F',lam,ind,gf
  b=where (lam ge lmin and lam le lmax and ind ge 26500000 and ind lt 26600000, count)
  print,count
  lam1=lam(b)
  ind1=ind(b)
  gf1=gf(b)
  c=sort(gf1)
  lam1=lam1(c)
  ind1=ind1(c)
  gf1=gf1(c)
  for i=0,n_elements(gf1)-1 do begin
    print,lam1(i),ind1(i),gf1(i)
  endfor
  return
end  
  
