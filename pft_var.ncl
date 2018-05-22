; Manipulate the params file for CLM5 to change PFT-dependent variables
; K. Dagon, October 2017

; Assumes the params files to modify have already been created by copying the default file
begin

; set number of variables
;v = 10

; template file
;template = addfile("clm5_params.c170913.nc","r")

; read variables to modify
; getfilevarnames could be useful here? but I only need a subset for this experiment
;displar = template->displar
;dleaf = template->dleaf

; repeated param names for min/max
arr=(/"displar","displar","dleaf","dleaf","froot_leaf","froot_leaf","kmax","kmax","krmax","krmax","leafcn","leafcn","medlynslope","medlynslope","psi50","psi50","rootprof_beta","rootprof_beta","z0mr","z0mr"/)

; param values (could be read by text file)
; IMPORTANT: some of these are values, some are multipliers (PFT variation)
val=(/0.4,0.95,0.0016,0.43,0.8,1.2,0.1,1.9,0.1,1.9,0.69,1.44,0.9,1.1,0.9,1.1,0.995,1.005,0.6,1.5/)

; loop through to create params files
; assumes the params files to modify have already been created by copying the default file
do i=1,dimsizes(arr)
 ;y=tostring(i)
 ;y=tostring(14+i)

 ; min/max determination using modulus
 m=i%2
 if m.eq.1 ; odd (min)
  opt="min"
 else ; even (max)
  opt="max"
 end if
 
 strs = (/"hydro_ensemble/hydro_ens_",arr(i-1),"_",opt,".nc"/)
 str_c = str_concat(strs)
 fin = addfile(str_c,"w") 

 ; trying to use the template without a priori creating the params files (didn't work)
 ;fout = addfile(str_c,"c")
 ;fout = template ; copy the variable data from template?

 var = fin->$arr(i-1)$

 ; absolute ranges
 if arr(i-1).eq."displar".or.arr(i-1).eq."dleaf"
  var(1:) = val(i-1)
 ; relative ranges; 2 variants
 else if arr(i-1).eq."kmax".or.arr(i-1).eq."psi50".or.arr(i-1).eq."rootprof_beta"
  var(:,1:) = var(:,1:)*val(i-1)
 else ; other relative ranges
  var(1:) = var(1:)*val(i-1)
 end if
 end if

 fin->$arr(i-1)$=var
 delete(var)
end do

end
