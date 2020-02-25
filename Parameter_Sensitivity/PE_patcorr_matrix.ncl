;****************************************
; PE_patcorr_matrix.ncl.ncl
;
; Calculate pattern correlation of the parameter effect (PE) for specified output
; Using all pairwise matches of parameters
; Save results as a matrix of patcorr values
;****************************************
begin
;****************************************
; specify some settings
;****************************************

 ; set output variable(s) of interest
 vars = (/"FPSN","QFLX_EVAP_TOT","TF","FSH","SOILWATER_10CM","SMTOT","ZWT"/)

 ; specify parameter names
 param=(/"displar","dleaf","froot_leaf","kmax","krmax","leafcn","medlynslope","psi50","rootprof_beta","z0mr","baseflow_scalar","maximum_leaf_wetted_fraction","interception_fraction","csoilc","cv","a","a2","zlnd","zsno","laidl","zdl","sy","fff","dewmx","psno","dmax","dint","kaccum","nmelt","kc25","ko25","cp25","fnr","act25"/)

 ; define "option" tag for each parameter
 ; Opt=0 means use existing min/max runs
 ; Opt=1 means use default for min
 ; Opt=2 means use default for max
 Opt = (/0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,2,1,0,0,1,0,2,0,0,0,0,0,1,0,0,0,0,2,0/)

 ; set number of calculations
 p_iter=dimsizes(param)
 v_iter=dimsizes(vars)

 ; set output path
 path = "/glade/scratch/kdagon/archive/"

;****************************************
; loop over variables
;**************************************** 
 do v=1,v_iter
  var=vars(v-1) 

 ; special output types
 if var.eq."TF"
  var1 = "QVEGT"
  var2 = "QFLX_EVAP_TOT"
 end if

 if var.eq."SMTOT"
  var1 = "SOILLIQ"
  var2 = "SOILICE"
 end if

 ; read in default output, throw out first 15 years (0016-0020 only)
 d_fils = systemfunc("ls "+path+"hydro_ensemble_reruns_default_params/lnd/hist/*{001[6-9],20-}*") 
 df = addfiles(d_fils,"r")
 if var.eq."TF" then
   v1_1 = df[:]->$var1$
   v1_2 = df[:]->$var2$
   v1_2_mask = mask(v1_2,v1_2.eq.0,False)
   d_var = v1_1 ; metadata
   d_var = v1_1/v1_2_mask
   delete(v1_1)
   delete(v1_2)
 else if var.eq."SMTOT" then
   v1_1 = df[:]->$var1$
   v1_2 = df[:]->$var2$
   v1_sum = v1_1 ; metadata
   v1_sum = v1_1 + v1_2
   d_var = dim_sum_n_Wrap(v1_sum,1)
   delete(v1_1)
   delete(v1_2)
 else
  d_var = df[:]->$var$
 end if
 end if
 ;printVarSummary(d_var)

 ; gauss weights for global mean
 lat = 46 ; number of latitudes at 4x5
 gaus_wt = gaus(lat/2) ; imput is PER hemisphere
 gw = tofloat(gaus_wt(:,1)) ; note that the generated latitudes do not EXACTLY match 4x5 grid
 ; type float needed for pattern_cor function

 ; get fixed variable output
 landf = df[0]->landfrac
 area = df[0]->area

 ; wgt variable for global mean
 arwt = area
 arwt = area * landf
 arwt = arwt/max(arwt)

 dvar_dims = dimsizes(d_var)
 PE_dims = (/dvar_dims(0),dvar_dims(1),dvar_dims(2),p_iter/)
 PE = new(PE_dims,float) 
 
;****************************************
; loop over ensemble; save PE for each param
;****************************************
 do i=1,p_iter
  y=param(i-1)
   
  if Opt(i-1).eq.1 ; use default as min
   v_min = d_var
  else 
   ; min output
   strs = (/path,"hydro_ensemble_reruns_",y,"_min/lnd/hist/*{001[6-9],20-}*"/)
   str_c = str_concat(strs)
   fils = systemfunc("ls "+str_c)
   f = addfiles(fils,"r")
   if var.eq."TF" then
    v1_1_min = f[:]->$var1$
    v1_2_min = f[:]->$var2$
    v1_2_min_mask = mask(v1_2_min,v1_2_min.eq.0,False)
    v_min = v1_1_min ; metadata
    v_min = v1_1_min/v1_2_min_mask
    delete(v1_1_min)
    delete(v1_2_min)
   else if var.eq."SMTOT" then
    v1_1_min = f[:]->$var1$
    v1_2_min = f[:]->$var2$
    v1_min_sum = v1_1_min ; metadata
    v1_min_sum = v1_1_min + v1_2_min
    v_min = dim_sum_n_Wrap(v1_min_sum,1)
    delete(v1_1_min)
    delete(v1_2_min)
   else
    v_min = f[:]->$var$
   end if
   end if
  end if ; end opt

  if Opt(i-1).eq.2 ; use default as max
   v_max = d_var
  else
   ; max output
   strs = (/path,"hydro_ensemble_reruns_",y,"_max/lnd/hist/*{001[6-9],20-}*"/)
   str_c = str_concat(strs)
   fils = systemfunc("ls "+str_c)
   f2 = addfiles(fils,"r")
   if var.eq."TF" then
    v1_1_max = f2[:]->$var1$
    v1_2_max = f2[:]->$var2$
    v1_2_max_mask = mask(v1_2_max,v1_2_max.eq.0,False)
    v_max = v1_1_max ; metadata
    v_max = v1_1_max/v1_2_max_mask
    delete(v1_1_max)
    delete(v1_2_max)
   else if var.eq."SMTOT" then
    v1_1_max = f2[:]->$var1$
    v1_2_max = f2[:]->$var2$
    v1_max_sum = v1_1_max ; metadata
    v1_max_sum = v1_1_max + v1_2_max
    v_max = dim_sum_n_Wrap(v1_max_sum,1) 
    delete(v1_1_max)
    delete(v1_2_max)
   else
    v_max = f2[:]->$var$
   end if
   end if
  end if ; end opt

  PE(:,:,:,i-1) = v_min ; metadata
  PE(:,:,:,i-1) = abs(v_max - v_min) ; save PE for each param
 end do ; end param loop

;****************************************
; create the matrix of patcors
;****************************************
 patcor = new((/p_iter,p_iter/), float)
 patcor_cor = patcor

 do i=1,p_iter
  do j=1,p_iter
   pc = pattern_cor(PE(:,:,:,i-1), PE(:,:,:,j-1), gw, 0) ; gw, centered
   pc_cor = pattern_cor(PE(:,:,:,i-1), PE(:,:,:,j-1), arwt, 0) ; arwt, centered
   ; absolute value the time mean; only care about magnitude of correlation
   patcor(i-1,j-1) = abs(dim_avg_n_Wrap(pc,0))
   patcor_cor(i-1,j-1) = abs(dim_avg_n_Wrap(pc_cor,0))
  end do
 end do

;****************************************
; write out patcor for each variable
;****************************************
 fname = str_concat((/var,".txt"/))
 opt = True
 opt@fout = fname
 write_matrix(patcor,"34f10.4",opt)

;****************************************
; keep track of sum of patcors
;****************************************
 if v.eq.1 then
  patcor_sum = patcor ; start with the first variable
  patcor_cor_sum = patcor_cor
 else 
  patcor_sum = patcor_sum + patcor ; incrementally sum patcor matrices
  patcor_cor_sum = patcor_cor_sum + patcor_cor
 end if

 end do ; variable loop

;****************************************
; write out sum as a matrix
;****************************************
 opt@fout = "sum.txt"
 write_matrix(patcor_sum,"34f10.4",opt)

 opt@fout = "sum_cor.txt"
 write_matrix(patcor_cor_sum,"34f10.4",opt)

end
