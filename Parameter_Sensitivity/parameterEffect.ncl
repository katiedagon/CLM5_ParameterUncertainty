;****************************************
; parameterEffect.ncl
;
; Calculate parameter effect for given output
; This script deals with PFT and namelist parameters
;****************************************
undef("parameterEffect")
function parameterEffect(var)
;****************************************
begin
;****************************************
; read in the files
;****************************************

 if var.eq."TF"
  var1 = "QVEGT"
  var2 = "QFLX_EVAP_TOT"
 else if var.eq."WUE"
  var1 = "FPSN"
  var2 = "QVEGT"
 else if var.eq."SMTOT"
  var1 = "SOILLIQ"
  var2 = "SOILICE"
 else
  var0 = var
 end if
 end if
 end if

 ; set parameter names
 param = (/"displar","dleaf","froot_leaf","kmax","krmax","leafcn","medlynslope","psi50","rootprof_beta","z0mr","baseflow_scalar","maximum_leaf_wetted_fraction","interception_fraction"/)

 ; set number of calculations
 iter=dimsizes(param)

 ; set output path
 path = "/glade/scratch/kdagon/archive/"

 ; read in default output, throw out first 15 years (0016-0020 only)
 d_fils = systemfunc("ls "+path+"hydro_ensemble_reruns_default_params/lnd/hist/*{001[6-9],20-}*")
 df = addfiles(d_fils,"r")

 if var.eq."TF".or.var.eq."WUE"
  d_var1 = df[:]->$var1$
  d_var2 = df[:]->$var2$
  d_var2_mask = d_var2
  d_var2_mask = mask(d_var2,d_var2.eq.0,False)
  d_var = d_var1 ; metadata
  d_var = d_var1/d_var2_mask

 else if var.eq."SMTOT"
  d_var1 = df[:]->$var1$
  d_var2 = df[:]->$var2$
  d_var_sum = d_var1 ; metadata
  d_var_sum = d_var1+d_var2
  ;printVarSummary(d_var_sum)
  d_var = dim_sum_n_Wrap(d_var_sum,1)
 else
  d_var = df[:]->$var0$
 end if
 end if

 PE = new(iter,float) ; array to save the PE for each variable
 PE_cor = PE

 ; gauss weights for global mean
 lat = 46 ; number of latitudes at 4x5
 gaus_wt = gaus(lat/2) ; imput is PER hemisphere
 gw = gaus_wt(:,1) ; note that the generated latitudes do not EXACTLY match 4x5 grid
 ;printVarSummary(gw)

 ; get fixed variable output
 landf = df[0]->landfrac
 area = df[0]->area

 ; wgt variable for global mean
 arwt = area
 arwt = area * landf
 arwt = arwt/max(arwt) 

;****************************************
; loop over ensemble; save PE for each param
;****************************************
 do i=1,iter
  y=param(i-1)
  ; min output
  strs = (/path,"hydro_ensemble_reruns_",y,"_min/lnd/hist/*{001[6-9],20-}*"/)   
  str_c = str_concat(strs)
  fils = systemfunc("ls "+str_c)
  f = addfiles(fils,"r")

  if var.eq."TF".or.var.eq."WUE"
   v1 = f[:]->$var1$
   v2 = f[:]->$var2$
   v2_mask = v2
   v2_mask = mask(v2,v2.eq.0,False)
   v_min = v1 ; metadata
   v_min = v1/v2_mask
  else if var.eq."SMTOT"
   v1 = f[:]->$var1$
   v2 = f[:]->$var2$
   v_min_sum = v1 ; metadata
   v_min_sum = v1+v2
   v_min = dim_sum_n_Wrap(v_min_sum,1)
  else
   v_min = f[:]->$var0$                                                                                                                        
  end if
  end if

  if param(i-1).eq."interception_fraction"
   v_max = d_var ; default as max
  else
   ; max output
   strs = (/path,"hydro_ensemble_reruns_",y,"_max/lnd/hist/*{001[6-9],20-}*"/)
   str_c = str_concat(strs)
   fils = systemfunc("ls "+str_c)
   f2 = addfiles(fils,"r")

   if var.eq."TF".or.var.eq."WUE"
    v1 = f2[:]->$var1$
    v2 = f2[:]->$var2$
    v2_mask = v2
    v2_mask = mask(v2,v2.eq.0,False)
    v_max = v1 ; metadata
    v_max = v1/v2_mask
   else if var.eq."SMTOT"
    v1 = f2[:]->$var1$
    v2 = f2[:]->$var2$
    v_max_sum = v1 ; metadata
    v_max_sum = v1+v2
    v_max = dim_sum_n_Wrap(v_max_sum,1)
   else
    v_max = f2[:]->$var0$
   end if
   end if 

  end if

  PE_all = abs(v_max - v_min)
  PE(i-1) = wgt_areaave_Wrap(dim_avg_n_Wrap(PE_all,0),gw,1,0)

  ; corrected global mean
  PE_cor(i-1) = wgt_areaave2(dim_avg_n_Wrap(PE_all,0),arwt,0)
 end do
;****************************************
; return value(s) of interest
;****************************************
 ;return(PE)
 return([/ PE,PE_cor /])
end
