; ***********************************************
; process_outputdata_GM.ncl
; 
; Process CLM PPE for global means
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set number of simulations
 niter = 100
 ;niter = 10

 ; set output path
 path = "/glade/scratch/kdagon/archive/"

 ; set variable(s) of interest
 ;var = "FPSN"
 ;var = "TSA"
 ;var = "QFLX_EVAP_TOT"
 ;var = "EFLX_LH_TOT"
 var = "FSH"

 ; unit conversion (if needed)
 u = 1
 ;u = (12 * (60 * 60 * 24 * 365) / 10^6) * 10^-15 ; umolC/m2/s to PgC/m2/yr
 ;u = (12 * (60 * 60 * 24 * 365) / 10^6)  ; umol/m2/s to gC/m2/yr

 ; gauss weights for global mean
 ;lats = 46 ; number of latitudes at 4x5
 ;gaus_wt = gaus(lats/2) ; imput is PER hemisphere
 ;gw = gaus_wt(:,1) ; note that the generated latitudes do not EXACTLY match 4x5 grid
 ;print(gw)
 ; alternate weighting: cos(latitude)
 ;glat = gaus_wt(:,0)
 ;rad = 4.0*atan(1.0)/180.0
 ;clat = cos(glat*rad)
 ;print(clat)

 ; sample file for fixed variables output
 strs_lat = str_concat((/path,"hydro_ensemble_LHC_1/lnd/hist/hydro_ensemble_LHC_1.clm2.h0.0016-01.nc"/))
 f_lat = addfile(strs_lat,"r")

 ; since gaus function does not generate the same lats as 4x5 grid
 ; use actual latitudes and cosine weighting for global mean
 ; but note this method does not account for landfrac (makes a difference, esp. with 4x5!)
 ;lat = f_lat->lat
 ;clat = cos(lat*rad)
 ;printMinMax(clat,1)

 ; get landfrac & area
 landf = f_lat->landfrac
 ;printVarSummary(landf)
 area = f_lat->area

 ; wgt variable
 arwt = area
 arwt = area * landf
 arwt = arwt/max(arwt)
 ;printVarSummary(arwt) 
 ;printMinMax(arwt,1)

 ; get obs datamask
 ;f_obs = addfile("../obs/MR_GPP_4x5_2000_2004.nc","r")
 ;f_obs = addfile("../obs/MR_LHF_4x5_2000_2004.nc","r")
 f_obs = addfile("../obs/GBAF_SH_4x5_2000_2004.nc","r")
 datamask_obs = f_obs->datamask

 ; establish output array
 out = new((/niter/),float)
 out!0 = "ens"
 ;out2 = out

 do i=1,niter
  y=tostring(i)
  print("Ensemble member "+y)
  strs = (/path,"hydro_ensemble_LHC_",y,"/lnd/hist/*{001[6-9],20-}*"/)
  ;strs = (/path,"future_climate_LHC_",y,"/lnd/hist/*{001[6-9],20-}*"/)
  str_c = str_concat(strs)
  fils = systemfunc("ls "+str_c)
  f = addfiles(fils,"r")
  v1 = f[:]->$var$
  ; second ensemble for difference
  strs = (/path,"future_climate_LHC_",y,"/lnd/hist/*{001[6-9],20-}*"/)
  str_c = str_concat(strs)
  fils = systemfunc("ls "+str_c)
  f = addfiles(fils,"r")
  v2 = f[:]->$var$
  o = v2 - v1 ; future - present
  ;o = v1 ; present only
  copy_VarMeta(v1,o)
  ; mask model output to match obs
  o_m = mask(o,datamask_obs.eq.0,False)
  copy_VarMeta(o,o_m)
  ; unit conversion
  o_c = o_m*u
  copy_VarMeta(o_m,o_c)
  ; global, annual mean
  ;v_gm_am_alt = dim_avg_n_Wrap(wgt_areaave_Wrap(v,clat,1,0),0)
  v_gm_am = dim_avg_n_Wrap(wgt_areaave2(o_c,arwt,0),0)
  print("Global mean="+v_gm_am)
  out(i-1) = v_gm_am
  ;out2(i-1) = v_gm_am_alt
  ; 5-year IAV
  ;v_am_sd_gm = wgt_areaave_Wrap(dim_stddev_n_Wrap(month_to_annual(v,1),0),clat,1,0)
  ;v_am_sd_gm = v_am_sd_gm*86400*365
  ;out(i-1) = v_am_sd_gm
 end do

 ;print(out)
 ;out@units = "g/m^2/yr"
 out@units = "W/m^2"
 printVarSummary(out)
;************************************************
; write output file
;************************************************
 ;alist = [/out/]
 ;write_table("outputdata_GPP.csv", "w", alist, "%f")
 ;write_table("outputdata_ET_IAV.csv", "w", alist, "%f")
 ;write_table("outputdata_ET.csv", "w", alist, "%f") 
 ;write_table("outputdata_TSA.csv", "w", alist, "%f") 
 ;write_table("outputdata_GPP_IAV.csv", "w", alist, "%f")

 ; writing as nc (preferred)
 ;ncdf = addfile("outputdata_GPP_GM_100.nc", "c")
 ;ncdf = addfile("outputdata_GPP_GM_100_diff.nc", "c")
 ;ncdf->GPP = out
 ;ncdf = addfile("outputdata_LHF_GM_100_diff.nc", "c")
 ;ncdf->LHF = out 
 ncdf = addfile("outputdata_FSH_GM_100_diff.nc", "c")
 ncdf->FSH = out

end
