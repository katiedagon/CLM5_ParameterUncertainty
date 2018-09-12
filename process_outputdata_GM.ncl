; ***********************************************
; process_outputdata_GM.ncl
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set number of simulations
 niter = 100
 ;niter = 2

 ; set output path
 path = "/glade/scratch/kdagon/archive/"

 ; set variable(s) of interest
 var = "FPSN"
 ;var = "TSA"
 ;var = "QFLX_EVAP_TOT"

 ; gauss weights for global mean
 ;lats = 46 ; number of latitudes at 4x5
 ;gaus_wt = gaus(lats/2) ; imput is PER hemisphere
 ;gw = gaus_wt(:,1) ; note that the generated latitudes do not EXACTLY match 4x5 grid
 ;print(gw)
 ; alternate weighting: cos(latitude)
 ;glat = gaus_wt(:,0)
 rad = 4.0*atan(1.0)/180.0
 ;clat = cos(glat*rad)
 ;print(clat)

 ; since gaus function does not generate the same lats as 4x5 grid
 ; use actual latitudes and cosine weighting for global mean
 strs_lat = str_concat((/path,"hydro_ensemble_LHC_0/lnd/hist/hydro_ensemble_LHC_0.clm2.h0.0001-01.nc"/))
 f_lat = addfile(strs_lat,"r")
 lat = f_lat->lat
 clat = cos(lat*rad)
 ;print(clat)
 
 ; establish output array
 out = new((/niter/),float)

 do i=1,niter
  y=tostring(i)
  strs = (/path,"hydro_ensemble_LHC_",y,"/lnd/hist/*{001[6-9],20-}*"/)
  str_c = str_concat(strs)
  fils = systemfunc("ls "+str_c)
  f = addfiles(fils,"r")
  v = f[:]->$var$
  ; global mean, annual mean
  ;v_gm_am = dim_avg_n_Wrap(wgt_areaave_Wrap(v,clat,1,0),0)
  ;v_gm_am = v_gm_am*86400*365
  ;v_gm_am = v_gm_am - 273
  ;out(i-1) = v_gm_am
  ; 5-year IAV
  v_am_sd_gm = wgt_areaave_Wrap(dim_stddev_n_Wrap(month_to_annual(v,1),0),clat,1,0)
  ;v_am_sd_gm = v_am_sd_gm*86400*365
  out(i-1) = v_am_sd_gm
 end do

 print(out)
 
;************************************************
; write output file
;************************************************
 alist = [/out/]
 ;write_table("outputdata_GPP.csv", "w", alist, "%f")
 ;write_table("outputdata_ET_IAV.csv", "w", alist, "%f")
 ;write_table("outputdata_ET.csv", "w", alist, "%f") 
 ;write_table("outputdata_TSA.csv", "w", alist, "%f") 
 write_table("outputdata_GPP_IAV.csv", "w", alist, "%f")

end
