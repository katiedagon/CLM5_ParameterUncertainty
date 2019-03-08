; ***********************************************
; process_obs_GM.ncl
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set obs path
 path = "~/CLM5hydro_LHC/obs/"

 ; set variable(s) of interest
 var = "GPP"

 ; set years
 nyrs = 5

 ; unit conversion (if needed)
 ;u = 365 * 10^-15 ; gC/m2/day to Pg/m2/year
 u = 10^6 / (86400 * 12) ; gC/m2/day to umolC/m2/s

 ; get obs file
 ;filename = "MR_GPP_0.9x1.25_ALLMONS_climo.nc"
 ;filename = "MR_GPP_0.9x1.25_2000_2004_amean.nc"
 filename = "MR_GPP_4x5_2000_2004_amean.nc"
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")

 ; since gaus function does not generate the same lats as model grid
 ; use actual latitudes and cosine weighting for global mean
 rad = 4.0*atan(1.0)/180.0
 lat = f_obs->lat
 clat = cos(lat*rad)
 ;print(clat)
 
 ; global mean calculation
 v_obs = f_obs->$var$
 ; unit conversion, copy metadata
 v_obs_c = v_obs*u 
 copy_VarMeta(v_obs,v_obs_c)
 ; global mean, annual mean
 v_gm_am = dim_avg_n_Wrap(wgt_areaave_Wrap(v_obs_c,clat,1,0),0)
 
 print(v_gm_am)

end
