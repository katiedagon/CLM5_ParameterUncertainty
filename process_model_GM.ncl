; ***********************************************
; process_model_GM.ncl
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set param set number
 pset = "001"

 ; set model path
 path = "/glade/scratch/kdagon/archive/"

 ; set variable(s) of interest
 var = "FPSN"

 ; set years
 nyrs = 5

 ; unit conversion (if needed)
 ;u = 365 * 10^-15 ; gC/m2/day to Pg/m2/year
 ;u = 10^6 / (86400 * 12) ; gC/m2/day to umolC/m2/s

 ; get hist files (last 5 years)
 strs = (/path,"test_paramset_",pset,"/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs)
 print(str_c)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v = f[:]->$var$

 ; since gaus function does not generate the same lats as model grid
 ; use actual latitudes and cosine weighting for global mean
 strs_lat = str_concat((/path,"hydro_ensemble_LHC_1/lnd/hist/hydro_ensemble_LHC_1.clm2.h0.0016-01.nc"/))
 print(strs_lat)
 f_lat = addfile(strs_lat,"r")
 lat = f_lat->lat
 rad = 4.0*atan(1.0)/180.0
 clat = cos(lat*rad)
 
 ; global mean calculation
 ; unit conversion, copy metadata
 ;v_c = v*u 
 ;copy_VarMeta(v,v_c)
 ; global mean, annual mean
 v_gm_am = dim_avg_n_Wrap(wgt_areaave_Wrap(v,clat,1,0),0)
 
 print(v_gm_am)

end
