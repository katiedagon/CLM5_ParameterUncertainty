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
 ;path = "/glade/p/cesm/lmwg_dev/oleson/ILAMB/ILAMB_all/DATA/gpp/FLUXNET-MTE/"
 ;path = "/gpfs/fs1/p/cesm/lmwg_dev/oleson/ILAMB/ILAMB_all/DATA/gpp/FLUXNET-MTE/"

 ; set variable(s) of interest
 var = "GPP" ; FLUXNET-MTE
 ;var = "gpp" ; ILAMB

 ; set years
 nyrs = 5

 ; Set some fixed dimensions
 start_year = 2000
 end_year = 2004
 nyrs = end_year - start_year + 1
 start_year_ind = start_year - 1982
 start_month = start_year_ind*12
 print(start_month)
 end_month = start_month + nyrs*12 - 1
 print(end_month)

 ; unit conversion (if needed)
 ;u = 365 * 10^-15 ; gC/m2/day to Pg/m2/year
 ;u = 10^6 / (86400 * 12) ; gC/m2/day to umolC/m2/s
 ;u = (10^3 * 10^6) / 12 ; kg/m2/s to umolC/m2/s
 u = 365 ; gC/m2/day to gC/m2/year
 ;u = 365*86400*1000 ; kgC/m2/s to gC/m2/year

 ; get obs file
 ;filename = "MR_GPP_0.9x1.25_ALLMONS_climo.nc"
 ;filename = "MR_GPP_0.9x1.25_2000_2004_amean.nc"
 filename = "MR_GPP_4x5_2000_2004.nc" ; FLUXNET-MTE
 ;filename = "gpp_0.5x0.5.nc" ; ILAMB - note not regridded
 
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")

 ; since gaus function does not generate the same lats as model grid
 ; use actual latitudes and cosine weighting for global mean
 rad = 4.0*atan(1.0)/180.0
 lat = f_obs->lat
 clat = cos(lat*rad)
 ;print(clat)
 
 ; sample file for fixed variables output
 fix_path = "/glade/scratch/kdagon/archive/"
 strs_lat = str_concat((/fix_path,"hydro_ensemble_LHC_1/lnd/hist/hydro_ensemble_LHC_1.clm2.h0.0016-01.nc"/))
 f_lat = addfile(strs_lat,"r")

 ; get landfrac & area
 landf = f_lat->landfrac
 ;printVarSummary(landf)
 area = f_lat->area
 ;area = f_obs->area ; this area is very different - WHY?

 ; wgt variable
 arwt = area
 arwt = area * landf
 arwt = arwt/max(arwt)
 ;printVarSummary(arwt) 
 ;printMinMax(arwt,1) 
 ; mask the arwt to match obs
 datamask_obs = f_obs->datamask
 arwt_m = mask(arwt,datamask_obs.eq.0,False)
 copy_VarMeta(arwt,arwt_m) 

 ; global mean calculation
 v_obs = f_obs->$var$
 ; unit conversion, copy metadata
 v_obs_c = v_obs*u 
 copy_VarMeta(v_obs,v_obs_c)
 ; global mean, annual mean
 ;v_gm_am_alt = dim_avg_n_Wrap(wgt_areaave_Wrap(v_obs_c,clat,1,0),0)
 v_gm_am = dim_avg_n_Wrap(wgt_areaave2(v_obs_c,arwt_m,0),0) 
 ; to subset years
 ;v_gm_am = dim_avg_n_Wrap(wgt_areaave2(v_obs_c(start_month:end_month,:,:),arwt,0),0) 
 ;v_gm_am_alt = dim_avg_n_Wrap(wgt_areaave_Wrap(v_obs_c(start_month:end_month,:,:),clat,1,0),0) 

 print("Global mean="+v_gm_am)
 ;print("Alt Global mean="+v_gm_am_alt)

 ; quick plot (5-year annual mean)
 v_obs_am = dim_avg_n_Wrap(v_obs_c,0)

 wks = gsn_open_wks("x11","test")

 res=True
 res@cnFillOn = True
 res@cnLinesOn           = False
 res@cnLineLabelsOn      = False
 res@cnFillPalette       = "MPL_YlOrRd"
 res@cnSpanFillPalette   = True

 ;plot = gsn_csm_contour_map(wks,landf,res)
 ;plot = gsn_csm_contour_map(wks,area,res)
 ;plot = gsn_csm_contour_map(wks,arwt,res)
 plot = gsn_csm_contour_map(wks,arwt_m,res)
 plot = gsn_csm_contour_map(wks,v_obs_am,res) 

 

end
