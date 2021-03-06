; ***********************************************
; process_obs_SVD.ncl
;
; Process obs for SVD
; ***********************************************
begin
;************************************************
; read in obs data
;************************************************
 ; set obs path
 path = "~/CLM5_ParameterUncertainty/obs/"

 ; set variable(s) of interest
 ;var = "GPP"
 var = "LHF"

 ; set years
 nyrs = 5
 ;nyrs = 27

 ; unit conversion (if needed)
 ;u = 365 * 10^-15 ; gC/m2/day to Pg/m2/year
 ;u = 10^-15 ; gC/m2/day to Pg/m2/day
 ;u = 1
 ;u = 86400 * 365 * 10^-12 ; kgC/m2/s to Pg/m2/year
 u = 10^-12 ; W to TW

 ; get obs file
 ; FLUXNET
 ;filename = "MR_GPP_4x5_2000_2004.nc"
 filename = "MR_LHF_4x5_2000_2004.nc"
 ;filename = "MR_GPP_4x5.nc"
 ;filename = "MR_LHF_4x5.nc"
 ; WECANN
 ;filename = "WECANN_GPP_LHF_4x5_2007_2015.nc"
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")

 ; get gridcell areas
 ; using model output for consistency
 f_md = addfile("/glade/scratch/kdagon/archive/hydro_ensemble_LHC_1/lnd/hist/hydro_ensemble_LHC_1.clm2.h0.0016-01.nc","r")
 grid_area = f_md->area
 ;grid_area = f_obs->area 
 grid_area_m = grid_area ; metadata
 grid_area_m = grid_area*10^6 ; km^2 to m^2

 ; get obs datamask
 datamask_obs = f_obs->datamask
 ;datamask_obs = f_obs->datamaskG
 ;datamask_obs = f_obs->datamaskL

 ; read in obs
 v_obs = f_obs->$var$
 ; unit conversion
 v_obs_c = v_obs*u
 ; copy metadata
 copy_VarMeta(v_obs,v_obs_c)
 ;printVarSummary(v_obs_c)
 ; WECANN - subset first 5 years
 ;v_obs_c_s = v_obs_c(:59,:,:)
 ;printVarSummary(v_obs_c_s)
 ;v_obs_am = dim_avg_n_Wrap(v_obs_c_s,0)
 ; 5-year annual mean
 v_obs_am = dim_avg_n_Wrap(v_obs_c,0) 
 ; test without area weighting
 ;v_obs_am_area = v_obs_am
 ; month to annual (to preserve all years)
 ;v_obs_am = month_to_annual(v_obs_c,1)
 ;printVarSummary(v_obs_am)
 ; area-weighted annual mean, copy metadata
 v_obs_am_area = v_obs_am * grid_area_m
 copy_VarMeta(v_obs_am,v_obs_am_area)
 ; area-weighted, preserve all years
 ;v_obs_am_area = v_obs_am ; metadata / placeholder
 ;do i=1,nyrs
 ; v_obs_am_area(i-1,:,:) = v_obs_am(i-1,:,:) * grid_area_m
 ;end do
 ;printVarSummary(v_obs_am_area)

;************************************************
; read in LHC ensemble output
;************************************************
 ;f_model = addfile("../outputdata/outputdata_GPP_forSVD_100.nc","r")
 ;f_model = addfile("../outputdata/outputdata_GPP_forSVD_100_WECANNmask.nc","r")
 f_model = addfile("../outputdata/outputdata_LHF_forSVD_100.nc","r")
 ;f_model = addfile("../outputdata/outputdata_LHF_forSVD_100_WECANNmask.nc","r")
 O = f_model->O
 ;printVarSummary(O)
 
 ; ensemble mean w/o obs
 O_em = dim_avg_n_Wrap(O,0)
 ;printVarSummary(O_em)
 ; unit conversion (temp)
 ;u_m = 1/365 ; Pg/yr to Pg/day
 ;O_em_c = O_em*u_m
 ;copy_VarMeta(O_em,O_em_c)

 ; obs is last ensemble member
 ;o_md_dims = dimsizes(O)
 ;O_mod = new((/o_md_dims(0)+1,o_md_dims(1),o_md_dims(2)/),"float")
 ;O_mod(:99,:,:) = O
 ;O_mod(100,:,:) = v_obs_am_area
 ;printVarSummary(O_mod)

 ; ensemble mean w/obs - technically this is not right
 ;O_em_alt = dim_avg_n_Wrap(O_mod,0)
 ;printVarSummary(O_em_alt)

;************************************************
; prepare input to SVD
;************************************************
 ; subtract ensemble mean from obs
 ; where ensemble mean is LHC ensemble mean
 v_obs_anom = v_obs_am_area ; metadata
 v_obs_anom = v_obs_am_area - O_em ; no obs
 ;v_obs_anom = v_obs_am_area - O_em_alt ; with obs
 ; preserve all years
 ;do i=1,nyrs
 ; v_obs_anom(i-1,:,:) = v_obs_am_area(i-1,:,:) - O_em
 ;end do
 ;printVarSummary(v_obs_anom)

 ; test alternate units
 ;v_obs_anom_test = v_obs_anom
 ;v_obs_anom_test = v_obs_anom*0.002739726027
 ;printVarSummary(v_obs_anom_test)

 ; update units
 ;v_obs_anom@units = "PgC/yr"
 ;v_obs_anom@units = "W"
 v_obs_anom@units = "TW"
 ;v_obs_anom@units = "PgC/day"
 ;printVarSummary(v_obs_anom)
 
 ; test plot

 wks = gsn_open_wks("x11","test")
 ;wks = gsn_open_wks("pdf","obs_GPP_4x5_anom_forSVD_withoutobs")
 ;wks = gsn_open_wks("pdf","obs_GPP_4x5_anom_forSVD_withobs")
 ;wks = gsn_open_wks("pdf","obs_LHF_4x5_anom_forSVD_withobs")  
 ;wks = gsn_open_wks("pdf","obs_LHF_4x5_anom_forSVD_withoutobs")

 res = True
 ;res@gsnDraw = False
 ;res@gsnFrame = False
 res@cnFillOn = True
 res@cnLinesOn           = False           ; turn off contour lines
 res@cnLineLabelsOn      = False           ; turn off contour line labels
 res@cnFillPalette       = "MPL_YlOrRd"
 res@cnSpanFillPalette   = True            ; Fully span color palette
 ;res@lbLabelBarOn        = False	   ; Turn off individual cb's
 res@gsnLeftString = var
 ;res@gsnRightString = v_obs_c_area_am@units
 res@gsnRightString = v_obs_anom@units
 res@tiMainString = "5-year annual mean obs, area-weighted, anomaly from LHC100 ensemble mean no obs"
 ;res@tiMainString = "5-year annual mean obs, area-weighted, anomaly from LHC100 ensemble mean with obs"
 ;res@tiMainString = "1982 annual mean obs, area-weighted, anomaly from LHC100 ensemble mean no obs"
 res@tiMainFontHeightF   = 0.015

 ; set the contour bounds (GPP)
 ;res@cnLevelSelectionMode = "ManualLevels"
 ;res@cnMinLevelValF  = -0.5
 ;res@cnMaxLevelValF  = 0.2
 ;res@cnLevelSpacingF = 0.1

 ; single plot
 ; 5-year mean
 plot = gsn_csm_contour_map(wks,v_obs_anom,res)
 ; all years
 ;plot = gsn_csm_contour_map(wks,v_obs_anom(0,:,:),res)

;************************************************
; write output file
;************************************************
 ; writing as nc (preferred)
 ;ncdf = addfile("obs_GPP_4x5_anom_forSVD.nc", "c")
 ;ncdf = addfile("obs_LHF_4x5_anom_forSVD.nc", "c")
 ;ncdf = addfile("obs_GPP_4x5_anom_forSVD_allyrs.nc", "c")
 ;ncdf = addfile("obs_LHF_4x5_anom_forSVD_allyrs.nc", "c")
 ;ncdf = addfile("WECANN_GPP_4x5_anom_forSVD.nc", "c")
 ;ncdf = addfile("WECANN_LHF_4x5_anom_forSVD.nc", "c")
 ncdf = addfile("obs_LHF_4x5_anom_forSVD_TW.nc", "c") ; sanity check LHF units

 ; define time as an unlimited dimension (optional?)
 ;filedimdef(ncdf,"time",-1,True) 
 ;ncdf->GPP = v_obs_anom
 ncdf->LHF = v_obs_anom
 ncdf->datamask = datamask_obs

end
