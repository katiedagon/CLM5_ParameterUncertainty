; ***********************************************
; process_model_SVD.ncl
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set param set number
 ;pset = "005"

 ; set model path
 path = "/glade/scratch/kdagon/archive/"

 ; set variable(s) of interest
 var = "FPSN"

 ; set years
 nyrs = 5

 ; unit conversion (if needed)
 u = (12 * (3600 * 24 * 365) / 1000000) * 10^-15 ; umol /m2 /s to Pg/m2/year

 ; get hist files (last 5 years)
 ;strs = (/path,"test_paramset_SVD_",pset,"/lnd/hist/*{001[6-9],20-}*"/)
 ;strs = (/path,"hydro_ensemble_LHC_default_params/lnd/hist/*{001[6-9],20-}*"/)
 strs = (/path,"hydro_ensemble_LHC_86/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs)
 ;print(str_c)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v = f[:]->$var$

 ; get gridcell areas
 grid_area = f[0]->area
 grid_area_m = grid_area ; metadata
 grid_area_m = grid_area*10^6 ; km^2 to m^2

 ; get obs datamask
 f_obs = addfile("../obs/MR_GPP_4x5_2000_2004.nc","r")
 ;f_obs = addfile("../obs/MR_LHF_4x5_2000_2004.nc","r")
 datamask_obs = f_obs->datamask

 ; mask model output to match obs
 v_m = mask(v,datamask_obs.eq.0,False)
 copy_VarMeta(v,v_m)
 ; unit conversion
 v_c = v_m*u
 copy_VarMeta(v_m,v_c)
 ; 5-year annual mean
 v_am = dim_avg_n_Wrap(v_c,0)
 ; area-weighted annual mean
 v_am_area = v_am * grid_area_m
 copy_VarMeta(v_am,v_am_area)
 ;printVarSummary(v_am_area)

 ; read LHC ensemble for anomaly calculation
 f_model = addfile("../outputdata/outputdata_GPP_forSVD_100.nc","r")
 ;f_model = addfile("../outputdata/outputdata_LHF_forSVD_100.nc","r")
 O = f_model->O
 ;printVarSummary(O)                                                                                                                         

 ; ensemble mean w/o test member
 O_em = dim_avg_n_Wrap(O,0)
 ;printVarSummary(O_em)

 ; test member is last ensemble member
 ;o_md_dims = dimsizes(O)
 ;O_mod = new((/o_md_dims(0)+1,o_md_dims(1),o_md_dims(2)/),"float")
 ;O_mod(:99,:,:) = O
 ;O_mod(100,:,:) = v_am_area
 ;printVarSummary(O_mod)

 ; ensemble mean w/obs
 ;O_em_alt = dim_avg_n_Wrap(O_mod,0)
 ;printVarSummary(O_em_alt) 

 ; prepare input to SVD
 ; subtract "ensemble" mean from test member
 ; where ensemble mean is LHC ensemble mean
 v_anom = v_am_area ; metadata
 v_anom = v_am_area - O_em ; no test member
 ;v_anom = v_am_area - O_em_alt ; with test member

 ; update units
 v_anom@units = "PgC/yr"
 ;v_anom@units = "W"
 ;printVarSummary(v_anom)

 ; test plot
 wks = gsn_open_wks("x11","test")
 ;wks = gsn_open_wks("pdf", "test_paramset_001_GPP_forSVD_withtest")
 ;wks = gsn_open_wks("pdf", "CLM_default_GPP_forSVD_withoutdefault")

 res = True
 ;res@gsnDraw = False
 ;res@gsnFrame = False
 res@cnFillOn = True
 res@cnLinesOn           = False           ; turn off contour lines
 res@cnLineLabelsOn      = False
 res@cnFillPalette       = "MPL_YlOrRd" 
 res@cnSpanFillPalette   = True            ; Fully span color palette
 ;res@lbLabelBarOn        = False           ; Turn off individual cb's
 res@lbBoxLinesOn = False ; no boxes
 res@gsnLeftString = var
 res@gsnRightString = v_anom@units
 res@tiMainString = "5-year annual mean test paramset, area-weighted, anomaly from LHC100 ensemble mean with test paramset"  
 ;res@tiMainString = "5-year annual mean default params, area-weighted, anomaly from LHC100 ensemble mean without default params"
 res@tiMainFontHeightF   = 0.012

 ; set the contour bounds (FPSN, 001)
 ;res@cnLevelSelectionMode = "ManualLevels"
 ;res@cnMinLevelValF  = -0.5
 ;res@cnMaxLevelValF  = 0.2
 ;res@cnLevelSpacingF = 0.1

 plot = gsn_csm_contour_map(wks,v_anom,res) 

 ; write output file
 ;ncdf = addfile("test_paramset_001_GPP_forSVD.nc", "c")
 ;ncdf = addfile("CLM_default_GPP_forSVD.nc", "c")
 ncdf = addfile("test_paramset_GPP_forSVD.nc", "c")
 ncdf->GPP = v_anom 
 ncdf->datamask = datamask_obs

end
