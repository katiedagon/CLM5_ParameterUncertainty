; ***********************************************
; compare_obs_output.ncl
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
 ;filename = "MR_GPP_4x5_2000_2004_amean.nc"
 filename = "MR_GPP_4x5_2000_2004.nc"
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")

 ; since gaus function does not generate the same lats as model grid
 ; use actual latitudes and cosine weighting for global mean
 ;rad = 4.0*atan(1.0)/180.0
 ;lat = f_obs->lat
 ;clat = cos(lat*rad)
 ;print(clat)
 
 ; global mean calculation
 v_obs = f_obs->$var$
 ;print(v_obs&lon)
 printVarSummary(v_obs)
 ; unit conversion, copy metadata
 v_obs_c = v_obs*u 
 copy_VarMeta(v_obs,v_obs_c)
 printVarSummary(v_obs_c)
 ; month to annual
 v_obs_amean = month_to_annual(v_obs_c,1)
 printVarSummary(v_obs_amean)
 ; global mean, annual mean
 ;v_gm_am = dim_avg_n_Wrap(wgt_areaave_Wrap(v_obs_c,clat,1,0),0)
 ;print(v_gm_am)

 ; get model output
 ; set param set number
 ;pset = "002"
 ;pset2 = "006"
 ;pset = "006"
 ;pset2 = "005"
 ;pset = "SVD_001"
 ;pset2 = "002"
 pset = "SVD_002"
 pset2 = "SVD_001"

 ; set model path
 path_model = "/glade/scratch/kdagon/archive/"

 ; set variable(s) of interest
 var_model = "FPSN"

 ; set years
 nyrs = 5

 ; unit conversion (if needed)
 ;u = 365 * 10^-15 ; gC/m2/day to Pg/m2/year
 ;u = 10^6 / (86400 * 12) ; gC/m2/day to umolC/m2/s

 ; get hist files (last 5 years)
 strs = (/path_model,"test_paramset_",pset,"/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs)
 ;print(str_c)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model = f[:]->$var_model$
 ;print(v_model&lon)
 v_model_amean = month_to_annual(v_model,1)
 printVarSummary(v_model_amean)

 ; get hist files from model run2 (last 5 years)
 strs = (/path_model,"test_paramset_",pset2,"/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model2 = f[:]->$var_model$

 ; compare obs / model
 diff = v_model_amean - v_obs_amean
 copy_VarMeta(v_model_amean, diff)
 ; mean across 5 years
 diff_amean = dim_avg_n_Wrap(diff, 0) 
 ; perdiff
 v_obs_amean_mask = mask(v_obs_amean,v_obs_amean.eq.0,False)
 perdiff = 100*diff/v_obs_amean_mask
 copy_VarMeta(diff,perdiff)
 perdiff_amean = dim_avg_n_Wrap(perdiff,0)

 ; plot diff
 ;plot = new(nyrs,graphic)
 plot = new(2,graphic)

 wks = gsn_open_wks("x11","test")
 ;wks = gsn_open_wks("pdf","model-minus-obs_4x5_2000-2004_amean_GPP_002")
 ;wks = gsn_open_wks("pdf","model-minus-obs_4x5_2000-2004_amean_GPP_006")
 ;wks = gsn_open_wks("pdf","model-minus-obs_4x5_2000-2004_amean_GPP_SVD001")

 res = True
 res@gsnDraw = False
 res@gsnFrame = False
 res@cnFillOn = True
 res@cnLinesOn           = False           ; turn off contour lines
 res@cnLineLabelsOn      = False           ; turn off contour line labels
 res@cnFillPalette       = "MPL_BrBG"
 res@cnSpanFillPalette   = True            ; Fully span color palette
 res@lbLabelBarOn        = False           ; Turn off individual cb's
 res@lbBoxLinesOn = False ; no boxes
 res@gsnLeftString = var
 ;res@gsnRightString = diff@units
 res@gsnRightString = "~F33~m~F21~mol m~S~-2~N~s~S~-1"

 ; set the contour bounds
 res@cnLevelSelectionMode = "ManualLevels"
 res@cnMinLevelValF  = -2
 res@cnMaxLevelValF  = 2
 res@cnLevelSpacingF = 0.2
 ;res@cnMinLevelValF  = -100
 ;res@cnMaxLevelValF  = 100
 ;res@cnLevelSpacingF = 5

 ; single plot
 ;plot = gsn_csm_contour_map(wks,diff_amean,res)
 ;plot = gsn_csm_contour_map(wks,perdiff_amean,res)

 ; multiple plots
 ;do i=1,nyrs
 ; y=tostring(i)
 ; title_strs = str_concat((/"Year ",y/))
 ; res@gsnCenterString = title_strs
 ; plot(i-1) = gsn_csm_contour_map(wks,diff(i-1,:,:),res)
 ;end do

 res@gsnCenterString = "Model with 'best' parameters - Observations"
 plot(0) = gsn_csm_contour_map(wks,diff_amean,res)

 ;resP                     = True                ; modify the panel plot
 ;resP@gsnMaximize         = True
 ;resP@gsnPanelMainString = "Model - Observations"
 ;resP@gsnPanelLabelBar    = True                ; add common colorbar
 ;resP@lbBoxLinesOn = False ; no boxes
 ;gsn_panel(wks,plot,(/3,2/),resP) 

 ; compare obs / model default
 strs_def = (/path_model,"hydro_ensemble_LHC_default_params/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs_def)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model_default = f[:]->$var_model$
 v_model_def_amean = month_to_annual(v_model_default,1)
 printVarSummary(v_model_def_amean) 

 diff_def = v_model_def_amean - v_obs_amean
 copy_VarMeta(v_model_def_amean, diff_def)
 ; mean across 5 years
 diff_def_amean = dim_avg_n_Wrap(diff_def, 0)

 res@gsnCenterString = "Model with default parameters - Observations"
 plot(1) = gsn_csm_contour_map(wks,diff_def_amean,res)

 resP                     = True                ; modify the panel plot
 resP@gsnMaximize         = True
 ;resP@gsnPanelMainString = "Model - Observations"
 resP@gsnPanelLabelBar    = True                ; add common colorbar
 resP@lbBoxLinesOn = False ; no boxes
 gsn_panel(wks,plot,(/1,2/),resP)

 ; compare regional seasonal cycles
 ; set point coordinates
 latp = -5
 lonp = 290

 v_obs_clim = clmMonTLL(v_obs_c)
 v_obs_time = v_obs_clim(:,{latp},{lonp})
 v_model_def_clim = clmMonTLL(v_model_default)
 v_model_def_time = v_model_def_clim(:,{latp},{lonp}) 
 v_model_clim = clmMonTLL(v_model)
 v_model_time = v_model_clim(:,{latp},{lonp})
 v_model2_clim = clmMonTLL(v_model2)
 v_model2_time = v_model2_clim(:,{latp},{lonp})

 dims = dimsizes(v_model_def_time)
 data = new((/4,dims(0)/),float)
 ;data = new((/3,dims(0)/),float)
 data(0,:) = v_obs_time
 data(1,:) = v_model_time
 data(2,:) = v_model2_time
 data(3,:) = v_model_def_time
 ;data(2,:) = v_model_def_time

 wks2 = gsn_open_wks("x11","test")
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt")
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt_006") 
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt_006_005")
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt_002_006")
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt_SVD001_002")

 plot2 = new(2, graphic)

 res2 = True
 res2@gsnDraw = False
 res2@gsnFrame = False
 
 ;res2@xyLineColors = (/"red","blue","black"/) ; obs, model test, model default
 res2@xyLineColors = (/"red","blue","green","black"/) ; obs, model test1, model test2, model default
 ;res@xyDashPatterns     = (/0,0,1,0/)
 res2@xyMonoDashPattern = True
 res2@xyDashPattern = 0
 res2@xyMonoLineThickness = True
 res2@xyLineThicknessF = 3.0

 res2@tiYAxisString = "GPP (~F33~m~F21~mol m~S~-2~N~s~S~-1~N~)"
 res2@tiXAxisString = "Months"
 res2@tiMainString = "Amazon" 

 ; legend settings
 res2@pmLegendDisplayMode    = "Never"            ; turn on legend
 res2@pmLegendSide           = "Top"               ; Change location of 
 res2@pmLegendParallelPosF   = 0.8                  ; move units right
 res2@pmLegendOrthogonalPosF = -0.8                ; move units down
 res2@pmLegendWidthF         = 0.15                ; Change width and
 res2@pmLegendHeightF        = 0.15                ; height of legend.
 res2@lgPerimOn              = False               ; turn off box around
 res2@lgLabelFontHeightF     = .02                 ; label font height
 ;res2@xyExplicitLegendLabels = (/"Obs","Model Preds","Model Default"/)         ; create explicit labels
 ;res2@xyExplicitLegendLabels = (/"Obs","Model Preds GM","Model Preds SVD","Model Default"/)

 ;years = ispan(1,5,1)
 ;plot(2) = gsn_csm_xy(wks,years,data,res)
 months = ispan(1,12,1)
 plot2(0) = gsn_csm_xy(wks2,months,data,res2)

 ; second region
 latp = 10
 lonp = 5
 v_obs_time = v_obs_clim(:,{latp},{lonp})
 v_model_def_time = v_model_def_clim(:,{latp},{lonp})
 v_model_time = v_model_clim(:,{latp},{lonp})
 v_model2_time = v_model2_clim(:,{latp},{lonp})
 data(0,:) = v_obs_time                                                                                                                        
 data(1,:) = v_model_time
 data(2,:) = v_model2_time
 data(3,:) = v_model_def_time
 ;data(2,:) = v_model_def_time
 res2@tiMainString = "Sahel"
 res2@tiYAxisString = ""
 ;res2@pmLegendDisplayMode    = "Never"
 ; legend fits better here
 ;res2@pmLegendDisplayMode    = "Always"
 plot2(1) = gsn_csm_xy(wks2,months,data,res2)

 ; third region
 ;latp = 40
 ;lonp = 260
 ;v_obs_time = v_obs_clim(:,{latp},{lonp})
 ;v_model_def_time = v_model_def_clim(:,{latp},{lonp})
 ;v_model_time = v_model_clim(:,{latp},{lonp})
 ;data(0,:) = v_obs_time
 ;data(1,:) = v_model_time
 ;data(2,:) = v_model_def_time
 ;res2@tiMainString = "Central US"
 ;plot2(2) = gsn_csm_xy(wks2,months,data,res2)

 res2P                     = True                ; modify the panel plot
 res2P@gsnMaximize         = True
 ;res2P@gsnPanelMainString = "Model - Observations"
 gsn_panel(wks2,plot2,(/1,2/),res2P) 

end
