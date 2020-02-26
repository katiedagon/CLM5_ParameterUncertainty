; ***********************************************
; compare_obs_output_likelihood.ncl
;
; Compare obs with model output using likelihood function
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
 u = 365 * 10^-15 ; gC/m2/day to Pg/m2/year
 ;u = 10^6 / (86400 * 12) ; gC/m2/day to umolC/m2/s

 ; get obs file
 filename = "MR_GPP_4x5_2000_2004.nc"
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")

 ; get gridcell areas
 grid_area = f_obs->area
 grid_area_m = grid_area ; metadata
 grid_area_m = grid_area*10^6 ; km^2 to m^2

 ; get obs datamask
 datamask_obs = f_obs->datamask

 ; process obs
 v_obs = f_obs->$var$
 ; unit conversion, copy metadata
 v_obs_c = v_obs*u 
 copy_VarMeta(v_obs,v_obs_c)
 ; month to annual
 v_obs_amean = month_to_annual(v_obs_c,1)

 ; calculate sd of obs (all years)
 filename_sd = "MR_GPP_4x5.nc"
 filepath_sd = str_concat((/path,filename_sd/))
 f_sd = addfile(filepath_sd,"r")

 ; process sd
 v_sd = f_sd->$var$
 ; unit conversion, copy metadata
 v_sd_c = v_sd*u
 copy_VarMeta(v_sd,v_sd_c)
 ; month to annual
 v_sd_amean = month_to_annual(v_sd_c,1)
 ; sd over years
 v_sd_amean_std = dim_stddev_n_Wrap(v_sd_amean,0)
 ;printVarSummary(v_sd_amean_std)

 ; get model output
 pset = "test_paramset_SVD_005"
 pset2 = "test_paramset_SVD_006"

 ; set model path
 path_model = "/glade/scratch/kdagon/archive/"

 ; set variable(s) of interest
 var_model = "FPSN"

 ; set years
 nyrs = 5

 ; unit conversion (if needed)
 u = (12 * (3600 * 24 * 365) / 1000000) * 10^-15 ; umolC/m2/s to Pg/m2/year

 ; get model hist files (last 5 years)
 strs = (/path_model,pset,"/lnd/hist/*{001[6-9],20-}*"/) 
 str_c = str_concat(strs)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model = f[:]->$var_model$
 ; mask model output to match obs
 v_model_m = mask(v_model,datamask_obs.eq.0,False)
 copy_VarMeta(v_model,v_model_m)
 v_model_c = v_model_m*u
 copy_VarMeta(v_model,v_model_c)
 v_model_amean = month_to_annual(v_model_c,1)

 ; get hist files from model run2 (last 5 years)
 strs = (/path_model,pset2,"/lnd/hist/*{001[6-9],20-}*"/) 
 str_c = str_concat(strs)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model2 = f[:]->$var_model$
 v_model2_m = mask(v_model2,datamask_obs.eq.0,False)
 copy_VarMeta(v_model2,v_model2_m)
 v_model2_c = v_model2_m*u
 copy_VarMeta(v_model2,v_model2_c)
 v_model2_amean = month_to_annual(v_model2_c,1)

 ; calculate likelihood (model 1)
 diff = v_model_amean - v_obs_amean
 copy_VarMeta(v_model_amean, diff)
 ; mean across 5 years
 L1 = (dim_avg_n_Wrap(diff,0)/v_sd_amean_std) 
 copy_VarMeta(v_sd_amean_std,L1)
 ; model run2
 diff2 = v_model2_amean - v_obs_amean
 copy_VarMeta(v_model2_amean, diff2)
 L2 = (dim_avg_n_Wrap(diff2,0)/v_sd_amean_std)
 copy_VarMeta(v_sd_amean_std,L2)

 ; plot diff
 ;plot = new(2,graphic)
 plot = new(3,graphic)

 wks = gsn_open_wks("x11","test")
 ;wks = gsn_open_wks("pdf","model-minus-obs_4x5_2000-2004_amean_GPP_002")

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
 ;res@gsnRightString = "~F33~m~F21~mol m~S~-2~N~s~S~-1"
 ;res@gsnRightString = "PgC m~S~-2~N~ yr~S~-1~N~"
 ;res@gsnRightString = "%"
 res@gsnRightString = "Normalized Error"

 ; set the contour bounds
 res@cnLevelSelectionMode = "ManualLevels"
 ;res@cnMinLevelValF  = -2
 ;res@cnMaxLevelValF  = 2
 ;res@cnLevelSpacingF = 0.2
 res@cnMinLevelValF  = -100
 res@cnMaxLevelValF  = 100
 res@cnLevelSpacingF = 10

 label_strs = str_concat((/pset, " relative to Observations"/))
 res@gsnCenterString = label_strs
 plot(0) = gsn_csm_contour_map(wks,L1,res)

 label_strs2 = str_concat((/pset2, " relative to Observations"/))
 res@gsnCenterString = label_strs2
 plot(1) = gsn_csm_contour_map(wks,L2,res)

 ; compare obs / model default
 strs_def = (/path_model,"hydro_ensemble_LHC_default_params/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs_def)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model_default = f[:]->$var_model$
 v_model_default_m = mask(v_model_default,datamask_obs.eq.0,False)
 copy_VarMeta(v_model_default,v_model_default_m)
 v_model_default_c = v_model_default_m*u
 copy_VarMeta(v_model_default,v_model_default_c)
 v_model_def_amean = month_to_annual(v_model_default_c,1)
 diff_def = v_model_def_amean - v_obs_amean
 copy_VarMeta(v_model_def_amean, diff_def)
 Ld = (dim_avg_n_Wrap(diff_def,0)/v_sd_amean_std)
 copy_VarMeta(v_sd_amean_std,Ld)

 res@gsnCenterString = "Model with default parameters relative to Observations"
 plot(2) = gsn_csm_contour_map(wks,Ld,res)

 resP                     = True                ; modify the panel plot
 resP@gsnMaximize         = True
 ;resP@gsnPanelMainString = "Model - Observations"
 resP@gsnPanelLabelBar    = True                ; add common colorbar
 resP@lbBoxLinesOn = False ; no boxes
 gsn_panel(wks,plot,(/2,2/),resP)
 ;gsn_panel(wks,plot,(/1,2/),resP)

 ; compare regional seasonal cycles
 ; set point coordinates
 latp = -5
 lonp = 290

 v_obs_clim = clmMonTLL(v_obs_c)
 v_obs_time = v_obs_clim(:,{latp},{lonp})
 v_model_def_clim = clmMonTLL(v_model_default_c)
 v_model_def_time = v_model_def_clim(:,{latp},{lonp}) 
 v_model_clim = clmMonTLL(v_model_c)
 v_model_time = v_model_clim(:,{latp},{lonp})
 v_model2_clim = clmMonTLL(v_model2_c)
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

 ;res2@tiYAxisString = "GPP (~F33~m~F21~mol m~S~-2~N~s~S~-1~N~)"
 res2@tiYAxisString = "GPP (PgC m~S~-2~N~ yr~S~-1~N~)"
 res2@tiXAxisString = "Months"
 res2@tiMainString = "Amazon" 

 ; legend settings
 ;res2@pmLegendDisplayMode    = "Never"            ; turn on legend
 res2@pmLegendDisplayMode    = "Always"
 res2@pmLegendSide           = "Top"               ; Change location of 
 res2@pmLegendParallelPosF   = 0.8                  ; move units right
 res2@pmLegendOrthogonalPosF = -0.8                ; move units down
 res2@pmLegendWidthF         = 0.15                ; Change width and
 res2@pmLegendHeightF        = 0.15                ; height of legend.
 res2@lgPerimOn              = False               ; turn off box around
 res2@lgLabelFontHeightF     = .02                 ; label font height
 ;res2@xyExplicitLegendLabels = (/"Obs","Model Preds","Model Default"/)         ; create explicit labels
 res2@xyExplicitLegendLabels = (/"Obs",pset,pset2,"Model Default"/)

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
 res2@pmLegendDisplayMode    = "Never"
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
