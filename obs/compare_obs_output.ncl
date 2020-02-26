; ***********************************************
; compare_obs_output.ncl
;
; Compare obs with output from optimized and default CLM
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set obs path
 path = "~/CLM5hydro_LHC/obs/"

 ; set variable(s) of interest
 ;var = "GPP"
 ;var = "LHF"
 var = "FSH"

 ; set years
 ;nyrs = 5

 ; unit conversion (if needed)
 ;u_obs = 365 * 10^-15 ; gC/m2/day to Pg/m2/year
 ;u_obs = 365 ; gC/m2/day to gC/m2/year
 ;u_obs = 10^6 / (86400 * 12) ; gC/m2/day to umolC/m2/s
 u_obs = 1

 ; get obs file
 ;filename = "MR_"+var+"_4x5_2000_2004.nc"
 filename = "GBAF_SH_4x5_2000_2004.nc"
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")
 datamask_obs = f_obs->datamask
 
 ; since gaus function does not generate the same lats as model grid
 ; use actual latitudes and cosine weighting for global mean
 ;rad = 4.0*atan(1.0)/180.0
 ;lat = f_obs->lat
 ;clat = cos(lat*rad)
 ;print(clat)
 
 v_obs = f_obs->$var$
 ;print(v_obs&lon)
 ;printVarSummary(v_obs)
 ; unit conversion, copy metadata
 v_obs_c = v_obs*u_obs 
 copy_VarMeta(v_obs,v_obs_c)
 ;printVarSummary(v_obs_c)
 ; 5-year  annual mean
 v_obs_amean = dim_avg_n_Wrap(v_obs_c,0)
 ;printVarSummary(v_obs_amean)
 ;printMinMax(v_obs_amean,1)

 ; global mean calculation - update for landfrac/area
 ; month to annual - 5-year maps
 ;v_obs_amean = month_to_annual(v_obs_c,1)
 ;printVarSummary(v_obs_amean)
 ; global mean, annual mean
 ;v_gm_am = dim_avg_n_Wrap(wgt_areaave_Wrap(v_obs_c,clat,1,0),0)
 ;print(v_gm_am)

 ; get model output - test simulations
 ; set param set number
 ;pset = "002"
 ;pset2 = "006"
 
 ; set test case name
 test_case = "test_paramset_SVD_v4"

 ; set PPE number
 ;PPE_mem = "92"

 ; set model output
 ;var_model = "FPSN"
 ;var_model = "EFLX_LH_TOT"
 var_model = "FSH"

 ; model unit conversion
 ;u_model = (12 * (60 * 60 * 24 * 365) / 10^6)  ; umol/m2/s to gC/m2/yr 
 ;u_model = (12 * (60 * 60 * 24) / 10^6) ; umolC/m2/s to gC/m2/day
 u_model = 1

 path_model = "/glade/scratch/kdagon/archive/"
 ;strs = (/path_model,"test_paramset_",pset,"/lnd/hist/*{001[6-9],20-}*"/)
 strs = (/path_model,test_case,"/lnd/hist/*{001[6-9],20-}*"/) 
 ;strs = (/path_model,"hydro_ensemble_LHC_",PPE_mem,"/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs)
 print(str_c)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model = f[:]->$var_model$
 ;print(v_model&lon)
 ;v_model_amean = month_to_annual(v_model,1)
 ;printVarSummary(v_model_amean)

 ; mask model to match obs
 v_model_m = mask(v_model,datamask_obs.eq.0,False)
 copy_VarMeta(v_model,v_model_m)
 
 ; unit conversion
 v_model_c = v_model_m*u_model
 copy_VarMeta(v_model_m,v_model_c)

 ; 5-year annual mean
 v_model_amean = dim_avg_n_Wrap(v_model_c,0)
 ;printVarSummary(v_model_amean)

 ; get hist files from model run2 (last 5 years)
 ;strs = (/path_model,"test_paramset_",pset2,"/lnd/hist/*{001[6-9],20-}*"/)
 ;strs = (/path_model,pset2,"/lnd/hist/*{001[6-9],20-}*"/) 
 ;str_c = str_concat(strs)
 ;fils = systemfunc("ls "+str_c)
 ;f = addfiles(fils,"r")
 ;v_model2 = f[:]->$var_model$
 ;v_model2_amean = month_to_annual(v_model2,1)

 ; get model output - PPE
 ; set number of simulations
 ;niter = 100
 ;niter = 20

 ; set years
 ;nyrs = 5

 ; get sample file for metadata
 f_md = addfile("/glade/scratch/kdagon/archive/hydro_ensemble_LHC_1/lnd/hist/hydro_ensemble_LHC_1.clm2.h0.0016-01.nc","r")
 ;o_md = f_md->$var_model$
 ;o_md_dims = dimsizes(o_md)
 landf = f_md->landfrac
 area = f_md->area
 ; wgt variable
 arwt = area
 arwt = area * landf
 arwt = arwt/max(arwt)
 arwt_m = mask(arwt,datamask_obs.eq.0,False) 
 copy_VarMeta(area,arwt_m)

 ; set up storage matrix
 ;v_model_am = new((/niter,o_md_dims(1),o_md_dims(2)/),"float")
 ;v_model_am!0 = "ens"
 ;v_model_am!1 = "lat"
 ;v_model_am&lat = o_md&lat
 ;v_model_am!2 = "lon"
 ;v_model_am&lon = o_md&lon

 ; get hist files (last 5 years)
 ;do i=1,niter
 ; y=tostring(i)
 ; print(y)
 ; strs = (/path_model,"hydro_ensemble_LHC_",y,"/lnd/hist/*{001[6-9],20-}*"/)
 ; str_c = str_concat(strs)
 ; fils = systemfunc("ls "+str_c)
 ; f = addfiles(fils,"r")
 ; v = f[:]->$var_model$
  ; mask model output to match obs
 ; v_c = mask(v,datamask_obs.eq.0,False)
 ; copy_VarMeta(v,v_c)
  ; unit conversion
 ; v_c = v_c*u_model
 ; v_am = dim_avg_n_Wrap(v_c,0) ; 5-year annual mean maps
 ; v_model_am(i-1,:,:) = v_am ; save map
 ;end do
 ;v_model_amean = dim_avg_n_Wrap(v_model_am,0) ; mean over PPE

 ; compare obs / model
 diff = v_model_amean - v_obs_amean
 copy_VarMeta(v_model_amean, diff)
 print("Test Case - Observations")
 printMinMax(diff,1)

 ; RMSE
 ;rmse = sqrt(dim_avg_n_Wrap((v_model_c - v_obs_c)^2,0))
 rmse = sqrt((v_model_amean - v_obs_amean)^2)
 ;printVarSummary(rmse)
 copy_VarMeta(v_model_amean,rmse)
 print("Test Case - Obs RMSE:")
 printMinMax(rmse,0)

 ; perdiff
 ;v_obs_amean_mask = mask(v_obs_amean,v_obs_amean.eq.0,False)
 ;v_obs_amean_mask = mask(v_obs_amean,v_obs_amean.lt.0.01,False)
 ;perdiff = 100*diff/v_obs_amean_mask
 ;copy_VarMeta(diff,perdiff)
 ;perdiff_amean = dim_avg_n_Wrap(perdiff,0)
 ;print("Test Case - Observations (%)")
 ;printMinMax(perdiff,1)

 ; model run2
 ;diff2 = v_model2_amean - v_obs_amean
 ;copy_VarMeta(v_model2_amean, diff2)
 ;diff2_amean = dim_avg_n_Wrap(diff2,0)
 ;perdiff2 = 100*diff2/v_obs_amean_mask
 ;copy_VarMeta(diff2,perdiff2)
 ;perdiff2_amean = dim_avg_n_Wrap(perdiff2,0)

 ; plot diff
 ;plot = new(nyrs,graphic)
 plot = new(2,graphic)
 ;plot = new(3,graphic)
 ;plot = new(4,graphic)

 wks = gsn_open_wks("x11","test")
 ;wks = gsn_open_wks("pdf","model-minus-obs_4x5_2000-2004_amean_GPP_002")
 ;wks = gsn_open_wks("pdf","model-minus-obs_4x5_2000-2004_amean_GPP_006")
 ;wks = gsn_open_wks("pdf","model-minus-obs_4x5_2000-2004_amean_GPP_SVD001")
 ;wks = gsn_open_wks("pdf","model-minus-obs_4x5_2000-2004_amean_GPP_LHC84_SVD004")
 ;wks = gsn_open_wks("pdf","model-minus-obs_4x5_2000-2004_amean_GPP_SVD005")
 ;wks = gsn_open_wks("pdf","PPE-minus-obs_modeldefault-minus-obs")
 ;wks = gsn_open_wks("pdf","RMSE_"+var+test_case+"_obs_default")
 ;wks = gsn_open_wks("pdf","Mean_bias_"+var+test_case+"_obs_default")

 res = True
 res@gsnDraw = False
 res@gsnFrame = False
 res@cnFillOn = True
 res@cnLinesOn           = False           ; turn off contour lines
 res@cnLineLabelsOn      = False           ; turn off contour line labels
 res@cnFillPalette       = "MPL_BrBG"
 ;res@cnFillPalette       = "MPL_YlOrRd"
 ;res@cnFillPalette       = "amwg256"
 res@cnSpanFillPalette   = True            ; Fully span color palette
 res@lbLabelBarOn        = False           ; Turn off individual cb's
 res@lbBoxLinesOn = False ; no boxes
 res@gsnLeftString = var
 ;res@gsnLeftString = var+" RMSE"
 ;res@gsnRightString = diff@units
 ;res@gsnRightString = "~F33~m~F21~mol m~S~-2~N~s~S~-1"
 ;res@gsnRightString = "gC m~S~-2~N~ yr~S~-1"
 ;res@gsnRightString = "gC m~S~-2~N~ day~S~-1"
 res@gsnRightString = "W m~S~-2"
 ;res@gsnRightString = "%"

 ; set the contour bounds
 res@cnLevelSelectionMode = "ManualLevels"

 ; GPP diffs (gC/m^2/day)
 ;res@cnMinLevelValF  = -3
 ;res@cnMaxLevelValF  = 3
 ;res@cnLevelSpacingF = 0.3

 ; GPP RMSE
 ;res@cnMinLevelValF  = 0.
 ;res@cnMaxLevelValF  = 4.
 ;res@cnLevelSpacingF = 0.25

 ; LHF/FSH diffs (W/m^2)
 res@cnMinLevelValF  = -30
 res@cnMaxLevelValF  = 30
 res@cnLevelSpacingF = 3

 ; LHF RMSE
 ;res@cnMinLevelValF  = 0.
 ;res@cnMaxLevelValF  = 40.
 ;res@cnLevelSpacingF = 2.

 ; % diffs
 ;res@cnMinLevelValF  = -100
 ;res@cnMaxLevelValF  = 100
 ;res@cnLevelSpacingF = 10

 ;label_strs = str_concat((/pset, " relative to Observations"/))
 ;res@gsnCenterString = label_strs
 ;res@gsnCenterString = "Model with 'best' parameters - Observations"
 res@gsnCenterString = "Optimized Model - Obs"
  
 ;plot(0) = gsn_csm_contour_map(wks,diff_amean,res)
 ;res@gsnCenterString = "PPE mean - Observations"
 plot(0) = gsn_csm_contour_map(wks,diff,res)
 ;plot(0) = gsn_csm_contour_map(wks,rmse,res)

 ; percent diff
 ;res@cnMinLevelValF  = -100
 ;res@cnMaxLevelValF  = 100
 ;res@cnLevelSpacingF = 10
 ;res@gsnRightString = "%"
 ;res@gsnCenterString = "PPE mean - Observations"
 ;res@gsnCenterString = "Model with 'best' parameters - Observations"
 ;plot(1) = gsn_csm_contour_map(wks,perdiff,res)

 ;resP                     = True                ; modify the panel plot
 ;resP@gsnMaximize         = True
 ;resP@gsnPanelMainString = "Model - Observations"
 ;resP@gsnPanelMainString = "PPE mean - Observations"
 ;resP@gsnPanelLabelBar    = True                ; add common colorbar
 ;resP@lbBoxLinesOn = False ; no boxes
 ;gsn_panel(wks,plot,(/3,2/),resP) 
 ;gsn_panel(wks,plot,(/1,2/),resP)

 ;label_strs2 = str_concat((/pset2, " relative to Observations"/))
 ;res@gsnCenterString = label_strs2
 ;plot(1) = gsn_csm_contour_map(wks,diff2_amean,res)
 ;plot(0) = gsn_csm_contour_map(wks,perdiff2_amean,res)

 ; compare obs / model default
 strs_def = (/path_model,"hydro_ensemble_LHC_default_params/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs_def)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model_default = f[:]->$var_model$
 v_model_default_m = mask(v_model_default,datamask_obs.eq.0,False)
 copy_VarMeta(v_model_default,v_model_default_m)
 v_model_default_c = v_model_default_m*u_model
 copy_VarMeta(v_model_default_m,v_model_default_c)
 v_model_def_amean = dim_avg_n_Wrap(v_model_default_c,0)
 ;v_model_def_amean = month_to_annual(v_model_default,1)
 ;printVarSummary(v_model_def_amean) 

 diff_def = v_model_def_amean - v_obs_amean
 copy_VarMeta(v_model_def_amean, diff_def)
 print("Default - Observations")
 printMinMax(diff_def,1)

 ;rmse_def = dim_rmsd_n_Wrap(v_model_default_c, v_obs_c, 0)
 ;rmse_def = dim_rmsd_Wrap(v_model_def_amean, v_obs_amean)
 rmse_def = sqrt((v_model_def_amean - v_obs_amean)^2)
 ;printVarSummary(rmse_def)
 copy_VarMeta(v_model_def_amean,rmse_def)
 print("Default / Obs RMSE:")
 printMinMax(rmse_def,0)

 ;perdiff_def = 100*diff_def/v_obs_amean_mask
 ;copy_VarMeta(diff_def,perdiff_def)
 ;perdiff_def_amean = dim_avg_n_Wrap(perdiff_def,0)
 ;print("Default - Observations (%)")
 ;printMinMax(perdiff_def,1)

 ;res@cnMinLevelValF  = -1000
 ;res@cnMaxLevelValF  = 1000
 ;res@cnLevelSpacingF = 100
 ;res@gsnRightString = "gC m~S~-2~N~ yr~S~-1"
 ;res@gsnCenterString = "Model with default parameters - Observations"
 res@gsnCenterString = "Default Model - Obs"  

 plot(1) = gsn_csm_contour_map(wks,diff_def,res)
 ;plot(1) = gsn_csm_contour_map(wks,rmse_def,res)

 ;res@cnMinLevelValF  = -100
 ;res@cnMaxLevelValF  = 100
 ;res@cnLevelSpacingF = 10
 ;res@gsnRightString = "%"
 ;res@gsnCenterString = "Model with default parameters - Observations"
 ;plot(3) = gsn_csm_contour_map(wks,perdiff_def,res)

 resP                     = True                ; modify the panel plot
 resP@gsnMaximize         = True
 ;resP@gsnPanelMainString = "Model - Observations"
 resP@gsnPanelLabelBar    = True                ; add common colorbar
 resP@lbBoxLinesOn = False ; no boxes
 ;gsn_panel(wks,plot,(/2,2/),resP)
 gsn_panel(wks,plot,(/1,2/),resP)

 ; compare regional seasonal cycles
 ; set point coordinates
 ;latp = -5
 ;lonp = 290

 ; set regional bounds
 latmin = -10
 latmax = 0
 lonmin = 290
 lonmax = 310

 v_obs_clim = clmMonTLL(v_obs_c)
 ;v_obs_time = v_obs_clim(:,{latp},{lonp})
 v_obs_time = wgt_areaave2(v_obs_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)
 printVarSummary(v_obs_clim(:,{latmin:latmax},{lonmin:lonmax}))
 v_model_def_clim = clmMonTLL(v_model_default_c)
 ;v_model_def_time = v_model_def_clim(:,{latp},{lonp}) 
 v_model_def_time = wgt_areaave2(v_model_def_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)
 v_model_clim = clmMonTLL(v_model_c)
 ;v_model_time = v_model_clim(:,{latp},{lonp})
 v_model_time = wgt_areaave2(v_model_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)
 ;v_model2_clim = clmMonTLL(v_model2)
 ;v_model2_time = v_model2_clim(:,{latp},{lonp})

 dims = dimsizes(v_model_def_time)
 ;data = new((/4,dims(0)/),float)
 data = new((/3,dims(0)/),float)
 data(0,:) = v_obs_time
 data(1,:) = v_model_time
 ;data(2,:) = v_model2_time
 ;data(3,:) = v_model_def_time
 data(2,:) = v_model_def_time

 wks2 = gsn_open_wks("x11","test")
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt")
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt_006") 
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt_006_005")
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt_002_006")
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt_SVD001_002")
 ;wks2 = gsn_open_wks("pdf","model-compare-obs_seasonal-cycle_GPP_pt_LHC84_SVD004")
 ;wks2 = gsn_open_wks("pdf",var+"_month_climo_"+test_case+"_obs_default")

 ;plot2 = new(2, graphic)
 plot2 = new(4, graphic)

 res2 = True
 res2@gsnDraw = False
 res2@gsnFrame = False
 
 res2@xyLineColors = (/"red","blue","black"/) ; obs, model test, model default
 ;res2@xyLineColors = (/"red","blue","green","black"/) ; obs, model test1, model test2, model default
 ;res@xyDashPatterns     = (/0,0,1,0/)
 res2@xyMonoDashPattern = True
 res2@xyDashPattern = 0
 res2@xyMonoLineThickness = True
 res2@xyLineThicknessF = 3.0

 ;res2@tiYAxisString = "GPP (~F33~m~F21~mol m~S~-2~N~s~S~-1~N~)"
 ;res2@tiYAxisString = "GPP (gC m~S~-2~N~ day~S~-1~N~)"
 ;res2@tiYAxisString = "LHF (W m~S~-2~N~)"
 res2@tiYAxisString = "SH (W m~S~-2~N~)" 
 ;res2@tiXAxisString = "Months"
 res2@tiMainString = "Amazon (0-10S, 70-50W)" 

 ; legend settings
 ;res2@pmLegendDisplayMode    = "Never"            ; turn on legend
 res2@pmLegendDisplayMode    = "Always"
 res2@pmLegendSide           = "Top"               ; Change location of 
 ; GPP
 ;res2@pmLegendParallelPosF   = 0.5                  ; move units right
 ; LHF
 res2@pmLegendParallelPosF   = 0.4
 res2@pmLegendOrthogonalPosF = -0.4                ; move units down
 res2@pmLegendWidthF         = 0.15                ; Change width and
 res2@pmLegendHeightF        = 0.15                ; height of legend.
 res2@lgPerimOn              = False               ; turn off box around
 res2@lgLabelFontHeightF     = .02                 ; label font height
 res2@xyExplicitLegendLabels = (/"Obs","Opt Model","Default Model"/)         ; create explicit labels
 ;res2@xyExplicitLegendLabels = (/"Obs","Model Preds GM","Model Preds SVD","Model Default"/)
 ;res2@xyExplicitLegendLabels = (/"Obs",pset,pset2,"Model Default"/)

 ;years = ispan(1,5,1)
 ;plot(2) = gsn_csm_xy(wks,years,data,res)
 months = ispan(1,12,1)

 ; explicit axis labels
 res2@trXMinF = min(months)

 res2@tmXBMode      = "Explicit" ; label independently
 res2@tmXBValues    = months
 res2@tmXBLabels    = (/"J","F","M","A","M","J","J","A","S","O","N","D"/)

 plot2(0) = gsn_csm_xy(wks2,months,data,res2)

 ; second region
 ;latp = 10
 ;lonp = 5
 latmin = 5
 latmax = 15
 lonmin = 0
 lonmax = 20
 ;v_obs_time = v_obs_clim(:,{latp},{lonp})
 ;v_model_def_time = v_model_def_clim(:,{latp},{lonp})
 ;v_model_time = v_model_clim(:,{latp},{lonp})
 ;v_model2_time = v_model2_clim(:,{latp},{lonp})
 v_obs_time = wgt_areaave2(v_obs_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)                           
 printVarSummary(v_obs_clim(:,{latmin:latmax},{lonmin:lonmax}))                                                                               
 v_model_def_time = wgt_areaave2(v_model_def_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)               
 v_model_time = wgt_areaave2(v_model_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0) 
 data(0,:) = v_obs_time                                                                                                                        
 data(1,:) = v_model_time
 ;data(2,:) = v_model2_time
 ;data(3,:) = v_model_def_time
 data(2,:) = v_model_def_time
 res2@tiMainString = "Sahel (5-15N, 0-20E)"
 res2@tiYAxisString = ""
 res2@tiXAxisString = ""
 res2@pmLegendDisplayMode    = "Never"
 ;res2@pmLegendDisplayMode    = "Always"
 plot2(1) = gsn_csm_xy(wks2,months,data,res2)

 ; third region
 ;latp = 40
 ;lonp = 260
 latmin = 33
 latmax = 45
 lonmin = 260
 lonmax = 280
 ;v_obs_time = v_obs_clim(:,{latp},{lonp})
 ;v_model_def_time = v_model_def_clim(:,{latp},{lonp})
 ;v_model_time = v_model_clim(:,{latp},{lonp})
 v_obs_time = wgt_areaave2(v_obs_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)
 printVarSummary(v_obs_clim(:,{latmin:latmax},{lonmin:lonmax}))
 v_model_def_time = wgt_areaave2(v_model_def_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)
 v_model_time = wgt_areaave2(v_model_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)
 data(0,:) = v_obs_time
 data(1,:) = v_model_time
 data(2,:) = v_model_def_time
 res2@tiMainString = "Eastern US (35-45N, 100-80W)"
 res2@tiXAxisString = "Months"
 ;res2@tiYAxisString = "GPP (gC m~S~-2~N~ day~S~-1~N~)"
 ;res2@tiYAxisString = "LHF (W m~S~-2~N~)"
 res2@tiYAxisString = "SH (W m~S~-2~N~)"  
 plot2(2) = gsn_csm_xy(wks2,months,data,res2)

 ; fourth region
 latmin = 45
 latmax = 55
 lonmin = 0
 lonmax = 20
 v_obs_time = wgt_areaave2(v_obs_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)
 printVarSummary(v_obs_clim(:,{latmin:latmax},{lonmin:lonmax}))
 v_model_def_time = wgt_areaave2(v_model_def_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)
 v_model_time = wgt_areaave2(v_model_clim(:,{latmin:latmax},{lonmin:lonmax}),arwt_m({latmin:latmax},{lonmin:lonmax}),0)
 data(0,:) = v_obs_time
 data(1,:) = v_model_time
 data(2,:) = v_model_def_time
 res2@tiMainString = "Western Europe (45-55N, 0-20E)"
 res2@tiXAxisString = "Months"
 res2@tiYAxisString = ""
 plot2(3) = gsn_csm_xy(wks2,months,data,res2) 


 res2P                     = True                ; modify the panel plot
 res2P@gsnMaximize         = True
 ;res2P@gsnPanelMainString = "Model - Observations"
 res2P@gsnPanelBottom     = 0.05 ; add space at bottom
 res2P@gsnPanelYWhiteSpacePercent = 5 ; add between panel y space
 gsn_panel(wks2,plot2,(/2,2/),res2P) 

end
