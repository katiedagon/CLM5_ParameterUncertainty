; ***********************************************
; future_warming_test.ncl
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ; control (default params) only has last 5 years (16-20)
 ;fils1 = systemfunc("ls /glade/scratch/kdagon/archive/hydro_ensemble_LHC_default_params/lnd/hist/*.nc")
 fils1 = systemfunc("ls /glade/scratch/kdagon/archive/hydro_ensemble_LHC_1/lnd/hist/*{001[6-9],20-}*")
 ;fils1 = systemfunc("ls /glade/scratch/kdagon/archive/future_climate_LHC_1/lnd/hist/*{001[6-9],20-}*")
 q1 = addfiles(fils1,"r")

 ; warming test currently has 20 years (1-20)
 ;fils2 = systemfunc("ls /glade/scratch/kdagon/archive/test_streams/lnd/hist/*{001[6-9],20-}*")
 fils2 = systemfunc("ls /glade/scratch/kdagon/archive/future_climate_LHC_1/lnd/hist/*{001[6-9],20-}*")
 q2 = addfiles(fils2,"r")

 ; get file var names
 vNames = getfilevarnames(q1[0])
 ;print(vNames)

 ;var = "TSA"
 ;var = "RH2M"
 ;var = "Q2M"
 ;var = "FPSN"
 ;var = "QFLX_EVAP_TOT"
 var = "EFLX_LH_TOT"
 ;var = "FSH" 
 ;var = "RAIN"
 ;var = "SOILWATER_10CM"
 ;var = "TBOT"
 ;var = "QBOT"

 ; read in all the files and join the timeseries together
 TSA1 = q1[:]->$var$
 TSA2 = q2[:]->$var$

 ; get landfrac & area
 landf = q1[0]->landfrac
 ;printVarSummary(landf)
 area = q1[0]->area

 ; wgt variable
 arwt = area
 arwt = area * landf
 arwt = arwt/max(arwt)
 ;printVarSummary(arwt)

 ; unit conversion (if needed)
 u = 1
 ;u = (12 * (3600 * 24 * 365) / 1000000) * 10^-15 ; umol /m2 /s to Pg/m2/year
 ;u = 86400 ; mm/s to mm/day
 TSA1 = TSA1*u
 TSA2 = TSA2*u

 ; annual means
 TSA1_am = dim_avg_n_Wrap(TSA1,0)
 TSA2_am = dim_avg_n_Wrap(TSA2,0)

 ; area weight (if needed)
 ar = 1
 ;ar = area*10^6 ; km^2 to m^2
 TSA1_am = TSA1_am*ar
 TSA2_am = TSA2_am*ar

 TSA_diff = TSA2_am - TSA1_am
 copy_VarMeta(TSA1_am, TSA_diff)

 ;TSA_perdiff = 100*TSA_diff/TSA1_am
 ;copy_VarMeta(TSA_diff, TSA_perdiff)

 ; global mean difference
 TSA_diff_gm = wgt_areaave2(TSA_diff,arwt,0)
 print(TSA_diff_gm)

 ;TSA_perdiff_gm = wgt_areaave2(TSA_perdiff,arwt,0)
 ;print(TSA_perdiff_gm)

;************************************************
; plot 2D panels
;************************************************
 wks_panel = gsn_open_wks("x11","test")
 ;wks_panel  = gsn_open_wks("eps","hydro_ensemble0_soil2D")
 ;wks_panel  = gsn_open_wks("eps","hydro_ensemble0_soil2D_annual")
 ;wks_panel  = gsn_open_wks("eps","hydro_ensemble0_soil1D_annual")
 ;wks_panel  = gsn_open_wks("eps","hydro_ensemble0_soil1D_annual_trends")

 res                     = True
 res@gsnMaximize         = True          ; maximize plot size

 ;plot = new(2,graphic)

 ;res@gsnDraw             = False           ; don't draw
 ;res@gsnFrame            = False           ; don't advance frame

 res@cnFillOn            = True            ; turn on color
 res@cnLinesOn           = False           ; turn off contour lines                                  
 ;res@cnLineLabelsOn      = False           ; turn off contour line labels
 ;res@lbBoxLinesOn        = False           ; turn off labelbar box lines
 ;res@cnFillPalette       = "BlWhRe"       ; color palette
 res@cnFillPalette       = "MPL_Reds"
 res@cnSpanFillPalette   = True           ; Fully span color palette
 ;res@lbLabelBarOn        = False           ; turn off individual cb's
 

 ; turn off top/right tickmarks
 res@tmXTOn = False
 res@tmYROn = False

 res@tiMainString      = "Changes with constant 2K warming"

 plot = gsn_csm_contour_map_ce(wks_panel,TSA_diff,res)

 ;res@cnLevelSelectionMode = "ManualLevels"
 ;res@cnMinLevelValF  = -100
 ;res@cnMaxLevelValF  = 100
 ;res@cnLevelSpacingF = 5 
 ;plot = gsn_csm_contour_map_ce(wks_panel,TSA_perdiff,res)

 ;plot(0) = gsn_csm_contour(wks_panel,transpose(TSOI_C_gm),res)
 ;plot(1) = gsn_csm_contour(wks_panel,transpose(H2OSOI_active_annual),res)


 ;resP                     = True                ; modify the panel plot
 ;resP@gsnPanelLabelBar    = True                ; add common colorbar
 ;resP@lbLabelFontHeightF  = 0.007               ; make labels smaller
 ;resP@gsnPanelYWhiteSpacePercent = 3            ; add white space between panels
 ;resP@gsnFrame            = False               ; don't advance frame
 ;resP@gsnPanelBottom = 0.05 ; force extra white space at bottom
 ;resP@gsnPanelTop = 0.95 ; and at top 
 ;gsn_panel(wks_panel,plot,(/2,1/),resP)               ; now draw as one plot

 ; now advance frame                                                                                   
 ;frame(wks_panel)

end
