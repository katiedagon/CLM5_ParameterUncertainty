; ***********************************************
; equilibrium.ncl
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ;fils = systemfunc("ls /glade/scratch/kdagon/archive/hydro_ensemble_LHC_default_params/lnd/hist/*.nc")
 ;fils = systemfunc("ls /glade/scratch/kdagon/archive/test_streams/lnd/hist/*.nc")
 fils = systemfunc("ls /glade/scratch/kdagon/archive/future_climate_LHC_1/lnd/hist/*.nc")
 q    = addfiles(fils,"r")
 
 ; read in all the files and join the timeseries together

 ;TSOI_10cm      = q[:]->TSOI_10CM
 ;printVarSummary(TSOI_10cm)
 ;H2OSOI_10cm    = q[:]->SOILWATER_10CM
 TSOI     	= q[:]->TSOI ; 4D variable (time x lev x lat x lon)
 ;printVarSummary(TSOI)
 H2OSOI		= q[:]->H2OSOI ; 4D variable (time x lev x lat x lon)
 ;printVarSummary(H2OSOI)

 SOILLIQ = q[:]->SOILLIQ 
 SOILICE = q[:]->SOILICE
 SOILWATER = SOILLIQ
 SOILWATER = SOILLIQ+SOILICE
 SOILWATER_TOT = dim_sum_n_Wrap(SOILWATER,1)
 ;printVarSummary(SOILWATER_TOT)

 TSOI_C = TSOI ; metadata
 TSOI_C = TSOI - 273 ; convert temp K->C
 TSOI_C&levgrnd = TSOI_C&levgrnd*1000 ; convert depth m->mm

 ;lev=10
 ;H2OSOI_active = H2OSOI(:,:lev-1,:,:)
 ;H2OSOI_active!1 = "levsoi"
 ;H2OSOI_active&levsoi = TSOI_C&levgrnd
 ;printVarSummary(H2OSOI_active)
 ;H2OSOI_active&levsoi = H2OSOI_active&levsoi*1000 ; convert depth m->mm

 ; get landfrac & area
 landf = q[0]->landfrac
 ;printVarSummary(landf)
 area = q[0]->area

 ; wgt variable
 arwt = area
 arwt = area * landf
 arwt = arwt/max(arwt)
 ;printVarSummary(arwt)

 ; global means
 TSOI_C_gm = wgt_areaave2(TSOI_C,arwt,0) 
 ;TSOI_C_gm&time = ispan(0,359,1) ; 30 years of monthly average data (30*12 = 360)
 ;printVarSummary(TSOI_C_gm)
 H2OSOI_active_gm = wgt_areaave2(H2OSOI,arwt,0)
 ;H2OSOI_active_gm&time = ispan(0,359,1)
 ;printVarSummary(H2OSOI_active_gm)

 ; annual means
 ;TSOI_C_annual = block_avg(TSOI_C_gm,12)
 ;TSOI_C_annual = month_to_annual(TSOI_C_gm,1)
 ;printVarSummary(TSOI_C_annual) 

 ;H2OSOI_active_annual = block_avg(H2OSOI_active_gm,12)
 ;H2OSOI_active_annual = month_to_annual(H2OSOI_active_gm,1)
 ;printVarSummary(H2OSOI_active_annual)

 ; monthly means
 ;TSOI_C_monthly = calculate_monthly_values(TSOI_C_gm,"avg",0,False)
 ;H2OSOI_active_monthly = calculate_monthly_values(H2OSOI_active_gm,"avg",0,False)
 ;printVarSummary(H2OSOI_active_monthly)

 ; total column sum (soil moisture)
 ;H2OSOI_active_annual_sum = dim_sum_n_Wrap(H2OSOI_active_annual,1)
 SOILWATER_gm = wgt_areaave2(SOILWATER_TOT,arwt,0)
 SOILWATER_gm!0 = "time"
 SOILWATER_gm&time = SOILWATER_TOT&time
 ;printVarSummary(SOILWATER_gm)
 SOILWATER_gm_am = month_to_annual(SOILWATER_gm,1)

 ; regression lines
 ;t = ispan(1,30,1)
 ;rc = regline(t,SOILWATER_gm_am)
 ;print(rc)
 ;reg = rc*t + rc@yintercept
 ;t15 = ispan(1,16,1)
 ;rc15 = regline(t15,SOILWATER_gm_am(14:))
 ;print(rc15)
 ;reg15 = rc15*t15 + rc15@yintercept

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

 ;res@tiXAxisString     = "~F21~Months"                                                                                              
 res@tiXAxisString     = "Years"
 res@tiYAxisString     = "~F21~Total Column Soil Moisture (kg m~S~-2~N~)"  
 res@xyLineThicknessF = 3

 ;data = new((/2,30/),float)
 ;data(0,:) = SOILWATER_gm_am
 ;data(1,:) = reg

 ;plot = gsn_csm_y(wks_panel,SOILWATER_gm,res) 
 plot = gsn_csm_y(wks_panel,SOILWATER_gm_am,res)
 ;plot = gsn_csm_y(wks_panel,data,res)

 ;data15 = new((/2,16/),float)
 ;data15(0,:) = SOILWATER_gm_am(14:)
 ;data15(1,:) = reg15

 ;wks_panel  = gsn_open_wks("eps","hydro_ensemble0_soil1D_annual_trends15")
 ;plot = gsn_csm_y(wks_panel,data15,res)

 delete(plot)
 wks_panel = gsn_open_wks("x11","test")
 plot = new(2,graphic)

 res@gsnDraw             = False           ; don't draw
 res@gsnFrame            = False           ; don't advance frame

 res@cnFillOn            = True            ; turn on color
 res@cnLinesOn           = False           ; turn off contour lines                                  
 ;res@cnLineLabelsOn      = False           ; turn off contour line labels
 ;res@lbBoxLinesOn        = False           ; turn off labelbar box lines
 res@cnFillPalette       = "BrownBlue12"       ; color palette
 res@cnSpanFillPalette   = True           ; Fully span color palette
 ;res@lbLabelBarOn        = False           ; turn off individual cb's
 
 res@trYReverse          = True                 ; reverse y-axis
 res@gsnYAxisIrregular2Log = True ; set irregular y-axis to log scale

 ; tick mark / label bar font
 res@tmXBLabelFont         = 21
 res@tmYLLabelFont         = 21
 res@lbLabelFont         = 21

 ; turn off top/right tickmarks
 res@tmXTOn = False
 res@tmYROn = False

 ; custom x-axis for monthly data
 ;res@tmXBMode = "Explicit"
 ;res@tmXBLabels = ispan(2,30,2)
 ;res@tmXBValues = ispan(24,360,24)

 res@lbOrientation        = "Vertical"   ; Rotate labelbar
 res@lbLabelStride        = 2 ; only label every other

 res@tiXAxisString     = ""
 res@tiYAxisString     = "~F21~Soil Depth (mm)"
 res@gsnLeftString   = ""
 res@gsnRightString  = ""
 res@tiMainString      = "~F21~Soil Temperature (~S~o~N~C)"

 ;printVarSummary(transpose(TSOI_C_gm)) 
 plot(0) = gsn_csm_contour(wks_panel,transpose(TSOI_C_gm),res)
 ;plot(0) = gsn_csm_contour(wks_panel,transpose(TSOI_C_annual),res) 

 res@gsnYAxisIrregular2Log = False
 res@tiXAxisString     = "~F21~Months from Start"
 ;res@tiXAxisString     = "~F21~Years from Start"
 res@tiYAxisString     = "~F21~Soil Layer"
 res@tiMainString      = "~F21~Soil Moisture (mm~S~3~N~mm~S~-3~N~)"
 plot(1) = gsn_csm_contour(wks_panel,transpose(H2OSOI_active_gm),res)   
 ;plot(1) = gsn_csm_contour(wks_panel,transpose(H2OSOI_active_annual),res)



 resP                     = True                ; modify the panel plot
 ;resP@txString            = "~F21~Central U.S. Control, 40~S~o~N~N, 97~S~o~N~W"
 ;resP@gsnPanelLabelBar    = True                ; add common colorbar
 ;resP@lbLabelFontHeightF  = 0.007               ; make labels smaller
 ;resP@gsnPanelYWhiteSpacePercent = 3            ; add white space between panels
 ;resP@gsnFrame            = False               ; don't advance frame
 resP@gsnPanelBottom = 0.05 ; force extra white space at bottom
 resP@gsnPanelTop = 0.95 ; and at top 
 gsn_panel(wks_panel,plot,(/2,1/),resP)               ; now draw as one plot

 ; now advance frame                                                                                   
 ;frame(wks_panel)

end
