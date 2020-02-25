;****************************************
; PE_plot_top.ncl 
;
; calculate and plot PE for top 6 parameters
;****************************************
begin
;****************************************
; specify some settings
;****************************************
 ; set output variable(s) of interest
 ; EF
 ;var1="QVEGT"
 ;var2="QFLX_EVAP_TOT"
 ; WUE
 ;var1="FPSN" ; FPSN=GPP in SP mode
 ;var2="GSSHA"                                                                                                                                 
 ;var3="GSSUN"
 ; GPP
 var = "FPSN"
 ; ET
 ;var = "QFLX_EVAP_TOT"
 ; SM10
 ;var = "SOILWATER_10CM"
 ; ZWT
 ;var = "ZWT"

 ; top 6 (final list)
 param = (/"medlynslope","kmax","fff","dint","dleaf","baseflow_scalar"/)

 ; define "option" tag for each parameter
 ; opt=0 means use existing min/max runs
 ; opt=1 means use default for min
 ; opt=2 means use default for max
 ;opt = (/0,0,0,0,0,0,0,0,0,0/) ; current top 6 both have min/max runs

 ; set number of calculations
 iter=dimsizes(param)

 ; set output path
 path = "/glade/scratch/kdagon/archive/"

;****************************************
; get panel plot ready
;****************************************
 plot = new(iter,graphic)

 ;wks  = gsn_open_wks("x11","panels")
 wks  = gsn_open_wks("pdf","PE_GPP_top6")

 res                     = True
 res@gsnDraw             = False           ; don't draw
 res@gsnFrame            = False           ; don't advance frame
 res@gsnMaximize         = True            ; maximize plot in frame
 res@cnFillOn            = True            ; turn on color fill
 res@cnLinesOn           = False           ; turn off contour lines
 res@cnLineLabelsOn      = False           ; turn off contour line labels 
 ;res@cnSpanFillPalette   = True            ; Fully span color palette
 ;res@cnFillPalette       = "WhiteYellowOrangeRed"
 res@lbLabelBarOn        = False           ; turn off individual cb's

 ; common label bar specs - need to specify or else plot will use first panel bounds
 ;res@cnLevelSelectionMode =  "ManualLevels"
 ;res@cnMinLevelValF       = 0
 ;res@cnMaxLevelValF       = 3
 ;res@cnLevelSpacingF      = 0.075

 ; explicit colorbar levels
 gsn_define_colormap(wks,"WhiteYellowOrangeRed")
 res@cnLevelSelectionMode =  "ExplicitLevels"
 res@cnLevels = (/0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0/) ; length n
 res@cnFillColors = ispan(0,250,10) ; length of colors array should be n+1 

;****************************************
; loop over ensemble; save PE for each param
;****************************************
 do i=1,iter
  y=param(i-1)
  ; min output
  strs = (/path,"hydro_ensemble_reruns_",y,"_min/lnd/hist/*{001[6-9],20-}*"/)
  str_c = str_concat(strs)
  fils = systemfunc("ls "+str_c)
  f = addfiles(fils,"r")
  v_min = f[:]->$var$

  ; max output
  strs = (/path,"hydro_ensemble_reruns_",y,"_max/lnd/hist/*{001[6-9],20-}*"/)
  str_c = str_concat(strs)
  fils = systemfunc("ls "+str_c)
  f2 = addfiles(fils,"r")
  v_max = f2[:]->$var$

  PE_all = v_min ; metadata
  PE_all = abs(v_max - v_min)
  ;PE_all = 86400*abs(v_max - v_min) ; ET [m/s] -> [mm/day]

;****************************************
; plot each param
;****************************************
  ; Parameter name label at top left of each panel
  res@gsnLeftStringFontHeightF = 0.04
  res@gsnLeftString  = param(i-1)
  res@gsnRightString   = ""

  plot(i-1) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(PE_all,0),res)
 end do

;****************************************
; panel plot
;****************************************
 resP                     = True                ; modify the panel plot
 resP@gsnMaximize         = True
 resP@gsnPanelMainString  = "Parameter Effect (PE) of GPP (~F33~m~F21~mol CO~B~2~N~ m~S~-2~N~s~S~-1~N~)"

 ; add white space
 resP@gsnPanelYWhiteSpacePercent = 5

 resP@gsnPanelLabelBar    = True                ; add common colorbar
 resP@lbBoxLinesOn = False ; no boxes

 gsn_panel(wks,plot,(/3,2/),resP)

end
