;---------------------------------------------
; datamask.ncl
;
; Plot and compare datamasks for obs products
;---------------------------------------------

begin
  ; Read in obs files
  FLUX_GPP_in = addfile("MR_GPP_4x5_2000_2004.nc","r")
  FLUX_LHF_in = addfile("MR_LHF_4x5_2000_2004.nc","r")
  WECANN_in = addfile("WECANN_GPP_LHF_4x5_2007_2015.nc","r")

  flux_gpp_dm = FLUX_GPP_in->datamask
  ;print(num(ismissing(flux_gpp_dm)))
  ;printVarSummary(flux_gpp_dm)
  print(flux_gpp_dm({0},:))
  flux_lhf_dm = FLUX_LHF_in->datamask
  wecann_gpp_dm = WECANN_in->datamaskG
  wecann_lhf_dm = WECANN_in->datamaskL

  ; Comparison Plots
  wks = gsn_open_wks("x11","test")
  ;wks = gsn_open_wks("pdf","FLUXNET_WECANN_GPP_LHF_4x5_datamasks")

  res=True
  res@cnFillOn    = True 
  res@gsnDraw  = False
  res@gsnFrame = False
 
  res@lbLabelBarOn        = False            ; turn off individual cb's
 
  res@cnLevelSelectionMode = "ExplicitLevels" 
  res@cnLevels         =   (/0.5/)
  res@lbLabelAlignment  = "BoxCenters"
  res@lbLabelStrings    = (/ "0","1"/)

  plot = new(4, graphic)

  res@gsnLeftString  = "FLUXNET GPP"
  plot(0) = gsn_csm_contour_map(wks,flux_gpp_dm,res)
  res@gsnLeftString  = "FLUXNET LHF"
  plot(1) = gsn_csm_contour_map(wks,flux_lhf_dm,res)
  res@gsnLeftString  = "WECANN GPP"
  plot(2) = gsn_csm_contour_map(wks,wecann_gpp_dm,res)
  res@gsnLeftString  = "WECANN LHF"
  plot(3) = gsn_csm_contour_map(wks,wecann_lhf_dm,res)

  ; plot panel
  resP                  = True                   ; modify the panel plot
  resP@gsnPanelLabelBar = True
  resP@gsnPanelYWhiteSpacePercent = 5

  gsn_panel(wks,plot,(/2,2/),resP)


end

