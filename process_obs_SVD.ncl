; ***********************************************
; process_obs_SVD.ncl
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

 ; get obs file
 ;filename = "MR_GPP_0.9x1.25_ALLMONS_climo.nc"
 filename = "MR_GPP_0.9x1.25_2000_2004_amean.nc"
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")

 ; get gridcell areas
 grid_area = f_obs->area 
 grid_area_m = grid_area ; metadata
 grid_area_m = grid_area*10^6 ; km^2 to m^2

 ; read in obs
 v_obs = f_obs->$var$
 ; unit conversion
 v_obs_c = v_obs*u
 ; copy metadata
 copy_VarMeta(v_obs,v_obs_c)
 printVarSummary(v_obs_c)
 ;area-weighted, copy metadata
 v_obs_c_area = v_obs_c
 do i=1,nyrs
  v_obs_c_area(i-1,:,:) = v_obs_c(i-1,:,:) * grid_area_m
 end do
 ; Full annual mean across nyrs
 v_obs_c_area_am = dim_avg_n_Wrap(v_obs_c_area,0)
 printVarSummary(v_obs_c_area_am)

 ; input to SVD - subtract "ensemble" mean
 v_obs_anom = v_obs_c_area ; metadata
 do i=1,nyrs
  v_obs_anom(i-1,:,:) = v_obs_c_area(i-1,:,:)-v_obs_c_area_am
 end do 

 ; update units
 v_obs_anom@units = "PgC/yr"
 printVarSummary(v_obs_anom)
 
 ; test plots
 plot = new(nyrs,graphic)

 ;wks = gsn_open_wks("x11","test")
 ;wks = gsn_open_wks("pdf","obs_GPP_0.9x1.25_forSVD")
 wks = gsn_open_wks("pdf","obs_GPP_0.9x1.25_anom_forSVD")

 res = True
 res@gsnDraw = False
 res@gsnFrame = False
 res@cnFillOn = True
 res@cnLinesOn           = False           ; turn off contour lines
 res@cnLineLabelsOn      = False           ; turn off contour line labels
 res@cnFillPalette       = "MPL_YlOrRd"
 res@cnSpanFillPalette   = True            ; Fully span color palette
 res@lbLabelBarOn        = False	   ; Turn off individual cb's
 res@gsnLeftString = var
 res@gsnRightString = v_obs_anom@units

 ; set the contour bounds
 ;res@cnLevelSelectionMode = "ManualLevels"
 ;res@cnMinLevelValF  = -1000
 ;res@cnMaxLevelValF  = 1000
 ;res@cnLevelSpacingF = 50

 do i=1,nyrs
  y=tostring(i)
  title_strs = str_concat((/"Year ",y/))
  res@gsnCenterString = title_strs
  plot(i-1) = gsn_csm_contour_map(wks,v_obs_anom(i-1,:,:),res)
 end do 

 resP                     = True                ; modify the panel plot
 resP@gsnMaximize         = True
 resP@gsnPanelMainString = "Annual mean, area-weighted, anomaly from 5-year mean"
 resP@gsnPanelLabelBar    = True                ; add common colorbar
 resP@lbBoxLinesOn = False ; no boxes
 gsn_panel(wks,plot,(/3,2/),resP) 


;************************************************
; write output file
;************************************************
 ; writing as nc (preferred)
 ncdf = addfile("obs/obs_GPP_anom_forSVD.nc", "c")
 ; define time as an unlimited dimension (optional?)
 ;filedimdef(ncdf,"time",-1,True) 
 ncdf->GPP = v_obs_anom
 

end
