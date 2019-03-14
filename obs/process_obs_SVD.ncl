; ***********************************************
; process_obs_SVD.ncl
; ***********************************************
begin
;************************************************
; read in obs data
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
 filename = "MR_GPP_4x5_2000_2004.nc"
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")

 ; get gridcell areas
 grid_area = f_obs->area 
 grid_area_m = grid_area ; metadata
 grid_area_m = grid_area*10^6 ; km^2 to m^2

 ; get obs datamask
 datamask_obs = f_obs->datamask

 ; read in obs
 v_obs = f_obs->$var$
 ; unit conversion
 v_obs_c = v_obs*u
 ; copy metadata
 copy_VarMeta(v_obs,v_obs_c)
 ;printVarSummary(v_obs_c)
 ; 5-year annual mean
 v_obs_am = dim_avg_n_Wrap(v_obs_c,0) 
 ; area-weighted annual mean, copy metadata
 v_obs_am_area = v_obs_am * grid_area_m
 copy_VarMeta(v_obs_am,v_obs_am_area)
 printVarSummary(v_obs_am_area)

;************************************************
; read in LHC ensemble output
;************************************************
 ; set number of simulations/ens members
 ;niter = 100

 ; set output path
 ;path_ens = "/glade/scratch/kdagon/archive/"

 ; set variable(s) of interest
 ;var_ens = "FPSN"

 ; unit conversion (if needed)
 ;u_ens = (12 * (3600 * 24 * 365) / 1000000) * 10^-15 ; umol /m2 /s to Pg/m2/year
 ;u = (3600 * 24 * 365) / 1000 ; mm/s to m/year
 ;u = 0.001 ; mm/s to m/s

 ; get sample file for metadata
 ;f_md = addfile("/glade/scratch/kdagon/archive/hydro_ensemble_LHC_1/lnd/hist/hydro_ensemble_LHC_1.clm2.h0.0016-01.nc","r")
 ;o_md = f_md->$var_ens$
 ;o_md_dims = dimsizes(o_md)

 ; get gridcell areas
 ;grid_area_ens = f_md->area
 ;grid_area_ens_m = grid_area_ens ; metadata
 ;grid_area_ens_m = grid_area_ens*10^6 ; km^2 to m^2

 ; set up storage matrix
 ;O = new((/niter+1,o_md_dims(1),o_md_dims(2)/),"float")
 ;O!0 = "ens"
 ;O!1 = "lat"
 ;O&lat = o_md&lat
 ;O!2 = "lon"
 ;O&lon = o_md&lon

 ;do i=1,niter
 ; y=tostring(i)
 ; print(y)
 ; strs = (/path_ens,"hydro_ensemble_LHC_",y,"/lnd/hist/*{001[6-9],20-}*"/)
 ; str_c = str_concat(strs)
 ; fils = systemfunc("ls "+str_c)
 ; f = addfiles(fils,"r")
 ; o = f[:]->$var_ens$
  ; unit conversion
 ; o_c = o*u_ens
  ; copy metadata
 ; copy_VarMeta(o,o_c)
  ; 5-year annual mean
 ; o_am = dim_avg_n_Wrap(o_c,0)
  ; area-weighted annual mean
 ; o_am_area = o_am * grid_area_ens_m
 ; copy_VarMeta(o_am,o_am_area)
  ; append to ensemble collection
 ; O(i-1,:,:) = o_am_area
  ;O(i-1,:,:) = o_am ; unweighted
 ;end do

 f_model = addfile("../outputdata/outputdata_GPP_forSVD_100.nc","r")
 O = f_model->O
 printVarSummary(O)
 
 ; ensemble mean w/o obs
 O_em = dim_avg_n_Wrap(O,0)
 printVarSummary(O_em)

 ; obs is last ensemble member
 o_md_dims = dimsizes(O)
 O_mod = new((/o_md_dims(0)+1,o_md_dims(1),o_md_dims(2)/),"float")
 O_mod(:99,:,:) = O
 O_mod(100,:,:) = v_obs_am_area
 printVarSummary(O_mod)

 ; ensemble mean w/obs
 O_em_alt = dim_avg_n_Wrap(O_mod,0)
 printVarSummary(O_em_alt)

;************************************************
; prepare input to SVD
;************************************************
 ; subtract "ensemble" mean from obs
 ; where ensemble mean is LHC ensemble mean
 v_obs_anom = v_obs_am_area ; metadata
 ;v_obs_anom = v_obs_am_area - O_em ; no obs
 v_obs_anom = v_obs_am_area - O_em_alt ; with obs

 ; update units
 ;v_obs_c_area_am@units = "PgC/yr"
 ;printVarSummary(v_obs_c_area_am)
 v_obs_anom@units = "PgC/yr"
 printVarSummary(v_obs_anom)
 
 ; test plot

 ;wks = gsn_open_wks("x11","test")
 ;wks = gsn_open_wks("pdf","obs_GPP_4x5_anom_forSVD")
 wks = gsn_open_wks("pdf","obs_GPP_4x5_anom_forSVD_withobs")

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
 ;res@tiMainString = "5-year annual mean obs, area-weighted, anomaly from LHC100 ensemble mean no obs"
 res@tiMainString = "5-year annual mean obs, area-weighted, anomaly from LHC100 ensemble mean with obs"
 res@tiMainFontHeightF   = 0.015

 ; set the contour bounds
 res@cnLevelSelectionMode = "ManualLevels"
 res@cnMinLevelValF  = -0.5
 res@cnMaxLevelValF  = 0.2
 res@cnLevelSpacingF = 0.1

 ; single plot
 plot = gsn_csm_contour_map(wks,v_obs_anom,res)

;************************************************
; write output file
;************************************************
 ; writing as nc (preferred)
 ncdf = addfile("obs_GPP_4x5_anom_forSVD.nc", "c")
 ; define time as an unlimited dimension (optional?)
 ;filedimdef(ncdf,"time",-1,True) 
 ncdf->GPP = v_obs_anom
 ncdf->datamask = datamask_obs

end
