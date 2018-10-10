; ***********************************************
; process_outputdata_SVD.ncl
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set number of simulations/ens members
 niter = 100
 ;niter = 10
 ;niter = 20

 ; set output path
 path = "/glade/scratch/kdagon/archive/"

 ; set variable(s) of interest
 var = "FPSN"
 ;var = "TSA"
 ;var = "QFLX_EVAP_TOT"

 ; unit conversion (if needed)
 u = (12 * (3600 * 24 * 365) / 1000000) * 10^-15 ; umol /m2 /s to Pg/m2/year
 ;u = 86400 ; mm/s to mm/day

 ; establish output array as U-vector from SVD
 ; dims = 10xniter
 ;out = new((/10,niter/),float)

 ; get sample file for metadata
 f_md = addfile("/glade/scratch/kdagon/archive/hydro_ensemble_LHC_1/lnd/hist/hydro_ensemble_LHC_1.clm2.h0.0016-01.nc","r")
 o_md = f_md->$var$
 o_md_dims = dimsizes(o_md)

 ; get gridcell areas
 grid_area = f_md->area 
 grid_area_m = grid_area ; metadata
 grid_area_m = grid_area*10^6 ; km^2 to m^2

 ; set up storage matrix
 O = new((/niter,o_md_dims(1),o_md_dims(2)/),"float")
 O!0 = "ens"
 O!1 = "lat"
 O&lat = o_md&lat
 O!2 = "lon"
 O&lon = o_md&lon

 do i=1,niter
  y=tostring(i)
  print(y)
  strs = (/path,"hydro_ensemble_LHC_",y,"/lnd/hist/*{001[6-9],20-}*"/)
  str_c = str_concat(strs)
  fils = systemfunc("ls "+str_c)
  f = addfiles(fils,"r")
  o = f[:]->$var$
  ; unit conversion
  o_c = o*u
  ; copy metadata
  copy_VarMeta(o,o_c)
  ; 5-year annual mean
  o_am = dim_avg_n_Wrap(o_c,0)
  ; area-weighted annual mean
  o_am_area = o_am * grid_area_m
  copy_VarMeta(o_am,o_am_area)
  ; append to ensemble collection
  O(i-1,:,:) = o_am_area
  ;O(i-1,:,:) = o_am ; unweighted
 end do

 ; ensemble mean
 O_em = dim_avg_n_Wrap(O,0)
 
 ; input to SVD - subtract ensemble mean
 X = new((/niter,o_md_dims(1),o_md_dims(2)/),"float")
 copy_VarMeta(O,X)
 do i=1,niter
  X(i-1,:,:) = O(i-1,:,:)-O_em
 end do
 X@units = "PgC/yr"
 ;printVarSummary(X)
 
 ; test plots
 plot = new(4,graphic)

 wks = gsn_open_wks("x11","test")
 ;wks = gsn_open_wks("pdf","hydro_ensemble_LHC_GPP_forSVD")

 res = True
 res@gsnDraw = False
 res@gsnFrame = False
 res@cnFillOn = True
 res@cnLinesOn           = False           ; turn off contour lines
 res@cnLineLabelsOn      = False           ; turn off contour line labels
 res@cnFillPalette       = "MPL_YlOrRd"
 res@cnSpanFillPalette   = True            ; Fully span color palette
 res@lbLabelBarOn        = False
 res@gsnLeftString = "GPP"
 res@gsnRightString = "PgC/yr"

 ; check some random ensemble members
 res@gsnCenterString = "Member 9 Anomaly"
 plot(0) = gsn_csm_contour_map(wks,X(8,:,:),res)
 res@gsnCenterString = "Member 17 Anomaly"
 plot(1) = gsn_csm_contour_map(wks,X(16,:,:),res)
 res@gsnCenterString = "Member 77 Anomaly"
 plot(2) = gsn_csm_contour_map(wks,X(76,:,:),res)
 res@gsnCenterString = "Member 99 Anomaly"
 plot(3) = gsn_csm_contour_map(wks,X(98,:,:),res)

 resP                     = True                ; modify the panel plot
 resP@gsnMaximize         = True
 resP@gsnPanelMainString = "5-year annual mean, area-weighted, anomaly from ensemble mean"
 resP@gsnPanelLabelBar    = True                ; add common colorbar
 resP@lbBoxLinesOn = False ; no boxes
 gsn_panel(wks,plot,(/2,2/),resP) 

 ; trying to remove missing values (non-land) 
 ;X_mod = X ; metadata
 ;X_mod = mask(X,X.eq.X@_FillValue,0)
 ;X_mod = where(ismissing(X),0,X) 
 ;printVarSummary(X_mod)
 ;plot = gsn_csm_contour_map(wks,X_mod(:,:,1),res)
 ;nmsg = num(ismissing(X_mod))
 ;print(nmsg)

 ; SVD calculations - not working with land-only data
 ;ncol = o_md_dims(1)*o_md_dims(2) ; number of colums = nlat*nlon
 ;nrow = niter ; number of rows = niter (ensemble members)
 ; pre-allocate space for returned arrays
 ;u = new((/ncol,nrow/),typeof(X))
 ;v = new((/ncol,ncol/),typeof(X))
 ;s = svd_lapack(X_mod,"S","S",0,u,v)

 ;printVarSummary(u)
 ;printVarSummary(v)
 ;print(s)
;************************************************
; write output file
;************************************************
 ; writing as csv (not preferred)
 ;alist = [/X/]
 ;write_table("outputdata_GPP_SVD.csv", "w", alist, "%f")

 ; writing as nc (preferred)
 ncdf = addfile("outputdata/outputdata_GPP_forSVD_100.nc", "c")
 ;ncdf = addfile("outputdata/outputdata_ET_forSVD_100.nc", "c")
 ; define time as an unlimited dimension (optional?)
 ;filedimdef(ncdf,"time",-1,True) 
 ncdf->X = X
 

end