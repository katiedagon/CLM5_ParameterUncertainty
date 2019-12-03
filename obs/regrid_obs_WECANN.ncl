;---------------------------------------------
; regrid_obs_WECANN.ncl
;
; Interpolating from one grid to another using bilinear interpolation
;---------------------------------------------
;

begin
;---Read in obs netCDF file
  in = addfile("WECANN_v1.0.nc","r")

;---Read in data
  ; assign FillValue
  ;gpp@_FillValue = 9.96921e+36
  ;print(num(ismissing(gpp(0,:,:))))
  ;print(num(ismissing(lhf(0,:,:))))

  ; correct coordinate variables
  lons = in->Longitude
  ;printVarSummary(lons)
  lon = lons(:,0)
  lon!0 = "lon"
  lon&lon = lon
  lon@units = "degrees_east"
  ; pivot lons so they run 0-360
  ;lon = lonPivot(lon,360)
  ;lon = lonFlip(lono)
  ;printVarSummary(lon)
  lats = in->Latitude
  ; reverse lats so they are monotonically increasing
  lat = lats(0,::-1)
  lat!0 = "lat"
  lat&lat = lat
  lat@units = "degrees_north"
  ;printVarSummary(lat)

  gpp_in = in->GPP
  ; re-order lat/lon and reverse lats
  gpp = gpp_in(t|:,lat|::-1,lon|:)
  gpp!1 = "lat"
  gpp!2 = "lon"
  gpp&lat = lat
  gpp&lon = lon
  ; lon flip
  gpp = lonFlip(gpp)
  ; deal with -nan
  if (any(isnan_ieee(gpp))) then
    gpp@_FillValue = 9.96921e+36
    replace_ieeenan (gpp, gpp@_FillValue, 0)
  end if
  ;printVarSummary(gpp)  
  ;print(gpp&lon)

  lhf_in = in->LE
  ; re-order lat/lon and reverse lats
  lhf = lhf_in(t|:,lat|::-1,lon|:)
  lhf!1 = "lat"
  lhf!2 = "lon"
  lhf&lat = lat
  lhf&lon = lon
  ; lon flip
  lhf = lonFlip(lhf)
  ; deal with -nan
  if (any(isnan_ieee(lhf))) then
    lhf@_FillValue = 9.96921e+36
    replace_ieeenan (lhf, lhf@_FillValue, 0)
  end if
  ;printVarSummary(lhf)  

  ;area = in->area
  ;datamask = in->datamask

;---Read in target resolution file
  in_res = addfile("../outputdata/outputdata_GPP_forSVD_100.nc","r")
  lon_res = in_res->lon
  ;print(lon_res)
  lat_res = in_res->lat
  ;print(lat_res)

;---------------------------------------------
; interpolate to new grid
;---------------------------------------------

 ; lon flip for coordinate
  ;lonF = lonFlip(lon)
  lonF = gpp&lon
  ;print(lonF)

  newgpp = linint2_Wrap(lonF,lat,gpp,True,lon_res,lat_res,0)
  newgpp!1   ="lat"
  newgpp!2   = "lon"
  newgpp&lat = lat_res
  newgpp&lon = lon_res
  ;printVarSummary(newgpp)

  newlhf = linint2_Wrap(lonF,lat,lhf,True,lon_res,lat_res,0)
  newlhf!1 = "lat"
  newlhf!2 = "lon"
  newlhf&lat = lat_res
  newlhf&lon = lon_res
  ;printVarSummary(newlhf)

  ; create datamask based on regridded obs data
  ; with the CAVEAT that this is based on the first time record
  ; but the datamask should not vary over time (or does it?)
  ;print(num(ismissing(newgpp(0,:,:))))
  ;print(num(ismissing(newgpp(1,:,:))))
  ;print(num(ismissing(newgpp(59,:,:))))
  newdatamG = where(ismissing(newgpp(0,:,:)),0.,1.)
  newdatamG!0 = "lat"
  newdatamG!1 = "lon"
  newdatamG&lat = lat_res
  newdatamG&lon = lon_res                                                                                                                      
  ;printVarSummary(newdatamG)

  ;print(num(ismissing(newlhf(0,:,:))))
  ;print(num(ismissing(newlhf(1,:,:))))
  ;print(num(ismissing(newlhf(59,:,:))))
  newdatamL = where(ismissing(newlhf(0,:,:)),0.,1.)
  newdatamL!0 = "lat"
  newdatamL!1 = "lon"
  newdatamL&lat = lat_res
  newdatamL&lon = lon_res
  ;printVarSummary(newdatamL)

;---------------------------------------------
; test plots
;---------------------------------------------
  ;wks = gsn_open_wks("x11","test")
  wks = gsn_open_wks("pdf","WECANN_1x1_4x5_2007_2015")  

  res                     = True               ; plot mods desired
  res@cnFillOn            = True
  res@cnLinesOn           = False              ; no contour lines
  res@cnLineLabelsOn      = False              ; no line labels
  res@gsnDraw  = False
  res@gsnFrame = False

  plot = new(4, graphic)

  ; plot native resolution (time average)
  res@gsnLeftString = "GPP (1x1)"
  plot(0) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(gpp,0),res)
  res@gsnLeftString = "LHF (1x1)"
  plot(1) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(lhf,0),res)

  ; plot new resolution (time average)
  res@gsnLeftString = "GPP (4x5)"
  plot(2) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(newgpp,0),res)
  res@gsnLeftString = "LHF (4x5)"
  plot(3) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(newlhf,0),res)
  
  ;plot = gsn_csm_contour_map(wks,newdatamG,res)
  ;plot = gsn_csm_contour_map(wks,newdatamL,res)

  ; plot panel
  gsn_panel(wks,plot,(/2,2/),False) 

  ; save out regridded output
  ncdf = addfile("WECANN_GPP_LHF_4x5_2007_2015.nc","c")

  ncdf->GPP = newgpp
  ncdf->LHF = newlhf
  ncdf->datamaskG = newdatamG
  ncdf->datamaskL = newdatamL
end
