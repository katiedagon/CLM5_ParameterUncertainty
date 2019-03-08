;---------------------------------------------
; regrid_obs.ncl
;
; Interpolating from one grid to another using bilinear interpolation
;---------------------------------------------
;
; This file still has to be loaded manually
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"  

begin
;---Read in obs netCDF file
  ;in = addfile("MR_GPP_0.9x1.25_2000_2004_amean.nc","r")
  in = addfile("MR_GPP_0.9x1.25_2000_2004.nc","r")

;---Read in data
  gpp = in->GPP
  printVarSummary(gpp)
  lon = in->lon
  lat = in->lat
  area = in->area
  datamask = in->datamask

;---Read in target resolution file
  in_res = addfile("../outputdata/outputdata_GPP_forSVD_100.nc","r")
  lon_res = in_res->lon
  lat_res = in_res->lat

;---------------------------------------------
; interpolate to new grid
;---------------------------------------------

  ;newlat = fspan(-60.,60,24)
  ;newlon = fspan(0.,355.,72)
  ;newlat@units = "degrees_north"
  ;newlon@units = "degrees_east"

  newgpp = linint2_Wrap(lon,lat,gpp,True,lon_res,lat_res,0)

  newgpp!1   ="lat"
  newgpp!2   = "lon"
  newgpp&lat = lat_res
  newgpp&lon = lon_res

  printVarSummary(newgpp)

  newarea = linint2_Wrap(lon,lat,area,True,lon_res,lat_res,0)
  newarea!0   ="lat"
  newarea!1   = "lon"
  newarea&lat = lat_res
  newarea&lon = lon_res

  printVarSummary(newarea)
 
  newdatam = linint2_Wrap(lon,lat,datamask,True,lon_res,lat_res,0)
  newdatam!0 = "lat"
  newdatam!1 = "lon"
  newdatam&lat = lat_res
  newdatam&lon = lon_res

  printVarSummary(newdatam)

  ; this function doesn't like missing values
  ;newgpp2 = g2gsh(gpp,(/46,72/),0)
  ;newgpp2!1   ="lat"
  ;newgpp2!2   = "lon"
  ;newgpp2&lat = lat_res
  ;newgpp2&lon = lon_res
  ;printVarSummary(newgpp2)

  wks = gsn_open_wks("x11","test")
  
  res                     = True               ; plot mods desired
  res@cnFillOn            = True
  res@cnLinesOn           = False              ; no contour lines
  res@cnLineLabelsOn      = False              ; no line labels

  ; plot native resolution
  plot = gsn_csm_contour_map(wks,gpp(0,:,:),res)

  ; plot new resolution
  plot = gsn_csm_contour_map(wks,newgpp(0,:,:),res)
  ;plot = gsn_csm_contour_map(wks,newgpp2(0,:,:),res)

  ; save out regridded output
  ;ncdf = addfile("MR_GPP_4x5_2000_2004_amean.nc","c")
  ncdf = addfile("MR_GPP_4x5_2000_2004.nc","c")
  ncdf->GPP = newgpp
  ncdf->area = newarea
  ncdf->datamask = newdatam
end
