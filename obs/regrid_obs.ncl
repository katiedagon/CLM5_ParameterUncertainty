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
  ; FLUXNET-MTE 1 degree, subset years, annual means
  ;in = addfile("MR_GPP_0.9x1.25_2000_2004_amean.nc","r")

  ; FLUXNET-MTE 1 degree, subset years
  ;in = addfile("MR_GPP_0.9x1.25_2000_2004.nc","r")
  ;in = addfile("MR_LHF_0.9x1.25_2000_2004.nc","r")

  ; FLUXNET-MTE 1/2 degree, subset years
  ;in = addfile("MR_GPP_0.5x0.5_2000_2004.nc","r")

  ; FLUXNET-MTE 1 degree, all years
  ;in = addfile("MR_GPP0.9x1.25.nc","r")
  ;in = addfile("MR_LHF0.9x1.25.nc","r")

  ; GBAF 1/2 degree, all years (1982-2008)
  ; NOTE: longitudes ordered -180/+180
  ;in = addfile("/glade/p/cesm/lmwg/diag/ILAMB/DATA/sh/GBAF/sh_0.5x0.5.nc","r")

  ; GRACE 1/2 degree, all years (2002-2014)
  ; NOTE: longitudes ordered -180/+180
  in = addfile("/glade/p/cesm/lmwg/diag/ILAMB/DATA/twsa/GRACE/twsa_0.5x0.5.nc","r")

;---Read in data
  ;gpp = in->GPP
  ;print(num(ismissing(gpp(0,:,:))))
  ;printVarSummary(gpp)
  ;lhf = in->LHF
  ;print(num(ismissing(lhf(0,:,:))))
  ;sh = in->sh
  ;printVarSummary(sh)
  twsa = in->twsa
  ;printVarSummary(twsa)

  lon = in->lon
  ;lon = in->LON
  lat = in->lat
  ;lat = in->LAT
  ;area = in->area
  ;datamask = in->datamask

;---Coordinate variables for native resolution
;---only needed when regridding original files with all years
  ;gpp!1 = "lat"
  ;gpp!2 = "lon"
  ;gpp&lat = lat
  ;gpp&lon = lon
  ;lhf!1 = "lat"
  ;lhf!2 = "lon"
  ;lhf&lat = lat
  ;lhf&lon = lon
  ;sh!1 = "lat"
  ;sh!2 = "lon"
  ;sh&lat = lat
  ;sh&lon = lon

;---Read in area from processing file
  ;in_area = addfile("MR_GPP_0.9x1.25_ALLMONS_climo.nc","r")
  ;area = in_area->area

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

  ;newgpp = linint2_Wrap(lon,lat,gpp,True,lon_res,lat_res,0)
  ;newgpp!1   ="lat"
  ;newgpp!2   = "lon"
  ;newgpp&lat = lat_res
  ;newgpp&lon = lon_res
  ;printVarSummary(newgpp)

  ;newlhf = linint2_Wrap(lon,lat,lhf,True,lon_res,lat_res,0)
  ;newlhf!1 = "lat"
  ;newlhf!2 = "lon"
  ;newlhf&lat = lat_res
  ;newlhf&lon = lon_res
  ;printVarSummary(newlhf)

  ; GBAF sh lons are -180/+180, need to lonFlip
  ;sh_flip = lonFlip(sh)
  ;printVarSummary(sh_flip)

  ; then just use flipped variable coordinates for interpolation
  ;lat_flip = sh_flip&lat
  ;lon_flip = sh_flip&lon

  ;newsh = linint2_Wrap(lon_flip,lat_flip,sh_flip,True,lon_res,lat_res,0)
  ;newsh!1 = "lat"                                                                                                                  
  ;newsh!2 = "lon"                                                                                                                  
  ;newsh&lat = lat_res                                                                                                              
  ;newsh&lon = lon_res
  ;printVarSummary(newsh) 

  ; GRACE twsa lons are -180/+180, need to lonFlip
  twsa_flip = lonFlip(twsa)
  ;printVarSummary(twsa_flip)

  ; then just use flipped variable coordinates for interpolation
  lat_flip = twsa_flip&lat
  lon_flip = twsa_flip&lon
 
  newtwsa = linint2_Wrap(lon_flip,lat_flip,twsa_flip,True,lon_res,lat_res,0)
  newtwsa!1 = "lat"                                                                                                                    
  newtwsa!2 = "lon"                                                                                                                    
  newtwsa&lat = lat_res                                                                                                                
  newtwsa&lon = lon_res                                                                                                                
  ;printVarSummary(newtwsa)

  ; pull out specific years
  ;obs_start_year = 1982
  obs_start_year = 2002
  ;start_year = 2000
  start_year = 2002
  end_year = 2004
  nyrs = end_year - start_year + 1
  start_year_ind = start_year - obs_start_year
  start_month = start_year_ind*12
  print(start_month)
  end_month = start_month + nyrs*12 - 1
  print(end_month) 
 
  ;newsh_subset = newsh(start_month:end_month,:,:)
  months = ispan(1,nyrs*12,1)
  ;newsh_subset!0 = "month"
  ;newsh_subset&month = months
  ;printVarSummary(newsh_subset)

  newtwsa_subset = newtwsa(start_month:end_month,:,:)
  newtwsa_subset!0 = "month"  
  newtwsa_subset&month = months 
  ;printVarSummary(newtwsa_subset)

  ; regridding the area seems highly suspect/WRONG - does not account for changing grid size
  ;newarea = linint2_Wrap(lon,lat,area,True,lon_res,lat_res,0)
  ;newarea!0   ="lat"
  ;newarea!1   = "lon"
  ;newarea&lat = lat_res
  ;newarea&lon = lon_res
  ;printVarSummary(newarea)
 
  ; regridding the datamask creates values between 0 and 1
  ;newdatam = linint2_Wrap(lon,lat,datamask,True,lon_res,lat_res,0)
  ; instead, create datamask based on regridded obs data
  ; with the CAVEAT that this is based on the first time record
  ; but the datamask should not vary over time (or does it?)
  ;print(num(ismissing(newgpp(0,:,:))))
  ;print(num(ismissing(newgpp(1,:,:))))
  ;print(num(ismissing(newgpp(59,:,:))))
  ;newdatam = where(ismissing(newgpp(0,:,:)),0.,1.)

  ;print(num(ismissing(newlhf(0,:,:))))
  ;print(num(ismissing(newlhf(1,:,:))))
  ;print(num(ismissing(newlhf(59,:,:))))
  ;newdatam = where(ismissing(newlhf(0,:,:)),0.,1.)
  ;printVarSummary(newdatam)

  ;print(num(ismissing(newsh_subset(0,:,:))))
  ;print(num(ismissing(newsh_subset(1,:,:))))
  ;print(num(ismissing(newsh_subset(59,:,:))))
  ;newdatam = where(ismissing(newsh_subset(0,:,:)),0.,1.) 

  ; note that 2002 Jan,Feb,Mar,May,June of GRACE TWSA are missing
  ; and June of year 2003
  ; so use 4th month to create datamask
  ; this also means time comparison with full model years isn't really fair...
  print(num(ismissing(newtwsa_subset(3,:,:))))                                                                                         
  print(num(ismissing(newtwsa_subset(4,:,:))))                                                                                         
  print(num(ismissing(newtwsa_subset(35,:,:))))
  newdatam = where(ismissing(newtwsa_subset(3,:,:)),0.,1.) 

  newdatam!0 = "lat"
  newdatam!1 = "lon"
  newdatam&lat = lat_res
  newdatam&lon = lon_res
  ;printVarSummary(newdatam)

  ; this function doesn't like missing values
  ;newgpp2 = g2gsh(gpp,(/46,72/),0)
  ;newgpp2!1   ="lat"
  ;newgpp2!2   = "lon"
  ;newgpp2&lat = lat_res
  ;newgpp2&lon = lon_res
  ;printVarSummary(newgpp2)

  wks = gsn_open_wks("x11","test")
  ;wks = gsn_open_wks("pdf","FLUXNET_GPP_0.9x1.25_4x5_2000_2004") 
  ;wks = gsn_open_wks("pdf","FLUXNET_LHF_0.9x1.25_4x5_2000_2004") 
  ;wks = gsn_open_wks("pdf","GBAF_SH_0.5x0.5_4x5_2000_2004")
  ;wks = gsn_open_wks("pdf", "GRACE_TWSA_0.5x0.5_4x5_2002_2004")

  res                     = True               ; plot mods desired
  res@cnFillOn            = True
  res@cnLinesOn           = False              ; no contour lines
  res@cnLineLabelsOn      = False              ; no line labels
  res@gsnDraw  = False
  res@gsnFrame = False

  ; LHF/SH bounds
  res@cnLevelSelectionMode = "ManualLevels"	; manually set the contour levels with the following 3 resources
  ;res@cnMinLevelValF  = 0.			; set the minimum contour level
  ;res@cnMaxLevelValF  = 100.			; set the maximum contour level
  ;res@cnLevelSpacingF = 10.

  ; TWSA bounds
  res@cnMinLevelValF  = -200.
  res@cnMaxLevelValF  = 200.
  res@cnLevelSpacingF = 40.

  plot = new(2, graphic)

  ; plot native resolution (time average)
  ;res@gsnLeftString = "GPP (0.9x1.25)"
  ;plot(0) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(gpp,0),res)
  ;res@gsnLeftString = "LHF (0.9x1.25)"
  ;plot(0) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(lhf,0),res)
  ;res@gsnLeftString = "SH (0.5x0.5)"
  ;plot(0) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(sh(start_month:end_month,:,:),0),res)
  res@gsnLeftString = "TWSA (0.5x0.5)"                                                                                                 
  plot(0) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(twsa(start_month:end_month,:,:),0),res)

  ;plot = gsn_csm_contour_map(wks,area,res)
  ;plot = gsn_csm_contour_map(wks,datamask,res)

  ; plot new resolution (time avg)
  ;res@gsnLeftString = "GPP (4x5)"
  ;plot(1) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(newgpp,0),res)
  ;res@gsnLeftString = "LHF (4x5)"
  ;plot(1) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(newlhf,0),res)
  ;res@gsnLeftString = "SH (4x5)"
  ;plot(1) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(newsh_subset,0),res)
  res@gsnLeftString = "TWSA (4x5)"
  plot(1) = gsn_csm_contour_map(wks,dim_avg_n_Wrap(newtwsa_subset,0),res)

  ;plot = gsn_csm_contour_map(wks,newarea,res)
  ;plot = gsn_csm_contour_map(wks,newdatam,res)

  ; plot panel
  gsn_panel(wks,plot,(/2,1/),False)

  ; save out regridded output
  ;ncdf = addfile("MR_GPP_4x5_2000_2004_amean.nc","c")
  ;ncdf = addfile("MR_GPP_4x5_2000_2004.nc","c")
  ;ncdf = addfile("MR_LHF_4x5_2000_2004.nc","c")
  ;ncdf = addfile("MR_GPP_4x5_2000_2004_v2.nc","c")
  ;ncdf = addfile("MR_GPP_4x5.nc","c")
  ;ncdf = addfile("MR_LHF_4x5.nc","c")
  ;ncdf = addfile("GBAF_SH_4x5_2000_2004.nc","c")
  ncdf = addfile("GRACE_TWSA_4x5_2002_2004.nc","c")

  ;ncdf->GPP = newgpp
  ;ncdf->LHF = newlhf
  ;ncdf->FSH = newsh_subset
  ncdf->TWSA = newtwsa_subset
  ;ncdf->area = newarea
  ncdf->datamask = newdatam
end
