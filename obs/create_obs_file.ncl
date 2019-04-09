; create_obs_file.ncl
; This program samples and creates an obs file from ILAMB data
; Katie Dagon, April 2019
;**************************************
begin
 
 ; Set some fixed dimensions
 start_year = 2000
 end_year = 2004
 nyrs = end_year - start_year + 1
 start_year_ind = start_year - 1982
 start_month = start_year_ind*12
 print(start_month)
 end_month = start_month + nyrs*12 - 1
 print(end_month)

 ; Data is monthly Jan, 1982 through Dec, 2008 (324 time samples, 27 years)
 ;cdf_filename = "/project/tss/bgc01/oleson/photosynthesis/FLUXNETGPP/MR_GPP0.9x1.25.nc"
 cdf_filename = "/glade/p/cesm/lmwg_dev/oleson/ILAMB/ILAMB_all/DATA/gpp/FLUXNET-MTE/gpp_0.5x0.5.nc"
 in_file = addfile(cdf_filename,"r")
 ;landmask = in_file->LANDMASK
 lat = in_file->lat
 nlat = dimsizes(lat)
 lon = in_file->lon
 nlon = dimsizes(lon)
 gpp = in_file->gpp
 printVarSummary(gpp)

 ; Sample obs
 ;gpp_sample = new((/nyrs,nlat,nlon/),"float") 
 gpp_sample_monthly = gpp(start_month:end_month,:,:)
 printVarSummary(gpp_sample_monthly)
 ;gpp_sample = month_to_annual(gpp_sample_monthly,1)
 gpp_sample = gpp_sample_monthly
 printVarSummary(gpp_sample)

 ; Metadata
 gpp_sample@_FillValue = -999.
 gpp_sample@long_name = "Gross Primary Production"
 gpp_sample@units = "kgC/m2/s"
 ;years = ispan(1,nyrs,1)
 ;gpp_sample!0 = "year"
 ;gpp_sample&year = years
 months = ispan(1,nyrs*12,1)
 gpp_sample!0 = "month"
 gpp_sample&month = months 
 gpp_sample!1 = "lat"
 gpp_sample&lat = lat
 gpp_sample!2 = "lon"
 gpp_sample&lon = lon
 printVarSummary(gpp_sample)

 ; Create a datamask for this resolution
 datamask = where(ismissing(gpp(0,:,:)),0.,1.)
 datamask!0 = "lat"                                                                                                                          
 datamask!1 = "lon"                                                                                                                          
 datamask&lat = lat                                                                                                                      
 datamask&lon = lon

 ; Replace this section with code that creates gridcell areas
 ; from /project/bgc01/oleson/photosynthesis/FLUXNETGPP/create_climo.ncl
 ;
 ; Get area from a 0.9x1.25 simulation
 cdf_filename = "/home/kdagon/CLM5hydro_LHC/MR_GPP_0.9x1.25_ANN_climo.nc"
 in_file = addfile(cdf_filename,"r")
 area = in_file->area
 ;

 ; Output file
 cdf_filename = "MR_GPP_0.5x0.5_2000_2004.nc"
 ;system("/bin/rm -f " + cdf_filename)
 out_file = addfile(cdf_filename,"c")
 setfileoption(out_file,"DefineMode",True)
 fAtt = True
 fAtt@title = "Sampled monthly means (2000-2004) of gross primary production from Beer, C., et al. (2010), Science, 329, 834-838" 
 fAtt@source = "/glade/p/cesm/lmwg_dev/oleson/ILAMB/ILAMB_all/DATA/gpp/FLUXNET-MTE/gpp_0.5x0.5.nc"
 fAtt@date = systemfunc("date")
 fileattdef( out_file, fAtt)

 ;dimNames = (/"year", "lat", "lon"/)                                                                                                      
 dimNames = (/"month", "lat", "lon"/)
 ;dimSizes = (/nyrs, nlat, nlon/)
 dimSizes = (/nyrs*12, nlat, nlon/)
 dimUnlim = (/False, False, False/)
 filedimdef(out_file,dimNames,dimSizes,dimUnlim)

 ;filevardef(out_file, "landmask", typeof(landmask), getvardims(landmask))
 filevardef(out_file, "lat", typeof(lat), getvardims(lat))
 filevardef(out_file, "lon", typeof(lon), getvardims(lon))
 filevardef(out_file, "GPP", typeof(gpp_sample), getvardims(gpp_sample))
 filevardef(out_file, "datamask", typeof(datamask), getvardims(datamask))
 filevardef(out_file, "area", typeof(area), getvardims(area))

 ;filevarattdef(out_file, "landmask", landmask)
 filevarattdef(out_file, "lat", lat)
 filevarattdef(out_file, "lon", lon)
 filevarattdef(out_file, "GPP", gpp_sample)
 filevarattdef(out_file, "datamask", datamask)
 filevarattdef(out_file, "area", area)

 setfileoption(out_file,"DefineMode",False)

 ;out_file->landmask = (/landmask/)
 out_file->lat = (/lat/)
 out_file->lon = (/lon/)
 out_file->GPP = (/gpp_sample/)
 out_file->datamask = (/datamask/)
 out_file->area = (/area/)

end
