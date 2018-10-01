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
 u = 12 * (3600 * 24 * 365) / 1000000 ; umol /m2 /s to g/m2/year

 ; establish output array as U-vector from SVD
 ; dims = 10xniter
 ;out = new((/10,niter/),float)

 ; get sample file for metadata
 f_md = addfile("/glade/scratch/kdagon/archive/hydro_ensemble_LHC_1/lnd/hist/hydro_ensemble_LHC_1.clm2.h0.0016-01.nc","r")
 o_md = f_md->$var$
 o_md_dims = dimsizes(o_md)

 ; get gridcell areas
 grid_area = f_md->area 

 ; set up storage matrix
 O = new((/o_md_dims(1),o_md_dims(2),niter/),"float")
 O!0 = "lat"
 O&lat = o_md&lat
 O!1 = "lon"
 O&lon = o_md&lon
 O!2 = "ens"

 do i=1,niter
  y=tostring(i)
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
  o_am_area = o_am ; metadata
  o_am_area = o_am * (1/grid_area)
  ; append to ensemble collection
  O(:,:,i-1) = o_am_area
  ;O(:,:,i-1) = o_am
 end do

 ; ensemble mean
 O_em = dim_avg_n_Wrap(O,2)
 
 ; input to SVD
 X = O ; metadata
 do i=1,niter
  X(:,:,i-1) = O(:,:,i-1)-O_em
 end do
 ;printVarSummary(X)
 
 ; test plots
 ;wks = gsn_open_wks("x11","test")
 ;res = True
 ;res@cnFillOn = True
 ;plot = gsn_csm_contour_map(wks,X(:,:,0),res)
 ;plot = gsn_csm_contour_map(wks,X(:,:,1),res)
 ;plot = gsn_csm_contour_map(wks,X(:,:,99),res)

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
 ;alist = [/X/]
 ;write_table("outputdata_GPP_SVD.csv", "w", alist, "%f")

 ;ncdf = addfile("outputdata/outputdata_GPP_forSVD.nc", "c")
 ;ncdf = addfile("outputdata/outputdata_GPP_forSVD_20.nc", "c")
 ;ncdf = addfile("outputdata/outputdata_GPP_forSVD_100.nc", "c")
 ;filedimdef(ncdf,"time",-1,True) 
 ;ncdf->X = X
 

end
