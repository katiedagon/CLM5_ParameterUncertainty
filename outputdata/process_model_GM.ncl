; ***********************************************
; process_model_GM.ncl
; ***********************************************
;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set param set number
 ;pset = "006"

 ; set model path
 path = "/glade/scratch/kdagon/archive/"

 ; set variable(s) of interest
 var = "FPSN"

 ; set years
 nyrs = 5

 ; unit conversion (if needed)
 ;u = 365 * 10^-15 ; gC/m2/day to Pg/m2/year
 ;u = 10^6 / (86400 * 12) ; gC/m2/day to umolC/m2/s

 ; get hist files (last 5 years)
 ;strs = (/path,"test_paramset_",pset,"/lnd/hist/*{001[6-9],20-}*"/)
 ;strs = (/path,"hydro_ensemble0/lnd/hist/*{001[6-9],20-}*"/)
 ;strs = (/path,"hydro_ensemble_rootprof_beta_min/lnd/hist/*{001[6-9],20-}*"/)
 ;strs = (/path,"hydro_ensemble_LHC_default_params/lnd/hist/*{001[6-9],20-}*"/)
 strs = (/path,"hydro_ensemble_LHC_v2_1/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs)
 print(str_c)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v = f[:]->$var$

 ; since gaus function does not generate the same lats as model grid
 ; use actual latitudes and cosine weighting for global mean
 strs_lat = str_concat((/path,"hydro_ensemble_LHC_1/lnd/hist/hydro_ensemble_LHC_1.clm2.h0.0016-01.nc"/))
 print(strs_lat)
 f_lat = addfile(strs_lat,"r")
 lat = f_lat->lat
 rad = 4.0*atan(1.0)/180.0
 clat = cos(lat*rad)

 ; get landfrac & area
 landf = f_lat->landfrac
 ;printVarSummary(landf)
 area = f_lat->area

 ; wgt variable
 arwt = area
 arwt = area * landf
 arwt = arwt/max(arwt)
 printVarSummary(arwt)
 
 ; global mean calculation
 ; unit conversion, copy metadata
 ;v_c = v*u 
 ;copy_VarMeta(v,v_c)
 ; global mean, annual mean
 v_gm_am = dim_avg_n_Wrap(wgt_areaave_Wrap(v,clat,1,0),0)
 print(v_gm_am)

 v_gm_am_alt = dim_avg_n_Wrap(wgt_areaave2(v,arwt,0),0)
 print(v_gm_am_alt)

 arwt2 = area
 arwt2 = area/max(area)
 v_gm_am_alt2 = dim_avg_n_Wrap(wgt_areaave2(v,arwt2,0),0)
 print(v_gm_am_alt2)

 ; annual mean across years
 v_am = dim_avg_n_Wrap(v,0)

 ;wks = gsn_open_wks("x11","test")

 res = True
 ;res@gsnDraw = False
 ;res@gsnFrame = False
 res@cnFillOn = True
 res@cnLinesOn           = False           ; turn off contour lines
 res@cnLineLabelsOn      = False
 res@cnSpanFillPalette   = True            ; Fully span color palette
 ;res@lbLabelBarOn        = False           ; Turn off individual cb's
 res@lbBoxLinesOn = False ; no boxes
 res@gsnLeftString = var
 res@gsnRightString = v@units

 plot = gsn_csm_contour_map(wks,v_am,res) 

end
