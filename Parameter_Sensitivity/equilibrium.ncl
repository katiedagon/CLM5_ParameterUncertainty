;************************************************
; equilibrium.ncl
;
; Test equilibration time for soil moisture
;************************************************
begin
;************************************************
; read in data
;************************************************
 fils = systemfunc("ls /glade/scratch/kdagon/archive/hydro_ensemble_reruns_default_params/lnd/hist/*.nc")
 q    = addfiles(fils,"r")
 
 ; read in all the files and join the timeseries together
 SOILLIQ = q[:]->SOILLIQ 
 SOILICE = q[:]->SOILICE
 SOILWATER = SOILLIQ ; metadata
 SOILWATER = SOILLIQ+SOILICE ; sum of liquid and ice
 ;printVarSummary(SOILWATER)
 SOILWATER_TOT = dim_sum_n_Wrap(SOILWATER,1) ; sum over depth
 ;printVarSummary(SOILWATER_TOT)

 TSOI = q[:]->TSOI ; 4D variable (time x lev x lat x lon) 
 ;printVarSummary(TSOI(:,{:2},:,:))
 TSOI_dmean = dim_avg_n_Wrap(TSOI(:,{:2},:,:),1) ; average over depth to 2m

 ; get fixed variable output
 landf = q[0]->landfrac
 area = q[0]->area
 
 ; wgt variable for global mean
 arwt = area
 arwt = area * landf
 arwt = arwt/max(arwt)

;************************************************
; calculate global, annual mean
;************************************************
 SOILWATER_gm_am = month_to_annual(wgt_areaave2(SOILWATER_TOT,arwt,0),1)
 TSOI_gm_am = month_to_annual(wgt_areaave2(TSOI_dmean,arwt,0),1)

;************************************************
; plot timeseries
;************************************************
 ;wks_panel = gsn_open_wks("x11","test")
 wks_panel  = gsn_open_wks("pdf","soil_moisture_spinup")

 res                     = True
 res@gsnMaximize         = True          ; maximize plot size

 res@tiXAxisString     = "~F21~Years from Start"                                                                                              
 res@tiYAxisString     = "~F21~Total Column Soil Moisture (kg m~S~-2~N~)"  
 res@xyLineThicknessF = 3
 
 time = ispan(1,30,1)
 res@trXMinF = 1

 plot = gsn_csm_xy(wks_panel,time,SOILWATER_gm_am,res) 

 ; optional plot of soil temperature timeseries
 ;plot = gsn_csm_xy(wks_panel,time,TSOI_gm_am,res)

end
