; ***********************************************
; mean_bias_maps.ncl
; ***********************************************
;************************************************
begin
;************************************************
; GPP
;************************************************
 ; set obs path
 path = "~/CLM5hydro_LHC/obs/"

 ; set variable(s) of interest
 var = "GPP"

 ; get obs file
 filename = "MR_"+var+"_4x5_2000_2004.nc"
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")
 datamask_obs = f_obs->datamask
 
 v_obs = f_obs->$var$
 ; 5-year  annual mean
 v_obs_amean = dim_avg_n_Wrap(v_obs,0)

 ; set test case name
 test_case = "test_paramset_SVD_v4"

 ; set model output
 var_model = "FPSN"

 ; model unit conversion (if needed)
 u_model = (12 * (60 * 60 * 24) / 10^6) ; umolC/m2/s to gC/m2/day

 path_model = "/glade/scratch/kdagon/archive/"
 strs = (/path_model,test_case,"/lnd/hist/*{001[6-9],20-}*"/) 
 str_c = str_concat(strs)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model = f[:]->$var_model$

 ; mask model to match obs
 v_model_m = mask(v_model,datamask_obs.eq.0,False)
 copy_VarMeta(v_model,v_model_m)
 
 ; unit conversion
 v_model_c = v_model_m*u_model
 copy_VarMeta(v_model_m,v_model_c)

 ; 5-year annual mean
 v_model_amean = dim_avg_n_Wrap(v_model_c,0)

 ; compare obs / model
 diff = v_model_amean - v_obs_amean
 copy_VarMeta(v_model_amean, diff)
 print(var+" CLM Test Case - Observations")
 printMinMax(diff,1)

 ; plot diff
 plotA = new(2,graphic)

 ;wks = gsn_open_wks("x11","test")
 wks = gsn_open_wks("pdf","Mean_bias_maps_GPP_LHF_FSH")

 res = True
 res@gsnDraw = False
 res@gsnFrame = False
 res@cnFillOn = True
 res@cnLinesOn           = False           ; turn off contour lines
 res@cnLineLabelsOn      = False           ; turn off contour line labels
 res@cnFillPalette       = "MPL_BrBG"
 ;res@cnFillPalette       = "MPL_YlOrRd"
 ;res@cnFillPalette       = "amwg256"
 res@cnSpanFillPalette   = True            ; Fully span color palette
 res@lbLabelBarOn        = False           ; Turn off individual cb's
 res@lbBoxLinesOn = False ; no boxes

 res@gsnLeftString = var
 res@gsnRightString = "gC m~S~-2~N~ day~S~-1"

 ; set the contour bounds
 res@cnLevelSelectionMode = "ManualLevels"

 ; GPP diffs (gC/m^2/day)
 res@cnMinLevelValF  = -3
 res@cnMaxLevelValF  = 3
 res@cnLevelSpacingF = 0.3

 res@gsnCenterStringFontHeightF = 0.025
 res@gsnCenterString = "Optimized CLM - Observations"
  
 plotA(0) = gsn_csm_contour_map(wks,diff,res)

 ; compare obs / model default
 strs_def = (/path_model,"hydro_ensemble_LHC_default_params/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs_def)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model_default = f[:]->$var_model$
 v_model_default_m = mask(v_model_default,datamask_obs.eq.0,False)
 copy_VarMeta(v_model_default,v_model_default_m)
 v_model_default_c = v_model_default_m*u_model
 copy_VarMeta(v_model_default_m,v_model_default_c)
 v_model_def_amean = dim_avg_n_Wrap(v_model_default_c,0)

 diff_def = v_model_def_amean - v_obs_amean
 copy_VarMeta(v_model_def_amean, diff_def)
 print(var+" Default CLM - Observations")
 printMinMax(diff_def,1)

 res@gsnCenterString = "Default CLM - Observations"  

 plotA(1) = gsn_csm_contour_map(wks,diff_def,res)

;************************************************
; LHF
;************************************************

 var = "LHF"

 ; get obs file
 filename = "MR_"+var+"_4x5_2000_2004.nc"
 filepath = str_concat((/path,filename/))
 f_obs = addfile(filepath,"r")
 datamask_obs = f_obs->datamask

 v_obs = f_obs->$var$
 ; 5-year  annual mean
 v_obs_amean = dim_avg_n_Wrap(v_obs,0)

 ; set model output
 var_model = "EFLX_LH_TOT"

 path_model = "/glade/scratch/kdagon/archive/"
 strs = (/path_model,test_case,"/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model = f[:]->$var_model$

 ; mask model to match obs
 v_model_m = mask(v_model,datamask_obs.eq.0,False)
 copy_VarMeta(v_model,v_model_m)

 ; 5-year annual mean
 v_model_amean = dim_avg_n_Wrap(v_model_m,0)

 ; compare obs / model
 diff = v_model_amean - v_obs_amean
 copy_VarMeta(v_model_amean, diff)
 print(var+" CLM Test Case - Observations")
 printMinMax(diff,1)

 ; plot diff
 plotB = new(2, graphic)

 res@gsnLeftString = var
 res@gsnRightString = "W m~S~-2"

 ; set the contour bounds
 res@cnLevelSelectionMode = "ManualLevels"

 ; LHF/FSH diffs (W/m^2)
 res@cnMinLevelValF  = -30
 res@cnMaxLevelValF  = 30
 res@cnLevelSpacingF = 3

 ;res@gsnCenterString = "Optimized CLM - Observations"
 res@gsnCenterString = ""

 plotB(0) = gsn_csm_contour_map(wks,diff,res)

 ; compare obs / model default
 strs_def = (/path_model,"hydro_ensemble_LHC_default_params/lnd/hist/*{001[6-9],20-}*"/)
 str_c = str_concat(strs_def)
 fils = systemfunc("ls "+str_c)
 f = addfiles(fils,"r")
 v_model_default = f[:]->$var_model$
 v_model_default_m = mask(v_model_default,datamask_obs.eq.0,False)
 copy_VarMeta(v_model_default,v_model_default_m)                                                                                               
 v_model_def_amean = dim_avg_n_Wrap(v_model_default_m,0)

 diff_def = v_model_def_amean - v_obs_amean
 copy_VarMeta(v_model_def_amean, diff_def)
 print(var+" Default CLM - Observations")
 printMinMax(diff_def,1)

 ;res@gsnCenterString = "Default CLM - Observations"

 plotB(1) = gsn_csm_contour_map(wks,diff_def,res)

;************************************************                                                                                              
; FSH
;************************************************                                                                                              

 var = "FSH"                                                                                                                                  

 ; get obs file                                                                                                                                
 filename = "GBAF_SH_4x5_2000_2004.nc"                                                                                                        
 filepath = str_concat((/path,filename/))                                                                                                      
 f_obs = addfile(filepath,"r")                                                                                                                 
 datamask_obs = f_obs->datamask                                                                                                                

 v_obs = f_obs->$var$                                                                                                                          
 ; 5-year  annual mean                                                                                                                         
 v_obs_amean = dim_avg_n_Wrap(v_obs,0)                                                                                                         

 ; set model output                                                                                                                            
 var_model = "FSH"                                                                                                                            

 path_model = "/glade/scratch/kdagon/archive/"                                                                                                 
 strs = (/path_model,test_case,"/lnd/hist/*{001[6-9],20-}*"/)                                                                                  
 str_c = str_concat(strs)                                                                                                                      
 fils = systemfunc("ls "+str_c)                                                                                                                
 f = addfiles(fils,"r")                                                                                                                        
 v_model = f[:]->$var_model$                                                                                                                   

 ; mask model to match obs                                                                                                                     
 v_model_m = mask(v_model,datamask_obs.eq.0,False)                                                                                             
 copy_VarMeta(v_model,v_model_m)                                                                                                               

 ; 5-year annual mean                                                                                                                          
 v_model_amean = dim_avg_n_Wrap(v_model_m,0)                                                                                                   

 ; compare obs / model                                                                                                                         
 diff = v_model_amean - v_obs_amean                                                                                                            
 copy_VarMeta(v_model_amean, diff)                                                                                                             
 print(var+" CLM Test Case - Observations")                                                                                                    
 printMinMax(diff,1)                                                                                                                           

 ; plot diff
 plotC = new(2, graphic)

 res@gsnLeftString = var
 res@gsnRightString = "W m~S~-2"

 ; set the contour bounds
 res@cnLevelSelectionMode = "ManualLevels"

 ; LHF/FSH diffs (W/m^2)
 res@cnMinLevelValF  = -30
 res@cnMaxLevelValF  = 30
 res@cnLevelSpacingF = 3
                                                                                                                                               
 ;res@gsnCenterString = "Optimized CLM - Observations"

 plotC(0) = gsn_csm_contour_map(wks,diff,res)

 ; compare obs / model default                                                                                                                 
 strs_def = (/path_model,"hydro_ensemble_LHC_default_params/lnd/hist/*{001[6-9],20-}*"/)                                                       
 str_c = str_concat(strs_def)                                                                                                                  
 fils = systemfunc("ls "+str_c)                                                                                                                
 f = addfiles(fils,"r")                                                                                                                        
 v_model_default = f[:]->$var_model$                                                                                                           
 v_model_default_m = mask(v_model_default,datamask_obs.eq.0,False)                                                                             
 copy_VarMeta(v_model_default,v_model_default_m)                                                                                               
 v_model_def_amean = dim_avg_n_Wrap(v_model_default_m,0)

 diff_def = v_model_def_amean - v_obs_amean                                                                                                    
 copy_VarMeta(v_model_def_amean, diff_def)                                                                                                     
 print(var+" Default CLM - Observations")
 printMinMax(diff_def,1)                                                                                                                       
                                                                                                                                               
 ;res@gsnCenterString = "Default CLM - Observations"                                                                                            

 plotC(1) = gsn_csm_contour_map(wks,diff_def,res)

;************************************************                                                                                              
; Panel plot
;************************************************
 presA                     = True                ; modify the panel plot
 presA@gsnFrame		   = False
 presA@gsnPanelTop         = 0.95
 presA@gsnPanelBottom      = 0.65
 presA@gsnPanelLabelBar    = True                ; add common colorbar
 presA@lbBoxLinesOn = False ; no boxes

 presB                     = True                ; modify the panel plot
 presB@gsnFrame            = False
 presB@gsnPanelTop         = 0.65
 presB@gsnPanelBottom      = 0.35
 presB@gsnPanelLabelBar    = True                ; add common colorbar
 presB@lbBoxLinesOn = False ; no boxes

 presC                     = True                ; modify the panel plot                                                                       
 presC@gsnFrame            = False                                                                                                             
 presC@gsnPanelTop         = 0.35                                                                                                               
 presC@gsnPanelBottom      = 0.05
 presC@gsnPanelLabelBar    = True                ; add common colorbar
 presC@lbBoxLinesOn = False ; no boxes 

 gsn_panel(wks,plotA,(/1,2/),presA)
 gsn_panel(wks,plotB,(/1,2/),presB) 
 gsn_panel(wks,plotC,(/1,2/),presC)

 frame(wks)

end
