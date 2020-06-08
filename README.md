# Project for OptML

Only Numpy and Matplotlib are required to be installed.  

## Translation Averaging

To show the demo of camera location recovery, please **directly run** `location_estimation.py` so that a plot of recovered 3D locations will be immediately drawn. The plot will be same as the first figure in Appendix of the report.  
To re-produce the plot of MAE versus noise variance in the report, modify the variable `robustness_test` to 1 in `location_estimation.py` (i.e. `robustness_test = 1`) and re-run this script. This would take some time.  
