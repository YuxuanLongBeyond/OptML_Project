# Project for OptML

Only Numpy and Matplotlib are required to be installed.  

## Translation Averaging

To show the demo of camera location recovery (i.e. translation averaging), please go to the directory `translation` and **directly run** `location_estimation.py` so that a plot of recovered 3D locations will be immediately drawn. The plot will be same as the first figure in Appendix of the report.  
To re-produce the plot of MAE versus noise variance in the report, modify `robustness_test = 1` to `robustness_test = 1` in the script `location_estimation.py`. Running this test would take some time.  
