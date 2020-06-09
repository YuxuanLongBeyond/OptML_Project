# Project for OptML

Only Numpy, Matplotlib and pytransform3d are required to be installed.  

## Rotation Averaging
To visualize a viewing graph, please go to the directory `rotation` and play around with `demo_test.ipynb`. Some mapping methods between different rotation representations are implemented as well.

To check the effectiveness of the interior point method, please go to the directory `rotation` and find `l1decoder.ipynb`. You can do different tests by changing the size of A matrix and adding different level of perturbations. Note: the level of perturbations should not be too high.

## Translation Averaging

To show the demo of camera location recovery (i.e. translation averaging), please go to the directory `translation` and **directly run** `location_estimation.py` so that a plot of recovered 3D locations will be immediately drawn. The plot will be same as the first figure in Appendix of the report.  
To re-produce the plot of MAE versus noise variance in the report, modify `robustness_test = 1` to `robustness_test = 1` in the script `location_estimation.py`. Running this test would take some time.  
