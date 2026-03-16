# MT

This repository contains all the code used for my Master Thesis. It does not include the datasets used to 

The repository is missing two large datafiles
- Dataset/Dataset_MSc/GFOC_RDCDFI.csv
- Dataset/Dataset_MSc/SWMA_RDAWFI.csv

It includes code from other sources, namely:


## Structure of Repository
In this section, the structure of the repository is displayed along with some explanatory text for each object.

- Analysis
    - modeling
        - Early_Tests

            Contains all code that was used during the early tests. 
        - logs
            - `multi_forecast_CV.log`
                all logs for the script `Analysis\modeling\multi_forecast_CV.py`
            - `tscv_training.log`
                all logs for the script `Analysis\modeling\create_tscv_models.py`
        - saved_models

            Contains all models created with the script `Analysis\modeling\multi_forecast_CV.py`
        - tscv_models

            Contains all models created with the script `Analysis\modeling\create_tscv_models.py`
        - `amplitude_finder.ipynb`

            **Description:** Analysis of the orbital period of *GRACE-FO-1* based on the orbital decay rate.
        - `correlation_matrix.ipynb`

            **Description:** Analysis the correlations between the lagged feature matrix `X` and the targets `y`.

        **Grid Search Analysis:** The following scripts were used for the grid search analysis:

        - `create_models.py`

            **Description:** Script for creating `MultiTaskLassoCV` models used for Grid Search (can also be used to generate `LinearRegression` and `Lasso` models).
        - `coef_analyis.ipynb`

            **Description:** Analysing coefficient matrices for generated models.
        - `compare_models.ipynb`

            **Description:** Compare different models created with `Analysis\modeling\multi_forecast_CV.py`.
        - `grid_search.ipynb`

            **Description:** Grid Search to determine the ideal regularisation strength of the trained `MultiTaskLassoCV` models.
        - `test_model.ipynb`

            **Description:** Run the models created with `Analysis\modeling\multi_forecast_CV.py` on the Test set.
        
        **Time Series Cross Validation (TSCV):** The following scripts were used for the TSCV analysis:

        - `create_tscv_models.py`

            **Description:** Create TSCV models.
        - `tscv_test.ipynb`

            **Description:** Initial test script for `Analysis\modeling\create_tscv_models.py`.
        - `tscv_time.ipynb`

            **Description:** Visualisations of the TSCV training and testing intervals
        - `tscv_results.ipynb`

            **Description:** Bar plot generated from the results (log file) of the training and testing during the creating of the models.
        - `tscv_subsets.ipynb`

            **Description:** Calculate R2 and R2 adjusted values for each trained model, create Barplots and visualisations of their evolution across different forecast horizons. Plot that shows a single forecast (for each model) at a specified point in time. Aggregated forecast horizons and created plots for different intervals.
        - `tscv_coef.ipynb`

            **Description:** Check coefficient values and lagged feature importance for a single model.
        - `tscv_compare.ipynb`

            **Description:** Compare coefficient values and lagged feature importance of different models.
        - `tscv_SHAP.ipynb`

            **Description:** SHAP-Based feature importance.
    - Orbital_Decay
        - generated

            **Description:** contains all the scripts for generating simulated orbital decay curves as well as their analysis.
        - plots

            **Description:**  Plots.
        - Trend_Removal

            **Description:** All the scripts for removing seasonal and trend components from the orbital decay rate. In particular the scripts labeled `S1` to `S4` are relevant to the thesis.

    - Spacecraft_Plots

        **Description:** Creating spacecraft position plots for the event catalogue data.
    - Subsets
        - `interesting_subsets.ipynb`
        
        **Description:** Create lists of intervals or flags for interesting subsets based on the change in orbital decay rate during one orbital period of *GRACE-FO-1* (**MeanStd**).
        - `interesting_subsets2.ipynb`
        
        **Description:** Create lists of intervals or flags for interesting subsets based on the Event flags from the 3 ICME catalogues (**Eflag**).
        - `interesting_subsets3.ipynb`
        
        **Description:**  Create lists of intervals or flags for interesting subsets based on the **Kp index**.
        - `interesting_subsets_merge.ipynb`
        
        **Description:** Merge the different subsets into a news subset/interval list.
        - `interesting_subsets_compare.ipynb`
        
        **Description:** Compare the different subsets and create visualisation plots.

- Dataset
    - Dataset_ICMECAT
    
    **Description:** Contains the scripts to download and process the *Helio4cast* ICME catalogue and the *R&C* ICME catalogue.
    - Dataset_IPshocks

    **Description:** Contains the script to convert the downloaded *IPShock* catalogue into csv format.
    - Dataset_MSc

    **Description:** Contains the scripts that were used for the inital evaluation of the Dataset provided by V. Mercea.
    - modeling
        - `setup_Dataset.py`

        **Description:** This script was used to create a trimmed-down version of the original dataset and adds some of the later used parameters (median decay rates, trend).
        - `resample_add.ipynb`

        **Description:** Used to resample the dataset from the original sampling rate (20s) to different sampling ratess (1min, 5min). Can also be used to add parameters to the dataset.
        - `create_Xy.py` 

        **Description:** Setup and create the lagged feature matrix `X` and the target matrix `y`.
        - `test.ipynb`

        **Description:** Used for testing the resulting dataset.
        - `test2.ipynb`

        **Description:** Used for testing the resulting Matrices `X` and `y`.

    - `create_flags.ipynb`

    **Description:** for creating binary event flags that can be added directly into the dataset (not used in the end)
    - `Dataset_Documentation.txt`

    **Description:** Descriptions of the different datasets.


- utils

    **Description:** Contains functions that were used in multiple scripts and important as packages. They are grouped into different scripts based on their function (data loading, plotting, modeling)
