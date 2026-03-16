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
            **Descripiton:** Analysis of the orbital period of *GRACE-FO-1* based on the orbital decay rate.
        - `correlation_matrix.ipynb`
            **Descripiton:** Analysis the correlations between the lagged feature matrix `X` and the targets `y`.

        **Grid Search Analysis:** The following scripts were used for the grid search analysis:

        - `create_models.py`
            **Descripiton:** Script for creating `MultiTaskLassoCV` models used for Grid Search (can also be used to generate `LinearRegression` and `Lasso` models).
        - `coef_analyis.ipynb`
            **Descripiton:** Analysing coefficient matrices for generated models.
        - `compare_models.ipynb`
            **Descripiton:** Compare different models created with `Analysis\modeling\multi_forecast_CV.py`.
        - `grid_search.ipynb`
            **Descripiton:** Grid Search to determine the ideal regularisation strength of the trained `MultiTaskLassoCV` models.
        - `test_model.ipynb`
            **Descripiton:** Run the models created with `Analysis\modeling\multi_forecast_CV.py` on the Test set.
        
        **Time Series Cross Validation (TSCV)** The following scripts were used for the TSCV analysis:

        - `create_tscv_models.py`
            **Descripiton:** Create TSCV models.
        - `tscv_test.ipynb`
            **Descripiton:** Initial test script for `Analysis\modeling\create_tscv_models.py`.
        - `tscv_time.ipynb`
            **Descripiton:** Visualisations of the TSCV training and testing intervals
        - `tscv_results.ipynb`
            **Descripiton:** Bar plot generated from the results (log file) of the training and testing during the creating of the models.
        - `tscv_subsets.ipynb`
            **Descripiton:** Calculate R2 and R2 adjusted values for each trained model, create Barplots and visualisations of their evolution across different forecast horizons. Plot that shows a single forecast (for each model) at a specified point in time. Aggregated forecast horizons and created plots for different intervals.
        - `tscv_coef.ipynb`
            **Descripiton:** Check coefficient values and lagged feature importance for a single model.
        - `tscv_compare.ipynb`
            **Descripiton:** Compare coefficient values and lagged feature importance of different models.
        - `tscv_SHAP.ipynb`
            **Descripiton:** SHAP-Based feature importance.
    - Orbital_Decay
    - Spacecraft_Plots
    - Subsets
