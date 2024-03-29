* Spotted error on flux. Set flux_error=0 for now. 
* Changed the number of energy bin (from 4 to 3).
* Refactor APDataset to be able to process a single time series and to extract sub-sequences.
* A bug in the simulation code has been spotted and fixed! 
* The dataset config file now supports 'single' and 'multiple' dataset types
* New 'single' datasets has been added to the agilehost3-prod5.yml
* A bug in the photometry code has been spotted and fixed. The bug caused fixed region integrations.
* Implementation of a custom loss using the Keras API.
* Upgrade of the 'predictions' plot (plot_predictions_v2).
* Implementation of RNN and CNN models and refactoring of the models classes.
* Implementation of test set generation
* Evaluation of the trained models with a jupyter notebook
* Implemented new Photometry class. It can extract from multiple reflected regions in multiprocessing and return numpy array instead of csv files.
* The p-value analysis code has been refactored. The pipeline will apply both the photometry and model prediction starting from the DL3.  
* Updated testing code. Load a named test-set composed by multiple timeseries.
* Major refactoring of OnlinePhotometry.
* Add a new DataManager class to generate train set e test set. 
* Spotted a bug that duplicate a region of each ring.
* Added module to extract p-values, thresholds and significance levels.
* P-value analysis using independent samples. 
* Spotted a bug in the generation of the pivot to separate normal samples to anomalous ones.
* Li&Ma vs AnomalyDetection notebook.