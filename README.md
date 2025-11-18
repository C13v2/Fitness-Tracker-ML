
# Fitness-Tracker-ML
Mini Master Project 

This repository contains a full data-processing → feature-extraction → modeling pipeline for barbell exercise recognition and repetition counting using accelerometer and gyroscope data collected from Mbientlab WristBand sensors.
The codebase has been reviewed and corrected (bug fixes, robustness improvements, clearer interfaces, logging, safer file handling).

What this project does:
Reads raw sensor CSV files and builds a cleaned time-indexed dataset.
Removes outliers using multiple methods (IQR, Chauvenet, LOF), and exports cleaned data.
Filters signals (Butterworth low-pass), computes PCA, temporal and frequency features (FFT-based).
Counts repetitions via peak detection on filtered signals.
Trains and evaluates multiple classifiers (Random Forest, Decision Tree, SVM, KNN, Naive Bayes, MLP), including automatic hyperparameter search.
Visualizes signals, outliers, feature distributions and model performance.
Saves final feature datasets and trained models for later prediction.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Quick start — environment & install

Create environment (recommended):
conda create -n barbell_ml python=3.9 -y
conda activate barbell_ml
# If you have conda_requirements.txt:
conda install --name barbell_ml --file conda_requirements.txt
# Then:
pip install -r pip_requirements.txt


If you don't have conda, just:
python -m venv venv
source venv/bin/activate   # linux/mac
venv\Scripts\activate      # windows
pip install -r pip_requirements.txt


Key packages required
numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib (for saving models), optionally graphviz (for tree export).

How to run (order matters)
Run scripts from the repository root.

Build dataset
python src/data/make_dataset.py                    # Output: data/interim/01_data_processed.pkl


Remove outliers
python src/features/remove_outliers.py            # Output: data/interim/02_outliers_removed_chauvenets.pkl


Build features
python src/features/build_features.py            # Output: data/interim/03_data_features.pkl


Train models & evaluate
python src/models/train_model.py                 # Trains multiple classifiers, runs forward feature selection, saves metrics/plots


Predict (apply saved models)
If you have src/models/predict_model.py, use it to load saved model(s) and run on new sessions. If not present, train_model.py includes examples to get predictions.

Visualization
python src/visualization/visualize.py            # Will create plots and save to reports/figures/

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Contact / attribution
Original project & inspiration: Machine Learning for the Quantified Self (Mark Hoogendoorn & Burkhardt Funk), and codebase by Dave Ebbelaar. 
