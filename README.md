# Gig Economy Rideshare Analytics

## Project Overview



Act as an expert Machine Learning engineer, Python developer, and academic project mentor. I am working on my 23CSE301 ML Capstone project titled **"Gig Economy — Driver Earnings & Burnout."**

I will provide my actual dataset, Review 2 guidelines, and Review 1 files if required. Carefully analyze all the provided files and generate the complete, correct, end-to-end Python code for my ML Project Review 2.


## 1. Project Details

Project title: Gig Economy — Driver Earnings & Burnout.

Dataset: Chicago Taxi Trips dataset.

Dataset file: 'data/chicago_taxi_sample.csv'

The dataset contains taxi-trip information, including:
- taxi_id
- trip_start_timestamp
- trip_end_timestamp
- trip_seconds
- trip_miles
- pickup_community_area
- dropoff_community_area
- fare
- tips
- tolls
- extras
- trip_total
- payment_type
- company
- pickup and dropoff latitude/longitude

The actual dataset may contain missing values, duplicate rows, invalid values, categorical variables, and numerical variables. Inspect the actual file before writing the code. Never assume that the sample and full dataset have identical distributions.

The project has three ML tracks:

1. Regression — predicting daily driver earnings.
2. Classification — predicting driver burnout-risk categories.
3. Clustering — discovering groups of driver-days using unsupervised learning.

The project also includes a Streamlit GUI as a bonus feature.

## 2. Review Guidelines

Before generating the code, read the entire Review 2 guidelines and identify every required task.

Create a requirement checklist covering:
- Regression requirements.
- Classification Part A and Part B requirements.
- All required classification algorithms.
- Clustering algorithms and preprocessing.
- Required evaluation metrics.
- Required plots and visualizations.
- Hyperparameter tuning.
- Model comparison.
- PCA or t-SNE requirements, if specified.
- Final model selection and justification.
- Model saving and loading.
- Streamlit GUI requirements.
- Both bonus-mark requirements.
- Any report, observation, conclusion, or submission requirements.

Follow the official guidelines exactly. If the guidelines differ from the project README, prioritize the official guidelines and explain the difference.

## 3. Dataset Loading and Exploratory Data Analysis

Provide complete executable code for:

1. Importing the required Python libraries.
2. Loading the actual CSV dataset.
3. Displaying the dataset shape, first five rows, column names, data types, and summary statistics.
4. Checking missing values and duplicate records.
5. Identifying invalid timestamps, negative trip durations, invalid distances, and other suspicious values where appropriate.
6. Converting timestamp columns to proper datetime formats.
7. Extracting date, hour, day of week, and weekend/night indicators when relevant.
8. Performing appropriate exploratory data analysis.
9. Creating meaningful visualizations with titles, axis labels, and legends.
10. Explaining important observations from the dataset.

Use the real column names and validate them before accessing them.

## 4. Driver-Day Aggregation and Target Creation

Convert trip-level data into driver-day-level data wherever required by the project.

Group the records using the appropriate driver identifier and calendar date. Follow the project's minimum-trip requirement if specified in the files.

Create meaningful daily features, such as:
- Number of trips.
- Total daily earnings.
- Total distance travelled.
- Total driving duration.
- Average trip distance.
- Average trip duration.
- Average fare.
- Total tips and other relevant earnings components.
- Night-time trip count or proportion.
- Weekend indicators.
- Other useful features supported by the actual dataset.

Ensure that daily earnings are calculated consistently and avoid double-counting fare, tips, tolls, extras, and trip totals.

Create the regression target `total_earnings` using the appropriate definition supported by the guidelines.

Create the classification target `burnout_risk` with Low, Medium, and High categories according to the prescribed method, such as tertiles of daily driving hours, if confirmed by the project instructions.

**Critical leakage prevention:** Do not use the exact driving-hours variable used to create `burnout_risk` as an input feature for predicting that target. Exclude any equivalent or directly derived leakage features as appropriate.

Explain the aggregation and target-creation logic.

## 5. Regression Track

Implement all regression algorithms required by the Review 1 and Review 2 guidelines.

If the guidelines require the ten algorithms documented in the project README, include:

1. Linear Regression.
2. Ridge Regression.
3. Lasso Regression.
4. ElasticNet Regression.
5. Polynomial Regression, degree 2.
6. K-Nearest Neighbors Regressor.
7. Support Vector Regressor.
8. Decision Tree Regressor.
9. Random Forest Regressor.
10. Gradient Boosting Regressor.

For every model:
- Use an appropriate train-test split.
- Apply scaling where required.
- Use suitable preprocessing pipelines.
- Train the model.
- Generate test predictions.
- Calculate R² score, RMSE, and MAE.
- Store results in a comparison table.
- Visualize and compare model performance.
- Plot actual versus predicted earnings where useful.

If required by the guidelines, perform cross-validation and hyperparameter tuning.

Select the best regression model based on appropriate validation results rather than assuming a model is best in advance.

Explain the final model's performance and limitations.

## 6. Classification Track — All Required Algorithms

Implement every classification algorithm required by the guidelines. If all ten algorithms in the project README are required, include:

1. Logistic Regression.
2. K-Nearest Neighbors Classifier.
3. Support Vector Machine.
4. Decision Tree Classifier.
5. Random Forest Classifier.
6. Gradient Boosting Classifier.
7. AdaBoost Classifier.
8. Bagging Classifier using a Decision Tree.
9. Gaussian Naive Bayes.
10. MLP Classifier.

For every classifier:
- Use a suitable stratified train-test split if appropriate.
- Handle numerical and categorical features correctly.
- Apply scaling where required.
- Prevent data leakage using preprocessing pipelines.
- Train the model and predict the test labels.
- Calculate accuracy, precision, recall, and F1-score.
- Use weighted averages for multiclass metrics when appropriate.
- Generate a confusion matrix.
- Calculate multiclass ROC-AUC using a suitable One-vs-Rest strategy when supported.
- Handle class imbalance and missing classes appropriately.
- Store all results in a single comparison table.
- Plot model performance comparisons.

Use the same split and evaluation protocol for fair comparison.

Do not fabricate model scores. Calculate all metrics from actual model predictions.

## 7. Classification Hyperparameter Tuning and Final Model

Perform hyperparameter tuning for the required final classifier using `GridSearchCV` or another method specified in the guidelines.

If SVM is required or selected based on validation:
- Compare the default SVM with the tuned SVM.
- Explore appropriate values of `C`, `kernel`, and other relevant parameters.
- Use a suitable scoring metric, such as weighted F1.
- Use cross-validation.
- Display the best parameters and cross-validation score.
- Evaluate the final tuned model on the held-out test set.
- Generate a classification report and confusion matrix.
- Calculate ROC-AUC where applicable.
- Justify final model selection using validation performance and relevant metrics.

Do not select SVM automatically if the official rubric requires a different selection procedure. The final choice must be supported by actual results.

## 8. Clustering Track — Mandatory Preprocessing Before Clustering

This section is extremely important.

**Never apply clustering directly to raw data. Complete and explain the preprocessing before fitting any clustering algorithm.**

Create a separate, complete clustering notebook or section with the following steps:

### A. Prepare clustering data
- Load the actual dataset.
- Perform appropriate cleaning and driver-day aggregation.
- Select meaningful numerical and/or encoded features.
- Exclude identifiers, timestamps in unsuitable raw form, target labels, and irrelevant columns.
- Do not include `burnout_risk` when fitting unsupervised clustering.
- Keep `burnout_risk` separately for post-clustering interpretation and comparison.
- Avoid using `total_earnings` as a clustering feature if it creates an inappropriate circular comparison with the regression target; explain the feature-selection decision.

### B. Handle missing values
- Inspect missing values in the selected clustering features.
- Impute numerical features appropriately.
- Encode categorical features if they are included.
- Ensure that the final clustering matrix contains valid numerical values.
- Verify that there are no remaining NaN or infinite values.

### C. Scale the features
- Use `StandardScaler` or another suitable scaling method.
- Fit the scaler on the training data only, then transform the test data.
- Keep the same feature columns and order for both datasets.
- Explain why scaling matters for distance-based algorithms.

### D. Optional dimensionality reduction
If required or useful, apply PCA after appropriate preprocessing and scaling.

Include:
- Explained variance ratio.
- Cumulative explained variance.
- A PCA visualization.
- A two-dimensional cluster plot using PCA components, where appropriate.

If t-SNE is required by the official guidelines, implement it separately for visualization and explain its limitations.

### E. K-Means clustering
Implement:
- The Elbow Method using inertia/WCSS.
- Silhouette analysis where appropriate.
- Selection of the number of clusters using evidence.
- K-Means model fitting.
- Cluster-label assignment for the training and test datasets.
- Cluster-size analysis.
- Cluster visualizations.
- Cluster profile tables using original, interpretable features.

Do not blindly fix the number of clusters to five. If the guidelines require five clusters, use five and explain the requirement. Otherwise, justify the selected value using the results.

### F. Agglomerative / Hierarchical clustering
Implement Agglomerative Clustering as required.

If the guidelines require linkage comparisons, compare:
- Ward.
- Complete.
- Average.
- Single.

Use a suitable distance metric and linkage combination. Explain any invalid combinations and avoid fitting unsupported configurations.

Generate a dendrogram where appropriate, considering computational limits. Compare the linkage results and justify the final selection.

### G. Evaluate clustering
Calculate the required internal clustering metrics, including:
- Silhouette Score.
- Davies–Bouldin Index.
- Calinski–Harabasz Index.

Compare the clustering results using a clear table.

Use internal metrics only on appropriately scaled feature matrices and valid cluster assignments. Handle degenerate solutions, such as one cluster or every sample becoming its own cluster.

### H. Compare clusters with burnout risk after fitting
After fitting the unsupervised models:
- Compare cluster assignments with `burnout_risk`.
- Display a cluster-versus-risk contingency table or heatmap.
- Calculate Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) if required.
- Explain that these are post-hoc comparisons and that the risk label was not used to fit the clustering model.

### I. Train-test consistency
If the official guidelines require a train-test comparison:
- Fit the clustering method on training data.
- Assign test observations using a valid out-of-sample procedure.
- For K-Means, use the fitted model's `predict` method.
- For Agglomerative Clustering, implement a justified assignment strategy, such as assigning test observations to the nearest training-cluster representative, or use another approach explicitly permitted by the guidelines.
- Do not pretend that standard Agglomerative Clustering directly supports prediction on unseen samples.
- Clearly explain how test-set metrics are computed.

## 9. Clustering Requirements

Carefully inspect the official guidelines for the two bonus marks.

If the bonuses correspond to dimensionality reduction and/or clustering visualizations:
- Implement PCA and t-SNE if required.
- Produce clear two-dimensional cluster visualizations.
- Include a cluster-profile comparison.
- Explain the results in understandable academic language.

If the bonuses specify different tasks, implement those exact tasks instead.

Do not assume that a suggested bonus task is mandatory unless supported by the supplied files.

## 10. Streamlit GUI Bonus

If the official guidelines or project requirements include a Streamlit GUI, provide the complete `app/app.py` implementation.

The application should include:
- A burnout-risk prediction interface.
- A daily-earnings prediction interface.
- Suitable user input controls.
- Prediction outputs.
- Class probabilities for burnout risk where supported.
- Model performance information.
- Input validation and helpful error messages.
- A clean, professional interface.

Ensure that the application uses the same feature names, feature order, preprocessing pipeline, and model versions as the saved models.

Save the complete fitted pipelines where possible so that the application can correctly preprocess user inputs.

Provide `requirements.txt`, commands for running the app locally, and deployment instructions if required.

Do not claim that deployment has succeeded unless it has actually been performed.

## 11. Model Saving and Loading

If required:
- Save the final trained models and preprocessing pipelines using `joblib` or an appropriate method.
- Use a consistent folder structure.
- Provide code to reload and test the saved models.
- Ensure that model inputs match the training feature schema.
- Mention the compatible Python and scikit-learn versions.

## 12. Code Quality Requirements

The generated implementation must:
- Be complete and executable.
- Use the actual dataset and its actual column names.
- Include all required imports.
- Define variables before use.
- Avoid missing functions and incomplete code.
- Avoid placeholder predictions and fabricated outputs.
- Use suitable preprocessing pipelines.
- Prevent target leakage.
- Handle missing values and unexpected categories.
- Use appropriate random seeds.
- Include meaningful comments.
- Avoid unnecessary repeated code.
- Be compatible with the specified Python environment.
- Include computational safeguards for expensive algorithms.
- Save outputs and plots if required.

If the dataset path is incorrect, explain exactly how to fix it.

If a library is missing, provide the required installation command.

Do not silently ignore errors or invent successful execution results.

## 13. Required Output Format

Organize the answer in the following order:

1. Analysis of the supplied files.
2. Complete Review 2 requirement checklist.
3. Dataset assumptions and preprocessing decisions.
4. Required library installation commands.
5. Complete regression implementation.
6. Complete classification implementation.
7. Complete classification model comparison and tuning.
8. Complete clustering preprocessing.
9. Complete K-Means implementation.
10. Complete Agglomerative Clustering implementation.
11. Clustering evaluation and visualization.
12. PCA/t-SNE or other required bonus tasks.
13. Streamlit GUI implementation, if required.
14. Model saving and loading.
15. Results tables and interpretation instructions.
16. Final conclusions.
17. Separate verification checklist for both bonus marks.
18. Final Review 2 submission checklist.

Write the code as numbered Jupyter Notebook cells in the correct execution order. Each code cell must be complete enough to run in sequence.

For every section, explain what the code does and what I should observe after running it.

## 14. Final Instructions

First inspect all supplied files. Then map every guideline to a specific implementation section. Finally, generate the full code.

If a requirement is missing or ambiguous, state that clearly. Do not invent requirements or results.

Prioritize correctness, complete guideline coverage, clustering preprocessing, proper evaluation, reproducibility, and the two official bonus-mark requirements.

**My goal is to obtain a complete, technically correct ML Project Review 2 implementation that I can run, understand, demonstrate, and explain during my academic evaluation.**
