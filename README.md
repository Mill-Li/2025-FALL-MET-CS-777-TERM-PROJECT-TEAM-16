# 2025-FALL-MET-CS-777-TERM-PROJECT-TEAM-16
Term Project at Boston University
# Transportation Mode Recognition Using Apache Spark and Machine Learning
The primary objective of this term project is to design and implement a large-scale transportation mode recognition system using the Microsoft Research Asia Geolife GPS Trajectory Dataset. This dataset contains over 17,000 trajectories collected from 182 users between April 2007 and August 2012, covering a total distance of 1.29 million kilometers and more than 50,000 hours of movement. Each trajectory is a sequence of time-stamped GPS points with information on latitude, longitude, altitude, and time, capturing diverse user activities such as walking, cycling, driving, and flying.

Using Apache Spark as the core computational platform, this project aims to develop a distributed machine learning pipeline for classifying different transportation modes based on spatiotemporal features, including speed, acceleration, altitude variation, and stop duration. Models including Random Forest, Gradient-Boosted Trees, and Logistic Regression will be implemented using Spark MLlib, enabling efficient processing and scalable classification of millions of GPS data points

The expected outcome is an end-to-end big data analytics framework capable of accurately identifying transportation modes and visualizing their spatial distribution. This work contributes to real-world applications in urban mobility analysis, intelligent transportation systems, and smart city development, while demonstrating strong technical depth in big data processing and distributed machine learning.

# Code Documentation
## Data Preprocessing
[Preprocessing Notebook](./TERM%20PROJECT/CODE/Data_Preprocessing.ipynb)
### Environment Setup
This Step in the project requires no local environment setup. 

All code runs in the following environment:

**Databricks Serverless Notebook Compute**
* **Environment:** Databricks Serverless Notebook Compute
* **Environment Version:** 4
* **Apache Spark Version:** 4.0.0
* **Language:** Python(PySpark)

**Required Library**

All required library is included in the Databricks Serverless Notebook:
* pyspark

**File Storage**

Input and output data are stored in ```/Volumes/workspace/default/metcs777termproject``` in Databricks Free Edition

### How to Run the Code
1. Import the [Preprocessing Notebook](./TERM%20PROJECT/CODE/Data_Preprocessing.ipynb) into Databricks Workspace
2. Upload the [raw data](./TERM%20PROJECT/DATA/) to Databricks Volume
3. Change the file path if required for both input and output
4. Run all cells with the serverless notebook compute
5. Temporary Views would be generated throughout the steps
6. Final result generated after all cells finished and stored under the specified path

### Result of Preprocessing Step
Two output folders contain ```.parquet``` files generated:

1. Labeled Aggregated Trajectory-level dataset
2. Unlabeled Aggregated Trajectory-level dataset

### Explanation of the Raw dataset

The raw dataset contains 182 users' multiple trajectory files with GPS tracking points.

Trajectory files are all in ```.plt``` format.

**PLT Format:**

Lines 1…6 are useless in this dataset and can be ignored. 

Points are described in the following lines, one for each line.
* Field 1: Latitude in decimal degrees.
* Field 2: Longitude in decimal degrees.
* Field 3: All set to 0 for this dataset.
* Field 4: Altitude in feet (-777 if not valid).
* Field 5: Date - number of days (with fractional part) that have passed since 12/30/1899.
* Field 6: Date as a string.
* Field 7: Time as a string.

Some users have a separate ```label.txt``` file that labels their trajectories with transportation mode.

**TXT Format:**

Line 1 is a header.
* Field 1: Start Time
* Field 2: End Time
* Field 3: Transportation Mode

### Explanation of the Preprocessed Result
Preprocessed Data is stored in ```.parquet``` format.

**Steps Taken for Preprocessing:**
1. **Load Raw Files**
   * Recursively load ```.plt``` files from directories
   * Parse Timestamps and GPS points
   * Assign Traj_id and User_id based on path
2. **Load Label Files**
   * Recursively load ```.txt``` files from directories
   * Parse Timestamps
   * Assign Traj_id and User_id based on path
3. **Feature Engineering**
   * $\Delta$ t(Changing time between steps/records)
   * Haversine Distance between steps/records
   * Speed and Acceleration based on distance and time
   * Stop duration per steps/records
4. **Mode Assignment**
   * Merge with the provided mode data
5. **Trajectory Aggregation**

   **Using PySpark:**
   * groupBy (user_id, traj_id, mode) or (user_id, traj_id)
   * Calculate descriptive statistics
   * Filter out invalid segments
6. **Save Final Output as ```.parquet```**

**Data Explanation**
- **user_id:** Unique identifier for the user.
- **traj_id:** Unique identifier for the trajectory.
- **mode:** Transportation mode label (e.g., walk, bike, drive).
- **total_distance_m:** Total distance traveled in the trajectory (meters).
- **max_speed:** Maximum speed observed.
- **median_speed:** Median speed across the trajectory.
- **var_speed:** Variance of speed across the trajectory.
- **mean_accel:** Average acceleration.
- **max_accel:** Maximum acceleration.
- **stop_duration_seconds:** Total time stopped (seconds).
- **start_time:** Timestamp of the first point in the segment.
- **end_time:** Timestamp of the final point in the segment.
- **duration_seconds:** Total duration of the trajectory (seconds).
- **mean_speed_calculated:** Total distance divided by duration (m/s), computed feature.

## Model Training, Evaluation, Prediction
[Model Notebook](./TERM%20PROJECT/CODE/Geolife_Mode_Classification.py)
### Environment Setup
This Step in the project requires no local environment setup.

All code runs in the Google Cloud Platform environment:

**Google Cloud Platform Environment**
* **Platform:** Google Cloud Platform
* **Service:** GCP Compute & GCS bucket
* **Machine Type:**
  * Machine type: n4-standard-4
    * vCPU: 4 vCPUs
    * Memory: 16 GB RAM
    * Boot Disk: 200 GB SSD (Hyperdisk Balanced)
  * Worker Node Configuration (× 2)
    * Machine type: n4-standard-4
    * vCPU: 4 vCPUs
    * Memory: 16 GB RAM
    * Boot Disk: 200 GB SSD (Hyperdisk Balanced)

**Required Library**

Libraries are selected and installed when creating the instance.
* pyspark
* pandas
* matplotlib
* seaborn

**File Storage**

Input and Output are stored in the Google Cloud Storage Bucket.

### How to Run the Code
1. Start an instance on GCP.
2. Import the [Python File](./TERM%20PROJECT/CODE/Geolife_Mode_Classification.py) into the started instance.
3. Upload the [Preprocessed Data](./TERM%20PROJECT/RESULT/PREPROCESSED%20RESULT/) to the bucket.
4. Submit a task with the corresponding system argument declaring Python files, the data files, and the output path
5. Check output directory for results after task finishes.

### Result of Model Training, Evaluation
1. Outputs saved as ```.csv``` or ```.png``` files.
2. The Confusion Matrix is evaluated for each model trained, results are saved as graphs.
3. Cross-comparison between three metrics on the models saved as ```.csv```.
4. Feature Importance is saved as a graph for the best model.

### Explanation of Model Training and Evaluation Results
**Step Taken for Model Training and Evaluation:**
1. **Load Preprocessed Data**
   * Read in ```.parquet``` files into pyspark dataframes
2. **Feature Preprocessing and Standardization**
   * Clean all the features, including removing nulls and filtering invalid data
   * Encoding Categorical data
   * Feature vector assembly
   * Standardization for feature vector
3. **Train/Test Data Split**
   * Split the data into two datasets
     * 80% training
     * 20% testing
     * Random State = 42
4. **Train & Evaluate Random Forest Model**
   * Set up Random Forest Model
     * numTrees = 200
     * maxDepth = 12
     * Random State = 42
   * Fit the model with the train data
   * Test the fitted model with the test data and generate a prediction list for evaluation
   * Evaluate the model using Accuracy and F1 scores.
5. **Train & Evaluate Logistic Regression Model**
   * Set up Logistic Regression Model
     * maxIter = 100
     * regParam = 0.01
     * L2
   * Fit the model with the train data
   * Test the fitted model with the test data and generate a prediction list for evaluation
   * Evaluate the model using Accuracy and F1 scores.
6. **Train & Evaluate Gradient Boost Tree Model**
   * Set up Basic Gradient Boost Tree Model
     * maxIter = 80
     * maxDepth = 6
     * stepSize = 0.1
     * Random State = 42
   * Set up One vs Rest since we have multiple features, as GBT only supports binary classification.
   * Fit the one vs rest model with the train data
   * Test the fitted model with the test data and generate a prediction list for evaluation
   * Evaluate the model using Accuracy and F1 scores.
7. **Build and plot multiclass Confusion Matrix for models**
   * Import MulticlassMetrics from pyspark.mllib.evaluation
   * Create a self-defined function to create a Confusion Matrix
   * Generate matrix for all three models
   * Using seaborn and matplotlib to generate a visualization of the confusion matrix
8. **Cross Comparison on Accuracy and F1 Score**
   * Simply build a pandas dataframe to print the result
9. **Plot the Accuracy Comparison and F1 Score Comparison Seperately**
    * Using seaborn and matplotlib to generate a visualization for each comparison in a bar graph.
10. **Plot the feature importance for the best model**
    * Using matplotlib to generate a visualization for the feature importance of the best model.
