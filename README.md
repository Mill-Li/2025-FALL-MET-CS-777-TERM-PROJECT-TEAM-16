# Transportation Mode Recognition Using Apache Spark and Machine Learning
Term Project at Boston University

Team Member: Tingchen Li, Kangjin Wang

# Introduction
As technology developed, smart city planning and optimizing public transit have become one of the unavoidable topics that people think of. In the development of planning and optimization, while several types of data are used to assist, mobility patterns play a key role in it.

Our project focuses on classifying the transportation modes of large-scale trajectory data. We would use PySpark to preprocess the raw data and extract the movement-related features, use Apache Spark to train Logistic Regression, Random Forest, and Gradient-Boosted Trees models, cross-compare the three models, and extract the top three movement-related features for transportation modes classification based on our validation results.

# Dataset

For the selection of the data set, the Microsoft Research Asia Geolife project contains rich movement signals such as speed, acceleration and stop patterns, which are important and high-quality data for our traffic pattern prediction. And this large-scale data is very suitable for distributed computing.

The raw dataset was collected in the Microsoft Research Asia Geolife project by 182 users. 

Each GPS trajectory in this dataset is represented by a sequence of time-stamped points.

The dataset contains 17621 trajectories in total, and most of the trajectories are logged every 1-5 seconds or every 5-10 meters per point.

## Trajectory Files
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

**Sample lines**

```39.984702,116.318417,0,492,39744.1201851852,2008-10-23,02:53:04```

```39.984683,116.31845,0,492,39744.1202546296,2008-10-23,02:53:10```

```39.984686,116.318417,0,492,39744.1203125,2008-10-23,02:53:15```

```39.984688,116.318385,0,492,39744.1203703704,2008-10-23,02:53:20```

```39.984655,116.318263,0,492,39744.1204282407,2008-10-23,02:53:25```

## Label Files
Some users have a separate ```label.txt``` file that labels their trajectories with transportation mode.

**TXT Format:**

Line 1 is a header.
* Field 1: Start Time
* Field 2: End Time
* Field 3: Transportation Mode

**Sample lines**

```2008/08/20 12:09:17	2008/08/20 12:45:05	walk```

```2008/08/20 12:45:05	2008/08/20 13:00:25	subway```

```2008/08/20 13:00:25	2008/08/20 13:07:25	walk```

```2008/08/20 13:07:25	2008/08/20 13:12:59	bus```

```2008/08/20 13:13:59	2008/08/20 13:24:23	walk```

# Methodology
As the raw data contains GPS data only, the coordinates in the GPS data need to be transformed to analyzable numerical features, like speed, distance, and acceleration, for model training.

While the trajectories are using timestamps, the label files are using time intervals, which need to be rematched for later training. 

As a result, data preprocessing plays an important role in our project. 

[Data Preprocessing & Feature Extraction](#data-preprocessing--feature-extraction)


For the models part, accuracy and weighted F1 scores are selected as the primary performance metrics.
* Accuracy provides an overall measure of correctness
* Weighted F1 scores accounting for class imbalance, which is preferable according to our dataset spread.

Logistic Regression is chosen as a baseline, since it is fast and interpretable. 

Random Forest and Gradient-Boosted Trees are chosen to dive deeper in seeking better accuracy and F1-scores.

[Model Training & Evaluation](#model-training--evaluation)

## Data Preprocessing & Feature Extraction
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

### Steps Taken for Preprocessing & Feature Extraction
1. **Load Raw Files**
   * Recursively load ```.plt``` files from directories
   * Parse Timestamps and GPS points
   * Assign Traj_id and User_id based on path
2. **Load Label Files**
   * Recursively load ```.txt``` files from directories
   * Parse Timestamps
   * Assign Traj_id and User_id based on path
3. **Feature Engineering**
   * $\Delta$ t (Changing time between steps/records)
   * Haversine Distance between steps/records
   * Speed and Acceleration based on distance and time
   * Stop duration per steps/records
4. **Mode Assignment**
   * Merge with the provided mode data
5. **Trajectory Aggregation Using PySpark:**
   * groupBy (user_id, traj_id, mode) or (user_id, traj_id)
   * Calculate descriptive statistics
   * Filter out invalid segments
6. **Save Final Output as** ```.parquet```

### Result of the Step
Two output folders contain ```.parquet``` files generated:

1. Labeled Aggregated Trajectory-level dataset
2. Unlabeled Aggregated Trajectory-level dataset

### Explanation of the Preprocessed & Feature Extraction Result
Preprocessed Data is stored in ```.parquet``` format.

**Result Explanation**
- **user_id:** Unique identifier for the user.
- **traj_id:** Unique identifier for the trajectory.
- **mode:** Transportation mode label (e.g., walk, bike, drive).
- **total_distance_m:** Total distance traveled across the trajectory (meters).
- **max_speed:** Maximum speed across the trajectory.
- **median_speed:** Median speed across the trajectory.
- **var_speed:** Variance of speed across the trajectory.
- **mean_accel:** Average acceleration across the trajectory.
- **max_accel:** Maximum acceleration across the trajectory.
- **stop_duration_seconds:** Total time stopped (seconds) across the trajectory.
- **start_time:** Timestamp of the first point in the segment.
- **end_time:** Timestamp of the final point in the segment.
- **duration_seconds:** Total duration of the trajectory (seconds).
- **mean_speed_calculated:** Total distance divided by duration (m/s), computed feature.

## Model Training & Evaluation
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

### Step Taken for Model Training and Evaluation
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

### Result of Model Training, Evaluation
1. Outputs saved as ```.csv``` or ```.png``` files.
2. The Confusion Matrix is evaluated for each model trained, results are saved as [graphs](./TERM%20PROJECT/RESULT/FINAL%20RESULT/ConfusionMatrix).
3. Cross-comparison between three metrics on the models saved as [file](./TERM%20PROJECT/RESULT/FINAL%20RESULT/ModelMetric).
4. Cross-comparison plotted and saved as [separate graphs](./TERM%20PROJECT/RESULT/FINAL%20RESULT/ModelComparison).
5. Feature Importance is saved as a [graph](./TERM%20PROJECT/RESULT/FINAL%20RESULT/FeatureImportance) for the best model.

# Result
## Explanation of final results
We conducted a comprehensive comparison of the three models under the same training set, test set, and the same feature selection. The results show that these models have their own advantages and disadvantages in different aspects. 
* Random Forest model performed best in terms of Accuracy and F1 Score in our results, reaching 0.8394 and 0.8386, and it can handle nonlinear features well.
* Logistic Regression, as a baseline of our choice, has the most stable overall performance, and can still maintain stable performance under the condition of low training cost, but its performance in prediction is very limited, and it is difficult to deal with nonlinear mobility patterns.
* Gradient-Boosted Trees, which is the most complex and costly model among the three, trains more slowly and requires careful parameter tuning, and performs better in some categories. Yet, the separability of different labels of transportation modes is very different in terms of the categories of the data itself. 

In our study, we found two pairs of transportation modes that are particularly prone to misclassification. The first is Walking and Running, which are misclassified mainly due to similar short-range speed fluctuation patterns. The second is Bike and Bus, because both show medium-speed segments and intermittent stops. In our final Feature Importance analysis, we found that speed and distance-related features contributed the most to the classification results, including median_speed, var_speed, and total_distance.

## Pros/Cons
In our project, we used Spark to distribute and process the large-scale GeoLife GPS Trajectory Dataset, which enabled our project to be completed efficiently and accurately. In addition, we used a variety of prediction models for comparison. This comparative experiment helped us to observe the advantages of different models in that part, so that we can comprehensively evaluate the characteristics of these three different algorithms. Finally, we saved the results of the Accuracy and the F1 Score, and made a lot of visual analysis, including performance comparison, Confusion Matrix, and importance analysis of each feature. These can help us better analyze and display in the future, and can specifically understand the pattern of misclassification in the training process, and enhance the interpretability of the model.

There are also some shortcomings in our project. First of all, the data set is imbalanced, which will affect the recognition effect of the traffic mode in the prediction process. And in the data, GPS signals contain a lot of noise and inconsistent sampling rates, all impacting speed and acceleration accuracy. Finally, for our Gradient-Boosted Trees, as mentioned before, this model requires a lot of training costs and is more sensitive to resources and parameters. If you want this model to perform better, you need to spend more time training it. Moreover, in the whole project, the prediction of the traffic type is only based on the features of the traffic itself, and the individual differences of users are not considered.

## Recommendation
In response to the shortcomings found in our project, we can make the following improvements later. 

First, for the problem of unbalanced data sets, we can use SMOTE or stratified sampling to improve to avoid the impact of this part on the results.

On the other hand, since the number of features currently used is not comprehensive enough, we should introduce more different types of features to help predict and improve the model's ability to distinguish. 

In terms of models, due to the limited resource allocation, we did not expand more types of models for comparative analysis. In the next step, we can choose some deep learning models, such as RNN and LSTM, for sequential GPS data.

# Conclusion
In this project, our main goal is to complete a complete end-to-end transportation mode recognition system, and use the original GPS data for distributed computing processes and for different model evaluations. And the final results show that Spark has obvious advantages in large-scale trajectory data processing and modeling. 

In this project, we have learned a lot. For example, in the data preprocessing and cleaning part, we found that whether the data is cleaned or not has a significant impact on the subsequent model training, and the training difficulty and model performance will be affected. And for feature selection part, a good feature selection can make the model better learn and predict the data, which is also a very important point. 

After completing the entire project, we also need to build a reproducible and scalable pipeline, which is of great help to the follow-up work. In general, we have a complete practice of how to complete a recognition system through this project.

# Contributions
Tingchen Li
* Building the [pre-processing pipeline](./TERM%20PROJECT/CODE/Data_Preprocessing.ipynb)
* Helping fine-tune the [model](./TERM%20PROJECT/CODE/Geolife_Mode_Classification.py)
* Write the corresponding part and format the whole part of [report](./README.md)

Kangjin Wang
* Building and tuning the [machine learning pipeline](./TERM%20PROJECT/CODE/Geolife_Mode_Classification.py)
* Validating the [preprocessed result](./TERM%20PROJECT/RESULT/PREPROCESSED%20RESULT/)
* Write the corresponding part of the [report](./README.md)

# Reference
[1] Data Source: Learning transportation mode from raw GPS data for geographic applications on the Web  <https://www.microsoft.com/en-us/research/project/geolife-dataset/>

[2] Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from GPS trajectories. In Proceedings of International conference on World Wild Web (WWW 2009), Madrid Spain. ACM Press: 791-800.

[3] Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, Wei-Ying Ma. Understanding Mobility Based on GPS Data. In Proceedings of ACM conference on Ubiquitous Computing (UbiComp 2008), Seoul, Korea. ACM Press: 312-321.

[4] Yu Zheng, Xing Xie, Wei-Ying Ma, GeoLife: A Collaborative Social Networking Service among User, location and trajectory. Invited paper, in IEEE Data Engineering Bulletin. 33, 2, 2010, pp. 32-40.

[5] Databricks Official Documentation <https://docs.databricks.com/>

[6] Google Cloud Dataproc Official Documentation <https://cloud.google.com/dataproc/docs>
