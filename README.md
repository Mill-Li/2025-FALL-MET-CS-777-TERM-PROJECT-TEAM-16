# 2025-FALL-MET-CS-777-TERM-PROJECT-TEAM-16
Term Project at Boston University
# Transportation Mode Recognition Using Apache Spark and Machine Learning
The primary objective of this term project is to design and implement a large-scale transportation mode recognition system using the Microsoft Research Asia Geolife GPS Trajectory Dataset. This dataset contains over 17,000 trajectories collected from 182 users between April 2007 and August 2012, covering a total distance of 1.29 million kilometers and more than 50,000 hours of movement. Each trajectory is a sequence of time-stamped GPS points with information on latitude, longitude, altitude, and time, capturing diverse user activities such as walking, cycling, driving, and flying.

Using Apache Spark as the core computational platform, this project aims to develop a distributed machine learning pipeline for classifying different transportation modes based on spatiotemporal features, including speed, acceleration, altitude variation, and stop duration. Models including Random Forest, Gradient-Boosted Trees, and Logistic Regression will be implemented using Spark MLlib, enabling efficient processing and scalable classification of millions of GPS data points

The expected outcome is an end-to-end big data analytics framework capable of accurately identifying transportation modes and visualizing their spatial distribution. This work contributes to real-world applications in urban mobility analysis, intelligent transportation systems, and smart city development, while demonstrating strong technical depth in big data processing and distributed machine learning.

# Code Documentation
## Data Preprocessing
[Open the preprocessing notebook](./TERM%20PROJECT/CODE/Data_Preprocessing.ipynb)
### Environment Setup
This Step in the project requires no local environment setup. 

All code runs in the following environment:

**Databricks Serverless Notebook Compute**
* **Environment:** Databricks Serverless Notebook Compute
* **Environment Version:** 4
* **Apache Spark Version:** 4.0.0
* **Language:** Python(PySpark)

**Required Library**

All required library is included in Databricks Runtime:
* PySpark

**File Storage**

Input and output data are stored in ```/Volumes/workspace/default/metcs777termproject``` in Databricks Free Edition

### How to Run the Code
1. Import the Preprocessing Notebook into Databricks Workspace
2. Upload the required raw data onto Databricks Volume
3. Change the file path if required for both input and output
4. Run all cells with the serverless notebook compute
5. Temporary Views would be generated throughout the steps
6. Final result generated after all cells finished and stored under the specified path
