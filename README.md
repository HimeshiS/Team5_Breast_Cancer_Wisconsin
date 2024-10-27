# Team5_Breast_Cancer_Wisconsin
Team 5's project exploring the Breast Cancer Wisconsin dataset to analyze and model patterns in cancer diagnosis. This repository contains data preprocessing, analysis, and machine learning models for classifying and predicting breast cancer outcomes.

## Content
* [Problem Definition](#problem-definition)
* [Approach to Solve the Problem](#approach-to-solve-the-problem)
* [Results](#results)
* [Team](#team)
* [Project Folder Structure](#project-folder-structure)

## Problem Definition

We defined our problem as follows:
Cost Optimization in Diagnostic Testing:  Assess whether all diagnostic features are needed to achieve high accuracy in breast cancer diagnosis. If a smaller subset of features provides similar accuracy, testing costs could be reduced by focusing on the most relevant ones.


## Approach to Solve the Problem

We think that our problem is a classification problem. We are using KNN method to propose a solution.
More specifically we suggest the following steps:

 - Standardize the dataset for the features.
 - Split the dataset into training and test sets with a 75-25 split.
 - Calculate the Chi-square score for each feature to understand its statistical significance relative to the target variable (Diagnosis).
 - Apply Forward Feature Selection to determine the most important features.
 - The general idea is to iteratively add features based on the performance of the KNN method, optimizing for accuracy.
 - Step 1: Run KNN independently for each feature to determine which single feature yields the highest accuracy score. This feature will be selected as feature1.
 - Step 2: Run KNN independently for feature1 combined with each other feature to determine which combination yields the highest accuracy score. The next feature selected in this step will be feature2.
 - Continue iteratively adding features in this way, up to a maximum of 15 features, aiming to reduce the initial 30 features by at least half to meet our business requirements.
 - Based on the results of these iterations, we will identify any features where adding them does not improve the performance metrics, including accuracy.

## Results

Based on the results from applying KNN, adding any features beyond the first nine does not improve any of the provided metrics, including the accuracy score. Please find below these features:
 - area_worst
 - compactness_se
 - concavity_mean
 - perimeter_worst
 - area_se
 - compactness_worst
 - concave_points_se
 - smoothness_worst
 - area_mean.

## Team
**Questions can be submitted to the following people:**
* **Alex**. Emails can be sent to Afeht8642@gmail.com
* **Beth**. Emails can be sent to cwlh07@gmail.com
* **Himeshi**. Emails can be sent to himeshis575@gmail.com
* **Ivan**. Emails can be sent to ivan.makushenko@gmail.com

## Project Folder Structure

Each team is responsible for creating their own Git repository for the Team Project. The following is a good starting point for the folder structure, however the structure is ultimately up to each team. You should structure your project in a way that makes sense for your business case, ensure it is clean, and remove any unused folders.

```markdown
|-- data
|---- processed
|---- raw
|---- sql
|-- experiments
|-- models
|-- reports
|-- src
|-- README.md
|-- .gitignore
```

* **Data:** Contains the raw, processed and final data. For any data living in a database, make sure to export the tables out into the `sql` folder, so it can be used by anyone else.
* **Experiments:** A folder for experiments
* **Models:** A folder containing trained models or model predictions
* **Reports:** Generated HTML, PDF etc. of your report
* **src:** Project source code
* README: This file!
* .gitignore: Files to exclude from this folder, specified by the Technical Facilitator
