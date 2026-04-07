# Customer Churn Data Preprocessing Pipeline

This project focuses on building a data preprocessing pipeline for a customer churn use case. The main objective is to transform raw customer data into a clean and structured dataset that is ready for machine learning tasks.

The pipeline is designed to be simple, reproducible, and easy to extend. It also includes a basic CI setup using GitHub Actions to automatically run preprocessing whenever changes are pushed to the repository.

---

## Overview

The workflow in this project covers:

- Reading raw data
- Validating required fields
- Cleaning and preparing the dataset
- Feature engineering
- Encoding categorical variables
- Scaling numerical features
- Exporting a final dataset ready for modeling

---

## Dataset

This project uses the **Telco Customer Churn dataset**, available on Kaggle:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains customer information such as demographics, services subscribed, billing details, and churn status.

---

## Preprocessing Pipeline

The preprocessing script performs the following steps:

- Drops unnecessary columns (e.g., `customerID`)
- Converts data types where needed (`TotalCharges`)
- Removes missing values and duplicates
- Creates a new feature (`tenure_group`)
- Encodes categorical variables using one-hot encoding
- Converts the target (`Churn`) into a binary format
- Applies standard scaling to numerical features

The final dataset is saved to:
preprocessing/dataset_preprocessing.csv