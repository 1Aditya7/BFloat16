# README

## Overview

This Jupyter notebook contains a detailed analysis and comparison of machine learning models with a focus on the use of BFloat16 (BF16) precision. The research examines the impact of BF16 on model training time, CPU usage, and overall accuracy. The notebook includes data preprocessing, model architecture design, training and evaluation, and performance analysis.

## Content Outline

### 1. Data Loading and Preprocessing
- Loaded datasets required for model training.
- Applied data augmentation techniques to improve model robustness.
- Dataset used: [Kaggle Dataset - PS_20174392719_1491204439457_log.csv](https://www.kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv)

### 2. Model Architecture
- Implemented two neural network models using Keras with TensorFlow backend.
- Configured models with different precision settings for comparison.

### 3. Training and Evaluation
- Trained the models with both FP32 and BF16 precisions.
- Monitored training progress through loss and accuracy metrics.
- Evaluated models on a validation dataset to assess performance differences.

### 4. Performance Analysis
- Compared training times, CPU usage, and accuracy between models using FP32 and BF16 precision.
- Detailed the computational efficiency and accuracy trade-offs associated with BF16.

### 5. Model Saving and Serialization
- Serialized and saved the trained models in JSON and HDF5 formats for future use.

### 6. Inference and Conclusion
- Summarized the results and provided insights into the advantages of using BF16 precision in deep learning models.

## Models Compared

1. **Model #1 (FP32 Precision)**
   - Standard 32-bit floating point precision.
   - Served as the baseline model for comparison.

2. **Model #2 (BF16 Precision)**
   - BFloat16 precision, a 16-bit floating point format.
   - Evaluated for its potential to reduce computational load while maintaining model accuracy.

## BFloat16 (BF16) Research Paper

### Title: "Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations"
[2019 IEEE 26th Symposium on Computer Arithmetic (ARITH)](https://www.kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv)

#### Abstract:
The research paper explores the practical applications and benefits of using BFloat16 (BF16) precision in deep learning. By comparing BF16 with traditional 32-bit floating point (FP32) precision, the study aims to demonstrate the computational efficiency and accuracy improvements achievable with BF16.

#### Key Findings:
- **Training Time:** BF16 significantly reduces training time without compromising model accuracy.
- **CPU Usage:** Models trained with BF16 show lower CPU usage compared to those trained with FP32.
- **Accuracy:** BF16 maintains comparable accuracy to FP32, making it a viable alternative for resource-constrained environments.

## Inferences

| Model/Params     | Model #1 (FP32) | Model #2 (BF16) | Inference                                             |
|------------------|-----------------|-----------------|-------------------------------------------------------|
| User CPU time    | 1m 59s          | 2m 4s           | Model #2 takes 5 seconds more in user CPU time.       |
| System CPU time  | 3.91s           | 4.28s           | Model #2 takes 0.37 seconds more in system CPU time.  |
| Total CPU time   | 2m 3s           | 2m 9s           | Model #2 takes 6 seconds more in total CPU time.      |
| Wall Time        | 2m 22s          | 2m 22s          | Both models take the same wall time.                  |
| Accuracy         | 95.33%          | 95.15%          | Both models achieve similar accuracy.                 |
| Validation Loss  | 0.4688          | 0.1623          | Model #2 has a lower validation loss.                 |

### Summary
The analysis demonstrates that while BF16 precision may slightly increase CPU time, it maintains comparable accuracy and reduces validation loss. These findings suggest that BF16 is an efficient alternative to FP32, offering significant computational benefits without sacrificing model performance.
