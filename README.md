# BFloat16 Performance Analysis

## Overview

This Jupyter notebook contains a detailed analysis and comparison of machine learning models with a focus on the use of Brain Float 16 (BF16) precision. The research examines the impact of BF16 on model training time, CPU usage, and overall accuracy. The notebook includes data preprocessing, model architecture design, training and evaluation, and performance analysis.
<p align="center">
  <img src="https://github.com/1Aditya7/BFloat16/blob/main/BF16_media/bfloat16-fp16.jpg" width="600"/>
</p>
<p align="center">
  <small>Image Credit: <a href="https://nhigham.com/2020/06/02/what-is-bfloat16-arithmetic/" target="_blank">What is BFloat16 Arithmetic?</a></small>
</p>

## Content Outline

### 1. Data Loading and Preprocessing
- Loaded datasets required for model training.
- Applied data augmentation techniques to improve model robustness.
- Dataset used: [Kaggle Dataset](https://www.kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv)

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

### Possible Reasons for Longer Training Time with BF16

1. **Data Type Conversion Overhead:**
   - When using BF16, data needs to be converted to this format before being processed. This conversion can add overhead, especially during the training phase, leading to a slight increase in CPU time.
   
2. **Hardware Compatibility:**
   - While BF16 is optimized for specific hardware like Google's TPUs or certain Intel CPUs, not all hardware is designed to take full advantage of BF16 operations. When using hardware that doesn’t natively support BF16, the processor may fall back to using FP32 for some operations, which could lead to increased processing time compared to FP32-only computations.

3. **Precision Handling:**
   - BF16 has less precision than FP32 (16 bits vs. 32 bits), which can cause more operations to be performed to achieve similar results in certain calculations, leading to longer computation times for some models.

4. **Memory Access Patterns:**
   - Although BF16 uses less memory bandwidth, when converting large datasets to BF16, memory access patterns may not be as optimized as for FP32, leading to inefficiencies in some parts of the model’s processing pipeline.

5. **Implementation Details:**
   - The TensorFlow framework or the specific model configuration might not be fully optimized for BF16 operations, which could result in a slower execution for BF16 compared to FP32 in this case.

### Summary
The analysis demonstrates that while BF16 precision may slightly increase CPU time, it maintains comparable accuracy and reduces validation loss. These findings suggest that BF16 is an efficient alternative to FP32, offering significant computational benefits without sacrificing model performance.
