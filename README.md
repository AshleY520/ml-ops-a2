# ML Experiments with Feature Store and Pipeline

*For reproducibility*

#### Technology Stack

- **MLflow 2.x** - Experiment tracking and model registry
- **Feast** - Feature store and versioning
- **CodeCarbon** - Carbon footprint monitoring
- **scikit-learn** - Random Forest implementation
- **Python 3.11.9 - Core development

#### Code

```bash
git clone ml-ops-a2
cd athletes-pipeline
pip install -r requirements.txt

mlflow run . -3 full_pipeline

mlflow ui --port 5000
```

## 1. Overview Results
I have run the four experiments
<img width="1436" height="530" alt="Screenshot 2025-07-14 at 7 00 57 PM" src="https://github.com/user-attachments/assets/a7497e3f-245e-43e2-a4d2-cd72ec4c8129" />

### Model Performance
- **Best Model**: v2_n200_d10 (R^2 = 1, MSE = 13.50)
- **Feature Engineering Impact**: +15 features improved R^2 by 10%
- **Hyperparameter Effect**: 200 trees vs 100 trees improved performance significantly

<img width="1391" height="604" alt="Screenshot 2025-07-14 at 7 01 59 PM" src="https://github.com/user-attachments/assets/b8507887-b339-442a-8de1-c808a9860b3a" />

### Carbon Footprint
- **Total Emissions**: 0.000009 kg CO2
- **Most Efficient**: v2_n200_d10
- **Environmental Impact**: Ultra-low carbon ML pipeline

## 2. Quantitative Analysis
- 4 model configurations compared
- Best: v2_n200_d10 (R^2 = 0.9998)
- Feature engineering impact: v2 (23 features) > v1 (8 features)

### Model Performance Comparison
| Model Configuration | R^2 Score | MSE | Features | Carbon (kg CO2) |
|---------------------|----------|-----|----------|------------------|
| RF_v2_n200_d10 | 1 | 13.5 | 23 | 4.54e-6 |
| RF_v2_n100_d6      | 0.997    | 237.7 | 23 | 1.53e-6 |
| RF_v1_n200_d10     | 0.997    | 205.8 | 8  | 1.31e-6 |
| RF_v1_n100_d6      | 0.989    | 800.7 | 8  | 6.92e-7 |


## Key Insights
1. Feature engineering (v2) dramatically improves model performance
2. 200 estimators vs 100 shows diminishing returns in v1 but significant gains in v2
3. Carbon footprint remains extremely low across all experiments
4. v2_n200_d10 achieves near-perfect prediction (R^2 = 1)


## 2. Qualitative Analysis: Performance comparison plots
### Different hyparameters
- Test R^2 
<img width="1399" height="676" alt="Screenshot 2025-07-14 at 6 46 10 PM" src="https://github.com/user-attachments/assets/8c7b45d4-144b-4aa8-95f4-3d713872e8f9" />
- n_estimators=100 (blue): R^2 range 0.989-0.997, wider variance
- n_estimators=200 (orange): R^2 range 0.997-1.000, tighter distribution, higher median
  
Comment:
a. 200 estimators consistently outperform 100 estimators
b. Large parameter increase (100 to 200) yields modest performance gain (~0.3-0.8%)
c. 200 estimators show less variance in performance

- Test MSE
<img width="1415" height="673" alt="Screenshot 2025-07-14 at 6 49 22 PM" src="https://github.com/user-attachments/assets/029c3480-cd80-4d3c-bf66-95b3b07ceab3" />
- n_estimators=100 (blue): MSE range 237-800, extremely wide variance
- n_estimators=200 (orange): MSE range 13-205, much tighter distribution, lower median
  
Comment:
a. 200 estimators dramatically reduce prediction errors - median MSE drops from ~525 to ~100
b. Large parameter increase (100 to 200) yields substantial error reduction - up to 95% MSE improvement (800 to 13)
c. 200 estimators show significantly less variance in MSE - error consistency improved by ~75%

### Different feature versions
- Test R^2 
<img width="1418" height="664" alt="Screenshot 2025-07-14 at 6 56 42 PM" src="https://github.com/user-attachments/assets/7a0bad7f-e9f4-4a2f-b58c-d0e5596a5004" />
feature_version=v2 (blue): R^2 range 0.997-1.000, tight distribution with high median
feature_version=v1 (orange): R^2 range 0.989-0.997, wider variance, lower median

Comment:
a. v2 features dramatically improve model performance - median R^2 increases from ~0.993 to ~0.9985 
b. Feature engineering (v1 to v2) yields substantial performance gain - up to 1.1% R^2 improvement (0.989→1.000) 
c. v2 features show significantly less variance in R^2 - performance consistency improved by ~75%

- Test MSE
<img width="1421" height="645" alt="Screenshot 2025-07-14 at 6 55 04 PM" src="https://github.com/user-attachments/assets/838f6ed1-6d71-4584-af24-3d6893c7d098" />
- feature_version=v2 (blue): MSE range 13-237, compact distribution with low median
- feature_version=v1 (orange): MSE range 205-800, extremely wide variance, high median

Comment:
a. v2 features dramatically reduce prediction errors - median MSE drops from ~500 to ~125 
b. Feature engineering (v1 to v2) yields substantial error reduction - up to 94% MSE improvement (800 to 13) 
c. v2 features show significantly less variance in MSE - error consistency improved by ~80%

## 3. Carbon Footprint:
- Total: 0.000009 kg CO2

<img width="1394" height="641" alt="Screenshot 2025-07-14 at 6 34 09 PM" src="https://github.com/user-attachments/assets/79c5b55e-e779-4424-a40c-973130359b57" />

### Quantitative analysis:
RF_v2_n200_d10: 4.541e-6 kg CO2  (4.54 μg CO2)
RF_v2_n100_d6:  1.534e-6 kg CO2  (1.53 μg CO2)  
RF_v1_n200_d10: 1.313e-6 kg CO2  (1.31 μg CO2)
RF_v1_n100_d6:  6.919e-7 kg CO2  (0.69 μg CO2)


### Qualitive analysis:

- 1. Feature Version Impact on Carbon Emissions:
<img width="2822" height="1632" alt="image" src="https://github.com/user-attachments/assets/3099836f-9357-4d30-a342-3d9be5e85fde" />
v2 feature group (blue): Emissions range 1.5-4.5 μg CO2, wider distribution
v1 feature group (orange): Emissions range 0.7-1.3 μg CO₂, compact distribution
Per-feature emission efficiency is identical (0.13 μg/feature)
**Feature engineering carbon cost**: Additional 15 features = +2.04 μg CO2

- 2. Hyperparameter Impact on Carbon Emissions:

<img width="1423" height="685" alt="image" src="https://github.com/user-attachments/assets/05bcb5a0-7403-4a5a-a991-52ffa631b956" />

n_estimators=200 (orange): Significantly higher carbon emissions
n_estimators=100 (blue): Relatively lower carbon emissions
100 to 200 estimators** on v1: +90% emissions, +0.8% performance
- **100 to 200 estimators** on v2: +197% emissions, +0.3% performance


