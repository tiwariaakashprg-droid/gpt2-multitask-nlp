# TMLR Statistical Robustness Report

This report is generated from the supplied dataset; no result values are hard-coded.

## Human-human agreement

- Annotators: 9
- Mean pairwise Pearson r: **0.967**
- Pairwise Pearson range: 0.939 to 0.990
- ICC(2,k), absolute agreement: **0.996**

## Response-length analysis

- Length unit: **characters**
- Mean response length: **129.59 characters**
- Median response length: **77.00 characters**
- Range: 3 to 365 characters

## Human vs llama3_mean

- n = 300
- Pearson r = **0.275**, p = 1.344e-06, 95% bootstrap CI = [0.145, 0.405]
- Spearman rho = **0.258**, p = 5.803e-06, 95% bootstrap CI = [0.145, 0.367]
- MAE = **27.71**, 95% bootstrap CI = [25.67, 29.89]
- Signed bias (judge - human) = **+23.04**, 95% bootstrap CI = [+20.30, +25.72]

### Response-length analysis

- Human score vs response length: r = 0.159 (p=0.00574), 95% CI [0.050, 0.290]
- llama3_mean score vs response length: r = 0.091 (p=0.1172), 95% CI [-0.001, 0.182]
- Human vs llama3_mean partial Pearson r controlling for response length: **0.265**, 95% bootstrap CI = [0.137, 0.391]

### Self-consistency

- Mean item SD across repeats: 0.277
- Exact consistency across repeats: 97.3%

### Decoding Config

- **high_temp** (n=100): r=0.264, 95% CI [0.057, 0.462], MAE=24.33
- **low_temp** (n=100): r=0.165, 95% CI [-0.051, 0.372], MAE=31.97
- **mid_temp** (n=100): r=0.372, 95% CI [0.152, 0.585], MAE=26.81

### Category

- **factual** (n=60): r=0.592, 95% CI [0.359, 0.762], MAE=25.79
- **instruction_following** (n=60): r=0.462, 95% CI [0.218, 0.679], MAE=24.64
- **math** (n=60): r=0.026, 95% CI [-0.057, 0.110], MAE=35.95
- **reasoning** (n=60): r=0.076, 95% CI [-0.117, 0.282], MAE=25.52
- **writing** (n=60): r=0.156, 95% CI [-0.103, 0.404], MAE=26.64

## Human vs qwen2_5_7b_mean

- n = 300
- Pearson r = **0.340**, p = 1.431e-09, 95% bootstrap CI = [0.178, 0.491]
- Spearman rho = **0.303**, p = 9.026e-08, 95% bootstrap CI = [0.194, 0.405]
- MAE = **18.64**, 95% bootstrap CI = [16.96, 20.45]
- Signed bias (judge - human) = **+9.54**, 95% bootstrap CI = [+7.04, +12.00]

### Response-length analysis

- Human score vs response length: r = 0.159 (p=0.00574), 95% CI [0.050, 0.290]
- qwen2_5_7b_mean score vs response length: r = -0.108 (p=0.06285), 95% CI [-0.195, -0.015]
- Human vs qwen2_5_7b_mean partial Pearson r controlling for response length: **0.364**, 95% bootstrap CI = [0.202, 0.511]

### Self-consistency

- Mean item SD across repeats: 0.395
- Exact consistency across repeats: 92.3%

### Decoding Config

- **high_temp** (n=100): r=0.184, 95% CI [-0.001, 0.400], MAE=14.74
- **low_temp** (n=100): r=0.314, 95% CI [0.016, 0.561], MAE=22.06
- **mid_temp** (n=100): r=0.436, 95% CI [0.152, 0.666], MAE=19.13

### Category

- **factual** (n=60): r=0.678, 95% CI [0.403, 0.848], MAE=19.46
- **instruction_following** (n=60): r=0.154, 95% CI [-0.059, 0.378], MAE=16.39
- **math** (n=60): r=0.506, 95% CI [-0.041, 0.800], MAE=17.32
- **reasoning** (n=60): r=0.130, 95% CI [-0.030, 0.270], MAE=18.34
- **writing** (n=60): r=0.132, 95% CI [-0.146, 0.397], MAE=21.71

## Judge-vs-judge agreement

- **llama3_mean_vs_qwen2_5_7b_mean**: Pearson r=0.531, Spearman rho=0.481, MAE=16.86
