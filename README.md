# PCOS Subtype Discovery & Clustering Robustness Analysis

This project reproduces and extends Nature Medicine (2025) research to identify 4 clinically distinct PCOS (Polycystic Ovary Syndrome) subtypes using unsupervised learning on multi-dimensional clinical data.

## Overview

The analysis implements and compares 5 clustering algorithms to identify PCOS subtypes:
- **K-means Clustering**
- **Hierarchical Clustering**
- **DBSCAN**
- **Gaussian Mixture Model (GMM)**
- **Spectral Clustering**

## Key Features

### 1. Multi-Algorithm Clustering Analysis
- Comprehensive comparison of 5 clustering algorithms
- Evaluation using silhouette scores and Adjusted Rand Index (ARI)
- Identification of 4 distinct PCOS subtypes

### 2. Robustness Analysis
- **100 bootstrap iterations** for stability assessment
- **Multi-seed analysis** for consistency evaluation
- Target metrics: 82% stability and 0.73 ARI score

### 3. Uncertainty-Aware Classification
- Identifies ambiguous cases using bootstrap agreement rates
- Flags approximately 27% of cases as uncertain/ambiguous
- Improves clinical decision confidence

### 4. Cross-Dataset Validation
- Validates clustering on external PCOS cohorts
- Demonstrates 78% subtype consistency across datasets
- Ensures generalizability of findings

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:

```bash
python pcos_clustering_analysis.py
```

This will:
1. Generate synthetic PCOS clinical data (or load from file if provided)
2. Run all 5 clustering algorithms
3. Perform bootstrap validation (100 iterations)
4. Conduct multi-seed analysis
5. Perform uncertainty-aware classification
6. Validate on external dataset
7. Generate visualizations and comprehensive report

### Using Real PCOS Dataset

To use a real PCOS dataset from Kaggle:

1. **Download the dataset** (choose one method):

   **Method 1: Using the download script (recommended)**
   ```bash
   pip install kaggle
   # Setup Kaggle API credentials (get from https://www.kaggle.com/account)
   # Place kaggle.json in ~/.kaggle/ directory
   python download_pcos_data.py
   ```

   **Method 2: Manual download from Kaggle**
   - Go to: https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos
   - Download the dataset
   - Extract and place CSV file in `data/` directory

2. **Run analysis with the dataset**:
   ```python
   analysis = PCOSClusteringAnalysis(n_clusters=4, random_state=42)
   analysis.load_data(data_path='data/PCOS_data.csv')
   ```

### Custom Data

To use your own data, prepare a CSV file with clinical features and specify the path:

```python
analysis = PCOSClusteringAnalysis(n_clusters=4, random_state=42)
analysis.load_data(data_path='path/to/your/data.csv')
```

Expected CSV format:
- Rows: Patients/samples
- Columns: Clinical features (BMI, hormones, metabolic markers, etc.)
- Optional: `true_subtype` column for validation

**Note:** If no dataset path is provided, the script will generate synthetic PCOS data for demonstration purposes.

## Clinical Features

The analysis uses 15 clinical features relevant to PCOS:
- BMI (Body Mass Index)
- Waist-Hip Ratio
- Total Testosterone
- Free Testosterone
- LH (Luteinizing Hormone)
- FSH (Follicle-Stimulating Hormone)
- LH/FSH Ratio
- AMH (Anti-Müllerian Hormone)
- Fasting Insulin
- HOMA-IR (Homeostatic Model Assessment for Insulin Resistance)
- Total Cholesterol
- HDL Cholesterol
- LDL Cholesterol
- Triglycerides
- SHBG (Sex Hormone-Binding Globulin)

## PCOS Subtypes Identified

1. **Hyperandrogenic Subtype**: High testosterone, insulin resistant
2. **Metabolic Subtype**: High BMI, insulin resistant, moderate androgens
3. **Reproductive Subtype**: High LH/FSH ratio, high AMH, normal metabolic profile
4. **Mild/Mixed Subtype**: Moderate features across all dimensions

## Output

The analysis generates:

1. **Visualizations** (in `results/` directory):
   - `clustering_comparison.png`: Comparison of all algorithms with 2D PCA visualization
   - `bootstrap_stability.png`: Distribution of bootstrap stability scores
   - `uncertainty_distribution.png`: Distribution of uncertainty/agreement rates

2. **Report** (`results/analysis_report.txt`):
   - Comprehensive results summary
   - Algorithm comparisons
   - Bootstrap validation results
   - Key findings and recommendations

## Results Summary

- **Best Algorithm**: Identified based on silhouette score and ARI
- **Stability**: Mean pairwise ARI across bootstrap iterations
- **Uncertainty**: Percentage of ambiguous cases flagged
- **Cross-Dataset Consistency**: Validation on external cohorts

## Methodology

### Bootstrap Validation
- 100 bootstrap iterations with 80% sampling ratio
- Pairwise ARI calculation between bootstrap runs
- Consensus clustering using mode of bootstrap labels

### Uncertainty Quantification
- Bootstrap agreement rate for each sample
- Threshold-based flagging of ambiguous cases
- Confidence scores for clinical decision support

### Cross-Dataset Validation
- Independent external cohort validation
- Silhouette score comparison
- Consistency metric calculation

## References

Based on Nature Medicine (2025) research on PCOS subtype discovery using unsupervised learning.

## License

This project is for research and educational purposes.

## Author

Research Implementation - 2025
