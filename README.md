# Lab 2: Association Rules & Clustering — Adult Census Income

## Dataset
Adult Census Income (`fetch_openml(name="adult", version=2)`) — 48,842 records.

## Selected Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| age | Continuous (17-90) | Age of respondent |
| education-num | Ordinal (1-16) | Education level (1=Preschool, 9=HS-grad, 13=Bachelors, 16=Doctorate) |
| hours-per-week | Continuous (1-99) | Weekly working hours |
| income | Binary | <=50K or >50K annual income |

## Notebooks
| # | File | Description |
|---|------|-------------|
| 1 | `01_preprocessing.ipynb` | Data loading, cleaning, EDA, discretization |
| 2 | `02_apriori.ipynb` | Apriori algorithm for association rules |
| 3 | `03_fp_growth.ipynb` | FP-Growth algorithm + comparison with Apriori |
| 4 | `04_kmeans.ipynb` | k-Means clustering |
| 5 | `05_kmedians.ipynb` | k-Medians clustering (pyclustering) |
| 6 | `06_hierarchical.ipynb` | Hierarchical Agglomerative clustering |
| 7 | `07_knn_clustering.ipynb` | KNN-based clustering |
| 8 | `08_dbscan.ipynb` | DBSCAN density-based clustering |
| 9 | `09_comparison.ipynb` | Comparative analysis of all algorithms |

## Setup
```bash
pip install -r requirements.txt
```

Run notebooks sequentially: `01_preprocessing.ipynb` first, then any others.
