### codes and data 

```bash
.
├── Data-split
│   ├── data_split.py
│   ├── ictcf
│   │   ├── nCT
│   │   │   ├── test_nCT.txt
│   │   │   ├── train_nCT.txt
│   │   │   └── val_nCT.txt
│   │   └── pCT
│   │       ├── test_pCT.txt
│   │       ├── train_pCT.txt
│   │       └── val_pCT.txt
│   └── UCSD
│       ├── COVID
│       │   ├── testCT_COVID.txt
│       │   ├── trainCT_COVID.txt
│       │   └── valCT_COVID.txt
│       └── NonCOVID
│           ├── testCT_NonCOVID.txt
│           ├── trainCT_NonCOVID.txt
│           └── valCT_NonCOVID.txt
├── model_backup
│   ├── ictcf
│   │   ├── DenseNet.pt
│   │   └── mp_DenseNet.pt
│   └── UCSD
│       ├── DenseNet.pt
│       └── mp_DenseNet.pt
├── model_code
│   ├── ictcf.py
│   └── UCSD.py
├── model_result
│   ├── ictcf
│   │   ├── DenseNet.txt
│   │   ├── mp_DenseNet.txt
│   │   ├── result_vis.R
│   │   ├── test_DenseNet.txt
│   │   └── test_mp_DenseNet.txt
│   └── UCSD
│       ├── DenseNet.txt
│       ├── mp_DenseNet.txt
│       ├── result_vis.R
│       ├── test_DenseNet.txt
│       └── test_mp_DenseNet.txt
├── notebook
│   ├── ictcf.ipynb
│   └── UCSD.ipynb
├── README.md
├── requirments.txt
└── riskscore
    ├── ictcf
    │   ├── risk_score_nCT.txt
    │   ├── risk_score_pCT.txt
    │   └── riskscore.py
    ├── riskscore.R
    └── UCSD
        ├── risk_score_nCT.txt
        ├── risk_score_pCT.txt
        └── riskscore.py

```

These are the codes and results for the project using 2D matrix profile algorithm to do COVID-19 CT imaging anomaly detection. I used **Python** to implement the 2D matrix profile algorithm and to build the model. I used **R** to visualize the results and test the risk score. 

