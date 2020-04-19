### Codes and data 

```bash
iMP
├── Data-split
│   ├── COVID
│   │   ├── testCT_COVID.txt
│   │   ├── trainCT_COVID.txt
│   │   └── valCT_COVID.txt
│   └── NonCOVID
│       ├── testCT_NonCOVID.txt
│       ├── trainCT_NonCOVID.txt
│       └── valCT_NonCOVID.txt
├── Images-processed
│   ├── CT_COVID.zip
│   └── CT_NonCOVID.zip
├── iMP_DenseNet.ipynb
├── iMP_DenseNet.py
├── model_backup
│   ├── DenseNet.pt
│   └── mp_DenseNet.pt
├── model_result
│   ├── DenseNet.txt
│   ├── mp_DenseNet.txt
│   ├── result_vis.R
│   ├── test_DenseNet.txt
│   ├── test_mp_DenseNet.txt
│   └── validation.png
├── README.md
└── requirments.txt
```

**Data-split** consists 6 txt files which indicating the name of the images used in training, valiadation, and testing steps in this study. The split is provided by the source of the the [raw data](https://github.com/UCSD-AI4H/COVID-CT).

**Images-processed** is storing the zipped raw data.

**iMP_denseNet.ipynb** is a Jupyter notebook which is embeded with some results for better visualization without actually run the code by yourself. Please note these results are just simple illustrations. The actual training had more epoches.

**iMP_denseNet.py** is the main code. Several requirments are needed for running this code. And it took around 10 hours in a GPU machine.

**model_backup** is storing the model architecture and weights. *DenseNet.pt* is the baseline model. *mp_DenseNet.pt* is the proposed one.

**model_result** is storing the excuted results of [iMP_denseNet.py](https://github.com/qianliu1219/iMP/blob/master/iMP_DenseNet.py) and the visualization of the models' performance on valiadation dataset([validation.png](https://github.com/qianliu1219/iMP/blob/master/model_result/validation.png)) generated using [result.R](https://github.com/qianliu1219/iMP/blob/master/model_result/result_vis.R). 

***Hint:*** Please check the Jupyter notebook [iMP_DenseNet.ipynb](https://github.com/qianliu1219/iMP/blob/master/iMP_DenseNet.ipynb) for visualizing the outputs without running the code by your self. If you want to run the code [iMP_denseNet.py](https://github.com/qianliu1219/iMP/blob/master/iMP_DenseNet.py), please make sure all [requirments](https://github.com/qianliu1219/iMP/blob/master/requirments.txt) are installed and the data is unzipped. 
