## Overview
### AdaPPI: Identification of novel protein functional modules via adaptive Graph Convolution Networks in a protein-protein interaction network, 2022
Here we provide the implementation of the AdaPPI method , along with an execution example (on the Krogan-core dataset). The repository is organized as follows:
+ `dataset/` contains the PPI network dataset file;
+ `evaluator/` contains the evaluations of protein functional modules (`compare_performance.py`);
+ `model/` contains:
  + the implementation of the adaptive graph convolution network based on core-attachment method (`adappi_model.py`);
  + the implementation of AGC based on core-attachment method(`protein_agc_model.py`);
  + the implementation of the core-attachment method(`core_attachment.py`);
  + `adappi/` contains:
    + the implementation of self-supervision in loss function (`loss_estimator.py`);
    + the implementation of the adaptive graph convolution network (`repres_learner.py`);
    + preparing to train the adaptive graph convolution network (`trainer.py`);
+ `util/` contains data preprocessing of PPI network dataset (`data_processor.py`);
+ `supplementary_files/` contains the p-value of (predicted but unmatched) potential candidate protein functional modules in two tables calculated by [GOTermFinder](https://go.princeton.edu/cgi-bin/GOTermFinder) tool.

Finally, `model/adappi_model.py` sets all hyperparameters and may be used to execute a full training run on PPI network dataset.

```bash
$ python model/adappi_model.py
```
## Architecture

![image](https://github.com/aI-area/AdaPPI/blob/main/framework.png)


## Dataset
| Type | Dataset | Protein Nodes | Interaction Edges | GO Attributes | Average Degree | Max Degree|
| :--: | :--: | :---: | :---: | :------: | :-----: | :-----: |
|C| Collins | 1,622 | 9,074 |  141   |    11.189    |127|
|C| Krogan-core | 2,708 | 7,123 | 143  |    5.261    | 141|
|C| Krogan14k | 3,581 | 14,076 | 143  |    7.861    |213|
|C| DIP | 4,928 | 17,201 |  143   |    6.981    |283|
|C| Biogrid | 5,640 | 59,748 |  143   |    21.187    |2,570|
|P| Humancyc | 1,648 | 53,122 |  151   |    32.234    |336|
|P| Panther | 1,802 | 36,956 |  142   |    20.508    |234|
|P| PID | 2,376 | 39,772 |  141   |    16.739    |226|
|Both| SGD | 5,808 | 454,396 |  181   |    78.326    |2,585|
#### Notes: C for only complex; P for only pathway; Both for complex & pathway.

## Dependencies
The script has been tested running under Python 3.7.9, with the following packages installed (along with their dependencies):
+ `numpy==1.20.1`
+ `scipy==1.6.0`
+ `tensorflow-gpu==1.14.0`
+ `scikit-learn==0.22.1`
+ `pandas==1.2.2`
+ `networkx==2.5.1`


## References
You may also be interested in the related articles：

+ Smoothness Sensor: Adaptive Smoothness-Transition Graph Convolutions for Attributed Graph Clustering [NASGC](https://github.com/aI-area/NASGC) ([DOI](https://doi.org/10.1109/tcyb.2021.3088880))
+ Attributed Graph Clustering via Adaptive Graph Convolution [AGC](https://github.com/karenlatong/AGC-master) 
+ Accelerated Attributed Network Embedding [AANE](https://github.com/xhuang31/AANE_Python)
+ Protein Complexes Identification Based on GO Attributed Network Embedding [GANE](https://github.com/LiKun-DLUT/GANE)
+ Text-associated DeepWalk [TADW](https://github.com/benedekrozemberczki/TADW)
+ Text-associated DeepWalk-Spectral Clustering [TADW-SC](https://github.com/kamalberahmand/TADW-SC)

## License

MIT