# Learnable Adaptive Cosine Estimator (LACE) for Image Classification:
**Learnable Adaptive Cosine Estimator (LACE) for Image Classification**

_Joshua Peeples, Connor McCurley, Sarah Walker, Dylan Stewart, and Alina Zare_

Note: If this code is used, cite it: Joshua Peeples, Connor McCurley, Sarah Walker, Dylan Stewart, and Alina Zare. 
(2021, October 5). GatorSense/LACE: Initial Release (Version v1.0). 
Zendo. https://doi.org/10.5281/zenodo.3731417
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3731417.svg)](https://doi.org/10.5281/zenodo.3731417)

[[`arXiv`](https://arxiv.org/abs/2001.00215)]

[[`BibTeX`](#CitingHist)]

In this repository, we provide the paper and code for the Learnable Adaptive Cosine Estimator (LACE) approach in "Learnable Adaptive Cosine Estimator (LACE) for Image Classification."

## Installation Prerequisites

This code uses python, pytorch, and barbar. 
Please use [[`Pytorch's website`](https://pytorch.org/get-started/locally/)] to download necessary packages.
Barbar is used to show the progress of model. Please follow the instructions [[`here`](https://github.com/yusugomori/barbar)]
to download the module.

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. To evaluate performance,
run `View_Results.py` (if results are saved out).

## Main Functions

The Learnable Adaptive Cosine Estimator (LACE) runs using the following functions. 

1. Intialize model  

```model, input_size = intialize_model(**Parameters)```

2. Prepare dataset(s) for model

 ```dataloaders_dict = Prepare_Dataloaders(**Parameters)```

3. Train model 

```train_dict = train_model(**Parameters)```

4. Test model

```test_dict = test_model(**Parameters)```


## Parameters
The parameters can be set in the following script or on the command line:

```Demo_Parameters.py```

## Inventory

```
└── root dir
    ├── demo.py   //Run this. Main demo file.
    ├── Demo_Parameters.py // Parameters file for demo.
    ├── Prepare_Data.py  // Load data for demo file.
    └── Utils  //utility functions
        ├── Embedding_Model.py  // Generates model with an encoder following the final layer (if necessary). 
        ├── loss_functions.py  // Contains LACE, angular softmax, and feature regularization methods for models.
        ├── Loss_Model.py  // Creates model with backbone and regularization loss.
        ├── Network_functions.py  // Contains functions to initialize, train, and test model. 
        ├── pytorchtools.py // Function for early stopping.
        ├── Save_Results.py  // Save results from demo script.
```


