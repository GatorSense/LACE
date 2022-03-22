# Learnable Adaptive Cosine Estimator (LACE) for Image Classification:
**Learnable Adaptive Cosine Estimator (LACE) for Image Classification**

_Joshua Peeples, Connor McCurley, Sarah Walker, Dylan Stewart, and Alina Zare_

Note: If this code is used, cite it: Joshua Peeples, Connor McCurley, Sarah Walker, Dylan Stewart, and Alina Zare. 
(2021, October 15). GatorSense/LACE: Initial Release (Version v1.0). 
Zendo. https://doi.org/10.5281/zenodo.5572704
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5572704.svg)](https://doi.org/10.5281/zenodo.5572704)

[[`IEEE Xplore`](https://ieeexplore.ieee.org/document/9706894)]

[[`WACV Repository`](https://openaccess.thecvf.com/content/WACV2022/html/Peeples_Learnable_Adaptive_Cosine_Estimator_LACE_for_Image_Classification_WACV_2022_paper.html)]

[[`arXiv`](https://arxiv.org/abs/2110.05324)]

[[`BibTeX`](#CitingLACE)]

In this repository, we provide the paper and code for the Learnable Adaptive Cosine Estimator (LACE) approach in "Learnable Adaptive Cosine Estimator (LACE) for Image Classification."

## Installation Prerequisites

This code uses python, pytorch, and barbar. 
Please use [[`Pytorch's website`](https://pytorch.org/get-started/locally/)] to download necessary packages.
Barbar is used to show the progress of model. Please follow the instructions [[`here`](https://github.com/yusugomori/barbar)]
to download the module.

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. 

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
The parameters can be set in the following script:

```Demo_Parameters.py```

## Inventory

```
https://github.com/GatorSense/LACE

└── root dir
    ├── demo.py   //Run this. Main demo file.
    ├── Demo_Parameters.py // Parameters file for demo.
    ├── Prepare_Data.py  // Load data for demo file.
    └── Utils  //utility functions
        ├── Embedding_Model.py  // Generates model with an encoder following the final layer (if necessary). 
        ├── loss_functions.py  // Contains LACE, angular softmax, and feature regularization methods for models.
        ├── Loss_Model.py  // Creates model with backbone and regularization loss.
        ├── Generating_Learning_Curves.py  // Plot training and validation accuracy and error measures.
        ├── Generate_TSNE_visual.py  // Create TSNE visual for results.
        ├── Network_functions.py  // Contains functions to initialize, train, and test model. 
        ├── pytorchtools.py // Function for early stopping.
        ├── Save_Results.py  // Save results from demo script.
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) 
file in the root directory of this source tree.

This product is Copyright (c) 2021 J. Peeples, C. McCurley, S. Walker, D. Stewart, 
and A. Zare. All rights reserved.

## <a name="CitingLACE"></a>Citing LACE

If you use the LACE code, please cite the following 
reference using the following entry.

**Plain Text:**

J. Peeples, C. McCurley, S. Walker, D. Stewart, and A. Zare, 
"Learnable Adaptive Cosine Estimator (LACE) for Image Classification," 
In Proceedings of the IEEE/CVF Winter Conference on Applications of 
Computer Vision (WACV), 2022, pp. 3479-3489, In Press.

**BibTex:**
```
@InProceedings{Peeples_2022_WACV,
  title = {Learnable Adaptive Cosine Estimator (LACE) for Image Classification},
  author = {Peeples, Joshua and McCurley, Connor and Walker, Sarah, and Stewart, Dylan, and Zare, Alina},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month = {January},
  year = {2022},
  pages = {3479-3489}
}
```
