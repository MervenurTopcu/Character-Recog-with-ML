# Character Recognition Using Machine Learning With EMNIST Dataset
 
 
 <h3>WHAT IS EMNIST DATASET?</h3>
 
 The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19 and converted to a 28x28 pixel image format and dataset structure that   directly matches the MNIST dataset. Further information on the dataset contents and conversion process can be found in the paper available at https://arxiv.org/abs/1702.05373v1.
 
 <h3>FORMAT</h3>
 
 There are six different splits provided in this dataset and each are provided in two formats:

1. Binary (see emnistsourcefiles.zip)
2. CSV (combined labels and images)
    - Each row is a separate image
    - 785 columns
    - First column = class_label (see mappings.txt for class label definitions)
    - Each column after represents one pixel value (784 total for a 28 x 28 image)

<h3>Balanced Dataset</h3>
The EMNIST Balanced dataset is meant to address the balance issues in the ByClass and ByMerge datasets. It is derived from the ByMerge dataset to reduce mis-classification errors due to capital and lower case letters and also has an equal number of samples per class. This dataset is meant to be the most applicable.

- train: 112,800
- test: 18,800
- total: 131,600
- classes: 47 (balanced)

![resim](https://user-images.githubusercontent.com/37351206/156732147-e632b8aa-d679-4dea-b700-33560f8908b6.png)

- 📫 Project is available on Kaggle [https://www.kaggle.com/mervenurtopcu/emnist-dataset](https://www.kaggle.com/mervenurtopcu/emnist-dataset)
