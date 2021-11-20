# Multiclass-Classification
Implementation of couple of machine learning algorithms for classification on the iris flowers data set.

1. [General](#General)
    - [Background](#background)
    - [Program Structure](https://github.com/tomershay100/Multiclass-Classification/blob/main/README.md#program-structure)
    - [About The Output File](https://github.com/tomershay100/Multiclass-Classification/blob/main/README.md#about-the-output-file)
    - [Running Instructions](https://github.com/tomershay100/Multiclass-Classification/blob/main/README.md#running-instructions)
2. [Dependencies](#dependencies) 
3. [Installation](#installation)

## General

### Background
Implementation of machine learning algorithms for the following multiclass classification: KNN, Perceptron, SVM and PA. The algorithms learn about a training set of iris flowers, and allow prediction of new flowers and belonging to their group.

There are three types of iris flowers in the training set, labeled by the numbers 0,1 and 2.

Each flower has five parameters (represented by float numbers). The various algorithms try to study the commonality of each type of flower, thus matching the flowers obtained in the test set to the type of flower that best suits it.

#### How does the algorithms work briefly?
**KNN Algorithm:** The algorithm predicts the flowers in the test set by the K flowers closest to it (by Euclidean distance).
***
**Perceptron Algorithm:** The algorithm multiplies each feature of the flower by an array of weights. Each labeling of the flowers has different weights, and the multiplication is done with each of them. The labeling chosen for a flower in the test set was chosen according to which weight resulted in a maximum value. When there is an error in the prediction of the training set, the weights change as follows:

<img src="https://render.githubusercontent.com/render/math?math=e^{i +\pi} =x%2B1">

![formula](https://render.githubusercontent.com/render/math?math=\color{white}\large\f(x)=sin(x))

```
insert an image
```
***
**SVM (Support Vector Machine) Algorithm:** Similar to the previous algorithm, the algorithm multiplies each feature of the flower by an array of weights. Each labeling of the flowers has different weights, and the multiplication is done with each of them. The labeling chosen for a flower in the test set was chosen according to which weight resulted in a maximum value. The difference between the two is the way of learning - in both algorithms, whenever there is an error in the labeling in the training set, the algorithm changes the weights that caused the error. In SVM the change happens as follows:
```
insert an image
```
***
**PA (Passive Aggressive) Algorithm:** Perfectly similar to Perceptron except for the weight change process performed as follows:
```
insert an image
```
