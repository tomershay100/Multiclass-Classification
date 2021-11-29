
# Multiclass-Classification
Implementation of couple of machine learning algorithms for classification on the iris flowers data set.

1. [General](#General)
    - [Background](#background)
    - [Program Structure](https://github.com/tomershay100/Multiclass-Classification/blob/main/README.md#program-structure)
    - [About The Output File](https://github.com/tomershay100/Multiclass-Classification/blob/main/README.md#about-the-output-file)
    - [Running Instructions](https://github.com/tomershay100/Multiclass-Classification/blob/main/README.md#running-instructions)
2. [Dependencies](#dependencies) 
3. [Installation](#installation)
4. [Footnote](#footnote)

## General

### Background
Implementation of the following machine learning algorithms for multiclass classification: ```KNN```, ```Perceptron```, ```SVM``` and ```PA```. The algorithms learn a training set of iris flowers, and make predictions of new set of flowers (test set).

There are three types of iris flowers in the training set, labeled by the numbers ``0``, ``1`` and ``2``.

Each flower has five parameters (represented by float numbers). The various algorithms try to learn the commonality of each type of flower, thus matching the flowers in the test set to the type of flower that best suits it.

#### About The Algorithms
* ```KNN Algorithm```: The algorithm predicts the flowers in the test set by the K flowers closest to it (by Euclidean distance).
	***
* ```Perceptron Algorithm:``` The algorithm multiplies each feature of the flower by an array of weights. Each label of the flowers has different weights, and the multiplication is done with each of them. The prediction that chosen for a flower in the test set was chosen according to which weight resulted in a maximum value. When there is an error in the prediction of the training set, the weights change as follows:

    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{w^y_{t%2B1}=w^y_{t}%2B\eta%20\cdot%20x})

    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{w^\hat{y}_{t%2B1}=w^{\hat{y}}_{t}-\eta%20\cdot%20x})

    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{w^{i\ne%20\hat{y},y}_{t%2B1}=w^{i\ne%20\hat{y},y}_{t}})

	***
* ```SVM (Support Vector Machine) Algorithm:``` Similar to the previous algorithm, the algorithm multiplies each feature of the flower by an array of weights. Each label of the flowers has different weights, and the multiplication is done with each of them. The prediction for each flower in the test set was chosen according to which weight resulted in a maximum value. The difference between the two is the way of learning -- in both algorithms, whenever there is an error in the labeling in the training set, the algorithm changes the weights that caused the error. In ```SVM``` the change happens as follows:

    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{w^y_{t%2B1}=\left(1-\eta\lambda\right)\cdot%20w^y_{t}%2B\eta%20\cdot%20x})
    
    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{w^\hat{y}_{t%2B1}=\left(1-\eta\lambda\right)\cdot%20w^\hat{y}_{t}-\eta%20\cdot%20x})
    
    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{w^{i\ne%20\hat{y},y}_{t%2B1}=\left(1-\eta\lambda\right)\cdot%20w^{i\ne%20\hat{y},y}_{t}})
    

	***
* ```PA (Passive Aggressive) Algorithm:``` Perfectly similar to Perceptron except for the weight change process performed as follows:

    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{w^y_{t%2B1}=w^y_{t}%2B\tau%20\cdot%20x})
    
    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{w^\hat{y}_{t%2B1}=w^{\hat{y}}_{t}-\tau%20\cdot%20x})
    
    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{w^{i\ne%20\hat{y},y}_{t%2B1}=w^{i\ne%20\hat{y},y}_{t}})
    
    ![formula](https://render.githubusercontent.com/render/math?math=\color{gray}\large{\text{where%20}\tau\text{is%20set%20to:%20%20}\tau=\frac{\ell\left(w,x,y\right)}{2\cdot\||x||^2}})
 
### Program Structure
The code is divided into 4 main functions. In fact, one function per algorithm. Given the training and test files, normalization is performed according to zscore normalization and feature selection is performed too. Then, the four algorithms are called and each in turn returns a list of predictions for each point in the test set. The program exports all the lists to an output file whose contents are briefly explained in the next section.

### About The Output File
All the predictions of each of the algorithms are exported to an output file whose name is received as input to the program. The file contains the data as follows:
```
knn: 0, perceptron: 0, svm: 0, pa: 1
knn: 1, perceptron: 2, svm: 1, pa: 1
knn: 2, perceptron: 2, svm: 2, pa: 1
...
```

### Running Instructions
In order to run the program, it must provide 4 arguments:
* First, a path to the ```txt``` file that contains the training data set.
* Second, a path to the ```txt``` file that contains the labels of each row in the training data set.
* Third, a path to the ```txt``` file that contains the test set.
* Fourth, a path to the output file to which the predictions of each of the algorithms will be written.

The training data set file should contain a number of lines, so that each line consists of 5 floating numbers separated by commas. So is the test set file. The label file consists of the same number of rows of the training set file and contains one numeric value (between 0 and 2) in each row.

## Dependencies
* [Python 3.6+](https://www.python.org/downloads/)
* Git
* [NumPy](https://numpy.org/install/)

## Installation

1. Open the terminal
2. Clone the project by:
	```
	$ git clone https://github.com/tomershay100/Multiclass-Classification.git
	```	
3. Run the ```main.py``` file:
	```
	$ python3 main.py train_x.txt train_y.txt test_x.txt output.txt
	 ```
	 
	 
## Footnote:
As you can see, there are a number of additional files. The files contain different graphs of different experiments in the program (for example, changing the value of the learning rate or changing the feature that is downloaded, etc.). In addition, it contains a report in the Hebrew language that explains the implementation of the algorithms, the experiments performed and the graphs.	 

In addition, there is also the label file of the test file, called test_y.txt. You can run the algorithm and check my accuracy percentages (which stand at ```96.66%``` in KNN, ```94.86%``` in perceptron, ```96.26%``` in SVM and ```94.25%``` in PA).
