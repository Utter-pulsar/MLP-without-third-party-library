# MLP-without-third-party-library
The whole derivation of MLP with both sigmoid and softmax activation function.

> This is a project structuring MLP with MSE and cross entropy loss function. This is an example of three layers MLP with one input layer, one hidden layer and one 
> output layer.

Sources
----
The datasets refer to [Fashion-Mnist](https://github.com/zalandoresearch/fashion-mnist).



Requirements
------------

Python 3.7.0

numpy 1.19.0

tqdm 4.36.1

Guidance
-----

Datasets are in the ```\data``` file. The exact file can be found in the [Fashion-Mnist](https://github.com/zalandoresearch/fashion-mnist), which is also shown in above.

Set Code Parameters
-------

In ```MLP_Sigmoid.py``` and ```MLP_Softmax.py```, the parameters can be changed from codes shown as below:

```python
# Pre-defined parameters
pre_trained = False             # pre_trained dataset
lay_1 = 30                      # number of nodes of the hidden layer
lr = 0.01                       # learning rate
batchsize = 8                   # batchsize
epoch = 50                      # epoch numbers
is_save = True                  # save the weights
```

"Pre_trained" means there is a ```sigmoid.npy``` or ```softmax.npy``` in the root directory.

"is_save" asks if you want to save the trained weights after training.

Theory Derivation
-------

The whole theory derivation will be illustrated in ```theory.pdf``` file.
