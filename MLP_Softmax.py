import os
import math
import gzip
import numpy as np
from tqdm import tqdm

# Define the weights matrix
class weights():
    def __init__(self, lw1, lw2, lb1, lb2):
        self.lw1 = lw1
        self.lw2 = lw2
        self.lb1 = lb1
        self.lb2 = lb2

# Define the copy of weights for calculation
class weights_tem():
    def __init__(self, lw1, lw2, lb1, lb2):
        self.lw1 = lw1
        self.lw2 = lw2
        self.lb1 = lb1
        self.lb2 = lb2

# Sigmoid activation function
def sigmoid_function(input):
    fz = []
    input = input.tolist()
    for num in input:
        if num >= 0:
            fz.append(1.0 / (1 + math.exp(-num)))
        else:
            fz.append(math.exp(num) / (1 + math.exp(num)))
    output = np.array(fz)
    return output

# Softmax activation function
def softmax_function(input):
    max_input = np.max(input)
    input = np.exp(input-max_input)
    sum_input = np.sum(input)
    output = input/sum_input
    return output

# Calculate the mean value for a batch
def mean_calculator(input):
    for i, k in enumerate(input):
        if i == 0:
            a = k
        else:
            a += k
    return a/len(input)

# Define how to load the dataset
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

# Define the forward process
def forward(weight, data):
    # print(weights.lw1)
    layer_1 = weight.lw1.dot(data)+weight.lb1
    layer_1 = sigmoid_function(layer_1)
    layer_2 = weight.lw2.dot(layer_1)+weight.lb2
    layer_2 = softmax_function(layer_2)
    return layer_1, layer_2

# Define the backward process and return the updated weights
def backward(weight, input, layer_1, layer_2, target, lr):

    sigma_out = layer_2-target
    sigma_out_reshape = sigma_out[:, None]
    y = layer_1.reshape((1, len(layer_1)))
    sigma_out_y = sigma_out_reshape.dot(y)
    weight.lw2 = weight.lw2-lr*sigma_out_y

    weight.lb2 = weight.lb2-lr*sigma_out

    sigma_layer = weight.lw2.T.dot(sigma_out)*(layer_1*(1-layer_1))
    sigma_layer_reshape = sigma_layer[:, None]
    y1 = input.reshape((1, len(input)))
    sigma_layer_y = sigma_layer_reshape.dot(y1)
    weight.lw1 = weight.lw1-lr*sigma_layer_y

    weight.lb1 = weight.lb1-lr*sigma_layer
    return weight



# Pre-defined parameters
pre_trained = False          # pre_trained dataset
lay_1 = 30                 # number of nodes of the hidden layer
lr = 0.01                   # learning rate
batchsize = 8             # batchsize
epoch = 50                  # epoch numbers
is_save = True             # save the weights

# Load the training dataset and test dataset
print('Loading the dataset...')
X_train, y_train = load_mnist('data', kind='train')
X_test, y_test = load_mnist('data', kind='t10k')

# Initial the weights
print('Initializing...')
if pre_trained is False:
    weights.lw1 = np.random.randn(lay_1, 784)/lay_1
    weights.lb1 = np.random.randn(lay_1)/lay_1
    weights.lw2 = np.random.randn(10, lay_1)/10
    weights.lb2 = np.random.randn(10)/10
else:
    weights_npy = np.load('softmax.npy', allow_pickle=True)
    weights.lw1 = weights_npy[0]
    weights.lw2 = weights_npy[1]
    weights.lb1 = weights_npy[2]
    weights.lb2 = weights_npy[3]

# Start to train
print('Training...')
weight1 = []
weight2 = []
b1 = []
b2 = []
record = 0  # record the highest accuracy
loss_show = []
accuracy_show = []


for number, i in enumerate(range(epoch)):
    print('The', number, 'epoch:')
    for index, label in enumerate(tqdm(y_train)):
        target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result1, result2 = forward(weights, X_train[index])
        target[label] = 1

        weights_tem.lw1 = weights.lw1.copy()
        weights_tem.lw2 = weights.lw2.copy()
        weights_tem.lb1 = weights.lb1.copy()
        weights_tem.lb2 = weights.lb2.copy()

        weight_tem = backward(weights_tem, X_train[index], result1, result2, target, lr)
        weight1.append(weight_tem.lw1)
        weight2.append(weight_tem.lw2)
        b1.append(weight_tem.lb1)
        b2.append(weight_tem.lb2)
        if len(weight1) == batchsize or index == len(y_train):
            weights.lw1 = mean_calculator(weight1)
            weights.lw2 = mean_calculator(weight2)
            weights.lb1 = mean_calculator(b1)
            weights.lb2 = mean_calculator(b2)
            weight1 = []
            weight2 = []
            b1 = []
            b2 = []

    result = []
    count = 0
    loss_list = []
    for index, test in enumerate(y_test):
        result1, result2 = forward(weights, X_test[index])
        pre = np.argmax(result2)
        if pre == test:
            count += 1
        loss_tem = -np.log(max(result2))
        loss_list.append(loss_tem)
    loss = sum(loss_list)/len(loss_list)

    accuracy = count/len(y_test)
    if record < accuracy and is_save:
        save = []
        save.append(weights.lw1)
        save.append(weights.lw2)
        save.append(weights.lb1)
        save.append(weights.lb2)
        np.save('softmax.npy', save)
        record = accuracy
    print('accuracy:',accuracy)
    print('loss:', loss)
    accuracy_show.append(accuracy)
    loss_show.append(loss)

# Draw the accuracy and loss
import matplotlib.pyplot as plt
lay_1 = str(lay_1)
lr = str(lr)
batchsize = str(batchsize)

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Softmax, batchsize:' + batchsize + ', lr:' + lr + ', hidden layer nodes:' + lay_1)
plt.plot(accuracy_show)

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Softmax, batchsize:' + batchsize + ', lr:' + lr + ', hidden layer nodes:' + lay_1)
plt.plot(loss_show)
plt.show()
