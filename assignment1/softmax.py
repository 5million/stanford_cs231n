import random
import numpy as np
from cs231n.data_utils import get_CIFAR10_data
from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.gradient_check import grad_check_sparse
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.classifiers.linear_classifier import Softmax
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Invoke the above function to get our data.
data_set = get_CIFAR10_data()
X_train = data_set["X_train"]
y_train = data_set["y_train"]
X_val = data_set["X_val"]
y_val = data_set["y_val"]
X_test = data_set["X_test"]
y_test = data_set["y_test"]

X_train = X_train.reshape((X_train.shape[0], -1))
X_val = X_val.reshape((X_val.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

X_mean = np.mean(X_train, axis=0)
X_train -= X_mean
X_val -= X_mean
X_test -= X_mean

X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

print(f"X_train:{X_train.shape}, y_train:{y_train.shape}")
print(f"X_val:{X_val.shape}, y_val:{y_val.shape}")
print(f"X_test:{X_test.shape}, y_test:{y_test.shape}")
# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(3073, 10) * 0.0001

results = {}
best_val = -1
best_softmax = None

# Provided as a reference. You may or may not want to change these hyperparameters
learning_rates = [1e-7, 5e-7, 1e-6, 5e-6]
regularization_strengths = [2.5e4, 4e5, 5e4, 8e4]

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

for lr in learning_rates:
    for reg in regularization_strengths:
        model = Softmax()
        model.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=120, batch_size=256)
        y_pred = model.predict(X_train)
        acc_train = np.mean(y_pred == y_train)
        y_pred = model.predict(X_val)
        acc_val = np.mean(y_pred == y_val)
        results[(lr, reg)] = (acc_train, acc_val)
        if best_val < acc_val:
            best_val = acc_val
            best_softmax = model


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)

# evaluate on test set
# Evaluate the best softmax on test set
y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))

# Visualize the learned weights for each class
w = best_softmax.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
plt.show()