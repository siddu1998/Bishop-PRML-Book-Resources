import numpy as np
import torch
import matplotlib.pyplot as plt

################# VECTORIZATION #############
def vector_multiplication(vector_1,vector_2):
    #ideally vector_1 and vector_2 in DL would be the features and vectors
    #two methods generally in literature you would see a transpose
    return (vector_1.T).dot(vector_2), vector_1.dot(vector_2) 

x = np.random.rand(1000) 
w = np.random.rand(1000)
print(vector_multiplication(x,w))


x = np.array([1,3,4])
w = np.array([1,3,4])
print(vector_multiplication(x,w))


################### PERCEPTRON ############
data = np.genfromtxt('perceptron_toydata.txt', delimiter='\t')

X = data[:,:2]
y = data[:,2]
y=y.astype(np.int)



#Understanding data
print('[INFO] Class label counts:', np.bincount(y))
print('[INFO] X shape',X.shape)
print('[INFO] Y shape',y.shape)



#shuffle the data
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]

X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]




#it is ideal to normalize the data. By normalizing we subtract the mean and divide by the variance
#ideally the mean and variance are taken form the training dataset and applied
mean = X_train.mean(axis=0) #note axis zero is finding mean of each column, while axis one be per row
sigma = X_train.std(axis=0)
X_train = (X_train - mean)/sigma
X_test = (X_test - mean)/sigma


plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')
plt.title('Training set')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend()
plt.show()

### Implementing Perceptron class ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Perceptron():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, 
                                   dtype=torch.float32, device=device)
        
        #so you have two feature vectors x1 and x2 and you need
        #to multiply them with w1 and w2, so in that case you need
        #something like [w1,w2] or [[w1],[w2]]
        #the above initialization gives [[w1],[w2]] i.e two rows and one column
        print('[INFO] Intialized weights of shape',self.weights)
        
        
        self.bias = torch.zeros(1, dtype=torch.float32, device=device)
        
        # placeholder vectors so they don't
        # need to be recreated each time
        self.ones = torch.ones(1)
        self.zeros = torch.zeros(1)

    def forward(self, x):
        #print(x.shape,self.weights.shape)
        #here it would be 1 row 2 columns, 2 rows 1 column
        #print(self.weights,self.weights.shape)
        linear = torch.mm(x, self.weights) + self.bias
        predictions = torch.where(linear > 0., self.ones, self.zeros)
        return predictions
        
    def backward(self, x, y):  
        predictions = self.forward(x)
        errors = y - predictions
        return errors
        
    def train(self, x, y, epochs):
        for e in range(epochs):
            for i in range(y.shape[0]):
                errors = self.backward(
                    x[i].reshape(1, self.num_features),
                    y[i]).reshape(-1)
                self.weights += (errors * x[i]).reshape(self.num_features, 1)
                self.bias += errors
                
    def evaluate(self, x, y):
        predictions = self.forward(x).reshape(-1)
        accuracy = torch.sum(predictions == y).float() / y.shape[0]
        return accuracy


ppn = Perceptron(num_features=2)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

ppn.train(X_train_tensor, y_train_tensor, epochs=5)

print('Model parameters:')
print('  Weights: %s' % ppn.weights)
print('  Bias: %s' % ppn.bias)  

X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

test_acc = ppn.evaluate(X_test_tensor, y_test_tensor)
print('Test set accuracy: %.2f%%' % (test_acc*100))