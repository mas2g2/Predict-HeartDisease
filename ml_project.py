import csv
import numpy as np

# This function reads data from csv files to a two dimensional numpy array

def load_data(filename):
    with open(filename,"r") as f:
        data = np.array(list(csv.reader(f)))
    # Removes headers from array
    data = data[1:,:]
    data = data.astype(float)
    return data

# Builds model

class NeuralNetwork:
    def __init__(self):
        self.weights = {}
        self.delta = {}
        self.error = {}
        self.num_layers = 1

    def new_layer(self, shape):
        np.random.seed(1)
        self.weights[self.num_layers] = 2*np.random.random(shape) -1
        self.delta[self.num_layers] = np.zeros(shape)
        self.num_layers += 1
        
    def sigmoid(self,x,deriv=False):
        if deriv == True:
            return x*(1-x)
        return 100/(1 + np.exp(-x))

    def max(self,x,y):
        if x >= y:
            return x
        return y

    def reLu(self, x):
        relu = np.zeros(x.shape)
        for i in range(len(x)):
            relu[i] = self.max(x[i],0)
        return relu

    def forward_prop(self,x):
        activation_val = {}
        data = x
        activation_val[1] = data
        for layer in range(2,self.num_layers+1):
            data = self.sigmoid(data.dot(self.weights[layer-1]))
            activation_val[layer] = data
        return activation_val

    def sum_squared_error(self, outputs, targets):
        return 0.5 * np.mean(np.sum(np.power(outputs - targets,2)))
   
    def back_prop(self, output, target):
        delta = {}
        delta[self.num_layers] = output[self.num_layers] - target
        for i in reversed(range(2,self.num_layers)):
            delta[i] = np.multiply(np.dot(self.weights[i],delta[i+1]),self.sigmoid(output[i],deriv=True))

        for i in range(1,self.num_layers):
            if output[i].ndim == 1 and delta[i+1].ndim == 1:
                output[i] = np.reshape(output[i],(len(output[i]),1))
                delta[i+1] = np.reshape(delta[i+1],(len(delta[i+1]),1))
            self.delta[i] += np.dot(delta[i+1],output[i].T).T

    def grad_desc(self, batch_size=0, learning_rate=1):
        for i in range(1,self.num_layers):
            partial_der = (1/batch_size)*self.delta[i]
            self.weights[i] += learning_rate * -partial_der

    def fit(self, inputs, targets, epochs = 1000, learning_rate=1):
        for iterat in range(epochs):
            for i in range(len(inputs)):
                x = inputs[i]
                y = targets[i]

                pred_y = self.forward_prop(x)

                loss = self.sum_squared_error(pred_y[self.num_layers],y)

                print("Error : ",loss)

                self.back_prop(pred_y,y)
            self.grad_desc(batch_size=i,learning_rate=learning_rate)

data = load_data("framingham.csv")
train, test = data[:200,:],data[200:,:]
train_x,train_y,test_x,test_y = train[:,:14],np.reshape(train[:,14],(len(train),1)),test[:,:14],test[:,14]

model = NeuralNetwork()
model.new_layer((14,5))
model.new_layer((5,10))
model.new_layer((10,1))
#print(model.weights)
predict = model.forward_prop(train_x)

#print(predict)
model.fit(train_x,train_y,epochs=1000,learning_rate=0.001)
