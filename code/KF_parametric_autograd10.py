# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:07:58 2020

@author: matth
"""


import autograd.numpy as np
from autograd import value_and_grad 
import math

from kernel_functions_autograd import kernels_dic

#%%
    
"""We define several useful functions"""
    
# Returns a random sample of the data, as a numpy array
def sample_selection(data, size):
    indices = np.arange(data.shape[0])
    sample_indices = np.sort(np.random.choice(indices, size, replace= False))
    
    return sample_indices

# This function creates a batch and associated sample
def batch_creation(data, batch_size, sample_proportion = 0.5):
    # If False, the whole data set is the mini-batch, otherwise either a 
    # percentage or explicit quantity.
    if batch_size == False:
        data_batch = data
        batch_indices = np.arange(data.shape[0])
    elif 0 < batch_size <= 1:
        batch_size = int(data.shape[0] * batch_size)
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
    else:
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
        

    # Sample from the mini-batch
    sample_size = math.ceil(data_batch.shape[0]*sample_proportion)
    sample_indices = sample_selection(data_batch, sample_size)
    
    return sample_indices, batch_indices


# Generate a prediction
def kernel_regression(X_train, X_test, Y_train, param, kernel_keyword = "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    t_matrix = kernel(X_test, X_train, param) 
    prediction = np.matmul(t_matrix, np.matmul(np.linalg.inv(k_matrix), Y_train)) 
    return prediction

# redicttimeseries

def kernel_extrapolate(X_train, X_test, Y_train, param, nsteps=1, kernel_keyword = "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    A=np.matmul(np.linalg.inv(k_matrix), Y_train)
    arr = np.array([])
    
    X_test0=X_test
    isteps=int(nsteps/(X_test.shape[1]))+1
    for i in range(isteps):
        X_test1=X_test0
        t_matrix = kernel(X_test1, X_train, param) 
        prediction = np.matmul(t_matrix, A) 
        X_test0=prediction
        arr = np.append(arr, np.array(prediction[0,:]))
    arr=arr[0:nsteps]
    return arr


def replace_nan(array):
    for i in range(array.shape[0]):
        if math.isnan(array[i]) == True:
            print("Found nan value, replacing by 0")
            array[i] = 0
    return array

def sample_size_linear(iterations, range_tuple):
    
    return np.linspace(range_tuple[0], range_tuple[1], num = iterations)[::-1]
            
#%% Rho function

# The pi or selection matrix
def pi_matrix(sample_indices, dimension):
    pi = np.zeros(dimension)
    
    for i in range(dimension[0]):
        pi[i][sample_indices[i]] = 1
    
    return pi


def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]
    
    kernel_matrix = kernel(matrix_data, matrix_data, parameters)
#    print(kernel_matrix.shape)
    pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0]))   
#    print(pi.shape)
    
    sample_matrix = np.matmul(pi, np.matmul(kernel_matrix, np.transpose(pi)))
#    print(sample_matrix.shape)
    
    Y_sample = Y_data[sample_indices]
#    print(Y_sample.shape)
    
    lambda_term = regu_lambda
    inverse_data = np.linalg.inv(kernel_matrix + lambda_term * np.identity(kernel_matrix.shape[0]))
    inverse_sample = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))
#    print(inverse_sample.shape)
#    B=np.matmul(inverse_sample, Y_sample)
#    print(B.shape)
    
    top = np.tensordot(Y_sample, np.matmul(inverse_sample, Y_sample))
    
   
    bottom = np.tensordot(Y_data, np.matmul(inverse_data, Y_data))
    
    print(1-top/bottom)
    return 1 - top/bottom

def l2(parameters, matrix_data, Y, batch_indices, sample_indices, kernel_keyword = "RBF"):
    X_sample = matrix_data[sample_indices]
    Y_sample = Y[sample_indices]
    
    not_sample = [x for x in batch_indices not in sample_indices]
    X_not_sample = matrix_data[not_sample]
    Y_not_sample = Y[not_sample]
    prediction = kernel_regression(X_sample, X_not_sample, Y_sample, kernel_keyword)
    
    return np.dot(Y_not_sample - prediction, Y_not_sample- prediction)

#%% Grad functions

""" We define the gradient calculator function.Like rho, the gradient 
calculator function accesses the gradfunctions via a keyword"""

# Gradient calculator function. Returns an array
def grad_kernel(parameters, X_data, Y_data, sample_indices, kernel_keyword= "RBF", regu_lambda = 0.000001):
    grad_K = value_and_grad(rho)
    rho_value, gradient = grad_K(parameters, X_data, Y_data, sample_indices, kernel_keyword, regu_lambda = regu_lambda)
    return rho_value, gradient


#%% The class version of KF
    
class KernelFlowsP():
    
    def __init__(self, kernel_keyword, parameters):
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.grad_hist = []
        self.para_hist = []
        
        self.LR = 0.1
        self.beta = 0.9
        self.regu_lambda = 0.0001
    
    def get_hist(self):
        return self.param_hist, self.gradients, self.rho_values
        
    
    def save_model(self):
        np.save("param_hist", self.param_hist)
        np.save("gradients", self.gradients)
        np.save("rho_values", self.rho_values)
        
    def get_parameters(self):
        return self.parameters
    
    def set_LR(self, value):
        self.LR = value
        
    def set_beta(self, value):
        self.beta = value
    def set_train(self, train):
        self.train = train
        
    
    def fit(self, X, Y, iterations, batch_size = False, optimizer = "SGD", 
            learning_rate = 0.1, beta = 0.9, show_it = 100, regu_lambda = 0.000001, 
            adaptive_size = False, adaptive_range = (), proportion = 0.5, reduction_constant = 0.0):            

        self.set_LR(learning_rate)
        self.set_beta(beta)
        self.regu_lambda = regu_lambda
        
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        momentum = np.zeros(self.parameters.shape, dtype = "float")
        
        # This is used for the adaptive sample decay
        rho_100 = []
        adaptive_mean = 0
        adaptive_counter = 0
        
        if adaptive_size == False or adaptive_size == "Dynamic":
            sample_size = proportion
        elif adaptive_size == "Linear":
            sample_size_array = sample_size_linear(iterations, adaptive_range) 
        else:
            print("Sample size not recognized")
            
        for i in range(iterations):
            if i % show_it == 0:
                print("parameters ", self.parameters)
            
            if adaptive_size == "Linear":
                sample_size = sample_size_array[i]
                
            elif adaptive_size == "Dynamic" and adaptive_counter == 100:
                if adaptive_mean != 0:
                    change = np.mean(rho_100) - adaptive_mean 
                else:
                    change = 0
                adaptive_mean = np.mean(rho_100)
                rho_100 = []
                sample_size += change - reduction_constant
                adaptive_counter= 0
                
            # Create a batch and a sample
            sample_indices, batch_indices = batch_creation(X, batch_size, sample_proportion = sample_size)
            X_data = X[batch_indices]
            Y_data = Y[batch_indices]
            

                
            # Changes parameters according to SGD rules
            if optimizer == "SGD":
                rho, grad_mu = grad_kernel(self.parameters, X_data, Y_data, 
                                           sample_indices, self.kernel_keyword, regu_lambda = regu_lambda)
                if  rho > 1 or rho < 0:
                    print("Warning, rho outside [0,1]: ", rho)
                else:
                    self.parameters -= learning_rate * grad_mu
                    
            
            # Changes parameters according to Nesterov Momentum rules     
            elif optimizer == "Nesterov":
                rho, grad_mu = grad_kernel(self.parameters - learning_rate * beta * momentum, 
                                               X_data, Y_data, sample_indices, self.kernel_keyword, regu_lambda = regu_lambda)
                if  rho > 1 or rho < 0:
                    print("Warning, rho outside [0,1]: ", rho)
                else:
                    momentum = beta * momentum + grad_mu
                    self.parameters -= learning_rate * momentum
                
            else:
                print("Error optimizer, name not recognized")
            
            # Update history 
            self.para_hist.append(np.copy(self.parameters))
            self.rho_values.append(rho)
            self.grad_hist.append(np.copy(grad_mu))
            
            rho_100.append(rho)
            adaptive_counter +=1
                
            
        # Convert all the lists to np arrays
        self.para_hist = np.array(self.para_hist) 
        self.rho_values = np.array(self.rho_values)
        self.grad_hist = np.array(self.grad_hist)
                
        return self.parameters
    
    def predict(self,test, regu_lambda = 0.0000001):
         
        X_train = self.X_train
        Y_train = self.Y_train
        prediction = kernel_regression(X_train, test, Y_train, self.parameters, self.kernel_keyword, regu_lambda = regu_lambda) 

        return prediction

    def extrapolate(self,test, nsteps=1,regu_lambda = 0.000001):
         
        X_train = self.X_train
        Y_train = self.Y_train
        prediction = kernel_extrapolate(X_train, test, Y_train, self.parameters, nsteps,self.kernel_keyword, regu_lambda = regu_lambda) 

        return prediction

#%%
        
# if __name__ == "__main__":
#     # Generating data according to RBF kernel, true gamma is 0.1
#     from autograd.numpy.random import uniform
#     def data_set_RBF(dimensions, mu_correct):
#         # a = normal(scale=0.2, size = size)
#         values = uniform(-10, 10, dimensions)
#         b = []
#         for element in values:
#             b.append( np.exp(-np.linalg.norm(element)**2 /(2*mu_correct[0]**2)))
#         b = np.array(b) #+ normal(0, 0.25)
#         return b, values

#     mu_correct = np.array([10.0])
#     Y, X = data_set_RBF((100, 1), mu_correct)
#     data_set = np.concatenate((X,np.expand_dims(Y, 1)), axis = 1)
    
#     mu_1 = np.array([1.0])
#     K = KernelFlowsP("RBF", mu_1)
#     mu_pred = K.fit(X, Y, 10000, optimizer = "Nesterov",  batch_size = 50, show_it = 5000)
#     print(mu_pred)
    
#     mu_2 = np.array([15.0])
#     K = KernelFlowsP("RBF", mu_2)
#     mu_pred = K.fit(X, Y, 10000, optimizer = "Nesterov", batch_size = 50, show_it = 5000)
#     print(mu_pred)


np.random.seed(1)
import matplotlib.pyplot as plt

# plt.close('all')

train_data = np.load('../DMET/Filtered/POD_Coeffs_train.npy')
test_data = np.load('../DMET/Filtered/POD_Coeffs_test.npy')

# plt.figure()
# plt.plot(train_data[0,:])
# plt.plot(train_data[1,:])
# plt.plot(train_data[2,:])
# plt.show()

# plt.figure()
# plt.plot(test_data[0,:])
# plt.plot(test_data[1,:])
# plt.plot(test_data[2,:])
# plt.show()

# np.shape(train_data)
# np.shape(test_data)

# plt.figure()
# plt.plot(train_data[0,0:30])
# plt.show()

lenX=len(train_data[0,:])
lenXt=len(test_data[0,:])
print(lenX,lenXt)

predicted_test_data=0*test_data

delay=test_data.shape[-1]//2

vdelay=delay*np.ones((train_data.shape[0],), dtype=int)
vregu_lambda=100.0*np.ones((train_data.shape[0],))

# vdelay[0]=200
# vregu_lambda[0]=100

# vdelay[1]=1250
# vregu_lambda[1]=1

# vdelay[2]=1250
# vregu_lambda[2]=1

# vdelay[3]=1250
# vregu_lambda[3]=1

# vdelay[4]=1250
# vregu_lambda[4]=1

# vdelay[5]=1250
# vregu_lambda[5]=1

# vdelay[6]=1250
# vregu_lambda[6]=1

#delay=240


# X=np.zeros((lenX-delay-1,delay))
# for i in range(lenX-delay-1):
#      X[i,:]=train_data[0,i:(i+delay)]


    
# a=np.arange(lenX+lenXt)
# X=np.zeros((lenX-2*delay,delay))
# Y=np.zeros((lenX-2*delay,delay))
# for i in range(lenX-2*delay):
#       X[i,:]=np.cos(np.pi*a[i:(i+delay)]/365)
#       Y[i,:]=np.cos(np.pi*a[(i+delay):(i+2*delay)]/365)
# nsteps=lenXt
# Z=np.cos(np.pi*a/365)
# Ztest=Z[lenX:(lenX+lenXt)]
# Xtest=np.zeros((1,delay))
# Xtest[0,:]=Y[i,:]


# normalize=10000
# train_data=train_data/normalize
# test_data=test_data/normalize


for mode in range(train_data.shape[0]):
    
    #regu_lambda=0.00000001
    #regu_lambda=100
    regu_lambda=vregu_lambda[mode]
    delay=vdelay[mode]
    
    
    normalize=np.amax(train_data[mode,:])
    # print(normalize)
    # normalize=np.mean(np.abs(train_data[mode,:]))
    # print(normalize)
    
    X=np.zeros((1+lenX-2*delay,delay))
    Y=np.zeros((1+lenX-2*delay,delay))
    for i in range(1+lenX-2*delay):
          X[i,:]=train_data[mode,i:(i+delay)]
          Y[i,:]=train_data[mode,(i+delay):(i+2*delay)]
    nsteps=lenXt      
    Ztest=test_data[mode,:]
    Xtest=np.zeros((1,delay))
    Xtest[0,:]=Y[i,:]
    
    X=X/normalize
    Y=Y/normalize
    Xtest=Xtest/normalize
    Ztest=Ztest/normalize
    
    # import sys
    # sys.exit()

    #mu_1 = np.array([1.0])
    mu_1 = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    kerneltype="anl"
    
    K = KernelFlowsP("anl2", mu_1)
    mu_pred = K.fit(X, Y, 100, optimizer = "Nesterov",  batch_size = 100, show_it = 500, regu_lambda=vregu_lambda[mode])
    mu_1=mu_pred
    
    X_train=X
    Y_train=Y
    param=mu_1

    X_test=Xtest
    
    kernel = kernels_dic[kerneltype]
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    A=np.matmul(np.linalg.inv(k_matrix), Y_train)
    arr = np.array([])
    
    X_test0=X_test
    isteps=int(nsteps/delay)+1
    for i in range(isteps):
        X_test1=X_test0
        t_matrix = kernel(X_test1, X_train, param) 
        prediction = np.matmul(t_matrix, A) 
        X_test0=prediction
        arr = np.append(arr, np.array(prediction[0,:]))
    arr=arr[0:nsteps]
        
    pred=arr*normalize
    Ztest=Ztest*normalize
    
    plt.figure()
    line_up, =plt.plot(pred,label='predicted')
    line_down, =plt.plot(Ztest,label='true')
    plt.legend(handles=[line_up, line_down])
    plt.title("mode "+str(mode))
    plt.show()
    
    predicted_test_data[mode,:]=pred
    
    nmin=0
    nmax=400
    plt.figure()
    line_up, =plt.plot(pred[nmin:nmax],label='predicted')
    line_down, =plt.plot(Ztest[nmin:nmax],label='true')
    plt.legend(handles=[line_up, line_down])
    plt.title("mode "+str(mode)+", tmin="+str(nmin)+", tmax="+str(nmax))
    plt.show()
    
    nmin=1200
    nmax=1600
    plt.figure()
    line_up, =plt.plot(pred[nmin:nmax],label='predicted')
    line_down, =plt.plot(Ztest[nmin:nmax],label='true')
    plt.legend(handles=[line_up, line_down])
    plt.title("mode "+str(mode)+", tmin="+str(nmin)+", tmax="+str(nmax))
    plt.show()
    
    
    
    # t=kernel(X_train, X_train, param)
    # prediction = np.matmul(t, A)
    # plt.figure()
    # plt.plot(prediction)
    # plt.show()

np.save("predicted_test_data",predicted_test_data)
