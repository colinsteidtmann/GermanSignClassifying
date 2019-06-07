# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
import random
import math
import copy

class SharedFunctions:
    def __init__(self, activations, weights_shapes, biases_shapes, scale_method=None, optimizer="adam", lr=0.01, lr_decay=0):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.

        ``activations`` is for the hidden layers, it must be a sting, either "sigmoid" or "tanh" or "relu" or "leaky_relu" or "linear"
        ``weights_shapes`` is a list, [w.shape for w in weights]
        ``biases_shapes`` is a list, [b.shape for b in biases]
        ``scale_method`` is to scale the data to lies in a smaller range
        ``optimizer`` is to optimize the speed that our neural network learns. 
        ``lr`` = learning rate, how fast our neural network learns the cost function gradient
        ``lr_decay`` is how fast we decay the learning rate

        """ 
        """ Check paramaters"""
        if not (scale_method == "normalize" or scale_method == "standardize" or scale_method is None):
            sys.exit('Ooops! scale_method must be "normalize" or "standardize" or None for none')
        if not (optimizer == "sgd" or optimizer == "momentum" or optimizer == "nesterov" or optimizer == "adagrad" or optimizer == "rmsprop" or optimizer == "adam" or optimizer == "adamax" or optimizer == "nadam"):
            sys.exit('Ooops! optimizer must be "sgd," "momentum," "nesterov," "adagrad," "rmsprop," "adam," "adamax," or "nadam" ')

        self.activations = activations
        self.weights_shapes = weights_shapes
        self.biases_shapes = biases_shapes
        self.scale_method = scale_method
        self.optimizer = optimizer
        self.initialize_optimizer_params()
        self.initilize_hyperparams(lr, lr_decay)
        self.regularizer = None
        self.reg_lambda = None
    

    def initialize_optimizer_params(self):
        """ Initilize different optimizer paramaters. """

        if (self.optimizer == "momentum"):
            """ 
                "With Momentum update, the parameter vector will build up velocity in any direction that has consistent gradient."
                mu --> momentum rate, typical values are [0.5, 0.9, 0.95, 0.99]
                vws/vbs --> velocity and direction of gradients
            """
            self.mu = 0.9 
            self.vws = [np.zeros(w_shape) for w_shape in self.weights_shapes]
            self.vbs = [np.zeros(b_shape) for b_shape in self.biases_shapes]
        
        elif (self.optimizer == "nesterov"):
            """
                Nesterov momentum has the same paramaters as Momentum, 
                except it's implementation will be different
            """
            self.mu = 0.9 
            self.vws = [np.zeros(w_shape) for w_shape in self.weights_shapes]
            self.vbs = [np.zeros(b_shape) for b_shape in self.biases_shapes]
        
        elif (self.optimizer == "adagrad"):
            """
                Adagrad is an adaptive learning rate method, "weights that 
                receive high gradients will have their effective learning rate reduced, 
                while weights that receive small or infrequent updates will have their 
                effective learning rate increased."
                
                eps --> avoids division by zero, typical values range from 1e-4 to 1e-8
                sws/sbs --> keeps track of per-parameter sum of squared gradients
            """
            self.eps = 1e-8
            self.sws = [np.zeros(w_shape) for w_shape in self.weights_shapes]
            self.sbs = [np.zeros(b_shape) for b_shape in self.biases_shapes]
        
        elif (self.optimizer == "rmsprop"):
            """ 
                "The RMSProp update adjusts the Adagrad method in a very simple way in an 
                attempt to reduce its aggressive, monotonically decreasing learning rate. 
                In particular, it uses a moving average of squared gradients"

                gamma --> decay rate, typical values are [0.9, 0.99, 0.999]
            """
            self.gamma = 0.9
            self.eps = 1e-8
            self.sws = [np.zeros(w_shape) for w_shape in self.weights_shapes]
            self.sbs = [np.zeros(b_shape) for b_shape in self.biases_shapes]

        elif (self.optimizer == "adam" or self.optimizer == "adamax" or self.optimizer == "nadam"):
            """
                - Adam is RMSProp with momentum
                - Adamax is a stable version of Adam that is more robust to big gradients, 
                  it is better for when paramter are updated sparsly 
                - Nadam is Adam RMSprop with Nesterov momentum.
                mws/mbs --> "smooth" verion of the gradient instead of the raw (and perhaps noisy) 
                       gradient vector dx. 
            """
            self.eps = 1e-8
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.vws = [np.zeros(w_shape) for w_shape in self.weights_shapes]
            self.vbs = [np.zeros(b_shape) for b_shape in self.biases_shapes]
            self.mws = [np.zeros(w_shape) for w_shape in self.weights_shapes]
            self.mbs = [np.zeros(b_shape) for b_shape in self.biases_shapes]
    
    def initilize_hyperparams(self, lr=0.01, lr_decay=0):
        """ ``iteration_count`` keeps a running tab on how many times SGD has iterated through the data, this 
              variable is used in some optimizers like Adam or in self.lr decay
            ``running_min``,``running_max``,``running_mean``, and ``running_var`` are all used in the scale_fn()
            ``lr`` = learning rate, how fast our neural network learns the cost function gradient
            ``lr_decay`` is how fast we decay the learning rate
        """
        self.running_min = 1.0
        self.running_max = 1.0
        self.running_mean = 1.0
        self.running_var = 1.0
        self.lr = lr
        self.lr_decay = lr_decay


    def optimize(self, nablaWs, nablaBs, weights, biases, iteration_count):
        lr = self.lr / (1 + self.lr_decay * iteration_count)
        """ Updates paramaters with gradients of cost function w.r.t paramaters """
        if (self.optimizer == "sgd"):
            new_weights = [w - (lr * nw) for w,nw in zip(weights,nablaWs)]
            new_biases = [b - (lr * nb) for b, nb in zip(biases, nablaBs)]
            
        elif (self.optimizer == "momentum"):
            self.vws = [(self.mu * vw) + (lr * nw) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [(self.mu * vb) + (lr * nb) for vb, nb in zip(self.vbs, nablaBs)]
            new_weights = [w - vw for w, vw in zip(weights, self.vws)]
            new_biases = [b - vb for b, vb in zip(biases, self.vbs)]

        elif (self.optimizer == "nesterov"):
            vws_prev = self.vws
            vbs_prev = self.vbs
            self.vws = [(self.mu * vw) + (lr * nw) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [(self.mu * vb) + (lr * nb) for vb, nb in zip(self.vbs, nablaBs)]
            new_weights = [w - ((self.mu * vw_prev) + ((1 - self.mu) * vw)) for w, vw_prev, vw in zip(weights, vws_prev, self.vws)]
            new_biases = [b - ((self.mu * vb_prev) + ((1 - self.mu) * vb)) for b, vb_prev, vb in zip(biases, vbs_prev, self.vbs)]

        elif (self.optimizer == "adagrad"):
            self.sws = [sw + (nw ** 2) for sw, nw in zip(self.sws, nablaWs)]
            self.sbs = [sb + (nb ** 2) for sb, nb in zip(self.sbs, nablaBs)]
            new_weights = [w - (lr * nw / (np.sqrt(sw) + self.eps)) for w, sw, nw in zip(weights, self.sws, nablaWs)]
            new_biases = [b - (lr * nb / (np.sqrt(sb) + self.eps)) for b, sb, nb in zip(biases, self.sbs, nablaBs)]
        
        elif (self.optimizer == "rmsprop"):
            self.sws = [(self.gamma * sw) + ((1 - self.gamma) * (nw ** 2)) for sw, nw in zip(self.sws, nablaWs)]
            self.sbs = [(self.gamma * sb) + ((1 - self.gamma) * (nb ** 2)) for sb, nb in zip(self.sbs, nablaBs)]
            new_weights = [w - (lr * nw / (np.sqrt(sw) + self.eps)) for w, sw, nw in zip(weights, self.sws, nablaWs)]
            new_biases = [b - (lr * nb / (np.sqrt(sb) + self.eps)) for b, sb, nb in zip(biases, self.sbs, nablaBs)]
        
        elif (self.optimizer == "adam"):
            self.mws = [(self.beta1 * mw) + ((1 - self.beta1) * nw) for mw, nw in zip(self.mws, nablaWs)]
            self.mbs = [(self.beta1 * mb) + ((1 - self.beta1) * nb) for mb, nb in zip(self.mbs, nablaBs)]
            mtws = [mw / (1 - (self.beta1 ** iteration_count)) for mw in self.mws]
            mtbs = [mb / (1 - (self.beta1 ** iteration_count)) for mb in self.mbs]
            self.vws = [(self.beta2 * vw) + ((1 - self.beta2) * (nw ** 2)) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [(self.beta2 * vb) + ((1 - self.beta2) * (nb ** 2)) for vb, nb in zip(self.vbs, nablaBs)]
            vtws = [vw / (1 - (self.beta2 ** iteration_count)) for vw in self.vws]
            vtbs = [vb / (1 - (self.beta2 ** iteration_count)) for vb in self.vbs]
            new_weights = [w - (lr * mtw / (np.sqrt(vtw) + self.eps)) for w, mtw, vtw in zip(weights, mtws, vtws)]
            new_biases = [b - (lr * mtb / (np.sqrt(vtb) + self.eps)) for b, mtb, vtb in zip(biases, mtbs, vtbs)]

        elif (self.optimizer == "adamax"):
            self.mws = [(self.beta1 * mw) + ((1 - self.beta1) * nw) for mw, nw in zip(self.mws, nablaWs)]
            self.mbs = [(self.beta1 * mb) + ((1 - self.beta1) * nb) for mb, nb in zip(self.mbs, nablaBs)]
            self.vws = [np.maximum((self.beta2 * vw), np.abs(nw)) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [np.maximum((self.beta2 * vb), np.abs(nb)) for vb, nb in zip(self.vbs, nablaBs)]
            new_weights = [w - ((lr / (1 - (self.beta1 ** iteration_count))) * (mw/vw)) for w, mw, vw in zip(weights, self.mws, self.vws)]
            new_biases = [b - ((lr / (1 - (self.beta1 ** iteration_count))) * (mb/vb)) for b, mb, vb in zip(biases, self.mbs, self.vbs)]
        
        elif (self.optimizer == "nadam"):
            self.mws = [(self.beta1 * mw) + ((1 - self.beta1) * nw) for mw, nw in zip(self.mws, nablaWs)]
            self.mbs = [(self.beta1 * mb) + ((1 - self.beta1) * nb) for mb, nb in zip(self.mbs, nablaBs)]
            mtws = [((self.beta1 * mw) / (1 - (self.beta1 ** iteration_count))) + (((1 - self.beta1) * nw ) / (1 - (self.beta1 ** iteration_count))) for mw, nw in zip(self.mws, nablaWs)]
            mtbs = [((self.beta1 * mb) / (1 - (self.beta1 ** iteration_count))) + (((1 - self.beta1) * nb ) / (1 - (self.beta1 ** iteration_count))) for mb, nb in zip(self.mbs, nablaBs)]
            self.vws = [(self.beta2 * vw) + ((1 - self.beta2) * (nw ** 2)) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [(self.beta2 * vb) + ((1 - self.beta2) * (nb ** 2)) for vb, nb in zip(self.vbs, nablaBs)]
            vtws = [(self.beta2 * vw) / (1 - (self.beta2 ** iteration_count)) for vw in self.vws]
            vtbs = [(self.beta2 * vb) / (1 - (self.beta2 ** iteration_count)) for vb in self.vbs]
            new_weights = [w - (lr * mtw / (np.sqrt(vtw) + self.eps)) for w, mtw, vtw in zip(weights, mtws, vtws)]
            new_biases = [b - (lr * mtb / (np.sqrt(vtb) + self.eps)) for b, mtb, vtb in zip(biases, mtbs, vtbs)]
        
        return new_weights, new_biases
            
    def activation_fn(self, z, layer_idx):

        z = copy.deepcopy(z)
        num_inputs = len(z)
        if self.activations[layer_idx] == "linear":
            return z
        elif self.activations[layer_idx] == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        elif self.activations[layer_idx] == "softmax":
            return np.exp(z-np.max(z,1,keepdims=True))/np.sum(np.exp(z-np.max(z,1,keepdims=True)),1,keepdims=True)
        elif self.activations[layer_idx] == "tanh":
            return np.tanh(z)
        elif self.activations[layer_idx] == "relu":
            return np.maximum(0, z)
        elif self.activations[layer_idx] == "leaky_relu":
            return np.maximum(0.01 * z, z)
    
    def activation_fn_prime(self, z, layer_idx):
        z = copy.deepcopy(z)
        if self.activations[layer_idx] == "linear":
            return np.ones(z.shape)
        elif self.activations[layer_idx] == "sigmoid":
            return self.activation_fn(z, layer_idx) * (1 - self.activation_fn(z, layer_idx))
        elif self.activations[layer_idx] == "softmax":
            return np.ones(z.shape)
        elif self.activations[layer_idx] == "tanh":
            return (1 - (np.tanh(z)** 2))
        elif self.activations[layer_idx] == "relu":
            z[z <= 0] = 0
            z[z > 0] = 1
            return z
        elif self.activations[layer_idx] == "leaky_relu":
            z[z <= 0] = 0.01
            z[z > 0] = 1
            return z

    def scale_fn(self, x, train=True):
        """ Scales input (x) using normalization or standardization
            if train=True, update running averages using data from mini batch
            else, scale using the running averages
        """
        x = copy.deepcopy(x)
        if type(x) != np.ndarray: x = np.array(x)

        if train:
            inputs = np.array(x, ndmin=4)
            self.running_min = np.min(inputs, axis=(0, 1, 2))
            self.running_max = np.min(inputs, axis=(0,1,2))
            self.running_mean = np.mean(inputs, axis=(0,1,2))
            self.running_var = np.var(inputs, axis=(0, 1, 2))

        if (self.scale_method == "standardize"):
            """ Standardizes data so min = 0 and max = 1 """
            return np.array((x - self.running_min) / (self.running_max - self.running_min + 1e-8), ndmin=2)
        elif (self.scale_method == "normalize"):
            """ Normalized data so μ = 0 and 𝛔 = 1 """
            return np.array((x - self.running_mean) / np.sqrt(self.running_var + 1e-8), ndmin=2)
    
    def weight_decay_deriv(self, w):
        """ 
            get derivatives of l1 or l2 regularizer (weight decay)
            l1 regularizer = old cost + lambda * |sum(w)|
            l2 regularizer = old cost + lambda * sum(w²)
            else, return 0's to not 
        """
        w = copy.deepcopy(w)
        if (self.regularizer == "l1"):
            w[w < 0] = -1
            w[w > 0] = 1
            w[w == 0] = 0
            return w * self.reg_lambda
        elif (self.regularizer == "l2"):
            return w * self.reg_lambda
        else:
            return np.zeros(w.shape)




    