# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from NeuralNets import SharedFunctions
import random
import math
import copy
class FullyConnected:
    def __init__(self, sizes, activations, scale_method=None, optimizer="adam", lr=0.01, lr_decay=0):
        """
        The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.

        ``activation`` is for the hidden layers, it must be a sting, either "sigmoid" or "softmax" or "tanh" or "relu" or "leaky_relu" or "linear"
        ``scale_method`` is to scale the data to lies in a smaller range
        ``optimizer`` is to optimize the speed that our neural network learns. 
        ``lr`` = learning rate, how fast our neural network learns the cost function gradient
        ``lr_decay`` is how fast we decay the learning rate

        """
        if (len(activations) != len(sizes)):
            sys.exit('Ooops! there must be an activation function for each layer in sizes')
        if (False in [True if ((activation == "linear" or activation == "sigmoid" or activation == "softmax" or activation == "tanh" or activation == "relu" or activation == "leaky_relu")) else False for activation in activations]):
            sys.exit('Ooops! activation function must be "linear" or "sigmoid" or "softmax" or "tanh" or "relu" or "leaky_relu"')
        if not (scale_method == "normalize" or scale_method == "standardize" or scale_method is None):
            sys.exit('Ooops! scale_method must be "normalize" or "standardize" or None for none')
        if not (optimizer == "sgd" or optimizer == "momentum" or optimizer == "nesterov" or optimizer == "adagrad" or optimizer == "rmsprop" or optimizer == "adam" or optimizer == "adamax" or optimizer == "nadam"):
            sys.exit('Ooops! optimizer must be "sgd," "momentum," "nesterov," "adagrad," "rmsprop," "adam," "adamax," or "nadam" ')
        elif (type(sizes) != list):
            sys.exit('Ooops! sized must be a list')
            
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activations = activations
        self.scale_method = scale_method
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay = lr_decay
        self.initialize_weights()
        self.shared_funcs = SharedFunctions.SharedFunctions(activations, [w.shape for w in self.weights], [b.shape for b in self.biases], scale_method, optimizer, lr, lr_decay)
        self.initilize_hyperparams()
        self.dropout_rate = None
        self.dropout_idxs = []
    
    def initialize_weights(self):
        """ Initlize our weights and biases with numbers drawn from a normal distribution 
            with mean=0 and std=1 
        """
        self.weights = [np.random.normal(0, (1/np.sqrt(inputSize)), (outputSize, inputSize)) for outputSize, inputSize in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [np.random.normal(0, 1, (outputSize, 1)) for outputSize in self.sizes[1:]]
        self.copy_of_weights = np.copy(self.weights)
        self.copy_of_biases = np.copy(self.biases)
    
    def initilize_hyperparams(self):
        """ ``iteration_count`` keeps a running tab on how many times SGD has iterated through the data, this 
              variable is used in some optimizers like Adam or in self.lr decay
        """
        self.iteration_count = 0

    def feedforward(self, inputs, scale=True, test=False):
        """ Return the outputs of the network if ``a`` is input
            Multiply by the dropoutProb (a number < 1 if dropout is used else just 1)
            ``scale`` is default True, but the get_losses() method will set it to false so that data is not rescaled
            if ``test`` is True then dropout will not be applied
        """
        if (self.scale_method is not None and scale == True and self.iteration_count > 0): inputs = self.shared_funcs.scale_fn(inputs, train=False)
        a = np.array(inputs,dtype=np.float64).reshape((-1, self.sizes[0], 1))
        """ dropout mask for first layer """
        if (0 in self.dropout_idxs and not test):
            d = np.random.binomial(1, (1 - self.dropout_rate), size=a.shape)/(1 - self.dropout_rate)
        else:
            d = 1
        a *= d
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.matmul(w, a) + b
            a = self.shared_funcs.activation_fn(z, (layer_idx + 1))
            """ create dropout mask """
            if ((layer_idx+1) in self.dropout_idxs and not test):
                d = np.random.binomial(1, (1 - self.dropout_rate), size=a.shape) / (1 - self.dropout_rate)
            else:
                d = 1
            a *= d
        return self.output_scaler(a)
    
    def apply_gradients(self, gradients):
        """ Applies gradients to weights and biases
            ``gradients`` is a tuple of lists ``([nablaWs], [nablaBs])`` representing for training data
        """
        self.iteration_count += 1
        for nablaWs, nablaBs in gradients:
            self.weights, self.biases = self.shared_funcs.optimize(nablaWs, nablaBs, self.weights, self.biases, self.iteration_count)
                
    
    def gradients(self, inputs, dx_layer, dx_type, grad_ys=None, return_new_grad_ys=False):
        """ Calculates derivitaves of dx_layer w.r.t dy_layer
            ``inputs`` is a normal feedforward input
            ``dx_layer`` is an integer, or list of intergers indicating dx_layer numbers, layer numbers start at 1, Weights and biases 
                        have the same layer number as the hidden or output layer ahead of them.
            ``dx_type`` is a string, or list of strings for the type of layer for dx_layer, it can be "weights", "biases", or "zLayer" 
            ``grad_ys`` represent the "starting" backprop value, ones if set to None, must be the same shape as output
            ``return_new_grad_ys`` default False, if True then it also returns d inputs w.r.t ``grad_ys``
            returns a list of gradients or single gradient (depending on if dx_layer is a list or not) for each dx_type in dx_type w.r.t output * grad_ys
        """
        dx_layer = np.array(dx_layer, ndmin=1)
        if type(dx_type) != list: dx_type = [dx_type]
        
        """ Check paramaters """
        if (dx_layer.dtype != int):
            sys.exit('Ooops! dx_layer be integer(s), layer numbers starting at 1. (Weights and biases have the same layer number as the hidden or output layer ahead of them.)')
        if not ("weights" in dx_type or "biases" in dx_type or "zLayer" in dx_type or "output" in dx_type ):
            sys.exit('Ooops! dx_type must equal "weights", "biases", "zLayer", or "output"')
        
        """ reshape inputs """
        x = np.array([inputs], dtype=np.float64).reshape((-1, self.sizes[0], 1))
        

        """ arrays to store outputs of each layer """
        dropoutLayers = []
        zLayers = [x]  
        aLayers = [x]

        """ dropout mask for first layer """
        if 0 in self.dropout_idxs:
            d = np.random.binomial(1, (1 - self.dropout_rate), size=x.shape)/(1 - self.dropout_rate)
        else:
            d = 1
        aLayers[0] *= d
        dropoutLayers.append(d)
        
        """ feedforward with input x, store outputs, z, and activations of z of each layer 
            (if using dropout regularzation store the the array with 0's (dropout mask), else just the default dropout array with 1's 
            so that no neurons are dropped)
        """
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.matmul(w, aLayers[-1]) + b
            zLayers.append(z)
            a = self.shared_funcs.activation_fn(z, (layer_idx + 1))

            """ create dropout mask """
            if (layer_idx + 1) in self.dropout_idxs:
                d = np.random.binomial(1, (1 - self.dropout_rate), size=a.shape)/(1 - self.dropout_rate)
            else:
                d = 1
            a *= d
            dropoutLayers.append(d)
            aLayers.append(a)
        aLayers[-1] = self.output_scaler(aLayers[-1])

        """ Begin Backpropagation
            Multiply grad_ys, "starting" backprop value, times delta
            get d of cost w.r.t final layer,  δᴸ = ∇ₐC ⊙ σ′(zᴸ) or ....
            get d of final layer w.r.t final layer,  δᴸ = σ′(zᴸ) 
            Multiply δᴸ * the output scale derivative            
        """
        num_outputs = aLayers[-1].shape[0]
        grad_ys = np.ones(np.array(aLayers[-1]).shape) if (grad_ys is None) else np.array(grad_ys, dtype=np.float64).reshape(aLayers[-1].shape)
        delta = (grad_ys * self.shared_funcs.activation_fn_prime(zLayers[-1], (self.num_layers - 1)).reshape(aLayers[-1].shape) * np.squeeze(self.output_scaler_deriv(zLayers[-1])).reshape(aLayers[-1].shape))
        delta *= dropoutLayers[-1]
        """  backpropagate error to each layer in nn, store each d of cost w.r.t weight layer as nabla_w
             δˡ = ((wˡ⁺¹)ᵀδˡ⁺¹) ⊙ σ′(zˡ).
             ∇w = δˡ(aˡ-¹)ᵀ
             ∇b = δˡ
             (if using weight decay regularzation add it to ∇w, if using dropout regularzation then zero ∇w & ∇b for 
             the biases and weights that were dropped in the feedforward)
             store dx w.r.t dy in nabla_dx_layer, for each layer in dx_layer 
        """ 
        nabla_dx_layer = [[0 for dx_layer in dx_layer] for dx_type in dx_type]
        """append output deriv if asked for """
        nabla_output = np.squeeze(np.argwhere(dx_layer == (self.num_layers+1)))
        if (("zLayer" in dx_type) and nabla_output.size > 0): nabla_dx_layer[dx_type.index("zLayer")][nabla_output] = delta
        for l in range(1, self.num_layers):
            """ nabla dx layer idx for weights and biases """
            idx_num = min(np.where(np.array(dx_type) == "weights"), np.where(np.array(dx_type) == "biases"))
            nabla_dwb_layer_idx = np.squeeze(np.argwhere(dx_layer == (self.num_layers - l + 1))[idx_num])
            """ nabla dx layer idx for hidden layers and inputs """
            idx_num = min(np.where(np.array(dx_type) == "zLayer"), np.array(2))
            nabla_dh_layer_idx = np.squeeze(np.argwhere(dx_layer == (self.num_layers - l))[idx_num])
            
            if nabla_dwb_layer_idx.size > 0:
                if "weights" in dx_type:
                    nabla_dx_layer[dx_type.index("weights")][nabla_dwb_layer_idx] = (np.matmul(delta, aLayers[-l - 1].transpose(0, 2, 1)) + self.shared_funcs.weight_decay_deriv(self.weights[-l])) 
                if "biases" in dx_type:
                    nabla_dx_layer[dx_type.index("biases")][nabla_dwb_layer_idx] =  delta 
            if (("zLayer" in dx_type) and nabla_dh_layer_idx.size > 0):
                nabla_dx_layer[dx_type.index("zLayer")][nabla_dh_layer_idx] = np.matmul(self.weights[-l].transpose(), delta) * (self.shared_funcs.activation_fn_prime(zLayers[-l - 1], layer_idx=(-l - 1)))

            delta = np.matmul(self.weights[-l].transpose(), delta) * self.shared_funcs.activation_fn_prime(zLayers[-l - 1], layer_idx=(-l - 1))
            delta *= dropoutLayers[-l - 1]
            
            

        """ remove extra gradient slots that are 0 and we didn't fill """
        nabla_dx_layer = [list(filter(lambda a: a is not 0, x)) for x in nabla_dx_layer]
        return (nabla_dx_layer, delta) if return_new_grad_ys else nabla_dx_layer

    def reset_params(self):
        """ Useful function for comparing different algorithms, resets the original paramater initializations
            so that comparisons can be accurate
        """
        self.weights = self.copy_of_weights
        self.biases = self.copy_of_biases
        self.shared_funcs = SharedFunctions.SharedFunctions(self.activations, [w.shape for w in self.weights], [b.shape for b in self.biases], self.scale_method, self.optimizer, self.lr, self.lr_decay)

    def get_weights(self):
        """ Returns neural network weights """
        return copy.deepcopy(self.weights)
    
    def get_biases(self):
        """ Returns neural network biases """
        return copy.deepcopy(self.biases)
    
    def set_weights(self, weights):
        """ Sets local neural network weights to copy of paramater weights """
        self.weights = copy.deepcopy(weights)
    
    def set_biases(self, biases):
        """ Sets local neural network biases to copy of paramater biases """
        self.biases = copy.deepcopy(biases)

    def output_scaler(self, x):
        """ Default scaler is 1*x, if custom output scaler is set then this will be replaced """
        return x
    
    def output_scaler_deriv(self, x):
        """ Default derivative of output scale is 1, if custom output scaler is set then this function 
            will be replaced
        """
        return np.ones(np.squeeze(x).shape)
    
    def custom_output_scaler(self, custom_func, custom_func_deriv):
        """ Set custom output scaler with your own ``custom_func``, ``custom_func_deriv`` must be
            present for backpropagation
        """
        self.output_scaler = custom_func
        self.output_scaler_deriv = custom_func_deriv
    
    def add_dropout(self, layer_idxs, rate):
        """ 
            dropout regularization
            ``layer_idxs`` are layers to apply dropout on in the input for
            ``rate`` specifies the dropout rate; 0.4, means 40% of the elements will be randomly dropped out during training.

            **IMPORTANT**
            For dropout to work correctly predictions and gradients must be called one after the other, otherwise the random dropout layers from predictions
            wont match the random dropout layers in backpropagation and then the gradients will be wrong.
        """
        if not (type(layer_idxs) is list or type(layer_idxs) is np.ndarray): layer_idxs = [layer_idxs]
        self.dropout_rate = rate
        self.dropout_idxs = layer_idxs

    def get_nprandomstate(self):
        return np.random.get_state()
    
    def set_nprandomstate(self, state):
        np.random.set_state(state)