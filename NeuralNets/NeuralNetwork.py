# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
import time
import math
from NeuralNets import FullyConnected as fc
from NeuralNets import ConvNet as cn
import copy

class NeuralNetwork:
    def __init__(self, sections, loss_fn="mean_squared_error"):
        self.sections = sections
        self.loss_fn = loss_fn
        self.regularizer = None
    
    def feedforward(self, inputs, scale=True, test=False):
        """ Return the outputs of feeding ``inputs`` through all the sections
            ``scale`` is default True, but the get_cost() method will set it to false so that data is not rescaled
            if scale is set false then we reset it to true after the first pass through a section, because the next section needs to scale it
            if ``test`` is true then things lke dropout will not be applied
        """
        a = copy.deepcopy(inputs)
        for section in self.sections:
            a = section.feedforward(a, scale, test)
            scale = True
        return a

    def sgd_fit(self, inputs, labels=None, grad_ys=None, batch_size=100, epochs=1, train_pct=1.0, print_epoch=False, shuffle_inputs=True, random_states=None):
        """ stochastic gradient descent - update weights at every step in batch 
             ``training_data`` is a list of tuples ``(x, y)`` representing the training inputs 
             and the desired outputs.
             ``batch_size`` = Size of the subset of examples to use when performing gradient descent during training.
             ``epochs`` = number of iterations we want to go through the training data
             ``train_pct`` is the split for training and testing data
             ``print_epoch`` is for early_stopping regularizer, it will print epoch when nn training stopped early
        """
        if (train_pct < 0.0 or train_pct > 1.0): sys.exit('Ooops! train_pct must be a float between 0 and 1 (inclusive)')
        self.sgd_test_costs = []
        self.sgd_train_costs = []
        random_index_order = np.arange(0, len(inputs))
        if shuffle_inputs: np.random.shuffle(random_index_order)
        inputs = inputs[random_index_order]
        if labels is not None: labels = labels[random_index_order]
        correct = []
        for epoch in range(epochs):
            for step in range(0, len(inputs), batch_size):
                if random_states is not None:
                    self.set_nprandomstates(random_states)
                else:
                    random_states = self.get_nprandomstates()
                train_x = inputs[step:step + math.floor(batch_size * train_pct)]
                train_y = None if labels is None else labels[step:step+math.floor(batch_size * train_pct)]
                self.set_nprandomstates(random_states)
                train_predictions = self.feedforward(train_x, scale=True)
                
                if labels is not None: 
                    test_x, test_y = inputs[step+math.floor(batch_size * train_pct):step+batch_size], labels[step+math.floor(batch_size * train_pct):step+batch_size]
                    
                    """ Update things: decay learning rate (if lr_decay > 0),
                                    update iteration_count,
                                    update train costs and test costs.
                    """
                    if (train_pct != 1.0):
                        self.set_nprandomstates(random_states)
                        test_predictions = self.feedforward(test_x, scale=True, test=True)
                        pct_correct = np.mean(np.squeeze(np.argmax(test_predictions, 1)) == np.argmax(test_y, 1))
                        correct.append(pct_correct)
                        avg_correct = sum(correct[-10:]) / len(correct[-10:])
                        if avg_correct > 0.95:break
                        print(test_predictions.shape, test_y.shape)
                        print(np.squeeze(np.argmax(test_predictions, 1)) == np.argmax(test_y,1), "\navg pct correct = ", avg_correct, "step =", (epoch + 1) * ((step+batch_size)/batch_size))
                        print(np.argmax(test_y, 1))
                        print(np.squeeze(np.argmax(test_predictions, 1)))

                        self.sgd_test_costs.append([(epoch + 1) * ((step+batch_size)/batch_size), np.sum(self.get_losses(test_predictions, test_y))])
                    else: self.sgd_test_costs.append([np.nan, np.nan])
                    self.sgd_train_costs.append([(epoch + 1) * ((step+batch_size)/batch_size), np.sum(self.get_losses(train_predictions,train_y))])
                
                """ Perform SGD on training data """
                train_loss_grad = self.loss_prime(train_predictions, train_y) if labels is not None else grad_ys
                grads = self.gradients(train_x, list(range(len(self.sections))), ["weights", "biases"], train_loss_grad, random_states)
                for section_idx, section in enumerate(self.sections):
                    nablaWs = grads[section_idx][0]
                    nablaBs = grads[section_idx][1]
                    """
                        nablaWs & nablaBs returns a multi-dimensional array where the first axis repersents batch_size,
                        so we average over the first axis
                    """
                    normalized_nablaWs = [np.mean(nabla_wLayer, 0) for nabla_wLayer in nablaWs]
                    normalized_nablaBs = [np.mean(nabla_bLayer, 0) for nabla_bLayer in nablaBs]
                    section.apply_gradients(zip([normalized_nablaWs], [normalized_nablaBs]))

                """ If regularizer is early_stopping, break from SGD when performance on testing data has not 
                    improved for a while and max ``patience`` has been reached
                """
                if (self.regularizer == "early_stopping"):
                    if ((np.min(np.array(self.sgd_test_costs)[:, 1]) < np.min(np.array(self.sgd_test_costs)[-self.patience:, 1])) and ((epoch + 1) * ((step+batch_size)/batch_size)) > self.patience):
                        if print_epoch: print("SGD stopped early at ", epoch)
                        break

    def gradients(self, inputs, section_layer, section_type, grad_ys=None, np_randomstates=None):
        """
            ``inputs`` = inputs
            ``section_layer`` is indexes of sections that you want gradients for
            ``section_type`` are strings, either "weights" or "biases" or "zLayer" that you want the gradients for
            ``grad_ys`` is starter gradient to start backpropagation with
            ``np_randomstates`` is a list of numpy random states, gotten before the feedforward was done for determining grad_ys
            returns list of gradients in form gradients[section_layer_idx][section_type_idx][gradients][numpy gradient array]
        """
        if not(type(section_layer) is list or type(section_layer) is np.ndarray): section_layer = [section_layer]
        if not (type(section_type) is list or type(section_type) is np.ndarray): section_type = [section_type]
        
        input_array = []
        a = copy.deepcopy(inputs)
        """ insures same output as outputs calculated for grad_ys"""
        if np_randomstates is not None: self.set_nprandomstates(np_randomstates)
        for section in self.sections:
            input_array.append(a)
            a = section.feedforward(a)

        returned_gradients = [[0 for section_type in section_type] for section_layer in section_layer]
        """ insures same output as outputs calculated for grad_ys"""
        if np_randomstates is not None: self.set_nprandomstates(np_randomstates)
        for section_idx, (section, section_inputs) in reversed(list(enumerate(zip(self.sections, input_array)))):
            dx_layer = []
            if type(section) is cn.ConvNet:
                if ("weights" in section_type or "biases" in section_type):
                    dx_layer.extend([idx + 2 for idx, layer_name in enumerate(section.layer_names) if layer_name is "conv"])
                if ("zLayer" in section_type):
                    dx_layer.extend([idx + 1 for idx, layer_name in enumerate(section.layer_names) if layer_name is "conv"])
            elif type(section) is fc.FullyConnected:
                if ("weights" in section_type or "biases" in section_type):
                    dx_layer.extend(np.arange(2, (len(section.activations) + 1)))
                if ("zLayer" in section_type):
                    dx_layer.extend(np.arange(0, (len(section.activations) + 2)))
            
            section_gradients, grad_ys = section.gradients(section_inputs, dx_layer, section_type, grad_ys, return_new_grad_ys=True)
            try:
                returned_gradients[section_idx] = section_gradients
            except Exception:
                pass 

        return returned_gradients


    def check_gradients(self, inputs, labels=None, grad_ys=None):
        """ 
            Checks gradients for each and every paramater in neural network.
            Prints updates along the way, returns nothing
        """
        epsilon = 1e-5
        num_inputs = inputs.shape[0]
        gradient_types = ["biases", "weights"]
        random_states = self.get_nprandomstates()
        if labels is None:
            calculated_gradients = self.gradients(inputs, [0, 1], gradient_types, grad_ys, random_states)
        else:
            self.set_nprandomstates(random_states)
            start_grad_y = self.loss_prime(self.feedforward(inputs), labels)
            calculated_gradients = self.gradients(inputs, [0, 1], gradient_types, start_grad_y, random_states)
        """ Check for each section """
        for section_idx, section in reversed(list(enumerate(self.sections))):
            """ Check for each gradient type """
            for gradient_type_idx, gradient_type in enumerate(gradient_types):
                if (gradient_type == "weights"):
                    section_params = section.get_weights()
                elif (gradient_type == "biases"):
                    section_params = section.get_biases()
                """ Check for each gradient type layer in section """
                for idx in range(len(section_params)):
                    if section_params[idx] is None: calculated_gradients[section_idx][gradient_type_idx].insert(idx, None)
                        
                for param_layer_idx, param_layer in enumerate(section_params):
                    if param_layer is not None: 
                        plus_section_params = copy.deepcopy(section_params)
                        minus_section_params = copy.deepcopy(section_params)
                        param_layer_shape = section_params[param_layer_idx].shape
                        flattened_param_layer = param_layer.flatten('F')
                        flattened_param_gradients = calculated_gradients[section_idx][gradient_type_idx][param_layer_idx].reshape((num_inputs, -1), order='F')
                        """ Check for each individual weight/bias in the gradient type layer of the current section """
                        for param_idx in range(len(flattened_param_layer)):
                            """ Positive change in x """
                            plus_params = copy.deepcopy(flattened_param_layer)
                            plus_params[param_idx] += epsilon
                            plus_params = plus_params.reshape(param_layer_shape, order='F')
                            plus_section_params[param_layer_idx] = plus_params
                            if (gradient_type == "weights"):
                                section.set_weights(plus_section_params)
                            elif (gradient_type == "biases"):
                                section.set_biases(plus_section_params)
                            self.set_nprandomstates(random_states)
                            plus_output = self.feedforward(inputs)
                            if labels is not None: plus_output = self.loss(plus_output, labels)
                            if grad_ys is not None: plus_output *= grad_ys

                            """ Negative change in x """
                            minus_params = copy.deepcopy(flattened_param_layer)
                            minus_params[param_idx] -= epsilon
                            minus_params = minus_params.reshape(param_layer_shape, order='F')
                            minus_section_params[param_layer_idx] = minus_params
                            if (gradient_type == "weights"):
                                section.set_weights(minus_section_params)
                            elif (gradient_type == "biases"):
                                section.set_biases(minus_section_params)
                            self.set_nprandomstates(random_states)
                            minus_output = self.feedforward(inputs)
                            if labels is not None: minus_output = self.loss(minus_output, labels)
                            if grad_ys is not None: minus_output *= grad_ys
  

                            """ Approx. gradient = (f(x+h)−f(x−h))/(2h) """
                            approximated_param_gradient = ((plus_output - minus_output) / (2 * epsilon))
                            approximated_grads = approximated_param_gradient.reshape((num_inputs,-1)).sum(1)

                            
                            
                            calculated_grads = np.squeeze(flattened_param_gradients[:, param_idx])
                            #print(flattened_param_gradients.shape, param_layer.shape, approximated_grads.sum())


                            """ If Approximated gradient does not equal calculated gradient with tolerence 1e-7 then we messed up """
                            if (np.allclose(np.squeeze(approximated_grads), np.squeeze(flattened_param_gradients[:, param_idx]), rtol=1e-7) is False):
                                print("Approximated gradients DO NOT equal Calculated gradients for {} in section:{}, layer:{}, param_idx:{}".format(gradient_type, section_idx, param_layer_idx, param_idx))
                                sys.exit("Approximated gradients:{}\nCalculated gradients:{}\nDifference:{}".format(approximated_grads, calculated_grads, (calculated_grads-approximated_grads)))
                                    
                                
                            
                            """ Reset paramaters to their original values """
                            if (gradient_type == "weights"):
                                section.set_weights(section_params)
                            elif (gradient_type == "biases"):
                                section.set_biases(section_params)

                            print("Approximated gradients equal Calculated gradients for {} in section:{}, layer:{}, param_idx:{}".format(gradient_type, section_idx, param_layer_idx, param_idx))

    def loss(self, a, y):
        if type(a) is not np.ndarray: a = np.array(a)
        if type(y) is not np.ndarray: y = np.array(y)
        a = copy.deepcopy(a)
        y = copy.deepcopy(y).reshape(a.shape)

        if (self.loss_fn == "mean_squared_error"):
            return ((y-a)**2)*0.5
        elif (self.loss_fn == "cross_entropy"):
            return -np.sum((y*np.log(np.maximum(a, 1e-9))),1,keepdims=True)
        elif (self.loss_fn == "huber"):
            return np.where(np.abs(y-a)<=1,0.5*(y-a)**2,1*(np.abs(y-a)-0.5*1))
        elif (self.loss_fn == "hinge"):
            return max(0, 1 - a * y)
        elif (self.loss_fn == "absolute_difference"):
            return np.abs((y-a))

    
    def loss_prime(self, a, y):
        if type(a) is not np.ndarray: a = np.array(a)
        if type(y) is not np.ndarray: y = np.array(y)
        a = copy.deepcopy(a)
        y = copy.deepcopy(y).reshape(a.shape)

        if (self.loss_fn == "mean_squared_error"):
            return (a-y)
        elif (self.loss_fn == "cross_entropy"):
            return (a-y)
        elif (self.loss_fn == "huber"):
            return np.where(np.abs(y-a)<=1,(a-y),-((y-a)/np.abs(y-a)))
        elif (self.loss_fn == "hinge"):
            return np.where((1 - a * y)<= 0, 0, -y)
        elif (self.loss_fn == "absolute_difference"):
            return -((y-a)/np.abs(y-a))
    
    def get_losses(self, predictions, labels):
        """ 
            gets average cost using current network to make predictions for inputs and comparing to labels 
        """
        if not (isinstance(predictions, (list, np.ndarray)) and isinstance(labels, (list, np.ndarray))):
            sys.exit('Ooops! inputs and labels for get_cost() must be passed as list or list in numpy array')
        costs = []
        for a, y in zip(predictions, labels):
            costs.append(self.loss(a, y))
        if (len(costs) > 0): return sum(costs)/len(costs)
        else: return 0
    
    def set_regularizer(self, regularizer, reg_lambda=0, patience=0):
        """ Set regularizer paramaters
            ``regularizer`` is a string, either "early_stopping" or "l1" or "l2"
            ``reg_lambda`` is the weight decay rate for "l1" or "l2" regularization
            ``patience`` is how many epochs or steps we will wait without getting a decrease in cost for "early_stopping" regularization

        """
        self.regularizer = regularizer
        self.patience = patience
        for section in self.sections:
            section.shared_funcs.regularizer = regularizer
            section.shared_funcs.reg_lambda = reg_lambda
    
    def get_weights(self):
        """
            return a list of weights for each section
        """
        weights = []
        for section in self.sections:
            weights.append(section.get_weights())
        return weights

    def set_weights(self, sectionWeights):
        """
            ``sectionWeights`` is a list of weights for each sections
            sets a list of weights for each section
        """
        for section,weights in zip(self.sections, sectionWeights):
            section.set_weights(weights)
    
    def get_biases(self):
        """
            return a list of biases for each section
        """
        biases = []
        for section in self.sections:
            biases.append(section.get_biases())
        return biases

    def set_biases(self, sectionBiases):
        """
            ``sectionBiases`` is a list of biases for each sections
            sets a list of biases for each section
        """
        for section,biases in zip(self.sections, sectionBiases):
            section.set_biases(biases)

    def get_nprandomstates(self):
        """
            return a list of np.random.get_state() for each section
        """
        np_randomstates = []
        for section in self.sections:
            np_randomstates.append(section.get_nprandomstate())
        return np_randomstates
    
    def set_nprandomstates(self, states):
        """
            ``states`` is a list of np.random.get_state() for each section
            applies each numpy random state to each section
        """
        for section, state in zip(self.sections, states):
            section.set_nprandomstate(state)

            

                

