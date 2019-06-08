# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from NeuralNets import SharedFunctions
import copy
class ConvNet:
    def __init__(self, conv_method, layer_names, num_filters, kernel_sizes, stride_sizes, pool_sizes, pool_fns, pad_fns, activations, input_channels, scale_method=None, optimizer="adam", lr=0.01, lr_decay=0):
        """
        ``conv_method`` must be a string, either "convolution" or "cross_correlation" --> (ex. conv_method="convolution")
        ``layer_names`` must be list, in order of operations that are wanted to be done, strings, either conv" or "pool, --> (ex. layer_names=["conv", "pool", "conv", "pool"])
        ``num_filters`` must be list, is number of kernel filters to apply for each conv_method in layer_names (order matters), if pooling layer then put None. --> (ex. num_filters=[32, None, 64, None])
        ``kernel_sizes`` must be list of lists with [kernel_height, kernel_width], if pooling layer then put None. --> (ex. kernel_sizes=[[5,5], None, [5,5], None])
        ``stride_sizes`` must be list of lists with [stride_height, stride_width] for each layer --> (ex. stride_sizes=[[1,1], [2,2], [1,1], [2,2]])
        ``pool_sizes`` must be a list of lists with [pool_height, pool_width] for each pooling layer, if conv layer then put None. --> (ex. pool_sizes=[None, [2,2], None, [2,2]])
        ``pool_fns`` must be a list of strings, either "max" or "mean", with pooling functions for each layer, if conv layer then put None. --> (ex. pool_fns=[None, "max", None, "mean"])
        ``pad_fns`` must be a list of strings, either "same" or "valid", for each layer in layer_names --> (ex. pad_fns=["valid", "same", "valid", "same"])
        ``activations`` must be a list of strings for each conv layer in layer_names, either "linear" "sigmoid" or "softmax" or "tanh" or "relu" or "leaky_relu", put None if pool layer --> (ex. activations=["relu", None, "relu", None])
        ``input_channels`` is an int, number of original input channels, (rgb images have 3 input_channels) --> (ex. input_channels=3)
        ``scale_method`` is a string, to scale the data so it in a smaller range, either "normalize" or "standardize" or None for none
        ``optimizer`` is to optimize the speed that our neural network learns, a string either "sgd," "momentum," "nesterov," "adagrad," "rmsprop," "adam," "adamax," or "nadam"
        ``lr`` = learning rate, how fast our neural network learns the cost function gradient, usually small like 0.01
        ``lr_decay`` is how fast we decay the learning rate, 0=No decay, 0.1=Fast decay

        Lists: ``layer_names, num_filters, kernel_sizes, stride_sizes, pool_sizes, pool_fns, pad_fns, activations`` should all be the same length

        full example:
        convnet = ConvNet(
            conv_method="convolution",
            layer_names=["conv", "pool", "conv", "pool"],
            num_filters=[32, None, 64, None],
            kernel_sizes=[[5,5], None, [5,5], None],
            stride_sizes=[[1,1], [2,2], [1,1], [2,2]],
            pool_sizes=[None, [2,2], None, [2,2]],
            pool_fns=[None, "max", None, "max"],
            pad_fns=["same", "valid", "same", "valid"],
            activations=["relu", None, "relu", None],
            input_channels=3,
            scale_method="normalize",
            optimizer="nadam",
            lr=0.01,
            lr_decay=0
        )
        """


        """ Check paramaters """
        if not (conv_method == "convolution" or conv_method == "cross_correlation"):
            sys.exit('Ooops! ``conv_method`` must equal "convolution" or "cross_correlation" ')
        if not (len(layer_names) == len(num_filters) == len(kernel_sizes) == len(stride_sizes) == len(pool_sizes) == len(pool_fns) == len(pad_fns) == len(activations)):
            sys.exit('Ooops! `layer_names, num_filters, kernel_sizes, stride_sizes, pool_sizes, pool_fns, pad_fns, activations`` should all be lists that are the same length ')
        if (False in [True if layer_name is "conv" or layer_name is "pool" else False for layer_name in layer_names]):
            sys.exit('Ooops! ``layer_names`` can only contain "conv" or "pool" ')
        if (False in [True if (((type(filterNum) is int) and (layer_name is "conv")) or ((filterNum is None) and (layer_name is "pool"))) else False for filterNum, layer_name in zip(num_filters, layer_names)]):
            sys.exit('Ooops! ``num_filters`` can only contain ints in the same indexes as "conv" in ``layer_names``, None if layer_name index is "pool" ')
        if (False in [True if (((type(kernelSize) is list and len(kernelSize) is 2) and (layer_name is "conv")) or ((kernelSize is None) and (layer_name is "pool"))) else False for kernelSize, layer_name in zip(kernel_sizes, layer_names)]):
            sys.exit('Ooops! ``kernel_sizes`` can only contain list[kernel_height, kernel_width] in the same indexes as "conv" in ``layer_names``, None if layer_name index is "pool" ')
        if (False in [True if (((type(strideSize) is list and len(strideSize) is 2))) else False for strideSize in stride_sizes]):
            sys.exit('Ooops! ``stride_sizes`` can only contain list of [stride_height, stride_width] for each layer_name in layer_names')
        if (False in [True if (((type(poolSize) is list and len(poolSize) is 2) and (layer_name is "pool")) or ((poolSize is None) and (layer_name is "conv"))) else False for poolSize, layer_name in zip(pool_sizes, layer_names)]):
            sys.exit('Ooops! ``pool_sizes`` can only contain lists of [pool_height, pool_width] in the same indexes as "pool" in ``layer_names``, None if layer_name index is "conv" ')
        if (False in [True if (((poolFn == "max" or poolFn == "min" or poolFn == "mean") and (layer_name is "pool")) or ((poolFn is None) and (layer_name is "conv"))) else False for poolFn, layer_name in zip(pool_fns, layer_names)]):
            sys.exit('Ooops! ``pool_fns`` can only contain lists of pool_fns (strings either "max", "mean", or "min") in the same indexes as "pool" in ``layer_names``, None if layer_name index is "conv" ')
        if (False in [True if (padFn == "same" or padFn == "valid") else False for padFn in pad_fns]):
            sys.exit('Ooops! ``pad_fns`` can only contain lists of pad_fns (strings either "same", or "valid")')
        if (False in [True if (((activation == "linear" or activation == "sigmoid" or activation == "softmax" or activation == "tanh" or activation == "relu" or activation == "leaky_relu") and (layer_name is "conv")) or ((activation is None) and (layer_name is "pool"))) else False for activation, layer_name in zip(activations, layer_names)]):
            sys.exit('Ooops! ``pool_fns`` can only contain lists of activations (strings either be "linear" or "sigmoid" or "softmax" or "tanh" or "relu" or "leaky_relu") in the same indexes as "conv" in ``layer_names``, None if layer_name index is "pool" ')
        if (type(input_channels) is not int):
            sys.exit('Ooops! ``input_channels`` must be an int (for rgb image, use 3)')

        self.conv_method = conv_method
        self.layer_names = layer_names
        self.num_filters = num_filters
        self.stride_sizes = stride_sizes
        self.pool_sizes = pool_sizes
        self.pool_fns = pool_fns
        self.pad_fns = pad_fns
        self.activations = activations
        self.scale_method = scale_method
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay = lr_decay
        self.initialize_weights(kernel_channels=[input_channels] + [(num_filters[self.get_last_convLayer_idx(idx)] if num_filters[idx] is not None else None) for idx in range(1,len(num_filters))], kernel_sizes=kernel_sizes)
        self.shared_funcs = SharedFunctions.SharedFunctions(activations, [k.shape for k in self.kernel_layers if k is not None], [b.shape for b in self.bias_layers if b is not None], scale_method, optimizer, lr, lr_decay)
        self.initilize_hyperparams()

    
    def initialize_weights(self, kernel_channels, kernel_sizes):
        self.kernel_layers = []
        self.bias_layers = []

        for layer_idx in range(len(self.layer_names)):
            if (self.layer_names[layer_idx] == "conv"):
                kernel_rows, kernel_cols, kernel_channel = kernel_sizes[layer_idx][0], kernel_sizes[layer_idx][1], kernel_channels[layer_idx]
                layer_filters_num = self.num_filters[layer_idx]
                self.kernel_layers.append(np.random.randn(layer_filters_num, kernel_rows, kernel_cols, kernel_channel)*np.sqrt(2/(layer_filters_num*kernel_rows*kernel_cols*kernel_channel)))
                self.bias_layers.append(np.random.rand(layer_filters_num, 1, 1, 1))
            else:
                self.kernel_layers.append(None)
                self.bias_layers.append(None)
        
        self.copy_of_kernel_layers = np.copy(self.kernel_layers)
        self.copy_of_bias_layers = np.copy(self.bias_layers)
    
    def initilize_hyperparams(self):
        self.iteration_count = 0
    
    def feedforward(self, inputs, scale=True, test=False):
        """ Return the outputs of the network for ``inputs``
            ``inputs`` is a 4d numpy array (batch_size, width, height, channels)
            ``scale`` is default True, but the get_losses() method will set it to false so that data is not rescaled
        """
        """ Check params """
        if (type(inputs) is not np.ndarray or (len(inputs.shape) is not 4)):
            sys.exit('Ooops! feedforward inputs must be 4d numpy array (batch_size, width, height, channels)')

        a = inputs
        if (self.scale_method is not None and scale == True and self.iteration_count > 0): inputs = self.shared_funcs.scale_fn(inputs, train=False)
        for idx, layer_name in enumerate(self.layer_names):
            if layer_name == "conv":
                a = self.pad(a, idx)
                a = self.convolution(a, idx)
                a = self.shared_funcs.activation_fn(a, idx)
            elif layer_name == "pool":
                a = self.pad(a, idx)
                a = self.pool(a, idx)
        return a

    def apply_gradients(self, gradients):
        """ Applies gradients to weights and biases
            ``gradients`` is a tuple of lists ``([nablaWs], [nablaBs])`` representing for training data
        """
        self.iteration_count += 1
        for nablaWs, nablaBs in gradients:
            new_kernels, new_biases = self.shared_funcs.optimize(nablaWs, nablaBs, [kernel_layer for kernel_layer in self.kernel_layers if kernel_layer is not None], [bias_layer for bias_layer in self.bias_layers if bias_layer is not None], self.iteration_count)
            self.kernel_layers, self.bias_layers = [new_kernels.pop(0) if kernel_layer is not None else None for kernel_layer in self.kernel_layers], [new_biases.pop(0) if bias_layer is not None else None for bias_layer in self.bias_layers]

    def gradients(self, inputs, dx_layer, dx_type, grad_ys=None, return_new_grad_ys=False):
        """ Calculates derivitaves of dx_layer w.r.t dy_layer
            ``inputs`` is a 4d numpy array (batch_size, width, height, channels)
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
        if (type(inputs) is not np.ndarray or (len(inputs.shape) is not 4)):
            sys.exit('Ooops! gradients inputs must be 4d numpy array (batch_size, width, height, channels)')
        if (dx_layer.dtype != int):
            sys.exit('Ooops! dx_layer must be integer or a list of integers, layer numbers starting at 1. (Weights and biases have the same layer number as the hidden or output layer ahead of them.)')
        if not ("weights" in dx_type or "biases" in dx_type or "zLayer" in dx_type ):
            sys.exit('Ooops! dx_type must equal "weights", "biases", or "zLayer"')

        """ feedforward with inputs, store outputs, z for each layer 
        """
        z = inputs
        z_layers = [inputs]
        for idx, layer_name in enumerate(self.layer_names):
            if layer_name == "conv":
                z = self.pad(z, idx)
                z_layers.append(z)
                z = self.convolution(z, idx)
                z_layers.append(z)
                z = self.shared_funcs.activation_fn(z, idx)
            elif layer_name == "pool":
                z = self.pad(z, idx)
                z_layers.append(z)
                z = self.pool(z, idx)
                z_layers.append(z)
        """ Begin Backpropagation
            Multiply grad_ys, "starting" backprop value, times delta, looping through layers backwards
            store dx w.r.t grad_ys in nabla_dx_layer, for each layer in dx_layer          
        """
        grad_ys = np.ones(np.array(z_layers[-1]).shape) if (grad_ys is None) else np.array(grad_ys).reshape(np.array(z_layers[-1]).shape)
        grad_ys *= self.shared_funcs.activation_fn_prime(z_layers[-1], self.get_last_convLayer_idx(len(self.layer_names)))
        delta_layers = [grad_ys]
        nabla_dx_layer = [[0 for dx_layer in dx_layer] for dx_type in dx_type]
        
        for layer_idx, layer_name in reversed(list(enumerate(self.layer_names))):
            """ nabla dx layer idx for weights and biases """
            nabla_dwb_layer_idx = np.squeeze(np.argwhere(dx_layer == layer_idx+2))
            """ nabla dx layer idx for hidden layers and inputs """
            nabla_dh_layer_idx = np.squeeze(np.argwhere(dx_layer == layer_idx+1))
            
            last_conv_layer_idx = self.get_last_convLayer_idx(layer_idx)
            if layer_name == "conv":
                idx = len(delta_layers)
                if last_conv_layer_idx != -1:
                    delta = self.get_delta_derivs(self.shared_funcs.activation_fn_prime(z_layers[-idx - 1], last_conv_layer_idx), delta_layers[-1], layer_idx)
                    if (nabla_dwb_layer_idx.size > 0 and "weights" in dx_type):
                        """Append kernel deriv to the correct nabla_dwb_layer_idx location """
                        nabla_dx_layer[dx_type.index("weights")][nabla_dwb_layer_idx] = (self.get_kernel_derivs(self.shared_funcs.activation_fn(z_layers[-idx - 1], last_conv_layer_idx), delta_layers[-1], layer_idx) + self.shared_funcs.weight_decay_deriv(self.kernel_layers[layer_idx]))
                else:
                    delta = self.get_delta_derivs(z_layers[-idx - 1], delta_layers[-1], layer_idx) 
                    if (nabla_dwb_layer_idx.size > 0 and "weights" in dx_type):
                        """Append kernel deriv to the correct nabla_dwb_layer_idx location """
                        nabla_dx_layer[dx_type.index("weights")][nabla_dwb_layer_idx] = (self.get_kernel_derivs(z_layers[-idx - 1], delta_layers[-1], layer_idx) + self.shared_funcs.weight_decay_deriv(self.kernel_layers[layer_idx]))

                if (nabla_dwb_layer_idx.size > 0 and "biases" in dx_type):
                    """Append bias deriv to the correct nabla_dwb_layer_idx location """
                    if (self.layer_names[-1] == "pool" and nabla_dwb_layer_idx == (len(self.layer_names)-1)):
                        nabla_dx_layer[dx_type.index("biases")][nabla_dwb_layer_idx] = np.sum(delta_layers[-3], axis=(1, 2)).reshape((-1,) + self.bias_layers[layer_idx].shape)
                    else:
                        """ Go back 3 to grad y if last layer is pool """
                        nabla_dx_layer[dx_type.index("biases")][nabla_dwb_layer_idx] = np.sum(delta_layers[-1], axis=(1, 2)).reshape((-1,) + self.bias_layers[layer_idx].shape)
                
                delta_layers.append(delta)

                idx = len(delta_layers)
                delta = self.get_pad_derivs(z_layers[-idx - 1], delta_layers[-1], layer_idx)                
                delta_layers.append(delta)
                if (nabla_dh_layer_idx.size > 0 and ("input" in dx_type or "zLayer" in dx_type)):
                    """Append delta deriv to the correct nabla_dh_layer_idx location """
                    nabla_dx_layer[dx_type.index("zLayer")][nabla_dh_layer_idx] = delta
            elif layer_name == "pool":
                idx = len(delta_layers)
                delta = self.get_pool_derivs(z_layers[-idx - 1], z_layers[-idx], delta_layers[-1], layer_idx)
                if last_conv_layer_idx != -1: delta *= self.shared_funcs.activation_fn_prime(z_layers[-idx - 1], last_conv_layer_idx)
                delta_layers.append(delta)
                
                idx = len(delta_layers)
                delta = self.get_pad_derivs(z_layers[-idx - 1], delta_layers[-1], layer_idx)
                delta_layers.append(delta)

        """ remove extra gradient slots that are 0 and we didn't fill """
        nabla_dx_layer = [list(filter(lambda a: a is not 0, x)) for x in nabla_dx_layer]
        return (nabla_dx_layer, delta) if return_new_grad_ys else nabla_dx_layer

    def convolution(self, input_layer, layer_idx):
        """
            ``input_layer`` == aˡ⁻¹
            ``layer_idx`` == layer index of "conv" in layer names
            returns convolved zˡ --> (wˡaˡ⁻¹ + bˡ)
        """
        """ Set local variables """
        kernel_layer = self.kernel_layers[layer_idx]
        bias_layer = self.bias_layers[layer_idx]
        stride_size = self.stride_sizes[layer_idx]

        """ Start convolving input layer """
        num_inputs, num_rows, num_cols = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2]
        num_filters, kernel_rows, kernel_cols, num_channels = kernel_layer.shape[0], kernel_layer.shape[1], kernel_layer.shape[2], kernel_layer.shape[3]
        stride_rows, stride_cols = stride_size[0], stride_size[1]
        output_rows, output_cols = int(((num_rows - kernel_rows) / stride_rows) + 1), int(((num_cols - kernel_cols) / stride_cols) + 1)
        output_layer = np.zeros((num_inputs, output_rows, output_cols, num_filters))
        for output_row in range(output_rows):
            for output_col in range(output_cols):
                z = np.array(np.tensordot(input_layer[:, stride_rows * output_row:stride_rows * output_row + kernel_rows, stride_cols * output_col:stride_cols * output_col + kernel_cols,:], np.rot90(kernel_layer, 2, axes=(1, 2)), axes=((1, 2, 3), (1, 2, 3))), ndmin=2)
                z += bias_layer.reshape((1, -1))
                output_layer[:, output_row, output_col, :] = z
        return output_layer

    def pool(self, input_layer, layer_idx):
        """
            ``input_layer`` == aˡ
            ``layer_idx`` == layer index of "pool" in layer names
            returns pooled aˡ using pool_fn[layer_idx]
        """
        """ Set local variables """
        pool_size = self.pool_sizes[layer_idx]
        stride_size = self.stride_sizes[layer_idx]
        pool_fn = self.pool_fns[layer_idx]

        """ Start pooling input layer """
        num_inputs, num_rows, num_cols, num_channels = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
        pool_rows, pool_cols, stride_rows, stride_cols = pool_size[0], pool_size[1], stride_size[0], stride_size[1]
        output_pool_rows = int(((num_rows - pool_rows) / stride_rows) + 1)
        output_pool_cols = int(((num_cols - pool_cols) / stride_cols) + 1)
        output_layer = np.zeros((num_inputs, output_pool_rows, output_pool_cols, num_channels))
        for output_pool_row_idx in range(output_pool_rows):
            for output_pool_col_idx in range(output_pool_cols):
                input_start_row, input_start_col = (stride_rows * output_pool_row_idx), (stride_cols * output_pool_col_idx)
                if (pool_fn == "max"):
                    z = np.squeeze(np.max(input_layer[:, input_start_row:(input_start_row + pool_rows), input_start_col:(input_start_col + pool_cols),:], axis=(1, 2))).reshape((num_inputs, num_channels))
                elif (pool_fn == "mean"):
                    z = np.squeeze(np.mean(input_layer[:, input_start_row:(input_start_row + pool_rows), input_start_col:(input_start_col + pool_cols),:], axis=(1, 2))).reshape((num_inputs, num_channels))
                output_layer[:, output_pool_row_idx, output_pool_col_idx,:] = z
        return output_layer

    def pad(self, input_layer, layer_idx):
        """
            ``input_layer`` == aˡ
            ``layer_idx`` == layer index of "pad" in layer names
            returns padded aˡ using pad_fn[layer_idx]
        """
        """ Set local variables """
        if (self.layer_names[layer_idx] == "conv"):
            filter_size = [self.kernel_layers[layer_idx].shape[1], self.kernel_layers[layer_idx].shape[2]]
        elif (self.layer_names[layer_idx] == "pool"):
            filter_size = self.pool_sizes[layer_idx]
        stride_size = self.stride_sizes[layer_idx]
        padding_fn = self.pad_fns[layer_idx]
        
        """ Start padding input layer """
        num_inputs, input_rows, input_cols, num_channels = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
        filter_rows, filter_cols, stride_rows, stride_cols = filter_size[0], filter_size[1], stride_size[0], stride_size[1]
        if (padding_fn == "same"):
            width_padding = (input_cols * stride_cols) + filter_cols - input_cols - stride_cols
            height_padding = (input_rows * stride_rows) + filter_rows - input_rows - stride_rows
            row_padding_right = int(np.ceil(width_padding / 2))
            row_padding_left = int(np.floor(width_padding / 2))
            col_padding_bottom = int(np.ceil(height_padding / 2))
            col_padding_top = int(np.floor(height_padding / 2))
            padded_inputs = np.pad(input_layer, [(0,0), (col_padding_top, col_padding_bottom), (row_padding_left, row_padding_right), (0,0)], mode='constant')
        elif (padding_fn == "valid"):
            max_num_rows = (int)(((input_rows - filter_rows) / stride_rows) + 1)
            max_num_cols = (int)(((input_cols - filter_cols) / stride_cols) + 1)
            padded_inputs = input_layer[:, :(filter_rows + (stride_rows * (max_num_rows - 1))), :(filter_cols + (stride_cols * (max_num_cols - 1))), :]
        return padded_inputs


    def get_delta_derivs(self, input_layer, delta_layer, layer_idx):
        """
            ``input_layer`` == aˡ
            ``delta_layer`` == δˡ⁺¹
            ``layer_idx`` == layer index of "conv" in layer names for location delta
            returns daˡ w.r.t δˡ⁺¹
        """
        """ Set local variables """
        kernel_layer = self.kernel_layers[layer_idx]
        stride_size = self.stride_sizes[layer_idx]

        """ Start getting d zˡ w.r.t δˡ⁺¹ --> producing  δˡ """
        num_inputs, input_rows, input_cols, input_channels = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
        num_filters, kernel_rows, kernel_cols = kernel_layer.shape[0], kernel_layer.shape[1], kernel_layer.shape[2]
        delta_rows, delta_cols = delta_layer.shape[1], delta_layer.shape[2]
        stride_rows, stride_cols = stride_size[0], stride_size[1] 
        delta_derivs = np.zeros(input_layer.shape)
        for delta_row in range(delta_rows):
            for delta_col in range(delta_cols):
                z = np.squeeze(np.tensordot(delta_layer[:, delta_row:delta_row + 1, delta_col:delta_col + 1,:], np.rot90(kernel_layer, 2, axes=(1, 2)), axes=((3), (0)))).reshape((num_inputs, kernel_rows, kernel_cols, input_channels))
                delta_derivs[:, stride_rows * delta_row:stride_rows * delta_row + kernel_rows, stride_cols * delta_col:stride_cols * delta_col + kernel_cols,:] += z
        return delta_derivs

    def get_kernel_derivs(self, input_layer, delta_layer, layer_idx):
        """
            ``input_layer`` == aˡ⁻¹
            ``delta_layer`` == δˡ
            ``layer_idx`` == layer index of "conv" in layer names for kernel_deriv
            returns d kernelsˡ w.r.t δˡ 

        """
        """ Set local variables """
        kernel_shape = self.kernel_layers[layer_idx].shape
        stride_size = self.stride_sizes[layer_idx]

        """ Start getting d kernelsˡ w.r.t δˡ """
        num_filters, kernel_rows, kernel_cols, kernel_channels = kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]
        num_inputs, delta_rows, delta_cols = delta_layer.shape[0], delta_layer.shape[1], delta_layer.shape[2]
        stride_rows, stride_cols = stride_size[0], stride_size[1]
        kernel_derivs = np.zeros((num_inputs, num_filters, kernel_rows, kernel_cols, kernel_channels))
        
        for delta_row in range(delta_rows):
            for delta_col in range(delta_cols):
                inputs = copy.deepcopy(input_layer)
                deltas = copy.deepcopy(delta_layer)
                inputs = np.repeat(np.array(inputs[:, stride_rows * delta_row:stride_rows * delta_row + kernel_rows, stride_cols * delta_col:stride_cols * delta_col + kernel_cols,:], ndmin=5), num_filters, 0)
                deltas = np.repeat(np.array(deltas[:, delta_row:delta_row + 1, delta_col:delta_col + 1,:], ndmin=5), kernel_channels, 0)
                inputs = np.swapaxes(inputs, 0, 1)
                deltas = np.swapaxes(np.swapaxes(deltas, 0, 1), 1, 4)
                z = (inputs * deltas)
                kernel_derivs += np.rot90(z,2,axes=(2, 3))
        return kernel_derivs

    def get_pool_derivs(self, input_layer, pool_layer, delta_layer, layer_idx):
        """ 
            ``input_layer`` == pre-pool aˡ
            ``pool_layer`` == post-pool aˡ
            ``delta_layer`` == dδˡ⁺¹ 
            ``layer_idx`` = layer index of "pool" in layer names 
            (``delta_layer`` shape == ``pool_layer`` shape)
            returns d pre-pool aˡ w.r.t δˡ⁺¹ 
        """
        """ Set local variables """
        pool_size = self.pool_sizes[layer_idx]
        stride_size = self.stride_sizes[layer_idx]
        pool_fn = self.pool_fns[layer_idx]

        """ Start getting d pre-pool aˡ w.r.t δˡ⁺¹ """
        pool_rows, pool_cols, stride_rows, stride_cols = pool_size[0], pool_size[1], stride_size[0], stride_size[1]
        num_inputs, num_pool_rows, num_pool_cols, num_channels = pool_layer.shape[0], pool_layer.shape[1], pool_layer.shape[2], pool_layer.shape[3]
        delta_rows, delta_cols = delta_layer.shape[1], delta_layer.shape[2]
        input_layer_derivs = np.zeros(input_layer.shape)
        for pool_row_idx in range(delta_rows):
            for pool_col_idx in range(delta_cols):
                delta_val = delta_layer[:, pool_row_idx, pool_col_idx, :]
                input_start_row, input_start_col = (stride_rows * pool_row_idx), (stride_cols * pool_col_idx)
                if (pool_fn == "max"):
                    i, r, c, ch = np.where(np.isin(input_layer[:, input_start_row:(input_start_row + pool_rows), input_start_col:(input_start_col + pool_cols),:], pool_layer[:, pool_row_idx, pool_col_idx,:]))
                    input_layer_derivs[i, input_start_row + r, input_start_col + c, ch] += delta_layer[i, pool_row_idx, pool_col_idx, ch]
                elif (pool_fn == "mean"):
                    input_layer_derivs[:, input_start_row:input_start_row+pool_rows, input_start_col:input_start_col+pool_cols, :] += (delta_layer[:, pool_row_idx, pool_col_idx, :]/(pool_rows*pool_cols)).reshape((num_inputs,1,1,num_channels))
        return input_layer_derivs

    def get_pad_derivs(self, input_layer, delta_layer, layer_idx):
        """ 
            ``input_layer`` == pre-pad aˡ
            ``delta_layer`` == dδˡ⁺¹ 
            ``layer_idx`` = layer index of input_layer
            (``delta_layer`` shape == post-pad aˡ shape)
            returns d pre-pad aˡ w.r.t δˡ⁺¹ 
        """
        """ Set local variables """
        if (self.layer_names[layer_idx] == "conv"):
            filter_size = [self.kernel_layers[layer_idx].shape[1], self.kernel_layers[layer_idx].shape[2]]
        elif (self.layer_names[layer_idx] == "pool"):
            filter_size = self.pool_sizes[layer_idx]
        stride_size = self.stride_sizes[layer_idx]
        padding_fn = self.pad_fns[layer_idx]

        """ Start getting d pre-pad aˡ w.r.t δˡ⁺¹  """
        num_inputs, input_rows, input_cols, num_channels = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
        filter_rows, filter_cols, stride_rows, stride_cols = filter_size[0], filter_size[1], stride_size[0], stride_size[1]
        if (padding_fn == "same" and input_layer.shape != delta_layer.shape):
            width_padding = (input_cols * stride_cols) + filter_cols - input_cols - stride_cols
            height_padding = (input_rows * stride_rows) + filter_rows - input_rows - stride_rows
            row_padding_right = int(np.ceil(width_padding / 2))
            row_padding_left = int(np.floor(width_padding / 2))
            col_padding_bottom = int(np.ceil(height_padding / 2))
            col_padding_top = int(np.floor(height_padding / 2))
            padded_inputs_derivs = delta_layer[:, col_padding_top:(-col_padding_bottom if col_padding_bottom != 0 else None), row_padding_left:(-row_padding_right if row_padding_right != 0 else None),:]
        elif (padding_fn == "valid" or input_layer.shape == delta_layer.shape):
            max_num_rows = (int)(((input_rows - filter_rows) / stride_rows) + 1)
            max_num_cols = (int)(((input_cols - filter_cols) / stride_cols) + 1)
            cut_bottom_rows = input_rows - (filter_rows + (stride_rows * (max_num_rows - 1)))
            cut_right_cols = input_cols - (filter_cols + (stride_cols * (max_num_cols - 1)))
            padded_inputs_derivs = np.pad(delta_layer, [(0,0), (0, cut_bottom_rows), (0, cut_right_cols), (0, 0)], mode='constant')
        return padded_inputs_derivs

    def get_last_convLayer_idx(self, idx):
        """ 
            ``idx`` == layer idx to start looking backwards from
            Returns last conv layer idx (in self.layer_names) going backwards from current layer idx
            Returns -1 if no previous conv layer
        """
        try:
            return (idx - list(reversed(self.layer_names[:idx])).index(next(filter(lambda i: i == "conv", list(reversed(self.layer_names[:idx]))))) - 1)
        except StopIteration:
            return - 1
    
    def get_next_convLayer_idx(self, idx):
        """ 
            ``idx`` == layer idx to start looking forwards from
            Returns next conv layer idx (in self.layer_names) going forwards from current layer idx
            Returns -1 if no previous conv layer
        """
        try:
            return (idx + list(self.layer_names[idx:]).index(next(filter(lambda i: i == "conv", list(self.layer_names[idx:])))))
        except StopIteration:
            return - 1
            
    def reset_params(self):
        """ Useful function for comparing different algorithms, resets the original paramater initializations
            so that comparisons can be accurate
        """
        self.kernel_layers = self.copy_of_kernel_layers
        self.bias_layers = self.copy_of_bias_layers
        self.shared_funcs = SharedFunctions.SharedFunctions(self.activations, [k.shape for k in self.kernel_layers if k is not None], [b.shape for b in self.bias_layers if b is not None], self.scale_method, self.optimizer, self.lr, self.lr_decay)

    def get_weights(self):
        """ Returns neural network kernel_layers """
        return copy.deepcopy(self.kernel_layers)
    
    def get_biases(self):
        """ Returns neural network biases """
        return copy.deepcopy(self.bias_layers)
    
    def set_weights(self, weights):
        """ Sets local neural network kernel_layers to copy of paramater weights """
        self.kernel_layers = copy.deepcopy(weights)
    
    def set_biases(self, biases):
        """ Sets local neural network bias_layers to copy of paramater biases """
        self.bias_layers = copy.deepcopy(biases)

    def get_nprandomstate(self):
        return np.random.get_state()
    
    def set_nprandomstate(self, state):
        np.random.set_state(state)

