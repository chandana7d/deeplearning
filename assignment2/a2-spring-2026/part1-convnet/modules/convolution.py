"""
2d Convolution Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias   = np.zeros(self.out_channels)

        self.dx     = None
        self.dw     = None
        self.db     = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        N, C, H, W = x.shape                                            #batch size, channels, height, width
        K = self.kernel_size                                            #kernel size
        S = self.stride                                                 #stride
        P = self.padding                                                #padding
        H_out = 1 + (H + 2 * P - K) // S                                        #output height
        W_out = 1 + (W + 2 * P - K) // S                                        #output width       
        out = np.zeros((N, self.out_channels, H_out, W_out), dtype=x.dtype)             #initialize output
        x_padded = np.pad(x, ((0,), (0,), (P,), (P,)), mode='constant')   #pad the input with zeros
        for n in range(N):                                              #iterate over batch size
            for f in range(self.out_channels):                                  #iterate over filters
                for i in range(H_out):                                  #iterate over output height
                    h_start = i * S                                     #calculate starting height index    
                    h_end = h_start + K                                 #calculate ending height index
                    for j in range(W_out):                              #iterate over output width
                        w_start = j * S                                 #calculate starting width index
                        w_end = w_start + K                             #calculate ending width index
                        window = x_padded[n, :, h_start:h_end, w_start:w_end]  #extract the window for convolution
                        out[n, f, i, j] = np.sum(window * self.weight[f]) + self.bias[f]  #compute the convolution and add bias to get the output value
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        N, C, H, W = x.shape                                            #batch size, channels, height, width
        K = self.kernel_size                                            #kernel size
        S = self.stride                                                 #stride
        P = self.padding                                                #padding
        H_out = 1 + (H + 2 * P - K) // S                            #output height 
        W_out = 1 + (W + 2 * P - K) // S                            #output width
        x_padded = np.pad(x, ((0,), (0,), (P,), (P,)), mode='constant')   #pad the input with zeros
        dx_padded = np.zeros_like(x_padded)                                           #initialize gradient with respect to padded input
        self.dw = np.zeros_like(self.weight)
        self.dx = np.zeros_like(x)
        self.db = np.sum(dout, axis=(0, 2, 3))                            #compute gradient with respect to bias
        for f in range(self.out_channels):                                  #       iterate over filters
            for c in range(C):                                          #iterate over channels
                for i in range(H_out):                                  #iterate over output height
                    h_start = i * S                                     #calculate starting height index    
                    h_end = h_start + K                                 #calculate ending height index
                    for j in range(W_out):                              #iterate over output width
                        w_start = j * S                                 #calculate starting width index
                        w_end = w_start + K                             #calculate ending width index
                        window = x_padded[:, c, h_start:h_end, w_start:w_end]  #extract the window for convolution
                        self.dw[f, c] += np.sum(window * dout[:, f, i, j][:, np.newaxis, np.newaxis], axis=0)  #compute the gradient with respect to weights
                        dx_padded[:, c, h_start:h_end, w_start:w_end] += self.weight[f, c] * dout[:, f, i, j][:, np.newaxis, np.newaxis]  #compute the gradient with respect to input
        self.dx = dx_padded[:, :, P:P+H, P:P+W]  #remove the padding from dx to get the final gradient with respect to input    
        return self.dx                 