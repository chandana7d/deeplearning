"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride      = stride
        self.cache       = None
        self.dx          = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        N, C, H, W = x.shape                                            #batch size, channels, height, width
        K = self.kernel_size                                            #kernel size
        S = self.stride                                                 #stride
        H_out = 1 + (H - K) // S                                        #output height
        W_out = 1 + (W - K) // S                                        #output width       
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)             #initialize output
        for n in range(N):                                              #iterate over batch size
            for c in range(C):                                          #iterate over channels
                for i in range(H_out):                                  #iterate over output height
                    h_start = i * S                                     #calculate starting height index    
                    h_end = h_start + K                                 #calculate ending height index
                    for j in range(W_out):                              #iterate over output width
                        w_start = j * S                                 #calculate starting width index
                        w_end = w_start + K                             #calculate ending width index
                        window = x[n, c, h_start:h_end, w_start:w_end]  #extract the window for max pooling
                        out[n, c, i, j] = np.max(window)                #store the max value in the output
        self.cache = (x, H_out, W_out)                                  #cache the input and output dimensions for backward pass
        return out                                                       #return the output of max pooling

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache                                    #retrieve cached input and output dimensions
        N, C, H, W = x.shape                                            #batch size, channels, height, width
        K = self.kernel_size
        S = self.stride                                                 #stride
        
        dx = np.zeros_like(x)                                           #initialize gradient with respect to input
        for n in range(N):                                              #iterate over batch size
            for c in range(C):                                          #iterate over channels
                for i in range(H_out):                                  #iterate over output height
                    h_start = i * S                                     #calculate starting height index
                    h_end = h_start + K                                 #calculate ending height index
                    for j in range(W_out):                              #iterate over output width
                        w_start = j * S                                 #calculate starting width index
                        w_end = w_start + K                             #calculate ending width index
                        window = x[n, c, h_start:h_end, w_start:w_end]  #extract the window for max pooling
                        max_idx = np.argmax(window)                     #find the index of the max value in the window
                        max_pos = np.unravel_index(max_idx, window.shape)  #convert the flat index to 2D index
                        dx[n, c, h_start + max_pos[0], w_start + max_pos[1]] += dout[n, c, i, j]  #propagate the gradient to the max value position
        self.dx = dx
        return dx
