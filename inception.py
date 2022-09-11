"""
Authors : Jordan Ngucho
(c) copyright 2022

Implementation of InceptionTime in Pytorch
https://link.springer.com/article/10.1007/s10618-020-00710-y 

All the credit goes to : 
 - https://github.com/TheMrGhostman/InceptionTime-Pytorch
 - https://github.com/hfawaz/InceptionTime
"""
import torch
import torch.nn as nn

def pass_through(x):
    """
    """
    return x

class Inception(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_size=40,
                 stride=1 
                 ):
        super(Inception, self).__init__()
        #kernel_sizes = [3, 5, 8,11,17]
        self.kernel_size= kernel_size - 1
        self.n_filters = n_filters
        self.activation=nn.ReLU()
        bottleneck_channels = 32

        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels=in_channels,
                                        out_channels=bottleneck_channels,
                                        kernel_size=1, stride=1, bias=False)
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1
        
        kernel_sizes = [self.kernel_size//(2**i) for i in range(3)]

        self.conv_from_bottleneck_1 = nn.Conv1d(in_channels=bottleneck_channels,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[0],
                                                stride=stride, 
                                                padding=kernel_sizes[0]//2,
                                                bias=False
                                                )
        self.conv_from_bottleneck_2 = nn.Conv1d(in_channels=bottleneck_channels,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[1],
                                                stride=stride,
                                                padding=kernel_sizes[1]//2,
                                                bias=False
                                               )
        self.conv_from_bottleneck_3 = nn.Conv1d(in_channels=bottleneck_channels,
                                                out_channels= n_filters,
                                                kernel_size=kernel_sizes[2],
                                                stride=stride,
                                                padding=kernel_sizes[2]//2,
                                                bias=False
                                               )
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_from_maxpool = nn.Conv1d(in_channels=in_channels,
                                           out_channels=n_filters,
                                           kernel_size=1, stride=1,
                                           padding=0, bias=False
                                          )
        self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)


    def forward(self, X):
        # step 1
        Z_bottleneck = self.bottleneck(X)
        Z_maxpool = self.max_pool(X)
        # step 2
        Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        Z4 = self.conv_from_maxpool(Z_maxpool)
        # step 3
        Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
        Z = self.activation(self.batch_norm(Z))
        return Z

        
class InceptionBlock(nn.Module):
    """
    """
    def __init__(self, in_channels, n_filters=32, kernel_size=40,
                 stride=1, use_residual=True):
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        self.activation = nn.ReLU()
        self.inception_1 = Inception(in_channels=in_channels, 
                                     n_filters=n_filters, 
                                     kernel_size=kernel_size, 
                                     stride=stride
                                    )
        self.inception_2 = Inception(in_channels=4*n_filters, 
                                     n_filters=n_filters, 
                                     kernel_size=kernel_size, 
                                     stride=stride
                                    )
        self.inception_3 = Inception(in_channels=4*n_filters,
                                     n_filters=n_filters, 
                                     kernel_size=kernel_size, 
                                     stride=stride
                                    )
        if self.use_residual:
            self.residual = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=4*n_filters,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0
                                                    ),
                                        nn.BatchNorm1d(num_features=4*n_filters)
                                        )
    def forward(self, X):
        Z = self.inception_1(X)
        Z = self.inception_2(Z)
        Z = self.inception_3(Z)
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        return Z

class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)
    
class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)

class InceptionTime(nn.Module):
    def __init__(self, nb_classes, n_filters=32, kernel_size=40,
                 stride=1) -> None:
        super(InceptionTime, self).__init__()
        self.nb_classes = nb_classes
        self.inceptionblock_1 = InceptionBlock(
                                        in_channels=1, 
                                        n_filters=n_filters, 
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        use_residual=True
                                    )
        self.inceptionblock_2 = InceptionBlock(
                                        in_channels=4*n_filters, 
                                        n_filters=n_filters, 
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        use_residual=True
                                    )        
        self.dense = nn.Linear(in_features=4*32*1, out_features=self.nb_classes)
    def forward(self, X):
        Z = Reshape(out_shape=(1, X.shape[-1]))(X)
        Z = self.inceptionblock_1(Z)
        Z = self.inceptionblock_2(Z)
        Z = nn.AdaptiveAvgPool1d(output_size=1)(Z)
        Z = Flatten(out_features=32*4*1)(Z)
        Z = self.dense(Z)
        Z = nn.Softmax(dim=1)(Z)
        return Z

    
if __name__ == '__main__':
    import torch.optim as optim
    import numpy as np
    # Loading the model into GPU
    device = torch.device("cpu")
    model = InceptionTime(nb_classes=10, n_filters=32, kernel_size=40,stride=1).to(device)

    # Initialize our optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lossFn = nn.CrossEntropyLoss()

    x = torch.randn(4, 500).to(device, dtype=torch.float)
    pred = model(x)
