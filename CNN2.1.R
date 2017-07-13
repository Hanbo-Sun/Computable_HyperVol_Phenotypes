#-----------------------------------
# CNN - simple model 1 - work but not good
#-----------------------------------
data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 50 )
relu_1 <- mx.symbol.Activation(data = conv_1, act_type = "relu")
pool_1 <- mx.symbol.Pooling(data = relu_1, pool_type = "max",
                            kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50 )
relu_2 <- mx.symbol.Activation(data = conv_2, act_type = "relu")
pool_2 <- mx.symbol.Pooling(data=relu_2, pool_type = "max",
                            kernel = c(2, 2), stride = c(2, 2))
#
flatten <- mx.symbol.Flatten(data = pool_2)
# model
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
relu_3 <- mx.symbol.Activation(data = fc_1, act_type = "relu")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = relu_3, num_hidden = 40)
# Output. Softmax output since we'd like to get some probabilities.
CNN_model <- mx.symbol.SoftmaxOutput(data = fc_2)
devices <- mx.cpu()
