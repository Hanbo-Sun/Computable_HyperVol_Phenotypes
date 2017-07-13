
#-----------------------------------
# CNN - simple model 1.1 
#-----------------------------------
data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv1_1 <- mx.symbol.Convolution(data = data, kernel = c(3, 3), num_filter = 64 )
relu1_1 <- mx.symbol.Activation(data = conv1_1, act_type = "relu")
conv1_2 <- mx.symbol.Convolution(data = relu1_1 , kernel = c(3, 3), num_filter = 64 )
relu1_2 <- mx.symbol.Activation(data = conv1_2 , act_type = "relu")
pool1 <- mx.symbol.Pooling(data = relu1_2 , pool_type = "max",
                           kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv2_1 <- mx.symbol.Convolution(data = pool1, kernel = c(3, 3), num_filter = 128 )
relu2_1 <- mx.symbol.Activation(data = conv2_1, act_type = "relu")
conv2_2 <- mx.symbol.Convolution(data = relu2_1 , kernel = c(3, 3), num_filter = 128 )
relu2_2 <- mx.symbol.Activation(data = conv2_2 , act_type = "relu")
pool2 <- mx.symbol.Pooling(data = relu2_2 , pool_type = "max",
                           kernel = c(2, 2), stride = c(2, 2))
#
flatten <- mx.symbol.Flatten(data = pool2)
# model
fc1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 512)
relu3 <- mx.symbol.Activation(data = fc1, act_type = "relu")
# 2nd fully connected layer
fc2 <- mx.symbol.FullyConnected(data = relu3, num_hidden = 512)
# Output. Softmax output since we'd like to get some probabilities.
CNN_model <- mx.symbol.SoftmaxOutput(data = fc2)
devices <- mx.cpu()

