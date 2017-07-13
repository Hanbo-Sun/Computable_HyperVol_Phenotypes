#-----------------------------------
# CNN - VGG16
#-----------------------------------
data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv1_1 <- mx.symbol.Convolution(data = data, kernel = c(3, 3), num_filter = 64, pad=c(1,1))
relu1_1 <- mx.symbol.Activation(data = conv1_1, act_type = "relu")
conv1_2 <- mx.symbol.Convolution(data = relu1_1, kernel = c(3, 3), num_filter = 64, pad=c(1,1))
relu1_2 <- mx.symbol.Activation(data = conv1_2, act_type = "relu")
pool1 <- mx.symbol.Pooling(data = relu1_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv2_1 <- mx.symbol.Convolution(data = pool1, kernel = c(3, 3), num_filter = 128, pad=c(1,1))
relu2_1 <- mx.symbol.Activation(data = conv2_1, act_type = "relu")
conv2_2 <- mx.symbol.Convolution(data = relu2_1, kernel = c(3, 3), num_filter = 128, pad=c(1,1))
relu2_2 <- mx.symbol.Activation(data = conv2_2, act_type = "relu")
pool2 <- mx.symbol.Pooling(data = relu2_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 3rd convolutional layer
conv3_1 <- mx.symbol.Convolution(data = pool2, kernel = c(3, 3), num_filter = 256,pad=c(1,1))
relu3_1 <- mx.symbol.Activation(data = conv3_1, act_type = "relu")
conv3_2 <- mx.symbol.Convolution(data = relu3_1, kernel = c(3, 3), num_filter = 256,pad=c(1,1) )
relu3_2 <- mx.symbol.Activation(data = conv3_2, act_type = "relu")
conv3_3 <- mx.symbol.Convolution(data = relu3_2, kernel = c(3, 3), num_filter = 256,pad=c(1,1) )
relu3_3 <- mx.symbol.Activation(data = conv3_3, act_type = "relu")
pool3 <- mx.symbol.Pooling(data = relu3_3, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 4th convolutional layer
conv4_1 <- mx.symbol.Convolution(data = pool3, kernel = c(3, 3), num_filter = 512,pad=c(1,1))
relu4_1 <- mx.symbol.Activation(data = conv4_1, act_type = "relu")
conv4_2 <- mx.symbol.Convolution(data = relu4_1, kernel = c(3, 3), num_filter = 512,pad=c(1,1) )
relu4_2 <- mx.symbol.Activation(data = conv4_2, act_type = "relu")
conv4_3 <- mx.symbol.Convolution(data = relu4_2, kernel = c(3, 3), num_filter = 512,pad=c(1,1) )
relu4_3 <- mx.symbol.Activation(data = conv4_3, act_type = "relu")
pool4 <- mx.symbol.Pooling(data = relu4_3, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 5th convolutional layer
conv5_1 <- mx.symbol.Convolution(data = pool4, kernel = c(3, 3), num_filter = 512,pad=c(1,1))
relu5_1 <- mx.symbol.Activation(data = conv5_1, act_type = "relu")
conv5_2 <- mx.symbol.Convolution(data = relu5_1, kernel = c(3, 3), num_filter = 512,pad=c(1,1) )
relu5_2 <- mx.symbol.Activation(data = conv5_2, act_type = "relu")
conv5_3 <- mx.symbol.Convolution(data = relu5_2, kernel = c(3, 3), num_filter = 512,pad=c(1,1) )
relu5_3 <- mx.symbol.Activation(data = conv5_3, act_type = "relu")
pool5 <- mx.symbol.Pooling(data = relu5_3, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# fully connected layer
flatten <- mx.symbol.Flatten(data = pool5)
fc1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 4096)
relu6 <- mx.symbol.Activation(data = fc1, act_type = "relu")
#dropout = mx.symbol.Dropout(relu_4, p = 0.2)
fc2 <- mx.symbol.FullyConnected(data = relu6, num_hidden = 4096)
relu7 <- mx.symbol.Activation(data = fc2, act_type = "relu")
# Output. Softmax
CNN_model <- mx.symbol.SoftmaxOutput(data = relu7)
devices <- mx.cpu()

