#-----------------------------------
# CNN - Train model
#-----------------------------------
model <- mx.model.FeedForward.create(
  CNN_model, X = tr.x, y = tr.y, eval.data = list(data=ts.x,label=ts.y),
  ctx = devices, num.round = 200, array.batch.size = 30,
  learning.rate = 0.001, eval.metric = mx.metric.accuracy, initializer=mx.init.normal(0.1),
  epoch.end.callback = mx.callback.log.train.metric(100)
)
# another train
model <- mx.model.FeedForward.create(
  NN_model, X = tr.x, y = tr.y, eval.data = list(data=ts.x,label=ts.y),
  ctx = devices, num.round = 20, array.batch.size = 40,
  learning.rate = 0.01, momentum = 0.9,
  eval.metric = mx.metric.accuracy, initializer=
    mx.init.uniform(0.7),
  epoch.end.callback = mx.callback.log.train.metric(100)
)