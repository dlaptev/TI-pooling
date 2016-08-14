-- Initialize dataset and method properties.
function define_constants()
  torch.setdefaulttensortype('torch.DoubleTensor')
  torch.manualSeed(1)

  local opt = {}
  opt.input_size = 32
  opt.real_size = 28
  opt.n_transformations = 24 -- TI-pooling parameter.

  opt.printing_interval = 2 -- Debug parameter
  opt.model_dump_name = 'saved_model'

  -- Optimization parameters.
  opt.batch_size = 64
  opt.weight_decay = 0 -- Regularization (0 means "not used").
  opt.adadelta_rho = 0.9
  opt.adadelta_eps = 1e-6
  opt.decrease_step_size = 200 -- Decrease step size every 200 epochs.

  return opt
end

-- Define the topology of the network. TI-pooling takes the maximum of a
-- feature over the transformed instances.
function define_model(input_size, n_transformations, n_outputs)
  local fully_connected_multiplier = 128
  local model = nn.Sequential()
  local number_of_filters = 40

  -- Standard model definition: stacked convolutions, ReLU and max-pooling.
  model:add(nn.SpatialConvolution(1, number_of_filters,
                                  3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialConvolution(number_of_filters, 2*number_of_filters,
                                  3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialConvolution(2*number_of_filters, 4*number_of_filters,
                                  3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Reshape(4*number_of_filters*input_size*input_size / (4*4*4)))
  model:add(nn.Linear(4*number_of_filters*input_size*input_size / (4*4*4),
                      fully_connected_multiplier*number_of_filters))
  model:add(nn.ReLU())
  model:add(nn.Reshape(1, fully_connected_multiplier*number_of_filters, 1))

  -- Put siamese replicas in parallel (replicate n_transformations times).
  local parallel_model = nn.Parallel(2, 4)
  for rotation_index = 1, n_transformations do
    parallel_model:add(model:clone())
  end

  -- TI-pooling (transformation-invariance pooling) layer.
  local full_model = nn.Sequential()
  full_model:add(parallel_model)
  -- Take the maximum output of siamese replicas over the transformations.
  full_model:add(nn.SpatialMaxPooling(n_transformations, 1, 1, 1))

  -- Add fully-connected and output layers on top of TI-pooling features.
  full_model:add(nn.Reshape(fully_connected_multiplier*number_of_filters))
  full_model:add(nn.Dropout())
  full_model:add(nn.Linear(fully_connected_multiplier*number_of_filters,
                           n_outputs))
  full_model:add(nn.LogSoftMax())
  full_model = full_model:cuda()

  -- Share all the parameters between siamese replicas.
  parallel_model = full_model:get(1)
  for rotation_index = 2, n_transformations do
    local current_module = parallel_model:get(rotation_index)
    current_module:share(parallel_model:get(1), 'weight', 'bias',
                         'gradWeight', 'gradBias')
  end

  full_model:training()
  return full_model
end
