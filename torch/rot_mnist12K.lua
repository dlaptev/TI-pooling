require 'torch'
require 'math'
require 'nn'
require 'cunn'
require 'image'
require 'optim'

require 'rot_mnist12K_model'
require 'tools'

-- TI-pooling example code for rot_mnist12k classification dataset.

-- The implementation mainly consists of two parts:
-- 1. the dataset is augmented with transformed samples (see get_transformed);
-- 2. the model contains SpatialMaxPooling, that selects the maximum output of
--    siamese network replicas over the transformations (see define_model).

-- For further details and more experiments please refer to the original paper:
-- "TI-pooling: transformation-invariant pooling for feature learning in
-- Convolutional Neural Networks"
-- D. Laptev, N. Savinov, J.M. Buhmann, M. Pollefeys, CVPR 2016.

local opt = define_constants()

-- Load and unzip "Rotated MNIST digits" dataset from the following address:
-- http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations
local train_file = '../mnist_all_rotation_normalized_float_train_valid.amat'
local test_file = '../mnist_all_rotation_normalized_float_test.amat'
local train = load_rotated_mnist(train_file, -1)
local test = load_rotated_mnist(test_file, -1)
local n_train_data = train.data:size(1) -- Number of training samples.
local n_outputs = train.labels:max()    -- Number of classes.

-- Augment the dataset with transformed (rotated) images.
train.data = get_transformed(train.data, opt)
test.data = get_transformed(test.data, opt)

-- Define the model and the objective function.
local model = define_model(opt.input_size, opt.n_transformations, n_outputs)
local criterion = nn.ClassNLLCriterion()
local criterion = criterion:cuda()

local parameters, gradParameters = model:getParameters()
local counter = 0
local epoch = 1
local new_epoch = false

-- The value and the gradient of the functional on one batch.
local batch_feval = function(x)
  -- Get the batch number and update the counters.
  if new_epoch then
    epoch = epoch + 1
    new_epoch = false
  end
  if x ~= parameters then
    parameters:copy(x)
  end
  local start_index = counter * opt.batch_size + 1
  local end_index = math.min((counter + 1) * opt.batch_size, n_train_data)
  if end_index == n_train_data then
    counter = 0
    new_epoch = true
  else
    counter = counter + 1
  end
  local batch_inputs = train.data[{{start_index, end_index}}]:cuda()
  local batch_targets = train.labels[{{start_index, end_index},1}]:cuda()

  -- Forward pass the batch, compute regularized loss.
  gradParameters:zero()
  local batch_outputs = model:forward(batch_inputs)
  local batch_loss = criterion:forward(batch_outputs, batch_targets) +
      opt.weight_decay * (parameters:norm()^2) / (2 * parameters:size(1))

  -- Backward pass the loss, compute gradients.
  local dloss_doutput = criterion:backward(batch_outputs, batch_targets)
  model:backward(batch_inputs, dloss_doutput)
  gradParameters:add(parameters * opt.weight_decay / parameters:size(1))

  return batch_loss, gradParameters
end

-- Main optimization cycle: iterate through epochs.
local test_errors = {}
local train_errors = {}
local optim_state = {}
optim_state.rho = opt.adadelta_rho
optim_state.eps = opt.adadelta_eps
while true do -- Cycle through the batches.
  if (counter == 0) and (epoch % opt.printing_interval == 0) then
    torch.manualSeed(100)
    train_errors[#train_errors + 1] = calculate_error(model, train, opt)
    test_errors[#test_errors + 1] = calculate_error(model, test, opt)
    print(string.format("epoch: %6s, train_error = %6.6f, test_error = %6.6f",
                        epoch, train_errors[#train_errors],
                        test_errors[#test_errors]))
    -- Save the model for testing or for further training.
    torch.save(opt.model_dump_name, model)
    torch.save(opt.model_dump_name .. '_state', optim_state)
    torch.manualSeed(epoch)
  end
  -- Make a step using AdaDelta optimization algorithm (updates parameters).
  optim.adadelta(batch_feval, parameters, optim_state)
  collectgarbage()
end
