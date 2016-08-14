-- This file contains the following functions:
--   load_rotated_mnist(file_name, count)
--   get_transformed(batch_inputs, opt)
--   calculate_error(model, data_to_check, opt)
--   convergent_adadelta(opfunc, x, config, state)

-- Loads the dataset from an .amat file.
function load_rotated_mnist(file_name, count)
  local loaded_data = {}
  for line in io.lines(file_name) do
    local chunks = {}
    for w in line:gmatch("%S+") do chunks[#chunks + 1] = tonumber(w) end
    loaded_data[#loaded_data + 1] = chunks
  end
  local loaded_data = torch.Tensor(loaded_data)
  local data = {}
  data.data = loaded_data[{{1, count}, {1, -2}}]
  data.labels = loaded_data[{{1, count}, {-1, -1}}]
  local shuffled_indices = torch.randperm(data.data:size(1)):long()
  data.data = data.data:index(1, shuffled_indices)
  data.labels = data.labels:index(1, shuffled_indices)
  data.labels:add(1)
  local real_size = math.sqrt(data.data:size(2))
  data.data = data.data:reshape(data.data:size(1), 1, real_size, real_size)
  print('--------------------------------')
  print('inputs', data.data:size())
  print('targets', data.labels:size())
  print('min target', data.labels:min())
  print('max target', data.labels:max())
  print('--------------------------------')
  return data
end

-- Augments the tensor along the second dimension with transformed instances
-- (rotation is used here, but various transformations can be used).
function get_transformed(batch_inputs, opt)
  local st = torch.LongStorage(5)
  st[1] = batch_inputs:size(1)  -- the number of images
  st[2] = opt.n_transformations
  st[3] = 1  -- the number of channels is 1
  st[4] = opt.input_size
  st[5] = opt.input_size
  local result = torch.Tensor(st)
  for index = 1, batch_inputs:size(1) do
    local padded_sample = torch.Tensor(opt.input_size, opt.input_size):zero()
    local offset = (opt.input_size - opt.real_size) / 2
    padded_sample[{{1 + offset, opt.input_size - offset},
                   {1 + offset, opt.input_size - offset}}] =
        batch_inputs[index]:squeeze()
    padded_sample = padded_sample:t():contiguous()
    for angle_index = 1, opt.n_transformations do
      result[index][angle_index][1] = image.rotate(padded_sample,
          2 * math.pi * angle_index / opt.n_transformations, 'bilinear')
    end
  end
  return result
end

-- Calculates the number of mispredictions of the trained model.
function calculate_error(model, data_to_check, opt)
  model:evaluate()
  local data_size = data_to_check.data:size(1)
  local batches_per_dataset = math.ceil(data_size / opt.batch_size)
  local error = 0
  for batch_index = 0, (batches_per_dataset - 1) do
    local start_index = batch_index * opt.batch_size + 1
    local end_index = math.min(data_size, (batch_index + 1) * opt.batch_size)
    local batch_targets =
        data_to_check.labels[{{start_index, end_index},1}]:cuda()
    local transformed_batch = data_to_check.data[{{start_index, end_index}}]
    local batch_inputs = transformed_batch:cuda()
    local logProbs = model:forward(batch_inputs)
    local classProbabilities = torch.exp(logProbs)
    local _, max_inds = torch.max(classProbabilities, 2)
    classPredictions = torch.Tensor():resize(max_inds:size(1))
        :copy(max_inds[{{1,max_inds:size(1)},1}]):cuda()
    error = error + classPredictions:ne(batch_targets):sum()
  end
  model:training()
  return error / data_size
end

-- Optimization subroutine: given a functional and its gradient, makes a step
-- minimizing the functional opfunc.
function convergent_adadelta(opfunc, x, config, state)
    -- (0) get/update state
    local state = state or {}
    state.rho = state.adadelta_rho or 0.9
    state.eps = state.adadelta_eps or 1e-6
    state.evalCounter = state.evalCounter or 0

    -- (1) evaluate f(x) and df/dx
    local fx,dfdx = opfunc(x)

    -- (2) parameter update
    if not state.paramVariance then
        state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        state.paramStd = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        state.delta = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        state.accDelta = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
        state.step = 1
    end
    state.paramVariance:mul(state.rho):addcmul(1 - state.rho,dfdx,dfdx)
    state.paramStd:resizeAs(state.paramVariance):copy(state.paramVariance)
        :add(state.eps):sqrt()
    state.delta:resizeAs(state.paramVariance):copy(state.accDelta)
        :add(state.eps):sqrt():cdiv(state.paramStd):cmul(dfdx)
    x:add(-state.step, state.delta)
    state.accDelta:mul(state.rho)
        :addcmul(1 - state.rho, state.delta, state.delta)

    -- (3) update evaluation counter
    state.evalCounter = state.evalCounter + 1

    -- return x*, f(x) before optimization
    return x,{fx}
end
