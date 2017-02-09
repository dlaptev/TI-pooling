import numpy as np
import math
from scipy.ndimage.interpolation import rotate

class DataLoader:
  def __init__(self, name, number_of_classes, number_of_rotations):
    loaded = np.loadtxt(name)
    # loaded = loaded[np.random.choice(loaded.shape[0], 1000, replace=False), :]
    self._x = self._transform(loaded, number_of_rotations)
    self._y = self._int_labels_to_one_hot(loaded[:, -1], number_of_classes)
    self._completed_epochs = -1
    self._new_epoch = False
    self._start_new_epoch()

  def _transform(self, loaded, number_of_rotations):
    # pad it and populate along the last dimension; then rotate
    padded = np.pad(np.reshape(loaded[:, :-1], [-1, 28, 28, 1]), [[0, 0], [2, 2], [2, 2], [0, 0]], 'constant', constant_values=0)
    tiled = np.tile(np.expand_dims(padded, 4), [number_of_rotations])
    for rotation_index in xrange(number_of_rotations):
      angle = 360.0 * rotation_index / float(number_of_rotations)
      tiled[:, :, :, :, rotation_index] = rotate(tiled[:, :, :, :, rotation_index],
                                                 angle,
                                                 axes=[1, 2],
                                                 reshape=False)
    print('finished rotating')
    return tiled

  def _int_labels_to_one_hot(self, int_labels, number_of_classes):
    offsets = np.arange(self._size()) * number_of_classes
    one_hot_labels = np.zeros((self._size(), number_of_classes))
    flat_iterator = one_hot_labels.flat
    for index in xrange(self._size()):
      flat_iterator[offsets[index] + int(int_labels[index])] = 1
    return one_hot_labels

  def _size(self):
    return self._x.shape[0]

  def _start_new_epoch(self):
    permuted_indexes = np.random.permutation(self._size())
    self._x = self._x[permuted_indexes, :]
    self._y = self._y[permuted_indexes]
    self._completed_epochs += 1
    self._index = 0
    self._new_epoch = True

  def get_completed_epochs(self):
    return self._completed_epochs

  def is_new_epoch(self):
    return self._new_epoch

  def next_batch(self, batch_size):
    if (self._new_epoch):
      self._new_epoch = False
    start = self._index
    end = start + batch_size
    if (end > self._size()):
      assert batch_size <= self._size()
      self._start_new_epoch()
      start = 0
      end = start + batch_size
    self._index += batch_size
    return self._x[start:end, :], self._y[start:end]

  def all(self):
    return self._x, self._y
