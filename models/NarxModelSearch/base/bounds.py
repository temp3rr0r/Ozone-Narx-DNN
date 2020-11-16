bounds = [(7, 1 * 31),  # batch_size (~ #days: week, month, year)
          (350, 600),  # epoch_size
          (0, 4),  # optimizer
          (64, 512),  # units
          (64, 512),
          (64, 512),
          (0.01, 0.25),  # dropout
          (0.01, 0.25),
          (0.01, 0.25),
          (0.01, 0.25),  # recurrent_dropout
          (0.01, 0.25),
          (0.01, 0.25),
          (0.1, 0.5),  # gaussian noise std
          (0.1, 0.5),
          (0.1, 0.5),
          (0, 1),  # batch normalization layers
          (0, 1),
          (0, 1),
          (0, 1),  # gaussian noise layer layers
          (0, 1),
          (0, 1),
          (0, 2),  # base layer types (plain: LSTM, GRU, Simple RNN)
          (0, 2),
          (0, 2),
          (0, 9),  # layer initializers, normal/uniform he/lecun,...
          (0, 9),
          (0, 9)]

# Model Search Space bounds
lb, ub = zip(*bounds)
lb = list(lb)  # Lower Bounds
ub = list(ub)  # Upper Bounds
