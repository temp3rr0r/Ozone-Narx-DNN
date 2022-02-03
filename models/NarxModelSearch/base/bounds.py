import json

bounds = [(7, 1 * 31),  # batch_size (~ #days: week, month, year)
          (350, 600),  # (5, 10),  # (350, 600)  # epoch_size
          (0, 4),  # optimizer
          (64, 512),  # (8, 64),  # (64, 512)  # units
          (64, 512),  # (8, 64),  # (64, 512)
          (64, 512),  # (8, 64),  # (64, 512)
          (0.01, 0.25),  # dropout
          (0.01, 0.25),
          (0.01, 0.25),
          (0.01, 0.25),  # recurrent_dropout  # TODO: disabled for MT, CY
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

# Benchmark 50 dimensions

# with open('settings/data_manipulation.json') as f:  # Read the settings json file
#     data_manipulation = json.load(f)
#
# bounds = [(0, 1)] * data_manipulation["benchmark_dimensions"]


# Model Search Space bounds
lb, ub = zip(*bounds)
lb = list(lb)  # Lower Bounds
ub = list(ub)  # Upper Bounds
