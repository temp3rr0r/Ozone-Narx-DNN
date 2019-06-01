
# Model Search Space bounds
bounds = [(7, 1 * 31),  # batch_size (~ #days: week, month, year)
          (350, 600), (0, 4),  # , 5)    # epoch_size, optimizer
          # (1023, 1024), (1023, 1024), (1023, 1024),  # units
          (64, 512), (64, 512), (64, 512),
          # (32, 512), (32, 196), (32, 384),
          (0.01, 0.25), (0.01, 0.25), (0.01, 0.25),  # dropout
          (0.01, 0.25), (0.01, 0.25), (0.01, 0.25),  # recurrent_dropout
          (0.01, 1), (0.01, 1), (0.01, 1),  # gaussian noise std
          (0, 1), (0, 1), (0, 1),  # gaussian_noise
          (0, 1), (0, 1), (0, 1),  # batch normalization
          (0, 5), (0, 5), (0, 5),  # base layer types (plain/bidirectional: LSTM, GRU, Simple RNN)
          (0, 9), (0, 9), (0, 9)]  # layer initializers, normal/uniform he/lecun,...

lb, ub = zip(*bounds)
lb = list(lb)  # Lower Bounds
ub = list(ub)  # Upper Bounds
