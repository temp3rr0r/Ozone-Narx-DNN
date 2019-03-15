
# Model Search Space bounds
# TODO: Add weights initializer search: https://keras.io/initializers/
bounds = [(7, 1 * 31),  # batch_size (~ #days: week, month, year)  # TODO: reduced batch size to try avoiding OOM
          (350, 600), (0, 4),  # , 5)    # epoch_size, optimizer
          # (1023, 1024), (1023, 1024), (1023, 1024),  # units
          (64, 512), (64, 512), (64, 512),
          # (32, 512), (32, 196), (32, 384),
          (0.01, 0.25), (0.01, 0.25), (0.01, 0.25),  # dropout
          (0.01, 0.25), (0.01, 0.25), (0.01, 0.25),  # recurrent_dropout
          (0.01, 1), (0.01, 1), (0.01, 1),  # gaussian noise std
          (0, 1), (0, 1), (0, 1),  # gaussian_noise
          (0, 1), (0, 1), (0, 1),  # batch normalization
          (0, 5), (0, 5), (0, 5)]  # base layer types (plain/bidirectional: LSTM, GRU, Simple RNN)  # TODO: Base layers

# Lower Bounds
lb = [bounds[0][0],  # batch_size
      bounds[1][0], bounds[2][0],  # epoch_size, optimizer
      bounds[3][0], bounds[4][0], bounds[5][0],  # units
      bounds[6][0], bounds[7][0], bounds[8][0],  # dropout
      bounds[9][0], bounds[10][0], bounds[11][0],  # recurrent_dropout
      bounds[12][0], bounds[13][0], bounds[14][0],  # gaussian noise std
      bounds[15][0], bounds[16][0], bounds[17][0],  # gaussian_noise
      bounds[18][0], bounds[19][0], bounds[20][0],  # batch normalization
      bounds[21][0], bounds[22][0], bounds[23][0]]  # base layer types

# Upper Bounds
ub = [bounds[0][1],  # batch_size
      bounds[1][1], bounds[2][1],  # epoch_size, optimizer
      bounds[3][1], bounds[4][1], bounds[5][1],  # units
      bounds[6][1], bounds[7][1], bounds[8][1],  # dropout
      bounds[9][1], bounds[10][1], bounds[11][1],  # recurrent_dropout
      bounds[12][1], bounds[13][1], bounds[14][1],  # gaussian noise std
      bounds[15][1], bounds[16][1], bounds[17][1],  # gaussian_noise
      bounds[18][1], bounds[19][1], bounds[20][1],  # batch normalization
      bounds[21][1], bounds[22][1], bounds[23][1]]  # base layer types
