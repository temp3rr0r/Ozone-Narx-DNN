import pandas as pd
# fromFileName = "BETN_12_66_73_121_51vars_O3_O3-1_19900101To2000101"
fromFileName = "ALL_BETN_51vars_O3_O3-1_19900101To2000101"
# fromFileName = "BETN073"
ts = pd.DataFrame.from_csv("data/{}.csv".format(fromFileName), header=None)
# toFileName = "BETN073"
# toFileName = "ALL_BETN_51vars_O3_O3-1_19900101To2000101"
toFileName = fromFileName

# Normalize/Standardize data before filling them
standardized_ts = (ts-ts.mean())/ts.std()
normalized_ts = (ts-ts.min())/(ts.max()-ts.min())

ts = ts.ffill().bfill()
ts.fillna(value=0.0, inplace=True)  # 0.0 if no value at all
ts.to_csv("data/{}_ts.csv".format(toFileName), sep=',')

# Standardize

standardized_ts = standardized_ts.ffill().bfill()
standardized_ts.fillna(value=0.0, inplace=True)  # 0.0 if no value at all
standardized_ts.to_csv("data/{}_ts_standardized.csv".format(toFileName), sep=',')

# Serialize-store the mean, min, max & std 1, for reverting
ts.mean().to_pickle("data/{}_ts_mean.pkl".format(toFileName))
ts.std().to_pickle("data/{}_ts_std.pkl".format(toFileName))
ts.min().to_pickle("data/{}_ts_min.pkl".format(toFileName))
ts.max().to_pickle("data/{}_ts_max.pkl".format(toFileName))

# Normalize

normalized_ts = normalized_ts.ffill().bfill()
normalized_ts.fillna(value=0.0, inplace=True)  # 0.0 if no value at all
normalized_ts.to_csv("data/{}_ts_normalized.csv".format(toFileName), sep=',')
sensor_mean = pd.read_pickle("data/{}_ts_mean.pkl".format(toFileName))
sensor_std = pd.read_pickle("data/{}_ts_std.pkl".format(toFileName))
sensor_min = pd.read_pickle("data/{}_ts_min.pkl".format(toFileName))
sensor_max = pd.read_pickle("data/{}_ts_max.pkl".format(toFileName))

print(sensor_mean)
print(sensor_std)
print(sensor_min)
print(sensor_max)
