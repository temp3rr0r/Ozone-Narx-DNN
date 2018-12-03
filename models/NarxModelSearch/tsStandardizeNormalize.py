import pandas as pd
# ts = pd.DataFrame.from_csv('data/BETN073.csv', header=None)
ts = pd.DataFrame.from_csv('data/BETN073_51vars_19900101To2000101.csv', header=None)

# Normalize/Standardize data before filling them
standardized_ts = (ts-ts.mean())/ts.std()
normalized_ts = (ts-ts.min())/(ts.max()-ts.min())

ts = ts.ffill().bfill()
ts.fillna(value=0.0, inplace=True)  # 0.0 if no value at all
# ts.to_csv("data/BETN073_ts.csv", sep=',')
ts.to_csv("data/BETN073_51vars_19900101To2000101_ts.csv", sep=',')

standardized_ts = standardized_ts.ffill().bfill()
standardized_ts.fillna(value=0.0, inplace=True)  # 0.0 if no value at all
# standardized_ts.to_csv("data/BETN073_ts_standardized.csv", sep=',')
standardized_ts.to_csv("data/BETN073_51vars_19900101To2000101_ts_standardized.csv", sep=',')

# TODO: serialize-store the mean, min, max & std 1, for reverting
# ts.mean().to_pickle("data/BETN073_ts_mean.pkl")
ts.mean().to_pickle("data/BETN073_51vars_19900101To2000101_ts_mean.pkl")
# ts.std().to_pickle("data/BETN073_ts_std.pkl")
ts.std().to_pickle("data/BETN073_51vars_19900101To2000101_ts_std.pkl")
# ts.min().to_pickle("data/BETN073_ts_min.pkl")
ts.min().to_pickle("data/BETN073_51vars_19900101To2000101_ts_min.pkl")
# ts.max().to_pickle("data/BETN073_ts_max.pkl")
ts.max().to_pickle("data/BETN073_51vars_19900101To2000101_ts_max.pkl")

normalized_ts = normalized_ts.ffill().bfill()
normalized_ts.fillna(value=0.0, inplace=True)  # 0.0 if no value at all
# normalized_ts.to_csv("data/BETN073_ts_normalized.csv", sep=',')
normalized_ts.to_csv("data/BETN073_51vars_19900101To2000101_ts_normalized.csv", sep=',')

# sensor_mean = pd.read_pickle("data/BETN073_ts_mean.pkl")
sensor_mean = pd.read_pickle("data/BETN073_51vars_19900101To2000101_ts_mean.pkl")
# sensor_std = pd.read_pickle("data/BETN073_ts_std.pkl")
sensor_std = pd.read_pickle("data/BETN073_51vars_19900101To2000101_ts_std.pkl")
# sensor_min = pd.read_pickle("data/BETN073_ts_min.pkl")
sensor_min = pd.read_pickle("data/BETN073_51vars_19900101To2000101_ts_min.pkl")
# sensor_max = pd.read_pickle("data/BETN073_ts_max.pkl")
sensor_max = pd.read_pickle("data/BETN073_51vars_19900101To2000101_ts_max.pkl")

print(sensor_mean)
print(sensor_std)
print(sensor_min)
print(sensor_max)