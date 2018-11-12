import pandas as pd
import numpy as np

# # TODO: Create a range of dates: 72 hours starting with midnight Jan 1st, 2011
# rng = pd.date_range('1/1/2011', periods=72, freq='H')
# print("rng: {}".format(rng[:5]))
#
# # TODO: Index pandas objects with dates:
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print("ts.head(): {}".format(ts.head()))
#
# # TODO: Change frequency and fill gaps: to 45 minute frequency and forward fill
# converted = ts.asfreq('45Min', method='pad')
# print("converted.head(): {}".format(converted.head()))
#
# # TODO: Resample the series to a daily frequency: Daily means
# daily_means = ts.resample('D').mean()
# print("daily_means: {}".format(daily_means))

# TODO: test data: only humidity downward trends (aka remove the 17;30 - 19:00 watering time steps)

df = pd.read_csv('hass_raw_dump.csv', parse_dates=['last_updated'])

dataFrameColumns = ['sensor.temperature', 'sensor.brightness', 'sensor.pressure', 'sensor.humidity',
    'sensor.flower1_temperature', 'sensor.flower1_conductivity', 'sensor.flower1_light_intensity',
    'sensor.flower1_moisture', 'sensor.yr_symbol', 'sun.sun', 'sensor.dark_sky_apparent_temperature',
    'sensor.dark_sky_cloud_coverage', 'sensor.dark_sky_humidity', 'sensor.dark_sky_temperature',
    'sensor.dark_sky_visibility', 'sensor.dark_sky_precip_intensity', 'sensor.dark_sky_precip_probability']
# dataFrameColumns = ['sensor.temperature', 'sensor.brightness']

dictTsSensors = {}
minDate = None
maxDate = None
for column_name in dataFrameColumns:
    _, sensor_ts = [x for _, x in df.groupby(df['entity_id'] == column_name)]
    sensor_ts['last_updated'] = sensor_ts['last_updated'].dt.round('1min')
    # TODO: store 'state' as float already, then remove 3xSTD outliers from sensor time-series?
    # df.a = df.a.astype(float)
    # TODO: sensor_ts.state = sensor_ts.state.astype(float)
    # TODO: how fill string_value == 'unavailable' or string_value == 'unknown' or string_value == 'below_horizon'? -> nan i guess
    # TODO: string_value == 'unavailable' or string_value == 'unknown' -> nan i guess
    # TODO: string_value == 'below_horizon' -> 0 string_value == 'above_horizon' -> 1 i guess
    dictTsSensors[column_name] = sensor_ts
    current_min_date = min(sensor_ts['last_updated'])
    current_max_date = max(sensor_ts['last_updated'])
    if minDate == None:
        minDate = current_min_date
    if maxDate == None:
        maxDate = current_max_date
    if current_max_date > maxDate:
        maxDate = current_max_date
    if current_min_date < minDate:
        minDate = current_min_date

# TODO: remove outliers before or after bfill & ffill?
# print("len(standardized_ts): {}", len(standardized_ts))
# standardized_ts[np.abs(standardized_ts - standardized_ts.mean()) <= (3*standardized_ts.std())]
# print("len(standardized_ts): {}", len(standardized_ts))
#
# print("len(standardized_ts): {}", len(standardized_ts))
# # TODO: Remote 3xStd outliers
# for column_name in dictTsSensors.keys():
#     standardized_ts[np.abs(standardized_ts[column_name] - standardized_ts[column_name].mean()) <= (3*standardized_ts[column_name].std())]
# print("len(standardized_ts): {}", len(standardized_ts))

ts = pd.DataFrame(np.nan, index=pd.date_range(start=minDate, end=maxDate, freq='Min'), columns=dataFrameColumns)

for column_name in dictTsSensors.keys():
    current_sensor_ts = dictTsSensors[column_name]
    if len(current_sensor_ts) > 0:
        for last_updated_time_second in current_sensor_ts['last_updated']:

            ts_sensor_value = current_sensor_ts.loc[current_sensor_ts['last_updated'] == last_updated_time_second, 'state']

            if len(ts_sensor_value) > 0:
                string_value = ts_sensor_value.iloc[-1]
                if string_value == 'unavailable' or string_value == 'unknown' or string_value == 'below_horizon':
                    ts_sensor_value = 0.0
                else:
                    if string_value == 'above_horizon':
                        ts_sensor_value = 1.0
                    else:
                        ts_sensor_value = float(string_value)

                ts.at[last_updated_time_second, column_name] = ts_sensor_value

# Normalize/Standardize data before filling them
standardized_ts = (ts-ts.mean())/ts.std()
normalized_ts = (ts-ts.min())/(ts.max()-ts.min())

ts = ts.ffill().bfill()
ts.fillna(value=0.0, inplace=True)  # 0.0 if no value at all
ts.to_csv("hass_ts.csv", sep=',')

standardized_ts = standardized_ts.ffill().bfill()
standardized_ts.fillna(value=0.0, inplace=True)  # 0.0 if no value at all
standardized_ts.to_csv("hass_ts_standardized.csv", sep=',')

# TODO: serialize-store the mean, min, max & std 1, for reverting
ts.mean().to_pickle("hass_ts_mean.pkl")
ts.std().to_pickle("hass_ts_std.pkl")
ts.max().to_pickle("hass_ts_min.pkl")
ts.min().to_pickle("hass_ts_max.pkl")

normalized_ts = normalized_ts.ffill().bfill()
normalized_ts.fillna(value=0.0, inplace=True)  # 0.0 if no value at all
normalized_ts.to_csv("hass_ts_normalized.csv", sep=',')
