import pandas as pd
# fromFileName = "ALL_BE_51vars_O3_O3-1_19900101To20121231"  # All stations
# fromFileName = "ALL_BETN_51vars_O3_O3-1_19900101To20121231"  # Background
fromFileName = "ALL_BE_51vars_PM10_PM10-1_19940101To20121231"
ts = pd.DataFrame.from_csv("data/{}.csv".format(fromFileName))
toFileName = "PM10_BETN"
# toFileName = fromFileName

# ts2a = ts[["PM10_BETN113", "PM10_BETN121", "PM10_BETN132"]]  # Prefix columns
# ts2b = ts[["PM10_BETN113-1", "PM10_BETN121-1", "PM10_BETN132-1"]]  # Suffix columns
# ts3 = ts[ts.columns.drop(list(ts.filter(regex='PM10_')))]
# ts = ts2a.join(ts3).join(ts2b)
# ts.to_csv("data/{}.csv".format(toFileName), sep=',')

# ts2a = ts[ts.columns.select(list(ts.filter(regex='PM10_BETN???')))]
ts2a = ts[["PM10_BETN043", "PM10_BETN045", "PM10_BETN052", "PM10_BETN054", "PM10_BETN060", "PM10_BETN063",
           "PM10_BETN066", "PM10_BETN067", "PM10_BETN070", "PM10_BETN073", "PM10_BETN085", "PM10_BETN093",
           "PM10_BETN100", "PM10_BETN113", "PM10_BETN121", "PM10_BETN132"]]
ts2b = ts[["PM10_BETN043-1", "PM10_BETN045-1", "PM10_BETN052-1", "PM10_BETN054-1", "PM10_BETN060-1", "PM10_BETN063-1",
           "PM10_BETN066-1", "PM10_BETN067-1", "PM10_BETN070-1", "PM10_BETN073-1", "PM10_BETN085-1", "PM10_BETN093-1",
           "PM10_BETN100-1", "PM10_BETN113-1", "PM10_BETN121-1", "PM10_BETN132-1"]]
ts3 = ts[ts.columns.drop(list(ts.filter(regex='PM10_')))]
ts = ts2a.join(ts3).join(ts2b)
ts.to_csv("data/{}.csv".format(toFileName), sep=',')
