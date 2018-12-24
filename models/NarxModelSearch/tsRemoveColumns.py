import pandas as pd
# fromFileName = "ALL_BE_51vars_O3_O3-1_19900101To20121231"  # All stations
fromFileName = "ALL_BETN_51vars_O3_O3-1_19900101To20121231"  # Background
ts = pd.DataFrame.from_csv("data/{}.csv".format(fromFileName))
toFileName = "BETN113_121_132_BG"
# toFileName = fromFileName

ts2a = ts[["O3_BETN113", "O3_BETN121", "O3_BETN132"]]  # Prefix columns
ts2b = ts[["O3_BETN113-1", "O3_BETN121-1", "O3_BETN132-1"]]  # Suffix columns
ts3 = ts[ts.columns.drop(list(ts.filter(regex='O3_')))]
ts = ts2a.join(ts3).join(ts2b)
ts.to_csv("data/{}.csv".format(toFileName), sep=',')
