import pandas as pd
# fromFileName = "ALL_BE_51vars_O3_O3-1_19900101To20121231"  # All stations
# fromFileName = "ALL_BETN_51vars_O3_O3-1_19900101To20121231"  # Background
# fromFileName = "ALL_BE_51vars_PM10_PM10-1_19940101To20121231"
#fromFileName = "ALL_BE_51vars_PM10_PM10-1_19950101To20181231"
# fromFileName = "ALL_BE_51vars_O3_O3-1_19900101To20181231"
# fromFileName = "ALL_BE_51vars_calendar_PM10_PM10-1_19950101To20181231"
fromFileName = "ALL_BE_51vars_calendar_O3_O3-7_19900101To20181231"
fromFileName = "ALL_BE_51vars_calendar_O3_O3-7_19900101To20181231"
ts = pd.DataFrame.from_csv("../data/{}.csv".format(fromFileName))
# toFileName = fromFileName

metric = "O3"

if metric == "PM10":
    toFileName = "PM10_BETN"
    ts2a = ts[["PM10_BETN043", "PM10_BETN045", "PM10_BETN052", "PM10_BETN054", "PM10_BETN060", "PM10_BETN063",
               "PM10_BETN066", "PM10_BETN067", "PM10_BETN070", "PM10_BETN073", "PM10_BETN085", "PM10_BETN093",
               "PM10_BETN100", "PM10_BETN113", "PM10_BETN121", "PM10_BETN132"]]
    ts2b = ts[["PM10_BETN043-1", "PM10_BETN045-1", "PM10_BETN052-1", "PM10_BETN054-1", "PM10_BETN060-1", "PM10_BETN063-1",
               "PM10_BETN066-1", "PM10_BETN067-1", "PM10_BETN070-1", "PM10_BETN073-1", "PM10_BETN085-1", "PM10_BETN093-1",
               "PM10_BETN100-1", "PM10_BETN113-1", "PM10_BETN121-1", "PM10_BETN132-1"]]
    ts3 = ts[ts.columns.drop(list(ts.filter(regex='PM10_')))]

    ts = ts2a.join(ts3).join(ts2b)
    ts.to_csv("../data/{}.csv".format(toFileName), sep=',')

elif metric == "O3":
    toFileName = "O3_BETN"
    # ts2a = ts[['O3_BE0312A', 'O3_BETAND3', 'O3_BETB004', 'O3_BETB006', 'O3_BETB011', 'O3_BETM705', 'O3_BETN012',
    #            'O3_BETN016', 'O3_BETN027', 'O3_BETN029', 'O3_BETN035', 'O3_BETN040', 'O3_BETN041', 'O3_BETN043',
    #            'O3_BETN045', 'O3_BETN046', 'O3_BETN050', 'O3_BETN051', 'O3_BETN052', 'O3_BETN054', 'O3_BETN060',
    #            'O3_BETN063', 'O3_BETN066', 'O3_BETN070', 'O3_BETN073', 'O3_BETN085', 'O3_BETN093', 'O3_BETN100',
    #            'O3_BETN113', 'O3_BETN121', 'O3_BETN132', 'O3_BETR001', 'O3_BETR012', 'O3_BETR201', 'O3_BETR222',
    #            'O3_BETR240', 'O3_BETR501', 'O3_BETR502', 'O3_BETR701', 'O3_BETR710', 'O3_BETR740', 'O3_BETR801',
    #            'O3_BETR811', 'O3_BETR831', 'O3_BETR841', 'O3_BETWOL1']]
    ts2a = ts[['O3_BETN073']]
    # ts2b = ts[['O3_BE0312A-1', 'O3_BETAND3-1', 'O3_BETB004-1', 'O3_BETB006-1', 'O3_BETB011-1', 'O3_BETM705-1',
    #            'O3_BETN012-1', 'O3_BETN016-1', 'O3_BETN027-1', 'O3_BETN029-1', 'O3_BETN035-1', 'O3_BETN040-1',
    #            'O3_BETN041-1', 'O3_BETN043-1', 'O3_BETN045-1', 'O3_BETN046-1', 'O3_BETN050-1', 'O3_BETN051-1',
    #            'O3_BETN052-1', 'O3_BETN054-1', 'O3_BETN060-1', 'O3_BETN063-1', 'O3_BETN066-1', 'O3_BETN070-1',
    #            'O3_BETN073-1', 'O3_BETN085-1', 'O3_BETN093-1', 'O3_BETN100-1', 'O3_BETN113-1', 'O3_BETN121-1',
    #            'O3_BETN132-1', 'O3_BETR001-1', 'O3_BETR012-1', 'O3_BETR201-1', 'O3_BETR222-1', 'O3_BETR240-1',
    #            'O3_BETR501-1', 'O3_BETR502-1', 'O3_BETR701-1', 'O3_BETR710-1', 'O3_BETR740-1', 'O3_BETR801-1',
    #            'O3_BETR811-1', 'O3_BETR831-1', 'O3_BETR841-1', 'O3_BETWOL1-1']]
    ts2b = ts[['O3_BETN073-1']]
    ts3 = ts[ts.columns.drop(list(ts.filter(regex='O3_')))]

    ts = ts2a.join(ts3).join(ts2b)

    ts = ts[ts.columns.drop(list(ts.filter(regex='Unnamed')))]
    ts.to_csv("../data/{}.csv".format(toFileName), sep=',')
