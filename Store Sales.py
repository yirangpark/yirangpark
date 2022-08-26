# %% [markdown]
# ### Store Sales - Time Series Forecasting
# ###### [Mission] predicting sales for each product family and store combinations<br>[Data]<br>1) Train: time series of the stores and the product families combination<br>2) Test<br>3) Store: information about stores(city, state, type, cluster)<br>4) Transactions: correlated with train's sales columns<br>5) Holidays and Events: meta data(past sales, trend, seasonality)<br>6) Daily Oil Price

# %% [markdown]
# ### Prepare data

# %%
# base
import pandas as pd
import numpy as np
import os
import gc
import warnings

# PACF-ACF
import statsmodels.api as sm

# visualization
%matplotlib inline
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# configurations
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')

# acquire data
train_data = pd.read_csv('C:/Users/1004/git/kaggle/Store Sales/train.csv')
test_data = pd.read_csv('C:/Users/1004/git/kaggle/Store Sales/test.csv')
holi = pd.read_csv('C:/Users/1004/git/kaggle/Store Sales/holidays_events.csv')
oil = pd.read_csv('C:/Users/1004/git/kaggle/Store Sales/oil.csv')
stores = pd.read_csv('C:/Users/1004/git/kaggle/Store Sales/stores.csv')
trans = pd.read_csv('C:/Users/1004/git/kaggle/Store Sales/transactions.csv')
combine = [train_data, test_data]

# %%
train_data.head()

# %%
test_data.head()

# %%
holi.head()

# %%
oil.head()

# %%
stores.head()

# %%
trans.head()

# %%
# datetime
train_data['date'] = pd.to_datetime(train_data.date)
test_data['date'] = pd.to_datetime(test_data.date)
trans['date'] = pd.to_datetime(trans.date)
oil['date'] = pd.to_datetime(oil.date)
holi['date'] = pd.to_datetime(holi.date)

# data types
train_data.onpromotion = train_data.onpromotion.astype("float16")
train_data.sales = train_data.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

# %% [markdown]
# ### Analyze by visualizing data(Transactions Data)

# %%
temp = pd.merge(train_data.groupby(['date', 'store_nbr']).sales.sum().reset_index(), trans, how = "left")
print("Spearman Correlation between Total Sales and Trancastions:{:,.4f}".format(temp.corr("spearman").sales.loc["transactions"]))
px.line(trans.sort_values(["store_nbr", "date"]), x = 'date', y = 'transactions', color = 'store_nbr', title = 'Transactions')

# %% [markdown]
# <h6>
# * Note(graph analysis)<br>
# - stable pattern in Transaction<br>
# - similar pattern except december from 2013 to 2017<br>
# - same pattern for each store in previous plot<br>
# - increased sales at the end of the year
# </h6>

# %%
a = trans.copy()
a["year"] = a.date.dt.year
a["month"] = a.date.dt.month
px.box(a, x = "year", y = "transactions", color = "month", title = "Transactions")

# %%
a = trans.set_index("date").resample("M").transactions.mean().reset_index()
a["year"] = a.date.dt.year
px.line(a, x = 'date', y = 'transactions', color = 'year', title = "Monthly Average Transactions")

# %%
px.scatter(temp, x = "transactions", y ="sales", trendline= "ols", trendline_color_override = "red")

# %%
a = trans.copy()
a["year"] = a.date.dt.year
a["dayofweek"] = a.date.dt.dayofweek+1
a = a.groupby(["year", "dayofweek"]).transactions.mean().reset_index()
px.line(a, x = "dayofweek", y = "transactions", color = "year", title = "Transactions")

# %% [markdown]
# ### Analyze by visualizing data(Oil Price Data)

# %%
# resample
oil = oil.set_index("date").dcoilwtico.resample("D").sum().reset_index()
# interpolate
oil["dcoilwtico"] = np.where(oil["dcoilwtico"]==0, np.nan, oil["dcoilwtico"])
oil["dcoilwtico_interpolated"] = oil.dcoilwtico.interpolate()
# plot
p = oil.melt(id_vars = ['date'] + list(oil.keys()[5:]), var_name = 'Legend')
px.line(p.sort_values(["Legend", "date"], ascending = [False, True]), x = 'date', y = 'value', color = 'Legend', title = 'Daily Oil Price')


# %%
temp = pd.merge(temp, oil, how = "left")
print("Correlation with Daily Oil Prices")
print(temp.drop(["store_nbr", "dcoilwtico"], axis = 1).corr("spearman").dcoilwtico_interpolated.loc[["sales", "transactions"]], "\n")

fig, axes = plt.subplots(1, 2, figsize = (15, 5))
temp.plot.scatter(x = "dcoilwtico_interpolated", y = "transactions", ax = axes[0])
temp.plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax = axes[1], color = "r")
axes[0].set_title('Daily oil Price & Transactions', fontsize = 15)
axes[1].set_title('Daily oil Price & Sales', fontsize = 15);

# %%
a = pd.merge(train_data.groupby(["date", "family"]).sales.sum().reset_index(), oil.drop("dcoilwtico", axis = 1), how = "left")
c = a.groupby("family").corr("spearman").reset_index()
c = c[c.level_1 == "dcoilwtico_interpolated"][["family", "sales"]].sort_values("sales")

fig, axes = plt.subplots(7, 5, figsize = (20,20))
for i, fam in enumerate(c.family):
    if i < 6:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[0, i-1])
        axes[0, i-1].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[0, i-1].axvline(x=70, color='r', linestyle='--')
    if i >= 6 and i<11:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[1, i-6])
        axes[1, i-6].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[1, i-6].axvline(x=70, color='r', linestyle='--')
    if i >= 11 and i<16:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[2, i-11])
        axes[2, i-11].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[2, i-11].axvline(x=70, color='r', linestyle='--')
    if i >= 16 and i<21:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[3, i-16])
        axes[3, i-16].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[3, i-16].axvline(x=70, color='r', linestyle='--')
    if i >= 21 and i<26:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[4, i-21])
        axes[4, i-21].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[4, i-21].axvline(x=70, color='r', linestyle='--')
    if i >= 26 and i < 31:
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[5, i-26])
        axes[5, i-26].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[5, i-26].axvline(x=70, color='r', linestyle='--')
    if i >= 31 :
        a[a.family == fam].plot.scatter(x = "dcoilwtico_interpolated", y = "sales", ax=axes[6, i-31])
        axes[6, i-31].set_title(fam+"\n Correlation:"+str(c[c.family == fam].sales.iloc[0])[:6], fontsize = 12)
        axes[6, i-31].axvline(x=70, color='r', linestyle='--')
        
        
plt.tight_layout(pad=5)
plt.suptitle("Daily Oil Product & Total Family Sales \n", fontsize = 20);
plt.show()

# %% [markdown]
# ### Analyze by visualizing data(Sales Data)

# %%
a = train_data[["store_nbr", "sales"]]
a["ind"] = 1
a["ind"] = a.groupby("store_nbr").ind.cumsum().values
a = pd.pivot(a, index = "ind", columns = "store_nbr", values = "sales").corr()
mask = np.triu(a.corr())
plt.figure(figsize = (20,20))
sns.heatmap(a,
        annot = True,
        fmt = '.1f',
        cmap = 'coolwarm',
        square = True,
        mask = mask,
        linewidths = 1,
        cbar = False)
plt.title("Correlations among stores", fontsize = 20)
plt.show()

# %% [markdown]
# <h6> * Note(graph analysis)<br>
# - Most of the stores are silmilar to each other<br>
# - Some stores(20, 21, 22, 52) may be a little different
# </h6>

# %%
a = train_data.set_index("date").groupby("store_nbr").resample("D").sales.sum().reset_index()
px.line(a, x = "date", y = "sales", color = "store_nbr", title = "Daily total sales of the stores")

# %%
print(train_data.shape)
train_data = train_data[~((train_data.store_nbr == 52) & (train_data.date < "2017-04-20"))]
train_data = train_data[~((train_data.store_nbr == 22) & (train_data.date < "2015-10-09"))]
train_data = train_data[~((train_data.store_nbr == 42) & (train_data.date < "2015-08-21"))]
train_data = train_data[~((train_data.store_nbr == 21) & (train_data.date < "2015-07-24"))]
train_data = train_data[~((train_data.store_nbr == 29) & (train_data.date < "2015-03-20"))]
train_data = train_data[~((train_data.store_nbr == 20) & (train_data.date < "2015-02-13"))]
train_data = train_data[~((train_data.store_nbr == 53) & (train_data.date < "2014-05-29"))]
train_data = train_data[~((train_data.store_nbr == 36) & (train_data.date < "2013-05-09"))]
train_data.shape

# %%
# Zero Forecasting
c = train_data.groupby(["store_nbr", "family"]).sales.sum().reset_index().sort_values(["family", "store_nbr"])
c = c[c.sales==0]
c

# %%
print(train_data.shape)
# Anti Join
outer_join = train_data.merge(c[c.sales == 0].drop("sales", axis = 1), how = 'outer', indicator = True)
train_data = outer_join[~(outer_join._merge == 'both')].drop('_merge', axis = 1)
del outer_join
gc.collect()
train_data.shape

# %%
zero_prediction = []
for i in range(0, len(c)):
    zero_prediction.append(
        pd.DataFrame({
            "date":pd.date_range("2017-08-16", "2017-08-31").tolist(),
            "store_nbr":c.store_nbr.iloc[i],
            "family":c.family.iloc[i],
            "sales":0
            })
    )
zero_prediction = pd.concat(zero_prediction)
del c
gc.collect()
zero_prediction

# %%
c = train_data.groupby(["family", "store_nbr"]).tail(60).groupby(["family", "store_nbr"]).sales.sum().reset_index()

# %%
c[c.sales == 0]

# %%
train_data

# %%
fig, ax = plt.subplots(1, 5, figsize = (20, 4))
train_data[(train_data.store_nbr == 10) & (train_data.family == "LAWN AND GARDEN")].set_index("date").sales.plot(ax = ax[0], title = "STORE 10 - LAWN AND GARDEN")
train_data[(train_data.store_nbr == 36) & (train_data.family == "LADIESWEAR")].set_index("date").sales.plot(ax = ax[1], title = "STORE 36 - LADIESWEAR")
train_data[(train_data.store_nbr == 6) & (train_data.family == "SCHOOL AND OFFICE SUPPLIES")].set_index("date").sales.plot(ax = ax[2], title = "STORE 6 - SCHOOL AND OFFICE SUPPLIES")
train_data[(train_data.store_nbr == 14) & (train_data.family == "BABY CARE")].set_index("date").sales.plot(ax = ax[3], title = "STORE 14 - BABY CARE")
train_data[(train_data.store_nbr == 53) & (train_data.family == "BOOKS")].set_index("date").sales.plot(ax = ax[4], title = "STORE 53 - BOOKS")
plt.show()

# %%
a = train_data.set_index("date").groupby("family").resample("D").sales.sum().reset_index()
px.line(a, x = "date", y = "sales", color = "family", title = "Daily total sales of the family")

# %%
a = train_data.groupby("family").sales.sum().sort_values(ascending = False).reset_index()
px.bar(a, y = "family", x = "sales", color = "family", title = "Which product family preferred more?")

# %%
print("Spearman Correlation between Sales and Onpromotion:{:,.4f}".format(train_data.corr("spearman").sales.loc["onpromotion"]))

# %%
d = pd.merge(train_data, stores)
d["store_nbr"] = d["store_nbr"].astype("int8")
d["year"] = d.date.dt.year
px.line(d.groupby(["city", "year"]).sales.mean().reset_index(), x = "year", y = "sales", color = "city")

# %% [markdown]
# ### Analyze data(hoildays and events)

# %%
# transferred holidays
tr1 = holi[(holi.type == "Holiday") & (holi.transferred == True)].drop("transferred", axis = 1).reset_index(drop = True)
tr2 = holi[(holi.type == "Transfer")].drop("transferred", axis = 1).reset_index(drop = True)
tr = pd.concat([tr1, tr2], axis = 1)
tr = tr.iloc[:, [5, 1, 2, 3, 4]]

holi = holi[(holi.transferred == False) & (holi.type != "Transfer")].drop("transferred", axis = 1)
holi = holi.append(tr).reset_index(drop = True)

# additional holidays
holi["description"] = holi["description"].str.replace("-", "").str.replace("+", "").str.replace("\d+", "")
holi["type"] = np.where(holi["type"] == "Additional", "Holiday", holi["type"])

# bridge holidays
holi["description"] = holi["description"].str.replace("Puente", "")
holi["type"] = np.where(holi["type"] == "Bridge", "Holiday", holi["type"])

# work day holidays, that is meant to payback the bridge
work_day = holi[holi.type == "Work Day"]
holi = holi[holi.type != "Work Day"]

# split
# events are national
events = holi[holi.type == "Event"].drop(["type", "locale", "locale_name"], axis = 1).rename({"description":"events"}, axis = 1)

holi = holi[holi.type != "Event"].drop("type", axis = 1)
regional = holi[holi.locale == "Regional"].rename({"locale_name":"state", "description":"holiday_regional"}, axis = 1).drop("locale", axis = 1).drop_duplicates()
national = holi[holi.locale == "National"].rename({"description":"holiday_national"}, axis = 1).drop(["locale", "locale_name"], axis = 1).drop_duplicates()
local = holi[holi.locale == "Local"].rename({"description":"holiday_local", "locale_name":"city"}, axis = 1).drop("locale", axis = 1).drop_duplicates()

d = pd.merge(train_data.append(test_data), stores)
d["store_nbr"] = d["store_nbr"].astype("int8")

# national holidays & events
d = pd.merge(d, national, how = "left")
# regional
d = pd.merge(d, regional, how = "left", on = ["date", "state"])
# local
d = pd.merge(d, local, how = "left", on = ["date", "city"])
# work day
d = pd.merge(d, work_day[["date", "type"]].rename({"type":"IsWorkDay"}, axis = 1), how = "left")
# events
events["events"] = np.where(events.events.str.contains("futbol"), "Futbol", events.events)

def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
    df = pd.get_dummies(df, columns = categorical_columns, dummy_na = nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    df.columns = df.columns.str.replace(" ", "_")
    return df, df.columns.tolist()

events, events_cat = one_hot_encoder(events, nan_as_category = False)
events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1, events["events_Dia_de_la_Madre"])
events = events.drop(239)

d = pd.merge(d, events, how = "left")
d[events_cat] = d[events_cat].fillna(0)

# new features
d["holiday_national_binary"] = np.where(d.holiday_national.notnull(), 1, 0)
d["holiday_local_binary"] = np.where(d.holiday_local.notnull(), 1, 0)
d["holiday_regional_binary"] = np.where(d.holiday_regional.notnull(), 1, 0)

#
d["national_independence"] = np.where(d.holiday_national.isin(['Batalla de Pichincha', 'Independencia de Cuenca', 'Independencia de Guayaquil', 'Independencia de Guayaquil', 'Primer Grito de Independencia']), 1, 0)
d["local_cantonizacio"] = np.where(d.holiday_local.str.contains("Cantonizacio"), 1, 0)
d["local_fundacion"] = np.where(d.holiday_local.str.contains("Fundacion"), 1, 0)
d["local_independencia"] = np.where(d.holiday_local.str.contains("Independencia"), 1, 0)

holi, holi_cat = one_hot_encoder(d[["holiday_national", "holiday_regional", "holiday_local"]], nan_as_category = False)
d = pd.concat([d.drop(["holiday_national", "holiday_regional", "holiday_local"], axis = 1), holi], axis = 1)

he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holi")].tolist() + d.columns[d.columns.str.startswith("national")].tolist() + d.columns[d.columns.str.startswith("local")].tolist()
d[he_cols] = d[he_cols].astype("int8")

d[["family", "city", "state", "type"]] = d[["family", "city", "state", "type"]].astype("category")

del holi, holi_cat, work_day, local, regional, national, events, events_cat, tr, tr1, tr2, he_cols
gc.collect()

d.head(10)

# %% [markdown]
# ### Apply an AB test to events and holidays features
# <h6> H0: The sales are equal(M1 = M2)<br>
# H1: The sales are not equal(M1 != M2)
# </h6>

# %%
# def AB_Test(dataframe, group, target):

#     # packasges
#     from scipy.stats import shapiro
#     import scipy.stats as stats

#     # split AB
#     groupA = dataframe[dataframe[group] == 1][target]
#     groupB = dataframe[dataframe[group] == 0][target]

#     # assumption: normality
#     ntA = shapiro(groupA)[1] < 0.05
#     ntB = shapiro(groupB)[1] < 0.05
#     # H0: distribution is normal! - False
#     # H1: distribution is not notmal! - True

#     if (ntA == False) & (ntB == False): # H0: normal distribution
#         # parametric test, assumption: homogeneity of variances
#         leveneTest = stats.levene(groupA, groupB)[1] < 0.05
#         # H0: homogeneity: False
#         # H1: heterogeneous: True

#         if leveneTest == False:
#             # homogeneity
#             ttest = stats.ttest_ind(groupA, groupB, equal_var = True)[1]
#             # H0: M1 == M2 - False
#             # H1: M1 != M2 - True
#         else:
#             # heterogeneous
#             ttest = stats.ttest_ind(groupA, groupB, equal_var = False)[1]
#             # H0: M1 == M2 - False
#             # H1: M1 != M2 - True
#     else:
#         # non_parametric test
#         ttest = stats.mannwhitneyu(groupA, groupB)[1]
#         # H0: M1 == M2 - False
#         # H1: M1 != M2 - True
    
#     # result
#     temp = pd.DataFrame({
#         "AB Hypothesis" : [ttest < 0.05],
#         "p-value":[ttest]
#     })
#     temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
#     temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
#     temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "A/B groups are similar!", "A/B groups are not similar!")
#     temp["Feature"] = group
#     temp["GroupA_mean"] = groupA.mean()
#     temp["GroupB_mean"] = groupB.mean()
#     temp["GroupA_median"] = groupA.median()
#     temp["GroupB_median"] = groupB.median()

#     # columns
#     if (ntA == False) & (ntB == False):
#         temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
#         temp = temp[["Feature", "Test Type", "Homogeneity", "AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
#     else:
#         temp = temp[["Feature", "Test Type", "AB Hypothesis", "p-value", "Comment", "GroupA_mean", "GroupB_mean", "GroupA_median", "GroupB_median"]]
#     return temp

# # apply A/B testing
# he_cols = d.columns[d.columns.str.startswith("events")].tolist() + d.columns[d.columns.str.startswith("holiday")].tolist() + d.columns[d.columns.str.startswith("national")].tolist() + d.columns[d.columns.str.startswith("local")].tolist()
# ab = []
# for i in he_cols:
#     ab.append(AB_Test(dataframe = d[d.sales.notnull()], group = i, target = "sales"))
# ab = pd.concat(ab)
# ab

# %%
d.groupby(["family", "events_Futbol"]).sales.mean()[:60]

# %% [markdown]
# ### Time related features

# %%
# time related features
def create_date_features(df):
    df['month'] = df.date.dt.month.astype("int8")
    df['day_of_month'] = df.date.dt.day.astype("int8")
    df['day_of_year'] = df.date.dt.dayofyear.astype("int16")
    df['week_of_month'] = (df.date.apply(lambda d:(d.day-1) // 7 + 1)).astype("int8")
    df['week_of_year'] = (df.date.dt.weekofyear).astype("int8")
    df['day_of_week']  = (df.date.dt.dayofweek + 1).astype("int8")
    df['year'] = df.date.dt.year.astype("int32")
    df['is_wknd'] = (df.date.dt.weekday // 4).astype("int8")
    df['quarter'] = df.date.dt.quarter.astype("int8")
    df['is_month_start'] = df.date.dt.is_month_start.astype("int8")
    df['is_month_end'] = df.date.dt.is_month_end.astype("int8")
    df['is_quarter_start'] = df.date.dt.is_quarter_start.astype("int8")
    df['is_year_start'] = df.date.dt.is_year_start.astype("int8")
    df['is_year_end'] = df.date.dt.is_year_end.astype("int8")
    # 0: winther 1: spring 2: summer 3: fall
    df["season"] = np.where(df.month.isin([12, 1, 2]), 0, 1)
    df["season"] = np.where(df.month.isin([6, 7, 8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
    return df
d = create_date_features(d)

# workday column
d["workday"] = np.where((d.holiday_national_binary == 1) | (d.holiday_local_binary == 1) | (d.holiday_regional_binary == 1) | (d['day_of_week'].isin([6, 7])), 0, 1)
d["workday"] = pd.Series(np.where(d.IsWorkDay.notnull(), 1, d["workday"])).astype("int8")
d.drop("IsWorkDay", axis = 1, inplace = True)

# wages in the public sector are paid every two weeks on the 15th and on th last day of the month
# supermarket sales could be affected by this
d["wageday"] = pd.Series(np.where((d['is_month_end'] == 1) | (d["day_of_month"] == 15), 1, 0)).astype("int8")

d.head(15)

# %% [markdown]
# ### Earhquake influence about the store sales

# %%
# march
d[(d.month.isin([4, 5]))].groupby(["year"]).sales.mean()

# %%
# april - may
pd.pivot_table(d[(d.month.isin([3]))], index = "year", columns = "family", values = "sales", aggfunc = "mean")

# %%
# june
pd.pivot_table(d[(d.month.isin([6]))], index = "year", columns = "family", values = "sales", aggfunc = "mean")

# %% [markdown]
# ### ACF & PACF for each family

# %%
a = d[(d.sales.notnull())].groupby(["date", "family"]).sales.mean().reset_index().set_index("date")
for num, i in enumerate(a.family.unique()):
    try:
        fig, ax = plt.subplots(1, 2, figsize = (15, 5))
        temp = a[(a.family == i)]
        sm.graphics.tsa.plot_acf(temp.sales, lags = 365, ax = ax[0], title = "AUTOCORRELATION\n" + i)
        sm.graphics.tsa.plot_pacf(temp.sales, lags = 365, ax = ax[1], title = "PARTIAL AUTOCORRELATION\n" + i)
    except:
        pass


# %%
a = d[d.year.isin([2016, 2017])].groupby(["year", "day_of_year"]).sales.mean().reset_index()
px.line(a, x = "day_of_year", y = "sales", color = "year", title = "Average sales for 2016 and 2017")

# %% [markdown]
# ### simple moving average

# %%
a = train_data.sort_values(["store_nbr", "family", "date"])
for i in [20, 30, 45, 60, 90, 120, 365, 730]:
    a["SMA" + str(i) + "_sales_lag16"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(16).values
    a["SMA" + str(i) + "_sales_lag30"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(30).values
    a["SMA" + str(i) + "_sales_lag60"] = a.groupby(["store_nbr", "family"]).rolling(i).sales.mean().shift(60).values
print("Correlation")
a[["sales"] + a.columns[a.columns.str.startswith("SMA")].tolist()].corr()

# %%
b = a[(a.store_nbr == 1)].set_index("date")
for i in b.family.unique():
    fig, ax = plt.subplots(2, 4, figsize = (20, 10))
    b[b.family == i][["sales", "SMA20_sales_lag16"]].plot(legend = True, ax = ax[0, 0], linewidth = 4)
    b[b.family == i][["sales", "SMA30_sales_lag16"]].plot(legend = True, ax = ax[0, 1], linewidth = 4)
    b[b.family == i][["sales", "SMA45_sales_lag16"]].plot(legend = True, ax = ax[0, 2], linewidth = 4)
    b[b.family == i][["sales", "SMA60_sales_lag16"]].plot(legend = True, ax = ax[0, 3], linewidth = 4)
    b[b.family == i][["sales", "SMA90_sales_lag16"]].plot(legend = True, ax = ax[1, 0], linewidth = 4)
    b[b.family == i][["sales", "SMA120_sales_lag16"]].plot(legend = True, ax = ax[1, 1], linewidth = 4)
    b[b.family == i][["sales", "SMA365_sales_lag16"]].plot(legend = True, ax = ax[1, 2], linewidth = 4)
    b[b.family == i][["sales", "SMA730_sales_lag16"]].plot(legend = True, ax = ax[1, 3], linewidth = 4)
    plt.suptitle("STORE 1 - " + i, fontsize = 15)
    plt.tight_layout(pad = 1.5)
    for j in range(0, 4):
        ax[0, j].legend(fontsize = "x-large")
        ax[1, j].legend(fontsize = "x-large")
    plt.show()

# %% [markdown]
# ### Exponential Moving Average

# %%
def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store_nbr", "family"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha = alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [16, 30, 60, 90]

a = ewm_features(a, alphas, lags)

# %%
a[(a.store_nbr == 1) & (a.family == "GROCERY I")].set_index("date")[["sales", "sales_ewm_alpha_095_lag_16"]].plot(title = "STROE 1 - GROCERY I");

# %% [markdown]
# ###### * 아래 링크를 참고 했으며, 개인적인 공부를 위한 코드입니다.<br>참고) https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide


