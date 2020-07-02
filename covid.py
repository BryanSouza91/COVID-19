# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/BryanSouza91/COVID-19/blob/master/COVID-19.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# # This is only the tested and reported cases John Hopkins CCSE has data for this is by no means a definitive view of the global epidemic.
# 
# ##### The repo is updated daily around 5:00pm PDT

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from datetime import date, timedelta


# %%
confirmed_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

recovered_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

deaths_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"


# %%
conf_df = pd.read_csv(confirmed_url) # ,index_col=['Province/State', 'Country/Region', 'Lat', 'Long']) 

recv_df = pd.read_csv(recovered_url) # ,index_col=['Province/State', 'Country/Region', 'Lat', 'Long'])

death_df = pd.read_csv(deaths_url) # ,index_col=['Province/State', 'Country/Region', 'Lat', 'Long'])


# %%
latest = conf_df.columns[-1]
latest


# %%
dates = conf_df.loc[:,'1/22/20':].columns


# %%
# create a differenced series function

def difference(dataset, interval=1):
    return pd.Series([dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))])

# %% [markdown]
# # Plots total confirmed cases by country
# 
# ##### Changing the logx=False to True shows the logarithmic scales of x-axis
# ##### Changing the logy=False to True shows the logarithmic scales of y-axis
# ##### Changing the loglog=False to True shows the logarithmic scales of both axes

# %%
conf_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == 'China'].sum().plot(figsize=(25,6),logx=False,logy=False,loglog=True);


# %%
conf_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == 'US'].sum().plot(figsize=(25,6),logx=False,logy=False,loglog=False);


# %%
conf_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == 'Japan'].sum().plot(figsize=(25,6),logx=False,logy=False,loglog=True);


# %%
conf_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == 'Italy'].sum().plot(figsize=(25,6),logx=False,logy=False,loglog=True);


# %%
conf_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == 'Iran'].sum().plot(figsize=(25,6),logx=False,logy=False,loglog=True);


# %%
conf_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == 'Russia'].sum().plot(figsize=(25,6),logx=False,logy=False,loglog=True);


# %%
conf_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == 'Greece'].sum().plot(figsize=(25,6),logx=False,logy=False,loglog=True);


# %%
conf_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == 'India'].sum().plot(figsize=(25,6),logx=False,logy=False,loglog=True);


# %%
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 Confirmed Cases")
sns.set_palette('colorblind')
sns.scatterplot(x='Long',y='Lat',size=latest,hue='Country/Region',data=conf_df,sizes=(10,10000),legend=False,edgecolor='k');


# %%
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 Recovered Cases")
sns.set_palette('colorblind')
sns.scatterplot(x='Long',y='Lat',size=latest,hue='Country/Region',data=recv_df,sizes=(10,10000),legend=False,edgecolor='k');


# %%
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 Deaths")
sns.set_palette('colorblind')
sns.scatterplot(x='Long',y='Lat',size=latest,hue='Country/Region',data=death_df,sizes=(10,10000),legend=False,edgecolor='k');


# %%
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 World Confirmed Cases")
sns.set_palette('colorblind')
ax = sns.stripplot(data=conf_df);
ax.set_xticks(np.arange(0, max(len(dates), 14)));
ax.set_xticklabels(pd.date_range(min(dates), max(dates),freq="14D"));


# %%
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 World Deaths")
sns.set_palette('colorblind')
sns.stripplot(data=death_df);


# %%
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 World Recovered Cases")
sns.set_palette('colorblind')
sns.stripplot(data=recv_df);


# %%
conf_df.loc[::,"1/22/20"::]

# %% [markdown]
# # Daily Reporting CSVs

# %%
def toStrftime(t):
    return t.strftime('%x').replace('/','-')


def getGroup(df, group, locale):
    grouped_df = df.groupby(group)
    return grouped_df.get_group(locale)


# %%
latest = (date.today() - timedelta(days=1))
latest = toStrftime(latest)
print(latest)


# %%
startDate = '04/12/20'

dateList = pd.date_range(startDate, latest, freq='d')
dateList = list(map(toStrftime,dateList))

# print(dateList)
print(len(dateList))


# %%
url_="https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_daily_reports_us/"


# %%
def daily(date):
    df = pd.read_csv(url_+date+"20.csv")
    return getGroup(df, 'Province_State', 'California')


# %%
fullDf = pd.concat(map(daily, dateList))


# %%
fullDf.tail()


# %%
fullDf.columns


# %%
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 California")
sns.set_palette('colorblind')
sns.lineplot(x="Last_Update", y="Incident_Rate", data=fullDf);
sns.lineplot(x="Last_Update", y="Testing_Rate", data=fullDf);


# %%
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 California")
sns.set_palette('colorblind')
sns.lineplot(x="Last_Update", y="Confirmed", data=fullDf);
sns.lineplot(x="Last_Update", y="Deaths", data=fullDf);
sns.lineplot(x="Last_Update", y="Active", data=fullDf);


# %%
fullDf7 = pd.DataFrame(fullDf.rolling(7).mean())


# %%
fullDf7.tail()


# %%
days = list(range(len(fullDf7)))
yVar = "Confirmed"
plt.figure(figsize=(26,13))
plt.title(f"SARS-Cov-2 COVID-19 California {yVar}")
sns.set_palette('colorblind')
sns.barplot(x=days, y=yVar, data=fullDf);
sns.lineplot(x=days, y=yVar, data=fullDf7);

# %% [markdown]
# # World report

# %%
# Create reusable series objects 
conf_sum = conf_df.loc[:,'1/22/20':].sum()
recv_sum = recv_df.loc[:,'1/22/20':].sum()
death_sum = death_df.loc[:,'1/22/20':].sum()

conf_sum_dif = difference(conf_sum).values
recv_sum_dif = difference(recv_sum).values
death_sum_dif = difference(death_sum).values


# %%
# Print world report
print("World numbers current as of {}".format(conf_df.columns[-1]))
print()
print("New cases:                                          {0} | {1:.3%}".format(conf_sum_dif[-1],conf_sum_dif[-1]/conf_sum[-1]))
print("Total confirmed cases:                              {0}".format(conf_sum[-1]))
print("New case 7- | 30- | 60-day Moving Average:          {0:.0f} | {1:.0f} | {2:.0f}".format(difference(conf_sum, 1).rolling(7).mean().values[-1],difference(conf_sum, 1).rolling(30).mean().values[-1],difference(conf_sum, 1).rolling(60).mean().values[-1]))
print("New recovered cases:                                {0} | {1:.3%}".format(recv_sum_dif[-1],recv_sum_dif[-1]/conf_sum[-1]))
print("Total recovered:                                    {0} | {1:.3%}".format(recv_sum[-1],recv_sum[-1]/conf_sum[-1]))
print("Recovered 7- | 30- | 60-day Moving Average:         {0:.0f} | {1:.0f} | {2:.0f}".format(difference(recv_sum, 1).rolling(7).mean().values[-1],difference(recv_sum, 1).rolling(30).mean().values[-1],difference(recv_sum, 1).rolling(60).mean().values[-1]))
print("New Deaths:                                         {0} | {1:.3%}".format(death_sum_dif[-1],death_sum_dif[-1]/conf_sum[-1]))
print("Total deaths:                                       {0} | {1:.3%}".format(death_sum[-1],death_sum[-1]/conf_sum[-1]))
print("Death 7- | 30- | 60-day Moving Average:             {0:.0f} | {1:.0f} | {2:.0f}".format(difference(death_sum, 1).rolling(7).mean().values[-1],difference(death_sum, 1).rolling(30).mean().values[-1],difference(death_sum, 1).rolling(60).mean().values[-1]))
print("Total Resolved Cases:                               {0} | {1:.3%}".format((recv_sum[-1] + death_sum[-1]),((recv_sum[-1] + death_sum[-1])/conf_sum[-1])))
print("Deaths as percentage of Total Resolved:             {0:.3%}".format((death_sum[-1]/(recv_sum[-1] + death_sum[-1]))))
print()
print("Growth rate above 1.0 is sign of exponential growth,")
print("but also skewed by increased testing.")
print("World Growth rate:                                  {0:.4}".format((conf_sum_dif[-1])/(conf_sum_dif[-2])))
print()
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 World Daily Change")
sns.set_palette('colorblind')
sns.lineplot(data=conf_sum_dif);
sns.lineplot(data=recv_sum_dif);
sns.lineplot(data=death_sum_dif);

# %% [markdown]
# # Report for each country reporting cases
# 

# %%
# define report function
def report(country):

    # Create reusable series objects 
    country_conf_sum = conf_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == country].sum()
    country_recv_sum = recv_df.loc[:,'1/22/20':].loc[recv_df['Country/Region'] == country].sum()
    country_death_sum = death_df.loc[:,'1/22/20':].loc[conf_df['Country/Region'] == country].sum()

    country_conf_sum_dif = difference(country_conf_sum).values
    country_recv_sum_dif = difference(country_recv_sum).values
    country_death_sum_dif = difference(country_death_sum).values

    print()
    print('_'*80)
    print("Numbers for {} current as of {}".format(country, country_conf_sum.index[-1]))
    print()
    print("New cases:                                          {0} | {1:.3%}".format(country_conf_sum_dif[-1],country_conf_sum_dif[-1]/country_conf_sum[-1]))
    print("Total confirmed cases:                              {0}".format(country_conf_sum[-1]))
    print("New case 7- | 30- | 60-day Moving Average:          {0:.0f} | {1:.0f} | {2:.0f}".format(difference(country_conf_sum, 1).rolling(7).mean().values[-1],difference(country_conf_sum, 1).rolling(30).mean().values[-1],difference(country_conf_sum, 1).rolling(60).mean().values[-1]))
    print("New recovered cases:                                {0} | {1:.3%}".format(country_recv_sum_dif[-1],(country_recv_sum_dif[-1]/country_conf_sum[-1])))
    print("Total recovered cases:                              {0} | {1:.3%}".format(country_recv_sum[-1],(country_recv_sum[-1]/country_conf_sum[-1])))
    print("New recovered 7- | 30- | 60-day Moving Average:     {0:.0f} | {1:.0f} | {2:.0f}".format(difference(country_recv_sum, 1).rolling(7).mean().values[-1],difference(country_recv_sum, 1).rolling(30).mean().values[-1],difference(country_recv_sum, 1).rolling(60).mean().values[-1]))
    print("New Deaths:                                         {0} | {1:.3%}".format(country_death_sum_dif[-1],country_death_sum_dif[-1]/country_conf_sum[-1]))
    print("Total deaths:                                       {0} | {1:.3%}".format(country_death_sum[-1],(country_death_sum[-1]/country_conf_sum[-1])))
    print("Death 7- | 30- | 60-day Moving Average:             {0:.0f} | {1:.0f} | {2:.0f}".format(difference(country_death_sum, 1).rolling(7).mean().values[-1],difference(country_death_sum, 1).rolling(30).mean().values[-1],difference(country_death_sum, 1).rolling(60).mean().values[-1]))
    print("Total Resolved Cases:                               {0} | {1:.3%}".format((country_recv_sum[-1] + country_death_sum[-1]),((country_recv_sum[-1] + country_death_sum[-1])/country_conf_sum[-1])))
    print("Deaths as percentage of Total Resolved:             {0:.3%}".format((country_death_sum[-1]/(country_recv_sum[-1] +
    country_death_sum[-1]))))
    print()
    print()
    print("Growth rate:                                        {0:.4}".format(country_conf_sum_dif[-1]/country_conf_sum_dif[-2]))
    print("_"*80)
    plt.figure(figsize=(26,13))
    plt.title(f"SARS-Cov-2 COVID-19 {country} Daily Change")
    # plt.show_legend()
    sns.set_palette('colorblind')
    sns.lineplot(data=country_conf_sum_dif);
    sns.lineplot(data=country_recv_sum_dif);
    sns.lineplot(data=country_death_sum_dif);


# %%
report('US')


# %%
report('New Zealand')


# %%
report('Italy')


# %%
report('United Kingdom')


# %%
report('Brazil')


# %%
report('France')


# %%
report('Sweden')


# %%
# for each in conf_df['Country/Region'].sort_values().unique():
    # report(each)

# %% [markdown]
# ## Make a graphing function similar to the reporting function.
# ### Moving Average Graphs
# ### Death as Percentage graph

# %%
plt.figure(figsize=(26,13))
plt.title("SARS-Cov-2 COVID-19 World Daily Change")
sns.set_palette('colorblind')
sns.lineplot(data=conf_sum_dif);
sns.lineplot(data=recv_sum_dif);
sns.lineplot(data=death_sum_dif);


# %%


