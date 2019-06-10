'''
Extract ERA5 (temp, wind, humidity, cloud) & NOAA GHCN (Global Historical Climatology Network) Daily Snow Depth data for Canadian Lake Ice Model (CLIMo).
TO MODIFY: region_code, rootdir, noaa_filedir and starty & endy if needed.
'''
import era_extract
import pandas as pd
import numpy as np

region_code = 'ALASKA'
rootdir = r'D:\ERA5&NOAA'
noaa_filedir = r'D:\ERA5&NOAA\NOAA_GHCN'    # downloaded NOAA GHCN data from https://www.ncdc.noaa.gov/cdo-web/datatools/findstation
era_filedir = rootdir + '\\' + region_code

(wind_df, wind_df_hr, humid_df, humid_df_hr, temp_df, temp_df_hr, snow_df, cloud_df, cloud_df_hr) = era_extract.eraextract(era_filedir, region_code, starty=1980, endy=2018)
df = pd.read_csv(noaa_filedir + '\\GHCN_' + region_code + '.csv', usecols=['DATE', 'SNWD'])
#fill missing date in NOAA GHCN data
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df.set_index(df.DATE, inplace=True)
df = df.resample("D").asfreq()

df = df.fillna(method='ffill')                              # fill missing value by preceding value (ffill: forward fill)
df = df[~((df.index.month == 2) & (df.index.day == 29))]    # delete date 02-29
#assign NOAA data to dataframe
snow_df['snow'] = df['SNWD'].values

# calculate snow accumulation (daily difference)
snow_df['snow'] = snow_df['snow'].diff(1)
snow_df.snow = np.where(snow_df.snow < 0, 0, snow_df.snow)
snow_df['snow'].iloc[0] = 0
snow_df['snow'] = snow_df['snow'] * 0.0254          # convert inch(standard unit) to meter

# delete unuseful columns
wind_df = wind_df.drop(['u10', 'v10'], axis=1)
wind_df_hr = wind_df_hr.drop(['u10', 'v10'], axis=1)
humid_df = humid_df.drop(['dew', 'air'], axis=1)
humid_df_hr = humid_df_hr.drop(['dew', 'air'], axis=1)

wind_df.to_csv(era_filedir + '\\wind-speed.txt', header=None, index=None, sep='\t')
humid_df.to_csv(era_filedir + '\\relative-humidity.txt', header=None, index=None, sep='\t')
temp_df.to_csv(era_filedir + '\\air-temperature.txt', header=None, index=None, sep='\t')
snow_df.to_csv(era_filedir + '\\snow-accumulation-rate.txt', header=None, index=None, sep='\t')
cloud_df.to_csv(era_filedir + '\\cloud-cover.txt', header=None, index=None, sep='\t')

wind_df_hr.to_csv(era_filedir + '\\wind-speed_hr.txt', header=None, index=None, sep='\t')
humid_df_hr.to_csv(era_filedir + '\\relative-humidity_hr.txt', header=None, index=None, sep='\t')
temp_df_hr.to_csv(era_filedir + '\\air-temperature_hr.txt', header=None, index=None, sep='\t')
cloud_df_hr.to_csv(era_filedir + '\\cloud-cover_hr.txt', header=None, index=None, sep='\t')

del df, wind_df, wind_df_hr, humid_df, humid_df_hr, temp_df, temp_df_hr, snow_df, cloud_df, cloud_df_hr