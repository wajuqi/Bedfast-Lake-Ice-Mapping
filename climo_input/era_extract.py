# Use for setting up ERA5 data to run in climo

from netCDF4 import Dataset
from statistics import mean
import pandas as pd
import numpy as np

def eraextract(filedir, code, starty, endy):
    # Create main dataframes for use in climo
    wind_df = pd.DataFrame(columns=['year', 'month', 'day', 'u10', 'v10', 'wind'])
    wind_df_hr = pd.DataFrame(columns=['year', 'month', 'day', 'hr', 'u10', 'v10', 'wind'])

    humid_df = pd.DataFrame(columns=['year', 'month', 'day', 'dew', 'air', 'humidity'])
    humid_df_hr = pd.DataFrame(columns=['year', 'month', 'day', 'hr', 'dew', 'air', 'humidity'])

    temp_df = pd.DataFrame(columns=['year', 'month', 'day', 'air'])
    temp_df_hr = pd.DataFrame(columns=['year', 'month', 'day', 'hr', 'air'])

    snow_df = pd.DataFrame(columns=['year', 'month', 'day', 'snow'])

    cloud_df = pd.DataFrame(columns=['year', 'month', 'day', 'cloud'])
    cloud_df_hr = pd.DataFrame(columns=['year', 'month', 'day', 'hr', 'cloud'])

    # Making a list of years with ERA Data
    years = range(starty, endy + 1)
    for year in years:
        print("\nYear %d" % year)
        # Open ERA netcdf associated with current year
        cfile = Dataset(filedir + '\\ERA5_{}_'.format(year) + code + '.nc')
        # List of variables in netcdf and total length of time index in netcdf
        vartemp = ['u10', 'v10', 'd2m', 't2m', 'tcc']

        time = range(0, len(cfile.variables['time']))
        # Create temporary dataframes
        windtemp_df = pd.DataFrame(columns=['year', 'month', 'day', 'u10', 'v10', 'wind'])
        windtemp_df_hr = pd.DataFrame(columns=['year', 'month', 'day', 'hr', 'u10', 'v10', 'wind'])

        humidtemp_df = pd.DataFrame(columns=['year', 'month', 'day', 'dew', 'air', 'humidity'])
        humidtemp_df_hr = pd.DataFrame(columns=['year', 'month', 'day', 'hr', 'dew', 'air', 'humidity'])

        temptemp_df = pd.DataFrame(columns=['year', 'month', 'day', 'air'])
        temptemp_df_hr = pd.DataFrame(columns=['year', 'month', 'day', 'hr', 'air'])

        snowtemp_df = pd.DataFrame(columns=['year', 'month', 'day', 'snow'])

        cloudtemp_df = pd.DataFrame(columns=['year', 'month', 'day', 'cloud'])
        cloudtemp_df_hr = pd.DataFrame(columns=['year', 'month', 'day', 'hr', 'cloud'])

        # Create lists of years, months, days to populate dataframes
        cdf_years = []
        cdf_months = []
        cdf_days = []
        cdf_years_d = []
        cdf_months_d = []
        cdf_days_d = []
        cdf_hours = []
        # Get Daily Data
        if (year / 4).is_integer():
            cdf_years += 366 * [year]
        else:
            cdf_years += 365 * [year]

        months = range(1, 13)
        for month in months:
            if month == 2:
                if (year / 4).is_integer():
                    cdf_months += 29 * [month]
                    cdf_days += range(1, 30)
                else:
                    cdf_months += 28 * [month]
                    cdf_days += range(1, 29)
            elif month == 4 or \
                    month == 6 or \
                    month == 9 or \
                    month == 11:
                cdf_months += 30 * [month]
                cdf_days += range(1, 31)
            else:
                cdf_months += 31 * [month]
                cdf_days += range(1, 32)
        # Get hourly data
        if (year / 4).is_integer():
            cdf_years_d += 8784 * [year]
        else:
            cdf_years_d += 8760 * [year]

        months = range(1, 13)
        for month in months:
            if month == 2:
                if (year / 4).is_integer():
                    cdf_months_d += 696 * [month]
                    for num in range(1, 30):
                        cdf_days_d += 24 * [num]
                    for num in range(1, 30):
                        for _ in range(1, 25):
                            cdf_hours += [_]
                else:
                    cdf_months_d += 672 * [month]
                    for num in range(1, 29):
                        cdf_days_d += 24 * [num]
                    for num in range(1, 29):
                        for _ in range(1, 25):
                            cdf_hours += [_]
            elif month == 4 or \
                    month == 6 or \
                    month == 9 or \
                    month == 11:
                cdf_months_d += 720 * [month]
                for num in range(1, 31):
                    cdf_days_d += 24 * [num]
                for num in range(1, 31):
                    for _ in range(1, 25):
                        cdf_hours += [_]
            else:
                cdf_months_d += 744 * [month]
                for num in range(1, 32):
                    cdf_days_d += 24 * [num]
                for num in range(1, 32):
                    for _ in range(1, 25):
                        cdf_hours += [_]

        # Populate dataframes
        windtemp_df['year'] = cdf_years
        windtemp_df['month'] = cdf_months
        windtemp_df['day'] = cdf_days
        humidtemp_df['year'] = cdf_years
        humidtemp_df['month'] = cdf_months
        humidtemp_df['day'] = cdf_days
        temptemp_df['year'] = cdf_years
        temptemp_df['month'] = cdf_months
        temptemp_df['day'] = cdf_days
        snowtemp_df['year'] = cdf_years
        snowtemp_df['month'] = cdf_months
        snowtemp_df['day'] = cdf_days
        cloudtemp_df['year'] = cdf_years
        cloudtemp_df['month'] = cdf_months
        cloudtemp_df['day'] = cdf_days

        windtemp_df_hr['year'] = cdf_years_d
        windtemp_df_hr['month'] = cdf_months_d
        windtemp_df_hr['day'] = cdf_days_d
        windtemp_df_hr['hr'] = cdf_hours
        humidtemp_df_hr['year'] = cdf_years_d
        humidtemp_df_hr['month'] = cdf_months_d
        humidtemp_df_hr['day'] = cdf_days_d
        humidtemp_df_hr['hr'] = cdf_hours
        temptemp_df_hr['year'] = cdf_years_d
        temptemp_df_hr['month'] = cdf_months_d
        temptemp_df_hr['day'] = cdf_days_d
        temptemp_df_hr['hr'] = cdf_hours
        cloudtemp_df_hr['year'] = cdf_years_d
        cloudtemp_df_hr['month'] = cdf_months_d
        cloudtemp_df_hr['day'] = cdf_days_d
        cloudtemp_df_hr['hr'] = cdf_hours

        print("Study Area:")
        print("\tLatitude:", cfile.variables['latitude'][:])
        print("\tLongitude:", cfile.variables['longitude'][:])

        # For each variable
        for vt in vartemp:
            print("Determining Daily Means for " + str(vt))
            hr_mean = []
            day_mean = []
            day_sum = []
            hr_mean_total = []

            # For each time in time index
            for t in time:
                area_mean = np.mean(cfile.variables[vt][t,:,:])
                hr_mean.append(area_mean)           # get hourly mean for the whole region
                hr_mean_total.append(area_mean)
                if len(hr_mean) == 24:              # when 24 values are in list ie. 1 day
                    day_mean.append(mean(hr_mean))  # get daily mean
                    day_sum.append(sum(hr_mean))    # get daily sum (daily average of snowfall would be sum of 24 hours)
                    hr_mean.clear()
                else:
                    continue
            # Append daily mean values to temporary dataframes for each variable
            if vt == "u10":
                windtemp_df['u10'] = day_mean
                windtemp_df_hr['u10'] = hr_mean_total
            elif vt == "v10":
                windtemp_df['v10'] = day_mean
                windtemp_df_hr['v10'] = hr_mean_total
            elif vt == "d2m":
                humidtemp_df['dew'] = day_mean
                humidtemp_df_hr['dew'] = hr_mean_total
            elif vt == "t2m":
                humidtemp_df['air'] = day_mean
                humidtemp_df_hr['air'] = hr_mean_total
                temptemp_df['air'] = day_mean
                temptemp_df_hr['air'] = hr_mean_total
            elif vt == 'tcc':
                cloudtemp_df['cloud'] = day_mean
                cloudtemp_df_hr['cloud'] = hr_mean_total

        # delete Feb 29th for CLIMo daily average
        if (year / 4).is_integer():
            windtemp_df = windtemp_df.drop(59)      # DOY 59: Feb 29
            humidtemp_df = humidtemp_df.drop(59)
            temptemp_df = temptemp_df.drop(59)
            snowtemp_df = snowtemp_df.drop(59)
            cloudtemp_df = cloudtemp_df.drop(59)

        # Append temporary dataframes to each main dataframe
        wind_df = wind_df.append(windtemp_df)
        humid_df = humid_df.append(humidtemp_df)
        temp_df = temp_df.append(temptemp_df)
        snow_df = snow_df.append(snowtemp_df)
        cloud_df = cloud_df.append(cloudtemp_df)

        wind_df_hr = wind_df_hr.append(windtemp_df_hr)
        humid_df_hr = humid_df_hr.append(humidtemp_df_hr)
        temp_df_hr = temp_df_hr.append(temptemp_df_hr)
        cloud_df_hr = cloud_df_hr.append(cloudtemp_df_hr)

    # calculate wind speed from u and v components
    wind_df['wind'] = np.sqrt(wind_df['u10'] ** 2 + wind_df['v10'] ** 2)
    wind_df_hr['wind'] = np.sqrt(wind_df_hr['u10'] ** 2 + wind_df_hr['v10'] ** 2)

    # convert from Kelvin to Celsius
    temp_df['air'] = temp_df['air'] - 273.15
    temp_df_hr['air'] = temp_df_hr['air'] - 273.15
    humid_df['dew'] = humid_df['dew'] - 273.15
    humid_df_hr['dew'] = humid_df_hr['dew'] - 273.15
    humid_df['air'] = humid_df['air'] - 273.15
    humid_df_hr['air'] = humid_df_hr['air'] - 273.15

    # Calculate humidity from dewpoint temp and air temp: http://andrew.rsmas.miami.edu/bmcnoldy/Humidity.html
    humid_df['humidity'] = 100 * (np.exp((17.625 * humid_df['dew'])/(243.04 + humid_df['dew'])) / np.exp((17.625 * humid_df['air']) / (243.04 + humid_df['air'])))
    humid_df_hr['humidity'] = 100 * (np.exp((17.625 * humid_df_hr['dew']) / (243.04 + humid_df_hr['dew'])) / np.exp((17.625 * humid_df_hr['air']) / (243.04 + humid_df_hr['air'])))

    return (wind_df, wind_df_hr, humid_df, humid_df_hr, temp_df, temp_df_hr, snow_df, cloud_df, cloud_df_hr)

