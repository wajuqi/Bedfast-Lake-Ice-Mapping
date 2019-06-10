from osgeo import gdal
import ogr
from numpy import *
import numpy as np
import glob
import osr
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
import datetime
import pandas as pd

Image.MAX_IMAGE_PIXELS = 1000000000

def transform_wgs84_to_utm(lon, lat, zone):
    # def get_utm_zone(longitude):
    #     return (int(1+(longitude+180.0)/6.0))

    def is_northern(latitude):
        """
        Determines if given latitude is a northern for UTM
        """
        if (latitude < 0.0):
            return 0
        else:
            return 1

    utm_coordinate_system = osr.SpatialReference()
    utm_coordinate_system.SetWellKnownGeogCS("WGS84") # Set geographic coordinate system to handle lat/lon
    # utm_coordinate_system.SetUTM(get_utm_zone(lon), is_northern(lat))
    utm_coordinate_system.SetUTM(zone, is_northern(lat))

    wgs84_coordinate_system = utm_coordinate_system.CloneGeogCS() # Clone ONLY the geographic coordinate system

    # create transform component
    wgs84_to_utm_transform = osr.CoordinateTransformation(wgs84_coordinate_system, utm_coordinate_system) # (<from>, <to>)
    return wgs84_to_utm_transform.TransformPoint(lon, lat, 0) # returns easting, northing, altitude

##get lake boundary extent
def lake_extent(shp):
    source_ds = ogr.Open(shp)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    return x_min, x_max, y_min, y_max

##threshold function
def threshold(a):   #a: incidence angle in degrees
    # t = 0.00335*a*a-0.4682*a-3.11495      #modified poly 2
    t = -0.2573484665035253*a-6.933835899301701
    return t

##incidence angle normalization
def normalization(x):
    y = -0.2573484665035253 * (x - 35)
    return y

##create landmask band: rasterize shapefile
def create_mask(filename, shp, cols_shp, rows_shp, x_start, y_start):
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    source_ds = ogr.Open(shp)
    source_layer = source_ds.GetLayer()

    output_mask = filename[:-4] + '_landmask.tif'
    target_ds = gdal.GetDriverByName('GTiff').Create(output_mask, cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(ds.GetGeoTransform())
    target_ds.SetProjection(ds.GetProjection())  ##sets same projection as input
    band_shp = target_ds.GetRasterBand(1)
    NoData_value = -999999
    band_shp.SetNoDataValue(NoData_value)
    band_shp.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], source_layer)
    target_ds = None
    band_shp = None

    # # the created landmask can be used for MAGIC (IRGS)
    # landmask = output_mask[:-3] + "bmp"
    # im = Image.open(output_mask)
    # out = im.convert("1")
    # out.save(landmask, "BMP")
    # os.access(output_mask, os.W_OK)
    # os.chmod(output_mask, 0o777)        #change the permission
    mask_array = gdal.Open(output_mask).ReadAsArray(x_start, y_start, cols_shp, rows_shp)
    os.remove(output_mask)


    return mask_array

## read images as array
def read_array(filename, x_min, x_max, y_min, y_max):
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    geotransform = ds.GetGeoTransform()
    originX = geotransform[0]  # top left x
    originY = geotransform[3]  # top left y
    pixelWidth = geotransform[1]  # w-e pixel resolution
    pixelHeight = -geotransform[5]  # n-s pixel resolution

    # read the study area into array
    x_start = int((x_min - originX) / pixelWidth)
    y_start = int((originY - y_max) / pixelHeight)
    cols_shp = abs(int((x_max - x_min) / pixelHeight))
    rows_shp = abs(int((y_max - y_min) / pixelWidth))

    # find polarization
    pol_start = filename.find('GRD') + 7
    pol_end = filename.find('_', pol_start)
    pols = filename[pol_start:pol_end]
    # print('Polarization:', pols)

    if pols == 'DV':    #'VH,VV'
        band = ds.GetRasterBand(2)
        band_theta = ds.GetRasterBand(ds.RasterCount)
    elif pols in ('DH','SH','SV','HH'):
        band = ds.GetRasterBand(1)
        band_theta = ds.GetRasterBand(ds.RasterCount)
    else:
        print("Polarization error!")

    backscatter_array = band.ReadAsArray(x_start, y_start, cols_shp, rows_shp)
    theta_array = band_theta.ReadAsArray(x_start, y_start, cols_shp, rows_shp)
    out_array = np.stack((backscatter_array,theta_array), axis = 2)
    return out_array, cols_shp, rows_shp, x_start, y_start

def plot_time_series(region, ice_type, lat, lon, year, ice_on_date, xaxis, temp, ice_thickness, snow_thickness):
    s = time_series_norm[lat, lon]
    s_mask = np.isfinite(s)

    fig = plt.figure()
    host = fig.add_subplot(111)

    par1 = host.twinx()
    par2 = host.twinx()

    host.set_xlim([datetime.date(year - 1, 9, 1), datetime.date(year, 8, 1)])
    host.set_ylim(-40, 0)
    par1.set_ylim(-60, 60)
    par2.set_ylim(-2.5, 2.5)

    host.set_xlabel("UTC Time")
    host.set_ylabel("Backscatter coefficient(dB)")
    par1.set_ylabel("ERA5 air temperature(◦C)")
    par2.set_ylabel("CLIMo ice thickness/snow depth(m)")

    color1 = plt.cm.viridis(0)
    color2 = plt.cm.viridis(0.5)
    color3 = 'orangered'

    p1, = host.plot(time_x[s_mask], s[s_mask], color=color1, label="Backscatter coefficient(dB)")
    p1_1 = host.axhline(y=threshold(35), color=color1, linestyle='--', label='Backscatter threshold')
    p1_2 = host.axvline(x=datetime.datetime(year - 1, 1, 1) + datetime.timedelta(ice_on_date - 1), linestyle='--',
                        color='blue', linewidth='1', label='ice-on date')
    host.text(datetime.datetime(year - 1, 1, 1) + datetime.timedelta(ice_on_date - 1) - datetime.timedelta(31), -3,
              'ice-on date')
    p1_3 = host.axvline(x=datetime.datetime(year, 5, 16), linestyle='--', color='orange', linewidth='1',label='melt onset date')
    host.text(datetime.datetime(year, 5, 16) - datetime.timedelta(31), -3, 'melt onset date')
    p2, = par1.plot(xaxis, temp, color=color2, label="ERA5 air temperature(◦C)")
    p2_1 = par1.axhline(y=0, color=color2, linestyle='--', label='0◦C')
    p3, = par2.plot(xaxis, ice_thickness, color=color3, label="CLIMo ice thickness/snow depth(m)")
    p3_1, = par2.plot(xaxis, snow_thickness, color=color3)

    lns = [p1_1, p2_1]
    host.legend(handles=lns, loc='best')

    # right, left, top, bottom
    par2.spines['right'].set_position(('outward', 45))

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    host.grid(which='major', linestyle=':', linewidth='0.5', color='black')
    plt.title(ice_type + ' ice time series ' + '(' + region +')')
    plt.gcf().autofmt_xdate()  # Rotation
    plt.tight_layout()

start_time = time.time()
code = 'C01'
rootdir = r'C01\2017-2018'
shp_dir = r'Shapefiles'
snow = '53%'
year = 2018
climo_start_year = 2013
climo_year = year - climo_start_year        #freeze-up year
utm_flag = 1    #1: if Sentinel-1 images are in UTM (needs to convert shp coordinates to utm)
ice_state = ['Bedfast', 'Floating']

'''
UTM Zone number and Central Meridian: http://www.jaworski.ca/utmzones.htm
#C01_Barrow(Alaska, USA): zone 4, -159.0
#C03_Teshekpuk(Alaska, USA): zone 5, -153.0
#C04_Mackenzie(Canada): zone 8, -135.0
#C06_Kytalyk(Russia): zone 55, 147.0
#C07_LenaDelta(Russia): zone 52, 129.0
#C08_Yamal(Russia): zone 42, 69.0
#CS12_1_Cape_Espenberg_Lowland(Alaska, USA): zone 3, -168.0
#CS12_2_Central_Seward_Peninsula(Alaska, USA): zone 3, -168.0
'''
if code == 'C01':
    region = 'Barrow, Alaska'
    shp = shp_dir + '\\Barrow.shp'
    zone = 4
elif code == 'C03':
    region = 'Teshekpuk, Alaska'
    # shp = shp_dir + '\\Teshekpuk_Lake2c.shp'
    shp = shp_dir + '\\Teshekpuk_Lake2c_clip.shp'
    zone = 5
elif code == 'C04':
    region = 'Mackenzie, Canada'
    shp = shp_dir + '\\Mackenzie Delta_NWTcmodify.shp'
    zone = 8
elif code == 'C06':
    region = 'Kytalyk, Russia'
    shp = shp_dir + '\\Kytalyk_SiberiaC2.shp'
    zone = 55
elif code == 'C07':
    region = 'Lena Delta, Russia'
    shp = shp_dir + '\\Lena_delta2c.shp'
    zone = 52
elif code == 'C08':
    region = 'Yamal, Russia'
    shp = shp_dir + '\\Yamal_SiberiaC2.shp'
    zone = 42
elif code == 'CS12_1':
    region = 'Cape Espenberg Lowland, Alaska'
    shp = shp_dir + '\\Cape_Espenberg_Lowland.shp'
    zone = 3
elif code == 'CS12_1':
    region = 'Central Seward Peninsula, Alaska'
    shp = shp_dir + '\\Central_Seward_Peninsula.shp'
    zone = 3
else:
    print("Region code not defined.")

## Transform wgs84 coordinates of lake extent to utm
x_min, x_max, y_min, y_max = lake_extent(shp)
if utm_flag == 1:
    x_min, y_min, altitude = transform_wgs84_to_utm(x_min, y_min, zone)
    x_max, y_max, altitude = transform_wgs84_to_utm(x_max, y_max, zone)

## Sort by dates rather than S1A/S1B
file_list = glob.glob(rootdir +'\*.tif')
def sort_by_datetime(x):
    d_start = x.find('GRD') + 10
    d_end = x.find('_', d_start)
    return x[d_start:d_end]
filenames = sorted(file_list, key = sort_by_datetime)

## x axis for plot
time_list = []
for x in filenames:
    time_list.append(sort_by_datetime(x))
time_list = [x[:4] + '-' + x[4:6] + '-' + x[6:8] + ' ' + x[9:11] + ':' + x[11:13] + ':' + x[13:15] for x in time_list]
time_format = '%Y-%m-%d %H:%M:%S'
time_xaxis = [datetime.datetime.strptime(i, time_format) for i in time_list]
time_x = np.asarray(time_xaxis)

all_arrays = []
all_arrays_theta = []
all_arrays_norm = []
all_arrays_thresh = []
vthreshold = np.vectorize(threshold)

for n in range(0, len(filenames)):
    print('Read image %d...'% (n+1))
    print(filenames[n])
    out_array, cols_shp, rows_shp, x_start, y_start = read_array(filenames[n], x_min, x_max, y_min, y_max)
    mask_array = create_mask(filenames[n], shp, cols_shp, rows_shp, x_start, y_start)

    ## Mask out land area (backscatter and incidence angle)
    out_array[:,:,0] = ma.masked_array(out_array[:,:,0], mask = ~mask_array.astype(bool)).filled(nan)
    out_array[:,:,1] = ma.masked_array(out_array[:,:,1], mask = ~mask_array.astype(bool)).filled(nan)

    out_array[:,:,1][out_array[:,:,1] == 0] = np.nan

    all_arrays.append(10 * log10(out_array[:, :, 0]))
    all_arrays_norm.append(10*log10(out_array[:,:,0]) - normalization(out_array[:,:,1]))
    all_arrays_theta.append(out_array[:,:,1])
    all_arrays_thresh.append(vthreshold(out_array[:,:,1]))

## Convert list of arrays into 3-dimensional array that stores the times series for all pixels(rows, cols) and all dates(axis)
time_series = np.dstack(all_arrays)
time_series_norm = np.dstack(all_arrays_norm)
time_series_theta = np.dstack(all_arrays_theta)
time_series_thresh = np.dstack(all_arrays_thresh)

## Extract ERA5 hourly air temperature data for each image
era_temp = pd.read_csv(r'D:\ERA5&NOAA\C01\air-temperature_hr.txt', sep='\t', header=None)
era_temp.columns = ["Year", "Month", "Day", "Hour","Temp"]
temp = []
for x in time_list:
    temp.append(era_temp.loc[(era_temp['Year'] == int(x[:4])) & (era_temp['Month'] == int(x[5:7])) & (era_temp['Day'] == int(x[8:10])) & (era_temp['Hour'] == int(x[11:13])), "Temp"].values)
temp = list(map(float64, temp))

#Extract climo simulated freeze-up date (ice-on date)
climo_ice_on = pd.read_csv(r'D:\Python scripts\climo\C01_3m\freezeup-0%.txt', delim_whitespace=True, engine='python')
climo_ice_on.columns = ['Year', ' ', 'Julian Day']
ice_on_date = climo_ice_on.loc[climo_ice_on['Year'] == climo_year, 'Julian Day'].values
ice_on_date = int(ice_on_date[-1])          #choose the last freeze-up date and first break-up date

#Extract climo simulated thickness
climo_thickness = pd.read_csv(r'D:\Python scripts\climo\C01_1m\thickness-' + snow + '.txt', delim_whitespace=True, engine='python', usecols=np.arange(6))
climo_thickness.columns = ['Day','Date','Net annual surface ice growth(m)','Net annual ice growth at bottom(m)','Ice Thickness(m)','Snow Thickness(m)']
ice_thickness = []
snow_thickness = []
for x in time_list:
    ice_thickness.append(climo_thickness.loc[climo_thickness['Date'] == x[:10], 'Ice Thickness(m)'].values)
    snow_thickness.append(climo_thickness.loc[climo_thickness['Date'] == x[:10], 'Snow Thickness(m)'].values)
ice_thickness = list(map(float64, ice_thickness))
ice_thickness = [i * (-1) for i in ice_thickness]
snow_thickness = list(map(float64, snow_thickness))

# For selected samples in SNAP
sample_path = r'H:\GlobPermafrost\Samples\time series samples' + '\\' + code
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
for file in os.listdir(sample_path):
    if file.endswith(".txt"):
        print(os.path.join(sample_path, file))
        sample = pd.read_csv(sample_path + '\\' + file, sep='\t',usecols=np.arange(2))
        sample.columns = ["lon","lat"]
        sample = sample.astype(int)
        file, file_extension = os.path.splitext(file)
        if file[0] == 'g':
            ice_type = ice_state[0]
        elif file[0] == 'f':
            ice_type = ice_state[1]
        else:
            ice_type = ice_state[1]
        for index, row in sample.iterrows():
            print(row['lat'], row['lon'])
            plot_time_series(region, ice_type, row['lat'], row['lon'], year, ice_on_date, time_xaxis, temp, ice_thickness, snow_thickness)
            fig_path = sample_path + '\\' + str(year-1) + '-' + str(year) + '-' + snow + '\\' + file + '\\'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(fig_path + str(row['lat']) + '_' + str(row['lon']))
            # plt.savefig(fig_path + str(row['lat']) + '_' + str(row['lon']) + '_ew')

print("--- %s seconds ---" % (time.time() - start_time))