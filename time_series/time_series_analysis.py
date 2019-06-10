from osgeo import gdal
import ogr
from numpy import *
import numpy as np
import glob
import osr
import os
from PIL import Image
import time
import datetime
from itertools import groupby, count
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

##write the classified image
def write_output(filename, cols_shp, rows_shp, x_min, y_max, code, result, output_path, percent_g, percent_f):
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    geotransform = ds.GetGeoTransform()
    pixelWidth = geotransform[1]  # w-e pixel resolution
    pixelHeight = -geotransform[5]  # n-s pixel resolution

    ## get date & mode
    d_start = filename.find('GRD') + 10
    d_end = filename.find('_', d_start)
    date_time = filename[d_start:d_end]
    mode_start = filename.find('GRD') - 7
    mode_end = filename.find('_GRD',mode_start)
    mode = filename[mode_start:mode_end]

    with open(output_path + "\\Percentage.txt", "a") as file:
        file.write(date_time + '\t' + str(percent_g) + '\t' + str(percent_f) + '\n')

    outfile = output_path + '\\H2O_THRESH_V01_' + date_time + '_' + mode + '_' + code + '.tif'
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile, cols_shp, rows_shp, 1, gdal.GDT_Byte)   #NaN supports float type
    outdata.SetGeoTransform((x_min, pixelWidth, 0, y_max, 0, -pixelHeight))
    outdata.SetProjection(ds.GetProjection())   # sets same projection as input
    outband = outdata.GetRasterBand(1)
    ct = gdal.ColorTable()
    # ct.SetColorEntry(0, (255,255,255,255))    # white background for NaN
    ct.SetColorEntry(0, (0, 0, 0, 255))         # RGBA is short for Red, Green, Blue, Alpha
    ct.SetColorEntry(1, (0, 0, 255, 255))       # grounded ice color(0,0,255), used to be (0,0,102)
    ct.SetColorEntry(2, (0, 255, 255, 255))     # floating ice color(0,255,255)
    outband.SetColorTable(ct)
    outband.WriteArray(result)
    outband.FlushCache()  ##saves to disk!!


start_time = time.time()
code = 'C01'
climo_snow = ['0%', '53%','100%']
'''Image date range: ice-on date to the date before melt date'''
rootdir = r'C01\2016-2017\ice season'
shp_dir = r'Shapefiles'
year = 2017
climo_start_year = 2013
climo_year = year - climo_start_year        #freeze-up year
output_path = rootdir + '\\TimeSeriesTHRESH'
bath_doy_path = rootdir + '\\DOY_BATH'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(bath_doy_path):
    os.makedirs(bath_doy_path)
utm_flag = 1    #1: if Sentinel-1 images are in UTM (needs to convert shp coordinates to utm)

if code == 'C01':
    shp = shp_dir + '\\Barrow.shp'
    zone = 4
else:
    print("Region code not defined.")

percent = output_path + "\\Percentage.txt"
with open(percent, "a") as file:
    file.write('THRESHOLD\n'+'date\tBedfast ice\tFloating ice'+'\n')

# Transform wgs84 coordinates of lake extent to utm
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

index_grounded = np.zeros((rows_shp, cols_shp), dtype=np.float16)
'''Image date range: ice-on date to the date before melt date'''
for i in range(0,rows_shp):
    for j in range(0,cols_shp):
        below_thresh = np.where(time_series[i,j] <= time_series_thresh[i,j])[0].tolist()                        #find the index of backscatter values that are below the threshold
        index_remain_below = [list(g) for k, g in groupby(below_thresh, key=lambda n, c=count(): n - next(c))]  #find the index of the consecutive numbers
        if len(index_remain_below) == 0:        # always above threshold
            index_grounded[i, j] = -1
        elif index_remain_below[-1][-1] == len(filenames)-1 and len(index_remain_below[-1]) >= 5:
            index_grounded[i, j] = index_remain_below[-1][0]
        else:
            index_grounded[i,j] = -1        #always remain afloat

index_grounded = ma.masked_array(index_grounded, mask = ~mask_array.astype(bool)).filled(nan)
index_grounded_int = ma.masked_array(index_grounded, mask = ~mask_array.astype(bool)).filled(-2).astype(int)
time_xaxis_int = [int(time.mktime(i.timetuple())) for i in time_xaxis]

def classify(a,n):
    if a == nan:
        return nan
    elif a == -1 or a > n:
        return 2  # floating ice value
    elif a <= n:
        return 1  # grounded ice value

def assign_date(a):
    if a == -1 or a == -2:
        return nan
    else:
        return time_xaxis_int[int(a)]

vclassify = np.vectorize(classify)
vassign_date = np.vectorize(assign_date)

## write classified images
for n in range(0, len(filenames)):
    result = vclassify(index_grounded,n)
    grounded = np.count_nonzero(result == 1)
    floating = np.count_nonzero(result == 2)
    percent_g = round(grounded/(grounded+floating), 4)
    percent_f = round(floating/(grounded+floating), 4)
    print(filenames[n])
    print ("grounded ice%: ", percent_g)
    print ("floating ice%: ", percent_f)
    write_output(filenames[n], cols_shp, rows_shp, x_min, y_max, code, result, output_path, percent_g, percent_f)

date_grounded = vassign_date(index_grounded_int)
doy_min = datetime.datetime.fromtimestamp(np.nanmin(date_grounded)).timetuple().tm_yday
doy_max = datetime.datetime.fromtimestamp(np.nanmax(date_grounded)).timetuple().tm_yday
print("DOY range:", doy_min, doy_max)

# scale to 1-255 for display, 0: nan
doy_range = (280,130)
date_range = ((datetime.datetime(year-1, 1, 1) + datetime.timedelta(doy_range[0] - 1)), (datetime.datetime(year, 1, 1) + datetime.timedelta(doy_range[1] - 1)))
doy_range_min = time.mktime(date_range[0].timetuple())
doy_range_max = time.mktime(date_range[1].timetuple())
date_grounded_norm = (date_grounded-doy_range_min)/(doy_range_max-doy_range_min) * 254 + 1

ds = gdal.Open(filenames[0], gdal.GA_ReadOnly)

## Write DOY map
outfile = bath_doy_path + '\\DOY_'+ str(year-1) + '-' + str(year) + '_' + code + '_' + str(doy_min) + '-'+ str(doy_max) + '_range'+ str(doy_range[0]) + '-'+ str(doy_range[1]) +'.tif'
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(outfile, cols_shp, rows_shp, 1, gdal.GDT_Byte)     #Byte will convert NaN to 0, however, color table only supports Byte or UInt16
outdata.SetGeoTransform((x_min, 40, 0, y_max, 0, -40))
outdata.SetProjection(ds.GetProjection())  ##sets same projection as input
outband = outdata.GetRasterBand(1)
outband.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

ct = gdal.ColorTable()
ct.CreateColorRamp(1,(0,128,255),int(255/2),(255,255,0))
ct.CreateColorRamp(int(255/2),(255,255,0),255,(255,0,0))
ct.SetColorEntry(0, (0,0,0,255)) #black background for NaN

outband.SetColorTable(ct)
outband.SetNoDataValue(0)
outband.WriteArray(date_grounded_norm)
outband.FlushCache() ##saves to disk!!
outdata = None
outband = None


## Write bathymetry map
for i in range(0, len(climo_snow)):
    ## Extract climo simulated thickness
    climo_thickness = pd.read_csv(r'D:\Python scripts\climo\C01_1m\thickness-' + climo_snow[i] + '.txt',
                                  delim_whitespace=True, engine='python', usecols=np.arange(6))
    climo_thickness.columns = ['Day', 'Date', 'Net annual surface ice growth(m)', 'Net annual ice growth at bottom(m)',
                               'Ice Thickness(m)', 'Snow Thickness(m)']
    ice_thickness = []
    for x in time_list:
        ice_thickness.append(climo_thickness.loc[climo_thickness['Date'] == x[:10], 'Ice Thickness(m)'].values)
    ice_thickness = list(map(float64, ice_thickness))
    ice_thickness = [i * (-1) for i in ice_thickness]

    def assign_thickness(a):
        if a == -2:
            return nan
        elif a == -1:
            return inf
        else:
            return -ice_thickness[int(a)]
    vassign_thickness = np.vectorize(assign_thickness)

    bathymetry = vassign_thickness(index_grounded_int)
    bath_min = np.nanmin(bathymetry)
    bath_max = np.nanmax(bathymetry[bathymetry != inf])
    print("Bathymetry range:", bath_min, bath_max)

    ## scale to 1-254 for display, 0: nan, 255: floating ice
    bathymetry_norm = (bathymetry - bath_min) / (bath_max - bath_min) * 253 + 1
    bathymetry_norm[bathymetry_norm == inf] = 255

    outfile = bath_doy_path + '\\Bathymetry_' + str(year-1) + '-' + str(year) + '_' + code + '_' + climo_snow[i] + '_' + str(
        bath_min) + '-' + str(bath_max) + '.tif'

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile, cols_shp, rows_shp, 1,
                            gdal.GDT_Byte)      # Byte will convert NaN to 0, however, color table only supports Byte or UInt16
    outdata.SetGeoTransform((x_min, 40, 0, y_max, 0, -40))
    outdata.SetProjection(ds.GetProjection())   # sets same projection as input
    outband = outdata.GetRasterBand(1)
    outband.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    ct = gdal.ColorTable()
    ct.CreateColorRamp(1, (255, 0, 0), int(254 / 2), (255, 255, 0))
    ct.CreateColorRamp(int(254 / 2), (255, 255, 0), 254,(0, 128, 255))
    ct.SetColorEntry(255, (0, 0, 255, 255))     # floating ice
    ct.SetColorEntry(0, (0, 0, 0, 255))         # black background for NaN

    outband.SetColorTable(ct)
    outband.SetNoDataValue(0)
    outband.WriteArray(bathymetry_norm)
    outband.FlushCache()  # saves to disk
    outdata = None
    outband = None

print("--- %s seconds ---" % (time.time() - start_time))