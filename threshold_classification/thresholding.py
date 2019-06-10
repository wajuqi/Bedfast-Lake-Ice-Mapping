# Only support 64-bit Python
# Parameters to provide: rootdir and shp_dir in main()
# Output image has the same extent as the input image.

from osgeo import gdal
import ogr
import numpy as np
import os
from os.path import join
from PIL import Image
import time

Image.MAX_IMAGE_PIXELS = 1000000000

## thresholding function
def threshold(a):       #a: incidence angle in degrees
    t = -0.25*a-6.93
    return t

## create landmask band: rasterize shapefile
def createmask(filename, shp):
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

    landmask = output_mask[:-3] + "bmp"
    im = Image.open(output_mask)
    out = im.convert("1")
    out.save(landmask, "BMP")
    os.remove(output_mask)

    mask_array = gdal.Open(landmask).ReadAsArray()
    return mask_array

## write the image
def writeoutput(filename, result, output_path, percent_g, percent_f):
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    ## get date & mode
    d_start = filename.find('GRD') + 10
    d_end = filename.find('_', d_start)
    date_time = filename[d_start:d_end]
    mode_start = filename.find('GRD') - 7
    mode_end = filename.find('_GRD', mode_start)
    mode = filename[mode_start:mode_end]

    with open(output_path + "\\Percentage.txt", "a") as file:
        file.write(date_time + '\t' + str(percent_g) + '\t' + str(percent_f) + '\n')

    outfile = output_path + '\\H2O_THRESH_V01_' + date_time + '_' + mode + '_' + '.tif'
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile, cols, rows, 1, gdal.GDT_Byte)
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())   # sets same projection as input
    outband = outdata.GetRasterBand(1)
    ct = gdal.ColorTable()
    ct.SetColorEntry(0, (0, 0, 0, 255))
    ct.SetColorEntry(1, (0, 0, 255, 255))       # bedfast ice color(0,0,255), used to be (0,0,102)
    ct.SetColorEntry(2, (0, 255, 255, 255))     # floating ice color(0,255,255)
    outband.SetColorTable(ct)
    outband.WriteArray(result)
    outband.FlushCache()

def classification(filename, shp, output_path):
    ds = gdal.Open(filename, gdal.GA_ReadOnly)

    ## find polarization
    pol_start = filename.find('GRD') + 7
    pol_end = filename.find('_', pol_start)
    pols = filename[pol_start:pol_end]
    print('Polarization:', pols)

    if pols == 'DV':  # 'VH,VV'
        band = ds.GetRasterBand(2)
        band_theta = ds.GetRasterBand(ds.RasterCount)
    elif pols in ('DH', 'SH', 'SV', 'HH'):
        band = ds.GetRasterBand(1)
        band_theta = ds.GetRasterBand(ds.RasterCount)
    else:
        print("Polarization error!")

    array = band.ReadAsArray()
    theta_array = band_theta.ReadAsArray()      # read the projected local incidence angle band into array

    ## create mask image
    mask_array = createmask(filename, shp)
    array = np.ma.masked_array(array, mask=~mask_array.astype(bool)).filled(0)

    ## convert to unit dB
    np.seterr(divide='ignore')
    array = np.where(array != 0, 10 * np.log10(array), 0)   # sigmaNought[dB] = 10*log10(sigmaNought)
    np.seterr(divide='warn')                                # ignore the zero warning

    def assign(a, b):
        if a == 0:
            return np.nan
        elif a > b:
            return 2  # floating ice value
        else:
            return 1  # bedfast ice value

    vassign = np.vectorize(assign)
    result = vassign(array, threshold(theta_array))

    bedfast = np.count_nonzero(result == 1)
    floating = np.count_nonzero(result == 2)
    percent_g = round(bedfast / (bedfast + floating), 4)
    percent_f = round(floating / (bedfast + floating), 4)
    print("bedfast ice%: ", percent_g)
    print("floating ice%: ", percent_f)

    ## write result in colour
    writeoutput(filename, result, output_path, percent_g, percent_f)

def main():
    start_time = time.time()
    rootdir = r's1_images'  # folder for preprocessed Sentinel-1 images
    shp = r'shapefiles\Barrow.shp'  # shapefile for study area
    output_path = rootdir + '\\THRESH'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ## write the percentage of each ice type
    percent = output_path + "\\Percentage.txt"
    with open(percent, "a") as file:
        file.write('THRESHOLD\n'+'Date\tBedfast ice\tFloating ice'+'\n')

    ## Classification
    for file in os.listdir(rootdir):
        if file.endswith(".tif"):
            start_time_every = time.time()
            print (os.path.join(rootdir, file))
            classification(os.path.join(rootdir, file), shp, output_path)
            print("--- %s seconds ---" % (time.time() - start_time_every))

    for item in os.listdir(output_path):
        if item.endswith(".xml"):
            os.remove(join(output_path, item))

    print("---Total time %s seconds ---" % (time.time() - start_time))

if __name__== "__main__":
    main()