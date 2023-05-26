import numpy as np
import h5py
from skimage import data, draw, io
import matplotlib.pyplot as plt
from img_list import prs_list,snt_list
from PIL import Image
import tifffile as tiff
from find_nearest import find_nearest
import xarray as xr
import pandas as pd
from itertools import chain
from osgeo import gdal, osr
import os

def read_prs_l2d(datapath, tstart, tend):
    """
    Open PRISMA L2D file and create GEOTIFF in a given directory

    Parameters:
    ----------
    datapath: str, folder with raw L2D files
    tstart: str, acquisition start time
    tend: str, acquisition end time

    Returns:
    -------
    lat: ndarray, latitude matrix
    lon: ndarray, longitude matrix
    cwv: ndarray, central wavelength vector
    vrf: ndarray, visible reflectance cube
    srf: ndarray, infrared reflectance cube
    """
    filename=datapath+'PRS_L2D_STD_'+tstart+'_'+tend+'_0001'
    pf = h5py.File(filename+'.he5','r')

    # Read wavelengths, drop zero ones and overlap
    attrs = pf.attrs
    img_id = attrs['Image_ID']
    vn_wvl = np.array([ wvl for wvl in attrs['List_Cw_Vnir'] ])
    sw_wvl = np.array([ wvl for wvl in attrs['List_Cw_Swir'] ])


    info = {}
    info['img_id'] = img_id

    # Read geometry information
    geom = pf['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geometric Fields']
    relang = np.array(geom['Rel_Azimuth_Angle'][:])
    obsang = np.array(geom['Observing_Angle'][:])
    szeang = np.array(geom['Solar_Zenith_Angle'][:])

    data = pf['HDFEOS']['SWATHS']['PRS_L2D_HCO']

    # Read geographical information
    lat = np.array(data['Geolocation Fields']['Latitude'][:])
    lon = np.array(data['Geolocation Fields']['Longitude'][:])

    vrf = np.array(data['Data Fields']['VNIR_Cube'][:]) 
    srf = np.array(data['Data Fields']['SWIR_Cube'][:]) 

    # Adjust visible bands and data
    vn_wvl = vn_wvl[3::]
    vrf = vrf[:,3::,:]
    vn_wvl = vn_wvl[::-1]
    vrf = vrf[:,::-1,:]

    # Adjust infrared bands and data
    srf = srf[:,::-1,:]
    sw_wvl = sw_wvl[::-1]
    srf = srf[:,6::,:]
    sw_wvl = sw_wvl[6::]

    # Reorder dimensions to space-bands
    vrf = np.moveaxis(vrf, 1, 2)
    srf = np.moveaxis(srf, 1, 2)
    # compute reflectance and merge cubes
    vrf,srf = reflectance_norm(datapath,vrf,srf,tstart,tend)
    cube = np.dstack((vrf,srf))
    print(np.shape(vrf),np.shape(srf))
    print('max vrf reflectance value: '+str(vrf.max())+'\n')
    print('min vrf reflectance value: '+str(vrf.min()))
    pf.close()
    
    #Here starts the tiff conversion
    #get minimum and maximum latitude and longitude
    xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]

    #get pixel spatial resolution
    xres = (xmax-xmin)/lat.shape[1]#lat.shape[1] gives the number of cols
    yres = (ymax-ymin)/lat.shape[0]#lat.shape[0] gives the number of rows

    #define coordinates
    geotransform=(xmin,xres,0,ymax,0, -yres)#zeros (third and fifth parameters) are for rotation

    #cube exytaction as GEOTIFF
    #define GeoTIFF structure and output filename
    output_raster = gdal.GetDriverByName('GTiff').Create(filename+"_reflectance.tif",cube.shape[1], cube.shape[0], cube.shape[2] ,gdal.GDT_Float32)  # Open the file
        
    #loop over all bands and write it to the GeoTIFF
    for b in range(1,cube.shape[2]):
        print("converting band",b)
        outband = output_raster.GetRasterBand(b) 
        outband.WriteArray(cube[:,:,b])
    #specify coordinates to WGS84
    output_raster.SetGeoTransform(geotransform)  
    srs = osr.SpatialReference()                 
    srs.ImportFromEPSG(4326)                                                               
    output_raster.SetProjection(srs.ExportToWkt())

    #clean memory     
    output_raster.FlushCache()
    """
    #SRF
    #define GeoTIFF structure and output filename
    output_raster_s = gdal.GetDriverByName('GTiff').Create(filename+"srf.tif",srf.shape[1], srf.shape[0], srf.shape[2] ,gdal.GDT_Float32)  # Open the file
        
    #loop over all bands and write it to the GeoTIFF
    for b_s in range(1,srf.shape[2]):
        print("converting band",b_s)
        outband_s = output_raster_s.GetRasterBand(b_s) 
        outband_s.WriteArray(srf[:,:,b_s])
    #specify coordinates to WGS84
    output_raster_s.SetGeoTransform(geotransform)  
    srs = osr.SpatialReference()                 
    srs.ImportFromEPSG(4326)                                                               
    output_raster_s.SetProjection(srs.ExportToWkt())

    #clean memory     
    output_raster_s.FlushCache()
    """
     #reduced cube creation
    #define GeoTIFF structure and output filename
    print('creating reduced cube for unmixing (cut to 2400nm)')
    cut_idx=216 #index where wavelength is 2400nm: biggest wavelegnth of aerial sensor
    output_raster = gdal.GetDriverByName('GTiff').Create(filename+"_reduced(unmixing).tif",cube.shape[1], cube.shape[0], cut_idx ,gdal.GDT_Float32)  # Open the file
        
    #loop over all bands and write it to the GeoTIFF
    for b in range(1,cut_idx):
        print("converting band",b)
        outband = output_raster.GetRasterBand(b) 
        outband.WriteArray(cube[:,:,b])
    #specify coordinates to WGS84
    output_raster.SetGeoTransform(geotransform)  
    srs = osr.SpatialReference()                 
    srs.ImportFromEPSG(4326)                                                               
    output_raster.SetProjection(srs.ExportToWkt())

    #clean memory     
    output_raster.FlushCache()
    print("Conversion from he5 PRISMA file to GeoTIFF complete.")

    return lat, lon, vn_wvl, sw_wvl, vrf, srf, info


def spherdist(zlon1,zlat1,zlon2,zlat2):
    """ Haversine distance between two points

    zlon1: float, longitude of first point in degrees
    zlat1: float, latitude of first point
    zlon2: float, longitude of second point
    zlat2: float, latitude of second point

    zd: float, distance (km) between points 1 and 2
    """
    d2r = np.pi/180.
    zhavelon = np.sin(0.5*(zlon2-zlon1)*d2r)**2
    zhavelat = np.sin(0.5*(zlat2-zlat1)*d2r)**2
    zcosfac = np.cos(zlat1*d2r)*np.cos(zlat2*d2r)
    zc = 2*np.arcsin(np.sqrt(zhavelat+(zcosfac*zhavelon)))
    zd = zc*6371000./1000.
    return zd

def make_rgb(ztstart, ztend, path):
    """
    Helper function to make RGB-like images
    from the visible cubes
    """
    wl_rgb = [640., 560., 480.]
    yy, xx, vwl, _, vrf, _, info = read_prs_l2d(path, ztstart, ztend)
    img = vrf.astype(float)

    # Get indices for the RGB channels
    idx_rgb = []
    for wl in wl_rgb:
        idx_rgb.append(np.abs(vwl-wl).argmin())

    rgb = img[:,:, idx_rgb]

    # Normalize frames for better visualization. 
    # Using np.max() is not good if there are clouds or bright features.
    for iw in range(len(wl_rgb)):
        rgb[:,:,iw] /= np.percentile(rgb[:,:,iw],99.5)
    
    return rgb

def make_rgb_dc(img, vwl):
    """
    Helper function to make RGB-like images
    from the visible cubes already coregistered
    """
    wl_rgb = [640., 560., 480.]

    img = img.astype(float) #vrf datacube here

    # Get indices for the RGB channels
    idx_rgb = []
    for wl in wl_rgb:
        idx_rgb.append(np.abs(vwl-wl).argmin())

    rgb = img[:,:, idx_rgb]

    # Normalize frames for better visualization. 
    # Using np.max() is not good if there are clouds or bright features.
    for iw in range(len(wl_rgb)):
        rgb[:,:,iw] /= np.percentile(rgb[:,:,iw],99.5)
    
    return rgb


def reflectance_norm(path_l2d,t1v,t1s,tstart1,tend1):
    
    pf1 = h5py.File(path_l2d+'PRS_L2D_STD_'+tstart1+'_'+tend1+'_0001.he5','r')
    attrs1 = pf1.attrs
    L2ScaleVnirMax1 = attrs1['L2ScaleVnirMax']
    L2ScaleVnirMin1 = attrs1['L2ScaleVnirMin']
    ref_v=L2ScaleVnirMin1 + t1v*(L2ScaleVnirMax1-L2ScaleVnirMin1)/65535 # scaling to get reflectance
    
    L2ScaleSwirMax1 = attrs1['L2ScaleSwirMax']
    L2ScaleSwirMin1 = attrs1['L2ScaleSwirMin']
    ref_s=L2ScaleSwirMin1 + t1s*(L2ScaleSwirMax1-L2ScaleSwirMin1)/65535 # scaling to get reflectance
    
    return ref_v,ref_s

def get_coord_id(path_l2d,img,tstart,tend):
    
    pf1 = h5py.File(path_l2d+'PRS_L2D_STD_'+tstart1+'_'+tend1+'_0001.he5','r')
    attrs1 = pf1.attrs
    lat = attrs1['Product_center_lat']
    lon = attrs1['Product_center_long']
    img_id = attrs1['Image_ID']

    return lat,lon,img_id
"""
def AOI(t1,t2,vwl,long=None,lat=None,key=None,coreg=True):
    #select an AOI in which to perform CVA
    y='n'
    if np.shape(t1)[2] == 230: #if it has 230 bands print PRISMA rgb
        rgb1 = make_rgb_dc(t1,vwl)
        rgb2 = make_rgb_dc(t2,vwl)
    else:                      #if not it's Landsat so print Landsat rgb
        R_idx=find_nearest(vwl,640)
        G_idx=find_nearest(vwl,560)
        B_idx=find_nearest(vwl,480)
        rgb1 = RGB_Landsat(t1,R_idx,G_idx,B_idx)
        rgb2 = RGB_Landsat(t2,R_idx,G_idx,B_idx)
        thresh=0.18
        rgb1[rgb1>thresh]=thresh
        rgb1=rgb1/thresh
        rgb2[rgb2>thresh]=thresh
        rgb2=rgb2/thresh
    while y=='n':
        if key==None:
            xtopleft,ytopleft,xbottomright,ybottomright=input('insert the 4 indexes of the square AOI separated by a space (order: xtopleft,ytopleft,xbottomright,ybottomright): ').split() 
        elif coreg:
            xtopleft,ytopleft,xbottomright,ybottomright=use_cases[key][1]
            y='y'
        else:
            xtopleft,ytopleft=use_cases[key][3]
            xbottomright=xtopleft+450
            ybottomright=ytopleft+450
            y='y'
        start=[int(ytopleft),int(xtopleft)]
        end=[ int(ybottomright), int(xbottomright)]
        row, col = draw.rectangle_perimeter(start=start, end=end)   
        fig, ax = plt.subplots(1, 1,figsize=(10, 10))
        ax.imshow(rgb1)
        ax.plot(col, row, "--y",linewidth=7,color='red')
        if y!='y':
            y=input('do you want to confirm the indices? (y/n): ')

    crop1=rgb1[start[0]:end[0],start[1]:end[1],:]
    crop2=rgb2[start[0]:end[0],start[1]:end[1],:]
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(crop1)
    plt.subplot(1,2,2)
    plt.imshow(crop2)
    t1_crop=t1[start[0]:end[0],start[1]:end[1],:]
    t2_crop=t2[start[0]:end[0],start[1]:end[1],:]

    print('the datacube is now cropped to the new AOI')
    print('those are the new sizes:')
    print(t1_crop.shape)
    print(t2_crop.shape)
    
    if np.any(long) and np.any(lat):
        xcrop=long[start[0]:end[0],start[1]:end[1]]
        ycrop=lat[start[0]:end[0],start[1]:end[1]]
        return crop1,crop2, t1_crop, t2_crop,xcrop, ycrop
    
    else:
        return crop1,crop2, t1_crop, t2_crop
"""
def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read()
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
        
def Landsat_cube(path,ts1,te1,L_bands):
    """
    inputs: 
    path->path to folder containing Landsat acquisitions
    ts1->acquisition start time 
    te1->acquisition end time (can be found in the image title)
    L_bands->vector containing band's numbers
    """
    for b in L_bands:
        img=tiff.imread(path+ts1+"_"+te1+"_59_Landsat_8-9_L2_B"+b+"_(Raw).tiff")
        #matrix with index 0 is the image, while index 1 is the mask
        single_band = np.array(img[:,:,0])
        if b=='01':
            cube=single_band
        else:
            cube=np.dstack((cube,single_band))
    return cube

def Sentinel2_cube(path,ts1,te1,L_bands):
    """
    inputs: 
    path->path to folder containing Sentinel-3 acquisitions
    ts1->acquisition start time 
    te1->acquisition end time (can be found in the image title)
    L_bands->vector containing band's numbers
    """
    for b in L_bands:
        img=tiff.imread(path+ts1+"_"+te1+"_Sentinel-2_L2A_B"+b+"_(Raw).tiff")
        #matrix with index 0 is the image, while index 1 is the mask
        print('band '+b+': '+str(np.shape(img)))
        single_band = np.array(img[:,:])
        if b=='01':
            cube=single_band
        else:
            cube=np.dstack((cube,single_band))
    return cube

def RGB_Sentinel2(img,R,G,B):
    RGB=np.dstack((img[:,:,R],img[:,:,G]))
    RGB=np.dstack((RGB,img[:,:,B]))
    return RGB

def make_rgb_aerial(img):
    RGB=np.dstack((img[:,:,37],img[:,:,22]))
    RGB=np.dstack((RGB,img[:,:,12]))
    return RGB

def band_pairing(to_pair,aerial_bands,wl_prs,wl_prs_reduced):
    """
    void function, creates a copy of a given vector (to_pair) with a selection of bands paired witht the aerial ones
    INPUTS:
        to_pair: ndarray, 1-D array of aerial values that needs to be paired with PRISMA (size: number of aerial bands)
        aerial bands: ndarray, aerial wavelengths 1-D array 
        wl_prs: ndarray, PRISMA wavelengths 1-D array 
        wl_prs_reduced: ndarray, PRISMA wavelengths 1-D array with a cut at 2400nm to match aerial bandwidth
    OUTPUT:
        mean_array: ndarray, 1-D array of values averaged to match the number of PRISMA bands (size: number of PRISMA bands)
    """
    index_array=np.zeros(np.shape(aerial_bands)[0]).astype(int)
    for idx,band in enumerate(aerial_bands):
        index_array[idx]=find_nearest(wl_prs,band)
    previous_value=index_array[0]
    buffer=np.array([])
    mean_array=np.zeros(np.shape(wl_prs_reduced)[0]).astype(float)
    for idx1,value in enumerate(index_array):
        if (value==previous_value):
            buffer=np.append(buffer,to_pair[idx1])
        else:
            avg=np.mean(buffer)
            mean_array[previous_value]=avg
            buffer=np.array([])
            buffer=np.append(buffer,to_pair[idx1])
            if (idx1==((np.shape(index_array)[0])-1)):
                avg=np.mean(buffer)
                mean_array[value]=avg
        previous_value=value
    return mean_array



def compute_index(ref,index,wave):
    if index=='CLRE':
        b790=ref[:,:,find_nearest(wave,780)]
        b705=ref[:,:,find_nearest(wave,705)]
        index_map=np.divide(b790,b705)
        index_map=index_map**(-1)
    elif index=='GNDVI':
        b545=ref[:,:,find_nearest(wave,545)]
        NIR=ref[:,:,find_nearest(wave,800)]
        index_map=(NIR-b545)/(NIR+b545)
    elif index=='NBRI':
        SWIR1=ref[:,:,find_nearest(wave,1500)]
        SWIR2=ref[:,:,find_nearest(wave,1959)]
        SWIR3=ref[:,:,find_nearest(wave,2400)]
        SWIR=np.dstack((SWIR1,SWIR2))
        SWIR=np.dstack((SWIR,SWIR3))
        SWIR=np.mean(SWIR,axis=2)
        NIR=ref[:,:,find_nearest(wave,800)]
        index_map=(NIR-SWIR)/(NIR+SWIR)
    elif index=='NDREI':
        rededge=ref[:,:,find_nearest(wave,710)]
        NIR=ref[:,:,find_nearest(wave,800)]
        index_map=(NIR-rededge)/(NIR+rededge)
    elif index=='NDVI':
        NIR=ref[:,:,find_nearest(wave,800)]
        red=ref[:,:,find_nearest(wave,640)]
        index_map=(NIR-red)/(NIR+red)
    elif index=='NRVI':
        red=ref[:,:,find_nearest(wave,640)]
        NIR=ref[:,:,find_nearest(wave,800)]
        RVI=NIR/red
        index_map=(RVI-1)/(RVI+1)
    elif index=='REIP':
        b670=ref[:,:,find_nearest(wave,670)]
        b700=ref[:,:,find_nearest(wave,700)]
        b740=ref[:,:,find_nearest(wave,740)]
        b780=ref[:,:,find_nearest(wave,780)]
        num=((b670+b780)/2)-b700
        den=b740-b700
        var=num/den
        index_map=700+40*var
    elif index=='REIP2':
        b667=ref[:,:,find_nearest(wave,667)]
        b702=ref[:,:,find_nearest(wave,702)]
        b742=ref[:,:,find_nearest(wave,742)]
        b782=ref[:,:,find_nearest(wave,782)]
        num=((b667+b782)/2)-b702
        den=b742-b702
        var=num/den
        index_map=702+40*var
    elif index=='REIP3':
        b665=ref[:,:,find_nearest(wave,665)]
        b705=ref[:,:,find_nearest(wave,705)]
        b740=ref[:,:,find_nearest(wave,740)]
        b783=ref[:,:,find_nearest(wave,783)]
        num=((b665+b783)/2)-b705
        den=b740-b705
        var=num/den
        index_map=705+45*var
    elif index=='SLAVI':
        red=ref[:,:,find_nearest(wave,640)]
        NIR=ref[:,:,find_nearest(wave,800)]
        SWIR1=ref[:,:,find_nearest(wave,1500)]
        SWIR2=ref[:,:,find_nearest(wave,1959)]
        SWIR3=ref[:,:,find_nearest(wave,2400)]
        SWIR=np.dstack((SWIR1,SWIR2))
        SWIR=np.dstack((SWIR,SWIR3))
        SWIR=np.mean(SWIR,axis=2)
        index_map=(NIR)/(red+SWIR)
    else:
        raise Exception('Please, insert a valid vegetation index name!')
    return index_map