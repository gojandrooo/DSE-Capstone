#! /usr/bin/env python

try:
    import fastfuels

except ModuleNotFoundError:
    print("Fastfuels not found.  Please visit https://pypi.org/project/fastfuels/ for installation instructions.")

try:
    import xarray as xr
    import rioxarray as rxr
    import dask
    import rasterio
    import pandas as pd
    import geopandas as gpd
    from pyproj import Transformer

except ModuleNotFoundError:
    print("Some package pertaining to the PanGeos ecosystem where not found." )
    print("Make sure xarray, rioxarray, dask, rasterio, and geopandas are installed.")
    print("Consider running the API inside a Docker container provided by the repository.")


import requests
from datetime import datetime
from os.path import isfile

class LandFire:
    def __init__(self, bpath="/qumulo/fuel/landfire/COG/", bpath_type="local", 
                 cog_mask="%s/L%s/US_%s_cog.tif"):

        """ Initialize LandFire module in FuelTransferAPI.
            bpath: Path to Landfire rasters in cloud-optimized Geotiff (COG) format,
            either a local directory or remote (http) host, port and path
            bpath_type: can be "local" or "http"
            cog_mask: file naming convention of COG files in bpath. should not need to be changed.
        """

        self.versions={
            210: (2019,f"2017-2019"),
            200: (2016,f"2013-2017"),
            140: (2014,f"2011-2015"),
            130: (2012,f"2007-2011"),
            120: (2010,f"2007-2011"),
            110: (2008,f"1984-2008"),
            105: (2001,None)
        }
    
        assert bpath_type in ["local", "http"], "Unsupported path type"

        self.fuels={
            "CH": "Canopy Height",
            "CC": "Canopy Cover",
            "CBD": "Canopy Bulk Density",
            "CBH": "Canopy Base Height",
            "FBFM40": "Surface Fuel FBFM40",
            "FBFM13": "Surface Fuel FBFM40",
            "EVT": "Existing Vegetation Type",
            "Elev": "Elevation"}

        self.bpath=bpath
        self.bpath_type=bpath_type
        self.cog_mask=cog_mask

        self.check_for_lf_data()

    def get_cog_file_path(self, version, fuel):
        """Helper function to get COG file path for version and fuel. """
        return self.cog_mask % (self.bpath, version, fuel)

    def check_for_lf_data(self):
        """Checks for available Landfire data."""
        self.avail_cog={}
        for fuel in list(self.fuels.keys()):
            self.avail_cog[fuel]=[]
            for version in list(self.versions.keys()):
                cogfile=self.get_cog_file_path(version, fuel)
                #assert isfile(cogfile), f"{cogfile} not found."
                #print(f"Found {cogfile}")

                if self.bpath_type == "local":
                    if isfile(cogfile):
                        self.avail_cog[fuel].append(version)
                else:
                    response=requests.head(cogfile)
                    if  response.status_code == 200:
                        self.avail_cog[fuel].append(version)

    
    def read_dataset(self, verstr, fuel_type,
            chunks=None, masked=False):
        """Returns Xarray DataArray with Landfire data for given version and fuel type."""

        assert fuel_type in self.avail_cog.keys(), f"Fuel type {fuel_type} not found"

        assert verstr in self.avail_cog[fuel_type], f"Version {verstr} not found for fuel type {fuel_type}"

        src=rxr.open_rasterio(self.get_cog_file_path(verstr, fuel_type), band=1, chunks=chunks, masked=masked)

        """
        out={
            "DataArray" : src,
            "FuelType" : fuel_type,
            "FuelDesc" : self.attributes[fuel_type]
        }
        return out
        """

        # The API-specified should appear first in the attributes
        # dictionary (which is ordered as by Python3.7)
        # So I'm creating a new dictionary with the Landfire attributes and
        # copy the COG-specified ones into it.

        myattr={}

        myattr["DataSouce"] = "LandFire"
        myattr["DataVersion"] = "L%d (%d, %s)" % \
               (verstr, self.versions[verstr][0], self.versions[verstr][1])

        myattr["FuelType"] = fuel_type
        myattr["FuelDesc"] = self.fuels[fuel_type]

        if fuel_type == "CH":
            myattr["Units"] = "meters * 10"
        else:
            myattr["Units"] = "not yet implemented."

        for key in src.attrs.keys():
            myattr[key] = src.attrs[key]
        
        src.attrs = myattr
                                
        return src

    def query(self, asrc, lon, lat, radius, verbose=False):
        """Query data in Xarray object asrc for rectangular region around (lon,lat) within radius """
        llcenter=gpd.GeoDataFrame(
            {"geometry": gpd.points_from_xy([lon], [lat])}, 
            crs=asrc.rio.crs.from_epsg(4326))

        aecenter=llcenter.to_crs(asrc.rio.crs).loc[0,"geometry"]

        ffext = [aecenter.x - radius,  #minx
                 aecenter.y + radius,  #miny
                 aecenter.x + radius,  #maxx
                 aecenter.y - radius]  #maxy
        
        if verbose:
            print(ffext)

        regclip=asrc.rio.clip_box(ffext[0], ffext[3], ffext[2], ffext[1]).compute()

        return regclip

    def select(self, asrc, lon, lat, method="nearest"):
        """Query data in Xarray object asrc near single point (lon, lat) """
        transformer = Transformer.from_crs("EPSG:4326", asrc.rio.crs, always_xy=True)
        xx, yy = transformer.transform(lon, lat)
        out=asrc.sel(x=xx, y=yy, method=method).compute().values[0]
        return out
        
    def query_all_versions(self, fueltype, lon, lat, radius):
        """Query data for fueltype from all available LandFire versions """
        out={}
        for ver in self.avail_cog[fueltype]:
            tmp=self.read_dataset(ver, fueltype)
            out[ver] = self.query(tmp, lon, lat, radius)
        return out

    def select_all_versions(self, fueltype, lon, lat, method="nearest"):
        """Select data for fueltype from all available LandFire versions """
        out={}
        for ver in self.avail_cog[fueltype]:
            tmp=self.read_dataset(ver, fueltype)
            out[ver] = self.select(tmp, lon, lat, method=method)
        return out

    def get_fbfm40_attributes(self, datatype):
        """Return dictionary attributes (shorthand and description) for given data type.
           datatype='FBFM40': Fire Behaviour Fuel Model 40"""

        assert datatype == 'FBFM40', 'datatype %s not (yet) implemented' % datatype

        abbrvs={}
        desc={}

        if self.bpath_type == "local":
            from os.path import dirname, isfile
            attrdir=dirname(__file__) + "/attributes/"
            attrfile=attrdir + "landfire_fbfm40_attributes.txt"
            descfile=attrdir + "landfire_fbfm40_display_attributes.txt"

            assert isfile(attrfile), attrfile + " not found"
            assert isfile(descfile), descfile + " not found"

            with open(attrfile, "r") as fid:
                for nl in fid.readlines():
                    if nl[0] != "#":
                        num=int(nl.split(":")[0])
                        abbrvs[num]=nl.split(":")[1].rstrip()
            
            with open(descfile, "r") as fid:
                for nl in fid.readlines():
                    if nl[0] != "#":
                        num=nl.split(":")[0]
                        desc[num]=nl.split(":")[1].rstrip()
        
        else:
            attrurl=self.bpath + "attributes/landfire_fbfm40_attributes.txt"
            descurl=self.bpath + "attributes/landfire_fbfm40_display_attributes.txt"

            attrresp=requests.head(attrurl).status_code
            descresp=requests.head(descurl).status_code

            assert attrresp == 200, attrurl + " not found, error code " + str(attrresp)
            assert descresp == 200, descurl + "not found, error code " + str(descresp)

            with requests.get(attrurl) as response:
                for nl in response.content.split(b"\n"):
                    nls=nl.decode('utf-8')
                    if len(nls) > 0:
                        if nls[0] != "#":
                            num=int(nls.split(":")[0])
                            abbrvs[num]=nls.split(":")[1].rstrip()
            
            with requests.get(descurl) as response:
                for nl in response.content.split(b"\n"):
                    nls=nl.decode('utf-8')
                    if len(nls) > 0:
                        if nl[0] != "#":
                            num=nls.split(":")[0]
                            desc[num]=nls.split(":")[1].rstrip()   
        odict={num: (abbrvs[num], desc[ab]) for num,ab in abbrvs.items()}
        return odict

class SALO:
    def __init__(self):
        self.fuels={"SurfaceFuels": "A model describing vegetation fuel type, size class, depth, moisture content and heat content.", 
            "LadderFuelDensity": "The amount of brush and/or understory fuels.", 
            "CanopyLayerCount": "The number of distinct vertical canopy layers",
            "CanopyBulkDensity": "The mass of available fuel that burns in a canopy fire - typically leaves and branches - divided by the volume of the crown",
            "CanopyBaseHeight": "The distance between the ground and the lowest branches in the canopy.",
            "CanopyHeight": "The distance between the ground and the top of the canopy.",
            "CanopyCover": "The horizontal area covered by tree canopies."}

        self.urlmask = "https://storage.googleapis.com/cfo-public/vegetation/California-Vegetation-%s-%04d-%s-%05dm.tif"

    def geturl(self, fuel, year, season, resol):
        return self.urlmask % (fuel, year, season, resol) 
        
    def check_data_availability(self, verbose=False):

        self.salosets=[]
        
        for fuel in self.fuels.keys():
            print(fuel)
            resol=10
            season="Summer"
            for year in range(2016, datetime.now().year+1):
                url=self.geturl(fuel, year, season, resol) 
                
                response=requests.head(url)

                if response.status_code == 200:
                    self.salosets.append({"fueltype": fuel,
                                 "year": year,
                                 "season": season,
                                 "resolution": resol,
                                 "url": url})
                else:
                    if (verbose):
                        print(f"{url} not found: {response.status_code}")

            #check for 3m resolution data, only available for selected years and spring/fall
            resol=3
            for year in [2016,2020]:
                for season in ["Spring", "Fall"]:
                    url=self.geturl(fuel, year, season, resol)

                    response=requests.head(url)
                    if response.status_code == 200:
                        self.salosets.append({"fueltype": fuel,
                                 "year": year,
                                 "season": season,
                                 "resolution": resol,
                                 "url": url})
                    else:
                        if (verbose):
                            print(f"{url} not found: {response.status_code}")

    def get_available_data(self):
        return pd.DataFrame(self.salosets)

class WifireFT:
    def __init__(self):
        fuel_sources={"Fastfuels": [{"version": "v1"}], 
            "Landfire": [{"Type": None}]}
