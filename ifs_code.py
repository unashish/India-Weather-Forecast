import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from ecmwf.opendata import Client
import numpy as np
import warnings
import time

# Suppress warnings for clean logs
warnings.filterwarnings("ignore")

# Use Azure for maximum reliability
client = Client(source="azure")

# --- CONFIGURATION ---
# India Bounding Box
lat_max, lat_min = 38.0, 6.0
lon_min, lon_max = 68.0, 98.0

# Path to your uploaded shapefile folder
SHAPEFILE_PATH = "India-Weather-Forecast/Admin2.shp"

# Time steps: 6h, 12h, 5 days, 10 days
steps = [6, 12, 120, 240]

# Variable Configuration: { "code": ["Display Name", "Colormap", "Unit", "Internal Key"] }
variables = {
    "2t": ["Temperature", "coolwarm", "°C", "t2m"],
    "tp": ["Total Precipitation", "YlGnBu", "mm", "tp"],
    "mucape": ["MUCAPE (Instability)", "inferno", "J/kg", "mucape"]
}

run_id = time.strftime("%Y%m%d_%H%M")

print(f"🚀 Launching India Intelligence Pipeline (Run ID: {run_id})")

# Load the custom shapefile once at the start
try:
    india_boundary = gpd.read_file(SHAPEFILE_PATH)
    print("✅ Custom Shapefile loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load shapefile at {SHAPEFILE_PATH}. Check folder name. Error: {e}")
    exit()

for var_code, info in variables.items():
    var_name, var_cmap, var_unit, data_key = info
    print(f"\nProcessing {var_name}...")
    
    for step in steps:
        target_grib = f"temp_{var_code}_{step}_{run_id}.grib"
        plot_img = f"plot_{var_code}_{step}.png"
        print(f"  - Step {step}h: ", end="", flush=True)
        
        try:
            # 1. Download
            client.retrieve(model="ifs", type="fc", param=var_code, step=step, target=target_grib)
            
            # 2. Load
            ds = xr.open_dataset(target_grib, engine="cfgrib")
            
            # Fix for scalar time
            raw_time = np.atleast_1d(ds.time.values)
            forecast_time = raw_time[0] + np.timedelta64(step, 'h')
            time_str = np.datetime_as_string(forecast_time, unit='h').replace('T', ' ')
            
            # 3. Slice Region
            data = ds[data_key].sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

            # Handle dimensions (squeeze 3D to 2D)
            if len(data.dims) > 2:
                other_dims = [d for d in data.dims if d not in ['latitude', 'longitude']]
                if other_dims:
                    data = data.isel({other_dims[0]: 0})

            # 4. Unit Conversions
            if var_code == "2t": 
                data = data - 273.15
            elif var_code == "tp": 
                data = data * 1000.0

            # 5. Plotting
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # Add Standard Map Features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#555555')
            
            # ADD YOUR CUSTOM SHAPEFILE BOUNDARY
            india_boundary.boundary.plot(ax=ax, color='black', linewidth=1.5)
            
            # Plot the Data
            data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=var_cmap, cbar_kwargs={'label': var_unit})
            
            plt.title(f"IFS {var_name}\n{time_str}", fontsize=14, fontweight='bold')
            plt.savefig(plot_img, dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Done")
            ds.close()

        except Exception as e:
            print(f"❌ FAILED: {e}")
        finally:
            # --- STRICT CLEANUP ---
            if os.path.exists(target_grib): os.remove(target_grib)
            if os.path.exists(target_grib + ".idx"): os.remove(target_grib + ".idx")
            # Clean all .idx files to keep repo clean
            for f in os.listdir('.'):
                if f.endswith('.idx'):
                    try: os.remove(f)
                    except: pass
        
        time.sleep(2)

print("\n🏁 All processes completed. Dashboard images are ready.")
