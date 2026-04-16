import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ecmwf.opendata import Client
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")
client = Client(source="azure")

# --- CONFIGURATION ---
steps = [6, 12, 120, 240]
lat_max, lat_min = 38.0, 6.0
lon_min, lon_max = 68.0, 98.0

variables = {
    "2t": ["Temperature", "coolwarm", "°C", "t2m"],
    "tp": ["Total Precipitation", "YlGnBu", "mm", "tp"],
    "mucape": ["MUCAPE (Instability)", "inferno", "J/kg", "mucape"]
}

print(f"🚀 Launching India Pipeline (Static Filenames Mode)")

for var_code, info in variables.items():
    var_name, var_cmap, var_unit, data_key = info
    print(f"\nProcessing {var_name}...")
    
    for step in steps:
        # FILENAMES ARE NOW STATIC (No timestamp)
        target_grib = f"temp_{var_code}_{step}.grib"
        plot_img = f"plot_{var_code}_{step}.png"
        
        print(f"  - Step {step}h: ", end="", flush=True)
        
        try:
            client.retrieve(model="ifs", type="fc", param=var_code, step=step, target=target_grib)
            ds = xr.open_dataset(target_grib, engine="cfgrib")
            
            raw_time = np.atleast_1d(ds.time.values)
            time_str = np.datetime_as_string(raw_time[0] + np.timedelta64(step, 'h'), unit='h').replace('T', ' ')
            
            data = ds[data_key].sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
            if len(data.dims) > 2:
                other_dims = [d for d in data.dims if d not in ['latitude', 'longitude']]
                if other_dims: data = data.isel({other_dims[0]: 0})

            if var_code == "2t": data = data - 273.15
            elif var_code == "tp": data = data * 1000.0

            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
            ax.add_feature(cfeature.BORDERS, linewidth=1.0)
            data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=var_cmap, cbar_kwargs={'label': var_unit})
            plt.title(f"IFS {var_name}\n{time_str}", fontsize=14, fontweight='bold')
            plt.savefig(plot_img, dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Done")
            ds.close()

        except Exception as e:
            print(f"❌ FAILED: {e}")
        finally:
            if os.path.exists(target_grib): os.remove(target_grib)
            if os.path.exists(target_grib + ".idx"): os.remove(target_grib + ".idx")
            for f in os.listdir('.'):
                if f.endswith('.idx'):
                    try: os.remove(f)
                    except: pass
        time.sleep(2)

print("\n🏁 All processes completed.")
