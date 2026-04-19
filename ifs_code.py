import json
import os
import signal
import time
import warnings
from datetime import datetime, timezone

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ecmwf.opendata import Client

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
lat_max, lat_min = 38.0, 6.0
lon_min, lon_max = 68.0, 98.0
SHAPEFILE_PATH = "Admin2.shp"
STEPS = [6, 12, 120, 240]
MAX_RETRIES = int(os.getenv("IFS_MAX_RETRIES", "3"))
RETRY_SLEEP_SECONDS = int(os.getenv("IFS_RETRY_SLEEP_SECONDS", "5"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("IFS_DOWNLOAD_TIMEOUT_SECONDS", "90"))

# Added additional forecast fields for a richer dashboard.
VARIABLES = {
    "2t": ["Temperature", "coolwarm", "°C", "t2m"],
    "tp": ["Total Precipitation", "YlGnBu", "mm", "tp"],
    "mucape": ["MUCAPE (Instability)", "inferno", "J/kg", "mucape"],
    "10si": ["10m Wind Speed", "viridis", "m/s", "si10"],
    "2r": ["2m Relative Humidity", "BrBG", "%", "r2"],
    "msl": ["Mean Sea-Level Pressure", "plasma", "hPa", "msl"],
}

SELECTED_STEPS = [
    int(part.strip())
    for part in os.getenv("IFS_STEPS", ",".join(str(step) for step in STEPS)).split(",")
    if part.strip()
]
SELECTED_VARIABLES = [
    part.strip() for part in os.getenv("IFS_VARIABLES", ",".join(VARIABLES.keys())).split(",") if part.strip()
]


def init_client() -> tuple[Client, str]:
    """Try Azure first, then fall back to ECMWF direct source."""
    sources = ["azure", "ecmwf"]
    last_error = None

    for source in sources:
        try:
            client = Client(source=source)
            return client, source
        except Exception as exc:
            last_error = exc
            print(f"⚠️ Client source '{source}' failed: {exc}")

    raise RuntimeError(f"Could not initialize any client source. Last error: {last_error}")


def retrieve_with_retries(client: Client, var_code: str, step: int, target_grib: str) -> None:
    """Download GRIB with retries for transient failures."""
    def _timeout_handler(_signum, _frame):
        raise TimeoutError(f"retrieve timeout after {DOWNLOAD_TIMEOUT_SECONDS}s")

    attempt = 1
    while attempt <= MAX_RETRIES:
        previous_handler = None
        try:
            previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(DOWNLOAD_TIMEOUT_SECONDS)
            client.retrieve(model="ifs", type="fc", param=var_code, step=step, target=target_grib)
            return
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Download failed after {MAX_RETRIES} attempts: {exc}") from exc
            print(f"    Retry {attempt}/{MAX_RETRIES - 1} after download error: {exc}")
            attempt += 1
            time.sleep(RETRY_SLEEP_SECONDS)
        finally:
            try:
                signal.alarm(0)
            except Exception:
                pass
            if previous_handler is not None:
                try:
                    signal.signal(signal.SIGALRM, previous_handler)
                except Exception:
                    pass


def cleanup_temp_files(target_grib: str) -> None:
    if os.path.exists(target_grib):
        os.remove(target_grib)
    if os.path.exists(target_grib + ".idx"):
        os.remove(target_grib + ".idx")
    for filename in os.listdir("."):
        if filename.endswith(".idx"):
            try:
                os.remove(filename)
            except OSError:
                pass


def process_variable_step(
    client: Client,
    india_map: gpd.GeoDataFrame,
    var_code: str,
    var_name: str,
    var_cmap: str,
    var_unit: str,
    data_key: str,
    step: int,
    run_id: str,
) -> tuple[bool, str]:
    target_grib = f"temp_{var_code}_{step}_{run_id}.grib"
    plot_img = f"plot_{var_code}_{step}.png"

    try:
        retrieve_with_retries(client, var_code, step, target_grib)
        ds = xr.open_dataset(target_grib, engine="cfgrib")

        raw_time = np.atleast_1d(ds.time.values)
        forecast_time = raw_time[0] + np.timedelta64(step, "h")
        time_str = np.datetime_as_string(forecast_time, unit="h").replace("T", " ")

        data = ds[data_key].sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).squeeze()

        if len(data.dims) > 2:
            other_dims = [dimension for dimension in data.dims if dimension not in ["latitude", "longitude"]]
            if other_dims:
                data = data.isel({other_dims[0]: 0})

        if var_code == "2t":
            data = data - 273.15
        elif var_code == "tp":
            data = data * 1000.0
        elif var_code == "msl":
            data = data / 100.0

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        india_map.boundary.plot(ax=ax, color="black", linewidth=1.5)

        data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=var_cmap, cbar_kwargs={"label": var_unit})

        plt.title(f"IFS {var_name}\n{time_str}", fontsize=14, fontweight="bold")
        plt.savefig(plot_img, dpi=150, bbox_inches="tight")
        plt.close()
        ds.close()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)
    finally:
        cleanup_temp_files(target_grib)


def write_run_metadata(status: str, source: str, run_id: str, summary: dict) -> None:
    payload = {
        "status": status,
        "run_id": run_id,
        "source": source,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps": SELECTED_STEPS,
        "variables": SELECTED_VARIABLES,
        "summary": summary,
    }

    with open("latest_run.json", "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


if __name__ == "__main__":
    run_id = time.strftime("%Y%m%d_%H%M")
    print(f"🚀 Starting India Intelligence Pipeline (Run ID: {run_id})")

    try:
        client, source = init_client()
        print(f"✅ Data client initialized via source: {source}")
    except Exception as exc:
        print(f"❌ ERROR: client initialization failed: {exc}")
        write_run_metadata(
            status="failed",
            source="unavailable",
            run_id=run_id,
            summary={"error": f"client_init_failed: {exc}"},
        )
        raise SystemExit(1)

    try:
        india_map = gpd.read_file(SHAPEFILE_PATH)
        print(f"✅ Shapefile '{SHAPEFILE_PATH}' loaded successfully.")
    except Exception as exc:
        print(
            f"❌ ERROR: Could not find {SHAPEFILE_PATH}. Make sure .shp, .shx, .dbf, .prj are in the main folder."
        )
        write_run_metadata(
            status="failed",
            source=source,
            run_id=run_id,
            summary={"error": f"shapefile_load_failed: {exc}"},
        )
        raise SystemExit(1)

    invalid_variables = [var for var in SELECTED_VARIABLES if var not in VARIABLES]
    if invalid_variables:
        print(f"❌ ERROR: unknown variables requested via IFS_VARIABLES: {invalid_variables}")
        write_run_metadata(
            status="failed",
            source=source,
            run_id=run_id,
            summary={"error": f"invalid_variables: {invalid_variables}"},
        )
        raise SystemExit(1)

    summary: dict[str, dict[str, str]] = {}
    total_ok = 0
    total_failed = 0

    for var_code in SELECTED_VARIABLES:
        info = VARIABLES[var_code]
        var_name, var_cmap, var_unit, data_key = info
        print(f"\nProcessing {var_name} ({var_code})...")

        summary[var_code] = {}
        for step in SELECTED_STEPS:
            print(f"  - Step {step}h: ", end="", flush=True)
            ok, message = process_variable_step(
                client=client,
                india_map=india_map,
                var_code=var_code,
                var_name=var_name,
                var_cmap=var_cmap,
                var_unit=var_unit,
                data_key=data_key,
                step=step,
                run_id=run_id,
            )

            if ok:
                total_ok += 1
                summary[var_code][str(step)] = "ok"
                print("✅ Done")
            else:
                total_failed += 1
                summary[var_code][str(step)] = f"failed: {message}"
                print(f"❌ FAILED: {message}")

            time.sleep(2)

    status = "success" if total_failed == 0 else "partial_success"
    write_run_metadata(status=status, source=source, run_id=run_id, summary=summary)

    print("\n🏁 All processes completed.")
    print(f"   Successful tasks: {total_ok}")
    print(f"   Failed tasks: {total_failed}")
    print("   Metadata written to latest_run.json")
