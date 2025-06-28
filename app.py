from flask import Flask, send_from_directory, jsonify, render_template, request
import os
import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_bounds

app = Flask(__name__)

DATA_PATH = 'data/rsmc_coast_ww3_20250626.nc'
TILE_DIR = 'static/tiles'
GEO_TIFF_FILENAME = 'wind_.tif'

# 🔍 Auto-detect coordinate names
def find_lat_lon_coords(ds):
    coords = list(ds.coords)
    lat_keys = ['lat', 'latitude', 'y', 'LAT', 'Y', 'IOYAXIS']
    lon_keys = ['lon', 'longitude', 'x', 'LON', 'X', 'IOXAXIS']
    for lat in lat_keys:
        for lon in lon_keys:
            if lat in coords and lon in coords:
                return lat, lon
    raise ValueError("Latitude or Longitude coordinate not found in dataset")

# 🗺️ Generate GeoTIFF with orientation fix
def generate_geotiff():
    try:
        os.makedirs(TILE_DIR, exist_ok=True)
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"NetCDF file not found at: {DATA_PATH}")

        ds = xr.open_dataset(DATA_PATH)
        variable = list(ds.data_vars)[0]
        time_dim = next((dim for dim in ds[variable].dims if 'time' in dim.lower()), None)
        data = ds[variable].isel({time_dim: 0}).values if time_dim else ds[variable].values

        lat_name, lon_name = find_lat_lon_coords(ds)
        lats = ds[lat_name].values
        lons = ds[lon_name].values

        if lats[0] < lats[-1]:
            lats = lats[::-1]
            data = data[::-1, :]
        if lons[0] > lons[-1]:
            lons = lons[::-1]
            data = data[:, ::-1]

        transform = from_bounds(lons.min(), lats.min(), lons.max(), lats.max(), data.shape[1], data.shape[0])
        geotiff_path = os.path.join(TILE_DIR, GEO_TIFF_FILENAME)
        with rasterio.open(geotiff_path, 'w', driver='GTiff',
                           height=data.shape[0], width=data.shape[1],
                           count=1, dtype='float32',
                           crs='EPSG:4326', transform=transform, nodata=np.nan) as dst:
            dst.write(data.astype('float32'), 1)

        ds.close()
        print(f"✅ GeoTIFF generated at: {geotiff_path}")
    except Exception as e:
        print(f"❌ Error generating GeoTIFF: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def serve_point_data():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))

        if not os.path.exists(DATA_PATH):
            return jsonify({"error": "Dataset file not found"}), 404

        ds = xr.open_dataset(DATA_PATH)
        lat_name, lon_name = find_lat_lon_coords(ds)
        lats = ds[lat_name].values
        lons = ds[lon_name].values

        # Flip lat/lon to match GeoTIFF and map
        if lats[0] < lats[-1]:
            lats = lats[::-1]
        if lons[0] > lons[-1]:
            lons = lons[::-1]

        if not (lats.min() <= lat <= lats.max()) or not (lons.min() <= lon <= lons.max()):
            ds.close()
            return jsonify({"error": "Requested lat/lon is outside dataset bounds"}), 400

        time_coord = next((c for c in ds.coords if 'time' in c.lower()), None)
        result = {}
        if time_coord:
            result["time"] = [str(np.datetime64(t)).replace('T', ' ') for t in ds[time_coord].values]
        else:
            result["time"] = []

        for var in ds.data_vars:
            try:
                sel_data = ds[var].sel({lat_name: lat, lon_name: lon}, method='nearest')
                values = sel_data.values.tolist() if time_coord and time_coord in sel_data.dims else [sel_data.values.item()]
                result[var] = values
            except Exception as e:
                result[var] = f"Error: {str(e)}"

        ds.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/wind')
def serve_wind_json():
    try:
        ds = xr.open_dataset(DATA_PATH)
        lat_name, lon_name = find_lat_lon_coords(ds)

        u_var = next((v for v in ds.data_vars if 'uwnd' in v.lower()), None)
        v_var = next((v for v in ds.data_vars if 'vwnd' in v.lower()), None)
        if not u_var or not v_var:
            raise ValueError("UWND or VWND variables not found")

        u_data = ds[u_var].isel({dim: 0 for dim in ds[u_var].dims if 'time' in dim.lower()}).values
        v_data = ds[v_var].isel({dim: 0 for dim in ds[v_var].dims if 'time' in dim.lower()}).values
        lats = ds[lat_name].values
        lons = ds[lon_name].values

        # Flip to match raster orientation
        if lats[0] < lats[-1]:
            lats = lats[::-1]
            u_data = u_data[::-1, :]
            v_data = v_data[::-1, :]
        if lons[0] > lons[-1]:
            lons = lons[::-1]
            u_data = u_data[:, ::-1]
            v_data = v_data[:, ::-1]

        wind_speed = np.sqrt(u_data**2 + v_data**2)
        wind_min = float(np.nanmin(wind_speed))
        wind_max = float(np.nanmax(wind_speed))

        def tolist_safe(arr):
            return [[float(x) if not np.isnan(x) else None for x in row] for row in arr]

        result = {
            "u_wind": tolist_safe(u_data),
            "v_wind": tolist_safe(v_data),
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "min": wind_min,
            "max": wind_max,
            "shape": {
                "u_wind": np.shape(u_data),
                "v_wind": np.shape(v_data)
            }
        }
        ds.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/tiles/<filename>')
def serve_tile(filename):
    return send_from_directory(TILE_DIR, filename)

if __name__ == '__main__':
    generate_geotiff()
    app.run(debug=True)
