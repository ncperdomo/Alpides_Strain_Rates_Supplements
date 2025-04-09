import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# WGS84 Ellipsoid Constants
a = 6378137.0  # Semi-major axis in meters
e = 0.0818191908426  # Eccentricity

# ----------------------------
# Helper Functions for Coordinate Conversion
# ----------------------------

def llh2local(llh, origin):
    """
    Converts from longitude and latitude to local coordinates (x, y) in kilometers.
    """
    # Convert degrees to radians
    llh = np.radians(llh)
    origin = np.radians(origin)
    
    # Calculate differences in latitudes and longitudes
    dlat = llh[1, :] - origin[1]
    dlon = llh[0, :] - origin[0]
    
    # Ensure longitude differences are within [-pi, pi]
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    
    # Compute average latitude and its sine/cosine
    lat_avg = (llh[1, :] + origin[1]) / 2
    lat_avg_sin = np.sin(lat_avg)
    lat_avg_cos = np.cos(lat_avg)
    
    # Meridian radius of curvature (M)
    M = a * (1 - e**2) / (np.sqrt(1 - e**2 * lat_avg_sin**2))**3
    
    # Prime vertical radius of curvature (N)
    N = a / np.sqrt(1 - e**2 * lat_avg_sin**2)
    
    # Compute local coordinates in meters
    x = dlon * N * lat_avg_cos  # Easting
    y = dlat * M                # Northing
    
    # Convert to kilometers
    x_km = x / 1000
    y_km = y / 1000
    
    return np.vstack([x_km, y_km])


def local2llh(xy, origin):
    """
    Converts from local coordinates to longitude and latitude.
    """
    # Convert origin to radians
    origin = np.radians(origin)
    
    # Unpack local coordinates and convert to meters
    x = xy[0, :] * 1000  # Easting in meters
    y = xy[1, :] * 1000  # Northing in meters
    
    # Initial guess for latitude: use origin latitude
    lat = origin[1]
    tol = 1e-10
    max_iter = 10
    iter_count = 0
    diff = np.inf
    
    # Iteratively solve for latitude
    while diff > tol and iter_count < max_iter:
        lat_old = lat
        lat_avg = (lat + origin[1]) / 2
        lat_avg_sin = np.sin(lat_avg)
        
        # Meridian radius of curvature (M)
        M = a * (1 - e**2) / (np.sqrt(1 - e**2 * lat_avg_sin**2))**3
        
        # Update latitude using the y local coordinate
        lat = origin[1] + y / M
        diff = np.max(np.abs(lat - lat_old))
        iter_count += 1
    
    # Compute longitude using updated latitude
    lat_avg = (lat + origin[1]) / 2
    N = a / np.sqrt(1 - e**2 * np.sin(lat_avg)**2)
    lon = origin[0] + x / (N * np.cos(lat_avg))
    
    # Convert longitude to [-180, 180] degrees
    lon = (lon + np.pi) % (2 * np.pi) - np.pi
    
    # Convert back to degrees
    lon = np.degrees(lon)
    lat = np.degrees(lat)
    
    return np.vstack([lon, lat])