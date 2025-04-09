# utils.py
# This is a collection of functions I've written during the past 2 years. I've tried to document them as much as possible.
# However, If you find any bugs or have any questions, please feel free to contact me at: jcastrop [at] iu [dot] edu 

import numpy as np
import matplotlib.pyplot as plt
import pygmt
import glob
import os
import pandas as pd
from shapely.geometry import Polygon, Point
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from input_data.modules.coord_conv import llh2local, local2llh
from collections import defaultdict
from input_data.modules.euler_pole import EulerPole
import xarray as xr
import shapely.geometry
import shapely.vectorized

# Constants
RADIUS_EARTH_KM = 6371.0  # Radius of Earth in kilometers

########################################################################################################################
########################################################################################################################
# Function to calculate the Haversine distance between two points
def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return RADIUS_EARTH_KM * c
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to get grid key for a location (rounded by grid size)
def get_grid_key(lon, lat, grid_size):
    return (round(lon / grid_size), round(lat / grid_size))
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to find nearby stations
def find_nearby_stations(strain_lon, strain_lat, grid_size, radius_threshold_km, stress_dict):
    nearby_stress_data = []
    # Get grid key for strain observation
    strain_grid_key = get_grid_key(strain_lon, strain_lat, grid_size)
    
    # Check neighboring grid cells including the current cell
    for dlon in [-1, 0, 1]:
        for dlat in [-1, 0, 1]:
            neighbor_key = (strain_grid_key[0] + dlon, strain_grid_key[1] + dlat)
            if neighbor_key in stress_dict:
                for stress_row in stress_dict[neighbor_key]:
                    distance = haversine_distance(strain_lon, strain_lat, stress_row['lon'], stress_row['lat'])
                    if distance <= radius_threshold_km:
                        nearby_stress_data.append(stress_row['azim'])
    
    return nearby_stress_data
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Create a continuous CPT file from Matplotlib's colormaps
def create_cpt_from_colormap(cmap_name, filename, num_colors=10, vmin=-90, vmax=90):
    """
    Create a continuous CPT file from a Matplotlib colormap and save it as a GMT-compatible CPT file.
    
    Parameters:
    - cmap_name: Name of the Matplotlib colormap (e.g., 'coolwarm')
    - filename: Name of the output CPT file (e.g., 'azimuth_diff.cpt')
    - num_colors: Number of color stops to define (fewer stops for continuous interpolation)
    - vmin: Minimum value for the CPT
    - vmax: Maximum value for the CPT
    """
    cmap = plt.get_cmap(cmap_name)
    values = np.linspace(vmin, vmax, num_colors)

    with open(filename, 'w') as f:
        for i, value in enumerate(values[:-1]):
            rgba_start = cmap(i / (num_colors - 1))
            rgba_end = cmap((i + 1) / (num_colors - 1))
            f.write(f"{value:.2f} {int(rgba_start[0]*255)} {int(rgba_start[1]*255)} {int(rgba_start[2]*255)} "
                    f"{values[i+1]:.2f} {int(rgba_end[0]*255)} {int(rgba_end[1]*255)} {int(rgba_end[2]*255)}\n")
        f.write(f"B 0 0 255\n")  # Background color
        f.write(f"F 255 255 255\n")  # Foreground color
        f.write(f"N 128 128 128\n")  # NaN color (gray)

########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to map azimuth differences to hex colors using the colormap
def get_hex_color(value, min_value=-90, max_value=90):
    # Create a colormap using matplotlib's colormap (for azimuth differences between -90 and 90 degrees)
    colormap = plt.get_cmap("twilight_shifted", 180)
    """Get hex color for a value based on a colormap range."""
    # Normalise the value to be between 0 and 1
    normalised_value = (value - min_value) / (max_value - min_value)
    # Get the RGBA color from the colormap
    rgba_color = colormap(normalised_value)
    # Convert RGBA to hex
    hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))
    return hex_color
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to plot GNSS velocity fields on a map
def plot_gps_velocity_fields(folder_path, figure_folder, colorbar_width=8.5, save_fig=True):
    # Find all CSV files in the output_coherence_analysis folder
    file_names = glob.glob(os.path.join(folder_path, '*.csv'))

    # Create a new figure for each file
    for file_name in file_names:
        # Create a new figure
        fig = pygmt.Figure()

        # Set the region and projection of the map
        #fig.basemap(region=[-15, 70, 5, 60], projection='M10c', frame='afg') # Only Euromediterranean and Middle East regions
        fig.basemap(region=[-20, 125, 5, 60], projection='M20c', frame='af') # Entire Alpine-Himalayan belt

        # Create a custom color palette for the relief shading
        #pygmt.makecpt(cmap="gray95,gray90,gray85", series=[-10000, 10000, 100])
        pygmt.makecpt(cmap="geo", series=[-8000, 8000, 100], output="input_data/cpts/alpides_topo.cpt")

        # Here I fixed an aesthetic issue when plotting the color palette for on-land topography values under the sea level (like the region around the Caspian Sea)
        # Read the CPT file
        with open("input_data/cpts/alpides_topo.cpt", "r") as file:
            lines = file.readlines()

        # Find the color corresponding to the interval [0, 100]
        for line in lines:
            if line.startswith("0\t0/97/71"):
                color_for_zero_to_100 = line.split()[1]
                break

        # Initialise a flag to modify values from -500 to 0 once
        new_lines = []

        # Modify the lines with elevations between -500 and 0 to match 0-100 color
        for line in lines:
            parts = line.split("\t")
            
            # Identify lines with elevations between -500 and 0
            if len(parts) > 2 and int(parts[0]) >= -500 and int(parts[2]) <= 0:
                if parts[1] != color_for_zero_to_100:  # Apply change only if it's not already modified
                    # Replace the color for negative elevations with the color for 0 to 100
                    parts[1] = color_for_zero_to_100
                    parts[3] = color_for_zero_to_100
                new_lines.append("\t".join(parts))
            else:
                new_lines.append(line)

        # Write the modified CPT file
        with open("input_data/cpts/alpides_topo_fixed.cpt", "w") as file:
            file.writelines(new_lines)

        pygmt.makecpt(cmap="input_data/cpts/alpides_topo_fixed.cpt", output="input_data/cpts/land_cbar.cpt", transparency=70, background='o', truncate=[0, 6000])

        # Add shaded topography with transparency
        fig.grdimage(grid="@earth_relief_01m", cmap="input_data/cpts/alpides_topo_fixed.cpt", shading=True, transparency=70) # nan_transparent=True results in error in the latest GMT version

        # Add coastlines
        fig.coast(water='white', shorelines="0.1p,black", area_thresh=4000, resolution='h')

        # Read the CSV file from output_coherence_analysis folder
        #df = pd.read_csv(file_name, sep=r'\s+', skiprows=1, header=None)
        #df.columns = ['Lon', 'Lat', 'E.vel', 'N.vel', 'E.adj', 'N.adj', 'E.sig', 'N.sig', 'Corr', 'U.vel', 'U.adj', 'U.sig', 'Stat']
        usecols = ['Lon', 'Lat', 'E.vel', 'N.vel', 'E.adj', 'N.adj', 'E.sig', 'N.sig', 'Corr', 'U.vel', 'U.adj', 'U.sig', 'Stat'] # I do this to ignore potential extra columns in the CSV file
        df = pd.read_csv(file_name, sep=r'\s+', skiprows=1, header=None, usecols=range(len(usecols)))
        df.columns = usecols

        # Check if the data frame is empty
        if df.shape[0] == 0:
            print(f"Skipping empty file: {file_name}")
            continue
        
        # Extract the coordinates, E and N velocity components, and E and N sig from the data frame
        lon = df['Lon']
        lat = df['Lat']
        e_vel = df['E.vel']
        n_vel = df['N.vel']
        e_sig = df['E.sig']
        n_sig = df['N.sig']

        # Calculate the velocity magnitude for scaling
        vel_mag = np.sqrt(e_vel**2 + n_vel**2)

        # Normalise the velocity magnitude to the range [0, 1]
        normalised_vel_mag = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min())

        # Create a list to store the vectors
        vectors = []

        # Iterate over each site
        for i in range(len(df)):
            x_start = lon[i]
            y_start = lat[i]
            direction_degrees = np.degrees(np.arctan2(n_vel[i], e_vel[i]))
            length = normalised_vel_mag[i] #* 0.5

            # Add the vector to the list
            vectors.append([x_start, y_start, direction_degrees, length])

        # Plot the GPS velocity vectors from output_coherence_analysis folder (blue)
        fig.plot(
            style='v0.1c+e+n0.15',
            data=vectors,
            fill='red',
            pen='black',
            label='Accepted vel.',
        )

        # Add scale bar
        with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.basemap(map_scale="JBR+o-9c/-0.8c+c0+w1000k+f+lkm")

        # Add scale vectors to the plot
        # Define origin of the scale vectors
        scale_origin_lon = 68  # Longitude of the scale vector origin
        scale_origin_lat = 16  # Latitude of the scale vector origin

        # Define scale vectors (30 mm/yr normalised appropriately)
        scale_vector_length = 30  # in mm/yr, this will be normalised

        # Calculate the normalised length for the scale vectors
        normalised_scale_length = (scale_vector_length - vel_mag.min()) / (vel_mag.max() - vel_mag.min())

        scale_vectors = [
            [scale_origin_lon, scale_origin_lat, 0, normalised_scale_length],  # Eastward vector
            [scale_origin_lon, scale_origin_lat, 90, normalised_scale_length]  # Northward vector
        ]

        # Plot the scale vectors
        fig.plot(
            style='v0.1c+e+n0.15',
            data=scale_vectors,
            fill='red',
            pen='black',
            label='Accepted vel.',
        )

        # Annotate the scale vectors
        fig.text(
            text=f'{scale_vector_length} mm/yr',
            x=scale_origin_lon - 5,
            y=scale_origin_lat,
            font='7p,black'
        )

        # Add color bar
        fig.colorbar(frame="af+lElevation (m)", position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef", cmap="input_data/cpts/land_cbar.cpt")

        # Get the base file name without extension
        base_name = os.path.splitext(os.path.basename(file_name))[0]

        # Change text to be printed based on base_name
        if 'eura' in base_name:
            display_name = 'Eurasia-fixed'
        elif 'nubi' in base_name:
            display_name = 'Nubia-fixed'
        elif 'indi' in base_name:
            display_name = 'India-fixed'
        elif 'arab' in base_name:
            display_name = 'Arabia-fixed'
        elif 'yang' in base_name:
            display_name = 'Yangtze-fixed'
        elif 'sina' in base_name:
            display_name = 'Sinai-fixed'
        elif 'aege' in base_name:
            display_name = 'Aegean Sea-fixed'
        elif 'anat' in base_name:
            display_name = 'East Anatolia-fixed'
        elif 'igb' in base_name:
            display_name = 'IGB14'
        elif 'myan' in base_name:
            display_name = 'Myanmar-fixed'
        elif 'amur' in base_name:
            display_name = 'Amuria-fixed'
        elif 'tbet' in base_name:
            display_name = 'Central Tibet-fixed'
        else:
            display_name = "Unknown"
        
        print(f"Plotting GNSS velocities in {display_name} reference frame:")

        # Show the figure
        fig.show()

        # Create a directory to store the figure files
        os.makedirs(figure_folder, exist_ok=True)

        # Save the figure
        if save_fig:
            figure_file_pdf = os.path.join(figure_folder, f'Alpides_{base_name}_map.pdf')
            fig.savefig(figure_file_pdf, dpi=300)
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to plot vertical GNSS velocities as a scatter plot on a map
def plot_vertical_velocity_fields(input_files, figure_folder, fault_file=None, save_fig=True):

    # Create a new figure
    fig = pygmt.Figure()

    # Set the region and projection of the map
    fig.basemap(region=[-20, 125, 5, 60], projection='M20c', frame='af') # The whole Alpine-Himalayan belt

    # Create a custom color palette for the relief shading
    pygmt.makecpt(cmap='grayC', series=[-1000, 7000, 100], output="input_data/cpts/alpides_grey.cpt", reverse=True)

    #pygmt.makecpt(cmap="geo", series=[-8000, 8000, 100], output="input_data/cpts/alpides_topo.cpt")

    # Here I fixed an aesthetic issue when plotting the color palette for on-land topography values under the sea level (like the region around the Caspian Sea)
    # Step 1: Create custom CPT (color palette table)
    pygmt.makecpt(cmap="geo", series=[-8000, 8000, 100], output="input_data/cpts/alpides_topo_fixed.cpt", transparency=70, background='o', truncate=[-8000, 8000])
    pygmt.makecpt(cmap="geo", series=[0, 8000, 100], output="input_data/cpts/land_cbar.cpt", transparency=70, background='o', truncate=[0, 8000])
    # Step 2: Modify CPT for land areas, including handling on-land topography below sea level (e.g., -500 to 0)
    with open("input_data/cpts/alpides_topo_fixed.cpt", 'r') as file:
        lines = file.readlines()

    # Find the color corresponding to the interval [0, 100] (which will be applied to the -500 to 0 range)
    for line in lines:
        if line.startswith("0\t"):
            color_for_zero_to_100 = line.split()[1]
            break

    new_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) > 2 and int(parts[0]) >= -500 and int(parts[2]) <= 0:
            if parts[1] != color_for_zero_to_100:
                parts[1] = color_for_zero_to_100
                parts[3] = color_for_zero_to_100
            new_lines.append("\t".join(parts))
        else:
            new_lines.append(line)

    # Write the modified CPT for land areas
    modified_land_cpt = "input_data/cpts/land_cpt_fixed.cpt"
    with open(modified_land_cpt, "w") as file:
        file.writelines(new_lines)


    # Add shaded topography with 30% transparency
    fig.grdimage(grid="@earth_relief_01m", cmap="input_data/cpts/alpides_grey.cpt", shading=True, transparency=70) # nan_transparent=True results in error in the latest GMT version

    # Create the custom colormap
    pygmt.makecpt(cmap='polar', series=[-4, 4, 0.01], continuous=True, background=True,)

    # Add coastlines
    fig.coast(water="white", shorelines="0.1p,black", area_thresh=4000, resolution='h') # borders="1/0.1p,gray90"

    # Add fault traces as very thin grey lines
    if fault_file:
        fig.plot(data=fault_file, pen="0.1p,grey")

    for input_file in input_files:

        # Read the CSV file
        df = pd.read_csv(input_file, skiprows=1, header=None)
        df.columns = ['Lon', 'Lat', 'U.vel', 'U.sig', 'Stat']
   
        # Create circles data with fixed size and colors based on u_vel
        circles = []
        for lon, lat, norm_u in zip(df['Lon'], df['Lat'], df['U.vel']):
            circles.append([lon, lat, norm_u, 0.07])

        # Plot circles with color based on u_vel and fixed size
        fig.plot(
            data=circles,
            style='cc',
            cmap=True,
            pen=None #pen='0.01p,black',
        )

    fig.coast(water=[], shorelines="0.1p,black", area_thresh=4000, resolution='h') # Clumsy way to avoid the coastlines being covered by the circles

    # Add a colorbar to the plot
    fig.colorbar(position="JMR+o0.5c/0c+w9c+v+e", cmap=True, frame=["a0.1", "+lVertical velocity (mm/yr)"])

    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale="JBR+o-9c/-0.8c+c0+w1000k+f+lkm")

    # Create a directory to store the figure 
    os.makedirs(figure_folder, exist_ok=True)

    # Save the figure
    if save_fig:
        figure_file_pdf = os.path.join(figure_folder, f'Alpides_vertical_velocity_field_map.pdf')
        fig.savefig(figure_file_pdf, dpi=300)

    # Show the figure
    fig.show()
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to plot the number of independent velocity estimates used to derive the combined velocity field
def plot_number_of_estimates(input_file, output_folder, save_fig=True):

    # Create a new figure
    fig = pygmt.Figure()

    # Set the region of the map
    region = [-20, 125, 5, 60]

    # Set the region and projection of the map
    fig.basemap(region=region, projection='M20c', frame='af')

    # Create a custom color palette for the relief shading
    pygmt.makecpt(cmap='grayC', series=[-1000, 7000, 100], output="input_data/cpts/alpides_grey.cpt", reverse=True)
    pygmt.makecpt(cmap="geo", series=[-5000, 5000, 100], output="input_data/cpts/alpides_topo.cpt")

    # Here I fixed an aesthetic issue when plotting the color palette for on-land topography values under the sea level (like the region around the Caspian Sea)
    # Read the CPT file
    with open("input_data/cpts/alpides_topo.cpt", "r") as file:
        lines = file.readlines()

     # Find the color corresponding to the interval [0, 100]
    for line in lines:
        if line.startswith("0\t0/97/71"):
            color_for_zero_to_100 = line.split()[1]
            break

    # Initialise a flag to modify values from -500 to 0 once
    new_lines = []

    # Modify the lines with elevations between -500 and 0 to match 0-100 color
    for line in lines:
        parts = line.split("\t")
            
        # Identify lines with elevations between -500 and 0
        if len(parts) > 2 and int(parts[0]) >= -500 and int(parts[2]) <= 0:
            if parts[1] != color_for_zero_to_100:  # Apply change only if it's not already modified
                # Replace the color for negative elevations with the color for 0 to 100
                parts[1] = color_for_zero_to_100
                parts[3] = color_for_zero_to_100
            new_lines.append("\t".join(parts))
        else:
            new_lines.append(line)

    # Write the modified CPT file
    with open("input_data/cpts/alpides_topo_fixed.cpt", "w") as file:
        file.writelines(new_lines)

    # Add shaded topography with transparency
    fig.grdimage(grid="@earth_relief_01m", cmap="input_data/cpts/alpides_grey.cpt", shading=True, transparency=70) # nan_transparent=True results in error in the latest GMT version

    # Add coastlines
    fig.coast(water='white', shorelines="0.1p,black", area_thresh=4000, resolution='h') # borders="1/0.1p,gray90",

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Cap the 'Num' values at 15
    df['Num'] = df['Num'].clip(upper=15)
    
    # Apply the log transformation
    df['Log_Num'] = np.log10(df['Num'])

    # Define the minimum and maximum for the colormap based on the transformed 'Log_Num' column
    min_val = df['Log_Num'].min()
    max_val = df['Log_Num'].max()

    # Adjust the colormap range based on the data
    pygmt.makecpt(cmap='turbo', series=[min_val, max_val, 0.05], continuous=False, background='o', log=True) # reverse=True
    
    # Create triangles with fixed size (e.g., 0.1 cm) and colors based on 'Num'
    triangles = [[lon, lat, num, 0.1] for lon, lat, num in zip(df['Lon'], df['Lat'], df['Num'])]
    
    # Plot triangles with color based on 'Num' and fixed size
    fig.plot(
        data=triangles,
        style='tc',
        cmap=True,
        pen='0.01p,black',
    )
    
    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale="JBR+o-9c/-0.8c+c0+w1000k+f+lkm")

    # Add a colorbar to the plot
    fig.colorbar(position="JMR+o0.5c/0c+w8.5c+v+ef", cmap=True, frame=["a5", "+lNumber of independent GNSS velocity solutions"])
    
    # Save the figure
    if save_fig:
        figure_file_pdf = os.path.join(output_folder, f'Alpides_number_of_estimates.pdf')
        fig.savefig(figure_file_pdf, dpi=300)

    # Show the figure
    fig.show()
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to load polygon from a file
def load_polygon(file_path):
    with open(file_path, 'r') as f:
        coords = [tuple(map(float, line.strip().split())) for line in f]
    return Polygon(coords)
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to plot a single GNSS velocity field on a map
def plot_velocity_field(input_file, region, figures_path, output_filename, path2faults, vel_scale_pos, vel_mag_ref_scale, vel_scaling_factor, 
                        vel_scale_label_offset, relief_colormap, path2_save_relief_colormap, topofile, map_scale_bar_position, map_scale_bar_length_km, 
                        colorbar_width, colormap_range, faults_pen="0.1p,grey50",map_projection='M20c',plot_label=False, label_text=None, 
                        label_font="12,Helvetica,black",label_offset=[-0.5, 0.5], euler_sites=None, euler_box=None, label_euler_box=None,
                        label_euler_sites=None, label_no_euler_sites=None, legend_settings=None, legend_box_settings=None, save_fig=True, 
                        save_euler_fig=True, output_filename_euler=None):
    """
    Plot synthetic velocity field using input velocity data.
    
    Args:
    - input_file (str): Path to the velocity field file (must contain columns 'Lon', 'Lat', 'E.vel', 'N.vel', 'E.adj', 'N.adj', 'E.sig', 'N.sig', 'Corr', 'U.vel', 'U.adj', 'U.sig', 'Stat').
    - region (list): Bounding box for the region in the format [west, east, south, north].
    - output_folder (str): Path to the folder where the output figures will be saved.
    
    Returns:
    - None
    """
    
    # Read the velocity data from the input file 
    # Define column names to match those on the input file 
    columns = ['Lon', 'Lat', 'E.vel', 'N.vel', 'E.adj', 'N.adj', 'E.sig', 'N.sig', 'Corr', 'U.vel', 'U.adj', 'U.sig', 'Stat', 'E.sig.scaled', 'N.sig.scaled']
    velocity_data = pd.read_csv(input_file, header=None, names=columns, skiprows=1, delimiter="\t")
    # Convert necessary columns to numeric and handle non-numeric entries
    velocity_data[['Lon', 'Lat', 'E.vel', 'N.vel', 'E.sig.scaled', 'N.sig.scaled']] = velocity_data[['Lon', 'Lat', 'E.vel', 'N.vel', 'E.sig.scaled', 'N.sig.scaled']].apply(pd.to_numeric, errors='coerce')
    # Create new column with velocity magnitude
    velocity_data['v_mag'] = np.sqrt(velocity_data['E.vel']**2 + velocity_data['N.vel']**2)

    # Drop any rows with NaNs in the key columns after conversion
    velocity_data.dropna(subset=['Lon', 'Lat', 'E.vel', 'N.vel', 'E.sig.scaled', 'N.sig.scaled'], inplace=True)

    #only keep the necessary columns 
    velocity_data = velocity_data[['Lon', 'Lat', 'E.vel', 'N.vel', 'E.sig.scaled', 'N.sig.scaled', 'v_mag']]

    # Create a list to store the vectors
    vectors = []

    # Iterate over each site
    for i in range(len(velocity_data)):
        # Get the coordinates of the site
        x_start = velocity_data.iloc[i]['Lon']
        y_start = velocity_data.iloc[i]['Lat']
        # Get the velocity components
        ve = velocity_data.iloc[i]['E.vel']
        vn = velocity_data.iloc[i]['N.vel']
        # Calculate the direction of the vector
        direction_degrees = np.degrees(np.arctan2(vn, ve))
        # Assign the length of the vector
        vel_mag = velocity_data.iloc[i]['v_mag']
        length = vel_mag
        # Add the vector to the list
        vectors.append([x_start, y_start, direction_degrees, length * vel_scaling_factor])

    # Start a PyGMT figure for Velocity magnitudes
    fig = pygmt.Figure()

    # Create a custom CPT (color palette table)
    #pygmt.makecpt(cmap=relief_colormap, series=[-4000, 4000], output=path2_save_relief_colormap, transparency=70, background='o', truncate=[-4000, 4000])
    # Step 1: Create custom CPT (color palette table)
    pygmt.makecpt(cmap=relief_colormap, series=[colormap_range[0], colormap_range[1], 100], output=path2_save_relief_colormap, transparency=70, background='o', truncate=[colormap_range[0], colormap_range[1]])
    pygmt.makecpt(cmap=relief_colormap, series=[0, colormap_range[1], 100], output="input_data/cpts/land_cbar.cpt", transparency=70, background='o', truncate=[0, colormap_range[1]])
    # Step 2: Modify CPT for land areas, including handling on-land topography below sea level (e.g., -500 to 0)
    with open(path2_save_relief_colormap, 'r') as file:
        lines = file.readlines()

    # Find the color corresponding to the interval [0, 100] (which will be applied to the -500 to 0 range)
    for line in lines:
        if line.startswith("0\t"):
            color_for_zero_to_100 = line.split()[1]
            break

    new_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) > 2 and int(parts[0]) >= -500 and int(parts[2]) <= 0:
            if parts[1] != color_for_zero_to_100:
                parts[1] = color_for_zero_to_100
                parts[3] = color_for_zero_to_100
            new_lines.append("\t".join(parts))
        else:
            new_lines.append(line)

    # Write the modified CPT for land areas
    modified_land_cpt = "input_data/cpts/land_cpt_fixed.cpt"
    with open(modified_land_cpt, "w") as file:
        file.writelines(new_lines)

    # Set the basemap and projection
    fig.basemap(region=region, projection=map_projection, frame='af')

    # Plot the relief grid (with transparency)
    fig.grdimage(grid=topofile, cmap=modified_land_cpt, shading=True, transparency=70)

    # Add coastline and shorelines
    fig.coast(shorelines="0.2p,black", area_thresh=4000, resolution='h', water='white')

    # Add fault traces as thin grey lines
    fig.plot(data=path2faults, pen=faults_pen)

    # Plot the velocity vectors
    fig.plot(
        style='v0.2c+e+n0.3/0.2',
        data=vectors,
        fill='gray25',
        pen='gray25',
    )

    # Add a map scale
    #with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
    #    fig.basemap(map_scale="JBR+o-6.5c/-0.8c+c0+w200k+f+lkm")
    
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")

    # Define scale vectors
    scale_origin_lon = vel_scale_pos[0]
    scale_origin_lat = vel_scale_pos[1]
    scale_vector_length = vel_mag_ref_scale * vel_scaling_factor  # in mm/yr
    scale_vector_length_text = vel_mag_ref_scale  # actual value

    scale_vectors = [
        [scale_origin_lon, scale_origin_lat, 0, scale_vector_length],  # Eastward
        [scale_origin_lon, scale_origin_lat, 90, scale_vector_length]  # Northward
    ]

    # Plot the scale vectors
    fig.plot(
        style='v0.2c+e+n0.3/0.2',
        data=scale_vectors,
        fill='black',
        pen='0.5,black',
    )

    # Annotate the scale vector
    fig.text(
        text=f'{scale_vector_length_text} mm/yr',
        x=scale_origin_lon + vel_scale_label_offset[0],
        y=scale_origin_lat + vel_scale_label_offset[1],
        font='7p,black'
    )

    # Add color bar
    fig.colorbar(frame="af+lElevation (m)", position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef", cmap="input_data/cpts/land_cbar.cpt")

    if plot_label and label_text is not None:
        # Let's add the label in the upper-right corner
        fig.text(
            x=region[1] - label_offset[0],  # Adjust near the right edge
            y=region[3] - label_offset[1],  # Adjust near the top edge
            text=label_text,
            font=label_font,
            fill="white",
            justify="TR",
            pen="0.1p,black"
        )

    # Save the figure
    if save_fig:
        output_path = f"{figures_path}/{output_filename}"
        fig.savefig(output_path, dpi=600)
    
    # Display the figure
    fig.show()

    print(f"Figure saved to {output_path}")

    # PLot a second figure if plot_euler_sites is defined
    if euler_sites is not None:
        # Start a PyGMT figure 
        fig = pygmt.Figure()

        # Set the basemap and projection
        fig.basemap(region=region, projection=map_projection, frame='af')

        # Plot the relief grid (with transparency)
        fig.grdimage(grid=topofile, cmap=modified_land_cpt, shading=True, transparency=70)

        # Add coastline and shorelines
        fig.coast(shorelines="0.2p,black", area_thresh=4000, resolution='h', water='white')

        # Add fault traces as thin grey lines
        fig.plot(data=path2faults, pen=faults_pen)

        # print initial sites used for iterative Euler pole estimation
        if euler_box is not None:
            fig.plot(x=euler_box[:, 0], y=euler_box[:, 1], pen='1.5p,orange', fill=None, label=label_euler_box)

        # Plot the GNSS sites as red triangles
        fig.plot(
            x=velocity_data['Lon'], y=velocity_data['Lat'], 
            style='t0.08i', fill='grey30', label=label_no_euler_sites
        )

        # Plot the GNSS sites as red triangles
        fig.plot(
            x=euler_sites[:, 1], y=euler_sites[:, 0], 
            style='t0.08i', fill='red', label=label_euler_sites
        )

        # Add a map scale
        with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")

        # Add a legend
        fig.legend(position=legend_settings, box=legend_box_settings)

        # Save the figure
        if save_euler_fig:
            output_path = f"{figures_path}/{output_filename_euler}"
            fig.savefig(output_path, dpi=600)

        # show the figure
        fig.show()
        print(f"Figure saved to {output_filename_euler}")
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
# Function to plot filtering results from 3 input files: accepted velocities, rejected (velocity coherence criterion), and rejected (velocity uncertainty criterion)
def plot_filtering_results(accepted_file, lognormal_file, coherence_file, study_region, figures_path, output_filename, path2faults, map_scale_pos, vel_mag_ref_scale, vel_scaling_factor, relief_colormap, topofile, lower_limit_topo, upper_limit_topo, topo_step, map_scale_params, save_fig=True):
    """
    Plot three velocity fields (accepted, lognormal filtered, and coherence filtered) with input velocity data, fault traces, and relief shading.
    
    Args:
    - accepted_file (str): Path to the accepted velocity field file.
    - lognormal_file (str): Path to the lognormal filtered velocity field file.
    - coherence_file (str): Path to the coherence filtered velocity field file.
    - region (list): Bounding box for the region in the format [west, east, south, north].
    - figures_path (str): Path to the folder where the output figure will be saved.
    - output_filename (str): Output filename for the velocity field map.
    - path2faults (str): Path to the fault traces file.
    - map_scale_pos (list): Position of the map scale on the map [longitude, latitude].
    - vel_mag_ref_scale (int): Reference scale for the velocity magnitude scale vectors in mm/yr.
    - vel_scaling_factor (float): Scaling factor for the velocity vectors.
    - relief_colormap (str): Colormap for the relief shading.
    - path2_save_relief_colormap (str): Path to the colormap file.
    - topofile (str): Path to the topography file (Earth relief).
    
    Returns:
    - None
    """
    
    # Define column names
    columns = ['Lon', 'Lat', 'E.vel', 'N.vel', 'E.adj', 'N.adj', 'E.sig', 'N.sig', 'Corr', 'U.vel', 'U.adj', 'U.sig', 'Stat']
    
    # Load the accepted velocity data
    accepted_vel_data = pd.read_csv(accepted_file, sep=r'\s+', header=None, names=columns)
    # Convert necessary columns to numeric and handle non-numeric entries
    accepted_vel_data[['Lon', 'Lat', 'E.vel', 'N.vel']] = accepted_vel_data[['Lon', 'Lat', 'E.vel', 'N.vel']].apply(pd.to_numeric, errors='coerce')
    # Create new column with velocity magnitude
    accepted_vel_data['v_mag'] = np.sqrt(accepted_vel_data['E.vel']**2 + accepted_vel_data['N.vel']**2)
    # Drop any rows with NaNs in the key columns after conversion
    accepted_vel_data.dropna(subset=['Lon', 'Lat', 'E.vel', 'N.vel'], inplace=True)
    #print(accepted_vel_data)

    # Load the lognormal filtered velocity data
    lognormal_data = pd.read_csv(lognormal_file, sep=r'\s+', header=None, names=columns)
    # Convert necessary columns to numeric and handle non-numeric entries
    lognormal_data[['Lon', 'Lat', 'E.vel', 'N.vel']] = lognormal_data[['Lon', 'Lat', 'E.vel', 'N.vel']].apply(pd.to_numeric, errors='coerce')
    # Create new column with velocity magnitude
    lognormal_data['v_mag'] = np.sqrt(lognormal_data['E.vel']**2 + lognormal_data['N.vel']**2)
    # Drop any rows with NaNs in the key columns after conversion
    lognormal_data.dropna(subset=['Lon', 'Lat', 'E.vel', 'N.vel'], inplace=True)
    #print(lognormal_data)
    
    # Load the coherence filtered velocity data
    coherence_data = pd.read_csv(coherence_file, sep=r'\s+', header=None, names=columns)
    # Convert necessary columns to numeric and handle non-numeric entries
    coherence_data[['Lon', 'Lat', 'E.vel', 'N.vel']] = coherence_data[['Lon', 'Lat', 'E.vel', 'N.vel']].apply(pd.to_numeric, errors='coerce')
    # Create new column with velocity magnitude
    coherence_data['v_mag'] = np.sqrt(coherence_data['E.vel']**2 + coherence_data['N.vel']**2)
    # Drop any rows with NaNs in the key columns after conversion
    coherence_data.dropna(subset=['Lon', 'Lat', 'E.vel', 'N.vel'], inplace=True)
    #print(coherence_data)

    # Create lists to store the vectors for plotting
    vectors_accepted = []
    vectors_lognormal = []
    vectors_coherence = []

    # Process the accepted velocity data
    for i in range(len(accepted_vel_data)):
        x_start = accepted_vel_data.iloc[i]['Lon']
        y_start = accepted_vel_data.iloc[i]['Lat']
        ve = accepted_vel_data.iloc[i]['E.vel']
        vn = accepted_vel_data.iloc[i]['N.vel']
        direction_degrees = np.degrees(np.arctan2(vn, ve))
        vel_mag = accepted_vel_data.iloc[i]['v_mag']
        vectors_accepted.append([x_start, y_start, direction_degrees, vel_mag * vel_scaling_factor])

    # Process the lognormal filtered velocity data
    for i in range(len(lognormal_data)):
        x_start = lognormal_data.iloc[i]['Lon']
        y_start = lognormal_data.iloc[i]['Lat']
        ve = lognormal_data.iloc[i]['E.vel']
        vn = lognormal_data.iloc[i]['N.vel']
        direction_degrees = np.degrees(np.arctan2(vn, ve))
        vel_mag = lognormal_data.iloc[i]['v_mag']
        vectors_lognormal.append([x_start, y_start, direction_degrees, vel_mag * vel_scaling_factor])

    # Process the coherence filtered velocity data
    for i in range(len(coherence_data)):
        x_start = coherence_data.iloc[i]['Lon']
        y_start = coherence_data.iloc[i]['Lat']
        ve = coherence_data.iloc[i]['E.vel']
        vn = coherence_data.iloc[i]['N.vel']
        direction_degrees = np.degrees(np.arctan2(vn, ve))
        vel_mag = coherence_data.iloc[i]['v_mag']
        vectors_coherence.append([x_start, y_start, direction_degrees, vel_mag * vel_scaling_factor])

    # Start a PyGMT figure for Velocity magnitudes
    fig = pygmt.Figure()

    # Set the basemap and projection
    fig.basemap(region=study_region, projection='M20c', frame='af')

    # Create a colormap for relief shading
    pygmt.makecpt(cmap=relief_colormap, series=[lower_limit_topo, upper_limit_topo, topo_step])
    fig.grdimage(grid=topofile, cmap=True, shading=True, transparency=70) 

    # Add fault traces as thin grey lines
    fig.plot(data=path2faults, pen="0.1p,grey50")

    # Plot the velocity vectors for each dataset
    fig.plot(style='v0.2c+e+n0.15', data=vectors_accepted, fill='gray25', pen='gray25', label=f'Accepted velocities+S0.5c')
    fig.plot(style='v0.2c+e+n0.15', data=vectors_lognormal, fill='orange', pen='orange', label=f'Filtered (uncertainty criterion)+S0.5c')
    fig.plot(style='v0.2c+e+n0.15', data=vectors_coherence, fill='red', pen='red', label=f'Filtered (coherence criterion)+S0.5c')

    # Add coastline and shorelines
    fig.coast(water='white', shorelines="0.1p,black", area_thresh=4000, resolution='h')

    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.basemap(map_scale=f"JBR+o{map_scale_params[0]}c/{map_scale_params[1]}c+c0+w{map_scale_params[2]}k+f+lkm")

    # Define the scale vectors (for reference velocity magnitude)
    scale_vector_length = vel_mag_ref_scale * vel_scaling_factor
    scale_vectors = [
        [map_scale_pos[0], map_scale_pos[1], 0, scale_vector_length],  # Eastward vector
        [map_scale_pos[0], map_scale_pos[1], 90, scale_vector_length]  # Northward vector
    ]

    # Plot the scale vectors
    fig.plot(style='v0.2c+e+n0.15', data=scale_vectors, fill='black', pen='black')

    # Annotate the scale vector
    fig.text(text=f'{vel_mag_ref_scale} mm/yr', x=map_scale_pos[0] - 1.5, y=map_scale_pos[1], font='7p,black')

    # Add a legend
    fig.legend(position='jTR+o0.2c', box='+gwhite+p1p')

    # Save the figure
    if save_fig:
        output_path = f"{figures_path}/{output_filename}"
        fig.savefig(output_path, dpi=600, crop=True)

    # Display the figure
    fig.show()

    print(f"Figure saved to {output_path}")
########################################################################################################################
########################################################################################################################

# Helper function to mask a region from user-defined polygons
def mask_observations_with_polygons(df: pd.DataFrame, mask_polygons: dict) -> pd.DataFrame:
    """
    Masks out observations in a pandas DataFrame by assigning NaN to rows
    where the (lon, lat) point falls inside any of the polygons provided.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'lon' and 'lat' columns.
        mask_polygons (dict): Dictionary with keys as polygon names and values as file paths.
                              Each file is assumed to be a two-column text file (lon, lat)
                              defining a closed polygon.
    
    Returns:
        pd.DataFrame: The DataFrame with rows inside any polygon set entirely to NaN.
    """
    # Loop over each polygon and mask the observations inside it.
    for poly_name, poly_path in mask_polygons.items():
        # Read polygon coordinates (assumes two-column: lon, lat)
        poly_coords = np.loadtxt(poly_path)
        # Create a Shapely Polygon from the coordinates
        polygon = shapely.geometry.Polygon(poly_coords)
        # Use vectorized contains to get a boolean array for rows inside the polygon
        inside = shapely.vectorized.contains(polygon, df['lon'].values, df['lat'].values)
        # Assign NaN to all columns for those rows that fall inside the polygon.
        df.loc[inside] = np.nan
    return df

########################################################################################################################
########################################################################################################################
# Function to plot strain rate parameters on a map
def plot_strain_rates_with_principal_directions(
    file_path, fault_file, focal_mechanism_file, output_folder,
    region, strain_grid_spacing, principal_directions_grid_spacing, output_filenames_prefix,
    max_shear_cpt, dilatation_cpt, sec_inv_cpt, strain_rate_style_cpt, azimuth_diff_cpt,
    path2_max_shear_cpt, path2_dilatation_cpt, path2_sec_inv_cpt, path2_strain_rate_style_cpt, path2_azimuth_diff_cpt,
    range_max_shear, range_dilatation, range_sec_inv, range_strain_rate_style, range_azimuth_max_shortening_rate, 
    mask_below_sec_inv_value, plot_moment_tensors, moment_tensors_scale, map_scale_bar_length_km, map_scale_bar_position, 
    strain_rate_cross_scale_positon, strain_rate_cross_scale_magnitude, strain_rate_cross_size, strain_rate_cross_label_offset, 
    colorbar_width, ocean_fill_color, shorelines_pen, include_kostrov = False, kostrov_file = None, plot_sks_kostrov_histogram=False, 
    plot_box_for_histogram=False, hist_shift=[0.3, 0.2], hist_projection="X3.2c/3.2c", 
    hist_frame=["ENsw+gwhite", "xf10a30+lSigned difference", "yf2.5a5+u%+lFrequency percent"], hist_annot_primary="6p", hist_font_label="6p",  
    box_histogram_position=[-3.2, 20], box_histogram_style="r4.6/4.2", box_histogram_fill="white", box_histogram_transparency=30,  
    plot_strain_comparison=False, strain_profile_file_no_creep=None, strain_profile_file_smoothed_no_creep=None, strain_profile_file_with_creep=None,
    strain_profile_file_smoothed_with_creep=None, plot_profile_box=False, box_coords=None, label_coords=None, faults_pen="0.1p,darkgrey", mask_polygons=None, 
    apply_mask=False, mask_dilatation_map=False, plot_mask_polygons=False, mask_fill="white", add_tectonic_map_labels=False, path2_tectonic_map_labels=None, 
    tectonic_map_labels_offset=[0,0], tectonic_map_labels_transparency=30, range_dilatation_sigma=[-50, 50], range_sec_inv_sigma=[0, 50], range_max_shear_sigma=[0, 50], 
    map_projection='M20c', scatter_style="c0.18c", save_fig=True
):
    """
    This function reads strain rates exported from BforStrain and estimates different strain rate tensor 
    parameters, including the second invariant, dilatation rates (first invariant), and maximum shear strain rates.
    It interpolates the strain rate tensor components into a regular grid and estimates the principal directions of strain rates.
    The function then plots the interpolated strain rates and the principal directions of strain rates.
    """

    # Define the base column names that are always expected
    base_columns = [
        'lon', 'lat', 'mean_maxshear', 'std_maxshear', 'mean_dilatation',
        'std_dilatation', 'Exx_mean', 'Exy_mean', 'Eyy_mean', 'Exx_std', 'Exy_std', 'Eyy_std'
    ]
    
    # Try loading the first few lines to check for additional columns
    sample_data = pd.read_csv(file_path, sep=r'\s+', nrows=5, header=None)
    
    # Check if the number of columns matches the expected number + 2 (for the additional columns)
    if sample_data.shape[1] == len(base_columns) + 2:
        # If there are extra columns, add the column names for mean and std of second invariant
        columns = base_columns + ['mean_second_inv', 'std_second_inv']
    else:
        # If not, stick with the base columns
        columns = base_columns
    
    # Now load the full dataset with the appropriate column names
    data = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)

    # Convert relevant columns to numeric, coercing errors to NaN
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop rows with NaN values in critical columns
    critical_columns = ['lon', 'lat', 'Exx_mean', 'Exy_mean', 'Eyy_mean']
    data.dropna(subset=critical_columns, inplace=True)
    
    # If the extra columns are not present, drop any NaN columns
    if 'mean_second_inv' in data.columns and data['mean_second_inv'].isna().all():
        data.drop(['mean_second_inv', 'std_second_inv'], axis=1, inplace=True)
    
    # if include_kostrov is True, load the Kostrov data
    if include_kostrov:
        # Load Kostrov file
        kostrov_data = pd.read_csv(kostrov_file, sep=r"\s+", header=None)

        # Define column names based on known structure
        column_names = [
            "err", "ert", "erp", "ett", "etp", "epp", "lon", "lat"
        ]
        additional_columns = [f"extra_{i}" for i in range(kostrov_data.shape[1] - len(column_names))]
        column_names.extend(additional_columns)
        kostrov_data.columns = column_names

        # Correct longitude values to be between -180 and 180
        kostrov_data.loc[kostrov_data["lon"] > 180, "lon"] -= 360

        # Calculate strain rate parameters
        kostrov_data["emean"] = (kostrov_data["ett"] + kostrov_data["epp"]) / 2.0
        kostrov_data["ediff"] = (kostrov_data["ett"] - kostrov_data["epp"]) / 2.0
        kostrov_data["taumax"] = np.sqrt(kostrov_data["erp"]**2 + kostrov_data["ediff"]**2)
        kostrov_data["emax"] = kostrov_data["emean"] + kostrov_data["taumax"]
        kostrov_data["emin"] = kostrov_data["emean"] - kostrov_data["taumax"]
        kostrov_data["sr_style"] = -(kostrov_data["emax"] + kostrov_data["emin"]) / (np.abs(kostrov_data["emax"]) + np.abs(kostrov_data["emin"]))

        # Here I remove NaNs and raise an error if no valid data is available
        valid_kostrov_data = kostrov_data[["lon", "lat", "sr_style"]].dropna()
        if valid_kostrov_data.empty:
            raise ValueError("No valid data available for interpolation")
        
        # Interpolate the Kostrov data to observations points from input strain rate data using griddata
        kostrov_interp_2_strain_tri_mesh = griddata(
            valid_kostrov_data[["lon", "lat"]].values,
            valid_kostrov_data["sr_style"].values,
            (data["lon"], data["lat"]),
            method="linear"
        )

        # Add the interpolated Kostrov strain rate style data to the strain rate data frame
        data["kostrov_style_interp_2_strain_obs"] = kostrov_interp_2_strain_tri_mesh

    # Convert all the strain rate units from microstrain/yr to nanostrain/yr
    data['mean_maxshear'] *= 1000
    data['std_maxshear'] *= 1000
    data['mean_dilatation'] *= 1000
    data['std_dilatation'] *= 1000
    data['Exx_mean'] *= 1000
    data['Exy_mean'] *= 1000
    data['Eyy_mean'] *= 1000
    data['Exx_std'] *= 1000
    data['Exy_std'] *= 1000
    data['Eyy_std'] *= 1000

    # If additional columns exist, convert the second invariant std to nanostrain/yr
    if 'std_second_inv' in data.columns:
        data['mean_second_inv'] *= 1000
        data['std_second_inv'] *= 1000
        
    # print data for debugging
    #print(data.head(10))

    ########################################################################################################################
    ########################################### Estimate the strain rate parameters ########################################
    ########################################################################################################################

    data['emean'] = (data['Exx_mean'] + data['Eyy_mean']) / 2.0
    data['ediff'] = (data['Exx_mean'] - data['Eyy_mean']) / 2.0
    data['taumax'] = np.sqrt(data['Exy_mean']**2 + data['ediff']**2)
    data['emax'] = data['emean'] + data['taumax']
    data['emin'] = data['emean'] - data['taumax']
    data['cov'] = np.pi / 180.0
    # Angle of Max shear, ranging from -90 to 90 in cartesian coordinates. Division by 2 accounts for 45 degrees difference
    data['azim'] = -np.arctan2(data['Exy_mean'], data['ediff']) / data['cov'] / 2.0 
    # Angle of max principal strain rate emax, ranging from 0 to 180, referenced to North
    data['azim'] = 90.0 + data['azim'] 
    data['dexazim'] = data['azim'] + 45.0 - 180.0
    data['dilat'] = data['Exx_mean'] + data['Eyy_mean']
    data['sec_inv'] = np.sqrt(data['Exx_mean']**2 + 2.0 * data['Exy_mean']**2 + data['Eyy_mean']**2)
    data['sr_style'] = -(data['emax'] + data['emin']) / (abs(data['emax']) + abs(data['emin']))

    ########################################################################################################################
    ########################## Estimate the azimuth of the maximum shortening rate direction ###############################
    ########################################################################################################################

    # Correct the azimuth for maximum shortening direction
    data['azim_shortening'] = np.where(
        data['emax'] > 0,  # Extension
        data['azim'] + 90,  # Add 90 degrees
        data['azim']  # Compression, keep azimuth
    )

    # Ensure azim_shortening stays within 0 to 360 degrees
    data['azim_shortening'] = np.mod(data['azim_shortening'], 360)

    # Adjust azimuth to be in the range of -90 to 90 degrees
    data['azim_shortening'] = np.where(
        data['azim_shortening'] > 180,
        data['azim_shortening'] - 360,
        data['azim_shortening']
    )

    # Now restrict the range to -90 to 90 degrees
    data['azim_shortening'] = np.where(
        data['azim_shortening'] < -90,
        data['azim_shortening'] + 180,
        data['azim_shortening']
    )

    data['azim_shortening'] = np.where(
        data['azim_shortening'] > 90,
        data['azim_shortening'] - 180,
        data['azim_shortening']
    )

    ########################################################################################################################
    ############## Interpolate second invariant, Max shear strain, and dilatation rates into regular grids #################
    ########################################################################################################################

    # Use blockmedian and surface to create grids
    def create_grid(data_col):
        blockmedian_output = pygmt.blockmedian(
            data=data[['lon', 'lat', data_col]],
            spacing=strain_grid_spacing,
            region=region
        )
        grid = pygmt.surface(
            data=blockmedian_output,
            spacing=strain_grid_spacing,
            region=region
        )
        return grid

    shear_grid = create_grid('mean_maxshear')
    dilatation_grid = create_grid('mean_dilatation')
    sec_inv_grid = create_grid('sec_inv')

    ########################################################################################################################
    ############ Interpolate strain rate tensor components into a regular grid to estimate principal directions ############
    ########################################################################################################################

    lon_grid = np.arange(region[0] - 1, region[1] + 1, principal_directions_grid_spacing)
    lat_grid = np.arange(region[2] - 1, region[3] + 1, principal_directions_grid_spacing)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate the strain rate tensor components
    points = np.array([data['lon'], data['lat']]).T
    Exx_mean_interp = griddata(
        points, data['Exx_mean'], (lon_mesh, lat_mesh), method='linear')
    Exy_mean_interp = griddata(
        points, data['Exy_mean'], (lon_mesh, lat_mesh), method='linear')
    Eyy_mean_interp = griddata(
        points, data['Eyy_mean'], (lon_mesh, lat_mesh), method='linear')
    
    # Interpolate Kostrov data to regular grid
    if include_kostrov:
        kostrov_interp = griddata(
            data[["lon", "lat"]].values,
            data["kostrov_style_interp_2_strain_obs"].values,
            (lon_mesh, lat_mesh),
            method="linear"
        )

    # Combine into a dataframe
    interp_data = pd.DataFrame({
        'lon': lon_mesh.ravel(),
        'lat': lat_mesh.ravel(),
        'Exx_mean': Exx_mean_interp.ravel(),
        'Exy_mean': Exy_mean_interp.ravel(),
        'Eyy_mean': Eyy_mean_interp.ravel()
    })

    if include_kostrov:
        interp_data["kostrov_sr_style"] = kostrov_interp.ravel()

    # Drop any NaNs
    interp_data.dropna(inplace=True)

    # Estimate strain rate parameters on the interpolated grid
    interp_data['emean'] = (interp_data['Exx_mean'] + interp_data['Eyy_mean']) / 2.0
    interp_data['ediff'] = (interp_data['Exx_mean'] - interp_data['Eyy_mean']) / 2.0
    interp_data['taumax'] = np.sqrt(interp_data['Exy_mean']**2 + interp_data['ediff']**2)
    interp_data['emax'] = interp_data['emean'] + interp_data['taumax']
    interp_data['emin'] = interp_data['emean'] - interp_data['taumax']
    interp_data['cov'] = np.pi / 180.0
    interp_data['azim'] = -np.arctan2(interp_data['Exy_mean'], interp_data['ediff']) / interp_data['cov'] / 2.0
    interp_data['azim'] = 90.0 + interp_data['azim']
    interp_data['dexazim'] = interp_data['azim'] + 45.0 - 180.0
    interp_data['dilat'] = interp_data['Exx_mean'] + interp_data['Eyy_mean']
    interp_data['sec_inv'] = np.sqrt(interp_data['Exx_mean']**2 + 2.0 * interp_data['Exy_mean']**2 + interp_data['Eyy_mean']**2)
    interp_data['sr_style'] = -(interp_data['emax'] + interp_data['emin']) / (abs(interp_data['emax']) + abs(interp_data['emin']))

    ########################################################################################################################

    # Apply the polygon mask to the interpolated data
    if mask_polygons is not None:
        interp_data = mask_observations_with_polygons(interp_data, mask_polygons)

    # Create a mask for sr_style to filter out data with low strain rates
    sr_style_mask = interp_data['sec_inv'] < mask_below_sec_inv_value
    interp_data.loc[sr_style_mask, 'sr_style'] = np.nan

    if include_kostrov and not valid_kostrov_data.empty:
        interp_data.loc[sr_style_mask, 'kostrov_sr_style'] = np.nan
        # Here I create a new entry for the difference strain rate style derived from GNSS strain rates and Kostrov-summed moment tensors
        # Compute the absolute difference
        interp_data['sr_style_abs_diff'] = abs(interp_data['sr_style'] - interp_data['kostrov_sr_style'])

        # Assign the sign based on which value is higher
        interp_data['sr_style_diff'] = np.where(
            interp_data['sr_style'] > interp_data['kostrov_sr_style'],
            interp_data['sr_style_abs_diff'],  # Positive if GNSS-derived is higher
            -interp_data['sr_style_abs_diff']  # Negative if Kostrov-summed is higher
        )

    # Create grid for sr_style
    sr_style_grid = pygmt.surface(
        data=interp_data[['lon', 'lat', 'sr_style']],
        spacing=strain_grid_spacing,
        region=region,
        maxradius='70k'
    )

    # Create grid for kostrov_sr_style
    if include_kostrov and not valid_kostrov_data.empty:
        kostrov_sr_style_grid = pygmt.surface(
            data=interp_data[['lon', 'lat', 'kostrov_sr_style']],
            spacing=strain_grid_spacing,
            region=region,
            maxradius='70k'
        )
    
    # Create grid for kostrov_sr_style difference
    if include_kostrov and not valid_kostrov_data.empty:
        sr_style_diff_grid = pygmt.surface(
            data=interp_data[['lon', 'lat', 'sr_style_diff']],
            spacing=strain_grid_spacing,
            region=region,
            maxradius='70k'
        )

    # Compute the azimuth of the maximum shortening rate direction
    # If emax > 0, the maximum shortening direction is 90 degrees away
    interp_data['azim_shortening'] = np.where(
        interp_data['emax'] > 0,  # Check if extensional
        interp_data['azim'] + 90,  # Add 90 degrees for extensional areas
        interp_data['azim']  # Leave as is if compressional
    )

    # Wrap azimuth values to the [-180°, 180°] range
    interp_data['azim_shortening'] = np.mod(interp_data['azim_shortening'] + 180, 360) - 180

    # Convert to the range [-90°, 90°]
    interp_data['azim_shortening'] = np.where(
        interp_data['azim_shortening'] < -90, 
        interp_data['azim_shortening'] + 180,  # Add 180 if azimuth is less than -90°
        interp_data['azim_shortening']
    )

    interp_data['azim_shortening'] = np.where(
        interp_data['azim_shortening'] > 90, 
        interp_data['azim_shortening'] - 180,  # Subtract 180 if azimuth is greater than 90°
        interp_data['azim_shortening']
    )

    # Mask azimuths for areas with low strain rates
    interp_data.loc[sr_style_mask, 'azim_shortening'] = np.nan  # Apply NaN masking

    # Use PyGMT surface to create a grid for the maximum shortening azimuth while keeping NaN values. 
    # Interpolating azimuths is a bad idea, so we compute the azimuth from the intepolated components!!!!!!!!!!!
    azim_shortening_grid = pygmt.surface(
        data=interp_data[['lon', 'lat', 'azim_shortening']],  # Include NaN values
        spacing=strain_grid_spacing,
        region=region,
        maxradius='70k'
    )

    ########################################################################################################################

    # Filter data based on the second invariant
    interp_data_filtered = interp_data[interp_data['sec_inv'] >= mask_below_sec_inv_value].copy()

    # Add 90 degrees to azimuths
    interp_data_filtered['azim_compression'] = interp_data_filtered['azim'] + 90.0
    interp_data_filtered['azim_extension'] = interp_data_filtered['azim'] + 90.0

    # Separate data into compression and extension components
    compression = interp_data_filtered[['lon', 'lat', 'emin', 'azim_compression']].copy()
    compression.columns = ['lon', 'lat', 'strain_rate', 'azimuth']
    compression['zero'] = 0

    extension = interp_data_filtered[['lon', 'lat', 'emax', 'azim_extension']].copy()
    extension.columns = ['lon', 'lat', 'strain_rate', 'azimuth']
    extension['zero'] = 0

    ########################################################################################################################
    ############################################# Plot Interpolated strain rates ###########################################
    ########################################################################################################################

    plot_region = [region[0], region[1], region[2], region[3]] # Just in case we need to modify the region for plotting... (not used here)

    ########################################## Plot Mean Maximum Shear Strain Rates ########################################
    print(f"Interpolated mean maximum shear strain rates:")
    fig = pygmt.Figure()
    pygmt.makecpt(cmap=max_shear_cpt, truncate=[0.05, 1], series=[range_max_shear[0], range_max_shear[1]], background='o', output="input_data/cpts/turbo_part.cpt")
    # Read in the Turbo part and prepend white for 0 to 0.01 microstrain/yr (i.e., 0 to 10 nanostrain/yr)
    with open("input_data/cpts/turbo_part.cpt") as file:
        turbo_colors = file.readlines()
    cropping_sting = f"0 white {range_max_shear[0]} white\n"
    turbo_colors.insert(0, cropping_sting)  # Add white from 0 to 0.01 microstrain/yr
    with open(path2_max_shear_cpt, "w") as file:
        file.writelines(turbo_colors)
    fig.basemap(region=plot_region, projection=map_projection, frame='af')
    if apply_mask and mask_polygons is not None:
        shear_grid = mask_grid_with_polygons(shear_grid, mask_polygons) # Apply polygon mask to the grid
    fig.grdimage(grid=shear_grid, cmap=path2_max_shear_cpt, shading=False, nan_transparent=True)

    # Plot masking polygons filled with white
    if plot_mask_polygons and mask_polygons is not None:
        for poly_name, poly_path in mask_polygons.items():
            poly_coords = np.loadtxt(poly_path)
            fig.plot(data=poly_coords, pen="0.1p,white", close=True, fill=mask_fill, transparency=50)

    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,
              area_thresh=4000, resolution='h')
    fig.colorbar(frame=['xaf+lMaximum shear strain rate', 'y+l10@+-9@+/yr'],
                 position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef")
    fig.plot(data=fault_file, pen="0.1p,darkgrey")
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
    
    if save_fig:
        fig.savefig(f"{output_folder}/{output_filenames_prefix}_Mean_Max_Shear_Strain_Rates_Map.pdf", dpi=600)
    fig.show()

    #################################### Plot Mean Dilatation Rates (first invariant) ######################################
    print(f"Interpolated mean dilatation rates (first invariant), (blue: shortening, red: extension):")
    fig = pygmt.Figure()
    pygmt.makecpt(cmap=dilatation_cpt, series=range_dilatation, background='o', output=path2_dilatation_cpt)
    fig.basemap(region=plot_region, projection=map_projection, frame='af')
    if mask_dilatation_map:
        dilatation_grid = mask_grid_with_polygons(dilatation_grid, mask_polygons) # Apply polygon mask to the grid
    fig.grdimage(grid=dilatation_grid, cmap=path2_dilatation_cpt, shading=False, nan_transparent=True)

    # Plot masking polygons filled with white
    if plot_mask_polygons and mask_dilatation_map and mask_polygons is not None:
        for poly_name, poly_path in mask_polygons.items():
            poly_coords = np.loadtxt(poly_path)
            fig.plot(data=poly_coords, pen="0.1p,white", close=True, fill=mask_fill, transparency=50)

    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,
              area_thresh=4000, resolution='h')
    fig.colorbar(frame=['x+lDilatation rate', 'y+l10@+-9@+/yr'],
                 position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+e")
    fig.plot(data=fault_file, pen=faults_pen)

    # Plot interpolated principal strain rate directions
    fig.basemap(region=plot_region, projection=map_projection, frame='af')

    # Plot compressional strains in blue
    fig.velo(
        data=compression[['lon', 'lat', 'zero', 'strain_rate', 'azimuth']], #.assign(scale=0.1),
        spec=f'x{strain_rate_cross_size}c',
        pen="thinnest,navy",
        fill='blue'
    )

    # Plot extensional strains in red
    fig.velo(
        data=extension[['lon', 'lat', 'strain_rate', 'zero', 'azimuth']], #.assign(scale=0.1),
        spec=f'x{strain_rate_cross_size}c',
        pen="thinnest,red2",
        fill='red'
    )

    # Add scale annotation
    strsclon, strsclat = strain_rate_cross_scale_positon
    fig.velo(data=[[strsclon, strsclat, 0, -strain_rate_cross_scale_magnitude, 90]], spec=f'x{strain_rate_cross_size}c',
             pen="thinnest,navy", fill='navy')
    fig.velo(data=[[strsclon, strsclat, strain_rate_cross_scale_magnitude, 0, 90]], spec=f'x{strain_rate_cross_size}c',
             pen="thinnest,red2", fill='red2')
    fig.text(text=f"{strain_rate_cross_scale_magnitude} nanostrain/yr", x=strsclon, y=strsclat,
             font="6p", justify="CB", offset=f"j{strain_rate_cross_label_offset[0]}c/{strain_rate_cross_label_offset[1]}c", fill="white")

    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm+ar")

    # Add tectonic map labels and annotations
    if add_tectonic_map_labels and path2_tectonic_map_labels is not None:
        fig.text(
            textfiles=path2_tectonic_map_labels,
            offset=f"j{tectonic_map_labels_offset[0]}c/{tectonic_map_labels_offset[1]}c",
            fill="white",
            justify=True,
            angle=True,
            font=True,
            transparency=tectonic_map_labels_transparency,
        )
    
    if add_tectonic_map_labels and path2_tectonic_map_labels is not None:
        fig.text(
            textfiles=path2_tectonic_map_labels,
            offset=f"j{tectonic_map_labels_offset[0]}c/{tectonic_map_labels_offset[1]}c",
            fill=None,
            justify=True,
            angle=True,
            font=True,
        )
        
    if save_fig:
        fig.savefig(f"{output_folder}/{output_filenames_prefix}_Mean_Dilatation_Rates_Map.pdf", dpi=600)
    fig.show()

    ################################## Plot Second Invariant of the Strain Rate Tensor #####################################
    print(f"Interpolated mean second invariant of the strain rate tensor:")
    fig = pygmt.Figure()
    pygmt.makecpt(cmap=sec_inv_cpt, truncate=[0.05, 1], series=[range_sec_inv[0], range_sec_inv[1]], background='o', output="input_data/cpts/turbo_part.cpt")
    # Read in the Turbo part and prepend white for 0 to 0.01 microstrain/yr (i.e., 0 to 10 nanostrain/yr)
    with open("input_data/cpts/turbo_part.cpt") as file:
        turbo_colors = file.readlines()
    cropping_sting = f"0 white {range_sec_inv[0]} white\n"
    turbo_colors.insert(0, cropping_sting)  # Add white from 0 to 0.01 microstrain/yr

    with open(path2_sec_inv_cpt, "w") as file:
        file.writelines(turbo_colors)

    fig.basemap(region=plot_region, projection=map_projection, frame='af')
    #if plot_strain_comparison and "EMED" in output_filenames_prefix:
    #    fig.plot(data=tri_mesh_strain_EMED, pen="0.1p", close=True, cmap=True)
    #else:
    if apply_mask and mask_polygons is not None:
        sec_inv_grid = mask_grid_with_polygons(sec_inv_grid, mask_polygons) # Apply polygon mask to the grid
    fig.grdimage(grid=sec_inv_grid, cmap=path2_sec_inv_cpt, shading=False, nan_transparent=True)

    # Plot masking polygons filled with white
    if plot_mask_polygons and mask_polygons is not None:
        for poly_name, poly_path in mask_polygons.items():
            poly_coords = np.loadtxt(poly_path)
            fig.plot(data=poly_coords, pen=None, close=True, fill=mask_fill, transparency=50)

    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,
              area_thresh=4000, resolution='h')
    fig.colorbar(frame=['xaf+lSecond invariant of the strain rate tensor', 'y+l10@+-9@+/yr'],
                 position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef")
    fig.plot(data=fault_file, pen=faults_pen)
    if "EMED" in output_filenames_prefix and plot_strain_comparison:
            with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
                fig.basemap(map_scale=f"g{map_scale_bar_position[0]-3}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
    else:
        with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")


    # Add tectonic map labels and annotations
    if add_tectonic_map_labels and path2_tectonic_map_labels is not None:
        fig.text(
            textfiles=path2_tectonic_map_labels,
            offset=f"j{tectonic_map_labels_offset[0]}c/{tectonic_map_labels_offset[1]}c",
            fill="white",
            justify=True,
            angle=True,
            font=True,
            transparency=tectonic_map_labels_transparency,
        )
    
    if add_tectonic_map_labels and path2_tectonic_map_labels is not None:
        fig.text(
            textfiles=path2_tectonic_map_labels,
            offset=f"j{tectonic_map_labels_offset[0]}c/{tectonic_map_labels_offset[1]}c",
            fill=None,
            justify=True,
            angle=True,
            font=True,
        )

    # --- BEGIN inset: strain rate comparison ---
    if plot_strain_comparison:
        # Load the strain profile data from the provided files
        try:
            data_raw_no_creep = np.loadtxt(strain_profile_file_no_creep)
            #data_smoothed_no_creep = np.loadtxt(strain_profile_file_smoothed_no_creep)
            data_raw_with_creep = np.loadtxt(strain_profile_file_with_creep)
            #data_smoothed_with_creep = np.loadtxt(strain_profile_file_smoothed_with_creep)
        except Exception as e:
            raise ValueError("Error loading strain profile files: " + str(e))
        
        # Plot the first and second boxes if coordinates are provided
        if plot_profile_box:
            if box_coords:
                box1_lon, box1_lat = box_coords
                fig.plot(
                    x=box1_lon,
                    y=box1_lat,
                    pen='0.85p,white',
                )
                if label_coords:
                    label1_lon, label1_lat = label_coords
                    fig.text(
                        text='①',
                        x=label1_lon,
                        y=label1_lat,
                        font='10p,white',
                        offset='-0.2c/0.13c'
                    )
        # If an optional second strain rate file is provided, load it (CSV assumed)
        #if optional_second_strain_rate_file is not None:
        #    data_jgr = np.loadtxt(optional_second_strain_rate_file, delimiter=",")

        # Determine the inset region from the available data
        x_all = np.concatenate([data_raw_no_creep[:, 0], data_raw_with_creep[:, 0]])
        y_all = np.concatenate([data_raw_no_creep[:, 1], data_raw_with_creep[:, 1]])
        #if optional_second_strain_rate_file is not None:
        #    x_all = np.concatenate([x_all, data_jgr[:, 0]])
        #    y_all = np.concatenate([y_all, data_jgr[:, 1]])
        x_min, x_max = np.min(x_all), np.max(x_all)
        y_min, y_max = np.min(y_all), np.max(y_all)
        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)
        inset_region = [x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin]

        # Create the inset in the bottom left (jBL) with width=5c and height=4c
        with pygmt.config(FONT_ANNOT_PRIMARY='9p', FONT_LABEL='9p', MAP_FRAME_PEN='0.5p,black', MAP_TICK_PEN='0.5p,black'):
            # Add transparent rectangles to improve visibility of the scatter plots
            fig.plot(x=45.2, y=35.3, style="r6.5/5.3", fill="white", pen=None, transparency=30)
            # Shift origin for the creeping segment profile
            fig.shift_origin(xshift="14.8c", yshift="0.3c")
            fig.basemap(region=inset_region, projection="X5c/4.1c", frame=['WNse+gwhite', 'xaf+lAcross-fault distance (km)', 'yaf+lSecond Invariant (nanostrain/yr)'])
            # Plot raw data for no-creep as blue crosses
            fig.plot(x=data_raw_no_creep[:, 0], y=data_raw_no_creep[:, 1],
                     style="c0.15c", fill="39/39/43", pen="0.5p,70", transparency=50, label=f'Creep ignored+S0.15c')
            
            fig.text(
                text='① İsmetpaşa',
                x=-55,
                y=700,
                font='10p,black',
                offset='0c/0c'
            )
            # Plot smoothed curve for no-creep as a grey line
            #fig.plot(x=data_smoothed_no_creep[:, 0], y=data_smoothed_no_creep[:, 1],
            #         pen="0.5p,gray")
            # Plot raw data for with-creep as red circles
            fig.plot(x=data_raw_with_creep[:, 0], y=data_raw_with_creep[:, 1],
                     style="c0.15c", fill="250/15/50", pen="0.05p,70", transparency=50, label=f'Creep accounted+S0.15c')
            # Plot smoothed curve for with-creep as a black line
            #fig.plot(x=data_smoothed_with_creep[:, 0], y=data_smoothed_with_creep[:, 1],
            #         pen="0.5p,black")
            # Optionally, plot the second strain rate dataset (e.g., from JGR) as green triangles
            #if optional_second_strain_rate_file is not None:
            #    fig.plot(x=data_jgr[:, 0], y=data_jgr[:, 1],
            #             style="t0.2c", color="green", pen="0.5p,green")
            # Add a short legend in the top right of the inset
            #legend_lines = []
            #legend_lines.append("S 0.2c x blue 0.5p,blue No Creep")
            #legend_lines.append("S 0.2c o red 0.5p,red Creep")
            #if optional_second_strain_rate_file is not None:
            #    legend_lines.append("S 0.2c t green 0.5p,green JGR")
            #legend_spec = "\n".join(legend_lines)
            #fig.legend(spec=legend_spec, position="jTR+o0.1c", box="+gwhite+p0.5p")

            # Add legend
            with pygmt.config(FONT_ANNOT_PRIMARY='7p', FONT_LABEL='7p', MAP_FRAME_PEN='0.5p,black', MAP_TICK_PEN='0.5p,black'):
                fig.legend(position='JTR+jTR', box='+gwhite+p0.5p,black')

    # --- END inset ---

    if save_fig:
        fig.savefig(f"{output_folder}/{output_filenames_prefix}_Second_Invariant_Strain_Rates_Map.pdf", dpi=600)
    fig.show()

    ########################################################################################################################
    ########################################################################################################################
    # Check if 'mean_second_inv' column is present, if so create additional grids and plots for it
    if 'mean_second_inv' in data.columns and 'std_second_inv' in data.columns:
        std_dilatation_grid = create_grid('std_dilatation')
        std_sec_inv_grid = create_grid('std_second_inv')
        std_max_shear_grid = create_grid('std_maxshear')

        if apply_mask and mask_polygons is not None:
            std_dilatation_grid = mask_grid_with_polygons(std_dilatation_grid, mask_polygons) # Apply polygon mask to the grid
            std_sec_inv_grid = mask_grid_with_polygons(std_sec_inv_grid, mask_polygons) # Apply polygon mask to the grid
            std_max_shear_grid = mask_grid_with_polygons(std_max_shear_grid, mask_polygons) # Apply polygon mask to the grid

        # Plot standard deviation of dilatation rates
        print(f"Interpolated 1-sigma standard deviation of dilatation rates:")
        fig = pygmt.Figure()
        fig.basemap(region=region, projection=map_projection, frame='af')
        pygmt.makecpt(cmap=max_shear_cpt, series=range_dilatation_sigma, background='o')
        if apply_mask and mask_polygons is not None:
            std_dilatation_grid = mask_grid_with_polygons(std_dilatation_grid, mask_polygons)
        fig.grdimage(grid=std_dilatation_grid, cmap=True, shading=False, nan_transparent=True)

        # Plot masking polygons filled with white
        if plot_mask_polygons and mask_polygons is not None:
            for poly_name, poly_path in mask_polygons.items():
                poly_coords = np.loadtxt(poly_path)
                fig.plot(data=poly_coords, pen=None, close=True, fill=mask_fill, transparency=50)

        fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,
              area_thresh=4000, resolution='h')
        #fig.colorbar(frame=['x+lDilatation rate 1@~\163@~ uncertainty', 'y+l10@+-9@+/yr'],
                 #position=f"JMR+o0.5c/0c+w{colorbar_width}c+v")
        fig.colorbar(frame=['x+lDilatation rate 1@~\163@~ uncertainty', 'y+l10@+-9@+/yr'],
                 position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef")
        fig.plot(data=fault_file, pen="0.1p,darkgrey")
        with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
        
        if save_fig:
            fig.savefig(f"{output_folder}/{output_filenames_prefix}_Std_Dilatation_Rates_Map.pdf", dpi=600)
        fig.show()

        # Plot standard deviation of second invariant
        print(f"Interpolated 1-sigma standard deviation of the second invariant:")
        fig = pygmt.Figure()
        fig.basemap(region=region, projection=map_projection, frame='af')
        pygmt.makecpt(cmap=sec_inv_cpt, series=range_sec_inv_sigma, background='o')
        if apply_mask and mask_polygons is not None:
            std_sec_inv_grid = mask_grid_with_polygons(std_sec_inv_grid, mask_polygons)
        fig.grdimage(grid=std_sec_inv_grid, cmap=True, shading=False, nan_transparent=True)

        # Plot masking polygons filled with white
        if plot_mask_polygons and mask_polygons is not None:
            for poly_name, poly_path in mask_polygons.items():
                poly_coords = np.loadtxt(poly_path)
                fig.plot(data=poly_coords, pen=None, close=True, fill=mask_fill, transparency=50)

        fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,
              area_thresh=4000, resolution='h')
        #fig.colorbar(frame=['xaf+lSecond invariant 1@~\163@~ uncertainty', 'y+l10@+-9@+/yr'],
        #         position=f"JMR+o0.5c/0c+w{colorbar_width}c+v")
        fig.colorbar(frame=['xaf+lSecond invariant 1@~\163@~ uncertainty', 'y+l10@+-9@+/yr'],
                 position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef")
        fig.plot(data=fault_file, pen="0.1p,darkgrey")
        with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
        
        if save_fig:
            fig.savefig(f"{output_folder}/{output_filenames_prefix}_Std_Second_Invariant_Map.pdf", dpi=600)
        fig.show()

        # Plot standard deviation of Mean Max. Shear Strain Rates
        print(f"Interpolated 1-sigma standard deviation of the mean max. shear strain rates:")
        fig = pygmt.Figure()
        fig.basemap(region=region, projection=map_projection, frame='af')
        pygmt.makecpt(cmap=max_shear_cpt, truncate=[0.05, 1], series=range_max_shear_sigma, background='o')
        if apply_mask and mask_polygons is not None:
            std_max_shear_grid = mask_grid_with_polygons(std_max_shear_grid, mask_polygons)
        fig.grdimage(grid=std_max_shear_grid, cmap=True, shading=False, nan_transparent=True)

        # Plot masking polygons filled with white
        if plot_mask_polygons and mask_polygons is not None:
            for poly_name, poly_path in mask_polygons.items():
                poly_coords = np.loadtxt(poly_path)
                fig.plot(data=poly_coords, pen=None, close=True, fill=mask_fill, transparency=50)

        fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,
              area_thresh=4000, resolution='h')
        fig.colorbar(frame=['xaf+lMaximum shear strain rate 1@~\163@~ uncertainty', 'y+l10@+-9@+/yr'],
                 position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef")
        #fig.colorbar(frame=['xaf+lMaximum shear strain rate 1@~\163@~ uncertainty', 'y+l10@+-9@+/yr'],
        #         position=f"JMR+o0.5c/0c+w{colorbar_width}c+v")
        fig.plot(data=fault_file, pen="0.1p,darkgrey")
        with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
        
        if save_fig:
            fig.savefig(f"{output_folder}/{output_filenames_prefix}_Std_Mean_Max_Shear_Strain_Rates_Map.pdf", dpi=600)
        fig.show()
    ########################################################################################################################
    ########################################################################################################################

    ############################# Plot interpolated strain rate style and focal mechanisms #################################
    print(f"Interpolated strain rate style (red: normal, green: strike-slip, blue: reverse):")
    pygmt.makecpt(cmap=strain_rate_style_cpt, series=[range_strain_rate_style[0], range_strain_rate_style[1]], background='o', output=path2_strain_rate_style_cpt, reverse=True, transparency=30)
    fig = pygmt.Figure()
    fig.basemap(region=plot_region, projection=map_projection, frame='af')
    if apply_mask and mask_polygons is not None:
        sr_style_grid = mask_grid_with_polygons(sr_style_grid, mask_polygons) # Apply polygon mask to the grid
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.grdimage(grid=sr_style_grid,
                    cmap=path2_strain_rate_style_cpt, shading=False, nan_transparent=True, transparency=30)
        
    # Plot masking polygons filled with white
    if plot_mask_polygons and mask_polygons is not None:
        for poly_name, poly_path in mask_polygons.items():
            poly_coords = np.loadtxt(poly_path)
            fig.plot(data=poly_coords, pen=None, close=True, fill=mask_fill, transparency=50)

    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,
              area_thresh=4000, resolution='h')
    fig.colorbar(frame='af+lStrain rate style -(@~\\145@~@-1@- + @~\\145@~@-2@-) / (@~\\174\\145@~@-1@-@~\\174@~ + @~\\174\\145@~@-2@-@~\\174@~)',
                 position=f"JMR+o0.5c/0c+w{colorbar_width}c+v")
    fig.plot(data=fault_file, pen="0.1p,darkgrey")
    if plot_moment_tensors:
        fig.meca(
            spec=focal_mechanism_file,
            scale=f"{moment_tensors_scale}c+f0",
            convention="mt",
            component="dc",
            pen="0.2p,black",
            transparency=30,
        )
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
    
    # Let's add the label in the upper-right corner only for the large-scale map
    if "Alpides" in output_filenames_prefix:
        fig.text(
            x=plot_region[1] - 1,  # Adjust near the right edge
            y=plot_region[3] - 0.5,  # Adjust near the top edge
            text="GNSS-derived",
            font="10p,Helvetica,black",
            fill="white",
            justify="TR",
            pen="0.1p,black"
        )

    if save_fig:
        fig.savefig(f"{output_folder}/{output_filenames_prefix}_Strain_Rate_Style_Map.pdf", dpi=600)
    fig.show()

    # I have considered the implications and concluded that plotting the interpolated Kostrov results after two consecutive interpolation steps is acceptable. 
    # After comparing with the original grid, the main features remain well-preserved, albeit with a degree of smoothing introduced by the interpolation process.
    # However, the key point I want to convey does not change (i.e., I did not see any concerning patterns, like strain rate styles changing from normal to reverse, etc.).
    if include_kostrov and not valid_kostrov_data.empty:
            ############################# Plot interpolated strain rate style and focal mechanisms #################################
        print(f"Interpolated strain rate style from Kostrov-summed GCMT (red: normal, green: strike-slip, blue: reverse):")
        pygmt.makecpt(cmap=strain_rate_style_cpt, series=[range_strain_rate_style[0], range_strain_rate_style[1]], background='o', output=path2_strain_rate_style_cpt, reverse=True, transparency=30)
        fig = pygmt.Figure()
        fig.basemap(region=plot_region, projection=map_projection, frame='af')
        if apply_mask and mask_polygons is not None:
            kostrov_sr_style_grid = mask_grid_with_polygons(kostrov_sr_style_grid, mask_polygons) # Apply polygon mask to the grid
        with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.grdimage(grid=kostrov_sr_style_grid,
                        cmap=path2_strain_rate_style_cpt, shading=False, nan_transparent=True, transparency=30)
            
        # Plot masking polygons filled with white
        if plot_mask_polygons and mask_polygons is not None:
            for poly_name, poly_path in mask_polygons.items():
                poly_coords = np.loadtxt(poly_path)
                fig.plot(data=poly_coords, pen=None, close=True, fill=mask_fill, transparency=50)

        fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,
                area_thresh=4000, resolution='h')
        fig.colorbar(frame='af+lStrain rate style -(@~\\145@~@-1@- + @~\\145@~@-2@-) / (@~\\174\\145@~@-1@-@~\\174@~ + @~\\174\\145@~@-2@-@~\\174@~)',
                    position=f"JMR+o0.5c/0c+w{colorbar_width}c+v")
        fig.plot(data=fault_file, pen="0.1p,darkgrey")
        if plot_moment_tensors:
            fig.meca(
                spec=focal_mechanism_file,
                scale=f"{moment_tensors_scale}c+f0",
                convention="mt",
                component="dc",
                pen="0.2p,black",
                transparency=30,
            )
        with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
        
        # Let's add the label in the upper-right corner
        fig.text(
            x=plot_region[1] - 1,  # Adjust near the right edge
            y=plot_region[3] - 0.5,  # Adjust near the top edge
            text="Kostrov-summed GCMT",
            font="10p,Helvetica,black",
            fill="white",
            justify="TR",
            pen="0.1p,black"
        )

        if save_fig:
            fig.savefig(f"{output_folder}/{output_filenames_prefix}_Strain_Rate_Style_Map_Kostrov.pdf", dpi=600)
        fig.show()

    # I'm curious to see the difference between the two strain rate styles, so I will plot the difference between the two interpolated strain rate styles.
    if include_kostrov and not valid_kostrov_data.empty:
        print(f"Interpolated strain rate style difference (GNSS-Kostrov). red: Kostrov > GNSS, blue: GNSS > Kostrov:")
        # Let's print the median and MAD of the strain rate style difference to be reported in the paper
        # Convert to numeric and drop any NaNs
        sr_diff = pd.to_numeric(interp_data['sr_style_diff'], errors='coerce').dropna()

        # Median of the difference
        median_gnss_kostrov = sr_diff.median()

        # Median Absolute Deviation (MAD)
        mad_gnss_kostrov = (sr_diff - median_gnss_kostrov).abs().median()

        # Print results
        print(f"Median difference: {median_gnss_kostrov:.1f}")
        print(f"Median Absolute Deviation (MAD): {mad_gnss_kostrov:.1f}")
        print("-" * 80)

        # Histogram-based mode estimation of strain rate style differences
        data_range = (-1, 1)  # Range of strain rate style differences
        num_bins = 18 # Number of bins for the histogram
        hist, bin_edges = np.histogram(sr_diff, bins=num_bins, range=data_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mode_hist = bin_centers[np.argmax(hist)]

        # Compute median absolute deviation (MAD) from the histogram mode
        mad_from_mode_hist = np.median(np.abs(sr_diff - mode_hist))

        print(f"Mode ({num_bins} bins) \u00B1 MAD strain rate style difference: {mode_hist:.1f} \u00B1 {mad_from_mode_hist:.1f} degrees")
        print("-" * 80)

        # Define the "vik" colormap with the range [-2, 2]
        pygmt.makecpt(cmap="vik", series=[-2, 2], reverse = True, output="input_data/cpts/vik.cpt")

        # Plot the difference grid
        fig = pygmt.Figure()
        fig.basemap(region=plot_region, projection=map_projection, frame="af")
        if apply_mask and mask_polygons is not None:
            sr_style_diff_grid = mask_grid_with_polygons(sr_style_diff_grid, mask_polygons) # Apply polygon mask to the grid
        with pygmt.config(FONT_ANNOT_PRIMARY="8p", FONT_LABEL="8p"):
            fig.grdimage(
                grid=sr_style_diff_grid,
                cmap="input_data/cpts/vik.cpt",
                shading=False,
                nan_transparent=True,  # Ensure NaNs are displayed as transparent
                transparency=30
            )

        # Plot masking polygons filled with white
        if plot_mask_polygons and mask_polygons is not None:
            for poly_name, poly_path in mask_polygons.items():
                poly_coords = np.loadtxt(poly_path)
                fig.plot(data=poly_coords, pen=None, close=True, fill=mask_fill, transparency=50)

        # Add coastlines and other map elements
        fig.coast(water=ocean_fill_color, shorelines=shorelines_pen, area_thresh=4000, resolution="h")

        # Add a colorbar for the difference
        fig.colorbar(
            frame="af+l Signed strain rate style difference (GNSS-Kostrov)",
            position=f"JMR+o0.5c/0c+w{colorbar_width}c+v",
            cmap="input_data/cpts/vik.cpt",
        )

        # I almost forgot to add the fault lines to the plot...
        fig.plot(data=fault_file, pen="0.1p,darkgrey")

        # Add a label to the map
        fig.text(
            x=plot_region[1] - 1,  # Adjust near the right edge
            y=plot_region[3] - 0.5,  # Adjust near the top edge
            text="GNSS - Kostrov",
            font="10p,Helvetica,black",
            fill="white",
            justify="TR",
            pen="0.1p,black"
        )

        # Here I add a histogram showing the distribution of the strain rate style differences
        if plot_sks_kostrov_histogram:
            # Let's define the histogram data removing any nans from the dataset
            histogram_data = pd.to_numeric(interp_data['sr_style_diff'], errors='coerce').values.tolist()
            histogram_data = [x for x in histogram_data if not np.isnan(x)] # Remove NaNs
            # Some debugging here... printing what is inside the histogram_data
            #print(histogram_data)

            if plot_box_for_histogram:
                fig.plot(x=box_histogram_position[0], y=box_histogram_position[1], style=box_histogram_style, fill=box_histogram_fill, pen=None, transparency=box_histogram_transparency)

            with pygmt.config(FONT_ANNOT_PRIMARY=hist_annot_primary, FONT_LABEL=hist_font_label, MAP_FRAME_PEN='0.3p,black', MAP_TICK_PEN='0.3p,black'):
                # Plot the histogram
                fig.shift_origin(xshift=f"{hist_shift[0]}c", yshift=f"{hist_shift[1]}c")
                fig.histogram(
                    data=histogram_data,
                    region=[-2, 2, 0, 0],  # Set the region for the histogram: x-range (-90 to 90), y-range (0 to 15)
                    projection=hist_projection,  # Set the projection for the histogram
                    frame=hist_frame,  # Add labels to the axes
                    series=[-2, 2, 0.2],  # Histogram range (x-range) and bin interval
                    cmap="input_data/cpts/vik.cpt",  # Use the same custom colormap for the histogram
                    pen="0.2p,black",  # Outline the histogram bars
                    histtype=1,  # Frequency percent
                )

        # Saving the figure
        if save_fig:
            fig.savefig(f"{output_folder}/{output_filenames_prefix}_Strain_Rate_Style_Difference.pdf", dpi=600)
        fig.show()

    ############################# Plot interpolated azimuths for principal compressive strain ###############################
    # debug: print min and max values of azim_shortening
    #print(f"Minimum azimuth of maximum shortening: {interp_data['azim_shortening'].min()}")
    #print(f"Maximum azimuth of maximum shortening: {interp_data['azim_shortening'].max()}")
    print(f"Interpolated azimuths for the maximum shortening strain:")
    fig = pygmt.Figure()
    fig.basemap(region=plot_region, projection=map_projection, frame='af')
    #if apply_mask and mask_polygons is not None:
    #    azim_shortening_grid = mask_grid_with_polygons(azim_shortening_grid, mask_polygons) # Apply polygon mask to the grid
    #fig.grdimage(grid=azim_shortening_grid,
    #             cmap=path2_azimuth_diff_cpt, shading=False, nan_transparent=True, transparency=0)
    
    # Plot masking polygons filled with white
    #if plot_mask_polygons and mask_polygons is not None:
    #    for poly_name, poly_path in mask_polygons.items():
    #            poly_coords = np.loadtxt(poly_path)
    #            fig.plot(data=poly_coords, pen=None, close=True, fill=mask_fill, transparency=50)

    # Plot the azimuth differences using the custom colormap as a scatter plot
    pygmt.makecpt(cmap=azimuth_diff_cpt, series=[range_azimuth_max_shortening_rate[0], range_azimuth_max_shortening_rate[1]], background='o', reverse=True, transparency=0) # output=path2_azimuth_diff_cpt,

    fig.plot(
        x=interp_data_filtered.lon,
        y=interp_data_filtered.lat,
        style=scatter_style,
        fill=interp_data_filtered.azim_shortening,
        cmap=True,
        pen=None
    )

    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,
              area_thresh=4000, resolution='h')
    fig.colorbar(frame='a15f15+lAzimuth of maximum shortening rate (@~\\145@~@-2@-)',
                 position=f"JMR+o0.5c/0c+w{colorbar_width}c+v")
    fig.plot(data=fault_file, pen="0.1p,darkgrey")
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
    
    # Let's add the label in the upper-right corner
    fig.text(
        x=plot_region[1] - 1,  # Adjust near the right edge
        y=plot_region[3] - 0.5,  # Adjust near the top edge
        text="GNSS-derived",
        font="10p,Helvetica,black",
        fill="white",
        justify="TR",
        pen="0.1p,black"
    )

    if save_fig:
        fig.savefig(f"{output_folder}/{output_filenames_prefix}_Strain_Azimuths_Map.pdf", dpi=600)
    fig.show()

    return data, interp_data
########################################################################################################################
########################################################################################################################
# Helper function to mask an xarray grid with user-defined polygons. I would have used GMT's grdmask, but it is not available in PyGMT
def mask_grid_with_polygons(grid: xr.DataArray, mask_polygons: dict) -> xr.DataArray:
    """
    Masks an xarray grid by setting grid cells that fall inside any user-defined polygons to NaN.
    
    Parameters:
        grid (xr.DataArray): Input grid with coordinates "x" (longitude) and "y" (latitude).
        mask_polygons (dict): Dictionary with keys as polygon names and values as file paths.
                              Each file should contain two-column (lon, lat) coordinates defining a closed polygon.
    
    Returns:
        xr.DataArray: The grid masked such that cells within any polygon are set to NaN.
    """
    if mask_polygons is not None:
        for poly_name, poly_path in mask_polygons.items():
            # Read the polygon coordinates (assumes two-column: lon, lat)
            poly_coords = np.loadtxt(poly_path)
            # Create a Shapely Polygon from the coordinates
            polygon = shapely.geometry.Polygon(poly_coords)
            
            # Extract grid coordinate arrays (assumes coordinates "x" and "y")
            lon = grid.coords["x"].values
            lat = grid.coords["y"].values
            
            # Create a meshgrid matching the grid dimensions.
            grid_lon, grid_lat = np.meshgrid(lon, lat)
            
            # Build a boolean mask where True indicates that the grid cell center is inside the polygon.
            inside_mask = shapely.vectorized.contains(polygon, grid_lon, grid_lat)
            
            # Use the xarray .where() method to assign NaN to grid cells inside the polygon.
            grid = grid.where(~inside_mask)
    return grid

########################################################################################################################
########################################################################################################################
# Function to plot SHMAX interpolated into the same grid as the strain rates
def plot_SHmax_with_moho_and_strain(
    moho_file_path, 
    wsm_file_path, 
    strain_rates_grid, 
    fault_file, 
    output_file, 
    strain_grid_spacing=0.1, 
    region=[-20, 125, 5, 60],
    map_scale_bar_length_km=1000,  # Length of the map scale bar in km
    map_scale_bar_position=[65, 10],  # Position (lon, lat) in degrees
    colorbar_width=8.5,  # Width of the colorbar in cm
    ocean_fill_color='white',  # Color for the ocean fill
    shorelines_pen="0.1p,black",  # Pen for the shorelines
    mask_below_sec_inv_value=0.01,  # Mask values below this second invariant value in microstrain/yr
    line_scale=0.1,  # Scale for the SHmax lines
    wsm_subsampling_for_plotting=5,  # Decimation factor for the SHmax data
    scatter_style="c0.18c",
    save_fig=True, # Save the figure to a file
    # mask_polygons=None  # Dictionary of polygon names to file paths for masking (optional)
):
    """
    Function to plot SHmax azimuths, filtered by Moho depths, on a map of strain rates. 
    The azimuths will be interpolated and visualised as anisotropy bars on top of a strain rate grid.
    
    Parameters:
    - moho_file_path (str): Path to the Moho depths CSV file.
    - wsm_file_path (str): Path to the World Stress Map (WSM) data file.
    - strain_rates_grid (pd.DataFrame): DataFrame containing strain rates data with columns 'lon', 'lat', and 'sec_inv'.
    - fault_file (str): Path to the fault line file.
    - output_file (str): Output path for the generated figure.
    - strain_grid_spacing (float): Spacing for the PyGMT surface interpolation.
    - region (list): List defining the bounding box region [west, east, south, north].
    """

    # Load Moho data
    moho_data = pd.read_csv(moho_file_path, header=None, names=['Lon', 'Lat', 'depth'])
    moho_data = moho_data.dropna().astype(float)
    
    # Crop Moho data to the specified region
    moho_data = moho_data[(moho_data['Lon'] > region[0]) & (moho_data['Lon'] < region[1]) & 
                          (moho_data['Lat'] > region[2]) & (moho_data['Lat'] < region[3])]

    # Load WSM data
    wsm_data = pd.read_csv(wsm_file_path, encoding='Latin-1', low_memory=False)
    wsm_data = wsm_data[['LON', 'LAT', 'DEPTH', 'AZI', 'QUALITY']]
    wsm_data = wsm_data[wsm_data['QUALITY'].isin(['A', 'B', 'C'])].dropna()

    # Interpolation grid for Moho data
    moho_points = np.array([moho_data['Lon'], moho_data['Lat']]).T
    moho_depths = moho_data['depth'].values

    # Interpolate Moho depth at WSM points
    wsm_points = np.array([wsm_data['LON'], wsm_data['LAT']]).T
    wsm_data['moho_depth'] = griddata(moho_points, moho_depths, wsm_points, method='linear')

    # Filter WSM data where depth is less than Moho depth
    wsm_data = wsm_data[wsm_data['DEPTH'] < wsm_data['moho_depth']].dropna(subset=['moho_depth'])

    # Remove entries with 999 values in the wsm_data
    wsm_data = wsm_data[wsm_data['AZI'] != 999]

    # Crop WSM data to the region
    wsm_data = wsm_data[(wsm_data['LON'] > region[0]) & (wsm_data['LON'] < region[1]) & 
                        (wsm_data['LAT'] > region[2]) & (wsm_data['LAT'] < region[3])]

    # Compute eastward and northward components of the azimuth
    wsm_data['eastward'] = np.sin(np.radians(wsm_data['AZI']))  # X component
    wsm_data['northward'] = np.cos(np.radians(wsm_data['AZI']))  # Y component

    # Subsample WSM data every 5th row (just for plotting/visualisation purposes). For interpolation and analysis I use the full dataset
    wsm_data_subsampled = wsm_data.iloc[::wsm_subsampling_for_plotting, :].reset_index(drop=True)

    # Convert azimuths to -90 to 90 degrees
    wsm_data['sh_max_azim'] = np.where(
        wsm_data['AZI'] > 90, 
        wsm_data['AZI'] - 180, 
        wsm_data['AZI']
    )

    # Interpolate SHmax azimuths to strain rates grid. 
    # I should have probably interpolated the components of the azimuths (northward and eastward) instead of the azimuths directly
    # to then compute the interpolated azimuths from the arctan2 of the components' ratio. 
    points = np.array([wsm_data['LON'], wsm_data['LAT']]).T
    #sh_max_azim_interp = griddata(
    #    points, 
    #    wsm_data['sh_max_azim'], 
    #    (strain_rates_grid['lon'], strain_rates_grid['lat']), 
    #    method='linear'
    #)

# interpolate eastward and northward components of the azimuths to the strain rates grid
    eastward_interp = griddata(points, wsm_data['eastward'], (strain_rates_grid['lon'], strain_rates_grid['lat']), method='linear')
    northward_interp = griddata(points, wsm_data['northward'], (strain_rates_grid['lon'], strain_rates_grid['lat']), method='linear')
    # Compute the interpolated azimuths from the components
    # Avoid division by zero
    eastward_interp = np.where(eastward_interp == 0, 1e-10, eastward_interp)
    sh_max_azim_interp = np.degrees(np.arctan2(eastward_interp, northward_interp))
    # Convert azimuths to -90 to 90 degrees
    sh_max_azim_interp = np.where(
        sh_max_azim_interp > 90,
        sh_max_azim_interp - 180,
        sh_max_azim_interp
    )

    # Assign the interpolated values to the grid
    strain_rates_grid['sh_max_azim'] = sh_max_azim_interp

    # Mask areas with low strain
    mask_low_strain = strain_rates_grid['sec_inv'] < mask_below_sec_inv_value
    strain_rates_grid.loc[mask_low_strain, 'sh_max_azim'] = np.nan
    strain_rates_grid = strain_rates_grid.dropna(subset=['sh_max_azim'])

    # Interpolate SHmax azimuths to a regular grid
    #sh_max_azim_grid = pygmt.surface(
    #    data=strain_rates_grid[['lon', 'lat', 'sh_max_azim']], 
    #    spacing=strain_grid_spacing, 
    #    region=region, 
    #    maxradius='70k'
    #)

    #sh_max_azim_grid = mask_grid_with_polygons(sh_max_azim_grid, mask_polygons) # Apply polygon mask to the grid

    # Plot the SHmax azimuth map
    fig = pygmt.Figure()
    fig.basemap(region=region, projection='M20c', frame='af')
    pygmt.makecpt(cmap="romaO", series=[-90, 90], background='o', transparency=0, reverse=True) #output="input_data/cpts/SHmax_azim.cpt",

    # Plot the interpolated SHmax azimuths
    #fig.grdimage(grid=sh_max_azim_grid, cmap="input_data/cpts/SHmax_azim.cpt", shading=False, nan_transparent=True, transparency=0)
    fig.plot(
        x=strain_rates_grid.lon,
        y=strain_rates_grid.lat,
        style=scatter_style,
        fill=strain_rates_grid.sh_max_azim,
        cmap=True,
        pen=None
    )

    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen, area_thresh=4000, resolution='h')

    # Plot fault lines
    fig.plot(data=fault_file, pen="0.1p,darkgrey")

    # Plot SHmax azimuth bars from subsampled WSM data
    fig.velo(
        data=wsm_data_subsampled[['LON', 'LAT', 'eastward', 'northward', 'AZI']],  
        spec="n1",  # Anisotropy bars
        scale=line_scale,  # Scale factor
        pen="4p,black",  # Line width and color
    )

    # Add colorbar and scale bar
    fig.colorbar(frame="a15f15+lAzimuth of maximum horizontal compressive stress (S@-Hmax@-)", position=f"JMR+o0.5c/0c+w{colorbar_width}c+v")
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
    
    # Let's add the label in the upper-right corner
    fig.text(
        x=region[1] - 1,  # Adjust near the right edge
        y=region[3] - 0.5,  # Adjust near the top edge
        text="WSM S@-Hmax@-",
        font="10p,Helvetica,black",
        fill="white",
        justify="TR",
        pen="0.1p,black"
    )

    # Save and show the plot
    if save_fig:
        fig.savefig(output_file, dpi=600)
    fig.show()

    return wsm_data
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to compute and plot vorticity rates based on an input GNSS velocity field

def compute_and_plot_vorticity_from_GNSS_data(
    velocity_file_path, # Path to the GNSS velocity field file
    fault_file, # Path to the fault file for plotting
    origin, # [lon, lat]
    region, # [west, east, south, north]
    grid_spacing=100,  # in kilometers
    rotation_cpt_file="input_data/cpts/rotation_rates.cpt", # Path to the output CPT file for rotation rates
    output_file="output_data/figures/Alpides_vorticity.pdf", # Path to save the output PDF plot
    map_scale_bar_length_km = 1000, # Length of the map scale bar in kilometers
    map_scale_bar_position = [65, 10], # Position of the map scale bar in [lon, lat]
    colorbar_width = 8.5, # Width of the colorbar in centimeters
    ocean_fill_color = 'white',  # Color for the ocean fill
    shorelines_pen = "0.1p,black",  # Pen for the shorelines
    colormap_series=[-0.1, 0.1],  # Series for the CPT file
    mask_polygons=None,  # Pass the mask polygons dictionary
    plot_mask_polygons=False,  # Plot the mask polygons
    mask_fill='white',  # Fill color for the mask polygons
    map_projection='M20c',  # Map projection
    save_fig = True, # Save the figure to a file
):
    """
    Function to compute and plot rotation rates (vorticity) based on GNSS velocities,
    interpolate the velocities, compute velocity gradients, and calculate vorticity.

    Parameters:
    - velocity_file_path (str): Path to the GNSS velocity field file.
    - fault_file (str): Path to the fault file for plotting.
    - origin (list): Origin for the local coordinate system in [Lon, Lat].
    - region (list): Geographic region in [west, east, south, north] for plotting and gridding.
    - grid_spacing (float): Grid spacing in kilometers.
    - rotation_cpt_file (str): Path to the CPT file for rotation rates.
    - output_file (str): Path to save the output PDF plot.
    """
    
    # Load the GNSS velocity field and add velocity magnitude
    velocity_data = pd.read_csv(velocity_file_path, delimiter='\t')
    velocity_data = velocity_data[['Lon', 'Lat', 'E.vel', 'N.vel', 'E.sig.scaled', 'N.sig.scaled']]
    velocity_data.columns = ['Lon', 'Lat', 'Ve', 'Vn', 'Se', 'Sn']
    velocity_data['Vmag'] = np.sqrt(velocity_data['Ve']**2 + velocity_data['Vn']**2)
    
    # Convert longitude and latitude to local Cartesian coordinates
    lon_lat = np.vstack([velocity_data['Lon'].values, velocity_data['Lat'].values])
    xy_coords = llh2local(lon_lat, origin)
    velocity_data['X'] = xy_coords[0, :]
    velocity_data['Y'] = xy_coords[1, :]

    # Define and convert region corners to Cartesian coordinates
    lon_corners = [region[0], region[1], region[1], region[0]]
    lat_corners = [region[2], region[2], region[3], region[3]]
    x_corners, y_corners = llh2local(np.array([lon_corners, lat_corners]), origin)

    # Define grid in Cartesian coordinates based on region and grid spacing
    x_min, x_max = x_corners.min(), x_corners.max()
    y_min, y_max = y_corners.min(), y_corners.max()
    x_grid = np.arange(x_min, x_max, grid_spacing)
    y_grid = np.arange(y_min, y_max, grid_spacing)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    # Interpolate East and North velocities onto the Cartesian grid
    points = np.vstack((velocity_data['X'], velocity_data['Y'])).T
    Ve_interp = griddata(points, velocity_data['Ve'], (x_mesh, y_mesh), method='linear')
    Vn_interp = griddata(points, velocity_data['Vn'], (x_mesh, y_mesh), method='linear')

    # Compute velocity magnitudes and gradients
    dVe_dy = np.gradient(Ve_interp, grid_spacing, axis=0)  # Gradient of Ve w.r.t. y
    dVn_dx = np.gradient(Vn_interp, grid_spacing, axis=1)  # Gradient of Vn w.r.t. x

    # Compute rotational rates
    rotation_rates = 0.5 * (dVe_dy - dVn_dx)
    
    # Mask for grid points more than 100 km away from the nearest GNSS observation
    gnss_coords = points  # GNSS data in (X, Y)
    tree = cKDTree(gnss_coords)
    grid_points = np.column_stack([x_mesh.flatten(), y_mesh.flatten()])
    distances, _ = tree.query(grid_points, k=1)
    distance_mask = distances.reshape(x_mesh.shape) > 100  # Mask points more than 100 km away

    # Apply mask to rotational rates
    rotation_rates_masked = np.copy(rotation_rates)
    rotation_rates_masked[distance_mask] = np.nan  # Set to NaN for points more than 100 km away

    # Convert local Cartesian coordinates back to lon/lat for plotting
    xy_grid = np.vstack([x_mesh.flatten(), y_mesh.flatten()])
    lon_lat_grid = local2llh(xy_grid, origin)
    lon_mesh_grid = lon_lat_grid[0, :].reshape(x_mesh.shape)
    lat_mesh_grid = lon_lat_grid[1, :].reshape(x_mesh.shape)

    # Flatten data for PyGMT surface
    lon_flat = lon_mesh_grid.flatten()
    lat_flat = lat_mesh_grid.flatten()
    rot_rate_flat = rotation_rates_masked.flatten()

    # Remove all the observations outside the region
    mask = (lon_flat >= region[0]) & (lon_flat <= region[1]) & (lat_flat >= region[2]) & (lat_flat <= region[3])
    lon_flat = lon_flat[mask]
    lat_flat = lat_flat[mask]
    rot_rate_flat = rot_rate_flat[mask]

    # Create a grid for rotation rates using PyGMT
    rotation_rates_grid = pygmt.surface(
        data=np.column_stack([lon_flat, lat_flat, rot_rate_flat]),
        spacing=0.1,
        region=region,
        #maxradius=grid_spacing # Noot needed anymore because I gridded the data 
    )

    # Create a custom CPT for the rotation rates
    pygmt.makecpt(cmap="polar", series=colormap_series, background='o', output=rotation_cpt_file)

    # Plot the results using PyGMT
    fig = pygmt.Figure()
    fig.basemap(region=region, projection=map_projection, frame='af')
    fig.grdimage(grid=rotation_rates_grid, cmap=rotation_cpt_file, shading=False, nan_transparent=False)
    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen, area_thresh=4000, resolution='h')

    # Plot masking polygons filled with white
    if plot_mask_polygons and mask_polygons is not None:
        for poly_name, poly_path in mask_polygons.items():
                poly_coords = np.loadtxt(poly_path)
                fig.plot(data=poly_coords, pen=None, close=True, fill=mask_fill, transparency=50)

    # Add fault traces
    fig.plot(data=fault_file, pen="0.1p,darkgrey")
    
    # Add colorbar
    fig.colorbar(frame="af+lVorticity (@~m@~strain/yr)", position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+e")
    
    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")

    # Save the plot
    if save_fig:
        fig.savefig(output_file, dpi=600)
    fig.show()
########################################################################################################################
########################################################################################################################

# Function to compute azimuthal differences between max. horizontal stress and max. shortening strain rate directions
########################################################################################################################
########################################################################################################################

def compute_azimuth_differences(
    strain_rates,
    wsm_data,
    RADIUS_THRESHOLD=50,          # kilometers
    STRAIN_RATE_THRESHOLD=0.01,   # microstrain/yr (10 nanostrain/yr)
    GRID_SIZE=1.0,                # degrees for grid spacing (coarse grid size to search nearby points)
    min_SHmax_obserrvations=2,     # minimum number of SHmax observations required to compute the azimuth difference
    mask_polygons=None
):
    """
    Compute the azimuth differences between the maximum shortening rate directions
    from strain rate data and the median maximum horizontal stress azimuths from stress data.

    Parameters:
    - strain_rates: DataFrame containing strain rate data with columns 'lon', 'lat', 'azim_shortening', 'sec_inv'.
    - wsm_data: DataFrame containing stress data with columns 'LON', 'LAT', 'sh_max_azim'.
    - RADIUS_THRESHOLD: Distance in kilometers to search for nearby stress stations (default is 50 km).
    - STRAIN_RATE_THRESHOLD: Minimum strain rate (second invariant) to consider (default is 0.01 microstrain/yr).
    - GRID_SIZE: Grid size in degrees for grouping stress data (default is 1.0 degrees).

    Returns:
    - DataFrame with columns 'lon', 'lat', 'dif_azim' representing the azimuth differences.
    """

    # Create a DataFrame with lon, lat, second invariant, and azimuth of the maximum shortening direction
    strain_rate_data = pd.DataFrame({
        'lon': strain_rates['lon'],
        'lat': strain_rates['lat'],
        'azim_shortening': strain_rates['azim_shortening'],  # azimuth of the maximum shortening rate direction (-90 to 90 degrees)
        'sec_inv': strain_rates['sec_inv']
    })

    # Create a DataFrame with lon, lat, and azimuth of the stress observations
    stress_data = pd.DataFrame({
        'lon': wsm_data['LON'],
        'lat': wsm_data['LAT'],
        'azim': wsm_data['sh_max_azim']  # azimuth of SHmax between -90 and 90 degrees
    })

    # Create a dictionary to hold stress data by grid cell
    stress_dict = defaultdict(list)

    # Populate the stress dictionary with stress stations grouped by their grid cell
    for _, stress_row in stress_data.iterrows():
        grid_key = get_grid_key(stress_row['lon'], stress_row['lat'], GRID_SIZE)
        stress_dict[grid_key].append(stress_row)

    # Prepare an empty list to collect the azimuth differences
    diff_azim_geod_stress_list = []

    # Loop over each strain rate observation
    for i, strain_row in strain_rate_data.iterrows():
        strain_lon = strain_row['lon']
        strain_lat = strain_row['lat']
        strain_azim = strain_row['azim_shortening']

        # Check if the second invariant is greater than or equal to the threshold
        if strain_row['sec_inv'] >= STRAIN_RATE_THRESHOLD:

            # Find stress observations within the radius threshold
            nearby_stress_data = find_nearby_stations(
                strain_lon,
                strain_lat,
                GRID_SIZE,
                RADIUS_THRESHOLD,
                stress_dict
            )

            # Check if there are more than min_SHmax_obserrvations stress observations nearby
            if len(nearby_stress_data) >= min_SHmax_obserrvations:
                # Compute the median stress azimuth
                median_stress_azim = np.median(nearby_stress_data)

                # Compute the azimuth difference (strain azim - median stress azim)
                azim_difference = strain_azim - median_stress_azim

                # Adjust the azimuth difference to be within -90 to +90 degrees
                azim_difference = np.where(azim_difference > 90, azim_difference - 180, azim_difference)
                azim_difference = np.where(azim_difference < -90, azim_difference + 180, azim_difference)

                # Collect the result in the list
                diff_azim_geod_stress_list.append({
                    'lon': strain_lon,
                    'lat': strain_lat,
                    'dif_azim': azim_difference
                })
            else:
                # If fewer than 2 stress data points are found within the radius, assign NaN
                diff_azim_geod_stress_list.append({
                    'lon': strain_lon,
                    'lat': strain_lat,
                    'dif_azim': np.nan
                })
        else:
            # If the second invariant is below the threshold, assign NaN
            diff_azim_geod_stress_list.append({
                'lon': strain_lon,
                'lat': strain_lat,
                'dif_azim': np.nan
            })

    # Convert the list to a DataFrame
    diff_azim_geod_stress = pd.DataFrame(diff_azim_geod_stress_list)

    # Merge the two DataFrames on 'lon' and 'lat' to associate azimuth differences with second invariant values
    merged_df = pd.merge(diff_azim_geod_stress, strain_rates, on=['lon', 'lat'], how='inner')

    # Select only relevant columns
    merged_df = merged_df[['lon', 'lat', 'dif_azim', 'sec_inv']]

    # Drop NaN values from the DataFrame
    merged_df = merged_df.dropna().reset_index(drop=True)

    # Remove NaN values before returning
    diff_azim_geod_stress = diff_azim_geod_stress.dropna().reset_index(drop=True)

    # Apply the polygon mask to the interpolated data
    if mask_polygons is not None:
        diff_azim_geod_stress = mask_observations_with_polygons(diff_azim_geod_stress, mask_polygons)
        merged_df = mask_observations_with_polygons(merged_df, mask_polygons)

    # Let's print the global median and MAD of azimuth differences to report them in the paper
    global_median = np.median(merged_df['dif_azim'])
    global_mad = np.median(np.abs(merged_df['dif_azim'] - global_median))

    print(f"Median azimuth difference:, {global_median:.1f}")
    print(f"Median Absolute Deviation (MAD) of azimuth differences:, {global_mad:.1f}")
    print("-" * 80)

    # Histogram-based mode estimation of azimuth differences
    data_range = (-90, 90)  # Range of azimuth differences
    num_bins = 18 # Number of bins for the histogram
    hist, bin_edges = np.histogram(merged_df['dif_azim'], bins=num_bins, range=data_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mode_hist = bin_centers[np.argmax(hist)]

    # Compute median absolute deviation (MAD) from the histogram mode
    mad_from_mode_hist = np.median(np.abs(merged_df['dif_azim'] - mode_hist))

    print(f"Mode ({num_bins} bins) \u00B1 MAD azimuth difference: {mode_hist:.1f} \u00B1 {mad_from_mode_hist:.1f} degrees")
    print("-" * 80)



    return diff_azim_geod_stress, merged_df


########################################################################################################################
########################################################################################################################

# Function to plot azimuth differences between max. horizontal stress and max. shortening strain rate directions
########################################################################################################################
########################################################################################################################

def plot_azimuth_differences(
    diff_azim_geod_stress,
    region=[-20, 125, 5, 60],  # Default region for the map
    cpt_name="vikO",  # Custom CPT file for the azimuth differences
    fault_file="input_data/datasets/afead_v2022.gmt",
    output_filename="output_data/figures/Alpides_azimuth_diff.pdf",
    cmap_name="vikO",  # Default colormap to use
    map_scale_bar_length_km=1000,  # Length of the map scale bar in km
    map_scale_bar_position=[65, 10],  # Position (lon, lat) in degrees
    colorbar_width=8.5,  # Width of the colorbar in cm
    ocean_fill_color='white',  # Color for the ocean fill
    shorelines_pen="0.1p,black",  # Pen for the shorelines
    scatter_style="c0.08c",  # Circle style with size 0.08 cm
    hist_shift=[0.3, 0.2], # Shift (x,y) for the histogram in cm
    hist_projection="X3.2c/3.2c", # Projection (xy dimentions) for the histogram in cm
    hist_frame=["ENsw+gwhite", "xf10a30+lAzimuth difference", "yf2.5a5+u%+lFrequency percent"], # Frame details for the histogram
    hist_annot_primary="6p",  # Primary annotation font size
    hist_font_label="6p",  # Font size for the histogram labels
    plot_box_for_histogram=True, # Plot the box around scale bar
    box_histogram_position=[109.5, 26],  # Position (lon, lat) in degrees
    box_histogram_style="r2.7/1.3",  # Style for the box scale bar
    box_histogram_fill="white",  # Fill color for the box scale bar
    box_histogram_transparency=30,  # Transparency for the box scale bar
    save_fig=True,  # Save the figure to a file
):
    """
    Plot azimuth differences between the maximum shortening rate and SHmax azimuths using PyGMT.

    Parameters:
    - diff_azim_geod_stress: DataFrame containing 'lon', 'lat', 'dif_azim' (azimuth differences).
    - region: The map region [west, east, south, north] (default is for Alpine-Himalayan region).
    - cpt_filename: File path for the custom CPT colormap (default is 'input_data/cpts/azimuth_diff.cpt').
    - fault_file: File path for fault traces (default is 'input_data/datasets/afead_v2022.gmt').
    - output_filename: File path to save the output figure (default is 'output_data/figures/Alpides_azimuth_diff.pdf').
    - histogram_bins: Number of bins for the histogram of azimuth differences (default is 18).
    - cmap_name: Name of the colormap to use (default is 'coolwarm').
    """
    
    # Create the custom CPT from the colormap with fewer stops (continuous interpolation)
    #create_cpt_from_colormap(cmap_name=cmap_name, filename=cpt_filename, num_colors=10)

    # Map azimuth differences to hex colors
    #diff_azim_geod_stress['hex_color'] = diff_azim_geod_stress['dif_azim'].apply(get_hex_color)

    # Create a PyGMT figure
    fig = pygmt.Figure()

    # Set up the map region and projection
    fig.basemap(region=region, projection="M20c", frame=True)

    # Plot coastlines and country borders for reference
    #fig.coast(water=ocean_fill_color, shorelines=shorelines_pen, area_thresh=4000, resolution="h")

    # Add fault traces as very thin grey lines
    fig.plot(data=fault_file, pen="0.1p,darkgrey")

    # This is a rather clumsy way to plot the azimuth differences using a for loop. I'll change it
    # Plot the azimuth differences using the hex colors calculated from the colormap
    #for i in range(len(diff_azim_geod_stress)):
    #    lon = diff_azim_geod_stress['lon'].iloc[i]
    #    lat = diff_azim_geod_stress['lat'].iloc[i]
    #    hex_color = diff_azim_geod_stress['hex_color'].iloc[i]
        
    #    fig.plot(
    #        x=[lon],
    #        y=[lat],
    #        style=scatter_style,  # Circle style with size 0.08 cm
    #        fill=hex_color,  # Use the hex color for each azimuth difference
    #        pen=None  # Black outline for symbols
    #    )

    # This is a more efficient way to plot the azimuth differences without relying on loops nor using Matplotlib colours 
    # Let's create the cpt first
    pygmt.makecpt(cmap=cpt_name, series=[-90, 90, 1], continuous=True)
    histogram_data = list(map(float, pd.to_numeric(diff_azim_geod_stress['dif_azim'], errors='coerce').values))
    # Plot the azimuth differences using the custom colormap as a scatter plot
    fig.plot(
        x=diff_azim_geod_stress.lon,
        y=diff_azim_geod_stress.lat,
        style=scatter_style,
        fill=histogram_data,
        cmap=True,
        pen=None
    )

    # Some debugging here.. printing diff_azim_geod_stress['dif_azim'] mean
    # print(f"Mean azimuth difference: {diff_azim_geod_stress['dif_azim'].mean()}")

    # Plot coastlines and country borders for reference
    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen, area_thresh=4000, resolution="h")

    # Load the custom CPT created from Matplotlib's colormap
    #pygmt.makecpt(cmap=cpt_filename, series=[-90, 90, 1], continuous=True)

    # Add a continuous color bar for the azimuth difference values
    fig.colorbar(
        cmap=True,  # Use the custom colormap for the colorbar
        frame='a30f10+lAzimuth difference (max shortening rate - S@-Hmax@-)',  # Label for the colorbar
        position=f"JMR+o0.5c/0c+w{colorbar_width}c+v"  # Position the colorbar
    )

    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")

    # Create a histogram of the azimuth differences
    #histogram_data = pd.to_numeric(diff_azim_geod_stress['dif_azim'], errors='coerce').values.tolist()

    # print mean of histogram_data
    #print(f"Mean azimuth difference: {np.mean(histogram_data)}")

    # convert the azimuth differences to a list
    #histogram_data = list(map(float, pd.to_numeric(diff_azim_geod_stress['dif_azim'], errors='coerce').values))
    #print(histogram_data)

    if plot_box_for_histogram:
        fig.plot(x=box_histogram_position[0], y=box_histogram_position[1], style=box_histogram_style, fill=box_histogram_fill, pen=None, transparency=box_histogram_transparency)

    with pygmt.config(FONT_ANNOT_PRIMARY=hist_annot_primary, FONT_LABEL=hist_font_label, MAP_FRAME_PEN='0.3p,black', MAP_TICK_PEN='0.3p,black'):
        # Plot the histogram
        fig.shift_origin(xshift=f"{hist_shift[0]}c", yshift=f"{hist_shift[1]}c")
        fig.histogram(
            data=histogram_data,  # Data for the histogram
            region=[-90, 90, 0, 0],  # Set the region for the histogram: x-range (-90 to 90), y-range (0 to 10)
            projection=hist_projection,  # Set the projection for the histogram
            frame=hist_frame,  # Add labels to the axes
            series=[-90, 90, 10],  # Histogram range (x-range) and bin interval
            cmap=True,  # Use the same custom colormap for the histogram
            pen="0.2p,black",  # Outline the histogram bars
            histtype=1,  # Frequency percent
        )

    # Show the figure
    fig.show()

    # Save the figure to a file
    if save_fig:
        fig.savefig(output_filename, dpi=600)
########################################################################################################################
########################################################################################################################

# Function to plot L curve from input data generated by the L-curve analysis in BForStrain
########################################################################################################################
########################################################################################################################

def plot_l_curve(
    input_filepath='input_data/datasets/l_curve_matrix_Alpides_JGR.txt', # Default input file with L-curve data
    output_filepath='output_data/figures/Alpides_L_curve.pdf', # Default output file for the plot
    damping_star_values=[7, 12.5, 20], # Default beta values to highlight with star markers
    cmap_name='turbo', # Default colormap for the plot
    figsize=(8, 6), # Default figure size
    ylim=(-150, 1600), # Default y-axis limits
    xlim=(0, 3.5e4), # Default x-axis limits
    dpi=300, # Default resolution for the saved figure
    save_fig=True # Save the figure to a file
):
    """
    Plot the L-curve for regularised solution norms and residual norms using different beta values.

    Parameters:
    - input_filepath: File path of the input data (default is 'input_data/datasets/l_curve_matrix_Alpides_JGR.txt').
    - output_filepath: File path for saving the plot (default is 'output_data/figures/Alpides_L_curve.pdf').
    - beta_star_values: List of beta values to highlight with star markers (default is [7, 12.5, 20]).
    - cmap_name: Colormap to use for the plot (default is 'turbo').
    - figsize: Size of the figure (default is (8, 6)).
    - ylim: Y-axis limits for the plot (default is (-150, 1600)).
    - xlim: X-axis limits for the plot (default is (0, 3.5e4)).
    - dpi: Dots per inch (resolution) for the saved figure (default is 300).
    """
    
    # Define column names because the file does not have a header
    columns = ['beta', 'chi_2', 'reg_norm', 'resid_Var', 'resid_norm']

    # Load data
    data = pd.read_csv(input_filepath, names=columns, delimiter='\t')

    # Calculate square of residual norm and regularised norm
    data['resid_norm_sq'] = data['resid_norm'] ** 2
    data['reg_norm_sq'] = data['reg_norm'] ** 2

    # Get colormap
    cmap = plt.get_cmap(cmap_name)

    # Normalise beta values for color mapping
    norm = plt.Normalize(data['beta'].min(), data['beta'].max())

    # Create a figure
    plt.figure(figsize=figsize)

    # Separate points based on the beta values
    star_markers = data[data['beta'].isin(damping_star_values)]
    circle_markers = data[~data['beta'].isin(damping_star_values)]

    # Plot circles for the rest of the beta values
    sc = plt.scatter(circle_markers['resid_norm_sq'], circle_markers['reg_norm_sq'], 
                     c=circle_markers['beta'], cmap=cmap_name, s=50, edgecolor='k', norm=norm, linewidths=0.5)

    # Plot stars for beta values from which the posterior was obtained
    plt.scatter(star_markers['resid_norm_sq'], star_markers['reg_norm_sq'], 
                c=star_markers['beta'], cmap=cmap_name, marker='*', s=150, edgecolor='k', norm=norm)

    # Set y-axis and x-axis limits
    plt.ylim(ylim)
    plt.xlim(xlim)

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\beta$ value (body-force damping)')

    # Set labels and title using LaTeX formatting
    plt.ylabel(r'Regularized solution norm squared $\| m \|^2$')
    plt.xlabel(r'Residual norm squared $\| Gm - d \|^2$')

    ax = plt.gca()  # get current axes
    ax.tick_params(axis='x', which='both', bottom=True, top=False)  # enable bottom ticks, disable top ticks
    ax.tick_params(axis='y', which='both', left=True, right=False)  # enable left ticks, disable right ticks


    # Save the figure as a high-resolution PDF
    if save_fig:
        plt.savefig(output_filepath, dpi=dpi, bbox_inches='tight')

    # Show the plot
    plt.show()

########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to create synthetic velocities from Euler poles and tectonic plate polygons for a grid of observation points
def create_synthetic_velocities_from_euler_poles_and_plate_polygons(
        euler_poles, polygons, eura_pole, output_filepath, lon_range=(-50, 150), lat_range=(-10, 85)
    ):
    """
    Creates synthetic velocities for a grid of observation points based on Euler poles and plate polygons.
    Saves the generated synthetic velocities as a CSV file.

    Parameters:
    - euler_poles: Dictionary containing EulerPole objects for each plate relative to ITRF14
    - polygons: Dictionary containing Polygon objects for each plate
    - eura_pole: Eurasian EulerPole object for reference
    - output_filepath: Path to save the generated synthetic velocities (CSV file)
    - lon_range: Longitude range for the grid of observation points (default: (-50, 150))
    - lat_range: Latitude range for the grid of observation points (default: (-10, 85))
    """

    # Subtract Eurasia's motion to get relative motion with Eurasia fixed
    relative_poles = {region: pole - eura_pole for region, pole in euler_poles.items()}

    # Here I correct the Euler poles for the Anatolian and Philippine Sea plates in the dictionary. For some reason,
    # the poles relative to ITRF14 led to wrong velocities. I have corrected them based on the correct values, this time
    # relative to Eurasia, as reported in the literature (I am subtracting Eurasia motion in the relative_poles dictionary).
    relative_poles['anatolia'] = EulerPole(lat=30.96, lon=34.22, rate=1.087) # From Bletery et al., 2020
    relative_poles['philippines'] = EulerPole(lat=54.38, lon=157.28, rate=-1.255) # From Nishimura et al., 2018

    # Create a grid of observation points from lat_range and lon_range, initialize with NaN
    lats = np.arange(lat_range[0], lat_range[1])  # Latitude range
    lons = np.arange(lon_range[0], lon_range[1])  # Longitude range
    grid_lon, grid_lat = np.meshgrid(lons, lats)  # Create a grid of latitudes and longitudes
    grid_east_vel = np.full(grid_lon.shape, np.nan)  # Initialize East velocities with NaN
    grid_north_vel = np.full(grid_lon.shape, np.nan)  # Initialize North velocities with NaN

    # Assign velocities for grid points inside polygons
    for region, poly in polygons.items():
        pole = relative_poles[region]  # Use the relative motion with Eurasia fixed
        for i in range(grid_lon.shape[0]):
            for j in range(grid_lon.shape[1]):
                point = Point(grid_lon[i, j], grid_lat[i, j])
                if poly.contains(point):
                    e, n, _ = pole.velocity_components(grid_lat[i, j], grid_lon[i, j])
                    if region in ['anatolia', 'philippines']:
                        grid_east_vel[i, j] = -e
                        grid_north_vel[i, j] = -n
                    else:
                        grid_east_vel[i, j] = e
                        grid_north_vel[i, j] = n

    # Create an array of points with lon, lat, east_vel, and north_vel
    points = []
    for i in range(grid_lon.shape[0]):
        for j in range(grid_lon.shape[1]):
            if not np.isnan(grid_east_vel[i, j]):
                points.append([grid_lon[i, j], grid_lat[i, j], grid_east_vel[i, j], grid_north_vel[i, j]])

    # Convert to numpy array, appending two columns of ones representing standard deviations
    points = np.array(points)
    ones = np.ones((len(points), 2))
    points = np.hstack((points, ones))

    # Convert to pandas DataFrame and save to CSV
    synthetic_vels = pd.DataFrame(points, columns=['Lon', 'Lat', 'E_vel', 'N_vel', 'E_sig', 'N_sig'])
    synthetic_vels.dropna(inplace=True)
    synthetic_vels.to_csv(output_filepath, index=False, float_format='%8.2f')
    return synthetic_vels

########################################################################################################################
########################################################################################################################
# Function to plot synthetic velocities from Euler poles and tectonic plate polygons

import numpy as np
import pandas as pd
import pygmt

def plot_synthetic_velocity_field(
    file_path, region, spacing, output_cpt, output_fig, scale_vector_length=25, scale_origin=(68, 16), save_fig=True
):
    """
    Load synthetic velocity field data, preprocess it, and plot a map of the velocity magnitudes and vectors.

    Parameters:
    - file_path: Path to the synthetic velocity data (CSV file).
    - region: List of [west, east, south, north] coordinates defining the region of interest.
    - spacing: Grid spacing in degrees for interpolation.
    - output_cpt: Path to the color palette file (CPT) for the velocity magnitude.
    - output_fig: Path to save the output figure.
    - scale_vector_length: Length of the scale vectors in mm/yr (default: 25 mm/yr).
    - scale_origin: Tuple (lon, lat) defining the origin for the scale vectors (default: (68, 16)).
    """
    
    # Define column names for the synthetic velocity field data
    columns = ['lon', 'lat', 've', 'vn', 'se', 'sn']
    
    # Load the data
    velocity_data = pd.read_csv(file_path, header=None, names=columns, skiprows=1)

    # Convert necessary columns to numeric and handle non-numeric entries
    velocity_data[['lon', 'lat', 've', 'vn', 'se', 'sn']] = velocity_data[['lon', 'lat', 've', 'vn', 'se', 'sn']].apply(pd.to_numeric, errors='coerce')

    # Create new column with velocity magnitude
    velocity_data['v_mag'] = np.sqrt(velocity_data['ve']**2 + velocity_data['vn']**2)

    # Drop rows with NaNs in key columns
    velocity_data.dropna(subset=['lon', 'lat', 've', 'vn', 'se', 'sn'], inplace=True)

    ####################################################################################################################
    # Interpolation to regular grid using PyGMT blockmedian and surface
    ####################################################################################################################

    # Blockmedian for preprocessing and creating a regular grid
    blockmedian_output = pygmt.blockmedian(
        data=velocity_data[['lon', 'lat', 've']],
        spacing=spacing,
        region=region
    )

    # Create grid for Eastward velocity
    ve_grid = pygmt.surface(
        data=blockmedian_output,
        spacing=spacing,
        region=region
    )

    blockmedian_output = pygmt.blockmedian(
        data=velocity_data[['lon', 'lat', 'vn']],
        spacing=spacing,
        region=region
    )

    # Create grid for Northward velocity
    vn_grid = pygmt.surface(
        data=blockmedian_output,
        spacing=spacing,
        region=region
    )

    blockmedian_output = pygmt.blockmedian(
        data=velocity_data[['lon', 'lat', 'v_mag']],
        spacing=spacing,
        region=region
    )

    # Create grid for velocity magnitude
    v_mag_grid = pygmt.surface(
        data=blockmedian_output,
        spacing=spacing,
        region=region
    )

    ####################################################################################################################
    # Plot the synthetic velocity field
    ####################################################################################################################

    # Normalise the velocity magnitude
    normalised_vel_mag = (velocity_data['v_mag'] - velocity_data['v_mag'].min()) / (velocity_data['v_mag'].max() - velocity_data['v_mag'].min())

    # Create a list to store the vectors
    vectors = []

    # Iterate over each site to get vector components
    for i in range(len(velocity_data)):
        x_start = velocity_data.iloc[i]['lon']
        y_start = velocity_data.iloc[i]['lat']
        ve = velocity_data.iloc[i]['ve']
        vn = velocity_data.iloc[i]['vn']
        direction_degrees = np.degrees(np.arctan2(vn, ve))
        length = normalised_vel_mag.iloc[i]  # Use .iloc for positional indexing
        vectors.append([x_start, y_start, direction_degrees, length])

    # Start a PyGMT figure
    fig = pygmt.Figure()

    # Create a custom CPT (color palette table) for velocity magnitudes
    pygmt.makecpt(cmap="turbo", truncate=[0.05, 1], series=[0, 50], background='o', output=output_cpt)

    # Plot the velocity magnitude grid
    fig.basemap(region=region, projection='M20c', frame='af')
    fig.grdimage(grid=v_mag_grid, cmap=output_cpt, shading=False)

    # Plot velocity vectors
    fig.plot(
        style='v0.1c+e+n0.15',
        data=vectors,
        fill='black',
        pen='black',
    )

    # Add coastlines and colorbar
    fig.coast(shorelines="0.2p,black", area_thresh=4000, resolution='h')
    fig.colorbar(frame="af+lVelocity magnitude (mm/yr)", position="JMR+o0.5c/0c+w8.5c+v+ef")

    # Add a transparent box for the scale bar
    fig.plot(x=64.8, y=12.3, style="r2/1.6", fill="white", pen=None, transparency=30)

    # Add scale vectors to the plot
    normalised_scale_length = (scale_vector_length - velocity_data['v_mag'].min()) / (velocity_data['v_mag'].max() - velocity_data['v_mag'].min())
    scale_vectors = [
        [scale_origin[0], scale_origin[1], 0, normalised_scale_length],  # Eastward vector
        [scale_origin[0], scale_origin[1], 90, normalised_scale_length]  # Northward vector
    ]
    fig.plot(
        style='v0.1c+e+n0.15',
        data=scale_vectors,
        fill='black',
        pen='black',
        label='Accepted vel.',
    )

    # Annotate the scale vector
    fig.text(
        text=f'{scale_vector_length} mm/yr',
        x=scale_origin[0] - 5,
        y=scale_origin[1],
        font='7p,black'
    )

    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale="JBR+o-9c/-0.8c+c0+w1000k+f+lkm")

    # Save and show the figure
    if save_fig:
        fig.savefig(output_fig, dpi=600)
    fig.show()
########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Functions to plot observed and modelled velocities on the same map, and optionally across-fault velocity profiles from input files

# This function is used to plot observed and predicted GNSS velocities along with optional velocity profiles in the EMED region
def plot_observed_and_predicted_velocities_EMED(
    data,
    obs_creeping,
    mod_creeping,
    mcmc_creeping,
    obs_locked,
    mod_locked,
    mcmc_locked,
    region,
    output_path,
    fault_file,
    scale_origin,
    vel_scaling_factor=0.03,
    plot_vel_profiles=True,
    box1_coords=None,
    box2_coords=None,
    label1_coords=None,  
    label2_coords=None,
    relief_colormap="geo",
    path2_save_relief_colormap="input_data/cpts/emed_geo.cpt",
    save_fig=True
):
    """
    Plot observed and predicted GNSS velocities along with optional velocity profiles.

    Args:
    - data (DataFrame): DataFrame containing the velocity data with columns ['lon', 'lat', 've_obs', 'vn_obs', 've_mod', 'vn_mod'].
    - obs_creeping (DataFrame): DataFrame for observed velocities in the creeping segment profile.
    - mod_creeping (DataFrame): DataFrame for modeled velocities in the creeping segment profile.
    - mcmc_creeping (DataFrame): DataFrame for MCMC elastic velocities in the creeping segment profile.
    - obs_locked (DataFrame): DataFrame for observed velocities in the locked segment profile.
    - mod_locked (DataFrame): DataFrame for modeled velocities in the locked segment profile.
    - mcmc_locked (DataFrame): DataFrame for MCMC elastic velocities in the locked segment profile.
    - region (list): List defining the region [west, east, south, north].
    - output_path (str): Path where the output figure will be saved.
    - fault_file (str): Path to the fault trace file for plotting.
    - scale_origin (tuple): Tuple defining the longitude and latitude for the scale vector origin.
    - vel_scaling_factor (float): Scaling factor for the velocity vectors (default is 0.03).
    - plot_vel_profiles (bool): Flag to control whether velocity profiles are plotted (default is True).
    - box1_coords (tuple): Coordinates for the first box as (box1_lon, box1_lat).
    - box2_coords (tuple): Coordinates for the second box as (box2_lon, box2_lat).
    - label1_coords (tuple, optional): Coordinates for the first label as (lon, lat).
    - label2_coords (tuple, optional): Coordinates for the second label as (lon, lat).

    Returns:
    - None
    """

    # Create the first figure: Observed and Predicted Velocities
    fig = pygmt.Figure()

    # Create a custom CPT for the relief shading
    #pygmt.makecpt(cmap="geo", series=[-4000, 4000], output="emed_geo.cpt", transparency=70, background='o', truncate=[-4000, 4000])

   # Step 1: Create custom CPT (color palette table)
    pygmt.makecpt(cmap=relief_colormap, series=[-3000, 3000, 100], output=path2_save_relief_colormap, transparency=70, background='o', truncate=[-3000, 3000])
    pygmt.makecpt(cmap=relief_colormap, series=[0, 3000, 100], output="input_data/cpts/land_cbar.cpt", transparency=70, background='o', truncate=[0, 3000])
    # Step 2: Modify CPT for land areas, including handling on-land topography below sea level (e.g., -500 to 0)
    with open(path2_save_relief_colormap, 'r') as file:
        lines = file.readlines()

    # Find the color corresponding to the interval [0, 100] (which will be applied to the -500 to 0 range)
    for line in lines:
        if line.startswith("0\t"):
            color_for_zero_to_100 = line.split()[1]
            break

    new_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) > 2 and int(parts[0]) >= -500 and int(parts[2]) <= 0:
            if parts[1] != color_for_zero_to_100:
                parts[1] = color_for_zero_to_100
                parts[3] = color_for_zero_to_100
            new_lines.append("\t".join(parts))
        else:
            new_lines.append(line)

    # Write the modified CPT for land areas
    modified_land_cpt = "input_data/cpts/land_cpt_fixed.cpt"
    with open(modified_land_cpt, "w") as file:
        file.writelines(new_lines)

    # Plot the base map
    fig.basemap(region=region, projection='M20c', frame='af')
    fig.grdimage(grid="@earth_relief_01m", cmap="input_data/cpts/land_cpt_fixed.cpt", shading=True, transparency=70)
    fig.coast(shorelines="0.2p,black", area_thresh=4000, resolution='h', water="white")
    fig.plot(data=fault_file, pen="0.1p,grey50")  # Add fault traces

    # Plot observed velocities (blue vectors)
    data['v_mag_obs'] = np.sqrt(data['ve_obs']**2 + data['vn_obs']**2)
    data['direction_obs'] = np.degrees(np.arctan2(data['vn_obs'], data['ve_obs']))
    data['length_obs'] = data['v_mag_obs'] * vel_scaling_factor  # Scale factor for plotting

    fig.plot(
        x=data['lon'],
        y=data['lat'],
        style='v0.2c+e',  # Vector style with head
        direction=[data['direction_obs'], data['length_obs']],
        fill='blue',
        pen='blue',
        label=f'Observed velocities+S0.5c'
    )

    # Plot predicted velocities (red vectors)
    data['v_mag_mod'] = np.sqrt(data['ve_mod']**2 + data['vn_mod']**2)
    data['direction_mod'] = np.degrees(np.arctan2(data['vn_mod'], data['ve_mod']))
    data['length_mod'] = data['v_mag_mod'] * vel_scaling_factor  # Scale factor for plotting

    fig.plot(
        x=data['lon'],
        y=data['lat'],
        style='v0.2c+e+n0.3/0.2',
        direction=[data['direction_mod'], data['length_mod']],
        fill='red',
        pen='red',
        label=f'Mean posterior velocities+S0.5c'
    )

    # Add legend
    fig.legend(position='JBL+jBL', box='+gwhite+p0.5p,black')

    # Plot the first and second boxes if coordinates are provided
    if plot_vel_profiles:
        if box1_coords:
            box1_lon, box1_lat = box1_coords
            fig.plot(
                x=box1_lon,
                y=box1_lat,
                pen='0.7p,black'
            )
            if label1_coords:
                label1_lon, label1_lat = label1_coords
                fig.text(
                    text='①',
                    x=label1_lon,
                    y=label1_lat,
                    font='10p,black',
                    offset='-0.2c/0.13c'
                )

        if box2_coords:
            box2_lon, box2_lat = box2_coords
            fig.plot(
                x=box2_lon,
                y=box2_lat,
                pen='0.7p,black'
            )
            if label2_coords:
                label2_lon, label2_lat = label2_coords
                fig.text(
                    text='②',
                    x=label2_lon,
                    y=label2_lat,
                    font='10p,black',
                    offset='-0.2c/0.12c'
                )
        
    # Add scale vectors to the plot
    scale_vector_length = 20 * vel_scaling_factor  # in mm/yr, this will be normalised
    scale_vector_length_text = 20  # in mm/yr, this will be displayed on the plot

    scale_vectors = [
        [scale_origin[0], scale_origin[1], 0, scale_vector_length],  # Eastward vector
        [scale_origin[0], scale_origin[1], 90, scale_vector_length]  # Northward vector
    ]

    fig.plot(
        style='v0.2c+e+n0.3/0.2',
        data=scale_vectors,
        fill='gray25',
        pen='gray25',
    )

    # Annotate the scale vectors
    fig.text(
        text=f'{scale_vector_length_text} mm/yr',
        x=scale_origin[0] - 1,
        y=scale_origin[1],
        font='7p,black'
    )

    # Add colorbar
    fig.colorbar(frame="af+lElevation (m)", position="JMR+o0.5c/0c+w10c+v+ef", cmap="input_data/cpts/land_cbar.cpt")

    # Plot velocity profiles if enabled
    if plot_vel_profiles:
        # Add transparent rectangles to improve visibility of the scatter plots
        fig.plot(x=43.9, y=43.2, style="r8/5.1", fill="white", pen=None, transparency=30)
        fig.plot(x=43.9, y=35.3, style="r8/5.1", fill="white", pen=None, transparency=30)

        # Set up fonts and line styles for the scatter plots
        with pygmt.config(FONT_ANNOT_PRIMARY='9p', FONT_LABEL='9p', MAP_FRAME_PEN='0.5p,black', MAP_TICK_PEN='0.5p,black'):
            # Shift origin for the creeping segment profile
            fig.shift_origin(xshift="13.2c", yshift="7.8c")
            # Define region and projection
            region_profile = [-105, 105, -3, 26]
            projection_profile = "X6.5c/4c"
            # Plot base map for the profile
            fig.basemap(
                region=region_profile,
                projection=projection_profile,
                frame=['WSne+gwhite', 'xaf+lAcross-fault distance (km)', 'yaf+lFault-parallal velocity (mm/yr)']
            )

            # Plot observed velocities (blue dots) for creeping segment
            fig.plot(
                x=obs_creeping['distance'],
                y=obs_creeping['velocity'],
                style='c0.1c',
                fill='blue',
                pen='blue',
                label='Observed'
            )
            # Plot modeled velocities (red crosses) for creeping segment
            fig.plot(
                x=mod_creeping['distance'],
                y=mod_creeping['velocity'],
                style='x0.15c',
                pen='red',
                fill='red',
                label='Mean posterior'
            )
            # Plot MCMC elastic velocities for creeping segment
            fig.plot(
                x=mcmc_creeping['distance'],
                y=mcmc_creeping['velocity'],
                pen='0.5p,black,--',
                label='Elastic model'
            )

            # Add legend and labels
            fig.legend(position='JTR+jTR+o0.1c/0.1c', box='+gwhite+p0.5p,black')
            fig.text(text='① Creeping segment', x=-40, y=1, font='10p,black')

            # Shift origin for the locked segment profile
            fig.shift_origin(xshift="0c", yshift="-6.8c")  # Shift down from current position
            # Plot base map for the locked segment profile
            fig.basemap(
                region=region_profile,
                projection=projection_profile,
                frame=['WSne+gwhite', 'xaf+lAcross-fault distance (km)', 'yaf+lFault-parallal velocity (mm/yr)']
            )
            # Plot observed velocities (blue dots) for locked segment
            fig.plot(
                x=obs_locked['distance'],
                y=obs_locked['velocity'],
                style='c0.1c',
                fill='blue',
                pen='blue',
                label='Observed'
            )
            # Plot modeled velocities (red crosses) for locked segment
            fig.plot(
                x=mod_locked['distance'],
                y=mod_locked['velocity'],
                style='x0.15c',
                pen='red',
                fill='red',
                label='Mean posterior'
            )
            # Plot MCMC elastic velocities for locked segment
            fig.plot(
                x=mcmc_locked['distance'],
                y=mcmc_locked['velocity'],
                pen='0.5p,black,--',
                label='Elastic model'
            )

            # Add legend and labels
            fig.legend(position='JTR+jTR+o0.1c/0.1c', box='+gwhite+p0.5p,black')
            fig.text(text='② Locked segment', x=-45, y=1, font='10p,black')

    # Save and display the first figure
    if save_fig:
        fig.savefig(output_path, dpi=600)
    fig.show()

def load_full_velocity_data(file_path):
    """
    Load full velocity data (observed, modeled, and residual velocities).
    
    Args:
        file_path (str): Path to the file containing full velocity data.
        
    Returns:
        pd.DataFrame: DataFrame containing the velocity data with columns:
                      ['lon', 'lat', 've_obs', 'vn_obs', 've_mod', 'vn_mod', 've_res', 'vn_res']
    """
    columns = ['lon', 'lat', 've_obs', 'vn_obs', 've_mod', 'vn_mod', 've_res', 'vn_res']
    return pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)


def load_velocity_profile_data(file_path):
    """
    Load velocity profile data for observed, modeled, and MCMC elastic data.
    
    Args:
        file_path (str): Path to the file containing velocity profile data.
        
    Returns:
        pd.DataFrame: DataFrame containing the profile data with columns:
                      ['distance', 'velocity']
    """
    columns = ['distance', 'velocity']
    return pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)

########################################################################################################################
########################################################################################################################

########################################################################################################################
# Functions to plot mean posterior creep rates 

def load_creep_data(creep_file):
    """Load the creeping estimates data from a file."""
    try:
        data = pd.read_csv(creep_file, sep='\t', header=None, names=['lon', 'lat', 'creep_rate'])
        return data
    except Exception as e:
        print(f"Error loading creep data: {e}")
        return pd.DataFrame()

def load_prior_creep_data(prior_creep_files):
    """Load the prior creep rates from a list of files."""
    data_list = []
    for file in prior_creep_files:
        try:
            data = pd.read_csv(file, sep=r'\s+', header=None, names=['lon', 'lat', 'prior_creep_rate'])
            data['lon'] = data['lon'].astype(float)
            data['lat'] = data['lat'].astype(float)
            data['prior_creep_rate'] = data['prior_creep_rate'].astype(float)
            data_list.append(data)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return data_list

def create_custom_cpts(cmap_geo, cmap_polar):
    """Create custom color palettes for plotting."""
    pygmt.makecpt(
        cmap=cmap_geo,
        series=[-4000, 4000, 100],
        output="input_data/cpts/geo.cpt",
        transparency=70,
        background='o',
        truncate=[-4000, 4000]
    )
    pygmt.makecpt(
        cmap=cmap_polar,
        series=[-15, 15],
        output='input_data/cpts/creep.cpt'
    )

def plot_creep_estimates(
    creep_file,
    prior_creep_files,
    map_annotations,
    profile_annotations,
    fault_file,
    rectangle_params,
    output_file="output_data/figures/creep_estimates.pdf",
    cmap_geo="geo",
    cmap_polar="polar",
    region=[20, 50, 32, 46],
    faults_pen="0.1p,grey50", # Pen for the fault traces
    save_fig=True
):
    """
    Plot the distribution of fault creep rates together with a profile of creep rates.

    Args:
    - creep_file (str): Path to the file containing creep estimates.
    - prior_creep_files (list): List of file paths for prior creep rates.
    - map_annotations (list): List of dictionaries with annotation coordinates for the map.
    - profile_annotations (list): List of dictionaries with annotation coordinates for the creep profile.
    - fault_file (str): Path to the fault trace file for plotting.
    - rectangle_params (dict): Dictionary specifying the rectangle's parameters (location, size, fill, transparency).
    - output_file (str): Path to save the output figure.
    - cmap_geo (str): Colormap for the base relief.
    - cmap_polar (str): Colormap for the creep rates.
    - region (list): List defining the map region [west, east, south, north].

    Returns:
    - None
    """
    # Load data
    creep_data = load_creep_data(creep_file)
    creep_prior_data_list = load_prior_creep_data(prior_creep_files)

    # Start a PyGMT figure
    fig = pygmt.Figure()

    # # Step 1: Create custom CPTs
    create_custom_cpts(cmap_geo, cmap_polar)
    
    # Step 2: Modify CPT for land areas, including handling on-land topography below sea level (e.g., -500 to 0)
    with open('input_data/cpts/geo.cpt', 'r') as file:
        lines = file.readlines()

    # Find the color corresponding to the interval [0, 100] (which will be applied to the -500 to 0 range)
    for line in lines:
        if line.startswith("0\t"):
            color_for_zero_to_100 = line.split()[1]
            break

    new_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) > 2 and int(parts[0]) >= -500 and int(parts[2]) <= 0:
            if parts[1] != color_for_zero_to_100:
                parts[1] = color_for_zero_to_100
                parts[3] = color_for_zero_to_100
            new_lines.append("\t".join(parts))
        else:
            new_lines.append(line)

    # Write the modified CPT for land areas
    modified_land_cpt = "input_data/cpts/land_cpt_fixed.cpt"
    with open(modified_land_cpt, "w") as file:
        file.writelines(new_lines)

    # Plot the base map
    fig.basemap(region=region, projection='M20c', frame='af')
    fig.grdimage(grid="@earth_relief_01m", cmap="input_data/cpts/land_cpt_fixed.cpt", shading=True, transparency=70)
    fig.coast(shorelines="0.2p,black", area_thresh=4000, resolution='h', water="white")
    fig.plot(data=fault_file, pen=faults_pen)  # Add fault traces

    # Plot the creeping rates as circles on the map
    fig.plot(
        x=creep_data['lon'],
        y=creep_data['lat'],
        style='c0.3c',  # Circle of size 0.3 cm
        fill=creep_data['creep_rate'],
        cmap='input_data/cpts/creep.cpt',
        pen=None
    )

    # Add scale bar and annotations
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale="JBR+o-6.5c/-0.8c+c0+w200k+f+lkm")

    # Add text annotations on the map (Longitude, Latitude)
    for annotation in map_annotations:
        fig.text(
            text=annotation['text'],
            x=annotation['lon'],
            y=annotation['lat'],
            font="8p,black",
            justify="LM",
            angle=annotation.get('angle', None)
        )

    # Add colorbar (only for the map)
    fig.colorbar(
        frame='af+lMean posterior creep rate (mm/yr)',
        position="JMR+o0.5c/0c+w10c+v",
        cmap='input_data/cpts/creep.cpt'
    )

    # Add a transparent rectangle behind the profile area
    if rectangle_params:
        x_rect = rectangle_params['x']
        y_rect = rectangle_params['y']
        width = rectangle_params['width']
        height = rectangle_params['height']
        fill = rectangle_params.get('fill', 'white')
        transparency = rectangle_params.get('transparency', 30)
        pen = rectangle_params.get('pen', None)

        fig.plot(
            x=x_rect,
            y=y_rect,
            style=f"r{width}/{height}",
            fill=fill,
            pen=pen,
            transparency=transparency
        )

    # Plot the creep profile
    plot_creep_profile(fig, creep_data, creep_prior_data_list, profile_annotations)

    # Save and display the figure
    if save_fig:
        fig.savefig(output_file, dpi=600)
    fig.show()

def plot_creep_profile(fig, creep_data, creep_prior_data_list, profile_annotations):
    """Plot the scatter plot of creep rates and prior lines as a subplot."""
    xmin = creep_data['lon'].min() - 1.5
    xmax = creep_data['lon'].max() + 2.5
    ymin = -10  # Adjust as needed
    ymax = 20   # Adjust as needed
    region_scatter = [xmin, xmax, ymin, ymax]

    with pygmt.config(
        FONT_ANNOT_PRIMARY='9p',
        FONT_LABEL='9p',
        MAP_FRAME_PEN='0.3p,black',
        MAP_TICK_PEN='0.3p,black'
    ):
        # Shift the origin to position the subplot correctly
        fig.shift_origin(xshift="1.3c", yshift="1.0c")

        # Define the region based on data and adjust projection to fit panel size
        fig.basemap(
            region=region_scatter,
            projection='X8.5c/5.5c',
            frame=['WSne+gwhite', 'xaf+lLongitude', 'ya5f2.5+lMean posterior creep rate (mm/yr)']
        )

        # Plot the posterior creep rates as a scatter plot
        fig.plot(
            x=creep_data['lon'],
            y=creep_data['creep_rate'],
            style='c0.2c',
            fill=creep_data['creep_rate'],
            cmap='input_data/cpts/creep.cpt',
            pen='0.1p,black',
        )

        # Plot the prior creep rates as lines
        for data in creep_prior_data_list:
            data_sorted = data.sort_values('lon').reset_index(drop=True)
            fig.plot(
                x=data_sorted['lon'],
                y=data_sorted['prior_creep_rate'],
                pen='0.5p,black,-',
            )

        # Add annotations to the profile (Longitude, Creep Rate)
        for annotation in profile_annotations:
            fig.text(
                text=annotation['text'],
                x=annotation['lon'],
                y=annotation['creep_rate'],
                font="8p,black",
                justify="LM",
                angle=annotation.get('angle', None)
            )

        # Create dummy plots for the legend entries outside the profile area
        xmin_dummy = xmin - 10
        ymin_dummy = ymin - 10

        fig.plot(
            x=[xmin_dummy], y=[ymin_dummy],  # Coordinates outside the plot area
            style='c0.2c',
            fill='white',
            pen='0.1p,black',
            label='Mean posterior creep rate'
        )
        fig.plot(
            x=[xmin_dummy, xmin_dummy - 1],  # Coordinates outside the plot area
            y=[ymin_dummy, ymin_dummy - 1],
            pen='0.5p,black,-',
            label='Prior creep rate'
        )

        # Add legend to the scatter plot
        fig.legend(position='JTR+jTR+o0c/0c', box='+gwhite+p0.5p,black')
########################################################################################################################
########################################################################################################################

# These are 2 helper functions needed later on to add profiles and labels to the map...

# This first function is used to add focus region boundaries and labels to the PyGMT figure
def create_focus_region_boundaries(fig, focus_regions):
    """
    Adds focus region boundaries and labels to the PyGMT figure.

    Parameters:
        fig (pygmt.Figure): The PyGMT figure to modify.
        focus_regions (dict): Dictionary defining focus regions with coordinates and labels.
    """
    for region_name, region_data in focus_regions.items():
        min_lon, max_lon, min_lat, max_lat = region_data["coordinates"]
        label_data = region_data.get("label", {})
        label_lon, label_lat = label_data.get("coordinates", (None, None))
        label_text = label_data.get("text", region_name)  # Default to region name if text is not provided

        # Draw the bounding box
        fig.plot(
            x=[min_lon, max_lon, max_lon, min_lon, min_lon],
            y=[min_lat, min_lat, max_lat, max_lat, min_lat],
            pen="1p,black",
        )

        # Add the label with transparency
        if label_lon is not None and label_lat is not None:
            fig.text(
                text=label_text,
                x=label_lon,
                y=label_lat,
                font="10p,Helvetica-Bold,black",
                fill="white",
                transparency=30,
                justify="CM",
            )
        
        # Add the label without transparency
        if label_lon is not None and label_lat is not None:
            fig.text(
                text=label_text,
                x=label_lon,
                y=label_lat,
                font="10p,Helvetica-Bold,black",
                justify="CM",
            )

# Now I define the second function to add profile bounding boxes and labels to the PyGMT figure
def calculate_initial_compass_bearing(lon1, lat1, lon2, lat2):
    """
    Calculate the azimuth (initial compass bearing) between two geographical points.
    """
    import math
    d_lon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    return (initial_bearing + 360) % 360


def destination_point(lon, lat, azimuth, distance_km):
    """
    Calculate a destination point given a starting point, azimuth, and distance using the haversine formula.
    """
    import math
    R = 6371.0  # Earth's radius in kilometers
    azimuth_rad = math.radians(azimuth)
    delta = distance_km / R

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(delta) + math.cos(lat1) * math.sin(delta) * math.cos(azimuth_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(azimuth_rad) * math.sin(delta) * math.cos(lat1),
        math.cos(delta) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lon2), math.degrees(lat2)


def create_profile_bounding_boxes(
    fig, profiles, selected_profile_names, profile_labels, temp_bbox_directory="./input_data/datasets/temp_bbox"
):
    """
    Adds profile bounding boxes and labels to the PyGMT figure, using temporary files for bounding boxes.

    Parameters:
        fig (pygmt.Figure): The PyGMT figure to modify.
        profiles (list): List of profile dictionaries.
        selected_profile_names (list): List of profile names to plot.
        profile_labels (dict): Dictionary defining profile labels with coordinates and text.
        temp_bbox_directory (str): Directory to save temporary bounding box files.
    """
    os.makedirs(temp_bbox_directory, exist_ok=True)

    for profile in profiles:
        if profile["name"] in selected_profile_names:
            start_lon, start_lat = profile["start_lon"], profile["start_lat"]
            end_lon, end_lat = profile["end_lon"], profile["end_lat"]
            width_km = profile["width_km"]
            profile_name = profile["name"]

            # Calculate azimuth and distance
            az12 = calculate_initial_compass_bearing(start_lon, start_lat, end_lon, end_lat)

            # Create bounding box polygon
            def create_profile_bbox(start_lon, start_lat, end_lon, end_lat, width_km):
                width_km_half = width_km / 2.0

                left_azimuth = (az12 - 90) % 360
                right_azimuth = (az12 + 90) % 360

                # Start point
                lon1_left, lat1_left = destination_point(
                    start_lon, start_lat, left_azimuth, width_km_half
                )
                lon1_right, lat1_right = destination_point(
                    start_lon, start_lat, right_azimuth, width_km_half
                )

                # End point
                lon2_left, lat2_left = destination_point(
                    end_lon, end_lat, left_azimuth, width_km_half
                )
                lon2_right, lat2_right = destination_point(
                    end_lon, end_lat, right_azimuth, width_km_half
                )

                # Polygon coordinates
                bbox_coords = [
                    (lon1_left, lat1_left),
                    (lon2_left, lat2_left),
                    (lon2_right, lat2_right),
                    (lon1_right, lat1_right),
                    (lon1_left, lat1_left),  # Close the polygon
                ]
                return Polygon(bbox_coords)

            bbox_polygon = create_profile_bbox(
                start_lon, start_lat, end_lon, end_lat, width_km
            )

            # Define the temporary file path for the bounding box
            temp_bbox_file_path = os.path.join(temp_bbox_directory, f"{profile_name}_bbox.dat")

            # Write the bounding box coordinates to the file
            with open(temp_bbox_file_path, "w") as tmpfile_bbox:
                for lon, lat in bbox_polygon.exterior.coords:
                    tmpfile_bbox.write(f"{lon} {lat}\n")

            # Plot profile bounding box on the map
            fig.plot(
                data=temp_bbox_file_path,
                pen="1p,magenta",
                fill="white",
                close=True,
                transparency=50,
            )

            # Add start and end profile labels
            profile_label = profile_labels.get(profile_name, {})

            # Start label (coordinates_1 and text_1)
            label_lon_1, label_lat_1 = profile_label.get("coordinates_1", (None, None))
            label_text_1 = profile_label.get("text_1", None)

            if label_lon_1 is not None and label_lat_1 is not None and label_text_1 is not None:
                fig.text(
                    text=label_text_1,
                    x=label_lon_1,
                    y=label_lat_1,
                    font="10p,Helvetica-Bold,black",
                    fill="white",
                    transparency=30,
                    justify="CM",
                )
                fig.text(
                    text=label_text_1,
                    x=label_lon_1,
                    y=label_lat_1,
                    font="10p,Helvetica-Bold,black",
                    justify="CM",
                )

            # End label (coordinates_2 and text_2)
            label_lon_2, label_lat_2 = profile_label.get("coordinates_2", (None, None))
            label_text_2 = profile_label.get("text_2", None)

            if label_lon_2 is not None and label_lat_2 is not None and label_text_2 is not None:
                fig.text(
                    text=label_text_2,
                    x=label_lon_2,
                    y=label_lat_2,
                    font="10p,Helvetica-Bold,black",
                    fill="white",
                    transparency=30,
                    justify="CM",
                )
                fig.text(
                    text=label_text_2,
                    x=label_lon_2,
                    y=label_lat_2,
                    font="10p,Helvetica-Bold,black",
                    justify="CM",
                )


# Function to plot triangular mesh for strain rates and (optionally) velocity profiles and focus region boundaries
def plot_tri_mesh(
    input_tri_mesh="input_data/datasets/meshes/alpides_mesh.gmt",  # Path to the input triangular mesh file
    alpides_region=[-20, 125, 5, 60], # Region of interest
    map_scale_bar_length_km=1000,  # Length of the map scale bar in km
    map_scale_bar_position=[65, 10],  # Position (x,y) in cm
    colorbar_width=8.5,  # Width of the colorbar in cm
    ocean_fill_color=[],  # Color for the ocean fill
    shorelines_pen="0.4p,black",  # Pen for the shorelines
    colormap_tri_mesh="turbo",  # Colormap for the normalised area values
    colormap_series=[0, 1, 0.1],
    colormap_continuous=True,
    colormap_truncate=[0.1, 1],
    plot_box_scale_bar=True,
    box_scale_bar_position=[64.8, 12.3],  # Position (lon, lat) in degrees
    box_scale_bar_style="r2/1.6",  # Style for the box scale bar
    box_scale_bar_fill="white",  # Fill color for the box scale bar
    box_scale_bar_transparency=30,  # Transparency for the box scale bar
    save_fig=True,  # Save the figure as a pdf
    output_filenames_prefix="Alpides",  # Prefix for the output filename
    plot_focus_region_boundaries=False,  # Plot the region boundaries? True / False
    focus_regions=None,  # Pass the focus regions dictionary
    plot_profiles=False, # plot profile bounding boxes? True / False
    profiles=None, # dictionary of profiles (used later one to plot velocity and strain rate profiles)
    selected_profile_names=["Aegean_Anatolia", "Cyprus_Anatolia", "Tibet_EW", "Himalayas_Baikal"],  # Only these profile will be plotted...
    profile_labels=None,  # dictionary of profile labels (used later one to plot velocity and strain rate profiles)
):

    fig = pygmt.Figure()

    # Set up the color palette for the normalised area values
    pygmt.makecpt(cmap=colormap_tri_mesh, series=colormap_series, continuous=colormap_continuous, truncate=colormap_truncate, transparency=15)

    fig.basemap(region=alpides_region, projection='M20c', frame='af')
    fig.plot(data=input_tri_mesh, pen="0.1p", close=True, cmap=True)
    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen,area_thresh=4000, resolution='h')

    # Add a transparent box for the scale bar
    if plot_box_scale_bar:
        fig.plot(x=box_scale_bar_position[0], y=box_scale_bar_position[1], style=box_scale_bar_style, fill=box_scale_bar_fill, pen=None, transparency=box_scale_bar_transparency)
        
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")

    # Add focus region boundaries and labels
    if plot_focus_region_boundaries and focus_regions:
        create_focus_region_boundaries(fig, focus_regions)

    # Add profile bounding boxes and labels
    if plot_profiles and profiles and selected_profile_names:
        create_profile_bounding_boxes(fig, profiles, selected_profile_names, profile_labels)

    # Add colorbar
    fig.colorbar(frame='af+lNormalized area',
                    position=f"JMR+o0.5c/0c+w{colorbar_width}c+v")

    # save the figure as pdf
    if save_fig:
        fig.savefig(f"output_data/figures/{output_filenames_prefix}_tri_mesh.pdf", dpi=300)

    fig.show()
########################################################################################################################
########################################################################################################################

def plot_observed_and_predicted_velocities(
    data,
    obs_fault1,
    mod_fault1,
    mcmc_fault1,
    obs_fault2,
    mod_fault2,
    mcmc_fault2,
    region,
    output_path,
    fault_file,
    vel_scale_pos = [66.7, 21.5],  # Position of the scale bar on the map (longitude, latitude)
    vel_mag_ref_scale = 35, # Reference scale for the velocity magnitude scale vectors
    vel_scaling_factor = 0.02,  # Scaling factor for the velocity vectors
    vel_scale_label_offset = [0.6, -0.7],  # Offset for the velocity scale label in degrees
    plot_vel_profiles=True,
    box1_coords=None,
    box2_coords=None,
    label1_coords=None,  
    label2_coords=None,
    relief_colormap="geo",
    path2_save_relief_colormap="input_data/cpts/emed_geo.cpt",
    map_legend_location="BL",
    origin_shift_profile_1=[0, 0],  # Shift the origin for the first profile
    origin_shift_profile_2=[0, 0],  # Shift the origin for the second profile
    profile_1_label="Fault 1",  # Label for the first profile
    profile_2_label="Fault 2",  # Label for the second profile
    profile_1_label_location=[-40, -1], # Label for the first velocity profile
    profile_2_label_location=[-45, -1], # Label for the second velocity profile
    profile_1_legend_location="TR", # Label for the first velocity profile
    profile_2_legend_location="BR", # Label for the second velocity profile
    region_profile1=[-100, 100, -3, 26],  # Region for the first velocity profile
    region_profile2=[-100, 100, -3, 26],  # Region for the second velocity profile
    profiles_size="X7.3c/4.3c",  # Size of the velocity profiles
    colormap_series = [-7000, 7000, 100], # Series for the colormap
    colormap_truncate = [-3000, 3000], # Truncate the colormap
    colormap_transparency = 70, # Transparency for the colormap
    colorbar_width = 10,  # Width of the colorbar in cm
    mcmc_params1=None,
    mcmc_params2=None,
    mcmc_params1_pos=None,
    mcmc_params2_pos=None,
    map_scale_bar_length_km=500,  # Length of the map scale bar in km
    map_scale_bar_position=[109.5, 26],  # Position (lon, lat) in degrees
    mcmc_results_line_spacing_profile_1=1.5,  # Line spacing for the MCMC results
    mcmc_results_line_spacing_profile_2=1.5,  # Line spacing for the MCMC results
    transparent_box1_params=None,
    transparent_box2_params=None,
    save_fig=True
):
    """
    Plot observed and predicted GNSS velocities along with optional velocity profiles.

    Args:
    - data (DataFrame): DataFrame containing the velocity data with columns ['lon', 'lat', 've_obs', 'vn_obs', 've_mod', 'vn_mod'].
    - obs_creeping (DataFrame): DataFrame for observed velocities in the creeping segment profile.
    - mod_creeping (DataFrame): DataFrame for modeled velocities in the creeping segment profile.
    - mcmc_creeping (DataFrame): DataFrame for MCMC elastic velocities in the creeping segment profile.
    - obs_locked (DataFrame): DataFrame for observed velocities in the locked segment profile.
    - mod_locked (DataFrame): DataFrame for modeled velocities in the locked segment profile.
    - mcmc_locked (DataFrame): DataFrame for MCMC elastic velocities in the locked segment profile.
    - region (list): List defining the region [west, east, south, north].
    - output_path (str): Path where the output figure will be saved.
    - fault_file (str): Path to the fault trace file for plotting.
    - scale_origin (tuple): Tuple defining the longitude and latitude for the scale vector origin.
    - vel_scaling_factor (float): Scaling factor for the velocity vectors (default is 0.03).
    - plot_vel_profiles (bool): Flag to control whether velocity profiles are plotted (default is True).
    - box1_coords (tuple): Coordinates for the first box as (box1_lon, box1_lat).
    - box2_coords (tuple): Coordinates for the second box as (box2_lon, box2_lat).
    - label1_coords (tuple, optional): Coordinates for the first label as (lon, lat).
    - label2_coords (tuple, optional): Coordinates for the second label as (lon, lat).

    Returns:
    - None
    """

    # Helper function to build text lines from a DataFrame row
    def build_text_lines(params):
        text_lines = [
            f"v@-0@- = {abs(params['v_0']):.1f} mm/yr",
            f"D = {params['D']:.2f} km",
        ]
        # Add optional parameters if present
        if 'v_c' in params:
            text_lines.append(f"v@-c@- = {abs(params['v_c']):.1f} mm/yr")
        if 'd_c' in params:
            text_lines.append(f"d@-c@- = {params['d_c']:.1f} km")
        return text_lines

    # Create the first figure: Observed and Predicted Velocities
    fig = pygmt.Figure()

    # Create a custom CPT for the relief shading
    #pygmt.makecpt(cmap="geo", series=[-4000, 4000], output="emed_geo.cpt", transparency=70, background='o', truncate=[-4000, 4000])

   # Step 1: Create custom CPT (color palette table)
    pygmt.makecpt(cmap=relief_colormap, series=colormap_series, output=path2_save_relief_colormap, transparency=colormap_transparency, background='o', truncate=colormap_truncate)
    pygmt.makecpt(cmap=relief_colormap, series=[0, colormap_series[1], colormap_series[2]], output="input_data/cpts/land_cbar.cpt", transparency=colormap_transparency, background='o', truncate=[0, colormap_truncate[1]])
    # Step 2: Modify CPT for land areas, including handling on-land topography below sea level (e.g., -500 to 0)
    with open(path2_save_relief_colormap, 'r') as file:
        lines = file.readlines()

    # Find the color corresponding to the interval [0, 100] (which will be applied to the -500 to 0 range)
    for line in lines:
        if line.startswith("0\t"):
            color_for_zero_to_100 = line.split()[1]
            break

    new_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) > 2 and int(parts[0]) >= -500 and int(parts[2]) <= 0:
            if parts[1] != color_for_zero_to_100:
                parts[1] = color_for_zero_to_100
                parts[3] = color_for_zero_to_100
            new_lines.append("\t".join(parts))
        else:
            new_lines.append(line)

    # Write the modified CPT for land areas
    modified_land_cpt = "input_data/cpts/land_cpt_fixed.cpt"
    with open(modified_land_cpt, "w") as file:
        file.writelines(new_lines)

    # Plot the base map
    fig.basemap(region=region, projection='M20c', frame='af')
    fig.grdimage(grid="@earth_relief_01m", cmap="input_data/cpts/land_cpt_fixed.cpt", shading=True, transparency=70)
    fig.coast(shorelines="0.2p,black", area_thresh=4000, resolution='h', water="white")
    fig.plot(data=fault_file, pen="0.1p,grey50")  # Add fault traces

    # Plot observed velocities (blue vectors)
    data['v_mag_obs'] = np.sqrt(data['ve_obs']**2 + data['vn_obs']**2)
    data['direction_obs'] = np.degrees(np.arctan2(data['vn_obs'], data['ve_obs']))
    data['length_obs'] = data['v_mag_obs'] * vel_scaling_factor  # Scale factor for plotting

    fig.plot(
        x=data['lon'],
        y=data['lat'],
        style='v0.2c+e',  # Vector style with head
        direction=[data['direction_obs'], data['length_obs']],
        fill='blue',
        pen='blue',
        label=f'Observed velocities+S0.5c'
    )

    # Plot predicted velocities (red vectors)
    data['v_mag_mod'] = np.sqrt(data['ve_mod']**2 + data['vn_mod']**2)
    data['direction_mod'] = np.degrees(np.arctan2(data['vn_mod'], data['ve_mod']))
    data['length_mod'] = data['v_mag_mod'] * vel_scaling_factor  # Scale factor for plotting

    fig.plot(
        x=data['lon'],
        y=data['lat'],
        style='v0.2c+e+n0.3/0.2',
        direction=[data['direction_mod'], data['length_mod']],
        fill='red',
        pen='red',
        label=f'Mean posterior velocities+S0.5c'
    )

    # Add legend
    fig.legend(position=f'J{map_legend_location}+j{map_legend_location}', box='+gwhite+p0.5p,black')

    # Plot the first and second boxes if coordinates are provided
    if plot_vel_profiles:
        if box1_coords:
            box1_lon, box1_lat = box1_coords
            fig.plot(
                x=box1_lon,
                y=box1_lat,
                pen='0.7p,black'
            )
            if label1_coords:
                label1_lon, label1_lat = label1_coords
                fig.text(
                    text='①',
                    x=label1_lon,
                    y=label1_lat,
                    font='10p,black',
                    offset='-0.2c/0.13c'
                )

        if box2_coords:
            box2_lon, box2_lat = box2_coords
            fig.plot(
                x=box2_lon,
                y=box2_lat,
                pen='0.7p,black'
            )
            if label2_coords:
                label2_lon, label2_lat = label2_coords
                fig.text(
                    text='②',
                    x=label2_lon,
                    y=label2_lat,
                    font='10p,black',
                    offset='-0.2c/0.12c'
                )
        
    # Add scale vectors to the plot
    scale_vector_length = vel_mag_ref_scale * vel_scaling_factor  # in mm/yr, this will be normalised
    scale_vector_length_text = vel_mag_ref_scale  # in mm/yr, this will be displayed on the plot

    scale_vectors = [
        [vel_scale_pos[0], vel_scale_pos[1], 0, scale_vector_length],  # Eastward vector
        [vel_scale_pos[0], vel_scale_pos[1], 90, scale_vector_length]  # Northward vector
    ]

    fig.plot(
        style='v0.2c+e+n0.3/0.2',
        data=scale_vectors,
        fill='gray25',
        pen='gray25',
    )

    # Annotate the scale vectors
    fig.text(
        text=f'{scale_vector_length_text} mm/yr',
        x=vel_scale_pos[0] + vel_scale_label_offset[0],
        y=vel_scale_pos[1] + vel_scale_label_offset[1],
        font='7p,black'
    )

    # Add colorbar
    fig.colorbar(frame="af+lElevation (m)", position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef", cmap="input_data/cpts/land_cbar.cpt")

    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")

    # Plot velocity profiles if enabled
    if plot_vel_profiles:
        # Add transparent rectangles to improve visibility of the scatter plots
        if transparent_box1_params:
            fig.plot(x=transparent_box1_params['x'], y=transparent_box1_params['y'], style=transparent_box1_params['style'], fill=transparent_box1_params['fill'], pen=transparent_box1_params['pen'], transparency=transparent_box1_params['transparency'])

        if transparent_box2_params:
            fig.plot(x=transparent_box2_params['x'], y=transparent_box2_params['y'], style=transparent_box2_params['style'], fill=transparent_box2_params['fill'], pen=transparent_box2_params['pen'], transparency=transparent_box2_params['transparency'])

        # Set up fonts and line styles for the scatter plots
        with pygmt.config(FONT_ANNOT_PRIMARY='9p', FONT_LABEL='9p', MAP_FRAME_PEN='0.5p,black', MAP_TICK_PEN='0.5p,black'):
            # Shift origin for the creeping segment profile
            fig.shift_origin(xshift=f"{origin_shift_profile_1[0]}c", yshift=f"{origin_shift_profile_1[1]}c")
            # Define region and projection
            projection_profile = profiles_size
            # Plot base map for the profile
            fig.basemap(
                region=region_profile1,
                projection=projection_profile,
                frame=['WSne+gwhite', 'xaf+lAcross-fault distance (km)', 'yaf+lFault-parallal velocity (mm/yr)']
            )

            # Plot observed velocities (blue dots) for creeping segment
            fig.plot(
                x=obs_fault1['distance'],
                y=obs_fault1['velocity'],
                style='c0.1c',
                fill='blue',
                pen='blue',
                label='Observed'
            )
            # Plot modeled velocities (red crosses) for creeping segment
            fig.plot(
                x=mod_fault1['distance'],
                y=mod_fault1['velocity'],
                style='x0.15c',
                pen='red',
                fill='red',
                label='Mean posterior'
            )
            # Plot MCMC elastic velocities for creeping segment
            fig.plot(
                x=mcmc_fault1['distance'],
                y=mcmc_fault1['velocity'],
                pen='0.5p,black,--',
                label='Elastic model'
            )

            # Add legend and labels
            fig.legend(position=f'J{profile_1_legend_location}+j{profile_1_legend_location}+o0.1c/0.1c', box='+gwhite+p0.5p,black')
            fig.text(text=f'{profile_1_label}', x=profile_1_label_location[0], y=profile_1_label_location[1], font='10p,black')

            # Add MCMC parameters to the first profile
            if mcmc_params1 is not None and mcmc_params1_pos is not None:
                params1 = mcmc_params1.iloc[0]
                text_lines1 = build_text_lines(params1)
                fig.text(
                    region=region_profile1,
                    projection="X7.3c/4.3c",
                    x=[mcmc_params1_pos[0]] * len(text_lines1),
                    y=[mcmc_params1_pos[1] - mcmc_results_line_spacing_profile_1*i for i in range(len(text_lines1))],
                    text=text_lines1,
                    font="9p,black",
                    justify="TL",
                    frame=False
                )

            # Shift origin for the locked segment profile
            fig.shift_origin(xshift=f"{origin_shift_profile_2[0]}c", yshift=f"{origin_shift_profile_2[1]}c")  # Shift down from current position
            # Plot base map for the locked segment profile
            fig.basemap(
                region=region_profile2,
                projection=projection_profile,
                frame=['WSne+gwhite', 'xaf+lAcross-fault distance (km)', 'yaf+lFault-parallal velocity (mm/yr)']
            )
            # Plot observed velocities (blue dots) for locked segment
            fig.plot(
                x=obs_fault2['distance'],
                y=obs_fault2['velocity'],
                style='c0.1c',
                fill='blue',
                pen='blue',
                label='Observed'
            )
            # Plot modeled velocities (red crosses) for locked segment
            fig.plot(
                x=mod_fault2['distance'],
                y=mod_fault2['velocity'],
                style='x0.15c',
                pen='red',
                fill='red',
                label='Mean posterior'
            )
            # Plot MCMC elastic velocities for locked segment
            fig.plot(
                x=mcmc_fault2['distance'],
                y=mcmc_fault2['velocity'],
                pen='0.5p,black,--',
                label='Elastic model'
            )

            # Add legend and labels
            fig.legend(position=f'J{profile_2_legend_location}+j{profile_2_legend_location}+o0.1c/0.1c', box='+gwhite+p0.5p,black')
            fig.text(text=f'{profile_2_label}', x=profile_2_label_location[0], y=profile_2_label_location[1], font='10p,black')

            # Add MCMC parameters to the second profile
            if mcmc_params2 is not None and mcmc_params2_pos is not None:
                params2 = mcmc_params2.iloc[0]
                text_lines2 = build_text_lines(params2)
                fig.text(
                    region=region_profile2,
                    projection="X7.3c/4.3c",
                    x=[mcmc_params2_pos[0]] * len(text_lines2),
                    y=[mcmc_params2_pos[1] - mcmc_results_line_spacing_profile_2*i for i in range(len(text_lines2))],
                    text=text_lines2,
                    font="9p,black",
                    justify="TL",
                    frame=False
                )

    # Save and display the first figure
    if save_fig:
        fig.savefig(output_path, dpi=600)
    fig.show()

def load_full_velocity_data(file_path):
    """
    Load full velocity data (observed, modeled, and residual velocities).
    
    Args:
        file_path (str): Path to the file containing full velocity data.
        
    Returns:
        pd.DataFrame: DataFrame containing the velocity data with columns:
                      ['lon', 'lat', 've_obs', 'vn_obs', 've_mod', 'vn_mod', 've_res', 'vn_res']
    """
    columns = ['lon', 'lat', 've_obs', 'vn_obs', 've_mod', 'vn_mod', 've_res', 'vn_res']
    return pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)


def load_velocity_profile_data(file_path):
    """
    Load velocity profile data for observed, modeled, and MCMC elastic data.
    
    Args:
        file_path (str): Path to the file containing velocity profile data.
        
    Returns:
        pd.DataFrame: DataFrame containing the profile data with columns:
                      ['distance', 'velocity']
    """
    columns = ['distance', 'velocity']
    return pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)

########################################################################################################################
########################################################################################################################

########################################################################################################################
########################################################################################################################
# Function to plot residual velocities from input data

def plot_residual_velocities(
    data,
    region,
    output_path,
    fault_file,
    vel_scale_pos = (32, 32.8),  # Position of the scale bar on the map (longitude, latitude)
    vel_mag_ref_scale = 20, # Reference scale for the velocity magnitude scale vectors
    vel_scaling_factor = 0.03,  # Scaling factor for the velocity vectors
    vel_scale_label_offset = [0.6, -0.3],  # Offset for the velocity scale label in degrees
    plot_residual_histograms=True,
    relief_colormap="geo",
    path2_save_relief_colormap="input_data/cpts/emed_geo.cpt",
    histogram_transparent_box_style=None,  # Parameters for the transparent box for the histogram
    histograms_style={
        'region':[-3.5, 3.5, 0, 40], # Region for the histograms
        'projection':"X3.7c/3.7c", # Projection for the histograms
        'frame_ve':["WNse+gwhite", "xf1a1+lV@-e@- residual (mm/yr)", "yf2.5a5+u%+lFrequency percent"], # Frame details for the histograms (East velocity residuals)
        'frame_vn':["Nwse+gwhite", "xf1a1+l\"V@-n@- residual (mm/yr)\"", "yf2.5a5+u%+lFrequency percent"],  # Frame details for the histograms (North velocity residuals)
        'series':[-4, 4, 0.5],  # Histogram range and bin interval
        'pen':"0.2p,black",  # Outline the histogram bars
        'histtype':1,  # Frequency percent
        'fill':'gray',  # Fill the histogram bars with gray color
        'distribution':None,  # Plot the distribution curve
        'cumulative':False,  # Plot the cumulative distribution
    },
    origin_shift_histogram1=[12.4, 0.2], # Shift applied to the origin of the first velocity profile (x, y) in cm
    origin_shift_histogram2=[3.7, 0], # Shift applied to the origin of the first velocity profile (x, y) in cm
    residual_legend_location="BL",  # Location for the residual velocity legend
    colormap_series = [-5000, 5000, 100], # Series for the colormap
    colormap_truncate = [-5000, 5000], # Truncate the colormap
    map_scale_bar_position=[109.5, 26],  # Position (lon, lat) in degrees
    map_scale_bar_length_km=500,  # Length of the map scale bar in km
    colorbar_width=10,  # Width of the colorbar in cm
    save_fig=True # Save the figure to a file
):
    """
    Plot GNSS residual velocities along with optional histograms.
    
    Args:
    - data (DataFrame): DataFrame containing the residual velocity data with columns ['lon', 'lat', 've_res', 'vn_res'].
    - region (list): List defining the region [west, east, south, north].
    - output_path (str): Path where the output figure will be saved.
    - fault_file (str): Path to the fault trace file for plotting.
    - scale_origin (tuple): Tuple defining the longitude and latitude for the scale vector origin.
    - vel_scaling_factor (float): Scaling factor for the velocity vectors (default is 0.1).
    - plot_residual_histograms (bool): Flag to control whether histograms are plotted (default is True).
    
    Returns:
    - None
    """

    # Create the second figure: Residual Velocities
    fig = pygmt.Figure()

    # Create a custom CPT for the relief shading
    #pygmt.makecpt(cmap="geo", series=[-4000, 4000], output="emed_geo.cpt", transparency=70, background='o', truncate=[-4000, 4000])

   # Step 1: Create custom CPT (color palette table)
    pygmt.makecpt(cmap=relief_colormap, series=colormap_series, output=path2_save_relief_colormap, transparency=70, background='o', truncate=colormap_truncate)
    pygmt.makecpt(cmap=relief_colormap, series=[0, colormap_series[1], 100], output="input_data/cpts/land_cbar.cpt", transparency=70, background='o', truncate=[0, colormap_truncate[1]])
    # Step 2: Modify CPT for land areas, including handling on-land topography below sea level (e.g., -500 to 0)
    with open(path2_save_relief_colormap, 'r') as file:
        lines = file.readlines()

    # Find the color corresponding to the interval [0, 100] (which will be applied to the -500 to 0 range)
    for line in lines:
        if line.startswith("0\t"):
            color_for_zero_to_100 = line.split()[1]
            break

    new_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) > 2 and int(parts[0]) >= -500 and int(parts[2]) <= 0:
            if parts[1] != color_for_zero_to_100:
                parts[1] = color_for_zero_to_100
                parts[3] = color_for_zero_to_100
            new_lines.append("\t".join(parts))
        else:
            new_lines.append(line)

    # Write the modified CPT for land areas
    modified_land_cpt = "input_data/cpts/land_cpt_fixed.cpt"
    with open(modified_land_cpt, "w") as file:
        file.writelines(new_lines)

    # Plot the base map
    fig.basemap(region=region, projection='M20c', frame='af')
    fig.grdimage(grid="@earth_relief_01m", cmap="input_data/cpts/land_cpt_fixed.cpt", shading=True, transparency=70)
    fig.coast(shorelines="0.2p,black", area_thresh=4000, resolution='h', water="white")
    fig.plot(data=fault_file, pen="0.1p,darkgrey")  # Add fault traces

    # Plot residual velocities (gray vectors)
    data['v_mag'] = np.sqrt(data['ve_res']**2 + data['vn_res']**2)
    data['direction'] = np.degrees(np.arctan2(data['vn_res'], data['ve_res']))
    data['length'] = data['v_mag'] * vel_scaling_factor  # Scale factor for plotting

    # Prepare data for plotting residual vectors
    #residual_vector_data = data[['lon', 'lat', 'direction', 'length']].values

    fig.plot(
        x=data['lon'],
        y=data['lat'],
        style='v0.2c+e+n0.3/0.2',
        direction=[data['direction'], data['length']],
        fill='gray25',
        pen='gray25',
        label=f'Residual velocities+S0.5c'
    )

    # Add legend
    fig.legend(position=f'J{residual_legend_location}+j{residual_legend_location}', box='+gwhite+p0.5p,black')

    # Add colorbar
    fig.colorbar(frame="af+lElevation (m)", position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef", cmap="input_data/cpts/land_cbar.cpt")

    # Add scale vectors to the plot
    scale_vector_length = vel_mag_ref_scale * vel_scaling_factor  # in mm/yr, this will be normalised
    scale_vector_length_text = vel_mag_ref_scale  # in mm/yr, this will be displayed on the plot

    scale_vectors = [
        [vel_scale_pos[0], vel_scale_pos[1], 0, scale_vector_length],  # Eastward vector
        [vel_scale_pos[0], vel_scale_pos[1], 90, scale_vector_length]  # Northward vector
    ]

    # Plot the scale vectors
    fig.plot(
        style='v0.2c+e+n0.3/0.2',
        data=scale_vectors,
        fill='gray25',
        pen='0.5p,gray25',
    )

    # Annotate the scale vectors
    fig.text(
        text=f'{scale_vector_length_text} mm/yr',
        x=vel_scale_pos[0] + vel_scale_label_offset[0],
        y=vel_scale_pos[1] + vel_scale_label_offset[1],
        font='7p,black'
    )

    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")

    # Add histograms for residual velocities if enabled
    if plot_residual_histograms:
        # Add transparent rectangle for histogram background
        if histogram_transparent_box_style:
            fig.plot(x=histogram_transparent_box_style['x'], y=histogram_transparent_box_style['y'], style=histogram_transparent_box_style['style'], fill=histogram_transparent_box_style['fill'], pen=histogram_transparent_box_style['pen'], transparency=histogram_transparent_box_style['transparency'])

        # Extract residual velocities
        ve_residuals = data['ve_res']
        vn_residuals = data['vn_res']

        # Plot histograms for East and North components of residual velocities
        with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p', MAP_FRAME_PEN='0.3p,black', MAP_TICK_PEN='0.3p,black'):
            # Plot the histogram for V_e residuals
            fig.shift_origin(xshift=f"{origin_shift_histogram1[0]}c", yshift=f"{origin_shift_histogram1[1]}c")
            fig.histogram(
                data=ve_residuals,
                region=histograms_style['region'],  # Set the region for the histogram
                projection=histograms_style['projection'],  # Set the projection for the histogram
                frame=histograms_style['frame_ve'],  # Labels
                series=histograms_style['series'],  # Histogram range and bin interval
                pen=histograms_style['pen'],   # Outline the histogram bars
                histtype=histograms_style['histtype'],  # Frequency percent
                fill=histograms_style['fill'], # Fill the histogram bars with gray color
                distribution=histograms_style['distribution'],  # Plot the distribution curve
                cumulative=histograms_style['cumulative'],  # Plot the cumulative distribution
            )

            # Plot the histogram for V_n residuals
            fig.shift_origin(xshift=f"{origin_shift_histogram2[0]}c", yshift=f"{origin_shift_histogram2[1]}c")
            fig.histogram(
                data=vn_residuals,
                region=histograms_style['region'],  # Set the region for the histogram
                projection=histograms_style['projection'],  # Set the projection for the histogram
                frame=histograms_style['frame_vn'],  # Labels
                series=histograms_style['series'],  # Histogram range and bin interval
                pen=histograms_style['pen'],  # Outline the histogram bars
                histtype=histograms_style['histtype'],  # Frequency percent
                fill=histograms_style['fill'],  # Fill the histogram bars with gray color
                distribution=histograms_style['distribution'],  # Plot the distribution curve
                cumulative=histograms_style['cumulative'],  # Plot the cumulative distribution
            )

    # Save and display the second figure
    if save_fig:
        fig.savefig(output_path, dpi=600)
    fig.show()

########################################################################################################################
########################################################################################################################

# Function to calculate the initial compass bearing between two points
def calculate_initial_compass_bearing(lon1, lat1, lon2, lat2):
    """
    Calculates the initial compass bearing (azimuth) between two points.
    """
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    diffLong = np.radians(lon2 - lon1)

    x = np.sin(diffLong) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2) -
         (np.sin(lat1) * np.cos(lat2) * np.cos(diffLong)))

    initial_bearing = np.arctan2(x, y)

    # Convert from radians to degrees and normalise to 0-360
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

# Function to calculate the destination point given start point, bearing, and distance
def destination_point(lon1, lat1, bearing, distance):
    """
    Given a start point, initial bearing, and distance, calculate the destination point.
    """
    radius = RADIUS_EARTH_KM  # Earth's radius in km

    bearing = np.radians(bearing)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    distance = distance / radius  # Convert distance to angular distance in radians

    lat2 = np.arcsin(np.sin(lat1) * np.cos(distance) +
                     np.cos(lat1) * np.sin(distance) * np.cos(bearing))

    lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(distance) * np.cos(lat1),
                             np.cos(distance) - np.sin(lat1) * np.sin(lat2))

    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)

    return lon2, lat2

# Updated plot_vel_strain_profiles function
def plot_vel_strain_profiles(
    horiz_vel_field_file,
    vert_vel_field_file,
    strain_rate_df,
    region,
    profiles,
    plot_vertical=True,
    plot_topography_profile=False,  # New parameter
    output_folder="output_data/figures",
    fault_file=None,
    map_scale_bar_length_km=500,
    map_scale_bar_position=[65, 10],
    colorbar_width=8.5,
    relief_colormap='grayC',
    path_to_cpts="input_data/cpts/",
    topography_grid="@earth_relief_30s",
    temp_bbox_directory='input_data/datasets/',
    delete_temp_files=True,
    save_fig=True,
):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_bbox_directory, exist_ok=True)

    # Read horizontal velocity data
    horiz_cols = [
        'Lon', 'Lat', 'E.vel', 'N.vel', 'E.adj', 'N.adj', 'E.sig', 'N.sig',
        'Corr', 'U.vel', 'U.adj', 'U.sig', 'Stat', 'E.sig.scaled', 'N.sig.scaled'
    ]
    horiz_vel_data = pd.read_csv(
        horiz_vel_field_file, sep=r'\s+', header=0, names=horiz_cols
    )

    # Read vertical velocity data
    vert_cols = ['Lon', 'Lat', 'Vz', 'sigma_vz', 'Stat'] #pd.read_csv(vert_vel_field_file, skiprows=1, header=None, , names=vert_cols)
    vert_vel_data = pd.read_csv(vert_vel_field_file, skiprows=1, header=None, names=vert_cols)

    # Normalise longitudes to the -180 to 180 range
    horiz_vel_data['Lon'] = horiz_vel_data['Lon'].apply(lambda x: x - 360 if x > 180 else x)
    vert_vel_data['Lon'] = vert_vel_data['Lon'].apply(lambda x: x - 360 if x > 180 else x)

    # Prepare strain rate data (columns: 'lon', 'lat', 'sec_inv', 'dilat')
    strain_rate_df = strain_rate_df[['lon', 'lat', 'sec_inv', 'dilat']].copy()

    # Create the map figure
    fig_map = pygmt.Figure()

    # Set the region and projection
    fig_map.basemap(region=region, projection='M20c', frame='af')

    # Create the topography colormap
    cpt_topo = os.path.join(path_to_cpts, 'alpides_grey.cpt')
    pygmt.makecpt(cmap=relief_colormap, series=[-1000, 7000, 100], output=cpt_topo, reverse=True)

    # Plot the topography
    fig_map.grdimage(grid=topography_grid, cmap=cpt_topo, shading=True, transparency=70)

    # Add coastlines
    fig_map.coast(
        shorelines="0.1p,black",
        area_thresh=4000,
        resolution='h',
        water='white',
    )

    # Add fault traces if provided
    if fault_file:
        fig_map.plot(data=fault_file, pen="0.1p,darkgrey")

    # Loop over each profile
    for profile in profiles:
        # Extract profile details
        start_lon = profile['start_lon'] - 360 if profile['start_lon'] > 180 else profile['start_lon']
        start_lat = profile['start_lat']
        end_lon = profile['end_lon'] - 360 if profile['end_lon'] > 180 else profile['end_lon']
        end_lat = profile['end_lat']
        width_km = profile['width_km']
        profile_name = profile.get('name', 'Profile')

        # Calculate azimuth and distance
        az12 = calculate_initial_compass_bearing(start_lon, start_lat, end_lon, end_lat)
        profile_length_km = haversine_distance(start_lon, start_lat, end_lon, end_lat)

        # Create bounding box polygon
        def create_profile_bbox(start_lon, start_lat, end_lon, end_lat, width_km):
            width_km_half = width_km / 2.0

            left_azimuth = (az12 - 90) % 360
            right_azimuth = (az12 + 90) % 360

            # Start point
            lon1_left, lat1_left = destination_point(start_lon, start_lat, left_azimuth, width_km_half)
            lon1_right, lat1_right = destination_point(start_lon, start_lat, right_azimuth, width_km_half)

            # End point
            lon2_left, lat2_left = destination_point(end_lon, end_lat, left_azimuth, width_km_half)
            lon2_right, lat2_right = destination_point(end_lon, end_lat, right_azimuth, width_km_half)

            # Polygon coordinates
            bbox_coords = [
                (lon1_left, lat1_left),
                (lon2_left, lat2_left),
                (lon2_right, lat2_right),
                (lon1_right, lat1_right),
                (lon1_left, lat1_left),  # Close the polygon
            ]
            bbox_polygon = Polygon(bbox_coords)
            return bbox_polygon

        bbox_polygon = create_profile_bbox(
            start_lon, start_lat, end_lon, end_lat, width_km
        )

        # Define the temporary file path for the bounding box
        temp_bbox_file_path = os.path.join(temp_bbox_directory, f"{profile_name}_bbox.dat")

        # Write the bounding box coordinates to the file
        with open(temp_bbox_file_path, 'w') as tmpfile_bbox:
            for lon, lat in bbox_polygon.exterior.coords:
                tmpfile_bbox.write(f"{lon} {lat}\n")

        # Plot profile bounding box on the map
        fig_map.plot(
            data=temp_bbox_file_path,
            pen="1p,red",
            close=True,
            transparency=50,
        )

        # Extract GNSS velocities within the bounding box
        def points_in_polygon(lon_series, lat_series, polygon):
            points = [Point(xy) for xy in zip(lon_series, lat_series)]
            return np.array([polygon.contains(point) for point in points])

        # Horizontal velocities
        in_bbox = points_in_polygon(
            horiz_vel_data['Lon'], horiz_vel_data['Lat'], bbox_polygon
        )
        horiz_vel_in_bbox = horiz_vel_data[in_bbox].copy()
        horiz_vel_in_bbox.dropna(subset=['E.vel', 'N.vel'], inplace=True)

        # Vertical velocities
        in_bbox_vert = points_in_polygon(
            vert_vel_data['Lon'], vert_vel_data['Lat'], bbox_polygon
        )
        vert_vel_in_bbox = vert_vel_data[in_bbox_vert].copy()
        vert_vel_in_bbox.dropna(subset=['Vz'], inplace=True)

        # Compute distances along profile for GNSS velocities
        def compute_distances_along_profile(lon_points, lat_points, start_lon, start_lat):
            distances = []
            for lon, lat in zip(lon_points, lat_points):
                lon_normalised = lon - 360 if lon > 180 else lon
                start_lon_normalised = start_lon - 360 if start_lon > 180 else start_lon
                distance = haversine_distance(start_lon_normalised, start_lat, lon_normalised, lat)
                distances.append(distance)
            return np.array(distances)

        horiz_vel_in_bbox['dist_along_profile'] = compute_distances_along_profile(
            horiz_vel_in_bbox['Lon'], horiz_vel_in_bbox['Lat'], start_lon, start_lat
        )

        vert_vel_in_bbox['dist_along_profile'] = compute_distances_along_profile(
            vert_vel_in_bbox['Lon'], vert_vel_in_bbox['Lat'], start_lon, start_lat
        )

        # Project velocities onto profile-parallel and profile-normal components
        profile_azimuth_rad = np.radians((90 - az12) % 360)
        pp_unit_vector = np.array([np.cos(profile_azimuth_rad), np.sin(profile_azimuth_rad)])
        pn_unit_vector = np.array([-np.sin(profile_azimuth_rad), np.cos(profile_azimuth_rad)])

        e_vel = horiz_vel_in_bbox['E.vel'].values
        n_vel = horiz_vel_in_bbox['N.vel'].values
        vel_vectors = np.vstack((e_vel, n_vel)).T

        pp_velocities = vel_vectors @ pp_unit_vector
        pn_velocities = vel_vectors @ pn_unit_vector

        horiz_vel_in_bbox['pp_vel'] = pp_velocities
        horiz_vel_in_bbox['pn_vel'] = pn_velocities

        # Extract strain rate data within the bounding box
        in_bbox_strain = points_in_polygon(
            strain_rate_df['lon'], strain_rate_df['lat'], bbox_polygon
        )
        strain_in_bbox = strain_rate_df[in_bbox_strain].copy()

        # Compute distances along profile for strain rate data
        strain_in_bbox['dist_along_profile'] = compute_distances_along_profile(
            strain_in_bbox['lon'], strain_in_bbox['lat'], start_lon, start_lat
        )

        # Sort strain data by distance along profile
        strain_in_bbox.sort_values(by='dist_along_profile', inplace=True)

        # Generate topography data along the profile
        if plot_topography_profile:
            spacing_km = 10.0  # Adjust the spacing as needed
            track_df = pygmt.project(
                center=[start_lon, start_lat],
                endpoint=[end_lon, end_lat],
                generate=f"{spacing_km}k",
                unit=True,  # Output distances in km
            )

            # Extract elevation data along the profile
            track_df = pygmt.grdtrack(
                points=track_df,
                grid=topography_grid,
                newcolname='elevation',
            )

        # Plot profiles using PyGMT
        fig_profile = pygmt.Figure()

        # Determine the number of subplots
        n_subplots = 5 if plot_vertical else 4
        if plot_topography_profile:
            n_subplots += 1

        # Create subplots with reduced margins
        with fig_profile.subplot(
            nrows=n_subplots,
            ncols=1,
            figsize=("10c", f"{n_subplots * 4}c"),
            margins=["0.5c", "0.0c"],  # Reduced vertical margin
            autolabel="(a)",
            frame="af",
            sharex="b",
        ):
            panel_index = 0

            # Profile-parallel velocities
            with fig_profile.set_panel(panel=panel_index):
                y_min = pp_velocities.min()
                y_max = pp_velocities.max()
                # find the maximum magnitude 
                max_val = max(abs(y_min), abs(y_max))
                fig_profile.basemap(
                    region=[0, profile_length_km, y_min-2, y_max+2],
                    projection="X?",
                    frame=[f'Wsrt+b"{profile_name}: Profile-Parallel Velocities"', "xaf", "yaf+lAlong-profile vel."],
                )
                # Create CPT
                pygmt.makecpt(cmap='roma', series=[-max_val, max_val], reverse=True)
                # Plot data with color
                fig_profile.plot(
                    x=horiz_vel_in_bbox['dist_along_profile'],
                    y=horiz_vel_in_bbox['pp_vel'],
                    style="c0.15c",
                    pen="0.1p,black",
                    fill=horiz_vel_in_bbox['pp_vel'],
                    cmap=True,
                )
                # Add colorbar to the right side
                with pygmt.config(FONT_ANNOT_PRIMARY='18p', FONT_LABEL='18p'):
                    fig_profile.colorbar(
                        frame='af+lmm/yr',
                        position="JMR+o0.3c/0c+w3.5c/0.3c",
                    )
            panel_index += 1

            # Profile-normal velocities
            with fig_profile.set_panel(panel=panel_index):
                y_min = pn_velocities.min()
                y_max = pn_velocities.max()
                # find the maximum magnitude 
                max_val = max(abs(y_min), abs(y_max))
                fig_profile.basemap(
                    region=[0, profile_length_km, y_min-2, y_max+2],
                    projection="X?",
                    frame=["Wsrt", "xaf", "yaf+lAcross-profile vel."],
                )
                # Create CPT
                pygmt.makecpt(cmap='cork', series=[-max_val, max_val])
                # Plot data with color
                fig_profile.plot(
                    x=horiz_vel_in_bbox['dist_along_profile'],
                    y=horiz_vel_in_bbox['pn_vel'],
                    style="c0.15c",
                    pen="0.1p,black",
                    fill=horiz_vel_in_bbox['pn_vel'],
                    cmap=True,
                )
                # Add colorbar to the right 
                with pygmt.config(FONT_ANNOT_PRIMARY='18p', FONT_LABEL='18p'):
                    fig_profile.colorbar(
                        frame='af+lmm/yr',
                        position="JMR+o0.3c/0c+w3.5c/0.3c",
                    )
            panel_index += 1

            # Vertical velocities
            # Check if vertical velocities are available and plot them
            has_vertical_data = not vert_vel_in_bbox.empty
            # I also check if there are at least 3 points for the vertical velocities

            if plot_vertical and has_vertical_data and vert_vel_in_bbox.shape[0] >= 3:
                with fig_profile.set_panel(panel=panel_index):
                    y_min = vert_vel_in_bbox['Vz'].min()
                    y_max = vert_vel_in_bbox['Vz'].max()
                    # find the maximum magnitude 
                    max_val = max(abs(y_min), abs(y_max))
                    fig_profile.basemap(
                        region=[0, profile_length_km, -8, 8],
                        projection="X?",
                        frame=["Wsrt", "xaf", "yaf+lVertical vel."],
                    )
                    # Create CPT
                    pygmt.makecpt(cmap='vik', series=[-6, 6])
                    # Plot data with color
                    fig_profile.plot(
                        x=vert_vel_in_bbox['dist_along_profile'],
                        y=vert_vel_in_bbox['Vz'],
                        style="c0.15c",
                        pen="0.1p,black",
                        fill=vert_vel_in_bbox['Vz'],
                        cmap=True,
                    )
                    # Add colorbar to the right side
                    with pygmt.config(FONT_ANNOT_PRIMARY='18p', FONT_LABEL='18p'):
                        fig_profile.colorbar(
                            frame='af+lmm/yr',
                            position="JMR+o0.3c/0c+w3.5c/0.3c+e",
                        )
                panel_index += 1

            # Strain rate second invariant
            with fig_profile.set_panel(panel=panel_index):
                y_min = strain_in_bbox['sec_inv'].min()
                y_max = strain_in_bbox['sec_inv'].max()
                fig_profile.basemap(
                    region=[0, profile_length_km, 0, y_max+0.1*y_max],
                    projection="X?",
                    frame=["Wsrt", "xaf", "yaf+lSecond invariant"],
                )
                # Create CPT
                pygmt.makecpt(cmap='turbo', series=[y_min, y_max], background='o')
                # Plot data with color
                fig_profile.plot(
                    x=strain_in_bbox['dist_along_profile'],
                    y=strain_in_bbox['sec_inv'],
                    style="c0.15c",
                    pen="0.1p,black",
                    fill=strain_in_bbox['sec_inv'],
                    cmap=True,
                )
                # Add colorbar to the right side
                with pygmt.config(FONT_ANNOT_PRIMARY='18p', FONT_LABEL='18p'):
                    fig_profile.colorbar(
                        frame='af+lnanostrain/yr',
                        position="JMR+o0.3c/0c+w3.5c/0.3c",
                    )
            panel_index += 1

            # Dilatation rate
            with fig_profile.set_panel(panel=panel_index):
                y_min = strain_in_bbox['dilat'].min()
                y_max = strain_in_bbox['dilat'].max()
                max_val = max(abs(y_min), abs(y_max))
                if plot_topography_profile:
                    frame_params = ["Wsrt", "xaf", "yaf+lDilatation rate"]
                else:
                    frame_params = ["WSbrt", "xaf+lDistance along profile (km)", "yaf+lDilatation rate"]
                fig_profile.basemap(
                    region=[0, profile_length_km, y_min-10, y_max+10],
                    projection="X?",
                    frame=frame_params,
                )
                # Create CPT
                pygmt.makecpt(cmap='polar', series=[-max_val, max_val], background='o')
                # Plot data with color
                fig_profile.plot(
                    x=strain_in_bbox['dist_along_profile'],
                    y=strain_in_bbox['dilat'],
                    style="c0.15c",
                    pen="0.1p,black",
                    fill=strain_in_bbox['dilat'],
                    cmap=True,
                )
                # Add colorbar to the right side
                with pygmt.config(FONT_ANNOT_PRIMARY='18p', FONT_LABEL='18p'):
                    fig_profile.colorbar(
                        frame='af+lnanostrain/yr',
                        position="JMR+o0.3c/0c+w3.5c/0.3c",
                    )
            panel_index += 1

            # Topography profile
            if plot_topography_profile:
                # Define the sea level
                sea_level = 0  # Sea level elevation (e.g., 0 meters)

                # Create the arrays for the polygon representing water masses
                water_x = np.concatenate((
                    [track_df['p'].iloc[0]],  # Start x-coordinate
                    track_df['p'],           # Main x-coordinates along the profile
                    [track_df['p'].iloc[-1]],# End x-coordinate
                    [track_df['p'].iloc[0]]  # Close the polygon
                ))

                water_y = np.concatenate((
                    [sea_level],                      # Start at sea level
                    np.minimum(track_df['elevation'], sea_level),  # Clip elevation to sea level
                    [sea_level],                      # Return to sea level
                    [sea_level]                       # Close the polygon
                ))

                with fig_profile.set_panel(panel=panel_index):
                    y_min = track_df['elevation'].min()
                    y_max = track_df['elevation'].max()
                    fig_profile.basemap(
                        region=[0, profile_length_km, y_min, y_max+1500],
                        projection="X?",
                        frame=["WSbrt", "xaf+lDistance along profile (km)", "yaf+lElevation (m)"],
                    )
                    # plot water masses
                    fig_profile.plot(
                        x=water_x,
                        y=water_y,
                        fill="lightblue",  # Fill water with light blue
                        pen=None,          # No outline for water masses
                    )
                    # Plot the topography profile
                    fig_profile.plot(
                        x=track_df['p'],
                        y=track_df['elevation'],
                        pen="1p,black",
                    )
                    # Fill under the curve
                    fig_profile.plot(
                        x=np.concatenate(([track_df['p'].iloc[0]], track_df['p'], [track_df['p'].iloc[-1]])),
                        y=np.concatenate(([y_min], track_df['elevation'], [y_min])),
                        fill="gray",
                    )
                    # Add text annotations (if provided)
                    if 'annotations' in profile and profile['annotations']:
                        for annotation in profile['annotations']:
                            fig_profile.text(
                                x=annotation['x'],
                                y=annotation['y'],
                                text=annotation['text'],
                                font=annotation.get('font', "10p,Times-Italic,black"),
                                offset=annotation.get('offset', "0.0c/0.0c"),
                                justify="CT",  # Center on top by default
                            )
                panel_index += 1

        # Display the profile figure
        fig_profile.show()

        # Save the figure
        if save_fig:
            profile_fig_path = os.path.join(output_folder, f"profile_{profile_name}.pdf")
            fig_profile.savefig(profile_fig_path)
            print(f"Profile saved as {profile_fig_path}")

        # Optionally delete the temporary file
        if delete_temp_files:
            os.remove(temp_bbox_file_path)

    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig_map.basemap(
            map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm"
        )

    # Add color bar for topography
    #fig_map.colorbar(
    #    cmap=cpt_topo,
    #    frame='af+l"Elevation (m)"',
    #    position=f"JMR+o0.5c/0c+w{colorbar_width}c+v+ef",
    #)

    # Display the map figure
    fig_map.show()

    # Save the map figure
    if save_fig:
        map_fig_path = os.path.join(output_folder, "profiles_map.pdf")
        fig_map.savefig(map_fig_path)
        print(f"Map with profiles saved to {map_fig_path}")
########################################################################################################################
########################################################################################################################
