import numpy as np
import pandas as pd
import pygmt
from scipy.interpolate import griddata
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors
import shapely.geometry
import shapely.vectorized
#from scipy.stats import circstd

# Helper functions
def get_grid_key(lon, lat, grid_size):
    return int(lon // grid_size), int(lat // grid_size)

def haversine_distance(lon1, lat1, lon2, lat2):
    # Calculates distance in km between two points on Earth
    R = 6371  # Radius of Earth in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def find_nearby_stations(strain_lon, strain_lat, grid_size, radius_threshold_km, data_dict, azimuth_field):
    nearby_azimuths = []
    # Get grid key for strain observation
    strain_grid_key = get_grid_key(strain_lon, strain_lat, grid_size)
    
    # Check neighboring grid cells including the current cell
    for dlon in [-1, 0, 1]:
        for dlat in [-1, 0, 1]:
            neighbor_key = (strain_grid_key[0] + dlon, strain_grid_key[1] + dlat)
            if neighbor_key in data_dict:
                for data_row in data_dict[neighbor_key]:
                    distance = haversine_distance(strain_lon, strain_lat, data_row['lon'], data_row['lat'])
                    if distance <= radius_threshold_km:
                        nearby_azimuths.append(data_row[azimuth_field])
    return nearby_azimuths

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

# Create_cpt_from_colormap function
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

# Your existing get_hex_color function
def get_hex_color(value, min_value=-90, max_value=90):
    # Create a colormap using matplotlib's colormap (for azimuth differences between -90 and 90 degrees)
    colormap = plt.get_cmap("coolwarm", 180)
    """Get hex color for a value based on a colormap range."""
    # Normalize the value to be between 0 and 1
    normalized_value = (value - min_value) / (max_value - min_value)
    # Ensure the normalized value is within [0, 1]
    normalized_value = np.clip(normalized_value, 0, 1)
    # Get the RGBA color from the colormap
    rgba_color = colormap(normalized_value)
    # Convert RGBA to hex
    hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba_color[0]*255), int(rgba_color[1]*255), int(rgba_color[2]*255))
    return hex_color

def adjust_sks_azimuths(sks_azimuths):
    """
    Adjust SKS azimuths from 0-360 degrees to -90 to 90 degrees.

    Parameters:
    - sks_azimuths: numpy array or pandas Series of SKS azimuths in degrees (0 to 360)

    Returns:
    - Adjusted SKS azimuths in degrees (-90 to 90)
    """
    adjusted_azimuths = sks_azimuths.copy()
    adjusted_azimuths = np.where(adjusted_azimuths > 180, adjusted_azimuths - 360, adjusted_azimuths)
    adjusted_azimuths = np.where(adjusted_azimuths < -180, adjusted_azimuths + 360, adjusted_azimuths)
    # Now adjust to -90 to 90 range
    adjusted_azimuths = np.where(adjusted_azimuths < -90, adjusted_azimuths + 180, adjusted_azimuths)
    adjusted_azimuths = np.where(adjusted_azimuths > 90, adjusted_azimuths - 180, adjusted_azimuths)
    return adjusted_azimuths


# Function to load strain rate data with variable columns (from previous code)
def load_strain_data(file_path):
    """
    Load strain rate data from a file, handling variable numbers of columns.
    
    Parameters:
    - file_path: Path to the data file.
    
    Returns:
    - data: pandas DataFrame with the appropriate columns.
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
    data = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns, engine='python')

    # Convert relevant columns to numeric, coercing errors to NaN
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop rows with NaN values in critical columns
    critical_columns = ['lon', 'lat', 'Exx_mean', 'Exy_mean', 'Eyy_mean']
    data.dropna(subset=critical_columns, inplace=True)
    
    # If the extra columns are not present, drop any NaN columns
    if 'mean_second_inv' in data.columns and data['mean_second_inv'].isna().all():
        data.drop(['mean_second_inv', 'std_second_inv'], axis=1, inplace=True)
    
    return data

######################

def compute_azimuth_difference(strain_azim, sks_azim):
    """
    Compute the azimuth difference between the strain azimuth and SKS azimuth,
    ensuring the result is between -90 and 90 degrees.

    Parameters:
    - strain_azim: Strain azimuth in degrees (-90 to 90)
    - sks_azim: SKS azimuth in degrees (-90 to 90)

    Returns:
    - Azimuth difference in degrees (-90 to 90)
    """
    # Compute the angular difference
    azim_diff = strain_azim - sks_azim

    # Adjust the difference to be within -180 to 180 degrees
    azim_diff = (azim_diff + 180) % 360 - 180

    # Adjust the difference to be within -90 to 90 degrees
    azim_diff = np.where(azim_diff < -90, azim_diff + 180, azim_diff)
    azim_diff = np.where(azim_diff > 90, azim_diff - 180, azim_diff)

    return azim_diff


#########################################

def plot_azimuths_with_differences(
    strain_data,
    sks_data,
    region,
    output_file,
    fault_file=None,
    STRAIN_RATE_THRESHOLD=0.01,
    RADIUS_THRESHOLD=50,  # km
    GRID_SIZE=1.0,  # degrees
    symbol_size=0.2,  # size of the circles in cm
    cmap_name='vikO',
    cpt_filename='azimuth_diff.cpt',
    map_scale_bar_length_km=1000,
    map_scale_bar_position=[65, 10],
    colorbar_width=8.5,
    ocean_fill_color=[],
    shorelines_pen="0.1p,black",
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
    additional_histograms=None,  # Dictionary for additional histograms
    box_additional_histograms=False,  # Box around additional histograms
    box_additionalhistogram_position=[60, 45],  # Position (lon, lat) in degrees
    box_additional_histograms_style="r4/4",  # Style for the box around additional histograms
    box_additional_histograms_fill="gray",  # Fill color for the box around additional histograms
    box_additional_histograms_transparency=30,  # Transparency for the box around additional histograms
    box_histogram_label = None,  # Label for the box around the histogram
    box_histogram_label_x = 0,  # X position for the label in histogram coordinates
    box_histogram_label_y = 10,  # Y position for the label in histogram coordinates
    cbar_frame='a30f10+lAzimuth difference (max. stretching rate - SKS)',
    save_fig=True,
):
    """
    Plot filled circles at the locations of the strain data points where both the strain azimuth and SKS azimuths are available.
    The circles are colored based on the azimuth difference between the principal extension direction and the median SKS azimuth.
    """
    # Adjust SKS azimuths to be within -90 to 90 degrees
    sks_data = sks_data.copy()
    sks_data['azi_adjusted'] = adjust_sks_azimuths(sks_data['azi'])

    # Filter strain data based on second invariant threshold
    strain_filtered = strain_data[strain_data['sec_inv'] >= STRAIN_RATE_THRESHOLD].copy()

    # Create a dictionary of SKS data points grouped by grid cell
    sks_dict = defaultdict(list)
    for _, sks_row in sks_data.iterrows():
        grid_key = get_grid_key(sks_row['lon'], sks_row['lat'], GRID_SIZE)
        sks_dict[grid_key].append(sks_row)

    # Prepare an empty list to collect the results
    results = []

    # Loop over each strain point
    for i, strain_row in strain_filtered.iterrows():
        strain_lon = strain_row['lon']
        strain_lat = strain_row['lat']
        strain_azim = strain_row['azim_extension']  # Already in -90 to 90 degrees

        # Find SKS observations within the radius threshold
        nearby_sks_rows = find_nearby_stations(
            strain_lon,
            strain_lat,
            GRID_SIZE,
            RADIUS_THRESHOLD,
            sks_dict,
            azimuth_field='azi_adjusted'  # Use the adjusted azimuth
        )

        if len(nearby_sks_rows) >= 1:
            # Collect the adjusted SKS azimuths
            nearby_sks_azims = [row for row in nearby_sks_rows]

            # Compute the median SKS azimuth
            median_sks_azim = np.median(nearby_sks_azims)

            # Compute azimuth difference
            azim_difference = compute_azimuth_difference(strain_azim, median_sks_azim)

            results.append({
                'lon': strain_lon,
                'lat': strain_lat,
                'azim_difference': float(azim_difference),
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Start plotting
    fig = pygmt.Figure()
    fig.basemap(region=region, projection='M20c', frame=True)

    # Add fault traces as very thin grey lines
    if fault_file:
        fig.plot(data=fault_file, pen="0.1p,darkgrey")

    # Create the custom CPT from the colormap
    #create_cpt_from_colormap(cmap_name=cmap_name, filename=cpt_filename, num_colors=256)

    # Load the CPT
    pygmt.makecpt(cmap=cmap_name, series=[-90, 90, 1], continuous=True)

    # Plot the filled circles
    fig.plot(
        x=results_df['lon'],
        y=results_df['lat'],
        style=f'c{symbol_size}c',  # Circle symbol with specified size in cm
        fill=results_df['azim_difference'].values,
        cmap=True,
        pen=None #'0.005p,black'  # Optional: outline the circles with a thin black pen
    )

    # Plot coastlines and country borders for reference
    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen, area_thresh=4000, resolution="h")

    # Add colorbar
    fig.colorbar(
        cmap=True,
        frame=cbar_frame,
        position=f"JMR+o0.5c/0c+w{colorbar_width}c+v"
    )

    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")


    if additional_histograms:
        for hist_name, hist_details in additional_histograms.items():
            # Extract region bounds
            lon_min, lon_max, lat_min, lat_max = hist_details['region']

            # Create the box outline using the region bounds
            fig.plot(
                x=[lon_min, lon_max, lon_max, lon_min, lon_min],  # Close the polygon
                y=[lat_min, lat_min, lat_max, lat_max, lat_min],
                pen="0.75p,black",  # Line style: 1-point thickness, black
            )

            # Plot the label if coordinates and text are provided
            if "map_label" in hist_details and "label_x" in hist_details and "label_y" in hist_details:
                fig.text(
                    x=hist_details["label_x"],
                    y=hist_details["label_y"],
                    text=hist_details["map_label"],
                    font="10p,Helvetica-Bold,black",  # Customize font style and size
                    justify="CM"  # Center alignment
                )

    # Create a histogram of the azimuth differences
    histogram_data = pd.to_numeric(results_df['azim_difference'], errors='coerce').values.tolist()

    if plot_box_for_histogram:
        fig.plot(x=box_histogram_position[0], y=box_histogram_position[1], style=box_histogram_style, fill=box_histogram_fill, pen=None, transparency=box_histogram_transparency)
        #fig.plot(x=box_histogram_position[0]+2, y=box_histogram_position[1], style=box_histogram_style, fill=box_histogram_fill, pen=None, transparency=box_histogram_transparency)
    if box_additional_histograms:
        fig.plot(x=box_additionalhistogram_position[0], y=box_additionalhistogram_position[1], style=box_additional_histograms_style, fill=box_additional_histograms_fill, pen=None, transparency=box_additional_histograms_transparency)
        #fig.plot(x=box_histogram_position[0]+2, y=box_histogram_position[1], style=box_histogram_style, fill=box_histogram_fill, pen=None, transparency=box_histogram_transparency)


    with pygmt.config(FONT_ANNOT_PRIMARY=hist_annot_primary, FONT_LABEL=hist_font_label, MAP_FRAME_PEN='0.3p,black', MAP_TICK_PEN='0.3p,black'):
        # Plot the histogram
        fig.shift_origin(xshift=f"{hist_shift[0]}c", yshift=f"{hist_shift[1]}c")
        fig.histogram(
            data=histogram_data,
            region=[-90, 90, 0, 11],  # Set the region for the histogram: x-range (-90 to 90), y-range (0 to 10)
            projection=hist_projection,  # Set the projection for the histogram
            frame=hist_frame,  # Add labels to the axes
            series=[-90, 90, 10],  # Histogram range (x-range) and bin interval
            cmap=True,  # Use the same custom colormap for the histogram
            pen="0.2p,black",  # Outline the histogram bars
            histtype=1,  # Frequency percent
        )

        # Add label to the box around the main histogram
        if box_histogram_label is not None:
            if not box_histogram_label_x and not box_histogram_label_y:
                print("Please provide the x and y coordinates for the histogram label")
            else:
                fig.text(
                    x=box_histogram_label_x,
                    y=box_histogram_label_y,
                    text=box_histogram_label,
                    font="7p,Helvetica,black",  # Customize font style and size
                )

    # Plot additional histograms
    if additional_histograms:
        for hist_name, hist_details in additional_histograms.items():
            region_data = results_df[
                (results_df['lon'] >= hist_details['region'][0]) &
                (results_df['lon'] <= hist_details['region'][1]) &
                (results_df['lat'] >= hist_details['region'][2]) &
                (results_df['lat'] <= hist_details['region'][3])
            ]['azim_difference']

            # Print region details
            print('-' * 80)
            print(f"{hist_details['label']}. Number of data points: {len(region_data)}")
            print('-' * 80)

            # Print median azimuth difference for the region
            median_azim_diff = region_data.median()
            # Compute MAD (Median Absolute Deviation)
            mad_azim_diff = np.median(np.abs(region_data - median_azim_diff))
            print(f"Median \u00B1 median absolute deviation azimuth difference: {median_azim_diff:.1f} \u00B1 {mad_azim_diff:.1f} degrees")

            # Experimenting with mode and MAD around the mode
            data_range = (-90, 90)

            # Histogram-based mode estimation
            num_bins = 18 # Number of bins for the histogram
            hist, bin_edges = np.histogram(region_data, bins=num_bins, range=data_range)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mode_hist = bin_centers[np.argmax(hist)]

            # Compute median absolute deviation (MAD) from the histogram mode
            mad_from_mode_hist = np.median(np.abs(region_data - mode_hist))

            print(f"Mode ({num_bins} bins) \u00B1 median absolute deviation azimuth difference: {mode_hist:.1f} \u00B1 {mad_from_mode_hist:.1f} degrees")

            # Convert the region's azimuth differences from degrees to radians
            angles_rad = np.radians(region_data)

            # Compute the circular mean
            sin_sum = np.mean(np.sin(angles_rad))
            cos_sum = np.mean(np.cos(angles_rad))
            mean_angle_rad = np.arctan2(sin_sum, cos_sum)
            mean_angle_deg = np.degrees(mean_angle_rad)

            # Compute the mean resultant length R
            R = np.sqrt(sin_sum**2 + cos_sum**2)

            # Compute the circular standard deviation in radians
            # Note: This formula assumes R > 0. For very dispersed data (R near 0),
            # the result might be large or undefined.
            circ_std_rad = np.sqrt(-2 * np.log(R))
            circ_std_deg = np.degrees(circ_std_rad)

            # Some debugging here because it seemed somewhat suspicious that the standard deviations were the same for 2 regions
            #print(f"Region: {hist_details['label']}")
            #print(f"Mean Angle (degrees): {mean_angle_deg:.2f}")
            #print(f"Resultant Length R: {R:.5f}")
            #print(f"Circular Std (degrees): {circ_std_deg:.2f}")
            # Compute circular standard deviation using scipy
            #circ_std_scipy = np.degrees(circstd(angles_rad))
            #print(f"Scipy Circular Std (degrees): {circ_std_scipy:.2f}")
            #print("-" * 40)

            print(f"Circular mean \u00B1 circular standard deviation azimuth difference: {mean_angle_deg:.1f} \u00B1 {circ_std_deg:.1f} degrees")
            print('-' * 80)

            # Shift the origin for the histogram
            fig.shift_origin(xshift=hist_details['shift'][0], yshift=hist_details['shift'][1])

            # Define frame based on histogram number
            #if hist_name == "hist2":  # Add annotations only for the second histogram
            #    histogram_frame = ["ENsw+gwhite", "xf10a30+lAzimuth difference", "yf2.5a5+u%+lFrequency percent"]
            #else:
            #    histogram_frame = ["ENsw+gwhite", "xf10a30+lAzimuth difference", "yf2.5a0"]  # Simplified frame without annotations

            # I know this is a rather unorthodox way to do this, but it works for now:
            if hist_name in ["hist1_with_y_axis_annotations_and_label", "hist2"]:  # Add annotations for specific histograms
                histogram_frame = [
                    "ENsw+gwhite",
                    "xf10a30+lAzimuth difference",
                    "yf2.5a5+u%+lFrequency percent"  # Y-axis annotations
                ]

            elif hist_name == "hist1_with_y_axis_annotations":
                histogram_frame = [
                    "ENsw+gwhite",
                    "xf10a30+lAzimuth difference",
                    "yf2.5a5+u%"  # Y-axis annotations
                ]
            
            else:  # No Y-axis annotations for other histograms
                histogram_frame = [
                    "ENsw+gwhite",
                    "xf10a30+lAzimuth difference",
                    "yf2.5a0"  # No annotations on Y-axis
                ]

            with pygmt.config(FONT_ANNOT_PRIMARY=hist_annot_primary, FONT_LABEL=hist_font_label, MAP_FRAME_PEN='0.3p,black', MAP_TICK_PEN='0.3p,black'):
                fig.histogram(
                    data=region_data, region=hist_details['xybounds'],
                    projection=hist_details['projection'], frame=histogram_frame, # Frame details for the histogram
                    series=[-90, 90, 10], cmap=True, pen="0.2p,black", histtype=1
                )
                # Add label if specified
                if 'label' in hist_details and 'x_pos' in hist_details and 'y_pos' in hist_details:
                    fig.text(
                        x=hist_details['x_pos'],
                        y=hist_details['y_pos'],
                        text=hist_details['label'],
                        font="7p,Helvetica,black"
                    )

    if save_fig:
        fig.savefig(output_file, dpi=600)
    fig.show()

    #return results_df['azim_difference']

#########################################
# Task 3: Plot SKS azimuth directions on a map with optional downsampling
def plot_sks_azimuths(sks_data, strain_data, region, output_file, step=1, step_strain=2,
                      fault_file=None,
                      map_scale_bar_length_km=500,
                      map_scale_bar_position=[65, 10],
                      ocean_fill_color=[],
                      shorelines_pen="0.1p,black",
                      line_scale=0.1,
                      STRAIN_RATE_THRESHOLD=0.01,
                      mask_polygons=None,
                      save_fig=True):
    """
    Plot the SKS azimuth directions from the SKS dataset on a map, allowing to downsample the data.

    Parameters:
    - sks_data: pandas DataFrame with columns 'lon', 'lat', 'azi'
    - region: List defining the region [west, east, south, north]
    - output_file: Path to save the output figure
    - step: Integer, downsampling step
    - fault_file: Path to fault data file
    """
    sks_data = sks_data.copy()
    sks_data_subsampled = sks_data.iloc[::step, :].reset_index(drop=True)

    # Compute eastward and northward components of the azimuth
    sks_data_subsampled['eastward'] = np.sin(np.radians(sks_data_subsampled['azi']))  # X component
    sks_data_subsampled['northward'] = np.cos(np.radians(sks_data_subsampled['azi']))  # Y component

    # Filter our masked strain rate data based on user-defined polygons
    if mask_polygons is not None:
        strain_data = mask_observations_with_polygons(strain_data, mask_polygons)

    # Compute eastward and northward components of the azimuth for extensional strain direction
    strain_data['eastward'] = np.sin(np.radians(strain_data['azim_extension']))  # X component
    strain_data['northward'] = np.cos(np.radians(strain_data['azim_extension']))  # Y component

    fig = pygmt.Figure()
    fig.basemap(region=region, projection='M20c', frame=True)
    if fault_file:
        fig.plot(data=fault_file, pen="0.1p,darkgrey")

    strain_filtered = strain_data[strain_data['sec_inv'] >= STRAIN_RATE_THRESHOLD].copy()
    # subsample strain data
    strain_filtered_subsampled = strain_filtered.iloc[::step_strain, :].reset_index(drop=True)

    # Plot extensional direction from strain rates
    fig.velo(
        data=strain_filtered_subsampled[['lon', 'lat', 'eastward', 'northward', 'azim_extension']],  
        spec="n1",  # Anisotropy bars
        scale=line_scale,  # Scale factor
        pen="4p,36/123/160",  # Line width and colour
    )
    # Plot SKS azimuth bars
    fig.velo(
        data=sks_data_subsampled[['lon', 'lat', 'eastward', 'northward', 'azi']],  
        spec="n1",  # Anisotropy bars
        scale=line_scale,  # Scale factor
        pen="4p,250/140/15",  # Line width and colour. How do we mix the colours? simple... we use https://coolors.co/ to get the RGB values that match the best ;)
    )
    fig.coast(shorelines=shorelines_pen, area_thresh=4000, resolution='h', water=ocean_fill_color)

    # Add scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")

    # Add legend line symbols outside the region. This is a workaround to add a legend to the plot because PyGMT does not support legends for anysortropy bars...
    legend_lon, legend_lat = region[0] - 5, region[2] - 5  # Place outside the region
    fig.plot(x=[legend_lon, legend_lon + 0.5], y=[legend_lat, legend_lat], pen="0.5p,250/140/15", label=f"SKS fast direction+S0.2")
    fig.plot(x=[legend_lon, legend_lon + 0.5], y=[legend_lat, legend_lat], pen="0.5p,36/123/160", label=f"Max. stretching rate+S0.2")

    # Add legend
    fig.legend(position='JTR+jTR', box='+gwhite+p0.5p,black')

    if save_fig:
        fig.savefig(output_file, dpi=600)
    fig.show()

############################################################################################################################################################################
# The functions below were adapted from K. Materna's Strain2D package: https://github.com/kmaterna/Strain_2D/blob/master/Strain_Tools/strain/strain_tensor_toolbox.py

def compute_derived_quantities(exx, exy, eyy):
    """
    Given the basic components of the strain tensor, compute derived quantities
    like 2nd invariant, azimuth of maximum extension, dilatation, etc.
    exx, eyy can be 1D arrays or 2D arrays.

    :param exx: strain component, float or 1D array or 2D array
    :param exy: strain component, float or 1D array or 2D array
    :param eyy: strain component, float or 1D array or 2D array
    :returns: list of derived quantities
    """
    # Compute derived quantities
    max_shear = 0.5 * np.sqrt(np.square(exx - eyy) + 4 * np.square(exy))
    #I2nd = 0.5 * (exx * eyy - np.square(exy)) # original line in K. Materna's code
    I2nd = np.sqrt(exx**2 + 2.0 * exy**2 + eyy**2) # Here I use the J2 instead of I2nd
    dilatation = exx + eyy

    # Compute eigenvalues and eigenvectors
    [e1, e2, v00, v01, v10, v11] = compute_eigenvectors(exx, exy, eyy)

    # Initialize azimuth array
    azimuth = np.zeros(np.shape(exx))

    dshape = np.shape(exx)
    if len(dshape) == 1:
        datalength = dshape[0]
        print("Computing strain invariants for dataset with length %d." % datalength)
        for i in range(datalength):
            azimuth[i] = compute_max_extension_azimuth(e1[i], e2[i], v00[i], v01[i], v10[i], v11[i])
    elif len(dshape) == 2:
        print("Computing strain invariants...")
        for j in range(dshape[0]):
            for i in range(dshape[1]):
                azimuth[j][i] = compute_max_extension_azimuth(e1[j][i], e2[j][i], v00[j][i], v01[j][i],
                                                               v10[j][i], v11[j][i])
    return [I2nd, max_shear, dilatation, azimuth]

def compute_eigenvectors(exx, exy, eyy):
    """
    Compute eigenvalues and eigenvectors of the strain tensor.

    :param exx: strain component, float or array
    :param exy: strain component, float or array
    :param eyy: strain component, float or array
    :returns: eigenvalues and eigenvector components
    """
    e1, e2 = np.zeros(np.shape(exx)), np.zeros(np.shape(exx))  # eigenvalues
    v00, v01 = np.zeros(np.shape(exx)), np.zeros(np.shape(exx))
    v10, v11 = np.zeros(np.shape(exx)), np.zeros(np.shape(exx))  # eigenvectors
    dshape = np.shape(exx)
    if len(dshape) == 1:
        for i in range(len(exx)):
            [e11, e22, v] = eigenvector_eigenvalue(exx[i], exy[i], eyy[i])
            e1[i], e2[i] = e11, e22
            v00[i], v10[i] = v[0][0], v[1][0]
            v01[i], v11[i] = v[0][1], v[1][1]
    elif len(dshape) == 2:
        for j in range(dshape[0]):
            for i in range(dshape[1]):
                [e11, e22, v] = eigenvector_eigenvalue(exx[j][i], exy[j][i], eyy[j][i])
                e1[j][i], e2[j][i] = e11, e22
                v00[j][i], v01[j][i] = v[0][0], v[0][1]
                v10[j][i], v11[j][i] = v[1][0], v[1][1]
    return [e1, e2, v00, v01, v10, v11]

def eigenvector_eigenvalue(exx, exy, eyy):
    """
    Compute eigenvalues and eigenvectors of the strain tensor at a point.

    :param exx: strain component
    :param exy: strain component
    :param eyy: strain component
    :returns: eigenvalues and eigenvectors
    """
    if np.isnan(np.sum([exx, exy, eyy])):
        v = [[np.nan, np.nan], [np.nan, np.nan]]
        return [0, 0, v]
    T = np.array([[exx, exy], [exy, eyy]])  # the tensor
    w, v = np.linalg.eig(T)  # Eigenvalues and eigenvectors
    return [w[0], w[1], v]

def compute_max_extension_azimuth(e1, e2, v00, v01, v10, v11):
    """
    Compute the azimuth of maximum extension.

    :param e1: eigenvalue 1
    :param e2: eigenvalue 2
    :param v00: eigenvector component (v[0][0])
    :param v01: eigenvector component (v[0][1])
    :param v10: eigenvector component (v[1][0])
    :param v11: eigenvector component (v[1][1])
    :returns: azimuth of maximum extension axis, in degrees clockwise from North
    """
    # Identify the eigenvector associated with the maximum eigenvalue (maximum extension)
    if e1 > e2:
        maxv = np.array([v00, v10])  # eigenvector corresponding to e1
    else:
        maxv = np.array([v01, v11])  # eigenvector corresponding to e2

    # Compute the azimuth
    strike = np.arctan2(maxv[1], maxv[0])  # in radians
    theta = 90 - np.degrees(strike)  # Convert to degrees clockwise from North

    # Adjust theta to be within 0 to 180 degrees
    if np.isnan(theta):
        return np.nan
    if theta < 0:
        theta += 180
    elif theta > 180:
        theta -= 180
    assert theta <= 180, "Error: computing an azimuth over 180 degrees."
    return theta
##############################################################################################################################################################################
def plot_apm_sks_azimuth_differences(
    apm_data,  # DataFrame with columns ['lon', 'lat', 'azi']
    sks_data,  # DataFrame with columns ['lon', 'lat', 'azi', 'dt']
    strain_data,  # DataFrame with several columns. Important ones ['lon', 'lat', 'sec_inv']
    mask_below_sec_inv_value,  # Mask out strain data below this value
    region,
    output_file,
    fault_file=None,
    RADIUS_THRESHOLD=50,  # km
    GRID_SIZE=1.0,  # degrees
    symbol_size=0.2,  # size of the circles in cm
    cmap_name='vikO',
    cpt_filename='azimuth_diff.cpt',
    map_scale_bar_length_km=1000,
    map_scale_bar_position=[65, 10],
    colorbar_width=8.5,
    ocean_fill_color=[],
    shorelines_pen="0.1p,black",
    hist_shift=[0.3, 0.2],  # Shift (x,y) for the histogram in cm
    hist_projection="X3.2c/3.2c",  # Projection (xy dimensions) for the histogram in cm
    hist_frame=["ENsw+gwhite", "xf10a30+lAzimuth difference", "yf2.5a5+u%+lFrequency percent"],
    hist_annot_primary="6p",  # Primary annotation font size
    hist_font_label="6p",  # Font size for the histogram labels
    plot_box_for_histogram=True,
    box_histogram_position=[109.5, 26],
    box_histogram_style="r2.7/1.3",
    box_histogram_fill="white",
    box_histogram_transparency=30,
    additional_histograms=None,
    box_additional_histograms=False,
    box_additionalhistogram_position=[60, 45],
    box_additional_histograms_style="r4/4",
    box_additional_histograms_fill="gray",
    box_additional_histograms_transparency=30,
    box_histogram_label=None,
    box_histogram_label_x=0,
    box_histogram_label_y=10,
    cbar_frame='a30f10+lAzimuth difference (APM - SKS)',
    save_fig=True,
):
    """
    Plot filled circles at the locations of the APM data points where both the APM azimuth 
    and SKS azimuths are available. The circles are colored based on the azimuth difference 
    between the APM azimuth (from the field 'azi') and the median SKS azimuth.
    
    This function expects:
      - apm_data: DataFrame with columns ['lon', 'lat', 'azi']
      - sks_data: DataFrame with columns ['lon', 'lat', 'azi', 'dt']
    """
    # Adjust SKS azimuths to be within -90 to 90 degrees
    sks_data = sks_data.copy()
    sks_data['azi_adjusted'] = adjust_sks_azimuths(sks_data['azi'])

    # Group SKS observations by grid cell
    from collections import defaultdict
    sks_dict = defaultdict(list)
    for _, sks_row in sks_data.iterrows():
        grid_key = get_grid_key(sks_row['lon'], sks_row['lat'], GRID_SIZE)
        sks_dict[grid_key].append(sks_row)

    apm_data['eastward'] = np.sin(np.radians(apm_data['azi']))  # X component
    apm_data['northward'] = np.cos(np.radians(apm_data['azi']))  # Y component

    # Interpolate APM eastward and northward components onto the strain grid
    points = np.array([apm_data['lon'], apm_data['lat']]).T
    eastward_interp = griddata(points, apm_data['eastward'],
                            (strain_data['lon'], strain_data['lat']), method='linear')
    northward_interp = griddata(points, apm_data['northward'],
                                (strain_data['lon'], strain_data['lat']), method='linear')

    # Avoid division by zero in case eastward_interp has any zero values
    eastward_interp = np.where(eastward_interp == 0, 1e-10, eastward_interp)

    # Compute the interpolated azimuth from the vector components in degrees
    azim_interp = np.degrees(np.arctan2(eastward_interp, northward_interp))
    azim_interp = np.where(azim_interp > 90, azim_interp - 180, azim_interp)

    # Create a new DataFrame using the strain grid coordinates and add the interpolated azimuth
    apm_interp = pd.DataFrame({
        'lon': strain_data['lon'],
        'lat': strain_data['lat'],
        'azi_interp': azim_interp,
        'sec_inv': strain_data['sec_inv']
    })

    # Mask regions where the strain invariant is below the threshold
    mask_low_strain = apm_interp['sec_inv'] < mask_below_sec_inv_value
    apm_interp.loc[mask_low_strain, 'azi_interp'] = np.nan
    apm_interp = apm_interp.dropna(subset=['azi_interp'])

    # Loop over each APM data point and compute azimuth differences
    results = []
    for i, apm_row in apm_interp.iterrows():
        apm_lon = apm_row['lon']
        apm_lat = apm_row['lat']
        apm_azi = apm_row['azi_interp']  # Interpolated azimuth, correcly computed from the components!!!!!

        # Find nearby SKS observations
        nearby_sks_rows = find_nearby_stations(
            apm_lon,
            apm_lat,
            GRID_SIZE,
            RADIUS_THRESHOLD,
            sks_dict,
            azimuth_field='azi_adjusted'
        )

        if len(nearby_sks_rows) >= 1:
            # Extract the adjusted SKS azimuths from nearby observations
            nearby_sks_azims = [row for row in nearby_sks_rows]
            # Compute the median SKS azimuth
            median_sks_azim = np.median(nearby_sks_azims)
            # Compute the difference between APM and median SKS azimuths
            azim_difference = compute_azimuth_difference(apm_azi, median_sks_azim)

            results.append({
                'lon': apm_lon,
                'lat': apm_lat,
                'azim_difference': float(azim_difference),
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Begin plotting using PyGMT
    fig = pygmt.Figure()
    fig.basemap(region=region, projection='M20c', frame=True)

    # Optionally, plot fault traces
    if fault_file:
        fig.plot(data=fault_file, pen="0.1p,darkgrey")

    # Create/load the custom CPT from the colormap
    # Uncomment the next line if you have a function to create the CPT file
    # create_cpt_from_colormap(cmap_name=cmap_name, filename=cpt_filename, num_colors=256)
    pygmt.makecpt(cmap=cmap_name, series=[-90, 90, 1], continuous=True)

    # Plot the filled circles using the azimuth difference values
    fig.plot(
        x=results_df['lon'],
        y=results_df['lat'],
        style=f'c{symbol_size}c',  # Circle symbol with specified size in cm
        fill=results_df['azim_difference'].values,
        cmap=True,
        pen=None
    )

    # Plot coastlines and country borders
    fig.coast(water=ocean_fill_color, shorelines=shorelines_pen, area_thresh=4000, resolution="h")

    # Add a colorbar
    fig.colorbar(
        cmap=True,
        frame=cbar_frame,
        position=f"JMR+o0.5c/0c+w{colorbar_width}c+v"
    )

    # Add a scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")

    # Optionally plot additional histograms on the map
    if additional_histograms:
        for hist_name, hist_details in additional_histograms.items():
            lon_min, lon_max, lat_min, lat_max = hist_details['region']
            fig.plot(
                x=[lon_min, lon_max, lon_max, lon_min, lon_min],
                y=[lat_min, lat_min, lat_max, lat_max, lat_min],
                pen="0.75p,black",
            )
            if "map_label" in hist_details and "label_x" in hist_details and "label_y" in hist_details:
                fig.text(
                    x=hist_details["label_x"],
                    y=hist_details["label_y"],
                    text=hist_details["map_label"],
                    font="10p,Helvetica-Bold,black",
                    justify="CM"
                )

    # Create a histogram of the azimuth differences
    histogram_data = pd.to_numeric(results_df['azim_difference'], errors='coerce').values.tolist()

    if plot_box_for_histogram:
        fig.plot(x=box_histogram_position[0], y=box_histogram_position[1],
                 style=box_histogram_style, fill=box_histogram_fill, pen=None,
                 transparency=box_histogram_transparency)
    if box_additional_histograms:
        fig.plot(x=box_additionalhistogram_position[0], y=box_additionalhistogram_position[1],
                 style=box_additional_histograms_style, fill=box_additional_histograms_fill, pen=None,
                 transparency=box_additional_histograms_transparency)

    with pygmt.config(FONT_ANNOT_PRIMARY=hist_annot_primary, FONT_LABEL=hist_font_label,
                        MAP_FRAME_PEN='0.3p,black', MAP_TICK_PEN='0.3p,black'):
        fig.shift_origin(xshift=f"{hist_shift[0]}c", yshift=f"{hist_shift[1]}c")
        fig.histogram(
            data=histogram_data,
            region=[-90, 90, 0, 12.5],
            projection=hist_projection,
            frame=hist_frame,
            series=[-90, 90, 10],
            cmap=True,
            pen="0.2p,black",
            histtype=1,
        )
        if box_histogram_label is not None:
            if not box_histogram_label_x and not box_histogram_label_y:
                print("Please provide the x and y coordinates for the histogram label")
            else:
                fig.text(
                    x=box_histogram_label_x,
                    y=box_histogram_label_y,
                    text=box_histogram_label,
                    font="7p,Helvetica,black",
                )

    # Additional histograms for specified regions
    if additional_histograms:
        for hist_name, hist_details in additional_histograms.items():
            region_data = results_df[
                (results_df['lon'] >= hist_details['region'][0]) &
                (results_df['lon'] <= hist_details['region'][1]) &
                (results_df['lat'] >= hist_details['region'][2]) &
                (results_df['lat'] <= hist_details['region'][3])
            ]['azim_difference']

            print('-' * 80)
            print(f"{hist_details['label']}. Number of data points: {len(region_data)}")
            print('-' * 80)

            median_azim_diff = region_data.median()
            mad_azim_diff = np.median(np.abs(region_data - median_azim_diff))
            print(f"Median ± median absolute deviation azimuth difference: {median_azim_diff:.1f} ± {mad_azim_diff:.1f} degrees")

            num_bins = 18
            hist, bin_edges = np.histogram(region_data, bins=num_bins, range=(-90, 90))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mode_hist = bin_centers[np.argmax(hist)]
            mad_from_mode_hist = np.median(np.abs(region_data - mode_hist))
            print(f"Mode ({num_bins} bins) ± median absolute deviation azimuth difference: {mode_hist:.1f} ± {mad_from_mode_hist:.1f} degrees")

            angles_rad = np.radians(region_data)
            sin_sum = np.mean(np.sin(angles_rad))
            cos_sum = np.mean(np.cos(angles_rad))
            mean_angle_rad = np.arctan2(sin_sum, cos_sum)
            mean_angle_deg = np.degrees(mean_angle_rad)
            R = np.sqrt(sin_sum**2 + cos_sum**2)
            circ_std_rad = np.sqrt(-2 * np.log(R))
            circ_std_deg = np.degrees(circ_std_rad)
            print(f"Circular mean ± circular standard deviation azimuth difference: {mean_angle_deg:.1f} ± {circ_std_deg:.1f} degrees")
            print('-' * 80)

            fig.shift_origin(xshift=hist_details['shift'][0], yshift=hist_details['shift'][1])
            if hist_name in ["hist1_with_y_axis_annotations_and_label", "hist2"]:
                histogram_frame = [
                    "ENsw+gwhite",
                    "xf10a30+lAzimuth difference",
                    "yf2.5a5+u%+lFrequency percent"
                ]
            elif hist_name == "hist1_with_y_axis_annotations":
                histogram_frame = [
                    "ENsw+gwhite",
                    "xf10a30+lAzimuth difference",
                    "yf2.5a5+u%"
                ]
            else:
                histogram_frame = [
                    "ENsw+gwhite",
                    "xf10a30+lAzimuth difference",
                    "yf2.5a0"
                ]

            with pygmt.config(FONT_ANNOT_PRIMARY=hist_annot_primary, FONT_LABEL=hist_font_label,
                                MAP_FRAME_PEN='0.3p,black', MAP_TICK_PEN='0.3p,black'):
                fig.histogram(
                    data=region_data, region=hist_details['xybounds'],
                    projection=hist_details['projection'], frame=histogram_frame,
                    series=[-90, 90, 10], cmap=True, pen="0.2p,black", histtype=1
                )
                if 'label' in hist_details and 'x_pos' in hist_details and 'y_pos' in hist_details:
                    fig.text(
                        x=hist_details['x_pos'],
                        y=hist_details['y_pos'],
                        text=hist_details['label'],
                        font="7p,Helvetica,black"
                    )

    if save_fig:
        fig.savefig(output_file, dpi=600)
    fig.show()
