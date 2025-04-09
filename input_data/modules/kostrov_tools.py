import pandas as pd
import numpy as np
import pygmt
from scipy.interpolate import griddata

def plot_strain_rate_and_moment_tensors(
    kostrov_file,
    region,
    output_file,
    strain_grid_spacing=0.5,
    strain_rate_style_cpt="coolwarm",
    mask_below_sec_inv_value=10, # nanostrain/yr
    plot_moment_tensors=True,  # Plot earthquake moment tensors from GCMT (True/False)
    focal_mechanism_file="input_data/datasets/alpides_psmeca_5.txt",
    moment_tensor_scale=0.5,
    water_color="skyblue",
    shoreline_pen="0.5p,black",
    map_scale_bar_length_km=1000,  # Length of the map scale bar in km
    map_scale_bar_position=[65, 10],  # Position (lon, lat) in degrees
    colorbar_width=8.5,  # Width of the colorbar in cm
    fault_file='input_data/datasets/afead_v2022.gmt',
    input_strain_rate_pd = None,
):
    """
    Plot strain rate style and moment tensors derived from a Kostrov summation file.

    Parameters:
        kostrov_file (str): Path to the Kostrov summation file containing strain rate tensor components.
        region (list): Map region [lon_min, lon_max, lat_min, lat_max].
        output_file (str): Path to save the output map.
        strain_grid_spacing (float): Grid spacing for strain rate interpolation.
        strain_rate_style_cpt (str): Color palette for strain rate style.
        mask_below_sec_inv_value (float): Threshold for masking low strain rates.
        moment_tensor_scale (float): Scale for moment tensors on the map.
        water_color (str): Color for ocean areas.
        shoreline_pen (str): Pen style for coastlines.
    """
    # Load Kostrov file
    data = pd.read_csv(kostrov_file, sep=r"\s+", header=None)

    # Define column names based on known structure
    column_names = [
        "err", "ert", "erp", "ett", "etp", "epp", "lon", "lat"
    ]
    additional_columns = [f"extra_{i}" for i in range(data.shape[1] - len(column_names))]
    column_names.extend(additional_columns)
    data.columns = column_names

    # Correct longitude values to be between -180 and 180
    data.loc[data["lon"] > 180, "lon"] -= 360

    # Calculate strain rate parameters
    data["emean"] = (data["ett"] + data["epp"]) / 2.0
    data["ediff"] = (data["ett"] - data["epp"]) / 2.0
    data["taumax"] = np.sqrt(data["erp"]**2 + data["ediff"]**2)
    data["emax"] = data["emean"] + data["taumax"]
    data["emin"] = data["emean"] - data["taumax"]
    data["sr_style"] = -(data["emax"] + data["emin"]) / (np.abs(data["emax"]) + np.abs(data["emin"]))

    # Interpolate strain rate style to a grid
    valid_data = data[["lon", "lat", "sr_style"]].dropna()
    if valid_data.empty:
        raise ValueError("No valid data available for interpolation after masking.")


    # Apply mask to remove strain rate style values when second invariant from the input strain rate dataframe (GNSS derived) is below threshold
    if input_strain_rate_pd is not None:
        # Ensure valid input strain rate data
        input_strain_rate_pd["lon"] = pd.to_numeric(input_strain_rate_pd["lon"], errors="coerce")
        input_strain_rate_pd["lat"] = pd.to_numeric(input_strain_rate_pd["lat"], errors="coerce")
        input_strain_rate_pd["sec_inv"] = pd.to_numeric(input_strain_rate_pd["sec_inv"], errors="coerce")
        input_strain_rate_pd.dropna(subset=["lon", "lat", "sec_inv"], inplace=True)

        # Crop input data to region
        #input_strain_rate_pd = input_strain_rate_pd[
        #    (input_strain_rate_pd["lon"] >= region[0]) &
        #    (input_strain_rate_pd["lon"] <= region[1]) &
        #    (input_strain_rate_pd["lat"] >= region[2]) &
        #    (input_strain_rate_pd["lat"] <= region[3])
        #]
        if input_strain_rate_pd.empty:
            raise ValueError("Input strain rate dataframe is empty after cropping to region.")

        # Interpolate sec_inv to the Kostrov grid
        strain_rate_interp = griddata(
            (input_strain_rate_pd["lon"], input_strain_rate_pd["lat"]),
            input_strain_rate_pd["sec_inv"],
            (valid_data["lon"], valid_data["lat"]),
            method="linear"
        )
        # Handle NaN values in interpolation
        #strain_rate_interp = np.nan_to_num(strain_rate_interp, nan=0)

        # Apply mask to valid_data
        valid_data["sec_inv"] = strain_rate_interp
        valid_data.loc[valid_data["sec_inv"] < mask_below_sec_inv_value, "sr_style"] = np.nan

    interp_region = [region[0]+1, region[1]-1, region[2], region[3]]

    grid = pygmt.surface(
        data=valid_data,
        spacing=strain_grid_spacing,
        region=interp_region,
        maxradius='150k',
    )

    # Create PyGMT figure
    fig = pygmt.Figure()
    fig.basemap(region=region, projection='M20c', frame='af')

    # Add strain rate style grid
    pygmt.makecpt(cmap=strain_rate_style_cpt, series=[-1, 1], reverse=True, background="o", transparency=30)

    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
            fig.grdimage(grid=grid,
                        cmap=True, shading=False, nan_transparent=True, transparency=30)

    fig.coast(shorelines=shoreline_pen, area_thresh=4000, resolution='h', water=water_color)
    with pygmt.config(FONT_ANNOT_PRIMARY='8p', FONT_LABEL='8p'):
        fig.basemap(map_scale=f"g{map_scale_bar_position[0]}/{map_scale_bar_position[1]}+w{map_scale_bar_length_km}k+f+lkm")
    fig.colorbar(frame='af+lStrain rate style -(@~\\145@~@-1@- + @~\\145@~@-2@-) / (@~\\174\\145@~@-1@-@~\\174@~ + @~\\174\\145@~@-2@-@~\\174@~)',
                 position=f"JMR+o0.5c/0c+w{colorbar_width}c+v")
    fig.plot(data=fault_file, pen="0.1p,darkgrey")

    if plot_moment_tensors:
        fig.meca(
            spec=focal_mechanism_file,
            scale=f"{moment_tensor_scale}c+f0",
            convention="mt",
            component="dc",
            pen="0.2p,black",
            transparency=30,
        )

    # Save and show figure
    fig.savefig(output_file, dpi=300)
    fig.show()