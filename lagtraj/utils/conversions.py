"""Conversions module"""
import numpy as np
import xarray as xr
import datetime

rg = 9.80665
cp = 1004.0
rlv = 2.5008e6

# Optional numba dependency
try:
    from numba import njit

    print("Running with numba")
except ImportError:

    def njit(numba_function):
        """Dummy numba function"""
        return numba_function

    print("Running without numba")


def add_dict_to_global_attrs(ds_to_add_to, dictionary):
    """Adds global attributes to datasets"""
    for attribute in dictionary:
        ds_to_add_to.attrs[attribute] = dictionary[attribute]


# This can probably be replaced by the generic
# No extrapolation is performed
# Time axis present, but not lat, lon
@njit
def steffen_1d_no_ep_time(
    input_data, input_levels, output_level_array,
):
    """ Performs Steffen interpolation on one individual column.
    Steffen, M. (1990). A simple method for monotonic interpolation in
    one dimension. Astronomy and Astrophysics, 239, 443. """
    t_max = input_data.shape[0]
    k_max = input_data.shape[1]
    k_max_output = output_level_array.shape[0]
    k_max_minus = k_max - 1
    linear_slope = np.empty((k_max))
    output_data = np.empty((t_max, k_max_output))
    # first point
    delta_lower = input_levels[1] - input_levels[0]
    delta_upper = input_levels[2] - input_levels[1]
    if delta_lower < 0:
        raise Exception("Non-montonic increase in input_levels")
    if delta_upper < 0:
        raise Exception("Non-montonic increase in input_levels")
    for time_index in range(t_max):
        slope_lower = (
            input_data[time_index, 1] - input_data[time_index, 0]
        ) / delta_lower
        slope_upper = (
            input_data[time_index, 2] - input_data[time_index, 1]
        ) / delta_upper
        weighted_slope = slope_lower * (
            1 + delta_lower / (delta_lower + delta_upper)
        ) - slope_upper * delta_lower / (delta_lower + delta_upper)
        if weighted_slope * slope_lower <= 0.0:
            linear_slope[0] = 0.0
        elif np.abs(weighted_slope) > 2 * np.abs(slope_lower):
            linear_slope[0] = 2.0 * slope_lower
        else:
            linear_slope[0] = weighted_slope

        # intermediate points
        for k in range(1, k_max_minus):
            delta_lower = input_levels[k] - input_levels[k - 1]
            delta_upper = input_levels[k + 1] - input_levels[k]
            slope_lower = (
                input_data[time_index, k] - input_data[time_index, k - 1]
            ) / delta_lower
            slope_upper = (
                input_data[time_index, k + 1] - input_data[time_index, k]
            ) / delta_upper
            weighted_slope = (slope_lower * delta_upper + slope_upper * delta_lower) / (
                delta_lower + delta_upper
            )

            if slope_lower * slope_upper <= 0.0:
                linear_slope[k] = 0.0
            elif np.abs(weighted_slope) > 2.0 * np.abs(slope_lower):
                linear_slope[k] = np.copysign(2.0, slope_lower) * min(
                    np.abs(slope_lower), np.abs(slope_upper)
                )
            elif np.abs(weighted_slope) > 2.0 * np.abs(slope_upper):
                linear_slope[k] = np.copysign(2.0, slope_lower) * min(
                    np.abs(slope_lower), np.abs(slope_upper)
                )
            else:
                linear_slope[k] = weighted_slope

        # last point
        delta_lower = input_levels[k_max_minus - 1] - input_levels[k_max_minus - 2]
        delta_upper = input_levels[k_max_minus] - input_levels[k_max_minus - 1]
        slope_lower = (
            input_data[time_index, k_max_minus - 1]
            - input_data[time_index, k_max_minus - 2]
        ) / delta_lower
        slope_upper = (
            input_data[time_index, k_max_minus]
            - input_data[time_index, k_max_minus - 1]
        ) / delta_upper
        weighted_slope = slope_upper * (
            1 + delta_upper / (delta_upper + delta_lower)
        ) - slope_lower * delta_upper / (delta_upper + delta_lower)
        if weighted_slope * slope_upper <= 0.0:
            linear_slope[k_max_minus] = 0.0
        elif np.abs(weighted_slope) > 2.0 * np.abs(slope_upper):
            linear_slope[k_max_minus] = 2.0 * slope_upper
        else:
            linear_slope[k_max_minus] = weighted_slope

        # loop over output points
        k_temp = 0
        for k_out in range(k_max_output):
            while (k_temp < k_max) and (
                input_levels[k_temp] < output_level_array[k_out]
            ):
                k_temp = k_temp + 1
            if 0 < k_temp < k_max:
                k_high = k_temp
                k_low = k_high - 1
                delta = input_levels[k_high] - input_levels[k_low]
                slope = (
                    input_data[time_index, k_high] - input_data[time_index, k_low]
                ) / delta
                a = (linear_slope[k_low] + linear_slope[k_high] - 2 * slope) / (
                    delta * delta
                )
                b = (3 * slope - 2 * linear_slope[k_low] - linear_slope[k_high]) / delta
                c = linear_slope[k_low]
                d = input_data[time_index, k_low]
                t_1 = output_level_array[k_out] - input_levels[k_low]
                t_2 = t_1 * t_1
                t_3 = t_2 * t_1
                output_data[time_index, k_out] = a * t_3 + b * t_2 + c * t_1 + d
            else:
                output_data[time_index, k_out] = np.nan
    return output_data


def central_estimate(a_in):
    """take a one-sided difference at the edges, and a central difference elsewhere"""
    return np.concatenate(([a_in[0]], 0.5 * (a_in[1:-1] + a_in[2:]), [a_in[-1]]))


def era5_units_change(da_era5, replacement_dictionary):
    """era5 units : replacement units
    we replace era5 units here"""
    these_units = da_era5.units
    if these_units in replacement_dictionary:
        return replacement_dictionary[these_units]
    else:
        return these_units


def racmo_from_era5(conversion_dict):
    """Obtain a racmo input file from era5 variable set at high resolution"""
    ds_filename = conversion_dict["input_file"]
    ds_era5 = xr.open_dataset(ds_filename)
    racmo_half_array = np.arange(0, 10000.0, 40.0)
    # Put full levels midway between half-levels, I think this is consistent with DALES
    # Change order of data, to confirm to other RACMO input
    racmo_full_array = 0.5 * (racmo_half_array[:-1] + racmo_half_array[1:])
    racmo_half_coord = {
        "nlevp1": (
            "nlevp1",
            (np.arange(len(racmo_half_array)) + 1)[::-1],
            {"long_name": "model half levels"},
        )
    }
    racmo_full_coord = {
        "nlev": (
            "nlev",
            (np.arange(len(racmo_full_array)) + 1)[::-1],
            {"long_name": "model full levels"},
        )
    }
    racmo_soil_coord = {
        "nlevs": ("nlevs", np.arange(4) + 1, {"long_name": "soil levels"})
    }
    ds_racmo = xr.Dataset(
        coords={
            "time": ds_era5.time,
            **racmo_full_coord,
            **racmo_half_coord,
            **racmo_soil_coord,
        }
    )
    # Variables from dictionary
    # Including unit checks
    for variable in racmo_from_era5_variables:
        era5_var = racmo_from_era5_variables[variable]
        da_era5 = ds_era5[era5_var]
        # perform units check
        unit_guess = era5_units_change(da_era5, era5_to_racmo_units)
        if not (unit_guess == racmo_variables[variable]["units"]):
            except_str = (
                "Incompatible units between ERA5 and RACMO for variable "
                + variable
                + ". Please fix using the fix_era5_to_racmo_units dictionary: ERA converted variable is "
                + unit_guess
                + ", RACMO variable is "
                + racmo_variables[variable]["units"]
            )
            raise Exception(except_str)
        # single level variable
        if np.ndim(da_era5.values) == 1:
            ds_racmo[variable] = (("time"), da_era5, racmo_variables[variable])
        # half level variable
        elif variable in ["zh", "presh"]:
            da_era5_on_half_levels = steffen_1d_no_ep_time(
                da_era5.values, ds_era5["lev"].values, racmo_half_array
            )
            da_era5_on_half_levels = da_era5.values
            ds_racmo[variable] = (
                ("time", "nlevp1"),
                da_era5_on_half_levels,
                racmo_variables[variable],
            )
        # full level variable
        else:
            da_era5_on_full_levels = steffen_1d_no_ep_time(
                da_era5.values, ds_era5["lev"].values, racmo_full_array
            )
            ds_racmo[variable] = (
                ("time", "nlev"),
                da_era5_on_full_levels,
                racmo_variables[variable],
            )
    # Simple unit fix fails for these variables
    # So these are added manually after checking
    # that units are compatible
    variables_to_manually_add = {
        "high_veg_type": "tvh",
        "low_veg_type": "tvl",
        "high_veg_lai": "lai_hv",
        "low_veg_lai": "lai_lv",
        "slor": "slor",
        "q_skin": "src",
        "snow": "sd",
        "lsm": "lsm",
    }
    for variable in variables_to_manually_add:
        ds_racmo[variable] = ds_era5[variables_to_manually_add[variable]]
        ds_racmo[variable].assign_attrs(racmo_variables[variable])
    # Soil moisture: combine levels
    swvl1 = ds_era5["swvl1"].values
    swvl2 = ds_era5["swvl2"].values
    swvl3 = ds_era5["swvl3"].values
    swvl4 = ds_era5["swvl4"].values
    q_soil = np.stack((swvl1, swvl2, swvl3, swvl4), axis=-1)
    ds_racmo["q_soil"] = (
        ("time", "nlevs"),
        q_soil,
        racmo_variables["q_soil"],
    )
    # Soil temperature: combine levels
    stl1 = ds_era5["stl1"].values
    stl2 = ds_era5["stl2"].values
    stl3 = ds_era5["stl3"].values
    stl4 = ds_era5["stl4"].values
    t_soil = np.stack((stl1, stl2, stl3, stl4), axis=-1)
    ds_racmo["t_soil"] = (
        ("time", "nlevs"),
        t_soil,
        racmo_variables["t_soil"],
    )
    # Soil thickness: combine levels
    h_soil = np.array([0.07, 0.21, 0.72, 1.89])
    ds_racmo["h_soil"] = (
        ("nlevs"),
        h_soil,
        racmo_variables["h_soil"],
    )
    # Orography: derive from surface geopotential
    ds_racmo["orog"] = ds_era5["z"] / rg
    ds_racmo["orog"].assign_attrs(racmo_variables["orog"])
    # Heat roughness, derive from "flsr" variable
    ds_racmo["heat_rough"] = np.exp(ds_era5["flsr"])
    ds_racmo["heat_rough"].assign_attrs(racmo_variables["heat_rough"])
    # Apply correction to t_skin (see output files)
    ds_racmo["t_skin"] = ds_era5["stl1"] + 1.0
    ds_racmo["t_skin"].assign_attrs(racmo_variables["t_skin"])
    # Surface fluxes: obtain from time mean in ERA data, do not change sign
    sfc_sens_flx = central_estimate(ds_era5["msshf"].values)
    ds_racmo["sfc_sens_flx"] = (
        ("time"),
        sfc_sens_flx,
        racmo_variables["sfc_sens_flx"],
    )
    sfc_lat_flx = central_estimate(ds_era5["mslhf"].values)
    ds_racmo["sfc_lat_flx"] = (
        ("time"),
        sfc_lat_flx,
        racmo_variables["sfc_lat_flx"],
    )
    # Final checks: are all variables present?
    ds_racmo["time_traj"] = (
        ds_era5["time"] - np.datetime64("1970-01-01T00:00")
    ) / np.timedelta64(1, "s")
    ds_racmo["time_traj"].assign_attrs(racmo_variables["time_traj"])
    ds_racmo["DS"] = "Trajectory origin"
    ds_racmo["DS"].assign_attrs(racmo_variables["DS"])
    ds_racmo["timDS"] = ds_era5.datetime_origin
    ds_racmo["timDS"].assign_attrs(racmo_variables["timDS"])
    ds_racmo["latDS"] = ds_era5.lat_origin
    ds_racmo["latDS"].assign_attrs(racmo_variables["latDS"])
    ds_racmo["lonDS"] = ds_era5.lon_origin
    ds_racmo["lonDS"].assign_attrs(racmo_variables["lonDS"])
    # Change order of data, to confirm to other RACMO input
    ds_racmo = ds_racmo.sortby("nlev", ascending=True)
    ds_racmo = ds_racmo.sortby("nlevp1", ascending=True)
    for var in racmo_variables:
        if var not in ds_racmo:
            print(var + " is missing in the RACMO formatted output")
    # Needs improvement
    racmo_dict = {
        "campaign": "NEEDS ADDING",
        "flight": "NEEDS ADDING",
        "date": ds_era5.datetime_origin,
        "source": "ERA5",
        "source_domain": "NEEDS ADDING",
        "source_grid": "grid0.1x0.1",
        "source_latsamp": ds_era5.averaging_width,
        "source_lonsamp": ds_era5.averaging_width,
        "creator": "https://github.com/EUREC4A-UK/lagtraj",
        "created": datetime.datetime.now().isoformat(),
        "wilting_point": 0.1715,
        "field_capacity": 0.32275,
        "t_skin_correct": "Skin temperature has been corrected by 1.000000. Motivation: value from IFS is actually the open SST, which is lower than the skin temperature.",
    }
    add_dict_to_global_attrs(ds_racmo, racmo_dict)
    ds_racmo.to_netcdf("ds_racmo.nc")


# racmo variable : era5 variable
# (we loop over racmo variables here)
racmo_from_era5_variables = {
    "lat": "latitude",
    "lon": "longitude",
    "zf": "height_h",
    "zh": "height_h",
    "ps": "sp",
    "pres": "p_h",
    "presh": "p_h",
    "u": "u",
    "v": "v",
    "t": "t",
    "q": "q",
    "ql": "clwc",
    "qi": "ciwc",
    "cloud_fraction": "cc",
    "omega": "w_pressure_corr",
    "o3": "o3",
    "t_local": "t_local",
    "q_local": "q_local",
    "ql_local": "clwc_local",
    "qi_local": "ciwc_local",
    "u_local": "u_local",
    "v_local": "v_local",
    "cc_local": "cc_local",
    "tadv": "t_advtend",
    "qadv": "q_advtend",
    "uadv": "u_advtend",
    "vadv": "v_advtend",
    "ug": "ug",
    "vg": "vg",
    "tladv": "t_l_advtend",
    "qladv": "clwc_advtend",
    "qiadv": "ciwc_advtend",
    "ccadv": "cc_advtend",
    "lat_traj": "lat_traj",
    "lon_traj": "lon_traj",
    "u_traj": "u_traj",
    "v_traj": "v_traj",
    "albedo": "fal",
    "mom_rough": "fsr",
    "t_snow": "tsn",
    "albedo_snow": "asn",
    "density_snow": "rsn",
    "t_sea_ice": "istl1",
    "open_sst": "sst",
    "high_veg_cover": "cvh",
    "low_veg_cover": "cvl",
    "sea_ice_frct": "siconc",
    "sdor": "sdor",
    "isor": "isor",
    "anor": "anor",
    "msnswrf": "msnswrf",
    "msnlwrf": "msnlwrf",
    "mtnswrf": "mtnswrf",
    "mtnlwrf": "mtnlwrf",
    "mtnswrfcs": "mtnswrfcs",
    "mtnlwrfcs": "mtnlwrfcs",
    "msnswrfcs": "msnswrfcs",
    "msnlwrfcs": "msnlwrfcs",
    "mtdwswrf": "mtdwswrf",
}

racmo_variables = {
    "lat": {"units": "degrees North", "long_name": "latitude"},
    "lon": {"units": "degrees East", "long_name": "longitude"},
    "zf": {"units": "m", "long_name": "full level height"},
    "zh": {"units": "m", "long_name": "half level height"},
    "ps": {"units": "Pa", "long_name": "surface pressure"},
    "pres": {"units": "Pa", "long_name": "full level pressure"},
    "presh": {"units": "Pa", "long_name": "half level pressure"},
    "u": {"units": "m/s", "long_name": "zonal wind (domain averaged)"},
    "v": {"units": "m/s", "long_name": "meridional wind (domain averaged)"},
    "t": {"units": "K", "long_name": "temperature (domain averaged)"},
    "q": {"units": "kg/kg", "long_name": "water vapor mixing ratio (domain averaged)"},
    "ql": {
        "units": "kg/kg",
        "long_name": "liquid water mixing ratio (domain averaged)",
    },
    "qi": {"units": "kg/kg", "long_name": "ice water mixing ratio (domain averaged)"},
    "cloud_fraction": {"units": "0-1", "long_name": "cloud fraction (domain averaged)"},
    "omega": {
        "units": "Pa/s",
        "long_name": "large-scale pressure velocity (domain averaged)",
    },
    "o3": {"units": "kg/kg", "long_name": "ozone mass mixing ratio (domain averaged)"},
    "t_local": {"units": "K", "long_name": "temperature (at domain midpoint)"},
    "q_local": {
        "units": "kg/kg",
        "long_name": "water vapor specific humidity (at domain midpoint)",
    },
    "ql_local": {
        "units": "kg/kg",
        "long_name": "liquid water specific humidity (at domain midpoint)",
    },
    "qi_local": {
        "units": "kg/kg",
        "long_name": "ice water specific humidity (at domain midpoint)",
    },
    "u_local": {"units": "m/s", "long_name": "zonal wind (at domain midpoint)"},
    "v_local": {"units": "m/s", "long_name": "meridional wind (at domain midpoint)"},
    "cc_local": {"units": "0-1", "long_name": "cloud fraction (at domain midpoint)"},
    "tadv": {
        "units": "K/s",
        "long_name": "tendency in temperature due to large-scale horizontal advection",
        "info": "derived at pressure levels",
        "lagrangian": "Lagrangian setup: horizontal advection calculated using velocity relative to wind on trajectory (u_traj,v_traj)",
    },
    "qadv": {
        "units": "kg/kg/s",
        "long_name": "tendency in water vapor due to large-scale horizontal advection",
        "info": "derived at pressure levels",
        "lagrangian": "Lagrangian setup: horizontal advection calculated using velocity relative to wind on trajectory (u_traj,v_traj)",
    },
    "uadv": {
        "units": "m/s2",
        "long_name": "tendency in zonal wind due to large-scale horizontal advection",
        "info": "derived at pressure levels",
        "lagrangian": "Lagrangian setup: horizontal advection calculated using velocity relative to wind on trajectory (u_traj,v_traj)",
    },
    "vadv": {
        "units": "m/s2",
        "long_name": "tendency in meridional wind due to large-scale horizontal advection",
        "info": "derived at pressure levels",
        "lagrangian": "Lagrangian setup: horizontal advection calculated using velocity relative to wind on trajectory (u_traj,v_traj)",
    },
    "ug": {
        "units": "m/s",
        "long_name": "geostrophic wind - zonal component",
        "info": "derived at pressure levels",
        "interpolation": "above 5 hPa the geostrophic wind is equal to the real wind",
    },
    "vg": {
        "units": "m/s",
        "long_name": "geostrophic wind -meridional component",
        "info": "derived at pressure levels",
        "interpolation": "above 5 hPa the geostrophic wind is equal to the real wind",
    },
    "tladv": {
        "units": "K/s",
        "long_name": "tendency in T_l due to large-scale horizontal advection",
    },
    "qladv": {
        "units": "kg/kg/s",
        "long_name": "tendency in liquid water spec hum due to large-scale horizontal advection",
    },
    "qiadv": {
        "units": "kg/kg/s",
        "long_name": "tendency in frozen water due to large-scale horizontal advection",
    },
    "ccadv": {
        "units": "1/s",
        "long_name": "tendency in cloud fraction due to large-scale horizontal advection",
    },
    "time_traj": {
        "units": "seconds since 1-1-1970 00:00",
        "long_name": "time values at trajectory waypoints",
    },
    "lat_traj": {
        "units": "degrees North",
        "long_name": "latitude of trajectory waypoints",
    },
    "lon_traj": {
        "units": "degrees East",
        "long_name": "longitude of trajectory waypoints",
    },
    "u_traj": {"units": "m/s", "long_name": "zonal wind at trajectory waypoints"},
    "v_traj": {"units": "m/s", "long_name": "meridional wind at trajectory waypoints"},
    "fradSWnet": {"units": "W/m2", "long_name": "radiative flux - net short wave"},
    "fradLWnet": {"units": "W/m2", "long_name": "radiative flux - net long wave"},
    "albedo": {"units": "0-1", "long_name": "albedo"},
    "mom_rough": {"units": "m", "long_name": "roughness length for momentum"},
    "heat_rough": {"units": "m", "long_name": "roughness length for heat"},
    "t_skin": {
        "units": "K",
        "long_name": "skin temperature",
        "t_skin_correct": "Skin temperature has been corrected by 1.000000. Motivation: value from IFS is actually the open SST, which is lower than the skin temperature.",
    },
    "q_skin": {"units": "skin reservoir content", "long_name": "m of water"},
    "snow": {"units": "m, liquid equivalent", "long_name": "snow depth"},
    "t_snow": {"units": "K", "long_name": "snow temperature"},
    "albedo_snow": {"units": "0-1", "long_name": "snow albedo"},
    "density_snow": {"units": "kg/m3", "long_name": "snow density"},
    "sfc_sens_flx": {"units": "W/m2", "long_name": "surface sensible heat flux"},
    "sfc_lat_flx": {"units": "W/m2", "long_name": "surface latent heat flux"},
    "h_soil": {"units": "m", "long_name": "soil layer thickness"},
    "t_soil": {"units": "K", "long_name": "soil layer temperature"},
    "q_soil": {"units": "m3/m3", "long_name": "soil moisture"},
    "lsm": {"units": "-", "long_name": "land sea mask"},
    "t_sea_ice": {"units": "K", "long_name": "sea ice temperature"},
    "open_sst": {"units": "K", "long_name": "open sea surface temperature"},
    "orog": {"units": "m", "long_name": "orography - surface height"},
    "DS": {"long_name": "label of trajectory reference point"},
    "timDS": {
        "units": "seconds since 1-1-1970 00:00",
        "long_name": "time at trajectory reference point",
        "info": "the reference point is the space-time coordinate from which the trajectory is calculated",
    },
    "latDS": {
        "units": "degrees North",
        "long_name": "latitude at trajectory reference point",
        "info": "the reference point is the space-time coordinate from which the trajectory is calculated",
    },
    "lonDS": {
        "units": "degrees East",
        "long_name": "longitude at trajectory reference point",
        "info": "the reference point is the space-time coordinate from which the trajectory is calculated",
    },
    "lat_grid": {
        "units": "degrees North",
        "long_name": "latitude of closest IFS gridpoint",
    },
    "lon_grid": {
        "units": "degrees East",
        "long_name": "longitude of closest IFS gridpoint",
    },
    "p_traj": {
        "units": "hPa",
        "long_name": "pressure level at which trajectory was calculated",
    },
    "sv": {"units": "whatever", "long_name": "tracers"},
    "fradSWnet": {"units": "W/m2", "long_name": "radiative flux - net short wave"},
    "fradLWnet": {"units": "W/m2", "long_name": "radiative flux - net long wave"},
    "high_veg_type": {"units": "-", "long_name": "high vegetation type"},
    "low_veg_type": {"units": "-", "long_name": "low vegetation type"},
    "high_veg_cover": {"units": "0-1", "long_name": "high vegetation cover"},
    "low_veg_cover": {"units": "0-1", "long_name": "low vegetation cover"},
    "high_veg_lai": {"units": "-", "long_name": "leaf area index of high vegetation"},
    "low_veg_lai": {"units": "-", "long_name": "leaf area index of low vegetation"},
    "sea_ice_frct": {"units": "0-1", "long_name": "sea ice fraction"},
    "sdor": {
        "units": "0-1",
        "long_name": "subgrid-scale orography - standard deviation",
    },
    "isor": {"units": "0-1", "long_name": "subgrid-scale orography - anisotropy"},
    "anor": {
        "units": "radians",
        "long_name": "subgrid-scale orography - orientation/angle of steepest gradient",
    },
    "slor": {"units": "m/m", "long_name": "subgrid-scale orography - mean slope"},
    "msnswrf": {
        "units": "W/m2",
        "long_name": "Mean surface net short-wave radiation flux",
    },
    "msnlwrf": {
        "units": "W/m2",
        "long_name": "Mean surface net long-wave radiation flux",
    },
    "mtnswrf": {
        "units": "W/m2",
        "long_name": "Mean top net short-wave radiation flux",
    },
    "mtnlwrf": {"units": "W/m2", "long_name": "Mean top net long-wave radiation flux"},
    "mtnswrfcs": {
        "units": "W/m2",
        "long_name": "Mean top net short-wave radiation flux, clear sky",
    },
    "mtnlwrfcs": {
        "units": "W/m2",
        "long_name": "Mean top net long-wave radiation flux, clear sky",
    },
    "msnswrfcs": {
        "units": "W/m2",
        "long_name": "Mean surface net short-wave radiation flux, clear sky",
    },
    "msnlwrfcs": {
        "units": "W/m2",
        "long_name": "Mean surface net long-wave radiation flux, clear sky",
    },
    "mtdwswrf": {
        "units": "W/m2",
        "long_name": "Mean top downward short-wave radiation flux",
    },
}


# era5 units : racmo units
# we replace era5 units here
era5_to_racmo_units = {
    "m s**-1": "m/s",
    "1": "0-1",
    "degrees_north": "degrees North",
    "degrees_east": "degrees East",
    "metres": "m",
    "kg kg**-1": "kg/kg",
    "Pa s**-1": "Pa/s",
    "K s**-1": "K/s",
    "kg kg**-1 s**-1": "kg/kg/s",
    "m s**-1 s**-1": "m/s2",
    "kg m**-3": "kg/m3",
    "s**-1": "1/s",
    "W m**-2": "W/m2",
}


# VARIABLES CURRENTLY LEFT OUT
# 't0' : {'long_name': 'Initial time'}
def hightune_from_era5(conversion_dict):
    """Obtain a hightune input file from era5 variable set at high resolution"""
    ds_filename = conversion_dict["input_file"]
    ds_era5 = xr.open_dataset(ds_filename)
    ds_hightune = xr.Dataset(coords={"time": ds_era5.time, "lev": ds_era5.lev,})
    # Variables from dictionary
    # Including unit checks
    for variable in hightune_from_era5_variables:
        era5_var = hightune_from_era5_variables[variable]
        da_era5 = ds_era5[era5_var]
        # perform units check
        unit_guess = era5_units_change(da_era5, era5_to_hightune_units)
        if not (unit_guess == hightune_variables[variable]["units"]):
            except_str = (
                "Incompatible units between ERA5 and hightune for variable "
                + variable
                + ". Please fix using the fix_era5_to_hightune_units dictionary: ERA converted variable is "
                + unit_guess
                + ", hightune variable is "
                + hightune_variables[variable]["units"]
            )
            raise Exception(except_str)
        ds_hightune[variable] = da_era5
    # TKE, set to zero
    ds_hightune["tke"] = 0.0 * (ds_era5["u"] * ds_era5["u"])
    ds_hightune["tke"].assign_attrs(hightune_variables["tke"])
    # Heat roughness, derive from "flsr" variable
    ds_hightune["z0h"] = np.exp(ds_era5["flsr"])
    ds_hightune["z0h"].assign_attrs(hightune_variables["z0h"])
    # Need to check t_skin correction
    ds_hightune["ts"] = ds_era5["stl1"] + 1.0
    ds_hightune["ts"].assign_attrs(hightune_variables["ts"])
    # Surface fluxes: obtain from time mean in ERA data, change sign for hightune!
    sfc_sens_flx = -central_estimate(ds_era5["msshf"].values)
    ds_hightune["sfc_sens_flx"] = (
        ("time"),
        sfc_sens_flx,
        hightune_variables["sfc_sens_flx"],
    )
    sfc_lat_flx = -central_estimate(ds_era5["mslhf"].values)
    ds_hightune["sfc_lat_flx"] = (
        ("time"),
        sfc_lat_flx,
        hightune_variables["sfc_lat_flx"],
    )
    wpthetap = (sfc_sens_flx / (cp * ds_era5["rho"].sel(lev=0.0))) * (
        ds_era5["theta"].sel(lev=0.0) / ds_era5["t"].sel(lev=0.0)
    )
    ds_hightune["wpthetap"] = (
        ("time"),
        wpthetap,
        hightune_variables["wpthetap"],
    )
    wpqvp = sfc_lat_flx / (rlv * ds_era5["rho"].sel(lev=0.0))
    ds_hightune["wpqvp"] = (
        ("time"),
        wpqvp,
        hightune_variables["wpqvp"],
    )
    wpqtp = wpqvp
    ds_hightune["wpqtp"] = (
        ("time"),
        wpqtp,
        hightune_variables["wpqtp"],
    )
    # This should be the same for all variables
    moisture_ratio = ds_era5["r_t"].sel(lev=0.0) / ds_era5["q_t"].sel(lev=0.0)
    wprvp = wpqvp * moisture_ratio
    ds_hightune["wprvp"] = (
        ("time"),
        wprvp,
        hightune_variables["wprvp"],
    )
    wprtp = wpqtp * moisture_ratio
    ds_hightune["wprtp"] = (
        ("time"),
        sfc_lat_flx,
        hightune_variables["wprtp"],
    )
    # Final checks: are all variables present?
    for var in hightune_variables:
        if var not in ds_hightune:
            print(var + " is missing in the hightune formatted output")
    # Needs improvement
    hightune_dictionary = {
        "Conventions": "CF-1.0",
        "comment": "Forcing and initial conditions for Lagrangian case",
        "reference": "NEEDS ADDING",
        "author": "NEEDS ADDING",
        "modifications": "NEEDS ADDING",
        "case": "NEEDS ADDING",
        "script": "https://github.com/EUREC4A-UK/lagtraj",
        "startDate": ds_hightune["time"][0],
        "endDate": ds_hightune["time"][-1],
        "tadv": 0,
        "tadvh": 1,
        "tadvv": 0,
        "rad_temp": 1,
        "qvadv": 0,
        "qvadvh": 1,
        "qvadvv": 0,
        "forc_omega": 0,
        "forc_w": 1,
        "forc_geo": 1,
        "nudging_u": 0,
        "nudging_v": 0,
        "nudging_t": 0,
        "nudging_q": 0,
        "zorog": 0.0,
        "surfaceType": "ocean",
        "surfaceForcing": "ts",
        "surfaceForcingWind": "z0_lagtraj",
    }
    ds_hightune.to_netcdf("ds_hightune.nc")


# hightune variable : era5 variable
# we loop over hightune variables here
hightune_from_era5_variables = {
    "lat": "latitude",
    "lon": "longitude",
    "ps": "sp",
    "height": "height_h",
    "pressure": "p_h",
    "u": "u",
    "v": "v",
    "temp": "t",
    "theta": "theta",
    "thetal": "theta_l",
    "qv": "q",
    "qt": "q_t",
    "rv": "r_v",
    "rt": "r_t",
    "rl": "r_l",
    "ri": "r_i",
    "ql": "clwc",
    "qi": "ciwc",
    "ps": "sp",
    "ps_forc": "sp",
    "height_forc": "height_h",
    "pressure_forc": "p_h",
    "ug": "ug",
    "vg": "vg",
    "temp_adv": "t_advtend",
    "theta_adv": "theta_advtend",
    "thetal_adv": "theta_l_advtend",
    "qv_adv": "q_advtend",
    "qt_adv": "q_t_advtend",
    "rv_adv": "r_v_advtend",
    "rt_adv": "r_t_advtend",
    "w": "w_corr",
    "temp_nudging": "t",
    "theta_nudging": "theta",
    "thetal_nudging": "theta_l",
    "qv_nudging": "q",
    "qt_nudging": "q_t",
    "rv_nudging": "r_v",
    "rt_nudging": "r_t",
    "u_nudging": "u",
    "v_nudging": "v",
    "z0m": "fsr",
}


hightune_variables = {
    "lat": {"long_name": "Latitude", "units": "degrees_north"},
    "lon": {"long_name": "Longitude", "units": "degrees_east"},
    "ps": {"long_name": "Surface pressure", "units": "Pa"},
    "height": {"long_name": "Height above ground", "units": "m"},
    "pressure": {"long_name": "Pressure", "units": "Pa"},
    "u": {"long_name": "Zonal wind", "units": "m s-1"},
    "v": {"long_name": "Meridional wind", "units": "m s-1"},
    "temp": {"long_name": "Temperature", "units": "K"},
    "theta": {"long_name": "Potential temperature", "units": "K"},
    "thetal": {"long_name": "Liquid potential temperature", "units": "K"},
    "qv": {"long_name": "Specific humidity", "units": "kg kg-1"},
    "qt": {"long_name": "Total water content", "units": "kg kg-1"},
    "rv": {"long_name": "Water vapor mixing ratio", "units": "kg kg-1"},
    "rt": {"long_name": "Total water mixing ratio", "units": "kg kg-1"},
    "rl": {"long_name": "Liquid water mixing ratio", "units": "kg kg-1"},
    "ri": {"long_name": "Ice water mixing ratio", "units": "kg kg-1"},
    "ql": {"long_name": "Liquid water content", "units": "kg kg-1"},
    "qi": {"long_name": "Ice water content", "units": "kg kg-1"},
    "tke": {"long_name": "Turbulent kinetic energy", "units": "m2 s-2"},
    "time": {"long_name": "Forcing time"},
    "ps_forc": {"long_name": "Surface pressure for forcing", "units": "Pa"},
    "height_forc": {"long_name": "Height for forcing", "units": "m"},
    "pressure_forc": {"long_name": "Pressure for forcing", "units": "Pa"},
    "ug": {"long_name": "Geostrophic zonal wind", "units": "m s-1"},
    "vg": {"long_name": "Geostrophic meridional wind", "units": "m s-1"},
    "temp_adv": {"long_name": "Temperature large-scale advection", "units": "K s-1"},
    "theta_adv": {
        "long_name": "Potential temperature large-scale advection",
        "units": "K s-1",
    },
    "thetal_adv": {
        "long_name": "Liquid potential temperature large-scale advection",
        "units": "K s-1",
    },
    "qv_adv": {
        "long_name": "Specific humidity large-scale advection",
        "units": "kg kg-1 s-1",
    },
    "qt_adv": {
        "long_name": "Total water content large-scale advection",
        "units": "kg kg-1 s-1",
    },
    "rv_adv": {
        "long_name": "Water vapor mixing ratio large-scale advection",
        "units": "kg kg-1 s-1",
    },
    "rt_adv": {
        "long_name": "Total water mixing ratio large-scale advection",
        "units": "kg kg-1 s-1",
    },
    "w": {"long_name": "Vertical velocity", "units": "m s-1"},
    "ts": {"long_name": "Surface temperature", "units": "K"},
    "ps": {"long_name": "Surface pressure", "units": "Pa"},
    "rh": {"long_name": "Relative humidity", "units": "%"},
    "temp_nudging": {"long_name": "Temperature for nudging", "units": "K"},
    "theta_nudging": {"long_name": "Potential temperature for nudging", "units": "K"},
    "thetal_nudging": {
        "long_name": "Liquid potential temperature for nudging",
        "units": "K",
    },
    "qv_nudging": {
        "long_name": "Specific humidity profile for nudging",
        "units": "kg kg-1",
    },
    "qt_nudging": {
        "long_name": "Total water content profile for nudging",
        "units": "kg kg-1",
    },
    "rv_nudging": {
        "long_name": "Water vapor mixing ratio profile for nudging",
        "units": "kg kg-1",
    },
    "rt_nudging": {
        "long_name": "Total water mixing ratio profile for nudging",
        "units": "kg kg-1",
    },
    "u_nudging": {"long_name": "Zonal wind profile for nudging", "units": "m s-1"},
    "v_nudging": {"long_name": "Meridional wind profile for nudging", "units": "m s-1"},
    "sfc_sens_flx": {
        "long_name": "Surface sensible heat flux (positive upward)",
        "units": "W m-2",
    },
    "sfc_lat_flx": {
        "long_name": "Surface latent heat flux (positive upward)",
        "units": "W m-2",
    },
    "wpthetap": {
        "long_name": "Surface flux of potential temperature",
        "units": "K m s-1",
    },
    "wpqvp": {
        "long_name": "Surface flux of water vapor specific humidity",
        "units": "m s-1",
    },
    "wpqtp": {
        "long_name": "Surface flux of total water specific humidity",
        "units": "m s-1",
    },
    "wprvp": {
        "long_name": "Surface flux of water vapor mixing ratio",
        "units": "m s-1",
    },
    "wprtp": {
        "long_name": "Surface flux of total water mixing ratio",
        "units": "m s-1",
    },
    "z0m": {"units": "m", "long_name": "roughness length for momentum"},
    "z0h": {"units": "m", "long_name": "roughness length for heat"},
}


# era5 units : hightune units
# we replace era5 units here
era5_to_hightune_units = {
    "m s**-1": "m s-1",
    "metres": "m",
    "kg kg**-1": "kg kg-1",
    "K s**-1": "K s-1",
    "kg kg**-1 s**-1": "kg kg-1 s-1",
}
