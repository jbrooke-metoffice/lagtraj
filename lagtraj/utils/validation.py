from lagtraj.forcings.profile_calculation import FORCING_VARS


def validate_trajectory(ds_traj):
    required_fields = [
        "lat",
        "lon",
        "time",
        "u_traj",
        "v_traj",
        "origin_lat",
        "origin_lon",
        "origin_datetime",
    ]
    missing_fields = list(filter(lambda f: f not in ds_traj, required_fields))

    if len(missing_fields) > 0:
        raise Exception(
            "The provided trajectory is missing the following"
            " fields: {}".format(", ".join(missing_fields))
        )

    required_attrs = ["name", "domain_name", "trajectory_type"]
    missing_attrs = list(filter(lambda f: f not in ds_traj.attrs, required_attrs))

    if len(missing_attrs) > 0:
        raise Exception(
            "The provided trajectory is missing the following"
            " attrs: {}".format(", ".join(missing_attrs))
        )


def validate_forcing_profiles(ds_forcing_profiles):
    required_vars = ["lat", "lon", "time", "level"]
    required_2d_fields = []
    for v in FORCING_VARS:
        v_fields_2d = [f"{v}_mean", f"{v}_local", f"d{v}dt_adv"]
        required_vars += v_fields_2d
        required_2d_fields += v_fields_2d

    missing_vars = list(filter(lambda f: f not in ds_forcing_profiles, required_vars))

    if len(missing_vars) > 0:
        raise Exception(
            "The provided forcing profiles are missing the"
            " following vars: {}".format(", ".join(missing_vars))
        )

    for v in required_2d_fields:
        assert ds_forcing_profiles[v].dims == ("time", "level")
