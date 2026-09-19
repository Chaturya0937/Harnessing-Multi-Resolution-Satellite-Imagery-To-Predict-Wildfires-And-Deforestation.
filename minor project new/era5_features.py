import numpy as np


def calculate_relative_humidity(
    temperature,
    dewpoint
):

    T = temperature - 273.15
    Td = dewpoint - 273.15

    rh = (
        100
        * np.exp(
            (17.625 * Td)
            / (243.04 + Td)
        )
        /
        np.exp(
            (17.625 * T)
            / (243.04 + T)
        )
    )

    return np.clip(
        rh,
        0,
        100
    )


def calculate_wind_speed(
    u,
    v
):

    return np.sqrt(
        u ** 2 + v ** 2
    )


def calculate_wind_direction(
    u,
    v
):

    direction = (
        270
        - np.degrees(
            np.arctan2(v, u)
        )
    ) % 360

    return direction