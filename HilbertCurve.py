#!/usr/bin/env python
from copy import deepcopy
from functools import cache
from typing import Iterable, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

base_curve = np.array(
    [
        [0.5, 0.5],
        [0.5, 1.5],
        [1.5, 1.5],
        [1.5, 0.5],
    ]
).T

rot90 = np.array([[0, -1], [1, 0]])
rot270 = rot90.T


def get_center_from_size(size: int) -> np.ndarray:
    """Get the corrdinates of the center point of a grid of size `size`.

    Args:
        size (int): The side length/width of the square grid.

    Returns:
        np.ndarray: A (2, 1) array with the coordinates of the center point.
    """
    return np.array([[size // 2] * 2]).T


def rotate_clockwise(a: np.ndarray, size: int) -> np.ndarray:
    """Rotate every point in `a` 90 degrees clockwise around the center of a grid with side length `size`.

    Args:
        a (np.ndarray): A (2, N) array of points.
        size (int): The side length of the grid.

    Returns:
        np.ndarray: An array of the same shape as `a`.
    """
    # Get the coordinates of the center point in the grid
    center = get_center_from_size(size=size)

    # # Translate every point in `a` so that the center lies in origo
    # a_trans = a - center
    # # Rotate every point in `a_trans` 90 degrees clockwise around origo
    # a_trans_rot = rot270 * a_trans
    # # Translate the rotated points back
    # a_rot = a_trans_rot + center

    # Condensed version:
    a_rot = rot270 @ (a - center) + center
    return a_rot


def rotate_counter_clockwise(a: np.ndarray, size: int) -> np.ndarray:
    """Rotate every point in `a` 90 degrees counter-clockwise around the center of a grid with side length `size`.

    Args:
        a (np.ndarray): A (2, N) array of points.
        size (int): The side length of the grid.

    Returns:
        np.ndarray: An array of the same shape as `a`.
    """
    # Get the coordinates of the center point in the grid
    center = get_center_from_size(size=size)

    # Rotate every point in `a` counter-clockwise around `center`.
    # See comments in rotate_clockwise for details on calculation
    a_rot = rot90 @ (a - center) + center
    return a_rot


@cache
def hilbert_curve(order: int = 1) -> np.ndarray:
    """Generate an array of coordinates that together define an Nth order Hilbert Curve.

    Since this function is recursive, the output for each order is cached to save on recalculation.

    Args:
        order (int): The order of the Hilbert Curve to calculate. Each order increases the size of the curve by 2*2.

    Returns:
        np.ndarray: A (2, N) array of points in the order that defines the Nth order Hilbert Curve.
    """
    if order < 1:
        raise ValueError(f"The lowest order Hilbert Curve has order 1, order {order} is not defined")

    elif order == 1:
        return base_curve

    else:
        # Calculate the size of the grid for the given order
        grid_size = 2 ** order

        # Get four copies of the Hilbert curve of the order n-1, all of which lies in the lower left quadrant
        lower_left, upper_left, upper_right, lower_right = [deepcopy(hilbert_curve(order=order - 1)) for _ in range(4)]

        # Rotate the lower left curve clockwise.
        # The size of each semi-curve is half that of the current grid size
        lower_left = rotate_clockwise(lower_left, size=(grid_size // 2))
        # Since the rotation has moved the starting point, we flip the order of the coordinates
        lower_left = np.fliplr(lower_left)

        # Rotate the lower right curve counter-clockwise.
        # The size of each semi-curve is half that of the current grid size
        lower_right = rotate_counter_clockwise(lower_right, size=(grid_size // 2))
        # Since the rotation has moved the starting point, we flip the order of the coordinates
        lower_right = np.fliplr(lower_right)

        # Translate the upper left semi-curve by the y-coordinate of the grid center
        upper_left[1] += get_center_from_size(size=grid_size)[1].item()

        # Translate the upper right semi-curve by the coordinates of the grid center
        upper_right += get_center_from_size(size=grid_size)

        # Translate the lower right semi-curve by the x-coordinate of the grid center
        lower_right[0] += get_center_from_size(size=grid_size)[0].item()

        # Create the complete curve by combining the semi-curves from the four quadrants
        curve = np.concatenate([lower_left, upper_left, upper_right, lower_right], axis=1)

        return curve


def plot_curve(curve: np.ndarray, order: int, show_axes: bool = False, ax: Optional[mpl.axes.Axes] = None) -> None:
    """Plot a Hilbert Curve, specified by `curve`, of order `order`.

    Args:
        curve (np.ndarray): A (2, N) array of coordinates that specifies the `order` order Hilbert Curve.
        order (int): The order of the Hilbert Curve to plot. Used to calculate grid size.
        show_axes (bool): If False, will not show the x and y axes.
        ax (Optional[mpl.axes.Axes]): The axes to plot the curve onto. If None, will create new axes.

    Returns:
        None: Plots curve to `ax`.
    """
    # Calculate the size of the grid for the given order
    grid_size = 2 ** order

    if ax is None:
        fig, ax = plt.subplots()

    # Divide the grid into segments
    ax.hlines(np.arange(0, grid_size + 1), xmin=0, xmax=grid_size, colors="k", linewidths=0.5)
    ax.vlines(np.arange(0, grid_size + 1), ymin=0, ymax=grid_size, colors="k", linewidths=0.5)

    # Plot the curve
    colorline(curve, ax=ax)

    # Set axes limits and title
    ax.set(xlim=(0, grid_size), ylim=(0, grid_size), title=f"Hilbert Curve of order {order}")

    # Make axes square
    plt.axis("square")
    if not show_axes:
        plt.axis("off")

    # Render figure
    plt.show(block=False)


def colorline(
    curve: np.ndarray,
    ax: mpl.axes.Axes,
    z: Optional[Iterable] = None,
    cmap: mpl.colors.LinearSegmentedColormap = plt.get_cmap("rainbow"),
    norm: mpl.colors.Normalize = plt.Normalize(0.0, 1.0),
    linewidth: float = 2.0,
    alpha: float = 0.5,
) -> None:
    """Plot a colored line with coordinates x and y as specified by `curve` onto `ax`.

    Optionally specify colors in the array z.
    Optionally specify a colormap, a norm function, a line width and an alpha.
    Source code from https://stackoverflow.com/a/25941474.

    Args:
        curve (np.ndarray): A (2, N) array of coordinates specifying each point on the curve.
        ax (mpl.axes.Axes): If specified, the curve will be plotted onto these axes.
        z (Optional[Iterable]): An iterable of colors that is passed to mpl.collections.LineCollection.
        cmap (mpl.colors.LinearSegmentedColormap): The colormap used to color each line segment.
            Passed to mpl.collections.LineCollection.
        norm (mpl.colors.Normalize): The normalization values used to normalize the values in `z`.
        linewidth (float): The line width of the line segments of the plot.
        alpha (float): The alpha of the line segments of the plot. Value between 0 and 1.

    Returns:
        None: Adds the generated line collection onto `ax`.
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, curve.shape[1])

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    # Create the line segments
    segments = make_segments(curve)
    lc = mpl.collections.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

    # Add collection of line segments to axes
    ax.add_collection(lc)


def make_segments(points: np.ndarray) -> np.ndarray:
    """Create list of line segments from x and y coordinates, in the correct format for mpl.collections.LineCollection.

    LineCollection expects an array of the form [numlines * (points per line) * 2 (x and y)].
    Source code from https://stackoverflow.com/a/25941474.

    Args:
        points (np.ndarray): A (2, N) array of points that should be connected with line segments.

    Returns:
        np.ndarray: An array of line segments that can be passed to mpl.collections.LineCollection.
    """

    points = points.T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(16, 10))

    for order in range(1, 9):
        curve = hilbert_curve(order=order)
        plot_curve(curve=curve, order=order, show_axes=False, ax=ax)

        input("Press any key to generate the next order curve...")
        ax.clear()
