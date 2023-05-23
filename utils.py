from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from matplotlib.colors import LinearSegmentedColormap
import ipywidgets as widgets
from IPython.display import display

YELLOW = "#DDAA33"
RED = "#BB5566"
BLUE = "#004488"
BLACK = "#000000"

NONWHITE_COLORMAP = LinearSegmentedColormap.from_list("YlOrBrNW", [
    "#FEE391",
    "#FEC44F",
    "#FB9A29",
    "#EC7014",
    "#CC4C02",
    "#993404",
    "#662506",
])

def plot_sail(ax: plt3d.Axes3D, vecs: np.ndarray):
    n = vecs.shape[0]

    X = np.stack([vecs[:,0] * i for i in np.linspace(1, 0, n//4)], axis=1)
    Y = np.stack([vecs[:,1] * i for i in np.linspace(1, 0, n//4)], axis=1)
    Z = np.stack([vecs[:,2] * i for i in np.linspace(1, 0, n//4)], axis=1)

    return ax.plot_surface(
        X, Y, Z, 
        cmap=NONWHITE_COLORMAP,
        alpha=0.8)

def plot_vector(ax: plt3d.Axes3D, dir: np.ndarray, arrow=True, **options):
    if arrow:
        return ax.quiver(
            *[ [0] for i in range(3) ],
            *[ [dir[i]] for i in range(3) ],
            length=np.linalg.norm(dir), normalize=False, **options)
    else:
        return ax.plot(
            *[ [0, dir[i]] for i in range(3) ],
            **{k: v for k, v in options.items() if k!="arrow_length_ratio"})
        
def plot_force_profile(
    ax: plt.Axes, 
    elevation: np.ndarray, 
    force_mag: np.ndarray, 
    force_normal_mag: np.ndarray, 
    force_tangent_mag: np.ndarray, 
    sunlight_elevation: float,
    lines: list | None = None
):
    elevation = np.rad2deg(elevation)
    sunlight_elevation = np.rad2deg(sunlight_elevation)

    if lines is None:
        lines = [
            ax.plot(elevation, force_mag, color=BLACK, label="Total")[0],
            ax.plot(elevation, force_normal_mag, color=RED, label="Normal")[0],
            ax.plot(elevation, force_tangent_mag, color=BLUE, label="Tangential")[0],
            ax.axvline(sunlight_elevation, linewidth=0.8, linestyle="--", color=BLACK)
        ]
    
        ax.set_ylabel("Normalized Force [-]")
        ax.set_xlabel("Sunlight Elevation [º]")
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 30, 60, 90])
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")
    
    else:
        lines[0].set_ydata(force_mag)
        lines[1].set_ydata(force_normal_mag)
        lines[2].set_ydata(force_tangent_mag)
        lines[3].set_xdata([sunlight_elevation, sunlight_elevation])

    return lines

def setup_sliders(update_sail):
    sliders = dict(
        sunlight_azimuth = widgets.FloatSlider(value=0, min=0, max=360, step=1, continuous_update=True),
        sunlight_elevation = widgets.FloatSlider(value=0, min=0, max=90, step=1, continuous_update=True),
        reflectivity = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=True),
        specularity = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=True),
        front_lambertian = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=True),
        front_emissivity = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=True),
        back_emissivity = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=True),
        back_lambertian = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=True),
        relative_billow = widgets.FloatSlider(value=0, min=0, max=0.2, step=0.001, continuous_update=False),
        relative_tip_displacement = widgets.FloatSlider(value=0, min=0, max=0.2, step=0.001, continuous_update=False),
    )
    
    out = widgets.interactive_output(update_sail, sliders)

    grid = widgets.GridspecLayout(5, 5, width="950px")
    keys = list(sliders.keys())
    for i in range(0, len(sliders)//2):
        for j in (0, 1):
            key = keys[i*2+j]
            grid[i, j*3] = widgets.Label(key.replace("_", " ").title()+":")
            grid[i, j*3+1] = sliders[key]


    display(grid)
    display(out)

@dataclass
class VecsCache:
    boom_half_length: float
    billow: float
    tip_displacement: float
    n_panels: int
    full: bool
    cached_value: np.ndarray

    def is_hit(self, **kwargs):
        for k, v in kwargs.items():
            if v != getattr(self, k):
                return False
            
        return True
