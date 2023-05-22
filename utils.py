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

    ax.plot_surface(
        X, Y, Z, 
        cmap=NONWHITE_COLORMAP,
        #linewidth=1,
        #edgecolor="black",
        alpha=0.8)
    
    ax.set_xlim(-1.2, 1.2, auto=False)
    ax.set_ylim(-1.2, 1.2, auto=False)
    ax.set_zlim(-1.2, 0.2, auto=False)
    ax.set_aspect("equal", "box")
    # ax.axis("off")

def plot_vector(ax: plt3d.Axes3D, dir: np.ndarray, arrow=True, **options):
    if arrow:
        ax.quiver(
            *[ [0] for i in range(3) ],
            *[ [dir[i]] for i in range(3) ],
            length=np.linalg.norm(dir), normalize=False, **options)
    else:
        ax.plot(
            *[ [0, dir[i]] for i in range(3) ],
            **{k: v for k, v in options.items() if k!="arrow_length_ratio"})
        
def plot_force_profile(
    ax: plt.Axes, 
    elevation: np.ndarray, 
    force_mag: np.ndarray, 
    force_normal_mag: np.ndarray, 
    force_tangent_mag: np.ndarray, 
    sunlight_elevation: float
):
    ax.plot(np.rad2deg(elevation), force_mag, color=BLACK, label="Total")
    ax.plot(np.rad2deg(elevation), force_normal_mag, color=RED, label="Normal")
    ax.plot(np.rad2deg(elevation), force_tangent_mag, color=BLUE, label="Tangential")
    ax.axvline(np.rad2deg(sunlight_elevation), linewidth=0.8, linestyle="--", color=BLACK)
    ax.set_ylabel("Normalized Force [-]")
    ax.set_xlabel("Sunlight Elevation [º]")
    ax.set_yticks([0, 1])
    ax.set_xticks([0, 30, 60, 90])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

def setup_widgets(update_sail):
    sliders = dict(
        sunlight_azimuth = widgets.FloatSlider(value=0, min=0, max=360, step=1, continuous_update=False),
        sunlight_elevation = widgets.FloatSlider(value=0, min=0, max=90, step=1, continuous_update=False),
        reflectivity = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=False),
        specularity = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=False),
        front_lambertian = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=False),
        front_emissivity = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=False),
        back_emissivity = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=False),
        back_lambertian = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, continuous_update=False),
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