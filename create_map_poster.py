from matplotlib.figure import Figure
from networkx import MultiDiGraph
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from geopy.geocoders import Nominatim
from opencage.geocoder import OpenCageGeocode
from opencage.geocoder import InvalidInputError, RateLimitExceededError, UnknownError
from tqdm import tqdm
import time
import json
import os
import sys
import asyncio
import argparse
import pickle
from typing import cast, Optional, Tuple, List, Dict
from geopandas import GeoDataFrame

# --- Configuration ---
CACHE_DIR = ".cache"
THEMES_DIR = "themes"
POSTERS_DIR = "posters"
FONTS_DIR = "fonts"

# Ensure directories exist
for d in [CACHE_DIR, THEMES_DIR, POSTERS_DIR]:
    os.makedirs(d, exist_ok=True)

# Load Fonts if available
FONTS = {}
if os.path.exists(FONTS_DIR):
    for f in os.listdir(FONTS_DIR):
        if f.endswith(".ttf"):
            if "Bold" in f.lower(): FONTS['bold'] = os.path.join(FONTS_DIR, f)
            elif "Light" in f.lower(): FONTS['light'] = os.path.join(FONTS_DIR, f)
            elif "Regular" in f.lower(): FONTS['regular'] = os.path.join(FONTS_DIR, f)

# Global Theme Placeholder
THEME = {}

class CacheError(Exception):
    pass

def cache_get(key: str):
    """Retrieve data from local pickle cache."""
    safe_key = "".join([c if c.isalnum() else "_" for c in key])
    cache_path = os.path.join(CACHE_DIR, f"{safe_key}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def cache_set(key: str, data):
    """Save data to local pickle cache."""
    safe_key = "".join([c if c.isalnum() else "_" for c in key])
    cache_path = os.path.join(CACHE_DIR, f"{safe_key}.pkl")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        raise CacheError(f"Failed to write cache: {e}")

def get_available_themes() -> List[str]:
    """Return list of theme names based on json files in themes directory."""
    themes = []
    if os.path.exists(THEMES_DIR):
        for f in os.listdir(THEMES_DIR):
            if f.endswith(".json"):
                themes.append(f[:-5])
    return sorted(themes)

def load_theme(theme_name: str) -> Dict:
    """Load theme configuration from JSON file."""
    theme = {
        "bg": "#000000",
        "text": "#FFFFFF",
        "water": "#001133",
        "parks": "#002211",
        "gradient_color": "#000000",
        "road_motorway": "#FFFFFF",
        "road_primary": "#DDDDDD",
        "road_secondary": "#BBBBBB",
        "road_tertiary": "#999999",
        "road_residential": "#666666",
        "road_default": "#666666"
    }
    path = os.path.join(THEMES_DIR, f"{theme_name}.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                custom = json.load(f)
                theme.update(custom)
        except Exception as e:
            print(f"Warning: Failed to load theme file {path}: {e}")
    return theme

def generate_output_filename(city: str, theme_name: str, fmt: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_city = city.lower().replace(" ", "_")
    return os.path.join(POSTERS_DIR, f"{safe_city}_{theme_name}_{timestamp}.{fmt}")

def get_edge_colors_by_type(G: MultiDiGraph) -> List[str]:
    colors = []
    for _, _, data in G.edges(data=True):
        highway = data.get("highway")
        if isinstance(highway, list): highway = highway[0]
        if highway in ["motorway", "motorway_link"]: colors.append(THEME.get("road_motorway"))
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]: colors.append(THEME.get("road_primary"))
        elif highway in ["secondary", "secondary_link"]: colors.append(THEME.get("road_secondary"))
        elif highway in ["tertiary", "tertiary_link"]: colors.append(THEME.get("road_tertiary"))
        elif highway in ["residential", "living_street", "unclassified"]: colors.append(THEME.get("road_residential"))
        else: colors.append(THEME.get("road_default"))
    return colors

def get_edge_widths_by_type(G: MultiDiGraph) -> List[float]:
    widths = []
    for _, _, data in G.edges(data=True):
        highway = data.get("highway")
        if isinstance(highway, list): highway = highway[0]
        if highway in ["motorway", "motorway_link"]: widths.append(1.2)
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]: widths.append(1.0)
        elif highway in ["secondary", "secondary_link"]: widths.append(0.8)
        elif highway in ["tertiary", "tertiary_link"]: widths.append(0.6)
        else: widths.append(0.4)
    return widths

def create_gradient_fade(ax, color: str, location: str = 'bottom', zorder: int = 10):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    if location == 'bottom':
        gradient = gradient.T
        extent = [0, 1, 0, 0.25]
    else:
        gradient = np.flip(gradient.T, axis=0)
        extent = [0, 1, 0.75, 1]
    rgb = mcolors.hex2color(color)
    cdict = {
        'red':   [(0.0, rgb[0], rgb[0]), (1.0, rgb[0], rgb[0])],
        'green': [(0.0, rgb[1], rgb[1]), (1.0, rgb[1], rgb[1])],
        'blue':  [(0.0, rgb[2], rgb[2]), (1.0, rgb[2], rgb[2])],
        'alpha': [(0.0, 1.0, 1.0),       (1.0, 0.0, 0.0)]
    }
    cmap = LinearSegmentedColormap('fade_cmap', cdict)
    ax.imshow(gradient, aspect='auto', extent=extent, cmap=cmap, zorder=zorder, transform=ax.transAxes)

def get_coordinates(city: str, country: str) -> Tuple[float, float]:
    coords_key = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords_key)
    if cached:
        print(f"✓ Using cached coordinates for {city}, {country}")
        return cached

    opencage_key = os.environ.get("OPENCAGE_API_KEY")
    if opencage_key:
        try:
            geocoder = OpenCageGeocode(opencage_key)
            results = geocoder.geocode(f"{city}, {country}", no_annotations='1', timeout=30)
            if results:
                lat, lon = results[0]['geometry']['lat'], results[0]['geometry']['lng']
                cache_set(coords_key, (lat, lon))
                return (lat, lon)
        except Exception as e:
            print(f"⚠ OpenCage Error: {e}")

    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    time.sleep(1)
    location = geolocator.geocode(f"{city}, {country}")
    if location:
        cache_set(coords_key, (location.latitude, location.longitude))
        return (location.latitude, location.longitude)
    raise ValueError(f"Could not find coordinates for {city}, {country}")

def get_crop_limits(G: MultiDiGraph, fig: Figure) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = [data['x'] for _, data in G.nodes(data=True)]
    ys = [data['y'] for _, data in G.nodes(data=True)]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    x_range, y_range = maxx - minx, maxy - miny
    fig_width, fig_height = fig.get_size_inches()
    desired_aspect, current_aspect = fig_width / fig_height, x_range / y_range
    center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
    if current_aspect > desired_aspect:
        desired_x_range = y_range * desired_aspect
        return (center_x - desired_x_range / 2, center_x + desired_x_range / 2), (miny, maxy)
    else:
        desired_y_range = x_range / desired_aspect
        return (minx, maxx), (center_y - desired_y_range / 2, center_y + desired_y_range / 2)

def fetch_graph(point: Tuple[float, float], dist: int) -> Optional[MultiDiGraph]:
    key = f"graph_{point[0]}_{point[1]}_{dist}"
    cached = cache_get(key)
    if cached: return cast(MultiDiGraph, cached)
    try:
        G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all')
        time.sleep(0.5)
        cache_set(key, G)
        return G
    except Exception as e:
        print(f"OSMnx error: {e}")
        return None

def fetch_features(point: Tuple[float, float], dist: int, tags: Dict, name: str) -> Optional[GeoDataFrame]:
    tag_str = "_".join(sorted(tags.keys()))
    key = f"{name}_{point[0]}_{point[1]}_{dist}_{tag_str}"
    cached = cache_get(key)
    if cached: return cast(GeoDataFrame, cached)
    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        time.sleep(0.3)
        cache_set(key, data)
        return data
    except Exception as e:
        print(f"OSMnx error: {e}")
        return None

def create_poster(city: str, country: str, point: Tuple[float, float], dist: int, output_file: str, output_format: str, country_label: str = None):
    print(f"\nGenerating map for {city}...")
    with tqdm(total=3, desc="Fetching data") as pbar:
        G = fetch_graph(point, dist)
        if not G: raise RuntimeError("Failed to fetch street network.")
        pbar.update(1)
        water = fetch_features(point, dist, {'natural': 'water', 'waterway': 'riverbank'}, 'water')
        pbar.update(1)
        parks = fetch_features(point, dist, {'leisure': 'park', 'landuse': 'grass'}, 'parks')
        pbar.update(1)

    fig, ax = plt.subplots(figsize=(12, 16), facecolor=THEME['bg'])
    ax.set_facecolor(THEME['bg'])
    ax.set_position((0, 0, 1, 1))
    G_proj = ox.project_graph(G)

    for data, color, z in [(water, THEME['water'], 1), (parks, THEME['parks'], 2)]:
        if data is not None and not data.empty:
            polys = data[data.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            if not polys.empty:
                try: polys = ox.projection.project_gdf(polys)
                except: polys = polys.to_crs(G_proj.graph['crs'])
                polys.plot(ax=ax, facecolor=color, edgecolor='none', zorder=z)

    ox.plot_graph(G_proj, ax=ax, bgcolor=THEME['bg'], node_size=0, edge_color=get_edge_colors_by_type(G_proj), edge_linewidth=get_edge_widths_by_type(G_proj), show=False, close=False)
    xlim, ylim = get_crop_limits(G_proj, fig)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    create_gradient_fade(ax, THEME['gradient_color'], 'bottom'); create_gradient_fade(ax, THEME['gradient_color'], 'top')

    f_bold = FontProperties(fname=FONTS.get('bold'), size=60) if 'bold' in FONTS else FontProperties(family='monospace', weight='bold', size=60)
    f_reg = FontProperties(fname=FONTS.get('regular'), size=14) if 'regular' in FONTS else FontProperties(family='monospace', size=14)
    
    spaced_city = "  ".join(list(city.upper()))
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes, color=THEME['text'], ha='center', fontproperties=f_bold, zorder=11)
    ax.text(0.5, 0.10, (country_label or country).upper(), transform=ax.transAxes, color=THEME['text'], ha='center', fontproperties=f_reg, zorder=11)
    
    plt.savefig(output_file, format=output_format.lower(), facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close()
    print(f"✓ Saved to {output_file}")

def list_themes():
    for t in get_available_themes(): print(f" - {t}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', '-c', required=True)
    parser.add_argument('--country', '-C', required=True)
    parser.add_argument('--theme', '-t', default='feature_based')
    parser.add_argument('--distance', '-d', type=int, default=10000)
    parser.add_argument('--format', '-f', default='png')
    parser.add_argument('--list-themes', action='store_true')
    args = parser.parse_args()

    if args.list_themes:
        list_themes()
        sys.exit(0)

    try:
        coords = get_coordinates(args.city, args.country)
        THEME = load_theme(args.theme)
        out = generate_output_filename(args.city, args.theme, args.format)
        create_poster(args.city, args.country, coords, args.distance, out, args.format)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)