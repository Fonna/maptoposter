from matplotlib.figure import Figure
from networkx import MultiDiGraph
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from opencage.geocoder import OpenCageGeocode
from opencage.geocoder import InvalidInputError, RateLimitExceededError, UnknownError
from tqdm import tqdm
import time
import json
import os
import sys

# ... (omitted) ...

def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country.
    
    Strategies:
    1. Cache: Check local cache first.
    2. Primary: OpenCage (Official SDK) - Commercial grade, requires API Key.
    3. Fallback: Nominatim (Geopy) - Free, rate-limited.
    """
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached:
        print(f"✓ Using cached coordinates for {city}, {country}")
        return cached

    print("Looking up coordinates...")
    
    # 1. Try OpenCage (if Key exists)
    opencage_key = os.environ.get("OPENCAGE_API_KEY")
    
    if opencage_key:
        print("Using OpenCage Geocoder (Official SDK)...")
        try:
            geocoder = OpenCageGeocode(opencage_key)
            query = f"{city}, {country}"
            
            # Using basic exception handling for network issues, 
            # specific OpenCage exceptions can be caught if needed.
            results = geocoder.geocode(query, no_annotations='1', timeout=30)
            
            if results and len(results):
                # OpenCage returns a list of dicts
                best_match = results[0]
                lat = best_match['geometry']['lat']
                lon = best_match['geometry']['lng']
                formatted_addr = best_match.get('formatted', f"{city}, {country}")
                
                print(f"✓ Found: {formatted_addr}")
                print(f"✓ Coordinates: {lat}, {lon}")
                
                try:
                    cache_set(coords, (lat, lon))
                except CacheError as e:
                    print(e)
                return (lat, lon)
            else:
                print(f"⚠ OpenCage could not find: {query}")
                # Fall through to Nominatim? Or raise error? 
                # Usually if commercial geocoder fails, free one likely will too, 
                # but we can let it fall through just in case.
        
        except RateLimitExceededError:
            print("✗ OpenCage Error: Rate limit exceeded (Payment required?)")
        except InvalidInputError:
            print("✗ OpenCage Error: Invalid input/Key")
        except Exception as e:
            print(f"⚠ OpenCage Error: {e}")
            print("Falling back to Nominatim...")

    # 2. Fallback to Nominatim (Geopy)
    print("Using Nominatim (Free/Rate-limited)...")
    if not opencage_key:
         print("  (Tip: Set OPENCAGE_API_KEY in .env for better stability)")
         
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)
    
    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}")

    # If geocode returned a coroutine in some environments, run it to get the result.
    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError:
             # If an event loop is already running...
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Async loop conflict.")
            location = loop.run_until_complete(location)
    
    if location:
        addr = getattr(location, "address", None)
        if addr:
            print(f"✓ Found: {addr}")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not find coordinates for {city}, {country}")
def get_crop_limits(G: MultiDiGraph, fig: Figure) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Determine cropping limits to maintain aspect ratio of the figure.

    This function calculates the extents of the graph's nodes and adjusts
    the x and y limits to match the aspect ratio of the provided figure.
    
    :param G: The graph to be plotted
    :type G: MultiDiGraph
    :param fig: The matplotlib figure object
    :type fig: Figure
    :return: Tuple of x and y limits for cropping
    :rtype: tuple[tuple[float, float], tuple[float, float]]
    """
    # Compute node extents in projected coordinates
    xs = [data['x'] for _, data in G.nodes(data=True)]
    ys = [data['y'] for _, data in G.nodes(data=True)]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    x_range = maxx - minx
    y_range = maxy - miny

    fig_width, fig_height = fig.get_size_inches()
    desired_aspect = fig_width / fig_height
    current_aspect = x_range / y_range

    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    if current_aspect > desired_aspect:
        # Too wide, need to crop horizontally
        desired_x_range = y_range * desired_aspect
        new_minx = center_x - desired_x_range / 2
        new_maxx = center_x + desired_x_range / 2
        new_miny, new_maxy = miny, maxy
        crop_xlim = (new_minx, new_maxx)
        crop_ylim = (new_miny, new_maxy)
    elif current_aspect < desired_aspect:
        # Too tall, need to crop vertically
        desired_y_range = x_range / desired_aspect
        new_miny = center_y - desired_y_range / 2
        new_maxy = center_y + desired_y_range / 2
        new_minx, new_maxx = minx, maxx
        crop_xlim = (new_minx, new_maxx)
        crop_ylim = (new_miny, new_maxy)
    else:
        # Otherwise, keep original extents (no horizontal crop)
        crop_xlim = (minx, maxx)
        crop_ylim = (miny, maxy)
    
    return crop_xlim, crop_ylim

def fetch_graph(point, dist) -> MultiDiGraph | None:
    lat, lon = point
    graph = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph)
    if cached is not None:
        print("✓ Using cached street network")
        return cast(MultiDiGraph, cached)

    try:
        G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all')
        # Rate limit between requests
        time.sleep(0.5)
        try:
            cache_set(graph, G)
        except CacheError as e:
            print(e)
        return G
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None

def fetch_features(point, dist, tags, name) -> GeoDataFrame | None:
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        # Rate limit between requests
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def fetch_graph(point, dist):
    lat, lon = point
    graph_key = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph_key)
    if cached is not None:
        print("✓ Using cached street network")
        return cached

    try:
        G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all')
        time.sleep(0.5)
        try:
            cache_set(graph_key, G)
        except CacheError as e:
            print(e)
        return G
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None


def fetch_features(point, dist, tags, name):
    lat, lon = point
    tag_str = "_".join(sorted(tags.keys()))
    features_key = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features_key)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cached

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        time.sleep(0.3)
        try:
            cache_set(features_key, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def create_poster(city, country, point, dist, output_file, output_format, country_label=None, name_label=None):
    print(f"\nGenerating map for {city}, {country}...")
    
    # Progress bar for data fetching
    with tqdm(total=3, desc="Fetching map data", unit="step", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        G = fetch_graph(point, dist)
        if G is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)
        
        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        water = fetch_features(point, dist, tags={'natural': 'water', 'waterway': 'riverbank'}, name='water')
        pbar.update(1)
        
        # 3. Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        parks = fetch_features(point, dist, tags={'leisure': 'park', 'landuse': 'grass'}, name='parks')
        pbar.update(1)
    
    print("✓ All data retrieved successfully!")
    
    # 2. Setup Plot
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=(12, 16), facecolor=THEME['bg'])
    ax.set_facecolor(THEME['bg'])
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    # Project graph to a metric CRS so distances and aspect are linear (meters)
    G_proj = ox.project_graph(G)
    
    # 3. Plot Layers
    # Layer 1: Polygons (filter to only plot polygon/multipolygon geometries, not points)
    if water is not None and not water.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        water_polys = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not water_polys.empty:
            # Project water features in the same CRS as the graph
            try:
                water_polys = ox.projection.project_gdf(water_polys)
            except Exception:
                water_polys = water_polys.to_crs(G_proj.graph['crs'])
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)
    
    if parks is not None and not parks.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        parks_polys = parks[parks.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not parks_polys.empty:
            # Project park features in the same CRS as the graph
            try:
                parks_polys = ox.projection.project_gdf(parks_polys)
            except Exception:
                parks_polys = parks_polys.to_crs(G_proj.graph['crs'])
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=2)
    
    # Layer 2: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(G_proj)
    edge_widths = get_edge_widths_by_type(G_proj)

    # Determine cropping limits to maintain the poster aspect ratio
    crop_xlim, crop_ylim = get_crop_limits(G_proj, fig)

    # Plot the projected graph and then apply the cropped limits
    ox.plot_graph(
        G_proj, ax=ax, bgcolor=THEME['bg'],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False, close=False
    )
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)
    
    # Layer 3: Gradients (Top and Bottom)
    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)
    
    # 4. Typography using Roboto font
    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=60)
        font_top = FontProperties(fname=FONTS['bold'], size=40)
        font_sub = FontProperties(fname=FONTS['light'], size=22)
        font_coords = FontProperties(fname=FONTS['regular'], size=14)
    else:
        # Fallback to system fonts
        font_main = FontProperties(family='monospace', weight='bold', size=60)
        font_top = FontProperties(family='monospace', weight='bold', size=40)
        font_sub = FontProperties(family='monospace', weight='normal', size=22)
        font_coords = FontProperties(family='monospace', size=14)
    
    spaced_city = "  ".join(list(city.upper()))
    
    # Dynamically adjust font size based on city name length to prevent truncation
    base_font_size = 60
    city_char_count = len(city)
    if city_char_count > 10:
        # Scale down font size for longer names
        scale_factor = 10 / city_char_count
        adjusted_font_size = max(base_font_size * scale_factor, 24)  # Minimum size of 24
    else:
        adjusted_font_size = base_font_size
    
    if FONTS:
        font_main_adjusted = FontProperties(fname=FONTS['bold'], size=adjusted_font_size)
    else:
        font_main_adjusted = FontProperties(family='monospace', weight='bold', size=adjusted_font_size)

    # --- BOTTOM TEXT ---
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_main_adjusted, zorder=11)
    
    country_text = country_label if country_label is not None else country
    ax.text(0.5, 0.10, country_text.upper(), transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)
    
    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0:
        coords = coords.replace("E", "W")
    
    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)
    
    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes, 
            color=THEME['text'], linewidth=1, zorder=11)

    # --- ATTRIBUTION (bottom right) ---
    if FONTS:
        font_attr = FontProperties(fname=FONTS['light'], size=8)
    else:
        font_attr = FontProperties(family='monospace', size=8)
    
    ax.text(0.98, 0.02, "© OpenStreetMap contributors", transform=ax.transAxes,
            color=THEME['text'], alpha=0.5, ha='right', va='bottom', 
            fontproperties=font_attr, zorder=11)

    # 5. Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    save_kwargs = dict(facecolor=THEME["bg"], bbox_inches="tight", pad_inches=0.05,)

    # DPI matters mainly for raster formats
    if fmt == "png":
        save_kwargs["dpi"] = 300

    plt.savefig(output_file, format=fmt, **save_kwargs)

    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid
  
  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline
  
  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads
  
  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout
  
  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula
  
  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split
  
  # List themes
  python create_map_poster.py --list-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --country-label   Override country text displayed on poster
  --theme, -t       Theme name (default: feature_based)
  --all-themes      Generate posters for all themes
  --distance, -d    Map radius in meters (default: 29000)
  --list-themes     List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")

def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return
    
    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, 'r') as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except:
            display_name = theme_name
            description = ''
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --list-themes
        """
    )
    
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--country-label', dest='country_label', type=str, help='Override country text displayed on poster')
    parser.add_argument('--theme', '-t', type=str, default='feature_based', help='Theme name (default: feature_based)')
    parser.add_argument('--all-themes', '--All-themes', dest='all_themes', action='store_true', help='Generate posters for all themes')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'svg', 'pdf'],help='Output format for the poster (default: png)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)
    
    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)
    
    # Validate required arguments
    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        sys.exit(1)
    
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        os.sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    else:
        if args.theme not in available_themes:
            print(f"Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            os.sys.exit(1)
        themes_to_generate = [args.theme]
    
    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)
    
    # Get coordinates and generate poster
    try:
        coords = get_coordinates(args.city, args.country)
        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(args.city, theme_name, args.format)
            create_poster(args.city, args.country, coords, args.distance, output_file, args.format, country_label=args.country_label)
        
        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
