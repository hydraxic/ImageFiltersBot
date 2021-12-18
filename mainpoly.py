from os import path, makedirs
from math import ceil
#import subprocess, sys
import sys
from PIL import Image, ImageDraw
from random import sample
from polylattice import PolyLattice
#from colors import palettes


def mainfunc(pal, intensity, def_res, ps):

    ## Configurations ##
    #palette = pal#'pastel_forest'
    mutation_intensity = intensity
    default_resolution = def_res#[1920, 1080]

    # Polygons have a fixed size in px. Higher resolution = more polygons
    poly_sizes = ps#(120, 100)

    ## Paths ##
    file_path = path.realpath(__file__)
    file_dir = file_path.rstrip("/mainpoly.py")

    # Create renders/ folder if necessary
    render_folder = file_dir + "/renders"
    makedirs(render_folder, exist_ok=True)

    render_file = render_folder + "/wallpaper.jpg"

    # Get resolution from program arguments, or use default resolution
    resolution = default_resolution

    if len(sys.argv) >= 2:
        try:
            # Try parsing resolution from argv[1]
            res_parse = sys.argv[1].split("x")

            if len(res_parse) != 2:
                raise ValueError()

            res_parse = [int(x) for x in res_parse]

            if any(x < 0 for x in res_parse):
                raise ValueError()

            resolution = res_parse

        except ValueError:
            sys.stderr.write('Resolution given in arguments must be written like "1920x1080". Using default resolution...')

    # Create an image of the size of the screen
    im = Image.new("RGB", resolution, 0)
    image_draw = ImageDraw.Draw(im)

    # Initialise a PolyLattice
    poly_count_x = (resolution[0] / poly_sizes[0])
    poly_count_y = (resolution[1] / poly_sizes[1])

    # Last polygons might be partly overflowing the image
    polylattice = PolyLattice(
        im.size,
        (ceil(poly_count_x), ceil(poly_count_y)),
        poly_sizes)
    polylattice.initialise(separate_in_triangles=True)

    # take two colours
    colors = sample(pal, 2)
    # Mutate PolyLattice and apply random gradient of colours
    polylattice.mutate(mutation_intensity)
    polylattice.gradient_colors_random_direction(colors[0], colors[1])

    # Draw the polylattice on the image
    polylattice.draw(image_draw)

    # Save image in renders
    im.save(render_file)