import pygame
import sys
import math
import numpy as np

def load():
    # path of player with different states
    PLAYER_PATH = (
            'assets/sprites/redbird-upflap.png',
            'assets/sprites/redbird-midflap.png',
            'assets/sprites/redbird-downflap.png'
    )

    # path of background
    BACKGROUND_PATH = 'assets/sprites/background-black.png'

    # path of pipe
    PIPE_PATH = 'assets/sprites/pipe-green.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    #SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    #SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    #SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    #SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    #SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of pixel coordinates
    along a line from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """  
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = np.array((y, x)) if is_steep else np.array((x, y))
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return np.array(points)

def raycast_fan(img, start, num_rays=15, fan_angle=math.pi, direction=0):
    '''
    # INPUT: img - the image to perform the raycasts on
    # INPUT: start - (x, y) tuple start location for raycasts
    # INPUT: num_rays - the number of rays to fire
    # INPUT: fan_angle - the angle over which the rays are spread
    # INPUT: direction - the direction, in radians, the raycast fan will be oriented
    #
    # OUTPUT: (hit_locations, distances)
    # OUTPUT: hit_locations - the hit locations of each ray (numpy arrays)
    # OUTPUT: distances - the distance along each ray to its hit location (floats)
    #
    Performs a number of raycasts in a fan pattern from the start location
    fanning out in the direction specifed.
    '''
    # Get line length long enough to span entire image
    line_length = math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    
    # Ray cast fan
    headings = [direction - fan_angle/2 + (fan_angle / (num_rays - 1)) * i for i in range(num_rays)] # Ray fan
    hit_locations = []
    distances = []
    for ray_idx in range(num_rays):
        heading = headings[ray_idx]
        end = (int(line_length * math.cos(heading) + start[0]), int(line_length * math.sin(heading) + start[1]))
        
        # Perform line trace from start to end (should be garanteed to hit something)
        out_hit = start
        line_coords = get_line(start, end)
        for pixel_coord in line_coords:
            # Keep track of current end of line for hit results
            out_hit = pixel_coord

            # Found hit if outside of image
            x, y = pixel_coord
            if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1]:
                break
                
            # Found hit if inside solid object
            if (img[pixel_coord[0], pixel_coord[1], 1] >= 128
            and img[pixel_coord[0], pixel_coord[1], 0] <= 228
            and img[pixel_coord[0], pixel_coord[1], 2] <= 139
            and img[pixel_coord[0], pixel_coord[1], 1] > img[pixel_coord[0], pixel_coord[1], 0]
            and img[pixel_coord[0], pixel_coord[1], 1] > img[pixel_coord[0], pixel_coord[1], 2]): # Check for mostly green
                break
            
        hit_locations.append(out_hit)
        distances.append(np.linalg.norm(out_hit - start))
    
    return np.array(hit_locations), np.array(distances)
