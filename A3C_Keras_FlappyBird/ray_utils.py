import numpy as np
import math

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
    x1, y1 = start.astype(int)
    x2, y2 = end.astype(int)
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

def raycast(img, starts, angles):
    # Get line length long enough to span entire image
    line_length = math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    
    # assert starts.shape[0] == angles.shape[0], "Error, number of start locations and number of angles must be the same"

    if starts.size < angles.size:
        starts = np.tile(starts.reshape(1, -1), [angles.size, 1])

    hit_locations = []
    distances = []
    for start, angle in zip(starts, angles):
        # Ray casts
        end = start + np.array([line_length * math.cos(angle), line_length * math.sin(angle)])
        
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
