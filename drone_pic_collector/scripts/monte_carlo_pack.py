#!/usr/bin/env python3
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class MonteCarloPack:
    def __init__(self):
        """
            creates calculator for distributing points in an ellipse
        """
        self.GRID_DENSITY = 100 # points per metre
        return
    def get_point_list(self, horiz_axis, vert_axis, num_points, iterations):
        """!
            creates list of tuples that are x,y coordinates
            relative to origin of ellipse that are as distantly
            spaced from eachother as possible using Monte Carlo
            method.

            @params vert_axis height in metres of ellipse
            @params horiz_axis width in metres of ellipse
        """
        # use grid with 10 000 points per metre each with mass = 1
        
        resolution_x = round(horiz_axis * self.GRID_DENSITY)
        resolution_y = round(vert_axis * self.GRID_DENSITY)
        a = round(horiz_axis * self.GRID_DENSITY/2)
        b = round(vert_axis * self.GRID_DENSITY/2)
        print(f"Ellipse size: a={horiz_axis}m, b={vert_axis}m")
        x = np.linspace(-a,a,resolution_x)
        y = np.linspace(-b,b, resolution_y)
        print("Creating meshgrid...")
        xv, yv = np.meshgrid(x, y)

        print("Masking points...")
        # filter points inside ellipse
        mask = (xv**2 / a**2 + yv**2 / b**2) <= 1
        background_points = np.vstack([xv[mask], yv[mask]]).T

        stops = background_points[np.random.choice(len(background_points), num_points, replace=False)]

        print("Iterating Lloyd's algo...")
        # iterate Lloyd's
        for j in range(iterations):
            print(f"Iteration number: {j}")
            tree = KDTree(stops)
            # find which stop is closest to each background point
            _, labels = tree.query(background_points)

            new_stops = []
            for i in range(num_points):
                region_points = background_points[labels == i]
                if len(region_points) > 0:
                    new_stops.append(region_points.mean(axis=0))
                else:
                    new_stops.append(background_points[np.random.randint(len(background_points))])
            
            stops = np.array(new_stops)
        scaled_stops = stops/self.GRID_DENSITY
        print("Scaled Stops:")
        print(scaled_stops)
        print("Found all stops...")
        return scaled_stops
    
    def plot_drone_stops(self, stops, a, b):
        print("Creating figure...")
        fig, ax = plt.subplots(figsize=(8,6))

        # draw ellipse boundary
        ellipse_border = Ellipse((0,0), width=a, height=b,
                                 edgecolor='blue', facecolor='none',
                                 linestyle='--', linewidth=2, label='Boundary')
        ax.add_patch(ellipse_border)

        # plot drone stops
        ax.scatter(stops[:,0], stops[:,1], c='red', marker='o', s=50, label='Stops')

        # formatting window
        ax.set_aspect('equal')
        ax.set_title(f'Drone Survey Stops: {len(stops)} points')
        ax.set_xlabel(f'Distance (m)')
        ax.set_ylabel(f'Distance (m)')
        ax.legend()
        ax.grid(True, which='both', linestyle=':', alpha=0.5)

        plt.show()
    
if __name__ == '__main__':
    monteCarloPack = MonteCarloPack()
    stops = monteCarloPack.get_point_list(2, 1, 10, 15)
    monteCarloPack.plot_drone_stops(stops, 2, 1)