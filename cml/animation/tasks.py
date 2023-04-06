from manim import *

import numpy as np


class PolyRegression(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=7,
            y_length=6,
            axis_config={
                "color": WHITE,
                "stroke_width": 2,
                "include_tip": False,
            },
            y_axis_config={
                "label_direction": LEFT,
            },
            x_axis_config={
                "label_direction": DOWN,
            },
            tips=False,
        )
        axes_labels = axes.get_axis_labels(
            x_label="x", y_label="y"
        )

        # Create points
        x_coords = [-1, -0.5, 0.5, 1]
        y_coords = [1, -1, 0.5, -0.5]
        points = VGroup()
        for x, y in zip(x_coords, y_coords):
            point = Dot(axes.coords_to_point(x, y))
            points.add(point)

        # Create polynomial function curves
        curves = VGroup()
        for order in range(1, 5):
            coef = np.polyfit(x_coords, y_coords, order)
            curve = axes.plot(
                lambda x: np.polyval(coef, x),
                x_range=[-1.5, 1.5],
                color=BLUE,
                stroke_width=2,
            )
            curves.add(curve)

        # Animate points and curves
        self.play(
            Write(axes),
            Write(axes_labels),
            FadeIn(points),
        )
        for curve in curves:
            self.play(
                Create(curve),
                points.animate.set_color(YELLOW),
                run_time=2,
            )
            self.wait(0.5)

        self.wait()