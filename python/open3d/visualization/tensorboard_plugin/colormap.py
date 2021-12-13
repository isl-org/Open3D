class Colormap:
    """This class is used to create a color map for visualization of points."""

    class Point:
        """Initialize the class.

        Args:
            value: The scalar value index of the point.
            color: The color associated with the value.
        """

        def __init__(self, value, color):
            assert (value >= 0.0)
            assert (value <= 1.0)

            self.value = value
            self.color = color

        def __repr__(self):
            """Represent the color and value in the colormap."""
            return "Colormap.Point(" + str(self.value) + ", " + str(
                self.color) + ")"

    # The value of each Point must be greater than the previous
    # (e.g. [0.0, 0.1, 0.4, 1.0], not [0.0, 0.4, 0.1, 1.0]
    def __init__(self, points):
        self.points = points

    def calc_u_array(self, values, range_min, range_max):
        """Generate the basic array based on the minimum and maximum range passed."""
        range_width = (range_max - range_min)
        return [
            min(1.0, max(0.0, (v - range_min) / range_width)) for v in values
        ]

    # (This is done by the shader now)
    def calc_color_array(self, values, range_min, range_max):
        """Generate the color array based on the minimum and maximum range passed.

        Args:
            values: The index of values.
            range_min: The minimum value in the range.
            range_max: The maximum value in the range.

        Returns:
            An array of color index based on the range passed.
        """
        u_array = self.calc_u_array(values, range_min, range_max)

        tex = [[1.0, 0.0, 1.0]] * 128
        n = float(len(tex) - 1)
        idx = 0
        for tex_idx in range(0, len(tex)):
            x = float(tex_idx) / n
            while idx < len(self.points) and x > self.points[idx].value:
                idx += 1

            if idx == 0:
                tex[tex_idx] = self.points[0].color
            elif idx == len(self.points):
                tex[tex_idx] = self.points[-1].color
            else:
                p0 = self.points[idx - 1]
                p1 = self.points[idx]
                dist = p1.value - p0.value
                # Calc weights between 0 and 1
                w0 = 1.0 - (x - p0.value) / dist
                w1 = (x - p0.value) / dist
                c = [
                    w0 * p0.color[0] + w1 * p1.color[0],
                    w0 * p0.color[1] + w1 * p1.color[1],
                    w0 * p0.color[2] + w1 * p1.color[2]
                ]
                tex[tex_idx] = c

        return [tex[int(u * n)] for u in u_array]

    # These are factory methods rather than class objects because
    # the user may modify the colormaps that are used.
    @staticmethod
    def make_greyscale():
        """Generate a greyscale colormap."""
        return Colormap([
            Colormap.Point(0.0, [0.0, 0.0, 0.0]),
            Colormap.Point(1.0, [1.0, 1.0, 1.0])
        ])

    @staticmethod
    def make_rainbow():
        """Generate the rainbow color array."""
        return Colormap([
            Colormap.Point(0.000, [0.0, 0.0, 1.0]),
            Colormap.Point(0.125, [0.0, 0.5, 1.0]),
            Colormap.Point(0.250, [0.0, 1.0, 1.0]),
            Colormap.Point(0.375, [0.0, 1.0, 0.5]),
            Colormap.Point(0.500, [0.0, 1.0, 0.0]),
            Colormap.Point(0.625, [0.5, 1.0, 0.0]),
            Colormap.Point(0.750, [1.0, 1.0, 0.0]),
            Colormap.Point(0.875, [1.0, 0.5, 0.0]),
            Colormap.Point(1.000, [1.0, 0.0, 0.0])
        ])
