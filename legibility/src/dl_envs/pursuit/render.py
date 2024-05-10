"""
2D rendering of the pursuit domain
"""

import math
import os
import sys
import numpy as np
import math
import six

from gym import error
from pathlib import Path

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size, grid_size=50, icon_size=50, visible=True, highlight=''):
        display = get_display(None)
        self.rows, self.cols = world_size
        self.highlight_prey = highlight
        self.grid_size = grid_size
        self.icon_size = icon_size

        self.width = self.cols * self.grid_size + 1
        self.height = self.rows * self.grid_size + 1
        self.window = pyglet.window.Window(width=self.width, height=self.height, display=display, visible=visible)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = Path(__file__).parent.absolute()

        pyglet.resource.path = [str(script_dir / 'data' / 'icons')]
        pyglet.resource.reindex()

        self.img_prey = pyglet.resource.image("prey.png")
        self.img_prey_target = pyglet.resource.image("prey_target.png")
        self.img_hunter = pyglet.resource.image("hunter.png")
    
    @property
    def highlight(self) -> str:
        return self.highlight_prey
    
    @highlight.setter
    def highlight(self, highlight: str) -> None:
        self.highlight_prey = highlight
    
    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley))

    def render(self, env, return_rgb_array=False):
        glClearColor(0, 0, 0, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_preys(env)
        self._draw_hunters(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        w_grid_size = self.window.width / self.cols
        h_grid_size = self.window.height / self.rows
        for r in range(self.rows + 1):
            batch.add(2, gl.GL_LINES, None, ("v2f", (0, h_grid_size * r, w_grid_size * self.cols, h_grid_size * r)), ("c3B", (*_WHITE, *_WHITE)))
        for c in range(self.cols + 1):
            batch.add(2, gl.GL_LINES, None, ("v2f", (w_grid_size * c, 0, w_grid_size * c, h_grid_size * self.rows)), ("c3B", (*_WHITE, *_WHITE)))
        batch.draw()

    def _draw_preys(self, env):
        preys = []
        batch = pyglet.graphics.Batch()
        w_grid_size = self.window.width / self.cols
        h_grid_size = self.window.height / self.rows

        for prey_id in env.prey_ids:
            prey = env.agents[prey_id]
            if prey.alive:
                row, col = prey.pos
                if prey_id == self.highlight_prey:
                    preys.append(pyglet.sprite.Sprite(self.img_prey_target, w_grid_size * col, self.window.height - h_grid_size * (row + 1), batch=batch))
                else:
                    preys.append(pyglet.sprite.Sprite(self.img_prey, w_grid_size * col, self.window.height - h_grid_size * (row + 1), batch=batch))
        for a in preys:
            a.update(scale_y=h_grid_size / a.height, scale_x=w_grid_size / a.width)
        batch.draw()
        for prey_id in env.prey_ids:
            prey = env.agents[prey_id]
            if prey.alive:
                self._draw_badge(*prey.pos, env.prey_ids.index(prey_id) + 1)

    def _draw_hunters(self, env):
        hunters = []
        batch = pyglet.graphics.Batch()
        w_grid_size = self.window.width / self.cols
        h_grid_size = self.window.height / self.rows

        for h_id in env.hunter_ids:
            hunter = env.agents[h_id]
            row, col = hunter.pos
            sprite = pyglet.sprite.Sprite(self.img_hunter, w_grid_size * col, self.window.height - h_grid_size * (row + 1), batch=batch)
            sprite.update(scale_y=h_grid_size / sprite.height, scale_x=w_grid_size / sprite.width)
            hunters.append(sprite)
        batch.draw()
        for h_id in env.hunter_ids:
            hunter = env.agents[h_id]
            self._draw_badge(*hunter.pos, env.hunter_ids.index(h_id) + 1)

    def _draw_badge(self, row, col, badge_id):
        resolution = 10
        w_grid_size = self.window.width / self.cols
        h_grid_size = self.window.height / self.rows
        radius_x = w_grid_size / 6
        radius_y = h_grid_size / 6

        lvl_badge_x = col * w_grid_size + (3 / 4) * w_grid_size
        lvl_badge_y = self.window.height - h_grid_size * (row + 1) + (1 / 4) * h_grid_size
        id_badge_x = col * w_grid_size + (1 / 4) * w_grid_size
        id_badge_y = self.window.height - h_grid_size * (row + 1) + (3 / 4) * h_grid_size
        

        # make a circle for each badge
        # verts = []
        # for i in range(resolution):
        #     angle = 2 * math.pi * i / resolution
        #     x = radius_x * math.cos(angle) + lvl_badge_x
        #     y = radius_y * math.sin(angle) + lvl_badge_y
        #     verts += [x, y]
        # circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        # glColor3ub(*_BLACK)
        # circle.draw(GL_POLYGON)
        # glColor3ub(*_WHITE)
        # circle.draw(GL_LINE_LOOP)
        # lvl_label = pyglet.text.Label(str(level), font_name="Times New Roman", font_size=12, x=lvl_badge_x, y=lvl_badge_y + 2,
        #                               anchor_x="center", anchor_y="center")
        # lvl_label.draw()
        
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius_x * math.cos(angle) + id_badge_x
            y = radius_y * math.sin(angle) + id_badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_BLACK)
        circle.draw(GL_POLYGON)
        glColor3ub(*_WHITE)
        circle.draw(GL_LINE_LOOP)
        id_label = pyglet.text.Label(str(badge_id), font_name="Times New Roman", font_size=11, x=id_badge_x, y=id_badge_y + 2, anchor_x="center",
                                     anchor_y="center")
        id_label.draw()
