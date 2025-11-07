import os
import imageio.v2 as imageio

folder = r"dump"
files = sorted(os.listdir(folder), key=lambda x: int(os.path.splitext(x)[0]))
images = [imageio.imread(os.path.join(folder, f)) for f in files]
imageio.mimsave("animation.gif", images, duration=1)
