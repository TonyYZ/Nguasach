import os

os.environ["QT_API"] = "pyqt5"

import matplotlib

matplotlib.use("module://mplcairo.tk")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

fig = plt.figure(figsize=(6, 6))
fig.text(.5, .5, "वर्धति",
         fontproperties=FontProperties(fname="Nirmala.ttf"))


plt.draw() # draw the plot
plt.pause(5) # show it for 5 seconds