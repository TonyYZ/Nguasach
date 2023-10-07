import matplotlib.pyplot as plt
import mplcursors
import matplotlib.patches as patches

# Load the image
image_path = 'path/to/your/image.jpg'
image = plt.imread(image_path)

# Create a figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(image)

# Define the text labels and their positions
labels = {
    'Label 1': (100, 100),
    'Label 2': (200, 200),
    'Label 3': (300, 300),
}

# Create patches for the rectangle lines
patches_dict = {}
for label, position in labels.items():
    rect = patches.Rectangle(position, 50, 50, linewidth=2, edgecolor='r', facecolor='none')
    patches_dict[label] = rect
    ax.add_patch(rect)
    rect.set_visible(False)

# Create the cursor hover event handler
def hover(event):
    for label, rect in patches_dict.items():
        contains, _ = rect.contains(event)
        rect.set_visible(contains)
    plt.draw()

# Register the cursor hover event
mplcursors.cursor(hover=True)

# Set up the text labels
for label, position in labels.items():
    ax.text(*position, label, fontsize=12, color='white', ha='center', va='center', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round'))

# Hide the default axis ticks and labels
ax.set_xticks([])
ax.set_yticks([])

# Show the plot
plt.show()
