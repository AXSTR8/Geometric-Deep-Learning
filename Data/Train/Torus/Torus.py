import bpy
import numpy as np 

# Delete default cube
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# Add Torus
bpy.ops.mesh.primitive_torus_add(
    location=(0, 0, 0),  # Location of the torus
    major_radius=1,     # Major radius (distance from the center of the torus to the center of a tube)
    minor_radius=0.3,   # Minor radius (radius of the tube)
    major_segments=48,  # Number of segments in the major circle
    minor_segments=12   # Number of segments in the minor circle
)
torus = bpy.context.object

# Scale the Torus along the y and z axes
torus.scale.y = 1.1
torus.scale.z = 1.6

# Set Torus color to White
mat = bpy.data.materials.new(name="White")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)  # Set color to white; for read  (1, 0, 0, 1)
torus.data.materials.append(mat)

# Set up camera
camera = bpy.data.objects['Camera']
camera.location = (12, 0, 0)  # Move camera along x-axis
camera.rotation_euler = (0, 1.5708,0)  # Set camera rotation to default

# Set up lighting
light = bpy.data.objects['Light']
light.location = (5.5, 0, 1.5)

# Define the number of rendered images
n_images = 40

# Create n_images randomly rotated pictures of Torus
for i in range(n_images):
    # The first image should not be rotated at all
    if i == 0:
        torus.rotation_euler = (0, 1.5708, 0)
    else:
        # Random array changing the position, with the angles restricted to [-1.1;1.1]
        position_arrow = np.random.uniform(low=-1.1, high=1.1, size=(3,))
        # Random array changing the rotation angle, with the angles restricted to [-3;3]
        rotation_arrow = np.random.uniform(low=-3, high=3, size=(3,))

        # Change the position of the Torus to a random vector
        torus.location = (0,0,0) + position_arrow

        # Rotate Torus by 90 degrees along the y-axis and add a random angle
        torus.rotation_euler = (0, 1.5708, 0) + rotation_arrow

    # Set up rendering
    bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles rendering engine
    bpy.context.scene.render.filepath = f"//torus_{i+58}.jpg"  # Output path for the rendered image
    bpy.context.scene.render.image_settings.file_format = 'JPEG'  # Output file format; For PNG it would  have 4 channels

    # Render the image
    bpy.ops.render.render(write_still=True)






