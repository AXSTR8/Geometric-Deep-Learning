import bpy
import numpy as np 

# Object is the default cube
#bpy.ops.object.select_all(action='DESELECT')
#cube = bpy.data.objects['Cube']
cube = bpy.context.object


# Deform the Cube along the y and z axes
cube.scale.y = 1.1
cube.scale.z = 1.6

# Set Cube color to White
mat = bpy.data.materials.new(name="White")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)  # Set color to white; for read  (1, 0, 0, 1)
cube.data.materials.append(mat)

# Set up camera
camera = bpy.data.objects['Camera']
camera.location = (12, 0, 0)  # Move camera along x-axis
camera.rotation_euler = (0, 1.5708,0)  # Set camera rotation to default

# Set up lighting
light = bpy.data.objects['Light']
light.location = (5.5, 0, 1.5)

# Define the number of rendered images
n_images = 50

# Create n_images randomly rotated pictures of Cubes
for i in range(n_images):
    # The first image should not be rotated at all
    if i == 0:
        cube.rotation_euler = (0, 0, 0)
    else:
        # Random array changing the position, with the angles restricted to [-1.1;1.1]
        position_arrow = np.random.uniform(low=-1.1, high=1.1, size=(3,))
        # Random array changing the rotation angle, with the angles restricted to [-3;3]
        rotation_arrow = np.random.uniform(low=-3, high=3, size=(3,))

        # Change the position of the Cubes to a random vector
        cube.location = (0,0,0) + position_arrow

        # Rotate Cube by a random angle
        cube.rotation_euler =  rotation_arrow

    # Set up rendering
    bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles rendering engine
    bpy.context.scene.render.filepath = f"//cube_{i}.jpg"  # Output path for the rendered image
    bpy.context.scene.render.image_settings.file_format = 'JPEG'  # Output file format; For PNG it would  have 4 channels

    # Render the image
    bpy.ops.render.render(write_still=True)






