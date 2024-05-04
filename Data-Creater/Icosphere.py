import bpy
import numpy as np 

# Delete default cube
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# Add Icosphere
bpy.ops.mesh.primitive_ico_sphere_add(radius=1, subdivisions=3, location=(0, 0, 0))
icosphere = bpy.context.object

# Scale the icosphere along the y and z axes
icosphere.scale.y = 1.1
icosphere.scale.z = 1.6
#icosphere.scale.x = 2
#icosphere.location = (0.3,0.7,0.5)

# Set Icosphere color to White
mat = bpy.data.materials.new(name="White")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)  # Set color to white; for read  (1, 0, 0, 1)
icosphere.data.materials.append(mat)

# Set up camera
camera = bpy.data.objects['Camera']
camera.location = (12, 0, 0)  # Move camera along x-axis
camera.rotation_euler = (0, 1.5708,0)  # Set camera rotation to default

# Set up lighting
light = bpy.data.objects['Light']
light.location = (5.5, 0, 1.5)

# Define the number of rendered images
n_images = 50

# Create n_images randomly rotated pictures of Icosphere
for i in range(n_images):
    # The first image should not be rotated at all
    if i == 0:
        icosphere.rotation_euler = (0, 0, 0)
    else:
        # Random array changing the position, with the angles restricted to [-1.1;1.1]
        position_arrow = np.random.uniform(low=-1.1, high=1.1, size=(3,))
        # Random array changing the rotation angle, with the angles restricted to [-3;3]
        rotation_arrow = np.random.uniform(low=-3, high=3, size=(3,))

        # Change the position of the Icosphere to a random vector
        icosphere.location = (0,0,0) + position_arrow

        # Rotate Suzanne by 90 degrees along the z-axis and add a random angle
        icosphere.rotation_euler = (0, 0, 1.5708) + rotation_arrow

    # Set up rendering
    bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles rendering engine
    bpy.context.scene.render.filepath = f"//icosphere_{i}.jpg"  # Output path for the rendered image
    bpy.context.scene.render.image_settings.file_format = 'JPEG'  # Output file format; For PNG it would  have 4 channels

    # Render the image
    bpy.ops.render.render(write_still=True)






