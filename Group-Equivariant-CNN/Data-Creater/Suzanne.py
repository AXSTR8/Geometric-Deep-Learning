import bpy
import numpy as np 

# Delete default cube
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select_set(True)
bpy.ops.object.delete()

# Add Suzanne
bpy.ops.mesh.primitive_monkey_add(location=(0, 0, 0))
suzanne = bpy.context.object

# Set monkey color to white
mat = bpy.data.materials.new(name="white")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)  # Set color to white; for read  (1, 0, 0, 1)
suzanne.data.materials.append(mat)

# Set up camera
camera = bpy.data.objects['Camera']
camera.location = (8, 0, 0)  # Move camera along x-axis
camera.rotation_euler = (0, 0, 0)  # Set camera rotation to default

# Point camera to Suzanne
camera_constraint = camera.constraints.new(type='TRACK_TO')
camera_constraint.target = suzanne
camera_constraint.track_axis = 'TRACK_NEGATIVE_Z'
camera_constraint.up_axis = 'UP_Y'

# Set up lighting
light = bpy.data.objects['Light']
light.location = (5.5, 0, 1.5)

# Create 50 randomly rotated pictures of "Suzanne"
for i in range(50):
    # The first image should not be rotated at all
    if i == 0:
        suzanne.rotation_euler = (0, 0, 1.5708)
    else:
        # Random array changing the rotation angle, with the angles restricted to [-1;1]
        rotation_arrow = np.random.uniform(low=-1, high=1, size=(3,))

        # Rotate Suzanne by 90 degrees along the z-axis
        suzanne.rotation_euler = (0, 0, 1.5708) + rotation_arrow

    # Set up rendering
    bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles rendering engine
    bpy.context.scene.render.filepath = f"//suzanne_{i}.jpg"  # Output path for the rendered image
    bpy.context.scene.render.image_settings.file_format = 'JPEG'  # Output file format; For PNG it would  have 4 channels

    # Render the image
    bpy.ops.render.render(write_still=True)