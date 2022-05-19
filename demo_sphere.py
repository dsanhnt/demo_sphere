from xmlrpc.client import Marshaller
import numpy as np
import argparse
from vispy                 import app, scene
from vispy.io              import imread, load_data_file, read_mesh, write_mesh
from vispy.scene.visuals   import Mesh, Sphere
from vispy.scene           import transforms
from vispy.geometry        import create_sphere
from vispy.visuals.filters import TextureFilter
from vispy.visuals.mesh    import  MeshVisual
from vispy.visuals.transforms   import STTransform
from vispy.visuals.filters import ShadingFilter, WireframeFilter
import pandas                   as pd
from math                       import sin, cos, pi, sqrt
import sys

data = []
scalefactor = 7.0
# data processing
def spherical_to_cartesian(object_positions, index):
    storage = []
    JulianDate, phi, theta, r = object_positions.to_numpy()[index]
    theta   *= pi/180
    phi     *= pi/180
    r       *= scalefactor
    x = r * cos(theta) * cos(phi)
    y = r * cos(theta) * sin(phi) 
    z = r * sin(theta)            
    storage.append((x,y,z))
    
    return JulianDate, storage


def get_sphere(texture_path, radius, translate,axial_tilt=0):
    # Generate sphere
    meshdata = create_sphere(rows=20,cols=40,radius=radius,method='latitude',offset=False)
    vertices = meshdata.get_vertices(None)
    faces = meshdata.get_faces()
    mesh = Mesh(vertices,faces, shading=None,color='white')
    mesh.transform = transforms.MatrixTransform()
    mesh.transform.translate(translate)
    view.add(mesh)
    
    if texture_path != None:    
        # Load texture file
        texture = np.flipud(imread(texture_path))
            
        # Applying Texture
        texcoords = get_texcoords(vertices,axial_tilt)
        texture_filter = TextureFilter(texture, texcoords)
        mesh.attach(texture_filter)
    else :
        wireframe_filter = WireframeFilter()
        mesh.attach(wireframe_filter)
    
    return mesh

def RotationMatrix(axis,theta):
    if   axis == 1:
        return [[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]]
    elif axis == 2:
        return [[cos(theta),0, sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]] 
    elif axis == 3:
        return [[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]]
    else:
        raise("Error")
#--------------------------------------------
# Compute Texture Coordinates  
def get_texcoords(vertices,axial_tilt=0):
    texcoords = []
    for vert in vertices:

        vert = RotationMatrix(1,axial_tilt/180*pi) @ vert
        # Compute position in uv-space
        radius = np.sqrt(vert[0]**2 + vert[1]**2 + vert[2]**2)
        # Convert to texture coordinates

        u = 0.5 + np.arctan2(-vert[0], vert[1]) / (2 * np.pi)
        v = 0.5 + np.arcsin(vert[2] / radius) / np.pi
        texcoords.append([u,v])
    return np.array(texcoords)
#------------------------------------------------

# Visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--shading', default='smooth', 
                            choices=['none','flat','smooth'],
                            help='shading mode')
args, _ = parser.parse_known_args()
        
# Create a canvas with a 3D viewport
canvas = scene.SceneCanvas(keys='interactive',
                                title='Applying Texture to geometry')

view = canvas.central_widget.add_view()

view.camera = scene.TurntableCamera()
view.camera.set_range((-1, 1), (-1, 1), (-1, 1))

Sun_texture_path    = load_data_file('2k_Earth.jpg',directory = './texture/')
Sphere_Sun          = get_sphere(Sun_texture_path, 1.0, translate = (0,0,0))


canvas.show()

if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()
