import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Wedge, Polygon, Ellipse 
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import AnchoredText    
import math


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon as Polyshape

import copy

class PolygonGenerator:
    """
    This class is used to generate a set generic 2D polygonal patch.

    """ 
    
    def __init__(self):
        self.__patches = []
    
    def add_polygon(self,_vert_coords):
        """
        This function create a polygonal patch from the vertices of a polygonal element.
        
        Args:
        - _vertices: np.array((n_vertices,2))
            Coordinates x and y of the vertices of the polygon. In order to avoid the problems in the generation, vertices must
            be provided ordinately so that sides are constructed with successive pair of points.
        """
        _polygon = Polygon(_vert_coords, closed = True)
        self.__patches.append(_polygon)
    
    def return_patches(self):
        """
        This function return the patches created previously.
        
        """
        results = self.__patches
        return results

class CircleGenerator:
    """
    This class is used to generate a set generic circular patches.
    
    """ 
    
    def __init__(self):
        self.__patches = []
    
    def add_circle(self,_center_coords, radius):
        """
        This function create a polygonal patch from the vertices of a polygonal element.
        
        Args:
        - _center_coords: tuple
            X - _center_coords[0] - and Y - _center_coords[1] - coordinates of the center of the circle.
        - radius: float
            Radius of the circle.
        """
        _circle = Circle(_center_coords, radius)
        self.__patches.append(_circle)
    
    def return_patches(self):
        """
        This function return the patches created previously.
        
        """
        results = self.__patches
        return results

class LatticeEnumeration:
    """
    
    This class will construct a np.array for numbering of the lattice elements.
    
    """
    def __init__(self):
        self.__lattice = {'lat_type':'square'}
                
    def add_dimension(self, x1_dim, x2_dim):
        """
        This class will create a numpy array of given lattice dimension.
        
        Args:
        
        x1_dim: int
            Lattice dimension across the first dimension
        x2_dim: int
            Lattice dimension across the first dimension
        """
        self.__lattice['lattice']=np.zeros((x1_dim,x2_dim),dtype=object)
        
    def set_type(self, lat_type='square'):
        """
        This function will set the lattice properties - i.e if it's a square or hexagonal lattice.
        'hexx' identifies hexagonal lattices with hexagons having 2 sides parallel to the y-axis.
        'hexy' identifies hexagonal lattices with hexagons having 2 sides parallel to the x-axis.
        
        Args:
        lat_type: str
            Lattice type identifier. Accepted value are 'square','hexx' or 'hexy'. Default value is 'square'.
        
        
        """
        if lat_type not in ['square','hexxc','hexyc']:
            print('Attention! Lattice type not recognized. Default value will be used.')
            lat_type = 'square'
        self.__lattice['lat_type'] = lat_type
        
    def add_enumeration(self,enum_type='num'):
        """
        This function will add an enumerating scheme to the lattice. Depending on the
        type of lattice different numbering scheme are constructed. Examples:
        Square lattices are numbered on the grid like schemes.
        self.__lattice['type']: 'square'
        self.__lattice['lattice'] = 
            [[1, 2, 3],                 [['A001', 'A002', 'A003'],
             [4, 5, 6],                  ['B001', 'B002', 'B003'],
             [7, 8, 9]]                  ['C001', 'C002', 'C003'],]
        
        Hexagonal lattices are numbered on ring like schemes
        self.__lattice['type']: 'hexxc' or 'hexyc'
        self.__lattice['lattice'] = 
            [                     [
            [0, 0, 0, 0, 0],      [0,      0,      0,      0, 0],
            [0, 0, 4, 3, 0],      [0,      0, 'B003', 'B002', 0],
            [0, 5, 1, 2, 0],      [0, 'B004', 'A001', 'B001', 0],
            [0, 6, 7, 0, 0],      [0, 'B005', 'B006',      0, 0],
            [0, 0, 0, 0, 0],      [0,      0,      0,      0, 0],
            ]                     ]
        
        Two different types of lattice enumeration are accepted. The first is the alpha-numuerical scheme, identified by
        'alphanum' which set a letter for a row-ring and a number for the position. The second numerical scheme, the default one,
        is the 'num' and uses the same schematic as before but the D is identified my a unique number.
        
        Args:
        enum_type: str
            Identifies the type of enumeration to be used. Accepts only 'alphanum' or 'enum'.
        """
        # --- # CHECKING THAT THE INPUT DATA ARE CONSISTENT # --- #
        if enum_type not in ['alphanum', 'num']:
            print('Type of enumaration not found in the database. Setting enumeration type to the default value.')
            enum_type = 'num'
        if 'lattice' not in self.__lattice.keys():
            raise ValueError('Error! Lattice dimension have not been submitted.')
        lat_dim = np.shape(self.__lattice['lattice'])
        # --- # ASSIGNING THE NUMERATION # --- #
        if self.__lattice['lat_type'] in ['hexxc','hexyc']:
            if lat_dim[0] % 2 == 0 or lat_dim[0] != lat_dim[1]:
                raise ValueError('Error! Unable to construct construct a numbering scheme when the lattice dimension provided.\n'+
                                 ' The function only accepts square arrays with odd number of elements')
            self.hexagonal_enumeration(enum_type)
        elif self.__lattice['lat_type'] in ['square']:
            self.cartesian_enumeration(enum_type)
        

    def hexagonal_enumeration(self,enum_type):
        """
        This function performs the enumeration on hexagonal lattices.
        
        Args:
        enum_type: str
            Type of enumeration to be performed. Must be either 'num' or 'alphanum'.
        
        """      
        if enum_type not in ['alphanum', 'num']:
            raise ValueError('Error! Enumeration technique not present in the database.')
        
        lat_dim = np.shape(self.__lattice['lattice'])
        center_idx = int((lat_dim[0]-1)/2)
        
        if enum_type == 'num':
            self.__lattice['lattice'][center_idx,center_idx] = 1
            counter = 2
            for i1 in range(1,center_idx):
                lside=i1
                
                for j1 in range(i1):
                    self.__lattice['lattice'][center_idx+j1,center_idx+i1-j1] = counter
                    counter = counter+1

                for j2 in range(i1):
                    self.__lattice['lattice'][center_idx+lside,center_idx+i1-lside-j2] = counter
                    counter = counter+1

                for j3 in range(i1):
                    self.__lattice['lattice'][center_idx+lside-j3,center_idx+i1-2*lside] = counter
                    counter = counter+1

                for j4 in range(i1):
                    self.__lattice['lattice'][center_idx-j4,center_idx+i1-2*lside+j4] = counter
                    counter = counter+1

                for j5 in range(i1):
                    self.__lattice['lattice'][center_idx-lside,center_idx+i1-lside+j5] = counter
                    counter = counter+1

                for j6 in range(i1):
                    self.__lattice['lattice'][center_idx-lside+j6,center_idx+i1] = counter
                    counter = counter+1
        elif enum_type == 'alphanum':
            self.__lattice['lattice'][center_idx,center_idx] = 'A001'
        
            for i1 in range(1,center_idx,1):
                lside=i1
                count_alpha = chr(ord('@')+i1+1)
                count_num = 1
                for j1 in range(i1):
                    num_id = '{0:0>3}'.format(count_num)
                    self.__lattice['lattice'][center_idx+j1,center_idx+i1-j1] = count_alpha+num_id
                    count_num = count_num + 1

                for j2 in range(i1):
                    num_id = '{0:0>3}'.format(count_num)
                    self.__lattice['lattice'][center_idx+lside,center_idx+i1-lside-j2] = count_alpha+num_id
                    count_num = count_num + 1

                for j3 in range(i1):
                    num_id = '{0:0>3}'.format(count_num)
                    self.__lattice['lattice'][center_idx+lside-j3,center_idx+i1-2*lside] = count_alpha+num_id
                    count_num = count_num + 1

                for j4 in range(i1):
                    num_id = '{0:0>3}'.format(count_num)
                    self.__lattice['lattice'][center_idx-j4,center_idx+i1-2*lside+j4] = count_alpha+num_id
                    count_num = count_num + 1
                for j5 in range(i1):
                    num_id = '{0:0>3}'.format(count_num)
                    self.__lattice['lattice'][center_idx-lside,center_idx+i1-lside+j5] = count_alpha+num_id
                    count_num = count_num + 1

                for j6 in range(i1):
                    num_id = '{0:0>3}'.format(count_num)
                    self.__lattice['lattice'][center_idx-lside+j6,center_idx+i1] = count_alpha+num_id
                    count_num = count_num + 1
                    
        if self.__lattice['lat_type'] == 'hexyc':
            self.__lattice['lattice'] = np.transpose(self.__lattice['lattice'])
         
    def cartesian_enumeration(self, enum_type):
        """
        This function performs the enumeration on square lattices.
        
        Args:
        enum_type: str
            Type of enumeration to be performed. Must be either 'num' or 'alphanum'.
            
        """
        if enum_type not in ['alphanum', 'num']:
            raise ValueError('Error! Enumeration technique not present in the database.')
        
        lat_dim = np.shape(self.__lattice['lattice'])
        
        if enum_type == 'num':
            counter = 1
            for i1 in range(lat_dim[0]):
                for i2 in range(lat_dim[1]):
                    self.__lattice['lattice'][i1,i2] = counter
                    counter = counter + 1
        elif enum_type == 'alphanum':
            for i1 in range(lat_dim[0]):
                count_num = i1
                num_id = '{0:0>3}'.format(count_num)
                for i2 in range(lat_dim[1]):
                    count_alpha = chr(ord('@')+i2)
                    self.__lattice[i1,i2] = count_alpha+num_id
    
    def get_lattice(self):
        """
        This function will return the lattice.
        
        """               
        results = copy.deepcopy(self.__lattice['lattice'])
        return results

class CenterToPolygon:
    """
    
    This class helps the user to evaluate the vertices of a polygon based on the center position.
    It returns an array contaning the coordinate of the vertices. If N is the number of vertices of
    polygon, then:
    
    Results: np.array((N,2))

    
    """
    @staticmethod
    def hexa_generator(center_coords, pitch, hexa_type, type_output = 'tuple'):
        """
        This function will evaluate the hexagon's vertices coordinates based on the center coordinates,
        the double of the apothem and its orientation.

        Args:
        center_coords: tuple
            Coordinate of the center. It accepts both 2D or 3D like structures - i.e.
            center_coords = (coord_x, coord_y) or center_coords = (coord_x, coord_y, coord_z)
        pitch: float
            Double of the apothem.
        hexa_type: str
            Hexagon orientation identifier.
            'hexxc' identifies hexagonal lattices with hexagons having 2 sides parallel to the y-axis.
            'hexyc' identifies hexagonal lattices with hexagons having 2 sides parallel to the x-axis.
        type_output: str
            Type of output to be returned. Default value is 'tuple'
            If type_output == 'tuple':
                The output is returned in a list, in which each element is a tuple contaning the x and y coordinate.
            If type_output == 'array':
                The output is returned in the form of np.array((6,2)).    
        """
        if type_output not in ['tuple', 'array']:
            print('Error! Output will be in tuple format.')
            type_output = 'tuple'
            vertices = []
        elif type_output == 'tuple':
            vertices = []
        else:
            vertices = np.zeros((6,2))
        
        radius = pitch/(2*np.cos(np.pi/6))
        
        for j1 in range(6):
            if hexa_type == 'hexyc':
                x_coord = center_coords[0]+radius*np.cos(j1*np.pi/3)
                y_coord = center_coords[1]+radius*np.sin(j1*np.pi/3)
            elif hexa_type == 'hexxc':
                x_coord = center_coords[0]+radius*np.cos(j1*np.pi/3+np.pi/6)
                y_coord = center_coords[1]+radius*np.sin(j1*np.pi/3+np.pi/6)
            
            if type_output == 'tuple':
                vertices.append((x_coord,y_coord))
            else:
                vertices[j1,0] = x_coord
                vertices[j1,1] = y_coord
        
        return vertices
    
    
    @staticmethod
    def square_generator(center_coords, pitch, type_output = 'tuple'):
        """
        This function will evaluate the square's vertices coordinates based on the center coordinates
        and the side length.

        Args:
        center_coords: tuple
            Coordinate of the center. It accepts both 2D or 3D like structures - i.e.
            center_coords = (coord_x, coord_y) or center_coords = (coord_x, coord_y, coord_z)
        pitch: float
            Side length
        type_output: str
            Type of output to be returned. Default value is 'tuple'
            If type_output == 'tuple':
                The output is returned in a list, in which each element is a tuple contaning the x and y coordinate.
            If type_output == 'array':
                The output is returned in the form of np.array((4,2)).    

        """
        if type_output not in ['tuple', 'array']:
            print('Error! Output will be in tuple format.')
            type_output = 'tuple'
            vertices = []
        elif type_output == 'tuple':
            vertices = []
        else:
            vertices = np.zeros((4,2))
        radius = pitch/4*np.sqrt(2)
        for j1 in range(4):
            x_coord = center_coords[0]+radius*np.cos((j1+1/4)*np.pi)
            y_coord = center_coords[1]+radius*np.sin((j1+1/4)*np.pi)
            if type_output == 'tuple':
                vertices.append((x_coord,y_coord))
            else:
                vertices[j1,0] = x_coord
                vertices[j1,1] = y_coord
        return vertices

    @staticmethod
    def square_prism_generator(center_coords, pitch, zmin, zmax, type_output = 'tuple'):
        """
        This function will evaluate the square's vertices coordinates based on the center coordinates
        and the side length.

        Args:
        center_coords: tuple
            Coordinate of the center. It accepts both 2D or 3D like structures - i.e.
            center_coords = (coord_x, coord_y) or center_coords = (coord_x, coord_y, coord_z)
        pitch: float
            Side length
        zmin: float
            z-coordinate of the lower plane of the prism.
        zmax: float
            z-coordinate of the upper plane of the prism.
        type_output: str
            Type of output to be returned. Default value is 'tuple'
            If type_output == 'tuple':
                The output is returned in a list, in which each element is a tuple contaning the x, y and z coordinate.
            If type_output == 'array':
                The output is returned in the form of np.array((8,3)).    
        """
        if type_output not in ['tuple', 'array']:
            print('Error! Output will be in tuple format.')
            type_output = 'tuple'
            vertices = []
        elif type_output == 'tuple':
            vertices = []
        else:
            vertices = np.zeros((8,3))
        radius = pitch/4*np.sqrt(2)
        counter_array = 0
        for z in [zmin,zmax]:
            for j1 in range(4):
                x_coord = center_coords[j1,0]+radius*np.cos((j1+1/4)*np.pi)
                y_coord = center_coords[j1,1]+radius*np.sin((j1+1/4)*np.pi)

                if type_output == 'tuple':
                    vertices.append((x_coord,y_coord,z))
                else:    
                    vertices[counter_array,0] = x_coord
                    vertices[counter_array,1] = y_coord
                    vertices[counter_array,2] = z
                    counter_array= counter_array+1
                   
        return vertices
    
    @staticmethod
    def hexa_prism_generator(center_coords, pitch, zmin, zmax, hexa_type, type_output = 'tuple'):
        """
        This function will evaluate the hexagon's vertices coordinates based on the center coordinates,
        the double of the apothem and its orientation.

        Args:
        center_coords: tuple
            Coordinate of the center. It accepts both 2D or 3D like structures - i.e.
            center_coords = (coord_x, coord_y) or center_coords = (coord_x, coord_y, coord_z)
        pitch: float
            Double of the apothem.
        zmin: float
            z-coordinate of the lower plane of the prism.
        zmax: float
            z-coordinate of the upper plane of the prism.
        hexa_type: str
            Hexagon orientation identifier.
            'hexxc' identifies hexagonal lattices with hexagons having 2 sides parallel to the y-axis.
            'hexyc' identifies hexagonal lattices with hexagons having 2 sides parallel to the x-axis.
        type_output: str
            Type of output to be returned. Default value is 'tuple'
            If type_output == 'tuple':
                The output is returned in a list, in which each element is a tuple contaning the x and y coordinate.
            If type_output == 'array':
                The output is returned in the form of np.array((12,3)).    
        """
        if type_output not in ['tuple', 'array']:
            print('Error! Output will be in tuple format.')
            type_output = 'tuple'
            vertices = []
        elif type_output == 'tuple':
            vertices = []
        else:
            vertices = np.zeros((12,3))
        
        radius = pitch/(2*np.cos(np.pi/6))
        counter_array = 0
        for z in [zmin,zmax]:
            for j1 in range(6):
                if hexa_type == 'hexyc':
                    x_coord = center_coords[0]+radius*np.cos(j1*np.pi/3)
                    y_coord = center_coords[1]+radius*np.sin(j1*np.pi/3)
                elif hexa_type == 'hexxc':
                    x_coord = center_coords[0]+radius*np.cos(j1*np.pi/3+np.pi/6)
                    y_coord = center_coords[1]+radius*np.sin(j1*np.pi/3+np.pi/6)

                if type_output == 'tuple':
                    vertices.append((x_coord,y_coord,z))
                else:    
                    vertices[counter_array,0] = x_coord
                    vertices[counter_array,1] = y_coord
                    vertices[counter_array,2] = z
                    counter_array= counter_array+1
        
        return vertices
    
class GeometryMask:
    
    
    """
    This function create the geometry mask for a set of input data and if prompted, will apply that to a set of data in order to elimate those elements,
    which do not respect the conditions.
    
    Args:
        _mask: dict;
            _mask.keys(): ['type','geom_params']
            _mask['type']: str;
                Type of mask. The currently accepted options are 'cyl' or 'prism'
                If _mask['type'] == 'prism':
                    The mask created is a prism-like polyhedra.
                If _mask['type'] == 'cyl':
                    The mask created is a cylinder
            _mask['geom_params']: dict;
                Dictionary storing the geometrical information to construct the geometrical mask.
                It has different keys depending on the type of _mask.
                If _mask['type'] == 'prism':
                    _mask['geom_params'].keys(): ['verts','zmin','zmax']
                        _mask['geom_params']['verts']: list;
                        List sotring coordinates of the vertices.
                            _mask['geom_params']['verts'][i]: tuple;
                                _mask['geom_params']['verts'][i][0]: float; x-coordinate of the vertex;
                                _mask['geom_params']['verts'][i][1]: float; y-coordinate of the vertex;
                If _mask['type'] == 'cyl':
                    _mask['geom_params'].keys(): ['center','radius','zmin','zmax'];
                        _mask['geom_params']['center']: tuple;
                            Coordinates of the center of the circle.
                            _mask['geom_params']['center'][0]: float; x-coordinate of the center;
                            _mask['geom_params']['center'][1]: float; y-coordinate of the center;
                _mask['geom_params']['zmin']: float;
                    z-coordinate of the horizontal plane constituting the lower boundary surface.
                    If the key is not present in the dictionary, the coordinates will not be checked with respect to this plane.
                _mask['geom_params']['zmax']: float;
                    z-coordinate of the horizontal plane constituting the jupper boundary surface.
                    If the key is not present in the dictionary, the coordinates will not be checked with respect to this plane.
    """
    def __init__(self, _mask):
        # --- # CHECK CONSISTENCY OF INPUT DATA # --- #
        self.__surface = {}
        if _mask['type'] not in ['cyl', 'prism']:
            raise ValueError('Error! Mask type not recognized.')
        else:
            self.__surface['type'] = _mask['type']
        if 'zmin' not in _mask['geom_params']:
            self.__surface['zmin'] =  -np.inf
        else:
            self.__surface['zmin'] = _mask['geom_params']['zmin']
        if 'zmax' not in _mask['geom_params']:
            self.__surface['zmax'] =  np.inf
        else:
            self.__surface['zmax'] = _mask['geom_params']['zmax']
        vertices = []    
        if _mask['type'] == 'prism':
            for vertex in _mask['geom_params']['verts']:
                vertices.append(vertex)
            surface = Polyshape(vertices)
            self.__surface['surface'] = surface
        if _mask['type'] == 'cyl':
            self.__surface['surface'] = {'radius':_mask['geom_params']['radius'],
                                         'center':_mask['geom_params']['center'],}
            
    def set_mask(self, _data, _data_coords, _where='inside'):
        """
        This class will modify a set of data so that elements which do not respect the condition imposed.
        The structure of input paramters is created on the basis of SerpentDetectorTools
        Args:
        _data: np.array((n_x1, n_x2, n_x3));
            Data to be masked;
        _data_coords((n_x1, n_x2, n_x3));
            Numpy array sting the coordinates associated to the points in _data
            _data_coords[j1,j2,j3]: tuple
                _data_coords[j1,j2,j3][0]: float; x-coordinate of the j1-th, j2-th, j3-th datum;
                _data_coords[j1,j2,j3][1]: float; x-coordinate of the j1-th, j2-th, j3-th datum;
                _data_coords[j1,j2,j3][2]: float; x-coordinate of the j1-th, j2-th, j3-th datum;
            
        _where: str
            Identifies which kind of discrimination is applied.
            if _where == "inside":
                all points contanined within the boundary surfaces, below the upper limits and above lower limits are retained.
                All other points are set to zero.
            if _where == "outside":
                all points contanined within the boundary surfaces, below the upper limits and above lower limits are retained.
                are set as zero. All other points are retained
            Optional. Default value is 'inside'
        """
        def check_prism(coords,polygon,zmin,zmax, where):
            # --- # CHECK CONSISTENCY OF INPUT DATA # --- #
            if len(coords) != 3:
                raise ValueError('Error! There is an inconsistency with the coordinate dimension.')
            if where not in ['inside','outside']:
                raise ValueError('Error! Outside or inside the surface?')
            
            if polygon.contains(Point(coords[0],coords[1])) == True and coords[2] >= zmin and coords[2]<= zmax:
                if where == 'inside':
                    res = True
                else:
                    res = False
            else:
                if where == 'inside':
                    res = False
                else:
                    res = True
            return res
            
        
        def check_cyl(coords, center, radius, zmin, zmax, where):
            if len(coords) != 3:
                raise ValueError('Error! There is an inconsistency with the coordinate dimension.')
            if where not in ['inside','outside']:
                raise ValueError('Error! Outside or inside the surface?')
            if np.sqrt((coords[0]-center[0])**2+(coords[1]-center[1])**2) <= radius and coords[2] >= zmin and coords[2] <= zmax:
                
                if where == 'inside':
                    res = True
                else:
                    res = False
            else:
                if where == 'inside':
                    res = False
                else:
                    res = True
            return res

        # --- # CHECKING CONSTISTENCY OF INPUT DATA # --- #
        if np.shape(_data) != np.shape(_data_coords):
            raise ValueError('Error! There is inconsistency between coordinate and data array dimensions.')
        
        array_shape = np.shape(_data)
        res_data = np.zeros(array_shape)
        dim_x1 = array_shape[0]
        dim_x2 = array_shape[1]
        dim_x3 = array_shape[2]
        for j1 in range(dim_x1):
            for j2 in range(dim_x2):
                for j3 in range(dim_x3):
                    if self.__surface['type'] == 'prism':
                        if check_prism(_data_coords[j1,j2,j3],self.__surface['surface'], self.__surface['zmin'],self.__surface['zmax'],_where) == True:
                            res_data[j1,j2,j3] = _data[j1,j2,j3]
                    elif self.__surface['type'] == 'cyl':
                        if check_cyl(_data_coords[j1,j2,j3],self.__surface['surface']['center'],
                                     self.__surface['surface']['radius'],self.__surface['zmin'],
                                     self.__surface['zmax'],_where) == True:
                            res_data[j1,j2,j3] = _data[j1,j2,j3]
                         
        return res_data
        