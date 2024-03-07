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
from package_utilities.geometrical_tools import CircleGenerator, PolygonGenerator
from scipy.spatial import ConvexHull, distance
"""
This set of tools helps for the creation of polygonal patches
"""

class PolygonGenerator2D:
    """
    This class is used to generate a generic 2D polygonal patch.
    
    Args:
    - _vertices: np.array((n_vertices,2))
        Vertices of the polygon. In order to avoid the problems in the generation, vertices must
        be provided ordinately so that sides are constructed with successive pair of points.
    """ 
    
    def __init__(self):
        
        self.__patches = []
    
    def add_polygon(self,_vertices):
        """
        This function create a polygonal patch from the vertices of a polygonal element.
        
        Args:
        - _vertices: np.array((n_vertices,2))
            Coordinates x and y of the vertices of the polygon. In order to avoid the problems in the generation, vertices must
            be provided ordinately so that sides are constructed with successive pair of points.
        """
        _polygon = Polygon(_vertices, closed = True)
        self.__patches.append(_polygon)
        
    def data_plotter(self,_data,_option):
        """
        This function diplays the data in polygonal meshes.
        
        Args:
        _data: np.array((n_x1))
            Data to be displayed.
        _option: dict
            Additional option of the plot.
            _option['property']: str
                Property to be displayed
            _option['unit_of_measure']: str
                Unit of measure of property
            _option['title']: str
                Title of the plot
            _option['text']: list, _option['text_on'][i]: str
                If _option['text'] == []
                    No text is displayed.
                If 'lat_pos' in _option['text']:
                    Lattice position is diplayed.
                If 'prop' in _option['text']:
                    The value of property is displayed in the figure.
                If 'err' in _option['text']:
                    The value of the error associated to property is displayed.
            _option['lattice']: np.array((n_element))
                Stores the denomination of the lattice position
            _option['err']: np.array((n_element))
                Stores the error associated to the data
            _option['loc_text']: np.array((n_element,2))
                Stores the coordinate of the text
            
            
        """
        if len(self.__patches) == np.size(_data):
            fig, ax = plt.subplots()
            ax.set_xlabel('x (cm)')
            ax.set_ylabel('y (cm)')
            ax.set_title(_option['title'])
            
            min_data = np.min(_data[_data > 0])
            max_data = np.max(_data[_data > 0])
            cmap = mpl.colormaps['autumn_r']
            patch_color = np.zeros((len(self.__patches)))
            plot_patches = []
            text_patch = []
            data = _data[_data > 0]
            text_loc = np.zeros((np.size(data),2))
            counter = 0 
            for poly in range(len(self.__patches)):
                if _data[poly] > 0:
                    text_p = []
                    plot_patches.append(self.__patches[poly])
                    if 'loc_text' in _option.keys() and len(_option['text']) > 0:
                        text_loc[counter,:] = _option['loc_text'][poly,:]
                        if 'lat_pos' in _option['text']:
                            text_p.append(str(_option['lattice'][poly]))
                        if 'prop' in _option['text']:
                            text_p.append('{0:.4g}'.format(_data[poly]))
                        if 'err' in _option['text']:
                            text_p.append('{0:.4g}'.format(_option['err'][poly]))
                        text_patch.append('\n'.join(text_p))
                    counter = counter + 1
                    if _data[poly] == max_data:
                        max_patch = [self.__patches[poly]]
                        text_max = ['Maximum value\n{0:.4g}'.format(max_data)]
                        if 'err' in _option.keys():
                            text_max.append(' +/- {0:.4g} '.format(_option['err'][poly])+_option['unit_of_measure'])
                        if 'lattice' in _option.keys():
                            text_max.append('\nlocated in '+str(_option['lattice'][poly]))    
                        text_annotation = ''.join(text_max)
                        
            polygon_plot = PatchCollection(plot_patches)
            polygon_plot.set_array(data)
            polygon_plot.set_edgecolor('w')
            polygon_plot.set_linewidth(2.0)
            polygon_plot.set_cmap('autumn_r')
            ax.add_collection(polygon_plot)
            
            max_plot = PatchCollection(max_patch)
            max_plot.set_array(max_data)
            max_plot.set_edgecolor('b')
            max_plot.set_facecolor('none')
            max_plot.set_linewidth(1.5)
            ax.add_collection(max_plot)
            
            at = AnchoredText(
            text_annotation, prop=dict(size=6), frameon=True, loc='lower right')
            
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            at.patch.set_facecolor('cyan')
            at.patch.set_edgecolor('b')

            ax.add_artist(at)
            
            text_colorbar = _option['property'] +' '+ _option['unit_of_measure']
            
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().y1-ax.get_position().y0])
            cbar = fig.colorbar(polygon_plot, ax= ax, cax= cax)
            cbar.ax.set_title(text_colorbar,loc= 'center')
            if 'loc_text' in _option.keys() and len(_option['text']) > 0:
                for text in range(len(text_patch)):
                    ax.text(s= text_patch[text],x= text_loc[text,0],y= text_loc[text,1],ha = 'center',va = 'center', fontsize= 5)
            ax.autoscale_view()
            ax.set_aspect('equal', adjustable='box')
            return fig, ax
        else:
            raise ValueError('Error! The number of polygon and data do not share the same dimension.')
            
        



class General2DPlotter:
    """
    This class allows to create pictures for square lattices.
    
    Returns a picture.
    
    """
    def __init__(self):
        self.__fig, self.__ax = plt.subplots()
        self.__legend_symbol = {}
        
        
        
    def add_label(self,_xlabel,_ylabel):
        """
        This function add labels to the plot.

        Args:
            _xlabel: str
                String to be applied to the x-label
            _ylabel: str
                String to be applied to the y-label
        """
        self.__ax.set_xlabel(_xlabel)
        self.__ax.set_ylabel(_ylabel)
    
    def add_boundary_patches(self, _patches):
        """
        This function will add boundary patches to the plot. For these patches only the black contour will be plotted.
        
        Args:
        _patches: list
            Patches list generated with either with circle generator or polygon generator.
        """
        bound_patch = PatchCollection(_patches)
        bound_patch.set_edgecolor('k')
        bound_patch.set_facecolor('none')
        bound_patch.set_linewidth(1.5)
        self.__ax.add_collection(bound_patch)
        
    
    def add_data_patches(self, _list_patches, _list_data,  _cbar_title):
        """
        This function will add boundary patches to the plot. For these patches only the black contour will be plotted.
        
        Args:
        _list_data: list
            List of data to be plotted.
        _list_patches: list
            Patches list generated with either with circle generator or polygon generator.
        """
        data_patch = PatchCollection(_list_patches)
        data_patch.set_array(_list_data)
        data_patch.set_edgecolor('w')
        data_patch.set_cmap('autumn_r')
        data_patch.set_linewidth(2.0)
        self.__ax.add_collection(data_patch)
        cax = self.__fig.add_axes([self.__ax.get_position().x1+0.01,self.__ax.get_position().y0,0.02,self.__ax.get_position().y1-self.__ax.get_position().y0])
        cbar = self.__fig.colorbar(data_patch, ax= self.__ax, cax= cax)
        cbar.ax.set_title(_cbar_title,loc= 'center')
    
    def add_text(self, _text_loc, _text, _option = {'fontsize':5,'ha':'center','va':'center'}):
        """
        This function will plot a certain text in the location of interest

        Args:
            _text_loc: tuple
                _text_loc[0] is the x-coordinate, _text_loc[1] is the y-coordinate
            _text: str
                Text to be displayed
            _option (optional): dict
                Option to be used for text plotting.
                _option['fontsize']:float
                    Size of the text. Size is the square number of points.
                _option['ha'], option['va']: text
                    Option for horizontal and vertical alignment of the text.
                Defaults to {'size':5,'ha':'center','va':'center'}.
        """
        self.__ax.text(s= _text,x= _text_loc[0],y= _text_loc[1],ha = _option['ha'],va = _option['ha'], fontsize= _option['fontsize'])
            
    def add_anchored_box(self, _text, _option={'frameon':True,'box_size':6,'loc':'lower right','boxstyle':"round,pad=0.,rounding_size=0.2",'facecolor':None,'edgecolor':'k'}):
        """
        This function creates a bounded anchored text.

        Args:
            _text: str
                Text to be displayed in the bounding box
            _option (optional): dict
                Dictionary storing optional plotting features.
                _option['frameon']: Bool; specifies if the frame of the box must plotted or not
                _option['boxsize']: float; Specifies the box dimension
                _option['loc']: str; Specifies where to plot the box
                _option['boxstyle']: str; Specifies the box styling.
                _option['facecolor']: None or color; Specifies the color of the box color
                _option['edgecolor']: None or color; Specifies the color of the edges
            Defaults to {'frameon':True,'box_size':6,'loc':'lower right','boxstyle':"round,pad=0.,rounding_size=0.2",'facecolor':None,'edgecolor':'k'}.
        """
        
        at = AnchoredText(_text, prop=dict(size= _option['box_size']), frameon= _option['frameon'], loc=_option['loc'])
        
        at.patch.set_boxstyle(_option['boxstyle'])
        at.patch.set_facecolor(_option['facecolor'])
        at.patch.set_edgecolor(_option['edgecolor'])
        self.__ax.add_artist(at)
        
    def add_legend(self):
        """
        This function will add a legend storing the symbols markers and labels.
        
        """
        handles=[]
        for key, res in self.__legend_symbol.items():
            handles.append(res)
            
        plt.legend(handles=handles)
        
        
    def add_title(self, title):
        """
        This function will add a title to the plot.
        
        Args:
        title: str
            Title to be plotted.
        """
        self.__ax.set_title(title)
        
    def add_symbols(self, loc, marker, label, option={'markersize':25,'color':'k','linewidth':1.0,'facecolor':None}):
        """
        This function will add symbols to the plot and will add their label to a collection of elements to be displayed in the legend.
        
        Args:
        loc: tuple
            Location of the symbol
        marker: str
            Marker to be displayed
        label: str
            Label to be displayed in the legend
        option: dict
            option['markersize']: float; marker dimension in squared pixels
            option['color']: str or color; Color of the marker. Can be used also to make custom symbols
            option['linewidth']: float; linewidth of the symbol
            option['facecolor']: str or color; Face color of the marker. In case None is used it allows to create allow symbols.
            
        """
        symb = self.__ax.scatter(x=loc[0],y=loc[1],
                          s= option['markersize'],
                          marker=marker,
                          label= label,
                          color= option['color'],
                          linewidth=option['linewidth'],
                          facecolor=option['facecolor'])
        if label not in self.__legend_symbol.keys():
            self.__legend_symbol[label]=symb
    
    def show_plot(self):
        """
        This function will display the plot without additional operation to it.
        """
        self.__ax.axis('auto')
        self.__ax.set_autoscale_on(True)
        plt.show(block=True)
    
    def get_fig(self):
        """
        This function will return the object self.__fig
        """
        results = self.__fig
        return results
    
    def get_ax(self):
        """
        This function will return the object self.__ax
        """
        results = self.__ax
        return results
    
    def save_plot(self, _path, name_figure):
        """
        This function will save the plot in a given directory.
        Args:
            _path: str
                Path of the output directory
            name_figure: str
                Name of the figure
        """
        self.__ax.axis('auto')
        self.__ax.set_autoscale_on(True)
        path_output= _path+'/'+name_figure+'.png'
        plt.savefig(path_output,format = 'png',dpi = 600, bbox_inches = 'tight')
        plt.close()

class General3DPlotter:
    """
    This class performs 3D plotting of data.
    
    """        
    def __init__(self):
        self.__fig = plt.figure()
        self.__ax = self.__fig.add_subplot(projection='3d')
        self.__legend_symbol = {}
    
    def add_label(self,_xlabel,_ylabel,_zlabel):
        """
        This function add labels to the plot.

        Args:
            _xlabel: str
                String to be applied to the x-label
            _ylabel: str
                String to be applied to the y-label
            _zlabel: str
                String to be applied to the z-label
        """
        self.__ax.set_xlabel(_xlabel)
        self.__ax.set_ylabel(_ylabel)
        self.__ax.set_zlabel(_zlabel)
    
    def add_title(self, title):
        """
        This function will add a title to the plot.
        
        Args:
        title: str
            Title to be plotted.
        """
        self.__ax.set_title(title)
    
    def add_scatterdata(self, data, data_coord, cmap_label, order=500,option={'marker':'o',
                                                                            'alpha': 1.0,
                                                                            'size':5,
                                                                            'linewidth':1.0},plot_zeros=False):
        """
        This function will plot  data in a scatter format.

        Args:
            data: np.array((n_x1,n_x2,n_x3))
                Data to be plotted.
                data[j1,j2,j3]: float
            data_coord: np.array((n_x1,n_x2,n_x3))
                Coordinates of the data to be plotted.
                data_coord[j1,j2,j3]: tuple or list;
                    data_coord[j1,j2,j3][0]: float; x-coordinate of the data;
                    data_coord[j1,j2,j3][1]: float; y-coordinate of the data;
                    data_coord[j1,j2,j3][2]: float; z-coordinate of the data;
            cmap_label: str
                Label to be written on the colorbar.
            order: int
                Order of plotting. Default is 1.
            option: dict;
                Dictionary storing the options for the plotting.
                Defaults to {'marker':'o', 'alpha': 1.0, 'size':5, 'linewidth':1.0}.
            plot_zeros: bool;
                Defines if the zeros must be plotted or not. Default value is False, for which zeros data are not plotted.
        """
        # --- #  CHECK CONSISTENCY OF INPUT DATA # --- #
        if np.shape(data) != np.shape(data_coord):
            raise ValueError('Error! There is inconsistency between the shape of the arrays.')
        array_shape = np.shape(data)
        dim_x1 = array_shape[0]
        dim_x2 = array_shape[1]
        dim_x3 = array_shape[2]
        list_data = []
        list_xcoord = []
        list_ycoord = []
        list_zcoord = []
        for j1 in range(dim_x1):
            for j2 in range(dim_x2):
                for j3 in range(dim_x3):
                    if len(data_coord[j1,j2,j3]) != 3:
                        raise ValueError('Error! The length of the coordinate list-tuple is different from 3.')
                    else:
                        if data[j1,j2,j3] != 0.0:
                            list_data.append(data[j1,j2,j3])
                            list_xcoord.append(data_coord[j1,j2,j3][0])
                            list_ycoord.append(data_coord[j1,j2,j3][1])
                            list_zcoord.append(data_coord[j1,j2,j3][2])
                        else:
                            if plot_zeros == True:
                                list_data.append(data[j1,j2,j3])
                                list_xcoord.append(data_coord[j1,j2,j3][0])
                                list_ycoord.append(data_coord[j1,j2,j3][1])
                                list_zcoord.append(data_coord[j1,j2,j3][2])
        cmap = mpl.colormaps['autumn']                        
        p = self.__ax.scatter(list_xcoord, list_ycoord, list_zcoord, c= list_data,
                            marker = option['marker'],s= option['size'], alpha = option['alpha'],
                            linewidth = option['linewidth'], cmap = cmap,zorder=order)
        self.__fig.colorbar(p, label = cmap_label, orientation= 'vertical', shrink=0.5)
        
    def add_boundary_polyhedra(self, vertices, order=1):
        """
        This function will create a polyhedra boundary patch to be plotted with the selected.
        The vertices must be generated with the appropriate methods in geomtry_tools,
        particular, hexa_prism_generator(), square_prism_generator()
        
        Args:
        vertices: np.array((12, 3)) or np.array((8, 3))
            Array storing the coordinates of the vertices.
            vertices[vertex, 0]: float; x-coordinate of the vertex: vert
            vertices[vertex, 1]: float; y-coordinate of the vertex: vert
            vertices[vertex, 2]: float; z-coordinate of the vertex: vert
        order: int
            Order of diplaying    
        """
        
        # --- # CHECKING CONSISTENCY OF INPUT DATA # --- #
        if np.shape(vertices) != (12,3) and np.shape(vertices) != (8,3):
            raise ValueError('Error! This method only works with the vertices generated by hexa_prism_generator(), square_prism_generator() in geometry_tools.')
        array_dim = np.shape(vertices)
        nsides = int(array_dim[0]/2)
        
        for v in range(nsides):
            
            # --- # handling vertical edges # --- #
            self.__ax.plot([vertices[v,0],vertices[v+nsides,0]],[vertices[v,1],vertices[v+nsides,1]],zs=[vertices[v,2],vertices[v+nsides,2]],color='k',linestyle='--',zorder= order)
            order = order+1
            # --- # HANDLING BASES # --- #
            if v != nsides-1:
                self.__ax.plot([vertices[v,0],vertices[v+1,0]],[vertices[v,1],vertices[v+1,1]],zs=[vertices[v,2],vertices[v+1,2]],color='k',linestyle='--',zorder= order)
                order= order+1
                self.__ax.plot([vertices[v+nsides,0],vertices[v+nsides+1,0]],[vertices[v+nsides,1],vertices[v+nsides+1,1]],zs=[vertices[v+nsides,2],vertices[v+nsides+1,2]],color='k',linestyle='--',zorder= order)
                order= order+1
            else:
                self.__ax.plot([vertices[v,0],vertices[0,0]],[vertices[v,1],vertices[0,1]],zs=[vertices[v,2],vertices[0,2]],color='k',linestyle='--',zorder= order)
                order= order+1
                self.__ax.plot([vertices[v+nsides,0],vertices[nsides,0]],[vertices[v+nsides,1],vertices[nsides,1]],zs=[vertices[v+nsides,2],vertices[nsides,2]],color='k',linestyle='--',zorder= order)
                order= order+1   
    def show_plot(self):
        """
        This function will display the plot without additional operation to it.
        """
        plt.show(block=True)
    
    def get_fig(self):
        """
        This function will return the object self.__fig
        """
        results = self.__fig
        return results
    
    def get_ax(self):
        """
        This function will return the object self.__ax
        """
        results = self.__ax
        return results
    
    def save_plot(self, _path, name_figure):
        """
        This function will save the plot in a given directory.
        Args:
            _path: str
                Path of the output directory
            name_figure: str
                Name of the figure
        """
        self.__ax.axis('auto')
        self.__ax.set_autoscale_on(True)
        path_output= _path+'/'+name_figure+'.png'
        plt.savefig(path_output,format = 'png',dpi = 600, bbox_inches = 'tight')
        plt.close()