# -*- coding: utf-8 -*-
"""
This script contains all the general functions that are used in the CVsim package
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

def checkkeys(funcname, kwargs, allowedkeys):
    '''
    This function determines if there is a key present in the kwargs that is
    not allowed. If that is the case, the functions will raise an error.
    '''
    added = kwargs.keys()-allowedkeys.keys()
    if len(added) > 0: #there is a keys that is not allowed
        for key in added:
            invalidkey = key
            break
        raise TypeError('{} got an unexpected keyword argument \'{}\' '.format(funcname, invalidkey))
    return

def find_nearest(array, value):
    '''
    Finds the nearest value and index in array.
    Input:
        array
        value
    Output:
        closest value
        index of closest values
    ''' 
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def fancyplot(data = [], *args, **kwargs):
    '''
    This function makes a fancy plot
    
    The input should be a np.array containing [[x1,y1],[x2,y2],etc].
    One can also make a nice looking figure without plotting any data. Just call GeneralFunctions.fancyplot().
    One can use plt.plot to add a graph to the nice looking figure.
    
    The options are:
        ylim = [float,float]
        xlim = [float,float]
        label = [x string,y string]
        title = string
        legend = [string]
        imagesize = [float,float]
        fontsize = int
        grid = bool
        saveas = string
        origin = bool
        linestyle = string
        color = string
        loc = string
        xscale = 'linear', 'log'
        yscale = 'linear', 'log'
        showfig = bool
        returnfig = bool
    '''
    #define some default parameters
    ylim = kwargs.get('ylim', None)
    xlim = kwargs.get('xlim', None)
    label = kwargs.get('label', ['',''])
    Title = kwargs.get('title', '')
    legend = kwargs.get('legend', None)
    ImageSize = kwargs.get('imagesize', [7.2,4.45])
    FontSize = kwargs.get('fontsize', 14)
    grid = kwargs.get('grid', True)
    saveas = kwargs.get('saveas', None)
    origin = kwargs.get('origin', True)
    style = kwargs.get('style', '-')
    Color = kwargs.get('color', None)
    loc = kwargs.get('loc', 'best')
    xscale = kwargs.get('xscale', 'linear')
    yscale = kwargs.get('yscale', 'linear')
    showfig = kwargs.get('showfig', True)
    returnfig = kwargs.get('returnfig', False)
    
    allowedkeys = {'ylim':None,'xlim':None,'label':None,'title':None,'legend':None,
                   'imagesize':None,'fontsize':None,'grid':None,'saveas':None,
                   'origin':None,'style':None,'loc':None,'color':None,
                   'xscale':None,'yscale':None,'showfig':None,'returnfig':None}
    checkkeys('fancyplot', kwargs, allowedkeys)
    
    if data == []:
        # if the data is emptty, the user only wants to have nice figure without a plot.
        showfig = False
        
    if ImageSize == 'small':
        ImageSize = [4.5,3]
    
    if showfig or returnfig:
        # generate the figure of a certain ImageSize
        fig = plt.figure(figsize=ImageSize)
        
    # change some standard settings
    params = {'legend.fontsize': FontSize-2,    # fontsize of the legend
              'font.family' : 'serif',          # font family
              'xtick.labelsize' : FontSize-2,   # fontsize of the tick labels
              'ytick.labelsize' : FontSize-2,   # fontsize of the tick labels
              'axes.titlesize' : FontSize,      # fontsize of title 
              'axes.labelsize': FontSize,       # fontsize of label 
              'lines.linewidth' : 2,            # linewidth
              'lines.markersize' : 6,           # markerwidth  
              'figure.dpi' : 100,               # set dpi
              'figure.figsize' : ImageSize}     # set imagesize
    
    
    plt.rcParams.update(params)
    
    if args: # for if you gave the data as fancyplot(x,y)
        if len(args) > 1:
            raise ValueError('Too much variables are given!')
        data = np.array(data)
        data = np.append([data],[args[0]],axis=0)
    else:
        #convert data array to np.array
        data = np.array(data)
    
    #check if there is only one curve given
    size = data.shape
    if size[0] == 2 and size[1] != 2:
        # there is only one curve
        if type(style) is list: style = style[0]
        x = data[0]
        y = data[1]
        datapoints = len(x)
        if Color != None: plt.plot(x,y, style, color=Color[0])
        else: plt.plot(x,y, style)
    else:
        i = 0
        datapoints = 0
        for x,y in data:
            if type(style) is list: #check if the style is a list or not
                try: #check if the color is not default
                    plt.plot(x,y, style[i], color=Color[i])
                except IndexError: plt.plot(x,y, style[-1]) #take the last element
                except: plt.plot(x,y, style[i]) 
            else:
                try: #check if the color is not default
                    plt.plot(x,y, style, color=Color[i])
                except: plt.plot(x,y, style)
            i += 1
            try: datapoints += len(x)
            except TypeError: pass
            #if the second list is just a single data point, it will raise an error, so neglect it by using try
            
    #display the title
    plt.title(Title)
    # set the range if given
    if ylim != None: plt.ylim(ylim)
    if xlim != None: plt.xlim(xlim)
    
    # displays the legend if given
    if legend != None: 
        if datapoints < 199990 or loc != 'best':
            if loc == 'outside right':
                if isinstance(legend,str): legend = [legend]
                plt.legend(legend,loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                if isinstance(legend,str): legend = [legend]
                plt.legend(legend,loc=loc)
        else:
            plt.legend(legend,loc='upper right')

    # displays the grids if wanted (default is yes)
    plt.grid(grid)
    
    #display the x and y labels
    plt.xlabel(label[0])
    plt.ylabel(label[1])

    # set the x and yscale, default is linear
    plt.xscale(xscale)
    try: plt.yscale(yscale)
    except UserWarning: 
        warnings.simplefilter("ignore", UserWarning)
        plt.yscale(yscale)
    
    # get the axis from the current graph, this line should be stated after the data has been plotted.
    ax=plt.gca()
    #plot thick black origin axis
    if origin:
        if yscale != 'log': ax.axhline(y=0, color='k', linewidth=1)
        ax.axvline(x=0, color='k', linewidth=1)
        
    # saves the figure in the current working directory as saveas
    if saveas != None: plt.savefig(fname=saveas, dpi=300, bbox_inches='tight')
    if showfig:
        # generate the figure of a certain ImageSize
        fig = plt.gcf()
        fig.set_size_inches(ImageSize)
        plt.show()       

    
    if returnfig: return fig, ax
    else: return


    
def find_nearest(array, value):
    '''
    Finds the nearest value and index in array.
    Input:
        array
        value
    Output:
        closest value
        index of closest values
    ''' 
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

    
def Animate(x,y, *args, **kwargs):
    '''
    This function returns an animation of the data set x, y[i]
    Input parameters:
    Required:
        x = 1D np.array
        y = 2D np.array(each row will be plotted as new step)
    or:
        x = 1D np.array
        y = [2D np.array, 2D np.array, etc]
        
    Optional:
        t_an: contains information about the time steps (1D np.array)
        E_app: contains information about the potential steps (1D np.array)
        frames: the amount of frames that will be displayed (int)
        interval: time between frames in ms (int)
        repeat: repeat the animation (bool) (defaul is True)
        t_wait: amount of seconds that the systems wait in order to continue the animation (float)
        t_unit: give the unit of t, deaulft is sec (string)
        saveas: save the animation using this filename (string)
        marker: specifies the marker (string)
        imagesize: imagesize of the animation (float,float)
        ylim = limits of the y axis [float,float]
        xlim = limits of the y axis [float,float]
        label = specifies the x and y labels [x string,y string]
        legend = specifies the legend [string1,string2..]
        fontsize = fontsize of the figure (int)
        grid = True to display grid, False otherwise (bool)
        origin = when true, the origin is shown by bold black lines (bool)
        style = specifies the linestyle of the lines (string)
        color = sets the color of the lines [string1, string2, ...]
        loc = sets the location of the legend (string)
        xscale = scale of the x-axis: choose from 'linear', 'log'
        yscale = scale of the y-axis: choose from 'linear', 'log'
        
    Output: animation
    '''
    frames = kwargs.get('frames', 100)
    interval = kwargs.get('interval', 50)
    saveanim = kwargs.get('saveas', None)
    repeat = kwargs.get('repeat', True)
    colors = kwargs.get('color', None)
    legend = kwargs.get('legend', None)
    loc = kwargs.get('loc', 'upper right')
    imagesizean = kwargs.get('imagesize', (6.5,5)) # different default compared to fancyplot to fit x-label 
    t_wait = kwargs.get('t_wait', 1.5)
    t_an  = kwargs.get('t_an', None) # if time_an is not specified, the iteration number i will be displayed in the title.
    E_app  = kwargs.get('E_app', None) # if E_app is specified, t_an also needs to be specified to display it
    t_unit = kwargs.get('t_unit', 's')
    marker = kwargs.get('marker', None)
    
    allowedkeys = {'frames':None,'ylim':None,'xlim':None,'interval':None,
                   'saveas':None,'repeat':None,'imagesize':None,
                   't_wait':None,'t_an':None,'t_unit':None,'marker':None,
                   'color':None,'legend':None,'loc':None, 'label':None,
                   'fontsize':None,'grid':None,'origin':None,'style':None,
                   'xscale':None,'yscale':None, 'E_app':None}
    checkkeys('animate', kwargs, allowedkeys)
    
    import time
    from matplotlib.animation import FuncAnimation
    
    options = {'returnfig': True,
               'saveas': None,
               'imagesize': imagesizean}
    if 'fontsize' in kwargs: options['fontsize'] = kwargs['fontsize']
    if 'grid' in kwargs: options['grid'] = kwargs['grid']
    if 'origin' in kwargs: options['origin'] = kwargs['origin']
    if 'style' in kwargs: options['style'] = kwargs['style']
    if 'xscale' in kwargs: options['xscale'] = kwargs['xscale']
    if 'yscale' in kwargs: options['yscale'] = kwargs['yscale']
    if 'label' in kwargs: options['label'] = kwargs['label']
    
    if 'xlim' in kwargs: options['xlim'] = kwargs['xlim']
    if type(x) is list: # multiple values of x are given
        if len(x) != len(y): # check if the len of x list is same as y list
            raise ValueError('The lists x and y must have the same length, but have length {} and {}'.format(len(x),len(y)))
        elif 'xlim' not in kwargs: # So x contains now more than one data set
            options['xlim'] = [min(np.array(np.array(x).flatten())), max(np.array(np.array(x).flatten()))]
    elif type(y) is list: # same x var for all y var
        if 'xlim' not in kwargs: options['xlim'] = [min(x),max(x)]
        x = [x] *len(y)
    else: # no list, just check x with y
        if len(x) != len(np.transpose(y)):
            raise ValueError('x and y must have same length, but have length {} and {}'.format(len(x),len(np.transpose(y))))
        if 'xlim' not in kwargs: options['xlim'] = [min(x),max(x)]

    # check if the ylim is specified
    if 'ylim' in kwargs: options['ylim'] = kwargs['ylim']
    elif type(y) is list: 
        options['ylim'] = [min(np.array(y).flatten()),max(np.array(y).flatten())]
    elif type(y) == type(np.ones(1)): 
        options['ylim'] = [min(y.flatten()),max(y.flatten())]
    else: raise ValueError('y needs to be a list or a np.array')
    
    # select number of frames
    if isinstance(y, list) and len(y[0]) > frames:
        idx = np.round(np.linspace(0, len(y[0]) - 1, frames)).astype(int)
        y = [el[idx] for el in y] 
        if t_an is not None: t_an = t_an[idx]
        if E_app is not None: E_app = E_app[idx]
    else:
        idx = np.round(np.linspace(0, len(y) - 1, frames)).astype(int)
        y = y[:][idx] 
        if t_an is not None: t_an = t_an[idx]
        if E_app is not None: E_app = E_app[idx]
            
    fig, ax = fancyplot(**options)
    
    plotoptions = {'lw': 3, 'marker': marker}
    
    if type(y) is list: # if there are multiple lines to plot
        lines = []
        ax.set_title('t = 0')
        for index,el in enumerate(y): 
            # generate a line object for each line that will be plotted
            try: # costom colors and legend      
                lines.append(ax.plot([], [], color=colors[index], label=legend[index], **plotoptions)[0])
            except: # costom legend
                try: lines.append(ax.plot([], [], label=legend[index], **plotoptions)[0])
                except: # costom color
                    try: lines.append(ax.plot([], [], color=colors[index], **plotoptions)[0])
                    except: #use no legend and default colors
                        lines.append(ax.plot([], [], **plotoptions)[0])
        def init():
            for line in lines:
                line.set_data([],[]) # set the init data for each line
            return lines

        def animate(i): # the function that will be iterated by FuncAnimation
            for lnum,line in enumerate(lines):
                line.set_data(x[lnum],y[lnum][i]) # plot the data of the line
            if t_an is None: ax.set_title(u"i = {}".format(i)) # add the frame number to the title
            elif E_app is not None: ax.set_title(u"t = {:.2f} s, E_app = {:.2f} V".format(t_an[i], E_app[i]))# add time and potential to title
            else: ax.set_title(u"t = {:.2f} ".format(t_an[i])+t_unit) # add only the time to the title
            if i == 1:
                time.sleep(t_wait)
            return tuple(lines) + (ax,)
    else: # if there is only one line to plot:
        if colors != None and type(colors) != str:
                raise TypeError('color should be a string!')
        elif legend != None and type(legend) != str:
                raise TypeError('legend should be a string!')
        line, = ax.plot([], [], color=colors, label=legend, **plotoptions)
        ax.set_title('t = 0')
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(i):
            line.set_data(x, y[i])
            if t_an is None: ax.set_title(u"i = {}".format(i)) # add the frame number to the title
            elif E_app is not None: ax.set_title(u"t = {:.2f} s, E_app = {:.2f} V".format(t_an[i], E_app[i]))# add time and potential to title
            else: ax.set_title(u"t = {:.2f} ".format(t_an[i])+t_unit) # add only the time to the title
            if i == 1:
                time.sleep(t_wait)
            return line, ax,
        
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=False, repeat=repeat)
    
    # displays the legend if given   
    if legend != None:
        if loc == 'outside right':
            if type(y) is list:
                plt.legend(tuple(lines),legend,loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        elif type(y) is list:
            plt.legend(tuple(lines),legend,loc=loc)
        else:
            plt.legend(loc=loc)
    
    if saveanim != None:
        print('Saving the animation ...')
        anim.save(saveanim, writer='imagemagick')
        print('Saved the animation!')
    plt.show()
    return anim    

def Animate_2D(mesh2D, z, *args, **kwargs):
    '''
    This function returns an animation of the data set 2Dmesh, z[i]
    Input parameters:
    Required:
        mesh2D: mesh specifying the x and y value (create with np.meshgrid)
        z: values of z, each first dimension contains the profile at differnt t (3D np.array)
        
    Optional:
        t_an: contains information about the time steps (1D np.array)
        E_app: contains information about the applied potential (1D np.array)
        frames: the amount of frames that will be displayed (int)
        interval: time between frames in ms (int)
        repeat: repeat the animation (bool) (defaul is True)
        t_wait: amount of seconds that the systems wait in order to continue the animation (float)
        t_unit: give the unit of t, deaulft is sec (string)
        saveas: save the animation using this filename (string)
        imagesize: imagesize of the animation (float,float)
        ylim: limits of the y axis [float,float]
        xlim: limits of the y axis [float,float]
        label: specifies the label of the x and y axis ([string,string])
        fontsize: fontsize of the figure (int)
        grid: True to display grid, False otherwise (bool: default False)
        origin: when true, the origin is shown by bold black lines (bool: default False)
        xscale: scale of the x-axis: choose from 'linear', 'log'
        yscale: scale of the y-axis: choose from 'linear', 'log'
        barlabel: label of the color bar (string)
        minbar: min value of color bar (string)
        maxbar: max value of color bar (string)
        cmap: color function for the color bar (string)
        shading: when true, the pixels in the 2D plot will be interpolated (bool) 
            (NOTE: there is some bug when using flat, the data is not correctly
            plotted in the graph)
        outline: draws an outline. (2D np.array)
            [[xel1, xel2, ...],[yel1, yel2, ...]]
            or
            [[xel1, xel2, ...],[yel1, yel2, ...] , [xel10, xel11, ...],[yel10, yel11, ...], ...]
        outlinecolor: the color of the outline
        
    Output: 
        animation
    '''
    frames = kwargs.get('frames', 100)
    ylim = kwargs.get('ylim',None)
    xlim = kwargs.get('xlim',None)
    interval = kwargs.get('interval', 50)
    saveanim = kwargs.get('saveas', None)
    repeat = kwargs.get('repeat', True)
    imagesizean = kwargs.get('imagesize', (6.5,5)) # different default compared to fancyplot to fit x-label 
    t_wait = kwargs.get('t_wait', 1.5)
    t_an  = kwargs.get('t_an', None) # if time_an is not specified, the iteration number i will be displayed in the title.
    t_unit = kwargs.get('t_unit', 's')
    barlabel = kwargs.get('barlabel','')
    minbar = kwargs.get('minbar', None)
    maxbar = kwargs.get('maxbar', None)
    cmap = kwargs.get('cmap', 'hsv')
    shadingbool = kwargs.get('shading', True)
    E_app  = kwargs.get('E_app', None) # if E_app is specified, t_an also needs to be specified to display it
    outline = kwargs.get('outline', None) # draws an outline
    outlinecolor = kwargs.get('outlinecolor', 'lime') # the color of the outline
    title_3D = kwargs.get('title 3D','')
    
    #check the keys
    allowedkeys = {'frames':None,'ylim':None,'xlim':None,'interval':None,
                   'saveas':None,'repeat':None,'imagesize':None,
                   't_wait':None,'t_an':None,'t_unit':None,'barlabel':None,
                   'minbar':None,'maxbar':None,'cmap':None,'shading':None,
                   'fontsize':None,'grid':None,'origin':None,'xscale':None,
                   'yscale':None,'label':None, 'E_app':None,'outline':None,
                   'outlinecolor':None, 'title 3D':None}
    checkkeys('animate_2D', kwargs, allowedkeys)
    
    import time
    from matplotlib.animation import FuncAnimation
    
    X,Y = np.array(mesh2D)
    x = X[0]
    y = Y[:,0]
    
    # check of the dimensions of the mesh correspond to the z values
    if z[0].shape != ( len(y), len(x)):
        raise ValueError('mesh2D and z has different shapes, {}, and {}'.format(mesh2D[0].shape,z[0].shape))
    #check if the t_an has the same length as z
    if t_an is not None and len(z) != len(t_an):
        raise ValueError('t_an and z has different lenghts, {}, and {}'.format(len(t_an),len(z)))

    options = {'grid': False,
               'origin': False,
               'returnfig': True,
               'saveas': None,
               'imagesize': imagesizean}
    if 'fontsize' in kwargs: options['fontsize'] = kwargs['fontsize']
    if 'label' in kwargs: options['label'] = kwargs['label']
    if 'grid' in kwargs: options['grid'] = kwargs['grid']
    if 'origin' in kwargs: options['origin'] = kwargs['origin']
    if 'style' in kwargs: options['style'] = kwargs['style']
    if 'xscale' in kwargs: options['xscale'] = kwargs['xscale']
    if 'yscale' in kwargs: options['yscale'] = kwargs['yscale']
    if xlim is None: options['xlim'] = [min(x),max(x)] # check if the xlim is specified
    else: options['xlim'] = xlim
    if ylim is None: options['ylim'] = [min(y),max(y)] # check if the ylim is specified
    else: options['ylim'] = ylim
	
    
    options2D = {'cmap': cmap}
    if shadingbool: options2D['shading'] = 'gouraud'
    if minbar is None: options2D['vmin'] = min(z.flatten()) # the limit was not specified
    else: options2D['vmin'] = minbar
    if maxbar is None: options2D['vmax'] = max(z.flatten()) # the limit was not specified
    else: options2D['vmax'] = maxbar
    
    # select number of frames
    if len(z) > frames:
        idx = np.round(np.linspace(0, len(z) - 1, frames)).astype(int)
        z = z[idx] 
        if t_an is not None: t_an = t_an[idx]
        if E_app is not None: E_app = E_app[idx]

    fig, ax = fancyplot(**options)
    quad = ax.pcolormesh(X, Y, z[0], **options2D)
    if outline is not None:
        x_outline = np.array([])
        y_outline = np.array([])
        for el in outline:
            x_outline = np.append(x_outline, X[tuple(el)])
            y_outline = np.append(y_outline, Y[tuple(el)])
        plt.scatter(x_outline,y_outline, marker='s', s=2, color=outlinecolor)
    line, = ax.plot([],[])
    bar = fig.colorbar(quad)
    bar.set_label(label=barlabel) #add a label to the colorbar

    def animate(i):
        quad.set_array(z[i].ravel())
        if t_an is None: ax.set_title(u"i = {}".format(i)+title_3D) # add the frame number to the title
        elif E_app is not None: ax.set_title(u"t = {:.2f} s, E_app = {:.2f} V".format(t_an[i], E_app[i])+title_3D)# add time and potential to title
        else: ax.set_title(u"t = {:.2f} ".format(t_an[i])+t_unit+title_3D) # add only the time to the title
        if i == 1:
            time.sleep(t_wait)
        return quad, ax,

    anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False, repeat=repeat)
    
    if saveanim != None:
        print('Saving the animation ...')
        anim.save(saveanim, writer='imagemagick')
        print('Saved the animation!')
    plt.show()
    return anim    

