import numpy as np
import h5py
import scipy.io
import os
import sys
from scipy.signal import medfilt2d
from scipy.ndimage import median_filter
from IPython.display import display, clear_output



class Dataset:
    def __init__(self, h5file, facility="SLAC"):
        self.facility = facility.upper()
        # Convention: ALWAYS END A PATH IN A SLAH
        self.header = os.path.dirname(h5file) + "/"
        # Read data
        try:
            self.data = h5py.File(h5file)
            self.file_reader="h5py"
            print("reading with h5py")
        except OSError:
            self.data = scipy.io.loadmat(h5file, simplify_cells=True)
            self.file_reader="scipy"
            print("reading with scipy")
        
        # Set the values of the dict as attributes to either this class or the NameSpace class
        # and set each value as a new object if type(value)==dict
        self._setattr(self.data, parent_class=self)
        if self.facility.upper()=="SLAC":
            self.data_struct.scalars.common_index -= 1

    def _setattr(self, nested_dict=None, parent_class=None):
        if self.file_reader=="scipy":
            for key, value in nested_dict.items():
                #print(key, value)
                # Store key as object that can have its own attributes
                if isinstance(value, dict):
                    sub_object = NameSpace()
                    sub_object.value = value
                    setattr(parent_class, key, sub_object)
                    self._setattr(value, parent_class=sub_object)
                else:
                    setattr(parent_class, key, value)
        else:
            raise NotImplementedError("Support for .mat files read by h5py not implemented")

    def find_image_path(self, camera) -> list:
        if self.facility=="SLAC":
            # Find the images
            cam = self.data_struct.images[camera]
            loc = cam.loc

            # Check that data is available
            is_iterable = hasattr(loc, "__iter__") and not isinstance(loc, str)

            if not is_iterable:
                locs = [loc]
            else:
                locs = list(loc)

            valid_locs=[]

            for i, loc in enumerate(locs):
                if os.path.exists(loc):
                    valid_locs.append(loc)
                else:
                    filename = loc.split("/")[-1]
                    new_loc = self.header + "images/" + camera + "/" + filename
                    if os.path.exists(new_loc):
                        valid_locs.append(new_loc)
                    else:
                        print(f"Couldn't find file/step nr {i}")
            
            return valid_locs            

    def get_images(self, camera, memory_efficient=False):
        if self.facility.upper()=="SLAC":
            locs: list = self.find_image_path(camera)
            metadata = self.data_struct.metadata[camera]
            common_index = self.data_struct.scalars.common_index
            steps = self.data_struct.scalars.steps
    

            if memory_efficient: # Dont keep the images stored in memory
                def _generator_function():
                    # Convert to set for instant O(1) lookups (do this outside the loop!)
                    valid_indices = set(common_index) 
                    abs_idx = 0 
                    
                    for loc in locs:
                        # Context manager ensures files actually close when you move to the next one
                        with h5py.File(loc, 'r') as f: 
                            im_obj = f['entry/data/data']
                            
                            for im in im_obj:
                                if abs_idx in valid_indices:
                                    im = im.astype(np.int16)
                                    if metadata.IS_ROTATED == 1:
                                        im = im.T
                                    if camera == "CHER":
                                        im[270:410, 1140:1190] = 0
                                    yield Image(im)
                                
                                abs_idx += 1
                return _generator_function
                        
            else:
                start_idx = 0
                if metadata.IS_ROTATED==1:
                    images = np.zeros((len(steps), metadata.SizeX_RBV, metadata.SizeY_RBV), dtype='int16')
                else:
                    images = np.zeros((len(steps), metadata.SizeY_RBV, metadata.SizeX_RBV), dtype='int16')
                print("Loading images")    
                for i, loc in enumerate(locs):
                    im_obj = h5py.File(loc)
                    ims = np.array(im_obj['entry/data/data'])
                    dims = np.shape(ims) # shots, y, x #If not rotated
                    if metadata.IS_ROTATED==1:
                        ims = np.transpose(ims, (0, 2, 1))
                    
                    images[start_idx:start_idx+dims[0], :, :] = ims
                    start_idx += dims[0]
                    
                        
                images = images[common_index]
                size_im = images.nbytes / 1024**3
                print(f"Size of images in memory: {size_im:.2f} GB")

                if camera=="CHER":
                    images[:, 270:410,1140:1190] = 0#median_filter(images, (1,9,9))
                    
                return Image(images)

    def get_single_image(self, camera, n_index, memory_efficient=False):
        """
        WARNING: DO NOT USE THIS WHEN LOOPING OVER IMAGES TO DO ANALYSIS!!
        The function to get a single image loops through the iterator to get to the index, 
        so looping over 1000 shots would mean going through 1000! loops. Most of which in C, but still.
        """
        from itertools import islice
        
        image_generator = self.get_images(camera, memory_efficient=True)
        
        return next(islice(image_generator(), n_index, n_index+1))  
            

    def clean_images(self, images, camera, subtract_background=True, medfilt=True, roi_x=None, roi_y=None):
        if not subtract_background and not medfilt:
            print("Doing nothing. What else is there to clean based on if not medfilt nor background subtraction??")
            return None

        if subtract_background:
            background = self.data_struct.backgrounds[camera].astype(np.int16)
            if self.data_struct.metadata[camera].IS_ROTATED==0:
                background = background.T
                
            images -= background
            images[images<0] = 0
            
        if images.ndim==3:    
            if roi_y:
                images = images[:, roi_y[0]:roi_y[1], :]
            if roi_x:
                images = images[:, :, roi_x[0]:roi_x[1]]
    
            if medfilt:
                print("Performing median filtering")
                images = median_filter(images, (1,3,3))
        else:
            if roi_y:
                images = images[roi_y[0]:roi_y[1], :]
            if roi_x:
                images = images[:, roi_x[0]:roi_x[1]]
                
            if medfilt:
                images = medfilt2d(images)        

        return images

            
    def flip_through_images(self):
        self.get_images        
                    

# Helper class for             
class NameSpace:
    def __init__(self):
        pass
    def __getitem__(self, value):
        return getattr(self, value)
    def __str__(self):
        return str(self.value)
        
    def __repr__(self):
        return str(self.value)

class Image(np.ndarray):
    def __new__(cls, array, axs=None):
        obj = np.asarray(array).view(cls)

        if axs is not None:
            obj.axs = axs
            
        else:
            if obj.ndim==2:
                obj.axs = [np.arange(obj.shape[0]), np.arange(obj.shape[1])]
            elif obj.ndim==3:
                obj.axs = [np.arange(obj.shape[1]), np.arange(obj.shape[2])]
                
                
        if obj.ndim==2:
            if len(obj.axs[0]) != obj.shape[0] or len(obj.axs[1]) != obj.shape[1]:
                raise ValueError("Axis lengths must match array dimensions.")
                
        elif obj.ndim==3:
            if len(obj.axs[0]) != obj.shape[1] or len(obj.axs[1]) != obj.shape[2]:
                raise ValueError("Axis lengths must match array dimensions.")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
                    
        self.axs = getattr(obj, "axs", None)

    def format_axes(self):
        import matplotlib.ticker as ticker

        ## Set x/y ticks and labels to plot ##
        """
        NOTE: You can not plot a cropped image, e.g. plt.imshow(self[100:200, :]) unless
        you use the same cropped image to get the formatted axes!
        """
        locator = ticker.MaxNLocator(nbins='auto')      
        
        y_ticks_plt = locator.tick_values(0, self.shape[0])
        x_ticks_plt = locator.tick_values(0, self.shape[1])

        y_ticks=[]
        x_ticks=[]
        
        y_ticklabels = []
        x_ticklabels = []
        
        for y, x in zip(y_ticks_plt, x_ticks_plt):
            if y>=0 and y<self.shape[0]:
                y_ticks.append(y)
                y_ticklabels.append(self.axs[0][int(y)])
                
            if x>=0 and x<self.shape[1]:
                x_ticks.append(x)
                x_ticklabels.append(self.axs[1][int(x)])

        return (y_ticks, y_ticklabels), (x_ticks, x_ticklabels)
        

    def __getitem__(self, index_array):
        cropped = super().__getitem__(index_array)

        if isinstance(index_array, tuple) and self.ndim==3:
            match len(index_array):
                case 1:
                    return cropped
                case 2:
                    cropped.axs = [self.axs[0][index_array[1]], self.axs[1]]
                case 3:
                    cropped.axs = [self.axs[0][index_array[1]], self.axs[1][index_array[2]]]
        elif isinstance(index_array, tuple) and self.ndim==2:
            match len(index_array):
                case 1:
                    cropped.axs = [self.axs[0][index_array[0]], self.axs[1]]
                case 2:
                    cropped.axs = [self.axs[0][index_array[0]], self.axs[1][index_array[1]]]
        elif self.ndim==2:
            cropped.axs = [self.axs[0][index_array], self.axs[1]]
                
            
        return cropped
        