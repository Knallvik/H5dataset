import numpy as np
import h5py
import scipy.io
import os
import sys
import re
import keyword
from pathlib import Path
from typing import Literal
from scipy.signal import medfilt2d
from scipy.ndimage import median_filter
from IPython.display import display, clear_output
from h5dataset.colors import FLASHForward


class Dataset:
    def __init__(self, dataset_path, facility = "", verbose=False):
        self.verbose: bool = verbose
        
        self.facility: str = facility.upper()
        # Convention: ALWAYS END A PATH IN A SLAH
        self.header = Path(os.path.dirname(dataset_path))
        # Read data
        try:
            self.data = h5py.File(dataset_path)
            self.file_reader="h5py"
            print("reading with h5py")
        except OSError:
            self.data = scipy.io.loadmat(dataset_path, simplify_cells=True)
            self.file_reader="scipy"
            print("reading with scipy")
        
        # Set the values of the dict as attributes to either this class or the NameSpace class
        # and set each value as a new object if type(value)==dict
        self._setattr(self.data, parent_class=self)
        if self.facility.upper()=="SLAC":
            self.data_struct.scalars.common_index -= 1

    @staticmethod
    def _to_valid_attr(name: str) -> str:
        # Strip leading/trailing non-identifier characters (e.g. slashes)
        cleaned: str = name.strip('/#')
        # Replace any non-identifier character with underscore
        cleaned = re.sub(r'[^0-9a-zA-Z_]', '_', cleaned)
        # Collapse consecutive underscores into one
        cleaned = re.sub(r'_+', '_', cleaned)
        # Prefix with underscore if it starts with a digit
        if cleaned and cleaned[0].isdigit():
            cleaned = '_' + cleaned
        # If the result is a keyword, append underscore
        if keyword.iskeyword(cleaned):
            cleaned += '_'
        # Empty fallback
        return cleaned or '_unnamed'

    def _check_chr_dataset(self, dataset: h5py.Dataset) -> bool:
        to_check: list = [b'char', b'string']

        mc = dataset.attrs.get('MATLAB_class', None)

        return mc in to_check
        

    def _handle_reference(self, key, value) -> np.ndarray:
        # 1. Capture the exact shape after squeezing to preserve dimensionality later
        squeezed_val: np.ndarray = np.squeeze(value)
        arr_val: np.ndarray = np.atleast_1d(squeezed_val)
        original_shape: tuple = arr_val.shape
        
        # 2. Hard flatten the array so we iterate exactly one HDF5 reference at a time
        references: np.ndarray = arr_val.flatten()

        # Temporary list to hold the flat decoded elements
        decoded_flat: list = []
        
        for reference in references:
            # Handle empty/null HDF5 references to prevent crashes
            if not reference:
                decoded_flat.append(None)
                continue

            dereferenced_obj = self.data[reference]

            if isinstance(dereferenced_obj, h5py.Dataset):
                # Check if this referenced dataset contains MORE references (nested cell arrays)
                if h5py.check_dtype(ref=dereferenced_obj.dtype) is h5py.Reference:
                    if self.verbose:
                        print(f"{key}: Found nested reference array, recursing")
                    
                    nested_decoded: np.ndarray = self._handle_reference(key=key, value=dereferenced_obj)
                    decoded_flat.append(nested_decoded)
                
                else:
                    # Check if it's a character/string array
                    if self._check_chr_dataset(dereferenced_obj):
                        if self.verbose:
                            print(f"{key}: Found data that encodes strings or chars, decoding")
                        raw_data: np.ndarray = np.squeeze(dereferenced_obj)
                        decoded_str: str = raw_data.tobytes().decode('utf-16-le', errors='replace').replace('\x00', '')
                        decoded_flat.append(decoded_str)
                    
                    else:
                        # Standard numeric dataset
                        if self.verbose:
                            print(f"{key}: Data not encoding strings or chars, keeping as is")
                        dtype = dereferenced_obj.dtype
                        decoded_num = np.squeeze(dereferenced_obj).astype(dtype)
                        decoded_flat.append(decoded_num)

            elif isinstance(dereferenced_obj, h5py.Group):
                # It's a struct/group inside a cell array. 
                if self.verbose:
                    print(f"{key}: Found Group reference, parsing into NameSpace")
                
                sub_object = NameSpace()
                
                # Populate the sub_object recursively.
                # DO NOT store dereferenced_obj as an attribute (e.g., .value) here.
                self._setattr(nested_dict=dereferenced_obj, parent_class=sub_object)
                
                decoded_flat.append(sub_object)

        # 3. Reconstruct the array to its original shape using an object array
        decoded_array = np.empty(len(decoded_flat), dtype=object)
        for i, item in enumerate(decoded_flat):
            decoded_array[i] = item
            
        return decoded_array.reshape(original_shape)


    def _setattr(self, nested_dict = None, parent_class=None) -> None:
        if self.file_reader=="scipy":
            for key, value in nested_dict.items():
                safe_key = self._to_valid_attr(key)
                
                if hasattr(parent_class, safe_key) and self.verbose:
                    print(f"WARNING: Collision detected! Overwriting existing attribute '{safe_key}'")

                if isinstance(value, dict):
                    sub_object = NameSpace()
                    
                    # Store as structured attributes, DO NOT keep the raw dictionary pointer.
                    setattr(parent_class, safe_key, sub_object)
                    self._setattr(value, parent_class=sub_object)
                else:
                    setattr(parent_class, safe_key, value)
                    
        elif self.file_reader=="h5py":
            for key, value in nested_dict.items():
                safe_key = self._to_valid_attr(key)

                if hasattr(parent_class, safe_key):
                    print(f"WARNING: Collision detected! Overwriting existing attribute '{safe_key}'")

                if isinstance(value, h5py.Group):
                    sub_object = NameSpace()
                    
                    # Store as structured attributes, DO NOT keep the raw HDF5 Group pointer.
                    setattr(parent_class, safe_key, sub_object)
                    self._setattr(value, parent_class=sub_object)

                elif isinstance(value, h5py.Dataset):
                    if h5py.check_dtype(ref=value.dtype) is h5py.Reference: 
                        decoded = self._handle_reference(key=safe_key, value=value)
                        setattr(parent_class, safe_key, decoded)

                    else:
                        check = self._check_chr_dataset(value)
                        dtype = value.dtype
                        value_arr = np.squeeze(value).astype(dtype)

                        if check:
                            value_arr = value_arr.tobytes().decode('utf-16-le', errors='replace').replace('\x00', '')

                        setattr(parent_class, safe_key, value_arr)

        else:
            raise NotImplementedError("Support for .mat files read by h5py not implemented")

    def find_image_path(self, camera: str, is_background=False) -> list:
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
                    new_loc = self.header / "images" / camera / filename
                    if os.path.exists(new_loc):
                        valid_locs.append(new_loc)
                    else:
                        print(f"Couldn't find file/step nr {i}")
            
            return valid_locs
        
        elif self.facility == "DESY":
            if is_background:
                path_list = np.atleast_1d(self.dataset.images[camera].noBeamBG.data)
            else:
                path_list = np.atleast_1d(self.dataset.images[camera].data)

            relative_paths = [self.header / path for path in path_list]

            return relative_paths

    def _get_axes(self, camera, shape_y, shape_x, is_rotated=False):
        if self.facility == "DESY":
            try:
                cam_node = self.dataset.images[camera]
                
                xroi = np.atleast_1d(cam_node.xroi)
                yroi = np.atleast_1d(cam_node.yroi)
                xcalib = abs(cam_node.xcalib)
                ycalib = abs(cam_node.ycalib)

                x_offset = cam_node.xoffset
                y_offset = cam_node.yoffset
                
                if len(xroi) == shape_x:
                    x_axis = xroi.astype(int)
                else:
                    x_axis = np.arange(shape_x) + int(xroi[0])
                    
                if len(yroi) == shape_y:
                    y_axis = yroi.astype(int)
                else:
                    y_axis = np.arange(shape_y) + int(yroi[0])
                    
                return [y_axis, x_axis], [(ycalib, y_offset), (xcalib, x_offset)]
                
            except Exception as e:
                if self.verbose:
                    print(f"Could not load global axes for DESY {camera}: {e}")
                return None, None
                
        elif self.facility == "SLAC":
            try:
                meta = self.data_struct.metadata[camera]
                
                min_x = int(np.squeeze(meta.MinX_RBV))
                min_y = int(np.squeeze(meta.MinY_RBV))
                size_x = int(np.squeeze(meta.SizeX_RBV))
                size_y = int(np.squeeze(meta.SizeY_RBV))
                
                try:
                    res = float(np.squeeze(meta.RESOLUTION))
                except AttributeError:
                    res = 1.0 
                
                x_axis = np.arange(size_x) + min_x
                y_axis = np.arange(size_y) + min_y
                
                if is_rotated:
                    return [x_axis, y_axis], [(res,0.0), (res,0.0)]
                else:
                    return [y_axis, x_axis], [(res,0.0), (res,0.0)]
                    
            except Exception as e:
                if self.verbose:
                    print(f"Could not load global axes for SLAC {camera}: {e}")
                return None, None

    def get_images(self, camera, memory_efficient=False, is_background=False):
        if self.facility=="SLAC":
            locs = self.find_image_path(camera)
            metadata = self.data_struct.metadata[camera]
            common_index = self.data_struct.scalars.common_index
            steps = self.data_struct.scalars.steps
            
            is_rotated = (metadata.IS_ROTATED == 1)
            shape_y = metadata.SizeX_RBV if is_rotated else metadata.SizeY_RBV
            shape_x = metadata.SizeY_RBV if is_rotated else metadata.SizeX_RBV
            
            axs, calibs = self._get_axes(camera, shape_y, shape_x, is_rotated=is_rotated)

            # If you dont want to store the entire 3D array in memory
            # you can use this generator to loop over the images by onyl calling a single shot
            # into memory at a time
            if memory_efficient:
                def _generator_function():
                    valid_indices = set(common_index) 
                    abs_idx = 0 
                    
                    for loc in locs:
                        with h5py.File(loc, 'r') as f: 
                            im_obj = f['entry/data/data']
                            
                            for im in im_obj:
                                if abs_idx in valid_indices:
                                    im = im.astype(np.int16)
                                    if is_rotated:
                                        im = im.T
                                    if camera == "CHER":
                                        im[270:410, 1140:1190] = 0
                                    yield Image(im, axs=axs, calibs=calibs)
                                
                                abs_idx += 1
                return _generator_function
                        
            else:
                start_idx = 0
                if is_rotated:
                    images = np.zeros((len(steps), metadata.SizeX_RBV, metadata.SizeY_RBV), dtype=np.uint16)
                else:
                    images = np.zeros((len(steps), metadata.SizeY_RBV, metadata.SizeX_RBV), dtype=np.uint16)
                
                print("Loading images")    
                for i, loc in enumerate(locs):
                    im_obj = h5py.File(loc)
                    ims = np.array(im_obj['entry/data/data'])
                    dims = np.shape(ims)
                    if is_rotated:
                        ims = np.transpose(ims, (0, 2, 1))
                    
                    images[start_idx:start_idx+dims[0], :, :] = ims
                    start_idx += dims[0]
                    
                images = images[common_index]
                size_im = images.nbytes / 1024**3
                print(f"Size of images in memory: {size_im:.2f} GB")

                if camera=="CHER":
                    images[:, 270:410,1140:1190] = 0
                    
                return Image(images, axs=axs, calibs=calibs)
            
        elif self.facility=="DESY":
            locs = self.find_image_path(camera=camera, is_background=is_background)
            xnum, ynum = int(self.dataset.images[camera].xnumpixels), int(self.dataset.images[camera].ynumpixels)
            images = np.empty((len(locs), ynum, xnum), dtype=np.uint16) 

            for i, loc in enumerate(locs):
                file_size = os.path.getsize(loc)
                offset: int = file_size - 2 * xnum * ynum
                image = np.fromfile(loc, offset=offset, dtype=np.uint16).reshape(ynum, xnum)
                images[i, :, :] = image
            
            axs, calibs = self._get_axes(camera, ynum, xnum, is_rotated=False)
            return Image(images, axs=axs, calibs=calibs)



    def get_single_image(self, camera, n_index):
        """
        WARNING: DO NOT USE THIS WHEN LOOPING OVER IMAGES TO DO ANALYSIS!!
        The function to get a single image loops through the iterator to get to the index, 
        so looping over 1000 shots would mean going through 1000! loops. Most of which in C, but still.

        Not the best way of implementing this. much faster to get image_obj from h5py and index the shot...
        """
        from itertools import islice
        
        image_generator = self.get_images(camera, memory_efficient=True)
        
        return next(islice(image_generator(), n_index, n_index+1))  
            

    def clean_images(self, images, camera, subtract_background=True, medfilt=True, roi_x=None, roi_y=None, medfilt_kernel=3):
        if not subtract_background and not medfilt and not roi_x and not roi_y:
            print("No cleaning parameters set. Returning original images.")
            return images

        if self.facility == "SLAC":
            if subtract_background:
                background = self.data_struct.backgrounds[camera].astype(np.int16)
                if self.data_struct.metadata[camera].IS_ROTATED == 0:
                    background = background.T
                    
                images = images.astype(np.int16)
                images -= background
                images[images < 0] = 0
                
            if images.ndim == 3:    
                if roi_y:
                    images = images[:, roi_y[0]:roi_y[1], :]
                if roi_x:
                    images = images[:, :, roi_x[0]:roi_x[1]]
        
                if medfilt:
                    print("Performing median filtering")
                    filtered = median_filter(images, size=(1, medfilt_kernel, medfilt_kernel))
                    # Re-inject data to preserve the Image subclass and cropped axes
                    images_copy = images.copy()
                    images_copy[:] = filtered
                    images = images_copy
            else:
                if roi_y:
                    images = images[roi_y[0]:roi_y[1], :]
                if roi_x:
                    images = images[:, roi_x[0]:roi_x[1]]
                    
                if medfilt:
                    filtered = medfilt2d(images, kernel_size=medfilt_kernel)
                    images_copy = images.copy()
                    images_copy[:] = filtered
                    images = images_copy

        elif self.facility == "DESY":
            if subtract_background:
                # 1. Fetch backgrounds and calculate median
                background = self.get_images(camera=camera, is_background=True)
                if background.ndim==3:
                    # 2. CRITICAL: np.median returns float64. Cast to int16 to prevent UFuncTypeError
                    background = np.median(background, axis=0).astype(np.int16)
                    
                images = images.astype(np.int16)
                images -= background
                images[images < 0] = 0
                
            if images.ndim == 3:    
                if roi_y:
                    images = images[:, roi_y[0]:roi_y[1], :]
                if roi_x:
                    images = images[:, :, roi_x[0]:roi_x[1]]
        
                if medfilt:
                    print("Performing median filtering")
                    filtered = median_filter(images, size=(1, medfilt_kernel, medfilt_kernel))
                    # Re-inject data to preserve the Image subclass and cropped axes
                    images_copy = images.copy()
                    images_copy[:] = filtered
                    images = images_copy
            else:
                if roi_y:
                    images = images[roi_y[0]:roi_y[1], :]
                if roi_x:
                    images = images[:, roi_x[0]:roi_x[1]]
                    
                if medfilt:
                    filtered = medfilt2d(images, kernel_size=medfilt_kernel)
                    images_copy = images.copy()
                    images_copy[:] = filtered
                    images = images_copy

        return images

            
    def flip_through_images(self,
                            camera,
                            clean=False,
                            memory_efficient=False,
                            cmap="viridis",
                            clim=None,
                            xlim=None,    # (xmin, xmax) in array column indices
                            ylim=None,    # (ymin, ymax) in array row indices
                            **clean_kwargs,
                            ):
        import matplotlib.pyplot as plt

        # ------------------------------------------------------------------ load
        images = self.get_images(camera, memory_efficient=memory_efficient)

        if callable(images) and not isinstance(images, np.ndarray):
            if self.verbose:
                print("Materialising generator into an in-memory stack for flipping…")
            images = np.stack([np.asarray(im) for im in images()])

        if clean:
            images = self.clean_images(images, camera=camera, **clean_kwargs)

        if images.ndim == 2:
            images = images[np.newaxis, ...]
        n = images.shape[0]
        if n == 0:
            print("No images to display.")
            return

        state = {"idx": 0}

        if self._is_jupyter():
            self._flip_jupyter(images, state, n, cmap, clim, xlim, ylim)
        else:
            self._flip_terminal(images, state, n, cmap, clim, xlim, ylim)


    @staticmethod
    def _is_jupyter() -> bool:
        """True if running in a Jupyter notebook / JupyterLab / QtConsole."""
        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell is None:
                return False
            cls_name = type(shell).__name__
            # ZMQInteractiveShell = notebook / lab / qtconsole
            return cls_name == "ZMQInteractiveShell"
        except Exception:
            return False


    # --------------------------------------------------------------------------
    # Jupyter backend
    # --------------------------------------------------------------------------
    def _flip_jupyter(self, images, state, n, cmap, clim, xlim, ylim):
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display

        out = widgets.Output()
        slider = widgets.IntSlider(value=0, min=0, max=n - 1,
                                   description="Image:", continuous_update=True)
        prev_btn = widgets.Button(description="◀ Prev", button_style="info")
        next_btn = widgets.Button(description="Next ▶", button_style="info")
        controls = widgets.HBox([prev_btn, next_btn, slider])

        def render(i):
            out.clear_output(wait=True)
            with out:
                fig, ax = plt.subplots()
                img = np.asarray(images[i])
                p = ax.pcolormesh(img, cmap=cmap)
                if clim is not None:
                    p.set_clim(*clim)
                if xlim is not None:
                    ax.set_xlim(*xlim)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.invert_yaxis()
                ax.set_title(f"Image {i} / {n - 1}")
                fig.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
                plt.show()
                plt.close(fig)

        def on_prev(_b):
            state["idx"] = max(0, state["idx"] - 1)
            slider.value = state["idx"]

        def on_next(_b):
            state["idx"] = min(n - 1, state["idx"] + 1)
            slider.value = state["idx"]

        def on_slider(change):
            if change["name"] != "value":
                return
            state["idx"] = int(change["new"])
            render(state["idx"])

        prev_btn.on_click(on_prev)
        next_btn.on_click(on_next)
        slider.observe(on_slider, names="value")

        render(0)
        display(controls, out)


    # --------------------------------------------------------------------------
    # Terminal / plt.show() backend
    # --------------------------------------------------------------------------
    def _flip_terminal(self, images, state, n, cmap, clim, xlim, ylim):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        img0 = np.asarray(images[0])
        p = ax.pcolormesh(img0, cmap=cmap)
        if clim is not None:
            p.set_clim(*clim)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.invert_yaxis()
        title = ax.set_title(f"Image 0 / {n - 1}")
        fig.colorbar(p, ax=ax, fraction=0.046, pad=0.04)

        def update():
            i = state["idx"]
            img = np.asarray(images[i])
            p.set_array(img.ravel())
            # Re-derive clim: use fixed components if given, else auto from this frame
            vmin = clim[0] if clim is not None else img.min()
            vmax = clim[1] if clim is not None else img.max()
            p.set_clim(vmin, vmax)
            title.set_text(f"Image {i} / {n - 1}")
            fig.canvas.draw_idle()


        def on_key(event):
            k = event.key
            if k in ("right", " "):
                state["idx"] = min(n - 1, state["idx"] + 1); update()
            elif k in ("left", "backspace"):
                state["idx"] = max(0, state["idx"] - 1); update()
            elif k in ("q", "escape"):
                plt.close(fig)
            elif k == "home":
                state["idx"] = 0; update()
            elif k == "end":
                state["idx"] = n - 1; update()

        def on_scroll(event):
            if event.button == "up":
                state["idx"] = min(n - 1, state["idx"] + 1)
            elif event.button == "down":
                state["idx"] = max(0, state["idx"] - 1)
            update()

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("scroll_event", on_scroll)

        print("Terminal viewer: ←/→ or scroll to navigate · q/Esc to quit")
        plt.show()


                    

# Helper class for _settatr
class NameSpace:
    def __init__(self):
        pass
        
    def __getitem__(self, key):
        return getattr(self, key)
        
    def __str__(self):
        # Filter out internal Python attributes and display the dynamically added data keys
        keys = [k for k in self.__dict__.keys() if not k.startswith('_')]
        return f"NameSpace(attributes={keys})"
        
    def __repr__(self):
        # Keep __repr__ identical to __str__ for consistent printing in lists/arrays
        keys = [k for k in self.__dict__.keys() if not k.startswith('_')]
        return f"NameSpace(attributes={keys})"

class Image(np.ndarray):
    def __new__(cls, array, axs=None, calibs=None):
        obj = np.asarray(array).view(cls)

        if axs is not None:
            obj.axs = axs
        else:
            if obj.ndim==2:
                obj.axs = [np.arange(obj.shape[0]), np.arange(obj.shape[1])]
            elif obj.ndim==3:
                obj.axs = [np.arange(obj.shape[1]), np.arange(obj.shape[2])]
                
        if calibs is not None:
            obj.calibs = calibs
        else:
            obj.calibs = [(1.0, 0.0), (1.0, 0.0)]
                
        if obj.ndim==2:
            if len(obj.axs[0]) != obj.shape[0] or len(obj.axs[1]) != obj.shape[1]:
                raise ValueError(f"Axis lengths must match array dimensions. Y: {len(obj.axs[0])} vs {obj.shape[0]}, X: {len(obj.axs[1])} vs {obj.shape[1]}")
                
        elif obj.ndim==3:
            if len(obj.axs[0]) != obj.shape[1] or len(obj.axs[1]) != obj.shape[2]:
                raise ValueError(f"Axis lengths must match array dimensions. Y: {len(obj.axs[0])} vs {obj.shape[1]}, X: {len(obj.axs[1])} vs {obj.shape[2]}")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.axs = getattr(obj, "axs", None)
        self.calibs = getattr(obj, "calibs", None)

    @property
    def calib_axs(self):
        """Dynamically applies the calibration scalars to the current raw axes."""
        if self.axs is None or self.calibs is None:
            return self.axs
        return [self.axs[0] * abs(self.calibs[0][0]) + self.calibs[0][1], self.axs[1] * abs(self.calibs[1][0]) + self.calibs[1][1]]

    def format_axes(self, calibrated=True):
        import matplotlib.ticker as ticker
        
        locator = ticker.MaxNLocator(nbins='auto')      
        
        y_ticks_plt = locator.tick_values(0, self.shape[0])
        x_ticks_plt = locator.tick_values(0, self.shape[1])

        y_ticks=[]
        x_ticks=[]
        
        y_ticklabels = []
        x_ticklabels = []
        
        # Select the correct axes based on user preference
        axes_to_use = self.calib_axs if calibrated else self.axs
        
        for y, x in zip(y_ticks_plt, x_ticks_plt):
            if y>=0 and y<self.shape[0]:
                y_ticks.append(y)
                y_ticklabels.append(f"{axes_to_use[0][int(y)]:.4g}")
                
            if x>=0 and x<self.shape[1]:
                x_ticks.append(x)
                x_ticklabels.append(f"{axes_to_use[1][int(x)]:.4g}")

        return (y_ticks, y_ticklabels), (x_ticks, x_ticklabels)

    def __getitem__(self, index_array):
        cropped = super().__getitem__(index_array)

        if not isinstance(cropped, Image) or getattr(self, 'axs', None) is None:
            return cropped

        # Copy original axes so we don't accidentally mutate the parent array's tracking
        new_axs = [self.axs[0].copy(), self.axs[1].copy()]

        def slice_axis(axis_array, slice_obj):
            try:
                sliced = axis_array[slice_obj]
                return sliced if isinstance(sliced, np.ndarray) else np.array([sliced])
            except Exception:
                return axis_array

        if isinstance(index_array, tuple):
            if self.ndim == 3:
                if len(index_array) > 1:
                    new_axs[0] = slice_axis(new_axs[0], index_array[1])
                if len(index_array) > 2:
                    new_axs[1] = slice_axis(new_axs[1], index_array[2])
            elif self.ndim == 2: 
                if len(index_array) > 0:
                    new_axs[0] = slice_axis(new_axs[0], index_array[0])
                if len(index_array) > 1:
                    new_axs[1] = slice_axis(new_axs[1], index_array[1])
        else:
            if self.ndim == 2:
                new_axs[0] = slice_axis(new_axs[0], index_array)
                
        cropped.axs = new_axs
        return cropped