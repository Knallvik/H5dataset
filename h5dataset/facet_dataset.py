from h5dataset.dataset import Dataset
import numpy as np
import os
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
from h5dataset.image import Image
import h5py

class FACET_dataset(Dataset):

    def __init__(self, dataset_path, verbose=False):
        super().__init__(dataset_path, verbose)
        self.data_struct.scalars.common_index -= 1


    def find_image_path(self, camera: str) -> list:
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

    def _get_axes(self, camera, is_rotated=False):
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
                

    def get_images(self, camera, memory_efficient=False):
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

        return images

            
    def flip_through_images(self,
                            camera=None,
                            images=None,
                            clean=False,
                            memory_efficient=False,
                            cmap="viridis",
                            clim=None,
                            xlim=None,    # (xmin, xmax) in array column indices
                            ylim=None,    # (ymin, ymax) in array row indices
                            **clean_kwargs,
                            ):
        """
        Interactive image viewer for Jupyter or Terminal.
        
        Can either fetch and optionally clean images automatically using a 'camera' string, 
        or accept an existing image array directly via the 'images' argument.
        """
        import matplotlib.pyplot as plt

        # ------------------------------------------------------------------ 
        # 1. Image Acquisition & Processing
        # ------------------------------------------------------------------
        if images is None:
            # Pipeline Mode: Fetch and process from dataset
            if camera is None:
                raise ValueError("You must provide either a 'camera' name or an 'images' array.")
            
            images = self.get_images(camera, memory_efficient=memory_efficient)

            if callable(images) and not isinstance(images, np.ndarray):
                if self.verbose:
                    print("Materialising generator into an in-memory stack for flipping…")
                images = np.stack([np.asarray(im) for im in images()])

            if clean:
                images = self.clean_images(images, camera=camera, **clean_kwargs)
        else:
            # Direct Mode: Use the provided images
            # (Safeguard in case the user passed a memory-efficient generator directly)
            if callable(images) and not isinstance(images, np.ndarray):
                if self.verbose:
                    print("Materialising provided generator into an in-memory stack for flipping…")
                images = np.stack([np.asarray(im) for im in images()])

        # ------------------------------------------------------------------ 
        # 2. Dimensionality Safeguards
        # ------------------------------------------------------------------
        # Ensure we are working with a NumPy array to safely access .ndim and .shape
        if not isinstance(images, np.ndarray):
            images = np.asarray(images)
            
        if images.ndim == 2:
            # A single 2D image is wrapped in a 3D container with 1 shot
            images = images[np.newaxis, ...]
        elif images.ndim != 3:
            raise ValueError(f"flip_through_images expects a 2D or 3D array, but got {images.ndim}D.")
            
        n = images.shape[0]
        if n == 0:
            print("No images to display.")
            return

        # ------------------------------------------------------------------ 
        # 3. GUI Initialization
        # ------------------------------------------------------------------
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