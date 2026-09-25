from h5dataset.dataset import Dataset
import numpy as np
import os
from scipy.ndimage import median_filter
from scipy.signal import medfilt2d
from h5dataset.image import Image

class DESY_dataset(Dataset):

    def __init__(self, dataset_path, verbose=False):
        super().__init__(dataset_path, verbose)


    def find_image_path(self, camera: str, is_background=False) -> list:

        if is_background:
            path_list = np.atleast_1d(self.dataset.images[camera].noBeamBG.data)
        else:
            path_list = np.atleast_1d(self.dataset.images[camera].data)

        relative_paths = [self.header / path for path in path_list]

        return relative_paths

    def _get_axes(self, camera, shape_y, shape_x):
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
                

    def get_images(self, camera, is_background=False):
           
        locs = self.find_image_path(camera=camera, is_background=is_background)
        xnum, ynum = int(self.dataset.images[camera].xnumpixels), int(self.dataset.images[camera].ynumpixels)
        images = np.empty((len(locs), ynum, xnum), dtype=np.uint16) 

        for i, loc in enumerate(locs):
            if not os.path.isfile(loc):
                # Handle DAQ dropped frames where loc is just the directory or missing
                images[i, :, :] = np.zeros((ynum, xnum), dtype=np.uint16)
            else:
                file_size = os.path.getsize(loc)
                offset: int = file_size - 2 * xnum * ynum
                image = np.fromfile(loc, offset=offset, dtype=np.uint16).reshape(ynum, xnum)
                images[i, :, :] = image
        
        axs, calibs = self._get_axes(camera, ynum, xnum)

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