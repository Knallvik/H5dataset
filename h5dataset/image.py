import numpy as np

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