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
    def __init__(self, dataset_path, verbose=False):
        self.verbose: bool = verbose
        
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

