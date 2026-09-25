Classes to read matlab datasets. Extra classes for DESY/SLAC are included to read images.

Use ex.:
    from root, do pip install -e .

    from desy_dataset import FACET_dataset
    file_path = "your/path/dataset.mat"
      
    data = FACET_dataset(file_path)
        
    scalars = data.data_struct.scalars (All data is in "data_struct")
    common_index = scalars.common_index
    camera = "DTOTR2"
    images = data.get_images(camera) (np.ndarray with shape (shots, y-dim, x-dim))

    

Each object in "data" is an object of type NameSpace (custom) in order to allow matlab dot-indexing.
