Classes to read matlab datasets. Extra classes for FLASHForward (DESY) / FACET are included to read images.

Use ex.:
    from root, do 
    pip install -e .

    from h5dataset.facet_dataset import FACET_dataset
    file_path = "your/path/dataset.mat"
      
    data = FACET_dataset(file_path)
        
    scalars = data.data_struct.scalars (All data is in "data_struct")
    common_index = scalars.common_index
    camera = "DTOTR2"
    images = data.get_images(camera) (np.ndarray with shape (shots, y-dim, x-dim))


for .mat datasets not from SLAC or FLASHForward:

    from dataset import Dataset
    file_path = "your/path/dataset.mat"
      
    data = Dataset(file_path)

there is also a function called "find_dataset(dataset_id, directory, depth) that finds the .mat-file with
the name "dataset_id" from "directory" going "depth" directories deep.
        
Each object in "data" is an object of type NameSpace (custom) in order to allow matlab dot-indexing.
