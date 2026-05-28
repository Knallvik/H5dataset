Class to read SLAC datasets.

Use ex.:
  from dataset import Dataset
  file_path = "your/path/dataset.mat"
  
  data = Dataset(file_path)
  
  scalars = data.data_struct.scalars (All data is in "data_struct")
  common_index = scalars.common_index
  camera = "DTOTR2"
  images = data.get_images(camera) np.ndarray with shape (shots, y-dim, x-dim)

Each object in "data" is an object of type NameSpace (custom) in order to allow matlab dot-indexing.
