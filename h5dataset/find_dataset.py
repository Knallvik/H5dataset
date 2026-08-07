"""
find_dataset.py
"""
import os
from typing import Union
from pathlib import Path

def find_dataset(data_id: Union[str, int], directory: str, depth: int = 8):
        directory = Path(directory)
        #list content of dir
        dirs = os.listdir(directory)
        #ensure data_id can be input as int
        data_id = str(data_id)
        #check if we reached our data_set
        final_path = None
        
        for content in dirs:
            #get name of files (dirs will remain the same)
            name = content.split('.')

            #name the path of the content we are inspecting
            path = directory / content
            
            #enter the directory if it is one
            if os.path.isdir(path) and depth>1:
                
                #get the path of the recursive function
                final_path = find_dataset(data_id, path, depth = depth-1)
                #if final path has returned an actual path (as it does when finding the dataset),
                # then return that dataset to the previous recursions.
                if final_path is not None:
                    return final_path
                    
            #if the content is not a directory and has the name of our data_id
            elif name[0] == data_id:
                try: 
                    #try to read the file-extension
                    name1 = name[1]
                except:
                    #if it is a file without an extension
                    name1 = ''
                finally:
                    #if it is our .mat file, return the path
                    if name1 == 'mat':
                        return path