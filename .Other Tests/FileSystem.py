import os

class FileSystem:
    def __init__(self) -> None: 
        pass
    
    def getFiles(self, folder_path = './Learning Data', extension = None):   
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(extension) and os.path.isfile(os.path.join(folder_path, f))
        ]
        sort_files = sorted(files, key=lambda x: os.path.getctime(x))
        
        return sort_files
    

    # Incremental File name selection
    def getNewFileName(self, extension):
        files = self.getFiles(extension = extension)
        lastFile =  None if len(files) <= 0 else os.path.splitext(os.path.basename(files[len(files)-1]))[0]
        
        newFileName = int(lastFile)+1 if lastFile is not None else 1

        return str(newFileName)