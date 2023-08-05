"""Provides tools for handling .jar files."""
import os
import zipfile
from pathlib import Path


class ArchiveHandler:
    """Handles .jar files related to Minecraft."""

    def __init__(self):
        self.data_path = os.path.join("..", "data")
        self.output_path = os.path.join("..", "out")

    def extract_textures(self, file_path: str):
        """Extracts .jar file textures."""
        output = os.path.join(self.data_path, "raw", Path(file_path).stem)
        os.makedirs(os.path.join('..', 'data', 'raw'), exist_ok=True)
        with zipfile.ZipFile(file_path, 'r') as archive:
            for file in archive.infolist():
                if file.filename.startswith('assets') and file.filename.endswith('.png'):
                    archive.extract(file, output)

    def create_texture_pack(self, dir_path: str):
        """Creates texture pack based on output from the extract function."""
        input_path = os.path.join(self.output_path, dir_path)

        with zipfile.ZipFile(dir_path + ".zip", 'w') as zip_file:
            for foldername, _, filenames in os.walk(input_path):
                for filename in filenames:
                    source_file_path = os.path.join(foldername, filename)
                    destination_path = source_file_path[
                        source_file_path.find("assets"):]
                    zip_file.write(source_file_path, destination_path)
