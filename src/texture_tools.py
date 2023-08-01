"""Provides tools for handling .jar files."""
import os
import shutil
import zipfile


class ArchiveHandler:
    """Handles .jar files related to Minecraft."""

    def __init__(self):
        self.aux_path = "ArchieveHandler_tmp"

    def extract_textures(self, file_path: str):
        """Extracts .jar file textures."""
        output_path = os.path.join(
            self.aux_path, os.path.basename(file_path[:len(file_path) - 4]))

        with zipfile.ZipFile(file_path, 'r') as archive:
            for file in archive.infolist():
                if file.filename.startswith('assets') and file.filename.endswith('.png'):
                    archive.extract(file, output_path)

    def create_texture_pack(self, dir_path: str):
        """Creates texture pack based on output from the extract function."""
        input_path = os.path.join(self.aux_path, dir_path)

        with zipfile.ZipFile(dir_path + ".zip", 'w') as zip_file:
            for foldername, _, filenames in os.walk(input_path):
                for filename in filenames:
                    source_file_path = os.path.join(foldername, filename)
                    destination_path = source_file_path[
                        source_file_path.find("assets"):]
                    zip_file.write(source_file_path, destination_path)

    def close(self):
        """Cleans up the temporary folder."""
        shutil.rmtree(self.aux_path)
