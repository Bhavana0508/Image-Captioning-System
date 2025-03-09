import os
from PIL import Image
import logging
from pathlib import Path
from .config import SUPPORTED_FORMATS, IMAGE_PROCESSING, UPLOAD_DIR

logger = logging.getLogger(__name__)

def is_valid_image(file_path: str) -> bool:
  """
  Check if the file is a valid image and meets the requirements.
  
  Args:
      file_path (str): Path to the image file
      
  Returns:
      bool: True if valid, False otherwise
  """
  try:
      # Check file extension
      ext = os.path.splitext(file_path)[1][1:].lower()
      if ext not in SUPPORTED_FORMATS:
          logger.warning(f"Unsupported file format: {ext}")
          return False
      
      # Check file size
      if os.path.getsize(file_path) > IMAGE_PROCESSING['max_size'] * 1024:  # Convert to bytes
          logger.warning(f"File too large: {os.path.getsize(file_path)} bytes")
          return False
      
      # Verify image can be opened
      with Image.open(file_path) as img:
          img.verify()
      return True
      
  except Exception as e:
      logger.error(f"Error validating image: {str(e)}")
      return False

def process_image_file(file_path: str) -> Path:
  """
  Process the uploaded image file.
  
  Args:
      file_path (str): Path to the image file
      
  Returns:
      Path: Path to the processed image
  """
  try:
      with Image.open(file_path) as img:
          # Convert to RGB if needed
          if img.mode != 'RGB':
              img = img.convert('RGB')
          
          # Resize if needed while maintaining aspect ratio
          max_size = IMAGE_PROCESSING['max_size']
          if max(img.size) > max_size:
              ratio = max_size / max(img.size)
              new_size = tuple(int(dim * ratio) for dim in img.size)
              img = img.resize(new_size, Image.Resampling.LANCZOS)
          
          # Save processed image
          output_path = UPLOAD_DIR / f"processed_{os.path.basename(file_path)}"
          img.save(
              output_path,
              quality=IMAGE_PROCESSING['quality'],
              optimize=True
          )
          
          return output_path
          
  except Exception as e:
      logger.error(f"Error processing image: {str(e)}")
      raise RuntimeError(f"Failed to process image: {str(e)}")

def cleanup_old_files(directory: Path, max_files: int = 100):
  """
  Clean up old files in the specified directory.
  
  Args:
      directory (Path): Directory to clean
      max_files (int): Maximum number of files to keep
  """
  try:
      files = list(directory.glob('*'))
      files.sort(key=lambda x: x.stat().st_mtime)
      
      if len(files) > max_files:
          for file in files[:-max_files]:
              file.unlink()
              logger.info(f"Cleaned up old file: {file}")
              
  except Exception as e:
      logger.error(f"Error cleaning up files: {str(e)}")