import sys
import os
from pathlib import Path

# Update Python path for pywin32 package
if sys.platform == 'win32':
    home_path = str(Path.home())
    pywin_path = (home_path + '\\Ikomia\\Python\\lib\\site-packages\\win32')
    if pywin_path not in sys.path:
        sys.path.append(pywin_path)

    pywin_path = (home_path + '\\Ikomia\\Python\\lib\\site-packages\\win32\\lib')
    if pywin_path not in sys.path:
        sys.path.append(pywin_path)

    pywin_path = (home_path + '\\Ikomia\\Python\\lib\\site-packages\\pythonwin')
    if pywin_path not in sys.path:
        sys.path.append(pywin_path)

    pywin32_path = (home_path + '\\Ikomia\\Python\\lib\\site-packages\\pywin32_system32')
    if pywin_path not in sys.path:
        sys.path.append(pywin32_path)

    os.environ['PATH'] += os.pathsep + home_path + '\\Ikomia\\Python\\lib\\site-packages\\pywin32_system32'