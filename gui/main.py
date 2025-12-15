# main.py

# IMPORTS
import sys
import widgets
# import widgets
from PySide6.QtWidgets import (QApplication)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = widgets.RobotGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()