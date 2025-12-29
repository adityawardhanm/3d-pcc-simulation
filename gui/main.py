# main.py

import sys


try:
    import widgets
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from PySide6.QtWidgets import QApplication

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

def main():
    try:
        app = QApplication(sys.argv)
        
        app.setStyle('Fusion')
        
        window = widgets.RobotGUI()
        
        window.show()
        
        sys.exit(app.exec())

    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()