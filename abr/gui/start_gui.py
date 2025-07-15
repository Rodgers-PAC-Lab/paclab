# Main script to run ABR session as a GUI

import sys
import signal
from . import main_window
import PyQt5.QtWidgets

# Apparently QApplication needs sys.argv for some reason
# https://stackoverflow.com/questions/27940378/why-do-i-need-sys-argv-to-start-a-qapplication-in-pyqt
app = PyQt5.QtWidgets.QApplication(sys.argv)

# Make CTRL+C work to close the GUI
# https://stackoverflow.com/questions/4938723/what-is-the-correct-way-to-make-my-pyqt-application-quit-when-killed-from-the-co
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Instantiate a MainWindow
win = main_window.MainWindow()

# Show it
win.show()

# Exit when app exec
sys.exit(app.exec())   