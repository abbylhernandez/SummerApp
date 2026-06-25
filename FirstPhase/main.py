"""Entry point — wires the modules together and launches the GUI.

Run:  python main.py
"""

import sys
import logging

from PyQt5 import QtWidgets

from app import TrialLoggerApp


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    app = QtWidgets.QApplication(sys.argv)
    w = TrialLoggerApp()
    w.resize(760, 820)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
