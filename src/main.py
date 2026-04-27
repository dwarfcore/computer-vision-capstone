"""
Facial Recognition Video Monitoring System
Main entry point - launches the GUI application
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from gui.app import MonitoringApp

if __name__ == "__main__":
    app = MonitoringApp()
    app.run()
