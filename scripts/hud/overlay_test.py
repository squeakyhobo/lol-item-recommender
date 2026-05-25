
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer

class LTAOverlay(QWidget):
    def __init__(self):
        super().__init__()
        # 1. Make the window transparent and always on top
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 2. Setup Layout
        self.layout = QVBoxLayout()
        self.label = QLabel("LTA ADVISOR: WAITING FOR GAME...")
        self.label.setStyleSheet("color: #00FF00; font-size: 18px; font-weight: bold; background-color: rgba(0, 0, 0, 150); padding: 10px; border-radius: 5px;")
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        
        # 3. Position the overlay (e.g., top right)
        self.setGeometry(100, 100, 400, 100)

    def update_recommendation(self, text):
        self.label.setText(f"LTA ADVISOR: {text}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = LTAOverlay()
    overlay.show()
    
    # Simulate an update
    QTimer.singleShot(3000, lambda: overlay.update_recommendation(">>> BUY SUNDERED SKY <<<"))
    
    sys.exit(app.exec_())

