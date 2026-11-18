"""Desktop demo. Pick a home and an away team and press the button.

    python -m eplpredict.app

Needs assets/first2.ui (Qt Designer file) with these widgets:
comboBox, comboBox_2, pushButton, textBrowser, textBrowser_2 and an optional label_7.
An optional assets/background.png is shown in label_7.
"""
import sys

from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow

from .config import ASSET_DIR
from .predict import Predictor

UI_FILE = ASSET_DIR / "first2.ui"
BACKGROUND = ASSET_DIR / "background.png"

# Labels of the Stage 1 statistics as shown in the window.
STAT_LABELS = [("possession", "점유율"), ("shot", "슈팅 수 비율"), ("touch", "볼터치 수 비율"),
               ("pass", "패스 수 비율"), ("clearance", "클리어링(수비) 비율")]


def patch_ui_includes(path) -> None:
    """Point resource includes of a Qt Designer file to .py modules. Safe to run twice."""
    text = path.read_text(encoding="utf-8")
    lines = [line.replace(".qrc", ".py") if "include location" in line else line
             for line in text.splitlines(keepends=True)]
    patched = "".join(lines)
    if patched != text:
        path.write_text(patched, encoding="utf-8")


def format_shares(prediction) -> str:
    lines = []
    for key, label in STAT_LABELS:
        home = prediction.home_shares[key] * 100
        lines.append(f"{label} : {home:.1f}% : {100 - home:.1f}%")
    return "\n\n".join(lines)


class MainWindow(QMainWindow):
    def __init__(self, predictor: Predictor):
        super().__init__()
        self.predictor = predictor
        uic.loadUi(str(UI_FILE), self)
        if BACKGROUND.exists() and hasattr(self, "label_7"):
            self.label_7.setPixmap(QPixmap(str(BACKGROUND)))
        self.pushButton.clicked.connect(self.on_predict)

    def on_predict(self) -> None:
        home = self.comboBox.currentText()
        away = self.comboBox_2.currentText()
        prediction = self.predictor.predict(home, away)
        self.textBrowser.setText(format_shares(prediction))
        self.textBrowser_2.setText(prediction.label)


def main() -> None:
    if not UI_FILE.exists():
        sys.exit(f"{UI_FILE} not found. Put the Qt Designer file there "
                 "or use `python -m eplpredict.predict` instead.")
    patch_ui_includes(UI_FILE)
    app = QApplication(sys.argv)
    window = MainWindow(Predictor())
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
