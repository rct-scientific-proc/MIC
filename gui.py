"""Optional PyQt5 GUI for the mic pipeline.

A thin command builder and runner over the three CLIs - the GUI never
re-implements them. Each tab assembles the exact `python train.py ...` /
`evaluate.py ...` / `inference.py ...` command (shown live in the preview
bar), launches it as a subprocess, and streams its console output into the
log pane; Stop kills the run. Only the everyday options get form fields -
anything else goes in the "extra CLI args" box, so every CLI option stays
reachable. Field values persist between launches.

Requires PyQt5 (an optional dependency, like optuna):

    pip install PyQt5

Then:

    python gui.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
INSTALL_HINT = "the GUI needs PyQt5 - to run it: pip install PyQt5"

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:  # main() prints the hint; the module stays importable
    QtCore = QtGui = QtWidgets = None


if QtWidgets is not None:

    class PathRow(QtWidgets.QWidget):
        """Line edit plus Browse button(s): kind is 'file', 'dir',
        'file+dir' (checkpoints: a .pt or a run directory), or 'files'
        (multiple paths joined with '; ')."""

        def __init__(self, kind="file", filt="All files (*)", placeholder=""):
            super().__init__()
            self.filt = filt
            self.edit = QtWidgets.QLineEdit()
            self.edit.setPlaceholderText(placeholder)
            lay = QtWidgets.QHBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.edit, 1)

            def button(label, fn):
                b = QtWidgets.QToolButton()
                b.setText(label)
                b.clicked.connect(fn)
                lay.addWidget(b)

            if kind in ("file", "file+dir"):
                button("File…", self._pick_file)
            if kind == "files":
                button("Files…", self._pick_files)
            if kind in ("dir", "file+dir", "files"):
                button("Dir…", self._pick_dir)

        def _start(self):
            return self.edit.text().split(";")[0].strip() or str(REPO)

        def _pick_file(self):
            p, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Choose file", self._start(), self.filt)
            if p:
                self.edit.setText(p)

        def _pick_files(self):
            ps, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Choose files", self._start(), self.filt)
            if ps:
                self.edit.setText("; ".join(ps))

        def _pick_dir(self):
            p = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose directory", self._start())
            if p:
                self.edit.setText(p)

        def text(self):
            return self.edit.text().strip()

        def setText(self, s):
            self.edit.setText(s)

    def _combo(*labels):
        c = QtWidgets.QComboBox()
        c.addItems(labels)
        return c

    def _edit(placeholder):
        e = QtWidgets.QLineEdit()
        e.setPlaceholderText(placeholder)
        return e

    class ToolTab(QtWidgets.QWidget):
        """One tab per CLI script. Subclasses fill `fields` (persisted),
        build their form, and implement argv()/missing()."""

        script = ""

        def __init__(self):
            super().__init__()
            self.fields: dict[str, QtWidgets.QWidget] = {}
            self.form = QtWidgets.QFormLayout(self)
            self.form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        def add(self, key, label, widget):
            self.fields[key] = widget
            self.form.addRow(label, widget)
            return widget

        def opt(self, a, flag, text):
            if text:
                a.extend([flag, text])

        def combo_opt(self, a, flag, combo):
            if combo.currentIndex() > 0:
                a.extend([flag, combo.currentText()])

        def argv(self) -> list[str]:
            raise NotImplementedError

        def missing(self) -> list[str]:
            return []

    class TrainTab(ToolTab):
        script = "train.py"

        def __init__(self):
            super().__init__()
            self.h5 = self.add("h5", "dataset (.h5)",
                               PathRow("file", "HDF5 (*.h5);;All files (*)"))
            self.config = self.add("config", "config (.json)",
                                   PathRow("file", "JSON (*.json)",
                                           "optional - CLI flags override it"))
            self.out = self.add("out", "output directory",
                                PathRow("dir", placeholder="default: runs/<timestamp>"))
            self.arch = self.add("arch", "architecture",
                                 _combo("(default: resnet18)", "resnet18",
                                        "resnet34", "resnet50"))
            self.epochs = self.add("epochs", "epochs",
                                   _edit("default: 50, or per smart level"))
            self.batch = self.add("batch", "batch size", _edit("default: 64"))
            self.lr = self.add("lr", "learning rate", _edit("default: 3e-4"))
            self.target = self.add("target", "target recall",
                                   _edit("default: 0.95"))
            self.smart = self.add("smart", "smart level",
                                  _combo("off", "1", "2", "3", "4", "5"))
            self.recall_first = self.add("recall_first", "recall-first level",
                                         _combo("off", "1", "2", "3", "4", "5"))
            self.optimizer = self.add("optimizer", "optimizer",
                                      _combo("(default: adamw)", "adamw", "sgd"))
            self.device = self.add("device", "device",
                                   _combo("auto", "cuda", "cpu"))
            self.amp = self.add("amp", "", QtWidgets.QCheckBox(
                "mixed precision (--amp, CUDA only)"))
            self.no_pretrained = self.add("no_pretrained", "", QtWidgets.QCheckBox(
                "random init (--no-pretrained)"))
            self.optuna = self.add("optuna", "optuna trials",
                                   _edit("blank = no search"))
            self.optuna_space = self.add("optuna_space", "optuna space (.json)",
                                         PathRow("file", "JSON (*.json)",
                                                 "optional - default space"))
            self.extra = self.add("extra", "extra CLI args",
                                  _edit("any other train.py flags, e.g. "
                                        "--workers 2 --threshold-mode per-class"))

        def argv(self):
            a = [self.h5.text()] if self.h5.text() else []
            self.opt(a, "--config", self.config.text())
            self.opt(a, "--out-dir", self.out.text())
            self.combo_opt(a, "--arch", self.arch)
            self.opt(a, "--epochs", self.epochs.text().strip())
            self.opt(a, "--batch-size", self.batch.text().strip())
            self.opt(a, "--lr", self.lr.text().strip())
            self.opt(a, "--target-recall", self.target.text().strip())
            self.combo_opt(a, "--smart", self.smart)
            self.combo_opt(a, "--recall-first", self.recall_first)
            self.combo_opt(a, "--optimizer", self.optimizer)
            self.combo_opt(a, "--device", self.device)
            if self.amp.isChecked():
                a.append("--amp")
            if self.no_pretrained.isChecked():
                a.append("--no-pretrained")
            self.opt(a, "--optuna", self.optuna.text().strip())
            self.opt(a, "--optuna-space", self.optuna_space.text())
            a.extend(shlex.split(self.extra.text()))
            a.append("--no-progress")
            return a

        def missing(self):
            if not self.h5.text() and not self.config.text():
                return ["dataset (.h5), or a config that provides it"]
            return []

    class EvaluateTab(ToolTab):
        script = "evaluate.py"
        SPLITS = {"train": "0", "validate": "1", "test": "2"}

        def __init__(self):
            super().__init__()
            self.ckpt = self.add("ckpt", "checkpoint (.pt or run dir)",
                                 PathRow("file+dir", "Checkpoint (*.pt);;All files (*)"))
            self.h5 = self.add("h5", "dataset (.h5)",
                               PathRow("file", "HDF5 (*.h5);;All files (*)"))
            self.split = self.add("split", "split",
                                  _combo("(default: test)", "train",
                                         "validate", "test"))
            self.out = self.add("out", "output directory",
                                PathRow("dir", placeholder="default: <checkpoint dir>/eval_<split>"))
            self.device = self.add("device", "device",
                                   _combo("auto", "cuda", "cpu"))
            self.extra = self.add("extra", "extra CLI args",
                                  _edit("any other evaluate.py flags, e.g. "
                                        "--no-report"))

        def argv(self):
            a = [p for p in (self.ckpt.text(), self.h5.text()) if p]
            if self.split.currentIndex() > 0:
                a.extend(["--split", self.SPLITS[self.split.currentText()]])
            self.opt(a, "--out-dir", self.out.text())
            self.combo_opt(a, "--device", self.device)
            a.extend(shlex.split(self.extra.text()))
            a.append("--no-progress")
            return a

        def missing(self):
            need = []
            if not self.ckpt.text():
                need.append("checkpoint")
            if not self.h5.text():
                need.append("dataset (.h5)")
            return need

    class InferenceTab(ToolTab):
        script = "inference.py"

        def __init__(self):
            super().__init__()
            self.ckpt = self.add("ckpt", "checkpoint (.pt or run dir)",
                                 PathRow("file+dir", "Checkpoint (*.pt);;All files (*)"))
            self.images = self.add("images", "images / directories",
                                   PathRow("files",
                                           "Images (*.png *.jpg *.jpeg *.bmp "
                                           "*.tif *.tiff);;All files (*)",
                                           "separate several paths with ;"))
            self.recursive = self.add("recursive", "", QtWidgets.QCheckBox(
                "scan directories recursively (--recursive)"))
            self.win_w = self.add("win_w", "window width (px)", _edit("required"))
            self.win_h = self.add("win_h", "window height (px)", _edit("required"))
            self.stride_x = self.add("stride_x", "stride x (px)", _edit("required"))
            self.stride_y = self.add("stride_y", "stride y (px)",
                                     _edit("default: same as stride x"))
            self.gt = self.add("gt", "ground truth (.json)",
                               PathRow("file", "JSON (*.json)",
                                       "optional - adds the GT report sections"))
            self.out = self.add("out", "output directory",
                                PathRow("dir", placeholder="default: runs/inference_<timestamp>"))
            self.batch = self.add("batch", "batch size", _edit("default: 64"))
            self.device = self.add("device", "device",
                                   _combo("auto", "cuda", "cpu"))
            self.grayscale = self.add("grayscale", "", QtWidgets.QCheckBox(
                "grayscale input (--grayscale)"))
            self.extra = self.add("extra", "extra CLI args",
                                  _edit("any other inference.py flags, e.g. "
                                        "--top-n 20 --no-report"))

        def argv(self):
            a = [self.ckpt.text()] if self.ckpt.text() else []
            a.extend(p.strip() for p in self.images.text().split(";") if p.strip())
            self.opt(a, "--window-width", self.win_w.text().strip())
            self.opt(a, "--window-height", self.win_h.text().strip())
            self.opt(a, "--stride-x", self.stride_x.text().strip())
            self.opt(a, "--stride-y", self.stride_y.text().strip())
            if self.recursive.isChecked():
                a.append("--recursive")
            self.opt(a, "--gt", self.gt.text())
            self.opt(a, "--out-dir", self.out.text())
            self.opt(a, "--batch-size", self.batch.text().strip())
            self.combo_opt(a, "--device", self.device)
            if self.grayscale.isChecked():
                a.append("--grayscale")
            a.extend(shlex.split(self.extra.text()))
            a.append("--no-progress")
            return a

        def missing(self):
            need = []
            if not self.ckpt.text():
                need.append("checkpoint")
            if not self.images.text():
                need.append("images")
            for label, e in (("window width", self.win_w),
                             ("window height", self.win_h),
                             ("stride x", self.stride_x)):
                if not e.text().strip():
                    need.append(label)
            return need

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, persist=True):
            super().__init__()
            self.persist = persist
            self.setWindowTitle("mic - train / evaluate / infer")
            self.resize(900, 720)
            self.proc: QtCore.QProcess | None = None

            self.train_tab = TrainTab()
            self.eval_tab = EvaluateTab()
            self.infer_tab = InferenceTab()
            self.tabs = QtWidgets.QTabWidget()
            self.tabs.addTab(self.train_tab, "Train")
            self.tabs.addTab(self.eval_tab, "Evaluate")
            self.tabs.addTab(self.infer_tab, "Inference")

            self.preview = QtWidgets.QLineEdit()
            self.preview.setReadOnly(True)
            mono = QtGui.QFont("Consolas")
            mono.setStyleHint(QtGui.QFont.Monospace)
            self.preview.setFont(mono)

            self.run_btn = QtWidgets.QPushButton("Run")
            self.stop_btn = QtWidgets.QPushButton("Stop")
            self.stop_btn.setEnabled(False)
            clear_btn = QtWidgets.QPushButton("Clear log")
            self.run_btn.clicked.connect(self.start)
            self.stop_btn.clicked.connect(self.stop)

            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(5000)
            self.log.setFont(mono)
            clear_btn.clicked.connect(self.log.clear)

            buttons = QtWidgets.QHBoxLayout()
            buttons.addWidget(self.run_btn)
            buttons.addWidget(self.stop_btn)
            buttons.addWidget(clear_btn)
            buttons.addStretch(1)

            central = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(central)
            lay.addWidget(self.tabs)
            lay.addWidget(QtWidgets.QLabel("command preview"))
            lay.addWidget(self.preview)
            lay.addLayout(buttons)
            lay.addWidget(QtWidgets.QLabel("console output"))
            lay.addWidget(self.log, 1)
            self.setCentralWidget(central)
            self.statusBar().showMessage("ready")

            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self.refresh_preview)
            self._timer.start(400)
            self.refresh_preview()
            if persist:
                self._load_settings()

        # ---- command preview / build ----------------------------------
        def current_tab(self) -> ToolTab:
            return self.tabs.currentWidget()

        def command(self) -> list[str]:
            tab = self.current_tab()
            return [sys.executable, "-u", str(REPO / tab.script)] + tab.argv()

        def refresh_preview(self):
            cmd = self.command()
            shown = ["python", Path(cmd[2]).name] + cmd[3:]
            self.preview.setText(subprocess.list2cmdline(shown))

        # ---- running --------------------------------------------------
        def start(self):
            if self.proc is not None:
                return
            tab = self.current_tab()
            need = tab.missing()
            if need:
                self.statusBar().showMessage("missing: " + ", ".join(need), 8000)
                return
            cmd = self.command()
            self.log.appendPlainText("$ " + subprocess.list2cmdline(
                ["python", Path(cmd[2]).name] + cmd[3:]))
            self.proc = QtCore.QProcess(self)
            self.proc.setWorkingDirectory(str(REPO))
            self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self.proc.readyReadStandardOutput.connect(self._read)
            self.proc.finished.connect(self._finished)
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.tabs.setEnabled(False)
            self.statusBar().showMessage(f"running {tab.script} …")
            self.proc.start(cmd[0], cmd[1:])

        def _read(self):
            text = bytes(self.proc.readAllStandardOutput()).decode(
                "utf-8", "replace")
            cursor = self.log.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.insertText(text)
            self.log.setTextCursor(cursor)
            self.log.ensureCursorVisible()

        def _finished(self, code, _status):
            self.statusBar().showMessage(
                "finished (exit 0)" if code == 0 else f"FAILED (exit {code})")
            self.proc = None
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.tabs.setEnabled(True)

        def stop(self):
            if self.proc is not None:
                self.statusBar().showMessage("stopping …")
                self.proc.kill()

        def closeEvent(self, event):
            if self.proc is not None:
                answer = QtWidgets.QMessageBox.question(
                    self, "Run in progress",
                    "A run is still active - stop it and quit?")
                if answer != QtWidgets.QMessageBox.Yes:
                    event.ignore()
                    return
                self.proc.kill()
            if self.persist:
                self._save_settings()
            event.accept()

        # ---- persistence ----------------------------------------------
        def _settings(self):
            return QtCore.QSettings("mic", "mic-gui")

        def _walk_fields(self):
            for prefix, tab in (("train", self.train_tab),
                                ("evaluate", self.eval_tab),
                                ("inference", self.infer_tab)):
                for key, w in tab.fields.items():
                    yield f"{prefix}/{key}", w

        def _save_settings(self):
            s = self._settings()
            for key, w in self._walk_fields():
                if isinstance(w, PathRow):
                    s.setValue(key, w.text())
                elif isinstance(w, QtWidgets.QLineEdit):
                    s.setValue(key, w.text())
                elif isinstance(w, QtWidgets.QComboBox):
                    s.setValue(key, w.currentIndex())
                elif isinstance(w, QtWidgets.QCheckBox):
                    s.setValue(key, int(w.isChecked()))
            s.setValue("tab", self.tabs.currentIndex())

        def _load_settings(self):
            s = self._settings()
            for key, w in self._walk_fields():
                v = s.value(key)
                if v is None:
                    continue
                if isinstance(w, PathRow):
                    w.setText(str(v))
                elif isinstance(w, QtWidgets.QLineEdit):
                    w.setText(str(v))
                elif isinstance(w, QtWidgets.QComboBox):
                    w.setCurrentIndex(int(v))
                elif isinstance(w, QtWidgets.QCheckBox):
                    w.setChecked(bool(int(v)))
            self.tabs.setCurrentIndex(int(s.value("tab", 0)))


def _self_test(win, out_dir: Path) -> None:
    """Offscreen render + command-construction assertions + screenshots -
    run by the smoke suite, no display or clicks needed."""
    t = win.train_tab
    t.h5.setText("data.h5")
    t.epochs.setText("5")
    t.smart.setCurrentIndex(3)
    t.amp.setChecked(True)
    assert t.argv() == ["data.h5", "--epochs", "5", "--smart", "3", "--amp",
                        "--no-progress"], t.argv()
    assert t.missing() == []

    e = win.eval_tab
    assert e.missing() == ["checkpoint", "dataset (.h5)"]
    e.ckpt.setText("runs/exp1")
    e.h5.setText("data.h5")
    e.split.setCurrentIndex(2)  # validate
    e.extra.setText("--no-report")
    assert e.argv() == ["runs/exp1", "data.h5", "--split", "1", "--no-report",
                        "--no-progress"], e.argv()

    i = win.infer_tab
    i.ckpt.setText("runs/exp1")
    i.images.setText("scans; extra.tif")
    i.win_w.setText("128")
    i.win_h.setText("128")
    i.stride_x.setText("64")
    i.recursive.setChecked(True)
    i.gt.setText("gt.json")
    assert i.argv() == ["runs/exp1", "scans", "extra.tif",
                        "--window-width", "128", "--window-height", "128",
                        "--stride-x", "64", "--recursive", "--gt", "gt.json",
                        "--no-progress"], i.argv()
    assert i.missing() == []

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tab in (("train", t), ("evaluate", e), ("inference", i)):
        win.tabs.setCurrentWidget(tab)
        win.refresh_preview()
        QtWidgets.QApplication.processEvents()
        win.grab().save(str(out_dir / f"{name}_tab.png"))
    print("GUI SELF-TEST PASSED")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", metavar="DIR", default=None,
                    help="render offscreen, assert the built commands, save "
                         "tab screenshots to DIR, and exit (used by the "
                         "smoke suite)")
    args = ap.parse_args(argv)
    if QtWidgets is None:
        raise SystemExit(INSTALL_HINT)
    if args.self_test is not None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    win = MainWindow(persist=args.self_test is None)
    if args.self_test is not None:
        win.show()
        _self_test(win, Path(args.self_test))
        return
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
