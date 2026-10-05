"""Optional PyQt5 dataset curator: browse an h5 snippet file by class and
split, inspect snippets, and remove bad ones from training entirely.

    python curate.py data.h5

--rows FILE opens a review list on top of the class views: a CSV with an
h5_row column (train.py / report.py write mined_hard_negatives.csv, the
hard negatives the miner found hardest - candidates for unlabelled
targets) or a plain text file of row numbers, shown in file order with
each row's class, split, and any difficulty column.

Removal never rewrites the (possibly huge, contiguous) images dataset:
Save writes a `removed` boolean dataset into the h5 (created on first
save; see h5_format.md) and every loader - training, evaluation, the
reports, optimize_h5 - skips flagged rows in every split from then on.
Removed snippets stay stored in the file, reviewable and restorable in
the "removed" view; purge them permanently with
`python optimize_h5.py in.h5 out.h5 --drop-removed`.

Requires PyQt5 (an optional dependency, like the run GUI):

    pip install PyQt5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from dataset import SPLIT_NAMES, set_display_channel, to_display_uint8, validate_h5


def read_rows_file(path, n: int) -> tuple[np.ndarray, dict[int, str]]:
    """--rows: a CSV with an h5_row column (plus optional difficulty) or
    one row number per line. Returns (rows in file order, {row: note})."""
    import csv
    text = Path(path).read_text(encoding="utf-8").strip().splitlines()
    rows, notes = [], {}
    if text and "h5_row" in text[0]:
        for rec in csv.DictReader(text):
            row = int(rec["h5_row"])
            rows.append(row)
            if rec.get("difficulty"):
                notes[row] = f"d {float(rec['difficulty']):.2f}"
    else:
        rows = [int(t.strip()) for t in text if t.strip() and not t.startswith("#")]
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and (rows.min() < 0 or rows.max() >= n):
        raise ValueError(f"--rows: row numbers must be in 0..{n - 1}")
    return rows, notes

REPO = Path(__file__).resolve().parent
INSTALL_HINT = "the curator needs PyQt5 - to run it: pip install PyQt5"
PAGE = 200
THUMB = 96

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:  # main() prints the hint; the module stays importable
    QtCore = QtGui = QtWidgets = None


if QtWidgets is not None:

    def _pixmap(arr: np.ndarray, side: int) -> QtGui.QPixmap:
        """HWC crop in any h5 image dtype (C = 1 or 3) -> QPixmap."""
        arr = np.ascontiguousarray(to_display_uint8(arr))
        h, w, c = arr.shape
        fmt = (QtGui.QImage.Format_Grayscale8 if c == 1
               else QtGui.QImage.Format_RGB888)
        img = QtGui.QImage(arr.data, w, h, w * c, fmt).copy()
        return QtGui.QPixmap.fromImage(img).scaled(
            side, side, QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)

    class Curator(QtWidgets.QMainWindow):
        def __init__(self, h5_path: str, display_channel=None, rows_file=None):
            super().__init__()
            self.h5_path = str(h5_path)
            validate_h5(self.h5_path)
            self.listed: np.ndarray | None = None  # --rows review list
            self.listed_notes: dict[int, str] = {}
            with h5py.File(self.h5_path, "r") as f:
                set_display_channel(display_channel, int(f["images"].shape[3]))
                self.classes = list(f["classes"].asstr()[:])
                self.labels = f["labels"][:].astype(np.int64)
                self.split = f["split"][:].astype(np.int64)
                self.n = len(self.labels)
                self.removed = (f["removed"][:].astype(bool)
                                if "removed" in f
                                else np.zeros(self.n, dtype=bool))
            self.saved_removed = self.removed.copy()
            if rows_file is not None:
                self.listed, self.listed_notes = read_rows_file(rows_file, self.n)
            self._h5 = h5py.File(self.h5_path, "r")
            self.page = 0

            # ---- left: split + class selection ----
            self.split_combo = QtWidgets.QComboBox()
            self.split_combo.addItems([SPLIT_NAMES[s] for s in sorted(SPLIT_NAMES)])
            self.split_combo.currentIndexChanged.connect(self._view_changed)
            self.class_list = QtWidgets.QListWidget()
            for _ in self.classes:
                self.class_list.addItem("")
            self.class_list.addItem("")  # trailing "removed" review row
            if self.listed is not None:
                self.class_list.addItem("")  # --rows review list
            self.class_list.setCurrentRow(0)
            self.class_list.currentRowChanged.connect(self._view_changed)

            left = QtWidgets.QVBoxLayout()
            left.addWidget(QtWidgets.QLabel("split"))
            left.addWidget(self.split_combo)
            left.addWidget(QtWidgets.QLabel("class"))
            left.addWidget(self.class_list, 1)
            left_w = QtWidgets.QWidget()
            left_w.setLayout(left)
            left_w.setMaximumWidth(260)

            # ---- right: pager + thumbnail grid ----
            self.grid = QtWidgets.QListWidget()
            self.grid.setViewMode(QtWidgets.QListWidget.IconMode)
            self.grid.setResizeMode(QtWidgets.QListWidget.Adjust)
            self.grid.setUniformItemSizes(True)
            self.grid.setIconSize(QtCore.QSize(THUMB, THUMB))
            self.grid.setGridSize(QtCore.QSize(THUMB + 22, THUMB + 40))
            self.grid.setSelectionMode(
                QtWidgets.QAbstractItemView.ExtendedSelection)
            self.grid.setSelectionRectVisible(True)  # drag a rubber band
            self.grid.itemDoubleClicked.connect(self._preview)
            self.grid.itemSelectionChanged.connect(self._selection_changed)

            self.page_label = QtWidgets.QLabel("")
            self.sel_label = QtWidgets.QLabel("")
            prev_btn = QtWidgets.QPushButton("< Prev")
            next_btn = QtWidgets.QPushButton("Next >")
            prev_btn.clicked.connect(lambda: self._page_step(-1))
            next_btn.clicked.connect(lambda: self._page_step(+1))
            select_page_btn = QtWidgets.QPushButton("Select page")
            select_page_btn.setToolTip(
                "select every snippet on this page (Ctrl+A); Ctrl/Shift-click "
                "or drag a rubber band for partial selections")
            select_page_btn.clicked.connect(self.grid.selectAll)
            self.remove_btn = QtWidgets.QPushButton("Remove selected (Del)")
            self.restore_btn = QtWidgets.QPushButton("Restore selected (Del)")
            self.save_btn = QtWidgets.QPushButton("Save")
            self.remove_btn.clicked.connect(self.remove_selected)
            self.restore_btn.clicked.connect(self.restore_selected)
            self.save_btn.clicked.connect(self.save)
            QtWidgets.QShortcut(QtGui.QKeySequence.Delete, self,
                                activated=self._delete_key)

            bar = QtWidgets.QHBoxLayout()
            bar.addWidget(prev_btn)
            bar.addWidget(self.page_label)
            bar.addWidget(next_btn)
            bar.addWidget(select_page_btn)
            bar.addWidget(self.sel_label)
            bar.addStretch(1)
            bar.addWidget(self.remove_btn)
            bar.addWidget(self.restore_btn)
            bar.addWidget(self.save_btn)

            right = QtWidgets.QVBoxLayout()
            right.addLayout(bar)
            right.addWidget(self.grid, 1)
            right_w = QtWidgets.QWidget()
            right_w.setLayout(right)

            splitter = QtWidgets.QSplitter()
            splitter.addWidget(left_w)
            splitter.addWidget(right_w)
            splitter.setStretchFactor(1, 1)
            self.setCentralWidget(splitter)
            self.resize(1080, 760)
            self._view_changed()

        # ---- view state ---------------------------------------------
        def in_removed_view(self) -> bool:
            return self.class_list.currentRow() == len(self.classes)

        def in_listed_view(self) -> bool:
            return (self.listed is not None
                    and self.class_list.currentRow() == len(self.classes) + 1)

        def current_rows(self) -> np.ndarray:
            """h5 row indices behind the current view: file order, or the
            --rows list's own order (its removed entries drop out)."""
            if self.in_removed_view():
                return np.flatnonzero(self.removed)
            if self.in_listed_view():
                return self.listed[~self.removed[self.listed]]
            cls = max(self.class_list.currentRow(), 0)
            sel = self.split_combo.currentIndex()
            return np.flatnonzero((self.split == sel)
                                  & (self.labels == cls) & ~self.removed)

        def _view_changed(self, *_):
            self.page = 0
            self._refresh()

        def _page_step(self, d: int):
            pages = max(1, -(-len(self.current_rows()) // PAGE))
            self.page = min(max(self.page + d, 0), pages - 1)
            self._populate()

        def _refresh(self):
            sel = self.split_combo.currentIndex()
            for c, name in enumerate(self.classes):
                shown = int(((self.split == sel) & (self.labels == c)
                             & ~self.removed).sum())
                gone = int(((self.labels == c) & self.removed).sum())
                text = f"{name}   ({shown}"
                text += f", {gone} removed)" if gone else ")"
                self.class_list.item(c).setText(text)
            self.class_list.item(len(self.classes)).setText(
                f"[removed - all splits]   ({int(self.removed.sum())})")
            if self.listed is not None:
                left_n = int((~self.removed[self.listed]).sum())
                self.class_list.item(len(self.classes) + 1).setText(
                    f"[--rows list]   ({left_n} of {len(self.listed)})")
            removed_view = self.in_removed_view()
            self.remove_btn.setVisible(not removed_view)
            self.restore_btn.setVisible(removed_view)
            self._populate()
            unsaved = int((self.removed != self.saved_removed).sum())
            star = " *" if unsaved else ""
            self.setWindowTitle(f"mic curate - {Path(self.h5_path).name}{star}")
            self.statusBar().showMessage(
                f"{self.h5_path}  ·  {self.n} snippets, "
                f"{int(self.removed.sum())} removed"
                + (f"  ·  {unsaved} unsaved change(s)" if unsaved else ""))

        def _populate(self):
            rows = self.current_rows()
            lo = self.page * PAGE
            chunk = rows[lo:lo + PAGE]
            self.grid.clear()
            for row in chunk:
                arr = self._h5["images"][int(row)]
                caption = f"#{row}"
                if self.in_removed_view() or self.in_listed_view():
                    caption += (f"  {self.classes[self.labels[row]]}"
                                f" · {SPLIT_NAMES[int(self.split[row])]}")
                if self.in_listed_view() and row in self.listed_notes:
                    caption += f" · {self.listed_notes[int(row)]}"
                item = QtWidgets.QListWidgetItem(
                    QtGui.QIcon(_pixmap(arr, THUMB)), caption)
                item.setData(QtCore.Qt.UserRole, int(row))
                self.grid.addItem(item)
            hi = lo + len(chunk)
            self.page_label.setText(
                f"{lo + 1}-{hi} of {len(rows)}" if len(rows) else "empty")

        def _selection_changed(self):
            k = len(self.grid.selectedItems())
            self.sel_label.setText(f"{k} selected" if k else "")
            verb = "Restore" if self.in_removed_view() else "Remove"
            btn = self.restore_btn if self.in_removed_view() else self.remove_btn
            btn.setText(f"{verb} {k} selected (Del)" if k
                        else f"{verb} selected (Del)")

        # ---- actions ------------------------------------------------
        def _selected_rows(self) -> list[int]:
            return [it.data(QtCore.Qt.UserRole)
                    for it in self.grid.selectedItems()]

        def remove_selected(self):
            rows = self._selected_rows()
            if rows and not self.in_removed_view():
                self.removed[rows] = True
                self._refresh()

        def restore_selected(self):
            rows = self._selected_rows()
            if rows and self.in_removed_view():
                self.removed[rows] = False
                self._refresh()

        def _delete_key(self):
            self.restore_selected() if self.in_removed_view() \
                else self.remove_selected()

        def save(self):
            self._h5.close()
            try:
                with h5py.File(self.h5_path, "r+") as f:
                    if "removed" in f:
                        f["removed"][:] = self.removed
                    else:
                        f.create_dataset("removed", data=self.removed)
            finally:
                self._h5 = h5py.File(self.h5_path, "r")
            self.saved_removed = self.removed.copy()
            self._refresh()
            self.statusBar().showMessage(
                f"saved: {int(self.removed.sum())} snippet(s) removed from "
                "training", 6000)

        def _preview(self, item):
            row = item.data(QtCore.Qt.UserRole)
            arr = self._h5["images"][int(row)]
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(
                f"#{row}  ·  {self.classes[self.labels[row]]}  ·  "
                f"{SPLIT_NAMES[int(self.split[row])]} split  ·  "
                f"{arr.shape[1]}x{arr.shape[0]} px")
            lab = QtWidgets.QLabel()
            lab.setPixmap(_pixmap(arr, 448))
            lay = QtWidgets.QVBoxLayout(dlg)
            lay.addWidget(lab)
            dlg.exec_()

        def closeEvent(self, event):
            if (self.removed != self.saved_removed).any():
                answer = QtWidgets.QMessageBox.question(
                    self, "Unsaved changes",
                    "Save removal changes to the h5 before closing?",
                    QtWidgets.QMessageBox.Save
                    | QtWidgets.QMessageBox.Discard
                    | QtWidgets.QMessageBox.Cancel)
                if answer == QtWidgets.QMessageBox.Cancel:
                    event.ignore()
                    return
                if answer == QtWidgets.QMessageBox.Save:
                    self.save()
            self._h5.close()
            event.accept()


def _self_test(win, out_dir: Path) -> None:
    """Offscreen render + real remove/save/restore round-trip against the
    given h5 (screenshots per view). Leaves the file with zero removals."""
    from dataset import SPLIT_TRAIN, H5SnippetDataset

    app = QtWidgets.QApplication.instance()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_train_len = len(H5SnippetDataset(win.h5_path, SPLIT_TRAIN))
    assert win.removed.sum() == 0

    win.split_combo.setCurrentIndex(SPLIT_TRAIN)
    win.class_list.setCurrentRow(0)
    app.processEvents()
    expect = int(((win.split == SPLIT_TRAIN) & (win.labels == 0)).sum())
    shown = len(win.current_rows())
    assert shown == expect and win.grid.count() == min(expect, PAGE), \
        (shown, expect, win.grid.count())
    win.grab().save(str(out_dir / "class_view.png"))

    win.grid.item(0).setSelected(True)
    win.grid.item(1).setSelected(True)
    win.remove_selected()
    assert win.removed.sum() == 2 and len(win.current_rows()) == expect - 2
    win.save()
    with h5py.File(win.h5_path, "r") as f:
        assert f["removed"][:].sum() == 2
    assert len(H5SnippetDataset(win.h5_path, SPLIT_TRAIN)) == base_train_len - 2
    assert validate_h5(win.h5_path)["counts"]["train"]["removed"] == 2

    win.class_list.setCurrentRow(len(win.classes))  # removed view
    app.processEvents()
    assert win.grid.count() == 2
    win.grab().save(str(out_dir / "removed_view.png"))
    win.grid.item(0).setSelected(True)
    win.restore_selected()
    win.save()
    with h5py.File(win.h5_path, "r") as f:
        assert f["removed"][:].sum() == 1
    win.grid.selectAll()  # "Select page" button routes here too
    app.processEvents()
    assert win.sel_label.text() == "1 selected", win.sel_label.text()
    assert "Restore 1 selected" in win.restore_btn.text()
    win.restore_selected()
    win.save()
    assert len(H5SnippetDataset(win.h5_path, SPLIT_TRAIN)) == base_train_len

    if win.listed is not None:  # --rows: the list view, in file order
        win.class_list.setCurrentRow(len(win.classes) + 1)
        app.processEvents()
        assert win.in_listed_view()
        assert list(win.current_rows()) == list(win.listed)
        assert win.grid.count() == min(len(win.listed), PAGE)
        first = win.grid.item(0).text()
        assert first.startswith(f"#{win.listed[0]}") and "·" in first, first
        win.grab().save(str(out_dir / "rows_view.png"))
        win.grid.item(0).setSelected(True)
        win.remove_selected()  # removal works from the list; it drops out
        assert list(win.current_rows()) == list(win.listed[1:])
        win.class_list.setCurrentRow(len(win.classes))  # restore from the removed view
        app.processEvents()
        win.grid.selectAll()
        win.restore_selected()
        assert win.removed.sum() == 0
        print(f"--rows list view: {len(win.listed)} rows in file order, "
              "remove/restore round-trip")
    print("CURATE SELF-TEST PASSED")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("h5", help="dataset .h5 file (h5_format.md)")
    ap.add_argument("--display-channel", type=int, default=None, metavar="C",
                    help="0-based band shown as a grayscale image (the second "
                         "channel is 1) - for bands not meant for the eye; "
                         "default: RGB as-is, 2-band files as false color")
    ap.add_argument("--rows", metavar="FILE", default=None,
                    help="review list: a CSV with an h5_row column (e.g. a "
                         "run's mined_hard_negatives.csv) or one row number "
                         "per line; adds a '[--rows list]' view in file "
                         "order, captioned with class, split and difficulty")
    ap.add_argument("--self-test", metavar="DIR", default=None,
                    help="render offscreen, run a remove/save/restore "
                         "round-trip on the given h5 (left clean), save "
                         "screenshots to DIR, and exit (smoke suite)")
    args = ap.parse_args(argv)
    if QtWidgets is None:
        raise SystemExit(INSTALL_HINT)
    if args.self_test is not None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    try:
        win = Curator(args.h5, display_channel=args.display_channel,
                      rows_file=args.rows)
    except (ValueError, OSError) as e:
        raise SystemExit(str(e))
    win.show()
    if args.self_test is not None:
        _self_test(win, Path(args.self_test))
        return
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
