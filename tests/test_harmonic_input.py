"""Harmonic parsing is safe and never rounds the user's fraction."""

import pytest

pytest.importorskip("PySide6")
from pyqtgraph.Qt import QtWidgets

from ssa.gui.harmonic_input import HarmonicInput, parse_harmonic


@pytest.mark.parametrize(
    "text,value",
    [("214", 214), ("214/214", 1), ("215/214", 215 / 214), (" 215 / 214 ", 215 / 214), ("1.25", 1.25), ("1e2", 100)],
)
def test_fraction_and_decimal_parsing(text, value):
    assert parse_harmonic(text) == value


@pytest.mark.parametrize("text", ["0", "-1", "1/0", "abc", "1/2/3", "float(215)/214", '__import__("os")', "nan", "inf"])
def test_invalid_inputs(text):
    with pytest.raises(ValueError):
        parse_harmonic(text)


def test_widget_keeps_expression_and_full_ratio():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = HarmonicInput()
    changes = []
    widget.valueChanged.connect(changes.append)
    widget.setDecimals(3)
    widget.setExpression("215/214")
    assert widget.expression() == "215/214"
    assert widget.value() == 215 / 214
    assert widget.value() != round(215 / 214, 3)
    assert changes[-1] == 215 / 214
    widget.setText("215/")
    assert not widget.hasAcceptableInput()
    with pytest.raises(ValueError):
        widget.value()
    assert len(changes) == 1
    widget.setExpression("214")
    assert widget.value() == 214
    app.processEvents()
    widget.close()
