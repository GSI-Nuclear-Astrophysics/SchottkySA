"""Positive harmonic coordinates entered as integers, decimals or fractions."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


def parse_harmonic(text: str) -> float:
    """Evaluate a number or integer fraction safely, with no expression eval."""
    compact = re.sub(r"\s+", "", str(text))
    try:
        value = float(Fraction(compact))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Enter a positive harmonic as an integer, decimal or fraction, e.g. 215/214.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Harmonic must be positive and finite; its denominator cannot be zero.")
    return value


class _HarmonicValidator(QtGui.QValidator):
    def validate(self, text: str, pos: int):
        try:
            value = parse_harmonic(text)
            if self.parent().minimum() <= value <= self.parent().maximum():
                return self.State.Acceptable, text, pos
        except ValueError:
            pass
        # Allow partially edited fractions/decimals, but never evaluate code.
        state = self.State.Intermediate if re.fullmatch(r"[\d\s+\-./eE]*", text) else self.State.Invalid
        return state, text, pos


class HarmonicInput(QtWidgets.QLineEdit):
    """Spin-box-compatible value access without rounding integer fractions.

    The entered expression is retained as text and the full float evaluation is
    returned by value(). The precision setting remains a GUI-API compatibility
    property; it never truncates the numerical harmonic ratio.
    """

    valueChanged = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum, self._maximum = 1e-12, 1e12
        self._decimals = 9
        self.setValidator(_HarmonicValidator(self))
        self.setText("1")
        self.textChanged.connect(self._edited)

    def setRange(self, minimum: float, maximum: float) -> None:
        self._minimum, self._maximum = float(minimum), float(maximum)

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum

    def setDecimals(self, decimals: int) -> None:
        self._decimals = int(decimals)

    def decimals(self) -> int:
        return self._decimals

    def value(self) -> float:
        value = parse_harmonic(self.text())
        if not self._minimum <= value <= self._maximum:
            raise ValueError(f"Harmonic must lie between {self._minimum:g} and {self._maximum:g}.")
        return value

    def setValue(self, value: float) -> None:
        self.setExpression(format(float(value), ".17g"))

    def setExpression(self, expression: str) -> None:
        value = parse_harmonic(expression)
        if not self._minimum <= value <= self._maximum:
            raise ValueError("Harmonic expression is outside the allowed range.")
        self.setText(str(expression).strip())

    def expression(self) -> str:
        return self.text().strip()

    def _edited(self, _text: str) -> None:
        valid = self.hasAcceptableInput()
        self.setStyleSheet("" if valid else "QLineEdit { border: 1px solid #d66; }")
        if valid:
            self.valueChanged.emit(self.value())
