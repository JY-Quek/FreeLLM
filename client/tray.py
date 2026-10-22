"""System tray controller for the FreeLLM client."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable


class TrayController:
    STATUS_TEXT = {
        "starting": "Service is starting…",
        "switching": "Switching inference engine…",
        "running": "Service is running",
        "error": "Service error",
    }
    STATUS_COLOR = {
        "starting": (240, 190, 0),
        "switching": (240, 190, 0),
        "running": (40, 170, 70),
        "error": (200, 50, 50),
    }

    def __init__(
        self,
        *,
        logo_file: Path,
        log_file: Path,
        logger,
        installed_engines: Callable[[], list[str]],
        engine_label: Callable[[str | None], str],
    ) -> None:
        self._logo_file = logo_file
        self._log_file = log_file
        self._log = logger
        self._installed_engines = installed_engines
        self._engine_label = engine_label
        self.status = "starting"
        self.error_msg = ""
        self._icon = None
        self._on_exit = None
        self._on_engine_change = None
        self._engine_getter = None

    def set_status(self, status: str, error_msg: str = "") -> None:
        if status not in self.STATUS_TEXT:
            return
        self.status = status
        self.error_msg = error_msg
        icon = self._icon
        if icon is not None:
            try:
                icon.icon = self._make_image()
                icon.title = self._tooltip()
                icon.update_menu()
            except Exception as exc:  # pragma: no cover - defensive
                self._log.debug("Tray update failed: %s", exc)

    def run(self, on_exit, on_engine_change=None, engine_getter=None) -> None:
        try:
            import pystray  # noqa: WPS433 (local import, optional dep)
        except Exception as exc:
            self._log.warning("pystray not available, running without tray icon: %s", exc)
            return

        self._on_exit = on_exit
        self._on_engine_change = on_engine_change
        self._engine_getter = engine_getter
        menu = pystray.Menu(
            pystray.MenuItem(lambda item: self._status_line(), None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("View logs", self._handle_view_logs),
            pystray.MenuItem("Settings", self._build_settings_menu(pystray)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self._handle_exit),
        )
        self._icon = pystray.Icon("FreeLLM", self._make_image(), self._tooltip(), menu)
        try:
            self._icon.run()
        finally:
            self._icon = None

    def _build_settings_menu(self, pystray):
        engines = self._installed_engines()
        if engines:
            engine_items = [
                pystray.MenuItem(
                    self._engine_label(name),
                    self._make_engine_action(name),
                    checked=self._make_engine_checker(name),
                    radio=True,
                )
                for name in engines
            ]
        else:
            engine_items = [
                pystray.MenuItem("No supported engines installed", None, enabled=False),
            ]
        return pystray.Menu(
            pystray.MenuItem("Inference engine", pystray.Menu(*engine_items)),
        )

    def _make_engine_action(self, engine: str):
        def action(icon, item):
            self._handle_engine_change(engine)

        return action

    def _make_engine_checker(self, engine: str):
        def checked(item):
            return self._is_active_engine(engine)

        return checked

    def _is_active_engine(self, engine: str) -> bool:
        if self._engine_getter is None:
            return False
        return self._engine_getter() == engine

    def _handle_engine_change(self, engine: str) -> None:
        if self._is_active_engine(engine):
            return
        if self._on_engine_change is None:
            return
        try:
            self._on_engine_change(engine)
        except Exception as exc:
            self._log.error("Engine change to %s failed: %s", engine, exc)

    def _status_line(self) -> str:
        base = self.STATUS_TEXT.get(self.status, "Service: unknown")
        if self.status == "error" and self.error_msg:
            return f"{base}: {self.error_msg}"
        return base

    def _tooltip(self) -> str:
        return f"FreeLLM Client — {self._status_line()}"

    def _make_image(self):
        from PIL import Image, ImageDraw

        color = self.STATUS_COLOR.get(self.status, (128, 128, 128))

        try:
            img = Image.open(self._logo_file).convert("RGBA").resize((64, 64), Image.LANCZOS)
        except Exception as exc:  # pragma: no cover - defensive
            self._log.debug("Could not load tray logo %s: %s", self._logo_file, exc)
            img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse((4, 4, 60, 60), fill=color, outline=(30, 30, 30, 255), width=2)
            return img

        draw = ImageDraw.Draw(img)
        draw.ellipse(
            (40, 40, 62, 62),
            fill=color + (255,),
            outline=(255, 255, 255, 255),
            width=3,
        )
        return img

    def _handle_view_logs(self, icon, item) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(str(self._log_file))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self._log_file)])
            else:
                subprocess.Popen(["xdg-open", str(self._log_file)])
        except Exception as exc:
            self._log.error("Could not open log file %s: %s", self._log_file, exc)

    def _handle_exit(self, icon, item) -> None:
        self._log.info("Tray exit requested")
        try:
            icon.visible = False
            icon.stop()
        except Exception:
            pass
        if self._on_exit:
            try:
                self._on_exit()
            except Exception as exc:
                self._log.error("Exit callback failed: %s", exc)