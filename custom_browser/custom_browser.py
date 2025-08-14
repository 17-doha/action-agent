import os
import logging
import socket
from typing import Optional, Any

# ---- browser-use 0.5.x stable bits ----
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig

# Playwright async types
from playwright.async_api import Playwright, Browser as PlaywrightBrowser

# ---- utilities (screen resolution) across 0.5.x layouts ----
try:
    # Primary path in your tree: browser_use/browser/utils/screen_resolution.py
    from browser_use.browser.utils.screen_resolution import (
        get_screen_resolution,
        get_window_adjustments,
    )
except Exception:
    # Older/simpler builds: browser_use/utils/screen_resolution.py
    try:
        from browser_use.utils.screen_resolution import (
            get_screen_resolution,
            get_window_adjustments,
        )
    except Exception:
        # Some builds used singular name; last-ditch aliases
        try:
            from browser_use.browser.utils.screen_resolution import (
                get_screen_resolution as _gsr,
                get_window_adjustment as _gwa,
            )
            get_screen_resolution = _gsr
            def get_window_adjustments(*args, **kwargs):
                return _gwa(*args, **kwargs)
        except Exception:
            # Safe fallbacks if nothing is present (won't crash at import-time)
            def get_screen_resolution():
                return (1280, 800)
            def get_window_adjustments(*args, **kwargs):
                return (0, 0)

# ---- chrome constants across 0.5.x layouts ----
CHROME_ARGS = []
CHROME_DETERMINISTIC_RENDERING_ARGS = []
CHROME_DISABLE_SECURITY_ARGS = []
CHROME_DOCKER_ARGS = []
CHROME_HEADLESS_ARGS = []

# Try deepest path first (matches your tree: browser_use/browser/browser/chrome.py)
try:
    from browser_use.browser.browser.chrome import (  # type: ignore
        CHROME_ARGS as _CARGS,
        CHROME_DETERMINISTIC_RENDERING_ARGS as _CDET,
        CHROME_DISABLE_SECURITY_ARGS as _CDIS,
        CHROME_DOCKER_ARGS as _CDOCK,
        CHROME_HEADLESS_ARGS as _CHEAD,
    )
    CHROME_ARGS = _CARGS
    CHROME_DETERMINISTIC_RENDERING_ARGS = _CDET
    CHROME_DISABLE_SECURITY_ARGS = _CDIS
    CHROME_DOCKER_ARGS = _CDOCK
    CHROME_HEADLESS_ARGS = _CHEAD
except Exception:
    # Next likely path used by some 0.5.x wheels
    try:
        from browser_use.browser.chrome import (  # type: ignore
            CHROME_ARGS as _CARGS2,
            CHROME_DETERMINISTIC_RENDERING_ARGS as _CDET2,
            CHROME_DISABLE_SECURITY_ARGS as _CDIS2,
            CHROME_DOCKER_ARGS as _CDOCK2,
            CHROME_HEADLESS_ARGS as _CHEAD2,
        )
        CHROME_ARGS = _CARGS2
        CHROME_DETERMINISTIC_RENDERING_ARGS = _CDET2
        CHROME_DISABLE_SECURITY_ARGS = _CDIS2
        CHROME_DOCKER_ARGS = _CDOCK2
        CHROME_HEADLESS_ARGS = _CHEAD2
    except Exception:
        # Legacy path that some users reported (rare)
        try:
            from browser_use.chrome import (  # type: ignore
                CHROME_ARGS as _CARGS3,
                CHROME_DETERMINISTIC_RENDERING_ARGS as _CDET3,
                CHROME_DISABLE_SECURITY_ARGS as _CDIS3,
                CHROME_DOCKER_ARGS as _CDOCK3,
                CHROME_HEADLESS_ARGS as _CHEAD3,
            )
            CHROME_ARGS = _CARGS3
            CHROME_DETERMINISTIC_RENDERING_ARGS = _CDET3
            CHROME_DISABLE_SECURITY_ARGS = _CDIS3
            CHROME_DOCKER_ARGS = _CDOCK3
            CHROME_HEADLESS_ARGS = _CHEAD3
        except Exception:
            # Leave empty lists; your _setup_builtin_browser() handles it
            pass

from .custom_context import CustomBrowserContext

logger = logging.getLogger(__name__)


def _to_dict(x: Any) -> dict:
    """Accept pydantic (v2/v1), dicts, SimpleNamespace, or plain objects."""
    if x is None:
        return {}
    if hasattr(x, "model_dump") and callable(getattr(x, "model_dump")):
        try:
            return x.model_dump()
        except Exception:
            pass
    if hasattr(x, "dict") and callable(getattr(x, "dict")):
        try:
            return x.dict()
        except Exception:
            pass
    if isinstance(x, dict):
        return x
    try:
        return vars(x)
    except TypeError:
        return {}


class CustomBrowser(Browser):
    async def new_context(
        self,
        config: Optional[BrowserContextConfig] = None
    ) -> CustomBrowserContext:
        """Create a browser-use BrowserContext using your custom context class."""
        browser_config = _to_dict(getattr(self, "config", None))
        context_config = _to_dict(config)
        merged_config = {**browser_config, **context_config}

        ctx = CustomBrowserContext.model_construct(
            browser=self,
            config=BrowserContextConfig(**merged_config),
            state=None,
        )
        if not hasattr(ctx, "browser_pid"):
            setattr(ctx, "browser_pid", None)

        await ctx.setup()
        return ctx


    async def _setup_builtin_browser(self, playwright: Playwright) -> PlaywrightBrowser:
        """Launch a Playwright browser with robust args; no IN_DOCKER usage."""
        assert getattr(self.config, "browser_binary_path", None) is None, (
            "browser_binary_path should be None when using builtin browsers"
        )

        # Window size & position
        if (
            not getattr(self.config, "headless", False)
            and hasattr(self.config, "new_context_config")
            and hasattr(self.config.new_context_config, "window_width")
            and hasattr(self.config.new_context_config, "window_height")
        ):
            screen_size = {
                "width": self.config.new_context_config.window_width,
                "height": self.config.new_context_config.window_height,
            }
            offset_x, offset_y = get_window_adjustments()
        elif getattr(self.config, "headless", False):
            screen_size = {"width": 1920, "height": 1080}
            offset_x, offset_y = 0, 0
        else:
            w, h = get_screen_resolution()
            screen_size = {"width": w, "height": h}
            offset_x, offset_y = get_window_adjustments()

        # IMPORTANT: use a LIST to preserve arg order
        chrome_args = [
            f'--remote-debugging-port={getattr(self.config, "chrome_remote_debugging_port", 9222)}',
            *CHROME_ARGS,
            *(CHROME_HEADLESS_ARGS if getattr(self.config, "headless", False) else []),
            *(CHROME_DISABLE_SECURITY_ARGS if getattr(self.config, "disable_security", False) else []),
            *(CHROME_DETERMINISTIC_RENDERING_ARGS if getattr(self.config, "deterministic_rendering", False) else []),
            f'--window-position={offset_x},{offset_y}',
            f'--window-size={screen_size["width"]},{screen_size["height"]}',
            *getattr(self.config, "extra_browser_args", []),
        ]

        # If remote-debugging-port is taken, remove that flag to avoid conflicts
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                port = int(getattr(self.config, "chrome_remote_debugging_port", 9222))
                if s.connect_ex(("localhost", port)) == 0:
                    port_flag = f'--remote-debugging-port={port}'
                    try:
                        chrome_args.remove(port_flag)
                    except ValueError:
                        pass
        except Exception:
            # Non-fatal; continue without removing the flag
            pass

        browser_class_name = getattr(self.config, "browser_class", "chromium")
        browser_class = getattr(playwright, browser_class_name)

        args_map = {
            "chromium": chrome_args,
            "firefox": [
                "-no-remote",
                *getattr(self.config, "extra_browser_args", []),
            ],
            "webkit": [
                "--no-startup-window",
                *getattr(self.config, "extra_browser_args", []),
            ],
        }

        proxy_obj = getattr(self.config, "proxy", None)
        proxy = _to_dict(proxy_obj) if proxy_obj is not None else None

        browser = await browser_class.launch(
            channel="chromium",                   # https://github.com/microsoft/playwright/issues/33566
            headless=bool(getattr(self.config, "headless", False)),
            args=args_map[browser_class_name],
            proxy=proxy,
            handle_sigterm=False,
            handle_sigint=False,
        )
        return browser
