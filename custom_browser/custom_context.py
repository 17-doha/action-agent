import logging
from enum import Enum
from typing import Optional

from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from playwright.async_api import (
    async_playwright,
    Browser as PlaywrightBrowser,
    BrowserContext as PlaywrightBrowserContext,
    Page as PlaywrightPage,
)

try:
    from browser_use.browser.context import BrowserContextState 
except Exception:
    class BrowserContextState(Enum):
        INIT = "INIT"
        READY = "READY"
        CLOSED = "CLOSED"

logger = logging.getLogger(__name__)


class CustomBrowserContext(BrowserContext):
    """
    A BrowserUse-compatible context that guarantees a Playwright browser/context/page
    so the rest of your pipeline can use it without falling back.
    """

    # These attrs are referenced by BrowserUse internals; define them for safety.
    playwright_browser: Optional[PlaywrightBrowser] = None
    playwright_context: Optional[PlaywrightBrowserContext] = None
    pages: list[PlaywrightPage] = []
    page: Optional[PlaywrightPage] = None
    _playwright = None

    def __init__(
        self,
        browser: "Browser",
        config: Optional[BrowserContextConfig] = None,
        state: Optional[BrowserContextState] = None,
    ):
        super().__init__(browser=browser, config=config, state=state)

    async def setup(self) -> "CustomBrowserContext":
        """
        Ensure there's an active Playwright browser/context and at least one page.
        This prevents: 'Could not obtain a Playwright Page from CustomBrowserContext'.
        """
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            logger.debug("[CustomBrowserContext] Started Playwright.")

        if self.playwright_browser is None:
            self.playwright_browser = await self.browser._setup_builtin_browser(self._playwright)
            logger.debug("[CustomBrowserContext] Launched Playwright browser via CustomBrowser.")

        if self.playwright_context is None:
            self.playwright_context = await self.playwright_browser.new_context()
            logger.debug("[CustomBrowserContext] Created Playwright context.")

        if not getattr(self, "pages", None):
            self.pages = []

        if not self.pages:
            self.page = await self.playwright_context.new_page()
            self.pages.append(self.page)
            logger.debug("[CustomBrowserContext] Opened initial page and stored as self.page.")
        else:
            self.page = self.pages[0]

        if not hasattr(self, "browser_pid"):
            setattr(self, "browser_pid", None)

        return self

    async def new_page(self) -> PlaywrightPage:
        """Create and return a new page within this context."""
        if self.playwright_context is None:
            await self.setup()
        p = await self.playwright_context.new_page()
        self.pages.append(p)
        self.page = p
        return p

    async def close(self) -> None:
        """Gracefully close page(s), context, browser, and stop Playwright."""
        try:
            if getattr(self, "pages", None):
                for p in list(self.pages):
                    try:
                        await p.close()
                    except Exception:
                        pass
                self.pages.clear()

            if self.playwright_context:
                try:
                    await self.playwright_context.close()
                except Exception:
                    pass
                self.playwright_context = None

            if self.playwright_browser:
                try:
                    await self.playwright_browser.close()
                except Exception:
                    pass
                self.playwright_browser = None

        finally:
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

            logger.debug("[CustomBrowserContext] Closed all resources.")
