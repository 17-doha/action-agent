import asyncio
import base64
import json
from typing import Generic, Optional, TypeVar
import logging

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from .utilities.config import config
from tools.utilities.llm import LLM
from tools.utilities.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    cleanup_in_progress: bool = Field(default=False, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.cleanup_in_progress:
            raise Exception("Browser is being cleaned up")
            
        if self.browser is None:
            browser_config_kwargs = {
                "headless": True,  # Always use headless for stability
                "disable_security": True,
                "extra_chromium_args": [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--memory-pressure-off",
                    "--max_old_space_size=1024",
                    "--single-process",
                    "--disable-web-security",
                    "--disable-extensions",
                    "--disable-plugins",
                    "--disable-images",
                    "--no-first-run",
                    "--disable-default-apps",
                ]
            }

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # Handle proxy settings
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security", 
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()

            # If there is context config in the config, use it
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute a specified browser action with proper error handling."""
        async with self.lock:
            try:
                if self.cleanup_in_progress:
                    return ToolResult(error="Browser tool is being cleaned up")

                # Add timeout for all operations
                context = await asyncio.wait_for(
                    self._ensure_browser_initialized(), 
                    timeout=30.0
                )

                # Get max content length from config
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()
                    try:
                        await asyncio.wait_for(page.goto(url), timeout=30.0)
                        await asyncio.wait_for(page.wait_for_load_state("domcontentloaded"), timeout=15.0)
                        return ToolResult(output=f"Navigated to {url}")
                    except asyncio.TimeoutError:
                        return ToolResult(error=f"Navigation to {url} timed out")

                elif action == "go_back":
                    try:
                        await asyncio.wait_for(context.go_back(), timeout=10.0)
                        return ToolResult(output="Navigated back")
                    except asyncio.TimeoutError:
                        return ToolResult(error="Go back operation timed out")

                elif action == "refresh":
                    try:
                        await asyncio.wait_for(context.refresh_page(), timeout=15.0)
                        return ToolResult(output="Refreshed current page")
                    except asyncio.TimeoutError:
                        return ToolResult(error="Page refresh timed out")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    # Navigate to Google and perform search
                    page = await context.get_current_page()
                    try:
                        await asyncio.wait_for(page.goto("https://www.google.com"), timeout=30.0)
                        await asyncio.wait_for(page.wait_for_load_state("domcontentloaded"), timeout=15.0)
                        
                        # Wait for search box and enter query
                        search_box = await page.wait_for_selector('input[name="q"]', timeout=10000)
                        await search_box.fill(query)
                        await search_box.press('Enter')
                        await page.wait_for_load_state("domcontentloaded", timeout=15000)
                        
                        return ToolResult(output=f"Performed web search for: {query}")
                    except asyncio.TimeoutError:
                        return ToolResult(error="Web search timed out")

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    try:
                        element = await asyncio.wait_for(
                            context.get_dom_element_by_index(index), 
                            timeout=10.0
                        )
                        if not element:
                            return ToolResult(error=f"Element with index {index} not found")
                        
                        download_path = await asyncio.wait_for(
                            context._click_element_node(element), 
                            timeout=10.0
                        )
                        output = f"Clicked element at index {index}"
                        if download_path:
                            output += f" - Downloaded file to {download_path}"
                        return ToolResult(output=output)
                    except asyncio.TimeoutError:
                        return ToolResult(error=f"Click element {index} timed out")

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    try:
                        element = await asyncio.wait_for(
                            context.get_dom_element_by_index(index), 
                            timeout=10.0
                        )
                        if not element:
                            return ToolResult(error=f"Element with index {index} not found")
                        
                        await asyncio.wait_for(
                            context._input_text_element_node(element, text), 
                            timeout=10.0
                        )
                        return ToolResult(
                            output=f"Input '{text}' into element at index {index}"
                        )
                    except asyncio.TimeoutError:
                        return ToolResult(error=f"Input text to element {index} timed out")

                # Scroll actions
                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size.get("height", 800)
                    )
                    try:
                        await asyncio.wait_for(
                            context.execute_javascript(
                                f"window.scrollBy(0, {direction * amount});"
                            ),
                            timeout=5.0
                        )
                        return ToolResult(
                            output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                        )
                    except asyncio.TimeoutError:
                        return ToolResult(error="Scroll operation timed out")

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await asyncio.wait_for(
                            locator.scroll_into_view_if_needed(), 
                            timeout=10.0
                        )
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except asyncio.TimeoutError:
                        return ToolResult(error=f"Scroll to text '{text}' timed out")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = min(seconds if seconds is not None else 3, 10)  # Max 10 seconds
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except asyncio.TimeoutError:
                return ToolResult(error=f"Browser action '{action}' timed out")
            except Exception as e:
                logger.error(f"Browser action '{action}' failed: {str(e)}")
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """Get the current browser state as a ToolResult with timeout protection."""
        try:
            if self.cleanup_in_progress:
                return ToolResult(error="Browser tool is being cleaned up")

            # Use provided context or fall back to self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await asyncio.wait_for(ctx.get_state(), timeout=15.0)

            # Create a viewport_info dictionary if it doesn't exist
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # Take a lightweight screenshot
            page = await ctx.get_current_page()
            await page.bring_to_front()

            screenshot = await asyncio.wait_for(
                page.screenshot(
                    full_page=False,  # Only viewport to save memory
                    animations="disabled", 
                    type="jpeg", 
                    quality=70  # Reduced quality
                ), 
                timeout=10.0
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed.",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()[:1000]  # Truncate to save memory
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=2, ensure_ascii=False),
                base64_image=screenshot,
            )
        except asyncio.TimeoutError:
            return ToolResult(error="Get browser state timed out")
        except Exception as e:
            logger.error(f"Failed to get browser state: {str(e)}")
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources with proper error handling."""
        if self.cleanup_in_progress:
            return
            
        self.cleanup_in_progress = True
        
        async with self.lock:
            try:
                # Clean up context first
                if self.context is not None:
                    try:
                        # Close all pages first
                        if hasattr(self.context, 'browser_session') and self.context.browser_session:
                            pages = getattr(self.context.browser_session, 'pages', [])
                            for page in pages:
                                try:
                                    await asyncio.wait_for(page.close(), timeout=5.0)
                                except Exception as e:
                                    logger.warning(f"Error closing page: {e}")
                        
                        await asyncio.wait_for(self.context.close(), timeout=10.0)
                        logger.info("Browser context closed successfully")
                    except asyncio.TimeoutError:
                        logger.warning("Browser context close timed out")
                    except Exception as e:
                        logger.warning(f"Error closing browser context: {e}")
                    finally:
                        self.context = None
                        self.dom_service = None

                # Clean up browser
                if self.browser is not None:
                    try:
                        await asyncio.wait_for(self.browser.close(), timeout=10.0)
                        logger.info("Browser closed successfully")
                    except asyncio.TimeoutError:
                        logger.warning("Browser close timed out")
                    except Exception as e:
                        logger.warning(f"Error closing browser: {e}")
                    finally:
                        self.browser = None

            except Exception as e:
                logger.error(f"Error during browser cleanup: {e}")
            finally:
                self.cleanup_in_progress = False

    def __del__(self):
        """Ensure cleanup when object is destroyed - with safe async handling."""
        if not self.cleanup_in_progress and (self.browser is not None or self.context is not None):
            try:
                # Try to get current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    loop.create_task(self.cleanup())
                else:
                    # If no loop is running, run cleanup in new loop
                    loop.run_until_complete(self.cleanup())
            except RuntimeError:
                # If we can't access the event loop, create a new one
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.cleanup())
                    loop.close()
                except Exception as e:
                    logger.warning(f"Failed to cleanup browser in __del__: {e}")

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool