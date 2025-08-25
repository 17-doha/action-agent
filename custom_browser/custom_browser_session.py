from pydantic import PrivateAttr
from browser_use.browser.session import BrowserSession as BaseBrowserSession

class CustomBrowserSession(BaseBrowserSession):
    _browser_pid: int = PrivateAttr(default=None)