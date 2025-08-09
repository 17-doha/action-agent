import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import imageio
from browser_use import Agent, BrowserSession
from browser_use.llm import ChatGoogle
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from typing import List

from langchain_core.language_models import BaseChatModel

import logging
import inspect
import pdb
from typing import Optional, Type, Callable, Dict, Any, Union, Awaitable, TypeVar
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.browser.context import BrowserContext
from browser_use.controller.registry.service import RegisteredAction, Registry
from browser_use.controller.service import Controller
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)

logger = logging.getLogger(__name__)
Context = TypeVar("Context")


def time_execution_sync(label):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

load_dotenv()

class Step(BaseModel):
    action: str = Field(description="The action taken in this step")
    description: str = Field(description="Description of what happened in this step")

class TestResult(BaseModel):
    steps: List[Step] = Field(description="List of steps taken by the agent")
    final_result: str = Field(description="The final result of the test")
    status: str = Field(description="Status: success or fail")

def ensure_dirs():
    os.makedirs("app_static/screenshots", exist_ok=True)
    print("[DEBUG] Created/ensured app_static/screenshots directory")
    os.makedirs("app_static/gifs", exist_ok=True)
    print("[DEBUG] Created/ensured app_static/gifs directory")
    os.makedirs("app_static/pdfs", exist_ok=True)
    print("[DEBUG] Created/ensured app_static/pdfs directory")

def generate_gif_from_images(image_paths, output_path):
    images = []
    target_size = None
    
    # First pass: determine the target size (use the first valid image)
    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                if target_size is None:
                    target_size = img.size
                break
            except Exception as e:
                print(f"[!] Failed to open {img_path}: {e}")
                continue
    
    if target_size is None:
        print("[!] No valid images found to determine target size.")
        return
    
    # Second pass: resize all images to the target size
    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                # Resize image to target size if it doesn't match
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    print(f"[→] Resized {img_path} from original size to {target_size}")
                images.append(img)
            except Exception as e:
                print(f"[!] Failed to process {img_path}: {e}")
                continue
    
    if len(images) >= 2:
        try:
            imageio.mimsave(output_path, images, fps=1)
            print(f"[✔] GIF generated: {output_path} (size: {os.path.getsize(output_path)} bytes)")
        except Exception as e:
            print(f"[ERROR] Failed to create GIF: {e}")
    else:
        print("[!] Not enough images to create a GIF.")




class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None,
                 ask_assistant_callback: Optional[Union[Callable[[str, BrowserContext], Dict[str, Any]], Callable[
                     [str, BrowserContext], Awaitable[Dict[str, Any]]]]] = None,
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()
        self.ask_assistant_callback = ask_assistant_callback
        self.mcp_client = None
        self.mcp_server_config = None

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action(
            "When executing tasks, prioritize autonomous completion. However, if you encounter a definitive blocker "
            "that prevents you from proceeding independently – such as needing credentials you don't possess, "
            "requiring subjective human judgment, needing a physical action performed, encountering complex CAPTCHAs, "
            "or facing limitations in your capabilities – you must request human assistance."
        )
        async def ask_for_assistant(query: str, browser: BrowserContext):
            if self.ask_assistant_callback:
                if inspect.iscoroutinefunction(self.ask_assistant_callback):
                    user_response = await self.ask_assistant_callback(query, browser)
                else:
                    user_response = self.ask_assistant_callback(query, browser)
                msg = f"AI ask: {query}. User response: {user_response['response']}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                return ActionResult(extracted_content="Human cannot help you. Please try another way.",
                                    include_in_memory=True)

        @self.registry.action(
            'Upload file to interactive element with file path ',
        )
        async def upload_file(index: int, path: str, browser: BrowserContext, available_file_paths: list[str]):
            if path not in available_file_paths:
                return ActionResult(error=f'File path {path} is not available')

            if not os.path.exists(path):
                return ActionResult(error=f'File {path} does not exist')

            dom_el = await browser.get_dom_element_by_index(index)

            file_upload_dom_el = dom_el.get_file_upload_element()

            if file_upload_dom_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            file_upload_el = await browser.get_locate_element(file_upload_dom_el)

            if file_upload_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            try:
                await file_upload_el.set_input_files(path)
                msg = f'Successfully uploaded file to index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f'Failed to upload file to index {index}: {str(e)}'
                logger.info(msg)
                return ActionResult(error=msg)

    @time_execution_sync('--act')
    async def act(self, action: ActionModel,
                  browser_context: Optional[BrowserContext] = None,
                  page_extraction_llm: Optional[BaseChatModel] = None,
                  sensitive_data: Optional[Dict[str, str]] = None,
                  available_file_paths: Optional[list[str]] = None,
                  context: Context | None = None) -> ActionResult:
        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    result = await self.registry.execute_action(
                        action_name,
                        params,
                        browser=browser_context,
                        page_extraction_llm=page_extraction_llm,
                        sensitive_data=sensitive_data,
                        available_file_paths=available_file_paths,
                        context=context,
                    )

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e
    # async def setup_mcp_client(self, mcp_server_config: Optional[Dict[str, Any]] = None):
    #     self.mcp_server_config = mcp_server_config
    #     if self.mcp_server_config:
    #         self.mcp_client = await setup_mcp_client_and_tools(self.mcp_server_config)
    #         self.register_mcp_tools()

    # def register_mcp_tools(self):
    #     if self.mcp_client:
    #         for server_name in self.mcp_client.server_name_to_tools:
    #             for tool in self.mcp_client.server_name_to_tools[server_name]:
    #                 tool_name = f"mcp.{server_name}.{tool.name}"
    #                 self.registry.registry.actions[tool_name] = RegisteredAction(
    #                     name=tool_name,
    #                     description=tool.description,
    #                     function=tool,
    #                     param_model=create_tool_param_model(tool),
    #                 )
    #                 logger.info(f"Add mcp tool: {tool_name}")
    #             logger.debug(f"Registered {len(self.mcp_client.server_name_to_tools[server_name])} mcp tools for {server_name}")
    #     else:
    #         logger.warning("MCP client not started.")

    # async def close_mcp_client(self):
    #     if self.mcp_client:
    #         await self.mcp_client.__aexit__(None, None, None)

async def run_agent_task(prompt: str):
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_dir = os.path.join("app_static/screenshots", f"run_{timestamp}")
    os.makedirs(screenshot_dir, exist_ok=True)
    gif_path = f"app_static/gifs/test_{timestamp}.gif"
    pdf_path = f"app_static/pdfs/test_{timestamp}.pdf"
    screenshots = []

    stop_capture_flag = [False]

    async with async_playwright() as playwright:
        try:
            print("[DEBUG] Starting persistent browser launch...")
            browser = await playwright.chromium.launch_persistent_context(
                user_data_dir="user_data",  
                headless=False,            
                args=["--start-maximized"]
            )
            print("[DEBUG] Persistent browser launched successfully")
        except Exception as e:
            print(f"[ERROR] Browser launch failed: {str(e)}")
            raise

        page = await browser.new_page()
        browser_session = BrowserSession(page=page)

        async def capture_loop():
            frame = 0
            while not stop_capture_flag[0]:
                try:
                    screenshot_path = os.path.join(screenshot_dir, f"frame_{frame}.png")
                    await page.screenshot(path=screenshot_path)
                    screenshots.append(screenshot_path)
                    print(f"[+] Captured screenshot: {screenshot_path} (total: {len(screenshots)})")
                    frame += 1
                except Exception as e:
                    print(f"[ERROR] Failed to capture screenshot: {e}")
                await asyncio.sleep(1)

        capture_task = asyncio.create_task(capture_loop())

        controller = CustomController(output_model=TestResult)

        agent = Agent(
            task=prompt,
            llm=ChatGoogle(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY")),
            browser_session=browser_session,
            controller=controller,
            max_steps=100
        )

        try:
            history = await agent.run()
            final_output = history.final_result()
            if final_output:
                structured_result = TestResult.model_validate_json(final_output)
                status = structured_result.status
            else:
                structured_result = TestResult(steps=[], final_result="No result", status="fail")
                status = "fail"
        finally:
            stop_capture_flag[0] = True
            await capture_task
            await browser.close()

    generate_gif_from_images(screenshots, gif_path)

    return {
        "text": structured_result.final_result,
        "gif_path": gif_path if os.path.exists(gif_path) else None,
        "status": status,
        "screenshots": screenshots,
        "timestamp": timestamp
    }


def run_prompt(prompt: str):
    return asyncio.run(run_agent_task(prompt))
