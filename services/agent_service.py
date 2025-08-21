import asyncio
import os
import json
from datetime import datetime
from types import SimpleNamespace

from browser_use import Agent, BrowserSession
from browser_use.llm import ChatGoogle
from playwright.async_api import async_playwright

from config.settings import GOOGLE_API_KEY, MCP_CONFIG_PATH
from models.result_models import TestResult
from utils.file_utils import ensure_dirs, generate_gif_from_images
from controllers.custom_controller import CustomController

async def run_agent_task(prompt: str):
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_dir = os.path.join("app_static/screenshots", f"run_{timestamp}")
    os.makedirs(screenshot_dir, exist_ok=True)
    gif_path = f"app_static/gifs/test_{timestamp}.gif"

    screenshots = []

    # Load MCP config if exists
    mcp_server_config = None
    if os.path.exists(MCP_CONFIG_PATH):
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                mcp_server_config = json.loads(content)

    controller = CustomController(output_model=TestResult)

    if mcp_server_config:
        await controller.setup_mcp_client(mcp_server_config)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch_persistent_context(
            user_data_dir="user_data",
            headless=False,
            args=["--start-maximized"]
        )
        page = await browser.new_page()
        browser_session = BrowserSession(page=page)

        agent = Agent(
            task=prompt,
            llm=ChatGoogle(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
            browser_session=browser_session,
            controller=controller,
            max_steps=100
        )

        history = await agent.run()
        final_output = history.final_result()
        structured_result = TestResult.model_validate_json(final_output) if final_output else TestResult(steps=[], final_result="No result", status="fail")

        await browser.close()

    generate_gif_from_images(screenshots, gif_path)

    return {
        "text": structured_result.final_result,
        "gif_path": gif_path if os.path.exists(gif_path) else None,
        "status": structured_result.status,
        "screenshots": screenshots,
        "timestamp": timestamp
    }

def run_prompt(prompt: str):
    return asyncio.run(run_agent_task(prompt))
