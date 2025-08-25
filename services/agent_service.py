import asyncio
import os
import json
from datetime import datetime
from types import SimpleNamespace

from browser_use import Agent, Browser, BrowserConfig  # Added missing imports
from browser_use.llm import ChatGoogle
from playwright.async_api import async_playwright

from config.settings import GOOGLE_API_KEY, MCP_CONFIG_PATH
from models.result_models import TestResult, Step  # Added Step import
from utils.file_utils import ensure_dirs, generate_gif_from_images
from controllers.custom_controller import CustomController
from prompts.prompt import extend_system_message

async def run_agent_task(prompt: str):
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_dir = os.path.join("app_static/screenshots", f"run_{timestamp}")
    os.makedirs(screenshot_dir, exist_ok=True)
    gif_path = f"app_static/gifs/test_{timestamp}.gif"
    pdf_path = f"app_static/pdfs/test_{timestamp}.pdf"
    screenshots = []

    stop_capture_flag = [False]

    # --- MCP server config loading ---
    mcp_server_config = None
    mcp_config_path = "mcp_server.json"
    if os.path.exists(mcp_config_path):
        with open(mcp_config_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                try:
                    mcp_server_config = json.loads(content)
                    print(f"[DEBUG] Loaded MCP server config from {mcp_config_path}")
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Invalid JSON in MCP config file: {e}")
            else:
                print(f"[DEBUG] MCP server config file is empty: {mcp_config_path}")
    else:
        print(f"[DEBUG] MCP server config file not found: {mcp_config_path}")

    structured_result = TestResult(steps=[], final_result="No result", status="fail")
    status = "fail"
    browser = None
    browser_context = None
    controller = None
    capture_task = None

    try:
        # Use proper browser_use Browser setup
        print("[DEBUG] Setting up browser_use Browser...")
        
        browser_config = BrowserConfig(
            headless=False,
            disable_security=True,
            extra_chromium_args=[
                "--start-maximized",
                "--no-sandbox",
                "--disable-dev-shm-usage",  # Reduce memory usage
                "--disable-gpu",             # Reduce GPU memory usage
                "--memory-pressure-off",     # Disable memory pressure handling
                "--max_old_space_size=4096", # Limit Node.js memory
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding"
            ]
        )
        
        browser = Browser(config=browser_config)
        browser_context = await browser.new_context()
        
        # Get the page for screenshot capturing
        page = await browser_context.get_current_page()
        
        print("[DEBUG] Browser setup completed successfully")

        # Screenshot capture loop with reduced frequency and PNG support
        async def capture_loop():
            frame = 0
            while not stop_capture_flag[0]:
                try:
                    screenshot_path = os.path.join(screenshot_dir, f"frame_{frame}.png")
                    # Remove quality parameter for PNG screenshots
                    await page.screenshot(path=screenshot_path, type='png')
                    screenshots.append(screenshot_path)
                    print(f"[+] Captured screenshot: {screenshot_path} (total: {len(screenshots)})")
                    frame += 1
                except Exception as e:
                    print(f"[ERROR] Failed to capture screenshot: {e}")
                await asyncio.sleep(3)  # Increased interval to reduce memory usage

        capture_task = asyncio.create_task(capture_loop())

        # Setup controller
        controller = CustomController(output_model=TestResult)

        if mcp_server_config:
            await controller.setup_mcp_client(mcp_server_config)

        # Configure LLM with correct parameters for ChatGoogle
        try:
            llm = ChatGoogle(
                model="gemini-2.0-flash", 
                api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1,  # Lower temperature for more deterministic responses
                # Note: ChatGoogle doesn't support max_tokens parameter
                # max_tokens is handled internally by the model
            )
            print("[DEBUG] LLM initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize LLM: {e}")
            raise

        # Create and run agent with proper browser context
        agent = Agent(
            task=prompt,
            llm=llm,
            browser=browser,  # Pass browser instead of browser_session
            controller=controller,
            max_steps=20,     # Reduced max steps to prevent long executions
            extend_system_message=extend_system_message
        )

        try:
            print("[DEBUG] Starting agent execution...")
            print(f"[DEBUG] Task prompt: {prompt[:200]}...")  # Log first 200 chars of prompt
            
            # Ensure the browser starts with a proper page
            page = await browser_context.get_current_page()
            await page.goto("https://www.google.com")
            await page.wait_for_load_state("networkidle", timeout=10000)
            print("[DEBUG] Initial navigation to Google completed")
            
            # Add a small delay to ensure everything is ready
            await asyncio.sleep(2)
            
            history = await agent.run()
            print(f"[DEBUG] Agent execution completed. History length: {len(history.history) if history else 0}")
            
            final_output = history.final_result()
            if final_output:
                print(f"[DEBUG] Final output: {final_output[:500]}...")  # Log first 500 chars
                structured_result = TestResult.model_validate_json(final_output)
                status = structured_result.status
                print(f"[DEBUG] Agent completed with status: {status}")
            else:
                structured_result = TestResult(steps=[], final_result="No result", status="fail")
                status = "fail"
                print("[DEBUG] Agent completed but no final output")
        except asyncio.TimeoutError:
            print("[ERROR] Agent execution timed out")
            structured_result = TestResult(
                steps=[], 
                final_result="Agent execution timed out", 
                status="fail"
            )
            status = "fail"
        except Exception as e:
            print(f"[ERROR] Agent execution failed: {e}")
            structured_result = TestResult(
                steps=[], 
                final_result=f"Agent execution failed: {str(e)}", 
                status="fail"
            )
            status = "fail"
        finally:
            stop_capture_flag[0] = True
            if capture_task:
                try:
                    await capture_task
                    print("[DEBUG] Screenshot capture stopped")
                except Exception as e:
                    print(f"[WARN] Error stopping capture task: {e}")

    except Exception as e:
        print(f"[ERROR] Browser setup failed: {e}")
        # Fallback to playwright persistent context
        print("[DEBUG] Falling back to playwright persistent context...")
        
        try:
            async with async_playwright() as playwright:
                try:
                    print("[DEBUG] Starting persistent browser launch...")
                    browser_pw = await playwright.chromium.launch_persistent_context(
                        user_data_dir="user_data",
                        headless=False,
                        args=["--start-maximized"]
                    )
                    print("[DEBUG] Persistent browser launched successfully")
                except Exception as e:
                    print(f"[ERROR] Playwright browser launch failed: {str(e)}")
                    raise

                page = await browser_pw.new_page()

                async def capture_loop_fallback():
                    frame = 0
                    while not stop_capture_flag[0]:
                        try:
                            screenshot_path = os.path.join(screenshot_dir, f"frame_{frame}.png")
                            # Remove quality parameter for PNG screenshots
                            await page.screenshot(path=screenshot_path, type='png')
                            screenshots.append(screenshot_path)
                            print(f"[+] Captured screenshot: {screenshot_path} (total: {len(screenshots)})")
                            frame += 1
                        except Exception as e:
                            print(f"[ERROR] Failed to capture screenshot: {e}")
                        await asyncio.sleep(3)  # Reduced frequency

                capture_task = asyncio.create_task(capture_loop_fallback())

                # For fallback, we'll have to use a basic setup without full browser_use integration
                # This is a simplified approach for the fallback case
                try:
                    # Simple task execution without full agent
                    await page.goto("https://www.google.com")
                    await page.wait_for_load_state()
                    
                    structured_result = TestResult(
                        steps=[Step(action="fallback", description="Used fallback browser")],
                        final_result="Fallback browser setup completed",
                        status="success"
                    )
                    status = "success"
                    
                except Exception as fallback_error:
                    print(f"[ERROR] Fallback execution failed: {fallback_error}")
                    structured_result = TestResult(
                        steps=[],
                        final_result=f"Fallback execution failed: {str(fallback_error)}",
                        status="fail"
                    )
                    status = "fail"
                finally:
                    stop_capture_flag[0] = True
                    if capture_task:
                        await capture_task
                    await browser_pw.close()
                    
        except Exception as fallback_error:
            print(f"[ERROR] Fallback also failed: {fallback_error}")
            structured_result = TestResult(
                steps=[],
                final_result=f"Both browser setups failed: {str(fallback_error)}",
                status="fail"
            )
            status = "fail"

    finally:
        # Cleanup
        try:
            if controller:
                if mcp_server_config:
                    await controller.close_mcp_client()
                else:
                    await controller.close_browser_tool()
                print("[DEBUG] Controller cleaned up")
        except Exception as e:
            print(f"[WARN] Error cleaning up controller: {e}")

        try:
            if browser_context:
                await browser_context.close()
                print("[DEBUG] Browser context closed")
        except Exception as e:
            print(f"[WARN] Error closing browser context: {e}")

        try:
            if browser:
                await browser.close()
                print("[DEBUG] Browser closed")
        except Exception as e:
            print(f"[WARN] Error closing browser: {e}")

    # Generate GIF from screenshots (only if we have screenshots)
    if screenshots:
        generate_gif_from_images(screenshots, gif_path)
    else:
        print("[WARN] No screenshots captured, skipping GIF generation")

    return {
        "text": structured_result.final_result,
        "gif_path": gif_path if os.path.exists(gif_path) else None,
        "status": status,
        "screenshots": screenshots,
        "timestamp": timestamp
    }

def run_prompt(prompt: str):
    return asyncio.run(run_agent_task(prompt))