import asyncio
import os
import json
from datetime import datetime
from types import SimpleNamespace

from browser_use import Agent, Browser, BrowserConfig
from browser_use.llm import ChatGoogle
from playwright.async_api import async_playwright

from config.settings import GOOGLE_API_KEY, MCP_CONFIG_PATH
from models.result_models import TestResult, Step
from utils.file_utils import ensure_dirs, generate_gif_from_images
from controllers.custom_controller import CustomController
from prompts.prompt import extend_system_message

async def run_agent_task(prompt: str):
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_dir = os.path.join("app_static/screenshots", f"run_{timestamp}")
    os.makedirs(screenshot_dir, exist_ok=True)
    gif_path = f"app_static/gifs/test_{timestamp}.gif"
    screenshots = []

    stop_capture_flag = [False]

    # Load MCP server config
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
        print("[DEBUG] Setting up browser_use Browser...")
        
        # Optimized browser config for memory efficiency
        browser_config = BrowserConfig(
            headless=True,
            disable_security=True,
            extra_chromium_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--memory-pressure-off",
                "--max_old_space_size=1024",  # Reduced from 2048
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images",  # Save memory by not loading images
                "--disable-css",     # Disable CSS parsing
                "--single-process",
                "--disable-web-security",
                "--disable-features=TranslateUI,BlinkGenPropertyTrees,VizDisplayCompositor",
                "--aggressive-cache-discard",
                "--memory-pressure-thresholds=0,0",
                "--disable-ipc-flooding-protection",
                "--disable-background-networking",
                "--disable-default-apps",
                "--disable-sync",
                "--no-first-run"
            ]
        )
        
        browser = Browser(config=browser_config)
        
        # Create browser context with timeout and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                browser_context = await asyncio.wait_for(
                    browser.new_context(), 
                    timeout=30.0  # Reduced timeout
                )
                print(f"[DEBUG] Browser context created successfully (attempt {attempt + 1})")
                break
            except asyncio.TimeoutError:
                print(f"[ERROR] Browser context creation timed out (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    raise Exception("Browser context creation timeout after retries")
                await asyncio.sleep(2)  # Wait before retry
        
        # Disable screenshot capture to save memory
        capture_task = None
        print("[DEBUG] Screenshot capture disabled to save memory")

        # Setup controller
        controller = CustomController(output_model=TestResult)

        if mcp_server_config:
            await controller.setup_mcp_client(mcp_server_config)

        # Configure LLM with optimized settings
        try:
            llm = ChatGoogle(
                model="gemini-2.0-flash-exp",  # Use experimental version for better performance
                api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.0,  # Deterministic responses
            )
            print("[DEBUG] LLM initialized successfully")
        except Exception as e:
            print(f"[ERROR] Failed to initialize LLM: {e}")
            raise

        # Create agent with optimized settings
        agent = Agent(
            task=prompt,
            llm=llm,
            browser=browser,
            controller=controller,
            max_steps=10,  # Reduced from 20 to prevent long executions
            max_failures=2,  # Add failure limit
            extend_system_message=extend_system_message,
            save_conversation_path=None,  # Disable conversation saving
            save_trace_path=None,  # Disable trace saving
        )

        try:
            print("[DEBUG] Starting agent execution...")
            print(f"[DEBUG] Task prompt: {prompt[:200]}...")
            
            # Start with a lightweight page
            page = await browser_context.get_current_page()
            await page.goto("about:blank")  # Start with blank page
            await asyncio.sleep(1)  # Brief pause
            
            print("[DEBUG] Initial setup completed")
            
            # Execute agent with timeout
            history = await asyncio.wait_for(agent.run(), timeout=300)  # 5 minute timeout
            print(f"[DEBUG] Agent execution completed. History length: {len(history.history) if history else 0}")
            
            final_output = history.final_result()
            if final_output:
                print(f"[DEBUG] Final output: {final_output[:500]}...")
                try:
                    structured_result = TestResult.model_validate_json(final_output)
                    status = structured_result.status
                    print(f"[DEBUG] Agent completed with status: {status}")
                except Exception as parse_error:
                    print(f"[WARN] Failed to parse final output: {parse_error}")
                    structured_result = TestResult(
                        steps=[], 
                        final_result=final_output, 
                        status="success"
                    )
                    status = "success"
            else:
                structured_result = TestResult(
                    steps=[], 
                    final_result="Agent completed but no structured output", 
                    status="partial"
                )
                status = "partial"
                print("[DEBUG] Agent completed but no final output")
                
        except asyncio.TimeoutError:
            print("[ERROR] Agent execution timed out")
            structured_result = TestResult(
                steps=[], 
                final_result="Agent execution timed out after 5 minutes", 
                status="timeout"
            )
            status = "timeout"
        except Exception as e:
            print(f"[ERROR] Agent execution failed: {e}")
            structured_result = TestResult(
                steps=[], 
                final_result=f"Agent execution failed: {str(e)}", 
                status="fail"
            )
            status = "fail"

    except Exception as e:
        print(f"[ERROR] Browser setup failed: {e}")
        structured_result = TestResult(
            steps=[],
            final_result=f"Browser setup failed: {str(e)}",
            status="fail"
        )
        status = "fail"

    finally:
        # Enhanced cleanup with proper error handling
        stop_capture_flag[0] = True
        
        if capture_task:
            try:
                capture_task.cancel()
                await asyncio.sleep(0.1)
                print("[DEBUG] Screenshot capture stopped")
            except Exception as e:
                print(f"[WARN] Error stopping capture task: {e}")

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
                # Properly close all pages first
                pages = browser_context.browser_session.pages
                for page in pages:
                    try:
                        await page.close()
                    except Exception:
                        pass
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

        # Force garbage collection
        import gc
        gc.collect()
        print("[DEBUG] Garbage collection performed")

    # Skip GIF generation to save resources
    if screenshots:
        try:
            generate_gif_from_images(screenshots, gif_path)
        except Exception as e:
            print(f"[WARN] GIF generation failed: {e}")
    else:
        print("[INFO] No screenshots captured, skipping GIF generation")

    return {
        "text": structured_result.final_result,
        "gif_path": gif_path if os.path.exists(gif_path) else None,
        "status": status,
        "screenshots": screenshots,
        "timestamp": timestamp
    }

def run_prompt(prompt: str):
    return asyncio.run(run_agent_task(prompt))