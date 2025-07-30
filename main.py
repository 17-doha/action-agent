import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import imageio
from browser_use import Agent, BrowserSession, Controller
from browser_use.llm import ChatGoogle
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from typing import List

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
            print("[DEBUG] Starting browser launch...")
            browser = await playwright.chromium.launch(headless=True)
            print("[DEBUG] Browser launched successfully")
        except Exception as e:
            print(f"[ERROR] Browser launch failed: {str(e)}")
            raise  # Re-raise for app.py to catch

        context = await browser.new_context()
        page = await context.new_page()
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

        controller = Controller(output_model=TestResult)
        

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

    # Debug: Check if files exist
    print(f"[DEBUG] GIF exists: {os.path.exists(gif_path)} (path: {gif_path})")

    return {
        "text": structured_result.final_result,
        "gif_path": gif_path if os.path.exists(gif_path) else None,
        "status": status
    }

def run_prompt(prompt: str):
    return asyncio.run(run_agent_task(prompt))
