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
from typing import List, Optional, Dict, Any
import json
from langchain_core.language_models import BaseChatModel
from utils.mcp_client import create_tool_param_model, setup_mcp_client_and_tools
import logging
from browser_use.config import CONFIG
from types import SimpleNamespace
from custom_browser.custom_browser import CustomBrowser

logger = logging.getLogger(__name__)
load_dotenv()

class Step(BaseModel):
    action: str = Field(description="The action taken in this step")
    description: str = Field(description="Description of what happened in this step")

class StepResult(BaseModel):
    steps: List[Step] = Field(description="List of steps taken by the agent")
    final_result: str = Field(description="The final result of the step")
    status: str = Field(description="Status: success or fail")
    verification_passed: bool = Field(description="Whether the step verification passed")

class BaseBookingStep:
    """Base class for all booking steps"""
    
    def __init__(self, browser_session: BrowserSession, controller, llm_model: str = "gemini-2.0-flash"):
        self.browser_session = browser_session
        self.controller = controller
        self.llm = ChatGoogle(model=llm_model, api_key=os.getenv("GOOGLE_API_KEY"))
        self.max_steps = 50
    
    async def execute(self, previous_step_data: Optional[Dict[str, Any]] = None) -> StepResult:
        """Execute this step"""
        prompt = self.get_prompt(previous_step_data)
        
        agent = Agent(
            task=prompt,
            llm=self.llm,
            browser_session=self.browser_session,
            controller=self.controller,
            max_steps=self.max_steps
        )
        
        try:
            history = await agent.run()
            final_output = history.final_result()
            
            if final_output:
                result = StepResult.model_validate_json(final_output)
            else:
                result = StepResult(
                    steps=[], 
                    final_result="No result", 
                    status="fail",
                    verification_passed=False
                )
            
            # Perform step-specific verification with fallback
            try:
                verification_result = await self.verify_step_completion()
            except AttributeError as e:
                logger.warning(f"Attribute error in verification: {e}. Falling back to agent result.")
                verification_result = (result.status == "success")
            except Exception as e:
                logger.error(f"Unexpected error in verification: {e}")
                verification_result = False
            
            result.verification_passed = verification_result
            
            if not verification_result:
                result.status = "fail"
                result.final_result += " - Step verification failed"
            
            return result
            
        except Exception as e:
            return StepResult(
                steps=[],
                final_result=f"Step execution failed: {str(e)}",
                status="fail",
                verification_passed=False
            )
    
    def get_prompt(self, previous_step_data: Optional[Dict[str, Any]] = None) -> str:
        """Get the prompt for this step - to be implemented by subclasses"""
        raise NotImplementedError
    
    async def verify_step_completion(self) -> bool:
        """Verify if the step completed successfully - to be implemented by subclasses"""
        raise NotImplementedError

class InitialAndLoginStep(BaseBookingStep):
    """Merged Step: Go to booking.com, check login status, and login if necessary"""
    
    def get_prompt(self, previous_step_data: Optional[Dict[str, Any]] = None) -> str:
        return """
        Go to booking.com.
        Check if the text 'tabaani-action' exists anywhere on the page.
        If it exists, you are already logged in. Return status 'success' with message 'User already logged in - tabaani-action found'.
        If it does not exist, proceed to login:
        1. Click on the "Sign in" button
        2. Choose Google as the login method
        3. Wait for the Google login window to appear
        4. Enter username: tabaaniaction@gmail.com
        5. Press Next
        6. Enter password: Tabaani@3072004
        7. Press Next
        8. After signing in, make sure you return to the Booking.com main window (not the Google popup)
        9. Verify that you are successfully logged in by checking if user account information is visible
        
        If login was needed and completed, return status 'success' with message 'Login completed successfully'.
        If any step fails, provide detailed error information.
        Make sure to thoroughly scan the entire page content for the text initially.
        """
    
    async def verify_step_completion(self) -> bool:
        try:
            page = self.browser_session.browser_context.pages[0]
            current_url = page.url
            if "booking.com" not in current_url.lower():
                return False
            
            page_content = await page.content()
            login_indicators = ["account", "profile", "sign out", "logout", "tabaani-action"]
            
            return any(indicator in page_content.lower() for indicator in login_indicators)
        except Exception as e:
            logger.error(f"Verification failed for InitialAndLoginStep: {e}")
            return False

class SearchStep(BaseBookingStep):
    """Step 3: Search for destination and dates"""
    
    def get_prompt(self, previous_step_data: Optional[Dict[str, Any]] = None) -> str:
        return """
        You are on booking.com main page and need to search for hotels.
        Follow these steps:
        1. Find the destination search field
        2. Enter "Japan" as the destination
        3. Set the check-in date to 19/8/2025 (August 19, 2025)
        4. Set the check-out date to 21/8/2025 (August 21, 2025)
        5. Click the Search button
        6. Wait for the search results page to load
        
        Make sure all fields are properly filled before clicking search.
        Return status 'success' only when you reach the search results page with hotels listed.
        """
    
    async def verify_step_completion(self) -> bool:
        try:
            page = self.browser_session.browser_context.pages[0]
            current_url = page.url
            page_content = await page.content()
            
            # Check if we're on search results page
            search_indicators = [
                "searchresults",
                "search results", 
                "hotels found",
                "properties found",
                "japan" in current_url.lower() or "japan" in page_content.lower()
            ]
            
            return any(indicator in current_url.lower() or indicator in page_content.lower() 
                      for indicator in search_indicators)
        except Exception as e:
            logger.error(f"Verification failed for SearchStep: {e}")
            return False

class HotelSelectionStep(BaseBookingStep):
    """Step 4: Select a hotel from search results"""
    
    def get_prompt(self, previous_step_data: Optional[Dict[str, Any]] = None) -> str:
        return """
        You are on the booking.com search results page showing hotels in Japan.
        Follow these steps:
        1. Look at the available hotels in the search results
        2. Choose any hotel from the results (preferably one that's available and reasonably priced)
        3. Click on the selected hotel
        4. Wait for the new tab/window to open with the hotel details
        5. Switch to the new tab/window that contains the hotel details
        6. Make sure the hotel detail page is fully loaded before proceeding
        
        Return status 'success' only when you successfully reach a hotel detail page.
        Include the hotel name in your final result if possible.
        """
    
    async def verify_step_completion(self) -> bool:
        try:
            page = self.browser_session.browser_context.pages[-1]  # Use last page for new tab
            current_url = page.url
            page_content = await page.content()
            
            # Check if we're on a hotel detail page
            hotel_page_indicators = [
                "hotel",
                "property",
                "book now",
                "reserve",
                "rooms",
                "amenities"
            ]
            
            return any(indicator in current_url.lower() or indicator in page_content.lower() 
                      for indicator in hotel_page_indicators)
        except Exception as e:
            logger.error(f"Verification failed for HotelSelectionStep: {e}")
            return False

class ReservationStep(BaseBookingStep):
    """Step 5: Complete the reservation process"""
    
    def get_prompt(self, previous_step_data: Optional[Dict[str, Any]] = None) -> str:
        return """
        You are on a hotel detail page on booking.com.
        Follow these steps to complete the reservation:
        
        1. Look for any button or link with text containing "Reserve", "Book now", "Book", or similar (case-insensitive)
        2. If found, click on it
        3. Choose any available room from the options presented
        4. Click "I'll reserve" or similar confirmation button
        5. Fill in the required guest information:
           - Phone number: 01087653765
           - Name: tabaani (use this for first name, you can use "Action" for last name if needed)
           - Email: tabaani-action@gmail.com
           - Country code: EG +20
        6. Scroll down to find the next section
        6. Look for a "Next", "Continue", or "Final details" button and click it if found
        7. Complete any additional required fields that appear
        
        Take your time with each step and make sure forms are properly filled.
        Return status 'success' only when you reach the final confirmation or payment page.
        If you encounter any errors or missing required fields, describe them in detail.
        """
    
    async def verify_step_completion(self) -> bool:
        try:
            page = self.browser_session.browser_context.pages[-1]  # Assume current is last page
            current_url = page.url
            page_content = await page.content()
            
            # Check if we're on confirmation/payment page
            completion_indicators = [
                "confirmation",
                "payment",
                "final details",
                "complete booking",
                "credit card",
                "booking summary",
                "review booking"
            ]
            
            return any(indicator in current_url.lower() or indicator in page_content.lower() 
                      for indicator in completion_indicators)
        except Exception as e:
            logger.error(f"Verification failed for ReservationStep: {e}")
            return False

class BookingWorkflow:
    """Main workflow coordinator for the booking process"""
    
    def __init__(self, controller, llm_model: str = "gemini-2.0-flash"):
        self.controller = controller
        self.llm_model = llm_model
        self.steps = []
        self.results = []
    
    async def execute_workflow(self, browser_session: BrowserSession) -> Dict[str, Any]:
        """Execute the complete booking workflow"""
        
        # Define the workflow steps
        workflow_steps = [
            InitialAndLoginStep(browser_session, self.controller, self.llm_model),
            SearchStep(browser_session, self.controller, self.llm_model),
            HotelSelectionStep(browser_session, self.controller, self.llm_model),
            ReservationStep(browser_session, self.controller, self.llm_model)
        ]
        
        previous_step_data = None
        overall_success = True
        
        for i, step in enumerate(workflow_steps):
            step_name = step.__class__.__name__
            logger.info(f"Executing step {i+1}/4: {step_name}")
            
            try:
                result = await step.execute(previous_step_data)
                self.results.append(result)
                
                if result.status == "fail" or not result.verification_passed:
                    logger.error(f"Step {step_name} failed: {result.final_result}")
                    overall_success = False
                    break
                
                logger.info(f"Step {step_name} completed successfully")
                previous_step_data = None  # Since next_step_data is removed
                
            except Exception as e:
                logger.error(f"Error executing step {step_name}: {str(e)}")
                overall_success = False
                break
        
        # Compile final results
        final_status = "success" if overall_success else "fail"
        all_steps = []
        final_messages = []
        
        for result in self.results:
            all_steps.extend(result.steps)
            final_messages.append(result.final_result)
        
        return {
            "text": " | ".join(final_messages),
            "status": final_status,
            "detailed_results": self.results,
            "total_steps": len(all_steps)
        }