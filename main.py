from dotenv import load_dotenv
import asyncio

from browser_use import Agent
from browser_use.llm import ChatGoogle  # or another model provider
import os
load_dotenv()  # Loads API keys from .env file

task_prompt = """
{
  "header": "You are a browser automation agent.",
  "title": "Test: Booking.com Hotel Reservation via Google Login",
  "objective": "Automate logging into Booking.com using a Google account, searching for a hotel in Giza, Egypt, and proceeding to reserve a room with free cancellation.",
  "preconditions": [
    "User has a valid Google account (username and password).",
    "Google sign-in is enabled and functional.",
    "System memory and performance are stable for execution."
  ],
  "steps": [
    {
      "step": "Navigate to https://www.booking.com.",
      "validation": {
        "pass": "Homepage loads successfully.",
        "fail": "Page failed to load or returned an error."
      }
    },
    {
      "step": "Click the 'Sign in' or 'Login' button.",
      "action": "click_element_by_index",
      "retry": 1,
      "validation": {
        "pass": "Login modal or redirection is triggered.",
        "fail": "Sign-in button click did not result in visible login UI."
      }
    },
    {
      "step": "Select 'Continue with Google' login option.",
      "action": "click_element_by_index",
      "retry": 2,
      "validation": {
        "pass": "Google login interface appears.",
        "fail": "Google login interface did not appear or timed out."
      }
    },
    {
      "step": "Enter Google account username.",
      "input": "dohaahemdan@gmail.com",
      "action": "fill_input_by_selector",
      "selector": "input[type='email']",
      "validation": {
        "pass": "'Next' button becomes available.",
        "fail": "Email field was not accepted or not interactable."
      }
    },
    {
      "step": "Click 'Next'.",
      "action": "click_element_by_text",
      "text": "Next"
    },
    {
      "step": "Enter Google account password.",
      "input": "Dohah#2771994",
      "action": "fill_input_by_selector",
      "selector": "input[type='password']",
      "validation": {
        "pass": "Login proceeds to Booking.com account.",
        "fail": "Password rejected or Google login failed."
      }
    },
    {
      "step": "Search for a hotel in Giza, Egypt for 1 adult from July 17 to July 20, 2025.",
      "action": "fill_search_form",
      "searchDetails": {
        "location": "Giza, Egypt",
        "checkIn": "2025-07-17",
        "checkOut": "2025-07-20",
        "guests": "1 adult"
      },
      "validation": {
        "pass": "Search results page displays hotels in Giza.",
        "fail": "No search results or incorrect destination/dates."
      }
    },
    {
      "step": "Filter results for good reviews (e.g. 8+ rating).",
      "optional": true
    },
    {
      "step": "Click the first available hotel with good reviews.",
      "action": "click_element_by_index",
      "index": 0,
      "validation": {
        "pass": "Hotel details page opens.",
        "fail": "Click failed or hotel page did not load."
      }
    },
    {
      "step": "Select a room with free cancellation (if available).",
      "action": "choose_room_with_condition",
      "condition": "Free cancellation",
      "validation": {
        "pass": "Room selected with free cancellation.",
        "fail": "No qualifying room available or selectable."
      }
    },
    {
      "step": "Proceed through booking until final confirmation page.",
      "action": "complete_booking_flow",
      "validation": {
        "pass": "Booking confirmation page is reached.",
        "fail": "Booking process was not completed."
      }
    }
  ],
  "expectedResult": [
    "User is successfully logged into Booking.com using Google.",
    "Search results for Giza are returned with the correct dates and filters.",
    "Room with free cancellation is selected.",
    "Booking confirmation is completed."
  ],
  "finalValidation": "If the confirmation page is not reached, report: ⚠️ 'Booking failed or confirmation not reached. Investigate login or room availability issues.'"
}

"""
async def main():
    agent = Agent(
        task=task_prompt,  # Any instruction
        llm=ChatGoogle(model="gemini-2.0-flash", api_key= "AIzaSyBiNdhGIUfAveXMIdqsZNMyrcsWQRLppw0"),
       
        max_steps=100
    )
    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())