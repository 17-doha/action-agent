from dotenv import load_dotenv
import asyncio

from browser_use import Agent
from browser_use.llm import ChatGoogle  # or another model provider
import os
load_dotenv()  # Loads API keys from .env file


task_prompt = """

You are an automated browser agent. Your goal is to fully book a hotel on Booking.com with free cancellation, logging in via Google first. Follow these steps exactly:

1. Navigate to https://www.booking.com.
2. Wait until the page is fully loaded.
3. Click on the “Sign in” or “Login” button in the header.
4. From the login options, select “Continue with Google.”
5. When the Google login popup appears:
   a. Enter the Google email: place here email
   b. Click “Next.”
   c. Enter the Google password: place here password
   d. Click “Next” and wait for authentication to complete.
6. Once returned to Booking.com and signed in, locate the main search form.
7. In the search input, type “China” and select the suggestion for “China” (or “Hotels in China”).
8. Open the date picker:
   a. Select a check‑in date on the first available day in August of the upcoming year.
   b. Select a check‑out date three nights later.
9. Click “Search” or the equivalent button.
10. When results load, open the filters panel:
    a. Under “Free cancellation,” check the box to only show refundable/free‑cancellation options.
11. Wait for the filtered results to update.
12. Sort results by “Price (lowest first)” (or your preferred sort).
13. From the top of the list, choose the first hotel that has:
    - A free‑cancellation badge.
    - A total price displayed.
14. Click “Reserve” or “Select room” for that property.
15. On the room selection page, pick the first available room with free cancellation.
16. Click “I’ll reserve” (or “Book now”) and wait for the booking form.
17. Fill in any required guest information fields if prompted (name, contact phone, etc.).
18. Review the reservation summary to ensure:
    - Dates are correct (3 nights in August).
    - Free cancellation is confirmed.
    - Total price is displayed.
19. Click “Complete booking” or “Confirm” to finalize.
20. Wait until you see the confirmation page with a booking reference number.
21. End.

Make sure to wait for each page and element to load before interacting.```




"""
planner_llm = ChatGoogle(model="gemini-2.0-flash", api_key= "AIzaSyBiNdhGIUfAveXMIdqsZNMyrcsWQRLppw0")
async def main():
    agent = Agent(
        task=task_prompt,  # Any instruction
        llm=ChatGoogle(model="gemini-2.0-flash", api_key= "AIzaSyBiNdhGIUfAveXMIdqsZNMyrcsWQRLppw0"),
        use_vision=True, 
        planner_llm=planner_llm,
        max_steps=100
    )
    result = await agent.run()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

