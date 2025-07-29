from gradio_client import Client, handle_file

# Connect to the web-ui API
client = Client("http://127.0.0.1:7788/")

# Structured browser task prompt
structured_prompt = '''
Open Booking.com and login using my Google account.  
Write username and password after clicking to login with google only.
Username: dohaahemdan@gmail.com
click next
Password: Dohah#2771994 
Search for a hotel in Giza, Egypt for 1 adult from July 17 to July 20, 2025.  
Click on the first available option with good reviews.  
Proceed to reserve the room with free cancellation if available.  
Complete the reservation process up to the final confirmation page.
'''



# Make a prediction using the /submit_wrapper endpoint
result = client.predict(
    "",  # param_0: Override system prompt (Textbox)
    "",  # param_1: Extend system prompt (Textbox)
    handle_file('C:/Users/Doha/action-agent/mcp_server.json'),  # param_2: MCP server JSON (File)
    "",  # param_3: MCP server (Textbox)
    "google",  # param_4: LLM Provider (Dropdown)
    "gemini-1.5-flash",  # param_5: LLM Model Name (Dropdown)
    0.6,  # param_6: LLM Temperature (Slider)
    True,  # param_7: Use Vision (Checkbox)
    16000,  # param_8: Ollama Context Length (Slider)
    "",  # param_9: Base URL (Textbox)
    "AIzaSyAcDEcM1tSpVNXmON4e0o9lGQ_NRQKNgkE",  # param_10: Google API Key (Textbox)
    "google",  # param_11: Planner LLM Provider (Dropdown)
    "gemini-1.5-flash",  # param_12: Planner LLM Model Name (Dropdown)
    0.6,  # param_13: Planner LLM Temperature (Slider)
    False,  # param_14: Use Vision (Planner LLM) (Checkbox)
    16000,  # param_15: Ollama Context Length (Planner) (Slider)
    "",  # param_16: Base URL (Planner) (Textbox)
    "",  # param_17: API Key (Planner) (Textbox)
    100,  # param_18: Max Run Steps (Slider)
    10,  # param_19: Max Number of Actions (Slider)
    128000,  # param_20: Max Input Tokens (Number)
    "auto",  # param_21: Tool Calling Method (Dropdown)
    "C:/Program Files/Google/Chrome/Application/chrome.exe",  # param_22: Browser Binary Path (Textbox)
    "C:/Users/Doha/AppData/Local/Google/Chrome/User Data",  # param_23: Browser User Data Dir (Textbox)
    False,  # param_24: Use Own Browser (Checkbox)
    True,  # param_25: Keep Browser Open (Checkbox)
    False,  # param_26: Headless Mode (Checkbox)
    False,  # param_27: Disable Security (Checkbox)
    1280,  # param_28: Window Width (Number)
    1100,  # param_29: Window Height (Number)
    "",  # param_30: CDP URL (Textbox)
    "",  # param_31: WSS URL (Textbox)
    "",  # param_32: Recording Path (Textbox)
    "",  # param_33: Trace Path (Textbox)
    "./tmp/agent_history",  # param_34: Agent History Save Path (Textbox)
    "./tmp/downloads",  # param_35: Save Directory for browser downloads (Textbox)
    [],  # param_36: Agent Interaction (Chatbot)
    structured_prompt,  # param_37: Your Task or Response (Textbox)
    None,  # param_38: Submit Button (Button, placeholder)
    None,  # param_39: Start Button (Button, placeholder)
    None,  # param_40: Stop Button (Button, placeholder)
    None,  # param_41: Pause/Resume Button (Button, placeholder)
    "<div style='width:100%; height:50vh; display:flex; justify-content:center; align-items:center; border:1px solid #ccc; background-color:#f0f0f0;'><p>Browser View (Requires Headless=True)</p></div>",  # param_42: Browser Live View (HTML)
    handle_file('C:/Users/Doha/action-agent/history.json'),  # param_43: Agent History JSON (File)
    handle_file('C:/Users/Doha/action-agent/placeholder.png'),  # param_44: Task Recording GIF (Image)
    api_name="/submit_wrapper"
)

# Print or process the result
print(result)