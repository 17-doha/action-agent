from services.agent_service import run_prompt


if __name__ == "__main__":
    result = run_prompt("Search hotels in Cairo")
    print(result)
