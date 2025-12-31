# 🤖 Action Agent

**Action Agent** is a powerful AI-driven browser automation tool built with [browser-use](https://github.com/browser-use/browser-use), Google's **Gemini 2.0 Flash**, and **Playwright**. It allows users to execute complex tasks in a real browser environment through simple natural language prompts.

---

## 📽️ Demo

Watch Action Agent in action:

!

https://github.com/user-attachments/assets/833036f4-d984-4460-b5ad-be755216630a



---

## ✨ Key Features

- **🚀 Autonomous Browsing**: Performs multi-step tasks like searching, data extraction, and form filling with minimal intervention.
- **🛠️ MCP Support**: Integrates with Model Context Protocol (MCP) servers to extend its capabilities with custom tools.
- **👥 Human-in-the-Loop**: Can ask for help when it encounters CAPTCHAs, requires credentials, or needs human judgment.
- **📁 File Handling**: Supports uploading and interacting with local files.
- **🐳 Docker Ready**: Easily deployable using Docker and Docker Compose.
- **🖥️ Modern Web UI**: A sleek Flask-based interface to monitor prompts, execution steps, and visual outputs (GIFs/Videos).

---

## 🛠️ Tech Stack

- **Core**: Python 3.10+
- **AI Model**: Gemini 2.0 Flash (via Google Generative AI)
- **Framework**: Flask
- **Browser Automation**: Playwright & browser-use
- **Environment**: Docker & Docker Compose

---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- A [Google AI Studio API Key](https://aistudio.google.com/app/apikey) (for Gemini 2.0 Flash)

### 📂 Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/action-agent.git
    cd action-agent
    ```

2.  **Configure Environment Variables**:
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```

3.  **Run with Docker Compose**:
    ```bash
    docker-compose up --build
    ```

4.  **Access the App**:
    Open your browser and navigate to `http://localhost:5000`.

---

## 📖 Usage

1.  Enter your task in the prompt input field (e.g., *"Find the best hotel in Cairo for under $100 and give me a summary"*).
2.  Click **Run**.
3.  Watch the agent explore the web in real-time.
4.  Once completed, the agent will provide a structured result and a playback recording of the process.

---

## 📂 Project Structure

```text
.
├── app.py                # Flask application entry point
├── main.py               # CLI entry point
├── services/             # Core agent logic
├── controllers/          # Custom browser-use controllers
├── models/               # Pydantic data models
├── templates/            # Web UI HTML templates
├── app_static/           # Generated screenshots/GIFs/Videos
├── custom_browser/       # Custom browser configurations
└── requirements.txt      # Python dependencies
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
