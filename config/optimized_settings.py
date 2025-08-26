# config/optimized_settings.py
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class OptimizedBrowserConfig:
    """Optimized browser configuration for memory efficiency."""
    headless: bool = True
    disable_security: bool = True
    max_memory_mb: int = 1024
    timeout_seconds: int = 30
    max_failures: int = 2
    disable_images: bool = True
    disable_javascript: bool = False  # Set to True if JS not needed
    single_process: bool = True
    
    def get_chrome_args(self) -> list[str]:
        """Get optimized Chrome arguments."""
        args = [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            f'--max_old_space_size={self.max_memory_mb}',
            '--memory-pressure-off',
            '--aggressive-cache-discard',
            '--memory-pressure-thresholds=0,0',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-extensions',
            '--disable-plugins',
            '--disable-web-security',
            '--disable-features=TranslateUI,BlinkGenPropertyTrees,VizDisplayCompositor',
            '--disable-ipc-flooding-protection',
            '--disable-background-networking',
            '--disable-default-apps',
            '--disable-sync',
            '--no-first-run',
            '--mute-audio',
        ]
        
        if self.single_process:
            args.append('--single-process')
        
        if self.disable_images:
            args.append('--disable-images')
            
        if self.disable_javascript:
            args.append('--disable-javascript')
            
        return args

@dataclass
class LLMConfig:
    """LLM configuration for optimal performance."""
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.0
    timeout_seconds: int = 30
    max_retries: int = 2
    enable_streaming: bool = False

@dataclass
class AgentConfig:
    """Agent configuration for stability."""
    max_steps: int = 10
    max_failures: int = 2
    step_timeout_seconds: int = 60
    total_timeout_seconds: int = 300  # 5 minutes
    save_conversations: bool = False
    save_traces: bool = False
    enable_screenshots: bool = False

class OptimizedSettings:
    """Central configuration manager for optimized browser automation."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.browser = OptimizedBrowserConfig()
        self.llm = LLMConfig()
        self.agent = AgentConfig()
        self._setup_logging()
        self._configure_environment()
        
    def _setup_logging(self):
        """Configure logging for better debugging."""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('browser_automation.log', mode='a')
            ]
        )
        
        # Reduce noise from verbose libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('playwright').setLevel(logging.WARNING)
        
    def _configure_environment(self):
        """Set environment variables for optimization."""
        env_vars = {
            'CHROME_FLAGS': ' '.join(self.browser.get_chrome_args()),
            'NODE_OPTIONS': f'--max_old_space_size={self.browser.max_memory_mb}',
            'PLAYWRIGHT_BROWSERS_PATH': str(Path.home() / '.cache' / 'ms-playwright'),
            'BROWSER_USE_DISABLE_TELEMETRY': '1',
            'PYTHONUNBUFFERED': '1',
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            
    def get_browser_config_dict(self) -> Dict[str, Any]:
        """Get browser configuration as dictionary for browser_use."""
        return {
            'headless': self.browser.headless,
            'disable_security': self.browser.disable_security,
            'extra_chromium_args': self.browser.get_chrome_args(),
        }
    
    def get_llm_config_dict(self) -> Dict[str, Any]:
        """Get LLM configuration as dictionary."""
        return {
            'model': self.llm.model,
            'temperature': self.llm.temperature,
            'timeout': self.llm.timeout_seconds,
        }
    
    def get_agent_config_dict(self) -> Dict[str, Any]:
        """Get agent configuration as dictionary."""
        return {
            'max_steps': self.agent.max_steps,
            'max_failures': self.agent.max_failures,
            'save_conversation_path': None if not self.agent.save_conversations else 'conversations',
            'save_trace_path': None if not self.agent.save_traces else 'traces',
        }
    
    def update_from_file(self, config_file: str):
        """Update settings from configuration file."""
        if os.path.exists(config_file):
            import json
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    
                # Update browser config
                if 'browser' in config_data:
                    browser_config = config_data['browser']
                    for key, value in browser_config.items():
                        if hasattr(self.browser, key):
                            setattr(self.browser, key, value)
                
                # Update LLM config
                if 'llm' in config_data:
                    llm_config = config_data['llm']
                    for key, value in llm_config.items():
                        if hasattr(self.llm, key):
                            setattr(self.llm, key, value)
                
                # Update agent config
                if 'agent' in config_data:
                    agent_config = config_data['agent']
                    for key, value in agent_config.items():
                        if hasattr(self.agent, key):
                            setattr(self.agent, key, value)
                            
                logging.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logging.warning(f"Failed to load config from {config_file}: {e}")

# Global settings instance
settings = OptimizedSettings()

# Example configuration file template
CONFIG_TEMPLATE = {
    "browser": {
        "headless": True,
        "disable_security": True,
        "max_memory_mb": 1024,
        "timeout_seconds": 30,
        "max_failures": 2,
        "disable_images": True,
        "disable_javascript": False,
        "single_process": True
    },
    "llm": {
        "model": "gemini-2.0-flash-exp",
        "temperature": 0.0,
        "timeout_seconds": 30,
        "max_retries": 2,
        "enable_streaming": False
    },
    "agent": {
        "max_steps": 10,
        "max_failures": 2,
        "step_timeout_seconds": 60,
        "total_timeout_seconds": 300,
        "save_conversations": False,
        "save_traces": False,
        "enable_screenshots": False
    }
}

def create_config_file(path: str = "optimized_config.json"):
    """Create a configuration file template."""
    import json
    with open(path, 'w') as f:
        json.dump(CONFIG_TEMPLATE, f, indent=2)
    print(f"Configuration template created at {path}")

if __name__ == "__main__":
    create_config_file()