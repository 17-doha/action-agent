import os
import inspect
import logging
from typing import Optional, Type, Callable, Dict, Any, Union, Awaitable, TypeVar

from pydantic import BaseModel
from browser_use.agent.views import ActionResult, ActionModel
from browser_use.browser.context import BrowserContext
from browser_use.controller.registry.service import RegisteredAction
from browser_use.controller.service import Controller
from browser_use.config import CONFIG
from langchain_core.language_models import BaseChatModel
from utils.mcp_client import create_tool_param_model, setup_mcp_client_and_tools
from tools.browser_use_tool import BrowserUseTool

logger = logging.getLogger(__name__)
Context = TypeVar("Context")


def time_execution_sync(label):
    """Decorator for timing function execution (sync version from first file)"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class CustomController(Controller):
    """
    Custom Controller that extends the base Controller
    to support extra actions (ask assistant, file upload),
    BrowserUseTool integration, and dynamic MCP tool registration.
    """

    def __init__(
        self,
        exclude_actions: list[str] = [],
        output_model: Optional[Type[BaseModel]] = None,
        ask_assistant_callback: Optional[
            Union[
                Callable[[str, BrowserContext], Dict[str, Any]],
                Callable[[str, BrowserContext], Awaitable[Dict[str, Any]]]
            ]
        ] = None,
    ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()
        self.ask_assistant_callback = ask_assistant_callback
        
        # Initialize BrowserUseTool (from first file)
        self.browser_tool = BrowserUseTool()
        
        # Initialize MCP client attributes
        self.mcp_client = None
        self.mcp_server_config = None

    def _register_custom_actions(self):
        """Register all custom browser actions."""

        @self.registry.action(
            "When executing tasks, prioritize autonomous completion. "
            "If blocked (credentials, human judgment, CAPTCHAs, etc.), request human assistance."
        )
        async def ask_for_assistant(query: str, browser: BrowserContext):
            if self.ask_assistant_callback:
                if inspect.iscoroutinefunction(self.ask_assistant_callback):
                    user_response = await self.ask_assistant_callback(query, browser)
                else:
                    user_response = self.ask_assistant_callback(query, browser)
                msg = f"AI ask: {query}. User response: {user_response['response']}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            return ActionResult(
                extracted_content="Human cannot help you. Please try another way.",
                include_in_memory=True,
            )

        @self.registry.action("Upload file to interactive element with file path")
        async def upload_file(
            index: int,
            path: str,
            browser: BrowserContext,
            available_file_paths: list[str],
        ):
            if path not in available_file_paths:
                return ActionResult(error=f'File path {path} is not available')
            if not os.path.exists(path):
                return ActionResult(error=f'File {path} does not exist')

            dom_el = await browser.get_dom_element_by_index(index)
            file_upload_dom_el = dom_el.get_file_upload_element()

            if not file_upload_dom_el:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            file_upload_el = await browser.get_locate_element(file_upload_dom_el)
            if not file_upload_el:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            try:
                await file_upload_el.set_input_files(path)
                msg = f'Successfully uploaded file to index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f'Failed to upload file to index {index}: {str(e)}'
                logger.info(msg)
                return ActionResult(error=msg)

    @time_execution_sync('--act')
    async def act(
        self,
        action: ActionModel,
        browser_context: Optional[BrowserContext] = None,
        page_extraction_llm: Optional[BaseChatModel] = None,
        sensitive_data: Optional[Dict[str, str]] = None,
        available_file_paths: Optional[list[str]] = None,
        context: Context | None = None,
        browser_session: Any = None,
        file_system: Any = None,
    ) -> ActionResult:
        """Execute actions with BrowserUseTool integration and registry fallback."""
        try:
            # Extract action_name and params from action (BrowserUseTool integration from first file)
            action_dict = action.model_dump(exclude_unset=True)
            if action_dict:
                action_name = next(iter(action_dict))
                params = action_dict[action_name]

                # Try to route through BrowserUseTool first
                try:
                    tool_result = await self.browser_tool.execute(action=action_name, **params)

                    # Convert ToolResult to ActionResult
                    if tool_result.error:
                        return ActionResult(error=tool_result.error)
                    else:
                        # Optionally get updated state
                        state_result = await self.browser_tool.get_current_state()
                        extracted = tool_result.output + "\nCurrent State: " + state_result.output
                        return ActionResult(
                            extracted_content=extracted, 
                            base64_image=state_result.base64_image
                        )
                except Exception as browser_tool_error:
                    logger.warning(f"BrowserUseTool failed for {action_name}: {browser_tool_error}")
                    # Fall back to registry execution
                    pass

                # Fallback to original registry-based execution
                for action_name_fallback, params_fallback in action_dict.items():
                    if params_fallback is not None:
                        result = await self.registry.execute_action(
                            action_name_fallback,
                            params_fallback,
                            browser_session=browser_session,
                            page_extraction_llm=page_extraction_llm,
                            file_system=file_system,
                            sensitive_data=sensitive_data,
                            available_file_paths=available_file_paths,
                            context=context,
                        )

                        if isinstance(result, str):
                            return ActionResult(extracted_content=result)
                        if isinstance(result, ActionResult):
                            return result
                        if result is None:
                            return ActionResult()
                        raise ValueError(
                            f'Invalid action result type: {type(result)} of {result}'
                        )
            return ActionResult()
        except Exception as e:
            logger.error(f"Error in act method: {e}")
            return ActionResult(error=str(e))

    async def close_browser_tool(self):
        """Clean up the browser tool properly (from first file)"""
        if hasattr(self, 'browser_tool') and self.browser_tool:
            try:
                await self.browser_tool.cleanup()
                logger.info("BrowserUseTool cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error cleaning up BrowserUseTool: {e}")

    async def setup_mcp_client(self, mcp_server_config: Optional[Dict[str, Any]] = None):
        """Initialize MCP client and register tools with proper error handling."""
        try:
            self.mcp_server_config = mcp_server_config
            if self.mcp_server_config:
                self.mcp_client, self.mcp_server_config = await setup_mcp_client_and_tools(
                    self.mcp_server_config
                )
                self.register_mcp_tools()
                logger.info("MCP client setup completed successfully")
        except Exception as e:
            logger.warning(f"Error setting up MCP client: {e}")
            self.mcp_client = None
            self.mcp_server_config = None

    def register_mcp_tools(self):
        """Register the MCP tools used by this controller."""
        if not self.mcp_client:
            logger.warning("MCP client not started.")
            return

        for server_name in self.mcp_client.server_name_to_tools:
            for tool in self.mcp_client.server_name_to_tools[server_name]:
                tool_name = f"mcp.{server_name}.{tool.name}"
                self.registry.registry.actions[tool_name] = RegisteredAction(
                    name=tool_name,
                    description=tool.description,
                    function=tool,
                    param_model=create_tool_param_model(tool),
                )
                logger.info(f"Add MCP tool: {tool_name}")
            logger.debug(
                f"Registered {len(self.mcp_client.server_name_to_tools[server_name])} "
                f"MCP tools for {server_name}"
            )

    async def close_mcp_client(self):
        """Clean up MCP client and browser tool (enhanced from first file)"""
        # Clean up browser tool first
        await self.close_browser_tool()
        
        # Then clean up MCP client
        if self.mcp_client:
            try:
                await self.mcp_client.__aexit__(None, None, None)
                logger.info("MCP client cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error cleaning up MCP client: {e}")