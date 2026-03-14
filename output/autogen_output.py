"""Generated AutoGen workflow from choreo-mini code."""

from autogen import AssistantAgent, UserProxyAgent
from choreo_mini.core.llm import CustomLLM

# Create the LLM
echo_llm = CustomLLM(lambda prompt, context=None, **kw: f"echo: {prompt}")

# Create agents
greeter = AssistantAgent(
    name="greeter",
    system_message="You are a greeter agent that echoes user input.",
    llm_config={"config_list": [{"model": "custom", "api_key": "dummy"}]},
    code_execution_config=False,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
)

# Register the custom LLM (this would need proper integration)
# For demonstration, we'll simulate the conversation

if __name__ == "__main__":
    # Simulate conversation
    user_message = "Hello"
    print(f"User: {user_message}")

    # In a real AutoGen setup, this would use the LLM
    response = echo_llm.generate(user_message)
    print(f"Greeter: {response}")