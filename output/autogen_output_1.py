"""Generated AutoGen workflow from choreo-mini code."""

from autogen import AssistantAgent, UserProxyAgent

import sys

from pathlib import Path

from choreo_mini.core.workflow import Workflow

from choreo_mini.core.nodes import AgentNode

from choreo_mini.core.llm import CustomLLM


# Create agents


greeter = AssistantAgent(
    name="greeter",
    system_message="You are a helpful assistant.",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": "your-key"}]}
)



# Create user proxy
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)

if __name__ == "__main__":
    # Execute based on workflow logic
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    # Profiling data would be collected here
    print("Profiling enabled but not implemented in AutoGen template")
    