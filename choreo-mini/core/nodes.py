from typing import Any, Dict, List, Optional, Union

# import LLM types for execution
from choreo_mini.core.llm import LLM, Message

class BaseNode:
    def __init__(self,
                 name: str, node_type: str,
                 properties: Optional[Dict[str, Any]] = None):
        self.name = name
        self.node_type = node_type
        self.properties = properties or {}
        self.children: List['BaseNode'] = []
        self.workflow: Optional['Workflow'] = None  # set when added to a workflow
        self.messages: List[Message] = []  # for agent nodes, to track conversation history


    def add_child(self, child_node: 'BaseNode'):
        self.children.append(child_node)

    def __repr__(self):
        return f"BaseNode(name={self.name}, type={self.node_type}, properties={self.properties})"

class AgentNode(BaseNode):
    def __init__(self,
                 name: str,
                 role: Optional[str] = None,
                 tasks: Optional[List[str]] = None,
                 goals: Optional[List[str]] = None,
                 backstory: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 properties: Optional[Dict[str, Any]] = None,
                 llm: LLM = None):
        super().__init__(name, "agent", properties)
        self.node_type = "agent"
        self.role: Optional[str] = role
        self.tasks: List[str] = tasks or []
        self.backstory: Optional[str] = backstory 
        self.system_prompt: Optional[str] = system_prompt
        self.goals: List[str] = goals or []
        self.llm: LLM = llm

        if not self.system_prompt:
            self.system_prompt = self.get_system_prompt()

    def get_system_prompt(self) -> str:
        if not self.system_prompt:
            prompt_parts = []
            if self.role: prompt_parts.append(f"Role: {self.role}")
            if self.backstory: prompt_parts.append(f"Backstory: {self.backstory}")
            if self.tasks: prompt_parts.append(f"Tasks: {', '.join(self.tasks)}")
            self.system_prompt = "\n".join(prompt_parts)
        if self.system_prompt: return self.system_prompt
        else:
            raise ValueError(f"Agent '{self.name}' does not have a system prompt or enough information to generate one.")
    
    def set_system_prompt(self, prompt: str):
        if prompt:
            self.system_prompt = prompt
    
    def __repr__(self):
        return f"AgentNode(name={self.name}, role={self.role}, tasks={self.tasks}, backstory={self.backstory})"
    
    def execute(
        self,
        context: Optional[Union[str, List[Message]]] = None,
        **kwds,
    ) -> Message:
        """Run the agent using an LLM.

        **Note:** conversation history is normally managed by a
        :class:`Workflow` instance.  When you invoke ``execute`` from a
        workflow, the ``context`` argument is filled automatically with the
        prior exchange.  Callers may pass ``context`` manually only if they
        wish to override or inspect the history themselves.

        ``context`` may be either a single string (appended as a user
        message) or a list of :class:`Message` objects representing a prior
        dialogue.  Additional keyword arguments are passed through to the
        model (temperature, max_tokens, etc.).

        LLM configuration is read from ``self.properties``.  At minimum
        the following keys are consulted:

        * ``provider`` – name registered with :func:`LLM.register_llm`
          (default ``"openai"``)
        * ``api_key`` – API key or credential for the service
        * ``model`` – model name (optional)
        * ``endpoint`` – base URL of the API (optional)

        The method returns the assistant message produced by the model.
        """
        # assemble provider arguments from properties
        self.messages.append(Message(role="user", content=context if isinstance(context, str) else ""  ))

        response = self.llm.chat(self.messages, **kwds)
        return response

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
    
