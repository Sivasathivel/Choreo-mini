"""Generated CrewAI workflow from choreo-mini code."""

from crewai import Agent, Task, Crew
from choreo_mini.core.llm import CustomLLM

# Create the LLM
echo_llm = CustomLLM(lambda prompt, context=None, **kw: f"echo: {prompt}")

# Create agents
greeter = Agent(
    role="greeter",
    goal="Respond to user messages",
    backstory="A simple agent that echoes user input",
    llm=echo_llm,
    verbose=True
)

# Create tasks
greet_task = Task(
    description="Respond to the user's message",
    agent=greeter,
    expected_output="A response that echoes the user's input"
)

# Create crew
crew = Crew(
    agents=[greeter],
    tasks=[greet_task],
    verbose=True
)

if __name__ == "__main__":
    # Run the crew
    result = crew.kickoff(inputs={"message": "Hello"})
    print("Response:", result)