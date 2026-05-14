"""
This is a Multi-Agent Reinforcement Learning (MARL) example using the Choreo Mini framework. It demonstrates how to set up a simple environment with multiple agents that learn to cooperate or compete based on their interactions. The agents will use a custom LLM (Language Model) for decision-making, and the environment will provide feedback to guide their learning process.

This experiment is designed to showcase the capabilities of Choreo Mini in handling complex multi-agent scenarios, and it can be extended to include more sophisticated agents, environments, and learning algorithms as needed.

The MARL setup includes:
- Worflow_USA representing USA in the CUSMA negotiotians, creating the possible actions that would maximize the USA's economic, social, and political benefits, as well as gain the political mileage for mid terms, asymmetric benefits for the funders, lobbyist and personal interests.
- Worflow_CANADA representing Canada in the CUSMA negotiotians, creating the possible actions that would maximize Canada's economic, social, and political benefits, as well as gain the political mileage for mid terms, asymmetric benefits for the funders, lobbyist and personal interests.
- Worflow_MEXICO representing Mexico in the CUSMA negotiotians, creating the possible actions that would maximize Mexico's economic, social, and political benefits, as well as gain the political mileage for mid terms, asymmetric benefits for the funders, lobbyist and personal interests.
- The 'environment' will simulate the negotiation process and provide feedback to the agents based on their actions, allowing them to learn and adapt their strategies over time.
- The actions taken by each workflow will be tracked as conversations for user to review and analyze the decision-making process of each agent.
- The experiment proceed until Nash equilibrium is reached where no agent can improve their outcome by unilaterally changing their strategy,  allowing us to observe the learning dynamics and outcomes of the multi-agent interactions.
"""