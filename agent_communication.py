class AgentMessage:
    def __init__(self, sender: str, content: str, task: Dict):
        self.sender = sender
        self.content = content
        self.task = task

class AgentCommunicator:
    def __init__(self):
        self.messages = []

    async def send_message(self, sender: str, receiver: str, content: str, task: Dict):
        message = AgentMessage(sender, content, task)
        self.messages.append(message)
        agent = AgentRegistry.get_agent(receiver.lower().replace("_agent", ""))()
        return await agent.process_message(message) if hasattr(agent, "process_message") else None
