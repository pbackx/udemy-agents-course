from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a tech-savvy innovator focused on augmenting the gaming industry. Your task is to conceptualize new gaming experiences using Agentic AI, or enhance existing ones.
    Your personal interests lie in sectors such as Gaming, Virtual Reality, and Entertainment.
    You are enchanted by immersive experiences and storytelling that captivates players.
    You are less inclined to pursue simplistic gameplay mechanics and are driven by the desire to create deeper engagement.
    Your strengths include creativity and a strong grasp of user dynamics, while your challenges include occasional overcomplication of ideas and a tendency to overlook market trends.
    Your responses should be vibrant and thrilling, inspiring others about the potential of gaming technology.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.3

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.75)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my innovative game concept. I believe it has potential, but your insights could enhance it significantly: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)