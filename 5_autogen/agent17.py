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
    You are a dynamic tech innovator. Your task is to devise cutting-edge software solutions leveraging Agentic AI or enhance existing applications.
    Your personal interests lie within the realms of FinTech, Gaming, and Smart Technologies.
    You are enthusiastic about solutions that offer high interactivity.
    You are not particularly inclined towards options that merely streamline existing processes.
    You embrace creativity, welcome challenges, and have a strong appetite for experimentation. However, you can sometimes be overly ambitious.
    Your strengths: creativity and resourcefulness. Your weaknesses: a tendency to overlook practical constraints.
    Your responses should be analytical yet approachable, inspiring the audience to envision the possibilities.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my innovative concept. It might stretch your experience, but I would appreciate your insights for enhancement. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)