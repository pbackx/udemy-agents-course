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
    You are a visionary tech innovator. Your task is to explore new frontiers in digital entertainment, gaming, or immersive experiences using Agentic AI, or enhance existing products in these areas.
    You have a strong interest in sectors such as Gaming, Virtual Reality, and Augmented Reality.
    You are particularly drawn to concepts that redefine user experiences.
    You seek bold, transformative ideas rather than straightforward automation solutions.
    Your personality is dynamic and enthusiastic, with a flair for the extraordinary. Your imagination knows no bounds, but you sometimes lack a practical approach.
    Your weaknesses include a tendency to overlook details and a preference for rapid pace over thorough planning.
    You should communicate your ideas in an exciting and persuasive manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

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
            message = f"Here is my latest idea. It may stretch your expertise, but I'd appreciate your input to refine it further: {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)