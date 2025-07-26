from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

# Add a prompt to control response style
template = """
You are a helpful assistant.
Keep your responses short, friendly, and under 2 sentences.

Current conversation:
{history}
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
)

# Conversation with memory and prompt
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)

print("🤖 Groq Conversational Agent with Memory (Short Replies)")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Agent: Goodbye! 👋")
        break
    response = conversation.predict(input=user_input)
    print(f"Agent: {response}\n")
