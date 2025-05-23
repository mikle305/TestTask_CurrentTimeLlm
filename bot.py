from datetime import datetime, timezone
from langgraph.graph import MessageGraph
from langchain.schema import HumanMessage
from groq import Groq
import os
from dotenv import load_dotenv


class Bot:
    def __init__(self, api_key: str, model_name: str):
        self.model_name = model_name
        self.llm = Groq(api_key=api_key)
        self.graph = MessageGraph()
        self.app = None

    def run(self):
        self.graph.add_node("process", self.process_message)
        self.graph.set_entry_point("process")
        self.graph.set_finish_point("process")
        self.app = self.graph.compile()

    def send_message(self, message: str) -> str:
        formatted_message = [HumanMessage(content=message)]
        response = self.app.invoke(formatted_message)
    
        clean_response = response[1].content.split('</think>')[-1].strip()
        clean_response = " ".join(clean_response.split()).strip()
        return clean_response

    def process_message(self, messages: list[HumanMessage]) -> str:
        last_msg = messages[-1].content
        message_words = last_msg.lower().split()
        if "time" in message_words:
            time_data = self.get_current_time()
            return f"Current UTC time: {time_data['utc']}"

        return self.groq_chat_completion(last_msg)

    def groq_chat_completion(self, user_query: str) -> str:
        response = self.llm.chat.completions.create(
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Keep responses short."},
                {"role": "user", "content": user_query}
            ],
            model = self.model_name,
        )
        return response.choices[0].message.content

    def get_current_time(self) -> dict:
        """Return the current UTC time in ISO-8601 format."""
        now = datetime.now(timezone.utc).isoformat()
        return {"utc": now.replace("+00:00", "Z")}


def run_interaction_loop(bot: Bot) -> None:
    print("Chat Bot activated. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Bot: Goodbye!")
                break
                
            response = bot.send_message(user_input)
            print(f"Bot: {response}")
        except KeyboardInterrupt:
            print("\nBot: Session interrupted.")
            break
        except Exception as e:
            print(f"Bot: Error occurred - {str(e)}")


def main():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    model_name = "deepseek-r1-distill-llama-70b"
    bot = Bot(api_key, model_name)
    bot.run()
    run_interaction_loop(bot)


if __name__ == "__main__":
    main()
