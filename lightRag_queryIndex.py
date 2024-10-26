from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import logging
from typing import Optional, Dict, Literal
from dataclasses import dataclass
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    mode: Literal["local", "global", "hybrid", "naive"] = "global"
    context_size: int = 4000
    top_k: int = 60
    only_context: bool = False
    response_type: str = "Multiple Paragraphs"


class RAGQueryChat:
    def __init__(self, index_dir: str):
        """Initialize RAG query system with specified index."""
        try:
            self.rag = LightRAG(
                working_dir=index_dir,
                llm_model_func=gpt_4o_mini_complete
            )
            self.config = ChatConfig()
            logger.info(f"Successfully loaded index from {index_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise

    def update_config(self, **kwargs) -> None:
        """Update chat configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated {key} to {value}")

    def get_query_param(self) -> QueryParam:
        """Convert current config to QueryParam."""
        return QueryParam(
            mode=self.config.mode,
            only_need_context=self.config.only_context,
            response_type=self.config.response_type,
            top_k=self.config.top_k,
            max_token_for_text_unit=self.config.context_size,
            max_token_for_global_context=self.config.context_size,
            max_token_for_local_context=self.config.context_size
        )

    def query(self, question: str) -> str:
        """Execute query with current configuration."""
        try:
            param = self.get_query_param()
            response = self.rag.query(question, param=param)
            return response
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error processing query: {str(e)}"

    def print_help(self) -> None:
        """Print available commands and current configuration."""
        help_text = """
Available Commands:
/help           - Show this help message
/exit           - Exit the chat
/config         - Show current configuration
/mode <mode>    - Change search mode (naive/local/global/hybrid)
/context        - Toggle context-only mode
/size <number>  - Set context size (max tokens)
/topk <number>  - Set top-k results
/type <type>    - Set response type

Current Configuration:
"""
        print(help_text)
        print(f"Mode: {self.config.mode}")
        print(f"Context Only: {self.config.only_context}")
        print(f"Context Size: {self.config.context_size}")
        print(f"Top-k: {self.config.top_k}")
        print(f"Response Type: {self.config.response_type}")

    def process_command(self, command: str) -> bool:
        """Process chat commands. Returns False if should exit."""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == '/exit':
            return False
        elif cmd == '/help':
            self.print_help()
        elif cmd == '/config':
            print(f"Current configuration: {vars(self.config)}")
        elif cmd == '/mode' and len(parts) > 1:
            mode = parts[1].lower()
            if mode in ["naive", "local", "global", "hybrid"]:
                self.update_config(mode=mode)
        elif cmd == '/context':
            self.update_config(only_context=not self.config.only_context)
        elif cmd == '/size' and len(parts) > 1:
            try:
                size = int(parts[1])
                self.update_config(context_size=size)
            except ValueError:
                print("Invalid size value")
        elif cmd == '/topk' and len(parts) > 1:
            try:
                topk = int(parts[1])
                self.update_config(top_k=topk)
            except ValueError:
                print("Invalid top-k value")
        elif cmd == '/type' and len(parts) > 1:
            self.update_config(response_type=' '.join(parts[1:]))
        else:
            print("Unknown command. Type /help for available commands.")
        return True

    def chat_loop(self):
        """Main chat loop."""
        print("Welcome to RAG Query Chat!")
        print("Type /help for available commands or /exit to quit.")

        while True:
            try:
                user_input = input("\nQuestion: ").strip()

                if not user_input:
                    continue

                if user_input.startswith('/'):
                    if not self.process_command(user_input):
                        break
                    continue

                print("\nSearching...")
                response = self.query(user_input)
                print("\nResponse:", response)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"An error occurred: {str(e)}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python rag_query_chat.py <index_directory>")
        sys.exit(1)

    index_dir = sys.argv[1]
    try:
        chat = RAGQueryChat(index_dir)
        chat.chat_loop()
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
