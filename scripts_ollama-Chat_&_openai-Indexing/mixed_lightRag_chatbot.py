from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import logging
import os
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    def __init__(self):
        self.index_path = None
        self.rag = None
        self.chat_history = []

        # Llama 3.2 3B specific configurations
        self.chunk_token_size = 512
        self.chunk_overlap_size = 50
        self.embedding_dim = 768  # nomic-embed-text dimension
        self.max_token_size = 8192

    def initialize_rag(self, index_path: str) -> bool:
        """Initialize RAG with specified index"""
        try:
            if not os.path.exists(index_path):
                logger.error(f"Index directory not found: {index_path}")
                return False

            self.index_path = index_path
            self.rag = LightRAG(
                working_dir=index_path,
                llm_model_func=ollama_model_complete,
                llm_model_name='llama3.2:3b',
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embedding_dim,
                    max_token_size=self.max_token_size,
                    func=lambda texts: ollama_embedding(
                        texts,
                        embed_model="nomic-embed-text"
                    )
                ),
                chunk_token_size=self.chunk_token_size,
                chunk_overlap_token_size=self.chunk_overlap_size
            )
            logger.info(f"Successfully loaded index from: {index_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            return False

    def search(self, query: str) -> Optional[str]:
        """Perform hybrid search with context"""
        try:
            if not self.rag:
                return "No index loaded. Please initialize first."

            # Create query parameters optimized for chat
            param = QueryParam(
                mode="hybrid",
                top_k=60,
                max_token_for_text_unit=4000,
                max_token_for_global_context=4000,
                max_token_for_local_context=4000,
                response_type="Conversational"
            )

            # Add chat history context if available
            if self.chat_history:
                context = "\n".join([f"User: {q}\nAssistant: {a}"
                                     for q, a in self.chat_history[-3:]])
                query = f"Previous context:\n{context}\n\nCurrent question: {query}"

            result = self.rag.query(query=query, param=param)

            if result:
                self.chat_history.append((query, result))
                return result
            return "I couldn't find relevant information to answer your question."

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"An error occurred: {str(e)}"


def list_available_indices(base_dir: str = "./rag_indices") -> list:
    """List all available indices"""
    if not os.path.exists(base_dir):
        return []
    return [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))]


def chat_interface():
    """Interactive chat interface"""
    chatbot = RAGChatbot()

    print("\nWelcome to RAG Chatbot!")
    print("=" * 50)

    # List available indices
    indices = list_available_indices()
    if not indices:
        print("No indices found. Please create an index first.")
        return

    print("\nAvailable indices:")
    for idx, index in enumerate(indices, 1):
        print(f"{idx}. {index}")

    # Select index
    while True:
        try:
            choice = input("\nSelect index number (or 'q' to quit): ")
            if choice.lower() == 'q':
                return

            index_num = int(choice) - 1
            if 0 <= index_num < len(indices):
                selected_index = indices[index_num]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Initialize RAG with selected index
    index_path = os.path.join("rag_indices", selected_index)
    if not chatbot.initialize_rag(index_path):
        print("Failed to initialize chatbot. Exiting.")
        return

    print(f"\nChatbot initialized with index: {selected_index}")
    print("\nYou can start chatting! (Type 'quit' to exit, 'clear' to clear history)")
    print("-" * 50)

    while True:
        try:
            query = input("\nYou: ").strip()

            if query.lower() == 'quit':
                break
            elif query.lower() == 'clear':
                chatbot.chat_history.clear()
                print("Chat history cleared!")
                continue
            elif not query:
                continue

            response = chatbot.search(query)
            print("\nAssistant:", response)
            print("-" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print("\nAn error occurred. Please try again.")


if __name__ == "__main__":
    try:
        chat_interface()
        print("\nThank you for using RAG Chatbot!")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        print("\nChat session ended.")
