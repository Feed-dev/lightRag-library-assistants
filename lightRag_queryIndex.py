from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
import os


class SimpleRAGQuery:
    def __init__(self, base_dir: str = "./rag_indices"):
        self.base_dir = base_dir
        self.rag = None
        self.current_index = None

    def list_indices(self) -> list:
        return [d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))]

    def switch_index(self, index_name: str) -> bool:
        index_path = os.path.join(self.base_dir, index_name)
        if not os.path.exists(index_path):
            print(f"Index {index_name} does not exist")
            return False

        self.rag = LightRAG(
            working_dir=index_path,
            llm_model_func=gpt_4o_mini_complete
        )
        self.current_index = index_name
        return True

    def query(self, question: str) -> str:
        try:
            response = self.rag.query(
                question,
                param=QueryParam(
                    mode="hybrid",
                    max_token_for_text_unit=4000
                )
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def run(self):
        while True:
            # List available indices
            print("\nAvailable indices:")
            indices = self.list_indices()
            for idx, index in enumerate(indices, 1):
                print(f"{idx}. {index}")

            # Select index
            if not self.current_index:
                choice = input("\nSelect index number (or 'quit' to exit): ")
                if choice.lower() == 'quit':
                    break

                try:
                    index_num = int(choice) - 1
                    if 0 <= index_num < len(indices):
                        self.switch_index(indices[index_num])
                    else:
                        print("Invalid index number")
                        continue
                except ValueError:
                    print("Please enter a valid number")
                    continue

            # Query loop
            while self.current_index:
                print(f"\nCurrent index: {self.current_index}")
                question = input("Enter your question (or 'switch'/'quit'): ")

                if question.lower() == 'quit':
                    return
                elif question.lower() == 'switch':
                    self.current_index = None
                    break
                elif question.strip():
                    print("\nSearching...")
                    response = self.query(question)
                    print("\nResponse:", response)


if __name__ == "__main__":
    rag_query = SimpleRAGQuery()
    rag_query.run()
