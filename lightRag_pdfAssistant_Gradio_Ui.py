import gradio as gr
import os
import logging
from typing import Optional, Tuple
from lightRag_pdfLibrary_indexer import LightRAGManager
from lightRag_chatbot import RAGChatbot

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LightRAGUI:
    def __init__(self):
        self.rag_manager = LightRAGManager()
        self.chatbot = RAGChatbot()
        self.current_index = None
        self.selected_files = []

    def list_indices(self) -> list:
        return self.rag_manager.list_indices()

    def handle_index_selection(self, value: str) -> tuple[str, str]:
        """Handle selection of existing index."""
        try:
            if self.rag_manager.switch_index(value):
                self.current_index = value
                return value, f"Switched to index: {value}"
            return value, f"Failed to switch to index: {value}"
        except Exception as e:
            if not value:  # Handle None or empty string
                return "", "Please select an index"
            logger.error(f"Error in handle_index_selection: {e}")
            return value, f"Error: {str(e)}"

    def create_new_index(self, name: str) -> tuple[list, str, str]:
        """Create a new index and return updated choices."""
        try:
            if not name:
                return self.list_indices(), "", "Please enter a name for the new index"

            if self.rag_manager.create_index(name):
                indices = self.list_indices()
                return (
                    indices,  # Updated choices for dropdown
                    "",  # Clear the input field
                    f"Successfully created index: {name}. You can now select it from the dropdown."  # Status message
                )
            return (
                self.list_indices(),
                name,
                f"Failed to create index: {name}"
            )
        except Exception as e:
            logger.error(f"Error in create_new_index: {e}")
            return self.list_indices(), name, f"Error: {str(e)}"

    def initialize_chat_index(self, index_name: str) -> str:
        """Initialize chatbot with selected index"""
        try:
            if not index_name:
                return "Please select an index first"

            index_path = os.path.join("./rag_indices", index_name)
            if self.chatbot.initialize_rag(index_path):
                return f"Chat initialized with index: {index_name}"
            return "Failed to initialize chat with selected index"
        except Exception as e:
            return f"Error initializing chat: {str(e)}"

    def handle_file_selection(self, files) -> tuple[str, list]:
        self.selected_files = files if isinstance(files, list) else [files]
        return f"Selected {len(self.selected_files)} files/directories", self.selected_files

    def process_selected_files(self) -> str:
        if not self.current_index:
            return "Please select or create an index first"

        if not self.selected_files:
            return "Please select files or directories to process"

        success_count = 0
        failed_count = 0

        try:
            for file in self.selected_files:
                if os.path.isfile(file.name) and file.name.lower().endswith('.pdf'):
                    if self.rag_manager.incremental_insert_pdf(file.name):
                        success_count += 1
                    else:
                        failed_count += 1
                elif os.path.isdir(file.name):
                    try:
                        self.rag_manager.batch_insert_pdfs(file.name)
                        success_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Error processing directory {file.name}: {e}")
                        continue

            return f"Processing complete: {success_count} successful, {failed_count} failed"
        except Exception as e:
            logger.error(f"Error in process_selected_files: {e}")
            return f"Error during processing: {str(e)}"

    def chat_query(self, message: str, history: list) -> tuple:
        if not self.current_index:
            return "", [{"role": "user", "content": message},
                        {"role": "assistant", "content": "Please select or create an index first"}]

        try:
            response = self.chatbot.search(message)
            return "", [{"role": "user", "content": message},
                        {"role": "assistant", "content": response}]
        except Exception as e:
            logger.error(f"Error in chat_query: {e}")
            return "", [{"role": "user", "content": message},
                        {"role": "assistant", "content": f"Error: {str(e)}"}]


def create_ui():
    ui = LightRAGUI()

    with gr.Blocks(title="LightRAG PDF Assistant") as demo:
        gr.Markdown("# LightRAG PDF Assistant")

        with gr.Tab("Index Management"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Select Existing Index")
                    index_dropdown = gr.Dropdown(
                        choices=ui.list_indices(),
                        label="Select Index",
                        interactive=True,
                        allow_custom_value=True
                    )
                    index_select_output = gr.Textbox(label="Selection Status", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### Create New Index")
                    new_index_input = gr.Textbox(
                        label="New Index Name",
                        placeholder="Type new index name here..."
                    )
                    create_index_btn = gr.Button("Create New Index")
                    create_index_output = gr.Textbox(label="Creation Status", interactive=False)

            gr.Markdown("### Process Files")
            with gr.Row():
                upload_button = gr.File(
                    label="Drop PDF files/folders or click to upload",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                    height=200
                )

            with gr.Row():
                selected_files = gr.Textbox(label="Selected Files", interactive=False)
                process_btn = gr.Button("Start Processing", variant="primary")
                process_output = gr.Textbox(label="Processing Status", interactive=False)

        with gr.Tab("Chat"):
            with gr.Row():
                chat_index_dropdown = gr.Dropdown(
                    choices=ui.list_indices(),
                    label="Select Index for Chat",
                    interactive=True
                )
                init_status = gr.Textbox(label="Initialization Status", interactive=False)

            chatbot = gr.Chatbot(height=500, type="messages")
            with gr.Row():
                msg = gr.Textbox(label="Message", placeholder="Type your message here...")
                send = gr.Button("Send")
                clear = gr.Button("Clear Chat")

        def handle_create_index(name):
            with gr.Row():
                gr.Markdown("Initializing new index...")
            indices, input_value, message = ui.create_new_index(name)
            return {
                index_dropdown: gr.update(choices=indices, value=None),  # Don't auto-select
                new_index_input: gr.update(value=""),
                create_index_output: message + "\nPlease select the new index from the dropdown to start using it."
            }

        def safe_chat_query(message: str, history: list) -> tuple:
            if not ui.chatbot.rag:
                return "", [{"role": "user", "content": message},
                            {"role": "assistant", "content": "Please select and initialize an index first"}]
            return ui.chat_query(message, history)

        # Event handlers
        index_dropdown.change(
            fn=ui.handle_index_selection,
            inputs=index_dropdown,
            outputs=[index_dropdown, index_select_output]
        )

        create_index_btn.click(
            fn=handle_create_index,
            inputs=new_index_input,
            outputs=[
                index_dropdown,
                new_index_input,
                create_index_output
            ]
        ).then(
            fn=lambda: gr.update(choices=ui.list_indices()),
            outputs=index_dropdown
        )

        upload_button.upload(
            fn=ui.handle_file_selection,
            inputs=upload_button,
            outputs=[selected_files, upload_button]
        )

        process_btn.click(
            fn=ui.process_selected_files,
            inputs=[],
            outputs=process_output
        )

        chat_index_dropdown.change(
            fn=ui.initialize_chat_index,
            inputs=chat_index_dropdown,
            outputs=init_status
        )

        send.click(
            fn=safe_chat_query,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )

        msg.submit(
            fn=safe_chat_query,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )

        clear.click(lambda: None, None, chatbot, queue=False)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=False)
