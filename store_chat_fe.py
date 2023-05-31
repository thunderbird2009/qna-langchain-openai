import gradio as gr
import random
import time
import sys
from store_chatbot import StoreChatBot

DEFAULT_OPENAI_API_KEY = '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'

# Parse command line arguments
args = sys.argv
print(f"arguments: {args}")
args = dict(arg.split('=') for arg in sys.argv[1:])
faq_embedding_store = args.get("--faq-embedding-store", 'faq-embeddings-store')
prod_embedding_store = args.get(
    "--prod-embedding-store", 'prod-embeddings-store')
openai_api_key = args.get("--openai-api-key", DEFAULT_OPENAI_API_KEY)

store_chat_bot = StoreChatBot(
    prod_embedding_store=prod_embedding_store,
    faq_embedding_store=faq_embedding_store,
    openai_api_key=openai_api_key
)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        bot_message = store_chat_bot.answer(message)
        chat_history.append((message, bot_message["output"]))
        time.sleep(1)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()