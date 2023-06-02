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
    gr.Markdown(
        """
    # Demo of a Web Store ChatBot based on LLM
    An experimental prototype of QnA bot using ChatGPT model to demonstrate the following pattern:

    ## User <---> Agent: (Rule-based-Agent, LLM-Agent) <---> Tools

    An Agent is used to detect user intents and extract intent-related entities from the
     input of user message and intermediate output (initially none), and select tools to process 
     them as intermediary steps. The outputs from tools are appended to the intermediary output
     to be processed by the Agent. This process repeats until the Agent decides to
     finalize an output message to the user.

    The Agent consists of two components:
    - A Rule-based Agent that process inputs and generates outputs based on pre-specified rules
    using traditional programming.
    - An LLM Agent that generates output from input with prompt, which controls the agent behavior.
 
    ### Tools
    There are current two tools implemented for demo purpose:
    - A tool of semantic search on a product catalog scraped from 
    https://www.webscraper.io/test-sites/e-commerce/allinone/
    - A tool of semantic search on a set of frequently-asked-question related to web stores. 
    This dataset is opensourced by EBay.

    Both tools use very small datasets and are meant to demo the basic architecture of the chatbot only.
    
    The Chatbot limit itself to the scope of those two tools and will not answer questions outside
    the scope.

    ### Limitation on Conversations
    - There is currently no session memory implemented yet. The chatbot does not consider
    previous messages from the user when answering the current message. This will be
    addressed soon.
    - The chatbot currently does not do greetings or small talks. This can be addressed
    """
    )

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

demo.launch(server_name="0.0.0.0", server_port=7860)
#demo.launch()