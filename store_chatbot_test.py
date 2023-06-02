from store_chatbot import StoreChatBot
import sys

DEFAULT_OPENAI_API_KEY = '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'

# Parse command line arguments
args = dict(arg.split('=') for arg in sys.argv[1:])
faq_embedding_store = args.get("--faq-embedding-store", 'faq-embeddings-store')
prod_embedding_store = args.get(
    "--prod-embedding-store", 'prod-embeddings-store')
openai_api_key = args.get("--openai-api-key", DEFAULT_OPENAI_API_KEY)

store_chat_bot = StoreChatBot(
    prod_embedding_store=prod_embedding_store,
    faq_embedding_store=faq_embedding_store,
    openai_api_key=openai_api_key,
    verbose=True
)


#print(store_chat_bot.answer("How to return an item?"))
#print(store_chat_bot.answer("How to return a product?"))
#print(store_chat_bot.answer("How do I reset my password?"))
#print(store_chat_bot.answer("Do you have any asus laptop?"))
print(store_chat_bot.answer("How are you?"))
#print(store_chat_bot.answer("I want to book a flight from SEA to TPA for tomorrow."))
#print(store_chat_bot.answer("In what cases, I can not return an item?"))