from store_chatbot import StoreChatBot
from store_chatbot2 import StoreChatBot2
import sys
import argparse

DEFAULT_OPENAI_API_KEY = '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'

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose mode')
parser.add_argument('--faq-embedding-store', type=str, help='FAQ embedding store')
parser.add_argument('--prod-embedding-store', type=str, help='Products embedding store')
parser.add_argument('--mode', type=int, default=1, help='Products embedding store')
parser.add_argument('--openai-api-key', type=str, default=DEFAULT_OPENAI_API_KEY, help='openAI api key')

# Parse the command-line arguments
args = parser.parse_args()

# args = dict(arg.split('=') for arg in sys.argv[1:])
# faq_embedding_store = args.get("--faq-embedding-store", 'faq-embeddings-store')
# prod_embedding_store = args.get(
#     "--prod-embedding-store", 'prod-embeddings-store')
# verbose = args.get(
#     "--verbose", 'prod-embeddings-store')
# openai_api_key = args.get("--openai-api-key", DEFAULT_OPENAI_API_KEY)

if args.mode == 1:
    store_chat_bot = StoreChatBot(
        prod_embedding_store=args.prod_embedding_store,
        faq_embedding_store=args.faq_embedding_store,
        openai_api_key=args.openai_api_key,
        verbose=args.verbose
    )
else:
    store_chat_bot = StoreChatBot2(
        prod_embedding_store=args.prod_embedding_store,
        faq_embedding_store=args.faq_embedding_store,
        openai_api_key=args.openai_api_key,
        verbose=args.verbose
    )

print(store_chat_bot.answer("I am looking for a cheap laptop"))
print(store_chat_bot.answer("How to return a product?"))
#print(store_chat_bot.answer("How do I reset my password?"))
print(store_chat_bot.answer("Do you have any asus laptop?"))
print(store_chat_bot.answer("How are you?"))
#print(store_chat_bot.answer("How are you today?"))
print(store_chat_bot.answer("I want to book a flight from SEA to TPA for tomorrow."))
print(store_chat_bot.answer("Are you a bot or a human?"))
#print(store_chat_bot.answer("In what cases, I can not return an item?"))

while True:
    sys.stdout.write('user: ')
    sys.stdout.flush()
    input_line = sys.stdin.readline().strip()

    # Process the input
    output = store_chat_bot.answer(input_line)

    # Output the result to stdout
    sys.stdout.write(f'bot: {output}\n')