# langchain-openai-examples
A very simple implementation of using chatgpt model to:
- route a request to a tool with input extracted from a user message.
- generate a response using observation (output from a tool).

## data/:
t-2.csv is a scaped product catalog of a toy e-commerce website.
taobao-shop1.csv is scraped product catalog from a taobao shop.

## Prepare data stores for retrieval
- python prepare-prod-store.py --prod-csv=data/t-2.csv --prod-embedding-store=run/prod-embeddings
- python prepare-faq-store.py --faq-dir=CustomerSupportFAQs --faq-embedding-store=run/faq-embeddings

## custom-prod-agent:
- python store-chat.py --prod-embedding-store=run/prod-embeddings --faq-embedding-store=run/faq-embeddings
