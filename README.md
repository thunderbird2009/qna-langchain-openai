# langchain-openai-examples
An experimental prototype of QnA bot using ChatGPT model to demonstrate the following pattern:
- Step 0: Receives a user input and calls LLM with it.
- Step 1: LLM identifies an intent and extract properties for the intent from the input.
- Step 2: Call a tool corresponding to the intent with the extracted properties.
- Step 3: Depends on the tool's output:
- - Either it is directly output to user.
- - Or it may be added to the intermediate output for next call to LLM together with the original user input.
- Step 4. LLM processes the complete input from step 3, and decides:
- - Either provides a final answer, that is output to user.
- - Or go to step 1, then the process repeats.

## Datasets
data/:
- t-2.csv is a scraped product catalog of a toy e-commerce website.
- taobao-shop1.csv is scraped product catalog from a taobao shop.

CustomerSupportFAQs/:  customer support FQAs in text format opensourced from EBay.

## Prepare data stores for retrieval
Generate an embedding vector store to support the tool of semantic search on products
- python prepare-prod-store.py --prod-csv=data/t-2.csv --prod-embedding-store=run/prod-embeddings

Generate an embedding vector store to support the tool of semantic search FAQs
- python prepare-faq-store.py --faq-dir=data/CustomerSupportFAQs --faq-embedding-store=run/faq-embeddings

## Run the chatbot
Run command line test
- python store_chat_test.py --prod-embedding-store=run/prod-embeddings --faq-embedding-store=run/faq-embeddings

Run a web frontend for chatbot
- python .\store_chat_fe.py --prod-embedding-store=run/prod-embeddings --faq-embedding-store=run/faq-embeddings
