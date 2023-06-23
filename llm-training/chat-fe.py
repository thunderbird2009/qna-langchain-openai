import gradio as gr
import random
import time
import sys
from string import Template
from peft import PeftConfig
from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

#model_name = "EleutherAI/gpt-neox-20b"
#model_name = "THUDM/chatglm-6b"
model_name = "tiiuae/falcon-7b-instruct"

#Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

import os 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

PEFT_MODEL = "experiments/model"
config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, return_dict=True, quantization_config=quant_config, device_map={"":0}, trust_remote_code=True)
model = PeftModel.from_pretrained(model, PEFT_MODEL)


def bot_answer(model, question):
    bot_answer.prompt_tmpl = Template(f"""
<human>: $question
<assistant>:
""".strip())
    
    gen_config = model.generation_config
    gen_config.max_new_tokens = 200
    gen_config.temperature = 0.7
    gen_config.top_p = 0.7
    gen_config.num_return_sequences = 1
    gen_config.pad_token_id = tokenizer.eos_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id

    device = "cuda:0"
    prompt = bot_answer.prompt_tmpl.substitute(question=question)
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids = encoding.input_ids,
            use_cache=True,
            do_sample=True,
            num_beams=1,
            num_beam_groups=1,
            generation_config = gen_config
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


#DEFAULT_OPENAI_API_KEY = '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'

# Parse command line arguments
# args = sys.argv
# print(f"arguments: {args}")
# args = dict(arg.split('=') for arg in sys.argv[1:])
# faq_embedding_store = args.get("--faq-embedding-store", 'faq-embeddings-store')
# prod_embedding_store = args.get(
#     "--prod-embedding-store", 'prod-embeddings-store')
# openai_api_key = args.get("--openai-api-key", DEFAULT_OPENAI_API_KEY)

# store_chat_bot = StoreChatBot(
#     prod_embedding_store=prod_embedding_store,
#     faq_embedding_store=faq_embedding_store,
#     openai_api_key=openai_api_key
# )

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Demo of a Chatbot using different LLMs. """)

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        bot_message = bot_answer(model, message)
        chat_history.append((message, bot_message))
        time.sleep(1)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860)
#demo.launch()