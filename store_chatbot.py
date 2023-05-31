from langchain.callbacks.manager import Callbacks
from typing import Tuple, Any
import re
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from typing import Optional, Type, Any
import json
import sys
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

DEFAULT_OPENAI_API_KEY = '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'

###########################  define my own tools #################################################

class ProdSearchTool(BaseTool):
    name = "prod_search"
    description = """Search product catalog. 
    Input: product name, category and description, etc. 
    Output: a list of products in JSon. Each product has the following fields: 
        category, subcategory, subcategory-link, product-link, name, price, description.
        Compose a snippet with product-link as a href for each product to show to customer.
    """
    docsearch: Optional[FAISS] = None

    def __init__(self, prod_embedding_store, embeddings, **data: Any) -> None:
        super().__init__(**data)
        self.docsearch = FAISS.load_local(
            folder_path=prod_embedding_store, embeddings=embeddings)

    def findProds(self, query) -> str:
        data_list = self.docsearch.similarity_search(query)
        # Define a mapping of old keys to new keys (including the NULL mappings)
        key_mapping = {
            'category-links': 'category',
            'subcategory-links': 'subcategory',
            'subcategory-links-href': 'subcategory-link',
            'product-links-href': 'product-link',
            'name': 'name',
            'price': 'price',
            'description': 'description'
        }
        updated_items = []
        for item in data_list:
            updated_item = {key_mapping.get(
                key, key): value for key, value in item.metadata.items() if key in key_mapping}
            updated_items.append(updated_item)

        json_data = {'type': 'prod_list', 'products': updated_items}
        return json.dumps(json_data)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.findProds(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("ProdSearchTool does not support async")


class CustServiceTool(BaseTool):
    name = "customer_service"
    description = """General customer service that handles questions about users' store experience,
    such as account, user profile, order, payment, shipment, etc.
    Input: customer request.
    Output: Text relevant to the question."""
    faqsearch: Optional[FAISS] = None

    def __init__(self, faq_embedding_store, embeddings) -> None:
        super().__init__()
        self.faqsearch = FAISS.load_local(
            folder_path=faq_embedding_store, embeddings=embeddings)

    def findFAQs(self, query) -> str:
        data_list = self.faqsearch.similarity_search_with_relevance_scores(
            query, k=1)
        if len(data_list) == 0 or data_list[0][1] < 0.5:
            json_data = {'type': 'final_msg',
                         'msg': 'Have not found an answer from our knowledge base. Will redirect you to an agent.'}
            return json.dumps(json_data)
        else:
            doc = data_list[0][0]
            json_data = {
                'type': 'kb_src', 'src': doc.metadata["source"], 'context': doc.page_content}
            return json.dumps(json_data)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.findFAQs(query)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CustServiceTool does not support async")


class DefaultTool(BaseTool):
    name = "default_tool"
    description = """Default tool to handle all questions or requests that can not be handled by
    other tools.
    Input: customer request.
    Output: final answer."""
    faqsearch: Optional[FAISS] = None

    DEFAULT_TOOL_MSG = """I am a chatbot that can handle product and store customer service questions. 
    Your question seems to be outside my scope. Could you rephrase it for me to understand better, 
    or ask a different question? Thx!
    """
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        json_data = {'type': 'final_msg', 'msg': self.DEFAULT_TOOL_MSG}
        return json.dumps(json_data)

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CustServiceTool does not support async")


# Set up a prompt template

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class MyLLMSingleActionAgent(LLMSingleActionAgent):
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if len(intermediate_steps) > 0:
            obs = json.loads(intermediate_steps[-1][1])
            if obs['type'] == 'prod_list':
                return AgentFinish(
                    return_values={"output": f'I found the following products:\n{json.dumps(obs["products"])}\n'}, log=""
                )
            elif obs['type'] == 'final_msg':
                return AgentFinish(return_values={"output": obs['msg']}, log="")
            elif obs['type'] == 'kb_src':
                intermediate_steps[-1] = [intermediate_steps[-1]
                                          [0], obs['context']]
                output = self.llm_chain.run(
                    intermediate_steps=intermediate_steps,
                    stop=self.stop,
                    callbacks=callbacks,
                    **kwargs,
                )
                output = output + f'\nsources:\n{obs["src"]}'
                return self.output_parser.parse(output)

        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)


class StoreChatBot:
    # Set up the base template
    template = """You are a chatbot of a Web store to answer customer questions. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Please also follow following rules regarding final answer:
    If the user's message is not a question or request, try to lead the conversation to a question by giving a Final Answer like
        "How are you. How can I help you today".
    If the user's question/request together with all the intermediary tuples of (Action, Action Input, Observation) 
        can not be address by any Action and you don't have a logical final answer either,
        give a Final Answer like "I don't seem to have an answer. Please rephrase the question for me to try again".

    Begin!

    Question: {input}
    {agent_scratchpad}"""

    def __init__(self, prod_embedding_store, faq_embedding_store, openai_api_key):
        # Get your embeddings engine ready
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        cst = CustServiceTool(faq_embedding_store, embeddings)
        #cst.init(faq_embedding_store, embeddings)
        pst = ProdSearchTool(prod_embedding_store, embeddings)
        tools = [pst, cst, DefaultTool()]
        tool_names = [tool.name for tool in tools]
        prompt = CustomPromptTemplate(
            template=self.template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )
        print(prompt)
        llm = OpenAI(temperature=0)
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        output_parser = CustomOutputParser()
        agent = MyLLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=False)

    def answer(self, user_msg) -> str:
        return self.agent_executor(user_msg)

