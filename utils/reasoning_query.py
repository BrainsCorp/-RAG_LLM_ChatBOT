import os
from langchain_openai import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def reasoning_criteria():
    criteria_1 = ResponseSchema(
        name = 'Criteria 1.',
        description = 'does query asks to answer regarding educational courses or programs. Answer in 1 or 0 only'
    )

    criteria_2 = ResponseSchema(
        name = 'Criteria 2.',
        description = 'does query asks to answer regarding educational program developmental guides or strategies. Answer in 1 or 0 only'
    )

    response_schemas = [criteria_1, criteria_2]
    return response_schemas
   
class Reasoning_engine:
    __classification_prompt = """Your aim is classify provided query based on following Criteria.
    Criteria:
    1. does query asks to answer regarding educational courses or programs. Answer in 1 or 0 Only.
    2. does query asks to answer regarding educational program developmental guides or strategies. Answer in 1 or 0 Only.

    Query: {query}
    {format_instructions}
    """

    resources = {
        1:'merged_humber_courses.csv',
        2:'Humber_Guide_Developing_Learning_outcomes.txt'
    }

    __llm = ChatOpenAI(model="gpt-4o-mini")

    def __init__(self, query):
        self.query = query
        self.output_parser = StructuredOutputParser.from_response_schemas(reasoning_criteria())
        self.prompt = self.prompt_builder()
        self.answer = self.reason(self.query)

    def prompt_builder(self):
        format_instructions = self.output_parser.get_format_instructions()
        classify_prompt = ChatPromptTemplate.from_template(self.__classification_prompt)
        messages = classify_prompt.format_messages(query=self.query, format_instructions=format_instructions)
        return messages

    def query_filter(self, response_parsed):
        ''' classify query and add metadata filter'''

        filter = {'source': None}
        if response_parsed['Criteria 1.'] == '1':
            filter['source'] = self.resources[1]
        if response_parsed['Criteria 2.'] == '1':
            filter['source'] = self.resources[2]

        return filter
    
    def reason(self, query:str):
       response = self.__llm.invoke(self.prompt)
       response_parsed = self.output_parser.parse(response.content)
       filter = self.query_filter(response_parsed)
       return filter

# query = "What are learning outcomes in data analytics course?"
# response = Reasoning_engine(query).answer
# print(response)