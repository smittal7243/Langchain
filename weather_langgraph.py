from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
os.environ["OPENWEATHERMAP_API_KEY"] = os.environ.get("OPENWEATHERMAP_API_KEY")

workflow = Graph()
weather = OpenWeatherMapAPIWrapper()

# Set the model as ChatOpenAI
model = ChatOpenAI(temperature=0)

# Define the functions for the workflow
def function_1(state):
    messages = state['messages']
    user_input = messages[-1]
    complete_query = "Your task is to provide only the city name based on the user query. \
                    Nothing more, just the city name mentioned. Following is the user query: " + user_input
    response = model.invoke(complete_query)
    state['messages'].append(response.content) 
    return state

def function_2(state):
    messages = state['messages']
    agent_response = messages[-1]
    weather = OpenWeatherMapAPIWrapper()
    weather_data = weather.run(agent_response)
    state['messages'].append(weather_data)
    return state

def function_3(state):
    messages = state['messages']
    user_input = messages[0]
    available_info = messages[-1]
    agent2_query = "Your task is to provide info concisely based on the user query and the available information from the internet. \
                        Following is the user query: " + user_input + " Available information: " + available_info
    response = model.invoke(agent2_query)
    return response.content

workflow = Graph()

workflow.add_node("agent", function_1)
workflow.add_node("tool", function_2)
workflow.add_node("responder", function_3)

workflow.add_edge('agent', 'tool')
workflow.add_edge('tool', 'responder')

workflow.set_entry_point("agent")
workflow.set_finish_point("responder")

app = workflow.compile()

city_name = input("Enter the city name: ")
state = {'messages': [city_name]} 
temperature_info = app.invoke(state)


first_sentence = temperature_info.split('.')[0] + '°C.'

# Extract temperature in Celsius
start_index = first_sentence.find("temperature of ") + len("temperature of ")
end_index = first_sentence.find("°C")
temperature_celsius = float(first_sentence[start_index:end_index])

# print("Celcius: ", temperature_celsius)
# Convert Celsius to Fahrenheit
temperature_fahrenheit = (temperature_celsius * 9/5) + 32

# print("Fahrenheit: ", temperature_fahrenheit)

# Replace Celsius with Fahrenheit in the first sentence
# first_sentence_fahrenheit = first_sentence.replace(f"{temperature_celsius}°C", f"{temperature_fahrenheit}°F")

# print(first_sentence_fahrenheit)
print("The weather in ", city_name, "is around ", temperature_fahrenheit, "°F.")