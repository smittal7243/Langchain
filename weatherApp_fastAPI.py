from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
os.environ["OPENWEATHERMAP_API_KEY"] = os.environ.get("OPENWEATHERMAP_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow
workflow = Graph()
weather = OpenWeatherMapAPIWrapper()
model = ChatOpenAI(temperature=0)

# Define the functions for the workflow
def function_1(state):
    messages = state['messages']
    user_input = messages[-1]
    complete_query = "Your task is to provide only the city name based on the user query. \
                    Nothing more, just the city name mentioned. Following is the user query: " + user_input
    response = model.invoke(complete_query)
    state['messages'].append(response.content)  # appending AIMessage response to the AgentState
    return state

def function_2(state):
    messages = state['messages']
    agent_response = messages[-1]
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

workflow.add_node("agent", function_1)
workflow.add_node("tool", function_2)
workflow.add_node("responder", function_3)

workflow.add_edge('agent', 'tool')
workflow.add_edge('tool', 'responder')

workflow.set_entry_point("agent")
workflow.set_finish_point("responder")

app_workflow = workflow.compile()

# Pydantic model for request body
class CityRequest(BaseModel):
    city: str

@app.post('/weather')
async def get_weather(request: CityRequest):
    city_name = request.city
    if not city_name:
        raise HTTPException(status_code=400, detail="City name is required")

    state = {'messages': [city_name]}  # Initialize state with the city name
    temperature_info = app_workflow.invoke(state)

    # Extract the first sentence
    first_sentence = temperature_info.split('.')[0] + '°C.'

    # Extract temperature in Celsius
    start_index = first_sentence.find("temperature of ") + len("temperature of ")
    end_index = first_sentence.find("°C")
    temperature_celsius = float(first_sentence[start_index:end_index])

    # Convert Celsius to Fahrenheit
    temperature_fahrenheit = (temperature_celsius * 9/5) + 32

    response = {
        'city': city_name,
        'temperature_fahrenheit': temperature_fahrenheit,
        'message': f"The weather in {city_name} is around {temperature_fahrenheit:.2f}°F."
    }

    return response

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index2.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
