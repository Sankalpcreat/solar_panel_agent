import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API client
client = OpenAI(api_key=openai_api_key)

# Tool definition
@tool
def compute_savings(monthly_cost: float) -> dict:
    """
    Computes potential savings with solar energy based on monthly electricity costs.
    """
    cost_per_kWh = 0.28  
    cost_per_watt = 1.50  
    sunlight_hours_per_day = 3.5  
    panel_wattage = 350  
    system_lifetime_years = 10  

    monthly_consumption_kWh = monthly_cost / cost_per_kWh
    daily_energy_production = monthly_consumption_kWh / 30
    system_size_kW = daily_energy_production / sunlight_hours_per_day
    number_of_panels = system_size_kW * 1000 / panel_wattage
    installation_cost = system_size_kW * 1000 * cost_per_watt
    annual_savings = monthly_cost * 12
    total_savings_10_years = annual_savings * system_lifetime_years
    net_savings = total_savings_10_years - installation_cost

    return {
        "number_of_panels": round(number_of_panels),
        "installation_cost": round(installation_cost, 2),
        "net_savings_10_years": round(net_savings, 2)
    }

# Utilities
def handle_tool_error(state) -> dict:
    """
    Handles errors during tool execution.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Creates a tool node with error handling.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# State definition
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State):
        latest_message = state["messages"][-1]
        result = self.runnable.invoke({"messages": [latest_message]})
        if result:
            state["messages"].append(SystemMessage(content=result["content"]))
        return {"messages": state["messages"]}

# Call OpenAI function
def call_openai(prompt: str) -> dict:
    """
    Calls the OpenAI API to process the prompt.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # Safely access the content
        content = response.choices[0].message.content if response.choices else "No response"
        return {"role": "assistant", "content": content}
    except Exception as e:
        return {"role": "assistant", "content": f"An error occurred: {str(e)}"}
class OpenAIRunnable(Runnable):
    def __init__(self, prompt_template: ChatPromptTemplate):
        self.prompt_template = prompt_template

    def invoke(self, state: State):
        serialized_messages = [
            {"role": "system", "content": msg.content}
            if isinstance(msg, SystemMessage)
            else {"role": "user", "content": msg.content}
            for msg in state["messages"]
        ]
        prompt = self.prompt_template.format(messages=serialized_messages)
        result = call_openai(prompt)
        return result

# Prompt template
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are a helpful assistant for Solar Panels Belgium.
            Gather the following:
            - Monthly electricity cost
            - Clarify if unsure
            Use tools after confirming all required details.
            '''
        ),
        ("placeholder", "{messages}"),
    ]
)

# Tools and Assistant Runnable
part_1_tools = [compute_savings]
part_1_assistant_runnable = OpenAIRunnable(primary_assistant_prompt)

# StateGraph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)