# AI Agents

A collection of AI agents with conversation memory and web search capabilities that represent a typical product team.

## Features

- Product Manager Agent with expertise in product strategy, roadmapping, and more
- Conversation memory to maintain context
- Automatic web search using Tavily API when current information is needed
- Interactive chat interface

## Setup

1. Install dependencies:
```bash
pip install google-genai tavily-python python-dotenv
```

2. Create a `.env` file with your API keys:
```
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

3. Run the notebook or Python scripts

## Usage

```python
# Create agent instance
pm_agent = ProductManagerAgent(model=model, client=client)

# Ask questions
response = pm_agent("What is the RICE framework?")
print(response)
```
