# Install Rasa
!pip install rasa

# Import necessary libraries
import json
import logging
from rasa.cli import train
from rasa.core.agent import Agent
from rasa.core.policies import FallbackPolicy, KerasPolicy
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.core.utils import EndpointConfig
from rasa.model import get_model

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the Rasa configuration file
config = """
language: "en"

pipeline:
  - name: "WhitespaceTokenizer"
  - name: "RegexFeaturizer"
  - name: "CRFEntityExtractor"
  - name: "EntitySynonymMapper"
  - name: "CountVectorsFeaturizer"
  - name: "EmbeddingIntentClassifier"

policies:
  - name: FallbackPolicy
    nlu_threshold: 0.3
    core_threshold: 0.2
    fallback_action_name: "utter_default"

  - name: KerasPolicy
    epochs: 200
    max_history: 5
"""

# Define the domain file
domain = """
intents:
  - greet
  - goodbye
  - thanks
  - help

responses:
  utter_greet:
    - text: "Hello! How can I assist you today?"

  utter_goodbye:
    - text: "Goodbye! Have a great day!"

  utter_thanks:
    - text: "You're welcome! Let me know if there's anything else I can help with."

  utter_default:
    - text: "I'm sorry, I didn't understand. Can you please rephrase your query?"

actions:
  - utter_greet
  - utter_goodbye
  - utter_thanks
  - utter_default

"""

# Define the stories file
stories = """
## greet
* greet
  - utter_greet

## goodbye
* goodbye
  - utter_goodbye

## thanks
* thanks
  - utter_thanks

## help
* help
  - utter_default
"""

# Save the configuration, domain, and stories to files
with open("config.yml", "w") as f:
    f.write(config)

with open("domain.yml", "w") as f:
    f.write(domain)

with open("data/stories.md", "w") as f:
    f.write(stories)

# Train the Rasa model
train.main(
    domain="domain.yml",
    config="config.yml",
    training_files="data",
    output="models",
    fixed_model_name="chatbot"
)

# Load the trained model
interpreter = RasaNLUInterpreter("models/nlu/default/chatbot")
action_endpoint = EndpointConfig(url="http://localhost:5055/webhook")
agent = Agent.load("models/dialogue", interpreter=interpreter, action_endpoint=action_endpoint)

# Define a function to handle user queries
def handle_user_query(query):
    responses = agent.handle_text(query)
    return [response['text'] for response in responses]

# Example user queries
user_queries = [
    "Hello",
    "What is your return policy?",
    "Thank you for your help!",
    "Goodbye"
]

# Process user queries and generate responses
chatbot_responses = []
for query in user_queries:
    response = handle_user_query(query)
    chatbot_responses.append(response)

# Print the chatbot responses
for query, response in zip(user_queries, chatbot_responses):
    print(f"User Query: {query}")
    print(f"Chatbot Response: {response}")
    print()
