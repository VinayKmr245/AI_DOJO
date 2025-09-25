import openai
import os
import json
import httpx
import csv
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tenacity import (
    retry, 
    stop_after_attempt,
    wait_random_exponential
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain_openai import AzureOpenAI
from typing import TypedDict, List, Union
from datetime import datetime

# Commenting out unresolved imports temporarily
# from langgraph.grapg import StateGraph,END
# from langgraph.prebuilt import create_react_agent
# from typings import Dict, List, Optional, Any, Union, Sequence

def get_access_token():
    auth = "https://api.uhg.com/oauth2/token"
    scope = "https://api.uhg.com/.default"
    grant_type = "client_credentials"
    
    with httpx.Client() as client:
        body = {
            "client_id":dbutils.secrets.get(scope="AIML_Training",key="client_id"),
            "client_secret":dbutils.secrets.get(scope="AIML_Training",key="client_secret"),
            "scope": scope,
            "grant_type": grant_type
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        response = client.post(auth, data=body, headers=headers)
        access_token = response.json().get("access_token")
        return access_token
    
    
load_dotenv('./Data/UIAS_vars.env')
endpoint = os.getenv("MODEL_ENDPOINT")
model_name = os.getenv("MODEL_NAME")
project_id = os.getenv("PROJECT_ID")
api_version = os.getenv("API_VERSION")

chat_client = openai.AzureOpenAI(
    azure_endpoint=endpoint,
    api_version=api_version,
    azure_deployment=model_name,
    azure_ad_token=get_access_token(),
    default_headers={"project_id": project_id}
)
#load files
reference_codes_path = "./data/reference_codes.json"
evaluation_dataset_path = "./data/evaluation_dataset.json"
test_dataset_path = "./data/test_dataset.json"
insurance_policies_path = "./data/insurance_policies.json"

with open(reference_codes_path, 'r') as file:
    reference_codes_data = json.load(file)

with open(evaluation_dataset_path, 'r') as file:
    evaluation_dataset_data = json.load(file)

with open(test_dataset_path, 'r') as file:
    test_dataset_data = json.load(file)
    
with open(insurance_policies_path, 'r') as file:
    insurance_policies_data = json.load(file)

# Define insurance_policies_data if required
insurance_policies_data = []  # Placeholder for actual data

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def query_llm(prompt_messages,max_tokens=4096,temperature=1.0,top_p=1.0):
    response = chat_client.chat.completions.create(
        model=model_name,
        messages=prompt_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return response.choices[0].message['content']

def summarize_patient_record(record_str: str):
    """
    Summarize the patient record in a structured format, enriched with human-readable descriptions
    of medical codes using ICD-10 and CPT mappings. The patient demographics section includes only
    name, gender, and age.
    """
    # Convert reference_codes_data to a JSON string for inclusion in the prompt
    reference_codes_json = json.dumps(reference_codes_data)

    # Parse the record string to extract relevant fields
    try:
        record = json.loads(record_str)
        name = record.get("name", "Unknown")
        gender = record.get("gender", "Unknown")
        date_of_birth = record.get("date_of_birth")
        date_of_service = record.get("date_of_service")

        # Calculate age using the calculate_age function
        age = calculate_age(date_of_birth, date_of_service) if date_of_birth and date_of_service else "Unknown"

        # Update the record string with only the required demographics
        record["patient_demographics"] = {
            "name": name,
            "gender": gender,
            "age": age
        }
        record_str = json.dumps(record)

    except Exception as e:
        return f"Error processing the patient record: {str(e)}"

    # Define the prompt for the LLM
    prompt_messages = [
        {
            "role": "developer",
            "content": (
                "You are a helpful assistant that summarizes patient records. "
                "Use the provided ICD-10 and CPT code mappings to include human-readable descriptions "
                "of medical codes. The summary should include the following sections:\n"
                "- Patient Demographics (only name, gender, and age)\n"
                "- Insurance Policy\n"
                "- Diagnoses (with descriptions)\n"
                "- Procedures (with descriptions)\n"
                "- Preauthorization Status\n"
                "- Billed Amount (in USD)\n"
                "- Date of Service\n\n"
                "The output must be in a bullet-point format or clearly labeled sections.\n\n"
                "Here are the ICD-10 and CPT code mappings for your reference:\n"
                f"{reference_codes_json}"
            ),
        },
        {
            "role": "user",
            "content": f"Summarize the following patient record:\n\n{record_str}\n\nSummary:",
        },
    ]

    # Query the LLM for the summary
    response = query_llm(prompt_messages)

    # Return the response directly without manual replacement
    return response

def summarize_policy_guidelines(policy_str: str):
    """
    Summarize the insurance policy guidelines in a structured format, enriched with human-readable
    descriptions of medical and procedure codes using ICD-10 and CPT mappings.
    """
    # Convert reference_codes_data to a JSON string for inclusion in the prompt
    reference_codes_json = json.dumps(reference_codes_data)

    # Define the prompt for the LLM
    prompt_messages = [
        {
            "role": "developer",
            "content": (
                "You are a helpful assistant that summarizes insurance policy guidelines. "
                "Use the provided ICD-10 and CPT code mappings to include human-readable descriptions "
                "of medical and procedure codes. The summary should include the following sections:\n"
                "- Policy Details: Include policy ID and plan name.\n"
                "- Covered Procedures: For each covered procedure, include:\n"
                "  - Procedure Code and Description (using CPT code mappings)\n"
                "  - Covered Diagnoses and Descriptions (using ICD-10 code mappings)\n"
                "  - Gender Restriction\n"
                "  - Age Range\n"
                "  - Preauthorization Requirement\n"
                "  - Notes on Coverage (if any)\n"
                "Each procedure should be presented as a separate entry under the 'Covered Procedures' section, "
                "with the required sub-points clearly listed.\n\n"
                "The output must be in a bullet-point format or clearly labeled sections.\n\n"
                "Here are the ICD-10 and CPT code mappings for your reference:\n"
                f"{reference_codes_json}"
            ),
        },
        {
            "role": "user",
            "content": f"Summarize the following insurance policy guidelines:\n\n{policy_str}\n\nSummary:",
        },
    ]

    # Query the LLM for the summary
    response = query_llm(prompt_messages)

    # Return the response directly without manual replacement
    return response

def check_claim_coverage(record_summary, policy_summary):
    """
    Evaluate claim coverage based on the provided patient record summary and policy guideline summary.
    The evaluation includes step-by-step checks and a final decision.
    """
    prompt_messages = [
        {
            "role": "developer",
            "content": (
                "You are an insurance claim processing assistant that evaluates claim coverage. "
                "Follow these steps to evaluate the claim:\n"
                "1. Check if the patient's diagnosis code(s) match the policy-covered diagnoses for the claimed procedure.\n"
                "2. Verify that the procedure code is explicitly listed in the policy and all associated conditions are satisfied.\n"
                "3. Ensure the patient's age falls within the policy's defined age range (inclusive of the lower bound, exclusive of the upper bound).\n"
                "4. Confirm that the patient's gender matches the policy's requirement for the procedure.\n"
                "5. If preauthorization is required by the policy, ensure it has been obtained.\n"
                "Only procedures and diagnoses explicitly listed in the patient record should be evaluated. "
                "For simplicity, assume there is only one procedure per patient.\n\n"
                "The output should include the following sections:\n"
                "- Coverage Review: Step-by-step analysis for the claimed procedure, detailing the checks performed.\n"
                "- Final Decision: For the procedure, return either 'APPROVE' or 'ROUTE FOR REVIEW' with a brief explanation of the reasoning behind it.\n"
            ),
        },
        {
            "role": "user",
            "content": f"""Evaluate the following claim:

Patient Record Summary:
{record_summary}

Insurance Policy Summary:
{policy_summary}

Please respond in JSON format with the following structure:
{{
    "coverage_review": "Step-by-step analysis of the checks performed.",
    "summary_of_findings":"summary of which coverage requirements were met or not met",
    "final_decision": {{
        "decision": "APPROVE or ROUTE FOR REVIEW",
        "reasoning": "Brief explanation of the decision."
    }}
}}
"""
        },
    ]

    # Query the LLM for the evaluation
    response = query_llm(prompt_messages)

    return response

# Function to calculate age based on date_of_birth and date_of_service
def calculate_age(date_of_birth: str, date_of_service: str) -> int:
    birth_date = datetime.strptime(date_of_birth, "%Y-%m-%d")
    service_date = datetime.strptime(date_of_service, "%Y-%m-%d")

    age = service_date.year - birth_date.year
    if (service_date.month, service_date.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age

@tool
def summarize_patient_record_tool(record_str: str) -> str:
    return summarize_patient_record(record_str)

@tool
def summarize_policy_guideline_tool(policy_str: str):
    return summarize_policy_guidelines(policy_str)

@tool
def check_claim_coverage_tool(record_summary: str, policy_summary: str) -> dict:
    return check_claim_coverage(record_summary, policy_summary)

# Define the system instruction prompt
SYSTEM_PROMPT = """
You are an AI assistant designed to process health insurance claims. You have access to the following tools:

1. summarize_patient_record_tool: Takes a patient record string and returns a concise summary.
2. summarize_policy_guideline_tool: Takes a policy ID and returns a summary of the insurance policy.
3. check_claim_coverage_tool: Takes a record summary and policy summary and determines if the claim should

Follow this exact workflow for each claim:
1. First, summarize the patient record using summarize_patient_record_tool
2. Next, get the relevant insurance policy summary using summarize_policy_guideline_tool
3. Finally, check if the claim should be covered using check_claim_coverage_tool

Your final response must follow this format:
Decision: [Approved/Denied]
Reason: [clear explanation of the decision based on the record and policy]

Make sure to complete all steps in order for every claim you process.
"""

# Create Langchain-compatible model
def create_langchain_model():
    """Create a Langchain-compatible model instance"""
    # Using a custom model wrapper to maintain compatibility with our existing authentication
    class CustomAzureOpenAI(AzureOpenAI):
        def __init__(self):
            # Configure with environment variables or parameters as needed
            super().__init__(
                azure_endpoint=endpoint,
                azure_deployment=model_name,
                api_key="dummy_key", # we'll use the token auth instead
                api_version=api_version,
                azure_ad_token_provider=get_access_token
            )
    return CustomAzureOpenAI()

# Define typed state for the agent
class AgentState(TypedDict):
    messages: List[Union[AIMessage, HumanMessage, SystemMessage]]
    # Additional state fields can be added as needed

# Create the React agent
def create_insurance_agent():
    """Create and return the insurance claims processing agent"""
    # Define the tools available to the agent
    tools = [
        summarize_patient_record_tool,
        summarize_policy_guideline_tool,
        check_claim_coverage_tool
    ]

    # Create the ReAct-style agent using LangGraph's create_react_agent function
    agent = create_react_agent(
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        single_agent=True  # Ensure it is configured as a single agent
    )
    
    return agent

def run_agent_validation():
    agent = create_insurance_agent()
    results = []
    for record in evaluation_dataset_data:
        try:
            patient_record = record["patient_record"]
            policy_guidelines = record["policy_guidelines"]

            # Use the agent to process the claim
            decision = agent.run({
                "patient_record": patient_record,
                "policy_guidelines": policy_guidelines
            })

            results.append({
                "patient_record": patient_record,
                "policy_guidelines": policy_guidelines,
                "decision": decision
            })
        except Exception as e:
            results.append({
                "patient_record": record["patient_record"],
                "policy_guidelines": record["policy_guidelines"],
                "decision": "ROUTE FOR REVIEW",
                "reasoning": str(e)
            })

    # Save results to a JSON file for debugging
    with open("agent_validation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results

# Add a new function to run the finalized agent on test_dataset.json and save results to submission.csv

def run_final_agent_evaluation():
    agent = create_insurance_agent()
    results = []
    for record in test_dataset_data:
        try:
            patient_record = record["patient_record"]
            policy_guidelines = record["policy_guidelines"]

            # Use the agent to process the claim
            decision = agent.run({
                "patient_record": patient_record,
                "policy_guidelines": policy_guidelines
            })

            results.append({
                "patient_record": patient_record,
                "policy_guidelines": policy_guidelines,
                "decision": decision["decision"],
                "reasoning": decision["reasoning"]
            })
        except Exception as e:
            results.append({
                "patient_record": record["patient_record"],
                "policy_guidelines": record["policy_guidelines"],
                "decision": "ROUTE FOR REVIEW",
                "reasoning": str(e)
            })

    # Save results to a CSV file
    with open("submission.csv", "w", newline="") as csvfile:
        fieldnames = ["patient_record", "policy_guidelines", "decision", "reasoning"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    return results



# Preprocess the test dataset to add the computed age to each patient record
for record in test_dataset_data:
    date_of_birth = record.get("date_of_birth")
    date_of_service = record.get("date_of_service")

    if date_of_birth and date_of_service:
        record["age"] = calculate_age(date_of_birth, date_of_service)

# Preprocess the evaluation dataset to add the computed age to each patient record
for record in evaluation_dataset_data:
    date_of_birth = record.get("date_of_birth")
    date_of_service = record.get("date_of_service")

    if date_of_birth and date_of_service:
        record["age"] = calculate_age(date_of_birth, date_of_service)

if __name__ == "__main__":
    import datetime

    print("Insurance Claim Processing Agent")
    print("")
    print(f"Loaded {len(reference_codes_data)} reference codes")
    print(f"Loaded {len(evaluation_dataset_data)} evaluation records")
    print(f"Loaded {len(test_dataset_data)} test records")
    print(f"Loaded {len(insurance_policies_data)} insurance policies")

    # Step 1: Standard validation (no human reference comparison)
    print("\nRunning standard agent validation...")
    validation_results = run_agent_validation()

    # Step 2: Evaluate against human reference responses
    print("\nRunning evaluation against human reference responses...")
    validation_csv_path = "./data/validation_reference_results.csv"
    evaluation_results = run_agent_evaluation_with_references(validation_csv_path)

    # Step 3: Generate improvement recommendations if evaluation was successful
    if evaluation_results and "metrics" in evaluation_results:
        print("\nGenerating agent improvement recommendations...")
        improvement_recommendations = improve_agent_based_on_evaluation(evaluation_results)

        # If we received valid recommendations with a suggested system prompt, update and test it
        if (improvement_recommendations and 
            "suggested_system_prompt" in improvement_recommendations and
            isinstance(improvement_recommendations["suggested_system_prompt"], str)):
            
            print("\nTesting improved system prompt...")
            # Update the global SYSTEM_PROMPT with the improved version
            original_prompt = SYSTEM_PROMPT

            SYSTEM_PROMPT = improvement_recommendations["suggested_system_prompt"]
            test_records = evaluation_dataset_data[:3] if len(evaluation_dataset_data) >= 3 else evaluation_dataset_data

# Create a new agent with the updated prompt
            improved_agent = create_insurance_agent()

            print("Testing improved agent on sample records...")
            for i, record in enumerate(test_records):
                record_str = json.dumps(record)
                input_message = HumanMessage(content=f"Process this patient record for insurance claim coverage: {record_str}")

                try:
                    improved_response = improved_agent.invoke({
                        "messages": [SystemMessage(content=SYSTEM_PROMPT), input_message]
                    })
                    print(f"Improved agent response for record {i+1}: {improved_response['messages'][-1].content}")
                except Exception as e:
                    print(f"Error with improved agent on record {i+1}: {str(e)}")

            print("\nImproved system prompt has been tested and saved to improved_system_prompt.txt")
            
            with open("improved_system_prompt.txt", "w") as f:
                f.write(SYSTEM_PROMPT)
            
            SYSTEM_PROMPT = original_prompt