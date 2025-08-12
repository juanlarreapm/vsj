import pandas as pd
from itertools import combinations
from collections import Counter
import networkx as nx
from community.community_louvain import best_partition
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security.api_key import APIKeyHeader
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import openai
from google.colab import userdata

# Define a secret API key (in a real application, this should be stored securely)
# For Render deployment, it's better to read this from environment variables
SECRET_API_KEY = "my-secret-api-key" # Replace with your actual secret key or read from env

# Define the API key header
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    """Validates the provided API key."""
    # In a real application, read SECRET_API_KEY from environment variables for security
    if api_key is None or api_key != SECRET_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

# Configure OpenAI API key (replace with your actual API key or use Colab Secrets)
# It's highly recommended to use Colab Secrets for your API key
openai.api_key = userdata.get("OPENAI_API_KEY")

def get_job_name_from_parts(part_names: List[str]) -> str:
    """
    Sends a list of part names to the LLM to generate a job name.
    """
    prompt = f"You are an automotive expert. Given the following list of automotive parts, suggest a concise job name (ex. 'Brake Job' or 'Oil Change') that describes the repair or maintenance task they are associated with:\n\n{', '.join(part_names)}\n\nJob Name:"

    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct", # You can choose a different model if needed
            prompt=prompt,
            max_tokens=20, # Limit the response length for a concise name
            temperature=0.5 # Adjust temperature for creativity (lower for more focused, higher for more creative)
        )
        job_name = response.choices[0].text.strip()
        return job_name
    except Exception as e:
        print(f"Error interacting with OpenAI: {e}")
        return "Could not generate job name."


# Define Pydantic models for response - Reordered for dependencies
class PartInfo(BaseModel):
    part_type_id: float
    part_type_name: str
    count: int
    frequency: float

class VehicleDetails(BaseModel):
    vehicle_year: int
    vehicle_make_name: str
    vehicle_model_name: str
    vehicle_submodel_name: str
    vehicle_engine_name: str
    vehicle_fuel_type_name: str

# Define job models (JobInfo first as FilteredJobInfo refers to PartInfo which is already defined)
class JobInfo(BaseModel):
    job_id: int
    num_orders: int
    parts: List[PartInfo]

class FilteredJobInfo(BaseModel):
    job_id: int
    num_orders: int
    parts: List[PartInfo] # This will contain parts filtered by frequency threshold


# Define response models that use the above models
class VehicleJobResponse(BaseModel):
    vehicle_details: VehicleDetails
    jobs: List[JobInfo]

class FilteredVehicleJobResponse(BaseModel):
    vehicle_details: VehicleDetails
    jobs: List[FilteredJobInfo] # This will contain jobs filtered by num_orders threshold

class JobPartsResponse(BaseModel):
    job_id: int
    parts: List[PartInfo]

class NumOrdersResponse(BaseModel):
    vehicle_id: int
    job_id: int
    num_orders: int

class FrequencyResponse(BaseModel):
    vehicle_id: int
    job_id: int
    part_type_id: float
    frequency: float
    count: int # Including count as it's often useful alongside frequency

# Define a model for the new data source
class NewDataModel(BaseModel):
    col1: str
    col2: int
    # Add fields based on the actual columns in your new CSV

class SuggestedJobNameResponse(BaseModel):
    job_id: int
    suggested_name: str


# Define the analysis functions (copied from previous cells)
def load_data(file_path):
    """Loads the dataset from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def perform_clustering_and_analysis(df):
    """
    Performs co-occurrence calculation and Louvain clustering for each vehicle,
    and generates the necessary dataframes for job and part analysis.
    """
    unique_vehicle_ids = df["vehicle_id"].unique()
    all_vehicles_part_jobs_list = []

    # Get unique vehicle details outside the loop for efficiency
    vehicle_details = df[["vehicle_id", "vehicle_year", "vehicle_make_name", "vehicle_model_name", "vehicle_submodel_name", "vehicle_engine_name", "vehicle_fuel_type_name"]].drop_duplicates()
    part_names = df[["part_type_id", "part_type_name"]].drop_duplicates()


    for vehicle_id in unique_vehicle_ids:
        df_vehicle = df[df["vehicle_id"] == vehicle_id].copy()

        basket_parts_vehicle = df_vehicle.groupby("order_group_id")["part_type_id"].apply(set)

        pair_counts_vehicle = Counter()
        for parts in basket_parts_vehicle:
            for pair in combinations(sorted(parts), 2):
                pair_counts_vehicle[pair] += 1

        cooc_df_vehicle = pd.DataFrame(
            [(a, b, count) for (a, b), count in pair_counts_vehicle.items()],
            columns=["part_type_a", "part_type_b", "count"]
        )

        if not cooc_df_vehicle.empty:
            G_vehicle = nx.Graph()
            for _, row in cooc_df_vehicle.iterrows():
                G_vehicle.add_edge(row["part_type_a"], row["part_type_b"], weight=row["count"])

            partition_vehicle = best_partition(G_vehicle, weight="weight")

            part_to_job_vehicle = pd.DataFrame.from_dict(partition_vehicle, orient="index", columns=["job_id"])
            part_to_job_vehicle["vehicle_id"] = vehicle_id
            part_to_job_vehicle["part_type_id"] = part_to_job_vehicle.index
            part_to_job_vehicle = part_to_job_vehicle.reset_index(drop=True)


            all_vehicles_part_jobs_list.append(part_to_job_vehicle)

    all_vehicle_part_jobs = pd.concat(all_vehicles_part_jobs_list, ignore_index=True)

    # Merge the df DataFrame with the all_vehicle_part_jobs DataFrame
    df_jobs_vehicle = df.merge(all_vehicle_part_jobs, on=["vehicle_id", "part_type_id"], how="left")

    # Group by Vehicle Id and job_id to count unique Order Group Ids
    jobs_by_vehicle_details_updated = df_jobs_vehicle.groupby(["vehicle_id", "job_id"])["order_group_id"].nunique().reset_index()
    jobs_by_vehicle_details_updated.rename(columns={"order_group_id": "num_orders"}, inplace=True)

    # Merge with vehicle_details
    jobs_by_vehicle_details_updated = jobs_by_vehicle_details_updated.merge(vehicle_details, on="vehicle_id")

    # Calculate part counts and frequencies per job per vehicle
    parts_by_job_by_vehicle = df_jobs_vehicle.groupby(["vehicle_id", "job_id", "part_type_id"]).size().reset_index(name="count")

    # Calculate frequency of each part within its job for each vehicle
    parts_by_job_by_vehicle["frequency"] = parts_by_job_by_vehicle.groupby(["vehicle_id", "job_id"])["count"].transform(lambda x: x / x.sum())

    # Merge with part_names to include Part Type Names
    parts_by_job_by_vehicle_with_names = parts_by_job_by_vehicle.merge(part_names, on="part_type_id", how="left")


    return all_vehicle_part_jobs, jobs_by_vehicle_details_updated, parts_by_job_by_vehicle_with_names


def get_vehicle_job_parts(vehicle_id, jobs_by_vehicle_details, parts_by_job_by_vehicle_with_names):
    """
    Retrieves job and part information for a specific vehicle ID.
    """
    vehicle_jobs_details = jobs_by_vehicle_details[jobs_by_vehicle_details["vehicle_id"] == int(vehicle_id)]

    result = {}
    if not vehicle_jobs_details.empty:
        vehicle_info = vehicle_jobs_details.iloc[0]
        result["vehicle_details"] = {
            "vehicle_year": vehicle_info["vehicle_year"],
            "vehicle_make_name": vehicle_info["vehicle_make_name"],
            "vehicle_model_name": vehicle_info["vehicle_model_name"],
            "vehicle_submodel_name": vehicle_info["vehicle_submodel_name"],
            "vehicle_engine_name": vehicle_info["vehicle_engine_name"],
            "vehicle_fuel_type_name": vehicle_info["vehicle_fuel_type_name"]
        }

        all_jobs_for_vehicle = vehicle_jobs_details.sort_values("num_orders", ascending=False)
        result["jobs"] = []

        if not all_jobs_for_vehicle.empty:
            for _, job_row in all_jobs_for_vehicle.iterrows():
                job_id = job_row["job_id"]
                num_orders = job_row["num_orders"]
                job_info = {"job_id": int(job_id), "num_orders": int(num_orders), "parts": []}

                parts_in_job_for_vehicle = parts_by_job_by_vehicle_with_names[
                    (parts_by_job_by_vehicle_with_names["vehicle_id"] == int(vehicle_id)) &
                    (parts_by_job_by_vehicle_with_names["job_id"] == job_id)
                ].sort_values("count", ascending=False)

                if not parts_in_job_for_vehicle.empty:
                    for _, part_row in parts_in_job_for_vehicle.iterrows():
                        job_info["parts"].append({
                            "part_type_id": part_row["part_type_id"],
                            "part_type_name": part_row["part_type_name"],
                            "count": int(part_row["count"]),
                            "frequency": float(part_row["frequency"])
                        })
                result["jobs"].append(job_info)
    return result

def get_job_parts_across_vehicles(job_id, parts_by_job_with_names_updated):
    """
    Retrieves part information for a specific job ID across all vehicles.
    """
    parts_for_job = parts_by_job_with_names_updated[parts_by_job_with_names_updated["job_id"] == float(job_id)]

    result = {"job_id": int(job_id), "parts": []}
    if not parts_for_job.empty:
        for _, part_row in parts_for_job.sort_values("count", ascending=False).iterrows():
             result["parts"].append({
                "part_type_id": part_row["part_type_id"],
                "part_type_name": part_row["part_type_name"],
                "count": int(part_row["count"]),
                "frequency": float(part_row["frequency"])
            })
    return result

# Load data and perform initial analysis
# In a production API, you might load data on startup or use a database
df = pd.read_csv("ORDERS_PAID (5).csv") # Make sure this file is available in your repo

# Perform clustering and analysis to get the necessary dataframes
all_vehicle_part_jobs, jobs_by_vehicle_details_updated, parts_by_job_by_vehicle_with_names = perform_clustering_and_analysis(df)
parts_by_job_with_names_updated = parts_by_job_by_vehicle_with_names # For consistency with the function

# Load the new data source
try:
    new_data_df = pd.read_csv("NEW_DATA_SOURCE.csv") # Replace with your new CSV file name
except FileNotFoundError:
    new_data_df = pd.DataFrame() # Handle missing file

# Define the FastAPI app instance
app = FastAPI()


# Define API endpoints
@app.get("/jobs_by_vehicle/{vehicle_id}", response_model=VehicleJobResponse, dependencies=[Depends(get_api_key)])
def read_jobs_by_vehicle(vehicle_id: int):
    """
    Retrieves job and part information for a specific vehicle ID.
    Requires a valid API key in the 'x-api-key' header.
    """
    return get_vehicle_job_parts(vehicle_id, jobs_by_vehicle_details_updated, parts_by_job_by_vehicle_with_names)

@app.get("/parts_by_job/{job_id}", response_model=JobPartsResponse, dependencies=[Depends(get_api_key)])
def read_parts_by_job(job_id: int):
    """
    Retrieves part information for a specific job ID across all vehicles.
    Requires a valid API key in the 'x-api-key' header.
    """
    # Note: This endpoint currently uses parts_by_job_with_names_updated which is aggregated across vehicles.
    # If vehicle-specific part details per job are needed for this endpoint too,
    # a similar approach to get_vehicle_job_parts would be required, possibly
    # requiring a vehicle_id parameter for this endpoint as well.
    return get_job_parts_across_vehicles(job_id, parts_by_job_with_names_updated)

@app.get("/num_orders/{vehicle_id}/{job_id}", response_model=NumOrdersResponse, dependencies=[Depends(get_api_key)])
def get_num_orders(vehicle_id: int, job_id: int):
    """
    Retrieves the number of orders for a specific vehicle and job.
    Requires a valid API key in the 'x-api-key' header.
    """
    filtered_data = jobs_by_vehicle_details_updated[
        (jobs_by_vehicle_details_updated["vehicle_id"] == vehicle_id) &
        (jobs_by_vehicle_details_updated["job_id"] == job_id)
    ]
    if filtered_data.empty:
        raise HTTPException(status_code=404, detail="Number of orders not found for the specified vehicle and job.")

    num_orders = int(filtered_data.iloc[0]["num_orders"])
    return NumOrdersResponse(vehicle_id=vehicle_id, job_id=job_id, num_orders=num_orders)

@app.get("/frequency/{vehicle_id}/{job_id}/{part_type_id}", response_model=FrequencyResponse, dependencies=[Depends(get_api_key)])
def get_frequency(vehicle_id: int, job_id: int, part_type_id: float):
    """
    Retrieves the frequency and count of a specific part within a specific job for a specific vehicle.
    Requires a valid API key in the 'x-api-key' header.
    """
    filtered_data = parts_by_job_by_vehicle_with_names[
        (parts_by_job_by_vehicle_with_names["vehicle_id"] == vehicle_id) &
        (parts_by_job_by_vehicle_with_names["job_id"] == job_id) &
        (parts_by_job_by_vehicle_with_names["part_type_id"] == part_type_id)
    ]
    if filtered_data.empty:
        raise HTTPException(status_code=404, detail="Frequency and count not found for the specified part, job, and vehicle.")

    frequency = float(filtered_data.iloc[0]["frequency"])
    count = int(filtered_data.iloc[0]["count"])
    return FrequencyResponse(vehicle_id=vehicle_id, job_id=job_id, part_type_id=part_type_id, frequency=frequency, count=count)

@app.get("/filtered_jobs_by_vehicle/{vehicle_id}", response_model=FilteredVehicleJobResponse, dependencies=[Depends(get_api_key)])
def get_filtered_jobs_by_vehicle(
    vehicle_id: int,
    frequency_threshold: float = Query(0.0, description="Minimum frequency of part within job"),
    num_orders_threshold: int = Query(0, description="Minimum number of orders for job within vehicle"),
):
    """
    Retrieves jobs and parts for a specific vehicle ID, filtered by number of orders and part frequency thresholds.
    Requires a valid API key in the 'x-api-key' header.
    """
    # Filter jobs by vehicle ID and num_orders threshold
    vehicle_jobs_details_filtered = jobs_by_vehicle_details_updated[
        (jobs_by_vehicle_details_updated["vehicle_id"] == vehicle_id) &
        (jobs_by_vehicle_details_updated["job_id"].notna()) & # Add this check
        (jobs_by_vehicle_details_updated["num_orders"] >= num_orders_threshold)
    ].sort_values("num_orders", ascending=False)


    if vehicle_jobs_details_filtered.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No jobs found for Vehicle ID: {vehicle_id} with num_orders >= {num_orders_threshold}"
        )

    # Get vehicle details
    vehicle_info = jobs_by_vehicle_details_updated[jobs_by_vehicle_details_updated["vehicle_id"] == vehicle_id].iloc[0]
    vehicle_details_response = VehicleDetails(
        vehicle_year=vehicle_info["vehicle_year"],
        vehicle_make_name=vehicle_info["vehicle_make_name"],
        vehicle_model_name=vehicle_info["vehicle_model_name"],
        vehicle_submodel_name=vehicle_info["vehicle_submodel_name"],
        vehicle_engine_name=vehicle_info["vehicle_engine_name"],
        vehicle_fuel_type_name=vehicle_info["vehicle_fuel_type_name"]
    )

    filtered_jobs_list = []
    for _, job_row in vehicle_jobs_details_filtered.iterrows():
        job_id = job_row["job_id"]
        num_orders = job_row["num_orders"]

        # Filter parts by vehicle ID, job ID, and frequency threshold
        filtered_parts_for_job = parts_by_job_by_vehicle_with_names[
            (parts_by_job_by_vehicle_with_names["vehicle_id"] == vehicle_id) &
            (parts_by_job_by_vehicle_with_names["job_id"] == job_id) &
            (parts_by_job_by_vehicle_with_names["frequency"] >= frequency_threshold)
        ].sort_values("count", ascending=False)

        parts_list = []
        if not filtered_parts_for_job.empty:
            for _, part_row in filtered_parts_for_job.iterrows():
                parts_list.append(PartInfo(
                    part_type_id=part_row["part_type_id"],
                    part_type_name=part_row["part_type_name"],
                    count=int(part_row["count"]),
                    frequency=float(part_row["frequency"])
                ))

        # Only add the job to the list if it has parts that meet the frequency threshold
        if parts_list:
             filtered_jobs_list.append(FilteredJobInfo(
                job_id=int(job_id),
                num_orders=int(num_orders),
                parts=parts_list
            ))

    if not filtered_jobs_list:
         raise HTTPException(
            status_code=404,
            detail=f"No jobs found for Vehicle ID: {vehicle_id} with num_orders >= {num_orders_threshold} and parts with frequency >= {frequency_threshold}"
        )


    return FilteredVehicleJobResponse(
        vehicle_details=vehicle_details_response,
        jobs=filtered_jobs_list
    )

@app.get("/new_data", dependencies=[Depends(get_api_key)])
def get_new_data():
    """
    Retrieves data from the new data source.
    Requires a valid API key in the 'x-api-key' header.
    """
    if new_data_df.empty:
        raise HTTPException(status_code=404, detail="New data source not loaded or is empty.")
    # Return the data as a list of dictionaries
    return new_data_df.to_dict(orient="records")

# New endpoint to get suggested job name from LLM
@app.get("/suggest_job_name/{job_id}", response_model=SuggestedJobNameResponse, dependencies=[Depends(get_api_key)])
def suggest_job_name(job_id: int):
    """
    Retrieves parts for a job ID and suggests a job name using an LLM.
    Requires a valid API key in the 'x-api-key' header.
    """
    # Retrieve parts for the given job ID
    parts_for_job = parts_by_job_by_vehicle_with_names[parts_by_job_by_vehicle_with_names["job_id"] == float(job_id)]

    if parts_for_job.empty:
        raise HTTPException(status_code=404, detail=f"No parts found for Job ID: {job_id}")

    # Get the list of part names
    part_names = parts_for_job["part_type_name"].tolist()

    # Call the LLM interaction function
    suggested_name = get_job_name_from_parts(part_names)

    # Return the suggested name
    return SuggestedJobNameResponse(job_id=job_id, suggested_name=suggested_name)
