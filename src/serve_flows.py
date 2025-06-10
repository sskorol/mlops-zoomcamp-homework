"""
Complete MLOps Homework 3 Solution using Flow.serve()
This serves a comprehensive pipeline that captures all homework answers

Run this to start serving your flows with complete homework documentation
"""

from prefect import serve
from data_flow import download_data_flow

# Import our enhanced pipeline that captures all homework answers
from ml_pipeline_flow import complete_ml_workflow_with_answers

# Import inspection flow for additional analysis
from model_utils import model_inspection_flow


def main():
    """
    Start serving all flows including the comprehensive homework solution
    """
    print("MLOps Homework 3 - Complete Solution Server")
    print("=" * 55)

    # Import verification
    print("✅ Successfully imported all flows:")
    print("   • download_data_flow (data management)")
    print("   • complete_ml_workflow_with_answers (main homework pipeline)")
    print("   • model_inspection_flow (additional analysis)")

    print("\n🚀 Starting comprehensive homework solution server...")

    # Create deployments using serve() with our enhanced pipeline
    serve(
        # Main homework solution - captures all answers
        complete_ml_workflow_with_answers.to_deployment(
            name="homework-complete-solution",
            description="Complete homework pipeline with all answers captured and displayed",
            tags=["homework", "complete", "answers", "mlops"],
            parameters={
                "year": 2023,
                "month": 3,
                "taxi_type": "yellow"
            }
        ),

        # Individual component flows for flexibility
        download_data_flow.to_deployment(
            name="data-download",
            description="Download and manage taxi data files",
            tags=["data", "download"],
            parameters={
                "year": 2023,
                "month": 3,
                "taxi_type": "yellow"
            }
        ),

        # Model inspection for deep analysis
        model_inspection_flow.to_deployment(
            name="model-inspection",
            description="Inspect models and provide detailed analysis",
            tags=["inspection", "analysis"],
            parameters={
                "experiment_name": "homework3-yellow-taxi-prefect"
            }
        ),

        # Server configuration
        limit=5,  # Maximum concurrent flow runs
        webserver=True,  # Enable web UI integration
    )


if __name__ == "__main__":
    main()