"""
BigQuery client utilities for fetching MMM data
"""
import pandas as pd
from google.cloud import bigquery
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BigQueryClient:
    """Client for interacting with BigQuery"""

    def __init__(self, project_id: str):
        """
        Initialize BigQuery client

        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def fetch_datamart(
        self,
        dataset: str,
        table: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch MMM datamart from BigQuery

        Args:
            dataset: BigQuery dataset name
            table: BigQuery table name
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with MMM data
        """
        query = f"""
        SELECT *
        FROM `{self.project_id}.{dataset}.{table}`
        WHERE 1=1
        """

        if start_date:
            query += f" AND date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"

        query += " ORDER BY date"

        logger.info(f"Fetching data from {dataset}.{table}")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Fetched {len(df)} rows")

        return df

    def save_results(
        self,
        df: pd.DataFrame,
        dataset: str,
        table: str,
        write_disposition: str = "WRITE_TRUNCATE"
    ) -> None:
        """
        Save results back to BigQuery

        Args:
            df: DataFrame to save
            dataset: BigQuery dataset name
            table: BigQuery table name
            write_disposition: Write mode (WRITE_TRUNCATE or WRITE_APPEND)
        """
        table_id = f"{self.project_id}.{dataset}.{table}"

        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
        )

        logger.info(f"Saving {len(df)} rows to {table_id}")
        job = self.client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        job.result()
        logger.info(f"Successfully saved to {table_id}")
