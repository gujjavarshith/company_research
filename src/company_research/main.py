#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from company_research.crew import CompanyResearch
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    company = input("Search the company :")
    inputs = {
        'topic': company,
        'current_year': str(datetime.now().year)
    }
    
    try:
        CompanyResearch().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")



