from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv
load_dotenv()

# src/company_research/crew.py

from company_research.tools import (
    serp_api_tool,
    wikipedia_tool,
    yahoo_finance_tool,
    news_api_tool,
    twitter_scraper_tool
)

@CrewBase
class CompanyResearch():
    """CompanyResearch crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def company_info_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['company_info_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def financial_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_analyst_agent'], # type: ignore[index]
            verbose=True
        )

    @agent
    def market_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['market_analyst_agent'], # type: ignore[index]
            verbose=True
        )
    @agent
    def sentiment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_agent'], # type: ignore[index]
            verbose=True
        )
    @agent
    def report_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['report_writer_agent'], # type: ignore[index]
            verbose=True
        )

    @task
    def gather_company_info(self) -> Task:
        return Task(
            config=self.tasks_config['gather_company_info'], # type: ignore[index]
        )

    @task
    def analyze_financials(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_financials'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def analyze_market_position(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_market_position'], # type: ignore[index]
        )

    @task
    def analyze_sentiment(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_sentiment'], # type: ignore[index]
        )

    @task
    def generate_report(self) -> Task:
        return Task(
            config=self.tasks_config['generate_report'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CompanyResearch crew"""

        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
