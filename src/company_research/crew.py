from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class CompanyResearch():
    """CompanyResearch crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
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

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
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
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
