#!/usr/bin/env python
import sys
import warnings
import os
import json

from datetime import datetime

from company_research.crew import CompanyResearch
from crewai import Crew, Process
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def read_report(report_path):
    """Read the generated report file."""
    try:
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    except Exception as e:
        print(f"Error reading report: {e}")
        return None

def read_feedback_file(feedback_path):
    """Read the feedback file."""
    try:
        if os.path.exists(feedback_path):
            with open(feedback_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    except Exception as e:
        print(f"Error reading feedback file: {e}")
        return ""

def append_to_feedback_file(feedback_path, new_section):
    """Append a new feedback section to the feedback file."""
    try:
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        # Read existing feedback
        existing_feedback = read_feedback_file(feedback_path)
        
        # Append new section
        if existing_feedback:
            updated_feedback = existing_feedback.rstrip() + "\n\n" + new_section + "\n"
        else:
            updated_feedback = new_section + "\n"
        
        # Write back
        with open(feedback_path, 'w', encoding='utf-8') as f:
            f.write(updated_feedback)
        return True
    except Exception as e:
        print(f"Error writing to feedback file: {e}")
        return False

def get_combined_report(report_path, feedback_path):
    """Get the combined report with all feedback."""
    report_content = read_report(report_path)
    feedback_content = read_feedback_file(feedback_path)
    
    if not report_content:
        return None
    
    # Clean up report content (remove markdown code blocks if present)
    content = report_content.strip()
    if content.startswith("```markdown"):
        content = content[11:].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    
    # Combine report and feedback
    if feedback_content:
        return content + "\n\n" + feedback_content
    return content

def run():
    """
    Run the crew with feedback loop.
    """
    company = input("Search the company :")
    inputs = {
        'topic': company,
        'current_year': str(datetime.now().year)
    }
    
    try:
        # Initial crew run to generate the report
        print("\n" + "="*50)
        print("Generating initial report...")
        print("="*50 + "\n")
        
        crew_instance = CompanyResearch()
        
        # Get agents by calling the agent methods directly
        company_info_agent = crew_instance.company_info_agent()
        financial_analyst_agent = crew_instance.financial_analyst_agent()
        market_analyst_agent = crew_instance.market_analyst_agent()
        sentiment_agent = crew_instance.sentiment_agent()
        report_writer_agent = crew_instance.report_writer_agent()
        
        agents_list = [
            company_info_agent,
            financial_analyst_agent,
            market_analyst_agent,
            sentiment_agent,
            report_writer_agent
        ]
        
        # Get tasks by calling the task methods directly
        gather_info_task = crew_instance.gather_company_info()
        analyze_financials_task = crew_instance.analyze_financials()
        analyze_market_task = crew_instance.analyze_market_position()
        analyze_sentiment_task = crew_instance.analyze_sentiment()
        generate_report_task = crew_instance.generate_report()
        revise_task = crew_instance.revise_report()
        
        # Initial tasks (all except revise_report)
        initial_tasks = [
            gather_info_task,
            analyze_financials_task,
            analyze_market_task,
            analyze_sentiment_task,
            generate_report_task
        ]
        
        initial_crew = Crew(
            agents=agents_list,
            tasks=initial_tasks,
            process=Process.sequential,
            verbose=True,
        )
        
        try:
            result = initial_crew.kickoff(inputs=inputs)
        except (ValueError, IndexError, Exception) as e:
            error_msg = str(e)
            if "Invalid response from LLM" in error_msg or "list index out of range" in error_msg:
                print(f"\n⚠️  Warning: LLM encountered an issue ({error_msg}).")
                print("Attempting to continue with partial results...")
                
                # Create default financials.json if it doesn't exist and task failed
                financials_path = "data/financials.json"
                if not os.path.exists(financials_path):
                    print("Creating default financials.json file...")
                    os.makedirs("data", exist_ok=True)
                    default_financials = {
                        "note": "Financial data tools were unavailable due to rate limiting or errors.",
                        "analysis": f"Unable to retrieve real-time financial data for {company}. This may be a private company or the financial data tools encountered issues.",
                        "recommendation": "Please try again later or check API keys for financial data services."
                    }
                    with open(financials_path, 'w', encoding='utf-8') as f:
                        json.dump(default_financials, f, indent=2)
                    print("✓ Default financials.json created.")
                
                # Check if report was generated despite the error
                report_path = f"reports/{company}_report.md"
                if os.path.exists(report_path):
                    print("✓ Report file found. Proceeding to feedback loop...")
                else:
                    print("✗ Report not generated. Please try again or check your API keys.")
                    raise Exception(f"Failed to generate report: {e}")
            else:
                raise
        
        # Feedback loop
        report_path = f"reports/{company}_report.md"
        feedback_path = f"reports/{company}_feedback.md"
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            # Read and display the main report (which already contains all merged feedback)
            report_content = read_report(report_path)
            if report_content:
                # Clean up report content (remove markdown code blocks if present)
                content = report_content.strip()
                if content.startswith("```markdown"):
                    content = content[11:].strip()
                if content.startswith("```"):
                    content = content[3:].strip()
                if content.endswith("```"):
                    content = content[:-3].strip()
                
                print("\n" + "="*50)
                print("CURRENT REPORT (with all feedback merged):")
                print("="*50)
                print(content[:1000] + "..." if len(content) > 1000 else content)
                print("="*50 + "\n")
            
            # Ask for feedback
            print("\n" + "="*50)
            print("FEEDBACK REQUEST")
            print("="*50)
            feedback = input("\nPlease provide your feedback on the report (or press Enter to finish): ").strip()
            
            if not feedback:
                print("\nReport finalized. All feedback has been merged into the main report.")
                print("Thank you for your feedback!")
                break
            
            # Incorporate feedback
            print("\n" + "="*50)
            print("Incorporating your feedback...")
            print("="*50 + "\n")
            
            # IMPORTANT: Read the current report content BEFORE running the crew
            # This ensures we preserve the original content even if CrewAI tries to write to the file
            current_report_before = read_report(report_path)
            if not current_report_before:
                print("⚠️  Warning: Could not read current report. Skipping feedback incorporation.")
                iteration += 1
                continue
            
            revision_inputs = {
                'topic': company,
                'current_year': str(datetime.now().year),
                'user_feedback': feedback
            }
            
            # Create a revision crew with only the revise_report task
            if revise_task:
                revision_crew = Crew(
                    agents=agents_list,
                    tasks=[revise_task],
                    process=Process.sequential,
                    verbose=True,
                )
                try:
                    result = revision_crew.kickoff(inputs=revision_inputs)
                    
                    # Extract the new section from the result
                    new_section = None
                    output = ""
                    
                    # Try different ways to extract the output
                    if hasattr(result, 'raw'):
                        output = str(result.raw)
                    elif hasattr(result, 'output'):
                        output = str(result.output)
                    elif hasattr(result, 'tasks_output'):
                        # Get output from the task
                        if result.tasks_output:
                            output = str(result.tasks_output[-1]) if isinstance(result.tasks_output, list) else str(result.tasks_output)
                    else:
                        output = str(result) if result else ""
                    
                    # Extract markdown content if wrapped in code blocks
                    if "```markdown" in output:
                        start = output.find("```markdown") + len("```markdown")
                        end = output.find("```", start)
                        if end != -1:
                            new_section = output[start:end].strip()
                    elif "```" in output:
                        start = output.find("```") + 3
                        end = output.find("```", start)
                        if end != -1:
                            new_section = output[start:end].strip()
                    else:
                        new_section = output.strip()
                    
                    # Clean up the section - remove any leading/trailing whitespace and ensure it starts with ##
                    if new_section:
                        new_section = new_section.strip()
                        # If it doesn't start with ##, try to find the first heading
                        if not new_section.startswith("##"):
                            # Look for the first ## heading
                            heading_pos = new_section.find("##")
                            if heading_pos != -1:
                                new_section = new_section[heading_pos:].strip()
                    
                    # If we got a new section, merge it into the main report immediately
                    if new_section and new_section.startswith("##"):
                        # First, append to feedback file (for backup/record)
                        append_to_feedback_file(feedback_path, new_section)
                        
                        # Use the report content we read BEFORE the crew ran (to preserve original)
                        # If the file was modified by CrewAI, we restore from our backup
                        current_content = current_report_before
                        
                        # Clean up report content (remove markdown code blocks if present)
                        content = current_content.strip()
                        if content.startswith("```markdown"):
                            content = content[11:].strip()
                        if content.startswith("```"):
                            content = content[3:].strip()
                        if content.endswith("```"):
                            content = content[:-3].strip()
                        
                        # Append new section to report
                        if not content.endswith("\n"):
                            content += "\n"
                        updated_content = content + "\n" + new_section + "\n"
                        
                        # Write back to main report file
                        os.makedirs(os.path.dirname(report_path), exist_ok=True)
                        with open(report_path, 'w', encoding='utf-8') as f:
                            f.write(updated_content)
                        print(f"✓ Feedback section merged into the main report.")
                    else:
                        print("⚠️  No valid new section was generated. Report not updated.")
                        if new_section:
                            print(f"Debug: Generated content (first 200 chars): {new_section[:200]}")
                        
                except (ValueError, IndexError, Exception) as e:
                    error_msg = str(e)
                    if "Invalid response from LLM" in error_msg or "list index out of range" in error_msg:
                        print(f"\n⚠️  Warning: LLM encountered an issue during revision ({error_msg}).")
                        print("The report may not have been updated. Please try again.")
                    else:
                        print(f"\n⚠️  Error during revision: {error_msg}")
            else:
                print("Error: Could not find revise_report task. Skipping revision.")
            
            iteration += 1
            print(f"\nRevision {iteration} completed. Reviewing updated report...\n")
        
        if iteration >= max_iterations:
            print("\nMaximum revision iterations reached. Finalizing report.")
            
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


