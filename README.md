# Intro

This project focuses on building a search and ranking component using machine learning to match talents with relevant job opportunities. The goal is to develop a solution that filters out irrelevant results and ranks the most suitable matches at the top. Code quality, project structure, environment management, and documentation are as important as the model's accuracy.

# Matching Talents and Jobs Using Machine Learning

The problem revolves around creating an efficient matching system for talents and job roles. Job seekers have specific attributes, such as degrees, job roles, salary expectations, and language skills, while job postings have corresponding requirements. The objective is to develop a model that:

- Filters out talent-job matches that don’t align.
- Ranks the results based on their relevance.

## Problem Statement

Given a dataset (see `data.json`) containing information about talents and job postings, the task is to create a machine learning model that predicts whether a talent is suitable for a job and provides a ranking score.

### Talent Dictionary:
- `degree` (str): Highest degree of the talent.
- `job_roles` (list[str]): Job roles the talent is interested in.
- `languages` (list[dict]): Languages the talent speaks with their respective levels.
- `salary_expectation` (int): Salary the talent expects from a new job.
- `seniority` (str): Seniority of the talent.

### Job Dictionary:
- `job_roles` (list[str]): Applicable job roles for the job.
- `languages` (list[dict]): Language requirements of the job.
- `max_salary` (int): Maximum salary the job offers.
- `min_degree` (str): Minimum degree required for the job.
- `seniorities` (list[str]): Seniority levels allowed for the job.

## Task

### Part 1: Build a Machine Learning Model
Create a machine learning model using Python that takes a talent profile and a job profile as input and returns:
- A label (`true` if the talent and job match, `false` otherwise).
- A ranking score (a float) to indicate the strength of the match.

Key steps:
- Data cleaning, transformation, and feature extraction.
- Use logistic regression (or any other sensible approach) to predict matches.
- Focus on interpretability and simplicity over high accuracy.

### Requirements:
- **Python** 
- **scikit-learn** as the machine learning library.


### Part 2: Implement a Search & Ranking Component
Create a search and ranking component that uses the model developed in Part 1 to rank talents based on their relevance to job opportunities. The provided template (`search.py`) includes methods to implement, and additional structuring of the project is encouraged for best practices.

## Solution Reasoning

This project uses **logistic regression** to match talents and jobs, with a focus on **interpretability**. Logistic regression allows for a clear understanding of how features (like seniority, degree, and salary) affect the prediction.

Strengths:
- Features are directly interpretable, making it easy to explain why a match was predicted.
- Ideal for initial exploration of the data.

Limitations:
- Logistic regression may not fully capture the complexity of the data. More sophisticated models could be considered to improve performance.
- Further enhancements, such as feature engineering and hyperparameter tuning, could increase accuracy.

## Conclusion

This project demonstrates how machine learning can be applied to match talents with job opportunities, with a focus on simplicity, interpretability, and clean code practices. The logistic regression model serves as a foundation, and more advanced models can be explored in the future for refinement.

