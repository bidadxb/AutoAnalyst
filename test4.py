import dspy

class memory_summarize_agent(dspy.Signature):
    """
    You are an AI agent which helps summarize other agent responses and user-input. 
    Keep these instructions in mind:

    - Analyze the provided text.
    - Present the extracted details in bullet points:
      - User Query: The user query/goal summarized, with only important information retained
      - Agent: Include agent name
      - Stack_Used: All python packages used
      - Actions: What actions did the agent_name take, summarize them like "Agent visualized a line chart using plotly"
   
    """
    agent_response = dspy.InputField(desc="What the agents output, commentary and code")
    user_goal = dspy.InputField(desc= "User query or intended goal")
    summary = dspy.OutputField(desc ="The summary generated in the format requested")

class error_memory_agent(dspy.Signature):
    """
Prompt for error_summarize Agent:

Agent Name: error_summarize

Purpose: To generate a concise summary of an error in Python code and provide a clear correction, along with relevant metadata and user query information. This summary will help in understanding the error and applying the correction.

Input Data:

Incorrect Python Code: (A snippet of code that produced an error)
Meta Data:
Agent Name: (Name of the agent that processed the code)
Agent Version: (Version of the agent that processed the code)
Timestamp: (When the error occurred)
User Query: (The query or task that led to the incorrect code execution)
Human-Defined Correction: (The corrected code or solution provided by a human expert)
Processing Instructions:

Error Analysis:

Analyze the incorrect Python code to determine the type of error and its cause.
Summary Creation:

Generate a brief summary of the error, highlighting the key issue in the code.
Provide a short explanation of the correction that resolves the issue.
Output Formatting:

Format the summary to include:
Error Summary: A concise description of the error.
Correction: A brief explanation of how to correct the error.
Integration:

Ensure the summary is clear and informative for future reference.
Output Data:

Error Summary:
Error Summary: (Brief description of the error)
Correction: (Concise explanation of the fix)
Example Output:

Error Summary: The IndexError occurred because the code attempted to access an element at an index that is out of range for the list.
Correction: Ensure the index is within the bounds of the list. For example, use if index < len(my_list): to check the index before accessing the list element. 
    """
    incorrect_code = dspy.InputField(desc="Error causing code")
    error_metadata = dspy.InputField(desc="The description of the error generated, with user/agent information for context")
    correction = dspy.InputField(desc="Correction suggested by AI or done manually by human")
    summary = dspy.OutputField(desc="The description which must contain information about the error and how to correct it")
    


import dspy
import streamlit as st




# Contains the DSPy agents

OPENAI_API_KEY="sk-jB37doZUJvUbVsEs3nCs0_TYSlDcMQVK6810TH1TqcT3BlbkFJEF2yImEbeyLQdNZTJJx7od9IVE8LW7OsFazuHc8NAA"



import dspy

class analytical_planner(dspy.Signature):
    # The planner agent which routes the query to Agent(s)
    # The output is like this Agent1->Agent2 etc
    """ You are data analytics planner agent. You have access to three inputs
    1. Datasets
    2. Data Agent descriptions
    3. User-defined Goal
    You take these three inputs to develop a comprehensive plan to achieve the user-defined goal from the data & Agents available.
    In case you think the user-defined goal is infeasible you can ask the user to redefine or add more description to the goal.

    Give your output in this format:
    plan: Agent1->Agent2->Agent3
    plan_desc = Use Agent 1 for this reason, then agent2 for this reason and lastly agent3 for this reason.

    You don't have to use all the agents in response of the query
    
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system, use this df,columns  set df as copy of df")
    Agent_desc = dspy.InputField(desc= "The agents available in the system")
    goal = dspy.InputField(desc="The user defined goal ")
    plan = dspy.OutputField(desc="The plan that would achieve the user defined goal", prefix='Plan:')
    plan_desc= dspy.OutputField(desc="The reasoning behind the chosen plan")

class goal_refiner_agent(dspy.Signature):
    # Called to refine the query incase user query not elaborate
    """You take a user-defined goal given to a AI data analyst planner agent, 
    you make the goal more elaborate using the datasets available and agent_desc"""
    dataset = dspy.InputField(desc="Available datasets loaded in the system, use this df,columns  set df as copy of df")
    Agent_desc = dspy.InputField(desc= "The agents available in the system")
    goal = dspy.InputField(desc="The user defined goal ")
    refined_goal = dspy.OutputField(desc='Refined goal that helps the planner agent plan better')

class preprocessing_agent(dspy.Signature):
    # Doer Agent which performs pre-processing like cleaning data, make new columns etc
    """ Given a user-defined analysis goal and a pre-loaded dataset df, 
    I will generate Python code using NumPy and Pandas to build an exploratory analytics pipeline.
      The goal is to simplify the preprocessing and introductory analysis of the dataset.

Task Requirements:

Identify and separate numeric and categorical columns into two lists: numeric_columns and categorical_columns.
Handle null values in the dataset, applying the correct logic for numeric and categorical columns.
Convert string dates to datetime format.
Create a correlation matrix that only includes numeric columns.
Use the correct column names according to the dataset.

The generated Python code should be concise, readable, and follow best practices for data preprocessing and introductory analysis. 
The code should be written using NumPy and Pandas libraries, and should not read the CSV file into the dataframe (it is already loaded as df).
When splitting numerical and categorical use this script:

categorical_columns = df.select_dtypes(include=[object, 'category']).columns.tolist()
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

DONOT 

Use this to handle conversion to Datetime
def safe_to_datetime(date):
    try:
        return pd.to_datetime(date,errors='coerce', cache=False)
    except (ValueError, TypeError):
        return pd.NaT

df['datetime_column'] = df['datetime_column'].apply(safe_to_datetime)

You will be given recent history as a hint! Use that to infer what the user is saying
You are logged in streamlit use st.write instead of print
If visualizing use plotly



    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system, use this df, column_names  set df as copy of df")
    goal = dspy.InputField(desc="The user defined goal could ")
    code = dspy.OutputField(desc ="The code that does the data preprocessing and introductory analysis")
    commentary = dspy.OutputField(desc="The comments about what analysis is being performed")
    


class statistical_analytics_agent(dspy.Signature):
    # Statistical Analysis Agent, builds statistical models using StatsModel Package
    """ 
    You are a statistical analytics agent. Your task is to take a dataset and a user-defined goal and output Python code that performs the appropriate statistical analysis to achieve that goal. Follow these guidelines:

Data Handling:

    Always handle strings as categorical variables in a regression using statsmodels C(string_column).
    Do not change the index of the DataFrame.
    Convert X and y into float when fitting a model.
Error Handling:

    Always check for missing values and handle them appropriately.
    Ensure that categorical variables are correctly processed.
    Provide clear error messages if the model fitting fails.
Regression:

    For regression, use statsmodels and ensure that a constant term is added to the predictor using sm.add_constant(X).
    Handle categorical variables using C(column_name) in the model formula.
    Fit the model with model = sm.OLS(y.astype(float), X.astype(float)).fit().
Seasonal Decomposition:

    Ensure the period is set correctly when performing seasonal decomposition.
    Verify the number of observations works for the decomposition.
Output:

    Ensure the code is executable and as intended.
    Also choose the correct type of model for the problem
    Avoid adding data visualization code.

Use code like this to prevent failing:
import pandas as pd
import numpy as np
import statsmodels.api as sm

def statistical_model(X, y, goal, period=None):
    try:
        # Check for missing values and handle them
        X = X.dropna()
        y = y.loc[X.index].dropna()

        # Ensure X and y are aligned
        X = X.loc[y.index]

        # Convert categorical variables
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category')

        # Add a constant term to the predictor
        X = sm.add_constant(X)

        # Fit the model
        if goal == 'regression':
            # Handle categorical variables in the model formula
            formula = 'y ~ ' + ' + '.join([f'C({col})' if X[col].dtype.name == 'category' else col for col in X.columns])
            model = sm.OLS(y.astype(float), X.astype(float)).fit()
            return model.summary()

        elif goal == 'seasonal_decompose':
            if period is None:
                raise ValueError("Period must be specified for seasonal decomposition")
            decomposition = sm.tsa.seasonal_decompose(y, period=period)
            return decomposition

        else:
            raise ValueError("Unknown goal specified. Please provide a valid goal.")

    except Exception as e:
        return f"An error occurred: {e}"

# Example usage:
result = statistical_analysis(X, y, goal='regression')
print(result)


    You may be give recent agent interactions as a hint! With the first being the latest
    You are logged in streamlit use st.write instead of print
If visualizing use plotly


    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system, use this df,columns  set df as copy of df")
    goal = dspy.InputField(desc="The user defined goal for the analysis to be performed")
    code = dspy.OutputField(desc ="The code that does the statistical analysis using statsmodel")
    commentary = dspy.OutputField(desc="The comments about what analysis is being performed")
    

class sk_learn_agent(dspy.Signature):
    # Machine Learning Agent, performs task using sci-kit learn
    """You are a machine learning agent. 
    Your task is to take a dataset and a user-defined goal, and output Python code that performs the appropriate machine learning analysis to achieve that goal. 
    You should use the scikit-learn library.


    Make sure your output is as intended!

    You may be give recent agent interactions as a hint! With the first being the latest
    You are logged in streamlit use st.write instead of print

    
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system, use this df,columns. set df as copy of df")
    goal = dspy.InputField(desc="The user defined goal ")
    code = dspy.OutputField(desc ="The code that does the Exploratory data analysis")
    commentary = dspy.OutputField(desc="The comments about what analysis is being performed")
    
    
    
class story_teller_agent(dspy.Signature):
    # Optional helper agent, which can be called to build a analytics story 
    # For all of the analysis performed
    """ You are a story teller agent, taking output from different data analytics agents, you compose a compelling story for what was done """
    agent_analysis_list =dspy.InputField(desc="A list of analysis descriptions from every agent")
    story = dspy.OutputField(desc="A coherent story combining the whole analysis")

class code_combiner_agent(dspy.Signature):
    # Combines code from different agents into one script
    """ You are a code combine agent, taking Python code output from many agents and combining the operations into 1 output
    You also fix any errors in the code. 


    Double check column_names/dtypes using dataset, also check if applied logic works for the datatype
    df.copy = df.copy()
    Change print to st.write
    Also add this to display Plotly chart
    st.plotly_chart(fig, use_container_width=True)



    Make sure your output is as intended!
        You may be give recent agent interactions as a hint! With the first being the latest
    You are logged in streamlit use st.write instead of print


    """
    dataset = dspy.InputField(desc="Use this double check column_names, data types")
    agent_code_list =dspy.InputField(desc="A list of code given by each agent")
    refined_complete_code = dspy.OutputField(desc="Refined complete code base")
    
    
class data_viz_agent(dspy.Signature):
    # Visualizes data using Plotly
    """
    You are AI agent who uses the goal to generate data visualizations in Plotly.
    You have to use the tools available to your disposal
    If row_count of dataset > 50000, use sample while visualizing 
    use this
    if len(df)>50000:
        .......
    Only this agent does the visualization
    Also only use x_axis/y_axis once in update layout
    {dataset}
    {styling_index}

    You must give an output as code, in case there is no relevant columns, just state that you don't have the relevant information
    
    Make sure your output is as intended! DO NOT OUTPUT THE DATASET/STYLING INDEX 
    ONLY OUTPUT THE CODE AND COMMENTARY. ONLY USE ONE OF THESE 'K','M' or 1,000/1,000,000. NOT BOTH

    You may be give recent agent interactions as a hint! With the first being the latest
    DONT INCLUDE GOAL/DATASET/STYLING INDEX IN YOUR OUTPUT!
    You can add trendline into a scatter plot to show it changes,only if user mentions for it in the query!
    You are logged in streamlit use st.write instead of print

    """
    goal = dspy.InputField(desc="user defined goal which includes information about data and chart they want to plot")
    dataset = dspy.InputField(desc=" Provides information about the data in the data frame. Only use column names and dataframe_name as in this context")
    styling_index = dspy.InputField(desc='Provides instructions on how to style your Plotly plots')
    code= dspy.OutputField(desc="Plotly code that visualizes what the user needs according to the query & dataframe_index & styling_context")
    commentary = dspy.OutputField(desc="The comments about what analysis is being performed, this should not include code")
    
    

class code_fix(dspy.Signature):
    # Called to fix unexecutable code
    """
    You are an AI which fixes the data analytics code from another agent, your fixed code should only fix the faulty part of the code, rest should remain the same
    You take the faulty code, and the error generated and generate the fixed code that performs the exact analysis the faulty code intends to do
    You are also give user given context that guides you how to fix the code!


    please reflect on the errors of the AI agent and then generate a 
   correct step-by-step solution to the problem.
   You are logged in streamlit use st.write instead of print


    """
    faulty_code = dspy.InputField(desc="The faulty code that did not work")
    previous_code_fixes = dspy.InputField(desc="User adds additional context that might help solve the problem")
    error = dspy.InputField(desc="The error generated")

    faulty_code = dspy.OutputField(desc="Only include the faulty code here")
    fixed_code= dspy.OutputField(desc="The fixed code")

class insights_agent(dspy.Signature):
    # For all of the analysis performed
    """ You are an analytical insights agent, using available data, you summarize data and prepare a list of key observations and insights. """
    goal = dspy.InputField(desc="user defined goal which includes information about data and chart they want to plot")
    dataset = dspy.InputField(desc=" Provides information about the data in the data frame. Only use column names and dataframe_name as in this context")
    insights = dspy.OutputField(desc="Insights summarizing the whole analysis")



# The ind module is called when agent_name is 
# explicitly mentioned in the query
class auto_analyst_ind(dspy.Module):
    # Only doer agents are passed
    def __init__(self,agents,retrievers):
        #Initializes all the agents, and makes retrievers
       
        #agents stores the DSPy module for each agent
        # agent_inputs contains all the inputs to use each agent
        # agent desc contains description on what the agent does 
        self.agents = {}
        self.agent_inputs ={}
        self.agent_desc =[]
        i =0
        #loops through to create module from agent signatures
        #creates a dictionary with the exact inputs for agents stored
        for a in agents:
            name = a.__pydantic_core_schema__['schema']['model_name']
            self.agents[name] = dspy.ChainOfThoughtWithHint(a)
            self.agent_inputs[name] ={x.strip() for x in str(agents[i].__pydantic_core_schema__['cls']).split('->')[0].split('(')[1].split(',')}
            self.agent_desc.append(str(a.__pydantic_core_schema__['cls']))
            i+=1
            
        # memory_summary agent builds a summary on what the agent does
        self.memory_summarize_agent = dspy.ChainOfThought(memory_summarize_agent)
        # two retrievers defined, one dataset and styling index
        self.dataset = retrievers['dataframe_index'].as_retriever(k=1)
        self.styling_index = retrievers['style_index'].as_retriever(similarity_top_k=1)


    def forward(self, query, specified_agent):
        
        # output_dict 
        dict_ ={}
        #dict_ is temporary store to be used as input into the agent(s)
        dict_['dataset'] = self.dataset.retrieve(query)[0].text
        dict_['styling_index'] = self.styling_index.retrieve(query)[0].text
        # short_term memory is stored as hint
        dict_['hint'] = st.session_state.st_memory
        dict_['goal']=query
        dict_['Agent_desc'] = str(self.agent_desc)
        st.write(f"User choose this {specified_agent} to answer this ")


        inputs = {x:dict_[x] for x in self.agent_inputs[specified_agent.strip()]}
        # creates the hint to passed into the agent(s)
        inputs['hint'] = str(dict_['hint']).replace('[','').replace(']','')
        # output dict stores all the information needed
        output_dict ={}
        # input sent to specified_agent
        output_dict[specified_agent.strip()]=self.agents[specified_agent.strip()](**inputs)
        # loops through the output Prediction object (converted as dict)
        for x in dict(output_dict[specified_agent.strip()]).keys():
            if x!='rationale':
                st.code(f"{specified_agent.strip()}[{x}]: {str(dict(output_dict[specified_agent.strip()])[x]).replace('#','#######')}")
                #append in messages for streamlit
                st.session_state.messages.append(f"{specified_agent.strip()}[{x}]: {str(dict(output_dict[specified_agent.strip()])[x])}")
        #sends agent output to memory
        output_dict['memory_'+specified_agent.strip()] = str(self.memory_summarize_agent(agent_response=specified_agent+' '+output_dict[specified_agent.strip()]['code']+'\n'+output_dict[specified_agent.strip()]['commentary'], user_goal=query).summary)
        # adds agent action summary as memory
        st.session_state.st_memory.insert(0,f"{'memory_'+specified_agent.strip()} : {output_dict['memory_'+specified_agent.strip()]}")


        return output_dict





# This is the auto_analyst with planner
class auto_analyst(dspy.Module):
    def __init__(self,agents,retrievers):
        #Initializes all the agents, and makes retrievers
       
        #agents stores the DSPy module for each agent
        # agent_inputs contains all the inputs to use each agent
        # agent desc contains description on what the agent does 
       

        self.agents = {}
        self.agent_inputs ={}
        self.agent_desc =[]
        i =0
        #loops through to create module from agent signatures
        #creates a dictionary with the exact inputs for agents stored
        for a in agents:
            name = a.__pydantic_core_schema__['schema']['model_name']
            self.agents[name] = dspy.ChainOfThought(a)
            self.agent_inputs[name] ={x.strip() for x in str(agents[i].__pydantic_core_schema__['cls']).split('->')[0].split('(')[1].split(',')}
            self.agent_desc.append(str(a.__pydantic_core_schema__['cls']))
            i+=1
        
        # planner agent routes and gives a plan
        # goal_refine is only sent when query is not routed by the planner
        # code_combiner agent helps combine different agent output as a single script
        self.planner = dspy.ChainOfThought(analytical_planner)
        self.refine_goal = dspy.ChainOfThought(goal_refiner_agent)
        self.code_combiner_agent = dspy.ChainOfThought(code_combiner_agent)
        self.story_teller = dspy.ChainOfThought(story_teller_agent)
        self.insights_agent = dspy.ChainOfThought(insights_agent)
        self.memory_summarize_agent = dspy.ChainOfThought(memory_summarize_agent)
                
        # two retrievers defined, one dataset and styling index
        self.dataset = retrievers['dataframe_index'].as_retriever(k=1)
        self.styling_index = retrievers['style_index'].as_retriever(similarity_top_k=1)
        
    def forward(self, query):
        dict_ ={}
        
        # output_dict 
        dict_ ={}
        #dict_ is temporary store to be used as input into the agent(s)
        dict_['dataset'] = self.dataset.retrieve(query)[0].text
        dict_['styling_index'] = self.styling_index.retrieve(query)[0].text
        # short_term memory is stored as hint
        dict_['hint'] = st.session_state.st_memory
        dict_['goal']=query
        dict_['Agent_desc'] = str(self.agent_desc)
        #percent complete is just a streamlit component
        percent_complete =0
        # output dict stores all the information needed

        output_dict ={}
        #tracks the progress
        my_bar = st.progress(0, text="**Planner Agent Working on devising a plan**")
        # sends the query to the planner agent to come up with a plan
        plan = self.planner(goal =dict_['goal'], dataset=dict_['dataset'], Agent_desc=dict_['Agent_desc'] )
        st.write("**This is the proposed plan**")
        st.session_state.messages.append(f"planner['plan']: {plan['plan']}")
        st.session_state.messages.append(f"planner['plan_desc']: {plan['plan_desc']}")

        len_ = len(plan.plan.split('->'))+2
        percent_complete += 1/len_
        my_bar.progress(percent_complete, text=" Delegating to Agents")


        output_dict['analytical_planner'] = plan
        plan_list =[]
        code_list =[]
        analysis_list = [plan.plan,plan.plan_desc]
        #splits the plan and shows it to the user
        if plan.plan.split('->'):
            plan_text = plan.plan
            plan_text = plan.plan.replace('Plan','').replace(':','').strip()
            st.write(plan_text)
            st.write(plan.plan_desc)
            plan_list = plan_text.split('->')
        else:
            # if the planner agent fails at routing the query to any agent this is triggered
            refined_goal = self.refine_goal(dataset=dict_['dataset'], goal=dict_['goal'], Agent_desc= dict_['Agent_desc'])
            st.session_state.messages.append(f"refined_goal: {refined_goal.refined_goal}")

            self.forward(query=refined_goal.refined_goal)
       #Loops through all of the agents in the plan
        for p in plan_list:
            # fetches the inputs
            inputs = {x:dict_[x] for x in self.agent_inputs[p.strip()]}
            output_dict[p.strip()]=self.agents[p.strip()](**inputs)
            code = output_dict[p.strip()].code
            
            # st.write("This is the generated Code"+ code)
            commentary = output_dict[p.strip()].commentary
            st.write('**'+p.strip().capitalize().replace('_','  ')+' -  is working on this analysis....**')
            st.session_state.messages.append(f"{p.strip()}['code']: {output_dict[p.strip()].code}")
            st.session_state.messages.append(f"{p.strip()}['commentary']: {output_dict[p.strip()].commentary}")


            st.write(commentary.replace('#',''))
            st.code(code)
            percent_complete += 1/len_
            my_bar.progress(percent_complete)
            # stores each of the individual agents code and commentary into seperate lists
            code_list.append(code)
            analysis_list.append(commentary)
        st.write("Combining all code into one")
        output_dict['code_combiner_agent'] = self.code_combiner_agent(agent_code_list = str(code_list), dataset=dict_['dataset'])
        st.session_state.messages.append(f"code_combiner_agent: {output_dict['code_combiner_agent']}")
        my_bar.progress(percent_complete + 1/len_, text=" Combining WorkFlow")

        my_bar.progress(100, text=" Compiling the story")
        # creates a summary from code_combiner agent
        output_dict['memory_combined'] = str(self.memory_summarize_agent(agent_response='code_combiner_agent'+'\n'+str(output_dict['code_combiner_agent'].refined_complete_code), user_goal=query).summary)
        st.session_state.st_memory.insert(0,f"{'memory_combined'} : {output_dict['memory_combined']}")

        return output_dict


# This file handles the data-preprocessing and creates retrievers

import pandas as pd
import numpy as np
import datetime

# instructions also stored here
instructions ="""
Here are the instructions for the AI system with the specified agents:

### AI System Instructions

#### Agents
- `@data_viz_agent`: Handles queries related to data visualization.
- `@sk_learn_agent`: Handles queries related to machine learning using scikit-learn.
- `@statistical_analytics_agent`: Handles queries related to statistical analysis.
- `@preprocessing_agent`: Handles queries related to data preprocessing
- `@insights_agent`: Handles queries related to insights, knowledge discovery and summarization.


#### Query Routing

1. **Direct Agent Routing**:
    - If the user specifies an agent in their query using `@agent_name`, the query will be directly routed to the specified agent.
    - Example: `@data_viz_agent Create a bar chart from the following data.`

2. **Planner-Based Routing**:
    - If the user does not specify an agent, the query will be routed to the system's planner.
    - The planner will analyze the query and determine the most appropriate agent to handle the request.
    - Example: `Generate a confusion matrix from this dataset.`

PLEASE READ THE INSTRUCTIONS! Thank you
"""

# For every column collects some useful information like top10 categories and min,max etc if applicable
def return_vals(df,c):
    if isinstance(df[c].iloc[10], (int, float, complex)):
        return {'max_value':max(df[c]),'min_value': min(df[c]), 'mean_value':np.mean(df[c])}
    elif(isinstance(df[c].iloc[10],datetime.datetime)):
        return {str(max(df[c])), str(min(df[c])), str(np.mean(df[c]))}
    else:
        return {'top_10_values':df[c].value_counts()[:10], 'total_categoy_count':len(df[c].unique())}
    
#removes `,` from numeric columns
def correct_num(df,c):
    try:
        df[c] = df[c].fillna('0').str.replace(',','').astype(float)
        return df[c]
    except:
        return df[c]



# does most of the pre-processing
def make_data(df, desc):
    dict_ = {}
    dict_['df_name'] = "The data is loaded as df"
    dict_['Description'] = desc
    dict_['dataframe_head_view'] = df.head(5).to_markdown()
    dict_['all_column_names'] = str(list(df.columns))

        
    for c in df.columns:

        df[c] = correct_num(df,c)
        

        try:
            dict_[c] = {'column_name':c,'type':str(type(df[c].iloc[0])), 'column_information':return_vals(df,c)}
        except:
            dict_[c] = {'column_name':c,'type':str(type(df[c].iloc[0])), 'column_information':'NA'}

    
    
    return dict_



# These are stored styling instructions for data_viz_agent, helps generate good graphs
styling_instructions =[
    """
        Dont ignore any of these instructions.
        For a line chart always use plotly_white template, reduce x axes & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label and try to use multiple colors if more than one line
        Annotate the min and max of the line
        Display numbers in thousand(K) or Million(M) if larger than 1000/100000 
        Show percentages in 2 decimal points with '%' sign
        Default size of chart should be height =1200 and width =1000
        
        """
        
   , """
        Dont ignore any of these instructions.
        For a bar chart always use plotly_white template, reduce x axes & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label 
        Always display numbers in thousand(K) or Million(M) if larger than 1000/100000. 
        Annotate the values of the bar chart
        If variable is a percentage show in 2 decimal points with '%' sign.
        Default size of chart should be height =1200 and width =1000
        """
        ,

          """
        For a histogram chart choose a bin_size of 50
        Do not ignore any of these instructions
        always use plotly_white template, reduce x & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label 
        Always display numbers in thousand(K) or Million(M) if larger than 1000/100000. Add annotations x values
        If variable is a percentage show in 2 decimal points with '%'
        Default size of chart should be height =1200 and width =1000
        """,


          """
        For a pie chart only show top 10 categories, bundle rest as others
        Do not ignore any of these instructions
        always use plotly_white template, reduce x & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label 
        Always display numbers in thousand(K) or Million(M) if larger than 1000/100000. Add annotations x values
        If variable is a percentage show in 2 decimal points with '%'
        Default size of chart should be height =1200 and width =1000
        """,

          """
        Do not ignore any of these instructions
        always use plotly_white template, reduce x & y axes line to 0.2 & x & y grid width to 1. 
        Always give a title and make bold using html tag axis label 
        Always display numbers in thousand(K) or Million(M) if larger than 1000/100000. Add annotations x values
        Don't add K/M if number already in , or value is not a number
        If variable is a percentage show in 2 decimal points with '%'
        Default size of chart should be height =1200 and width =1000
        """,
"""
    For a heat map
    Use the 'plotly_white' template for a clean, white background. 
    Set a chart title 
    Style the X-axis with a black line color, 0.2 line width, 1 grid width, format 1000/1000000 as K/M
    Do not format non-numerical numbers 
    .style the Y-axis with a black line color, 0.2 line width, 1 grid width format 1000/1000000 as K/M
    Do not format non-numerical numbers 

    . Set the figure dimensions to a height of 1200 pixels and a width of 1000 pixels.
""",
"""
    For a Histogram, used for returns/distribution plotting
    Use the 'plotly_white' template for a clean, white background. 
    Set a chart title 
    Style the X-axis  1 grid width, format 1000/1000000 as K/M
    Do not format non-numerical numbers 
    .style the Y-axis, 1 grid width format 1000/1000000 as K/M
    Do not format non-numerical numbers 
    
    Use an opacity of 0.75

     Set the figure dimensions to a height of 1200 pixels and a width of 1000 pixels.
"""
       
         ]




# This file is for the streamlit frontend

# All imports
#from agents import *

import streamlit as st
import os
import statsmodels.api as sm
from streamlit_feedback import streamlit_feedback
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from io import StringIO
import traceback
import contextlib
import sys
import plotly as px
import matplotlib.pyplot as plt
import distutils


def reset_everything():
    st.cache_data.clear()

# stores std output generated on the console to be shown to the user on streamlit app
@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

# Markdown changes made to the sidebar
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ff8080;
        color: #ffffff;
        title-color:#ffffff
    }
</style>
<style>
  [data-testid=stSidebar] h1 {
    color: #ffffff

   }
</style>
""", unsafe_allow_html=True)


# Load the agents into the system
agent_names= [insights_agent,sk_learn_agent,statistical_analytics_agent,preprocessing_agent, data_viz_agent]

# Configure the LLM to be ChatGPT-4o-mini
# You can change this to use your particular choice of LLM
dspy.configure(lm = dspy.OpenAI(model='gpt-4o-mini',api_key=os.environ['sk-jB37doZUJvUbVsEs3nCs0_TYSlDcMQVK6810TH1TqcT3BlbkFJEF2yImEbeyLQdNZTJJx7od9IVE8LW7OsFazuHc8NAA'], max_tokens=16384))

# dspy.configure(lm =dspy.GROQ(model='llama3-70b-8192', api_key =os.environ.get("GROQ_API_KEY"),max_tokens=10000 ) )

# sets the embedding model to be used as OpenAI default
Settings.embed_model = OpenAIEmbedding(api_key=os.environ["OPENAI_API_KEY"])

# Imports images
st.image('./images/Auto-analysts icon small.png', width=70)
st.title("Auto-Analyst")
    

# asthetic features for streamlit app
st.logo('./images/Auto-analysts icon small.png')
st.sidebar.title(":white[Auto-Analyst] ")
st.sidebar.text("Have all your Data Science ")
st.sidebar.text("Analysis Done!")

# creates the file_uploader
uploaded_file = st.file_uploader("Upload your file here...", on_change=reset_everything())
st.write("You can upload your own data or use sample data by clicking the button below")
# creates a button to use the sample data
sample_data = st.button("Use Sample Data")
if sample_data:

    uploaded_file = "Housing.csv"

# more markdown changes
st.markdown(
    """
    <style>
    .css-1l1u5u8 code {
        color: black; /* Change this to your desired color */
        background-color: #f5f5f5; /* Optional: change background color if needed */
        padding: 2px 4px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# displays the instructions for the user
st.markdown(instructions)


retrievers = {}
# df = pd.read_csv('open_deals_min2.csv')
#initializes the uploaded_df or sample df into a dataframe
# also caches that data for performance
@st.cache_data
def initialize_data(button_pressed=False):
    if button_pressed==False:
        uploaded_df = pd.read_csv(uploaded_file, parse_dates=True)
    else:
        uploaded_df = pd.read_csv("Housing.csv")
        st.write("LOADED")
    return uploaded_df


#initializes the planner based system
@st.cache_resource
def intialize_agent():

    return auto_analyst(agents=agent_names,retrievers=retrievers)

#intializes the independent components
@st.cache_resource
def initial_agent_ind():
    return auto_analyst_ind(agents=agent_names,retrievers=retrievers)

#initializes the two retrievers one for data, the other for styling to be used by visualization agent
@st.cache_data(hash_funcs={StringIO: StringIO.getvalue})
def initiatlize_retrievers(_styling_instructions, _doc):
    retrievers ={}
    style_index =  VectorStoreIndex.from_documents([Document(text=x) for x in _styling_instructions])
    retrievers['style_index'] = style_index
    retrievers['dataframe_index'] =  VectorStoreIndex.from_documents([Document(text=x) for x in _doc])

    return retrievers
    
# a simple function to save the output 
def save():
    filename = 'output2.txt'
    outfile = open(filename, 'a')
    
    outfile.writelines([str(i)+'\n' for i in st.session_state.messages])
    outfile.close()


# Defines how the chat system works
def run_chat():
    # defines a variable df (agent code often refers to the dataframe as that)
    if 'df' in st.session_state:
        df = st.session_state['df']
        if df is not None:
            st.write(df.head(5))
            if "show_placeholder" not in st.session_state:
                st.session_state.show_placeholder = True
        else:
            st.error("No data uploaded yet, please upload a file or use sample data")
   
    # Placeholder text to display above the chat box
    placeholder_text = "Welcome to Auto-Analyst, How can I help you? You can use @agent_name to call a specific agent or let the planner route the query!"

    # Display the placeholder text above the chat box
    if "show_placeholder" in st.session_state and st.session_state.show_placeholder:
        st.markdown(f"**{placeholder_text}**")

    # User input taken here    
    user_input = st.chat_input("What are the summary statistics of the data?")

    # Once the user enters a query, hide the placeholder text
    if user_input:
        st.session_state.show_placeholder = False


    # If user has given input or query
    if user_input:
        # this chunk displays previous interactions
        if st.session_state.messages!=[]:
            for m in st.session_state.messages:
                if '-------------------------' not in m:
                    st.write(m.replace('#','######'))






        st.session_state.messages.append('\n------------------------------------------------NEW QUERY------------------------------------------------\n')
        st.session_state.messages.append(f"User: {user_input}")
        
        #all the agents the user mentioned by name to be stored in this list
        specified_agents = []
        # checks for each agent if it is mentioned in the query
        for a in agent_names: 
            if a.__pydantic_core_schema__['schema']['model_name'] in user_input.lower():
                specified_agents.insert(0,a.__pydantic_core_schema__['schema']['model_name'])

    # this is triggered when user did not mention any of the agents in the query
    # this is the planner based routing
        if specified_agents==[]:



            # Generate response in a chat message object
            with st.chat_message("Auto-Anlyst Bot",avatar="🚀"):
                st.write("Responding to "+ user_input)
                # sends the query to the chat system
                output=st.session_state['agent_system_chat'](query=user_input)
                #only executes output from the code combiner agent
                execution = output['code_combiner_agent'].refined_complete_code.split('```')[1].replace('#','####').replace('python','')
                st.markdown(output['code_combiner_agent'].refined_complete_code)
                
                # Tries to execute the code and display the output generated from the console
                try:
                    
                    with stdoutIO() as s:
                        exec(execution)
                       
                    st.write(s.getvalue().replace('#','########'))

                    
                # If code generates an error (testing code fixing agent will be added here)
                except:

                    e = traceback.format_exc()
                    st.markdown("The code is giving an error on excution "+str(e)[:1500])
                    st.write("Please help the code fix agent with human understanding")
                    user_given_context = st.text_input("Help give additional context to guide the agent to fix the code", key='user_given_context')
                    st.session_state.messages.append(user_given_context)

    # this is if the specified_agent list is not empty, send to individual mentioned agents
        else:
            for spec_agent in specified_agents:
                with st.chat_message(spec_agent+" Bot",avatar="🚀"):
                    st.markdown("Responding to "+ user_input)
                    # only sends to the specified agents 
                    output=st.session_state['agent_system_chat_ind'](query=user_input, specified_agent=spec_agent)

                    # Fail safe sometimes code output not structured correctly
                    if len(output[spec_agent].code.split('```'))>1:
                        execution = output[spec_agent].code.split('```')[1].replace('#','####').replace('python','').replace('fig.show()','st.plotly_chart(fig)')
                    else:
                        execution = output[spec_agent].code.split('```')[0].replace('#','####').replace('python','').replace('fig.show()','st.plotly_chart(fig)')


                    # does the code execution and displays it to the user
                    try:
                        
                        with stdoutIO() as s:
                            exec(execution)
                    

                        st.write(s.getvalue().replace('#','########'))


                        
                # If code generates an error (testing code fixing agent will be added here)

                    except:

                        e = traceback.format_exc()
                        st.markdown("The code is giving an error on excution "+str(e)[:1500])
                        st.write("Please help the code fix agent with human understanding")
                        user_given_context = st.text_input("Help give additional context to guide the agent to fix the code", key='user_given_context')
                        st.session_state.messages.append(user_given_context)


# simple feedback form to capture the user's feedback on the answers
        with st.form('form'):
            streamlit_feedback(feedback_type="thumbs", optional_text_label="Do you like the response?", align="flex-start")

            st.session_state.messages.append('\n---------------------------------------------------------------------------------------------------------\n')
            st.form_submit_button('Save feedback',on_click=save())






# initializes some variables in the streamlit session state
# messages used for storing query and agent responses
if "messages" not in st.session_state:
    st.session_state.messages = []
# thumbs used to store user feedback
if "thumbs" not in st.session_state:
    st.session_state.thumbs = ''
#stores df
if "df" not in st.session_state:
    st.session_state.df = None
#stores short-term memory
if "st_memory" not in st.session_state:
    st.session_state.st_memory = []

# if user has uploaded a file or used our sample data
if uploaded_file or sample_data:
    # intializes the dataframe
    st.session_state['df'] = initialize_data()
    
    st.write(st.session_state['df'].head())
    # if user asked for sample data
    if sample_data:
        desc = "Housing Dataset"
        doc=[str(make_data(st.session_state['df'],desc))]
    # if user uploaded their own data
    else:
        # They give a small description so the LLM/Agent can be given additional context
        desc = st.text_input("Write a description for the uploaded dataset")
        doc=['']
        if st.button("Start The Analysis"):

            dict_ = make_data(st.session_state['df'],desc)
            doc = [str(dict_)]

# this initializes the retrievers 
    if doc[0]!='':
        retrievers = initiatlize_retrievers(styling_instructions,doc)
        
        st.success('Document Uploaded Successfully!')
        st.session_state['agent_system_chat'] = intialize_agent()
        st.session_state['agent_system_chat_ind'] = initial_agent_ind()
        st.write("Begin")
    



# saves user feedback if given
if st.session_state['thumbs']!='':
    filename = 'output2.txt'
    outfile = open(filename, 'a',encoding="utf-8")
    
    outfile.write(str(st.session_state.thumbs)+'\n')
    outfile.write('\n------------------------------------------------END QUERY------------------------------------------------\n')

    outfile.close()
    st.session_state['thumbs']=''
    st.write("Saved your Feedback")





run_chat()

#shortens the short-term memory to only include previous 10 interactions
if len(st.session_state.st_memory)>10:
    st.session_state.st_memory = st.session_state.st_memory[:10]


