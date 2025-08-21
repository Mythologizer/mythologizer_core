from mythologizer_core import AgentAttribute
import random

agent_attributes = [
    AgentAttribute(
        name='Age',
        description='Age of the agent',
        d_type=int,
        min=0,
        epoch_change_function= lambda x: x + 1
    ),
    AgentAttribute(
        name='Confidence',
        description='The confidence of the agent',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function= lambda x: x + random.random()
    ),
    AgentAttribute(
        name='Emotionality',
        description='The emotionality of the agent with 0 representing a very emotionless person and 1 representing a very emotional person',
        d_type=float,
        min=0.0,
        max=1.0)
    ]