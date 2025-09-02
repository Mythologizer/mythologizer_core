from mythologizer_core import AgentAttribute
import random

random_fluctation_capped_between_0_1 = lambda x: x + random.random() * (1 - x)

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
        epoch_change_function= random_fluctation_capped_between_0_1
    ),
    AgentAttribute(
        name='Emotionality',
        description='The emotionality of the agent with 0 representing a very emotionless person and 1 representing a very emotional person',
        d_type=float,
        min=0.0,
        max=1.0),
    AgentAttribute(
        name='Creativity',
        description='The creativity of the agent with 0 representing a non-creative person and 1 representing a very creative person',
        d_type=float,
        min=0.0,
        max=1.0),
    AgentAttribute(
        name='Talkativeness',
        description='The talkativeness of the agent with 0 representing a non-talkative person and 1 representing a very talkative person',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function=random_fluctation_capped_between_0_1
    ),
    AgentAttribute(
        name='Funiness',
        description='The funniness of the agent with 0 representing an unfunny person and 1 representing a very funny person',
        d_type=float,
        min=0.0,
        max=1.0),
    AgentAttribute(
        name='Loneliness',
        description='The loneliness of the agent with 0 representing a person who is more of a loner and 1 representing a very community-oriented person',
        d_type=float,
        min=0.0,
        max=1.0),
    AgentAttribute(
        name='Stubbornness',
        description='The stubbornness of the agent with 0 representing a very submissive person and 1 representing a very stubborn person',
        d_type=float,
        min=0.0,
        max=1.0),
    AgentAttribute(
        name='Mysteriousness',
        description='The level of mysteriousness of the agent with 0 representing a '
                    'non-mysterious person and 1 representing a very mysterious person',
        d_type=float,
        min=0.0,
        max=1.0),
    AgentAttribute(
        name='Superstitiousness',
        description='The level of superstition of the agent with 0 representing a very '
                    'rational person and 1 representing a very superstitious person',
        d_type=float,
        min=0.0,
        max=1.0),
    AgentAttribute(
        name='Clumsiness',
        description='The level of clumsiness of the agent with 0 representing a well-handled '
                    'person and 1 representing a very clumsy person',
        d_type=float,
        min=0.0,
        max=1.0),
    AgentAttribute(
        name='Absurdity',
        description='The level of absurdity of the agent with 0 representing a logical person '
                    'and 1 representing a very absurd person',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function=random_fluctation_capped_between_0_1
    ),
    AgentAttribute(
        name='Rebelliousness',
        description='The level of rebelliousness of the agent with 0 representing a '
                    'rule-following person and 1 representing a very rebellious person',
        d_type=float,
        min=0.0,
        max=1.0,
        epoch_change_function=random_fluctation_capped_between_0_1
    ),
    AgentAttribute(
        name='Closeness to Nature',
        description='The level of how close an agent is to nature with 0 '
                    'representing an indoors person and 1 representing a very active '
                    'and outdoorsy person',
        d_type=float,
        min=0.0,
        max=1.0)
    ]