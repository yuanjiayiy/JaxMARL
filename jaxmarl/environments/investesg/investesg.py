from dataclasses import field, replace
from functools import partial
from typing import Any, Dict, List, Tuple
import jax
import jax.numpy as jnp
from jax import random
from jax import jit
from jax import vmap
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
import numpy as np
import chex
from flax import struct

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns
import itertools
jax.config.update('jax_numpy_dtype_promotion', 'strict')

@struct.dataclass
class Company:
    bankrupt: bool = False
    initial_capital: float = 6.0
    capital: float = 6.0
    beta: float = 0.1667
    initial_resilience: float = 0.07
    resilience: float = 0.07
    resilience_incr_rate: float = 3.0
    cumu_mitigation_amount: float = 0.0
    cumu_greenwash_amount: float = 0.0
    cumu_resilience_amount: float = 0.0
    margin: float = 0.0
    capital_gain: float = 0.0
    mitigation_pc: float = 0.0
    greenwash_pc: float = 0.0
    resilience_pc: float = 0.0
    mitigation_amount: float = 0.0
    greenwash_amount: float = 0.0
    resilience_amount: float = 0.0
    esg_score: float = 0.0
    greenwash_esg_coef: float = 2.0

    # def __init__(self, capital=6, climate_risk_exposure = 0.07, beta = 0.1667, greenwash_esg_coef = 2):
    #     self.initial_capital = capital                      # initial capital, in trillion USD
    #     self.capital = capital                              # current capital, in trillion USD
    #     self.beta = beta                                    # Beta risk factor against market performance

    #     self.initial_resilience \
    #         = climate_risk_exposure                         # initial climate risk exposure
    #     self.resilience \
    #         = climate_risk_exposure                         # capital loss ratio when a climate event occurs
        
    #     self.resilience_incr_rate = 3                 # increase rate of climate resilience
    #     self.cumu_mitigation_amount = 0    # cumulative amount invested in emissions mitigation, in trillion USD
    #     self.cumu_greenwash_amount = 0      # cumulative amount invested in greenwashing, in trillion USD
    #     self.cumu_resilience_amount = 0                   # cumulative amount invested in resilience, in trillion USD

    #     self.margin = 0                                     # single period profit margin
    #     self.capital_gain = 0                               # single period capital gain
        
    #     self.mitigation_pc = 0            # single period investment in emissions mitigation, in percentage of total capital
    #     self.greenwash_pc = 0                             # single period investment in greenwashing, in percentage of total capital
    #     self.resilience_pc = 0                      # single period investment in resilience, in percentage of total capital
        
    #     self.mitigation_amount = 0        # amount of true emissions mitigation investment, in trillion USD
    #     self.greenwash_amount = 0                # amount of greenwashing investment, in trillion USD
    #     self.resilience_amount = 0               # amount of resilience investment, in trillion USD
    #     self.esg_score = 0                                  # signal to be broadcasted to investors: emissions mitigation investment / total capital,
    #                                                         # adjusted for greenwashing
    #     self.bankrupt = False

    #     self.greenwash_esg_coef = greenwash_esg_coef       # coefficient of greenwashing_pc on ESG score

    def receive_investment(self, amount):
        """Receive investment from investors."""
        return self.replace(capital=self.capital + amount)

    def lose_investment(self, amount):
        """Lose investment due to climate event."""
        return self.replace(capital=self.capital - amount)
    
    def make_esg_decision(self):
        """Make a decision on how to allocate capital."""
        
        # Calculate updated investment amounts for a single period
        mitigation_amount = self.mitigation_pc * self.capital
        greenwash_amount = self.greenwash_pc * self.capital
        resilience_amount = self.resilience_pc * self.capital

        # Calculate updated cumulative investment
        cumu_mitigation_amount = self.cumu_mitigation_amount + mitigation_amount
        cumu_greenwash_amount = self.cumu_greenwash_amount + greenwash_amount
        cumu_resilience_amount = self.cumu_resilience_amount + resilience_amount

        # Update resilience
        resilience = self.initial_resilience * jnp.exp(
            -self.resilience_incr_rate * (cumu_resilience_amount / self.capital)
        )

        # Update ESG score
        esg_score = self.mitigation_pc + self.greenwash_pc * self.greenwash_esg_coef

        # Return a new instance of the object with updated values
        return self.replace(
            mitigation_amount=mitigation_amount,
            greenwash_amount=greenwash_amount,
            resilience_amount=resilience_amount,
            cumu_mitigation_amount=cumu_mitigation_amount,
            cumu_greenwash_amount=cumu_greenwash_amount,
            cumu_resilience_amount=cumu_resilience_amount,
            resilience=resilience,
            esg_score=esg_score
        )


    def update_capital(self, state):
        """Update the capital based on market performance and climate event."""
        # add a random disturbance to market performance
        company_performance = random.normal(random.PRNGKey(0), shape=()) * self.beta + state.market_performance
        # New capital considering mitigation, resilience, and greenwashing investments
        new_capital = self.capital * (1 - self.mitigation_pc - self.resilience_pc - self.greenwash_pc) * company_performance
        # Climate event impact on capital

        new_capital = jax.lax.cond(
            state.climate_event_occurrence > 0,
            lambda _: new_capital * (1.0 - self.resilience * state.climate_event_occurrence.astype('float32')),  # if num_investments is 0, return 0
            lambda _: new_capital,  # else calculate the investment amount
            None
        )

        # Calculate margin and capital gain
        capital_gain = new_capital - self.capital
        margin = capital_gain / self.capital
        # Check if bankrupt
        bankrupt = new_capital <= 0.0
        # Return a new object with updated capital, gain, margin, and bankruptcy status
        return self.replace(
            capital=new_capital,
            capital_gain=capital_gain,
            margin=margin,
            bankrupt=bankrupt
        )
    
    
    def reset(self):
        """Reset the company to the initial state."""
        return self.replace(
            capital=self.initial_capital,
            resilience=self.initial_resilience,
            mitigation_pc=.0,
            mitigation_amount=.0,
            greenwash_pc=.0,
            greenwash_amount=.0,
            resilience_pc=.0,
            resilience_amount=.0,
            cumu_resilience_amount=.0,
            cumu_mitigation_amount=.0,
            cumu_greenwash_amount=.0,
            margin=.0,
            capital_gain=.0,
            esg_score=.0,
            bankrupt=False
        )

@struct.dataclass
class Investor:
    investments: chex.Array 
    initial_capital: float = 6.0
    cash: float = 6.0
    capital: float = 6.0
    esg_preference: float = 0.0
    utility: float = 0.0
    
    def initial_investment(self, environment):
        """Invest in all companies at the beginning of the simulation."""
        # Create a new dictionary with investments set to 0 for each company
        investments = jnp.zeros(environment.num_companies)
        # Return a new object with the updated investments dictionary
        return self.replace(investments=investments)

    
    def invest(self, amount: float, company_idx: int):
        """Invest a certain amount in a company. 
        At the end of each period, investors collect all returns and then redistribute capital in next round."""
        # Update cash
        new_cash = self.cash - amount
        # Update investment for the specified company immutably
        new_investments = self.investments.at[company_idx].add(amount)
        # Return a new Investor object with updated cash and investments
        return self.replace(cash=new_cash, investments=new_investments)

    def update_investment_returns(self, state):
        """Update the capital based on market performance and climate event."""
        # Extract investments and margins into arrays
        investments = self.investments # Convert investments to JAX array
        margins = jnp.array([company.margin for company in state.companies])
        # Define the investment update function
        def update_investment(investment, margin):
            return jnp.maximum(investment * (1 + margin), 0) # Ensure no negative investment
        # Use vmap to apply update_investment over all investments and margins
        new_investments = vmap(update_investment)(investments, margins)
        return self.replace(investments=new_investments)
    


    def divest(self, company_idx: int, state) -> Tuple["Investor", "State"]:
        """Divest from a company."""
        # Get the current investment return from the company
        investment_return = self.investments[company_idx]
        # Update cash by adding the investment return (immutably)
        new_cash = self.cash + investment_return
        # Update the investments array immutably using JAX's index_update
        new_investments = self.investments.at[company_idx].set(0)
        # Update the company in the environment immutably
        # Immutably update the company in the environment
        updated_company = state.companies[company_idx].lose_investment(investment_return)
        # Create a new dictionary of companies, where only the updated company is changed
        new_companies = [updated_company if idx == company_idx else state.companies[idx] for idx in range(len(state.companies))]
        return self.replace(cash=new_cash, investments=new_investments), state.replace(companies=new_companies)

    
    def calculate_utility(self, state):
        """Calculate reward based on market performance and ESG preferences."""
        def get_utility():
            # Convert investments and company indices to JAX arrays
            investments = self.investments
            # Extract ESG scores for the companies
            esg_scores = jnp.array([company.esg_score for company in state.companies])
            # Define a helper function to calculate investment balance and ESG reward
            def investment_calc(investment, esg_score):
                return investment, esg_score * investment
            
            # Vectorize the function to apply to all investments
            invest_balances, esg_rewards = vmap(investment_calc)(investments, esg_scores)
            # Calculate total investment balance and ESG reward, skipping zero investments
            total_invest_balance = jnp.sum(invest_balances)
            total_esg_reward = jnp.sum(esg_rewards)
            # Calculate new capital and average ESG reward
            new_capital = total_invest_balance + self.cash
            avg_esg_reward = total_esg_reward / new_capital
            # Calculate profit rate
            profit_rate = (new_capital - self.capital) / self.capital
            return new_capital, profit_rate + self.esg_preference * avg_esg_reward
        # Check if capital is zero
        capital, utility = jax.lax.cond(
                    self.capital > .0,
                    lambda _: get_utility(),  # if num_investments is 0, return 0
                    lambda _: (jnp.array(0.0), jnp.array(0.0)), # else calculate the investment amount
                    None
        )

        return self.replace(capital=capital, utility=utility)
    
    def reset(self):
        """Reset the investor to the initial state."""
        return self.replace(
            capital=self.initial_capital,
            cash=self.initial_capital,
            utility=0
        )


@struct.dataclass
class State:

    time: int
    terminal: bool
    heat_prob: float
    precip_prob: float
    drought_prob: float
    climate_risk: float
    companies: List[Company]
    investors: List[Investor]
    market_performance: float = 1.0
    climate_event_occurrence: int = 0
    
    
    
class InvestESG(MultiAgentEnv):
    """
    JAX Compatible version of ESG investment environment.
    """

    def __init__(
        self,
        company_attributes=None,
        investor_attributes=None,
        num_companies=5,
        num_investors=5,
        initial_heat_prob = 0.28,
        initial_precip_prob = 0.13,
        initial_drought_prob = 0.17,
        max_steps=100,
        market_performance_baseline=1.1, 
        market_performance_variance=0.0,
        allow_resilience_investment=False,
        allow_greenwash_investment=False,
        action_capping=0.1
    ):
        self.max_steps = max_steps
        self.timestamp = 0

        # initialize companies and investors based on attributes if not None
        if company_attributes is not None:
            self.companies = [Company(**attributes) for attributes in company_attributes]
            self.num_companies = len(company_attributes)
        else:
            self.companies = [Company() for _ in range(num_companies)]
            self.num_companies = num_companies
        
        if investor_attributes is not None:
            self.investors = [Investor(**attributes, investments=jnp.zeros(self.num_companies)) for attributes in investor_attributes]
            self.num_investors = len(investor_attributes)
        else:
            self.num_investors = num_investors
            self.investors = [Investor(investments=jnp.zeros(self.num_companies)) for _ in range(num_investors)]

        self.agents = [f"company_{i}" for i in range(self.num_companies)] + [f"investor_{i}" for i in range(self.num_investors)]
        self.n_agents = len(self.agents)
        self.possible_agents = self.agents[:]
        self.market_performance_baseline = market_performance_baseline # initial market performance
        self.market_performance_variance = market_performance_variance # variance of market performance
        self.allow_resilience_investment = allow_resilience_investment # whether to allow resilience investment by companies
        self.allow_greenwash_investment = allow_greenwash_investment # whether to allow greenwash investment by companies

        self.initial_heat_prob = initial_heat_prob # initial probability of heat wave
        self.initial_precip_prob = initial_precip_prob # initial probability of precipitation
        self.initial_drought_prob = initial_drought_prob # initial probability of drought
        self.heat_prob = initial_heat_prob # current probability of heat wave
        self.precip_prob = initial_precip_prob # current probability of precipitation
        self.drought_prob = initial_drought_prob # current probability of drought
        self.initial_climate_risk = 1 - (1-initial_heat_prob)*(1-initial_precip_prob)*(1-initial_drought_prob) # initial probability of at least one climate event
        self.climate_risk = self.initial_climate_risk # current probability of climate event

        self.climate_event_occurrence = 0 # number of climate events occurred in the current step
        self.action_capping = action_capping # action capping for company action
        # initialize investors with initial investments dictionary
        for idx, investor in enumerate(self.investors):
            self.investors[idx] = investor.initial_investment(self)

        # initialize historical data storage
        self.history = {
            "esg_investment": [],
            "greenwash_investment": [],
            "resilience_investment": [],
            "climate_risk": [],
            "climate_event_occurs": [],
            "market_performance": [],
            "market_total_wealth": [],
            "company_rewards": [[] for _ in range(self.num_companies)],
            "investor_rewards": [[] for _ in range(self.num_investors)],
            "company_capitals": [[] for _ in range(self.num_companies)],
            "company_climate_risk": [[] for _ in range(self.num_companies)],
            "investor_capitals": [[] for _ in range(self.num_investors)],
            "investor_utility": [[] for _ in range(self.num_investors)],
            "investment_matrix": np.zeros((self.num_investors, self.num_companies)),
            "company_mitigation_amount": [[] for _ in range(self.num_companies)],
            "company_greenwash_amount": [[] for _ in range(self.num_companies)],
            "company_resilience_amount": [[] for _ in range(self.num_companies)],
            "company_esg_score": [[] for _ in range(self.num_companies)],
            "company_margin": [[] for _ in range(self.num_companies)],
            "company_rewards": [[] for _ in range(self.num_companies)],
            "investor_rewards": [[] for _ in range(self.num_investors)],
        }

    def action_space(self, agent):
        ## Each company makes 3 decisions:
        ## 1. Amount to invest in emissions mitigation (continuous)
        ## 2. amount to invest in greenwash (continuous)
        ## 3. amount to invest in resilience (continuous)
        ## Each investor has num_companies possible*2 actions: for each company, invest/not invest
        # if agent is a company
        if agent.startswith("company"):
            return spaces.Box(low=0, high=self.action_capping, shape=(3,))
        else:
            return spaces.MultiDiscrete(self.num_companies*[2]) # 0: not invest, 1: invest
    
    def observation_space(self):
        # all agents have access to the same information, namely the capital, climate resilience, ESG score, and margin of each company
        # of all companies and the investment in each company and remaining cash of each investor
        observation_size = self.num_companies * 4 + self.num_investors * (self.num_companies + 1)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,))
        return observation_space
    
    def get_companies(self, state: State):
        return {f'company_{k}': v for k, v in enumerate(state.companies)}
    

    def get_investors(self, state: State):
        return {f'investor_{k}': v for k, v in enumerate(state.investors)}
    
    def lose_investment(self, state, company_idx, amount):
        """Lose investment due to climate event."""
        capital = state.companies_capitals.at[company_idx].get()
        return state.replace(companies_capitals=state.companies_capitals.at[company_idx].set(capital - amount))
    
    def receive_investment(self, state, company_idx, amount):
        """Receive investment from investors."""
        capital = state.companies_capitals.at[company_idx].get()
        return state.replace(companies_capitals=state.companies_capitals.at[company_idx].set(capital + amount))
    
    def make_esg_decision(self, state, company_idx, mitigation_pc, greenwash_pc, resilience_pc):
        """Make a decision on how to allocate capital."""
        # Calculate updated investment amounts for a single period
        capital = state.companies_capitals.at[company_idx].get()
        mitigation_amount = mitigation_pc * capital
        greenwash_amount = greenwash_pc * capital
        resilience_amount = resilience_pc * capital

        # Calculate updated cumulative investment
        cumu_mitigation_amount = self.cumu_mitigation_amount + mitigation_amount
        cumu_greenwash_amount = self.cumu_greenwash_amount + greenwash_amount
        cumu_resilience_amount = self.cumu_resilience_amount + resilience_amount

        # Update resilience
        resilience = self.initial_resilience * jnp.exp(
            -self.resilience_incr_rate * (cumu_resilience_amount / self.capital)
        )

        # Update ESG score
        esg_score = mitigation_pc + greenwash_pc * self.companies[company_idx].greenwash_esg_coef

        # Return a new instance of the object with updated values
        return self.replace(
            mitigation_amount=mitigation_amount,
            greenwash_amount=greenwash_amount,
            resilience_amount=resilience_amount,
            cumu_mitigation_amount=cumu_mitigation_amount,
            cumu_greenwash_amount=cumu_greenwash_amount,
            cumu_resilience_amount=cumu_resilience_amount,
            resilience=resilience,
            esg_score=esg_score
        )


    def update_capital(self, state, company_idx):
        """Update the capital based on market performance and climate event."""
        # add a random disturbance to market performance
        company_performance = random.normal(random.PRNGKey(0), shape=()) * self.beta + state.market_performance
        # New capital considering mitigation, resilience, and greenwashing investments
        new_capital = self.capital * (1 - self.mitigation_pc - self.resilience_pc - self.greenwash_pc) * company_performance
        # fdsjfpo j[ Climate event impact on capital
        if state.climate_event_occurrence > 0:
            new_capital *= (1 - self.resilience) ** state.climate_event_occurrence
        # Calculate margin and capital gain
        capital_gain = new_capital - self.capital
        margin = capital_gain / self.capital
        # Check if bankrupt
        bankrupt = new_capital <= 0
        # Return a new object with updated capital, gain, margin, and bankruptcy status
        return self.replace(
            capital=new_capital,
            capital_gain=capital_gain,
            margin=margin,
            bankrupt=bankrupt
        )
    
    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal
    
    def step_env(
            self,
            key: chex.PRNGKey,
            state,
            actions: Dict[str, chex.Array],
    ):

        rng_heat = jax.random.PRNGKey(state.time*100) # random number generator for climate event
        rng_precip = jax.random.PRNGKey(state.time*500) # random number generator for climate event
        rng_drought = jax.random.PRNGKey(state.time*1000) # random number generator for climate event

        """Step function for the environment."""
        ## unpack actions
        # first num_companies actions are for companies, the rest are for investors
        companys_actions = {k: v for k, v in actions.items() if k.startswith("company_")}
        remaining_actions = {k: v for k, v in actions.items() if k not in companys_actions}
        # Reindex investor actions to start from 0
        investors_actions = {f"investor_{i}": action for i, (k, action) in enumerate(remaining_actions.items())}
        companies = self.get_companies(state)
        ## action masks
        # if company is brankrupt, it cannot invest in ESG or greenwashing
        for name, company in companies.items():
            companys_actions[name] = jax.lax.cond(
                company.bankrupt == True,
                lambda _: jnp.array([0.0, 0.0, 0.0]),
                lambda _: companys_actions[name],
                None
            )

        # 0. investors divest from all companies and recollect capital
        # Vectorize the divestment process for all investors
        investors = list(self.get_investors(state).values())

        for investor_idx, investor in enumerate(investors):
            state, investors[investor_idx] = self._divest_investor(state, investors[investor_idx])

        # Update the environment and investors
        state = state.replace(investors=investors)

        # 1. investors allocate capital to companies (binary decision to invest/not invest)
        investors = state.investors
        companies = state.companies
        for i, investor in enumerate(state.investors):
            investor_action = investors_actions[f"investor_{i}"]
            # number of companies that the investor invests in
            num_investments = jnp.sum(investor_action)
            # equal investment in each company; round down to nearest integer to avoid exceeding capital
            def calculate_investment_amount(investor_cash, num_investments):
                # Use jax.lax.cond to check if num_investments is 0
                investment_amount = jax.lax.cond(
                    num_investments > 0,
                    lambda _: jnp.floor(investor_cash / num_investments.astype('float32')),  # if num_investments is 0, return 0
                    lambda _: jnp.array(0.0),  # else calculate the investment amount
                    None
                )
                return investment_amount
            
            investment_amount = calculate_investment_amount(investor.cash, num_investments)
            
            for j, company in enumerate(state.companies):
                def i_invest_in_j(i, j, investment_amount):
                    investor, company = jax.lax.cond(
                        investor_action[j] > 0,
                        lambda _: (investors[i].invest(investment_amount, j), companies[j].receive_investment(investment_amount)), # if not invest, return the same
                        lambda _: (investors[i], companies[j]), # else calculate the investment amount
                        None
                    )
                    return investor, company
                investors[i], companies[j] = i_invest_in_j(i, j, investment_amount)
        state = state.replace(companies=companies, investors=investors)
                   
        # 2. companies invest in ESG/greenwashing/none, report margin and esg score
        # Vectorize the process_company function across all companies and their actions
        for company_idx, company in enumerate(companies):
            companies[company_idx] = self._process_company(company, jnp.array(companys_actions[f"company_{company_idx}"]))
        state = state.replace(companies=companies)

        # 3. update probabilities of climate event based on cumulative ESG investments across companies
        total_mitigation_investment = jnp.sum(jnp.array([company.cumu_mitigation_amount for company in self.companies]))
        self.heat_prob = self.initial_heat_prob + 0.0083*state.time/(1+0.0222*total_mitigation_investment)
        self.precip_prob = self.initial_precip_prob + 0.0018*state.time/(1+0.0326*total_mitigation_investment)
        self.drought_prob = self.initial_drought_prob + 0.003*state.time/(1+0.038*total_mitigation_investment)
        self.climate_risk = 1 - (1-self.heat_prob)*(1-self.precip_prob)*(1-self.drought_prob)
        state = state.replace(
            heat_prob = self.heat_prob,
            precip_prob = self.precip_prob,
            drought_prob = self.drought_prob,
            climate_risk = self.climate_risk
        )

        # 4. market performance and climate event evolution
        rng_key = key
        rng_key, rng_key1, rng_key2 = random.split(rng_key, 3)
        new_market_performance = random.normal(rng_key1) * self.market_performance_variance + self.market_performance_baseline
        heat_event = (random.uniform(random.split(rng_heat)[1]) < self.heat_prob).astype(int)
        precip_event = (random.uniform(random.split(rng_precip)[1]) < self.precip_prob).astype(int)
        drought_event = (random.uniform(random.split(rng_drought)[1]) < self.drought_prob).astype(int)
        climate_event_occurrence = heat_event + precip_event + drought_event
        state = state.replace(
            market_performance=new_market_performance,
            climate_event_occurrence=climate_event_occurrence
        )

        # 5. companies update capital based on market performance and climate event
        for company_idx, company in enumerate(companies):
            companies[company_idx] = self._update_company_capital(company, state)
        state = state.replace(companies=companies)
         # 6. investors calculate returns based on market performance
        for investor_idx, investor in enumerate(investors):
            investors[investor_idx] = self._calculate_investor_utility(self._update_investor_returns(investor, state), state)
        state = state.replace(investors=investors)

        # 7. termination and truncation
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        dones = {"agent_0": done, "agent_1": done, "__all__": done}

        # 8. update history
        # self = self.replace(history=self._update_history())
        # if state.terminal:
        #     state = state.reset()

        return (self._get_observation(state), state, self._get_reward(state), dones, None)


    def _divest_investor(self, state, investor):
        # jax.debug.print("Carry: {}, investor: {}", state, investor)
        # Vectorized function to apply divestment across all companies
        def divest_company(company_idx, state):
            return investor.divest(company_idx, state)
        
        for i in range(self.num_companies):
            investor, state = divest_company(i, state)
        # Vectorize divestment across all companies in investor's investments
        # new_investor, new_state = jax.lax.scan(
        #     lambda carry, company_idx: divest_company(company_idx, carry),
        #     state,
        #     np.arange(num_companies)
        # )
        return state, investor
    
    def _process_investor_actions(self, investor: Investor, investor_action: jnp.array, companies: List[Company]):
        """Process investor actions and update company investments."""
        num_investments = jnp.sum(investor_action)
        if num_investments > 0:
            investment_amount = jnp.floor(investor.cash / num_investments)
            def invest_in_company(company, action):
                return company.receive_investment(investment_amount) if action else company
            # Use vmap to vectorize the investment in each company
            updated_companies = vmap(invest_in_company)(companies, investor_action)
            return updated_companies
        else:
            return companies
    
    def _process_company(self, company, action):
        """Update company's actions and make ESG decisions."""

        def process_company_action():
            # Unpack actions and apply conditions for greenwash and resilience investment
            mitigation_pc, greenwash_pc, resilience_pc = action
            greenwash_pc = greenwash_pc if self.allow_greenwash_investment else 0.0
            resilience_pc = resilience_pc if self.allow_resilience_investment else 0.0

            # Update the company's percentages and make ESG decisions immutably
            updated_company = company.replace(
                mitigation_pc=mitigation_pc,
                greenwash_pc=greenwash_pc,
                resilience_pc=resilience_pc
            )
            # Call the ESG decision-making method (must return an updated company)
            return updated_company.make_esg_decision()
        
        updated_company = jax.lax.cond(
            company.bankrupt,
            lambda _: company,
            lambda _: process_company_action(),
            None
        )
        return updated_company

    def _update_company_capital(self, company, state):
        updated_company = jax.lax.cond(
            company.bankrupt,
            lambda _: company,
            lambda _: company.update_capital(state),
            None
        )
        return updated_company
    
    def _update_investor_returns(self, investor, state):
        return investor.update_investment_returns(state)
    
    def _calculate_investor_utility(self, investor, state):
        return investor.calculate_utility(state)
    
    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> chex.Array:
        obs_batch = self._get_observation(state)
        return obs_batch

    def _get_observation(self, state: State):
        """Get observation for each company and investor. Public information is shared across all agents."""

        # Vectorized function to map over companies
        companies = self.get_companies(state)
        investors = self.get_investors(state)

        def vectorized_get_company_obs():
            capital = jnp.array([company.capital for company in companies.values()])
            resilience = jnp.array([company.resilience for company in companies.values()])
            esg_score = jnp.array([company.esg_score for company in companies.values()])
            margin = jnp.array([company.margin for company in companies.values()])
            # TODO: Add more information such as avg esg, esg spending, resilience spending
            # additional_info = jnp.array((3, len(companies.values())))
            return jnp.stack([capital, resilience, esg_score, margin], axis=1)

        # Vectorized function to map over investors
        def vectorized_get_vestor_obs():
            investments = jnp.array([investor.investments for investor in investors.values()])
            capitals = jnp.array([investor.capital for investor in investors.values()])[:, jnp.newaxis]
            return jnp.concatenate([investments, capitals], axis=1)

        company_obs = vectorized_get_company_obs()
        investor_obs = vectorized_get_vestor_obs()
        
        # TODO: Add climate data
        # climate_obs = jnp.zeros(3)
        full_obs = jnp.concatenate([company_obs.flatten(), investor_obs.flatten()])
        return {agent: full_obs for agent in self.agents}

    def _get_infos(self):
        return {}

    def _get_reward(self, state):
        """Get reward for all agents."""
        # Helper function to get company rewards
        def get_company_reward(i):
            return (f"company_{i}", state.companies[i].capital_gain)
        # Helper function to get investor rewards
        def get_investor_reward(i):
            return (f"investor_{i}", state.investors[i].utility)
        # Use vmap to vectorize over companies and investors
        company_rewards = [get_company_reward(company_idx) for company_idx in range(len(self.companies))]
        investor_rewards = [get_investor_reward(investor_idx) for investor_idx in range(len(self.investors))]
        # Combine both company and investor rewards into a dictionary
        rewards = {**dict(company_rewards), **dict(investor_rewards)}
        return rewards
    
    def reset(self, key=chex.PRNGKey):
        """Reset the environment."""
        
        # Helper function to reset each company
        def reset_company(company):
            return company.reset()

        # Helper function to reset each investor
        def reset_investor(investor):
            return investor.reset()

        # Reset all companies and investors using vmap
        updated_companies = [reset_company(company) for company in self.companies]

        # updated_companies = jax.vmap(reset_company)(jnp.array(self.companies))
        updated_investors = [reset_investor(investor) for investor in self.investors]
        # jax.vmap(reset_investor)(jnp.array(self.investors))

        # Reset the environment attributes immutably
        agents = dict()
        agents.update({f"company_{i}": updated_companies[i] for i in range(self.num_companies)})
        agents.update({f"investor_{i}": updated_investors[i] for i in range(self.num_investors)})
        
        # Reset historical data
        history = {
            "esg_investment": [],
            "greenwash_investment": [],
            "resilience_investment": [],
            "climate_risk": [],
            "climate_event_occurs": [],
            "market_performance": [],
            "market_total_wealth": [],
            "company_capitals": [[] for _ in range(self.num_companies)],
            "company_climate_risk": [[] for _ in range(self.num_companies)],
            "investor_capitals": [[] for _ in range(self.num_investors)],
            "investor_utility": [[] for _ in range(self.num_investors)],
            "investment_matrix": jnp.zeros((self.num_investors, self.num_companies)),
            "company_mitigation_amount": [[] for _ in range(self.num_companies)],
            "company_greenwash_amount": [[] for _ in range(self.num_companies)],
            "company_resilience_amount": [[] for _ in range(self.num_companies)],
            "company_esg_score": [[] for _ in range(self.num_companies)],
            "company_margin": [[] for _ in range(self.num_companies)],
            "company_rewards": [[] for _ in range(self.num_companies)],
            "investor_rewards": [[] for _ in range(self.num_investors)],
        }

        state = State(
            time=0,
            terminal=False,
            market_performance=1.0,
            heat_prob=self.initial_heat_prob,
            precip_prob=self.initial_precip_prob,
            drought_prob=self.initial_drought_prob,
            climate_risk=self.initial_climate_risk,
            climate_event_occurrence=0,
            companies=updated_companies,
            investors=updated_investors
        )

        # Return a new environment object with updated state
        return self._get_observation(state), state


    @property
    def name(self) -> str:
        """Environment name."""
        return "InvestESG"

    def render(self, mode='human', fig='fig'):
        # import pdb; pdb.set_trace()
        
        if not hasattr(self, 'fig') or self.fig is None:
            # Initialize the plot only once
            self.fig = Figure(figsize=(32, 18))
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.subplots(3, 4)  # Adjusted to 2 rows and 6 columns
            plt.subplots_adjust(hspace=0.5, wspace=1)  # Increased wspace from 0.2 to 0.3
            plt.ion()  # Turn on interactive mode for plotting

            # Generate a color for each company
            self.company_colors = plt.cm.rainbow(np.linspace(0, 1, self.num_companies))
            self.investor_colors = plt.cm.rainbow(np.linspace(0, 1, self.num_investors))
        # Ensure self.ax is always a list of axes
        if not isinstance(self.ax, np.ndarray):
            self.ax = np.array([self.ax])

        # Clear previous figures to update with new data
        for row in self.ax:
            for axis in row:
                axis.cla()

        # Subplot 1: Overall ESG Investment and Climate Risk over time
        ax1 = self.ax[0][0]
        ax2 = ax1.twinx()  # Create a secondary y-axis

        ax1.plot(self.history["esg_investment"], label='Cumulative ESG Investment', color='blue')
        ax2.plot(self.history["climate_risk"], label='Climate Risk', color='orange')
        # Add vertical lines for climate events
        for i, event in enumerate(self.history["climate_event_occurs"]):
            if event==1:
                ax1.axvline(x=i, color='orange', linestyle='--', alpha=0.5)
            if event>1:
                ax1.axvline(x=i, color='red', linestyle='--', alpha=0.5)

        ax1.set_title('Overall Metrics Over Time')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Investment in ESG')
        ax2.set_ylabel('Climate Event Probability')
        ax2.set_ylim(0, 2)  # Set limits for Climate Event Probability

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Subplot 2: Company Decisions
        ax = self.ax[0][1]
        for i in range(self.num_companies):
            mitigation = self.history["company_mitigation_amount"][i]
            ax.plot(mitigation, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Mitigation Investments Over Time')
        ax.set_ylabel('Mitigation Investment')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 3: Company Greenwash Decisions
        ax = self.ax[0][2]
        for i in range(self.num_companies):
            greenwash = self.history["company_greenwash_amount"][i]
            ax.plot(greenwash, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Greenwash Investments Over Time')
        ax.set_ylabel('Greenwash Investment')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 4: Company Resilience Decisions
        ax = self.ax[0][3]
        for i in range(self.num_companies):
            resilience = self.history["company_resilience_amount"][i]
            ax.plot(resilience, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Resilience Investments Over Time')
        ax.set_ylabel('Resilience Investment')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 5: Company Climate risk exposure over time
        ax = self.ax[1][0]  
        for i, climate_risk_history in enumerate(self.history["company_climate_risk"]):
            ax.plot(climate_risk_history, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Climate Risk Exposure Over Time')
        ax.set_ylabel('Climate Risk Exposure')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 6: Company Capitals over time
        ax = self.ax[1][1]
        for i, capital_history in enumerate(self.history["company_capitals"]):
            ax.plot(capital_history, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company Capitals Over Time')
        ax.set_ylabel('Capital')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 7: Company ESG Score over time
        ax = self.ax[1][2]
        for i, esg_score_history in enumerate(self.history["company_esg_score"]):
            ax.plot(esg_score_history, label=f'Company {i}', color=self.company_colors[i])
        ax.set_title('Company ESG Score Over Time')
        ax.set_ylabel('ESG Score')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 8: Investment Matrix
        investment_matrix = self.history["investment_matrix"]
        ax = self.ax[1][3]
        sns.heatmap(investment_matrix, ax=ax, cmap='Reds', cbar=True, annot=True, fmt='g')

        ax.set_title('Investment Matrix')
        ax.set_ylabel('Investor ID')
        ax.set_xlabel('Company ID')

         # Subplot 9: Investor Capitals over time
        ax = self.ax[2][0]
        for i, capital_history in enumerate(self.history["investor_capitals"]):
            ax.plot(capital_history, label=f'Investor {i}', color=self.investor_colors[i])
        ax.set_title('Investor Capitals Over Time')
        ax.set_ylabel('Capital')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 10: Investor Utility over time
        ax = self.ax[2][1]
        for i, utility_history in enumerate(self.history["investor_utility"]):
            ax.plot(utility_history, label=f'Investor {i}', color=self.investor_colors[i])
        ax.set_title('Investor Utility Over Time')
        ax.set_ylabel('Utility')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 11: Cumulative Investor Utility over time
        ax = self.ax[2][2]
        for i, utility_history in enumerate(self.history["investor_utility"]):
            cumulative_utility_history = list(itertools.accumulate(utility_history))
            ax.plot(cumulative_utility_history, label=f'Investor {i}', color=self.investor_colors[i])
        ax.set_title('Cumulative Investor Utility Over Time')
        ax.set_ylabel('Cumulative Utility')
        ax.set_xlabel('Timestep')
        ax.legend(loc='upper right')

        # Subplot 12: Market Total Wealth over time
        ax = self.ax[2][3]
        ax.plot(self.history["market_total_wealth"], label='Total Wealth', color='green')
        ax.set_title('Market Total Wealth Over Time')
        ax.set_ylabel('Total Wealth')
        ax.set_xlabel('Timestep')
        ax.legend()

        self.fig.tight_layout()

        # Update the plots
        self.canvas.draw()
        self.canvas.flush_events()
        plt.pause(0.001)  # Pause briefly to update plots

        # TODO: Consider generate videos later
        if mode == 'human':
            plt.show(block=False)
        elif mode == 'rgb_array':
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return img
        
        
if __name__ == "__main__":
    env = InvestESG()
    print(env.action_space())
    print(env.observation_space())

