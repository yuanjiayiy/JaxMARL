from dataclasses import field, replace
from typing import Dict, List
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

@struct.dataclass
class Company:
    initial_capital: float = 6
    capital: float = 6
    beta: float = 0.1667
    initial_resilience: float = 0.07
    resilience: float = 0.07
    resilience_incr_rate: float = 3
    cumu_mitigation_amount: float = 0
    cumu_greenwash_amount: float = 0
    cumu_resilience_amount: float = 0
    margin: float = 0
    capital_gain: float = 0
    mitigation_pc: float = 0
    greenwash_pc: float = 0
    resilience_pc: float = 0
    mitigation_amount: float = 0
    greenwash_amount: float = 0
    resilience_amount: float = 0
    esg_score: float = 0
    bankrupt: bool = False
    greenwash_esg_coef: float = 2

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
    
    def make_esg_decision(self, strategy):
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


    def update_capital(self, environment):
        """Update the capital based on market performance and climate event."""
        # add a random disturbance to market performance
        company_performance = random.normal(random.PRNGKey(0), shape=()) * self.beta + environment.market_performance
        # New capital considering mitigation, resilience, and greenwashing investments
        new_capital = self.capital * (1 - self.mitigation_pc - self.resilience_pc - self.greenwash_pc) * company_performance
        # Climate event impact on capital
        if environment.climate_event_occurrence > 0:
            new_capital *= (1 - self.resilience) ** environment.climate_event_occurrence
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
    
    def reset(self):
        """Reset the company to the initial state."""
        return self.replace(
            capital=self.initial_capital,
            resilience=self.initial_resilience,
            mitigation_pc=0,
            mitigation_amount=0,
            greenwash_pc=0,
            greenwash_amount=0,
            resilience_pc=0,
            resilience_amount=0,
            cumu_resilience_amount=0,
            cumu_mitigation_amount=0,
            cumu_greenwash_amount=0,
            margin=0,
            capital_gain=0,
            esg_score=0,
            bankrupt=False
        )

@struct.dataclass
class Investor:
    initial_capital: float = 6
    cash: float = 6
    capital: float = 6
    investments: Dict[int, float] = field(default_factory=dict)
    esg_preference: float = 0
    utility: float = 0
    
    def initial_investment(self, environment):
        """Invest in all companies at the beginning of the simulation."""
        # Create a new dictionary with investments set to 0 for each company
        investments = {i: 0 for i in range(environment.num_companies)}
        # Return a new object with the updated investments dictionary
        return self.replace(investments=investments)

    
    def invest(self, amount, company_idx):
        """Invest a certain amount in a company. 
        At the end of each period, investors collect all returns and then redistribute capital in next round."""
        if self.cash < amount:
            raise ValueError("Investment amount exceeds available capital.")
        else:
            # Update cash
            new_cash = self.cash - amount
            # Convert investments dict to array
            investments_array = jnp.array(list(self.investments.values()))
            # Update investment for the specified company immutably
            new_investments_array = jax.ops.index_update(investments_array, company_idx, investments_array[company_idx] + amount)
            # Convert array back to dict
            new_investments = dict(zip(self.investments.keys(), new_investments_array))
            # Return a new Investor object with updated cash and investments
            return self.replace(cash=new_cash, investments=new_investments)

    def update_investment_returns(self, environment):
        """Update the capital based on market performance and climate event."""
        # Extract investments and margins into arrays
        investments = jnp.array(list(self.investments.values()))  # Convert investments to JAX array
        company_indices = jnp.array(list(self.investments.keys()))
        margins = jnp.array([environment.companies[idx].margin for idx in company_indices])
        # Define the investment update function
        def update_investment(investment, margin):
            return jnp.maximum(investment * (1 + margin), 0) # Ensure no negative investment
        # Use vmap to apply update_investment over all investments and margins
        updated_investments = vmap(update_investment)(investments, margins)
        # Update self.investments with the new values
        new_investments = dict(zip(company_indices, updated_investments))
        return self.replace(investments=new_investments)

    def divest(self, investor_idx, company_idx, environment):
        """Divest from a company."""
        # Get the current investment return from the company
        investment_return = self.investments[company_idx]
        # Update cash by adding the investment return (immutably)
        new_cash = self.cash + investment_return
        # Convert investments dict to an array for better compatibility with JAX
        investments_array = jnp.array(list(self.investments.values()))
        # Update the investments array immutably using JAX's index_update
        new_investments_array = jax.ops.index_update(investments_array, company_idx, 0)
        # Convert the updated array back to a dictionary
        new_investments = dict(zip(self.investments.keys(), new_investments_array))
        # Update the company in the environment immutably
        # Immutably update the company in the environment
        updated_company = environment.companies[company_idx].lose_investment(investment_return)
        # Create a new dictionary of companies, where only the updated company is changed
        new_companies = {
            idx: (updated_company if idx == company_idx else environment.companies[idx])
            for idx in environment.companies
        }
        return self.replace(cash=new_cash, investments=new_investments), environment.replace(companies=new_companies)

    
    def calculate_utility(self, environment):
        """Calculate reward based on market performance and ESG preferences."""
        # Check if capital is zero
        if self.capital == 0:
            return self.replace(utility=0)
        # Convert investments and company indices to JAX arrays
        investments = jnp.array(list(self.investments.values()))
        company_indices = jnp.array(list(self.investments.keys()))
        # Extract ESG scores for the companies
        esg_scores = jnp.array([environment.companies[idx].esg_score for idx in company_indices])
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
        return self.replace(capital=new_capital, utility=profit_rate + self.esg_preference * avg_esg_reward)
    
    def reset(self):
        """Reset the investor to the initial state."""
        return self.replace(
            capital=self.initial_capital,
            cash=self.initial_capital,
            investments={i: 0 for i in self.investments},
            utility=0
        )


@struct.dataclass
class State:

    time: int
    terminal: bool
    market_performance: float = 1
    climate_event_probability: float = 0.5
    climate_event_occurrence: int = 0
    companies_capitals: jnp.array = field(default_factory=lambda: jnp.array([]))
    companies_capital_gains: jnp.array = field(default_factory=lambda: jnp.array([]))
    companies_resiliences: jnp.array = field(default_factory=lambda: jnp.array([]))
    companies_esg_scores: jnp.array = field(default_factory=lambda: jnp.array([]))
    companies_margins: jnp.array = field(default_factory=lambda: jnp.array([]))
    companies_bankrupts: jnp.array = field(default_factory=lambda: jnp.array([]))
    investor_investments: jnp.array = field(default_factory=lambda: jnp.array([]))
    investor_cash: jnp.array = field(default_factory=lambda: jnp.array([]))
    investor_capitals: jnp.array = field(default_factory=lambda: jnp.array([]))
    investor_utilities: jnp.array = field(default_factory=lambda: jnp.array([]))
    
    
    
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
        initial_climate_event_probability=0.5,
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
            self.investors = [Investor(**attributes) for attributes in investor_attributes]
            self.num_investors = len(investor_attributes)
        else:
            self.num_investors = num_investors
            self.investors = [Investor() for _ in range(num_investors)]

        self.agents = [f"company_{i}" for i in range(self.num_companies)] + [f"investor_{i}" for i in range(self.num_investors)]
        self.n_agents = len(self.agents)
        self.possible_agents = self.agents[:]
        self.market_performance_baseline = market_performance_baseline # initial market performance
        self.market_performance_variance = market_performance_variance # variance of market performance
        self.allow_resilience_investment = allow_resilience_investment # whether to allow resilience investment by companies
        self.allow_greenwash_investment = allow_greenwash_investment # whether to allow greenwash investment by companies
        self.initial_climate_event_probability = initial_climate_event_probability # initial probability of climate event
        self.climate_event_probability = initial_climate_event_probability # current probability of climate event
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


    def update_capital(self, environment):
        """Update the capital based on market performance and climate event."""
        # add a random disturbance to market performance
        company_performance = random.normal(random.PRNGKey(0), shape=()) * self.beta + environment.market_performance
        # New capital considering mitigation, resilience, and greenwashing investments
        new_capital = self.capital * (1 - self.mitigation_pc - self.resilience_pc - self.greenwash_pc) * company_performance
        # Climate event impact on capital
        if environment.climate_event_occurrence > 0:
            new_capital *= (1 - self.resilience) ** environment.climate_event_occurrence
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
    
    def divest(self, state, investor_idx, company_idx):
        """Divest from a company."""
        investments = state.investor_investments[investor_idx]
        cash = state.investor_cash[investor_idx]
        # Get the current investment return from the company
        investment_return = investments[company_idx]
        # Update cash by adding the investment return (immutably)
        new_investor_cash = state.investor_cash.at[investor_idx].set(cash + investment_return)
        # Update the investments array immutably using JAX's index_update
        new_investments = jax.ops.index_update(state.investor_investments, jax.ops.index[investor_idx, company_idx], 0)
        # Update the company in the environment immutably
        # Immutably update the company in the environment
        new_state = self.lose_investment(state, company_idx, investment_return)
        return new_state.replace(investor_cash=new_investor_cash, 
                                 investments=new_investments)


    def step_env(
            self,
            key: chex.PRNGKey,
            state,
            actions: Dict[str, chex.Array],
    ):
        """Step function for the environment."""
        ## unpack actions
        # first num_companies actions are for companies, the rest are for investors
        companys_actions = {k: v for k, v in actions.items() if k.startswith("company_")}
        remaining_actions = {k: v for k, v in actions.items() if k not in companys_actions}
        # Reindex investor actions to start from 0
        investors_actions = {f"investor_{i}": action for i, (k, action) in enumerate(remaining_actions.items())}
        import pdb; pdb.set_trace()
        ## action masks
        # if company is brankrupt, it cannot invest in ESG or greenwashing
        for i, company_bankrupt in enumerate(state.companies_bankrupts):
            if company_bankrupt:
                companys_actions[f"company_{i}"] = jnp.array([0.0, 0.0, 0.0])

        # 0. investors divest from all companies and recollect capital
        # Vectorize the divestment process for all investors
        # updated_investors, updated_env = jax.lax.scan(
        #     lambda carry, investor: self._divest_investor(investor, carry),
        #     self,
        #     self.investors
        # )
        updated_state = self._divest_investor(state=state)

        # Update the environment and investors
        # self = updated_env.replace(investors=updated_investors)

        # 1. investors allocate capital to companies (binary decision to invest/not invest)
        # Use vmap to process all investors
        updated_companies = vmap(self._process_investor_actions)(self.investors, investors_actions, self.companies)
        self = self.replace(companies=updated_companies)
                   
        # 2. companies invest in ESG/greenwashing/none, report margin and esg score
        # Vectorize the process_company function across all companies and their actions
        updated_companies = jax.vmap(self._process_company)(
            self, jnp.array(self.companies), jnp.array([companys_actions[f"company_{i}"] for i in range(len(self.companies))])
        )
        # Update the environment with the new list of companies
        self = self.replace(companies=updated_companies)

        # 3. update probabilities of climate event based on cumulative ESG investments across companies
        total_mitigation_investment = jnp.sum(jnp.array([company.cumu_mitigation_amount for company in self.companies]))
        climate_event_probability =  self.initial_climate_event_probability + 0.014*self.timestamp/(1+0.028*total_mitigation_investment)
        self = self.replace(climate_event_probability=climate_event_probability)

        # 4. market performance and climate event evolution
        rng_key = random.PRNGKey(self.timestamp)
        rng_key, rng_key1, rng_key2 = random.split(rng_key, 3)
        new_market_performance = random.normal(rng_key1) * self.market_performance_variance + self.market_performance_baseline
        climate_event_occurrence = int(self.climate_event_probability) + (random.uniform(rng_key2) < self.climate_event_probability % 1).astype(int)
        self = self.replace(
            market_performance=new_market_performance,
            climate_event_occurrence=climate_event_occurrence
        )

        # 5. companies update capital based on market performance and climate event
        updated_companies = jax.vmap(self._update_company_capital)(jnp.array(self.companies), self)
        updated_investors = jax.vmap(self._update_investor_returns)(jnp.array(self.investors), self)

        # 6. investors calculate returns based on market performance
        final_investors = jax.vmap(self._calculate_investor_utility)(updated_investors, self)
        self = self.replace(companies=updated_companies, investors=final_investors)

        # 7. termination and truncation
        self = self.replace(timestamp=self.timestamp + 1)
        termination = {agent: self.timestamp >= self.max_steps for agent in self.agents}

        # 8. update history
        self = self.replace(history=self._update_history())
        if any(termination.values()):
            self = self.reset()

        return self._get_observation(), self._get_reward(), termination, None


    def _divest_investor(self, state):
        # Vectorized function to apply divestment across all companies
        def divest_company(state, investor_idx, company_idx):
            return self.divest(state=state, investor_idx=investor_idx, company_idx=company_idx)
        
        # Vectorize divestment across all companies in investor's investments
        new_state = jax.lax.scan(
            divest_company,
            state,
            jnp.arrange(len(self.investors)),
            jnp.arrange(len(self.companies))
        )
        
        return new_state
    
    def _process_investor_actions(investor, investor_action, companies):
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
    
    def _process_company(environment, company, action):
        """Update company's actions and make ESG decisions."""
        
        # Skip if the company is bankrupt
        if company.bankrupt:
            return company

        # Unpack actions and apply conditions for greenwash and resilience investment
        mitigation_pc, greenwash_pc, resilience_pc = action
        greenwash_pc = greenwash_pc if environment.allow_greenwash_investment else 0.0
        resilience_pc = resilience_pc if environment.allow_resilience_investment else 0.0

        # Update the company's percentages and make ESG decisions immutably
        updated_company = company.replace(
            mitigation_pc=mitigation_pc,
            greenwash_pc=greenwash_pc,
            resilience_pc=resilience_pc
        )

        # Call the ESG decision-making method (must return an updated company)
        return updated_company.make_esg_decision()

    def _update_company_capital(company, environment):
        return company.update_capital(environment) if not company.bankrupt else company
    
    def _update_investor_returns(investor, environment):
        return investor.update_investment_returns(environment)
    
    def _calculate_investor_utility(investor, environment):
        return investor.calculate_utility(environment)

    def _get_observation(self, state: State):
        """Get observation for each company and investor. Public information is shared across all agents."""

        # Vectorized function to map over companies
        def vectorized_get_company_obs(state):
            capital, resilience, esg_score, margin = state.companies_capitals, state.companies_resiliences, state.companies_esg_scores, state.companies_margins
            return jnp.stack([capital, resilience, esg_score, margin], axis=1)

        # Vectorized function to map over investors
        def vectorized_get_vestor_obs(state):
            investments, capitals = state.investor_investments, state.investor_capitals[:, jnp.newaxis]
            return jnp.concatenate([investments, capitals], axis=1)

        company_obs = vectorized_get_company_obs(state)
        investor_obs = vectorized_get_vestor_obs(state)
        full_obs = jnp.concatenate([company_obs.flatten(), investor_obs.flatten()])
        return {agent: full_obs for agent in self.agents}

    def _get_infos(self):
        return {}

    def _get_reward(self):
        """Get reward for all agents."""
        # Helper function to get company rewards
        def get_company_reward(i, company):
            return (f"company_{i}", company.capital_gain)
        # Helper function to get investor rewards
        def get_investor_reward(i, investor):
            return (f"investor_{i}", investor.utility)
        # Use vmap to vectorize over companies and investors
        company_indices = jnp.arange(len(self.companies))
        investor_indices = jnp.arange(len(self.investors))
        company_rewards = jax.vmap(get_company_reward)(company_indices, jnp.array(self.companies))
        investor_rewards = jax.vmap(get_investor_reward)(investor_indices, jnp.array(self.investors))
        # Combine both company and investor rewards into a dictionary
        rewards = {**dict(company_rewards), **dict(investor_rewards)}
        return rewards

    def reset(self, key=None):
        """Reset the environment."""
        state = State(
            time=0,
            terminal=False,
            market_performance=1,
            climate_event_probability=self.initial_climate_event_probability,
            climate_event_occurrence=0
        )
        
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
        agents = [f"company_{i}" for i in range(self.num_companies)] + [f"investor_{i}" for i in range(self.num_investors)]
        
        
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

        new_state = state.replace(
            companies_capitals=jnp.array([company.capital for company in self.companies]),
            companies_capital_gains=jnp.array([company.capital_gain for company in self.companies]),
            companies_resiliences=jnp.array([company.resilience for company in self.companies]),
            companies_esg_scores=jnp.array([company.esg_score for company in self.companies]),
            companies_margins=jnp.array([company.margin for company in self.companies]),
            companies_bankrupts=jnp.array([company.bankrupt for company in self.companies]),
            investor_investments=jnp.array([list(investor.investments.values()) for investor in self.investors]),
            investor_cash=jnp.array([investor.capital for investor in self.investors]),
            investor_capitals=jnp.array([investor.capital for investor in self.investors]),
            investor_utilities=jnp.array([investor.utility for investor in self.investors]),
        )

        # Return a new environment object with updated state
        return new_state, self._get_observation(new_state), self._get_infos()


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
        
    def is_terminal(self) -> bool:
        """Check if the environment has reached a terminal state."""
        if self.timestamp >= self.max_steps:
            return True
        
        


if __name__ == "__main__":
    env = InvestESG()
    print(env.action_space())
    print(env.observation_space())

