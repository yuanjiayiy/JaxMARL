import jax
import jax.numpy as jnp
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
import numpy as np
import matplotlib.pyplot as plt

class Company:
    def __init__(self, capital=10000, climate_risk_exposure = 0.5):
        self.initial_capital = capital      # initial capital
        self.capital = capital              # current capital
        
        self.initial_climate_risk_exposure \
            = climate_risk_exposure         # initial climate risk exposure
        self.climate_risk_exposure \
            = climate_risk_exposure         # capital loss ratio when a climate event occurs
        
        self.exposure_decay_rate = 0.1      # decay rate of climate risk exposure
        self.esg_invested = 0               # cumulative amount invested in ESG
        self.margin = 0                     # single period profit margin
        self.capital_gain = 0               # single period capital gain
        self.strategy = None                # 1: "mitigation", 2: "greenwashing", 0: "none"
        self.esg_score = None               # signal to be broadcasted to investors: "esg-friendly":1, "none":0

    def receive_investment(self, amount):
        """Receive investment from investors."""
        self.capital += amount

    def lose_investment(self, amount):
        """Lose investment due to climate event."""
        self.capital -= amount
    
    def make_decision(self, strategy):
        """Make a decision on how to allocate capital."""
        self.strategy = strategy
        if strategy == 1:
            self.invest_in_esg(self.capital*0.2)  # TODO: this is a hardcoded value, should be a parameter
        elif strategy == 2:
            self.invest_in_greenwash(self.capital*0.1)  # TODO: this is a hardcoded value, should be a parameter
        else:
            self.esg_score = 0

    def invest_in_esg(self, amount):
        """Invest a certain amount in ESG."""
        self.esg_invested += amount
        self.capital -= amount
        # climate risk exposure is an exponential decay function of the amount invested in ESG
        self.climate_risk_exposure = self.climate_risk_exposure \
            * jnp.exp(-self.exposure_decay_rate * self.esg_invested)
        self.esg_score = 1

    def invest_in_greenwash(self, amount):
        """Invest a certain amount in greenwashing."""
        self.capital -= amount
        self.esg_score = 1

    def update_capital(self, environment):
        """Update the capital based on market performance and climate event."""
        # add a random disturbance to market performance
        company_performance = environment.market_performance + jax.random.normal(jax.random.PRNGKey(0), shape=(1,))[0]*0.25 #TODO: ranges from 0.5 to 1.5 most of time
        new_capital = self.capital * company_performance
        if environment.climate_event_occurred:
            new_capital *= (1 - self.climate_risk_exposure)
        self.capital_gain = new_capital - self.capital
        self.margin = self.capital_gain/self.capital
        self.capital = new_capital

    def report(self):
        """Report the current status of the company."""
        return {
            "capital": self.capital,
            "climate_risk_exposure": self.climate_risk_exposure,
            "esg_score": self.esg_score,
            "margin": self.margin
        }
    
    def reset(self):
        """Reset the company to the initial state."""
        self.capital = self.initial_capital
        self.climate_risk_exposure = self.initial_climate_risk_exposure
        self.esg_invested = 0
        self.margin = 0
        self.capital_gain = 0
        self.strategy = None
        self.esg_score = None
    
class Investor:
    def __init__(self, esg_preference, capital=10000):
        self.initial_capital = capital      # initial capital
        self.capital = capital              # current capital
        self.investments = {}               # dictionary to track investments in different companies
        self.esg_preference = esg_preference # the weight of ESG in the investor's decision making
        self.utility = 0                     # single-period reward
    
    def initial_investment(self, environment):
        """Invest in all companies at the beginning of the simulation."""
        self.investments = {i: 0 for i in range(environment.num_companies)}
    
    def invest(self, amount, company_idx):
        """Invest a certain amount in a company. 
        At the end of each period, investors collect all returns and then redistribute capital in next round."""
        if self.capital < amount:
            raise ValueError("Investment amount exceeds available capital.")
        else:
            self.capital -= amount
            self.investments[company_idx] += amount

    def divest(self, company_idx, environment):
        """Divest from a company."""
        investment_return = self.investments[company_idx]
        self.capital += investment_return
        environment.companies[company_idx].lose_investment(investment_return)
        self.investments[company_idx] = 0
    
    def calculate_utility(self, environment):
        """Calculate reward based on market performance and ESG preferences."""
        returns = 0
        esg_reward = 0
        for company_idx, investment in self.investments.items():
            company = environment.companies[company_idx]
            returns += investment * company.margin
            esg_reward += company.esg_score
            # update value of investment based on returns
            self.investments[company] = investment * (1 + company.margin)

        overall_return_rate = returns/self.capital
        utility = overall_return_rate + self.esg_preference * esg_reward
        self.capital += returns
        self.utility = utility
    
    def report(self):
        """Report the current status of the investor."""
        return self.investments
    
    def reset(self):
        """Reset the investor to the initial state."""
        self.capital = self.initial_capital
        self.investments = {i: 0 for i in range(self.num_companies)}
        self.utility = 0


class InvestESG(MultiAgentEnv):
    """
    JAX Compatible version of ESG investment environment.
    """

    def __init__(
        self,
        num_companies=10,
        num_investors=10,
        company_starting_capital=10000,
        investor_starting_capital=10000,
        investor_esg_preference=0.5,
        initial_climate_event_probability=0.1,
        max_steps=100,
        egocentric=False, # TODO: what is this? Do we need it?
        cnn=False
    ):
        self.max_steps = max_steps
        self.current_step = 0
        self.num_companies = num_companies
        self.num_investors = num_investors
        self.companies = [Company(capital = company_starting_capital) for _ in range(num_companies)]
        self.investors = [Investor(investor_esg_preference, investor_starting_capital) for _ in range(num_investors)]
        self.market_performance = 1 # initial market performance
        self.initial_climate_event_probability = initial_climate_event_probability # initial probability of climate event
        self.climate_event_probability = initial_climate_event_probability # current probability of climate event
        self.climate_event_occurred = False # whether a climate event has occurred in the current step
        # initialize investors with initial investments dictionary
        for investor in self.investors:
            investor.initial_investment(self)

    def action_space(self):
        # each company has 3 possible actions: "mitigation", "greenwashing", "none"
        # each investor has num_companies possible*2 actions: for each company, invest/not invest
        company_actions = spaces.Discrete(3)
        investor_actions = spaces.MultiDiscrete(self.num_companies*[2]) # 0: not invest, 1: invest
        return {"company": company_actions, "investor": investor_actions}
    
    def observation_space(self):
        # each company has 4 observable features: capital, climate risk exposure, ESG score, margin
        # each investor has num_companies+1 observable features: investment in each company, remaining capital
        company_obs = spaces.Box(low=0, high=1, shape=(4,), dtype=jnp.float32)
        investor_obs = spaces.Box(low=0, high=1, shape=(self.num_companies+1,), dtype=jnp.float32)
        return {"company": company_obs, "investor": investor_obs}

    
    def step(self, actions):
        """Step function for the environment."""
        rng_key = jax.random.PRNGKey(0)
        rng_key, subkey = jax.random.split(rng_key)

        # unpack actions
        companys_actions = actions['company']
        investors_actions = actions['investor']

        # 0. investors divest from all companies and recollect capital
        for investor in self.investors:
            for company in investor.investments:
                investor.divest(company)

        # 1. investors allocate capital to companies (binary decision to invest/not invest)
        for i, investor in enumerate(self.investors):
            investor_action = investors_actions[i]
            # number of companies that the investor invests in
            num_investments = jnp.sum(investor_action)
            if num_investments > 0:
                investment_amount = investor.capital/num_investments
                for j, company in enumerate(self.companies):
                    if investor_action[j]:
                        investor.invest(investment_amount, j)
                        # company receives investment
                        company.receive_investment(investment_amount)
                   
        # 2. companies invest in ESG/greenwashing/none, report margin and esg score
        for i, company in enumerate(self.companies):
            company.make_decision(companys_actions[i])

        # 3. update probabilities of climate event based on cumulative ESG investments across companies
        total_esg_investment = jnp.sum(jnp.array([company.esg_invested for company in self.companies]))
        self.climate_event_probability =  self.initial_climate_event_probability * jnp.exp(-0.1 * total_esg_investment)

        # 4. market performance and climate event evolution
        self.market_performance = jax.random.normal(subkey, shape=(1,))[0]*0.25   # ranges from 0.5 to 1.5 most of time
        # TODO: consider other distributions and time-correlation of market performance
        self.climate_event_occurred = jax.random.bernoulli(subkey, self.climate_event_probability)

        # 5. companies update capital based on market performance and climate event
        for company in self.companies:
            company.update_capital(self)

        # 6. investors calculate returns based on market performance
        for investor in self.investors:
            investor.calculate_utility(self)

        # 7. update observation for each company and investor
        return self._get_observation(), self._get_reward(), None, None

    def _get_observation(self):
        """Get observation for each company and investor. Public information is shared across all agents."""
        company_obs = [company.report() for company in self.companies]
        investor_obs = [investor.report() for investor in self.investors]
        return {"company": company_obs, "investor": investor_obs}

    def _get_reward(self):
        """Get reward for all agents."""
        company_rewards = [company.capital_gain for company in self.companies]
        investor_rewards = [investor.utility for investor in self.investors]
        return {"company": company_rewards, "investor": investor_rewards}

    def reset(self):
        """Reset the environment."""
        for company in self.companies:
            company.reset()
        for investor in self.investors:
            investor.reset()
        self.market_performance = 1
        self.climate_event_probability = self.initial_climate_event_probability
        self.climate_event_occurred = False
        self.current_step = 0
        return self._get_observation()

    @property
    def name(self) -> str:
        """Environment name."""
        return "InvestESG"

    def render(self, mode='human'):
        if not hasattr(self, 'fig') or self.fig is None:
            # Initialize the plot only once
            self.fig, self.ax = plt.subplots(4, 1, figsize=(12, 18))
            plt.subplots_adjust(hspace=0.4)
        
        # Clear previous figures to update with new data
        for axis in self.ax:
            axis.cla()

        # Data to plot
        company_capitals = [company.capital for company in self.companies]
        investor_capitals = [investor.capital for investor in self.investors]
        esg_investments = [company.esg_invested for company in self.companies]
        company_decisions = [company.strategy for company in self.companies]
        
        # Subplot 1: Company Capitals
        self.ax[0].bar(range(len(company_capitals)), company_capitals, color='blue')
        self.ax[0].set_title('Company Capitals')
        self.ax[0].set_ylabel('Capital')
        self.ax[0].set_xlabel('Company ID')
        
        # Subplot 2: Investor Capitals
        self.ax[1].bar(range(len(investor_capitals)), investor_capitals, color='green')
        self.ax[1].set_title('Investor Capitals')
        self.ax[1].set_ylabel('Capital')
        self.ax[1].set_xlabel('Investor ID')

        # Subplot 3: ESG Investments
        self.ax[2].bar(range(len(esg_investments)), esg_investments, color='orange')
        self.ax[2].set_title('ESG Investments by Companies')
        self.ax[2].set_ylabel('Total ESG Investment')
        self.ax[2].set_xlabel('Company ID')

        # Subplot 4: Company Decisions
        decision_labels = ['None', 'Mitigation', 'Greenwashing']  # Assuming 0: None, 1: Mitigation, 2: Greenwashing
        decision_colors = ['grey', 'green', 'red']  # Color coding for decisions
        decision_colors_mapped = [decision_colors[d] if d is not None else 'grey' for d in company_decisions]
        self.ax[3].bar(range(len(company_decisions)), [1]*len(company_decisions), color=decision_colors_mapped)
        self.ax[3].set_title('Company Decisions')
        self.ax[3].set_ylabel('Decision Type')
        self.ax[3].set_xlabel('Company ID')
        self.ax[3].set_yticks([])  # Hide y-axis as it's not meaningful here
        self.ax[3].set_xticks(range(len(decision_labels)))
        self.ax[3].set_xticklabels(decision_labels)


        # Update the plots
        plt.pause(0.001)  # Pause briefly to update plots

        if mode == 'rgb_array':
            # Return RGB array of the plot for video etc.
            self.fig.canvas.draw()
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return img

        elif mode == 'human':
            plt.show()

    def is_terminal(self) -> bool:
        """Check if the environment has reached a terminal state."""
        if self.current_step >= self.max_steps:
            return True
        
        


if __name__ == "__main__":
    env = InvestESG()
    print(env.action_space())
    print(env.observation_space())

