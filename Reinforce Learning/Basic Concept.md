# Basic Concept

## 1. State
A representation of the environment at a particular point in time. It contains all the information needed by the agent to decide on an action.

## 2. Action
A choice made by the agent, which potentially changes the state of the environment.

## 3. State Transition
The process by which the environment moves from one state to another state in response to an action.

## 4. Reward
A numerical signal given to the agent that indicates the desirability of a particular state-action transition.

## 5. Reward Probability
A probabilistic model that specifies the likelihood of certain rewards being received for a given state-action pair.

## 6. Trajectory
A sequence of states, actions, and rewards that the agent experiences over time.

## 7. Episode
A complete sequence of states, actions, and rewards starting from an initial state and ending in a terminal state (e.g., when the task completes).

## 8. Return
The sum of rewards obtained from a specific time step until the end of the episode.

## 9. Discounted Return
A weighted sum of future rewards, where each reward is multiplied by a discount factor (γ, 0 ≤ γ ≤ 1) that reduces the importance of rewards further in the future.

## 10. Markov Decision Process (MDP)
A **memoryless** formal framework for RL problems, defined by:
- A set of states
- A set of actions
- A transition probability distribution
- A reward function
- A discount factor (optional in some definitions)