# Financial Agent Workshop Exercises

These exercises build on concepts from Days 1-4 and are designed to be completed during the 2-hour workshop. Choose the exercises that align with your interests and skill level.

## Core Exercises (Choose at least 2)

### 1. Implement Reflection Mechanism
Complete the `reflect_on_response` method in the `FinancialAdvisorAgent` class. This method should:
- Analyze whether the advice is evidence-based and well-reasoned
- Identify any missing important considerations
- Assess the confidence level in the recommendation
- Propose improvements to the response

**Building on:** Day 4's reasoning frameworks (self-correction and reflection)

### 2. Add Portfolio Diversification Tool
Create a new tool that provides portfolio diversification recommendations based on:
- Risk tolerance (conservative, moderate, aggressive)
- Investment horizon (short-term, medium-term, long-term)
- Asset class allocation (stocks, bonds, cash, alternatives)

**Building on:** Day 2's financial queries and Day 4's tool definition patterns

### 3. Handle Multi-Part Financial Queries
Enhance the agent to handle complex queries that require multiple pieces of information, such as:
- "Compare AAPL and MSFT stocks and recommend which is better for a 5-year investment"
- "What is the current inflation rate and how should I adjust my investment strategy?"

**Building on:** Day 4's ReAct pattern and sequential reasoning

### 4. Error Handling and Uncertainty
Improve the agent to gracefully handle situations where:
- Financial data is unavailable or outdated
- Queries require speculation about future market conditions
- Information is insufficient to provide a confident recommendation

**Building on:** Day 2's error handling and Day 4's reasoning frameworks

## Advanced Exercises (Optional)

### 5. Financial Planning Chain-of-Thought
Implement a specialized chain-of-thought reasoning pattern for retirement planning that considers:
- Current age and retirement age
- Current savings and monthly contribution capacity
- Expected inflation and investment returns
- Desired retirement income

**Building on:** Day 4's chain-of-thought reasoning pattern

### 6. Personalized Financial Advice
Add a mechanism to maintain a user profile and customize advice based on:
- Risk tolerance and investment goals
- Existing portfolio and financial situation
- Tax considerations and jurisdiction
- Ethical preferences (ESG investing)

**Building on:** Day 1's agent personalization concepts

### 7. Financial Claim Attribution
Enhance the agent to cite sources for all factual claims and explain the confidence level for each recommendation, including:
- Historical data citations
- Disclosure of assumptions
- Confidence scoring for predictions
- Explicit uncertainty acknowledgment

**Building on:** Day 4's reasoning frameworks and Day 2's response parsers

## Group Challenge (Time Permitting)

### Multi-Agent Financial Advisory System
Design a multi-agent system where specialist agents collaborate:
- Market Analyst: Focuses on market data and trends
- Tax Specialist: Handles tax implications of investments
- Risk Manager: Assesses and explains investment risks
- Portfolio Manager: Optimizes asset allocation
- Coordinator: Orchestrates the specialists and synthesizes advice

**Building on:** Day 1's multi-agent concepts and Day 4's agent architectures

## Submission and Evaluation

After completing your chosen exercises:
1. Run your enhanced agent against the test queries in the evaluation framework
2. Score your agent's performance using the provided metrics
3. Prepare a brief (2-minute) demonstration of your agent's capabilities
4. Document the improvements you made and their impact on performance