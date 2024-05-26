# !pip install instructor groq openai

import instructor
import openai
import groq
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import json

inference_provider = "openai"   # "openai" or "groq"
client = instructor.from_openai(openai.OpenAI()) if inference_provider == "openai" else instructor.from_groq(groq.Groq())

class Score(BaseModel):
    score: float = Field(description=f"""
        **Score**
        
        Score for the strategy evaluation.
    """)
    explanation: str = Field(description=f"""
        **Explanation**
        
        Explanation / justification for the score.
    """)

class Objective(BaseModel):
    name: str = Field(description=f"""
        **Objective Name**
        
        Name of the strategic objective derived from the strategy document.
    """)
    description: str = Field(description=f"""
        **Objective Description**
        
        Description of the strategic objective derived from the strategy document.
    """)
    relevance_to_objective_score: Score = Field(description=f"""
        **Relevance to Objective Score (1-5 integer)**
        
        Rate the relevance of this objective on a 5-point scale, from 1 (least relevant) to 5 (most relevant).

        **Justification for Relevance to Objective Score**
        
        Provide a detailed explanation justifying the assigned relevance score, including its alignment with the organization's mission and goals.
        
        - **Impact on Members**: How does this objective serve the members?
        - **Growth of the Game**: In what way does it contribute to the growth of golf?
        - **Alignment with North Stars**: How does it align with the organization's North Star objectives?
    """)

class MarketAnalysis(BaseModel):
    market_need_score: Score = Field(description=f"""
        **Market Need Score (1-5 integer)**
        
        Rate the market need for the business idea on a 5-point scale, from 1 (low need) to 5 (high need).

        **Justification for Market Need Score**
        
        Provide a detailed explanation justifying the market need score, including the pain points and needs addressed by the business idea.
        
        - **Assessment of Demand**: What is the evidence of demand for this idea?
        - **Member Feedback**: Have members expressed a need for this?
    """)
    market_size_score: Score = Field(description=f"""
        **Market Size Score (1-5 integer)**
        
        Rate the market size on a 5-point scale, from 1 (small market) to 5 (large market).

        **Justification for Market Size Score**
        
        Provide a detailed explanation justifying the market size score, including market size estimates and growth potential.
        
        - **Market Research**: What does market research indicate about the size and potential growth?
        - **Competitive Landscape**: How crowded is the market?
    """)
    competitive_landscape_score: Score = Field(description=f"""
        **Competitive Landscape Score (1-5 integer)**
        
        Rate the competitive landscape on a 5-point scale, from 1 (high competition) to 5 (low competition).

        **Justification for Competitive Landscape Score**
        
        Provide a detailed explanation justifying the competitive landscape score, including key competitors and market gaps.
        
        - **Competitive Dynamics**: Who are the key players and how competitive is the sector?
        - **Differentiation**: How does this business idea stand out from competitors?
    """)

class FinancialAnalysis(BaseModel):
    revenue_projection_score: Score = Field(description=f"""
        **Revenue Projection Score (1-5 integer)**
        
        Rate the revenue projection on a 5-point scale, from 1 (low projection) to 5 (high projection).

        **Justification for Revenue Projection Score**
        
        Provide a detailed explanation justifying the revenue projection score, including potential revenue streams and growth expectations.
        
        - **Revenue Channels**: What are the main sources of revenue from this idea?
        - **Growth Trajectory**: What are the short-term and long-term revenue projections?
    """)
    cost_analysis_score: Score = Field(description=f"""
        **Cost Analysis Score (1-5 integer)**
        
        Rate the cost analysis on a 5-point scale, from 1 (high costs) to 5 (low costs).

        **Justification for Cost Analysis Score**
        
        Provide a detailed explanation justifying the cost analysis score, including initial investments, operational costs, and ongoing expenses.
        
        - **Start-up Costs**: What are the initial investments required?
        - **Operational Efficiency**: How efficient is the cost structure?
    """)
    profitability_score: Score = Field(description=f"""
        **Profitability Score (1-5 integer)**
        
        Rate the profitability on a 5-point scale, from 1 (low profitability) to 5 (high profitability).

        **Justification for Profitability Score**
        
        Provide a detailed explanation justifying the profitability score, including potential margins and return on investment.
        
        - **Margin Analysis**: What are the expected profit margins?
        - **Return on Investment**: How quickly can we expect to break even and see returns?
    """)
    funding_requirements_score: Score = Field(description=f"""
        **Funding Requirements Score (1-5 integer)**
        
        Rate the funding requirements on a 5-point scale, from 1 (high funding needs) to 5 (low funding needs).

        **Justification for Funding Requirements Score**
        
        Provide a detailed explanation justifying the funding requirements score, including potential funding sources and financial sustainability.
        
        - **Capital Needs**: How much capital is required and for what?
        - **Funding Sources**: What are potential sources of funding, both internal and external?
    """)

class OperationalFeasibility(BaseModel):
    resource_availability_score: Score = Field(description=f"""
        **Resource Availability Score (1-5 integer)**
        
        Rate the availability of necessary resources on a 5-point scale, from 1 (low availability) to 5 (high availability).

        **Justification for Resource Availability Score**
        
        Provide a detailed explanation justifying the resource availability score, including human, technological, and financial resources.
        
        - **Human Resources**: Are the required skills and personnel available?
        - **Technological Resources**: Are the necessary technologies in place and accessible?
    """)
    operational_requirements_score: Score = Field(description=f"""
        **Operational Requirements Score (1-5 integer)**
        
        Rate the operational requirements on a 5-point scale, from 1 (high complexity) to 5 (low complexity).

        **Justification for Operational Requirements Score**
        
        Provide a detailed explanation justifying the operational requirements score, including the complexity and feasibility of operational processes.
        
        - **Operational Complexity**: What operational processes are required and how complex are they?
        - **Feasibility**: How feasible is it to implement these processes given current capabilities?
    """)
    scalability_score: Score = Field(description=f"""
        **Scalability Score (1-5 integer)**
        
        Rate the scalability on a 5-point scale, from 1 (low scalability) to 5 (high scalability).

        **Justification for Scalability Score**
        
        Provide a detailed explanation justifying the scalability score, including the potential to scale the business and handle growth.
        
        - **Scalability Potential**: How scalable is the business model?
        - **Growth Capacity**: What infrastructure is needed to support growth?
    """)

class RiskAnalysis(BaseModel):
    market_risk_score: Score = Field(description=f"""
        **Market Risk Score (1-5 integer)**
        
        Rate the market risk on a 5-point scale, from 1 (high risk) to 5 (low risk).

        **Justification for Market Risk Score**
        
        Provide a detailed explanation justifying the market risk score, including external risks such as market fluctuations and competition.
        
        - **Market Volatility**: What is the level of market fluctuation risk?
        - **Competitive Risk**: How does competition pose a risk?
    """)
    operational_risk_score: Score = Field(description=f"""
        **Operational Risk Score (1-5 integer)**
        
        Rate the operational risk on a 5-point scale, from 1 (high risk) to 5 (low risk).

        **Justification for Operational Risk Score**
        
        Provide a detailed explanation justifying the operational risk score, including internal risks such as resource constraints and execution challenges.
        
        - **Execution Risk**: What are the risks associated with executing the operational plans?
        - **Resource Risk**: How likely are resource shortages or constraints?
    """)
    financial_risk_score: Score = Field(description=f"""
        **Financial Risk Score (1-5 integer)**
        
        Rate the financial risk on a 5-point scale, from 1 (high risk) to 5 (low risk).

        **Justification for Financial Risk Score**
        
        Provide a detailed explanation justifying the financial risk score, including financial risks such as funding and revenue variability.
        
        - **Funding Risk**: What are the risks associated with securing necessary funding?
        - **Revenue Risk**: How stable are the anticipated revenue streams?
    """)
    legal_regulatory_risk_score: Score = Field(description=f"""
        **Legal and Regulatory Risk Score (1-5 integer)**
        
        Rate the legal and regulatory risk on a 5-point scale, from 1 (high risk) to 5 (low risk).

        **Justification for Legal and Regulatory Risk Score** 
        
        Provide a detailed explanation justifying the legal and regulatory risk score, including compliance with relevant laws and regulations.
        
        - **Compliance**: What are the legal compliance requirements and risks?
        - **Regulatory Changes**: How susceptible is the business to changes in regulations?
    """)

class ImpactAnalysis(BaseModel):
    member_impact_score: Score = Field(description=f"""
        **Member Impact Score (1-5 integer)**
        
        Rate the impact on members on a 5-point scale, from 1 (low impact) to 5 (high impact).

        **Justification for Member Impact Score**
        
        Provide a detailed explanation justifying the member impact score, including how the business idea benefits PGA members.
        
        - **Member Services**: How does this idea improve services for members?
        - **Member Satisfaction**: What is the potential impact on member satisfaction and engagement?
    """)
    game_growth_score: Score = Field(description=f"""
        **Game Growth Impact Score (1-5 integer)**
        
        Rate the impact on the growth of the game on a 5-point scale, from 1 (low impact) to 5 (high impact).

        **Justification for Game Growth Impact Score**
        
        Provide a detailed explanation justifying the game growth impact score, including how the business idea contributes to the growth of golf.
        
        - **Participation Increase**: How will this idea drive participation in golf?
        - **Outreach and Engagement**: How does it help in reaching new demographics or expanding current ones?
    """)
    brand_impact_score: Score = Field(description=f"""
        **Brand Impact Score (1-5 integer)**

        Rate the impact on the PGA brand on a 5-point scale, from 1 (low impact) to 5 (high impact).

        **Justification for Brand Impact Score**
        
        Provide a detailed explanation justifying the brand impact score, including potential impacts on the PGA brand and reputation.
        
        - **Brand Equity**: How does this idea enhance or maintain brand equity?
        - **Public Perception**: What are the potential effects on public perception of the PGA?
    """)
    sustainability_score: Score = Field(description=f"""
        **Sustainability Impact Score (1-5 integer)**
        
        Rate the sustainability impact on a 5-point scale, from 1 (low impact) to 5 (high impact).

        **Justification for Sustainability Impact Score**
        
        Provide a detailed explanation justifying the sustainability impact score, including contributions to environmental and social sustainability goals.
        
        - **Environmental Impact**: Does this idea support environmental sustainability initiatives?
        - **Social Responsibility**: What are the social implications and benefits?
    """)
    sustainability_justification: str = Field(description=f"""
    """)

class StrategyEvaluation(BaseModel):
    title: str = Field(description=f"""
        **Strategy Document Title** 
        
        Title of the strategy document being evaluated.
    """)
    objectives: List[Objective] = Field(description=f"""
        **Objectives and Relevance Scores** 
        
        List of strategic objectives and their respective relevance scores and justifications.
    """)
    market_analysis: MarketAnalysis = Field(description=f"""
        **Market Analysis Scores and Justifications** 
        
        Market analysis scores and detailed justifications, including market need, market size, and competitive landscape.
    """)
    financial_analysis: FinancialAnalysis = Field(description=f"""
        **Financial Analysis Scores and Justifications** 
        
        Financial analysis scores and detailed justifications, including revenue projections, cost analysis, profitability, and funding requirements.
    """)
    operational_feasibility: OperationalFeasibility = Field(description=f"""
        **Operational Feasibility Scores and Justifications** 
        
        Operational feasibility scores and detailed justifications, including resource availability, operational requirements, and scalability.
    """)
    risk_analysis: RiskAnalysis = Field(description=f"""
        **Risk Analysis Scores and Justifications** 
        
        Risk analysis scores and detailed justifications, including market risk, operational risk, financial risk, and legal/regulatory risk.
    """)
    impact_analysis: ImpactAnalysis = Field(description=f"""
        **Impact Analysis Scores and Justifications** 
        
        Impact analysis scores and detailed justifications, including member impact, game growth impact, brand impact, and sustainability impact.
    """)
    all_scores: List[Score] = Field(description=f"""
        **All Scores**
        
        List of all scores, including strategic objectives, market analysis, financial analysis, operational feasibility, risk analysis, and impact analysis.
    """)
    overall_score: float = Field(description=f"""
        **Overall Score**
        
        Overall score (an average of all scores, rounded to one decimal place), including strategic objectives, market analysis, financial analysis, operational feasibility, risk analysis, and impact analysis.
    """)
    overall_scores_narrative: str = Field(description=f"""
        **Overall Scores Narrative**
        
        Narrative summary of most significant drivers / factors for the overall score.
    """)



text = f"""
Strategic Partnership Proposal: PGA and Casino Company for the Senior PGA Championship

Objectives:
1. Enhance Fan Engagement and Experience:
   - Create immersive and interactive experiences to captivate attendees.
   - Offer exclusive VIP packages combining golf and casino entertainment.
   - Introduce engaging activities to entertain fans throughout the event.

2. Increase Revenue Streams:
   - Generate additional revenue through co-branded sponsorship deals.
   - Enhance ticket sales with premium and VIP packages.
   - Drive in-event spending through exclusive casino zones and merchandise.

3. Expand Audience Reach:
   - Attract a new demographic of casino-goers by increasing visibility and appeal.
   - Utilize integrated marketing campaigns to reach a broader network of potential attendees.
   - Address challenges in attracting non-traditional golf fans.

4. Boost Brand Visibility:
   - Elevate the PGA brand through high-profile casino associations.
   - Increase media coverage and promotional opportunities.
   - Offer elite and diverse entertainment options to enhance the prestige of the Senior PGA Championship.

Market Analysis:
- Market Need Score: Highlight the significant need for innovative fan experiences.
- Market Size Score: Emphasize the substantial market size and niche nature of senior golf.
- Competitive Landscape Score: Address the moderately challenging competitive landscape and the need for differentiation.

Overall Strategy:
- Ensure a well-rounded and promising strategy with significant potential benefits.
- Address challenges such as competitive landscape, cost management, and operational complexity.
- Focus on enhancing fan engagement, increasing revenue streams, and positive brand impact.

Improve the proposal text to achieve a higher score next time by incorporating these enhancements.
"""

result = client.chat.completions.create(
    model="llama3-70b-8192" if inference_provider == "groq" else "gpt-4o",
    response_model=StrategyEvaluation, # this is Instructor at work!
    temperature=0.0,
    messages=[{"role": "user", "content": text}]
    )

print(result.model_dump_json(indent=4))




