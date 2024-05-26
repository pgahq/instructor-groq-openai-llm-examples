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
Executive Summary
As the landscape of sports entertainment evolves, so too must the experiences we provide our fans. The PGA of America is poised to revolutionize the Senior PGA Championship through a strategic partnership with a premier casino company. This proposal outlines a comprehensive strategy to combine the sophistication of golf with the excitement of casino entertainment, creating an unparalleled experience for our audience and opening new revenue streams.

1. Introduction
1.1. Background
The Senior PGA Championship is one of the most prestigious events in senior golf, attracting a global audience of golf enthusiasts and casual fans alike. The integration of casino entertainment into this revered tournament offers a unique opportunity to enhance fan engagement, attract a diverse demographic, and generate substantial revenue.

1.2. Rationale for Partnership
The casino industry embodies excitement, entertainment, and luxury—qualities that align perfectly with the aspirational and elite image of professional golf. By uniting these two industries, we can create a multifaceted experience that appeals to a broader audience, enhances the event's prestige, and drives financial growth for both partners.

2. Objectives
2.1 Enhance Fan Engagement and Experience
Create immersive and interactive experiences that captivate attendees.
Offer exclusive VIP packages that combine golf and casino entertainment.
Introduce new and engaging activities to keep fans entertained throughout the event.
2.2 Increase Revenue Streams
Generate additional revenue through co-branded sponsorship deals.
Enhance ticket sales with premium and VIP packages.
Drive in-event spending through exclusive casino zones and merchandise.
2.3 Expand Audience Reach
Attract a new demographic of casino-goers who may not typically engage with golf.
Increase visibility and appeal through integrated marketing campaigns.
Utilize casino partnerships to gain access to a broader network of potential attendees.
2.4 Boost Brand Visibility
Elevate the PGA brand through high-profile casino associations.
Increase media coverage and promotional opportunities.
Enhance the prestige of the Senior PGA Championship by offering elite and diverse entertainment options.
3. Partnership Details
3.1 Partner Selection Criteria
Reputation: Partner with a casino company known for its luxury, quality, and corporate responsibility.
Infrastructure: Ensure the casino has the capacity to host high-profile events and manage large crowds.
Alignment: Select a partner whose brand values and target audience align with those of the PGA.
3.2 Negotiation Points
Revenue Sharing: Define clear terms for revenue sharing from ticket sales, sponsorships, and in-event spending.
Branding Rights: Establish guidelines for co-branding, including logo placements, joint marketing, and promotional activities.
Sponsorship & Activation: Agree on sponsorship tiers, activation rights, and access to PGA players for promotional events.
4. Fan Engagement Initiatives
4.1 On-Site Casino Zones
Gaming Simulators: Install state-of-the-art gaming simulators where fans can experience virtual casino games.
Lounge Areas: Create luxurious lounge areas for fans to relax, socialize, and enjoy top-tier hospitality.
Betting Kiosks: Integrate discreet and responsible betting kiosks where fans can place wagers on tournament outcomes.
4.2 VIP Packages
Exclusive Access: Offer VIP passes that include backstage access, meet-and-greet opportunities with players, and access to exclusive hospitality suites.
Golf & Casino Experiences: Develop packages that bundle premium seating at the tournament with casino credits, exclusive shows, and more.
Luxury Accommodations: Partner with the casino to offer top-tier accommodation packages that blend golf and luxury entertainment experiences.
4.3 Special Promotions
Pre-Tournament Events: Host exclusive pre-tournament events at the casino to build excitement and offer fans a sneak peek.
Charity Golf Games: Organize charity golf games featuring PGA pros, with proceeds benefiting local non-profits.
Post-Tournament Celebrations: Plan grand post-tournament celebrations, including award ceremonies and gala dinners at the casino.
5. Marketing Strategy
5.1 Integrated Marketing Campaign
Cross-Promotion: Leverage PGA and casino marketing channels to amplify reach and engagement.
Digital Marketing: Utilize social media, email marketing, and digital ads to promote the event and VIP packages.
Content Creation: Develop engaging co-branded content, including videos, behind-the-scenes footage, and player interviews.
5.2 Media Relations
Press Releases: Issue joint press releases to announce the partnership and highlight key events.
Media Events: Host media events to provide exclusive previews and interviews with PGA players and casino representatives.
Broadcast Integration: Integrate casino branding and features into broadcast coverage of the tournament.
6. Operational Plan
6.1 Venue Setup
Logistics: Coordinate with the tournament venue to ensure seamless integration of casino zones and fan engagement areas.
Security: Implement robust security measures to ensure the safety and enjoyment of all attendees.
Staffing: Train and deploy staff from both the PGA and the casino to manage various event components.
6.2 Timeline
Phase 1: Planning (Months 1-3): Finalize partnership agreements, develop event plans, and draft marketing strategies.
Phase 2: Marketing (Months 3-6): Launch the integrated marketing campaign, sell early bird packages, and engage media.
Phase 3: Pre-Event Preparations (Months 6-9): Set up venue, train staff, and finalize operational details.
Phase 4: Event Week (Month 10): Execute the Senior PGA Championship with all integrated elements.
Phase 5: Post-Event (Month 11): Conduct a comprehensive review, gather feedback, and prepare reports.
7. Financial Projections
7.1 Revenue Streams
Ticket Sales: Increase ticket sales through VIP and premium packages.
Sponsorships: Secure co-branded sponsorship deals to drive revenue.
In-Event Spending: Generate additional income from casino zones, merchandise, and food and beverage sales.
7.2 Cost Considerations
Venue Setup: Allocate budget for the setup of casino zones and fan engagement areas.
Marketing: Invest in integrated marketing campaigns and promotional activities.
Staffing & Logistics: Cover costs related to staffing, security, and operational logistics.
7.3 ROI
Projections: Estimate a 20% increase in total revenue compared to previous years.
Break-Even Analysis: Conduct a break-even analysis to ensure financial viability and sustainability.
8. Risk Management
8.1 Potential Risks
Regulatory Compliance: Ensure compliance with all legal and regulatory requirements related to casino activities.
Brand Alignment: Maintain the PGA’s prestigious image by carefully selecting casino partners and activities.
Fan Concerns: Address potential concerns from traditional golf fans regarding the integration of casino entertainment.
8.2 Mitigation Strategies
Compliance Team: Establish a dedicated team to oversee regulatory adherence and mitigate legal risks.
Selective Partnerships: Partner only with casinos that align with PGA values and quality standards.
Transparent Communication: Clearly communicate the benefits of the partnership to stakeholders and fans to alleviate concerns.
9. Conclusion
The proposed partnership between the PGA and a premier casino company for the Senior PGA Championship represents an innovative approach to elevate fan engagement, diversify revenue streams, and enhance the brand prestige of both partners. By merging the sophistication of golf with the excitement of casino entertainment, we can create a uniquely compelling experience that appeals to a wide range of audiences. This strategic alliance promises significant benefits, driving growth and setting new standards for sports entertainment.
"""

result = client.chat.completions.create(
    model="llama3-70b-8192" if inference_provider == "groq" else "gpt-4o",
    response_model=StrategyEvaluation, # this is Instructor at work!
    temperature=0.0,
    messages=[{"role": "user", "content": text}]
    )

print(result.model_dump_json(indent=4))



