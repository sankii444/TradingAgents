from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    opponent_argument_or_opening,
    report_or_absent,
)


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = opponent_argument_or_opening(
            investment_debate_state.get("current_response", ""), "bear analyst"
        )
        market_research_report = report_or_absent(state["market_report"], "market")
        sentiment_report = report_or_absent(state["sentiment_report"], "sentiment")
        news_report = report_or_absent(state["news_report"], "news")
        fundamentals_report = report_or_absent(state["fundamentals_report"], "fundamentals")
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        if asset_type == "forex":
            target_label = "currency pair"
            role_intro = (
                "You are a Bull Analyst making the case for the currency pair. "
                "Build a strong, evidence-based case from the supplied research."
            )
            fundamentals_label = "Corporate fundamentals (not applicable to spot FX)"
            focus_guidance = (
                "- Relative interest rates, central-bank policy, inflation, and economic releases\n"
                "- How those factors support the base currency relative to the quote currency\n"
                "- Market trends and positive evidence in the supplied reports\n"
            )
        elif asset_type == "commodity":
            target_label = "commodity"
            role_intro = (
                "You are a Bull Analyst making the case for the commodity. "
                "Build a strong, evidence-based case from the supplied research."
            )
            fundamentals_label = "Corporate fundamentals (not applicable to commodities)"
            focus_guidance = (
                "- Supply constraints, falling inventories, or rising end-user demand\n"
                "- Production, seasonality, and relevant futures-market conditions\n"
                "- Macroeconomic drivers and positive evidence in the supplied reports\n"
            )
        else:
            target_label = "stock" if asset_type == "stock" else "asset"
            role_intro = (
                f"You are a Bull Analyst advocating for investing in the {target_label}. "
                "Your task is to build a strong, evidence-based case emphasizing growth potential, "
                "competitive advantages, and positive market indicators. Leverage the provided research "
                "and data to address concerns and counter bearish arguments effectively."
            )
            fundamentals_label = (
                "Company fundamentals report"
                if asset_type == "stock"
                else "Asset fundamentals report (may be unavailable for crypto)"
            )
            focus_guidance = (
                "- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.\n"
                "- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.\n"
                "- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.\n"
            )

        prompt = f"""{role_intro}

Key points to focus on:
{focus_guidance}
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Resources available:
{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position.
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
