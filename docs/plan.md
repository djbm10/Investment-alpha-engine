**INSTITUTIONAL-GRADE**

**Adaptive Algorithmic**

**Trading System**

*Graph Signal Processing × Topological Data Analysis × Deep Learning*

*with Self-Correcting Feedback Loops*

Comprehensive Build Plan & Execution Roadmap

March 2026

**CONFIDENTIAL**

**TABLE OF CONTENTS**

Executive Summary

This document is a complete, phase-by-phase engineering plan to build a
self-correcting algorithmic trading system that manages real capital.
The system combines three complementary analytical layers: graph signal
processing to identify mispriced assets relative to their peers,
topological data analysis to detect regime changes in market structure,
and temporal convolutional networks to predict mean-reversion timing. A
fourth meta-layer dynamically allocates capital across sub-strategies
based on real-time performance, and a fifth layer implements continuous
learning from the system's own mistakes.

The plan is organized into eight phases spanning approximately 6--9
months of part-time development. Each phase has explicit entry criteria,
deliverables, validation gates, and failure conditions. No phase may
begin until the prior phase's gate is passed. This sequencing exists
specifically to prevent the most common failure mode in quantitative
trading: building complex systems on untested foundations.

+-----------------------------------------------------------------------+
| **CRITICAL RISK ACKNOWLEDGMENT**                                      |
|                                                                       |
| Algorithmic trading systems can and do lose money. Backtested results |
| are not indicative of future performance. This plan includes          |
| extensive risk controls, but no system eliminates the possibility of  |
| significant capital loss. Never deploy capital you cannot afford to   |
| lose. The self-correcting mechanisms described in this document       |
| reduce but do not eliminate the risk of drawdowns.                    |
+-----------------------------------------------------------------------+

System Architecture Overview

The system operates as five stacked analytical layers, each feeding into
the next. Understanding the full architecture before building any
component prevents the common mistake of optimizing individual pieces
that don't integrate well.

The Five Layers

  ------------ ------------------ ------------------ -----------------------
  **Layer**    **Function**       **Input**          **Output**

  1\. Graph    Models asset       Daily price        Residual signal vector
  Engine       relationships as a returns for the    (e) showing which
               network and        asset universe     assets are out of line
               identifies                            
               deviations from                       
               peer behavior                         

  2\. Regime   Monitors the shape Rolling            Regime state label
  Detector     of market          correlation        (stable, transitioning,
               correlations over  matrices           new regime) and
               time and flags                        confidence score
               structural shifts                     

  3\.          Forecasts whether  Historical         Predicted next-period
  Prediction   deviations will    residuals,         residual and confidence
  Engine       revert or persist  features, regime   interval
                                  state              

  4\.          Scores             Strategy           Position sizes per
  Portfolio    sub-strategies and predictions,       asset per strategy
  Allocator    distributes        uncertainty, costs 
               capital                               
               dynamically                           

  5\. Learning Analyzes mistakes, Realized vs.       Updated parameters,
  Engine       recalibrates       predicted          retrained models,
               parameters, and    outcomes, trade    performance reports
               retrains models    logs               
  ------------ ------------------ ------------------ -----------------------

Data Flow Architecture

Every trading day, the system executes the following pipeline in
sequence. Each step depends on the previous step's output, and the
entire pipeline must complete before any trades are placed. This is
non-negotiable: partial execution leads to incoherent positions.

1.  Market close data is ingested and validated. Missing data, stock
    splits, dividends, and delistings are handled automatically. This
    happens at 4:15 PM ET, after the close.

2.  The Graph Engine rebuilds the correlation network using the most
    recent rolling window of returns. The graph Laplacian is computed
    and the residual signal e is generated for every asset in the
    universe.

3.  The Regime Detector computes today's persistence diagram and
    compares it to the recent history of diagrams. It outputs a regime
    label and transition probability.

4.  The Prediction Engine takes the residual vector, features, and
    regime state as input and outputs predicted residuals and confidence
    intervals for the next period.

5.  The Portfolio Allocator scores each sub-strategy's opportunities
    using the utility function, applies risk constraints, and generates
    target position sizes.

6.  Orders are generated by comparing target positions to current
    positions. Orders are staged, sanity-checked against risk limits,
    and then submitted to the broker API.

7.  Post-trade, the Learning Engine logs all predictions, decisions, and
    outcomes. Weekly and monthly, it triggers recalibration and
    retraining cycles.

+-----------------------------------------------------------------------+
| **DESIGN PRINCIPLE: DEFENSE IN DEPTH**                                |
|                                                                       |
| Every component that can fail has an independent fallback. If the     |
| Regime Detector fails, the system defaults to conservative            |
| positioning. If the Prediction Engine fails, the system falls back to |
| the raw Graph Engine signals with wider thresholds. If the Portfolio  |
| Allocator fails, positions are frozen. No single component failure    |
| can cause uncontrolled trading.                                       |
+-----------------------------------------------------------------------+

Phase 1: Infrastructure & Data Pipeline

+-----------------------------------------------------------------------+
| **PHASE 1** \| Weeks 1--3                                             |
|                                                                       |
| **Infrastructure & Data Pipeline**                                    |
|                                                                       |
| *Build the data ingestion, storage, and validation layer that         |
| everything else depends on. This is unglamorous but the most          |
| important phase. Bad data kills strategies faster than bad math.*     |
+-----------------------------------------------------------------------+

1.1 Development Environment

Set up a reproducible, version-controlled development environment. Every
experiment, every parameter change, every model version must be
trackable. Without this, you will lose track of what works within weeks.

Required Stack

**Language:** Python 3.11+. All components will be in Python. This is
non-negotiable for the ecosystem of quant libraries available.

**Version Control:** Git with a private GitHub/GitLab repo. Every
experiment gets a branch. The main branch always has a working, tested
system.

**Environment Management:** Poetry or Conda for dependency management.
Pin every dependency version. A broken numpy upgrade at the wrong time
can corrupt your entire backtest history.

**Database:** PostgreSQL for structured data (prices, trades, logs).
TimescaleDB extension for time-series optimization. SQLite is fine for
development but will not scale.

**Configuration:** YAML files for all parameters. Never hardcode a
parameter. Every number that can change (lookback windows, thresholds,
alpha, etc.) lives in config. Use Hydra or OmegaConf for config
management.

**Logging:** Python's logging module with structured JSON output. Every
decision the system makes gets logged with a timestamp, the inputs that
produced it, and the output. This log is what the Learning Engine
analyzes.

**Scheduling:** APScheduler or cron for daily pipeline execution. The
system must run without human intervention.

1.2 Data Ingestion

You need two categories of data: price data for the asset universe, and
reference data for corporate actions (splits, dividends, delistings).
Bad data handling is the #1 source of backtest-to-live performance
divergence.

Price Data Sources

  --------------- ------------------- --------------------- --------------------
  **Source**      **Cost**            **Quality**           **Use Case**

  Yahoo Finance   Free                Adequate for          Development and
  (yfinance)                          development;          prototyping only
                                      occasional errors in  
                                      adjusted close        

  Alpaca Market   Free with account   Good; split-adjusted; Paper trading and
  Data                                real-time and         initial live trading
                                      historical            

  Polygon.io      \$29--\$199/month   Institutional         Production system
                                      quality; tick-level   once profitable
                                      data available        

  Tiingo          \$10--30/month      Good quality;         Affordable
                                      reliable historical   production
                                      data                  alternative
  --------------- ------------------- --------------------- --------------------

Data Validation Rules

Every data point must pass these checks before entering the system.
Failing any check triggers an alert and the affected asset is excluded
from that day's trading signals.

1.  Return magnitude check: Absolute daily return exceeding 50% is
    flagged as a likely data error. True moves of this magnitude exist
    but are rare enough to warrant manual review.

2.  Volume check: Zero volume days indicate a data gap or halted stock.
    Exclude from signal generation.

3.  Continuity check: Gaps of more than 3 consecutive trading days for
    any asset trigger an investigation. The asset may have been
    delisted.

4.  Split detection: Compare adjusted and unadjusted close.
    Discrepancies indicate a corporate action that needs to be applied
    to historical data.

5.  Cross-source validation: For production, compare price data from two
    independent sources. Disagreements exceeding 0.5% trigger a flag.

1.3 Asset Universe Selection

The choice of which assets to trade is more important than the
sophistication of your model. The graph-based approach requires assets
that are moderately correlated (so the graph has meaningful structure)
and liquid (so you can actually trade the signals without slippage
eating your edge).

Recommended Starting Universes

  ---------------------- ----------- ------------------------------------------
  **Universe**           **Count**   **Why**

  S&P 500 Sector ETFs    11          Liquid, moderate cross-correlation, low
  (XLK, XLF, XLE, XLV,               transaction costs, good for the graph
  XLI, XLP, XLU, XLC,                approach because sector relationships are
  XLY, XLRE, XLB)                    meaningful and persistent

  S&P 100 Individual     \~95 liquid More signal but more noise. Use this as a
  Stocks                             second universe once sector ETFs are
                                     working.

  Country/Region ETFs    15--20      Different correlation structure from US
  (EWJ, EWG, EFA, EEM,               sectors. Good for diversification of
  etc.)                              strategy.
  ---------------------- ----------- ------------------------------------------

+-----------------------------------------------------------------------+
| **WHY START WITH SECTOR ETFs**                                        |
|                                                                       |
| Individual stocks have earnings events, management changes, lawsuits, |
| and other idiosyncratic shocks that create noise the graph model      |
| can't distinguish from real signals. ETFs average out this noise.     |
| Your graph Laplacian filter is looking for peer-group deviations;     |
| ETFs give you cleaner peer-group relationships to work with. Graduate |
| to individual stocks after the ETF version is validated.              |
+-----------------------------------------------------------------------+

1.4 Phase 1 Validation Gate

You may not proceed to Phase 2 until all of the following are true:

-   You can pull 5+ years of clean, split-adjusted daily data for your
    entire universe with a single command

-   All validation checks are automated and produce a daily data quality
    report

-   Data is stored in PostgreSQL with proper indexing on date and ticker

-   You have a working config system where all parameters are
    externalized

-   Git repo is set up with a clean directory structure

Phase 2: Graph Signal Processing Engine

+-----------------------------------------------------------------------+
| **PHASE 2** \| Weeks 4--6                                             |
|                                                                       |
| **Graph Signal Processing Engine**                                    |
|                                                                       |
| *Build the core analytical engine that models asset relationships as  |
| a network and generates mean-reversion signals from peer-group        |
| deviations. This is the backbone of the entire system.*               |
+-----------------------------------------------------------------------+

2.1 Correlation Graph Construction

The first step is converting raw price data into a weighted network
where each node is an asset and each edge represents the strength of the
relationship between two assets.

Step-by-Step Construction

1.  Compute daily log returns for each asset: r_t = ln(P_t / P\_{t-1}).
    Log returns are preferred over simple returns because they are
    additive over time and approximately normally distributed.

2.  Compute the pairwise Pearson correlation matrix ρ over a rolling
    window. Start with a 60-trading-day window (approximately 3 months).
    This window length balances responsiveness to changing relationships
    against stability.

3.  Convert correlations to distances: d_ij = √(2(1 − ρ_ij)). This is a
    proper metric (satisfies triangle inequality) and maps perfectly
    correlated assets to distance 0 and uncorrelated assets to distance
    √2. This is the standard metric used in financial TDA literature.

4.  Convert distances to edge weights using a Gaussian kernel: w_ij =
    exp(−d_ij² / 2σ²). The bandwidth parameter σ controls how quickly
    the weight decays with distance. Set σ to the median of all pairwise
    distances as a starting point (this is called the median heuristic
    and is well-justified theoretically).

5.  Apply a sparsification threshold: set weights below a minimum (e.g.,
    0.1) to zero. This removes weak edges that add noise without
    information. A sparse graph is computationally cheaper and produces
    cleaner signals.

+-----------------------------------------------------------------------+
| **UPGRADE PATH: DCC-GARCH CORRELATIONS**                              |
|                                                                       |
| The rolling-window Pearson correlation approach described above is    |
| the baseline. Your existing DCC-GARCH modeling experience is a        |
| genuine advantage here. DCC (Dynamic Conditional Correlation)         |
| produces time-varying correlations that update daily and adapt to     |
| volatility clustering. This is strictly better than rolling windows   |
| because it doesn't have the arbitrary lookback window choice and it   |
| responds faster to correlation shifts. Implement the rolling window   |
| first to validate the concept, then upgrade to DCC-GARCH correlations |
| in Phase 7 (Optimization).                                            |
+-----------------------------------------------------------------------+

2.2 Graph Laplacian Computation

With the weight matrix W constructed, you now compute the graph
Laplacian, which is the mathematical object that enables graph-based
filtering.

**Weight Matrix W:** n × n symmetric matrix where W_ij is the edge
weight between assets i and j. Diagonal is zero (no self-loops).

**Degree Matrix D:** n × n diagonal matrix where D_ii = Σ_j W_ij (sum of
each row of W). This measures how \"connected\" each asset is to the
rest of the network.

**Unnormalized Laplacian:** L = D − W. This is the simplest form but
gives disproportionate influence to highly connected nodes.

**Normalized Laplacian:** L_norm = I − D\^{−1/2} W D\^{−1/2}. Use this
version. It normalizes for connectivity differences, so a mega-cap stock
with many connections doesn't dominate the filter output.

2.3 Graph Signal Filtering

This is the core equation from both Instagram videos. It takes today's
return vector and decomposes it into a "peer-consensus" component and a
"deviation" component.

**The Filter:** h = (I − αL)\^J × x

Where x is today's return vector (one number per asset), α is the
diffusion rate (how strongly each asset's return is pulled toward its
neighbors), J is the number of diffusion steps (how far the smoothing
propagates through the network), and h is the smoothed output
representing the "expected" return for each asset based on its peers.

**The Residual:** e = x − h

This is your trading signal. Large positive e means the asset
outperformed what its peer network predicted. Large negative e means it
underperformed. The trading thesis is that these deviations are
temporary and will revert.

Parameter Selection

  --------------- ------------ ------------- ---------------------------------
  **Parameter**   **Starting   **Range to    **Effect**
                  Value**      Explore**     

  α (diffusion    0.05         0.01 -- 0.15  Higher α = more smoothing = the
  rate)                                      consensus is more influenced by
                                             neighbors. Too high and
                                             everything looks the same; too
                                             low and the filter has no effect.

  J (diffusion    3            1 -- 8        Higher J = smoothing propagates
  steps)                                     further through the network. J=1
                                             only considers direct neighbors;
                                             J=5+ considers second and
                                             third-degree connections.

  σ (kernel       Median of    0.5× to 2×    Controls how fast edge weights
  bandwidth)      distances    median        decay with distance. Smaller σ =
                                             only very similar assets
                                             connected; larger σ = more
                                             connections.

  Lookback window 60 days      40 -- 120     For rolling correlation. Shorter
                               days          = more responsive but noisier.
                                             Longer = more stable but slower
                                             to adapt.
  --------------- ------------ ------------- ---------------------------------

2.4 Signal Generation Rules

The raw residual e needs to be converted into actionable trading
signals. This requires standardization and thresholding.

1.  Standardize e: Compute the rolling mean and standard deviation of
    e_i for each asset over the trailing 60 days. Compute z_i = (e_i −
    mean(e_i)) / std(e_i). This puts all assets on the same scale.

2.  Apply thresholds: When z_i \> +1.5, the asset has deviated
    significantly above its peer consensus. This is a short signal (or
    underweight signal for a long-only version). When z_i \< −1.5, this
    is a long signal. Between −1.5 and +1.5 is the neutral zone with no
    action.

3.  Size positions proportionally to z-score magnitude: The further from
    zero, the larger the position. Use a linear or sigmoid sizing
    function. Cap maximum position size at 10% of portfolio per asset
    (20% for ETF universe).

4.  Apply a holding period: Do not exit a position until either (a) the
    z-score crosses back through zero (reversion complete), (b) a
    stop-loss is hit, or (c) a maximum holding period of 10 trading days
    is reached.

2.5 Backtesting Framework

Every signal generation rule above must be validated against historical
data before proceeding. But backtesting is where most people fool
themselves, so the framework must be designed to prevent self-deception.

Walk-Forward Validation Protocol

Never use a single train/test split. Use expanding-window walk-forward
validation: train on data from the start through month M, test on month
M+1. Then train on data through month M+1, test on month M+2. Repeat.
This simulates the actual experience of trading the strategy in real
time, where you only have access to past data.

Metrics to Track

  ---------------- ------------------ ------------------------------------
  **Metric**       **Target for       **What It Tells You**
                   Sector ETFs**      

  Sharpe Ratio     \> 1.0             Risk-adjusted return. Below 0.5 is
  (annualized)     out-of-sample      noise; 1.0+ is a legitimate signal.

  Maximum Drawdown \< 15%             Worst peak-to-trough loss. If you
                                      can't stomach the max drawdown,
                                      you'll panic-exit at the worst time.

  Win Rate         \> 50%             Percentage of trades that are
                                      profitable. Below 45% for a
                                      mean-reversion strategy suggests the
                                      signal isn't working.

  Profit Factor    \> 1.5             Gross profits / gross losses. Below
                                      1.2 means edge is too thin to
                                      survive transaction costs.

  Avg Holding      2--10 days         Should match mean-reversion time
  Period                              scale. If trades take 30+ days to
                                      resolve, the signal is too slow.

  Annual Turnover  \< 3000% for ETFs  Total value traded / portfolio
                                      value. Excessive turnover means
                                      transaction costs will eat returns.
  ---------------- ------------------ ------------------------------------

2.6 Transaction Cost Modeling

This is where most backtests lie. You must model realistic costs or your
backtest will show phantom profits.

  ------------- --------------------- ------------------------------------
  **Cost        **Realistic           **How to Model**
  Component**   Estimate**            

  Commission    \$0 for Alpaca/most   Zero for equities and ETFs on modern
                brokers               brokers

  Bid-Ask       1--3 bps for liquid   Apply half the spread as cost on
  Spread        ETFs, 5--15 bps for   each side (entry and exit). Use
                mid-cap stocks        historical spread data if available;
                                      otherwise use conservative fixed
                                      estimate.

  Market Impact 1--5 bps for small    Model as a function of order size
                orders (\<\$50K),     relative to average daily volume.
                5--20 bps for larger  Use the square-root impact model:
                                      impact = k × √(shares / ADV).

  Slippage      1--3 bps              Difference between expected
                                      execution price and actual. Model as
                                      random noise around the midpoint.
  ------------- --------------------- ------------------------------------

+-----------------------------------------------------------------------+
| **COST REALITY CHECK**                                                |
|                                                                       |
| For sector ETFs with a \~\$100K portfolio, total round-trip cost is   |
| approximately 3--8 basis points per trade. If your average trade      |
| profit is 15 bps, costs consume 20--50% of gross profits. If your     |
| average trade profit is 5 bps, costs consume 60--160% and the         |
| strategy is not viable. Always check this ratio before getting        |
| excited about backtest returns.                                       |
+-----------------------------------------------------------------------+

2.7 Phase 2 Validation Gate

You may not proceed to Phase 3 until:

-   The graph engine produces signals that achieve a Sharpe ratio above
    0.7 on out-of-sample data (walk-forward), after transaction costs

-   The strategy is profitable in at least 60% of out-of-sample months

-   Maximum drawdown in any walk-forward window does not exceed 20%

-   **If the Sharpe is below 0.5, STOP. Do not proceed. Re-examine
    universe selection, parameter choices, and cost assumptions. The
    graph engine alone must show a legitimate signal before you layer
    complexity on top.**

Phase 3: Topological Regime Detection

+-----------------------------------------------------------------------+
| **PHASE 3** \| Weeks 7--9                                             |
|                                                                       |
| **Topological Regime Detection**                                      |
|                                                                       |
| *Build the early warning system that detects when market correlation  |
| structure is changing. This is what prevents the graph engine from    |
| trading stale signals during regime transitions.*                     |
+-----------------------------------------------------------------------+

3.1 Persistent Homology Pipeline

Persistent homology is a technique from algebraic topology that
identifies structural features (clusters, loops, voids) in data and
measures how robust they are. Applied to stock correlations, it reveals
the hidden geometry of market structure.

The Pipeline

1.  Start with today's distance matrix (same one from Phase 2, Step 3).
    This is an n × n matrix where each entry is the correlation-based
    distance between two assets.

2.  Build the Vietoris-Rips complex. Start with threshold ε = 0. Each
    asset is an isolated point. Gradually increase ε. When ε reaches the
    distance between two assets, connect them with an edge. When three
    mutually-connected assets form a triangle, fill it in. Continue for
    higher-dimensional shapes.

3.  Track the birth and death of topological features. A connected
    component is "born" when a point appears and "dies" when it merges
    with another component. A loop is "born" when a cycle forms and
    "dies" when the cycle is filled in. Each feature gets a (birth,
    death) pair.

4.  Collect all (birth, death) pairs into a persistence diagram. This is
    a scatter plot where the x-axis is birth time and the y-axis is
    death time. Points far from the diagonal line y = x represent
    features that persisted across a wide range of ε values and are
    therefore meaningful structure, not noise.

5.  Repeat daily over a rolling 60-day window. You now have a time
    series of persistence diagrams --- one per day.

3.2 Regime Change Detection

With a time series of persistence diagrams, you can measure how much the
market's structure is changing from day to day.

**Distance Metric:** Use the Wasserstein-1 distance (also called Earth
Mover's Distance) between consecutive persistence diagrams. This
measures the minimum "work" required to transform one diagram into the
next. Low distance = stable structure. High distance = structural shift.

**Regime Classification:** Compute the rolling 20-day mean and standard
deviation of the daily Wasserstein distances. Classify today's regime
as: STABLE if today's distance is within 1 standard deviation of the
mean; TRANSITIONING if between 1 and 2 standard deviations; and NEW
REGIME if beyond 2 standard deviations.

3.3 Integration with Graph Engine

The regime detector modulates the graph engine's behavior:

  --------------- --------------------------------- -------------------------
  **Regime**      **Graph Engine Behavior**         **Position Sizing**

  STABLE          Trade normally using standard     100% of target sizing
                  thresholds                        

  TRANSITIONING   Widen entry thresholds by 50%     50% of target sizing
                  (require stronger signals). Begin 
                  shortening the correlation        
                  lookback window to give more      
                  weight to recent data.            

  NEW REGIME      Halt all new entries. Close       Reduce to 25% of target
                  positions that have been open     sizing. No new positions
                  longer than 3 days. Rebuild the   until regime returns to
                  correlation graph using only the  STABLE.
                  most recent 20 days of data.      
  --------------- --------------------------------- -------------------------

+-----------------------------------------------------------------------+
| **WHY THIS MATTERS**                                                  |
|                                                                       |
| The 2022 bear market is a perfect example. Correlations that held for |
| years broke down over weeks. Growth stocks that used to move together |
| started diverging wildly as the market repriced interest rate         |
| sensitivity. A graph engine without regime detection would have       |
| traded stale correlations and gotten destroyed. The TDA layer would   |
| have flagged the structural shift within days and reduced exposure.   |
+-----------------------------------------------------------------------+

3.4 Phase 3 Validation Gate

You may not proceed to Phase 4 until:

-   The regime detector correctly identifies at least 70% of historical
    drawdown periods as TRANSITIONING or NEW REGIME (test against March
    2020, Q1 2022, SVB crisis March 2023, August 2024 vol spike)

-   Adding regime detection to the graph engine reduces maximum drawdown
    by at least 20% relative (e.g., from 15% to 12%) without reducing
    annualized return by more than 10% relative

-   The combined system (graph + TDA) achieves a Sharpe above 0.8
    out-of-sample after costs

Phase 4: Prediction Engine (TCN)

+-----------------------------------------------------------------------+
| **PHASE 4** \| Weeks 10--13                                           |
|                                                                       |
| **Prediction Engine**                                                 |
|                                                                       |
| *Add a neural network that predicts whether deviations will revert or |
| persist, improving trade timing and reducing false signals.*          |
+-----------------------------------------------------------------------+

4.1 Feature Engineering

The prediction engine's input features determine its ceiling. No model
can extract signal that isn't in the features. Every feature must be
computable from data available at the time of prediction (no
look-ahead).

Feature Categories

  ----------------- ------------------------- ---------------------------------
  **Category**      **Features**              **Rationale**

  Graph Residuals   Current residual e_i,     Direct measure of how much the
                    5-day rolling mean of     asset is deviating from peers
                    e_i, 5-day rolling std of 
                    e_i, z-score of e_i       

  Price Momentum    1-day, 5-day, 10-day,     Captures trend that may cause the
                    20-day returns for the    deviation to persist rather than
                    asset                     revert

  Volatility        20-day realized           High volatility = wider expected
                    volatility, ratio of      deviations = need higher
                    5-day to 20-day           thresholds
                    volatility (vol regime)   

  Volume            Volume relative to 20-day Unusual volume often accompanies
                    average, volume trend     persistent moves, not reverting
                    (5-day avg / 20-day avg)  ones

  Network Position  Node degree (how          A highly connected asset
                    connected the asset is),  deviating from many peers is a
                    average neighbor return,  stronger signal than one
                    graph centrality          deviating from few

  Regime State      Current regime label      The model needs to know the
                    (one-hot encoded),        current market structural context
                    Wasserstein distance      
                    (continuous), regime      
                    duration (days since last 
                    regime change)            

  Cross-Sectional   Rank of residual among    Whether the deviation is
                    all assets, % of assets   idiosyncratic (tradeable) or
                    with same-sign residual,  systematic (not tradeable)
                    dispersion of residuals   
  ----------------- ------------------------- ---------------------------------

4.2 TCN Architecture

A Temporal Convolutional Network processes sequences of feature vectors
and outputs a prediction for the next time step. It uses dilated causal
convolutions, meaning it can only look at past and present data (never
the future), and the dilation allows it to efficiently capture
long-range dependencies.

Recommended Architecture

**Input:** Tensor of shape (batch_size, sequence_length, n_features).
Use sequence_length = 20 trading days. n_features is the total feature
count from the table above (approximately 25--35 per asset).

**Layers:** 4 residual blocks, each containing 2 dilated causal
convolution layers with dilation factors \[1, 2, 4, 8\]. Channel width
of 64. Dropout of 0.2 between blocks.

**Pooling:** Global average pooling across the time dimension to produce
a fixed-length vector.

**Output Head:** Linear layer outputting 2 values per asset: predicted
next-day residual and predicted uncertainty (modeled as the log-variance
of the prediction). This is a heteroscedastic regression setup that
gives you both a point estimate and a confidence measure.

**Loss Function:** Negative log-likelihood under a Gaussian with
predicted mean and variance. This trains the model to be well-calibrated
in its uncertainty estimates, not just accurate in its point
predictions. Well-calibrated uncertainty is critical for the Portfolio
Allocator.

4.3 Training Protocol

Neural networks for financial prediction are extremely prone to
overfitting. The training protocol must be designed to prevent this.

Walk-Forward Training

Use an expanding window: train on years 1--N, validate on year N+0.5,
test on year N+1. Advance by 6 months. Retrain from scratch each time
(no warm-starting from previous model). This gives you multiple
independent out-of-sample test periods.

Overfitting Prevention

-   Early stopping: Monitor validation loss and stop training when it
    hasn't improved for 10 epochs

-   Weight decay: L2 regularization with coefficient 1e-4

-   Dropout: 0.2 in all blocks, 0.3 before the output head

-   Label smoothing: Add small noise (std = 0.01) to target residuals to
    prevent the model from memorizing specific return values

-   Ensemble: Train 5 models with different random seeds. Use the
    average prediction. If models disagree significantly (high
    prediction variance across ensemble), treat the prediction as
    low-confidence.

4.4 Phase 4 Validation Gate

-   TCN predictions improve the Sharpe ratio of the combined system by
    at least 0.15 over the graph + TDA system alone, out-of-sample

-   The predicted uncertainty is well-calibrated: approximately 68% of
    realized residuals fall within the predicted 1-sigma interval

-   **If the TCN does not improve out-of-sample performance, do not use
    it. Revert to the Phase 3 system. Complexity without improvement is
    a net negative (more failure modes, harder to debug).**

Phase 5: Dynamic Portfolio Allocation

+-----------------------------------------------------------------------+
| **PHASE 5** \| Weeks 14--16                                           |
|                                                                       |
| **Dynamic Portfolio Allocation**                                      |
|                                                                       |
| *Build the meta-layer that runs multiple sub-strategies               |
| simultaneously and dynamically shifts capital toward what's working.* |
+-----------------------------------------------------------------------+

5.1 Sub-Strategy Definitions

Run three sub-strategies simultaneously, each targeting a different
signal:

  ---------------- -------------- --------------------- -----------------------
  **Strategy**     **Universe**   **Signal**            **Expected Character**

  A: Sector ETF    11 S&P sector  Graph Laplacian       Low frequency (2--5
  Mean-Reversion   ETFs           residual with TDA     trades/week), moderate
                                  regime overlay        Sharpe, very low costs

  B: Large-Cap     S&P 100 stocks Same graph approach   Higher frequency (5--15
  Stock                           on individual stocks  trades/week),
  Mean-Reversion                  with TCN timing       potentially higher
                                                        Sharpe, higher costs

  C: Cross-Asset   Mix of equity, Trend-following       Uncorrelated to A and
  Momentum         bond,          overlay using 50/200  B. Profitable in
                   commodity, and day moving averages   trending markets when
                   currency ETFs                        mean-reversion
                   (SPY, TLT,                           struggles.
                   GLD, UUP)                            
  ---------------- -------------- --------------------- -----------------------

+-----------------------------------------------------------------------+
| **WHY STRATEGY C EXISTS**                                             |
|                                                                       |
| Mean-reversion strategies have a known weakness: they lose money in   |
| strong trending markets where deviations persist instead of           |
| reverting. Strategy C is a momentum/trend-following strategy that     |
| profits in exactly those conditions. The combination provides         |
| structural diversification: when A and B are losing, C tends to be    |
| winning, and vice versa.                                              |
+-----------------------------------------------------------------------+

5.2 The Utility Function

Each sub-strategy receives a dynamic capital allocation based on a
utility score that balances expected return against uncertainty and
transaction costs:

**Utility Score:** ν_t\^s = η̂\_t\^s − λ_U × U_t\^s − λ_C × ĉ\_t\^s

Where η̂ is the predicted return for strategy s at time t, U is the
prediction uncertainty (from the TCN ensemble disagreement or rolling
forecast error), ĉ is the estimated transaction cost for the required
rebalancing, λ_U controls the penalty for uncertainty (start at 1.0),
and λ_C controls the penalty for costs (start at 2.0; costs should be
penalized more heavily because they are guaranteed losses while returns
are uncertain).

Capital Allocation

Convert utility scores to capital weights using a softmax function with
temperature parameter τ: w_s = exp(ν_s / τ) / Σ exp(ν_j / τ).
Temperature τ controls how concentrated the allocation is. High τ =
equal allocation regardless of scores. Low τ = winner-take-all. Start
with τ = 1.0.

Allocation Constraints

-   Minimum allocation per strategy: 10% of total capital. This prevents
    the allocator from completely abandoning a strategy that may
    recover.

-   Maximum allocation per strategy: 60%. This prevents
    over-concentration in a single strategy.

-   Rebalancing frequency: Weekly. Daily rebalancing of capital between
    strategies generates excessive costs with minimal benefit.

Phase 6: The Self-Correcting Learning Engine

+-----------------------------------------------------------------------+
| **PHASE 6** \| Weeks 17--22                                           |
|                                                                       |
| **The Learning Engine**                                               |
|                                                                       |
| *This is the system you specifically asked for: a mechanism that      |
| makes the strategy learn from its own mistakes and improve over time. |
| This is the hardest component to build correctly and the most         |
| valuable.*                                                            |
+-----------------------------------------------------------------------+

6.1 What "Learning From Mistakes" Actually Means

There are three fundamentally different types of mistakes a trading
system can make, and each requires a different corrective mechanism.
Conflating them is the #1 reason adaptive systems fail.

  ------------ ------------------- --------------------- -----------------
  **Mistake    **Example**         **Correction          **Frequency**
  Type**                           Mechanism**           

  Parameter    The optimal α for   Bayesian parameter    Monthly
  Drift        the graph filter    optimization on a     
               was 0.05 last year  rolling basis         
               but should be 0.08                        
               now because                               
               correlations have                         
               tightened                                 

  Model        The TCN was trained Scheduled retraining  Quarterly
  Staleness    on 2019--2023 data  with fresh data       
               and doesn't capture                       
               post-2024 market                          
               dynamics                                  

  Structural   The mean-reversion  Strategy-level        Continuously
  Failure      thesis itself stops performance           monitored, rare
               working for a       monitoring with       action
               specific asset      automatic             
               class because       deactivation triggers 
               market                                    
               microstructure                            
               changed                                   
  ------------ ------------------- --------------------- -----------------

6.2 Mistake Detection: The Trade Journal

Every trade the system makes is logged with the full context that
produced it. This is the raw material the Learning Engine analyzes.

Data Captured Per Trade

  ------------------------------ ----------------------------------------
  **Field**                      **Purpose**

  Entry date, exit date, holding Timing analysis
  period                         

  Asset, direction (long/short), Basic trade record
  entry price, exit price        

  Graph residual z-score at      Signal strength at time of entry
  entry                          

  Regime state at entry and exit Whether regime changed during the trade

  TCN prediction and confidence  Model's expected outcome
  at entry                       

  Actual P&L (gross and net of   Realized outcome
  costs)                         

  Prediction error = predicted   Core learning signal
  residual − actual residual     

  Concurrent positions and       Crowding and correlation risk context
  portfolio-level exposure       
  ------------------------------ ----------------------------------------

6.3 Learning Loop 1: Bayesian Parameter Optimization

Instead of fixing parameters like α, J, and σ permanently, the system
continuously estimates the best parameters given recent performance
data.

How It Works

1.  Define a prior distribution over each parameter. For α, start with a
    Gaussian prior centered on 0.05 with std 0.03. For J, a discrete
    uniform prior over {1, 2, 3, 4, 5}. For σ, a Gaussian prior centered
    on the median distance with std equal to 25% of the median.

2.  Every month, evaluate the last 60 trading days of performance for a
    grid of parameter combinations. The likelihood of each parameter
    combination is proportional to the Sharpe ratio it would have
    achieved.

3.  Update the posterior distribution using Bayes' rule: posterior ∝
    prior × likelihood. The prior acts as a regularizer, preventing the
    system from chasing noise in recent performance.

4.  Set the active parameters to the posterior mean (not the posterior
    mode --- the mean is more robust to multi-modal posteriors). Update
    is gradual: new_param = 0.7 × old_param + 0.3 × posterior_mean. This
    exponential smoothing prevents wild parameter swings.

+-----------------------------------------------------------------------+
| **WHY BAYESIAN, NOT GRID SEARCH**                                     |
|                                                                       |
| Grid search picks the single best parameter combination on recent     |
| data. This is extremely prone to overfitting. Bayesian optimization   |
| maintains a distribution over parameters, which means it inherently   |
| accounts for parameter uncertainty. If the data doesn't strongly      |
| favor any particular parameter value, the posterior stays wide and    |
| the parameters barely change. If the data strongly favors a shift,    |
| the posterior narrows and the parameters update. This is              |
| self-regulating in exactly the right way.                             |
+-----------------------------------------------------------------------+

6.4 Learning Loop 2: Scheduled Model Retraining

The TCN and any other ML models are retrained on a fixed schedule with
expanding training windows.

**Retraining Cadence:** Quarterly (every 63 trading days). More frequent
retraining risks overfitting to recent noise. Less frequent risks model
staleness.

**Training Window:** Expanding window from system inception through the
most recent trading day, minus a 20-day holdout buffer (to prevent
information leakage from the most recent period used for validation).

**Model Selection:** Each retraining produces a candidate model. Compare
the candidate against the incumbent model on the 20-day holdout period.
The candidate replaces the incumbent only if it achieves a higher Sharpe
ratio on the holdout. If the candidate is worse, keep the incumbent.
This prevents a bad retraining from degrading live performance.

**Ensemble Update:** When a new model passes the selection gate, it
replaces the oldest model in the ensemble (not all of them). This
provides continuity: the ensemble always contains models trained on
different vintages of data.

6.5 Learning Loop 3: Mistake Pattern Analysis

This is the most sophisticated learning mechanism. It identifies
systematic patterns in the trades that lose money and adjusts behavior
to avoid repeating them.

Weekly Mistake Categorization

Every Friday, the system runs an automated analysis of all losing trades
from the past week:

  ---------------- ------------------------ ----------------------------------
  **Mistake        **Detection Criteria**   **Corrective Action**
  Category**                                

  False            Trade entered on a       Increase the momentum filter
  Mean-Reversion   strong residual signal   weight in the TCN features. For
                   but the deviation        the next month, require that the
                   persisted or widened.    5-day momentum oppose the
                   The asset was trending,  deviation before entering.
                   not mean-reverting.      

  Regime-Stale     Trade entered during     Tighten the TRANSITIONING
  Signal           STABLE regime but a      threshold by 10%. The regime
                   regime transition        detector was too slow to flag the
                   occurred within 3 days   shift.
                   of entry.                

  Cost-Killed Edge Trade was profitable     Increase the minimum z-score
                   before costs but         threshold for entry by 0.1 for
                   unprofitable after       that asset's cost profile. Some
                   costs.                   assets are simply too expensive to
                                            trade at current position sizes.

  Crowded Exit     Many positions exited on Stagger exits over 2--3 days when
                   the same day, causing    portfolio turnover exceeds 20% in
                   realized slippage        a single day.
                   exceeding the modeled    
                   estimate.                

  Correlation      An asset that was highly Exclude assets whose node degree
  Breakdown        connected in the graph   drops below the 20th percentile of
                   suddenly disconnected    the universe for 5+ consecutive
                   (node degree dropped).   days.
                   Signals during           
                   disconnection were       
                   unreliable.              

  Volatility       Position size was based  Add a volatility-scaling layer:
  Mismatch         on normal volatility but reduce position size
                   realized volatility      proportionally when recent vol
                   during the trade was 2×+ exceeds 1.5× the 60-day average.
                   higher, amplifying the   
                   loss.                    
  ---------------- ------------------------ ----------------------------------

6.6 Learning Loop 4: Strategy-Level Kill Switch

The most drastic learning mechanism: automatically reducing or
deactivating a sub-strategy that is consistently underperforming.

**Trigger:** If a sub-strategy's rolling 60-day Sharpe ratio drops below
−0.5 (i.e., it's consistently losing money on a risk-adjusted basis),
its allocation is reduced to the minimum (10%).

**Quarantine:** If the rolling 120-day Sharpe is below −0.5, the
strategy is quarantined: it continues to generate signals and track
paper performance, but no real capital is allocated. This allows you to
observe whether the strategy recovers without risking money.

**Reactivation:** A quarantined strategy is reactivated when its paper
Sharpe rises above 0.0 for 40 consecutive trading days.

**Permanent Kill:** If a strategy remains quarantined for 6 months, it
is presumed structurally broken and must be redesigned, not just
re-parameterized.

6.7 The Learning Dashboard

Build a dashboard (Streamlit or Dash) that displays, in real time:

-   Current parameter values and their drift over time (are parameters
    trending in a direction, or oscillating?)

-   Mistake categorization breakdown (pie chart of losing trade
    categories this month vs. last month)

-   Prediction calibration plot (predicted vs. actual residuals with
    confidence intervals)

-   Sub-strategy allocation history and utility scores

-   Current regime state and transition history

-   Equity curve with drawdown overlay for each strategy and the
    combined portfolio

Phase 7: Risk Management & Execution

+-----------------------------------------------------------------------+
| **PHASE 7** \| Weeks 23--26                                           |
|                                                                       |
| **Risk Management & Live Execution**                                  |
|                                                                       |
| *Build the safety layer that prevents catastrophic losses and connect |
| to the broker for live trading. This is where the system becomes      |
| real.*                                                                |
+-----------------------------------------------------------------------+

7.1 Risk Limits (Non-Negotiable)

These limits are hardcoded circuit breakers that cannot be overridden by
any signal, model, or allocation logic. They exist because models fail,
and when they fail, they fail fast.

  ---------------- ------------------ ------------------------------------
  **Limit**        **Threshold**      **Action When Breached**

  Maximum Daily    2% of portfolio    All positions closed. No new trades
  Loss                                until next trading day. Alert sent
                                      to phone.

  Maximum Weekly   5% of portfolio    All positions closed. System paused
  Loss                                for 3 trading days. Full diagnostic
                                      required before resumption.

  Maximum Monthly  10% of portfolio   All positions closed. System paused
  Loss                                indefinitely. Manual review and
                                      explicit restart required.

  Maximum Single   10% of portfolio   Position capped at limit. Excess
  Position         (ETFs: 20%)        orders rejected.

  Maximum Gross    200% of portfolio  No new positions until exposure
  Exposure         (100% long + 100%  drops. Most relevant if using
                   short)             margin/leverage.

  Maximum Net      50% long or 50%    Prevents unintended directional
  Exposure         short              bets. The strategy should be roughly
                                      market-neutral.

  Maximum          \|0.3\| over       If portfolio returns correlate too
  Correlation to   trailing 20 days   highly with the market, the strategy
  SPY                                 has become a disguised beta bet, not
                                      an alpha strategy. Reduce positions.

  Order Size       No single order \> Reject order. Log and alert. Likely
  Sanity           5% of portfolio    a bug.
  ---------------- ------------------ ------------------------------------

+-----------------------------------------------------------------------+
| **THESE LIMITS ARE NOT NEGOTIABLE**                                   |
|                                                                       |
| Do not modify these limits based on backtest performance. Backtests   |
| do not capture tail events, system failures, data errors, or flash    |
| crashes. The limits exist to protect you from scenarios that don't    |
| appear in historical data. If a limit is triggered, the correct       |
| response is always to stop and investigate, never to widen the limit. |
+-----------------------------------------------------------------------+

7.2 Broker Integration (Alpaca)

Alpaca provides commission-free equity and ETF trading with a
well-documented API. Use the alpaca-trade-api Python library.

Implementation Sequence

6.  Paper trading account: Open immediately. All development and testing
    uses paper trading. The API is identical to live trading, so code
    requires zero changes to switch.

7.  Order management: Build an order management layer that translates
    target positions into orders. Compare current holdings to target
    holdings. Generate market orders for the difference. Use limit
    orders only if you implement a more sophisticated execution layer
    later.

8.  Order validation: Before submitting any order, verify it passes all
    risk limits. Log the order, the risk check results, and the
    submission timestamp.

9.  Fill monitoring: After submission, poll for fill confirmation. Log
    fill price, time, and quantity. Compute and log slippage (difference
    between expected and actual fill price).

10. Reconciliation: Every morning before the pipeline runs, reconcile
    the system's internal position records against the broker's reported
    positions. Any discrepancy halts trading until resolved.

7.3 Paper Trading Validation

**Minimum Duration:** 90 calendar days of paper trading before going
live. No exceptions.

**What You're Measuring:** Not whether the strategy is profitable
(backtests already told you that). You're measuring whether the live
system behaves as expected: Do fills match expected prices? Do positions
match targets? Do risk limits trigger correctly? Does the pipeline run
reliably every day without manual intervention?

**Go-Live Criteria:** 90 days of uninterrupted daily execution,
portfolio tracking error vs. backtest simulation below 5% annualized,
zero risk limit breaches caused by system errors, all learning loops
executing on schedule.

Phase 8: Live Deployment & Scaling

+-----------------------------------------------------------------------+
| **PHASE 8** \| Week 27+                                               |
|                                                                       |
| **Live Deployment**                                                   |
|                                                                       |
| *Put real money behind the system. Start small, scale slowly, and let |
| the learning engine prove itself.*                                    |
+-----------------------------------------------------------------------+

8.1 Capital Deployment Schedule

Never go from \$0 to full allocation in one step. Scale in gradually,
using each step as a live validation of the system's behavior.

  ------------- -------------------- ------------------------------------------
  **Weeks After **Capital Deployed** **Purpose**
  Go-Live**                          

  Weeks 1--4    \$5,000--\$10,000    Minimum viable deployment. Verify
                                     execution, fills, and slippage. Compare
                                     live P&L to paper trading P&L. If
                                     divergence exceeds 2% annualized,
                                     investigate.

  Weeks 5--12   \$15,000--\$25,000   First scaling step. Learning engine should
                                     have its first monthly recalibration
                                     cycle. Verify parameter updates are
                                     sensible.

  Weeks 13--24  \$25,000--\$50,000   Second scaling step. By now you have a
                                     full quarter of live data. Compare live
                                     Sharpe to backtest Sharpe. If live Sharpe
                                     is less than 50% of backtest Sharpe, pause
                                     and investigate.

  Weeks 25+     Scale to target      Only after 6 months of live trading with
                allocation           consistent performance tracking. Target
                                     allocation depends on your total
                                     investable assets and risk tolerance. A
                                     common guideline: no more than 25--40% of
                                     liquid net worth in a single algorithmic
                                     strategy.
  ------------- -------------------- ------------------------------------------

8.2 Ongoing Operations

Once live, the system requires ongoing monitoring and maintenance. This
is not set-and-forget.

Daily (5 minutes)

Check that the pipeline ran successfully. Review the daily summary email
(which the system should send automatically): positions, P&L, any risk
limit approaches, any data quality flags.

Weekly (30 minutes)

Review the Learning Dashboard. Check mistake categorization trends.
Review parameter drift. Verify the regime detector's current state
matches your qualitative assessment of market conditions.

Monthly (2 hours)

Full performance review. Compare live performance to backtest
expectations. Review the Bayesian parameter updates. Check that the
portfolio allocator's capital shifts are sensible. Review transaction
costs vs. estimates.

Quarterly (half day)

TCN retraining cycle. Evaluate whether new features should be added.
Review universe composition (should any assets be added or removed?).
Tax loss harvesting review.

8.3 Legal and Tax Considerations

+-----------------------------------------------------------------------+
| **EMPLOYMENT AGREEMENT**                                              |
|                                                                       |
| Before deploying real capital in an algorithmic trading strategy,     |
| review your employment agreement with a lawyer. Some employers        |
| (particularly in finance and pharma) have restrictions on outside     |
| trading activities, required pre-clearance for trades, or blackout    |
| periods. Automated trading that generates hundreds of trades per year |
| may trigger compliance requirements you're not aware of.              |
+-----------------------------------------------------------------------+

From a tax perspective, the strategy will likely generate mostly
short-term capital gains (holding periods under 1 year), which are taxed
as ordinary income. With your existing LETF simulation work covering
wash sale rules and tax lot tracking, you're already familiar with the
relevant complexities. Consider running the strategy in a tax-advantaged
account (Roth IRA via Alpaca) for a portion of the capital if feasible.

Appendix A: Complete Technology Stack

  --------------- ---------------------- ---------------------------------
  **Component**   **Technology**         **Purpose**

  Language        Python 3.11+           All system code

  Data Storage    PostgreSQL +           Price data, trade logs, learning
                  TimescaleDB            records

  Config          Hydra / OmegaConf      Externalized parameters
  Management                             

  Graph           NetworkX, NumPy, SciPy Graph construction and Laplacian
  Processing                             computation

  TDA             giotto-tda, ripser     Persistent homology and regime
                                         detection

  Deep Learning   PyTorch                TCN implementation and training

  Backtesting     vectorbt or custom     Walk-forward validation

  Broker API      alpaca-trade-api       Paper and live trading

  Dashboard       Streamlit or Plotly    Learning dashboard and monitoring
                  Dash                   

  Scheduling      APScheduler            Daily pipeline automation

  Optimization    Optuna (for Bayesian   Parameter optimization in
                  hyperparameter search) Learning Engine

  Version Control Git + GitHub           Code and experiment tracking

  Logging         Python logging +       Structured JSON logs for all
                  structlog              decisions

  Alerting        Twilio or Pushover     Phone alerts for risk limit
                                         breaches and system failures
  --------------- ---------------------- ---------------------------------

Appendix B: Key Academic References

*Gidea, M. & Katz, Y. (2018).* Topological Data Analysis of Financial
Time Series: Landscapes of Crashes. Physica A, 491, 820--834. The
foundational paper for applying persistent homology to financial regime
detection.

*Ortega, A. et al. (2018).* Graph Signal Processing: Overview,
Challenges, and Applications. Proceedings of the IEEE, 106(5), 808--828.
Comprehensive reference for graph Laplacians and graph filtering, which
is the core of the signal generation engine.

*Bai, S., Kolter, J.Z. & Koltun, V. (2018).* An Empirical Evaluation of
Generic Convolutional and Recurrent Networks for Sequence Modeling. The
paper that introduced TCNs and demonstrated their superiority over LSTMs
for many sequence tasks.

*Engle, R. (2002).* Dynamic Conditional Correlation: A Simple Class of
Multivariate GARCH Models. Journal of Business & Economic Statistics,
20(3), 339--350. The DCC-GARCH paper, relevant for upgrading the
correlation estimation in Phase 7.

*de Prado, M. López (2018).* Advances in Financial Machine Learning.
Wiley. Comprehensive treatment of backtesting pitfalls, feature
engineering, and model validation for financial ML. Required reading
before building the TCN.

*Cont, R. (2001).* Empirical Properties of Asset Returns: Stylized Facts
and Statistical Issues. Quantitative Finance, 1, 223--236. Essential
background on the statistical properties of financial returns that your
models must account for.

Appendix C: Master Timeline

  ---------------- ----------- ------------------------ ---------------------------
  **Phase**        **Weeks**   **Deliverable**          **Gate Criteria**

  1\.              1--3        Working data pipeline    5+ years clean data,
  Infrastructure               with validated           automated validation,
                               historical data          PostgreSQL storage

  2\. Graph Engine 4--6        Graph-based signal       Sharpe \> 0.7 OOS after
                               generator with full      costs; profitable 60%+ of
                               backtest                 months

  3\. TDA Regime   7--9        Regime detector          Catches 70%+ of drawdowns;
                               integrated with graph    reduces max DD 20%+; Sharpe
                               engine                   \> 0.8

  4\. TCN          10--13      Trained ensemble of TCN  Improves Sharpe by 0.15+;
  Prediction                   models                   calibrated uncertainty; or
                                                        revert to Phase 3

  5\. Portfolio    14--16      Multi-strategy           Combined Sharpe \> 1.0 OOS;
  Allocator                    allocation system        max DD \< 12%

  6\. Learning     17--22      All four learning loops  Parameters update sensibly;
  Engine                       operational              mistake categories tracked;
                                                        retraining pipeline works

  7\. Risk &       23--26      Broker-connected system  90 days paper trading;
  Execution                    with risk limits         tracking error \< 5%; zero
                                                        system-caused breaches

  8\. Live         27+         Real capital deployed    6 months live before full
  Deployment                   and scaling              allocation; live Sharpe \>
                                                        50% of backtest Sharpe
  ---------------- ----------- ------------------------ ---------------------------

*End of Document*
