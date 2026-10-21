# energy-derivatives-pricing

Pricing library for energy and commodity derivatives: Black-76 for options on
forwards and futures, Black-Scholes-Merton for spot-based contracts, full
analytic Greeks, implied volatility, and Monte Carlo simulation.

[![tests](https://github.com/Amitav507/energy-derivatives-pricing/actions/workflows/ci.yml/badge.svg)](https://github.com/Amitav507/energy-derivatives-pricing/actions/workflows/ci.yml)

The closed-form pricing core has no dependencies beyond the standard library.
Simulation uses numpy. The optional market data layer pulls real forward
curves and rate curves from free public sources.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from engine import Black76, OptionType

# A three month call on a gas forward at 3.40, struck at 3.50, 45% vol.
option = Black76(
    forward=3.40,
    strike=3.50,
    time_to_expiry=0.25,
    rate=0.04,
    volatility=0.45,
    option_type=OptionType.CALL,
)

print(option.price())          # 0.259
print(option.greeks().delta)   # 0.489

# Recover the volatility the market is quoting.
print(option.implied_volatility(market_price=0.30))
```

Greeks come back in raw units. Divide vega by 100 for a one volatility point
move, theta by 365 for a day of decay, and rho by 100 for a 1% rate move.

## Why Black-76 rather than Black-Scholes

Energy and commodity options are written on forwards and futures, not on
spot. Black-Scholes-Merton prices spot plus a cost of carry, which for a
commodity means financing, storage and convenience yield. Convenience yield
is unobservable and moves exactly when it matters, and for power the whole
framework collapses, because power cannot be stored and no spot-forward
arbitrage exists to define a carry at all.

Black-76 sidesteps this by pricing off the forward the exchange already
quotes. That quote has priced the carry for you, so it never has to be
estimated. Formally it is the zero carry case of Black-Scholes-Merton, and
both are implemented here.

One consequence the library gets right and a naive substitution does not.
Under Black-76 the rate enters only through the discount factor, never
through `d1` or `d2`, so rho is `-T * price` and is always negative. Pricing
Black-76 by setting `dividend_yield = rate` in a Black-Scholes engine
reproduces the price and four of the Greeks exactly, but reports rho with the
wrong sign, because it differentiates the rate while holding the dividend
yield fixed. `Black76` is a real implementation rather than that
substitution, and `tests/test_black76.py` pins the difference.

## Validation

Correctness is checked three independent ways rather than against the
library's own formulas.

**Against a reference implementation.** Prices agree with `py_vollib` to
**2.5e-14** across a 144 point grid of forward, strike, maturity and
volatility. At the prices in that grid one floating point unit is about
1.4e-14, so the residual is under two units and comes from the normal CDF
itself: `py_vollib` goes through `scipy.stats.norm.cdf`, this library uses
`math.erf`.

**Against simulation.** Monte Carlo and the closed form are independent
numerical routes to the same number. `tests/test_monte_carlo.py` asserts they
agree within four standard errors, which is a statistical test rather than an
arbitrary tolerance.

**Against no-arbitrage identities.** Put-call parity in both the spot and
futures forms, no-arbitrage price bounds, Greek sign and bound conditions,
and textbook reference values.

Two real pricing bugs were found and fixed this way, both documented in the
tests: a zero-volatility branch that returned undiscounted spot intrinsic
instead of the discounted payoff on the forward, and an implied volatility
solver whose lower bound rejected valid deep in-the-money European put quotes,
which legitimately trade below `K - S` because they cannot be exercised early.

## Monte Carlo

```python
from engine import price_european, OptionType

result = price_european(
    forward=3.40, strike=3.50, time_to_expiry=0.25,
    rate=0.04, volatility=0.45,
    option_type=OptionType.CALL,
    paths=200_000, seed=42,
)
print(result)   # price +/- standard error, with a 95% confidence interval
```

Antithetic sampling is on by default. Because antithetic draws are not
independent, the standard error is computed from the pair averages rather
than from all payoffs, which the naive calculation would overstate.

Simulation exists here to cross-check the closed forms, and as the route to
contracts that have no closed form: path dependent payoffs, average price
options, and the volume optionality inside a flexible supply contract.

## Market data

The optional data layer assembles pricing inputs from free public sources.

```bash
pip install -e ".[data]"
python notebooks/multi_asset_demo.py
```

| Module | Source |
|---|---|
| `marketdata.futures_prices` | Yahoo Finance continuous futures: gas, crude, metals |
| `marketdata.rates` | FRED, SOFR and Treasury constant maturity |
| `marketdata.realized_vol` | Rolling annualised volatility from log returns |
| `marketdata.entsoe_power` | UK day-ahead power prices, needs a free ENTSOE key |

Rates are converted to continuous compounding before use. Treasury constant
maturity yields are quoted bond-equivalent and SOFR is simple annualised, and
mixing those conventions into a pricing model is a roughly 12 basis point
error at 5% rate levels.

## Project layout

```
engine/
  black76.py        Options on forwards and futures
  black_scholes.py  Spot-based, with continuous dividend yield
  monte_carlo.py    Simulation, antithetic sampling, standard errors
  _solver.py        Bracketed Newton implied volatility, shared
  cli.py            Command line pricer
marketdata/         Forward curves, rate curves, realized volatility
tests/              Pricing tests, run in CI
notebooks/          End to end demo across several underlyings
```

## Known limitations

Stated plainly, because they are the difference between a pricing library and
a pricing system.

- **Volatility is a scalar.** One number per underlying, so there is no
  smile, no skew and no term structure. Real commodity surfaces have all
  three, including the Samuelson effect and hard seasonality in gas and
  power. A strike and tenor indexed surface is the main piece of work
  outstanding.
- **Volatility comes from realized, not implied.** Realized volatility is
  backward looking and contains no variance risk premium, so it sits
  systematically below implied. Calibrating to a quoted surface needs vendor
  data that is not freely available.
- **Lognormal only.** Power prices go negative and WTI settled negative in
  April 2020. A lognormal model cannot represent that at all. A Bachelier
  model is the standard answer and is not implemented yet.
- **European exercise only.** NYMEX crude options are American style.
- **Vanilla payoffs only.** No Asian options, which dominate oil products,
  and no spread options, which is how spark and dark spreads are priced.
- **Forward curve is a single point.** The front month close stands in for
  the whole curve, discarding contango, backwardation and seasonal shape.
- **Continuous futures contracts.** Yahoo `=F` series splice successive
  contracts without back-adjusting, so roll dates can inject volatility that
  is not a real market move. Checked on the current sample and roll artifacts
  were not driving the estimate, but the exposure is real.
- **The market data layer is not covered by CI**, because it calls live
  external APIs. Only the pricing tests run there.

## Licence

MIT.
