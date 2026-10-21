"""Command-line interface for pricing a single European option.

Example:
    python -m engine.cli --spot 100 --strike 105 --time 0.5 --rate 0.03 --vol 0.25 --type call
"""

from __future__ import annotations

import argparse

from .black_scholes import BlackScholes, OptionType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Black-Scholes option pricer")
    parser.add_argument("--spot", type=float, required=True, help="underlying price")
    parser.add_argument("--strike", type=float, required=True, help="strike price")
    parser.add_argument("--time", type=float, required=True, help="time to expiry in years")
    parser.add_argument("--rate", type=float, required=True, help="risk-free rate, e.g. 0.03")
    parser.add_argument("--vol", type=float, help="volatility, e.g. 0.25 (omit if using --market-price)")
    parser.add_argument("--dividend", type=float, default=0.0, help="continuous dividend yield")
    parser.add_argument("--type", choices=["call", "put"], default="call")
    parser.add_argument(
        "--market-price",
        type=float,
        default=None,
        help="if given, solve implied volatility instead of pricing",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    option_type = OptionType(args.type)

    if args.market_price is not None:
        engine = BlackScholes(
            spot=args.spot,
            strike=args.strike,
            time_to_expiry=args.time,
            rate=args.rate,
            volatility=args.vol or 0.2,
            option_type=option_type,
            dividend_yield=args.dividend,
        )
        iv = engine.implied_volatility(args.market_price)
        print(f"Implied volatility: {iv:.6f} ({iv * 100:.2f}%)")
        return

    if args.vol is None:
        raise SystemExit("--vol is required unless --market-price is given")

    engine = BlackScholes(
        spot=args.spot,
        strike=args.strike,
        time_to_expiry=args.time,
        rate=args.rate,
        volatility=args.vol,
        option_type=option_type,
        dividend_yield=args.dividend,
    )
    price = engine.price()
    g = engine.greeks()

    print(f"Price: {price:.6f}")
    print(f"Delta: {g.delta:.6f}")
    print(f"Gamma: {g.gamma:.6f}")
    print(f"Vega:  {g.vega / 100:.6f}  (per 1 vol point)")
    print(f"Theta: {g.theta / 365:.6f}  (per day)")
    print(f"Rho:   {g.rho / 100:.6f}  (per 1% rate move)")


if __name__ == "__main__":
    main()
