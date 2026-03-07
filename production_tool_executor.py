"""
Production Tool Executor for PULSE Trading Platform.

Implements deterministic, explainable signal logic. Core technical tools (1-5)
are enhanced with options microstructure-aware tools (6-10) that use Greeks,
GEX, Charm, and Vanna concepts from the repository's quant engines.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

try:
    from core.trainer.engines import GreekEngines
except Exception:  # pragma: no cover - defensive fallback
    GreekEngines = None


TOOL_NAMES = {
    1: "RSI Momentum",
    2: "MACD Signal",
    3: "Volume Spike",
    4: "Moving Average Cross",
    5: "VWAP Bias",
    6: "Gamma Regime Drift",
    7: "Charm Flow Pressure",
    8: "Vanna Shock Reversal",
    9: "Gamma Flip Proximity",
    10: "Dealer Flow Composite",
}


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _extract_float(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _extract_float_list(data: Dict[str, Any], key: str) -> List[float]:
    value = data.get(key, [])
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(0.0)
    return out




def _infer_time_bucket(tte_days: float) -> str:
    if tte_days <= 1:
        return "0dte"
    if tte_days <= 7:
        return "weekly"
    if tte_days <= 45:
        return "swing"
    return "long_dated"


def _infer_market_regime(data: Dict[str, Any]) -> str:
    vix = _extract_float(data, "vix", 18.0)
    ret_1d = _extract_float(data, "ret_1d", 0.0)
    realized_vol = _extract_float(data, "realized_vol", 0.2)

    if vix >= 30 or realized_vol >= 0.45:
        return "high_volatility"
    if abs(ret_1d) >= 0.015:
        return "trend"
    return "range"


def _select_pricing_model(data: Dict[str, Any]) -> str:
    tte_days = _extract_float(data, "tte_days", 1.0)
    bucket = _infer_time_bucket(tte_days)
    if bucket == "0dte":
        return "intraday_microstructure_greeks"
    if bucket == "weekly":
        return "black_scholes_greeks"
    if bucket == "swing":
        return "black_scholes_term_structure_proxy"
    return "vol_surface_long_dated_proxy"


def _build_context_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    tte_days = _extract_float(data, "tte_days", 1.0)
    return {
        "time_bucket": _infer_time_bucket(tte_days),
        "market_regime": _infer_market_regime(data),
        "pricing_model": _select_pricing_model(data),
        "tte_days": tte_days,
    }

def _signal_from_rsi(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    rsi = _extract_float(data, "rsi", 50.0)
    if rsi <= 30:
        signal = 1
        score = _clamp_score((30 - rsi) / 30 + 0.55)
    elif rsi >= 70:
        signal = -1
        score = _clamp_score((rsi - 70) / 30 + 0.55)
    else:
        signal = 0
        score = _clamp_score(1 - abs(rsi - 50) / 20)
    return signal, score, {"rsi": rsi}


def _signal_from_macd(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    macd = _extract_float(data, "macd", 0.0)
    signal_line = _extract_float(data, "signal_line", 0.0)
    gap = macd - signal_line
    if gap > 0:
        signal = 1
    elif gap < 0:
        signal = -1
    else:
        signal = 0
    score = _clamp_score(min(1.0, abs(gap) * 2 + 0.5))
    return signal, score, {"macd": macd, "signal_line": signal_line, "gap": round(gap, 6)}


def _signal_from_volume(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    volume = _extract_float(data, "volume", 0.0)
    avg_volume = _extract_float(data, "avg_volume", max(volume, 1.0))
    price = _extract_float(data, "price", 0.0)
    prev_price = _extract_float(data, "prev_price", price)
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
    if volume_ratio >= 1.5:
        signal = 1 if price >= prev_price else -1
    else:
        signal = 0
    score = _clamp_score(min(1.0, volume_ratio / 2))
    return signal, score, {
        "volume": volume,
        "avg_volume": avg_volume,
        "volume_ratio": round(volume_ratio, 4),
        "price": price,
        "prev_price": prev_price,
    }


def _signal_from_ma_cross(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    sma_fast = _extract_float(data, "sma_fast", 0.0)
    sma_slow = _extract_float(data, "sma_slow", 0.0)
    if sma_fast > sma_slow:
        signal = 1
    elif sma_fast < sma_slow:
        signal = -1
    else:
        signal = 0
    denom = abs(sma_slow) if abs(sma_slow) > 1e-9 else 1.0
    cross_strength = abs(sma_fast - sma_slow) / denom
    score = _clamp_score(min(1.0, cross_strength * 20 + 0.5))
    return signal, score, {
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "cross_strength": round(cross_strength, 6),
    }


def _signal_from_vwap(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    price = _extract_float(data, "price", 0.0)
    vwap = _extract_float(data, "vwap", price)
    if price > vwap:
        signal = 1
    elif price < vwap:
        signal = -1
    else:
        signal = 0
    denom = abs(vwap) if abs(vwap) > 1e-9 else 1.0
    bias = abs(price - vwap) / denom
    score = _clamp_score(min(1.0, bias * 25 + 0.5))
    return signal, score, {"price": price, "vwap": vwap, "bias": round(bias, 6)}


def _compute_chain_metrics(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute options microstructure metrics with repo Greek engines."""
    if GreekEngines is None:
        return None

    spot = _extract_float(data, "spot", _extract_float(data, "price", 0.0))
    if spot <= 0:
        return None

    strikes = _extract_float_list(data, "strikes")
    oi_call = _extract_float_list(data, "oi_call")
    oi_put = _extract_float_list(data, "oi_put")
    if not strikes:
        return None

    n = min(len(strikes), len(oi_call), len(oi_put))
    strikes = strikes[:n]
    oi_call = oi_call[:n]
    oi_put = oi_put[:n]
    if n == 0:
        return None

    sigma = _extract_float(data, "iv", 0.25)
    r = _extract_float(data, "risk_free_rate", 0.045)
    tte_days = _extract_float(data, "tte_days", 1.0)
    tte_years = max(tte_days / 365.0, 1 / (365 * 24))

    net_gex_by_strike: List[Tuple[float, float]] = []
    total_abs_gex = 0.0
    near_abs_gex = 0.0
    charm_pressure = 0.0
    vanna_exposure = 0.0

    near_window = max(spot * 0.02, 1.0)
    for k, c_oi, p_oi in zip(strikes, oi_call, oi_put):
        gamma = GreekEngines.gamma(spot, k, tte_years, r, sigma)
        charm = GreekEngines.charm(spot, k, tte_years, r, sigma)
        vanna = GreekEngines.vanna(spot, k, tte_years, r, sigma)

        # Same directional formulation used in mm_hedge engines docs.
        net_gex = (-gamma * c_oi + gamma * p_oi) * (spot ** 2)
        net_gex_by_strike.append((k, net_gex))

        abs_gex = abs(net_gex)
        total_abs_gex += abs_gex
        if abs(k - spot) <= near_window:
            near_abs_gex += abs_gex

        charm_pressure += (-(charm * c_oi) - (charm * p_oi)) * 100
        vanna_exposure += (vanna * (c_oi - p_oi)) * 100

    net_gex_total = sum(v for _, v in net_gex_by_strike)
    regime = "long_gamma" if net_gex_total >= 0 else "short_gamma"
    concentration = near_abs_gex / total_abs_gex if total_abs_gex > 0 else 0.0

    sorted_points = sorted(net_gex_by_strike, key=lambda x: x[0])
    flip_strike = spot
    for i in range(len(sorted_points) - 1):
        left_k, left_gex = sorted_points[i]
        right_k, right_gex = sorted_points[i + 1]
        if left_gex < 0 <= right_gex:
            flip_strike = (left_k + right_k) / 2
            break
    flip_distance_pct = abs(spot - flip_strike) / spot if spot > 0 else 0.0

    return {
        "spot": spot,
        "net_gex_total": net_gex_total,
        "gamma_regime": regime,
        "gex_concentration": concentration,
        "charm_pressure": charm_pressure,
        "vanna_exposure": vanna_exposure,
        "flip_strike": flip_strike,
        "flip_distance_pct": flip_distance_pct,
    }


def _signal_gamma_regime(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    metrics = _compute_chain_metrics(data)
    if not metrics:
        return 0, 0.5, {"note": "Missing options chain metrics"}

    ret_5m = _extract_float(data, "ret_5m", 0.0)
    regime = metrics["gamma_regime"]
    if regime == "short_gamma":
        signal = 1 if ret_5m >= 0 else -1
    else:
        signal = -1 if ret_5m >= 0 else 1

    score = _clamp_score(0.45 + metrics["gex_concentration"] + min(0.35, abs(ret_5m) * 100))
    return signal, score, {
        "gamma_regime": regime,
        "net_gex_total": round(metrics["net_gex_total"], 2),
        "gex_concentration": round(metrics["gex_concentration"], 4),
        "ret_5m": ret_5m,
    }


def _signal_charm_flow(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    metrics = _compute_chain_metrics(data)
    if not metrics:
        return 0, 0.5, {"note": "Missing options chain metrics"}

    charm = metrics["charm_pressure"]
    # Positive charm pressure implies dealer sell pressure (bearish), negative is bullish.
    if charm > 0:
        signal = -1
    elif charm < 0:
        signal = 1
    else:
        signal = 0

    score = _clamp_score(0.4 + min(0.6, abs(charm) / 2e8))
    return signal, score, {"charm_pressure": round(charm, 2)}


def _signal_vanna_shock(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    metrics = _compute_chain_metrics(data)
    if not metrics:
        return 0, 0.5, {"note": "Missing options chain metrics"}

    iv_change = _extract_float(data, "iv_change", 0.0)
    vanna = metrics["vanna_exposure"]

    if iv_change > 0:
        signal = 1 if vanna > 0 else -1
    elif iv_change < 0:
        signal = -1 if vanna > 0 else 1
    else:
        signal = 0

    score = _clamp_score(0.45 + min(0.35, abs(iv_change) * 10) + min(0.2, abs(vanna) / 1e7))
    return signal, score, {
        "vanna_exposure": round(vanna, 2),
        "iv_change": iv_change,
    }


def _signal_gamma_flip(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    metrics = _compute_chain_metrics(data)
    if not metrics:
        return 0, 0.5, {"note": "Missing options chain metrics"}

    ret_5m = _extract_float(data, "ret_5m", 0.0)
    flip_distance = metrics["flip_distance_pct"]
    regime = metrics["gamma_regime"]

    if flip_distance <= 0.01:
        signal = 1 if (regime == "short_gamma" and ret_5m >= 0) else -1
    else:
        signal = 0

    score = _clamp_score(0.35 + max(0.0, 0.35 - flip_distance * 20) + min(0.3, abs(ret_5m) * 100))
    return signal, score, {
        "flip_strike": round(metrics["flip_strike"], 4),
        "flip_distance_pct": round(flip_distance, 6),
        "gamma_regime": regime,
    }


def _signal_dealer_composite(data: Dict[str, Any]) -> tuple[int, float, Dict[str, Any]]:
    gamma_sig, gamma_score, gamma_meta = _signal_gamma_regime(data)
    charm_sig, charm_score, charm_meta = _signal_charm_flow(data)
    vanna_sig, vanna_score, vanna_meta = _signal_vanna_shock(data)

    weighted = (
        gamma_sig * gamma_score * 0.4
        + charm_sig * charm_score * 0.3
        + vanna_sig * vanna_score * 0.3
    )

    if weighted > 0.15:
        signal = 1
    elif weighted < -0.15:
        signal = -1
    else:
        signal = 0

    confidence = _clamp_score(abs(weighted))
    return signal, confidence, {
        "weighted_bias": round(weighted, 6),
        "components": {
            "gamma": {"signal": gamma_sig, "score": gamma_score, **gamma_meta},
            "charm": {"signal": charm_sig, "score": charm_score, **charm_meta},
            "vanna": {"signal": vanna_sig, "score": vanna_score, **vanna_meta},
        },
    }


def execute_tool(tool_id: int, data: Dict[str, Any], symbol: str = "SPY") -> Optional[Dict[str, Any]]:
    """Execute a tool and return deterministic signal output."""
    if not (1 <= tool_id <= 52):
        return None

    tool_logic = {
        1: _signal_from_rsi,
        2: _signal_from_macd,
        3: _signal_from_volume,
        4: _signal_from_ma_cross,
        5: _signal_from_vwap,
        6: _signal_gamma_regime,
        7: _signal_charm_flow,
        8: _signal_vanna_shock,
        9: _signal_gamma_flip,
        10: _signal_dealer_composite,
    }

    logic = tool_logic.get(tool_id)
    if logic is None:
        signal = 0
        score = 0.5
        meta = {"note": "Tool logic not implemented yet; neutral output returned."}
    else:
        signal, score, meta = logic(data)

    context_meta = _build_context_meta(data)
    merged_meta = {**context_meta, **meta}

    return {
        "tool_id": tool_id,
        "tool_name": TOOL_NAMES.get(tool_id, f"Tool {tool_id}"),
        "signal": signal,
        "score": score,
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "metadata": merged_meta,
    }


def get_consensus(tool_ids: List[int], data: Dict[str, Any], symbol: str = "SPY") -> Dict[str, Any]:
    """Compute consensus across multiple tools."""
    results = []
    weighted_sum = 0.0
    total_weight = 0.0

    for tool_id in tool_ids:
        result = execute_tool(tool_id=tool_id, data=data, symbol=symbol)
        if result is None:
            continue
        results.append(result)
        weight = result["score"]
        weighted_sum += result["signal"] * weight
        total_weight += weight

    normalized = weighted_sum / total_weight if total_weight > 0 else 0.0
    if normalized > 0.2:
        consensus_signal = 1
    elif normalized < -0.2:
        consensus_signal = -1
    else:
        consensus_signal = 0

    return {
        "symbol": symbol,
        "tool_count": len(results),
        "consensus_signal": consensus_signal,
        "confidence": _clamp_score(abs(normalized)),
        "normalized_score": round(normalized, 6),
        "tools": results,
        "timestamp": datetime.now().isoformat(),
    }
