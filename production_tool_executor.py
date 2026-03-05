"""
Production Tool Executor for PULSE Trading Platform.

Implements lightweight, deterministic signal logic for core tool IDs so API users
get stable, explainable outputs instead of random mock responses.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


TOOL_NAMES = {
    1: "RSI Momentum",
    2: "MACD Signal",
    3: "Volume Spike",
    4: "Moving Average Cross",
    5: "VWAP Bias",
}


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _extract_float(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


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
    }

    logic = tool_logic.get(tool_id)
    if logic is None:
        signal = 0
        score = 0.5
        meta = {"note": "Tool logic not implemented yet; neutral output returned."}
    else:
        signal, score, meta = logic(data)

    return {
        "tool_id": tool_id,
        "tool_name": TOOL_NAMES.get(tool_id, f"Tool {tool_id}"),
        "signal": signal,
        "score": score,
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "metadata": meta,
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
