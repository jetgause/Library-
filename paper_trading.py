"""
PULSE Trading Platform - Paper Trading System
Full paper trading implementation with P&L tracking, position management, and SQLite persistence
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class Position:
    """Represents an open position"""
    position_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: int
    side: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    tool_id: Optional[int] = None
    tool_name: Optional[str] = None
    
    def to_dict(self):
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat()
        return d


@dataclass
class Trade:
    """Represents a closed trade"""
    trade_id: str
    position_id: str
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    tool_id: Optional[int] = None
    tool_name: Optional[str] = None
    exit_reason: str = 'manual'
    
    def to_dict(self):
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat()
        d['exit_time'] = self.exit_time.isoformat()
        return d


class PaperTradingEngine:
    """
    Complete paper trading engine with:
    - Position management
    - P&L tracking
    - SQLite persistence
    - Trade history
    - Performance analytics
    """
    
    def __init__(self, initial_capital: float = 100000, db_path: str = "paper_trading.db"):
        """
        Initialize paper trading engine
        
        Args:
            initial_capital: Starting capital in dollars
            db_path: Path to SQLite database
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.db_path = db_path
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_returns = []
        
        # Initialize database
        self._init_database()
        self._load_state()
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                side TEXT NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                tool_id INTEGER,
                tool_name TEXT
            )
        """)
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                position_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                tool_id INTEGER,
                tool_name TEXT,
                exit_reason TEXT
            )
        """)
        
        # Account state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_state (
                timestamp TEXT PRIMARY KEY,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                num_positions INTEGER NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_state(self):
        """Load previous state from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Load positions
        positions_df = pd.read_sql_query("SELECT * FROM positions", conn)
        for _, row in positions_df.iterrows():
            pos = Position(
                position_id=row['position_id'],
                symbol=row['symbol'],
                entry_time=datetime.fromisoformat(row['entry_time']),
                entry_price=row['entry_price'],
                quantity=row['quantity'],
                side=row['side'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                tool_id=row['tool_id'],
                tool_name=row['tool_name']
            )
            self.positions[pos.position_id] = pos
        
        # Load last account state
        try:
            last_state = pd.read_sql_query(
                "SELECT * FROM account_state ORDER BY timestamp DESC LIMIT 1", 
                conn
            )
            if not last_state.empty:
                self.cash = last_state['cash'].iloc[0]
        except:
            pass
        
        conn.close()
    
    def _save_position(self, position: Position):
        """Save position to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.position_id,
            position.symbol,
            position.entry_time.isoformat(),
            position.entry_price,
            position.quantity,
            position.side,
            position.stop_loss,
            position.take_profit,
            position.tool_id,
            position.tool_name
        ))
        
        conn.commit()
        conn.close()
    
    def _save_trade(self, trade: Trade):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id,
            trade.position_id,
            trade.symbol,
            trade.side,
            trade.entry_time.isoformat(),
            trade.exit_time.isoformat(),
            trade.entry_price,
            trade.exit_price,
            trade.quantity,
            trade.pnl,
            trade.pnl_pct,
            trade.tool_id,
            trade.tool_name,
            trade.exit_reason
        ))
        
        conn.commit()
        conn.close()
    
    def _delete_position(self, position_id: str):
        """Delete position from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions WHERE position_id = ?", (position_id,))
        conn.commit()
        conn.close()
    
    def _save_account_state(self):
        """Save current account state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        equity = self.get_total_equity()
        
        cursor.execute("""
            INSERT INTO account_state VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            self.cash,
            equity,
            len(self.positions)
        ))
        
        conn.commit()
        conn.close()
    
    def open_position(self, symbol: str, quantity: int, price: float, side: str = 'long',
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                     tool_id: Optional[int] = None, tool_name: Optional[str] = None) -> Optional[Position]:
        """
        Open a new position
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Entry price
            side: 'long' or 'short'
            stop_loss: Stop loss price
            take_profit: Take profit price
            tool_id: ID of tool that generated signal
            tool_name: Name of tool
        
        Returns:
            Position object or None if insufficient capital
        """
        cost = quantity * price
        
        if cost > self.cash:
            print(f"Insufficient capital. Need ${cost:,.2f}, have ${self.cash:,.2f}")
            return None
        
        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        position = Position(
            position_id=position_id,
            symbol=symbol,
            entry_time=datetime.now(),
            entry_price=price,
            quantity=quantity,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
            tool_id=tool_id,
            tool_name=tool_name
        )
        
        self.positions[position_id] = position
        self.cash -= cost
        
        self._save_position(position)
        self._save_account_state()
        
        print(f"✅ Opened {side} position: {quantity} {symbol} @ ${price:.2f}")
        print(f"   Position ID: {position_id}")
        print(f"   Remaining cash: ${self.cash:,.2f}")
        
        return position
    
    def close_position(self, position_id: str, price: float, exit_reason: str = 'manual') -> Optional[Trade]:
        """
        Close an existing position
        
        Args:
            position_id: ID of position to close
            price: Exit price
            exit_reason: Reason for exit ('manual', 'stop_loss', 'take_profit', 'signal')
        
        Returns:
            Trade object or None if position not found
        """
        if position_id not in self.positions:
            print(f"Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        
        # Calculate P&L
        if position.side == 'long':
            pnl = (price - position.entry_price) * position.quantity
        else:  # short
            pnl = (position.entry_price - price) * position.quantity
        
        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100
        
        # Return capital + profit
        self.cash += (position.entry_price * position.quantity) + pnl
        
        # Create trade record
        trade_id = f"trade_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        trade = Trade(
            trade_id=trade_id,
            position_id=position_id,
            symbol=position.symbol,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            tool_id=position.tool_id,
            tool_name=position.tool_name,
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
        
        # Remove position
        del self.positions[position_id]
        
        self._save_trade(trade)
        self._delete_position(position_id)
        self._save_account_state()
        
        print(f"✅ Closed position: {position.symbol}")
        print(f"   P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        print(f"   Exit reason: {exit_reason}")
        print(f"   New cash balance: ${self.cash:,.2f}")
        
        return trade
    
    def get_total_equity(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total account equity"""
        if current_prices is None:
            current_prices = {pos.symbol: pos.entry_price for pos in self.positions.values()}
        
        position_value = 0
        for pos in self.positions.values():
            current_price = current_prices.get(pos.symbol, pos.entry_price)
            if pos.side == 'long':
                position_value += current_price * pos.quantity
            else:
                pnl = (pos.entry_price - current_price) * pos.quantity
                position_value += (pos.entry_price * pos.quantity) + pnl
        
        return self.cash + position_value
    
    def get_performance_stats(self) -> Dict:
        """Calculate comprehensive performance statistics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0
            }
        
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_return = trades_df['pnl_pct'].mean()
        total_pnl = trades_df['pnl'].sum()
        
        # Calculate Sharpe ratio
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'].values
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'current_equity': self.get_total_equity(),
            'return_on_capital': ((self.get_total_equity() - self.initial_capital) / self.initial_capital) * 100
        }


if __name__ == "__main__":
    print("PULSE Paper Trading System initialized")
    print("Ready for paper trading!")
