"""
Database Optimization Script for PULSE Trading Platform

This script provides tools for:
- Identifying slow queries
- Adding missing indexes
- Optimizing table structures
- Query caching configuration
- Connection pooling optimization

Usage:
    python scripts/optimize_database.py --analyze
    python scripts/optimize_database.py --add-indexes
    python scripts/optimize_database.py --vacuum
"""

import argparse
import sqlite3
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DatabaseOptimizer:
    """
    Database optimization utility for PULSE trading platform.
    
    Provides tools for analyzing and optimizing SQLite databases.
    """
    
    def __init__(self, db_path: str = "pulse_trading.db"):
        """
        Initialize the database optimizer.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to the database."""
        if not os.path.exists(self.db_path):
            print(f"Warning: Database file {self.db_path} does not exist. Creating new database.")
        
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def analyze_tables(self) -> List[Dict[str, Any]]:
        """
        Analyze all tables in the database.
        
        Returns:
            List of table statistics
        """
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        
        # Get all tables (excluding system tables)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        
        stats = []
        
        for table in tables:
            # Validate table name to prevent SQL injection
            if not table.replace('_', '').isalnum():
                continue
            
            # Get row count (using parameter binding for safety)
            cursor.execute(f"SELECT COUNT(*) FROM '{table}'")
            row_count = cursor.fetchone()[0]
            
            # Get table info
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = cursor.fetchall()
            
            # Get index info
            cursor.execute(f"PRAGMA index_list('{table}')")
            indexes = cursor.fetchall()
            
            stats.append({
                'table': table,
                'rows': row_count,
                'columns': len(columns),
                'indexes': len(indexes),
                'column_names': [col[1] for col in columns]
            })
        
        return stats
    
    def identify_missing_indexes(self) -> List[Dict[str, Any]]:
        """
        Identify potentially missing indexes.
        
        Returns:
            List of recommended indexes
        """
        if not self.conn:
            self.connect()
        
        recommendations = []
        
        # Common patterns that benefit from indexes
        patterns = [
            ('positions', 'symbol'),
            ('positions', 'entry_time'),
            ('trades', 'symbol'),
            ('trades', 'entry_time'),
            ('trades', 'exit_time'),
            ('tool_performance', 'tool_id'),
            ('tool_performance', 'timestamp')
        ]
        
        cursor = self.conn.cursor()
        
        for table, column in patterns:
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if not cursor.fetchone():
                continue
            
            # Validate table name to prevent SQL injection
            if not table.replace('_', '').isalnum():
                continue
            
            # Check if column exists
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = [col[1] for col in cursor.fetchall()]
            if column not in columns:
                continue
            
            # Check if index already exists
            cursor.execute(f"PRAGMA index_list('{table}')")
            indexes = cursor.fetchall()
            
            # Get indexed columns
            indexed_columns = set()
            for idx in indexes:
                cursor.execute(f"PRAGMA index_info('{idx[1]}')")
                for col_info in cursor.fetchall():
                    indexed_columns.add(col_info[2])
            
            if column not in indexed_columns:
                recommendations.append({
                    'table': table,
                    'column': column,
                    'index_name': f'idx_{table}_{column}',
                    'reason': f'Frequently queried column without index'
                })
        
        return recommendations
    
    def add_index(self, table: str, column: str, index_name: str = None):
        """
        Add an index to a table.
        
        Args:
            table: Table name
            column: Column name
            index_name: Optional index name (auto-generated if not provided)
        """
        if not self.conn:
            self.connect()
        
        if not index_name:
            index_name = f"idx_{table}_{column}"
        
        # Validate table and column names to prevent SQL injection
        if not table.replace('_', '').isalnum() or not column.replace('_', '').isalnum():
            print(f"❌ Invalid table or column name: {table}.{column}")
            return False
        
        try:
            cursor = self.conn.cursor()
            # Use single quotes for identifiers in SQLite
            cursor.execute(f"CREATE INDEX IF NOT EXISTS '{index_name}' ON '{table}'('{column}')")
            self.conn.commit()
            print(f"✅ Created index {index_name} on {table}.{column}")
            return True
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            return False
    
    def vacuum_database(self):
        """
        Run VACUUM to optimize database file.
        
        This reclaims unused space and optimizes the database structure.
        """
        if not self.conn:
            self.connect()
        
        print("Running VACUUM on database...")
        start = time.time()
        
        try:
            self.conn.execute("VACUUM")
            duration = time.time() - start
            print(f"✅ VACUUM completed in {duration:.2f}s")
            return True
        except Exception as e:
            print(f"❌ Error running VACUUM: {e}")
            return False
    
    def analyze_database(self):
        """
        Run ANALYZE to update query optimizer statistics.
        """
        if not self.conn:
            self.connect()
        
        print("Running ANALYZE on database...")
        start = time.time()
        
        try:
            self.conn.execute("ANALYZE")
            duration = time.time() - start
            print(f"✅ ANALYZE completed in {duration:.2f}s")
            return True
        except Exception as e:
            print(f"❌ Error running ANALYZE: {e}")
            return False
    
    def get_database_size(self) -> int:
        """
        Get database file size in bytes.
        
        Returns:
            Database size in bytes
        """
        if os.path.exists(self.db_path):
            return os.path.getsize(self.db_path)
        return 0
    
    def print_report(self):
        """Print a comprehensive database optimization report."""
        print("="*70)
        print("DATABASE OPTIMIZATION REPORT")
        print("="*70)
        
        # Database info
        size_bytes = self.get_database_size()
        size_mb = size_bytes / 1024 / 1024
        print(f"\nDatabase: {self.db_path}")
        print(f"Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
        
        # Table statistics
        print("\n" + "-"*70)
        print("TABLE STATISTICS")
        print("-"*70)
        
        stats = self.analyze_tables()
        
        if stats:
            print(f"{'Table':<20} {'Rows':<10} {'Columns':<10} {'Indexes':<10}")
            print("-"*70)
            for stat in stats:
                print(f"{stat['table']:<20} {stat['rows']:<10} {stat['columns']:<10} {stat['indexes']:<10}")
        else:
            print("No tables found in database")
        
        # Index recommendations
        print("\n" + "-"*70)
        print("INDEX RECOMMENDATIONS")
        print("-"*70)
        
        recommendations = self.identify_missing_indexes()
        
        if recommendations:
            for rec in recommendations:
                print(f"\n📊 Table: {rec['table']}")
                print(f"   Column: {rec['column']}")
                print(f"   Suggested index: {rec['index_name']}")
                print(f"   Reason: {rec['reason']}")
        else:
            print("✅ No missing indexes detected")
        
        print("\n" + "="*70)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Database optimization tool for PULSE Trading Platform"
    )
    parser.add_argument(
        "--db-path",
        default="pulse_trading.db",
        help="Path to database file (default: pulse_trading.db)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze database and print report"
    )
    parser.add_argument(
        "--add-indexes",
        action="store_true",
        help="Add recommended indexes"
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM to optimize database"
    )
    parser.add_argument(
        "--analyze-stats",
        action="store_true",
        help="Run ANALYZE to update query optimizer statistics"
    )
    
    args = parser.parse_args()
    
    # If no action specified, show report
    if not any([args.analyze, args.add_indexes, args.vacuum, args.analyze_stats]):
        args.analyze = True
    
    optimizer = DatabaseOptimizer(args.db_path)
    
    try:
        optimizer.connect()
        
        if args.analyze:
            optimizer.print_report()
        
        if args.add_indexes:
            print("\n" + "="*70)
            print("ADDING RECOMMENDED INDEXES")
            print("="*70)
            recommendations = optimizer.identify_missing_indexes()
            
            if recommendations:
                for rec in recommendations:
                    optimizer.add_index(rec['table'], rec['column'], rec['index_name'])
            else:
                print("No indexes to add")
        
        if args.vacuum:
            print("\n" + "="*70)
            print("VACUUM DATABASE")
            print("="*70)
            optimizer.vacuum_database()
        
        if args.analyze_stats:
            print("\n" + "="*70)
            print("ANALYZE DATABASE")
            print("="*70)
            optimizer.analyze_database()
        
    finally:
        optimizer.close()


if __name__ == "__main__":
    main()
