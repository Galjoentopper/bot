#!/usr/bin/env python3
"""
Price Sync Monitoring Script

Run this script periodically to monitor price synchronization health.
It checks for common issues and provides actionable insights.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add paper_trader to path
sys.path.append(str(Path(__file__).parent))

from paper_trader.config.settings import TradingSettings
from paper_trader.data.bitvavo_collector import BitvavoDataCollector

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def monitor_price_sync():
    """Monitor price synchronization health."""
    
    print(f"🔍 Price Sync Health Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test symbols - common crypto pairs
    test_symbols = ["BTC-EUR", "ETH-EUR", "ADA-EUR"]
    
    try:
        settings = TradingSettings()
        collector = BitvavoDataCollector(
            api_key="dummy",
            api_secret="dummy", 
            interval="15m",
            settings=settings
        )
        
        issues_found = 0
        
        for symbol in test_symbols:
            print(f"\n📊 Checking {symbol}:")
            
            try:
                # Test both price methods
                current_price = await collector.get_current_price(symbol)
                trading_price = await collector.get_current_price_for_trading(symbol)
                
                if current_price and trading_price:
                    diff_pct = abs(current_price - trading_price) / current_price * 100
                    
                    if diff_pct > 0.1:  # More than 0.1% difference
                        print(f"   ⚠️ Price discrepancy: {diff_pct:.3f}%")
                        issues_found += 1
                    else:
                        print(f"   ✅ Prices synchronized (diff: {diff_pct:.3f}%)")
                
                else:
                    print(f"   ❌ Failed to fetch prices")
                    issues_found += 1
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                issues_found += 1
        
        # Overall health assessment
        print(f"\n📋 Health Summary:")
        if issues_found == 0:
            print(f"   🎉 All systems healthy - no price sync issues detected")
        elif issues_found <= len(test_symbols) // 2:
            print(f"   ⚠️ Minor issues detected ({issues_found}/{len(test_symbols)} symbols)")
            print(f"   💡 Consider monitoring logs for price validation warnings")
        else:
            print(f"   🚨 Multiple issues detected ({issues_found}/{len(test_symbols)} symbols)")
            print(f"   🔧 Review network connectivity and API rate limits")
        
        return issues_found
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return -1

def print_monitoring_tips():
    """Print tips for ongoing monitoring."""
    
    print(f"\n💡 Monitoring Tips:")
    print("=" * 40)
    print("1. Run this script every 30 minutes during trading hours")
    print("2. Check logs for 'Price discrepancy' warnings")
    print("3. Monitor trade execution rates in bot logs")
    print("4. Watch for 'unrealistic price change' rejections")
    print("5. Set up alerts for multiple consecutive failures")
    
    print(f"\n🔧 Troubleshooting:")
    print("- High discrepancies: Check network latency")
    print("- API failures: Verify rate limits and connectivity")  
    print("- Validation rejections: Review threshold settings")
    print("- Buffer staleness: Check websocket connection")

async def main():
    """Run price sync monitoring."""
    
    issues = await monitor_price_sync()
    print_monitoring_tips()
    
    # Exit codes for automation
    if issues == 0:
        print(f"\n✅ Exit code: 0 (healthy)")
        sys.exit(0)
    elif issues > 0:
        print(f"\n⚠️ Exit code: 1 (issues detected)")
        sys.exit(1)
    else:
        print(f"\n❌ Exit code: 2 (check failed)")
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())