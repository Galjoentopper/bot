#!/usr/bin/env python3
"""
Demonstration script to show the improved high-frequency optimization can generate viable parameters
"""

import sys
from optimized_variables import ScientificOptimizer
import random

def demo_parameter_generation():
    """Demonstrate that we can now generate sensible high-frequency parameters"""
    print("🚀 HIGH-FREQUENCY TRADING PARAMETER GENERATION DEMO")
    print("=" * 70)
    
    # Initialize optimizer for high-frequency trading
    optimizer = ScientificOptimizer(
        symbols=['BTCEUR'], 
        optimization_mode='high_frequency',
        objective='profit_factor',
        verbose=True
    )
    
    print("\n🎯 Generating 5 sample parameter sets for high-frequency trading:")
    print("-" * 70)
    
    for i in range(5):
        params = optimizer._sample_random_params()
        
        # Calculate key metrics
        spread = params['buy_threshold'] - params['sell_threshold']
        tp_sl_ratio = params['take_profit_pct'] / params['stop_loss_pct']
        risk_return = params['take_profit_pct'] / params['risk_per_trade']
        
        print(f"\n📊 Parameter Set #{i+1}:")
        print(f"   Buy Threshold:     {params['buy_threshold']:.5f}")
        print(f"   Sell Threshold:    {params['sell_threshold']:.5f}")
        print(f"   Spread:            {spread:.5f} (smaller = more sensitive)")
        print(f"   LSTM Delta:        {params['lstm_delta_threshold']:.2e}")
        print(f"   Position Size:     {params['max_capital_per_trade']:.1%}")
        print(f"   Risk per Trade:    {params['risk_per_trade']:.1%}")
        print(f"   Stop Loss:         {params['stop_loss_pct']:.1%}")
        print(f"   Take Profit:       {params['take_profit_pct']:.1%}")
        print(f"   TP/SL Ratio:       {tp_sl_ratio:.2f}x")
        print(f"   Risk/Return:       {risk_return:.2f}")
        print(f"   Max Positions:     {params['max_positions']:.0f}")
        print(f"   Max Trades/Hour:   {params['max_trades_per_hour']:.0f}")
        
        # Quality assessment
        if spread < 0.01 and tp_sl_ratio > 1.0:
            print("   ✅ EXCELLENT: Very sensitive with positive expectancy")
        elif spread < 0.015 and tp_sl_ratio > 1.0:
            print("   ✅ GOOD: Sensitive with positive expectancy") 
        else:
            print("   ⚠️  OKAY: May need refinement")

def demo_objective_function():
    """Demonstrate the enhanced objective function"""
    print("\n\n🎯 ENHANCED OBJECTIVE FUNCTION DEMONSTRATION")
    print("=" * 70)
    
    optimizer = ScientificOptimizer(
        symbols=['BTCEUR'],
        optimization_mode='high_frequency', 
        objective='profit_factor',
        verbose=False
    )
    
    scenarios = [
        {
            "name": "Traditional Low-Frequency (OLD PROBLEM)",
            "metrics": {
                "total_trades": 16,
                "trades_per_day": 0.01,  # 16 trades over 5 years = ~0.01/day
                "win_rate": 0.0,         # 0% win rate (the original issue)
                "profit_factor": 0.0,
                "total_return": -0.074   # -7.4% (original -742.26 on 10k capital)
            }
        },
        {
            "name": "Improved High-Frequency (TARGET)",
            "metrics": {
                "total_trades": 150,
                "trades_per_day": 5.5,   # Target: 5+ trades per day
                "win_rate": 0.58,        # Improved win rate
                "profit_factor": 1.4,
                "total_return": 0.15     # 15% return
            }
        },
        {
            "name": "Optimal High-Frequency",
            "metrics": {
                "total_trades": 250,
                "trades_per_day": 8.2,   # High frequency achieved
                "win_rate": 0.62,        # Good win rate
                "profit_factor": 1.6,
                "total_return": 0.25     # 25% return
            }
        }
    ]
    
    print("Comparing scenarios with the new objective function:")
    print("-" * 70)
    
    for scenario in scenarios:
        metrics = scenario["metrics"]
        obj_value = optimizer._calculate_objective(metrics)
        
        print(f"\n📈 {scenario['name']}:")
        print(f"   Total Trades:      {metrics['total_trades']}")
        print(f"   Trades per Day:    {metrics['trades_per_day']:.2f}")
        print(f"   Win Rate:          {metrics['win_rate']:.1%}")
        print(f"   Profit Factor:     {metrics['profit_factor']:.1f}")
        print(f"   Total Return:      {metrics['total_return']:+.1%}")
        print(f"   🎯 Objective Score: {obj_value:.2f}")
        
        if obj_value <= -990:
            print("   ❌ REJECTED: Insufficient trades or poor performance")
        elif obj_value < 1:
            print("   ⚠️  POOR: Below average performance")
        elif obj_value < 2:
            print("   ✅ GOOD: Above average performance")
        else:
            print("   🚀 EXCELLENT: High-performance configuration")

def main():
    """Run the demonstration"""
    try:
        demo_parameter_generation()
        demo_objective_function()
        
        print("\n" + "=" * 70)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("\n✅ Key Achievements:")
        print("   • Fixed restrictive parameter ranges that caused 0 trades")
        print("   • Implemented explicit 5+ trades/day optimization target")
        print("   • Added robust parameter validation (take-profit > stop-loss)")
        print("   • Enhanced objective function rewards trade frequency")
        print("   • Immediate adaptation when zero trades detected")
        print("\n🚀 The system is now optimized for high-frequency trading!")
        print("   Use: python optimized_variables.py --symbols BTCEUR --mode high_frequency")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())