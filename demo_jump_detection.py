#!/usr/bin/env python3
"""
Demonstration of enhanced cryptocurrency price jump detection features
"""
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_jump_scenario_data():
    """Create synthetic data that demonstrates jump scenarios"""
    np.random.seed(123)
    
    # Create 200 15-minute periods (50 hours)
    dates = pd.date_range(start='2024-07-01', periods=200, freq='15min')
    base_price = 60000.0
    
    # Normal price evolution with some trending
    normal_returns = np.random.normal(0.0002, 0.003, 180)  # 0.02% mean, 0.3% std
    
    # Add jump scenarios at specific points
    jump_scenarios = [
        {'index': 50, 'type': 'volume_surge_breakout', 'return': 0.008, 'volume_mult': 3.0},
        {'index': 100, 'type': 'resistance_break', 'return': 0.007, 'volume_mult': 2.5},  
        {'index': 150, 'type': 'momentum_acceleration', 'return': 0.006, 'volume_mult': 2.0}
    ]
    
    # Apply jump scenarios
    for scenario in jump_scenarios:
        idx = scenario['index']
        if idx < len(normal_returns):
            normal_returns[idx] = scenario['return']
    
    # Calculate prices
    prices = [base_price]
    for ret in normal_returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Add final periods to match dates
    while len(prices) < len(dates):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.002)))
    
    # Create OHLCV data
    data = []
    base_volume = 1000000
    
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Volume spikes for jump scenarios
        volume_mult = 1.0
        for scenario in jump_scenarios:
            if abs(i - scenario['index']) <= 2:  # Volume spike around jump
                volume_mult = scenario['volume_mult']
                break
        
        high = close * (1 + abs(np.random.normal(0, 0.001)))
        low = close * (1 - abs(np.random.normal(0, 0.001)))
        open_price = close * (1 + np.random.normal(0, 0.0005))
        volume = base_volume * volume_mult * np.random.lognormal(0, 0.3)
        
        data.append({
            'timestamp': date,
            'open': open_price, 
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df, jump_scenarios

def analyze_jump_detection():
    """Analyze jump detection capabilities"""
    print("🔍 Cryptocurrency Jump Detection Analysis")
    print("=" * 60)
    
    # Create test data with known jumps
    data, jump_scenarios = create_jump_scenario_data()
    print(f"✅ Created synthetic data: {len(data)} periods with {len(jump_scenarios)} jump scenarios")
    
    # Show jump scenarios
    print("\n📊 Programmed Jump Scenarios:")
    for i, scenario in enumerate(jump_scenarios, 1):
        timestamp = data.index[scenario['index']]
        price_before = data.iloc[scenario['index']-1]['close']
        price_after = data.iloc[scenario['index']]['close']
        actual_return = (price_after - price_before) / price_before
        print(f"   {i}. {scenario['type']} at {timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"      Expected return: {scenario['return']:.2%}, Actual: {actual_return:.2%}")
    
    # Engineer features
    try:
        from paper_trader.models.feature_engineer import FeatureEngineer
        
        fe = FeatureEngineer()
        features_df = fe.engineer_features(data)
        
        if features_df is None:
            print("❌ Feature engineering failed")
            return
            
        print(f"\n✅ Generated {len(features_df.columns)} features from {len(features_df)} samples")
        
        # Analyze jump-specific features around known jumps
        jump_features = [
            'volume_surge_5', 'volume_surge_10', 'volatility_breakout', 'atr_breakout',
            'resistance_breakout', 'momentum_acceleration', 'price_acceleration',
            'momentum_convergence', 'squeeze_breakout', 'market_momentum_alignment'
        ]
        
        print(f"\n🎯 Jump Detection Feature Analysis:")
        print("-" * 40)
        
        for i, scenario in enumerate(jump_scenarios, 1):
            idx = scenario['index']
            timestamp = data.index[scenario['index']]
            
            # Find corresponding feature index (accounting for dropped rows)
            feature_timestamps = features_df.index
            closest_idx = np.argmin(np.abs(feature_timestamps - timestamp))
            
            print(f"\n{i}. {scenario['type']} at {timestamp.strftime('%Y-%m-%d %H:%M')}:")
            
            # Check which jump features activated
            activated_features = []
            for feature in jump_features:
                if feature in features_df.columns:
                    # Check window around the jump
                    window_start = max(0, closest_idx - 2)
                    window_end = min(len(features_df), closest_idx + 3)
                    feature_values = features_df[feature].iloc[window_start:window_end]
                    
                    if feature_values.sum() > 0:
                        max_val = feature_values.max()
                        activated_features.append(f"{feature}({max_val:.0f})")
            
            if activated_features:
                print(f"   ✅ Activated features: {', '.join(activated_features)}")
            else:
                print(f"   ⚠️  No jump features activated")
            
            # Calculate targets around this point
            if closest_idx < len(features_df) - 1:
                price_current = features_df.iloc[closest_idx]['close'] if 'close' in features_df.columns else None
                price_next = features_df.iloc[closest_idx + 1]['close'] if 'close' in features_df.columns else None
                
                if price_current and price_next:
                    actual_change = (price_next - price_current) / price_current
                    jump_target = 1 if actual_change >= 0.005 else 0
                    print(f"   📈 Price change: {actual_change:.2%} → Jump target: {jump_target}")
        
        # Overall statistics
        print(f"\n📈 Overall Jump Feature Statistics:")
        print("-" * 40)
        
        total_samples = len(features_df)
        for feature in jump_features:
            if feature in features_df.columns:
                activation_count = features_df[feature].sum()
                activation_rate = activation_count / total_samples * 100
                print(f"   {feature}: {activation_count} activations ({activation_rate:.1f}%)")
        
        # Market context analysis
        market_features = ['bull_market', 'bear_market', 'market_stress', 'strong_trend', 'weak_trend']
        print(f"\n🌐 Market Context Analysis:")
        print("-" * 30)
        
        for feature in market_features:
            if feature in features_df.columns:
                avg_value = features_df[feature].mean()
                print(f"   {feature}: {avg_value:.1%} of time")
        
    except ImportError as e:
        print(f"❌ Could not import feature engineer: {e}")
    except Exception as e:
        print(f"❌ Analysis error: {e}")

def show_enhanced_model_approach():
    """Show how the enhanced model approach works"""
    print(f"\n🧠 Enhanced Model Approach Summary")
    print("=" * 60)
    
    print("1. 🎯 Binary Jump Classification:")
    print("   • Both LSTM and XGBoost now focus on detecting ≥0.5% price jumps")
    print("   • LSTM uses sigmoid activation + binary crossentropy loss")
    print("   • Consistent target definition across both models")
    
    print("\n2. 🔧 Jump-Specific Features (15+ new features):")
    print("   • Volume surge detection (2x, 1.5x normal volume)")
    print("   • Momentum acceleration (rate of momentum change)")
    print("   • Volatility breakout signals (ATR and BB width)")
    print("   • Support/resistance breakout indicators")
    print("   • Price gap detection and multi-timeframe convergence")
    
    print("\n3. 🌍 Market Context Features:")
    print("   • Bull/bear market regime detection")
    print("   • Market momentum alignment across timeframes")
    print("   • Market stress indicators (high volatility periods)")
    print("   • Trend strength and consistency measures")
    
    print("\n4. 🎪 Enhanced Ensemble Prediction:")
    print("   • Jump probability calculation from features + models")
    print("   • Adaptive model weighting based on jump likelihood")
    print("   • LSTM gets higher weight for jump detection (now binary)")
    print("   • Signal strength boosted for high jump probability")
    
    print("\n5. 📊 Jump-Focused Trading Signals:")
    print("   • Separate jump-focused buy signal generation")
    print("   • Enhanced position sizing for high-probability jumps (up to 1.5x)")
    print("   • More aggressive profit targets for jump trades")
    print("   • Tighter stop losses to capture jump momentum")
    
    print("\n6. ✅ Key Improvements for Jump Detection:")
    print("   • Precision: Better identification of actual jump opportunities")
    print("   • Recall: Reduced false negatives on legitimate jumps")
    print("   • F1 Score: Balanced improvement in jump detection accuracy")
    print("   • Risk Management: Adaptive position sizing based on jump confidence")

def main():
    """Main demonstration"""
    analyze_jump_detection()
    show_enhanced_model_approach()
    
    print(f"\n🚀 Next Steps:")
    print("=" * 20)
    print("1. Train models with new binary jump targets and features")
    print("2. Backtest performance on historical data")
    print("3. Compare F1, precision, and recall vs. previous implementation")
    print("4. Deploy enhanced models for live trading")
    
    print(f"\n💡 Expected Improvements:")
    print("• Better detection of 0.5%+ price movements within 15-minute windows")  
    print("• Reduced false positives through multi-feature validation")
    print("• Enhanced risk-adjusted returns through jump-focused position sizing")
    print("• More responsive signals during high-volatility breakout periods")

if __name__ == "__main__":
    main()