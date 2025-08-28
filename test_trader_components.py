#!/usr/bin/env python3
"""
Comprehensive Trader Component Test Suite
Runs all isolated component tests and provides summary report
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_test_script(script_name):
    """Run a test script and capture results"""
    script_path = project_root / script_name
    
    if not script_path.exists():
        return {
            'success': False,
            'error': f"Test script {script_name} not found",
            'output': '',
            'duration': 0
        }
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Print output in real-time style
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        success = result.returncode == 0
        
        return {
            'success': success,
            'error': result.stderr if result.stderr else None,
            'output': result.stdout,
            'duration': duration,
            'return_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            'success': False,
            'error': f"Test timed out after {duration:.1f} seconds",
            'output': '',
            'duration': duration
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'error': str(e),
            'output': '',
            'duration': duration
        }

def generate_test_report(test_results):
    """Generate a comprehensive test report"""
    report = []
    report.append("\n" + "="*80)
    report.append("TRADER COMPONENT TEST REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result['success'])
    failed_tests = total_tests - passed_tests
    
    report.append("SUMMARY:")
    report.append(f"  Total Tests: {total_tests}")
    report.append(f"  Passed: {passed_tests}")
    report.append(f"  Failed: {failed_tests}")
    report.append(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    report.append("")
    
    # Individual test results
    report.append("INDIVIDUAL TEST RESULTS:")
    report.append("-" * 40)
    
    for test_name, result in test_results.items():
        status = "PASS" if result['success'] else "FAIL"
        duration = result['duration']
        
        report.append(f"{test_name:<30} {status:<6} ({duration:.2f}s)")
        
        if not result['success'] and result['error']:
            # Show first few lines of error
            error_lines = result['error'].split('\n')[:3]
            for line in error_lines:
                if line.strip():
                    report.append(f"    ERROR: {line.strip()}")
    
    report.append("")
    
    # Detailed analysis
    report.append("DETAILED ANALYSIS:")
    report.append("-" * 40)
    
    if failed_tests > 0:
        report.append("\nFAILED TESTS:")
        for test_name, result in test_results.items():
            if not result['success']:
                report.append(f"\n{test_name}:")
                report.append(f"  Duration: {result['duration']:.2f}s")
                report.append(f"  Return Code: {result.get('return_code', 'N/A')}")
                if result['error']:
                    report.append(f"  Error: {result['error'][:200]}...")
    
    # Recommendations
    report.append("\nRECOMMENDations:")
    if failed_tests == 0:
        report.append("  All tests passed! System components are functioning correctly.")
    else:
        report.append(f"  {failed_tests} component(s) need attention:")
        
        for test_name, result in test_results.items():
            if not result['success']:
                if 'config' in test_name.lower():
                    report.append("    - Fix configuration loading issues")
                elif 'feature' in test_name.lower():
                    report.append("    - Resolve feature alignment problems")
                elif 'model' in test_name.lower():
                    report.append("    - Address model loading failures")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

def main():
    """Main test runner"""
    print("TRADER COMPONENT DIAGNOSTIC SUITE")
    print("=" * 50)
    print(f"Starting comprehensive component testing...")
    print(f"Project root: {project_root}")
    
    # Define test scripts to run
    test_scripts = [
        'test_config_loader.py',
        'test_feature_engine.py', 
        'test_model_loading.py'
    ]
    
    # Run all tests
    test_results = {}
    
    for script in test_scripts:
        test_name = script.replace('.py', '').replace('test_', '')
        result = run_test_script(script)
        test_results[test_name] = result
    
    # Generate and display report
    report = generate_test_report(test_results)
    print(report)
    
    # Save report to file
    report_file = project_root / 'test_trader_components_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Return overall success
    overall_success = all(result['success'] for result in test_results.values())
    
    if overall_success:
        print("\n✅ ALL TESTS PASSED - System components are healthy")
    else:
        failed_count = sum(1 for result in test_results.values() if not result['success'])
        print(f"\n❌ {failed_count} TEST(S) FAILED - System needs attention")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)