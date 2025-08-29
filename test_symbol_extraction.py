import yaml
import sys

def extract_symbols(config_file):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        symbols = config.get('data_acquisition', {}).get('symbols', [])
        if not symbols:
            symbols = config.get('data', {}).get('symbols', [])
        if not symbols:
            symbols = config.get('symbols', [])
        
        if symbols:
            print('SYMBOLS_FOUND:' + ','.join(symbols))
            return 0
        else:
            print('NO_SYMBOLS_FOUND')
            return 1
    except Exception as e:
        print(f'ERROR: {e}')
        return 1

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python test_symbol_extraction.py <config_file>')
        sys.exit(1)
    
    config_file = sys.argv[1]
    sys.exit(extract_symbols(config_file))