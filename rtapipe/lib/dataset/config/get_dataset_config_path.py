import os
from pathlib import Path

def main():
    return Path(__file__).parent.resolve().joinpath(f"agilehost3-prod5.yml")
    
if __name__=='__main__':
    main()