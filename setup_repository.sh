#!/bin/bash
# Setup script for MSLSTM Trading Simulation Repository

echo "🚀 Setting up MSLSTM Trading Simulation Repository..."

# Initialize git repository
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git branch -M main
else
    echo " Git repository already initialized"
fi

# Create required directories
echo " Creating directory structure..."
mkdir -p trading_results
mkdir -p logs
mkdir -p results
mkdir -p models
mkdir -p data

# Set up virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    echo " Virtual environment created. Activate with: source venv/bin/activate"
else
    echo " Virtual environment already exists"
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements_sota.txt" ]; then
    pip install -r requirements_sota.txt
    pip install openpyxl xlsxwriter
elif [ -f "requirements_phase1.txt" ]; then
    pip install -r requirements_phase1.txt
    pip install openpyxl xlsxwriter
else
    echo "⚠️  No requirements file found. Installing basic dependencies..."
    pip install pandas numpy yfinance scikit-learn tensorflow matplotlib seaborn plotly openpyxl xlsxwriter
fi

# Test basic functionality
echo "🧪 Testing basic functionality..."
python -c "
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
print(' All basic imports working')

# Test yfinance
try:
    data = yf.download('TCS.NS', period='5d', progress=False)
    print(f' Yahoo Finance connection working - Got {len(data)} days of TCS data')
except Exception as e:
    print(f' Yahoo Finance test failed: {e}')

print('🎯 Basic setup complete!')
"

# Initial git commit
echo "💾 Creating initial commit..."
git add .
git commit -m "🎉 Initial commit: MSLSTM Trading Simulation setup"

# Display next steps
echo ""
echo "🎉 Setup Complete!"
echo "="*50
echo ""
echo "📋 Next Steps:"
echo "1. Train your models (if not already done):"
echo "   python MSLSTM_paper_implementation.py --ticker TCS.NS --model ensemble"
echo ""
echo "2. Test single stock simulation:"
echo "   python real_trading_simulator.py --ticker TCS.NS --days 1"
echo ""
echo "3. Run portfolio simulation:"
echo "   python real_trading_simulator.py --portfolio --days 1"
echo ""
echo "4. Set up GitHub repository:"
echo "   git remote add origin <your-github-repo-url>"
echo "   git push -u origin main"
echo ""
echo "5. Enable GitHub Actions:"
echo "   - Go to your GitHub repository"
echo "   - Go to Actions tab"
echo "   - Enable workflows"
echo ""
echo "🔧 Configuration:"
echo "- Daily automation: 6:30 PM IST (1:00 PM UTC)"
echo "- Investment per stock: ₹25,000"
echo "- Results saved in: trading_results/"
echo ""
echo " Repository ready for daily automated trading simulation!"
