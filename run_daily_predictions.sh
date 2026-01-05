#!/bin/bash

# Daily Stock Predictions Runner
# This script activates the virtual environment and runs daily predictions

# Set working directory to script location
cd "$(dirname "$0")"

# Activate virtual environment (adjust path if needed)
source masters/bin/activate

# Set PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run daily predictions with logging
echo "$(date): Starting daily predictions" >> daily_predictions.log

python daily_predictions.py >> daily_predictions.log 2>&1

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "$(date): Daily predictions completed successfully" >> daily_predictions.log
else
    echo "$(date): Daily predictions failed with exit code $exit_code" >> daily_predictions.log
fi

# Optional: Send notification (uncomment if you want email notifications)
# if [ $exit_code -eq 0 ]; then
#     echo "Daily predictions completed successfully at $(date)" | mail -s "Stock Predictions Success" your-email@example.com
# else
#     echo "Daily predictions failed at $(date)" | mail -s "Stock Predictions Failed" your-email@example.com
# fi

exit $exit_code
