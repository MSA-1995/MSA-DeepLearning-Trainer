# 🧠 Deep Learning Trainer

Advanced Deep Learning model for MSA Trading Bot.

## Features
- 🎓 Trains AI Brain, Exit Strategy, Pattern Recognition, Coin Ranking
- 🤖 Deep Neural Network for predictions
- 💾 Saves knowledge to PostgreSQL Database
- ⏰ Runs every 12 hours automatically

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```
DATABASE_URL=postgresql://user:password@host:port/database
```

3. Run:
```bash
python deep_trainer.py
```

## How it works
1. Loads historical trades from database
2. Trains Deep Learning model
3. Trains all 4 advisors (AI Brain, Exit, Pattern, Ranking)
4. Saves predictions and knowledge to database
5. Trading bot reads from database and uses the knowledge

## Requirements
- Python 3.10+
- PostgreSQL Database
- 512MB RAM minimum
