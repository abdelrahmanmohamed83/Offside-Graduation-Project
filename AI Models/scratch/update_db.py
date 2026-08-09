import json
import os

db_path = r"d:\offside\data\players_database.json"

# Existing famous players
players = {
    "1": "Alisson Becker",
    "4": "Sergio Ramos",
    "5": "Zinedine Zidane",
    "7": "Cristiano Ronaldo",
    "8": "Toni Kroos",
    "9": "Erling Haaland",
    "10": "Lionel Messi",
    "11": "Mohamed Salah",
    "17": "Kevin De Bruyne",
    "22": "Jude Bellingham"
}

# Fill the rest from 0 to 100
for i in range(101):
    num_str = str(i)
    if num_str not in players:
        players[num_str] = f"Player #{num_str}"

# Sort by number (optional but nice)
sorted_players = dict(sorted(players.items(), key=lambda item: int(item[0])))

with open(db_path, 'w', encoding='utf-8') as f:
    json.dump(sorted_players, f, indent=4, ensure_ascii=False)

print(f"Successfully updated {db_path} with numbers 0-100.")
