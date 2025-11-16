from datetime import datetime
from render_all_of_today import main as render_all_of_today_main
year = 2025
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
days = [1, 5, 10, 15, 20, 25]

for month in months:
    for day in days:
        date = datetime(year, month, day)
        date_string = date.strftime("%Y-%m-%d")
        render_all_of_today_main([
            "--date",
            date_string,
        ])
