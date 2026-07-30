@echo off
cd D:\news-classifier\scrapers
call ..\venv\Scripts\activate.bat
python scraper.py >> ..\data\scraper.log 2>&1