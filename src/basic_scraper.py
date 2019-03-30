from recipe_scrapers import scrape_me
import pickle
import time
import os

# Create an empty dictionary or load the existing dictionary to append more data
all_recipes_dictionary = {}
fName = 'all_recipes_dictionary.pickle'
try:
    with open(fName, 'rb') as handle:
        uall_recipes_dictionary = pickle.load(handle)
except IOError:
    all_recipes_dictionary = {}

# We can iterate through allrecipes.com using its id number
# Some recipe pages have been deleted an its id number not reassigned
def page_scraper(pg_num):
    scraper = scrape_me('https://www.allrecipes.com/recipe/' + str(pg_num))
    a = scraper.title()
    b = scraper.total_time()
    c = scraper.ingredients()
    d = scraper.instructions()
    f = scraper.ratings()
    # Append to the dictionary if the page exists, and not currently in the dictionary
    if pg_num not in all_recipes_dictionary and f > -1:
        all_recipes_dictionary[pg_num] = (a,b,c,d,f)

# Our objective is to scrape all data where page id number is between 10,000 and 20,0000
# We implement a sleep function in accord with allrecipes's scraping rules
# We save after looping through every 25 entry and pause the scraper for 1,000 seconds
# Begin to continue scrape from the last id
if not all_recipes_dictionary:
    startid = 10001
else:
    startid = max(all_recipes_dictionary)
for i in range(startid,20000):
    if i%25 == 0: 
        with open('all_recipes_dictionary.pickle', 'wb') as handle:
            pickle.dump(all_recipes_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        time.sleep(1000)
    time.sleep(1.7)
    page_scraper(i)
