
import requests

import re
from bs4 import BeautifulSoup


def scrape_one_liners(url):
    # Send a GET request to the URL
    response = requests.get(url)
    print(response)
# Navigate to the grading system (replace with the URL of your grading system)
    jokes = []
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        #print(soup)
        # Find the <div> elements containing the one-liners
        ol_elements = soup.find_all(re.compile('ol'))
        # Extract and print the text of each <li> element within the <ol>
        for index, ol_element in enumerate(ol_elements, start=1):
            li_elements = ol_element.find_all('li')
            for li_index, li_element in enumerate(li_elements, start=1):
                one_liner_text = li_element.get_text(strip=True)
                jokes.append(one_liner_text)
    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")
    return jokes

url_to_scrape = 'https://bestlifeonline.com/funny-one-liners/'
scraped_jokes = scrape_one_liners(url_to_scrape)
print(len(scraped_jokes))

