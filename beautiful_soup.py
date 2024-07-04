from bs4 import BeautifulSoup

html_doc = """
<html>
<head><title>Test Page</title></head>
<body>
<p class="title"><b>Test Page</b></p>
<p class="content">This is a test page.</p>
</body>
</html>
"""

soup = BeautifulSoup(html_doc, 'html.parser')

# Extract the title
title = soup.title.string

# Extract the content of the paragraph with class 'content'
content = soup.find('p', class_='content').text

print("Title:", title)
print("Content:", content)
