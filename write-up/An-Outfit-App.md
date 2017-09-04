# WhatFits: An iOS Outfit Recommender
The goal of this article is to build an Outfit iOS app. The app suggests pieces of clothing or an entire outfit based on a user photo of a piece of clothing. I will scrape outfits in the form of images from [Polyvore.com](https://www.polyvore.com). I will then use machine learning to build the outfit recommendation model. Lastly, I will deploy the model in an iOS app using [CoreML](https://developer.apple.com/documentation/coreml).

## Scraping the data
I am using the Python package [Scrapy](https://doc.scrapy.org/en/latest/index.html). I got started using the tutorial from [PyImageSearch](http://www.pyimagesearch.com/2015/10/12/scraping-images-with-python-and-scrapy/).

### Getting started with Scarpy
For macOS users:

* ```brew install openssl```
* ```pip install scrapy```
* ```pip install pillow```

should do it for dependencies. For more information refer to the [tutorial](http://www.pyimagesearch.com/2015/10/12/scraping-images-with-python-and-scrapy/).

### Setting up your Scrapy spider
Scrapy's shell tool is useful to getting familiar with Scarpy. To use the shell enter ```scrapy shell``` in your command-line and fetch a website using ```fetch("http://your/url/")```. Scrapy will get the website as an object called ```response```. You can access the source via ```response.text``` or specific CSS objects via ```response.css```. The [documentation](https://doc.scrapy.org/en/latest/intro/tutorial.html#extracting-data) has an excellent tutorial on this.

Create the spider by running ```scrapy startproject outfitspider``` in the working directory. Scrapy will make a bunch of files for you. Make the following changes:
#### In settings.py

```python
# in settings.py
ROBOTSTXT_OBEY = False
ITEM_PIPELINES = {
   'outfitspider.pipelines.MyImagesPipeline': 1,
}
IMAGES_STORE = "your/working/directory"
```
The first line tells Scrapy not to obey robot's text. Unfortunately, without setting this your spider will be blocked and you won't be able to scrape anything. The second variable sets your custom pipeline which we will define in the next code listings. The third variable defines where you want to store downloaded images.

#### In pipelines.py
The code is inspired by [this](https://stackoverflow.com/a/22263951) answer on [stackoverflow.com](https://stackoverflow.com/). The class ```MyImagesPipeline``` defines methods that are called when processing each item. I let the path be a Scrapy field (see below) so ```file_path``` just needs to get that field.

```python
import scrapy
from scrapy.contrib.pipeline.images import ImagesPipeline
from scrapy.http import Request
from scrapy.exceptions import DropItem

class MyImagesPipeline(ImagesPipeline):
    def file_path(self, request, response=None, info=None):
        image_guid = request.meta['the_path']
        return 'full/%s' % (image_guid)

    def get_media_requests(self, item, info):
        for i in range(len(item['image_urls'])):
            yield Request(item['image_urls'][i], meta=item)

class OutfitspiderPipeline(object):
    def process_item(self, item, spider):
        return item
```

#### In outfitspider.py
This is where I added most of my own code.


```python
def start_requests(self):
    token = 30
    for _ in range(3):
        url = 'https://www.polyvore.com/cgi/search.sets?.in=json&.out=jsonx&request=%7B%22item_count.to%22%3A%2210%22%2C%22date%22%3A%22day%22%2C%22item_count.from%22%3A%224%22%2C%22page%22%3Anull%2C%22.passback%22%3A%7B%22next_token%22%3A%7B%22limit%22%3A%2230%22%2C%22start%22%3A{}%7D%7D%7D'.format(token)
        yield scrapy.Request(url=url, callback=self.parse)
        token += 60
```

```start_requests``` defines the entry point for the spider. We let this be the [outfit page](https://www.polyvore.com/outfits/search.sets?date=day&item_count.from=4&item_count.to=10) on Polyvore.com. However, this page has an infinite scroll which Scrapy won't see. Turn off Java-script to get an idea of what Scrapy will see. We need to capture where new source-code is coming from when we scroll down the page. The developer tools on Chrome come in useful for that. Go to Settings->More Tools->Developer Tools and the tab Network. Click the Clear symbol in the toolbar. Now scroll down the page until it loads the next set of items. You will notice this by a short jerk or by the fact that the table is filled with items now. Go up to the item that starts with "search.sets". This is the request that the page is sending to the server. If you copy the REQUEST Url and paste it into a new tab it should display a bunch of HTML source. If you scroll down again and again you will notice that only one number in the Request URL changes. So instead of going to the actual Polyvore Outfit page we will just use the request URL and get our image linkes from that. Once we have processed one page we can increase the number at ```%3A__%```.


```python
def parse(self, response):
    t = response.text
    links=re.findall('<div class=\\\\"title\\\\">\\\\n            <a href=\\\\"..([\w\d?=/]*)', t)
    for outfit_url in links:
        outfit_url = '{}{}'.format('https://www.polyvore.com', outfit_url)
        yield scrapy.Request(outfit_url, self.parse_outfit)
```

The ```parse``` method defines what you do with the source above once you get it. From inspecting the source you can see that all outfit names are in a div class "title" and then have a link. So we can use regular expressions to search for this text and grab the link. Since the source uses relative links we need to replace ```..``` with ```https://www.polyvore.com```. Then yield a scarpy request which means to follow that link.

```python
def parse_outfit(self, response):
        img_urls = [img.extract() for img in response.xpath('//img[@class="img_size_m"]/@src')[:5]]
        name = response.xpath('//title/text()').extract_first()
        name = name[:-11] if name else None

        num_views, num_likes = re.findall('hours ago. ([0-9]*) views. ([0-9]*) likes.',response.text)[0]


        for i, img_url in enumerate(img_urls):
            the_path = '{}_views{}_likes{}_pic{}'.format(name, num_views, num_likes, str(i))
            the_path = the_path.title().replace(' ', '')
            the_path = the_path.translate({ord(c): "" for c in "!@#$%^&*[]{};,./<>?\|`~-=+"})
            the_path += '.jpg'
            yield OutfitspiderItem(outfit_title=name,
                                the_path=the_path,
                                num_views=num_views,
                                num_likes=num_likes,
                                item_number=i,
                                image_urls=[img_url])
```
The ```parse_outfit``` method extracts image items and other information from the outfit page. An example of an outfit page would be [this](https://www.polyvore.com/school_look_stylebest/set?id=226905517). ```parse_outfit``` gets the first 5 images of class "img_size_m". It also extracts the name, number of views and number of likes. From the meta information it constructs a file-name and submits all this information for each image individually. The full ```OutfitSpider``` class is shown below.

```python
from outfitspider.items import OutfitspiderItem
import datetime
import scrapy
import re


class OutfitSpider(scrapy.Spider):
    name = "my-outfit-spider"

    def start_requests(self):
        token = 30
        for _ in range(3):
            url = 'https://www.polyvore.com/cgi/search.sets?.in=json&.out=jsonx&request=%7B%22item_count.to%22%3A%2210%22%2C%22date%22%3A%22day%22%2C%22item_count.from%22%3A%224%22%2C%22page%22%3Anull%2C%22.passback%22%3A%7B%22next_token%22%3A%7B%22limit%22%3A%2230%22%2C%22start%22%3A{}%7D%7D%7D'.format(token)
            yield scrapy.Request(url=url, callback=self.parse)
            token += 60

    def parse(self, response):
        t = response.text
        links=re.findall('<div class=\\\\"title\\\\">\\\\n            <a href=\\\\"..([\w\d?=/]*)', t)
        for outfit_url in links:
            outfit_url = '{}{}'.format('https://www.polyvore.com', outfit_url)
            yield scrapy.Request(outfit_url, self.parse_outfit)


    def parse_outfit(self, response):

        img_urls = [img.extract() for img in response.xpath('//img[@class="img_size_m"]/@src')[:5]]
        name = response.xpath('//title/text()').extract_first()
        name = name[:-11] if name else None

        num_views, num_likes = re.findall('hours ago. ([0-9]*) views. ([0-9]*) likes.',response.text)[0]


        for i, img_url in enumerate(img_urls):
            the_path = '{}_views{}_likes{}_pic{}'.format(name, num_views, num_likes, str(i))
            the_path = the_path.title().replace(' ', '')
            the_path = the_path.translate({ord(c): "" for c in "!@#$%^&*[]{};,./<>?\|`~-=+"})
            the_path += '.jpg'
            yield OutfitspiderItem(outfit_title=name,
                                the_path=the_path,
                                num_views=num_views,
                                num_likes=num_likes,
                                item_number=i,
                                image_urls=[img_url])
```

You are now ready to run the spider. 